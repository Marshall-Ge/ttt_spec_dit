# -*- coding: utf-8 -*-
"""Explicit PixArt-α transformer (text-to-image, ada_norm_single).

This module exposes a hand-written transformer whose forward signature is
explicit about the SpecA cache state::

    PixArtTransformer2D.forward(x, encoder_hidden_states, t, current, cache_dic,
                                added_cond_kwargs, ...)

The block loop and the full/Taylor branching are visible line-by-line; there
is no monkeypatching. Submodules reuse the diffusers
``PixArtTransformer2DModel`` building blocks (``BasicTransformerBlock``-style
attn1/attn2/ff), so the existing state_dict keys line up exactly.

Block module-name convention (keys into ``cache.cache[-1][layer]``):
  - 'attn1' : self-attention output (modulated via norm1)
  - 'attn2' : cross-attention output (raw hidden_states, no gate)
  - 'ff'    : feed-forward output (modulated via norm2)
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from diffusers.models.transformers.pixart_transformer_2d import PixArtTransformer2DModel
from diffusers import PixArtAlphaPipeline

from accelerators.speca import (
    SpecACache,
    SpecAState,
    speca_cal_type,
    taylor_cache_init,
    derivative_approximation,
    cache_step_pixart,
    compute_error_gate,
)
from accelerators.teacache import (
    teacache_decide,
    teacache_cache_residual,
    teacache_apply_residual,
    compute_modulated_input,
)

# VFL (Verification Feedback Loop) — delegates to shared module.
# All globals and hooks live in ``verification_feedback_loop.vfl_state``.
# We re-export the public setters/getters for backward compatibility and keep
# only PixArt-specific constants + thin recording wrappers that pass ``model="pixart"``.
from verification_feedback_loop.vfl_state import (
    set_vfl_buffer,
    set_vfl_calibrator,
    get_vfl_buffer,
    set_vfl_step_info,
    record_speca_event,
    record_teacache_event,
)

_VFL_PROBE_LAYER = 24  # TeaCache + SpecA check layer for PixArt


def _vfl_record_speca_event(layer_id, timestep_val, step_idx, num_steps,
                              predicted_hidden, full_hidden, error_value,
                              module_name="",
                              latent_input=None, encoder_hidden_states=None):
    """Record a SpecA verification event for PixArt."""
    record_speca_event(
        layer_id=layer_id, timestep_val=timestep_val,
        step_idx=step_idx, num_steps=num_steps,
        predicted_hidden=predicted_hidden, full_hidden=full_hidden,
        error_value=error_value,
        model="pixart", module_name=module_name,
        latent_input=latent_input,
        encoder_hidden_states=encoder_hidden_states,
    )


def _vfl_record_teacache_event(layer_id, timestep_val, step_idx, num_steps,
                                 predicted_hidden, true_hidden,
                                 raw_diff: float = 0.0,
                                 latent_input=None, encoder_hidden_states=None):
    """Record a TeaCache probe event for PixArt."""
    record_teacache_event(
        layer_id=layer_id, timestep_val=timestep_val,
        step_idx=step_idx, num_steps=num_steps,
        predicted_hidden=predicted_hidden, true_hidden=true_hidden,
        model="pixart", raw_diff=raw_diff,
        latent_input=latent_input,
        encoder_hidden_states=encoder_hidden_states,
    )


class PixArtTransformer2D(nn.Module):
    """Explicit PixArt-α transformer with SpecA-aware forward.

    The submodule tree is built by instantiating a diffusers
    ``PixArtTransformer2DModel`` and borrowing its children, so state_dict
    keys line up exactly with the released checkpoint.
    """

    def __init__(self,
                 num_attention_heads: int = 16,
                 attention_head_dim: int = 72,
                 in_channels: int = 4,
                 out_channels: int = 8,
                 num_layers: int = 28,
                 sample_size: int = 64,
                 patch_size: int = 2,
                 cross_attention_dim: int = 1152,
                 use_additional_conditions: bool = False,
                 caption_channels: int = 4096):
        super().__init__()

        # Build a stock diffusers PixArtTransformer2DModel purely to harvest
        # its submodule tree (names + shapes match the checkpoint exactly).
        ref = PixArtTransformer2DModel(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            sample_size=sample_size,
            patch_size=patch_size,
            cross_attention_dim=cross_attention_dim,
            attention_bias=True,
            activation_fn="gelu-approximate",
            norm_elementwise_affine=False,
            norm_type="ada_norm_single",
            use_additional_conditions=use_additional_conditions,
            caption_channels=caption_channels,
        )

        # Borrow the subtrees so our attribute names match the checkpoint keys.
        self.pos_embed = ref.pos_embed
        self.transformer_blocks = ref.transformer_blocks
        self.norm_out = ref.norm_out
        self.proj_out = ref.proj_out
        self.scale_shift_table = ref.scale_shift_table
        self.adaln_single = ref.adaln_single
        self.caption_projection = ref.caption_projection

        # Expose the diffusers config object.
        self.config = ref.config
        self.out_channels = out_channels

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                timestep: torch.Tensor,
                current: Optional[SpecAState] = None,
                cache_dic: Optional[SpecACache] = None,
                teacache_state: Optional[dict] = None,
                added_cond_kwargs: Optional[Dict[str, Any]] = None,
                attention_mask: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                return_dict: bool = True):
        """Explicit PixArt forward with SpecA + TeaCache branching.

        Parameters
        ----------
        hidden_states : (B, C, H, W) latent
        encoder_hidden_states : (B, seq, dim) T5 caption embeddings
        timestep : (B,) timestep indices
        current, cache_dic : SpecA state dicts (None = vanilla / TeaCache).
        teacache_state : TeaCache state dict (None = vanilla / SpecA).
            When set, the step counter is NOT advanced here — the caller
            must call ``teacache_step`` after this forward returns.
        added_cond_kwargs : dict with ``resolution`` / ``aspect_ratio`` keys
        attention_mask : (B, key_tokens) or (B, 1, key_tokens) self-attn mask
        encoder_attention_mask : (B, seq) or (B, 1, seq) cross-attn mask
        """
        # 1. Decide step type once for the whole stack.
        use_speca = current is not None and cache_dic is not None
        use_teacache = teacache_state is not None and not use_speca

        if use_speca:
            speca_cal_type(cache_dic, current, calibrator=_vfl_calibrator)
        vanilla = not use_speca and not use_teacache

        # Preprocess attention masks (replicate diffusers convention).
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 2. Patch embed + adaln_single.
        batch_size = hidden_states.shape[0]
        height, width = (
            hidden_states.shape[-2] // self.config.patch_size,
            hidden_states.shape[-1] // self.config.patch_size,
        )

        # VFL: snapshot the transformer's raw latent input (B, C, H, W) before
        # pos_embed transforms it. Used as replay context for L3 training.
        _vfl_latent_input = hidden_states

        hidden_states = self.pos_embed(hidden_states)

        # adaln_single returns (timestep_emb, embedded_timestep)
        timestep_emb, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )

        # VFL: snapshot the raw T5 embeddings BEFORE caption_projection —
        # re-running forward will call caption_projection again, so we must
        # pass the un-projected tensor to avoid double projection.
        _vfl_encoder_hidden_states = encoder_hidden_states

        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, -1, hidden_states.shape[-1])

        # ---- TeaCache: decide at pos_embed output, before blocks ----
        if use_teacache:
            modulated = compute_modulated_input(
                self, hidden_states, timestep_emb)
            should_calc, _ = teacache_decide(teacache_state, modulated,
                                            calibrator=_vfl_calibrator,
                                            probe_layer=_VFL_PROBE_LAYER)

            if not should_calc:
                hidden_states = teacache_apply_residual(
                    teacache_state, hidden_states)
                # Jump to tail
                shift, scale = (
                    self.scale_shift_table[None]
                    + embedded_timestep[:, None].to(self.scale_shift_table.device)
                ).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states)
                hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(hidden_states.device)
                hidden_states = self.proj_out(hidden_states)
                hidden_states = hidden_states.squeeze(1)
                hidden_states = hidden_states.reshape(
                    shape=(-1, height, width, self.config.patch_size,
                           self.config.patch_size, self.out_channels)
                )
                hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
                output = hidden_states.reshape(
                    shape=(-1, self.out_channels,
                           height * self.config.patch_size, width * self.config.patch_size)
                )
                if not return_dict:
                    return (output,)
                from diffusers.models.modeling_outputs import Transformer2DModelOutput
                return Transformer2DModelOutput(sample=output)

            # Should calc: save input for residual caching
            ori_hidden = hidden_states.clone()

        # 3. Block loop (visible, no monkeypatch).
        for layer_idx, block in enumerate(self.transformer_blocks):
            if use_speca:
                current.layer = layer_idx
            step_type = 'full' if (vanilla or use_teacache) else current.type

            # ------------------- ada_norm_single projection ------------------
            proj = (block.scale_shift_table[None]
                    + timestep_emb.reshape(batch_size, 6, -1))
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                proj.chunk(6, dim=1)

            if step_type == 'full':
                # --- submodule 1: self-attention (norm1-modulated) ---
                if use_speca:
                    current.module = 'attn1'
                    taylor_cache_init(cache_dic, current)
                norm_hidden = block.norm1(hidden_states)
                modulated = norm_hidden * (1 + scale_msa) + shift_msa
                attn1_out = block.attn1(modulated)
                if use_speca:
                    derivative_approximation(cache_dic, current, attn1_out)
                hidden_states = hidden_states + gate_msa * attn1_out

                # --- submodule 2: cross-attention (raw hidden, no gate) ---
                if use_speca:
                    current.module = 'attn2'
                    taylor_cache_init(cache_dic, current)
                # attn2 takes raw hidden_states as query (PixArt convention)
                attn2_out = block.attn2(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                )
                if use_speca:
                    derivative_approximation(cache_dic, current, attn2_out)
                hidden_states = hidden_states + attn2_out  # no gate

                # --- submodule 3: feed-forward (norm2-modulated) ---
                if use_speca:
                    current.module = 'ff'
                    taylor_cache_init(cache_dic, current)
                norm_ff = block.norm2(hidden_states)
                modulated_ff = norm_ff * (1 + scale_mlp) + shift_mlp
                ff_out = block.ff(modulated_ff)
                if use_speca:
                    derivative_approximation(cache_dic, current, ff_out)
                hidden_states = hidden_states + gate_mlp * ff_out

            elif step_type == 'Taylor':
                distance = current.step - current.activated_steps[-1]
                check_layer = cache_dic.check_layer
                do_check = (layer_idx == check_layer and cache_dic.check)
                if do_check:
                    full_hidden = hidden_states.clone()

                hidden_states = cache_step_pixart(
                    hidden_states,
                    cache_dic.cache[-1][layer_idx]['attn1'],
                    cache_dic.cache[-1][layer_idx]['attn2'],
                    cache_dic.cache[-1][layer_idx]['ff'],
                    gate_msa, gate_mlp, distance,
                )

                # Optional error probe: recompute full block on saved input.
                if do_check:
                    proj_full = (block.scale_shift_table[None]
                                 + timestep_emb.reshape(batch_size, 6, -1))
                    fsh_msa, fsc_msa, fg_msa, fsh_mlp, fsc_mlp, fg_mlp = \
                        proj_full.chunk(6, dim=1)

                    # attn1
                    norm_h = block.norm1(full_hidden)
                    mod_h = norm_h * (1 + fsc_msa) + fsh_msa
                    a1 = block.attn1(mod_h)
                    full_hidden = full_hidden + fg_msa * a1
                    # attn2
                    a2 = block.attn2(
                        full_hidden,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=encoder_attention_mask,
                    )
                    full_hidden = full_hidden + a2
                    # ff
                    norm_ff_f = block.norm2(full_hidden)
                    mod_ff_f = norm_ff_f * (1 + fsc_mlp) + fsh_mlp
                    ff_f = block.ff(mod_ff_f)
                    full_hidden = full_hidden + fg_mlp * ff_f

                    gate_value, _ = compute_error_gate(
                        hidden_states, full_hidden,
                        metric=cache_dic.error_metric,
                    )
                    current.last_layer_error = gate_value

                    # ---- VFL: record SpecA verification event ----
                    _vfl_record_speca_event(
                        layer_id=layer_idx,
                        timestep_val=_vfl_step_idx,
                        step_idx=_vfl_step_idx,
                        num_steps=_vfl_num_steps,
                        predicted_hidden=hidden_states,
                        full_hidden=full_hidden,
                        error_value=gate_value,
                        module_name="block",
                        latent_input=_vfl_latent_input,
                        encoder_hidden_states=_vfl_encoder_hidden_states,
                    )

        # ---- TeaCache: save residual after blocks complete ----
        if use_teacache:
            teacache_cache_residual(teacache_state, hidden_states, ori_hidden)

            # ---- VFL: TeaCache probe — record (predicted_via_skip, true_full) pair ----
            _vfl_record_teacache_event(
                layer_id=_VFL_PROBE_LAYER,
                timestep_val=_vfl_step_idx,
                step_idx=_vfl_step_idx,
                num_steps=_vfl_num_steps,
                predicted_hidden=teacache_apply_residual(
                    teacache_state, ori_hidden),
                true_hidden=hidden_states,
                raw_diff=teacache_state.get("last_raw_diff", 0.0),
                latent_input=_vfl_latent_input,
                encoder_hidden_states=_vfl_encoder_hidden_states,
            )

        # 4. Output (norm_out + proj_out + unpatchify) — always runs.
        shift, scale = (
            self.scale_shift_table[None]
            + embedded_timestep[:, None].to(self.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(hidden_states.device)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.config.patch_size,
                   self.config.patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels,
                   height * self.config.patch_size, width * self.config.patch_size)
        )

        if not return_dict:
            return (output,)
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
        return Transformer2DModelOutput(sample=output)

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, repo: str, cache_dir: Optional[str] = None,
                        dtype: torch.dtype = torch.float16,
                        **kwargs) -> Tuple["PixArtTransformer2D", Any, Any, Any]:
        """Load weights from a diffusers PixArtAlphaPipeline.

        Returns ``(model, vae, tokenizer, text_encoder)`` so the caller
        can manage all pipeline sub-components.

        Parameters
        ----------
        repo : str
            HuggingFace repo id or local path (e.g. "PixArt-alpha/PixArt-XL-2-512x512").
        cache_dir : str, optional
            HF cache directory.
        dtype : torch.dtype
            Model dtype (default float16).
        """
        pipe = PixArtAlphaPipeline.from_pretrained(
            repo, cache_dir=cache_dir, torch_dtype=dtype,
            local_files_only=True,
        )

        orig = pipe.transformer
        cfg = orig.config

        model = cls(
            num_attention_heads=cfg.get("num_attention_heads", 16),
            attention_head_dim=cfg.get("attention_head_dim", 72),
            in_channels=cfg.get("in_channels", 4),
            out_channels=cfg.get("out_channels", 8),
            num_layers=cfg.get("num_layers", 28),
            sample_size=cfg.get("sample_size", 64),
            patch_size=cfg.get("patch_size", 2),
            cross_attention_dim=cfg.get("cross_attention_dim", 1152),
            use_additional_conditions=getattr(orig, "use_additional_conditions", False),
            caption_channels=cfg.get("caption_channels", 4096),
        )

        # Copy state from the pipeline's transformer into our newly-built one.
        model.load_state_dict(orig.state_dict(), strict=True)

        return model, pipe.vae, pipe.tokenizer, pipe.text_encoder
