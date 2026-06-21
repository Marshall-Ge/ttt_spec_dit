# -*- coding: utf-8 -*-
"""Explicit DiT-2-256 transformer (class-conditional, ada_norm_zero).

This module exposes a hand-written transformer whose forward signature is
explicit about acceleration state::

    DiTTransformer2D.forward(x, t, current, cache_dic, teacache_state, class_labels)

The block loop and full/Taylor/TeaCache branching are visible line-by-line;
there is no monkeypatching. Submodules reuse the diffusers ``Transformer2DModel``
building blocks (``AdaLayerNormZero``, ``BasicTransformerBlock``-style attn/ff),
so the existing ``diffusion_pytorch_model.bin`` checkpoint loads directly and
``eval/latency.py``'s tail-profiler (which indexes
``transformer.transformer_blocks[0].norm1.emb`` / ``proj_out_1`` / ``proj_out_2``)
keeps working unchanged.

Block module-name convention (keys into ``cache_dic['cache'][-1][layer]``):
  - 'attn' : self-attention output (modulated)
  - 'mlp'  : feed-forward output (modulated)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.transformers.transformer_2d import Transformer2DModel

from accelerators.speca import (
    speca_cal_type,
    taylor_cache_init,
    derivative_approximation,
    cache_step_dit,
    compute_error_gate,
)
from accelerators.teacache import (
    teacache_decide,
    teacache_cache_residual,
    teacache_apply_residual,
    compute_modulated_input_dit,
)


class DiTTransformer2D(nn.Module):
    """Explicit DiT-2-256 transformer with SpecA + TeaCache-aware forward.

    The submodule tree is built by instantiating a diffusers
    ``Transformer2DModel`` (norm_type='ada_norm_zero') and borrowing its
    children, so state_dict keys line up exactly with the released
    ``diffusion_pytorch_model.bin``.
    """

    def __init__(self,
                 num_attention_heads: int = 16,
                 attention_head_dim: int = 72,
                 in_channels: int = 4,
                 out_channels: int = 8,
                 num_layers: int = 28,
                 sample_size: int = 32,
                 patch_size: int = 2,
                 num_embeds_ada_norm: int = 1000):
        super().__init__()

        # Build a stock diffusers Transformer2DModel purely to harvest its
        # submodule tree (names + shapes match the checkpoint exactly).
        ref = Transformer2DModel(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            sample_size=sample_size,
            patch_size=patch_size,
            num_embeds_ada_norm=num_embeds_ada_norm,
            activation_fn="gelu-approximate",
            attention_bias=True,
            norm_elementwise_affine=False,
            norm_type="ada_norm_zero",
            cross_attention_dim=None,
        )

        # Borrow the subtrees so our attribute names match the checkpoint keys.
        self.pos_embed = ref.pos_embed
        self.transformer_blocks = ref.transformer_blocks
        self.norm_out = ref.norm_out
        self.proj_out_1 = ref.proj_out_1
        self.proj_out_2 = ref.proj_out_2

        # Expose the diffusers config object so FLOPs tail-profiler
        # (which reads .patch_size / .out_channels) keeps working.
        self.config = ref.config
        self.out_channels = out_channels
        self.patch_size = patch_size

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self,
                hidden_states: torch.Tensor,
                timestep: torch.Tensor,
                current: Optional[dict] = None,
                cache_dic: Optional[dict] = None,
                teacache_state: Optional[dict] = None,
                class_labels: Optional[torch.Tensor] = None,
                return_dict: bool = True):
        """Explicit DiT forward with SpecA + TeaCache branching.

        Parameters
        ----------
        hidden_states : (B, C, H, W) latent
        timestep : (B,) timestep indices
        current, cache_dic : SpecA state dicts (None = vanilla / TeaCache).
        teacache_state : TeaCache state dict (None = vanilla / SpecA).
            Only one of (current+cache_dic) or teacache_state should be set.
            When set, the step counter is NOT advanced here — the caller
            must call ``teacache_step`` after this forward returns.
        class_labels : (B,) ImageNet class indices
        """
        # Determine mode
        use_speca = current is not None and cache_dic is not None
        use_teacache = teacache_state is not None and not use_speca

        if use_speca:
            speca_cal_type(cache_dic, current)
        vanilla = not use_speca and not use_teacache

        # 2. Patch embed + positional embed.
        height, width = (
            hidden_states.shape[-2] // self.patch_size,
            hidden_states.shape[-1] // self.patch_size,
        )
        hidden_states = self.pos_embed(hidden_states)

        # ---- TeaCache: decide at pos_embed output, before blocks ----
        if use_teacache:
            modulated = compute_modulated_input_dit(
                self, hidden_states, timestep, class_labels)
            should_calc, _ = teacache_decide(teacache_state, modulated)

            if not should_calc:
                # Skip all blocks — apply cached residual
                hidden_states = teacache_apply_residual(
                    teacache_state, hidden_states)
                # Jump directly to tail
                conditioning = self.transformer_blocks[0].norm1.emb(
                    timestep, class_labels, hidden_dtype=hidden_states.dtype)
                shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
                hidden_states = self.proj_out_2(hidden_states)
                hidden_states = hidden_states.reshape(
                    shape=(-1, height, width, self.patch_size,
                           self.patch_size, self.out_channels)
                )
                hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
                output = hidden_states.reshape(
                    shape=(-1, self.out_channels,
                           height * self.patch_size, width * self.patch_size)
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
                current['layer'] = layer_idx
            step_type = 'full' if (vanilla or use_teacache) else current['type']

            # adaLN-Zero: returns (norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp)
            norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(
                hidden_states, timestep=timestep, class_labels=class_labels,
                hidden_dtype=hidden_states.dtype,
            )

            if step_type == 'full':
                if use_speca:
                    current['module'] = 'attn'
                    taylor_cache_init(cache_dic, current)
                attn_out = block.attn1(norm_hidden)
                if use_speca:
                    derivative_approximation(cache_dic, current, attn_out)
                hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_out

                if use_speca:
                    current['module'] = 'mlp'
                    taylor_cache_init(cache_dic, current)
                norm_ff = block.norm3(hidden_states)
                modulated_ff = norm_ff * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                ff_out = block.ff(modulated_ff)
                if use_speca:
                    derivative_approximation(cache_dic, current, ff_out)
                hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_out

            elif step_type == 'Taylor':
                distance = current['step'] - current['activated_steps'][-1]
                check_layer = cache_dic['check_layer']
                do_check = (layer_idx == check_layer and cache_dic['check'])
                if do_check:
                    full_hidden = hidden_states.clone()

                hidden_states = cache_step_dit(
                    hidden_states,
                    cache_dic['cache'][-1][layer_idx]['attn'],
                    cache_dic['cache'][-1][layer_idx]['mlp'],
                    gate_msa, gate_mlp, distance,
                )

                if do_check:
                    fnh, fgate_msa, fshift_mlp, fscale_mlp, fgate_mlp = block.norm1(
                        full_hidden, timestep=timestep, class_labels=class_labels,
                        hidden_dtype=full_hidden.dtype,
                    )
                    attn_full = block.attn1(fnh)
                    full_hidden = full_hidden + fgate_msa.unsqueeze(1) * attn_full
                    norm_ff = block.norm3(full_hidden)
                    modulated_ff = norm_ff * (1 + fscale_mlp[:, None]) + fshift_mlp[:, None]
                    ff_full = block.ff(modulated_ff)
                    full_hidden = full_hidden + fgate_mlp.unsqueeze(1) * ff_full

                    gate_value, _ = compute_error_gate(
                        hidden_states, full_hidden,
                        metric=cache_dic['error_metric'],
                    )
                    current['last_layer_error'] = gate_value

        # ---- TeaCache: save residual after blocks complete ----
        if use_teacache:
            teacache_cache_residual(teacache_state, hidden_states, ori_hidden)

        # 4. Output (norm_out + proj_out_1/2 + unpatchify) — always runs.
        conditioning = self.transformer_blocks[0].norm1.emb(
            timestep, class_labels, hidden_dtype=hidden_states.dtype)
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        hidden_states = self.proj_out_2(hidden_states)

        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.patch_size,
                   self.patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels,
                   height * self.patch_size, width * self.patch_size)
        )

        if not return_dict:
            return (output,)
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
        return Transformer2DModelOutput(sample=output)

    # ------------------------------------------------------------------
    # Classifier-free guidance wrapper.
    # ------------------------------------------------------------------

    def forward_with_cfg(self,
                         hidden_states: torch.Tensor,
                         timestep: torch.Tensor,
                         current: Optional[dict],
                         cache_dic: Optional[dict],
                         teacache_state: Optional[dict] = None,
                         class_labels: torch.Tensor = None,
                         cfg_scale: float = 4.0):
        half = hidden_states[: len(hidden_states) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep, current, cache_dic,
                                 teacache_state=teacache_state,
                                 class_labels=class_labels, return_dict=False)[0]
        # CFG only on noise channels; variance (learned sigma) passes through.
        # noise channels = config.in_channels (= 4 for SD VAE latent).
        eps, rest = model_out[:, :self.config.in_channels], model_out[:, self.config.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "DiTTransformer2D":
        """Load weights from a diffusers ``diffusion_pytorch_model.bin``.

        ``path`` may point at the ``.bin`` file, the transformer directory,
        or the pipeline root (containing a ``transformer/`` subdirectory).
        """
        import os
        if os.path.isdir(path):
            t_dir = os.path.join(path, "transformer")
            if os.path.isdir(t_dir):
                path = t_dir

        if os.path.isdir(path):
            bin_path = os.path.join(path, "diffusion_pytorch_model.bin")
            cfg_path = os.path.join(path, "config.json")
        else:
            bin_path = path
            cfg_path = os.path.join(os.path.dirname(path), "config.json")

        import json
        with open(cfg_path) as f:
            cfg = json.load(f)
        model = cls(
            num_attention_heads=cfg.get("num_attention_heads", 16),
            attention_head_dim=cfg.get("attention_head_dim", 72),
            in_channels=cfg.get("in_channels", 4),
            out_channels=cfg.get("out_channels", 8),
            num_layers=cfg.get("num_layers", 28),
            sample_size=cfg.get("sample_size", 32),
            patch_size=cfg.get("patch_size", 2),
            num_embeds_ada_norm=cfg.get("num_embeds_ada_norm", 1000),
        )

        state_dict = torch.load(bin_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        return model


# ------------------------------------------------------------------
# Deferred re-export for eval/latency.py compatibility.
# ------------------------------------------------------------------

def __getattr__(name):
    if name == 'DiTGenerator':
        from run_dit import DiTGenerator as _Gen
        return _Gen
    raise AttributeError(f"module 'models.dit' has no attribute {name!r}")
