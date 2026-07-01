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

Block module-name convention (keys into ``cache.cache[-1][layer]``):
  - 'attn' : self-attention output (modulated)
  - 'mlp'  : feed-forward output (modulated)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.transformers.transformer_2d import Transformer2DModel

from accelerators.speca import (
    SpecACache,
    SpecAState,
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

# VFL (Verification Feedback Loop) — delegates to shared module.
# All globals and hooks live in ``verification_feedback_loop.vfl_state``.
# We re-export the public setters/getters for backward compatibility and keep
# only DiT-specific constants + thin recording wrappers that pass ``model="dit"``.
from verification_feedback_loop.vfl_state import (
    set_vfl_buffer,
    set_vfl_calibrator,
    get_vfl_buffer,
    get_vfl_calibrator,
    set_vfl_step_info,
    get_vfl_step_idx,
    get_vfl_num_steps,
    set_vfl_sample_id,
    record_speca_event,
    record_teacache_event,
)

_VFL_PROBE_LAYER = 20  # TeaCache + SpecA check layer for DiT


def _vfl_record_speca_event(layer_id, timestep_val, step_idx, num_steps,
                              predicted_hidden, full_hidden, error_value,
                              module_name="",
                              latent_input=None, class_labels=None):
    """Record a SpecA verification event for DiT."""
    record_speca_event(
        layer_id=layer_id, timestep_val=timestep_val,
        step_idx=step_idx, num_steps=num_steps,
        predicted_hidden=predicted_hidden, full_hidden=full_hidden,
        error_value=error_value,
        model="dit", module_name=module_name,
        latent_input=latent_input,
        class_labels=class_labels,
    )


def _vfl_record_teacache_event(layer_id, timestep_val, step_idx, num_steps,
                                 predicted_hidden, true_hidden,
                                 raw_diff: float = 0.0,
                                 latent_input=None, class_labels=None):
    """Record a TeaCache probe event for DiT."""
    record_teacache_event(
        layer_id=layer_id, timestep_val=timestep_val,
        step_idx=step_idx, num_steps=num_steps,
        predicted_hidden=predicted_hidden, true_hidden=true_hidden,
        model="dit", raw_diff=raw_diff,
        latent_input=latent_input,
        class_labels=class_labels,
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
                current: Optional[SpecAState] = None,
                cache_dic: Optional[SpecACache] = None,
                teacache_state: Optional[dict] = None,
                class_labels: Optional[torch.Tensor] = None,
                return_dict: bool = True,
                ttt_state: Optional[dict] = None):
        """Explicit DiT forward with SpecA + TeaCache + Session-TTT branching.

        Parameters
        ----------
        hidden_states : (B, C, H, W) latent
        timestep : (B,) timestep indices
        current, cache_dic : SpecA state dicts (None = vanilla / TeaCache / TTT).
        teacache_state : TeaCache state dict (None = vanilla / SpecA).
            Only one of (current+cache_dic) or teacache_state should be set.
            When set, the step counter is NOT advanced here — the caller
            must call ``teacache_step`` after this forward returns.
        class_labels : (B,) ImageNet class indices
        ttt_state : Session-TTT state dict (from ``ttt_state_init``), optional.
            When set TOGETHER with ``teacache_state``, activates the dual
            Teacher/Student path: calc steps distil the backbone into the
            persistent plugin ($\\phi$) via one AdamW step; skip steps route
            the stale cache through $\\phi$. Backbone ($\\Theta$) is frozen
            (gradient flows ONLY through $\\phi$). See ``models/ttt_plugin.py``.
        """
        # Determine mode
        use_speca = current is not None and cache_dic is not None
        use_teacache = teacache_state is not None and not use_speca
        use_ttt = use_teacache and ttt_state is not None

        if use_speca:
            speca_cal_type(cache_dic, current, calibrator=get_vfl_calibrator())
        vanilla = not use_speca and not use_teacache

        # VFL: snapshot the transformer's raw latent input (B, C, H, W) before
        # pos_embed transforms it. Used as replay context for L3 training.
        _vfl_latent_input = hidden_states

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
            should_calc, _ = teacache_decide(teacache_state, modulated,
                                              calibrator=get_vfl_calibrator(),
                                              probe_layer=_VFL_PROBE_LAYER)

            # ===========================================================
            # Session-TTT: persistent plugin modulation of the cache.
            #
            # The plugin (φ) is the ONLY learnable object — the backbone (Θ)
            # is frozen (requires_grad=False at the runner level). On calc
            # steps we distil the full 28-block teacher Z_true into φ; on
            # skip steps we let φ modulate the stale cache. The ambient
            # autograd context (set by _denoise_loop_ttt, which is NOT under
            # @torch.no_grad) determines whether the plugin builds a graph.
            # ===========================================================
            if use_ttt:
                return self._forward_ttt(
                    hidden_states, timestep, class_labels, teacache_state,
                    ttt_state, should_calc, height, width, return_dict,
                )

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
                current.layer = layer_idx
            step_type = 'full' if (vanilla or use_teacache) else current.type

            # adaLN-Zero: returns (norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp)
            norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(
                hidden_states, timestep=timestep, class_labels=class_labels,
                hidden_dtype=hidden_states.dtype,
            )

            if step_type == 'full':
                if use_speca:
                    current.module = 'attn'
                    taylor_cache_init(cache_dic, current)
                attn_out = block.attn1(norm_hidden)
                if use_speca:
                    derivative_approximation(cache_dic, current, attn_out)
                hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_out

                if use_speca:
                    current.module = 'mlp'
                    taylor_cache_init(cache_dic, current)
                norm_ff = block.norm3(hidden_states)
                modulated_ff = norm_ff * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                ff_out = block.ff(modulated_ff)
                if use_speca:
                    derivative_approximation(cache_dic, current, ff_out)
                hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_out

            elif step_type == 'Taylor':
                distance = current.step - current.activated_steps[-1]
                check_layer = cache_dic.check_layer
                do_check = (layer_idx == check_layer and cache_dic.check)
                if do_check:
                    full_hidden = hidden_states.clone()

                hidden_states = cache_step_dit(
                    hidden_states,
                    cache_dic.cache[-1][layer_idx]['attn'],
                    cache_dic.cache[-1][layer_idx]['mlp'],
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
                        metric=cache_dic.error_metric,
                    )
                    current.last_layer_error = gate_value

                    # ---- VFL: record SpecA verification event ----
                    _vfl_record_speca_event(
                        layer_id=layer_idx,
                        timestep_val=get_vfl_step_idx(),
                        step_idx=get_vfl_step_idx(),
                        num_steps=get_vfl_num_steps(),
                        predicted_hidden=hidden_states,
                        full_hidden=full_hidden,
                        error_value=gate_value,
                        module_name="block",
                        latent_input=_vfl_latent_input,
                        class_labels=class_labels,
                    )

        # ---- TeaCache: save residual after blocks complete ----
        if use_teacache:
            teacache_cache_residual(teacache_state, hidden_states, ori_hidden)

            # ---- VFL: TeaCache probe — record (predicted_via_skip, true_full) pair ----
            # Only on calc steps (the block loop just ran), compare what the
            # skip path *would have* given against the just-computed true output.
            # This produces layer-level supervision from TeaCache's step-level
            # decisions — using the full-stack residual as the "prediction".
            _vfl_record_teacache_event(
                layer_id=_VFL_PROBE_LAYER,
                timestep_val=get_vfl_step_idx(),
                step_idx=get_vfl_step_idx(),
                num_steps=get_vfl_num_steps(),
                predicted_hidden=teacache_apply_residual(
                    teacache_state, ori_hidden),
                true_hidden=hidden_states,
                raw_diff=teacache_state.get("last_raw_diff", 0.0),
                latent_input=_vfl_latent_input,
                class_labels=class_labels,
            )

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
    # Session-TTT forward path (Teacher/Student dual mode)
    # ------------------------------------------------------------------

    def _run_full_blocks(self, hidden_states: torch.Tensor,
                         timestep: torch.Tensor,
                         class_labels: torch.Tensor) -> torch.Tensor:
        """Run the full 28-block stack (vanilla, 'full' step type).

        Factored out so it can be called under ``torch.no_grad`` from the
        TTT teacher path WITHOUT affecting the SpecA / TeaCache block loop
        below. Returns the post-block hidden state (B, seq, hidden_dim).
        """
        for block in self.transformer_blocks:
            norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(
                hidden_states, timestep=timestep, class_labels=class_labels,
                hidden_dtype=hidden_states.dtype,
            )
            attn_out = block.attn1(norm_hidden)
            hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_out
            norm_ff = block.norm3(hidden_states)
            modulated_ff = norm_ff * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            ff_out = block.ff(modulated_ff)
            hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_out
        return hidden_states

    def _forward_ttt(self, hidden_states, timestep, class_labels,
                     teacache_state, ttt_state, should_calc,
                     height, width, return_dict):
        """Session-TTT dual-mode forward, sitting atop the TeaCache decision.

        **calc step (Teacher Mode):**
            1. Run the frozen 28-block backbone under ``no_grad`` → Z_true.
            2. Build the stale cached input = (hidden + prev_residual), detached.
            3. Run the plugin (φ) on the stale input → Z_pred (graph-enabled).
            4. Stash (z_pred, z_true) for the loop's ``ttt_train_step``.
            5. Cache the residual against Z_true and emit Z_true through the
               shared tail — so the emitted image uses the high-fidelity
               teacher output (the plugin only *learns* here, it does not yet
               *drive* the output).

        **skip step (Student/Inference Mode):**
            1. Build the stale cached input = (hidden + prev_residual).
            2. Run φ on it → Z_pred under ``no_grad`` (no training signal).
            3. Emit Z_pred through the shared tail. The 28 blocks are bypassed
               entirely — this is where session-level adaptation pays off: a
               well-trained φ sustains high skip ratios without fidelity loss.

        Both branches converge on the shared output tail (norm_out + proj_out +
        unpatchify), so no tail code is duplicated. The TeaCache step counter
        is advanced by the caller (consistent with the vanilla TeaCache path).
        """
        plugin = ttt_state["plugin"]

        # Shared timestep embedding (same signal the tail projector uses).
        t_emb = self.transformer_blocks[0].norm1.emb(
            timestep, class_labels, hidden_dtype=hidden_states.dtype)

        # ---- Plugin runs in fp32 for numerical stability ----
        # The hidden-state MSE can reach ~1e5 (e.g. 170K), which overflows
        # fp16 (max 65504) during the squared-difference and produces NaNs
        # that then corrupt φ. The plugin is tiny (<1M params), so we keep it
        # in fp32 and cast the backbone's fp16 inputs up at the boundary,
        # casting the output back down. This is mixed-precision training done
        # right: backbone stays fp16 (fast inference), plugin trains in fp32.
        plugin_dtype = next(plugin.parameters()).dtype
        backbone_dtype = hidden_states.dtype

        if should_calc:
            # ---------- Teacher Mode ----------
            ori_hidden = hidden_states.clone()

            # 1. Frozen-teacher ground truth: full 28-block pass, no graph.
            #    Wrap in autocast(fp32) so the teacher signal is numerically
            #    clean — fp16 activations in the 28-block stack can spike
            #    beyond 65504 and produce NaN, which would corrupt φ.
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float32):
                    z_true = self._run_full_blocks(
                        hidden_states, timestep, class_labels)

            # Safety net: if the fp16 backbone still produces NaN (e.g.
            # extreme activation values), fall back to the stale cached
            # residual and skip this training opportunity.
            if torch.isnan(z_true).any():
                import warnings
                warnings.warn(
                    f"[TTT] NaN detected in teacher z_true at step "
                    f"{teacache_state.get('cnt', '?')}; "
                    f"falling back to stale cache, skipping training.")
                z_true = teacache_apply_residual(
                    teacache_state, hidden_states).clone()

            # 2. Stale cached input that the plugin must learn to correct.
            #    teacache_apply_residual returns hidden unchanged when no
            #    residual is cached yet (first step) — correct identity.
            cached = teacache_apply_residual(teacache_state, hidden_states).detach()

            # 3. Micro-Epoch distillation on z_true (sample-starvation fix).
            #    The expensive 28-block teacher signal costs ~675M FLOPs to
            #    produce; its value is fully extracted by reusing it multiple
            #    times. Plugin (<1M params) forward+backward is ~100× cheaper
            #    than the backbone — squeezing z_true dry before discarding it.
            micro_epochs = ttt_state.get("micro_epochs", 1)
            z_true_target = z_true.detach().to(plugin_dtype)

            plugin.train()
            for me in range(micro_epochs):
                ttt_state["optimizer"].zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=torch.float32):
                    z_pred = plugin(cached.to(plugin_dtype),
                                    t_emb.to(plugin_dtype))
                    loss = F.mse_loss(z_pred, z_true_target)
                loss.backward()
                ttt_state["optimizer"].step()
            plugin.eval()

            # Telemetry: log the final epoch's loss.
            loss_val = float(loss.detach().item())
            ttt_state["losses"].append(loss_val)
            ttt_state["session_losses"].append(loss_val)
            ttt_state["n_calc"] += 1
            ttt_state["trained_steps"] += 1

            # Clear stash: training was done in the micro-epoch loop; the
            # ambient _denoise_loop_ttt must NOT double-train (ttt_train_step
            # returns 0.0 when z_pred/z_true are None).
            ttt_state["z_pred"] = None
            ttt_state["z_true"] = None

            # 5. Update TeaCache residual against the teacher, emit teacher.
            #    z_true is fp32 from autocast; cast back to backbone dtype
            #    for the tail (which is fp16) and for caching.
            z_true_bd = z_true.to(backbone_dtype)
            teacache_cache_residual(teacache_state, z_true_bd, ori_hidden)
            hidden_states = z_true_bd

        else:
            # ---------- Student/Inference Mode ----------
            # Bypass all 28 blocks; φ modulates the stale cache. No training.
            cached = teacache_apply_residual(teacache_state, hidden_states)
            with torch.no_grad():
                z_pred = plugin(cached.detach().to(plugin_dtype),
                                t_emb.to(plugin_dtype))
            hidden_states = z_pred.to(backbone_dtype)

        # Shared output tail (identical to the vanilla / TeaCache path).
        conditioning = t_emb
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
                         current: Optional[SpecAState],
                         cache_dic: Optional[SpecACache],
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

    def forward_with_cfg_ttt(self,
                             hidden_states: torch.Tensor,
                             timestep: torch.Tensor,
                             teacache_state: Optional[dict],
                             ttt_state: Optional[dict],
                             class_labels: torch.Tensor,
                             cfg_scale: float):
        """CFG wrapper for the Session-TTT path.

        Mirrors ``forward_with_cfg`` (takes the cond half, duplicates it to
        [cond, cond], runs ONE forward), threading the TTT state so the
        plugin's teacher/student logic fires inside ``forward``. CFG is
        applied to the noise channels exactly as in the vanilla path; it is
        orthogonal to the plugin, which operates in the pre-tail hidden space.
        """
        half = hidden_states[: len(hidden_states) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep,
                                 current=None, cache_dic=None,
                                 teacache_state=teacache_state,
                                 class_labels=class_labels, return_dict=False,
                                 ttt_state=ttt_state)[0]
        eps, rest = (model_out[:, :self.config.in_channels],
                     model_out[:, self.config.in_channels:])
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
