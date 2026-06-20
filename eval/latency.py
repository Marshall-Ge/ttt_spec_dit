# -*- coding: utf-8 -*-
"""Latency, FLOPs, and Speedup metrics.

- LatencyMetric:   wall-clock latency (s) per image, vanilla vs accelerated
- FLOPsMetric:     profiled FLOPs per step, accumulated via TeaCache decisions
- SpeedupMetric:   derived speedup ratio from latency or FLOPs
"""

import numpy as np
import torch
from .base import Metric


# ===========================================================================
# Latency
# ===========================================================================

class LatencyMetric(Metric):
    """Collects per-image wall-clock latencies (vanilla + accelerated).

    Call ``add_pair(vanilla_s, accel_s)`` for each image pair.
    """

    def __init__(self):
        self._vanilla: list = []
        self._accel: list = []

    def add_pair(self, vanilla_s: float, accel_s: float):
        self._vanilla.append(vanilla_s)
        self._accel.append(accel_s)

    def add_pairs_batch(self, vanilla_list, accel_list):
        """Add latency pairs for a batch in one call.

        Parameters
        ----------
        vanilla_list : list of float
            Length B, vanilla latencies.
        accel_list : list of float
            Length B, accelerated latencies.
        """
        self._vanilla.extend(vanilla_list)
        self._accel.extend(accel_list)

    def add(self, image: torch.Tensor, prompt: str = None,
            reference: torch.Tensor = None):
        pass  # use add_pair() instead

    def compute(self) -> dict:
        if not self._vanilla:
            return {
                "latency_vanilla_mean": float("nan"),
                "latency_accel_mean": float("nan"),
                "speedup_latency": float("nan"),
            }
        v = np.array(self._vanilla)
        a = np.array(self._accel)
        per_image_speedup = v / a
        return {
            "latency_vanilla_mean": float(v.mean()),
            "latency_vanilla_std": float(v.std()),
            "latency_accel_mean": float(a.mean()),
            "latency_accel_std": float(a.std()),
            "speedup_latency_mean": float(per_image_speedup.mean()),
            "speedup_latency_std": float(per_image_speedup.std()),
        }

    def reset(self):
        self._vanilla.clear()
        self._accel.clear()


# ===========================================================================
# FLOPs
# ===========================================================================

class FLOPsMetric(Metric):
    """Profiles the transformer once, then counts FLOPs per generation using
    TeaCache calc/skip decisions.

    FLOPs are reported in **TeraFLOPs** (÷ 1e12).

    Parameters
    ----------
    generator : PixArtGenerator
        Must already be loaded (``.load()`` called).
    """

    def __init__(self, generator):
        self._gen = generator
        self._flops_full: float = 0.0   # FLOPs for one full step
        self._flops_skip: float = 0.0   # FLOPs for one skip step
        self._flops_vanilla_step: float = 0.0  # full step (always the same)
        self._profiled: bool = False

        # Accumulators
        self._total_vanilla: float = 0.0
        self._total_accel: float = 0.0
        self._n: int = 0

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    def profile(self):
        """Run profiling now. Call BEFORE TeaCache is installed."""
        self._profile_once()

    def _profile_once(self):
        if self._profiled:
            return

        from torch.utils.flop_counter import FlopCounterMode
        from models.dit import DiTGenerator
        transformer = self._gen.transformer
        device = self._gen.device
        dtype = self._gen.dtype
        is_dit = isinstance(self._gen, DiTGenerator)

        # Build scheduler to get a valid timestep
        self._gen._build_scheduler()
        timesteps = self._gen.scheduler.timesteps
        shape = self._gen.latent_shape

        # Latent input
        gen_torch = torch.Generator(device=device).manual_seed(0)
        latents = torch.randn(shape, device=device, dtype=dtype,
                              generator=gen_torch) * self._gen.scheduler.init_noise_sigma
        t = timesteps[0]
        latent_input = self._gen.scheduler.scale_model_input(latents, t)
        current_t = t.expand(1).to(torch.int64)

        if is_dit:
            # DiT: forward(hidden_states, timestep, class_labels)
            class_labels = self._gen.encode_prompt(0)  # class 0
            with FlopCounterMode(display=False) as fcm:
                _ = transformer(
                    latent_input,
                    timestep=current_t,
                    class_labels=class_labels,
                    return_dict=False,
                )
            self._flops_full = fcm.get_total_flops()
            self._flops_skip = _profile_tail_flops_dit(
                transformer, latent_input, current_t, class_labels, device, dtype)
        else:
            # PixArt
            prompt_embeds, attn_mask = self._gen.encode_prompt("test")
            added = {"resolution": None, "aspect_ratio": None}

            with FlopCounterMode(display=False) as fcm:
                _ = transformer(
                    latent_input,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=attn_mask,
                    timestep=current_t,
                    added_cond_kwargs=added,
                    return_dict=False,
                )
            self._flops_full = fcm.get_total_flops()
            self._flops_skip = _profile_tail_flops(transformer, latent_input,
                                                    prompt_embeds, attn_mask,
                                                    current_t, added, device, dtype)

        self._flops_vanilla_step = self._flops_full
        self._profiled = True

        print(f"  [FLOPs] profiled — full={self._flops_full/1e9:.3f} GFLOPs, "
              f"skip={self._flops_skip/1e9:.3f} GFLOPs "
              f"(reduction={1 - self._flops_skip/self._flops_full:.0%})")

    # ------------------------------------------------------------------
    # Accumulate
    # ------------------------------------------------------------------

    def add(self, image: torch.Tensor, prompt: str = None,
            reference: torch.Tensor = None):
        pass  # use add_generation()

    def add_generation(self, teacache):
        """Accumulate FLOPs from one generation's TeaCache decisions."""
        self._profile_once()
        n_calc = sum(1 for d in teacache.decisions if d == "calc")
        n_skip = sum(1 for d in teacache.decisions if d == "skip")
        total_steps = n_calc + n_skip
        self._total_vanilla += total_steps * self._flops_full
        self._total_accel += n_calc * self._flops_full + n_skip * self._flops_skip
        self._n += 1

    def add_vanilla_steps(self, n_steps: int):
        """Accumulate FLOPs for a vanilla-only generation (no TeaCache)."""
        self._profile_once()
        self._total_vanilla += n_steps * self._flops_full
        self._total_accel += n_steps * self._flops_full  # vanilla = full every step
        self._n += 1

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def compute(self) -> dict:
        if self._n == 0:
            return {
                "flops_vanilla_T": float("nan"),
                "flops_accel_T": float("nan"),
                "flops_reduction": float("nan"),
                "speedup_flops": float("nan"),
            }
        v = self._total_vanilla / self._n
        a = self._total_accel / self._n
        return {
            "flops_vanilla_T": v / 1e12,
            "flops_accel_T": a / 1e12,
            "flops_reduction": 1.0 - a / v if v > 0 else 0.0,
            "speedup_flops": v / a if a > 0 else float("nan"),
        }

    def reset(self):
        self._total_vanilla = 0.0
        self._total_accel = 0.0
        self._n = 0


# ---------------------------------------------------------------------------
# Internal: profile FLOPs of the tail (norm_out + proj_out + unpatchify)
# ---------------------------------------------------------------------------

def _profile_tail_flops(transformer, latent_input, prompt_embeds, attn_mask,
                        current_t, added, device, dtype):
    """Run a forward that skips all transformer blocks, counting only the tail."""
    from torch.utils.flop_counter import FlopCounterMode

    batch_size = latent_input.shape[0]
    height = latent_input.shape[-2] // transformer.config.patch_size
    width = latent_input.shape[-1] // transformer.config.patch_size

    # Replicate the stock forward, but skip the block loop
    def tail_only_forward(hidden_states):
        """Only run pos_embed + adaln_single + tail (no blocks)."""
        hidden_states = transformer.pos_embed(hidden_states)

        timestep_emb, embedded_timestep = transformer.adaln_single(
            current_t, added, batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )

        if transformer.caption_projection is not None:
            eh = transformer.caption_projection(prompt_embeds)
            eh = eh.view(batch_size, -1, hidden_states.shape[-1])
        else:
            eh = prompt_embeds

        # ── BLOCKS SKIPPED ──
        # hidden_states stays unchanged except for a tiny add (modeled below)

        # Step counter: advance to a non-zero state so the decision logic
        # doesn't interfere (we're profiling, not caching).

        # Tail (always runs)
        shift, scale = (
            transformer.scale_shift_table[None]
            + embedded_timestep[:, None].to(transformer.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = transformer.norm_out(hidden_states)
        hidden_states = (hidden_states * (1 + scale.to(hidden_states.device))
                         + shift.to(hidden_states.device))
        hidden_states = transformer.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, transformer.config.patch_size,
                   transformer.config.patch_size, transformer.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, transformer.out_channels,
                   height * transformer.config.patch_size,
                   width * transformer.config.patch_size)
        )
        return output

    with FlopCounterMode(display=False) as fcm:
        _ = tail_only_forward(latent_input)
    return fcm.get_total_flops()


def _profile_tail_flops_dit(transformer, latent_input, current_t, class_labels,
                             device, dtype):
    """Run a forward that skips all DiT transformer blocks, counting only the tail."""
    from torch.utils.flop_counter import FlopCounterMode
    from torch.nn.functional import silu

    height = latent_input.shape[-2] // transformer.patch_size
    width = latent_input.shape[-1] // transformer.patch_size

    def tail_only_forward(hidden_states):
        """Only run pos_embed + tail for DiT (no blocks)."""
        hidden_states = transformer.pos_embed(hidden_states)

        # ── BLOCKS SKIPPED ──

        # DiT tail
        conditioning = transformer.transformer_blocks[0].norm1.emb(
            current_t, class_labels, hidden_dtype=hidden_states.dtype)
        shift, scale = transformer.proj_out_1(silu(conditioning)).chunk(2, dim=1)
        hidden_states = transformer.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        hidden_states = transformer.proj_out_2(hidden_states)

        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, transformer.patch_size,
                   transformer.patch_size, transformer.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, transformer.out_channels,
                   height * transformer.patch_size, width * transformer.patch_size)
        )
        return output

    with FlopCounterMode(display=False) as fcm:
        _ = tail_only_forward(latent_input)
    return fcm.get_total_flops()


# ===========================================================================
# Speedup (derived)
# ===========================================================================

class SpeedupMetric(Metric):
    """Aggregates speedup from per-image pair ratios.

    Call ``add_pair(vanilla_s, accel_s)`` for each image pair.
    """

    def __init__(self):
        self._vanilla: list = []
        self._accel: list = []

    def add_pair(self, vanilla_s: float, accel_s: float):
        self._vanilla.append(vanilla_s)
        self._accel.append(accel_s)

    def add(self, image: torch.Tensor, prompt: str = None,
            reference: torch.Tensor = None):
        pass

    def compute(self) -> dict:
        if not self._vanilla:
            return {
                "speedup_cuda": float("nan"),
                "speedup_wall": float("nan"),
            }
        v = np.array(self._vanilla)
        a = np.array(self._accel)
        # Total-time speedup (sum first, then divide)
        speedup_total = v.sum() / a.sum() if a.sum() > 0 else float("nan")
        # Per-image speedup stats
        per = v / a
        return {
            "speedup_total": float(speedup_total),
            "speedup_mean": float(per.mean()),
            "speedup_std": float(per.std()),
        }

    def reset(self):
        self._vanilla.clear()
        self._accel.clear()
