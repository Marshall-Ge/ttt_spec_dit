# -*- coding: utf-8 -*-
"""Latency, FLOPs, and Speedup metrics.

- LatencyMetric:   wall-clock latency (s) per image, vanilla vs accelerated
- FLOPsMetric:     profiled FLOPs per step, accumulated via TeaCache decisions
- SpeedupMetric:   derived speedup ratio from latency or FLOPs
"""

import numpy as np
import torch

from utils.flops import _profile_tail_flops, _profile_tail_flops_dit
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

    def add_speca_generation(self, full_steps: int, taylor_steps: int,
                             probe_full_blocks: int, num_layers: int):
        """Accumulate SpecA FLOPs using full-block-equivalent accounting."""
        self._profile_once()
        total_steps = full_steps + taylor_steps
        block_flops = ((self._flops_full - self._flops_skip) / num_layers
                       if num_layers > 0 else 0.0)
        self._total_vanilla += total_steps * self._flops_full
        self._total_accel += (
            full_steps * self._flops_full
            + taylor_steps * self._flops_skip
            + probe_full_blocks * block_flops
        )
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
