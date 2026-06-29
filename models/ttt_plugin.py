# -*- coding: utf-8 -*-
"""Session-Level Test-Time Training (TTT) plugin for DiT-2-256.

Phase 3 architectural pivot: instead of single-image residual prediction we
attach a microscopic, *persistent* plugin ($\\phi$) to the frozen DiT backbone
($\\Theta$, $\\nabla_\\Theta = 0$). During a "session" — a sequential stream of
semantically-related images that share the same ImageNet class but have
independent Gaussian noise initializations $z_0^{(i)}$ — the plugin acts as an
online student:

  * **calc step (Teacher Mode)**: the full 28-block backbone yields the ground
    truth $Z_{\\text{true}}$. The stale cached state is passed through the plugin
    to give $Z_{\\text{pred}}$. We minimise $\\mathcal{L}_{\\text{TTT}} =
    \\mathrm{MSE}(Z_{\\text{pred}}, Z_{\\text{true}})$ with exactly one AdamW
    step, updating *only* $\\phi$.
  * **skip step (Student/Inference Mode)**: the 28 blocks are bypassed and
    $Z_{\\text{cached}}$ is routed through the plugin to modulate the stale cache.

As the session progresses, $\\phi$ learns the class-specific semantic manifold,
enabling the "Flywheel Effect": later images can sustain extreme skip ratios
(>80%) without fidelity collapse.

This module is dual-parametric and **zero-initialised** so that Image 1, step 0
of the plugin is exactly the static TeaCache baseline:

.. math::
    Z_{\\text{out}} = Z_{\\text{cached}} \\odot (1 + \\Delta\\gamma_{\\phi}(t))
                       + \\Delta\\beta_{\\phi}(t)

with $\\Delta\\gamma, \\Delta\\beta \\equiv 0$ at initialisation.

Backbone freezing (the strict $\\nabla_\\Theta = 0$ constraint) is enforced
explicitly by the session runner via ``requires_grad_(False)`` on the
transformer/VAE — NOT by ``@torch.no_grad`` decorators (which would also
suppress the plugin's gradient).
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


# ===========================================================================
# Persistent plugin — SessionAdaLNModulator
# ===========================================================================

class SessionAdaLNModulator(nn.Module):
    """Lightweight time-conditioned adaLN modulator for session-level TTT.

    Produces per-timestep scale (Δγ) and shift (Δβ) vectors for the hidden
    dimension, conditioned on the stale cached hidden state and the timestep
    embedding (sourced from the DiT block-0 ``norm1.emb`` — the same signal
    the tail projector uses).

    Parameters
    ----------
    hidden_dim : int
        DiT hidden dimension (1152 for DiT-2-256). The cached hidden state has
        shape ``(B, seq_len, hidden_dim)`` with ``seq_len = (32/2)**2 = 256``.
    mid_dim : int
        Bottleneck width. Default 192 keeps total params < 1M.

    Parameter budget (hidden_dim=1152, mid_dim=192)
        fc_t    : 1152·192 + 192   = 221,376
        fc_h_in : 1152·192 + 192   = 221,376
        fc_h    :  192·192 + 192   =  37,056   (SiLU-activated)
        fc_out  :  192·2304 + 2304 = 444,672   → chunk into (Δγ, Δβ)
        -----------------------------------------------------------
        total   : ≈ 0.92 M parameters (strictly under the 1 M budget)

    Forward
    -------
    cached_hidden_state : (B, seq_len, hidden_dim)
    timestep_emb        : (B, hidden_dim)   — output of block0.norm1.emb
    Returns
        Z_out : (B, seq_len, hidden_dim)
            = Z_cached ⊙ (1 + Δγ[:, None]) + Δβ[:, None]
    """

    def __init__(self, hidden_dim: int = 1152, mid_dim: int = 192):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mid_dim = mid_dim

        # Time-conditioning MLP: project the timestep embedding into the
        # bottleneck space, then mix with the cached-state summary.
        self.fc_t = nn.Linear(hidden_dim, mid_dim)
        # Project the pooled cached state (hidden_dim) into mid_dim so it can
        # be added to the timestep branch inside the bottleneck.
        self.fc_h_in = nn.Linear(hidden_dim, mid_dim)
        self.fc_h = nn.Linear(mid_dim, mid_dim)
        # Final projection → 2·hidden_dim (Δγ and Δβ concatenated).
        self.fc_out = nn.Linear(mid_dim, 2 * hidden_dim)

        self.act = nn.SiLU()

        self._init_zero_output()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_zero_output(self):
        """Initialise fc_out weights AND bias to strictly 0.0.

        This guarantees an Identity mapping at step 0 of Image 1: the plugin
        emits Δγ = Δβ = 0, so Z_out == Z_cached, i.e. the system starts exactly
        at the static TeaCache baseline. The upstream fc_t / fc_h layers are
        left at PyTorch defaults so they become expressive immediately once
        gradients flow.
        """
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self,
                cached_hidden_state: torch.Tensor,
                timestep_emb: torch.Tensor) -> torch.Tensor:
        """Time-conditioned scale/shift modulation of the cached state.

        Parameters
        ----------
        cached_hidden_state : (B, seq_len, hidden_dim)
            The stale hidden state at the TeaCache residual-application site
            (pos_embed output of DiT, i.e. ``hidden + previous_residual``).
        timestep_emb : (B, hidden_dim)
            Timestep embedding from ``transformer_blocks[0].norm1.emb``.

        Returns
        -------
        modulated : (B, seq_len, hidden_dim)
        """
        # Pool the cached state across the sequence dimension (mean) so the
        # modulator produces per-token scale/shift that is globally informed
        # by the current sample, not per-token (keeps params tiny).
        pooled = cached_hidden_state.mean(dim=1)        # (B, hidden_dim)

        t = self.act(self.fc_t(timestep_emb))           # (B, mid_dim)
        h_in = self.fc_h_in(pooled)                     # (B, mid_dim)
        h = self.act(self.fc_h(t + h_in))               # (B, mid_dim)  — mix
        delta = self.fc_out(h)                          # (B, 2·hidden_dim)

        delta_gamma, delta_beta = delta.chunk(2, dim=1) # each (B, hidden_dim)

        # Broadcast over the sequence dimension (B, 1, hidden_dim).
        out = (cached_hidden_state * (1.0 + delta_gamma[:, None])
               + delta_beta[:, None])
        return out

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def num_parameters(self) -> int:
        """Total trainable parameter count (excluding buffers)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# Pure-function state API (plain-dict state, mirrors accelerators/ style)
# ===========================================================================
#
# The state dict is owned by the session runner (continual_inference_runner.py)
# and threaded through the model forward + denoise loop. The model's forward is
# agnostic to the TTT training mechanics — it only reads ``plugin`` and writes
# ``z_pred`` / ``z_true``; the *loop* calls ``ttt_train_step``.

def ttt_state_init(num_steps: int,
                   plugin: SessionAdaLNModulator,
                   lr: float = 1e-4,
                   micro_epochs: int = 3) -> Dict:
    """Allocate the persistent TTT state dict.

    The AdamW optimizer is constructed here with the plugin's parameters ONLY
    (the backbone is frozen at the runner level via ``requires_grad_(False)``).
    Both the plugin weights and the optimizer momentum persist across images;
    they are deliberately NOT reset between images.

    Parameters
    ----------
    num_steps : int
        Denoising step count for a single image generation.
    plugin : SessionAdaLNModulator
        The plugin module to optimise.
    lr : float
        AdamW learning rate.
    micro_epochs : int
        Per calc-step micro-epoch count (sample-starvation fix). The expensive
        28-block teacher signal is reused this many times for plugin training.
        Default 3 gives ~75 effective updates during burn-in (5 imgs × 5 calc
        steps × 3 me). 1 = single-pass legacy behaviour.

    Returns
    -------
    state : dict
        Plain dict carrying the plugin, optimizer, and per-image telemetry.
    """
    optimizer = torch.optim.AdamW(plugin.parameters(), lr=lr)

    return {
        # ---- config ----
        "num_steps": num_steps,
        "lr": lr,
        "micro_epochs": micro_epochs,

        # ---- persistent learned objects (NEVER reset across images) ----
        "plugin": plugin,
        "optimizer": optimizer,

        # ---- per-image telemetry (reset every image via ttt_reset_for_image) ----
        "losses": [],          # list[float] — per-calc-step loss this image
        "n_calc": 0,           # calc steps this image
        "n_skip": 0,           # skip steps this image

        # ---- transient handoff between forward() and the loop ----
        # Forward writes (z_pred, z_true) here on calc steps; the loop reads
        # them in ttt_train_step and clears them immediately.
        "z_pred": None,
        "z_true": None,

        # ---- cumulative training telemetry ----
        "trained_steps": 0,    # total optimizer steps across the whole session
        "session_losses": [],  # running list of every step's loss (all images)
    }


def ttt_train_step(state: Dict) -> float:
    """Execute exactly one AdamW step on the plugin, using the (z_pred, z_true)
    stashed by the model's forward on the just-completed calc step.

    The teacher target z_true was produced under ``torch.no_grad`` and is
    already detached; only the plugin's graph (z_pred) carries gradient. After
    the step, gradients are cleared and the stashed tensors are released.

    Parameters
    ----------
    state : dict
        TTT state (from ``ttt_state_init``).

    Returns
    -------
    loss_value : float
        The MSE loss before the optimizer step (for telemetry).
    """
    z_pred = state["z_pred"]
    z_true = state["z_true"]
    if z_pred is None or z_true is None:
        return 0.0

    # MSE between student prediction and frozen-teacher ground truth.
    loss = torch.nn.functional.mse_loss(z_pred, z_true.detach())

    optimizer = state["optimizer"]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    loss_value = float(loss.detach().item())

    # Telemetry
    state["losses"].append(loss_value)
    state["session_losses"].append(loss_value)
    state["n_calc"] += 1
    state["trained_steps"] += 1

    # Release the transient tensors.
    state["z_pred"] = None
    state["z_true"] = None
    return loss_value


def ttt_record_skip(state: Dict) -> None:
    """Increment the per-image skip counter (no training on skip steps)."""
    state["n_skip"] += 1


def ttt_reset_for_image(state: Dict) -> None:
    """Reset per-image telemetry for a new image, keeping plugin weights and
    optimizer momentum intact (the persistence that drives the Flywheel Effect).
    """
    state["losses"] = []
    state["n_calc"] = 0
    state["n_skip"] = 0
    state["z_pred"] = None
    state["z_true"] = None


def ttt_avg_loss(state: Dict) -> float:
    """Mean plugin loss over the current image's calc steps (NaN if none)."""
    if not state["losses"]:
        return float("nan")
    return float(sum(state["losses"]) / len(state["losses"]))


def ttt_skip_ratio(state: Dict) -> float:
    """Skip ratio (%) over the current image's steps."""
    total = state["n_calc"] + state["n_skip"]
    if total == 0:
        return 0.0
    return 100.0 * state["n_skip"] / total


def ttt_session_stats(state: Dict) -> Dict:
    """Aggregate statistics over the entire session so far."""
    return {
        "trained_steps": state["trained_steps"],
        "session_loss_mean": (float(sum(state["session_losses"])
                                 / len(state["session_losses"]))
                             if state["session_losses"] else float("nan")),
        "session_loss_first": (state["session_losses"][0]
                               if state["session_losses"] else float("nan")),
        "session_loss_last": (state["session_losses"][-1]
                              if state["session_losses"] else float("nan")),
        "plugin_params": state["plugin"].num_parameters(),
    }
