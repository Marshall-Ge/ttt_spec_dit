# -*- coding: utf-8 -*-
"""M6: Async Trainer — 异步训练 worker.

触发条件 (先满足者触发):
  * buffer 新增样本数 >= trigger_min_samples
  * 距上次训练 >= trigger_min_interval_s

每次触发:
  1. 从 StratifiedReplayBuffer 按配置比例采 batch
  2. 对 LoRA adapter 参数做 M 步梯度更新
  3. 产出带版本号的 candidate checkpoint
  4. 记录 base_model_version、buffer 快照、loss 曲线

训练不与推理同步 — 在独立调用中执行, 不阻塞 denoising loop。

Usage::

    trainer = AsyncTrainer(transformer, buffer, calibrator, config)
    # Call periodically (e.g. after each batch or image):
    trainer.maybe_train()
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from verification_feedback_loop.config import VFLConfig, DEFAULT_VFL_CONFIG
from verification_feedback_loop.curvature_loss import compute_training_loss
from verification_feedback_loop.lora_adapter import (
    attach_lora,
    detach_lora,
    save_lora_checkpoint,
    get_lora_params,
    freeze_backbone,
    select_top_k_layers,
    count_lora_params,
)
from verification_feedback_loop.replay_buffer import StratifiedReplayBuffer


class AsyncTrainer:
    """Trigger-based async training worker for L3 LoRA adapters.

    Does NOT spawn a real OS thread/process — instead, ``maybe_train()``
    is called synchronously from the inference loop but only triggers
    training when conditions are met, keeping inference latency bounded.
    """

    def __init__(self,
                 transformer,
                 buffer: StratifiedReplayBuffer,
                 config: VFLConfig = DEFAULT_VFL_CONFIG,
                 output_dir: str = "./output/vfl_checkpoints",
                 base_model_version: str = "unknown"):
        self.transformer = transformer
        self.buffer = buffer
        self.config = config
        self.output_dir = output_dir
        self.base_model_version = base_model_version

        # Internal state
        self._last_train_time: float = 0.0
        self._last_buffer_sample_count: int = 0
        self._train_step: int = 0
        self._candidate_version: int = 0
        self._layer_wrappers: Optional[Dict[int, Any]] = None
        self._attached_layers: List[int] = []
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._loss_history: List[float] = []

        # Deferred: layers selected after sufficient collection
        self._layers_selected: bool = False

        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Trigger logic
    # ------------------------------------------------------------------

    def should_train(self) -> Tuple[bool, str]:
        """Check whether training should be triggered now.

        Returns (should, reason).
        """
        current_samples = self.buffer.total_samples
        new_samples = current_samples - self._last_buffer_sample_count

        if new_samples >= self.config.trigger_min_samples:
            return True, f"new_samples={new_samples} >= {self.config.trigger_min_samples}"

        elapsed = time.time() - self._last_train_time
        if (self._last_train_time > 0
                and elapsed >= self.config.trigger_min_interval_s
                and new_samples > 0):
            return True, f"elapsed={elapsed:.0f}s >= {self.config.trigger_min_interval_s}s"

        return False, ""

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def maybe_train(self) -> Optional[Dict]:
        """Check trigger and train if conditions are met.

        Returns training summary dict if training happened, None otherwise.
        """
        should, reason = self.should_train()
        if not should:
            return None

        print(f"\n[VFL:AsyncTrainer] Triggered: {reason}")
        return self._do_train()

    def _do_train(self) -> Dict:
        """Execute one training cycle."""
        t0 = time.time()
        cfg = self.config

        # ---- Step 0: Convert to fp32 for training stability ----
        # The 28-block DiT in fp16 can overflow on random inputs, producing NaN
        # that corrupts LoRA training.  fp32 conversion is safe because training
        # runs asynchronously AFTER all generation (no concurrent inference).
        orig_dtype = next(self.transformer.parameters()).dtype
        self.transformer.float()

        # ---- Step 1: Select layers (first time only) ----
        if not self._layers_selected:
            self._select_and_attach_layers()

        # ---- Step 2: Sample training batch ----
        batch_size = 16  # configurable in future
        events, anchors = self.buffer.sample_training_batch(
            batch_size, ratio=cfg.batch_ratio)

        # Only events carrying replay context (latent_input) can drive the
        # new buffer-driven loss. Older events recorded before the replay
        # context fields existed are skipped by compute_training_loss, but
        # if NONE have it we bail out early to avoid a no-op training cycle.
        usable_events = [e for e in events
                         if getattr(e, "latent_input", None) is not None]

        if len(usable_events) < 4:
            print(f"  [VFL:AsyncTrainer] Insufficient usable events "
                  f"({len(usable_events)}/{len(events)} have latent_input), skipping")
            return {"status": "skipped",
                    "reason": "insufficient_usable_events",
                    "total_events": len(events),
                    "usable_events": len(usable_events)}

        # ---- Step 3: Training loop ----
        if self._optimizer is None:
            lora_params = get_lora_params(self.transformer)
            self._optimizer = torch.optim.AdamW(lora_params, lr=1e-4)

        losses = []
        for step in range(cfg.trainer_steps_per_trigger):
            self._optimizer.zero_grad(set_to_none=True)

            # Buffer-driven loss: supervised MSE on event.true_feature +
            # curvature on per-(sample, layer) trajectories + anchor
            # diffusion loss on real samples. Forward re-runs per event
            # with hooks capturing the LoRA-modified target-layer output.
            loss_scaled = compute_training_loss(
                self.transformer,
                curvature_events=usable_events,
                anchor_samples=anchors if anchors else None,
                lambda_curvature=cfg.lambda_curvature,
                curvature_order=cfg.curvature_order,
            )

            # Backward + step
            if torch.isfinite(loss_scaled) and loss_scaled.item() > 0:
                loss_scaled.backward()
                torch.nn.utils.clip_grad_norm_(
                    get_lora_params(self.transformer), max_norm=1.0)
                self._optimizer.step()

            losses.append(float(loss_scaled.detach().item()))
            self._train_step += 1

        # ---- Restore original dtype so inference stays in fp16 ----
        if orig_dtype == torch.float16:
            self.transformer.half()

        # ---- Step 4: Save candidate checkpoint ----
        self._candidate_version += 1
        ckpt_path = os.path.join(
            self.output_dir,
            f"lora_candidate_v{self._candidate_version:03d}.pt",
        )

        if self._layer_wrappers:
            save_lora_checkpoint(
                self._layer_wrappers, ckpt_path,
                version=f"candidate_v{self._candidate_version:03d}",
                base_model_version=self.base_model_version,
                metadata={
                    "train_step": self._train_step,
                    "buffer_samples": self.buffer.total_samples,
                    "loss_mean": float(sum(losses) / max(1, len(losses))),
                    "loss_last": losses[-1] if losses else 0.0,
                    "attached_layers": self._attached_layers,
                    "lambda_curvature": cfg.lambda_curvature,
                },
            )

        # ---- Step 5: Update state ----
        self._last_train_time = time.time()
        self._last_buffer_sample_count = self.buffer.total_samples
        self._loss_history.extend(losses)

        elapsed = time.time() - t0
        summary = {
            "status": "trained",
            "candidate_version": self._candidate_version,
            "checkpoint_path": ckpt_path,
            "steps": len(losses),
            "loss_mean": float(sum(losses) / max(1, len(losses))),
            "loss_last": losses[-1] if losses else 0.0,
            "elapsed_s": elapsed,
            "buffer_samples": self.buffer.total_samples,
            "attached_layers": self._attached_layers,
        }

        print(f"  [VFL:AsyncTrainer] Done in {elapsed:.1f}s: "
              f"loss={summary['loss_mean']:.6f}, "
              f"layers={self._attached_layers}, "
              f"→ {ckpt_path}")

        # Save summary alongside checkpoint
        summary_path = ckpt_path.replace(".pt", "_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    # ------------------------------------------------------------------
    # Layer selection
    # ------------------------------------------------------------------

    def _select_and_attach_layers(self):
        """Select top-K layers by reject frequency and attach LoRA."""
        cfg = self.config
        top_k = select_top_k_layers(self.buffer, k=cfg.top_k_layers)

        if len(top_k) == 0:
            # Fallback: use layers 18, 20, 24 (empirically high-rejection)
            top_k = [18, 20, 24]
            print(f"  [VFL:AsyncTrainer] No reject data yet, "
                  f"using fallback layers: {top_k}")

        self._attached_layers = top_k
        self._layer_wrappers = attach_lora(
            self.transformer, top_k,
            rank=cfg.loRA_rank, alpha=cfg.loRA_alpha,
        )
        freeze_backbone(self.transformer)

        n_params = count_lora_params(self._layer_wrappers)
        print(f"  [VFL:AsyncTrainer] LoRA attached to layers {top_k}: "
              f"{n_params:,} params (rank={cfg.loRA_rank})")

        self._layers_selected = True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_checkpoint(self, checkpoint_path: str):
        """Load a pre-trained LoRA checkpoint and attach to transformer."""
        from verification_feedback_loop.lora_adapter import load_lora_checkpoint
        wrappers, meta = load_lora_checkpoint(self.transformer, checkpoint_path)
        self._layer_wrappers = wrappers
        self._attached_layers = sorted(wrappers.keys())
        self._layers_selected = True
        freeze_backbone(self.transformer)
        n_params = sum(
            w.lora_A.numel() + w.lora_B.numel()
            for wdict in wrappers.values() for w in wdict.values()
        )
        print(f"  [VFL:AsyncTrainer] Loaded LoRA checkpoint: {checkpoint_path}")
        print(f"    Layers: {self._attached_layers}, {n_params:,} params, "
              f"base_model={meta.get('checkpoint_base_model', '?')}")
        return meta

    def on_base_model_swap(self, new_version: str):
        """Handle base model version change: detach old LoRA, reset selection."""
        if self._layer_wrappers:
            detach_lora(self.transformer, self._layer_wrappers)
            self._layer_wrappers = None

        self._attached_layers = []
        self._layers_selected = False
        self._optimizer = None
        self._candidate_version = 0
        self.base_model_version = new_version

        print(f"  [VFL:AsyncTrainer] Base model swapped to {new_version}, "
              f"LoRA detached, pending re-selection.")

    def get_status(self) -> Dict:
        """Return current trainer status for monitoring."""
        return {
            "train_step": self._train_step,
            "candidate_version": self._candidate_version,
            "layers_selected": self._layers_selected,
            "attached_layers": self._attached_layers,
            "buffer_samples": self.buffer.total_samples,
            "last_train_elapsed_s": (time.time() - self._last_train_time
                                     if self._last_train_time > 0 else -1),
            "loss_history_len": len(self._loss_history),
            "loss_last_10": (self._loss_history[-10:]
                             if len(self._loss_history) >= 10
                             else self._loss_history),
        }
