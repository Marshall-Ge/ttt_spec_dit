# -*- coding: utf-8 -*-
"""M6 (Phase 2): AsyncTrainingWorker — 真异步后台训练 worker.

Phase 1 的 ``AsyncTrainer`` 把 ``maybe_train()`` 留在推理循环内同步调用,
训练触发时直接阻塞推理。Phase 2 改成真异步:

  * 后台 daemon 线程独立循环, 检查 buffer → 训练 → 保存 checkpoint;
  * 推理线程只负责往 buffer 里写 events / anchors, **完全不感知训练**;
  * 训练在 ``self._train_model`` (原模型的 fp32 深拷贝) 上进行, 推理用
    原模型 fp16, 两者权重 / dtype / autograd 状态互不干扰;
  * 训练产出的 LoRA checkpoint 写到 ``output_dir``, 推理可在下一轮启动
    时通过 ``find_latest_checkpoint`` 加载 (跨 run 飞轮)。

触发条件 (新):
  基于 **信号质量** 而非 raw count — 至少 ``buffer_ready_min_strata``
  个 (layer, bucket) stratum 各有 ``buffer_ready_min_per_stratum`` 个
  events, 且 anchor 样本数 ≥ ``buffer_ready_min_anchors``。这避免了
  Phase 1 首次触发时 buffer 稀疏选层不可靠的问题 (Phase 2 已挂全部 28
  层所以选层本身消失, 但稀疏 buffer 仍然会让梯度噪声过大)。

容错:
  ``_train_once`` 整段被 ``try/except`` 包裹, 任何异常 (CUDA OOM、
  NaN、tensor shape mismatch…) 都被捕获并记录, **不会传播到推理线程**。
  daemon 线程在主进程退出时自动终止。

Usage::

    worker = AsyncTrainingWorker(transformer, buffer, config, output_dir)
    worker.start()                  # spawn 后台线程
    # ... 推理循环只写 buffer, 不调用 worker 任何方法 ...
    worker.stop()                   # 主进程退出前优雅停止
    ckpt = worker.get_latest_checkpoint()

Backward-compat:
  旧的 ``AsyncTrainer`` 类仍保留 (deprecated), 仅供 ``demo_e2e.py`` /
  ``run_session2_flywheel.py`` 等同步调用 ``maybe_train()`` 的脚本继续
  使用。新代码应一律使用 ``AsyncTrainingWorker``。
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from verification_feedback_loop.config import VFLConfig, DEFAULT_VFL_CONFIG
from verification_feedback_loop.curvature_loss import compute_training_loss
from verification_feedback_loop.lora_adapter import (
    attach_lora,
    attach_lora_all_layers,
    detach_lora,
    save_lora_checkpoint,
    load_lora_checkpoint,
    find_latest_checkpoint,
    get_lora_params,
    freeze_backbone,
    count_lora_params,
)
from verification_feedback_loop.replay_buffer import StratifiedReplayBuffer


# ===========================================================================
# Phase 2: AsyncTrainingWorker
# ===========================================================================


class AsyncTrainingWorker:
    """后台线程驱动的真异步 LoRA 训练 worker.

    与推理线程解耦:
      * 推理线程 → ``buffer.add()`` / ``buffer.add_anchor()`` 写;
      * 训练线程 → ``buffer.sample_training_batch()`` 读, 在
        ``self._train_model`` 上跑 forward/backward, 保存 checkpoint。
    所有共享状态 (buffer) 由 ``StratifiedReplayBuffer`` 自带的锁保护。
    """

    def __init__(self,
                 transformer,
                 buffer: StratifiedReplayBuffer,
                 config: VFLConfig = DEFAULT_VFL_CONFIG,
                 output_dir: str = "./output/vfl_checkpoints",
                 base_model_version: str = "unknown",
                 train_batch_size: int = 16):
        """
        Parameters
        ----------
        transformer : nn.Module
            推理用的 transformer (通常 fp16, 在 GPU 上)。会立刻被
            ``deepcopy`` 一份并转 fp32 作为训练副本 — 原模型不受影响。
        buffer : StratifiedReplayBuffer
            推理线程写、训练线程读的共享 buffer (已自带锁)。
        config : VFLConfig
            训练 / 触发配置。
        output_dir : str
            checkpoint 输出目录。
        base_model_version : str
            用作 checkpoint 元数据, 标识训练时的 base 模型版本。
        train_batch_size : int
            单次训练周期从 buffer 采样的 batch 大小。
        """
        self.buffer = buffer
        self.config = config
        self.output_dir = output_dir
        self.base_model_version = base_model_version
        self.train_batch_size = train_batch_size
        os.makedirs(output_dir, exist_ok=True)

        # ---- 1. Keep a reference to the inference model; deepcopy is LAZY ----
        # We defer ``copy.deepcopy(transformer)`` + ``.float()`` + LoRA attach
        # to the first ``_train_once()`` call.  In many benchmark runs the
        # buffer never reaches the readiness threshold, so the ~2.7 GB fp32
        # copy would be pure waste.  When training DOES fire the copy happens
        # inside the background thread — inference continues unaffected on the
        # original fp16 model.
        if not getattr(transformer, "transformer_blocks", None):
            raise ValueError(
                "AsyncTrainingWorker expects a transformer with "
                "`.transformer_blocks` (DiT/PixArt). Got: "
                f"{type(transformer).__name__}")
        self._inference_model = transformer
        self._train_model: Optional[torch.nn.Module] = None
        self._layer_wrappers: Optional[Dict[int, Any]] = None
        self._model_ready: bool = False

        # ---- 2. 训练状态 ----
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._candidate_version: int = 0
        self._total_updates: int = 0
        self._train_step: int = 0
        self._loss_history: List[float] = []

        # ---- 4. 后台线程控制 ----
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._latest_checkpoint_path: Optional[str] = None
        # 训练线程 crash 不传染推理 — 用 _last_error 暴露给主线程做诊断
        self._last_error: Optional[str] = None
        self._crash_count: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """启动后台训练线程 (daemon=True, 主进程退出时自动终止)。"""
        if self._thread is not None and self._thread.is_alive():
            print("[VFL:AsyncTrainingWorker] thread already running — skip")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._train_loop,
            name="vfl-async-trainer",
            daemon=True,
        )
        self._thread.start()
        print(f"[VFL:AsyncTrainingWorker] background thread started "
              f"(poll_interval={self.config.poll_interval_s}s)")

    def stop(self, timeout: float = 30.0):
        """请求停止后台线程, 等待当前训练周期完成 (最多 timeout 秒)。

        不会强行中断训练 — 设 ``timeout`` 大于一次 ``_train_once`` 的预期
        耗时即可优雅退出。超时后线程仍在跑 (daemon, 主进程退出时被杀)。
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                print(f"[VFL:AsyncTrainingWorker] stop timeout ({timeout}s) — "
                      f"thread still running, will be killed on process exit")
            else:
                print("[VFL:AsyncTrainingWorker] background thread joined")
        self._thread = None

    # ------------------------------------------------------------------
    # Public introspection (主线程可调用)
    # ------------------------------------------------------------------

    def get_latest_checkpoint(self) -> Optional[str]:
        """返回最近一次成功训练周期产出的 checkpoint 路径 (或 None)。"""
        return self._latest_checkpoint_path

    @property
    def total_updates(self) -> int:
        """返回已完成的成功训练周期数。"""
        return self._total_updates

    @property
    def total_train_steps(self) -> int:
        """返回累计梯度更新步数 (跨周期累计)。"""
        return self._train_step

    @property
    def crash_count(self) -> int:
        """训练线程中捕获的异常次数 (诊断用)。"""
        return self._crash_count

    @property
    def last_error(self) -> Optional[str]:
        """最近一次训练异常的字符串描述 (None 表示无异常或尚未触发)。"""
        return self._last_error

    def get_status(self) -> Dict:
        """Snapshot 状态字典, 给主线程做日志/聚合。"""
        return {
            "train_step": self._train_step,
            "candidate_version": self._candidate_version,
            "total_updates": self._total_updates,
            "crash_count": self._crash_count,
            "last_error": self._last_error,
            "buffer_samples": self.buffer.total_samples,
            "loss_history_len": len(self._loss_history),
            "loss_last_10": (self._loss_history[-10:]
                             if len(self._loss_history) >= 10
                             else self._loss_history),
            "latest_checkpoint": self._latest_checkpoint_path,
            "thread_alive": (self._thread is not None
                             and self._thread.is_alive()),
        }

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _train_loop(self):
        """后台线程主循环: poll buffer → 训练 → 休眠 → 直到 stop()。"""
        while not self._stop_event.is_set():
            try:
                if self._buffer_ready():
                    ckpt = self._train_once()
                    if ckpt:
                        self._latest_checkpoint_path = ckpt
            except Exception as e:
                # 任何异常都吞掉 — 训练线程 crash 不能影响推理。
                self._crash_count += 1
                self._last_error = f"{type(e).__name__}: {e}"
                print(f"[VFL:AsyncTrainingWorker] _train_once crashed "
                      f"(#{self._crash_count}): {e}")
                traceback.print_exc()
            # 用 wait 而不是 sleep, 这样 stop() 能立刻唤醒
            self._stop_event.wait(self.config.poll_interval_s)

        print("[VFL:AsyncTrainingWorker] loop exiting (stop requested)")

    # ------------------------------------------------------------------
    # Lazy model init — deepcopy only on first training trigger
    # ------------------------------------------------------------------

    def _ensure_train_model(self):
        """Deep-copy the inference model and attach LoRA (once).

        Called lazily from ``_train_once`` so that benchmark runs where the
        buffer never crosses the readiness threshold pay zero GPU-memory cost
        for the training copy (~2.7 GB fp32 for DiT-2-256).
        """
        if self._model_ready:
            return
        self._train_model = copy.deepcopy(self._inference_model)
        self._train_model.float()
        self._train_model.train()

        self._layer_wrappers = attach_lora_all_layers(
            self._train_model,
            rank=self.config.loRA_rank,
            alpha=self.config.loRA_alpha,
        )
        freeze_backbone(self._train_model)

        n_params = count_lora_params(self._layer_wrappers)
        n_layers = len(self._layer_wrappers)
        print(f"[VFL:AsyncTrainingWorker] lazy init: "
              f"train_model=fp32 deepcopy, LoRA on {n_layers} layers "
              f"({n_params:,} params, rank={self.config.loRA_rank})")
        self._model_ready = True

    # ------------------------------------------------------------------
    # Buffer readiness — 信号质量而非 raw count
    # ------------------------------------------------------------------

    def _buffer_ready(self) -> bool:
        """基于信号质量判断是否应该训练。

        条件:
          1. 至少 ``buffer_ready_min_strata`` 个 stratum 各有
             ``buffer_ready_min_per_stratum`` 个 events
             (避免稀疏 buffer 噪声梯度);
          2. anchor 样本 ≥ ``buffer_ready_min_anchors``
             (L_anchor 没数据时退化为纯 supervised, 容易过拟合到 reject)。
        """
        cfg = self.config
        stats = self.buffer.stats()

        strata = stats.get("strata", {})
        strata_ready = sum(
            1 for s in strata.values()
            if s.get("size", 0) >= cfg.buffer_ready_min_per_stratum
        )
        anchors_ok = stats.get("total_anchors", 0) >= cfg.buffer_ready_min_anchors

        ready = (strata_ready >= cfg.buffer_ready_min_strata) and anchors_ok
        if ready:
            print(f"[VFL:AsyncTrainingWorker] buffer ready: "
                  f"{strata_ready}/{cfg.buffer_ready_min_strata} strata, "
                  f"{stats.get('total_anchors', 0)} anchors")
        return ready

    # ------------------------------------------------------------------
    # One training cycle
    # ------------------------------------------------------------------

    def _train_once(self) -> Optional[str]:
        """一次完整的训练周期: sample → M 步梯度 → save checkpoint.

        返回 checkpoint 路径 (成功) 或 None (跳过 / 无可用样本)。
        异常由调用方 ``_train_loop`` 的 try/except 兜底。
        """
        # ---- 0. Lazy init: deepcopy + LoRA attach on first trigger ----
        self._ensure_train_model()

        t0 = time.time()
        cfg = self.config

        # ---- 1. 采样训练 batch ----
        events, anchors = self.buffer.sample_training_batch(
            self.train_batch_size, ratio=cfg.batch_ratio)

        # 只保留带 replay context 的 events (compute_training_loss 的硬要求)
        usable_events = [e for e in events
                         if getattr(e, "latent_input", None) is not None]
        if len(usable_events) < 4:
            print(f"  [VFL:AsyncTrainingWorker] skip: only "
                  f"{len(usable_events)}/{len(events)} events have "
                  f"latent_input (need ≥4)")
            return None

        # ---- 2. 初始化 optimizer (lazy, 等 LoRA 挂好后) ----
        if self._optimizer is None:
            lora_params = get_lora_params(self._train_model)
            if not lora_params:
                print("  [VFL:AsyncTrainingWorker] no LoRA params — skip")
                return None
            self._optimizer = torch.optim.AdamW(lora_params, lr=1e-4)

        # ---- 3. M 步梯度更新 ----
        losses: List[float] = []
        for _ in range(cfg.trainer_steps_per_trigger):
            self._optimizer.zero_grad(set_to_none=True)
            loss = compute_training_loss(
                self._train_model,
                curvature_events=usable_events,
                anchor_samples=anchors if anchors else None,
                lambda_curvature=cfg.lambda_curvature,
                curvature_order=cfg.curvature_order,
            )
            if not torch.isfinite(loss) or loss.item() <= 0:
                # 零 loss (e.g. 所有 events shape mismatch) → 跳过 step 但
                # 不算 crash
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                get_lora_params(self._train_model), max_norm=1.0)
            self._optimizer.step()
            losses.append(float(loss.detach().item()))
            self._train_step += 1

        if not losses:
            print("  [VFL:AsyncTrainingWorker] no productive gradient steps "
                  "this cycle — skip checkpoint save")
            return None

        # ---- 4. 保存 checkpoint ----
        self._candidate_version += 1
        ckpt_path = os.path.join(
            self.output_dir,
            f"lora_candidate_v{self._candidate_version:03d}.pt",
        )
        save_lora_checkpoint(
            self._layer_wrappers, ckpt_path,
            version=f"candidate_v{self._candidate_version:03d}",
            base_model_version=self.base_model_version,
            metadata={
                "train_step": self._train_step,
                "buffer_samples": self.buffer.total_samples,
                "loss_mean": float(sum(losses) / max(1, len(losses))),
                "loss_last": losses[-1],
                "attached_layers": sorted(self._layer_wrappers.keys()),
                "lambda_curvature": cfg.lambda_curvature,
                "phase": 2,
            },
        )

        # ---- 5. 更新统计 ----
        self._total_updates += 1
        self._loss_history.extend(losses)

        elapsed = time.time() - t0
        print(f"  [VFL:AsyncTrainingWorker] cycle #{self._candidate_version} "
              f"done in {elapsed:.1f}s: "
              f"loss_mean={sum(losses)/len(losses):.6f}, "
              f"steps={len(losses)} → {ckpt_path}")

        # 顺便写一个 summary.json 方便离线分析
        try:
            summary_path = ckpt_path.replace(".pt", "_summary.json")
            with open(summary_path, "w") as f:
                json.dump({
                    "candidate_version": self._candidate_version,
                    "checkpoint_path": ckpt_path,
                    "train_step": self._train_step,
                    "loss_mean": float(sum(losses) / len(losses)),
                    "loss_last": losses[-1],
                    "elapsed_s": elapsed,
                    "buffer_samples": self.buffer.total_samples,
                    "crash_count": self._crash_count,
                }, f, indent=2)
        except OSError:
            pass

        return ckpt_path


# ===========================================================================
# Backward-compat: legacy AsyncTrainer (deprecated)
# ===========================================================================
#
# 保留旧 API 仅供 demo_e2e.py / run_session2_flywheel.py 等同步调用方继续
# 使用。新代码请使用 AsyncTrainingWorker。maybe_train() 仍是同步触发, 不会
# spawn 线程; layer 选择走老的 select_top_k_layers 路径。


from verification_feedback_loop.lora_adapter import select_top_k_layers  # noqa: E402


class AsyncTrainer:
    """Deprecated synchronous trainer (Phase 1 API).

    Retained for backward compatibility with scripts that call
    ``maybe_train()`` from the inference loop. New code should use
    ``AsyncTrainingWorker`` instead.
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

        self._last_train_time: float = 0.0
        self._last_buffer_sample_count: int = 0
        self._train_step: int = 0
        self._candidate_version: int = 0
        self._layer_wrappers: Optional[Dict[int, Any]] = None
        self._attached_layers: List[int] = []
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._loss_history: List[float] = []
        self._layers_selected: bool = False

        os.makedirs(output_dir, exist_ok=True)

    def should_train(self) -> Tuple[bool, str]:
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

    def maybe_train(self) -> Optional[Dict]:
        should, reason = self.should_train()
        if not should:
            return None
        print(f"\n[VFL:AsyncTrainer] Triggered: {reason}")
        return self._do_train()

    def _do_train(self) -> Dict:
        t0 = time.time()
        cfg = self.config
        orig_dtype = next(self.transformer.parameters()).dtype
        self.transformer.float()

        if not self._layers_selected:
            self._select_and_attach_layers()

        batch_size = 16
        events, anchors = self.buffer.sample_training_batch(
            batch_size, ratio=cfg.batch_ratio)
        usable_events = [e for e in events
                         if getattr(e, "latent_input", None) is not None]
        if len(usable_events) < 4:
            print(f"  [VFL:AsyncTrainer] Insufficient usable events "
                  f"({len(usable_events)}/{len(events)} have latent_input), skipping")
            if orig_dtype == torch.float16:
                self.transformer.half()
            return {"status": "skipped",
                    "reason": "insufficient_usable_events",
                    "total_events": len(events),
                    "usable_events": len(usable_events)}

        if self._optimizer is None:
            lora_params = get_lora_params(self.transformer)
            self._optimizer = torch.optim.AdamW(lora_params, lr=1e-4)

        losses = []
        for step in range(cfg.trainer_steps_per_trigger):
            self._optimizer.zero_grad(set_to_none=True)
            loss_scaled = compute_training_loss(
                self.transformer,
                curvature_events=usable_events,
                anchor_samples=anchors if anchors else None,
                lambda_curvature=cfg.lambda_curvature,
                curvature_order=cfg.curvature_order,
            )
            if torch.isfinite(loss_scaled) and loss_scaled.item() > 0:
                loss_scaled.backward()
                torch.nn.utils.clip_grad_norm_(
                    get_lora_params(self.transformer), max_norm=1.0)
                self._optimizer.step()
            losses.append(float(loss_scaled.detach().item()))
            self._train_step += 1

        if orig_dtype == torch.float16:
            self.transformer.half()

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
              f"layers={self._attached_layers}, → {ckpt_path}")
        try:
            with open(ckpt_path.replace(".pt", "_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
        except OSError:
            pass
        return summary

    def _select_and_attach_layers(self):
        cfg = self.config
        top_k = select_top_k_layers(self.buffer, k=cfg.top_k_layers)
        if len(top_k) == 0:
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

    def load_checkpoint(self, checkpoint_path: str):
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
