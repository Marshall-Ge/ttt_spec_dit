# -*- coding: utf-8 -*-
"""M3: Stratified Replay Buffer — 分层回放缓冲区.

按 (layer_id, timestep_bucket) 分 stratum, 每个 stratum 用固定容量的
reservoir sampling ring buffer。

三类样本按配置比例存取:
  - hard_negative : reject 事件 (全部记录)
  - normal        : accept 事件 (低采样率记录)
  - anchor        : 独立于 verification 机制之外的 anchor 样本
                     (真实 prompt + 标准 diffusion loss target),
                     用于在训练 L3 adapter 时锚定生成质量, 防止 catastrophic forgetting

设计要点:
  1. Reservoir sampling 保证每个 stratum 内长期均匀采样
  2. 三类样本同一 stratum 共享总容量, 按比例抽样
  3. Anchor 样本不参与 per-stratum 容量限制 (独立存储, 通常很小)
"""

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


# ===========================================================================
# Anchor sample
# ===========================================================================


@dataclass
class AnchorSample:
    """标准 diffusion loss 锚定样本。

    独立于 verification 机制之外, 定期从真实数据流采样。
    用于 L3 训练中的 L_diffusion anchor loss, 防止模型 "拉懒"。
    """

    prompt: Any                    # prompt / class label (模型输入)
    latent: torch.Tensor           # 噪声 latent (B, C, H, W)
    timestep: torch.Tensor         # 采样的 timestep (B,)
    target: torch.Tensor           # diffusion target — v_pred 或 noise target (B, C, H, W)
    model: str = ""                # "dit" | "pixart"
    base_model_version: str = ""   # base model 版本标识

    def detach(self) -> "AnchorSample":
        """返回所有 tensor detached 的副本 (安全存入 buffer)。"""
        return AnchorSample(
            prompt=self.prompt,
            latent=self.latent.detach().cpu(),
            timestep=self.timestep.detach().cpu(),
            target=self.target.detach().cpu(),
            model=self.model,
            base_model_version=self.base_model_version,
        )


# ===========================================================================
# Stratum — single (layer_id, timestep_bucket) ring buffer
# ===========================================================================


class _Stratum:
    """单个 stratum 的 reservoir sampling ring buffer。

    不区分样本类型 — 所有三类样本存入同一个 ring, 带 kind 标签。
    采样时按 ratio 分别从三类中抽取。
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self._buffer: List[Tuple[Any, str]] = []  # [(sample, kind), ...]
        self._total_seen: Dict[str, int] = defaultdict(int)  # per-kind 累计看到数
        self._wraps: int = 0  # 绕回次数 (ring full 后 +1 每次绕回)

    def add(self, sample: Any, kind: str) -> None:
        """Reservoir sampling 写入。"""
        self._total_seen[kind] += 1
        total = sum(self._total_seen.values())

        if len(self._buffer) < self.capacity:
            self._buffer.append((sample, kind))
        else:
            # Reservoir sampling: 以 capacity/total 概率替换随机位置
            idx = random.randint(0, total - 1)
            if idx < self.capacity:
                self._buffer[idx] = (sample, kind)
                if idx == 0:
                    self._wraps += 1

    def sample(self, n: int, ratio: Dict[str, float]) -> List[Any]:
        """按 ratio 从三类中各采 n_i 个样本。

        返回按 ratio 比例混合的样本列表 (不带 kind 标签)。
        """
        # 按 kind 分桶
        by_kind: Dict[str, List[Any]] = defaultdict(list)
        for sample, kind in self._buffer:
            by_kind[kind].append(sample)

        result: List[Any] = []
        for kind, r in ratio.items():
            pool = by_kind.get(kind, [])
            n_i = max(0, int(n * r))
            if pool and n_i > 0:
                result.extend(random.choices(pool, k=min(n_i, len(pool))))

        # 如果某种类样本不足, 用其他类补齐到 n
        if len(result) < n and by_kind:
            all_pool = [s for s, _ in self._buffer]
            remaining = n - len(result)
            result.extend(random.choices(all_pool, k=min(remaining, len(all_pool))))

        random.shuffle(result)
        return result[:n]

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        return len(self._buffer) == 0

    def stats(self) -> Dict:
        by_kind = defaultdict(int)
        for _, kind in self._buffer:
            by_kind[kind] += 1
        return {
            "size": self.size,
            "capacity": self.capacity,
            "wraps": self._wraps,
            "by_kind": dict(by_kind),
            "total_seen": dict(self._total_seen),
        }


# ===========================================================================
# StratifiedReplayBuffer
# ===========================================================================


class StratifiedReplayBuffer:
    """按 (layer_id, timestep_bucket) 分层的回放缓冲区。

    每 stratum 独立 reservoir sampling, anchor 样本全局存储。
    """

    def __init__(self,
                 capacity_per_stratum: int = 1000,
                 num_layers: int = 28,
                 num_timestep_buckets: int = 3,
                 batch_ratio: Optional[Dict[str, float]] = None):
        """
        Parameters
        ----------
        capacity_per_stratum : int
            每个 (layer, bucket) stratum 的最大容量。
        num_layers : int
            Transformer block 数量 (DiT: 28, PixArt: 28)。
        num_timestep_buckets : int
            时间桶数量 (默认 3: early/mid/late)。
        batch_ratio : dict
            训练采样比例 {"hard_negative": 0.5, "normal": 0.3, "anchor": 0.2}。
        """
        self.capacity_per_stratum = capacity_per_stratum
        self.num_layers = num_layers
        self.num_timestep_buckets = num_timestep_buckets
        self.batch_ratio = batch_ratio or {
            "hard_negative": 0.5, "normal": 0.3, "anchor": 0.2,
        }

        # (layer_id, timestep_bucket) → _Stratum
        self._strata: Dict[Tuple[int, int], _Stratum] = {}
        self._anchor_samples: List[AnchorSample] = []

        # 累计统计
        self._total_added: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Stratum key
    # ------------------------------------------------------------------

    def _key(self, layer_id: int, timestep_bucket: int) -> Tuple[int, int]:
        return (layer_id, timestep_bucket)

    def _get_or_create_stratum(self, layer_id: int,
                                timestep_bucket: int) -> _Stratum:
        key = self._key(layer_id, timestep_bucket)
        if key not in self._strata:
            self._strata[key] = _Stratum(capacity=self.capacity_per_stratum)
        return self._strata[key]

    # ------------------------------------------------------------------
    # Add
    # ------------------------------------------------------------------

    def add(self, event, kind: str) -> None:
        """存入一个 verification event。

        Parameters
        ----------
        event : VerificationEvent
        kind : str
            "hard_negative" (reject 事件) 或 "normal" (accept 低采样)。
        """
        stratum = self._get_or_create_stratum(
            event.layer_id, event.timestep_bucket)
        stratum.add(event, kind)
        self._total_added[kind] += 1

    def add_anchor(self, sample: AnchorSample) -> None:
        """存入一个 anchor 样本 (独立存储, 不限容量)。"""
        self._anchor_samples.append(sample.detach())
        # 容量保护: 保留最近 500 个
        if len(self._anchor_samples) > 500:
            self._anchor_samples = self._anchor_samples[-500:]

    def add_anchor_from_tensors(self, prompt, latent, timestep, target,
                                 model="dit", base_model_version="unknown"):
        """从 denoising loop 中收集 anchor 样本 (便捷方法)。

        Parameters
        ----------
        prompt : class label or text prompt
        latent : (B, C, H, W) noisy latent input
        timestep : (B,) timestep tensor
        target : (B, C, H, W) noise prediction target (model output)
        """
        sample = AnchorSample(
            prompt=prompt,
            latent=latent.detach().cpu(),
            timestep=timestep.detach().cpu(),
            target=target.detach().cpu(),
            model=model,
            base_model_version=base_model_version,
        )
        self.add_anchor(sample)

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------

    def sample_training_batch(self,
                               batch_size: int,
                               ratio: Optional[Dict[str, float]] = None
                               ) -> Tuple[List, List[AnchorSample]]:
        """按比例从各 stratum 采样训练 batch。

        Parameters
        ----------
        batch_size : int
            总 batch 大小 (包含所有三类样本)。
        ratio : dict, optional
            覆盖默认采样比例。

        Returns
        -------
        (events, anchors) : (list of VerificationEvent, list of AnchorSample)
            events 包含 hard_negative 和 normal 两类。
            anchors 是独立的 anchor 样本列表。
        """
        ratio = ratio or self.batch_ratio

        # 计算各类配额
        n_hard = max(1, int(batch_size * ratio.get("hard_negative", 0.5)))
        n_normal = max(1, int(batch_size * ratio.get("normal", 0.3)))
        n_anchor = max(1, int(batch_size * ratio.get("anchor", 0.2)))
        # 调整使总和 = batch_size
        total_alloc = n_hard + n_normal + n_anchor
        if total_alloc != batch_size:
            n_hard += batch_size - total_alloc

        # 从有数据的 strata 中均匀采样
        events = []
        hard_pools = [(k, s) for k, s in self._strata.items()
                       if any(kind == "hard_negative" for _, kind in s._buffer)]
        normal_pools = [(k, s) for k, s in self._strata.items()
                         if any(kind == "normal" for _, kind in s._buffer)]

        # hard_negative
        if hard_pools:
            per_stratum = max(1, n_hard // len(hard_pools))
            for _, s in hard_pools:
                events.extend(s.sample(per_stratum,
                                       {"hard_negative": 1.0, "normal": 0.0, "anchor": 0.0}))

        # normal
        if normal_pools:
            per_stratum = max(1, n_normal // len(normal_pools))
            for _, s in normal_pools:
                events.extend(s.sample(per_stratum,
                                       {"hard_negative": 0.0, "normal": 1.0, "anchor": 0.0}))

        # anchor — 独立采样
        anchors = []
        if self._anchor_samples:
            anchors = random.choices(
                self._anchor_samples,
                k=min(n_anchor, len(self._anchor_samples)),
            )

        return events, anchors

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        """全缓冲区统计快照。"""
        strata_stats = {}
        for (layer_id, bucket), s in self._strata.items():
            strata_stats[f"L{layer_id}_B{bucket}"] = s.stats()

        # 按 layer 聚合
        per_layer_hard: Dict[int, int] = defaultdict(int)
        per_layer_normal: Dict[int, int] = defaultdict(int)
        for (layer_id, bucket), s in self._strata.items():
            for kind, cnt in s.stats()["by_kind"].items():
                if kind == "hard_negative":
                    per_layer_hard[layer_id] += cnt
                elif kind == "normal":
                    per_layer_normal[layer_id] += cnt

        return {
            "num_strata": len(self._strata),
            "num_strata_nonempty": sum(1 for s in self._strata.values() if not s.is_empty),
            "total_samples": sum(s.size for s in self._strata.values()),
            "total_anchors": len(self._anchor_samples),
            "total_added": dict(self._total_added),
            "per_layer_hard_negative": dict(sorted(per_layer_hard.items())),
            "per_layer_normal": dict(sorted(per_layer_normal.items())),
            "strata": strata_stats,
        }

    @property
    def total_samples(self) -> int:
        return sum(s.size for s in self._strata.values())
