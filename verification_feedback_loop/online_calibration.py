# -*- coding: utf-8 -*-
"""M2: Online Calibrator — 在线统计量校准 (无梯度, 常驻).

对每个 ``(layer_id, timestep_bucket)`` 维护 **EMA 分位数阈值** —
动态调整拒绝阈值 τ, 使其跟随当前 base model 的真实误差分布,
替代 SpecA 的固定 ``base_threshold * decay^progress``。

RLS-based online rescale 已移除 — 在线拟合出的 proxy_diff→true_error 映射
产出的值太小, 不适合 TeaCache 的 accumulate-vs-threshold 机制。

更新成本极低 (标量运算), 可以直接在 verification hook 同步路径里调用,
不算违反"梯度不同步"约束 (无 backprop, 只是统计量更新)。

Interface::

    cal = OnlineCalibrator()
    cal.update(event)                    # 每个 VerificationEvent 调用一次
    thresh = cal.get_threshold(5, 1)     # → float
    cal.on_base_model_swap("v2.0")       # 缩短 EMA 窗口加速重收敛
"""

import math
from typing import Dict, Tuple

import torch


# ===========================================================================
# EMA threshold estimator
# ===========================================================================


class _EMAThreshold:
    """指数滑动窗口的均值+标准差估计, 用于动态阈值。

    维护两个 EMA:
      - error_mean: 误差均值
      - error_var:  误差方差 (EMA of squared deviation)

    阈值 = error_mean + k * sqrt(error_var)

    k 控制了阈值在分布中的位置:
      - k=2 → ~95th percentile (正态假设)
      - k=3 → ~99.7th percentile

    ``window`` 参数控制 EMA 的半衰期:
      - window=500 → alpha=0.998 (缓慢, 适合长期稳定运行)
      - window=50  → alpha=0.980 (快速, 适合版本切换后短期)
    """

    def __init__(self, window: int = 500, k: float = 2.0):
        self.window = window
        self.alpha = math.exp(-1.0 / window) if window > 0 else 0.0
        self.k = k

        self.error_mean: float = 0.0
        self.error_var: float = 1e-6  # 小初始值防止除零
        self.n_updates: int = 0

    def update(self, error_value: float):
        """更新 EMA 统计量。"""
        self.n_updates += 1
        alpha = self.alpha
        delta = error_value - self.error_mean
        self.error_mean = alpha * self.error_mean + (1 - alpha) * error_value
        self.error_var = alpha * self.error_var + (1 - alpha) * delta ** 2

    def get_threshold(self, min_floor: float = 0.001) -> float:
        """返回当前阈值: mean + k·std, 不低于 min_floor。"""
        std = math.sqrt(max(0.0, self.error_var))
        return max(min_floor, self.error_mean + self.k * std)

    @property
    def mean(self) -> float:
        return self.error_mean

    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self.error_var))

    def set_window(self, window: int):
        """动态调整 EMA 窗口 (用于版本切换后加速重收敛)。"""
        self.window = window
        self.alpha = math.exp(-1.0 / window) if window > 0 else 0.0

    def is_ready(self, min_updates: int = 50) -> bool:
        return self.n_updates >= min_updates

    def reset(self):
        self.error_mean = 0.0
        self.error_var = 1e-6
        self.n_updates = 0


# ===========================================================================
# OnlineCalibrator — main class
# ===========================================================================


class OnlineCalibrator:
    """对每个 ``(layer_id, timestep_bucket)`` 维护 EMA 阈值估计器。

    使用方式::

        cal = OnlineCalibrator(
            ema_window=500,
            threshold_k=2.0,
        )

        # 每个 VFL event 调用一次 (在 M1 record_event 的同一位置)
        cal.update(event)

        # 在 TeaCache decide / SpecA cal_type 中查询
        threshold = cal.get_threshold(layer_id, bucket)
    """

    def __init__(self,
                 ema_window: int = 500,
                 ema_window_short: int = 50,
                 threshold_k: float = 3.0,
                 min_floor: float = 0.001):
        """
        Parameters
        ----------
        ema_window : int
            正常运行 EMA 窗口 (半衰期步数)。
        ema_window_short : int
            版本切换后的短 EMA 窗口。
        threshold_k : float
            阈值 = mean + k·std 中的 k。
        min_floor : float
            阈值下限。
        """
        self.ema_window_long = ema_window
        self.ema_window_short = ema_window_short
        self.threshold_k = threshold_k
        self.min_floor = min_floor

        # (layer_id, bucket) → _EMAThreshold
        self._ema: Dict[Tuple[int, int], _EMAThreshold] = {}

        self._total_updates: int = 0

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, event) -> None:
        """用 VerificationEvent 更新 EMA 统计量。

        每次 event 更新对应 stratum 的 EMA 误差分布 → 动态阈值。
        """
        key = (event.layer_id, event.timestep_bucket)

        if key not in self._ema:
            self._ema[key] = _EMAThreshold(
                window=self.ema_window_long, k=self.threshold_k)

        self._ema[key].update(event.error_value)
        self._total_updates += 1

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_threshold(self, layer_id: int, timestep_bucket: int,
                      default: float = 0.25) -> float:
        """获取 ``(layer, bucket)`` 的当前动态阈值。

        如果该 stratum 的 EMA 尚未就绪 (样本不足), 返回 default。
        就绪后, 在线阈值不低于 default (静态公式作为地板),
        防止冷启动阶段阈值过低导致质量退化。
        在 exploit_mode 下, 允许阈值低于静态默认值 (用于飞轮第二阶段)。
        """
        key = (layer_id, timestep_bucket)
        ema = self._ema.get(key)
        if ema is None or not ema.is_ready():
            return default
        online = ema.get_threshold(min_floor=self.min_floor)
        if getattr(self, '_exploit_mode', False):
            return online  # 飞轮模式: 允许更激进的阈值
        return max(online, default)  # 保守模式: 静态默认值做地板

    def get_stats(self, layer_id: int, timestep_bucket: int) -> Dict:
        """获取 ``(layer, bucket)`` 的诊断统计。"""
        key = (layer_id, timestep_bucket)
        ema = self._ema.get(key)

        stats = {
            "layer_id": layer_id,
            "timestep_bucket": timestep_bucket,
            "ema_ready": ema.is_ready() if ema else False,
            "ema_updates": ema.n_updates if ema else 0,
        }
        if ema is not None:
            stats["ema_mean"] = ema.mean
            stats["ema_std"] = ema.std
            stats["threshold"] = ema.get_threshold(min_floor=self.min_floor)

        return stats

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_base_model_swap(self, new_version: str):
        """Base model 版本切换回调。

        将所有 EMA 窗口缩短至 ema_window_short, 加速重收敛。
        """
        for ema in self._ema.values():
            ema.set_window(self.ema_window_short)

    def on_converged(self):
        """恢复正常运行参数 (在版本切换重新收敛后调用)。"""
        for ema in self._ema.values():
            ema.set_window(self.ema_window_long)

    def set_exploit_mode(self, enabled: bool = True):
        """启用飞轮 exploit 模式: 允许阈值低于静态默认值, 降低 k 值。"""
        self._exploit_mode = enabled
        if enabled:
            for ema in self._ema.values():
                ema.k = 1.5  # 更激进的阈值 (从 3.0 降到 1.5)

    @property
    def total_updates(self) -> int:
        return self._total_updates

    @property
    def num_strata(self) -> int:
        return len(self._ema)
