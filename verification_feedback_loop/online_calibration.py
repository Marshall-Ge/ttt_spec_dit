# -*- coding: utf-8 -*-
"""M2: Online Calibrator — 在线统计量校准 (无梯度, 常驻).

对每个 ``(layer_id, timestep_bucket)`` 维护两组在线估计器:

1. **RLS (递归最小二乘)** — 拟合 ``proxy_diff → true_error`` 映射,
   替代 TeaCache 的离线 poly4 rescale 函数。指数遗忘, 自适应分布漂移。

2. **EMA 分位数阈值** — 动态调整拒绝阈值 τ, 使其跟随当前 base model
   的真实误差分布, 替代 SpecA 的固定 ``base_threshold * decay^progress``。

更新成本极低 (标量运算), 可以直接在 verification hook 同步路径里调用,
不算违反"梯度不同步"约束 (无 backprop, 只是统计量更新)。

Interface::

    cal = OnlineCalibrator()
    cal.update(event)                    # 每个 VerificationEvent 调用一次
    thresh = cal.get_threshold(5, 1)     # → float
    rescale = cal.get_rescale_fn(5, 1)   # → Callable[[float], float]
    cal.on_base_model_swap("v2.0")       # 缩短 EMA 窗口加速重收敛
"""

import math
from collections import defaultdict
from typing import Callable, Dict, Optional, Tuple

import torch


# ===========================================================================
# RLS estimator (exponential forgetting)
# ===========================================================================


class _RLSEstimator:
    """单变量输入的递归最小二乘, 带指数遗忘。

    拟合 ``y = θ^T · φ(x)``, 其中 φ(x) 是多项式基函数。
    默认使用二次基 ``[x², x, 1]``, 与离线 poly4 保持同族但更低阶 (更稳定)。

    算法 (exponential forgetting RLS):
        P = P / λ
        K = P·φ / (1 + φ^T·P·φ)
        θ = θ + K·(y - φ^T·θ)
        P = P - K·φ^T·P
    """

    def __init__(self, degree: int = 2, forget_factor: float = 0.995,
                 eps: float = 1e-4):
        """
        Parameters
        ----------
        degree : int
            多项式基函数的阶数 (默认 2: [x², x, 1])。
        forget_factor : float
            指数遗忘因子 λ ∈ (0, 1]。越小遗忘越快。
        eps : float
            初始协方差矩阵的对角值 (控制先验不确定度)。
        """
        self.degree = degree
        self.forget_factor = forget_factor
        self.dim = degree + 1  # [x^d, ..., x, 1]

        # 参数向量 θ, 初始化为零 (预测 error=0)
        self.theta = torch.zeros(self.dim)

        # 逆协方差矩阵 P = eps⁻¹ · I (大先验不确定度)
        self.P = torch.eye(self.dim) / eps

        self.n_updates: int = 0

    def _phi(self, x: float) -> torch.Tensor:
        """构造多项式基向量: [x^d, ..., x, 1]."""
        return torch.tensor([x ** k for k in range(self.degree, -1, -1)])

    def update(self, x: float, y: float) -> float:
        """用单对 (x, y) 更新 RLS 估计器。

        Returns
        -------
        pred_error : float
            更新前的预测误差 |y - ŷ| (用于诊断)。
        """
        phi = self._phi(x)  # (dim,)
        theta_old = self.theta.clone()
        y_pred = float(torch.dot(theta_old, phi))

        # Exponential forgetting
        self.P = self.P / self.forget_factor

        # Kalman gain
        P_phi = self.P @ phi  # (dim,)
        denom = 1.0 + float(torch.dot(phi, P_phi))
        K = P_phi / denom  # (dim,)

        # Update
        innovation = y - y_pred
        self.theta = theta_old + K * innovation
        self.P = self.P - torch.outer(K, phi @ self.P)  # (dim, dim)

        self.n_updates += 1
        return abs(innovation)

    def predict(self, x: float) -> float:
        """预测 ŷ = θ^T · φ(x)。"""
        phi = self._phi(x)
        return float(torch.dot(self.theta, phi))

    def get_linear_coeffs(self) -> Tuple[float, float]:
        """返回线性近似系数 (slope, intercept), 用于日志/诊断。

        即使模型是二次的, 也返回在 x=0 附近的一阶泰勒近似:
        ŷ ≈ θ[-1] + θ[-2]·x (截距 + 斜率)。
        """
        d = self.degree
        if d >= 2:
            intercept = float(self.theta[-1])      # θ₀
            slope = float(self.theta[-2])           # θ₁
        else:
            intercept = float(self.theta[-1])
            slope = float(self.theta[-2]) if d >= 1 else 0.0
        return slope, intercept

    def is_ready(self, min_updates: int = 10) -> bool:
        """是否已有足够样本用于预测 (避免冷启动噪声)。"""
        return self.n_updates >= min_updates

    def reset(self):
        """重置为初始状态 (用于 base model 版本隔离时可选择性重置)。"""
        self.theta = torch.zeros(self.dim)
        self.P = torch.eye(self.dim) / 1e-4
        self.n_updates = 0


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
    """对每个 ``(layer_id, timestep_bucket)`` 维护 RLS + EMA 估计器。

    使用方式::

        cal = OnlineCalibrator(
            forget_factor=0.995,
            ema_window=500,
            threshold_k=2.0,
            rls_degree=2,
        )

        # 每个 VFL event 调用一次 (在 M1 record_event 的同一位置)
        cal.update(event)

        # 在 TeaCache decide / SpecA cal_type 中查询
        threshold = cal.get_threshold(layer_id, bucket)
        rescale_fn = cal.get_rescale_fn(layer_id, bucket)
    """

    def __init__(self,
                 forget_factor: float = 0.995,
                 ema_window: int = 500,
                 ema_window_short: int = 50,
                 threshold_k: float = 3.0,
                 rls_degree: int = 2,
                 min_floor: float = 0.001):
        """
        Parameters
        ----------
        forget_factor : float
            RLS 遗忘因子 λ (0<λ≤1)。
        ema_window : int
            正常运行 EMA 窗口 (半衰期步数)。
        ema_window_short : int
            版本切换后的短 EMA 窗口。
        threshold_k : float
            阈值 = mean + k·std 中的 k。
        rls_degree : int
            RLS 多项式阶数。
        min_floor : float
            阈值下限。
        """
        self.forget_factor = forget_factor
        self.ema_window_long = ema_window
        self.ema_window_short = ema_window_short
        self.threshold_k = threshold_k
        self.rls_degree = rls_degree
        self.min_floor = min_floor

        # (layer_id, bucket) → _RLSEstimator
        self._rls: Dict[Tuple[int, int], _RLSEstimator] = {}
        # (layer_id, bucket) → _EMAThreshold
        self._ema: Dict[Tuple[int, int], _EMAThreshold] = {}

        self._total_updates: int = 0

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, event) -> None:
        """用 VerificationEvent 更新统计量。

        每次 event 更新对应 stratum 的:
          - RLS: proxy=event.error_value, target=event.error_value (自回归校准)
                但实际上我们想要的是 proxy_diff→error 映射。
                对于 TeaCache, proxy=raw_diff (从 teacache_state 取),
                target=event.error_value (VFL probe 测得的 hidden-state 误差)。
          - EMA: 更新误差分布 → 动态阈值。

        Note: 对于 TeaCache 的 rescale 校准, 需要额外的 raw_diff 信息。
        如果 event 携带了 proxy_value, 用它; 否则用 error_value 自身做
        分布估计 (仅更新 EMA 阈值)。
        """
        key = (event.layer_id, event.timestep_bucket)

        # RLS
        if key not in self._rls:
            self._rls[key] = _RLSEstimator(
                degree=self.rls_degree, forget_factor=self.forget_factor)

        # EMA
        if key not in self._ema:
            self._ema[key] = _EMAThreshold(
                window=self.ema_window_long, k=self.threshold_k)

        self._rls[key].update(event.error_value, event.error_value)
        self._ema[key].update(event.error_value)
        self._total_updates += 1

    def update_with_proxy(self, event, proxy_value: float) -> None:
        """用 VerificationEvent + 外部 proxy 值更新 RLS。

        这是 TeaCache 的主要更新路径:
          - proxy_value = raw_diff (teacache_decide 中算出的 relative L1)
          - target = event.error_value (VFL probe 的 hidden-state error)

        拟合 proxy_value → true_error 的映射, 替代离线 poly4。
        """
        key = (event.layer_id, event.timestep_bucket)

        if key not in self._rls:
            self._rls[key] = _RLSEstimator(
                degree=self.rls_degree, forget_factor=self.forget_factor)
        if key not in self._ema:
            self._ema[key] = _EMAThreshold(
                window=self.ema_window_long, k=self.threshold_k)

        self._rls[key].update(proxy_value, event.error_value)
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

    def get_rescale_fn(self, layer_id: int, timestep_bucket: int
                       ) -> Callable[[float], float]:
        """返回 ``(layer, bucket)`` 的在线 rescale 函数。

        如果 RLS 尚未就绪, 返回恒等函数 (no rescaling)。

        Returns
        -------
        fn : Callable[[float], float]
            输入 raw_diff, 输出预测的 true_error (clamped to ≥ 0)。
        """
        key = (layer_id, timestep_bucket)
        rls = self._rls.get(key)
        if rls is None or not rls.is_ready():
            return lambda x: max(0.0, x)  # identity fallback

        # 返回闭包: 用当前 RLS 参数预测, clamp 到 ≥0
        def _rescale(x: float) -> float:
            return max(0.0, rls.predict(x))

        return _rescale

    def get_stats(self, layer_id: int, timestep_bucket: int) -> Dict:
        """获取 ``(layer, bucket)`` 的诊断统计。"""
        key = (layer_id, timestep_bucket)
        rls = self._rls.get(key)
        ema = self._ema.get(key)

        stats = {
            "layer_id": layer_id,
            "timestep_bucket": timestep_bucket,
            "rls_ready": rls.is_ready() if rls else False,
            "rls_updates": rls.n_updates if rls else 0,
            "ema_ready": ema.is_ready() if ema else False,
            "ema_updates": ema.n_updates if ema else 0,
        }
        if ema is not None:
            stats["ema_mean"] = ema.mean
            stats["ema_std"] = ema.std
            stats["threshold"] = ema.get_threshold(min_floor=self.min_floor)
        if rls is not None and rls.is_ready():
            slope, intercept = rls.get_linear_coeffs()
            stats["rls_slope"] = slope
            stats["rls_intercept"] = intercept
            # 在典型 raw_diff 范围内的预测样例
            for x_test in [0.0, 0.1, 0.5, 1.0]:
                stats[f"predict_at_{x_test}"] = rls.predict(x_test)

        return stats

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_base_model_swap(self, new_version: str):
        """Base model 版本切换回调。

        将所有 EMA 窗口缩短至 ema_window_short, 加速重收敛。
        RLS 不清空 (历史信息仍有部分参考价值), 但临时增加遗忘速度。
        """
        for ema in self._ema.values():
            ema.set_window(self.ema_window_short)
        # RLS: 临时提高遗忘率 (降低 forget_factor)
        old_ff = self.forget_factor
        for rls in self._rls.values():
            rls.forget_factor = 0.9  # 短期快速遗忘
        # 恢复 (下次训练 worker 启动时调用 on_converged() 恢复)
        self.forget_factor = old_ff

    def on_converged(self):
        """恢复正常运行参数 (在版本切换重新收敛后调用)。"""
        for ema in self._ema.values():
            ema.set_window(self.ema_window_long)
        for rls in self._rls.values():
            rls.forget_factor = self.forget_factor

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
        return len(self._rls)
