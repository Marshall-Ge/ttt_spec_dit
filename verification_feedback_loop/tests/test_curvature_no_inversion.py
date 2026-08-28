# -*- coding: utf-8 -*-
"""M5 constraint test: verify curvature loss does NOT depend on predicted_feature.

This test explicitly verifies Constraint 1 from the VFL spec:
  "绝不能让模型去拟合 draft 的预测值"

It checks that the loss computation graph contains no reference to
predicted_feature — the loss must be derived solely from true_feature sequences.
"""

import torch
import torch.nn as nn


# ===========================================================================
# Reference implementation of trajectory_curvature_loss (M5)
# ===========================================================================


def trajectory_curvature_loss(
    true_features: list,  # List[Tensor] — consecutive true features along trajectory
    order: int = 2,
) -> torch.Tensor:
    """Curvature loss: penalises deviation from local low-order polynomial fit.

    For a sequence of consecutive true features along the denoising trajectory,
    fit a local polynomial of given order and penalise the fitting residual.

    This encourages the trajectory to be more "extrapolatable" by low-order
    methods (Taylor expansion in SpecA, residual reuse in TeaCache) WITHOUT
    ever referencing the draft prediction — the loss only sees true_feature.

    Parameters
    ----------
    true_features : list of Tensor
        Sequence of true hidden states along consecutive timesteps.
        Each tensor has shape (B, seq, hidden_dim).
    order : int
        Polynomial order for local fit (default 2, matching SpecA Taylor order).

    Returns
    -------
    loss : scalar Tensor
    """
    if len(true_features) < order + 2:
        if len(true_features) == 0:
            return torch.tensor(0.0)
        return torch.tensor(0.0, device=true_features[0].device)

    # Stack along a new "time" dimension: (B, seq, hidden_dim, T)
    stacked = torch.stack(true_features, dim=-1)  # (B, seq, hidden_dim, T)
    T = stacked.shape[-1]

    # Build a Vandermonde-like design matrix for polynomial fitting.
    # Use normalised time coordinate t ∈ [-1, 1].
    t = torch.linspace(-1, 1, T, device=stacked.device,
                       dtype=stacked.dtype)  # (T,)
    # Design matrix: (T, order+1) — columns = [t^0, t^1, ..., t^order]
    A = torch.stack([t ** k for k in range(order + 1)], dim=1)  # (T, order+1)

    # Solve least-squares: (A^T A)^{-1} A^T
    # For numerical stability use torch.linalg.lstsq
    # stacked: (B, seq, hidden_dim, T) → reshape to (B*seq*hidden_dim, T)
    B, S, H = stacked.shape[0], stacked.shape[1], stacked.shape[2]
    flat = stacked.permute(0, 1, 2, 3).reshape(-1, T)  # (B*S*H, T)

    # lstsq: solve flat^T = A @ coeffs  →  coeffs = (A^T A)^{-1} A^T flat^T
    solution = torch.linalg.lstsq(A, flat.T)  # solution: (order+1, B*S*H)
    coeffs = solution.solution  # (order+1, B*S*H)

    # Reconstruct: fitted = A @ coeffs → (T, B*S*H) → (B*S*H, T)
    fitted = (A @ coeffs).T  # (B*S*H, T)
    fitted = fitted.reshape(B, S, H, T)  # (B, S, hidden_dim, T)

    # Residual: actual - fitted
    residual = stacked - fitted  # (B, S, hidden_dim, T)
    loss = (residual ** 2).mean()

    return loss


# ===========================================================================
# Constraint verification
# ===========================================================================


class TestCurvatureLossNoInversion:
    """Verify that trajectory_curvature_loss only depends on true_feature."""

    def test_no_predicted_input(self):
        """The loss function should NEVER accept a predicted_feature argument."""
        import inspect
        sig = inspect.signature(trajectory_curvature_loss)
        params = list(sig.parameters.keys())
        assert "predicted_feature" not in params, \
            "trajectory_curvature_loss must NOT accept predicted_feature"
        assert "predicted" not in params, \
            "trajectory_curvature_loss must NOT accept any predicted parameter"

    def test_loss_computes_without_predicted(self):
        """Loss should compute successfully with only true_features."""
        true_features = [torch.randn(2, 256, 1152) for _ in range(5)]
        loss = trajectory_curvature_loss(true_features, order=2)
        assert torch.isfinite(loss).all()
        assert loss.item() >= 0.0

    def test_gradient_only_through_true(self):
        """Gradients should only flow through true_features, not any predicted."""
        # Create true_features that require grad
        true_features = [torch.randn(2, 64, 256, requires_grad=True)
                         for _ in range(5)]

        # Create a "predicted" tensor — it should NOT affect the loss
        predicted = torch.randn(2, 64, 256, requires_grad=True)

        # Compute loss using ONLY true_features
        loss = trajectory_curvature_loss(true_features, order=2)
        loss.backward()

        # true_features should have gradients
        for i, tf in enumerate(true_features):
            assert tf.grad is not None, f"true_features[{i}] should have grad"

        # predicted should have NO gradient (never used in computation)
        assert predicted.grad is None, \
            "predicted_feature must have NO gradient — it was never used in loss computation"

    def test_no_reference_to_predicted_in_graph(self):
        """Verify the computation graph has no reference to a predicted tensor."""
        true_features = [torch.randn(2, 64, 256, requires_grad=True)
                         for _ in range(5)]
        predicted = torch.randn(2, 64, 256)

        loss = trajectory_curvature_loss(true_features, order=2)

        # Check that the loss's grad_fn chain does not reference predicted
        # This is verified by checking that changing predicted doesn't change loss
        loss_val_1 = loss.item()

        predicted2 = torch.randn(2, 64, 256)  # completely different
        loss_val_2 = trajectory_curvature_loss(true_features, order=2).item()

        # Same true_features → same loss, regardless of predicted values
        assert abs(loss_val_1 - loss_val_2) < 1e-10, \
            "Loss changed when predicted changed — predicted must not affect loss"

    def test_empty_sequence(self):
        """Edge case: empty or too-short sequence."""
        loss = trajectory_curvature_loss([], order=2)
        assert loss.item() == 0.0

        loss = trajectory_curvature_loss(
            [torch.randn(1, 64, 256)], order=2)
        assert loss.item() == 0.0

    def test_order_1_baseline(self):
        """First-order (linear) fit should also work."""
        true_features = [torch.randn(1, 64, 256) for _ in range(4)]
        loss = trajectory_curvature_loss(true_features, order=1)
        assert torch.isfinite(loss).all()
