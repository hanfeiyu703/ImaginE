"""
Anti-collapse regularizer for ImaginE imagination representations.

Replaces the original EP-statistic-based SIGReg with a VICReg-style
variance-covariance regularization that is numerically well-behaved:

  L_reg = L_var + L_cov

  L_var: hinge loss on per-dimension std — penalizes dimensions whose
         standard deviation falls below a target threshold (default 1.0).
         This directly prevents representation collapse.

  L_cov: penalizes off-diagonal elements of the covariance matrix,
         encouraging dimensions to be decorrelated (prevents redundancy).

Reference:
    - VICReg (Bardes et al., 2022)
    - Adapted for ImaginE's imagination representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGReg(nn.Module):
    """Variance-Covariance anti-collapse regularizer.

    Keeps the original class name for backward compatibility with
    ImaginELoss and config references.
    """

    def __init__(self):
        super().__init__()
        self.std_target = 1.0
        self.cov_weight = 1.0  # no extra down-scaling needed after proper normalization

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute variance + covariance regularization loss.

        Args:
            z: (N, D) embedding matrix
        Returns:
            scalar loss (always >= 0)
        """
        if z.dim() > 2:
            z = z.reshape(-1, z.size(-1))

        N, D = z.shape
        if N < 4:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        # --- Variance term: hinge on per-dimension std ---
        std = z.std(dim=0)  # (D,)
        var_loss = F.relu(self.std_target - std).mean()

        # --- Covariance term: penalize off-diagonal correlations ---
        z_centered = z - z.mean(dim=0, keepdim=True)
        cov = (z_centered.T @ z_centered) / (N - 1)  # (D, D)
        cov_sq = cov.pow(2)
        diag_mask = torch.eye(D, device=z.device, dtype=torch.bool)
        # Normalize by the number of off-diagonal elements: D*(D-1)
        cov_loss = cov_sq.masked_fill(diag_mask, 0.0).sum() / (D * (D - 1)) * self.cov_weight

        # Expected magnitude: var_loss ~ 0..1, cov_loss ~ 0..0.1
        # With gamma=0.05, contribution to total loss is small (~0.005)
        return var_loss + cov_loss
