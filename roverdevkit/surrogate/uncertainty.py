"""Uncertainty quantification and calibration utilities.

Two UQ paths, matching the two primary model families:

1. **Quantile regression** for gradient-boosted trees — fit three models at
   τ ∈ {0.05, 0.5, 0.95} and take the outer quantiles as the 90 % PI.
2. **Deep ensembles / MC dropout** for NNs — aggregate over ≥ 5 members,
   report mean and std.

Calibration check: on the held-out test set, 90 % prediction intervals
should cover ~90 % of points. If systematically over/underconfident, apply
a post-hoc scalar rescaling (Platt-style) before publication.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def prediction_interval_coverage(
    y_true: NDArray[np.float64],
    y_lo: NDArray[np.float64],
    y_hi: NDArray[np.float64],
) -> float:
    """Fraction of points where ``y_lo ≤ y_true ≤ y_hi``."""
    return float(np.mean((y_true >= y_lo) & (y_true <= y_hi)))


def calibrate_intervals(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    y_std: NDArray[np.float64],
    target_coverage: float = 0.9,
) -> float:
    """Return a scalar scaling factor for ``y_std`` that hits target coverage."""
    raise NotImplementedError("Implement in Week 8 per project_plan.md §6.")
