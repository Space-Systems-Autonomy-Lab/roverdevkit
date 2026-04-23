"""ML correction model: predicts SCM-minus-Bekker-Wong residuals.

The multi-fidelity composition is::

    prediction = bekker_wong(x) + correction(x)

Learning the correction is a much easier regression problem than learning
SCM from scratch because most of the signal — the first-order pressure-
sinkage trend — is already captured by the analytical model. SCM only has
to explain the residual.

This module is populated in Week 8 once (a) the Bekker-Wong model is
validated and (b) the SCM dataset from Week 7 is available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class CorrectionModel(Protocol):
    """Structural interface for any correction-layer regressor."""

    def fit(self, x: NDArray[np.float64], y_residual: NDArray[np.float64]) -> "CorrectionModel":
        ...

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        ...

    def save(self, path: Path) -> None:
        ...


def train_correction_model(
    scm_csv: Path,
    bekker_predictions_csv: Path,
    output_path: Path,
) -> CorrectionModel:
    """Train the correction model on paired (SCM, Bekker-Wong) points."""
    raise NotImplementedError("Implement in Week 8 per project_plan.md §6.")
