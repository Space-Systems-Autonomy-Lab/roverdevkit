"""Surrogate model architectures.

Three model families in scope:

1. **Linear regression** — baseline, sanity check.
2. **Gradient-boosted trees (XGBoost)** — primary model. Handles mixed
   integer/continuous inputs well, fast to train on 50k rows, good built-in
   quantile-regression support for UQ.
3. **Neural net ensemble (PyTorch)** — used for UQ via deep ensembles or
   MC dropout. Optional — only trained if time permits in Week 8.

Multi-fidelity composition::

    multi_fidelity(x) = bekker_wong_forward(x) + correction_model(x)

is implemented here as a thin wrapper so the tradespace layer treats it as
just another predictor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Surrogate(Protocol):
    """Common interface for all surrogate models."""

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> Surrogate: ...

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def predict_with_uncertainty(
        self, x: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(mean, std)`` predictions."""
        ...

    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> Surrogate: ...


class MultiFidelitySurrogate:
    """Composes the analytical Bekker-Wong + a learned correction model."""

    def __init__(self, base_surrogate: Surrogate, correction: Surrogate) -> None:
        raise NotImplementedError("Implement in Week 8 per project_plan.md §6.")

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError
