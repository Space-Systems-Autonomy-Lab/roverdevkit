"""Layered error-budget compilation for the paper.

Combines the per-layer validation numbers (project_plan.md §7) into a
single table::

    "Surrogate predicts mission range with ±X% error relative to the
     analytical evaluator, which itself agrees with published wheel
     testbed data to within ±Y%, giving end-to-end error of ±Z% on
     real-rover comparison."

Produces both a terse CSV (for the paper table) and a narrative string
(for the abstract).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ErrorBudgetRow:
    layer: str
    metric: str
    error_pct: float
    n_samples: int
    source: str


def compile_error_budget() -> pd.DataFrame:
    """Aggregate all layered validation results into one tidy DataFrame."""
    raise NotImplementedError("Implement in Week 9 per project_plan.md §6.")
