"""SHAP-based design-rule extraction.

Feeds the trained surrogate into SHAP to generate:

- Global feature importance per output metric.
- Partial-dependence / SHAP-dependence plots for the most important features.
- Interaction plots for known-coupled pairs (wheel radius × slope,
  solar area × latitude, etc.).

The target is qualitative, interpretable statements like "below 15 kg total
mass, 4-wheel configurations dominate; above, 6-wheel becomes Pareto-optimal
because [reason]" (project_plan.md §6 W12).
"""

from __future__ import annotations

import pandas as pd


def compute_shap_values(surrogate, x: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe of per-sample, per-feature SHAP values."""
    raise NotImplementedError("Implement in Week 12 per project_plan.md §6.")


def extract_design_rules(surrogate, x: pd.DataFrame, *, top_k: int = 5) -> list[str]:
    """Extract human-readable design rules from SHAP patterns."""
    raise NotImplementedError("Implement in Week 12 per project_plan.md §6.")
