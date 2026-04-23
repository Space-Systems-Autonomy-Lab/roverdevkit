"""Parametric sweep engine.

User fixes a subset of design variables, sweeps others on a grid, and the
evaluator (or surrogate) returns a dataframe of metrics over the grid.
Intended to be driven interactively from a Jupyter notebook via
``ipywidgets``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from roverdevkit.schema import MissionScenario


def sweep(
    fixed: Mapping[str, float | int],
    swept: Mapping[str, Sequence[float]],
    scenario: MissionScenario,
    *,
    use_surrogate: bool = True,
) -> pd.DataFrame:
    """Run a grid sweep over ``swept`` variables with ``fixed`` held constant.

    Parameters
    ----------
    fixed
        Mapping of design-variable name to fixed value.
    swept
        Mapping of design-variable name to array of grid values. Cartesian
        product is taken over all swept variables.
    scenario
        Mission context for every grid cell.
    use_surrogate
        If True, evaluate via the surrogate (fast, default). If False, call
        the full evaluator (slow but exact).
    """
    raise NotImplementedError("Implement in Week 10 per project_plan.md §6.")
