"""NSGA-II multi-objective optimization via pymoo.

Three-objective Pareto front::

    maximize range_km
    minimize total_mass_kg
    maximize slope_capability_deg

Constraints (from project_plan.md §6 W10):
    - motor torque within limits,
    - mass budget (if specified by the scenario),
    - thermal_survival must be True,
    - slope_capability ≥ scenario.max_slope_deg for climbing scenarios.

Runs surrogate-in-the-loop — NSGA-II calls the fitted surrogate thousands
of times per generation, which is only tractable because the surrogate is
millisecond-latency.
"""

from __future__ import annotations

import pandas as pd

from roverdevkit.schema import MissionScenario


def run_nsga2(
    scenario: MissionScenario,
    *,
    population_size: int = 100,
    n_generations: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Run NSGA-II and return the Pareto-front design-and-metric dataframe."""
    raise NotImplementedError("Implement in Week 11 per project_plan.md §6.")
