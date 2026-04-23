"""The headline validation: does the optimizer rediscover real rover designs?

Procedure (project_plan.md §7 Layer 5):

1. Read a real rover's mass budget + mission profile + terrain from
   :file:`data/published_rovers.csv`.
2. Run NSGA-II with matching constraints.
3. Take the Pareto-optimal designs in the neighborhood of the real rover's
   mass.
4. Measure geometric distance in the design space between the nearest
   Pareto point and the real rover.

Acceptance criteria: within ~25–30 % on key dimensions (wheel diameter,
chassis mass, solar area), and the real rover lies on or close to the
Pareto front — not interior-dominated by the optimizer's suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RediscoveryResult:
    rover_name: str
    nearest_pareto_index: int
    design_space_distance: float
    per_variable_errors_pct: dict[str, float]
    pareto_dominated: bool


def rediscover(rover_name: str) -> RediscoveryResult:
    """Run the rediscovery test for one real rover (e.g. 'Rashid', 'Pragyan')."""
    raise NotImplementedError("Implement in Week 12 per project_plan.md §6.")
