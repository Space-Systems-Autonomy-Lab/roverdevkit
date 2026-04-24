"""Mission evaluator: the primary artifact of the project.

- :mod:`.evaluator` — top-level ``evaluate(design, scenario) → metrics``.
- :mod:`.scenarios` — the four canonical mission scenarios as configs.
- :mod:`.traverse_sim` — time-stepped traverse loop integrating
  terramechanics, power, battery, mass, and thermal.
- :mod:`.capability` — static mobility capability metrics (max slope).
"""

from roverdevkit.mission.capability import max_climbable_slope_deg
from roverdevkit.mission.evaluator import evaluate
from roverdevkit.mission.scenarios import list_scenarios, load_scenario
from roverdevkit.mission.traverse_sim import TraverseLog, run_traverse

__all__ = [
    "TraverseLog",
    "evaluate",
    "list_scenarios",
    "load_scenario",
    "max_climbable_slope_deg",
    "run_traverse",
]
