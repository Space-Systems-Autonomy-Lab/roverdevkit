"""Top-level mission evaluator.

This is the **primary artifact** of the project (project_plan.md §2). The
surrogate layer is a fast approximation of this function; the tradespace
and validation layers consume its outputs. Every ML claim in the paper is
grounded in what this function computes.

Public API::

    from roverdevkit.mission.evaluator import evaluate
    from roverdevkit.mission.scenarios import load_scenario
    from roverdevkit.schema import DesignVector

    metrics = evaluate(design_vector, scenario)
"""

from __future__ import annotations

from roverdevkit.schema import DesignVector, MissionMetrics, MissionScenario


def evaluate(
    design: DesignVector,
    scenario: MissionScenario,
    *,
    use_scm_correction: bool = False,
) -> MissionMetrics:
    """Run the full mission evaluator on one design in one scenario.

    Pipeline:

    1. Mass model → total mass, per-wheel vertical load.
    2. Thermal survival check → binary constraint.
    3. Traverse sim → time-stepped loop of terramechanics + power + battery
       updates until traverse complete or battery depleted.
    4. Aggregate time-series into mission-level metrics.

    Parameters
    ----------
    design
        12-D design vector.
    scenario
        Mission context (latitude, terrain, distance, sun geometry).
    use_scm_correction
        If True, apply the learned Bekker-Wong → SCM correction (Path 2).
        Requires the correction model to be loaded. Default False so the
        analytical path always works.
    """
    raise NotImplementedError("Implement in Week 4 per project_plan.md §6.")
