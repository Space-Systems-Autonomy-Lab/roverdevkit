"""End-to-end mission-evaluator integration tests.

Fleshed out in Weeks 4–5. The Week 5 acceptance test loads Yutu-2 /
Pragyan / Rashid parameters and checks that the evaluator predicts daily
traverse distance and power profile in the right order of magnitude.
"""

from __future__ import annotations

import pytest

from roverdevkit.mission.evaluator import evaluate
from roverdevkit.schema import DesignVector, MissionScenario


@pytest.mark.integration
@pytest.mark.xfail(reason="evaluator.py is implemented in Week 4 (project_plan.md §6).")
def test_evaluator_runs_on_equatorial_scenario(
    rashid_like_design: DesignVector, equatorial_scenario: MissionScenario
) -> None:
    metrics = evaluate(rashid_like_design, equatorial_scenario)
    assert metrics.total_mass_kg > 0
    assert metrics.range_km >= 0
