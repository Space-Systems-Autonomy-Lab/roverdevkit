"""The four canonical mission scenarios (project_plan.md §3.2).

1. ``equatorial_mare_traverse`` — Apollo-17-like terrain, 14-day mission.
2. ``polar_prospecting`` — high latitude, long shadows, intermittent sun.
3. ``highland_slope_capability`` — up to 25° slopes, minimum mass climber.
4. ``crater_rim_survey`` — short traverse, lots of slope changes, energy-optimal.

Scenarios are serialized as YAML in :file:`roverdevkit/mission/scenarios/*.yaml`
to make them easy for users and reviewers to inspect. The loader validates
via the :class:`MissionScenario` pydantic model.
"""

from __future__ import annotations

from pathlib import Path

from roverdevkit.schema import MissionScenario, ScenarioName

SCENARIO_DIR = Path(__file__).parent / "configs"


def load_scenario(name: ScenarioName) -> MissionScenario:
    """Load a named canonical scenario from its YAML config."""
    raise NotImplementedError("Implement in Week 4 per project_plan.md §6.")


def list_scenarios() -> list[ScenarioName]:
    """List all available scenario names."""
    raise NotImplementedError("Implement in Week 4 per project_plan.md §6.")
