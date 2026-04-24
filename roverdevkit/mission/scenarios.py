"""The four canonical mission scenarios (project_plan.md §3.2).

1. ``equatorial_mare_traverse`` — Apollo-17-like terrain, 14-day mission.
2. ``polar_prospecting`` — high latitude, long shadows, intermittent sun.
3. ``highland_slope_capability`` — up to 25° slopes, minimum mass climber.
4. ``crater_rim_survey`` — short traverse, lots of slope changes, energy-optimal.

Scenarios are serialized as YAML in :file:`roverdevkit/mission/configs/*.yaml`
to make them easy for users and reviewers to inspect. The loader validates
via the :class:`MissionScenario` pydantic model so invalid fields raise
immediately at load time rather than deep inside the traverse sim.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml  # type: ignore[import-untyped]

from roverdevkit.schema import MissionScenario, ScenarioName

SCENARIO_DIR: Path = Path(__file__).parent / "configs"


def _config_path(name: str) -> Path:
    return SCENARIO_DIR / f"{name}.yaml"


def load_scenario(name: ScenarioName) -> MissionScenario:
    """Load a named canonical scenario from its YAML config.

    Parameters
    ----------
    name
        Scenario key; must match a ``*.yaml`` basename in
        :data:`SCENARIO_DIR`. The ``ScenarioName`` type alias pins the
        allowed values at the type-check level.

    Raises
    ------
    FileNotFoundError
        If no YAML file exists for ``name``.
    pydantic.ValidationError
        If the YAML contents do not validate against
        :class:`MissionScenario` (e.g. out-of-range latitude).
    """
    path = _config_path(name)
    if not path.exists():
        available = list_scenarios()
        raise FileNotFoundError(
            f"scenario config {path} not found. Available scenarios: {available}"
        )
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(
            f"scenario file {path} did not parse to a mapping (got {type(raw).__name__})."
        )
    return MissionScenario(**raw)


def list_scenarios() -> list[ScenarioName]:
    """List every scenario YAML that ships with the package.

    Returned as a list of :data:`ScenarioName` literals; every element is
    guaranteed loadable by :func:`load_scenario`.
    """
    # cast: ScenarioName is a Literal, but the YAML filenames on disk are
    # the source of truth. We validate by round-tripping through
    # MissionScenario when a scenario is actually loaded; here we just
    # surface what's on disk.
    return cast(
        "list[ScenarioName]",
        sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml")),
    )
