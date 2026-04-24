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

_CANONICAL_NAMES: set[str] = {
    "equatorial_mare_traverse",
    "polar_prospecting",
    "highland_slope_capability",
    "crater_rim_survey",
}
"""Canonical tradespace scenarios that `list_scenarios` returns.

Validation-only scenarios (Week 5 real-rover cross-check) also live in
:data:`SCENARIO_DIR` but are excluded from the tradespace listing so
Phase-3 sweeps never accidentally pick them up."""


def _config_path(name: str) -> Path:
    return SCENARIO_DIR / f"{name}.yaml"


def load_scenario(name: str) -> MissionScenario:
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
    """List the canonical tradespace scenarios that ship with the package.

    Validation-only scenarios (e.g. ``chandrayaan3_pragyan``) are kept
    out of this list so Phase-3 sweeps never pick them up. Returned as
    a list of :data:`ScenarioName` literals; every element is guaranteed
    loadable by :func:`load_scenario`.
    """
    on_disk = {p.stem for p in SCENARIO_DIR.glob("*.yaml")}
    return cast(
        "list[ScenarioName]",
        sorted(on_disk & _CANONICAL_NAMES),
    )
