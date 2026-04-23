"""Tests for the terramechanics sub-package.

These are skeleton tests: most raise ``NotImplementedError`` today and will
be filled in during Week 1 against worked examples from Wong's *Theory of
Ground Vehicles* (4th ed., chapters 2--4).
"""

from __future__ import annotations

import pytest

from roverdevkit.terramechanics.bekker_wong import (
    SoilParameters,
    WheelGeometry,
    single_wheel_forces,
)


def test_soil_and_wheel_dataclasses_are_constructable() -> None:
    soil = SoilParameters(n=1.0, k_c=1.4, k_phi=820.0, cohesion_kpa=1.0, friction_angle_deg=45.0)
    wheel = WheelGeometry(radius_m=0.1, width_m=0.06, grouser_height_m=0.005, grouser_count=12)
    assert soil.n == 1.0
    assert wheel.radius_m == 0.1


@pytest.mark.xfail(reason="Bekker-Wong is implemented in Week 1 (project_plan.md §6).")
def test_single_wheel_matches_wong_textbook_example() -> None:
    """Placeholder: replace with a Wong worked example once implemented."""
    soil = SoilParameters(n=1.0, k_c=1.4, k_phi=820.0, cohesion_kpa=1.0, friction_angle_deg=45.0)
    wheel = WheelGeometry(radius_m=0.1, width_m=0.06)
    single_wheel_forces(wheel, soil, vertical_load_n=50.0, slip=0.2)
