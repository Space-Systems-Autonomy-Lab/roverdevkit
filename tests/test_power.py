"""Tests for the power sub-package.

Fleshed out in Week 2 against published Yutu-2 noon-power numbers and
NASA Glenn battery curves.
"""

from __future__ import annotations

import pytest

from roverdevkit.power.solar import panel_power_w, sun_elevation_deg


@pytest.mark.xfail(reason="solar.py is implemented in Week 2 (project_plan.md §6).")
def test_noon_sun_at_equator_is_90_deg() -> None:
    assert sun_elevation_deg(latitude_deg=0.0, lunar_hour_angle_deg=0.0) == pytest.approx(90.0)


@pytest.mark.xfail(reason="solar.py is implemented in Week 2 (project_plan.md §6).")
def test_panel_power_is_zero_below_horizon() -> None:
    assert panel_power_w(panel_area_m2=1.0, panel_efficiency=0.3, sun_elevation_deg=-10.0) == 0.0
