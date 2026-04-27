"""Pure-Python unit tests for :mod:`roverdevkit.tradespace.sweeps`.

These tests exercise the grid expansion and backend-pick logic
without touching joblib / xgboost / FastAPI -- they only need
:mod:`roverdevkit.schema` and numpy.
"""

from __future__ import annotations

import numpy as np
import pytest

from roverdevkit.schema import DesignVector
from roverdevkit.tradespace.sweeps import (
    EVALUATOR_AUTO_THRESHOLD,
    EVALUATOR_HARD_LIMIT,
    SURROGATE_HARD_LIMIT,
    SweepAxis,
    SweepSpec,
    expand_grid,
    pick_backend,
)


def _base_design() -> DesignVector:
    return DesignVector(
        wheel_radius_m=0.10,
        wheel_width_m=0.10,
        grouser_height_m=0.012,
        grouser_count=14,
        n_wheels=6,
        chassis_mass_kg=20.0,
        wheelbase_m=0.6,
        solar_area_m2=0.5,
        battery_capacity_wh=100.0,
        avionics_power_w=15.0,
        nominal_speed_mps=0.04,
        drive_duty_cycle=0.15,
    )


def test_sweep_axis_values_endpoints_inclusive() -> None:
    axis = SweepAxis(variable="wheel_radius_m", lo=0.08, hi=0.18, n_points=11)
    vals = axis.values()
    assert vals[0] == pytest.approx(0.08)
    assert vals[-1] == pytest.approx(0.18)
    assert len(vals) == 11


def test_sweep_axis_rejects_n_points_below_two() -> None:
    with pytest.raises(ValueError, match="n_points must be >= 2"):
        SweepAxis("wheel_radius_m", 0.08, 0.18, 1).values()


def test_sweep_axis_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match="hi must be > lo"):
        SweepAxis("wheel_radius_m", 0.18, 0.08, 5).values()


def test_sweep_spec_rejects_non_primary_target() -> None:
    with pytest.raises(ValueError, match="not a primary regression target"):
        SweepSpec(
            target="not_a_real_metric",
            x_axis=SweepAxis("wheel_radius_m", 0.08, 0.18, 5),
            y_axis=None,
        )


def test_sweep_spec_rejects_duplicate_axis_variables() -> None:
    axis = SweepAxis("wheel_radius_m", 0.08, 0.18, 5)
    with pytest.raises(ValueError, match="must sweep different variables"):
        SweepSpec(target="range_km", x_axis=axis, y_axis=axis)


def test_sweep_spec_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="not in"):
        SweepSpec(
            target="range_km",
            x_axis=SweepAxis("wheel_radius_m", 0.08, 0.18, 5),
            y_axis=None,
            backend="cuda",  # type: ignore[arg-type]
        )


def test_expand_grid_1d_overrides_just_x() -> None:
    spec = SweepSpec(
        target="range_km",
        x_axis=SweepAxis("wheel_radius_m", 0.08, 0.18, 5),
        y_axis=None,
    )
    designs = expand_grid(spec, _base_design())
    assert len(designs) == 5
    radii = [d.wheel_radius_m for d in designs]
    assert radii[0] == pytest.approx(0.08)
    assert radii[-1] == pytest.approx(0.18)
    # All other dims unchanged
    for d in designs:
        assert d.wheel_width_m == pytest.approx(0.10)
        assert d.solar_area_m2 == pytest.approx(0.5)


def test_expand_grid_2d_row_major_y_outer_x_inner() -> None:
    spec = SweepSpec(
        target="range_km",
        x_axis=SweepAxis("wheel_radius_m", 0.08, 0.18, 3),
        y_axis=SweepAxis("solar_area_m2", 0.4, 0.8, 2),
    )
    designs = expand_grid(spec, _base_design())
    assert len(designs) == 6
    # Row-major: first three share y[0], next three share y[1].
    ys = [d.solar_area_m2 for d in designs]
    assert ys[:3] == pytest.approx([0.4, 0.4, 0.4])
    assert ys[3:] == pytest.approx([0.8, 0.8, 0.8])
    xs = [d.wheel_radius_m for d in designs]
    np.testing.assert_allclose(xs[:3], [0.08, 0.13, 0.18])
    np.testing.assert_allclose(xs[3:], [0.08, 0.13, 0.18])


def test_expand_grid_rounds_integer_variable() -> None:
    spec = SweepSpec(
        target="range_km",
        x_axis=SweepAxis("grouser_count", 0.0, 24.0, 5),
        y_axis=None,
    )
    designs = expand_grid(spec, _base_design())
    counts = [d.grouser_count for d in designs]
    # Linear grid is [0, 6, 12, 18, 24]; all integers already.
    assert counts == [0, 6, 12, 18, 24]


def test_pick_backend_auto_uses_evaluator_below_threshold() -> None:
    spec = SweepSpec(
        target="range_km",
        x_axis=SweepAxis("wheel_radius_m", 0.08, 0.18, EVALUATOR_AUTO_THRESHOLD),
        y_axis=None,
    )
    assert pick_backend(spec) == "evaluator"


def test_pick_backend_auto_promotes_to_surrogate_above_threshold() -> None:
    n = EVALUATOR_AUTO_THRESHOLD + 1
    spec = SweepSpec(
        target="range_km",
        x_axis=SweepAxis("wheel_radius_m", 0.08, 0.18, n),
        y_axis=None,
    )
    assert pick_backend(spec) == "surrogate"


def test_pick_backend_explicit_evaluator_hard_limit() -> None:
    n = EVALUATOR_HARD_LIMIT + 1
    spec = SweepSpec(
        target="range_km",
        x_axis=SweepAxis("wheel_radius_m", 0.08, 0.18, n),
        y_axis=None,
        backend="evaluator",
    )
    with pytest.raises(ValueError, match="evaluator hard limit"):
        pick_backend(spec)


def test_pick_backend_explicit_surrogate_hard_limit() -> None:
    # 200 × 201 = 40_200 > SURROGATE_HARD_LIMIT (40_000).
    spec = SweepSpec(
        target="range_km",
        x_axis=SweepAxis("wheel_radius_m", 0.08, 0.18, 200),
        y_axis=SweepAxis("solar_area_m2", 0.4, 0.8, 201),
        backend="surrogate",
    )
    assert spec.n_cells() > SURROGATE_HARD_LIMIT
    with pytest.raises(ValueError, match="surrogate hard limit"):
        pick_backend(spec)
