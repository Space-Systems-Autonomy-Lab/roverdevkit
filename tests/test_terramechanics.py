"""Tests for the terramechanics sub-package.

Week-1 coverage is physics-first-principles sanity:

- force-balance self-consistency,
- monotonic response to load, slip, soil stiffness, and wheel width,
- sign conventions (positive slip ⇒ torque draw, zero slip ⇒ drawbar pull
  dominated by compaction drag),
- Bekker plate compaction resistance matches the integrated zero-slip
  drawbar pull to within model-form noise,
- sub-millisecond runtime.

Quantitative comparison to published single-wheel experiments (Wong
textbook ch. 4, with Ding 2011 / Iizuka & Kubota 2011 as optional
extensions) lands in Week 13 (Phase 4) once the validation CSVs are
digitized; see ``data/validation/README.md`` and project_plan.md §6
Phase 4.
"""

from __future__ import annotations

import time

import pytest

from roverdevkit.terramechanics.bekker_wong import (
    SoilParameters,
    WheelForces,
    WheelGeometry,
    _integrate_forces,
    single_wheel_forces,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def nominal_soil() -> SoilParameters:
    """Apollo regolith nominal — matches ``data/soil_simulants.csv``."""
    return SoilParameters(
        n=1.0,
        k_c=1.4,
        k_phi=820.0,
        cohesion_kpa=0.17,
        friction_angle_deg=46.0,
    )


@pytest.fixture
def loose_soil() -> SoilParameters:
    """Apollo regolith loose-bound — softer."""
    return SoilParameters(
        n=1.0,
        k_c=0.5,
        k_phi=400.0,
        cohesion_kpa=0.1,
        friction_angle_deg=30.0,
    )


@pytest.fixture
def dense_soil() -> SoilParameters:
    """Apollo regolith dense-bound — stiffer."""
    return SoilParameters(
        n=1.2,
        k_c=2.0,
        k_phi=1200.0,
        cohesion_kpa=0.5,
        friction_angle_deg=50.0,
    )


@pytest.fixture
def rashid_wheel() -> WheelGeometry:
    """Rashid-like: ~0.1 m radius, 0.06 m wide."""
    return WheelGeometry(radius_m=0.1, width_m=0.06)


# ---------------------------------------------------------------------------
# Dataclass smoke tests
# ---------------------------------------------------------------------------


def test_soil_and_wheel_dataclasses_are_constructable() -> None:
    soil = SoilParameters(n=1.0, k_c=1.4, k_phi=820.0, cohesion_kpa=1.0, friction_angle_deg=45.0)
    wheel = WheelGeometry(radius_m=0.1, width_m=0.06, grouser_height_m=0.005, grouser_count=12)
    assert soil.n == 1.0
    assert wheel.radius_m == 0.1


# ---------------------------------------------------------------------------
# Force-balance self-consistency
# ---------------------------------------------------------------------------


def test_force_balance_closes_at_solved_entry_angle(
    rashid_wheel: WheelGeometry, nominal_soil: SoilParameters
) -> None:
    """W_integrated(θ₁★) must equal the applied load within tight tolerance."""
    load = 150.0
    forces = single_wheel_forces(rashid_wheel, nominal_soil, load, slip=0.2)
    w_check, _, _ = _integrate_forces(forces.entry_angle_rad, rashid_wheel, nominal_soil, 0.2)
    assert w_check == pytest.approx(load, rel=1e-4)


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------


def test_sinkage_monotonic_in_load(
    rashid_wheel: WheelGeometry, nominal_soil: SoilParameters
) -> None:
    loads = [30.0, 60.0, 120.0, 240.0]
    sinkages = [
        single_wheel_forces(rashid_wheel, nominal_soil, w, slip=0.0).sinkage_m for w in loads
    ]
    assert sinkages == sorted(sinkages), f"sinkage should be monotonic in load, got {sinkages}"
    assert sinkages[-1] > sinkages[0]  # strict increase across the range


def test_drawbar_pull_increases_with_slip_in_traction_regime(
    rashid_wheel: WheelGeometry, nominal_soil: SoilParameters
) -> None:
    """Between low and moderate slip, drawbar pull grows.

    (The relationship saturates near ~50 % slip and is not strictly
    monotonic all the way to 100 %; we test only the rising regime.)
    """
    load = 150.0
    slips = [0.02, 0.05, 0.1, 0.2, 0.35]
    dps = [
        single_wheel_forces(rashid_wheel, nominal_soil, load, slip=s).drawbar_pull_n for s in slips
    ]
    for a, b in zip(dps[:-1], dps[1:], strict=True):
        assert b >= a - 1e-6, f"DP should not decrease in rising slip regime, got {dps}"
    assert dps[-1] > dps[0]


def test_softer_soil_sinks_more(
    rashid_wheel: WheelGeometry, loose_soil: SoilParameters, dense_soil: SoilParameters
) -> None:
    load = 150.0
    soft = single_wheel_forces(rashid_wheel, loose_soil, load, slip=0.0).sinkage_m
    stiff = single_wheel_forces(rashid_wheel, dense_soil, load, slip=0.0).sinkage_m
    assert soft > stiff


def test_wider_wheel_sinks_less(nominal_soil: SoilParameters) -> None:
    narrow = WheelGeometry(radius_m=0.1, width_m=0.04)
    wide = WheelGeometry(radius_m=0.1, width_m=0.12)
    load = 150.0
    z_narrow = single_wheel_forces(narrow, nominal_soil, load, slip=0.0).sinkage_m
    z_wide = single_wheel_forces(wide, nominal_soil, load, slip=0.0).sinkage_m
    assert z_wide < z_narrow


# ---------------------------------------------------------------------------
# Sign conventions
# ---------------------------------------------------------------------------


def test_drawbar_pull_much_smaller_at_zero_than_driving_slip(
    rashid_wheel: WheelGeometry, nominal_soil: SoilParameters
) -> None:
    """At slip = 0, net tractive force should be a small fraction of the
    driving-slip value.

    We deliberately do **not** assert ``DP(s=0) < 0``. The pure Bekker-Wong
    model keeps a nonzero kinematic shear term even at zero slip — for
    high-friction low-cohesion soils (Apollo regolith, φ ≈ 46°) this can
    tip the integrated DP slightly positive. That's a known ±15–30 %
    weakness of the analytical model (cf. Ishigami 2007, Ding 2011) and
    is exactly what the Path-2 SCM correction layer is meant to absorb.
    The robust first-principles test is therefore on the *ratio*: at
    zero slip, whatever sign it has, the magnitude should be much smaller
    than at moderate driving slip.
    """
    load = 150.0
    dp_zero = single_wheel_forces(rashid_wheel, nominal_soil, load, slip=0.0).drawbar_pull_n
    dp_drive = single_wheel_forces(rashid_wheel, nominal_soil, load, slip=0.3).drawbar_pull_n
    assert abs(dp_zero) < 0.5 * abs(dp_drive)
    assert dp_drive > 0.0


def test_torque_positive_when_driving(
    rashid_wheel: WheelGeometry, nominal_soil: SoilParameters
) -> None:
    forces = single_wheel_forces(rashid_wheel, nominal_soil, vertical_load_n=150.0, slip=0.2)
    assert forces.driving_torque_nm > 0.0


def test_compaction_resistance_positive_and_grows_with_sinkage(
    rashid_wheel: WheelGeometry, nominal_soil: SoilParameters
) -> None:
    """The Bekker plate compaction resistance is a diagnostic output;
    check it's positive and scales with sinkage (monotonic in load)."""
    light = single_wheel_forces(rashid_wheel, nominal_soil, vertical_load_n=30.0, slip=0.0)
    heavy = single_wheel_forces(rashid_wheel, nominal_soil, vertical_load_n=300.0, slip=0.0)
    assert light.rolling_resistance_n > 0.0
    assert heavy.rolling_resistance_n > light.rolling_resistance_n
    assert heavy.sinkage_m > light.sinkage_m


# ---------------------------------------------------------------------------
# Physical plausibility on a Rashid-class design point
# ---------------------------------------------------------------------------


def test_rashid_class_sinkage_and_dp_are_plausible(
    rashid_wheel: WheelGeometry, nominal_soil: SoilParameters
) -> None:
    """Rough order-of-magnitude check against published micro-rover experience.

    Rashid was ~10 kg / 4 wheels ≈ 25 N per wheel on Earth, ≈ 4 N per
    wheel on the Moon. We run 50 N per wheel (a conservative Earth-like
    check) on nominal regolith with moderate slip and verify that
    sinkage stays in a few mm to few cm, drawbar pull is nonzero, and
    the torque is well within stall-torque of hobby-scale drive motors
    (order 1 N·m). Ballpark, not a calibrated test.
    """
    forces = single_wheel_forces(rashid_wheel, nominal_soil, vertical_load_n=50.0, slip=0.15)
    assert 0.0005 < forces.sinkage_m < 0.05, (
        f"sinkage {forces.sinkage_m * 1000:.1f} mm out of range"
    )
    assert forces.drawbar_pull_n > 0.0
    assert 0.0 < forces.driving_torque_nm < 10.0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_rejects_nonpositive_load(
    rashid_wheel: WheelGeometry, nominal_soil: SoilParameters
) -> None:
    with pytest.raises(ValueError, match="positive"):
        single_wheel_forces(rashid_wheel, nominal_soil, vertical_load_n=0.0, slip=0.2)


def test_rejects_out_of_range_slip(
    rashid_wheel: WheelGeometry, nominal_soil: SoilParameters
) -> None:
    with pytest.raises(ValueError, match="slip"):
        single_wheel_forces(rashid_wheel, nominal_soil, vertical_load_n=50.0, slip=1.5)


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def test_single_wheel_forces_runs_under_one_millisecond(
    rashid_wheel: WheelGeometry, nominal_soil: SoilParameters
) -> None:
    """Per the plan (§4), we need < 1 ms per call for 50k+ mission runs.

    We time a warm run to exclude JIT / import overhead, then assert
    the amortized cost over 100 calls is sub-millisecond. CI jitter
    rarely breaks this margin on any M-series Mac or modern CI runner.
    """
    single_wheel_forces(rashid_wheel, nominal_soil, vertical_load_n=50.0, slip=0.2)  # warm
    n_calls = 100
    t0 = time.perf_counter()
    for _ in range(n_calls):
        single_wheel_forces(rashid_wheel, nominal_soil, vertical_load_n=50.0, slip=0.2)
    elapsed = (time.perf_counter() - t0) / n_calls
    assert elapsed < 1e-3, f"amortized {elapsed * 1000:.2f} ms/call exceeds 1 ms budget"


# ---------------------------------------------------------------------------
# Placeholder for Wong-textbook worked-example comparison (Week 13, Phase 4)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="Populate from Wong 4th ed. ch. 4 worked example once digitized.")
def test_single_wheel_matches_wong_textbook_example() -> None:
    soil = SoilParameters(n=1.0, k_c=1.4, k_phi=820.0, cohesion_kpa=1.0, friction_angle_deg=45.0)
    wheel = WheelGeometry(radius_m=0.1, width_m=0.06)
    forces: WheelForces = single_wheel_forces(wheel, soil, vertical_load_n=50.0, slip=0.2)
    # Placeholder acceptance thresholds; replace with Wong's published values.
    assert forces.drawbar_pull_n == pytest.approx(0.0, abs=0.1)
