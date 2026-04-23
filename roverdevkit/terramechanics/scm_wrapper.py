"""PyChrono Soil Contact Model wrapper for single-wheel high-fidelity runs.

Path 2 of the three-path data generation strategy (see project_plan.md §5).
This module is **optional** — if PyChrono fails to install on the target
platform, the rest of the package runs unmodified on the analytical path.

Usage pattern::

    from roverdevkit.terramechanics.scm_wrapper import run_scm_single_wheel
    result = run_scm_single_wheel(wheel, soil, vertical_load_n=50, slip=0.2)

We target ~30 seconds per run on an M2 MacBook, 4–5 parallel workers via
``multiprocessing``, and ~2,000 total runs concentrated in regions where
Bekker-Wong is known to be weak: high slip, grousered wheels, sloped
terrain.
"""

from __future__ import annotations

from dataclasses import dataclass

from .bekker_wong import SoilParameters, WheelGeometry


@dataclass(frozen=True)
class SCMResult:
    """Outputs of a single PyChrono SCM single-wheel simulation."""

    drawbar_pull_n: float
    driving_torque_nm: float
    sinkage_m: float
    wall_clock_s: float
    chrono_version: str


def pychrono_available() -> bool:
    """Return True if PyChrono is importable in the current environment."""
    try:
        import pychrono  # noqa: F401
    except ImportError:
        return False
    return True


def run_scm_single_wheel(
    wheel: WheelGeometry,
    soil: SoilParameters,
    vertical_load_n: float,
    slip: float,
    slope_deg: float = 0.0,
    sim_time_s: float = 5.0,
) -> SCMResult:
    """Run a single-wheel PyChrono SCM simulation and return settled forces.

    Raises
    ------
    ImportError
        If PyChrono is not installed in the current environment.
    """
    raise NotImplementedError("Implement in Week 7 per project_plan.md §6.")
