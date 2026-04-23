"""Time-stepped traverse simulator.

At each step::

    power_in  = solar.panel_power_w(...)
    power_out = (rolling_resistance × v + avionics_draw + heater_draw)
    battery.step(power_in - power_out, dt)
    position += v × dt × drive_duty_cycle
    log(t, pos, soc, slip, sinkage, ...)

Terminates on: traverse distance reached, battery depleted below DoD, or
thermal constraint violated. Returns the full time-series log for
downstream aggregation by :func:`roverdevkit.mission.evaluator.evaluate`.

Target: ~300 lines of Python, <50 ms per mission on the analytical path
(project_plan.md §4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class TraverseLog:
    """Per-step traverse-sim history arrays."""

    t_s: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    position_m: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    state_of_charge: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    power_in_w: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    power_out_w: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    slip: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    sinkage_m: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    terminated_reason: str = ""


def run_traverse(*args, **kwargs) -> TraverseLog:
    """Run the traverse loop until completion or early termination.

    Concrete signature will be pinned down in Week 4; we leave ``*args`` /
    ``**kwargs`` here only to avoid premature commitment to a parameter list
    that's tangled in the schema + physics work of Weeks 1–3.
    """
    raise NotImplementedError("Implement in Week 4 per project_plan.md §6.")
