"""Battery state-of-charge model with efficiency and temperature derating.

Simple coulomb-counting SOC with:
    - round-trip charge/discharge efficiency,
    - depth-of-discharge limit (prevents draining below a floor),
    - temperature-dependent usable-capacity derating.

Calibration data: NASA Glenn lithium-ion performance reports at lunar
temperatures. Validation target (Week 2): confirm a 100 Wh battery
delivers ~85 Wh usable at nominal lunar operating temperature.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BatteryState:
    """Mutable battery state updated each traverse-sim step."""

    capacity_wh: float
    """Nominal capacity at 25 °C and full DoD, Wh."""

    state_of_charge: float
    """Fraction in [0, 1]."""

    temperature_c: float = 20.0

    min_state_of_charge: float = 0.2
    """Depth-of-discharge limit; don't discharge below this."""

    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95


def step(state: BatteryState, power_net_w: float, dt_s: float) -> BatteryState:
    """Advance the battery state by one time step.

    Parameters
    ----------
    state
        Current battery state.
    power_net_w
        Net power into the battery, W (positive = charging).
    dt_s
        Time step length, s.
    """
    raise NotImplementedError("Implement in Week 2 per project_plan.md §6.")


def usable_capacity_wh(state: BatteryState) -> float:
    """Usable capacity after DoD limit and temperature derating."""
    raise NotImplementedError("Implement in Week 2 per project_plan.md §6.")
