"""Sub-model and mission-evaluator comparison against published data.

Covers validation layers 3 and 4 from project_plan.md §7:

- Bekker-Wong vs single-wheel testbed data (Ding 2011, Iizuka & Kubota 2011,
  Wong textbook datasets).
- Solar model vs published rover power profiles.
- Battery model vs datasheet discharge curves.
- Full mission evaluator vs published Yutu-2 / Pragyan traverse data.

Each comparison returns a tidy DataFrame of (experiment, predicted,
measured, percent_error) for inclusion in the paper's validation section.
"""

from __future__ import annotations

import pandas as pd


def compare_bekker_wong_to_experiments(dataset_name: str) -> pd.DataFrame:
    raise NotImplementedError("Implement in Week 9 per project_plan.md §6.")


def compare_mission_evaluator_to_real_rover(rover_name: str) -> pd.DataFrame:
    raise NotImplementedError("Implement in Week 5 per project_plan.md §6.")
