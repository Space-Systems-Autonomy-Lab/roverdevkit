"""Terramechanics sub-models.

- :mod:`.bekker_wong` — Bekker-Wong pressure-sinkage with Janosi-Hanamoto
  shear. Fast analytical path (Path 1).
- :mod:`.pychrono_scm` — single-wheel PyChrono Soil Contact Model driver,
  drop-in signature analog of :func:`.bekker_wong.single_wheel_forces`.
  Heavier but more physically detailed (Path 2).
- :mod:`.soils` — name -> :class:`SoilParameters` lookup backed by
  :file:`data/soil_simulants.csv`.
- :mod:`.scm_wrapper` — Week-7 batch orchestration around
  :mod:`.pychrono_scm`: parallel runs, resumable work queue, CSV I/O.
- :mod:`.correction_model` — ML correction model that predicts the delta
  between SCM and Bekker-Wong over the design space.
"""

from roverdevkit.terramechanics.bekker_wong import (
    SoilParameters,
    WheelForces,
    WheelGeometry,
    single_wheel_forces,
)
from roverdevkit.terramechanics.soils import (
    SoilSimulantRecord,
    get_soil_parameters,
    list_soil_simulants,
    load_soil_catalogue,
)

__all__ = [
    "SoilParameters",
    "SoilSimulantRecord",
    "WheelForces",
    "WheelGeometry",
    "get_soil_parameters",
    "list_soil_simulants",
    "load_soil_catalogue",
    "single_wheel_forces",
]
