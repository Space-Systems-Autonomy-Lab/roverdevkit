"""Terramechanics sub-models.

- :mod:`.bekker_wong` — Bekker-Wong pressure-sinkage with Janosi-Hanamoto
  shear. Fast analytical path (Path 1).
- :mod:`.pychrono_scm` — single-wheel PyChrono Soil Contact Model driver,
  drop-in signature analog of :func:`.bekker_wong.single_wheel_forces`.
  Heavier but more physically detailed (Path 2).
- :mod:`.scm_wrapper` — Week-7 batch orchestration around
  :mod:`.pychrono_scm`: parallel runs, resumable work queue, CSV I/O.
- :mod:`.correction_model` — ML correction model that predicts the delta
  between SCM and Bekker-Wong over the design space.
"""
