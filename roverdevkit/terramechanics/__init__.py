"""Terramechanics sub-models.

- :mod:`.bekker_wong` — Bekker-Wong pressure-sinkage with Janosi-Hanamoto
  shear.
- :mod:`.scm_wrapper` — thin wrapper around PyChrono's Soil Contact Model
  for single-wheel high-fidelity runs (optional Path 2).
- :mod:`.correction_model` — ML correction model that predicts the delta
  between SCM and Bekker-Wong over the design space.
"""
