# Pretrained surrogates

Ships the canonical surrogate so that users can do tradespace exploration
without installing PyChrono and without regenerating the 50k-sample
analytical dataset.

Files expected once Week 8 is complete:

- `default_surrogate.pkl` — XGBoost multi-output model trained on the full
  analytical + SCM-correction dataset. Small enough to ship in git.
- `metadata.json` — training commit hash, dataset hash, hold-out accuracy
  table, calibration scaling factor. Written by
  `roverdevkit.surrogate.train`.

Regenerate with::

    python -m roverdevkit.surrogate.train \\
        --data data/analytical/lhs_50k.parquet \\
        --model xgboost \\
        --output pretrained/default_surrogate.pkl
