"""Surrogate training entry point.

Generates the training dataset (if needed), does an 80/10/10 split, trains
the baseline and primary models, and reports per-output R² and RMSE on the
held-out test set.

Typical CLI usage (once implemented)::

    python -m roverdevkit.surrogate.train \\
        --data data/analytical/lhs_50k.parquet \\
        --model xgboost \\
        --output pretrained/default_surrogate.pkl
"""

from __future__ import annotations

from pathlib import Path


def generate_lhs_dataset(
    n_samples: int,
    output_path: Path,
    scenarios: list[str] | None = None,
) -> None:
    """Latin-hypercube-sample the design space and run the analytical evaluator.

    Target: 50,000 samples in a few hours on a MacBook (project_plan.md §6 W6).
    """
    raise NotImplementedError("Implement in Week 6 per project_plan.md §6.")


def train_surrogate(
    data_path: Path,
    model_name: str,
    output_path: Path,
) -> dict[str, float]:
    """Train a surrogate and return a per-output metrics dict."""
    raise NotImplementedError("Implement in Week 6 per project_plan.md §6.")
