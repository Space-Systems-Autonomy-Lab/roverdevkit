"""Shared plotting utilities for tradespace figures.

Matplotlib is the primary backend (for publication figures); plotly is used
for interactive 3-D Pareto fronts inside notebooks. Keep the matplotlib
defaults consistent across the paper — set up a shared ``rcParams`` helper.
"""

from __future__ import annotations

import pandas as pd


def plot_pareto_front_3d(front_df: pd.DataFrame, *, backend: str = "plotly"):
    """Plot the (range, mass, slope) Pareto front. Returns the figure object."""
    raise NotImplementedError("Implement in Week 11 per project_plan.md §6.")


def plot_real_rovers_overlay(front_df: pd.DataFrame, rovers_df: pd.DataFrame):
    """Overlay real-rover design points on a Pareto front (the headline plot)."""
    raise NotImplementedError("Implement in Week 12 per project_plan.md §6.")
