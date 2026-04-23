"""Tests for the surrogate sub-package.

Fleshed out in Weeks 6–8 once the analytical dataset is generated.
"""

from __future__ import annotations

from roverdevkit.surrogate.features import FEATURE_COLUMNS, TARGET_COLUMNS


def test_feature_and_target_lists_are_non_empty() -> None:
    assert len(FEATURE_COLUMNS) == 12
    assert "range_km" in TARGET_COLUMNS
    assert "total_mass_kg" in TARGET_COLUMNS
    assert set(FEATURE_COLUMNS).isdisjoint(set(TARGET_COLUMNS))
