"""Light smoke tests for the surrogate sub-package's column inventories.

Detailed dataset/sampler tests live in test_surrogate_sampling.py and
test_surrogate_dataset.py; this module only checks the cross-cutting
column-list invariants exposed by features.py so a stale rename is
caught at the smallest possible scope.
"""

from __future__ import annotations

from roverdevkit.surrogate.features import (
    CLASSIFICATION_TARGETS,
    DESIGN_FEATURE_COLUMNS,
    FEASIBILITY_COLUMN,
    INPUT_COLUMNS,
    PRIMARY_REGRESSION_TARGETS,
    REGRESSION_TARGETS,
    SCENARIO_CATEGORICAL_COLUMNS,
    SCENARIO_NUMERIC_COLUMNS,
)


def test_design_feature_count() -> None:
    assert len(DESIGN_FEATURE_COLUMNS) == 12


def test_input_columns_compose_from_groups() -> None:
    expected = DESIGN_FEATURE_COLUMNS + SCENARIO_NUMERIC_COLUMNS + SCENARIO_CATEGORICAL_COLUMNS
    assert expected == INPUT_COLUMNS
    assert len(INPUT_COLUMNS) == 25  # 12 + 9 + 4


def test_regression_targets_include_primaries() -> None:
    for col in PRIMARY_REGRESSION_TARGETS:
        assert col in REGRESSION_TARGETS
    assert "range_km" in PRIMARY_REGRESSION_TARGETS
    assert "total_mass_kg" in PRIMARY_REGRESSION_TARGETS


def test_inputs_disjoint_from_targets() -> None:
    assert set(INPUT_COLUMNS).isdisjoint(set(REGRESSION_TARGETS))
    assert set(INPUT_COLUMNS).isdisjoint(set(CLASSIFICATION_TARGETS))


def test_feasibility_classifier_is_motor_torque_only() -> None:
    """v2 schema: thermal_survival is no longer a surrogate target.

    See data/analytical/SCHEMA.md (v1 -> v2) and project_log.md for the
    rationale (thermal is a free design lever in the current mass model).
    """
    assert CLASSIFICATION_TARGETS == ["motor_torque_ok"]
    assert FEASIBILITY_COLUMN == "motor_torque_ok"
    assert "thermal_survival" not in CLASSIFICATION_TARGETS
    assert "thermal_survival" not in REGRESSION_TARGETS
