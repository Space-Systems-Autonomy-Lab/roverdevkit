"""Validation harnesses (project_plan.md §7).

- :mod:`.rover_registry`    — published-rover design vectors, scenarios,
  and truth numbers (Week 5).
- :mod:`.rover_comparison`  — run the evaluator on the registry and score
  vs truth (Week 5, Layer 4).
- :mod:`.rover_rediscovery` — the headline validation (Layer 5, Week 12).
- :mod:`.cross_scenario`    — robustness checks across the four scenario
  families (Week 5, Layer 6).

Layer-3 sub-model validation against published wheel-testbed data and
the consolidated layered error budget were originally scoped to a
dedicated Week 9. Post-W7.7 reframe these collapsed to a single
Wong-textbook tolerance test (in :mod:`tests.test_terramechanics`)
plus a markdown writeup at ``reports/error_budget.md``; both land in
Week 13 alongside the publication figures (project_plan.md §6 Phase 4).
"""

from roverdevkit.validation.rover_comparison import (
    ComparisonSummary,
    RoverComparisonResult,
    acceptance_gate,
    compare_all,
    compare_one,
    format_report,
)
from roverdevkit.validation.rover_registry import (
    PublishedTruth,
    RoverRegistryEntry,
    flown_registry,
    load_truth_table,
    registry,
    registry_by_name,
    truth_by_rover,
)

__all__ = [
    "ComparisonSummary",
    "PublishedTruth",
    "RoverComparisonResult",
    "RoverRegistryEntry",
    "acceptance_gate",
    "compare_all",
    "compare_one",
    "flown_registry",
    "format_report",
    "load_truth_table",
    "registry",
    "registry_by_name",
    "truth_by_rover",
]
