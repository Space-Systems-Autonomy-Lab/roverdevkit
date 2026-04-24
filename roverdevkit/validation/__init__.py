"""Validation harnesses (project_plan.md §7).

- :mod:`.rover_registry`        — published-rover design vectors,
  scenarios, and truth numbers (Week 5).
- :mod:`.rover_comparison`      — run the evaluator on the registry and
  score vs truth (Week 5, Layer 4).
- :mod:`.rover_rediscovery`     — the headline validation (Layer 5, Week 12).
- :mod:`.experimental_comparison` — sub-model vs published single-wheel
  testbed data (Layer 3, Week 9).
- :mod:`.error_budget`          — layered error-budget compilation (Week 9).
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
    "format_report",
    "load_truth_table",
    "registry",
    "registry_by_name",
    "truth_by_rover",
]
