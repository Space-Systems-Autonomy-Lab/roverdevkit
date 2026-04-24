"""Soil-simulant lookup: resolve a scenario's ``soil_simulant`` name to a
:class:`SoilParameters` record from ``data/soil_simulants.csv``.

Why this module exists
----------------------
Mission scenarios (``roverdevkit/mission/configs/*.yaml``) reference soils
by name (e.g. ``Apollo_regolith_nominal``). The Bekker-Wong model takes a
:class:`SoilParameters` dataclass. This module is the bridge: it loads the
CSV once, exposes the catalogue as a dict, and maps name -> parameters.

The CSV is the single source of truth for soil parameters across the
project (terramechanics tests, traverse simulator, validation notebooks).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from roverdevkit.terramechanics.bekker_wong import SoilParameters

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
SOIL_CSV_PATH: Path = _REPO_ROOT / "data" / "soil_simulants.csv"


# ---------------------------------------------------------------------------
# Public record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SoilSimulantRecord:
    """One row from :file:`data/soil_simulants.csv` plus derived parameters."""

    simulant: str
    citation: str
    notes: str
    density_kg_per_m3: float
    parameters: SoilParameters


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_soil_catalogue(csv_path: Path | str | None = None) -> dict[str, SoilSimulantRecord]:
    """Load every row of ``soil_simulants.csv`` keyed by simulant name.

    Cached: the first call parses the CSV, subsequent calls return the
    same dict in O(1). Pass a different ``csv_path`` to bypass the cache
    (only useful for unit tests that supply a temp CSV).
    """
    path = Path(csv_path) if csv_path is not None else SOIL_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"soil-simulant catalogue not found at {path}. "
            "Expected data/soil_simulants.csv in the repo."
        )

    catalogue: dict[str, SoilSimulantRecord] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        required = {
            "simulant",
            "n",
            "k_c_kN_per_m_n_plus_1",
            "k_phi_kN_per_m_n_plus_2",
            "cohesion_kPa",
            "friction_angle_deg",
            "density_kg_per_m3",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"soil CSV missing required columns: {sorted(missing)}")

        for row in reader:
            params = SoilParameters(
                n=float(row["n"]),
                k_c=float(row["k_c_kN_per_m_n_plus_1"]),
                k_phi=float(row["k_phi_kN_per_m_n_plus_2"]),
                cohesion_kpa=float(row["cohesion_kPa"]),
                friction_angle_deg=float(row["friction_angle_deg"]),
            )
            name = row["simulant"].strip()
            catalogue[name] = SoilSimulantRecord(
                simulant=name,
                citation=row.get("citation", "").strip(),
                notes=row.get("notes", "").strip(),
                density_kg_per_m3=float(row["density_kg_per_m3"]),
                parameters=params,
            )

    if not catalogue:
        raise ValueError(f"soil catalogue at {path} has no rows.")
    return catalogue


def get_soil_parameters(simulant_name: str) -> SoilParameters:
    """Return the Bekker-Wong :class:`SoilParameters` for a named simulant.

    Raises
    ------
    KeyError
        If ``simulant_name`` is not in the catalogue. The error message
        lists the valid names.
    """
    catalogue = load_soil_catalogue()
    if simulant_name not in catalogue:
        available = sorted(catalogue.keys())
        raise KeyError(f"unknown soil simulant {simulant_name!r}. Known simulants: {available}")
    return catalogue[simulant_name].parameters


def list_soil_simulants() -> list[str]:
    """Return sorted list of simulant names available in the catalogue."""
    return sorted(load_soil_catalogue().keys())
