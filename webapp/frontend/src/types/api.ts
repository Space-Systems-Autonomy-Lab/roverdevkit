/**
 * TypeScript mirrors of the FastAPI Pydantic schemas.
 *
 * These are deliberately hand-written rather than generated from the
 * OpenAPI doc: the public surface is small, the manual definitions
 * give us better doc comments at the call sites, and they double as
 * the source of truth for design-vector bounds in the form UI. If
 * the API surface grows past ~10 routes we should switch to
 * `openapi-typescript` codegen as a build step.
 */

/** Lunar mission scenarios the surrogate is calibrated for. */
export type ScenarioName =
  | "equatorial_mare_traverse"
  | "polar_prospecting"
  | "highland_slope_capability"
  | "crater_rim_survey";

/** Subset of `MissionScenario.terrain_class` exposed via the API. */
export type TerrainClass =
  | "mare_nominal"
  | "mare_loose"
  | "highland_dense"
  | "polar_regolith";

export type SunGeometry = "continuous" | "diurnal" | "polar_intermittent";

/** Mirror of `roverdevkit.schema.DesignVector`. */
export interface DesignVector {
  wheel_radius_m: number;
  wheel_width_m: number;
  grouser_height_m: number;
  grouser_count: number;
  n_wheels: 4 | 6;
  chassis_mass_kg: number;
  wheelbase_m: number;
  solar_area_m2: number;
  battery_capacity_wh: number;
  avionics_power_w: number;
  nominal_speed_mps: number;
  drive_duty_cycle: number;
}

/** Mirror of `roverdevkit.schema.MissionScenario`. */
export interface MissionScenario {
  name: string;
  latitude_deg: number;
  traverse_distance_m: number;
  terrain_class: TerrainClass;
  soil_simulant: string;
  mission_duration_earth_days: number;
  max_slope_deg: number;
  sun_geometry: SunGeometry;
}

export interface SoilParametersOut {
  simulant: string;
  n: number;
  k_c: number;
  k_phi: number;
  cohesion_kpa: number;
  friction_angle_deg: number;
  shear_modulus_k_m: number;
}

export interface ScenarioWithSoil {
  scenario: MissionScenario;
  soil: SoilParametersOut;
}

export interface ScenarioListResponse {
  scenarios: ScenarioWithSoil[];
}

export type PrimaryTarget =
  | "range_km"
  | "energy_margin_raw_pct"
  | "slope_capability_deg"
  | "total_mass_kg";

/**
 * Canonical row order used everywhere the four primary targets are
 * rendered. Matches `roverdevkit.surrogate.features.PRIMARY_REGRESSION_TARGETS`
 * so the Python and TypeScript layers agree by construction.
 */
export const PRIMARY_REGRESSION_TARGET_ORDER: readonly PrimaryTarget[] = [
  "range_km",
  "energy_margin_raw_pct",
  "slope_capability_deg",
  "total_mass_kg",
] as const;

export interface PredictTarget {
  target: PrimaryTarget;
  q05: number;
  q50: number;
  q95: number;
}

export interface FeatureRow {
  columns: string[];
  values: unknown[];
}

export interface PredictRequest {
  design: DesignVector;
  scenario_name: string;
  repair_crossings?: boolean;
}

export interface PredictResponse {
  scenario_name: string;
  quantiles: [number, number, number];
  predictions: PredictTarget[];
  feature_row: FeatureRow;
}

/**
 * Mirror of the FastAPI `EvaluateRequest`. Drives the deterministic
 * corrected mission evaluator on a single design × canonical scenario.
 * Used by the single-design panel as the source of truth for the
 * median value of each performance metric; the surrogate's quantile
 * heads supply the prediction-interval band around it.
 */
export interface EvaluateRequest {
  design: DesignVector;
  scenario_name: string;
}

export interface EvaluateMetric {
  target: PrimaryTarget;
  value: number;
}

/**
 * Mirror of the FastAPI `ThermalDiagnosticOut`. Every numeric field is
 * already in the user's display units (°C, W, m²) so the panel and
 * dialog can render them without conversion.
 */
export interface ThermalDiagnostic {
  survives: boolean;
  peak_sun_temp_c: number;
  lunar_night_temp_c: number;
  min_operating_temp_c: number;
  max_operating_temp_c: number;
  rhu_power_w: number;
  hibernation_power_w: number;
  surface_area_m2: number;
  hot_case_ok: boolean;
  cold_case_ok: boolean;
}

/** Mirror of the FastAPI `MotorTorqueDiagnosticOut`. */
export interface MotorTorqueDiagnostic {
  survives: boolean;
  peak_torque_nm: number;
  ceiling_nm: number;
  rover_stalled: boolean;
  torque_ok: boolean;
}

export interface EvaluateResponse {
  scenario_name: string;
  metrics: EvaluateMetric[];
  thermal: ThermalDiagnostic;
  motor_torque: MotorTorqueDiagnostic;
  used_scm_correction: boolean;
  elapsed_ms: number;
}

/**
 * Merged per-target row consumed by the chart and the panel table.
 *
 * - `value` is the deterministic median from the evaluator (ground truth).
 * - `q05`/`q95` are the surrogate's calibrated 90% prediction interval
 *   wrapping that median. Both may be `undefined` while the
 *   corresponding request is in flight or has failed.
 */
export interface PredictionRow {
  target: PrimaryTarget;
  value: number;
  q05: number | null;
  q95: number | null;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  surrogate_loaded: boolean;
  surrogate_targets: string[];
  quantile_bundles_path: string;
}

export interface VersionResponse {
  api_version: string;
  package_version: string;
  dataset_version: string;
  quantile_bundles_path: string;
}

/**
 * Mirror of the FastAPI `SweepAxisIn` schema. A sweep axis defines a
 * linearly-spaced grid `[lo, hi]` over a single design-vector field
 * with `n_points` cells (inclusive at both ends).
 */
export interface SweepAxisIn {
  variable: SweepableVariable;
  lo: number;
  hi: number;
  n_points: number;
}

/**
 * Subset of `DesignVector` keys the sweep page lets the user vary on
 * a grid axis. Mirrors `roverdevkit.tradespace.sweeps.SWEEPABLE_VARIABLES`;
 * `n_wheels` is excluded because it is binary.
 */
export type SweepableVariable =
  | "wheel_radius_m"
  | "wheel_width_m"
  | "grouser_height_m"
  | "grouser_count"
  | "chassis_mass_kg"
  | "wheelbase_m"
  | "solar_area_m2"
  | "battery_capacity_wh"
  | "avionics_power_w"
  | "nominal_speed_mps"
  | "drive_duty_cycle";

export const SWEEPABLE_VARIABLES: readonly SweepableVariable[] = [
  "wheel_radius_m",
  "wheel_width_m",
  "grouser_height_m",
  "grouser_count",
  "chassis_mass_kg",
  "wheelbase_m",
  "solar_area_m2",
  "battery_capacity_wh",
  "avionics_power_w",
  "nominal_speed_mps",
  "drive_duty_cycle",
] as const;

export type SweepBackend = "auto" | "evaluator" | "surrogate";

export interface SweepRequest {
  target: PrimaryTarget;
  x_axis: SweepAxisIn;
  y_axis?: SweepAxisIn | null;
  base_design: DesignVector;
  scenario_name: string;
  backend?: SweepBackend;
}

export interface SweepResponse {
  target: PrimaryTarget;
  scenario_name: string;
  x_variable: SweepableVariable;
  y_variable: SweepableVariable | null;
  x_values: number[];
  y_values: number[] | null;
  /** 1-D `(n_x,)` for a 1-D sweep, 2-D `(n_y, n_x)` for a 2-D sweep. */
  z_values: number[] | number[][];
  backend_used: "evaluator" | "surrogate";
  backend_requested: SweepBackend;
  used_scm_correction: boolean;
  n_cells: number;
  elapsed_ms: number;
  sensitivity: SweepSensitivity;
}

/**
 * Per-axis spread of the swept metric. Powers the inline sensitivity hint
 * shown under the chart so the user can quickly tell when a metric is
 * effectively flat across the chosen grid (saturation), or when one axis
 * dominates the other by an order of magnitude (visual masking).
 */
export interface SweepSensitivity {
  /** max(z) - min(z) over the whole grid, in target units. */
  total_spread: number;
  /** total_spread / max(|max|, |min|, eps); dimensionless. */
  relative_spread: number;
  /** Median marginal x-spread (1-D = total_spread). */
  axis_spread_x: number;
  /** Median marginal y-spread; null for 1-D sweeps. */
  axis_spread_y: number | null;
}

export interface RegistryEntrySummary {
  rover_name: string;
  is_flown: boolean;
  design: DesignVector;
  scenario: MissionScenario;
  gravity_m_per_s2: number;
  thermal_architecture: Record<string, unknown>;
  panel_efficiency: number;
  panel_dust_factor: number;
  imputation_notes: string;
}

export interface RegistryListResponse {
  rovers: RegistryEntrySummary[];
}

/**
 * Static design-space bounds, kept aligned with
 * `roverdevkit/schema.py::DesignVector`. The form uses these for
 * range validation, slider extents, and step sizes; if the Python
 * schema bounds change we update them here too (caught at runtime
 * by FastAPI's 422 response, but a same-day visual diff is nicer).
 */
export interface FieldBounds {
  min: number;
  max: number;
  step: number;
  unit: string;
  label: string;
  description: string;
}

export const DESIGN_BOUNDS: Record<keyof DesignVector, FieldBounds> = {
  wheel_radius_m: {
    min: 0.05,
    max: 0.2,
    step: 0.005,
    unit: "m",
    label: "Wheel radius",
    description: "R, mobility wheel radius.",
  },
  wheel_width_m: {
    min: 0.03,
    max: 0.2,
    step: 0.005,
    unit: "m",
    label: "Wheel width",
    description: "W, mobility wheel width.",
  },
  grouser_height_m: {
    min: 0.0,
    max: 0.02,
    step: 0.001,
    unit: "m",
    label: "Grouser height",
    description: "h_g, soil-engaging tooth height.",
  },
  grouser_count: {
    min: 0,
    max: 24,
    step: 1,
    unit: "",
    label: "Grouser count",
    description: "N_g, grousers per wheel.",
  },
  n_wheels: {
    min: 4,
    max: 6,
    step: 2,
    unit: "",
    label: "Wheel count",
    description: "N_w, mobility wheel count (4 or 6).",
  },
  chassis_mass_kg: {
    min: 3,
    max: 50,
    step: 0.5,
    unit: "kg",
    label: "Chassis mass",
    description:
      "m_c, dry chassis mass (subsystem masses are added by the model).",
  },
  wheelbase_m: {
    min: 0.3,
    max: 1.2,
    step: 0.05,
    unit: "m",
    label: "Wheelbase",
    description: "L_wb, longitudinal wheel separation.",
  },
  solar_area_m2: {
    min: 0.1,
    max: 1.5,
    step: 0.05,
    unit: "m^2",
    label: "Solar area",
    description: "A_s, deployable solar array area.",
  },
  battery_capacity_wh: {
    min: 20,
    max: 500,
    step: 5,
    unit: "Wh",
    label: "Battery capacity",
    description: "C_b, usable battery capacity.",
  },
  avionics_power_w: {
    min: 5,
    max: 40,
    step: 0.5,
    unit: "W",
    label: "Avionics power",
    description: "P_a, continuous avionics draw.",
  },
  nominal_speed_mps: {
    min: 0.01,
    max: 0.1,
    step: 0.005,
    unit: "m/s",
    label: "Nominal drive speed",
    description: "v, drive speed when commanded.",
  },
  drive_duty_cycle: {
    min: 0.02,
    max: 0.6,
    step: 0.01,
    unit: "",
    label: "Drive duty cycle",
    description: "δ, fraction of mission spent driving.",
  },
};

/** User-facing display metadata for the four predicted performance metrics. */
export const TARGET_META: Record<
  PrimaryTarget,
  { label: string; unit: string; description: string }
> = {
  range_km: {
    label: "Range",
    unit: "km",
    description:
      "Distance the rover can traverse during the scenario at its commanded drive duty cycle.",
  },
  energy_margin_raw_pct: {
    label: "Energy margin",
    unit: "%",
    description:
      "(Energy generated − energy used) / energy used over the traverse. Positive values mean surplus solar generation; large positive values are typical for energy-rich designs.",
  },
  slope_capability_deg: {
    label: "Slope capability",
    unit: "deg",
    description:
      "Steepest slope the rover can sustain on the scenario's soil at its commanded speed.",
  },
  total_mass_kg: {
    label: "Total mass",
    unit: "kg",
    description:
      "Estimated dry mass of the rover, including chassis, motors, structure, power, and avionics.",
  },
};
