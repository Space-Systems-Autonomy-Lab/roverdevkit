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
    description: "m_c, dry chassis mass before mass-up.",
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

/** Display metadata for the four primary surrogate targets. */
export const TARGET_META: Record<
  PrimaryTarget,
  { label: string; unit: string; description: string }
> = {
  range_km: {
    label: "Range",
    unit: "km",
    description:
      "Capability-at-designed-duty traverse over the scenario window.",
  },
  energy_margin_raw_pct: {
    label: "Energy margin (raw)",
    unit: "%",
    description: "Unbounded (E_gen − E_used) / E_used over the traverse.",
  },
  slope_capability_deg: {
    label: "Slope capability",
    unit: "deg",
    description: "Maximum sustainable slope at the sampled soil.",
  },
  total_mass_kg: {
    label: "Total mass",
    unit: "kg",
    description: "Mass-up estimate including motors, structure, payload.",
  },
};
