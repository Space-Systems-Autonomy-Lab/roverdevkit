import { create } from "zustand";

import type { DesignVector, ScenarioName } from "@/types/api";

/**
 * Local UI state for the single-design panel.
 *
 * Pulled into Zustand rather than React component state so future
 * panels (sweep, Pareto explorer) can share the same "currently
 * selected design + scenario" without prop-drilling. TanStack Query
 * still owns *server* state (scenarios list, predict response, etc.).
 */

/**
 * Default design vector used as the form's starting point.
 *
 * Rough Yutu-2 / mid-class lunar micro-rover (matches the backend's
 * predict-test fixture). All twelve fields sit comfortably inside
 * the v3-widened LHS bounds so the surrogate sees an in-distribution
 * input on first render.
 */
export const DEFAULT_DESIGN: DesignVector = {
  wheel_radius_m: 0.1,
  wheel_width_m: 0.1,
  grouser_height_m: 0.012,
  grouser_count: 14,
  n_wheels: 6,
  chassis_mass_kg: 20,
  wheelbase_m: 0.6,
  solar_area_m2: 0.5,
  battery_capacity_wh: 100,
  avionics_power_w: 15,
  nominal_speed_mps: 0.04,
  drive_duty_cycle: 0.15,
};

interface DesignState {
  design: DesignVector;
  scenarioName: ScenarioName;
  setDesignField: <K extends keyof DesignVector>(
    key: K,
    value: DesignVector[K],
  ) => void;
  setDesign: (design: DesignVector) => void;
  setScenario: (name: ScenarioName) => void;
  resetDesign: () => void;
}

export const useDesignStore = create<DesignState>()((set) => ({
  design: DEFAULT_DESIGN,
  scenarioName: "equatorial_mare_traverse",
  setDesignField: (key, value) =>
    set((state) => ({ design: { ...state.design, [key]: value } })),
  setDesign: (design) => set({ design }),
  setScenario: (name) => set({ scenarioName: name }),
  resetDesign: () => set({ design: DEFAULT_DESIGN }),
}));
