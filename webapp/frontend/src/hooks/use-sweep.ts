import { useMutation } from "@tanstack/react-query";

import { api } from "@/lib/api";
import type { SweepRequest, SweepResponse } from "@/types/api";

/**
 * `POST /sweep` mutation. Used by the parametric-sweep page to fetch
 * a 1-D line or 2-D heatmap of one performance metric over a grid of
 * one (or two) design-vector fields. The backend picks evaluator vs
 * surrogate based on grid size unless the user forces a backend.
 */
export function useSweep() {
  return useMutation<SweepResponse, Error, SweepRequest>({
    mutationKey: ["sweep"],
    mutationFn: (req) => api.sweep(req),
  });
}
