import { useMutation, useQueries } from "@tanstack/react-query";

import { api } from "@/lib/api";
import type {
  DesignVector,
  EvaluateRequest,
  EvaluateResponse,
} from "@/types/api";

/**
 * `POST /evaluate` mutation. Returns the deterministic corrected
 * mission evaluator's output for a single design × scenario. The
 * design panel triggers this in parallel with `usePredict()` so the
 * chart can show the evaluator's value as the median diamond and the
 * surrogate's q05/q95 as the prediction-interval band wrapping it.
 */
export function useEvaluate() {
  return useMutation<EvaluateResponse, Error, EvaluateRequest>({
    mutationKey: ["evaluate"],
    mutationFn: (req) => api.evaluate(req),
  });
}

/**
 * Per-rover evaluator queries used by the registry overlay. Mirrors
 * `useRegistryPredictions` but hits `/evaluate` so the overlay
 * markers are ground-truth physics outputs (matching the candidate
 * design's median diamond) rather than surrogate regressions.
 *
 * Each rover is its own query so cache hits are reused across
 * re-renders and overlay toggles.
 */
export function useRegistryEvaluations(
  rovers: Array<{ rover_name: string; design: DesignVector }>,
  scenarioName: string,
) {
  return useQueries({
    queries: rovers.map((r) => ({
      queryKey: [
        "registry-evaluate",
        r.rover_name,
        scenarioName,
        JSON.stringify(r.design),
      ] as const,
      queryFn: (): Promise<EvaluateResponse> =>
        api.evaluate({ design: r.design, scenario_name: scenarioName }),
      staleTime: Infinity,
      gcTime: 60 * 60 * 1000,
      refetchOnWindowFocus: false,
    })),
    combine: (results) => ({
      results,
      isPending: results.some((r) => r.isPending),
      isError: results.some((r) => r.isError),
    }),
  });
}
