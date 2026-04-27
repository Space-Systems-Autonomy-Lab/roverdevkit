import { useQueries, useQuery } from "@tanstack/react-query";

import { api } from "@/lib/api";
import type { DesignVector, PredictResponse } from "@/types/api";

/**
 * Lazily fetched real-rover registry. The list is small and immutable
 * for the lifetime of the backend, so we cache forever and never
 * refetch on focus.
 */
export function useRegistry() {
  return useQuery({
    queryKey: ["registry"],
    queryFn: () => api.listRegistry(),
    staleTime: Infinity,
    gcTime: Infinity,
    refetchOnWindowFocus: false,
  });
}

/**
 * Per-rover prediction queries used by the overlay. Each rover is a
 * separate query so cache hits are reused across re-renders and
 * overlay toggles. We pass the user's currently selected scenario so
 * the overlay is an apples-to-apples comparison: "Pragyan, run on the
 * same mission you just configured." A future revision can offer a
 * toggle for "use the rover's own published scenario instead."
 */
export function useRegistryPredictions(
  rovers: Array<{ rover_name: string; design: DesignVector }>,
  scenarioName: string,
) {
  return useQueries({
    queries: rovers.map((r) => ({
      queryKey: [
        "registry-predict",
        r.rover_name,
        scenarioName,
        // Hash a stable shape of the design too in case a future
        // revision lets the user edit registry rover designs.
        JSON.stringify(r.design),
      ] as const,
      queryFn: (): Promise<PredictResponse> =>
        api.predict({ design: r.design, scenario_name: scenarioName }),
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
