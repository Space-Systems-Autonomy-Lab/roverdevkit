import { useQuery } from "@tanstack/react-query";

import { api } from "@/lib/api";

/** Liveness + version probe, polled lazily once on app mount. */
export function useHealth() {
  return useQuery({
    queryKey: ["healthz"],
    queryFn: () => api.healthz(),
    staleTime: 60 * 1000,
  });
}

export function useVersion() {
  return useQuery({
    queryKey: ["version"],
    queryFn: () => api.version(),
    staleTime: 60 * 60 * 1000,
  });
}
