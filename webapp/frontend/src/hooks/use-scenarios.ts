import { useQuery } from "@tanstack/react-query";

import { api } from "@/lib/api";

/** Cached fetch of the canonical four scenarios. Stale after 5 min. */
export function useScenarios() {
  return useQuery({
    queryKey: ["scenarios"],
    queryFn: () => api.listScenarios(),
    staleTime: 5 * 60 * 1000,
  });
}
