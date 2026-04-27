import { useMutation } from "@tanstack/react-query";

import { api } from "@/lib/api";
import type { PredictRequest, PredictResponse } from "@/types/api";

/**
 * `POST /predict` mutation. Kept as a mutation rather than a query
 * so the user explicitly drives evaluations from the form rather
 * than triggering a request on every keystroke.
 */
export function usePredict() {
  return useMutation<PredictResponse, Error, PredictRequest>({
    mutationKey: ["predict"],
    mutationFn: (req) => api.predict(req),
  });
}
