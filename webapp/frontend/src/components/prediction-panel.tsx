import { Loader2 } from "lucide-react";

import { PredictionChart } from "@/components/prediction-chart";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import type { PredictResponse } from "@/types/api";
import { TARGET_META, type PrimaryTarget } from "@/types/api";

interface PredictionPanelProps {
  data: PredictResponse | undefined;
  isPending: boolean;
  error: Error | null;
}

/** Right-hand panel: chart + numeric summary table for the latest prediction. */
export function PredictionPanel({
  data,
  isPending,
  error,
}: PredictionPanelProps) {
  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle>Surrogate prediction</CardTitle>
        <CardDescription>
          Median (♦) and 90 % prediction interval from the W8 step-4
          quantile-XGBoost surrogate trained on{" "}
          <code className="text-xs">lhs_v4</code> (40 k corrected-evaluator
          rows).
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {isPending ? (
          <div className="flex items-center gap-2 text-sm text-[var(--color-muted-foreground)]">
            <Loader2 className="h-4 w-4 animate-spin" />
            Calling /predict…
          </div>
        ) : null}

        {error ? (
          <div className="rounded border border-[var(--color-destructive)] bg-[var(--color-destructive)]/5 p-3 text-sm text-[var(--color-destructive)]">
            {error.message}
          </div>
        ) : null}

        {data ? (
          <>
            <PredictionChart predictions={data.predictions} />
            <div className="overflow-hidden rounded-md border">
              <table className="w-full text-sm">
                <thead className="bg-[var(--color-muted)]/40">
                  <tr>
                    <th className="px-3 py-2 text-left font-medium">Target</th>
                    <th className="px-3 py-2 text-right font-medium">q₀₅</th>
                    <th className="px-3 py-2 text-right font-medium">median</th>
                    <th className="px-3 py-2 text-right font-medium">q₉₅</th>
                  </tr>
                </thead>
                <tbody>
                  {data.predictions.map((p) => {
                    const meta = TARGET_META[p.target as PrimaryTarget];
                    return (
                      <tr key={p.target} className="border-t">
                        <td className="px-3 py-2">
                          <div className="font-medium">{meta.label}</div>
                          <div className="text-xs text-[var(--color-muted-foreground)]">
                            {meta.description}
                          </div>
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          {fmt(p.q05)} {meta.unit}
                        </td>
                        <td className="px-3 py-2 text-right font-semibold tabular-nums">
                          {fmt(p.q50)} {meta.unit}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          {fmt(p.q95)} {meta.unit}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </>
        ) : isPending ? null : (
          <p className="text-sm text-[var(--color-muted-foreground)]">
            Submit a design vector on the left to evaluate it.
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function fmt(x: number): string {
  if (!Number.isFinite(x)) return "n/a";
  if (Math.abs(x) >= 100) return x.toFixed(1);
  if (Math.abs(x) >= 1) return x.toFixed(2);
  return x.toFixed(3);
}
