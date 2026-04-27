import { Loader2 } from "lucide-react";
import type { ReactNode } from "react";

import { ConstraintDetailsButton } from "@/components/constraint-details-dialog";
import {
  PredictionChart,
  type OverlayPrediction,
} from "@/components/prediction-chart";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import type {
  MotorTorqueDiagnostic,
  PredictionRow,
  ThermalDiagnostic,
} from "@/types/api";
import { TARGET_META } from "@/types/api";

export interface PredictionPanelMeta {
  /** Whether the SCM-corrected physics evaluator was used (vs BW-only fallback). */
  used_scm_correction: boolean;
  /** Mission-evaluator wall-clock in milliseconds (for the user's current design). */
  evaluator_ms: number;
  /**
   * Structured constraint diagnostics from the deterministic evaluator.
   * Used by the footer chips to show pass/fail and to back the
   * click-for-details dialog explaining *why* a flag fired.
   */
  thermal: ThermalDiagnostic;
  motor_torque: MotorTorqueDiagnostic;
}

interface PredictionPanelProps {
  /** Merged rows: evaluator median + (optional) surrogate q05/q95. */
  rows: PredictionRow[] | undefined;
  meta: PredictionPanelMeta | undefined;
  /** Whether either request (evaluate or predict) is in flight. */
  isPending: boolean;
  /** Highest-priority error from the evaluator or surrogate calls. */
  error: Error | null;
  /** Whether the surrogate's PI band is still loading after the evaluator returned. */
  surrogatePending?: boolean;
  overlays?: OverlayPrediction[];
  overlayLoading?: boolean;
}

/**
 * Right-hand panel: chart + numeric summary table for the latest prediction.
 *
 * The median value is the deterministic output of the corrected mission
 * evaluator (BW + wheel-level SCM correction); the q05/q95 columns and
 * the chart's blue bars are the surrogate's calibrated 90% prediction
 * interval wrapping that median.
 */
export function PredictionPanel({
  rows,
  meta,
  isPending,
  error,
  surrogatePending = false,
  overlays = [],
  overlayLoading = false,
}: PredictionPanelProps) {
  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle>Predicted performance</CardTitle>
        <CardDescription>
          Median (♦) is the physics evaluator&rsquo;s deterministic output;
          the blue bar shows the surrogate&rsquo;s calibrated 90% prediction
          interval around it.
          {overlays.length > 0 ? (
            <>
              {" "}
              Coloured circles show the same evaluator output for selected
              real rovers run on this scenario, for comparison.
            </>
          ) : null}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {isPending ? (
          <div className="flex items-center gap-2 text-sm text-[var(--color-muted-foreground)]">
            <Loader2 className="h-4 w-4 animate-spin" />
            Evaluating mission&hellip;
          </div>
        ) : null}

        {error ? (
          <div className="rounded border border-[var(--color-destructive)] bg-[var(--color-destructive)]/5 p-3 text-sm text-[var(--color-destructive)]">
            {error.message}
          </div>
        ) : null}

        {rows && meta ? (
          <>
            <PredictionChart rows={rows} overlays={overlays} />
            {surrogatePending ? (
              <div className="flex items-center gap-2 text-xs text-[var(--color-muted-foreground)]">
                <Loader2 className="h-3 w-3 animate-spin" />
                Loading prediction interval&hellip;
              </div>
            ) : null}
            {overlayLoading ? (
              <div className="flex items-center gap-2 text-xs text-[var(--color-muted-foreground)]">
                <Loader2 className="h-3 w-3 animate-spin" />
                Loading rover comparisons&hellip;
              </div>
            ) : null}
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
                  {rows.map((row) => {
                    const m = TARGET_META[row.target];
                    return (
                      <tr key={row.target} className="border-t">
                        <td className="px-3 py-2">
                          <div className="font-medium">{m.label}</div>
                          <div className="text-xs text-[var(--color-muted-foreground)]">
                            {m.description}
                          </div>
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          {row.q05 === null ? "—" : `${fmt(row.q05)} ${m.unit}`}
                        </td>
                        <td className="px-3 py-2 text-right font-semibold tabular-nums">
                          {fmt(row.value)} {m.unit}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          {row.q95 === null ? "—" : `${fmt(row.q95)} ${m.unit}`}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <PanelFooter meta={meta} />
          </>
        ) : isPending ? null : (
          <p className="text-sm text-[var(--color-muted-foreground)]">
            Configure a rover on the left and click <em>Predict performance</em>{" "}
            to evaluate it.
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function PanelFooter({ meta }: { meta: PredictionPanelMeta }) {
  const thermalOk = meta.thermal.survives;
  const motorOk = meta.motor_torque.survives;
  const allOk = thermalOk && motorOk;
  return (
    <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-[var(--color-muted-foreground)]">
      <span>
        Evaluator: {meta.evaluator_ms.toFixed(0)} ms
        {meta.used_scm_correction ? " · SCM-corrected" : " · Bekker–Wong only"}
      </span>
      {allOk ? (
        <span className="flex flex-wrap items-center gap-2">
          <ConstraintChip
            label="thermal survival ok"
            ok
            details={
              <ConstraintDetailsButton
                variant="thermal"
                thermal={meta.thermal}
                motorTorque={meta.motor_torque}
                failed={false}
              />
            }
          />
          <ConstraintChip
            label="motor torque ok"
            ok
            details={
              <ConstraintDetailsButton
                variant="motor_torque"
                thermal={meta.thermal}
                motorTorque={meta.motor_torque}
                failed={false}
              />
            }
          />
        </span>
      ) : (
        <span className="flex flex-wrap items-center gap-2">
          {!thermalOk ? (
            <ConstraintChip
              label="thermal survival fails"
              ok={false}
              details={
                <ConstraintDetailsButton
                  variant="thermal"
                  thermal={meta.thermal}
                  motorTorque={meta.motor_torque}
                  failed
                />
              }
            />
          ) : null}
          {!motorOk ? (
            <ConstraintChip
              label="motor torque exceeded"
              ok={false}
              details={
                <ConstraintDetailsButton
                  variant="motor_torque"
                  thermal={meta.thermal}
                  motorTorque={meta.motor_torque}
                  failed
                />
              }
            />
          ) : null}
        </span>
      )}
    </div>
  );
}

function ConstraintChip({
  label,
  ok,
  details,
}: {
  label: string;
  ok: boolean;
  details: ReactNode;
}) {
  const cls = ok
    ? "inline-flex items-center gap-1.5 rounded-full bg-emerald-500/10 px-2 py-0.5 text-emerald-700"
    : "inline-flex items-center gap-1.5 rounded-full bg-[var(--color-destructive)]/10 px-2 py-0.5 text-[var(--color-destructive)]";
  return (
    <span className={cls}>
      {label}
      {details}
    </span>
  );
}

function fmt(x: number): string {
  if (!Number.isFinite(x)) return "n/a";
  if (Math.abs(x) >= 100) return x.toFixed(1);
  if (Math.abs(x) >= 1) return x.toFixed(2);
  return x.toFixed(3);
}
