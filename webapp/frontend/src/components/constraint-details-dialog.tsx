import { Info } from "lucide-react";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import type {
  MotorTorqueDiagnostic,
  ThermalDiagnostic,
} from "@/types/api";

/**
 * "Why did this constraint fire?" dialog opened from the panel chips.
 *
 * The footer in `prediction-panel.tsx` renders one trigger per failed
 * constraint; this component is the dialog body for either thermal or
 * motor torque. Both diagnostics arrive from `/evaluate` so the
 * numbers shown here are deterministic ground truth (not surrogate
 * predictions).
 */
export function ConstraintDetailsButton({
  variant,
  thermal,
  motorTorque,
  failed,
}: {
  variant: "thermal" | "motor_torque";
  thermal: ThermalDiagnostic;
  motorTorque: MotorTorqueDiagnostic;
  failed: boolean;
}) {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <button
          type="button"
          aria-label={`${variant === "thermal" ? "Thermal" : "Motor torque"} details`}
          className={
            "inline-flex h-4 w-4 items-center justify-center rounded-full text-current/70 hover:text-current focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-ring)]"
          }
        >
          <Info className="h-3.5 w-3.5" aria-hidden />
        </button>
      </DialogTrigger>
      <DialogContent>
        {variant === "thermal" ? (
          <ThermalBody thermal={thermal} failed={failed} />
        ) : (
          <MotorTorqueBody mt={motorTorque} failed={failed} />
        )}
      </DialogContent>
    </Dialog>
  );
}

function ThermalBody({
  thermal,
  failed,
}: {
  thermal: ThermalDiagnostic;
  failed: boolean;
}) {
  const rows: {
    label: string;
    temp: number;
    limit: number;
    ok: boolean;
    direction: "above" | "below";
    description: string;
  }[] = [
    {
      label: "Hot case · peak sun",
      temp: thermal.peak_sun_temp_c,
      limit: thermal.max_operating_temp_c,
      ok: thermal.hot_case_ok,
      direction: "above",
      description:
        "Steady-state interior temperature with the sun at its peak elevation for the scenario latitude and avionics drawing nominal operating power.",
    },
    {
      label: "Cold case · lunar night",
      temp: thermal.lunar_night_temp_c,
      limit: thermal.min_operating_temp_c,
      ok: thermal.cold_case_ok,
      direction: "below",
      description:
        "Steady-state interior temperature during lunar night with the rover hibernating (~2 W) and any RHU dissipation. No solar input.",
    },
  ];

  return (
    <>
      <DialogHeader>
        <DialogTitle>
          Thermal survival —{" "}
          {failed ? (
            <span className="text-[var(--color-destructive)]">fails</span>
          ) : (
            <span className="text-emerald-700">passes</span>
          )}
        </DialogTitle>
        <DialogDescription>
          Single-node radiative balance for the avionics enclosure at
          steady state. The rover survives only if the hot case stays
          under the operating ceiling <em>and</em> the cold case stays
          above the operating floor.
        </DialogDescription>
      </DialogHeader>

      <div className="overflow-hidden rounded-md border">
        <table className="w-full text-sm">
          <thead className="bg-[var(--color-muted)]/40">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Case</th>
              <th className="px-3 py-2 text-right font-medium">Temperature</th>
              <th className="px-3 py-2 text-right font-medium">Limit</th>
              <th className="px-3 py-2 text-right font-medium">Status</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.label} className="border-t align-top">
                <td className="px-3 py-2">
                  <div className="font-medium">{row.label}</div>
                  <div className="text-xs text-[var(--color-muted-foreground)]">
                    {row.description}
                  </div>
                </td>
                <td className="px-3 py-2 text-right tabular-nums">
                  {fmt1(row.temp)} °C
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-[var(--color-muted-foreground)]">
                  {row.direction === "above" ? "≤ " : "≥ "}
                  {fmt1(row.limit)} °C
                </td>
                <td className="px-3 py-2 text-right">
                  <span
                    className={
                      row.ok
                        ? "rounded-full bg-emerald-500/10 px-2 py-0.5 text-xs text-emerald-700"
                        : "rounded-full bg-[var(--color-destructive)]/10 px-2 py-0.5 text-xs text-[var(--color-destructive)]"
                    }
                  >
                    {row.ok ? "ok" : "fails"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="space-y-2 text-xs text-[var(--color-muted-foreground)]">
        <p>
          <span className="font-medium text-[var(--color-foreground)]">
            How it&rsquo;s computed.
          </span>{" "}
          Closed-form Stefan&ndash;Boltzmann balance:{" "}
          <code>T = (T_sink⁴ + Q_in / (ε σ A))^(1/4)</code>. The hot case
          uses absorbed solar power plus avionics; the cold case uses
          hibernation power plus any RHU. Sink temperature is 250 K hot,
          100 K cold; ε = 0.85, α = 0.3, sunlit-area fraction 0.25.
        </p>
        <p>
          <span className="font-medium text-[var(--color-foreground)]">
            Assumed thermal hardware.
          </span>{" "}
          Surface area ≈ {fmt2(thermal.surface_area_m2)} m² (rebuilt from
          the chassis-mass cube-root proxy). Hibernation power{" "}
          {fmt1(thermal.hibernation_power_w)} W. RHU power{" "}
          {fmt1(thermal.rhu_power_w)} W.
        </p>
        {!thermal.cold_case_ok ? (
          <p>
            <span className="font-medium text-[var(--color-foreground)]">
              Why this design fails the cold case.
            </span>{" "}
            With 0 W of RHU power and only{" "}
            {fmt1(thermal.hibernation_power_w)} W of hibernation
            heating, the enclosure radiates to ~
            {fmt1(thermal.lunar_night_temp_c)} °C during lunar night —
            below the {fmt1(thermal.min_operating_temp_c)} °C operating
            floor. Real lunar micro-rovers (Pragyan, Yutu&ndash;2,
            Rashid&ndash;1, MoonRanger) carry RHUs or supplemental
            heaters precisely to close this gap. RHU mass is not part
            of the design vector in this study, so we expose this as a
            diagnostic flag rather than a free design lever.
          </p>
        ) : null}
        {!thermal.hot_case_ok ? (
          <p>
            <span className="font-medium text-[var(--color-foreground)]">
              Why this design fails the hot case.
            </span>{" "}
            Peak-sun absorbed power plus avionics dissipation drives the
            enclosure to ~{fmt1(thermal.peak_sun_temp_c)} °C — above
            the {fmt1(thermal.max_operating_temp_c)} °C operating
            ceiling. Reducing avionics power, lowering solar
            absorptivity, or adding radiator area would bring the hot
            case down.
          </p>
        ) : null}
      </div>
    </>
  );
}

function MotorTorqueBody({
  mt,
  failed,
}: {
  mt: MotorTorqueDiagnostic;
  failed: boolean;
}) {
  const margin = mt.peak_torque_nm > 0 ? mt.ceiling_nm - mt.peak_torque_nm : 0;
  return (
    <>
      <DialogHeader>
        <DialogTitle>
          Motor torque —{" "}
          {failed ? (
            <span className="text-[var(--color-destructive)]">fails</span>
          ) : (
            <span className="text-emerald-700">passes</span>
          )}
        </DialogTitle>
        <DialogDescription>
          Compares the peak per-wheel torque observed during the
          traverse to the closed-form sizing ceiling the mass model
          assumed when sizing the drive motors.
        </DialogDescription>
      </DialogHeader>

      <div className="overflow-hidden rounded-md border">
        <table className="w-full text-sm">
          <thead className="bg-[var(--color-muted)]/40">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Quantity</th>
              <th className="px-3 py-2 text-right font-medium">Value</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-t">
              <td className="px-3 py-2">
                <div className="font-medium">Peak per-wheel torque</div>
                <div className="text-xs text-[var(--color-muted-foreground)]">
                  Largest absolute torque observed during traverse simulation.
                </div>
              </td>
              <td className="px-3 py-2 text-right tabular-nums">
                {fmt2(mt.peak_torque_nm)} N·m
              </td>
            </tr>
            <tr className="border-t">
              <td className="px-3 py-2">
                <div className="font-medium">Sizing ceiling</div>
                <div className="text-xs text-[var(--color-muted-foreground)]">
                  <code>sf · μ · (m·g/N) · R</code> — safety factor ×
                  peak friction × per-wheel weight × wheel radius.
                </div>
              </td>
              <td className="px-3 py-2 text-right tabular-nums">
                {fmt2(mt.ceiling_nm)} N·m
              </td>
            </tr>
            <tr className="border-t">
              <td className="px-3 py-2 font-medium">Margin (ceiling − peak)</td>
              <td
                className={
                  "px-3 py-2 text-right tabular-nums " +
                  (mt.torque_ok ? "text-emerald-700" : "text-[var(--color-destructive)]")
                }
              >
                {fmt2(margin)} N·m
              </td>
            </tr>
            <tr className="border-t">
              <td className="px-3 py-2 font-medium">Rover stalled?</td>
              <td className="px-3 py-2 text-right">
                <span
                  className={
                    mt.rover_stalled
                      ? "rounded-full bg-[var(--color-destructive)]/10 px-2 py-0.5 text-xs text-[var(--color-destructive)]"
                      : "rounded-full bg-emerald-500/10 px-2 py-0.5 text-xs text-emerald-700"
                  }
                >
                  {mt.rover_stalled ? "yes" : "no"}
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="space-y-2 text-xs text-[var(--color-muted-foreground)]">
        <p>
          The constraint fires if peak torque exceeds the ceiling
          <em> or</em> the traverse loop ever stalls (zero forward
          progress on a slope). To make a borderline design pass:
          increase wheel radius, add wheels, or reduce total mass.
        </p>
      </div>
    </>
  );
}

function fmt1(x: number): string {
  return Number.isFinite(x) ? x.toFixed(1) : "n/a";
}

function fmt2(x: number): string {
  return Number.isFinite(x) ? x.toFixed(2) : "n/a";
}
