import type { Data, Layout } from "plotly.js";

import Plot from "@/lib/plotly";
import type { PredictionRow, PrimaryTarget } from "@/types/api";
import { TARGET_META } from "@/types/api";

/** Per-target overlay value for a real rover, projected onto the same metric grid. */
export interface OverlayMetric {
  target: PrimaryTarget;
  value: number;
}

export interface OverlayPrediction {
  /** Display name from the registry. */
  rover_name: string;
  /** Marker color (hex or rgba string). */
  color: string;
  /** Deterministic evaluator output, one entry per primary target. */
  metrics: OverlayMetric[];
}

interface PredictionChartProps {
  /** Merged rows: evaluator median + (optional) surrogate q05/q95. */
  rows: PredictionRow[];
  overlays?: OverlayPrediction[];
}

/**
 * Horizontal "median + 90 % PI" chart, one row per primary target.
 *
 * Each row shows the deterministic evaluator output as a diamond
 * marker (the candidate design's median) plus, when the surrogate's
 * quantile heads have returned, a horizontal line from q05 to q95
 * representing the calibrated 90 % prediction interval. Optional
 * coloured circles per overlay rover sit on the same axis at each
 * rover's evaluator-computed value, so candidate-vs-flown
 * comparisons are apples-to-apples ground truth.
 *
 * Targets get separate x-axes because their units don't commensurate
 * (km vs % vs deg vs kg).
 */
export function PredictionChart({ rows, overlays = [] }: PredictionChartProps) {
  if (rows.length === 0) return null;

  const traces: Data[] = [];
  const layout: Partial<Layout> = {
    grid: { rows: rows.length, columns: 1, pattern: "independent" },
    showlegend: overlays.length > 0,
    legend: {
      orientation: "h",
      y: -0.15,
      x: 0,
      xanchor: "left",
      yanchor: "top",
      bgcolor: "rgba(0,0,0,0)",
    },
    height: 100 + rows.length * 80 + (overlays.length > 0 ? 40 : 0),
    margin: { l: 170, r: 30, t: 20, b: 30 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "ui-sans-serif, system-ui, sans-serif", size: 12 },
  };

  rows.forEach((row, idx) => {
    const meta = TARGET_META[row.target];
    const xref = idx === 0 ? "x" : (`x${idx + 1}` as const);
    const yref = idx === 0 ? "y" : (`y${idx + 1}` as const);
    const xaxisKey = idx === 0 ? "xaxis" : (`xaxis${idx + 1}` as const);
    const yaxisKey = idx === 0 ? "yaxis" : (`yaxis${idx + 1}` as const);
    const label = `${meta.label} (${meta.unit || "·"})`;

    // PI bar (only when both q05 and q95 are available).
    if (row.q05 !== null && row.q95 !== null) {
      traces.push({
        type: "scatter",
        mode: "lines",
        x: [row.q05, row.q95],
        y: [label, label],
        xaxis: xref,
        yaxis: yref,
        showlegend: false,
        line: { color: "rgba(65, 105, 225, 0.55)", width: 6 },
        hovertemplate: `90 %% PI: [${fmt(row.q05)}, ${fmt(row.q95)}] ${meta.unit}<extra></extra>`,
      });
    }
    // Candidate median diamond (evaluator's deterministic value).
    traces.push({
      type: "scatter",
      mode: "markers",
      x: [row.value],
      y: [label],
      xaxis: xref,
      yaxis: yref,
      name: "Your design",
      // Only show the candidate in the legend on the first row to avoid
      // four duplicate entries.
      showlegend: idx === 0 && overlays.length > 0,
      legendgroup: "candidate",
      marker: {
        symbol: "diamond",
        size: 14,
        color: "rgba(40, 75, 180, 1)",
        line: { color: "white", width: 1.5 },
      },
      hovertemplate: `your design · ${fmt(row.value)} ${meta.unit}<extra></extra>`,
    });

    // One marker per overlay at this target's evaluator value.
    overlays.forEach((overlay) => {
      const overlayMetric = overlay.metrics.find(
        (om) => om.target === row.target,
      );
      if (!overlayMetric) return;
      traces.push({
        type: "scatter",
        mode: "markers",
        x: [overlayMetric.value],
        y: [label],
        xaxis: xref,
        yaxis: yref,
        name: overlay.rover_name,
        showlegend: idx === 0,
        legendgroup: overlay.rover_name,
        marker: {
          symbol: "circle",
          size: 11,
          color: overlay.color,
          line: { color: "white", width: 1.5 },
        },
        hovertemplate: `${overlay.rover_name}: ${fmt(overlayMetric.value)} ${meta.unit}<extra></extra>`,
      });
    });

    (layout as Record<string, unknown>)[xaxisKey] = {
      automargin: true,
      gridcolor: "rgba(0,0,0,0.08)",
      zerolinecolor: "rgba(0,0,0,0.15)",
      ticks: "outside",
      ticklen: 4,
    };
    (layout as Record<string, unknown>)[yaxisKey] = {
      automargin: true,
      ticks: "",
      showgrid: false,
    };
  });

  return (
    <Plot
      data={traces}
      layout={layout}
      style={{ width: "100%" }}
      useResizeHandler
      config={{ displayModeBar: false, responsive: true }}
    />
  );
}

function fmt(x: number): string {
  if (!Number.isFinite(x)) return "n/a";
  if (Math.abs(x) >= 100) return x.toFixed(1);
  if (Math.abs(x) >= 1) return x.toFixed(2);
  return x.toFixed(3);
}
