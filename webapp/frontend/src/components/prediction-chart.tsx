import type { Data, Layout } from "plotly.js";

import Plot from "@/lib/plotly";
import type { PredictTarget } from "@/types/api";
import { TARGET_META, type PrimaryTarget } from "@/types/api";

interface PredictionChartProps {
  predictions: PredictTarget[];
}

/**
 * Horizontal "median + 90 % PI" chart, one row per primary target.
 *
 * Each row is a horizontal line from q05 to q95 with a diamond
 * marker at q50. Targets get separate x-axes because their units
 * don't commensurate (km vs % vs deg vs kg). All layout properties
 * are hand-typed against `@types/plotly.js`.
 */
export function PredictionChart({ predictions }: PredictionChartProps) {
  if (predictions.length === 0) return null;

  const traces: Data[] = [];
  const layout: Partial<Layout> = {
    grid: { rows: predictions.length, columns: 1, pattern: "independent" },
    showlegend: false,
    height: 100 + predictions.length * 80,
    margin: { l: 170, r: 30, t: 20, b: 30 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "ui-sans-serif, system-ui, sans-serif", size: 12 },
  };

  predictions.forEach((p, idx) => {
    const meta = TARGET_META[p.target as PrimaryTarget];
    const xref = idx === 0 ? "x" : (`x${idx + 1}` as const);
    const yref = idx === 0 ? "y" : (`y${idx + 1}` as const);
    const xaxisKey = idx === 0 ? "xaxis" : (`xaxis${idx + 1}` as const);
    const yaxisKey = idx === 0 ? "yaxis" : (`yaxis${idx + 1}` as const);
    const label = `${meta.label} (${meta.unit || "·"})`;

    traces.push({
      type: "scatter",
      mode: "lines",
      x: [p.q05, p.q95],
      y: [label, label],
      xaxis: xref,
      yaxis: yref,
      line: { color: "rgba(65, 105, 225, 0.55)", width: 6 },
      hovertemplate: `90 %% PI: [${fmt(p.q05)}, ${fmt(p.q95)}] ${meta.unit}<extra></extra>`,
    });
    traces.push({
      type: "scatter",
      mode: "markers",
      x: [p.q50],
      y: [label],
      xaxis: xref,
      yaxis: yref,
      marker: {
        symbol: "diamond",
        size: 14,
        color: "rgba(40, 75, 180, 1)",
        line: { color: "white", width: 1.5 },
      },
      hovertemplate: `median: ${fmt(p.q50)} ${meta.unit}<extra></extra>`,
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
