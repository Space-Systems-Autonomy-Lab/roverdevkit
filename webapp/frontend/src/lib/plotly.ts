import type { ComponentType } from "react";

import Plotly from "plotly.js-dist-min";
import * as factoryModule from "react-plotly.js/factory";
import type { PlotParams } from "react-plotly.js";

/**
 * Local re-export of `react-plotly.js` bound to the slim
 * `plotly.js-dist-min` build so we don't ship the full ~5 MB
 * Plotly bundle to the browser.
 *
 * `react-plotly.js/factory` is a Babel-compiled CJS module:
 *   exports.__esModule = true;
 *   exports.default = plotComponentFactory;
 *
 * Depending on which interop layer touches it (esbuild dev-server,
 * Rolldown prod, or Vite's `import * as` namespace shim), the same
 * import statement can land on the function itself, on
 * `{ default: fn }`, or even on `{ default: { default: fn } }`
 * when two CJS-interop layers run in series. `unwrapDefault` walks
 * those wrappers until it finds the callable.
 */
type PlotlyFactory = (plotly: unknown) => ComponentType<PlotParams>;

function unwrapDefault(value: unknown, depth = 5): unknown {
  let current = value;
  for (let i = 0; i < depth; i++) {
    if (typeof current === "function") return current;
    if (current && typeof current === "object" && "default" in current) {
      current = (current as { default: unknown }).default;
    } else {
      return current;
    }
  }
  return current;
}

const createPlotlyComponent = unwrapDefault(factoryModule) as PlotlyFactory;

if (typeof createPlotlyComponent !== "function") {
  console.error("react-plotly.js/factory module:", factoryModule);
  throw new Error(
    "react-plotly.js/factory did not resolve to a callable factory; " +
      "got: " +
      typeof createPlotlyComponent +
      " — see console.error above for the raw module shape.",
  );
}

const Plot = createPlotlyComponent(Plotly);

export default Plot;
