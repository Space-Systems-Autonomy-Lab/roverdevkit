// `plotly.js-dist-min` ships an untyped UMD bundle. We only consume
// it through `react-plotly.js/factory`, which accepts an opaque
// object, so an `unknown` default export is enough; trace and layout
// typing comes from `@types/plotly.js`.
declare module "plotly.js-dist-min" {
  const Plotly: unknown;
  export default Plotly;
}
