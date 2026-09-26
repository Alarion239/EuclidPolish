/* Plot data types (re-exported by Plot.tsx, where pages import them from). */
import type { Tick } from "../ticks";

export type { Tick };
export type AxisScale = "linear" | "log";

export type Series = {
  x: number[];
  y: (number | null)[];
  low?: (number | null)[];
  high?: (number | null)[];
  errorLow?: (number | null)[];
  errorHigh?: (number | null)[];
  color: string;
  mode?: "line" | "histogram" | "scatter";
  marker?: "filled" | "ring" | "diamond";
  width?: number;
  dash?: number[];
  dots?: boolean;
  markerEvery?: number;
  hatch?: boolean;
  alpha?: number;
  fillAlpha?: number;
  /** In-plot text drawn ON the line (contour-style). */
  label?: string;
  labelAt?: number;
  /** Name in the tooltip readout and the auto legend (default: `label`). */
  name?: string;
  /** Legend / visibility group: series sharing a key toggle together
   *  (default: `name`, then `label`, then the series index). */
  key?: string;
};

export type Guide = {
  axis: "x" | "y";
  v: number;
  color?: string;
  dash?: number[];
  width?: number;
  alpha?: number;
  label?: string;
  labelSide?: "before" | "after";
};

export type Band = {
  axis: "x" | "y";
  from: number;
  to: number;
  color: string;
  alpha?: number;
  hatch?: boolean;
  label?: string;
};

/* Optional density layer drawn UNDER the grid/guides/series — a 2D histogram
   painted as a heatmap (log-count normalised by default, empty cells
   transparent). Edges are in the same data units as the axes. */
export type Heat = {
  z: number[][];        // z[i][j] = count in x-bin i, y-bin j
  xEdges: number[];     // length z.length + 1
  yEdges: number[];     // length z[0].length + 1
  max?: number;         // log-norm ceiling (default = max positive count)
  min?: number;         // linear-norm floor (default = min finite value)
  scale?: "log" | "linear"; // default log (pixel-density diagnostics)
  colorTicks?: Tick[];  // values/labels in z units
  colorLabel?: string;
  /** t∈[0,1] → css color (default viridis). A drawn input compared by
   *  identity: a new closure redraws the plot, so memoise it. */
  color?: (t: number) => string;
};

/** A histogram cell selected in a heat plot (onHeatClick / highlight). */
export type Cell = { i: number; j: number };

export type LegendItem = {
  label: string;
  color: string;
  /** Visibility key (default: label) — matches `Series.key ?? name ?? label`. */
  key?: string;
  dash?: boolean;
  histogram?: boolean;
  filled?: boolean;
  hatch?: boolean;
  line?: boolean;
  marker?: "filled" | "ring" | "diamond";
};

/** Zoomed axis domains (null axis = the prop domain). */
export type PlotView = { x: [number, number] | null; y: [number, number] | null };

export type PlotProps = {
  xDomain: [number, number];
  yDomain: [number, number];
  xScale?: AxisScale;
  /** Log y (domain must be positive; non-positive values are gaps). */
  yScale?: AxisScale;
  xTicks?: Tick[];
  yTicks?: Tick[];
  xLabel?: string;
  yLabel?: string;
  title?: string;
  series: Series[];
  bands?: Band[];
  guides?: Guide[];
  heat?: Heat;
  /** Heat plots: a click reports the histogram cell; `highlight` outlines one. */
  onHeatClick?: (cell: Cell) => void;
  onPlotClick?: (point: { x: number; y: number }) => void;
  highlight?: Cell | null;
  height?: number;      /* fixed px; omit → aspect-driven */
  aspect?: number;      /* height = width * aspect (default 0.5) */

  /* ── v2 interaction ── */
  /** Hover crosshair + nearest-point tooltip (default true). */
  tooltip?: boolean;
  /** Box-zoom (drag), pan (shift-drag / arrows), wheel zoom (Ctrl/⌘ held, or
   *  the plot focused from the keyboard; a mouse click never turns the page
   *  wheel into zoom), double-click / Esc / 0 to reset (default true). */
  zoom?: boolean;
  /** Axes that zoom and pan (default "xy"). */
  zoomAxes?: "x" | "y" | "xy";
  /** Controlled zoom (with onViewChange); omit for internal zoom state. */
  view?: PlotView | null;
  onViewChange?: (view: PlotView | null) => void;
  /** Plots sharing a key show each other's cursor (linked crosshair). */
  syncKey?: string;
  /** Built-in interactive legend under the plot; "auto" lists the series
   *  that have a `name`. */
  legend?: LegendItem[] | "auto";
  /** Clicking a built-in legend entry toggles its series (default true).
   *  Applies only when `legend` is given; there is no legend without it
   *  (use `legend="auto"` for one built from the named series). */
  legendToggle?: boolean;
  /** Controlled hidden series keys (with onHiddenChange). */
  hidden?: string[];
  onHiddenChange?: (hidden: string[]) => void;
  /** Series key to emphasise (others dim), e.g. from an external legend. */
  emphasis?: string | null;
  /** Enables the PNG and CSV export buttons; the files are `<exportName>.png/.csv`. */
  exportName?: string;
  /** Tooltip / generated-tick formatting (zoomed ticks are compared by their
   *  labels, so an inline formatter does not force a redraw). */
  xFormat?: (v: number) => string;
  yFormat?: (v: number) => string;
  /** Accessible name (default: the title). */
  "aria-label"?: string;
};
