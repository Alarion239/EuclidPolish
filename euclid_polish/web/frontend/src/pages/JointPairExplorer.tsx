import { useEffect, useState } from "react";
import Plot, { type Guide, type Heat, type Series, type Tick } from "../charts/Plot";
import { useResource } from "../hooks";
import { Button, Checkbox, Chip, Empty, Spinner } from "../ui";
import {
  niceTicks,
  type CornerContour,
  type CornerDiagonal,
  type CornerVariable,
} from "./GalaxyCorner";

type Layer = { rows: number; density: number[][]; contours: CornerContour[] };
type PairView = {
  available: boolean;
  detail?: string;
  kind?: "joint" | "marginal";
  x?: CornerVariable;
  y?: CornerVariable;
  x_edges?: number[];
  y_edges?: number[];
  q1?: Layer;
  model?: Layer;
  diagonal?: CornerDiagonal;
  contour_mass_fractions?: number[];
  q1_rows?: number;
  model_draws?: number;
  vis_range?: [number, number];
};
type Source = "q1" | "model";
type Shading = Source | "none";

const SOURCE: Record<Source, { label: string; color: string; rgb: [number, number, number] }> = {
  q1: { label: "Euclid Q1", color: "#2478d4", rgb: [36, 120, 212] },
  model: { label: "Model draws", color: "#e25543", rgb: [226, 85, 67] },
};
const DEFAULT_LEVELS = [0.5, 0.8, 0.95];

/* Shading is an opaque blend over the card surface: translucent heat cells
   overlap their neighbours slightly and would double their alpha in seams. */
function surfaceRgb(): [number, number, number] {
  const raw = getComputedStyle(document.documentElement).getPropertyValue("--surface-1").trim();
  const match = /^#([0-9a-f]{6})$/i.exec(raw);
  if (!match) return [255, 255, 255];
  const value = parseInt(match[1], 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

const axisTicks = (domain: [number, number]): Tick[] => {
  const { values, decimals } = niceTicks(domain, 7);
  return values.map((v) => ({ v, label: v.toFixed(decimals) }));
};
const axisLabel = (variable: CornerVariable) => `${variable.label} [${variable.unit}]`;
const percent = (fraction: number) => `${Math.round(100 * fraction)}%`;

function binIndex(edges: number[], value: number): number {
  if (!(value >= edges[0] && value <= edges[edges.length - 1])) return -1;
  let index = 0;
  while (index < edges.length - 2 && edges[index + 1] <= value) index++;
  return index;
}

/* Smallest enclosed-mass region containing the cell: the fraction of the
   layer's mass in cells at least as dense as it. */
function enclosingFraction(layer: Layer | undefined, i: number, j: number): number | null {
  const value = layer?.density[i]?.[j];
  if (!layer || value == null || !(value > 0)) return null;
  let total = 0;
  let denser = 0;
  for (const column of layer.density) {
    for (const cell of column) {
      total += cell;
      if (cell >= value) denser += cell;
    }
  }
  return total > 0 ? denser / total : null;
}

function stepSeries(edges: number[], density: number[]): { x: number[]; y: number[] } {
  const x: number[] = [];
  const y: number[] = [];
  density.forEach((value, bin) => {
    x.push(edges[bin], edges[bin + 1]);
    y.push(value, value);
  });
  return { x, y };
}

export default function JointPairExplorer({
  variables, revision,
}: {
  variables?: CornerVariable[];
  revision: string;
}) {
  const [xKey, setXKey] = useState("vis");
  const [yKey, setYKey] = useState("log_re");
  const [visible, setVisible] = useState<Record<Source, boolean>>({ q1: true, model: true });
  const [shading, setShading] = useState<Shading>("q1");
  const [levels, setLevels] = useState<number[]>(DEFAULT_LEVELS);
  const [probe, setProbe] = useState<{ x: number; y: number } | null>(null);
  const resource = useResource<PairView>(
    variables?.length
      ? `/api/galaxy-distributions/joint-pair?x=${xKey}&y=${yKey}&r=${encodeURIComponent(revision)}`
      : null,
    [],
    { ttl: 60_000 },
  );
  // The last drawn pair stays on screen while a new selection loads, so the
  // card keeps its height instead of collapsing on every chip click.
  const [shown, setShown] = useState<PairView | null>(null);
  useEffect(() => {
    if (resource.data) setShown(resource.data);
  }, [resource.data]);

  if (!variables?.length) {
    return <Empty>Rebuild cached plots to draw the joint distributions.</Empty>;
  }
  const view = resource.data ?? shown;
  const loadingSelection = resource.loading && !resource.data && shown != null;
  const chooseAxis = (axis: "x" | "y", key: string) => {
    setProbe(null);
    if (axis === "x") setXKey(key); else setYKey(key);
  };
  const swapAxes = () => { setProbe(null); setXKey(yKey); setYKey(xKey); };
  const toggleLevel = (fraction: number) => setLevels((current) => (
    current.includes(fraction)
      ? current.filter((value) => value !== fraction)
      : [...current, fraction]
  ));
  const fractions = [...(view?.contour_mass_fractions ?? [])].sort((a, b) => a - b);

  const controls = <div className="pair-explorer__controls">
    <section>
      <h4>x axis</h4>
      <div className="pair-explorer__chips">
        {variables.map((variable) => <Chip key={variable.key} on={variable.key === xKey}
          onClick={() => chooseAxis("x", variable.key)}>{variable.label}</Chip>)}
      </div>
    </section>
    <section>
      <h4>y axis <Button size="sm" variant="ghost" onClick={swapAxes}>Swap axes</Button></h4>
      <div className="pair-explorer__chips">
        {variables.map((variable) => <Chip key={variable.key} on={variable.key === yKey}
          onClick={() => chooseAxis("y", variable.key)}>{variable.label}</Chip>)}
      </div>
    </section>
    <section>
      <h4>display</h4>
      <div className="pair-explorer__chips">
        {(["q1", "model"] as Source[]).map((source) => <Checkbox key={source}
          checked={visible[source]}
          onChange={(checked: boolean) => setVisible((current) => ({ ...current, [source]: checked }))}
        >{SOURCE[source].label}</Checkbox>)}
      </div>
      <div className="pair-explorer__chips">
        <span className="pair-explorer__hint">shading</span>
        {(["q1", "model", "none"] as Shading[]).map((choice) => <Chip key={choice}
          on={shading === choice} onClick={() => setShading(choice)}
          dot={choice === "none" ? undefined : SOURCE[choice].color}
        >{choice === "none" ? "off" : SOURCE[choice].label}</Chip>)}
      </div>
      {fractions.length > 0 && <div className="pair-explorer__chips">
        <span className="pair-explorer__hint">contours</span>
        {fractions.map((fraction) => <Chip key={fraction}
          on={levels.includes(fraction)} onClick={() => toggleLevel(fraction)}
        >{percent(fraction)}</Chip>)}
      </div>}
    </section>
  </div>;

  if (!view) {
    return <>{controls}<Empty>{resource.loading ? <><Spinner /> reading joint distribution…</> : "Joint distribution unavailable."}</Empty></>;
  }
  if (!view.available || !view.x || !view.y) {
    return <>{controls}<Empty>{view.detail ?? "Joint distribution unavailable."}</Empty></>;
  }

  if (view.kind === "marginal" && view.diagonal) {
    const diagonal = view.diagonal;
    const edges = diagonal.edges;
    const xDomain: [number, number] = [edges[0], edges[edges.length - 1]];
    const peak = Math.max(...diagonal.q1, ...diagonal.model, 1e-12) * 1.08;
    const yDomain: [number, number] = [0, peak];
    const series: Series[] = [];
    (["q1", "model"] as Source[]).forEach((source) => {
      if (!visible[source]) return;
      const step = stepSeries(edges, diagonal[source]);
      series.push({
        x: step.x,
        y: step.y,
        ...(source === "q1" ? { low: step.y.map(() => 0), high: step.y, fillAlpha: 0.14 } : {}),
        color: SOURCE[source].color,
        width: source === "q1" ? 2 : 2.2,
        dash: source === "model" ? [7, 4] : undefined,
      });
    });
    return <>
      {controls}
      <Plot xDomain={xDomain} yDomain={yDomain}
        xTicks={axisTicks(xDomain)} yTicks={axisTicks(yDomain)}
        xLabel={axisLabel(view.x)} yLabel="probability density (unit area)"
        series={series} aspect={0.5} />
      <p className="pair-explorer__readout">
        {loadingSelection
          ? "Loading the selected pair…"
          : "Same variable on both axes: the two samples' 1-D distributions at unit area (Euclid Q1 filled, model dashed)."}
      </p>
    </>;
  }

  const xEdges = view.x_edges ?? [];
  const yEdges = view.y_edges ?? [];
  const xDomain: [number, number] = [xEdges[0], xEdges[xEdges.length - 1]];
  const yDomain: [number, number] = [yEdges[0], yEdges[yEdges.length - 1]];
  const series: Series[] = [];
  (["q1", "model"] as Source[]).forEach((source) => {
    const layer = view[source];
    if (!visible[source] || !layer) return;
    layer.contours
      .filter((contour) => levels.some((level) => Math.abs(level - contour.mass_fraction) < 1e-9))
      .forEach((contour) => {
        const longest = contour.paths.reduce((best, path) => (
          path.x.length > best.x.length ? path : best
        ), contour.paths[0]);
        contour.paths.forEach((path) => series.push({
          x: path.x,
          y: path.y,
          color: SOURCE[source].color,
          width: contour.mass_fraction <= 0.5 ? 2.3 : contour.mass_fraction <= 0.8 ? 1.8 : 1.3,
          alpha: contour.mass_fraction >= 0.99 ? 0.6 : 1,
          dash: source === "model" ? [7, 4] : undefined,
          label: path === longest ? percent(contour.mass_fraction) : undefined,
          labelAt: path === longest ? (source === "q1" ? 0.2 : 0.62) : undefined,
        }));
      });
  });

  let heat: Heat | undefined;
  if (shading !== "none" && view[shading]?.density.length) {
    const tint = SOURCE[shading].rgb;
    const base = surfaceRgb();
    heat = {
      z: view[shading]!.density.map((column) => column.map((value) => (value > 0.002 ? value : NaN))),
      xEdges,
      yEdges,
      scale: "linear",
      min: 0,
      max: 1,
      color: (t: number) => {
        const weight = 0.05 + 0.6 * t;
        return `rgb(${base.map((channel, k) => Math.round(channel + (tint[k] - channel) * weight)).join(", ")})`;
      },
      colorTicks: [{ v: 0, label: "0" }, { v: 0.5, label: "0.5" }, { v: 1, label: "1" }],
      colorLabel: "density / peak",
    };
  }

  const guides: Guide[] = probe ? [
    { axis: "x", v: probe.x, color: "#17202e", dash: [3, 3], width: 1, alpha: 0.7 },
    { axis: "y", v: probe.y, color: "#17202e", dash: [3, 3], width: 1, alpha: 0.7 },
  ] : [];
  const probeText = (() => {
    if (!probe) return "Click the plot to read which contour a point falls inside.";
    const i = binIndex(xEdges, probe.x);
    const j = binIndex(yEdges, probe.y);
    const describe = (source: Source) => {
      const fraction = enclosingFraction(view[source], i, j);
      return fraction == null
        ? `outside the populated ${SOURCE[source].label} region`
        : `inside the ${percent(fraction)} ${SOURCE[source].label} contour`;
    };
    return `${view.x!.label} = ${probe.x.toFixed(3)}, ${view.y!.label} = ${probe.y.toFixed(3)}: ${describe("q1")}; ${describe("model")}.`;
  })();

  return <>
    {controls}
    <Plot xDomain={xDomain} yDomain={yDomain}
      xTicks={axisTicks(xDomain)} yTicks={axisTicks(yDomain)}
      xLabel={axisLabel(view.x)} yLabel={axisLabel(view.y)}
      series={series} heat={heat} guides={guides}
      onPlotClick={(point) => setProbe(point)}
      aspect={0.62} />
    <p className="pair-explorer__readout">{loadingSelection ? "Loading the selected pair…" : probeText}</p>
    <div className="galaxy-plot__definitions">
      <span><i style={{ background: SOURCE.q1.color }} />
        Euclid Q1 (solid): {(view.q1?.rows ?? 0).toLocaleString()} colour-model rows inside the window, raw colours.
      </span>
      <span><i style={{ background: SOURCE.model.color }} />
        Model draws (dashed): {(view.model?.rows ?? 0).toLocaleString()} draws inside the window{view.vis_range ? ` (VIS ${view.vis_range[0].toFixed(2)}–${view.vis_range[1].toFixed(2)})` : ""}, deconvolved colours.
      </span>
      <span>Contours enclose the labelled share of each sample's in-window mass; shading is that sample's smoothed density relative to its peak.</span>
    </div>
  </>;
}
