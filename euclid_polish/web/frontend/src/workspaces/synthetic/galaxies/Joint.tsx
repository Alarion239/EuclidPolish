/* Galaxies › joint: the three joint views, all on the Synthetic chart kit
   (jointColor, contourStyle / contourSeries, contourMassLabel, stepSeries,
   enclosingFraction, tintRamp):
   - the corner plot of the six model variables (Euclid Q1 rows below the
     diagonal, model draws above; a cell opens that pair in the explorer);
   - the pair explorer (any two variables, GET /api/galaxy-distributions/
     joint-pair, shading + contour levels, a click reads which contour holds a
     point, "Swap axes");
   - the magnitude × radius maps (gray Q1, blue dashed generated, red solid
     model contours labelled by enclosed mass). */
import { useEffect, useId, useState, type KeyboardEvent } from "react";
import Plot, { type Guide, type Heat, type Series } from "../../../charts/Plot";
import { C } from "../../../colors";
import { formatCount, formatNumber } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { linearTicks, type Tick } from "../../../ticks";
import { Button, Caption, Chip, EmptyState, Segmented, Skeleton, Switch } from "../../../ui";
import { useJointPair, type CornerData, type CornerVariable, type JointMaps, type PairView } from "../api";
import {
  binIndex, contourMassLabel, contourSeries, contourStyle, enclosingFraction, isLog10Label, jointColor,
  log10AxisTicks, physicalFromLog10, physicalLogAxisLabel, stepSeries, surfaceColor, tintRamp,
} from "../chartKit";
import { Info, Swatch } from "../common";
import { modelColourWording } from "./galaxyText";
import { CONTOUR_LEVELS, jointMapSeries } from "./model";

type Source = "q1" | "model";
const SOURCES: Source[] = ["q1", "model"];
const SOURCE_LABEL: Record<Source, string> = { q1: "Euclid Q1", model: "Model draws" };
const DEFAULT_LEVELS = [0.5, 0.8, 0.95];
const pct = (fraction: number) => `${Math.round(100 * fraction)}%`;

/** Axis label of a corner variable; log10 variables read physically. */
export function variableAxis(variable: CornerVariable): { label: string; log10: boolean } {
  if (isLog10Label(variable.label)) return { label: physicalLogAxisLabel(`${variable.label} (${variable.unit})`), log10: true };
  return { label: `${variable.label} [${variable.unit}]`, log10: false };
}

const axisTicks = (domain: [number, number], log10: boolean): Tick[] =>
  log10 ? log10AxisTicks(domain, 7) : linearTicks(domain, { count: 7 });

/* ─── corner ────────────────────────────────────────────────────────────── */

const CELL = 128;
const GAP = 6;
const MARGIN = { top: 28, right: 44, bottom: 30, left: 78 };

export function CornerPlot({ data, active, onPick }: {
  data?: CornerData; active?: { x: string; y: string }; onPick?: (x: string, y: string) => void;
}) {
  const clipPrefix = `corner-${useId().replace(/:/g, "")}`;
  if (!data?.available || !data.variables?.length || !data.cells || !data.diagonal) {
    return <EmptyState compact icon="activity" title={data?.detail ?? "Rebuild cached plots to draw the joint distributions"} />;
  }
  const variables = data.variables;
  const n = variables.length;
  const width = MARGIN.left + n * CELL + (n - 1) * GAP + MARGIN.right;
  const height = MARGIN.top + n * CELL + (n - 1) * GAP + MARGIN.bottom;
  const cellX = (col: number) => MARGIN.left + col * (CELL + GAP);
  const cellY = (row: number) => MARGIN.top + row * (CELL + GAP);
  const frac = (value: number, [a, b]: [number, number]) => (value - a) / (b - a);
  const px = (col: number, value: number) => cellX(col) + frac(value, variables[col].domain) * CELL;
  const py = (row: number, value: number) => cellY(row) + CELL - frac(value, variables[row].domain) * CELL;
  const ticks = variables.map((variable) => linearTicks(variable.domain, { count: 3 })
    .filter((t) => t.v >= variable.domain[0] && t.v <= variable.domain[1]));
  const cellByPosition = new Map(data.cells.map((cell) => [`${cell.row},${cell.col}`, cell]));
  const color = { q1: jointColor("q1"), model: jointColor("model") };
  const fractions = [...(data.contour_mass_fractions ?? [])].sort((a, b) => a - b);
  const offWindow = variables.filter((v) => (v.outside_fraction?.q1 ?? 0) >= 0.01 || (v.outside_fraction?.model ?? 0) >= 0.01);

  const grid = (row: number, col: number, horizontal: boolean) => (
    <g className="rl-corner__grid">
      {ticks[col].map((t) => <line key={`x${t.v}`} x1={px(col, t.v)} x2={px(col, t.v)} y1={cellY(row)} y2={cellY(row) + CELL} />)}
      {horizontal && ticks[row].map((t) => <line key={`y${t.v}`} x1={cellX(col)} x2={cellX(col) + CELL} y1={py(row, t.v)} y2={py(row, t.v)} />)}
    </g>
  );

  const diagonalCell = (index: number) => {
    const diagonal = data.diagonal![index];
    const peak = Math.max(...diagonal.q1, ...diagonal.model, 1e-12) * 1.08;
    const base = cellY(index) + CELL;
    const path = (density: number[]) => {
      const step = stepSeries(diagonal.edges, density);
      const pts = step.x.map((x, i) => `L${px(index, x).toFixed(1)},${(base - (step.y[i] / peak) * CELL).toFixed(1)}`);
      return `M${px(index, diagonal.edges[0]).toFixed(1)},${base.toFixed(1)}${pts.join("")}`
        + `L${px(index, diagonal.edges[diagonal.edges.length - 1]).toFixed(1)},${base.toFixed(1)}`;
    };
    return (
      <g clipPath={`url(#${clipPrefix}-${index}-${index})`}>
        {grid(index, index, false)}
        <path d={path(diagonal.q1)} fill={color.q1} fillOpacity={0.16} stroke={color.q1} strokeWidth={1.2} />
        <path d={path(diagonal.model)} fill="none" stroke={color.model} strokeWidth={1.4} />
      </g>
    );
  };

  const jointCell = (row: number, col: number) => {
    const cell = cellByPosition.get(`${row},${col}`);
    const stroke = color[cell?.source ?? (row > col ? "q1" : "model")];
    return (
      <g clipPath={`url(#${clipPrefix}-${row}-${col})`}>
        {grid(row, col, true)}
        {cell?.contours.map((contour) => {
          const style = contourStyle(contour.mass_fraction);
          return contour.paths.map((path, i) => (
            <path key={`${contour.mass_fraction}-${i}`} fill="none" stroke={stroke} strokeWidth={style.width}
              strokeOpacity={style.opacity} strokeLinejoin="round"
              d={path.x.map((x, k) => `${k ? "L" : "M"}${px(col, x).toFixed(1)},${py(row, path.y[k]).toFixed(1)}`).join("")} />
          ));
        })}
        {cell && !cell.contours.length && <text className="rl-corner__empty" x={cellX(col) + CELL / 2} y={cellY(row) + CELL / 2}>no data</text>}
      </g>
    );
  };

  const positions = Array.from({ length: n * n }, (_, i) => [Math.floor(i / n), i % n] as const);
  const pick = (row: number, col: number) => onPick?.(variables[col].key, variables[row].key);
  const onKey = (e: KeyboardEvent, row: number, col: number) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(row, col); }
  };
  return (
    <div className="rl-corner">
      <div className="rl-corner__frame">
        <svg viewBox={`0 0 ${width} ${height}`} role="group"
          aria-label="Corner plot of VIS magnitude, SFR, radius and three colours: Euclid Q1 below the diagonal (lower triangle), model draws above (upper triangle)">
          <defs>
            {positions.map(([row, col]) => (
              <clipPath key={`${row}-${col}`} id={`${clipPrefix}-${row}-${col}`}>
                <rect x={cellX(col)} y={cellY(row)} width={CELL} height={CELL} />
              </clipPath>
            ))}
          </defs>
          {positions.map(([row, col]) => {
            const selected = active && variables[col].key === active.x && variables[row].key === active.y;
            const triangle = row === col ? "diagonal" : row > col ? "lower" : "upper";
            const name = row === col ? variables[row].label : `${variables[col].label} vs ${variables[row].label}`;
            // The cell's sample size is read on hover (its tooltip), not printed in every cell.
            const cell = row === col ? undefined : cellByPosition.get(`${row},${col}`);
            const hover = cell ? `${name} · n = ${formatCount(cell.rows)} ${cell.source === "q1" ? "Q1 rows" : "model draws"}` : name;
            return (
              <g key={`c${row}-${col}`} className="rl-corner__cell" data-triangle={triangle} data-active={selected || undefined}
                role={onPick ? "button" : undefined} tabIndex={onPick ? 0 : undefined}
                aria-label={onPick ? `Explore ${name} (${triangle === "lower" ? "Euclid Q1" : triangle === "upper" ? "model draws" : "both samples"})` : undefined}
                onClick={onPick ? () => pick(row, col) : undefined} onKeyDown={onPick ? (e) => onKey(e, row, col) : undefined}>
                <title>{hover}</title>
                <rect className="rl-corner__bg" x={cellX(col)} y={cellY(row)} width={CELL} height={CELL} />
                {row === col ? diagonalCell(row) : jointCell(row, col)}
              </g>
            );
          })}
          {variables.map((variable, col) => (
            <text key={`t${col}`} className="rl-corner__title" x={cellX(col) + CELL / 2} y={MARGIN.top - 10}>{variable.label} [{variable.unit}]</text>
          ))}
          {variables.map((variable, row) => {
            const y = cellY(row) + CELL / 2;
            return <text key={`r${row}`} className="rl-corner__title" x={14} y={y} transform={`rotate(-90 14 ${y})`}>{variable.label}</text>;
          })}
          {variables.map((_v, col) => ticks[col].map((t) => (
            <text key={`bx${col}-${t.v}`} className="rl-corner__tick" data-side="bottom" x={px(col, t.v)} y={cellY(n - 1) + CELL + 13}>{t.label}</text>
          )))}
          {variables.map((_v, row) => (row === 0 ? null : ticks[row].map((t) => (
            <text key={`ly${row}-${t.v}`} className="rl-corner__tick" data-side="left" x={MARGIN.left - 6} y={py(row, t.v)}>{t.label}</text>
          ))))}
          {variables.map((_v, row) => (row === n - 1 ? null : ticks[row].map((t) => (
            <text key={`ry${row}-${t.v}`} className="rl-corner__tick" data-side="right" x={cellX(n - 1) + CELL + 6} y={py(row, t.v)}>{t.label}</text>
          ))))}
        </svg>
      </div>
      <div className="rl-defs">
        <span><Swatch color={color.q1} />lower triangle · {formatCount(data.q1_rows)} Euclid Q1 colour-model rows, raw colours</span>
        <span><Swatch color={color.model} />upper triangle · {formatCount(data.model_draws)} model draws
          {data.vis_range ? ` inside VIS ${data.vis_range[0].toFixed(2)}–${data.vis_range[1].toFixed(2)}` : ""}, {modelColourWording(data.model_noise)}</span>
        <span>contours enclose {fractions.map((f) => Math.round(100 * f)).join(" / ")}% · click a cell to explore it</span>
        {offWindow.length > 0 && (
          <span title="PHZ log SFR below −5 (effectively zero SFR) is always off-window">
            off-window (Q1 / model): {offWindow.map((v) => `${v.label} ${(100 * (v.outside_fraction?.q1 ?? 0)).toFixed(1)}% / ${(100 * (v.outside_fraction?.model ?? 0)).toFixed(1)}%`).join(" · ")}
          </span>
        )}
      </div>
    </div>
  );
}

/* ─── pair explorer ─────────────────────────────────────────────────────── */

export function usePairKeys(): { x: string; y: string; set: (x: string, y: string) => void } {
  const [x, setX] = useUrlState("px", "vis");
  const [y, setY] = useUrlState("py", "log_re");
  return { x, y, set: (nx, ny) => { setX(nx); setY(ny); } };
}

function probeText(view: PairView, probe: { x: number; y: number } | null, xLog: boolean, yLog: boolean): string {
  if (!probe || !view.x || !view.y) return "Click the plot to read which contour a point falls inside.";
  const i = binIndex(view.x_edges ?? [], probe.x);
  const j = binIndex(view.y_edges ?? [], probe.y);
  const describe = (source: Source) => {
    const fraction = enclosingFraction(view[source]?.density, i, j);
    return fraction == null ? `outside the populated ${SOURCE_LABEL[source]} region` : `inside the ${pct(fraction)} ${SOURCE_LABEL[source]} contour`;
  };
  const value = (v: number, log: boolean) => (log ? physicalFromLog10(v) : formatNumber(v, { digits: 3 }));
  return `${view.x.label} = ${value(probe.x, xLog)}, ${view.y.label} = ${value(probe.y, yLog)}: ${describe("q1")}; ${describe("model")}.`;
}

export function PairExplorer({ variables, revision }: { variables?: CornerVariable[]; revision: string }) {
  const pair = usePairKeys();
  const [hidden, setHidden] = useUrlState<string[]>("phide", []);
  const [shading, setShading] = useUrlState<"q1" | "model" | "none">("pshade", "q1");
  const [levels, setLevels] = useUrlState<number[]>("plev", DEFAULT_LEVELS, {
    parse: (raw) => raw.split(",").map(Number).filter((v) => v > 0 && v < 1),
    serialize: (v) => v.join(","),
  });
  const [probe, setProbe] = useState<{ x: number; y: number } | null>(null);
  const resource = useJointPair(variables?.length ? pair.x : null, pair.y, revision);
  // The last drawn pair stays on screen while a new one loads (no collapse).
  const [shown, setShown] = useState<PairView | null>(null);
  useEffect(() => { if (resource.data) setShown(resource.data); }, [resource.data]);
  useEffect(() => { setProbe(null); }, [pair.x, pair.y]);

  if (!variables?.length) return <EmptyState compact icon="activity" title="Rebuild cached plots to draw the joint distributions" />;
  const view = resource.data ?? shown;
  const loadingNew = resource.loading && !resource.data && shown != null;
  const fractions = [...(view?.contour_mass_fractions ?? [])].sort((a, b) => a - b);
  const toggleLevel = (f: number) => setLevels(levels.includes(f) ? levels.filter((l) => l !== f) : [...levels, f]);
  const toggleSource = (s: Source, on: boolean) => setHidden(on ? hidden.filter((h) => h !== s) : [...hidden, s]);
  const visible = (s: Source) => !hidden.includes(s);

  const controls = (
    <div className="rl-pair__controls">
      <div className="rl-pair__axis">
        <span className="rl-subhead">x</span>
        <div className="rl-chips" role="group" aria-label="x variable">
          {variables.map((v) => <Chip key={v.key} on={v.key === pair.x} onClick={() => pair.set(v.key, pair.y)}>{v.label}</Chip>)}
        </div>
      </div>
      <div className="rl-pair__axis">
        <span className="rl-subhead">y</span>
        <div className="rl-chips" role="group" aria-label="y variable">
          {variables.map((v) => <Chip key={v.key} on={v.key === pair.y} onClick={() => pair.set(pair.x, v.key)}>{v.label}</Chip>)}
        </div>
        <Button size="sm" variant="ghost" icon="reset" onClick={() => pair.set(pair.y, pair.x)}>Swap axes</Button>
      </div>
      <div className="rl-pair__axis">
        {SOURCES.map((s) => (
          <Switch key={s} size="sm" checked={visible(s)} onChange={(on) => toggleSource(s, on)}>
            <span className="rl-legend-cell"><Swatch color={jointColor(s)} dash={s === "model"} />{SOURCE_LABEL[s]}</span>
          </Switch>
        ))}
        <Segmented size="sm" aria-label="Shading" value={shading} onChange={setShading}
          options={[{ value: "q1", label: "Q1 shading" }, { value: "model", label: "model shading" }, { value: "none", label: "off" }]} />
      </div>
      {fractions.length > 0 && (
        <div className="rl-pair__axis">
          <span className="rl-subhead">contours</span>
          <div className="rl-chips" role="group" aria-label="Contour levels">
            {fractions.map((f) => <Chip key={f} on={levels.includes(f)} onClick={() => toggleLevel(f)}>{pct(f)}</Chip>)}
          </div>
        </div>
      )}
    </div>
  );

  if (!view) {
    return <>{controls}{resource.loading ? <Skeleton height={280} /> : (
      <EmptyState compact icon="warn" title="Joint distribution unavailable">{resource.error?.message}</EmptyState>
    )}</>;
  }
  if (!view.available || !view.x || !view.y) {
    return <>{controls}<EmptyState compact icon="activity" title={view.detail ?? "Joint distribution unavailable"} /></>;
  }
  const xAxis = variableAxis(view.x);
  const yAxis = variableAxis(view.y);

  if (view.kind === "marginal" && view.diagonal) {
    const { edges } = view.diagonal;
    const xDomain: [number, number] = [edges[0], edges[edges.length - 1]];
    const yDomain: [number, number] = [0, Math.max(...view.diagonal.q1, ...view.diagonal.model, 1e-12) * 1.08];
    const series: Series[] = SOURCES.filter(visible).map((s) => {
      const step = stepSeries(edges, view.diagonal![s]);
      return {
        x: step.x, y: step.y, color: jointColor(s), width: s === "q1" ? 2 : 2.2, dash: s === "model" ? [7, 4] : undefined,
        ...(s === "q1" ? { low: step.y.map(() => 0), high: step.y, fillAlpha: 0.14 } : {}), name: SOURCE_LABEL[s], key: s,
      };
    });
    return (
      <>
        {controls}
        <Plot xDomain={xDomain} yDomain={yDomain} xTicks={axisTicks(xDomain, xAxis.log10)} yTicks={linearTicks(yDomain, { count: 5 })}
          xLabel={xAxis.label} yLabel="probability density (unit area)" series={series} aspect={0.5}
          xFormat={xAxis.log10 ? (v) => physicalFromLog10(v) : undefined} exportName={`galaxy-pair-${view.x.key}`}
          aria-label={`${view.x.label} marginal: Euclid Q1 and model draws`} />
        <p className="rl-faint" aria-live="polite">
          {loadingNew ? "Loading the selected pair…" : "Same variable on both axes: both samples' 1-D distributions at unit area (Q1 filled, model dashed)."}
        </p>
      </>
    );
  }

  const xEdges = view.x_edges ?? [];
  const yEdges = view.y_edges ?? [];
  const xDomain: [number, number] = [xEdges[0], xEdges[xEdges.length - 1]];
  const yDomain: [number, number] = [yEdges[0], yEdges[yEdges.length - 1]];
  const series: Series[] = SOURCES.filter((s) => visible(s) && view[s]).flatMap((s) => contourSeries(view[s]!.contours, {
    color: jointColor(s), dash: s === "model" ? [7, 4] : undefined, levels, name: SOURCE_LABEL[s], key: s,
    labelAt: () => (s === "q1" ? 0.2 : 0.62), widthScale: 1.25,
  }));
  let heat: Heat | undefined;
  const layer = shading !== "none" ? view[shading] : undefined;
  if (layer?.density.length) {
    heat = {
      z: layer.density.map((column) => column.map((v) => (v > 0.002 ? v : NaN))), xEdges, yEdges,
      scale: "linear", min: 0, max: 1, color: tintRamp(jointColor(shading as Source), surfaceColor()),
      colorTicks: [{ v: 0, label: "0" }, { v: 0.5, label: "0.5" }, { v: 1, label: "1" }], colorLabel: "density / peak",
    };
  }
  const guides: Guide[] = probe ? [
    { axis: "x", v: probe.x, color: C.cross, dash: [3, 3], width: 1, alpha: 0.7 },
    { axis: "y", v: probe.y, color: C.cross, dash: [3, 3], width: 1, alpha: 0.7 },
  ] : [];
  return (
    <>
      {controls}
      <Plot xDomain={xDomain} yDomain={yDomain} xTicks={axisTicks(xDomain, xAxis.log10)} yTicks={axisTicks(yDomain, yAxis.log10)}
        xLabel={xAxis.label} yLabel={yAxis.label} series={series} heat={heat} guides={guides}
        onPlotClick={setProbe} aspect={0.62} legend="auto"
        xFormat={xAxis.log10 ? (v) => physicalFromLog10(v) : undefined}
        yFormat={yAxis.log10 ? (v) => physicalFromLog10(v) : undefined}
        exportName={`galaxy-pair-${view.x.key}-${view.y.key}`}
        aria-label={`Joint ${view.x.label} × ${view.y.label}: Euclid Q1 and model contours`} />
      <p className="rl-faint" aria-live="polite">{loadingNew ? "Loading the selected pair…" : probeText(view, probe, xAxis.log10, yAxis.log10)}</p>
      <div className="rl-defs">
        <span><Swatch color={jointColor("q1")} />Euclid Q1 (solid): {formatCount(view.q1?.rows ?? 0)} rows in window, raw colours</span>
        <span><Swatch color={jointColor("model")} dash />Model draws (dashed): {formatCount(view.model?.rows ?? 0)} in window
          {view.vis_range ? ` (VIS ${view.vis_range[0].toFixed(2)}–${view.vis_range[1].toFixed(2)})` : ""}, {modelColourWording(view.model_noise)}</span>
      </div>
    </>
  );
}

/* ─── magnitude × radius maps ───────────────────────────────────────────── */

export function JointMapsView({ data }: { data?: JointMaps }) {
  if (!data?.available || !data.maps?.length) {
    return <EmptyState compact icon="activity" title={data?.detail ?? "Rebuild the galaxy statistics to make the joint maps"} />;
  }
  const q1 = data.maps.find((m) => m.key === "q1");
  const overlays = data.maps.filter((m) => m.key === "synthetic" || m.key === "model");
  const synthetic = overlays.find((m) => m.key === "synthetic");
  if (!q1 || !overlays.length) {
    return <EmptyState compact icon="activity" title="The Q1 density or the generated / model contour layers are unavailable" />;
  }
  const xDomain: [number, number] = [data.magnitude_edges[0], data.magnitude_edges[data.magnitude_edges.length - 1]];
  const yDomain: [number, number] = [10 ** data.log_radius_edges[0], 10 ** data.log_radius_edges[data.log_radius_edges.length - 1]];
  const series = jointMapSeries(data);
  const levels = (data.contour_mass_fractions?.length ? data.contour_mass_fractions : CONTOUR_LEVELS).map(contourMassLabel).join(" / ");
  return (
    <div className="rl-maps">
      <Plot xDomain={xDomain} yDomain={yDomain} yScale="log" xTicks={linearTicks(xDomain, { count: 8 })}
        yTicks={log10AxisTicks([Math.log10(yDomain[0]), Math.log10(yDomain[1])]).map((t) => ({ v: 10 ** t.v, label: t.label }))}
        xLabel="VIS 2FWHM AB magnitude" yLabel="Circularized Sérsic Rₑ (arcsec, log scale)" series={series}
        legend="auto" aspect={0.45} exportName="galaxy-joint-maps"
        yFormat={(v) => `${formatNumber(v, { sig: 3 })}″`}
        aria-label="Magnitude × radius contours: Q1 gray, generated blue dashed, model red solid" />
      <div className="rl-defs">
        {[q1, ...overlays].map((m) => (
          <span key={m.key}><Swatch color={jointColor(m.key, "maps")} dash={m.key === "synthetic"} />{m.label} · {m.detail}</span>
        ))}
      </div>
      <Caption>
        {[`Q1 shading: ${formatNumber(q1.surface_density_arcmin2, { sig: 3 })} objects arcmin⁻² inside the map`,
          synthetic?.rows != null ? `${formatCount(synthetic.rows)} generated galaxies` : null,
          `contours enclose ${levels} of each sample`].filter(Boolean).join(" · ")}
      </Caption>
    </div>
  );
}

export function JointHelp() {
  return (
    <Info label="About the joint views">
      <p>The corner plot pairs VIS brightness, SFR, size and the three NISP/VIS colours: Euclid Q1 rows below the
        diagonal, fitted-model draws above it, both samples at unit area on the diagonal.</p>
      <p>Contours enclose the labelled share of each sample's in-window mass; the explorer's shading is that sample's
        smoothed density relative to its peak.</p>
    </Info>
  );
}
