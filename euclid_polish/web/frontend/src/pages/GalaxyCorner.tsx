import { useId } from "react";
import { Empty } from "../ui";

export type CornerVariable = {
  key: string;
  label: string;
  unit: string;
  domain: [number, number];
  outside_fraction?: { q1: number; model: number };
};
export type CornerContour = {
  mass_fraction: number;
  paths: Array<{ x: number[]; y: number[] }>;
};
export type CornerCell = {
  row: number;
  col: number;
  source: "q1" | "model";
  rows: number;
  contours: CornerContour[];
};
export type CornerDiagonal = { edges: number[]; q1: number[]; model: number[] };
export type CornerData = {
  available: boolean;
  detail?: string;
  variables?: CornerVariable[];
  contour_mass_fractions?: number[];
  diagonal?: CornerDiagonal[];
  cells?: CornerCell[];
  q1_rows?: number;
  model_draws?: number;
  vis_range?: [number, number];
};

const CELL = 128;
const GAP = 6;
const MARGIN = { top: 28, right: 44, bottom: 30, left: 78 };
const SOURCE_COLOR = { q1: "#2478d4", model: "#e25543" } as const;

const contourStyle = (fraction: number) => fraction <= 0.5
  ? { width: 1.7, opacity: 1 }
  : fraction <= 0.8
    ? { width: 1.2, opacity: 0.8 }
    : { width: 0.8, opacity: 0.55 };

/* Two to four round ticks inside the domain, with their label precision. */
function niceTicks([a, b]: [number, number]): { values: number[]; decimals: number } {
  const span = b - a;
  if (!(span > 0)) return { values: [], decimals: 0 };
  const power = 10 ** Math.floor(Math.log10(span / 3));
  const multiple = [1, 2, 2.5, 5, 10].find((m) => span / (m * power) <= 4) ?? 10;
  const step = multiple * power;
  const values: number[] = [];
  for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) {
    values.push(Math.abs(v) < step * 1e-6 ? 0 : v);
  }
  const decimals = Math.max(0, Math.ceil(-Math.log10(step) - 1e-9))
    + (multiple === 2.5 ? 1 : 0);
  return { values, decimals };
}

export default function GalaxyCorner({ data }: { data?: CornerData }) {
  const clipPrefix = `corner-${useId().replace(/:/g, "")}`;
  if (!data?.available || !data.variables?.length || !data.cells || !data.diagonal) {
    return <Empty>{data?.detail ?? "Rebuild cached plots to draw the joint distributions."}</Empty>;
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
  const ticks = variables.map((variable) => niceTicks(variable.domain));
  const cellByPosition = new Map(data.cells.map((cell) => [`${cell.row},${cell.col}`, cell]));
  const fractions = data.contour_mass_fractions ?? [];
  const offWindow = variables.filter((variable) => (variable.outside_fraction?.q1 ?? 0) >= 0.01
    || (variable.outside_fraction?.model ?? 0) >= 0.01);

  const grid = (row: number, col: number, horizontal: boolean) => <g className="corner-grid">
    {ticks[col].values.map((value) => <line key={`x${value}`}
      x1={px(col, value)} x2={px(col, value)} y1={cellY(row)} y2={cellY(row) + CELL} />)}
    {horizontal && ticks[row].values.map((value) => <line key={`y${value}`}
      x1={cellX(col)} x2={cellX(col) + CELL} y1={py(row, value)} y2={py(row, value)} />)}
  </g>;

  const diagonalCell = (index: number) => {
    const diagonal = data.diagonal![index];
    const peak = Math.max(...diagonal.q1, ...diagonal.model, 1e-12) * 1.08;
    const base = cellY(index) + CELL;
    const step = (density: number[]) => {
      let d = `M${px(index, diagonal.edges[0]).toFixed(1)},${base.toFixed(1)}`;
      density.forEach((value, bin) => {
        const top = base - (value / peak) * CELL;
        d += `L${px(index, diagonal.edges[bin]).toFixed(1)},${top.toFixed(1)}`
          + `L${px(index, diagonal.edges[bin + 1]).toFixed(1)},${top.toFixed(1)}`;
      });
      return `${d}L${px(index, diagonal.edges[diagonal.edges.length - 1]).toFixed(1)},${base.toFixed(1)}`;
    };
    return <g key={`d${index}`} clipPath={`url(#${clipPrefix}-${index}-${index})`}>
      {grid(index, index, false)}
      <path d={step(diagonal.q1)} fill={SOURCE_COLOR.q1} fillOpacity={0.16}
        stroke={SOURCE_COLOR.q1} strokeWidth={1.2} />
      <path d={step(diagonal.model)} fill="none"
        stroke={SOURCE_COLOR.model} strokeWidth={1.4} />
    </g>;
  };

  const jointCell = (row: number, col: number) => {
    const cell = cellByPosition.get(`${row},${col}`);
    const color = SOURCE_COLOR[cell?.source ?? (row > col ? "q1" : "model")];
    return <g key={`c${row}-${col}`} clipPath={`url(#${clipPrefix}-${row}-${col})`}>
      {grid(row, col, true)}
      {cell?.contours.map((contour) => {
        const style = contourStyle(contour.mass_fraction);
        return contour.paths.map((path, pathIndex) => <path
          key={`${contour.mass_fraction}-${pathIndex}`}
          d={path.x.map((x, i) => `${i ? "L" : "M"}${px(col, x).toFixed(1)},${py(row, path.y[i]).toFixed(1)}`).join("")}
          fill="none" stroke={color} strokeWidth={style.width} strokeOpacity={style.opacity}
          strokeLinejoin="round" />);
      })}
      {cell && !cell.contours.length && <text className="corner-empty"
        x={cellX(col) + CELL / 2} y={cellY(row) + CELL / 2}>no data</text>}
      {cell?.source === "q1" && <text className="corner-count"
        x={cellX(col) + 5} y={cellY(row) + 11}>n = {cell.rows.toLocaleString()}</text>}
    </g>;
  };

  const positions = Array.from({ length: n * n }, (_, i) => [Math.floor(i / n), i % n] as const);
  const formatTick = (value: number, decimals: number) => value.toFixed(decimals);

  return <div className="galaxy-corner">
    <div className="galaxy-corner__frame">
      <svg viewBox={`0 0 ${width} ${height}`} role="img"
        aria-label="Corner plot of VIS magnitude, SFR, radius, and three colours: Euclid Q1 below the diagonal, model draws above">
        <defs>
          {positions.map(([row, col]) => <clipPath key={`${row}-${col}`} id={`${clipPrefix}-${row}-${col}`}>
            <rect x={cellX(col)} y={cellY(row)} width={CELL} height={CELL} />
          </clipPath>)}
        </defs>
        {positions.map(([row, col]) => <rect key={`bg${row}-${col}`} className="corner-cell"
          x={cellX(col)} y={cellY(row)} width={CELL} height={CELL} />)}
        {positions.map(([row, col]) => row === col ? diagonalCell(row) : jointCell(row, col))}

        {variables.map((variable, col) => <text key={`t${col}`} className="corner-title"
          x={cellX(col) + CELL / 2} y={MARGIN.top - 10}>{variable.label} [{variable.unit}]</text>)}
        {variables.map((variable, row) => {
          const y = cellY(row) + CELL / 2;
          return <text key={`r${row}`} className="corner-title"
            x={14} y={y} transform={`rotate(-90 14 ${y})`}>{variable.label}</text>;
        })}
        {variables.map((_variable, col) => ticks[col].values.map((value) => <text
          key={`bx${col}-${value}`} className="corner-tick corner-tick--bottom"
          x={px(col, value)} y={cellY(n - 1) + CELL + 13}
        >{formatTick(value, ticks[col].decimals)}</text>))}
        {variables.map((_variable, row) => row === 0 ? null : ticks[row].values.map((value) => <text
          key={`ly${row}-${value}`} className="corner-tick corner-tick--left"
          x={MARGIN.left - 6} y={py(row, value)}
        >{formatTick(value, ticks[row].decimals)}</text>))}
        {variables.map((_variable, row) => row === n - 1 ? null : ticks[row].values.map((value) => <text
          key={`ry${row}-${value}`} className="corner-tick corner-tick--right"
          x={cellX(n - 1) + CELL + 6} y={py(row, value)}
        >{formatTick(value, ticks[row].decimals)}</text>))}
      </svg>
    </div>
    <div className="galaxy-plot__definitions galaxy-corner__legend">
      <span><i style={{ background: SOURCE_COLOR.q1 }} />
        Lower triangle: {data.q1_rows?.toLocaleString() ?? "—"} Euclid Q1 colour-model rows, raw forced-photometry colours. SFR panels use PHZ-valid rows and Rₑ panels resolved Sérsic fits; n counts rows inside each panel window.
      </span>
      <span><i style={{ background: SOURCE_COLOR.model }} />
        Upper triangle: {data.model_draws?.toLocaleString() ?? "—"} draws from the fitted model{data.vis_range ? ` inside VIS ${data.vis_range[0].toFixed(2)}–${data.vis_range[1].toFixed(2)}` : ""}, deconvolved colours.
      </span>
      <span>
        Diagonal: both samples at unit area. Contours enclose {fractions.map((f) => Math.round(100 * f)).sort((a, b) => a - b).join(" / ")}% of the mass inside each window. Each cell plots its column variable (x) against its row variable (y).
      </span>
      {offWindow.length > 0 && <span>
        Outside the plotted window (Q1 / model): {offWindow.map((variable) => `${variable.label} ${(100 * (variable.outside_fraction?.q1 ?? 0)).toFixed(1)}% / ${(100 * (variable.outside_fraction?.model ?? 0)).toFixed(1)}%`).join(" · ")}. PHZ log SFR values below −5 (effectively zero SFR) are always off-window.
      </span>}
    </div>
  </div>;
}
