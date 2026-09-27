/* Real-field diagnostics (GET /api/inference/diagnostics.json) matched to the
 * synthetic STARFULL evaluation (GET /ensemble/evals.json): model–model
 * angular cross-correlation r_ij(d), member σ vs brightness and the RBF
 * combiners' pixel occupancy. Pure shaping — the plots live in
 * FieldDiagnostics.tsx. Both sides are put on SHARED axes and a common
 * (coarser) binning so the real and synthetic panels compare cell by cell. */
import type { Evals } from "../../ensemble/api";

export type Num = number | null;

export type FieldManifest = {
  field_id: string; ra: number; dec: number; field_size?: number; grid_side?: number;
  count: number; member_labels?: string[]; combiner_kinds?: string[];
  /** The members that ran (default: only the production gate's); older
   *  caches ran every member and lack these. */
  run_member_labels?: string[]; member_scope?: "gate" | "all";
};
export type FieldStatus = { field: FieldManifest | null; field_size?: number };

export type ModelPower = {
  k: Num[]; r_pairs: Num[][]; r_cross: Num[];
  pair_indices?: [number, number][]; samples?: number; pixel_scale_arcsec: number;
};
export type Occupancy = {
  mode: "histogram" | "heat"; x_edges: number[]; y_edges?: number[];
  counts: number[] | number[][]; x_label: string; y_label?: string; pixel_count: number;
};
export type FieldDiagnostics = {
  version: number; member_labels: string[]; member_scope?: "gate" | "all"; n_ensemble_members?: number;
  model_power: ModelPower;
  std_brightness: { x_edges: number[]; y_edges: number[]; counts: number[][]; x_label: string; y_label: string };
  combiners: Record<string, Occupancy>;
};

export type Pt = { x: number; y: number };
export type Curve = { x: number[]; y: number[] };

/** The two RBF combiners whose occupancy has a synthetic counterpart. */
export const RBF_KINDS: Record<string, string> = {
  raw_incremental_minmeanmax_rbf: "RBF",
  raw_incremental_frozen_minmeanmax_rbf: "frozen RBF",
};

/** The r(d) plots end at this separation (the last measured bin is held). */
export const CROSS_X_DOMAIN: [number, number] = [0.05, 10];

/** `(transform(x), y)` for positive finite x and finite y, sorted by x. */
/** How many members made the field's SR and diagnostics: "20 of 30
 *  members (the production gate's)", or "30 members" when every one ran. */
export function fieldMembersText(f: Pick<FieldManifest, "member_labels" | "run_member_labels" | "member_scope">): string {
  const total = f.member_labels?.length ?? 0;
  const ran = f.run_member_labels?.length ?? total;
  const noun = (n: number) => `member${n === 1 ? "" : "s"}`;
  if (ran >= total) return `${total} ${noun(total)}`;
  return `${ran} of ${total} ${noun(total)}${f.member_scope === "gate" ? " (the production gate's)" : ""}`;
}

export function transformedSeries(xs: readonly Num[], ys: readonly Num[], transform: (v: number) => number): Pt[] {
  const out: Pt[] = [];
  xs.forEach((v, i) => {
    const y = ys[i];
    if (v == null || !Number.isFinite(v) || v <= 0 || y == null || !Number.isFinite(y)) return;
    const x = transform(v);
    if (Number.isFinite(x)) out.push({ x, y });
  });
  return out.sort((a, b) => a.x - b.x);
}

/** Holds the last value out to `endX` (a flat tail beyond the last bin). */
export function extendTo(points: Pt[], endX: number): Pt[] {
  if (!points.length || points[points.length - 1].x >= endX) return points;
  return [...points, { x: endX, y: points[points.length - 1].y }];
}

const curve = (points: Pt[]): Curve => ({ x: points.map((p) => p.x), y: points.map((p) => p.y) });

/** Pairs + median r(d) of one side. The real field is in spatial frequency k
 *  [cycles/″] → d = 0.5 / k; the synthetic evals are already in θ [″]. */
export function crossCurves(xs: readonly Num[], pairs: readonly (readonly Num[])[], median: readonly Num[],
  inFrequency: boolean): { pairs: Curve[]; median: Curve } | null {
  const tf = inFrequency ? (k: number) => 0.5 / k : (d: number) => d;
  const med = extendTo(transformedSeries(xs, median, tf), CROSS_X_DOMAIN[1]);
  if (!med.length) return null;
  return {
    pairs: pairs.map((row) => extendTo(transformedSeries(xs, row, tf), CROSS_X_DOMAIN[1])).filter((p) => p.length).map(curve),
    median: curve(med),
  };
}

/** Extent of every finite value of the arrays ([0, 1] when there is none). */
export function sharedDomain(...arrays: readonly (readonly Num[])[]): [number, number] {
  let lo = Infinity, hi = -Infinity;
  for (const a of arrays) for (const v of a) if (v != null && Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (lo === Infinity) return [0, 1];
  return lo === hi ? [lo, lo + 1] : [lo, hi];
}

/** `bins` equal bins over the domain (bins + 1 edges). */
export function displayEdges(domain: [number, number], bins: number): number[] {
  const n = Math.max(1, Math.floor(bins));
  return Array.from({ length: n + 1 }, (_, i) => domain[0] + (domain[1] - domain[0]) * i / n);
}

function edgeBin(edges: readonly number[], v: number): number {
  const n = edges.length - 1;
  if (n < 1 || v < edges[0] || v > edges[n]) return -1;
  if (v === edges[n]) return n - 1;
  const i = Math.floor((v - edges[0]) / (edges[1] - edges[0]));
  return i >= 0 && i < n ? i : -1;
}

/** Re-bins a 2-D histogram onto uniform target edges by cell centre (the
 *  counts of cells whose centre falls outside the target are dropped). */
export function rebinHeat(z: readonly (readonly number[])[], srcX: readonly number[], srcY: readonly number[],
  dstX: readonly number[], dstY: readonly number[]): number[][] {
  const out = Array.from({ length: dstX.length - 1 }, () => Array<number>(dstY.length - 1).fill(0));
  z.forEach((row, i) => {
    const ti = edgeBin(dstX, (srcX[i] + srcX[i + 1]) / 2);
    if (ti < 0) return;
    row.forEach((count, j) => {
      const tj = edgeBin(dstY, (srcY[j] + srcY[j + 1]) / 2);
      if (tj >= 0 && Number.isFinite(count)) out[ti][tj] += count;
    });
  });
  return out;
}

const finite = (a: readonly Num[] | undefined | null): number[] =>
  (a ?? []).filter((v): v is number => v != null && Number.isFinite(v));

export type HeatPair = {
  xDomain: [number, number]; yDomain: [number, number]; xEdges: number[]; yEdges: number[];
  real: number[][]; synthetic: number[][] | null; xLabel: string; yLabel: string;
};

/** Real and synthetic 2-D histograms on shared axes and the coarser of the
 *  two binnings. `synthetic` is null when that side has no usable histogram. */
export function heatPair(
  real: { z: number[][]; xEdges: number[]; yEdges: number[] },
  syn: { z: number[][]; xEdges: readonly Num[]; yEdges: readonly Num[] } | null,
  labels: { x: string; y: string },
): HeatPair {
  const sx = finite(syn?.xEdges), sy = finite(syn?.yEdges);
  const synOk = !!syn && sx.length >= 2 && sy.length >= 2 && !!syn.z.length;
  const xDomain = sharedDomain(real.xEdges, synOk ? sx : []);
  const yDomain = sharedDomain(real.yEdges, synOk ? sy : []);
  const xBins = Math.min(real.xEdges.length - 1, synOk ? sx.length - 1 : Infinity);
  const yBins = Math.min(real.yEdges.length - 1, synOk ? sy.length - 1 : Infinity);
  const xEdges = displayEdges(xDomain, xBins), yEdges = displayEdges(yDomain, yBins);
  return {
    xDomain, yDomain, xEdges, yEdges, xLabel: labels.x, yLabel: labels.y,
    real: rebinHeat(real.z, real.xEdges, real.yEdges, xEdges, yEdges),
    synthetic: synOk && syn ? rebinHeat(syn.z, sx, sy, xEdges, yEdges) : null,
  };
}

/** σ-vs-brightness: the real field against the synthetic `bright_std`. */
export function brightnessPair(d: FieldDiagnostics, evals: Evals | null | undefined): HeatPair | null {
  const s = d.std_brightness;
  if (!s?.counts?.length || s.x_edges.length < 2 || s.y_edges.length < 2) return null;
  const b = evals?.bright_std;
  return heatPair({ z: s.counts, xEdges: s.x_edges, yEdges: s.y_edges },
    b ? { z: b.hist, xEdges: b.bright_edges, yEdges: b.std_edges } : null,
    { x: s.x_label, y: s.y_label });
}

export type OccupancyView =
  | { kind: string; label: string; mode: "histogram"; pixels: number; x: number[]; y: number[]; xDomain: [number, number]; xLabel: string }
  | { kind: string; label: string; mode: "heat"; pixels: number; heat: HeatPair };

/** The RBF combiners' pixel occupancy on the real field (a histogram or a
 *  2-D min/max-member map, the latter paired with the synthetic one). */
export function occupancyViews(d: FieldDiagnostics, evals: Evals | null | undefined): OccupancyView[] {
  const axis = evals?.combiner_feature_error?.axes?.min_max;
  return Object.entries(d.combiners ?? {}).filter(([kind]) => kind in RBF_KINDS).map(([kind, c]): OccupancyView => {
    const label = RBF_KINDS[kind];
    if (c.mode === "histogram") {
      const counts = c.counts as number[];
      return {
        kind, label, mode: "histogram", pixels: c.pixel_count, xLabel: c.x_label,
        x: counts.map((_, i) => (c.x_edges[i] + c.x_edges[i + 1]) / 2),
        y: counts.map((n) => Math.log10(n + 1)),
        xDomain: [c.x_edges[0], c.x_edges[c.x_edges.length - 1]],
      };
    }
    const model = axis?.models?.[kind];
    return {
      kind, label, mode: "heat", pixels: c.pixel_count,
      heat: heatPair({ z: c.counts as number[][], xEdges: c.x_edges, yEdges: c.y_edges ?? [0, 1] },
        model && axis ? { z: model.counts, xEdges: axis.edges[0] ?? [], yEdges: axis.edges[1] ?? [] } : null,
        { x: c.x_label, y: c.y_label ?? "feature" }),
    };
  });
}

/** `n + 1` evenly spaced ticks over [lo, hi]. */
export function evenTicks(lo: number, hi: number, n = 4): { v: number; label: string }[] {
  return Array.from({ length: n + 1 }, (_, i) => {
    const v = lo + (hi - lo) * i / n;
    return { v, label: Number.isInteger(v) ? String(v) : v.toFixed(1) };
  });
}
