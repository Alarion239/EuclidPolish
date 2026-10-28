/* Models › Knees › Pixel types, drawn by the console itself: the shapes of
   the JSON the analysis and comparison jobs write next to their PNGs
   (helpers/knee_pixel_types_report.py › browser-native data), and the pure
   helpers the renderers use — the file names, the stamp decoding (8-bit
   pixels on the row's asinh scale, row 0 = top: the scene's brighter side is
   up) and the stamp colour maps. A folder without these files (an analysis
   from before them) falls back to its PNGs. */
import type { FieldSet, KneeBand } from "./knees";

export type Quant = (number | null)[] | null;
export type Basis = "raw" | "learned";

export type CurveBin = { q20: number | null; median: number | null; q80: number | null; mech: number | null; n: number } | null;

export type TypeView = {
  index: number; prevalence: number | null; n: number; mean_share: (number | null)[]; log_q: number | null;
  bright_median: number | null; dist: number | null; classes: Record<string, number | null>; hist: (number | null)[];
};

export type AnalysisBandView = {
  sigma: number | null; blank_fraction: number | null; n: number;
  /** log₁₀ knee / σ of every knee. */
  logk: (number | null)[];
  heat: { x_edges: number[]; y_edges: number[]; z: (number | null)[][] };
  curve: CurveBin[];
  /** basis → group → feature key → per brightness decade. */
  features: Record<Basis, Record<string, Record<string, (number | null)[]>>>;
  types: TypeView[];
  type_mix: ((number | null)[] | null)[];
  /** basis → decade index → ρ per radius. */
  look: Record<string, Record<string, (number | null)[]>>;
  mech: { edges: number[]; z: (number | null)[][]; gap: (number | null)[] };
};

export type AnalysisView = {
  kind: FieldSet; model: string; knees: number[]; edges_sigma: number[]; centres_log: number[]; bin_labels: string[];
  radii: number[]; type_hist_x: number[]; features: { key: string; label: string; log: boolean }[]; groups: string[];
  bands: Partial<Record<KneeBand, AnalysisBandView>>;
};

export type StampCell = {
  sr: string; ref: string; occ: string; radial: (number | null)[];
  n?: number; log_q?: number | null; learned?: number | null; dist?: number | null; norm?: number[];
};
export type StampRow = { bin: number; label: string; norm: number[]; cells: Record<string, StampCell> };
export type StampsView = { side: number; radial_px: number[]; groups: Partial<Record<Basis, StampRow[]>>; types: StampCell[] };

export type ModelCurves = { knee: Quant[]; preference: Quant[]; mech: (number | null)[] };
export type CompareView = {
  kind: FieldSet; models: string[]; centres_log: number[]; bin_labels: string[]; radii: number[];
  curves: Partial<Record<KneeBand, Record<string, ModelCurves>>>;
  share_diff?: Partial<Record<KneeBand, { knees: number[]; matrix: (number | null)[][] }>>;
  /** band → model → decade label → ρ per radius. */
  look?: Partial<Record<KneeBand, Record<string, Record<string, (number | null)[]>>>>;
};

export type PairCell = { n: number; delta: number | null; sr_a: string; sr_b: string; ref: string };
export type PairsView = { side: number; models: [string, string]; n: number; rows: { bin: number; label: string; norm: number[]; cells: Record<string, PairCell> }[] };

/* ── files ─────────────────────────────────────────────────────────────── */

export const viewFile = (kind: FieldSet) => `view_${kind}.json`;
export const stampsFile = (kind: FieldSet, band: KneeBand) => `stamps_${kind}_${band}.json`;
export const pairsFile = (kind: FieldSet, band: KneeBand) => `pairs_${kind}_${band}.json`;

type Obj = Record<string, unknown>;
const isObj = (v: unknown): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);
/** The server's view, or null for anything else (a 404 body, an old file). */
export const asAnalysisView = (raw: unknown): AnalysisView | null =>
  (isObj(raw) && isObj(raw.bands) && Array.isArray(raw.centres_log) ? (raw as AnalysisView) : null);
export const asStamps = (raw: unknown): StampsView | null =>
  (isObj(raw) && typeof raw.side === "number" && isObj(raw.groups) ? (raw as StampsView) : null);
export const asCompareView = (raw: unknown): CompareView | null =>
  (isObj(raw) && isObj(raw.curves) && Array.isArray(raw.models) ? (raw as CompareView) : null);
export const asPairs = (raw: unknown): PairsView | null =>
  (isObj(raw) && typeof raw.side === "number" && Array.isArray(raw.rows) ? (raw as PairsView) : null);

/* ── stamps ────────────────────────────────────────────────────────────── */

/** base64 → bytes (one per pixel). */
export function decodeStamp(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** matplotlib's magma and cividis, ten anchors each (theme-independent: a
 *  stamp is an image, it reads the same on light and dark). */
export const MAGMA = ["#000004", "#180f3d", "#440f76", "#721f81", "#9e2f7f", "#cd4071", "#f1605d", "#fd9668", "#feca8d", "#fcfdbf"];
export const CIVIDIS = ["#00224e", "#123570", "#3b496c", "#575d6d", "#707173", "#8a8779", "#a69d75", "#c4b56c", "#e4cf5b", "#fee838"];

/** A 256-entry RGB table interpolated over `anchors`. */
export function lut(anchors: readonly string[]): Uint8ClampedArray {
  const rgb = anchors.map((h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)));
  const out = new Uint8ClampedArray(256 * 3);
  for (let v = 0; v < 256; v++) {
    const x = (v / 255) * (rgb.length - 1);
    const i = Math.min(rgb.length - 2, Math.floor(x));
    const t = x - i;
    for (let c = 0; c < 3; c++) out[v * 3 + c] = Math.round(rgb[i][c] + (rgb[i + 1][c] - rgb[i][c]) * t);
  }
  return out;
}

/** Bytes → RGBA pixels through `table`. */
export function stampRgba(bytes: Uint8Array, table: Uint8ClampedArray): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(new ArrayBuffer(bytes.length * 4));
  for (let i = 0; i < bytes.length; i++) {
    const v = bytes[i] * 3;
    out[i * 4] = table[v];
    out[i * 4 + 1] = table[v + 1];
    out[i * 4 + 2] = table[v + 2];
    out[i * 4 + 3] = 255;
  }
  return out;
}

/** "−0.3 … 42σ (asinh)": a row's colour scale in words. */
export function normText(norm: readonly number[] | undefined): string {
  if (!norm || norm.length < 2) return "";
  const f = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toPrecision(2));
  return `${f(norm[0])} … ${f(norm[1])}σ, asinh`;
}

/* ── curves ────────────────────────────────────────────────────────────── */

/** A quantile triple per decade → Plot's y / low / high (null where thin). */
export function quantSeries(q: readonly Quant[]): { y: (number | null)[]; low: (number | null)[]; high: (number | null)[] } {
  return {
    y: q.map((v) => (v ? v[1] ?? null : null)),
    low: q.map((v) => (v ? v[0] ?? null : null)),
    high: q.map((v) => (v ? v[2] ?? null : null)),
  };
}

/** "10⁻²σ"-style text of a log₁₀ σ value. */
export function sigmaText(logv: number | null | undefined): string {
  if (logv == null || !Number.isFinite(logv)) return "—";
  const v = 10 ** logv;
  return `${v >= 100 ? v.toFixed(0) : v >= 1 ? v.toPrecision(2) : v.toPrecision(1)}σ`;
}
