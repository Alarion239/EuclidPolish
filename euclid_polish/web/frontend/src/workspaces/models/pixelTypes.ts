/* Models › Knees › Pixel types: the pure model of the view (spec
   docs/superpowers/specs/2026-10-15-knee-attribution-design.md › Pixel
   types). The figures are drawn by the server's analysis and comparison jobs
   (helpers/knee_pixel_types_report.py); this module names them, picks the
   file for a figure × field set × band × grouping, lists the models a
   comparison can take and says whether two of them sample the same pixels. */
import type { KneeRunSummary, PixelAnalysis, PixelComparison, PixelReports } from "./kneesApi";
import { modelShort, type FieldSet, type KneeBand } from "./knees";

/** Groups by the effective knee itself, or by its excess over the chain-rule
 *  baseline (what the network itself prefers). */
export type GroupBasis = "learned" | "raw";
export type PixelFigure = "map" | "groups" | "features" | "types" | "mix" | "look" | "chain";
export type CompareFigure = "curves" | "diff" | "look" | "pairs";

type FigureSpec<K extends string> = {
  key: K; label: string; title: string;
  /** One figure per band (else one for all four). */
  perBand: boolean;
  /** Drawn twice: by the knee and by the network's own preference. */
  basis: boolean;
  file: (kind: FieldSet, band: KneeBand, basis: GroupBasis) => string;
  caption: string;
};

const learned = (basis: GroupBasis) => (basis === "learned" ? "_learned" : "");

export const PIXEL_FIGURES: readonly FigureSpec<PixelFigure>[] = [
  { key: "map", label: "Knee vs brightness", perBand: false, basis: false,
    title: "The knee each pixel is read through against its brightness, with the chain-rule baseline",
    file: (kind) => `knee_map_${kind}.png`,
    caption: "Each pixel's effective knee (Σ share × log knee, its own band's local 3×3 split) against its brightness, both in "
      + "the band's noise σ; grey is how common such pixels are. The dotted line is the chain-rule baseline: the knee a network "
      + "with the same gradient on every knee would get from the same LR pixels. The gap above it is what the network chose." },
  { key: "groups", label: "Pixel groups", perBand: true, basis: true,
    title: "Pixels as bright as each other, read through a lower, the usual or a higher knee",
    file: (kind, band, basis) => `groups${learned(basis)}_${kind}_${band}.png`,
    caption: "Within each decade of brightness, the 20 % of pixels read through the lowest knee, the middle 60 % and the 20 % "
      + "read through the highest: median stamps (σ units, each turned so the brighter side is up) of the SR and the "
      + "truth (synthetic) or the LR (real), and where sources sit around them (the fraction of stamps above 1σ)." },
  { key: "features", label: "What sets them apart", perBand: false, basis: true,
    title: "How the lower- and higher-knee pixels differ from pixels as bright",
    file: (kind, _band, basis) => `group_features${learned(basis)}_${kind}.png`,
    caption: "Stamp measures of the three groups per brightness decade, on the truth (synthetic) or the smoothed LR (real): the "
      + "distance to the nearest pixel above 1σ, the brightest pixel nearby, whether the pixel is a local peak, the slope." },
  { key: "types", label: "Pixel types", perBand: true, basis: false,
    title: "Knee types: k-means of the knee-share profile, ordered by effective knee",
    file: (kind, band) => `types_${kind}_${band}.png`,
    caption: "The recurring knee profiles (k-means of the shares' log-ratios, every brightness decade weighted alike), each "
      + "with how common it is, its classes, its median stamps and its brightness spread." },
  { key: "mix", label: "Types by brightness", perBand: false, basis: false,
    title: "Which knee types make up each brightness decade",
    file: (kind) => `types_by_brightness_${kind}.png`,
    caption: "The share of each knee type in every brightness decade (weighted by the pixels each sample stands for)." },
  { key: "look", label: "Look-around", perBand: false, basis: false,
    title: "How far around a pixel the knee choice looks",
    file: (kind) => `look_around_${kind}.png`,
    caption: "At fixed brightness, the rank correlation of a pixel's knee with the brightest scene pixel within r. A curve "
      + "that rises with r says the knee follows the light around the pixel, not the pixel itself." },
  { key: "chain", label: "Chain rule", perBand: false, basis: false,
    title: "How much of the knee choice the chain rule alone explains",
    file: (kind) => `mechanics_${kind}.png`,
    caption: "Each pixel's effective knee against the chain-rule baseline's: on the diagonal the knee is the stretch's "
      + "derivative alone; above it the network leans on higher knees than that." },
];

export const COMPARE_FIGURES: readonly Omit<FigureSpec<CompareFigure>, "basis">[] = [
  { key: "curves", label: "Knee curves", perBand: false, title: "Both models' effective knee and own preference per brightness decade",
    file: (kind) => `compare_knee_${kind}.png`,
    caption: "The median effective knee (20–80 % band) per brightness decade, both models overlaid; below, the excess over the "
      + "chain-rule baseline (each model's own preference)." },
  { key: "diff", label: "Share difference", perBand: false, title: "Mean knee share of B minus A, per brightness decade and knee",
    file: (kind) => `compare_share_diff_${kind}.png`,
    caption: "Red: model B leans on that knee more than A at that brightness. The knees below the noise are near-copies of one "
      + "log stretch, so shifts among them are as large between two seeds of one recipe (±0.6 dex): read them with care." },
  { key: "look", label: "Look-around", perBand: false, title: "Both models' look-around, overlaid",
    file: (kind) => `compare_look_around_${kind}.png`,
    caption: "Each model's rank correlation of the knee with the brightest scene pixel within r (solid A, dashed B)." },
  { key: "pairs", label: "Same pixels", perBand: true, title: "The same pixels, read by B through a lower, the same or a higher knee",
    file: (kind, band) => `compare_pairs_${kind}_${band}.png`,
    caption: "The pixels both runs sampled, grouped within each brightness decade by Δ = log q̄_B − log q̄_A: median stamps of "
      + "both models' SR and of the scene." },
];

export const PIXEL_FIGURE_KEYS = PIXEL_FIGURES.map((f) => f.key);
export const COMPARE_FIGURE_KEYS = COMPARE_FIGURES.map((f) => f.key);
export const parsePixelFigure = (v: string): PixelFigure =>
  ((PIXEL_FIGURE_KEYS as readonly string[]).includes(v) ? (v as PixelFigure) : "groups");
export const parseCompareFigure = (v: string): CompareFigure =>
  ((COMPARE_FIGURE_KEYS as readonly string[]).includes(v) ? (v as CompareFigure) : "curves");
export const parseBasis = (v: string): GroupBasis => (v === "raw" ? "raw" : "learned");
export const pixelFigure = (key: PixelFigure) => PIXEL_FIGURES.find((f) => f.key === key) ?? PIXEL_FIGURES[1];
export const compareFigure = (key: CompareFigure) => COMPARE_FIGURES.find((f) => f.key === key) ?? COMPARE_FIGURES[0];

/* ── normalisation ─────────────────────────────────────────────────────── */

type Obj = Record<string, unknown>;
const isObj = (v: unknown): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);
const str = (v: unknown, d = "") => (typeof v === "string" ? v : v == null ? d : String(v));
const strs = (v: unknown) => (Array.isArray(v) ? v.map((x) => str(x)) : []);
const kinds = (v: unknown) => strs(v).filter((k): k is FieldSet => k === "synthetic" || k === "real");
const record = (v: unknown): Record<string, string> =>
  (isObj(v) ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, str(x)])) : {});

export function normalisePixelReports(raw: unknown): PixelReports {
  const o = isObj(raw) ? raw : {};
  const analyses = (Array.isArray(o.analyses) ? o.analyses : []).filter(isObj).filter((a) => a.folder != null).map((a) => ({
    folder: str(a.folder), run_id: str(a.run_id), model: str(a.model), created: str(a.created),
    kinds: kinds(a.kinds), files: strs(a.files), params: isObj(a.params) ? (a.params as PixelAnalysis["params"]) : {},
  }));
  const asComparisons = (list: unknown) => (Array.isArray(list) ? list : []).filter(isObj).filter((c) => c.folder != null).map((c) => ({
    folder: str(c.folder), models: record(c.models), runs: record(c.runs), created: str(c.created), tag: str(c.tag),
    kinds: kinds(c.kinds), files: strs(c.files),
  }));
  return { analyses, comparisons: asComparisons(o.comparisons), overviews: asComparisons(o.overviews) };
}

/* ── labels ────────────────────────────────────────────────────────────── */

const plural = (n: number, w: string) => `${n} ${w}${n === 1 ? "" : "s"}`;

/** "#249 · 25 Oct 07:02 · 30 synthetic + 7 real · mixed". */
export function analysisLabel(a: PixelAnalysis, when: string): string {
  const p = a.params ?? {};
  const fields = [p.synthetic ? `${p.synthetic} synthetic` : null, p.tiles?.length ? `${p.tiles.length} real` : null]
    .filter(Boolean).join(" + ");
  return [modelShort(a.model), when, fields || null, p.sampling && p.sampling !== "class" ? p.sampling : null]
    .filter(Boolean).join(" · ");
}

/** The comparison's two specs, A first. */
export const comparisonSpecs = (c: PixelComparison): [string, string] => {
  const specs = Object.values(c.models);
  return [specs[0] ?? "", specs[1] ?? ""];
};

/** "#238 vs #237 · 25 Oct 08:26 · floor". */
export function comparisonLabel(c: PixelComparison, when: string): string {
  const [a, b] = comparisonSpecs(c);
  return [`${modelShort(b)} vs ${modelShort(a)}`, when, c.tag || null].filter(Boolean).join(" · ");
}

/* ── starting jobs ─────────────────────────────────────────────────────── */

/** A model of a finished run, as a comparison side (`value` = the POSTed
 *  `<run id>|<spec>`). */
export type ModelRef = {
  value: string; runId: string; spec: string; seed: number | null; sampling: string; version: number | null; label: string;
};

/** Every model of every finished run (not running, not failed), newest run
 *  first: what an analysis or a comparison can take. */
export function modelRefs(runs: readonly KneeRunSummary[], when: (r: KneeRunSummary) => string): ModelRef[] {
  return runs.filter((r) => r.status !== "running" && r.status !== "failed").flatMap((r) => r.models.map((spec) => ({
    value: `${r.id}|${spec}`, runId: r.id, spec, seed: r.seed ?? null, sampling: String(r.sampling ?? "class"),
    version: r.version ?? null, label: `${modelShort(spec)} · ${when(r)}${r.seed != null ? ` · seed ${r.seed}` : ""}`,
  })));
}

export type PairCheck = { ok: boolean; error?: string; notes: string[] };

/** Whether B can be compared with A pixel by pixel: two different models
 *  (or one model in two runs) under one seed. Notes what pairs few pixels. */
export function pairCheck(a: ModelRef | null | undefined, b: ModelRef | null | undefined): PairCheck {
  if (!a || !b) return { ok: false, error: "Pick both models", notes: [] };
  if (a.value === b.value) return { ok: false, error: "Pick two different models (or one model in two runs)", notes: [] };
  if (a.seed != null && b.seed != null && a.seed !== b.seed) {
    return { ok: false, error: `The runs used different seeds (${a.seed} and ${b.seed}), so they sampled different pixels`, notes: [] };
  }
  const notes: string[] = [];
  if (a.sampling === "class" || b.sampling === "class") {
    notes.push("A class-sampled run pairs only where both drew the same class pixels: mixed or uniform runs pair their whole uniform draw.");
  }
  if (a.runId !== b.runId && ((a.version ?? 0) < 5 || (b.version ?? 0) < 5)) {
    notes.push("Runs before record version 5 may have drawn other pixels per model: few may pair.");
  }
  return { ok: true, notes };
}

/** "local job: 2 models, ~4 min" (~2 min a model on this machine). */
export function analyseEstimate(nModels: number): string {
  return `local job: ${plural(nModels, "model")}, ~${Math.max(1, Math.round(2 * nModels))} min`;
}

/* ── one analysis' summary.json ────────────────────────────────────────── */

export type BandFacts = { n: number; sigma: number | null; blank: number | null; types: number };

/** The per-band facts of an analysis summary (null when absent). */
export function bandFacts(summary: unknown, kind: FieldSet, band: KneeBand): BandFacts | null {
  const s = isObj(summary) ? summary : {};
  const k = isObj(s[kind]) ? (s[kind] as Obj) : {};
  const b = isObj(k[band]) ? (k[band] as Obj) : null;
  if (!b) return null;
  const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);
  return { n: num(b.n) ?? 0, sigma: num(b.noise_sigma_e), blank: num(b.blank_fraction), types: Array.isArray(b.types) ? b.types.length : 0 };
}

/** The bands whose figure (for a per-band figure) exists in `files`. */
export function bandsWithFile(files: readonly string[], file: (band: KneeBand) => string, bands: readonly KneeBand[]): KneeBand[] {
  return bands.filter((b) => files.includes(file(b)));
}
