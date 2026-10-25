/* Pure shaping behind Models › Knees (knee attribution: which asinh knees a
   multi-knee model relies on, for which pixels). No React here:
   - normalise the server's payloads (runs, models, the run record) so a
     partial record (a run still writing) never breaks the view;
   - the brightness curves (bin centres = geometric means of the edges, null
     bins skipped), the class bars, the gradient-vs-ablation agreement, the
     cross-band matrix and the field explorer's dots (dominant knee);
   - the split drawn (the receptive-field gradient, the local 3×3 gradient
     or the ablation) and the input band whose own knee split is drawn (all:
     the σ-weighted split over every input band), with their fallbacks (a
     record before the local or the per-input split; the ablation has no
     per-input split), the per-electron (version 1) warning and the field's
     noise σ line;
   - one colour per knee value on a sequential ramp (faint knee → bright
     knee), the same value the same colour in every panel;
   - the run form: defaults, validation (the sampling and its samples per
     band too), the confirm's estimate and the POSTed fields; the field
     image URL (the backend's slug rule). */
import { viridis } from "../../colors";
import { formatNumber, formatPow10 } from "../../format";
import type {
  ClassShares, KneeAggregate, KneeField, KneeModelChoice, KneeModelResult, KneeRunParams, KneeRunRecord, KneeRunSummary, KneeSamples,
  KneeSampling,
} from "./kneesApi";
import { memberNumber } from "./model";

export const KNEE_BANDS = ["VIS", "Y_E", "J_E", "H_E"] as const;
export type KneeBand = (typeof KNEE_BANDS)[number];
export const BAND_LABEL: Record<KneeBand, string> = { VIS: "VIS", Y_E: "Y", J_E: "J", H_E: "H" };

/** The synthetic truth classes, in the spec's priority order. */
export const SYNTHETIC_CLASSES = ["saturated", "star core", "star wing", "galaxy core", "galaxy outskirts", "faint", "sky"] as const;
/** The real fields' classes (the predicted-image bins, then blank). */
export const REAL_CLASSES = ["sky", "faint", "mid", "bright", "core", "blank"] as const;

export type FieldSet = "synthetic" | "real";
/** The split: the gradient summed over the receptive field (`gradient`) or
 *  over the 3×3 LR pixels under the sample (`local`), or the ablation. */
export type ShareKind = "gradient" | "local" | "ablation";
export type ImageLayer = "sr" | "lr";

export const FIELD_SETS: readonly FieldSet[] = ["synthetic", "real"];
export const MAX_SYNTHETIC = 100;
export const MAX_PER_CLASS = 256;
export const DEFAULT_PER_CLASS = 24;
export const MAX_SEED = 2 ** 31 - 1;
/** Models overlaid at once (small multiples). */
export const MAX_OVERLAY = 4;
/** The confirm's cost model, per field per model: a fixed part (the SR and the
 *  classes) plus a part per sample class (gradients and ablation, linear in
 *  samples per class). Measured on an M4 CPU: ~19 s per field at 24 per class. */
export const SECONDS_PER_FIELD = 2;
export const SECONDS_PER_CLASS_SAMPLE = 0.7;
/** Per sampled pixel (all four bands count): with the ablation check, and
 *  the gradient alone (measured: 1024 per band + 48 per class ≈ 52 s/field). */
export const SECONDS_PER_SAMPLE = { ablation: 0.025, gradient: 0.009 };
/** How pixels are sampled: per class, uniformly, or both pooled. */
export const SAMPLINGS: readonly KneeSampling[] = ["class", "uniform", "mixed"];
export const DEFAULT_PER_BAND = 1024;
export const MAX_PER_BAND = 8192;

/* ── normalisation ─────────────────────────────────────────────────────── */

type Obj = Record<string, unknown>;
const isObj = (v: unknown): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);
const obj = (v: unknown): Obj => (isObj(v) ? v : {});
const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const numOr = (v: unknown, d: number) => (finite(v) ? v : d);
const numOrNull = (v: unknown): number | null => (finite(v) ? v : null);
const str = (v: unknown, d = "") => (typeof v === "string" ? v : v == null ? d : String(v));
const strs = (v: unknown): string[] => (Array.isArray(v) ? v.filter((x) => x != null).map((x) => String(x)) : []);
/** A share vector: non-finite entries read 0. */
const shares = (v: unknown): number[] => (Array.isArray(v) ? v.map((x) => (finite(x) ? x : 0)) : []);
const numbers = (v: unknown): number[] => (Array.isArray(v) ? v.map((x) => (finite(x) ? x : NaN)) : []);
/** A C × K share matrix (rows as `shares`); null when absent or empty. */
const matrix = (v: unknown): number[][] | null =>
  (Array.isArray(v) && v.length ? v.map((row) => (Array.isArray(row) ? shares(row) : [])) : null);
/** A share vector, null when absent. */
const vector = (v: unknown): number[] | null => (Array.isArray(v) ? shares(v) : null);
/** Per bin a share vector (null: no sample); the list null when absent. */
const binVectors = (v: unknown): (number[] | null)[] | null =>
  (Array.isArray(v) ? v.map((s) => (Array.isArray(s) ? shares(s) : null)) : null);
/** A band × band matrix (missing cells null, a non-array row empty). */
const crossMatrix = (v: unknown): (number | null)[][] | null =>
  (Array.isArray(v) ? v.map((row) => (Array.isArray(row) ? row.map(numOrNull) : [])) : null);

export function normaliseRuns(raw: unknown): KneeRunSummary[] {
  const list = Array.isArray(obj(raw).runs) ? (obj(raw).runs as unknown[]) : [];
  return list.filter(isObj).filter((r) => r.id != null).map((r) => ({
    id: str(r.id), created: r.created == null ? null : str(r.created), finished: r.finished == null ? null : str(r.finished),
    status: str(r.status, "done"), models: strs(r.models), synthetic: numOr(r.synthetic, 0), tiles: strs(r.tiles),
    per_class: numOrNull(r.per_class),
    sampling: r.sampling == null ? "class" : str(r.sampling), per_band: numOrNull(r.per_band),
    seed: numOrNull(r.seed), version: numOrNull(r.version),
  }));
}

export function normaliseModels(raw: unknown): KneeModelChoice[] {
  const list = Array.isArray(obj(raw).models) ? (obj(raw).models as unknown[]) : [];
  return list.filter(isObj).filter((m) => m.spec != null).map((m) => ({
    spec: str(m.spec), label: str(m.label, str(m.spec)), knees: numbers(m.knees).filter(Number.isFinite),
    depth: numOrNull(m.depth), kind: str(m.kind, "member"), available: m.available === true,
    reason: m.reason == null ? null : str(m.reason),
  }));
}

function normaliseClassShares(raw: unknown): ClassShares {
  const r = obj(raw);
  return {
    knee_share: shares(r.knee_share), ablation_share: shares(r.ablation_share), band_share: shares(r.band_share),
    knee_share_by_input: matrix(r.knee_share_by_input),
    knee_share_local: vector(r.knee_share_local), band_share_local: vector(r.band_share_local),
    knee_share_by_input_local: matrix(r.knee_share_by_input_local), n: numOr(r.n, 0),
  };
}

export function normaliseAggregate(raw: unknown): KneeAggregate | null {
  if (!isObj(raw)) return null;
  const byClass: KneeAggregate["by_class"] = {};
  for (const [band, classes] of Object.entries(obj(raw.by_class))) {
    byClass[band] = {};
    for (const [cls, v] of Object.entries(obj(classes))) byClass[band][cls] = normaliseClassShares(v);
  }
  const bb = obj(raw.by_brightness);
  const bands: KneeAggregate["by_brightness"]["bands"] = {};
  for (const [band, v] of Object.entries(obj(bb.bands))) {
    const b = obj(v);
    bands[band] = {
      knee_share: binVectors(b.knee_share) ?? [],
      knee_share_by_input: Array.isArray(b.knee_share_by_input) ? b.knee_share_by_input.map(matrix) : null,
      knee_share_local: binVectors(b.knee_share_local),
      knee_share_by_input_local: Array.isArray(b.knee_share_by_input_local) ? b.knee_share_by_input_local.map(matrix) : null,
      n: Array.isArray(b.n) ? b.n.map((x) => numOr(x, 0)) : [],
    };
  }
  return {
    by_class: byClass,
    by_brightness: { edges_e: numbers(bb.edges_e), bands },
    cross_band: crossMatrix(raw.cross_band) ?? [],
    cross_band_local: crossMatrix(raw.cross_band_local),
    n_samples: numOr(raw.n_samples, 0),
  };
}

const EMPTY_SAMPLES: KneeSamples = {
  band: [], y: [], x: [], class: [], value_e: [], knee_share: [], ablation_share: [], band_share: [], sensitivity: [], knee_share_local: [],
};

function normaliseSamples(raw: unknown): KneeSamples {
  if (!isObj(raw)) return EMPTY_SAMPLES;
  const mat = (v: unknown) => (Array.isArray(v) ? v.map(shares) : []);
  return {
    band: numbers(raw.band), y: numbers(raw.y), x: numbers(raw.x), class: strs(raw.class), value_e: numbers(raw.value_e),
    knee_share: mat(raw.knee_share), ablation_share: mat(raw.ablation_share), band_share: mat(raw.band_share),
    sensitivity: numbers(raw.sensitivity), knee_share_local: mat(raw.knee_share_local),
  };
}

function normaliseField(raw: Obj): KneeField {
  const shape = numbers(raw.shape);
  return {
    ref: str(raw.ref), kind: raw.kind === "real" ? "real" : "synthetic",
    shape: [Number.isFinite(shape[0]) ? shape[0] : 0, Number.isFinite(shape[1]) ? shape[1] : 0],
    image: typeof raw.image === "string" && raw.image ? raw.image : null,
    noise_sigma_e: Array.isArray(raw.noise_sigma_e) && raw.noise_sigma_e.length ? numbers(raw.noise_sigma_e) : null,
    samples: normaliseSamples(raw.samples),
  };
}

function normaliseModelResult(raw: unknown): KneeModelResult {
  const r = obj(raw);
  const agg = obj(r.aggregate);
  return {
    knees: numbers(r.knees).filter(Number.isFinite),
    bands: strs(r.bands).length ? strs(r.bands) : [...KNEE_BANDS],
    depth: numOrNull(r.depth),
    aggregate: { synthetic: normaliseAggregate(agg.synthetic), real: normaliseAggregate(agg.real) },
    fields: (Array.isArray(r.fields) ? r.fields : []).filter(isObj).map(normaliseField),
  };
}

/** The run record with every collection present (models in `params.models`
 *  order, then any others the record holds). */
export function normaliseRecord(raw: unknown): KneeRunRecord | null {
  if (!isObj(raw) || raw.id == null) return null;
  const p = obj(raw.params);
  const params: KneeRunParams = {
    models: strs(p.models), synthetic: numOr(p.synthetic, 0), tiles: strs(p.tiles),
    per_class: numOr(p.per_class, DEFAULT_PER_CLASS), seed: numOr(p.seed, 0),
  };
  const models: Record<string, KneeModelResult> = {};
  const raws = obj(raw.models);
  const order = [...params.models.filter((m) => m in raws), ...Object.keys(raws).filter((m) => !params.models.includes(m))];
  for (const spec of order) models[spec] = normaliseModelResult(raws[spec]);
  const errors: Record<string, string> = {};
  for (const [k, v] of Object.entries(obj(raw.errors))) errors[k] = str(v);
  return {
    version: numOrNull(raw.version),
    id: str(raw.id), created: raw.created == null ? null : str(raw.created), finished: raw.finished == null ? null : str(raw.finished),
    status: str(raw.status, "done"), params, knee_grid_note: raw.knee_grid_note == null ? null : str(raw.knee_grid_note),
    models, errors,
  };
}

/* ── labels ────────────────────────────────────────────────────────────── */

/** "10⁻³ e⁻" for a power of ten, else "3.2 e⁻". */
export function kneeLabel(q: number): string {
  if (!(q > 0)) return `${formatNumber(q)} e⁻`;
  const e = Math.log10(q);
  const r = Math.round(e);
  return Math.abs(e - r) < 1e-9 ? `${formatPow10(r)} e⁻` : `${formatNumber(q, { sig: 2 })} e⁻`;
}

/** A brightness in e⁻ (tooltips, hover text): "0.003 e⁻", "1.2k e⁻". */
export function electronText(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a >= 1e3 && a < 1e6) return `${+(v / 1e3).toPrecision(2)}k e⁻`;
  return `${formatNumber(v, { sig: 2 })} e⁻`;
}

/** `universal:member_222` → "#222", `member:member_196` → "#196 (member)". */
export function modelShort(spec: string): string {
  const [kind, name = ""] = spec.includes(":") ? [spec.slice(0, spec.indexOf(":")), spec.slice(spec.indexOf(":") + 1)] : ["", spec];
  const n = memberNumber(name);
  if (!n) return spec;
  return kind === "member" ? `#${n} (member)` : `#${n}`;
}

export const sharePct = (v: number | null | undefined, digits = 0) =>
  v == null || !Number.isFinite(v) ? "—" : `${(v * 100).toFixed(digits)}%`;

/* ── colours ───────────────────────────────────────────────────────────── */

/** The knee values of several models, ascending and de-duplicated. */
export function kneeUnion(lists: readonly (readonly number[])[]): number[] {
  const all = lists.flat().filter((q) => Number.isFinite(q) && q > 0);
  return [...new Set(all.map((q) => +q.toPrecision(6)))].sort((a, b) => a - b);
}

/** The stretch of the viridis ramp the knees use per theme: viridis' dark
 *  purple end vanishes on a dark surface and its yellow end on a light one,
 *  so each theme drops the end that lacks contrast. */
export const KNEE_RAMP: Record<"light" | "dark", [number, number]> = { light: [0.04, 0.84], dark: [0.2, 1] };

/** Knee → colour on a sequential ramp (viridis, faint knee dark → bright
 *  knee light) by its rank among `union`, so the same knee value has the same
 *  colour in every panel. `range`: the part of the ramp used (KNEE_RAMP). */
export function kneeRamp(union: readonly number[], range: [number, number] = KNEE_RAMP.light): (q: number) => string {
  const n = union.length;
  const [lo, hi] = range;
  return (q: number) => {
    const key = +q.toPrecision(6);
    let i = union.indexOf(key);
    if (i < 0) i = union.findIndex((u) => u >= key);
    if (i < 0) i = n - 1;
    return viridis(n > 1 ? lo + ((hi - lo) * i) / (n - 1) : (lo + hi) / 2);
  };
}

/* ── aggregates ────────────────────────────────────────────────────────── */

export const bandIndex = (band: string) => Math.max(0, (KNEE_BANDS as readonly string[]).indexOf(band));

/** Bin centres: the geometric mean of each pair of positive edges (the
 *  arithmetic mean otherwise). */
export function binCentres(edges: readonly number[]): number[] {
  const out: number[] = [];
  for (let i = 0; i + 1 < edges.length; i++) {
    const a = edges[i], b = edges[i + 1];
    out.push(a > 0 && b > 0 ? Math.sqrt(a * b) : (a + b) / 2);
  }
  return out;
}

/* ── the split and the input band ──────────────────────────────────────── */

/** The splits (URL `?split=`), in the control's order. */
export const SHARE_KINDS: readonly ShareKind[] = ["gradient", "local", "ablation"];

/** Whether the aggregate carries the local (3×3) splits (version 3). */
export function hasLocalSplit(agg: KneeAggregate | null | undefined): boolean {
  if (!agg) return false;
  return agg.cross_band_local != null
    || Object.values(agg.by_brightness.bands).some((b) => b.knee_share_local != null)
    || Object.values(agg.by_class).some((classes) => Object.values(classes).some((v) => v.knee_share_local != null));
}

export type SplitView = { split: ShareKind; missing: boolean };

/** The split drawn: the URL's pick, except the receptive-field gradient for
 *  `local` when an aggregate shown predates the local split (`missing`). */
export function effectiveSplit(picked: ShareKind, aggs: readonly (KneeAggregate | null | undefined)[]): SplitView {
  const shown = aggs.filter((a): a is KneeAggregate => !!a);
  const missing = picked === "local" && shown.length > 0 && !shown.every(hasLocalSplit);
  return { split: missing ? "gradient" : picked, missing };
}

/** Whose knee split is drawn (URL `?input=`): "all" = the σ-weighted split
 *  over every input band, else one input band's own split (its row of the
 *  C × K by-input matrices, rows in KNEE_BANDS order). */
export const KNEE_INPUTS = ["all", "VIS", "Y", "J", "H"] as const;
export type KneeInput = (typeof KNEE_INPUTS)[number];
/** The record version whose shares are noise-weighted (and split per input). */
export const NOISE_WEIGHTED_VERSION = 2;

/** The input's row in a by-input matrix (null: all input bands). */
export const inputIndex = (input: KneeInput): number | null => (input === "all" ? null : KNEE_INPUTS.indexOf(input) - 1);

/** Whether the aggregate carries the per-input knee split (version 2; the
 *  local one with `local`, version 3). */
export function hasInputSplit(agg: KneeAggregate | null | undefined, local = false): boolean {
  if (!agg) return false;
  const key = local ? "knee_share_by_input_local" : "knee_share_by_input";
  return Object.values(agg.by_brightness.bands).some((b) => b[key] != null)
    || Object.values(agg.by_class).some((classes) => Object.values(classes).some((v) => v[key] != null));
}

export type InputView = { input: KneeInput; disabled: boolean; missing: boolean };

/** The input split drawn under the split `kind` (the effective one): the
 *  URL's pick, except "all" under the ablation split (it removes a knee in
 *  every input band at once: `disabled`) and when an aggregate shown predates
 *  the per-input split of that gradient (`missing`). */
export function effectiveInput(picked: KneeInput, kind: ShareKind, aggs: readonly (KneeAggregate | null | undefined)[]): InputView {
  const disabled = kind === "ablation";
  const shown = aggs.filter((a): a is KneeAggregate => !!a);
  const missing = !disabled && picked !== "all" && shown.length > 0 && !shown.every((a) => hasInputSplit(a, kind === "local"));
  return { input: disabled || missing ? "all" : picked, disabled, missing };
}

/** A record before the noise weighting (version 1 or none): its shares are
 *  per input electron and lean toward NISP. */
export const isPerElectron = (rec: KneeRunRecord | null | undefined): boolean =>
  !!rec && !(rec.version != null && rec.version >= NOISE_WEIGHTED_VERSION);

/** The field's noise σ per input band in one line, "noise σ: VIS 17.6 · Y
 *  0.87 · J 0.98 · H 0.90 e⁻"; null when the record has none. */
export function noiseText(sigma: readonly number[] | null | undefined): string | null {
  if (!sigma?.some((v) => Number.isFinite(v))) return null;
  const fmt = (v: number) => (!Number.isFinite(v) ? "—" : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));
  return `noise σ: ${KNEE_BANDS.map((b, c) => `${BAND_LABEL[b]} ${fmt(sigma[c] ?? NaN)}`).join(" · ")} e⁻`;
}

export type BrightnessLine = { knee: number; index: number; x: number[]; y: number[]; n: number[] };

/** One line per knee: its mean share by pixel brightness in `band`, over the
 *  bins that hold samples (null / n = 0 bins skipped). `input`: one input
 *  band's own split (its row of each bin's by-input matrix) instead of the
 *  all-input one; `kind` "local": the local (3×3) split (the ablation has no
 *  brightness curves: it draws the receptive-field gradient). An aggregate
 *  without the local or the per-input split (an older record) draws the one
 *  it has (effectiveSplit / effectiveInput say so). */
export function brightnessLines(
  agg: KneeAggregate | null | undefined, band: string, knees: readonly number[], input: KneeInput = "all", kind: ShareKind = "gradient",
): BrightnessLine[] {
  if (!agg) return [];
  const per = agg.by_brightness.bands[String(bandIndex(band))];
  if (!per) return [];
  const centres = binCentres(agg.by_brightness.edges_e);
  const row = inputIndex(input);
  const local = kind === "local" ? per.knee_share_local ?? null : null;
  const inputs = local ? per.knee_share_by_input_local : per.knee_share_by_input;
  const byInput = row == null ? null : inputs ?? null;
  return knees.map((knee, k) => {
    const line: BrightnessLine = { knee, index: k, x: [], y: [], n: [] };
    centres.forEach((c, i) => {
      const s = byInput && row != null ? byInput[i]?.[row] : (local ?? per.knee_share)[i];
      const n = per.n[i] ?? 0;
      if (!s || !(n > 0) || !Number.isFinite(c) || !(c > 0)) return;
      const v = s[k];
      if (v == null || !Number.isFinite(v)) return;
      line.x.push(c); line.y.push(v); line.n.push(n);
    });
    return line;
  });
}

/** The samples behind a brightness panel (the sum of its bins' n). */
export function brightnessSamples(agg: KneeAggregate | null | undefined, band: string): number {
  const per = agg?.by_brightness.bands[String(bandIndex(band))];
  return per ? per.n.reduce((s, n) => s + (n > 0 ? n : 0), 0) : 0;
}

/** The x extent of the bins with samples (for shared axes), or null. */
export function brightnessExtent(lines: readonly BrightnessLine[]): [number, number] | null {
  let lo = Infinity, hi = -Infinity;
  for (const l of lines) for (const x of l.x) { lo = Math.min(lo, x); hi = Math.max(hi, x); }
  return hi >= lo ? [lo, hi] : null;
}

export type BinLead = { x: number; knee: number; share: number };

/** The knee that leads the faintest and the brightest sampled bins (the
 *  page's one-line answer); null ends when no bin has samples. */
export function edgeLeads(lines: readonly BrightnessLine[]): { faint: BinLead | null; bright: BinLead | null } {
  const xs = [...new Set(lines.flatMap((l) => l.x))].sort((a, b) => a - b);
  const lead = (x: number | undefined): BinLead | null => {
    if (x == null) return null;
    let best: BinLead | null = null;
    for (const l of lines) {
      const i = l.x.indexOf(x);
      if (i >= 0 && (!best || l.y[i] > best.share)) best = { x, knee: l.knee, share: l.y[i] };
    }
    return best;
  };
  return { faint: lead(xs[0]), bright: xs.length > 1 ? lead(xs[xs.length - 1]) : null };
}

export const classOrder = (set: FieldSet): readonly string[] => (set === "real" ? REAL_CLASSES : SYNTHETIC_CLASSES);

export type ClassRow = { cls: string; n: number; shares: number[]; gradient: number[]; ablation: number[]; band: number[] };

/** The classes sampled in `band` (n > 0), in the set's order (unknown classes
 *  after, alphabetically); `shares` = the split `kind`: the receptive-field
 *  gradient, the local (3×3) one (the receptive field's when the class has
 *  none) or the ablation. `input` picks one input band's own gradient split
 *  (the class's by-input row; the all-input split when the class has none);
 *  the ablation split is over all input bands at once and ignores it.
 *  `gradient` stays the receptive-field, all-input split (the agreement
 *  compares it with the ablation). */
export function classRows(
  agg: KneeAggregate | null | undefined, band: string, set: FieldSet, kind: ShareKind, input: KneeInput = "all",
): ClassRow[] {
  const per = agg?.by_class[String(bandIndex(band))];
  if (!per) return [];
  const known = classOrder(set);
  const names = [...known.filter((c) => c in per), ...Object.keys(per).filter((c) => !known.includes(c)).sort()];
  const row = inputIndex(input);
  return names.filter((c) => (per[c]?.n ?? 0) > 0).map((cls) => {
    const v = per[cls];
    const local = kind === "local" ? v.knee_share_local ?? null : null;
    const own = row == null ? undefined : (local ? v.knee_share_by_input_local : v.knee_share_by_input)?.[row];
    return {
      cls, n: v.n, gradient: v.knee_share, ablation: v.ablation_share, band: v.band_share,
      shares: kind === "ablation" ? v.ablation_share : own ?? local ?? v.knee_share,
    };
  });
}

/** The index of the largest share (−1 when empty or all zero/invalid). */
export function dominantKnee(s: readonly number[] | null | undefined): number {
  let best = -1, max = 0;
  (s ?? []).forEach((v, i) => { if (Number.isFinite(v) && v > max) { max = v; best = i; } });
  return best;
}

/** How far the gradient and the ablation split agree: 1 − ½ Σ|g − a| over
 *  the shares renormalised to 1 (1 = the same split, 0 = disjoint; null when
 *  either is all zero). */
export function agreement(g: readonly number[], a: readonly number[]): number | null {
  const sum = (v: readonly number[]) => v.reduce((s, x) => s + (Number.isFinite(x) && x > 0 ? x : 0), 0);
  const sg = sum(g), sa = sum(a);
  if (!(sg > 0) || !(sa > 0)) return null;
  const n = Math.max(g.length, a.length);
  let d = 0;
  for (let i = 0; i < n; i++) d += Math.abs(Math.max(0, g[i] ?? 0) / sg - Math.max(0, a[i] ?? 0) / sa);
  return Math.max(0, 1 - d / 2);
}

/** The 4×4 mean band-share matrix (rows: output band, cols: input band);
 *  missing cells null. `kind` "local": the local (3×3) band shares (the
 *  receptive field's when the aggregate has none). */
export function crossBandMatrix(agg: KneeAggregate | null | undefined, kind: ShareKind = "gradient"): (number | null)[][] {
  const m = (kind === "local" ? agg?.cross_band_local : null) ?? agg?.cross_band;
  return KNEE_BANDS.map((_, r) => KNEE_BANDS.map((__, c) => {
    const v = m?.[r]?.[c];
    return v == null || !Number.isFinite(v) ? null : v;
  }));
}

/* ── field explorer ────────────────────────────────────────────────────── */

export type FieldDot = {
  i: number; x: number; y: number; cls: string; value: number | null;
  shares: number[]; dominant: number; sensitivity: number | null;
};

/** The field's samples in `band` as dots, each with its dominant knee under
 *  the chosen split (`local`: the local 3×3 gradient split, the receptive
 *  field's when the field has none). */
export function fieldDots(field: KneeField | null | undefined, band: string, kind: ShareKind = "gradient"): FieldDot[] {
  if (!field) return [];
  const s = field.samples;
  const b = bandIndex(band);
  const local = kind === "local" && s.knee_share_local?.length ? s.knee_share_local : null;
  const out: FieldDot[] = [];
  s.band.forEach((sb, i) => {
    if (sb !== b) return;
    const x = s.x[i], y = s.y[i];
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    const sh = (kind === "ablation" ? s.ablation_share[i] : (local ?? s.knee_share)[i]) ?? [];
    const v = s.value_e[i];
    const sens = s.sensitivity[i];
    out.push({ i, x, y, cls: s.class[i] ?? "", value: Number.isFinite(v) ? v : null, shares: sh, dominant: dominantKnee(sh),
      sensitivity: Number.isFinite(sens) ? sens : null });
  });
  return out;
}

/** The dot's hover text: class, brightness and the knee shares. */
export function dotTitle(d: FieldDot, knees: readonly number[]): string {
  const parts = d.shares.map((v, k) => `${kneeLabel(knees[k] ?? NaN)} ${sharePct(v)}`).join(" · ");
  return `${d.cls} · ${electronText(d.value)} · (x ${d.x}, y ${d.y})\n${parts}`;
}

/** The backend's path-safe slug of a model spec or a field ref. */
export function slug(text: string): string {
  const s = String(text).replace(/[^A-Za-z0-9._-]+/g, "__").replace(/^_+|_+$/g, "");
  return s || "x";
}

/** The field image (asinh PNG) at `band` and `layer`: the record's `image`
 *  URL when it has one, else built from the slugs. */
export function fieldImageUrl(runId: string, spec: string, field: KneeField, band: string, layer: ImageLayer): string {
  const base = field.image || `/api/knee-attribution/${encodeURIComponent(runId)}/field/${encodeURIComponent(slug(spec))}/${encodeURIComponent(slug(field.ref))}.png`;
  const [path, query = ""] = base.split("?");
  const q = new URLSearchParams(query);
  q.set("band", band);
  q.set("layer", layer);
  return `${path}?${q.toString()}`;
}

/* ── the run form ──────────────────────────────────────────────────────── */

/** The models preselected: the `n` newest (highest member number) available
 *  universal models, else the newest available models of any kind. */
export function defaultModelPick(models: readonly KneeModelChoice[], n = 3): string[] {
  const num = (m: KneeModelChoice) => Number(memberNumber(m.spec.slice(m.spec.indexOf(":") + 1)) ?? -1);
  const newest = (list: KneeModelChoice[]) => [...list].sort((a, b) => num(b) - num(a)).slice(0, n).map((m) => m.spec);
  const avail = models.filter((m) => m.available);
  const universal = avail.filter((m) => m.kind === "universal");
  return newest(universal.length ? universal : avail);
}

export type RunForm = {
  models: string[]; synthetic: string; tiles: string[]; perClass: string; seed: string;
  /** Absent: per class (the default), with the ablation check. */
  sampling?: KneeSampling; perBand?: string; ablation?: boolean;
};
export const DEFAULT_FORM: Omit<RunForm, "models" | "tiles"> = { synthetic: "20", perClass: String(DEFAULT_PER_CLASS), seed: "0" };

const wholeIn = (raw: string, lo: number, hi: number): number | null => {
  const t = raw.trim();
  if (!/^\d+$/.test(t)) return null;
  const v = Number(t);
  return v >= lo && v <= hi ? v : null;
};

export type RunCheck = {
  ok: boolean; synthetic: number | null; perClass: number | null; seed: number | null;
  /** Null under class sampling (no uniform draw). */
  perBand: number | null;
  errors: Partial<Record<"models" | "synthetic" | "perClass" | "perBand" | "seed" | "fields", string>>;
};

/** Validate the form (the backend's ranges): ok when every number is a whole
 *  number in range, a model is picked and some field (synthetic or real). */
export function checkRunForm(f: RunForm): RunCheck {
  const synthetic = wholeIn(f.synthetic, 0, MAX_SYNTHETIC);
  const perClass = wholeIn(f.perClass, 1, MAX_PER_CLASS);
  const seed = wholeIn(f.seed, 0, MAX_SEED);
  const uniform = !!f.sampling && f.sampling !== "class";
  const perBand = uniform ? wholeIn(f.perBand ?? "", 1, MAX_PER_BAND) : null;
  const errors: RunCheck["errors"] = {};
  if (uniform && perBand == null) errors.perBand = `A whole number from 1 to ${MAX_PER_BAND}`;
  if (!f.models.length) errors.models = "Pick at least one model";
  if (synthetic == null) errors.synthetic = `A whole number from 0 to ${MAX_SYNTHETIC}`;
  if (perClass == null) errors.perClass = `A whole number from 1 to ${MAX_PER_CLASS}`;
  if (seed == null) errors.seed = "A whole number ≥ 0";
  if (synthetic === 0 && !f.tiles.length) errors.fields = "Pick synthetic test fields or real tiles";
  return { ok: Object.keys(errors).length === 0, synthetic, perClass, seed, perBand, errors };
}

/** The POSTed form fields (POST /api/knee-attribution): the sampling and
 *  its samples per band only off the default (per class), `ablation=0`
 *  only when the check is off. */
export function runFormData(f: {
  models: readonly string[]; synthetic: number; tiles: readonly string[]; perClass: number; seed: number;
  sampling?: KneeSampling; perBand?: number | null; ablation?: boolean;
}): Record<string, string> {
  const out: Record<string, string> = {
    models: f.models.join(","), synthetic: String(f.synthetic), tiles: f.tiles.join(","),
    per_class: String(f.perClass), seed: String(f.seed),
  };
  if (f.sampling && f.sampling !== "class") {
    out.sampling = f.sampling;
    out.per_band = String(f.perBand ?? DEFAULT_PER_BAND);
  }
  if (f.ablation === false) out.ablation = "0";
  return out;
}

/** "local job: 3 models × 22 fields, ~3 min": per field and model a fixed
 *  part plus the samples (per class: ~28 per class count; uniform: 4 bands ×
 *  per band), each slower with the ablation check. */
export function estimateText(nModels: number, nFields: number, perClass = DEFAULT_PER_CLASS,
  opts: { sampling?: KneeSampling; perBand?: number | null; ablation?: boolean } = {}): string {
  const ablation = opts.ablation !== false;
  const sampling = opts.sampling ?? "class";
  const classPart = sampling === "uniform" ? 0
    : (ablation ? SECONDS_PER_CLASS_SAMPLE : SECONDS_PER_CLASS_SAMPLE * SECONDS_PER_SAMPLE.gradient / SECONDS_PER_SAMPLE.ablation)
      * Math.max(0, perClass);
  const uniformPart = sampling === "class" ? 0
    : 4 * Math.max(0, opts.perBand ?? DEFAULT_PER_BAND) * (ablation ? SECONDS_PER_SAMPLE.ablation : SECONDS_PER_SAMPLE.gradient);
  const perField = SECONDS_PER_FIELD + classPart + uniformPart;
  const s = perField * Math.max(0, nModels) * Math.max(0, nFields);
  const min = s / 60;
  const time = min < 90 ? `~${Math.max(1, Math.round(min))} min` : `~${+(min / 60).toPrecision(2)} h`;
  const plural = (n: number, w: string) => `${n} ${w}${n === 1 ? "" : "s"}`;
  return `local job: ${plural(nModels, "model")} × ${plural(nFields, "field")}, ${time}`;
}

/* ── runs ──────────────────────────────────────────────────────────────── */

/** A run's one-line description for the picker. */
export function runText(r: KneeRunSummary, when: string): string {
  const fields = [r.synthetic ? `${r.synthetic} synthetic` : null, r.tiles.length ? `${r.tiles.length} real` : null].filter(Boolean).join(" + ");
  const models = r.models.length <= 3 ? r.models.map(modelShort).join(", ") : `${r.models.length} models`;
  const status = r.status === "done" ? "" : ` · ${r.status}`;
  return `${when} · ${models}${fields ? ` · ${fields}` : ""}${status}`;
}

/** The record's sets that hold an aggregate for any of `specs`. */
export function setsWithData(rec: KneeRunRecord | null, specs: readonly string[]): FieldSet[] {
  if (!rec) return [];
  return FIELD_SETS.filter((s) => specs.some((m) => rec.models[m]?.aggregate[s]));
}

/** The models drawn: the URL's pick (those in the run, at most
 *  MAX_OVERLAY), else the run's first MAX_OVERLAY models. */
export function overlayModels(rec: KneeRunRecord | null, picked: readonly string[]): string[] {
  if (!rec) return [];
  const inRun = Object.keys(rec.models);
  const mine = picked.filter((m) => inRun.includes(m)).slice(0, MAX_OVERLAY);
  return mine.length ? mine : inRun.slice(0, MAX_OVERLAY);
}

/** Errors as rows: `model` or `model › field` with the reason. */
export function errorRows(errors: Record<string, string>): { where: string; reason: string }[] {
  return Object.entries(errors).map(([k, reason]) => {
    const [model, field] = k.split("|");
    return { where: field ? `${modelShort(model)} › ${field}` : modelShort(model), reason };
  });
}

/* ── URL codecs ────────────────────────────────────────────────────────── */

export const parseBand = (v: string): KneeBand => ((KNEE_BANDS as readonly string[]).includes(v) ? (v as KneeBand) : "VIS");
export const parseSet = (v: string): FieldSet => (v === "real" ? "real" : "synthetic");
export const parseShare = (v: string): ShareKind => (SHARE_KINDS.includes(v as ShareKind) ? (v as ShareKind) : "gradient");
export const parseLayer = (v: string): ImageLayer => (v === "lr" ? "lr" : "sr");
export const parseInput = (v: string): KneeInput => ((KNEE_INPUTS as readonly string[]).includes(v) ? (v as KneeInput) : "all");
export const parseSampling = (v: string): KneeSampling => ((SAMPLINGS as readonly string[]).includes(v) ? (v as KneeSampling) : "class");
