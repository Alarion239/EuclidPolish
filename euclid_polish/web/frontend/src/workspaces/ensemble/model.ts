/* Pure logic of the Ensemble workspace (unit-tested in model.test.ts): member
   names, knee descriptions, facets for colouring, the knee-PSNR leaderboard
   over a selectable integration range, member-list parsing and the headline
   formatting; the image-first pass's helpers (member search and the
   disagreement status, gate usage over bands, held-out loss comparability,
   the one-experiment real-data benchmark, the shared e⁻ number format).
   No React, no DOM. */
import type { ExperimentSummary, KneeInfo, KneeModel } from "./api";

/* ── member names ──────────────────────────────────────────────────────── */

const MEMBER_RE = /^(?:member_)?(\d{1,6})(?:·psnr)?$/;

/** "196·psnr" / "member_196" / "196" → "196" (zero-padded to 2 digits like
 *  the member directories), or null. */
export function memberNumber(raw: string | null | undefined): string | null {
  const m = MEMBER_RE.exec(String(raw ?? "").trim());
  if (!m) return null;
  return String(Number(m[1])).padStart(2, "0");
}

export const memberName = (raw: string): string | null => {
  const n = memberNumber(raw);
  return n == null ? null : `member_${n}`;
};

export const memberLabel = (raw: string): string | null => {
  const n = memberNumber(raw);
  return n == null ? null : `${n}·psnr`;
};

/** "170, 171 180-182 member_190" → member names (ranges expanded, de-duplicated,
 *  in order); unparseable tokens are returned in `bad`. */
export function parseMemberList(text: string): { names: string[]; bad: string[] } {
  const names: string[] = [];
  const bad: string[] = [];
  for (const token of text.split(/[\s,;]+/).filter(Boolean)) {
    const range = /^(\d{1,6})-(\d{1,6})$/.exec(token);
    if (range) {
      const [a, b] = [Number(range[1]), Number(range[2])];
      if (b < a || b - a > 500) { bad.push(token); continue; }
      for (let i = a; i <= b; i++) names.push(`member_${String(i).padStart(2, "0")}`);
      continue;
    }
    const name = memberName(token);
    if (name) names.push(name); else bad.push(token);
  }
  return { names: [...new Set(names)], bad };
}

/* ── knees ─────────────────────────────────────────────────────────────── */

export const DEFAULT_KNEE_E = 100;

const knum = (v: number) => (Math.abs(v) >= 1000 ? `${+(v / 1000).toPrecision(3)}k` : `${+v.toPrecision(3)}`);

export type KneeKind = "multi" | "single" | "default";

/** How a member was trained with respect to the asinh knee:
 *  multi-knee members read "multi ×6 → 10" (one output image scored at every
 *  knee, stretched at 10 e⁻) or "multi ×6 heads" (one image per knee) — never
 *  the "100e" the old table showed; single-knee members their knee; members
 *  without a knee the per-band default. */
export function kneeText(m: KneeInfo): { text: string; kind: KneeKind; title: string; sort: number } {
  const knees = Array.isArray(m.asinh_knees) ? m.asinh_knees.filter((v) => Number.isFinite(v)) : [];
  if (knees.length > 1) {
    const span = `${knum(Math.min(...knees))}–${knum(Math.max(...knees))} e⁻`;
    const loss = m.knee_loss && m.knee_loss !== "plain" ? `, ${m.knee_loss} channels` : "";
    if (m.output_knee != null) {
      return { text: `multi ×${knees.length} → ${knum(m.output_knee)}`, kind: "multi", sort: 1e6 + m.output_knee,
        title: `Trained at ${knees.length} knees (${span}) at once${loss}; one output image stretched at ${knum(m.output_knee)} e⁻` };
    }
    return { text: `multi ×${knees.length} heads`, kind: "multi", sort: 2e6,
      title: `Trained at ${knees.length} knees (${span})${loss}; one output image per knee` };
  }
  if (m.asinh_knee != null && Number.isFinite(m.asinh_knee)) {
    return { text: `${knum(m.asinh_knee)} e⁻`, kind: "single", sort: m.asinh_knee, title: `Asinh knee ${m.asinh_knee} e⁻` };
  }
  return { text: `${DEFAULT_KNEE_E} e⁻`, kind: "default", sort: DEFAULT_KNEE_E, title: "The per-band default knee (100 e⁻)" };
}

/* ── colour facets ─────────────────────────────────────────────────────── */

export type ColorBy = "loss" | "depth" | "knee" | "multi" | "uniform";
export const COLOR_BY_OPTIONS: { value: ColorBy; label: string }[] = [
  { value: "loss", label: "Loss" }, { value: "depth", label: "Depth" },
  { value: "knee", label: "Knee" }, { value: "multi", label: "Multi-knee" },
  { value: "uniform", label: "Uniform" },
];

type FacetSource = KneeInfo & { loss?: string | null; loss_norm?: string | null; blocks?: number | null };

/** The facet value a member is coloured by (a stable string key). */
export function facetOf(m: FacetSource, by: ColorBy): string {
  switch (by) {
    case "loss": return (m.loss_norm ?? m.loss ?? "l1").toLowerCase();
    case "depth": return m.blocks != null ? `${m.blocks} blocks` : "unknown depth";
    case "knee": return kneeText(m).text;
    case "multi": {
      const k = kneeText(m);
      return k.kind === "multi" ? (m.output_knee != null ? "multi-knee, 1 image" : "multi-knee, heads") : "single knee";
    }
    default: return "members";
  }
}

/** Distinct facet values in display order (numbers numerically; knees by knee). */
export function facetValues<T extends FacetSource>(rows: readonly T[], by: ColorBy): string[] {
  const seen = new Map<string, number>();
  for (const r of rows) {
    const key = facetOf(r, by);
    const sort = by === "knee" ? kneeText(r).sort : by === "depth" ? (r.blocks ?? 1e9) : 0;
    if (!seen.has(key)) seen.set(key, sort);
  }
  return [...seen.entries()]
    .sort((a, b) => a[1] - b[1] || a[0].localeCompare(b[0], undefined, { numeric: true }))
    .map(([k]) => k);
}

/* ── knee-integrated PSNR ──────────────────────────────────────────────── */

/** Mean of a PSNR-vs-knee curve (K × C) over log10(knee) between `from` and
 *  `to` (trapezoid rule; the ends are interpolated linearly in log knee), per
 *  band. Over the whole grid this is the backend's `integrated_psnr`. */
export function integrateKnee(curve: number[][], knees: number[], from: number, to: number): (number | null)[] {
  const nb = curve[0]?.length ?? 0;
  const lk = knees.map((k) => Math.log10(k));
  const lo = Math.log10(Math.max(Math.min(from, to), knees[0]));
  const hi = Math.log10(Math.min(Math.max(from, to), knees[knees.length - 1]));
  const out: (number | null)[] = [];
  for (let c = 0; c < nb; c++) {
    if (!(hi > lo)) {
      out.push(interp(lk, curve.map((r) => r[c]), lo));
      continue;
    }
    const xs = [lo, ...lk.filter((x) => x > lo && x < hi), hi];
    const ys = xs.map((x) => interp(lk, curve.map((r) => r[c]), x));
    if (ys.some((y) => y == null)) { out.push(null); continue; }
    let area = 0;
    for (let i = 1; i < xs.length; i++) area += 0.5 * ((ys[i] as number) + (ys[i - 1] as number)) * (xs[i] - xs[i - 1]);
    out.push(area / (hi - lo));
  }
  return out;
}

function interp(xs: number[], ys: (number | null | undefined)[], x: number): number | null {
  for (let i = 1; i < xs.length; i++) {
    if (x <= xs[i] + 1e-12) {
      const a = ys[i - 1], b = ys[i];
      if (a == null || b == null || !Number.isFinite(a) || !Number.isFinite(b)) return null;
      const t = xs[i] === xs[i - 1] ? 0 : (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
      return a + Math.max(0, Math.min(1, t)) * (b - a);
    }
  }
  const last = ys[ys.length - 1];
  return last != null && Number.isFinite(last) ? last : null;
}

export type LeaderRow = {
  id: string; label: string; kind: KneeModel["kind"]; model: KneeModel;
  bands: (number | null)[]; mean: number | null; rank: number | null;
  /** Rank over the full knee grid (0.1–1e4 e⁻). */
  fullRank: number | null;
  /** fullRank − rank: > 0 means the model climbs in the selected range. */
  rankDelta: number | null;
  vsMean: number | null;
};

const meanOf = (vs: (number | null)[]) => {
  const f = vs.filter((v): v is number => v != null && Number.isFinite(v));
  return f.length === vs.length && f.length ? f.reduce((a, b) => a + b, 0) / f.length : null;
};

function ranks(values: (number | null)[]): (number | null)[] {
  const order = values.map((v, i) => [v, i] as const).filter(([v]) => v != null)
    .sort((a, b) => (b[0] as number) - (a[0] as number));
  const out: (number | null)[] = values.map(() => null);
  order.forEach(([, i], r) => { out[i] = r + 1; });
  return out;
}

/** The leaderboard: every model's integrated PSNR per band + mean over the
 *  selected range (and bands), its rank, its rank over the full grid and the
 *  change between the two, and its gain over the plain mean. */
export function kneeLeaderboard(models: readonly KneeModel[], knees: number[], range: [number, number],
  bandMask: readonly boolean[] | null = null): LeaderRow[] {
  const full: [number, number] = [knees[0], knees[knees.length - 1]];
  const pick = (vs: (number | null)[]) => (bandMask ? vs.filter((_, i) => bandMask[i]) : vs);
  const sel = models.map((m) => integrateKnee(m.psnr, knees, range[0], range[1]));
  const selMean = sel.map((b) => meanOf(pick(b)));
  const fullMean = models.map((m) => meanOf(pick(integrateKnee(m.psnr, knees, full[0], full[1]))));
  const r = ranks(selMean);
  const fr = ranks(fullMean);
  const meanIndex = models.findIndex((m) => m.kind === "mean");
  const meanRef = meanIndex >= 0 ? selMean[meanIndex] : null;
  return models.map((m, i) => ({
    id: m.id, label: m.label, kind: m.kind, model: m,
    bands: sel[i], mean: selMean[i], rank: r[i], fullRank: fr[i],
    rankDelta: r[i] != null && fr[i] != null ? (fr[i] as number) - (r[i] as number) : null,
    vsMean: selMean[i] != null && meanRef != null ? (selMean[i] as number) - meanRef : null,
  }));
}

/** A curve relative to a reference curve (same grid), per knee and band. */
export function relativeTo(curve: number[][], ref: number[][] | null | undefined): number[][] {
  if (!ref) return curve;
  return curve.map((row, k) => row.map((v, c) => v - (ref[k]?.[c] ?? NaN)));
}

/* ── formatting ────────────────────────────────────────────────────────── */

export function db(v: number | null | undefined, digits = 2): string {
  return v == null || !Number.isFinite(v) ? "—" : v.toFixed(digits);
}

export function dbDelta(v: number | null | undefined, digits = 2): string {
  if (v == null || !Number.isFinite(v)) return "—";
  const s = v.toFixed(digits);
  return v > 0 ? `+${s}` : s.replace("-", "−");
}

export const deltaTone = (v: number | null | undefined): "good" | "bad" | "neutral" =>
  v == null || !Number.isFinite(v) || Math.abs(v) < 5e-3 ? "neutral" : v > 0 ? "good" : "bad";

/** "52k / 70k" style step progress. */
export function stepsText(step: number | null | undefined, target: number | null | undefined): string {
  const k = (v: number) => (v >= 1000 ? `${+(v / 1000).toFixed(1)}k` : String(v));
  if (step == null) return "—";
  return target ? `${k(step)} / ${k(target)}` : k(step);
}

/** Series `[[step, v]…]` → Plot-ready x/y arrays. */
export function xy(series: readonly (readonly [number, number])[] | null | undefined): { x: number[]; y: number[] } {
  const x: number[] = [];
  const y: number[] = [];
  for (const [a, b] of series ?? []) { x.push(a); y.push(b); }
  return { x, y };
}

/** A trailing moving average (window in points; 1 = raw), NaN-safe. */
export function smooth(y: readonly number[], window: number): number[] {
  const w = Math.max(1, Math.floor(window));
  if (w <= 1) return [...y];
  const out: number[] = [];
  let sum = 0, n = 0;
  const q: number[] = [];
  for (const v of y) {
    q.push(v);
    if (Number.isFinite(v)) { sum += v; n++; }
    if (q.length > w) {
      const old = q.shift() as number;
      if (Number.isFinite(old)) { sum -= old; n--; }
    }
    out.push(n ? sum / n : NaN);
  }
  return out;
}

/** Combiner method / variant id → a short display name. */
export function variantLabel(name: string): string {
  if (name === "mean") return "mean";
  if (name === "rbf" || name.startsWith("raw_incremental")) return "RBF";
  const dir = name.replace(/^gate:/, "");
  if (dir === "spatial_gate_combiner") return "production";
  return dir.replace(/^spatial_gate_/, "");
}

/** An e⁻ (or any physical) value at 3 significant figures, in exponent form
 *  at the extremes (≥ 1000 or < 0.01): the back-trace stamps and the
 *  diagnostics cell labels. */
/** The asinh knee of one back-trace row (PixelTrace): the traced pixel's
 *  own level — the largest finite |target|, σ and |err| — so its LR / target
 *  / SR stamps show the structure around a few-e⁻ pixel (a 100 e⁻ knee left
 *  them black). Floored at 0.25 e⁻. */
export function stampKnee(s: { hr_val: number; std_val: number; err_val: number }): number {
  const levels = [s.hr_val, s.std_val, s.err_val].map((v) => Math.abs(v)).filter((v) => Number.isFinite(v));
  return Math.max(0.25, ...levels);
}

export function formatE(v: number): string {
  if (!Number.isFinite(v)) return "—";
  return Math.abs(v) >= 1000 || (Math.abs(v) > 0 && Math.abs(v) < 0.01)
    ? v.toExponential(1) : String(Number(v.toPrecision(3)));
}

/* ── gate usage (members table) ────────────────────────────────────────── */

/** The Euclid bands in canonical order and their short names (as the chips
 *  read). Kept here so this module stays free of the hooks in ./api. */
export const GATE_BANDS: readonly { band: string; short: string }[] = [
  { band: "VIS", short: "VIS" }, { band: "Y_E", short: "Y" }, { band: "J_E", short: "J" }, { band: "H_E", short: "H" },
];

export type GateUsage = { mean: number | null; max: number | null; bands: { band: string; short: string; v: number | null }[] };

/** A member's share of the production gate's weight, per band and summed up:
 *  the mean over the bands that have a value (the table's "Gate use") and the
 *  largest band. The VIS weight alone hid members the gate uses for NISP
 *  (#190: 0.02 % of VIS but ~35 % of Y/J/H). */
export function gateUsage(usage: Record<string, number | null | undefined> | null | undefined): GateUsage {
  const bands = GATE_BANDS.map(({ band, short }) => {
    const v = usage?.[band];
    return { band, short, v: v != null && Number.isFinite(v) ? v : null };
  });
  const vs = bands.map((b) => b.v).filter((v): v is number => v != null);
  return {
    mean: vs.length ? vs.reduce((a, b) => a + b, 0) / vs.length : null,
    max: vs.length ? Math.max(...vs) : null,
    bands,
  };
}

/* ── disagreement member picker ────────────────────────────────────────── */

/** Does a member match the picker's search text? Every word must appear in
 *  its number (with or without "#"), loss, knee description or label. */
export function memberMatches(m: { num: string; loss?: string | null; knee?: string | null; label?: string | null }, query: string): boolean {
  const words = query.toLowerCase().split(/\s+/).map((w) => w.replace(/^#/, "")).filter(Boolean);
  if (!words.length) return true;
  const hay = `${m.num} ${m.loss ?? ""} ${m.knee ?? ""} ${m.label ?? ""}`.toLowerCase();
  return words.every((w) => hay.includes(w));
}

/** The line that says what the disagreement viewer shows for a selection. */
export function movieStatus(sel: readonly string[], shown = 5): string {
  if (!sel.length) return "Pick members: one shows its SR, two or more play the disagreement movie";
  if (sel.length === 1) return `Showing member #${sel[0]}`;
  const head = sel.slice(0, shown).map((n) => `#${n}`).join(", ");
  const rest = sel.length > shown ? ` and ${sel.length - shown} more` : "";
  return `Movie over ${sel.length} members: ${head}${rest}`;
}

/* ── combiners ─────────────────────────────────────────────────────────── */

const lossDef = (v: { fit?: Record<string, unknown> } | null | undefined) => {
  const l = v?.fit?.loss;
  return typeof l === "string" && l.trim() ? l.trim() : null;
};

/** Is a variant's held-out loss on production's scale? Only when both name
 *  the same loss definition (v1's band-weighted error read ~4× "better"
 *  than production's relative MSE). Without a production definition there
 *  is nothing to compare against, so nothing is flagged. */
export function heldOutComparable(v: { fit?: Record<string, unknown> }, production: { fit?: Record<string, unknown> } | null | undefined): boolean {
  const ref = lossDef(production);
  if (!ref) return true;
  return lossDef(v) === ref;
}

export type Bench = { holeMean: number | null; holeMax: number | null; medianR: number | null; rLt08: number | null; nTiles: number | null };
export type Benchmark = { expId: string; label?: string; created?: string; nTiles: number | null; bySpec: Map<string, Bench> };

/** The real-data benchmark of the Combiners table, from ONE experiment so
 *  every variant is scored on the same tiles: the newest experiment that ran
 *  production (else the newest one). Variants it did not run stay blank. */
export function benchmarkExperiment(exps: readonly ExperimentSummary[] | null | undefined): Benchmark | null {
  const sorted = [...(exps ?? [])].sort((a, b) => String(b.created ?? "").localeCompare(String(a.created ?? "")));
  const e = sorted.find((x) => x.summary && "production" in x.summary) ?? sorted[0];
  if (!e) return null;
  const bySpec = new Map<string, Bench>();
  let nTiles: number | null = null;
  for (const [spec, agg] of Object.entries(e.summary ?? {})) {
    const a = agg as { n_tiles?: number; summary?: Record<string, number | null> };
    const n = a.n_tiles ?? null;
    if (n != null) nTiles = Math.max(nTiles ?? 0, n);
    bySpec.set(spec, { holeMean: a.summary?.hole_pct_mean ?? null, holeMax: a.summary?.hole_pct_max ?? null,
      medianR: a.summary?.median_R ?? null, rLt08: a.summary?.pct_R_lt_0p8 ?? null, nTiles: n });
  }
  return { expId: e.id, label: e.label, created: e.created, nTiles, bySpec };
}
