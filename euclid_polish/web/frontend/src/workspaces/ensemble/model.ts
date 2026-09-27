/* Pure logic of the Ensemble workspace (unit-tested in model.test.ts): member
   names, knee descriptions, facets for colouring, the knee-PSNR leaderboard
   over a selectable integration range, member-list parsing and the headline
   formatting; the image-first pass's helpers (member search and the
   disagreement status, gate usage over bands, held-out loss comparability,
   the one-experiment real-data benchmark, the shared e⁻ number format).
   No React, no DOM. */
import { formatDate } from "../../format";
import type { ExperimentSummary, Headline, KneeInfo, KneeModel, KneePayload } from "./api";

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

/** A knee-PSNR model's display name (the Knee tab, its tracking note):
 *  "#196", "plain mean", "production gate", else the combiner's label. */
export function kneeModelName(m: { id: string; kind: string; label: string }): string {
  if (m.kind === "member") return `#${memberNumber(m.label) ?? m.label}`;
  if (m.kind === "mean") return "plain mean";
  return m.id === "spatial_gate" ? "production gate" : m.label;
}

/* ── the Overview comparison ──────────────────────────────────────────── */

export type ComparisonRow = { id: "gate" | "mean" | "best"; label: string; integrated: number | null; bands: (number | null)[] };
export type Comparison = { rows: ComparisonRow[]; source: "knee" | "headline"; bestNumber: string | null };

const finiteOr = (v: number | null | undefined): number | null => (v != null && Number.isFinite(v) ? v : null);
const bandMeanOf = (vs: readonly (number | null | undefined)[] | null | undefined) => {
  const f = (vs ?? []).filter((v): v is number => v != null && Number.isFinite(v));
  return f.length ? f.reduce((a, b) => a + b, 0) / f.length : null;
};
const N_BANDS = 4;

/** The Overview's gate / plain mean / best member rows on ONE metric — the
 *  knee-integrated PSNR over the full grid (`integrated`, one value per band;
 *  ∫PSNR = their mean) — so "best member" is the best by that same number.
 *  From knee-psnr.json when it is readable; else from the overview headline
 *  (band means, plus the gate's per-band values). RBF rows are never read. */
export function overviewComparison(knee: KneePayload | null | undefined, head: Headline["knee"] | null | undefined): Comparison {
  const models = knee?.available ? knee.models ?? [] : [];
  if (models.length) {
    const nb = knee?.bands?.length || N_BANDS;
    const bands = (m: KneeModel | undefined) => Array.from({ length: nb }, (_, i) => finiteOr(m?.integrated?.[i]));
    const gate = models.find((m) => m.kind === "combiner" && m.id === "spatial_gate");
    const mean = models.find((m) => m.kind === "mean");
    let best: KneeModel | undefined;
    let bestValue: number | null = null;
    for (const m of models) {
      if (m.kind !== "member") continue;
      const v = bandMeanOf(m.integrated);
      if (v != null && (bestValue == null || v > bestValue)) { best = m; bestValue = v; }
    }
    const bestNumber = best ? memberNumber(best.label) ?? best.label : null;
    const rows: ComparisonRow[] = [];
    if (gate) rows.push({ id: "gate", label: "Production gate", integrated: bandMeanOf(gate.integrated), bands: bands(gate) });
    if (mean) rows.push({ id: "mean", label: "Plain mean", integrated: bandMeanOf(mean.integrated), bands: bands(mean) });
    if (best) rows.push({ id: "best", label: `Best member (#${bestNumber})`, integrated: bestValue, bands: bands(best) });
    return { rows, source: "knee", bestNumber };
  }
  const empty = () => Array.from({ length: N_BANDS }, () => null);
  const rows: ComparisonRow[] = [];
  if (!head?.available) return { rows, source: "headline", bestNumber: null };
  const bestNumber = head.best_member_label ? memberNumber(head.best_member_label) ?? head.best_member_label : null;
  if (finiteOr(head.production) != null) {
    const pb = head.production_bands ?? [];
    rows.push({ id: "gate", label: "Production gate", integrated: finiteOr(head.production),
      bands: Array.from({ length: N_BANDS }, (_, i) => finiteOr(pb[i])) });
  }
  if (finiteOr(head.mean) != null) rows.push({ id: "mean", label: "Plain mean", integrated: finiteOr(head.mean), bands: empty() });
  if (finiteOr(head.best_member) != null) {
    rows.push({ id: "best", label: bestNumber ? `Best member (#${bestNumber})` : "Best member", integrated: finiteOr(head.best_member), bands: empty() });
  }
  return { rows, source: "headline", bestNumber };
}

/** How far the cross-member σ under-states a field's error: the median over
 *  the test fields of RMSE / mean σ (fields with no σ or no RMSE skipped). */
export function fieldErrorRatio(std: readonly (number | null)[], rmse: readonly (number | null)[]): { ratio: number | null; n: number } {
  const r: number[] = [];
  std.forEach((s, i) => {
    const e = rmse[i];
    if (s != null && e != null && Number.isFinite(s) && Number.isFinite(e) && s > 0 && e > 0) r.push(e / s);
  });
  if (!r.length) return { ratio: null, n: 0 };
  r.sort((a, b) => a - b);
  const mid = r.length >> 1;
  return { ratio: r.length % 2 ? r[mid] : 0.5 * (r[mid - 1] + r[mid]), n: r.length };
}

/** A knee in e⁻ as the axes read it: 0.1, 100, 1k, 10k. */
export const kneeNum = knum;

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

/** The pixel back-trace stamp's backing store: the largest whole number of
 *  device pixels that fits the stamp's laid-out (possibly fractional) CSS
 *  width, and the CSS size that shows exactly those device pixels — so the
 *  pixelated canvas is never rescaled (a 141.5 css px cell at dpr 2 is 283
 *  device px; a 284 backing squeezed into it dropped a row). */
export function stampBacking(cssWidth: number, dpr: number): { side: number; css: number } | null {
  if (!(cssWidth > 0) || !(dpr > 0)) return null;
  const side = Math.max(1, Math.floor(cssWidth * dpr + 1e-6));
  return { side, css: side / dpr };
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

/** A member's peak share of the gate's weight (the largest over the bands and
 *  brightness bins) and where it is. `gate_usage_peak` is a bare number or
 *  `{value, band?, bin?}`; `where` reads "cores", "Y bright"… or null. */
export type GatePeak = { v: number | null; where: string | null };
type PeakField = number | { value?: number | null; band?: string | null; bin?: string | null } | null;

const BIN_TEXT: Record<string, string> = { core: "cores" };

export function gatePeak(row: { gate_usage_peak?: PeakField }): GatePeak {
  const raw = row.gate_usage_peak;
  const obj = typeof raw === "object" ? raw : null;
  const v = typeof raw === "number" ? raw : obj?.value;
  if (v == null || !Number.isFinite(v)) return { v: null, where: null };
  const band = obj?.band ? GATE_BANDS.find((b) => b.band === obj.band)?.short ?? obj.band : null;
  const bin = obj?.bin ? BIN_TEXT[obj.bin] ?? obj.bin : null;
  return { v, where: [band, bin].filter(Boolean).join(" ") || null };
}

const share = (v: number | null, digits = 1) => (v == null || !Number.isFinite(v) ? "—" : `${(100 * v).toFixed(digits)}%`);

/** "0.0% mean · 48% peak (cores)": the all-pixel mean over the bands hides a
 *  member the gate leans on in a few bright pixels (#195: 0.0% mean, 48% of
 *  the core weight). Just the mean when the payload has no peak. */
export function gateUseText(usage: GateUsage, peak: GatePeak): string {
  if (peak.v == null) return share(usage.mean);
  const p = share(peak.v, peak.v >= 0.1 ? 0 : 1);
  return `${share(usage.mean)} mean · ${p} peak${peak.where ? ` (${peak.where})` : ""}`;
}

/** Does the production gate read this member (`used_by_gate`)? null when
 *  the payload does not say. */
export function usedByGate(row: { used_by_gate?: boolean | null }): boolean | null {
  return typeof row.used_by_gate === "boolean" ? row.used_by_gate : null;
}

/** How many members production SR runs: "Runs 20 of 30 members: those with
 *  ≥ 0.5% of the gate's weight somewhere" for a pruned gate (the rule when
 *  the payload records its share threshold), else "Runs all 30 members". */
export function productionRunsText(reads: number, total: number, threshold?: number | null): string {
  const noun = (n: number) => `member${n === 1 ? "" : "s"}`;
  if (reads >= total) return total === 1 ? "Runs its 1 member" : `Runs all ${total} ${noun(total)}`;
  const why = threshold != null && Number.isFinite(threshold)
    ? `those with ≥ ${Number((100 * threshold).toPrecision(3))}% of the gate's weight somewhere`
    : "the ones the gate reads";
  return `Runs ${reads} of ${total} ${noun(total)}: ${why}`;
}

/** The share threshold a pruned gate's members were picked by (a fraction:
 *  0.005 = 0.5%) from a fit record or model details (`prune_threshold`, or
 *  the "used by the gate" rule's `used_threshold`), or null when not recorded. */
export function pruneThreshold(v: { fit?: Record<string, unknown> | null }): number | null {
  return shareThreshold(v.fit);
}

export function shareThreshold(rec: Record<string, unknown> | null | undefined): number | null {
  const t = rec?.prune_threshold ?? rec?.used_threshold;
  return typeof t === "number" && Number.isFinite(t) && t > 0 && t < 1 ? t : null;
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

/** One band's real-data hole % (the share of bright LR pixels the SR blanks). */
export type BandHole = { band: string; short: string; pct: number | null };
export type Bench = {
  holeMean: number | null; holeMax: number | null; medianR: number | null; rLt08: number | null; nTiles: number | null;
  /** Per band (VIS Y J H) when the experiment recorded `per_band`, else empty. */
  bands: BandHole[];
  /** The band with the most holes (null without per-band values). */
  worst: BandHole | null;
};
export type Benchmark = {
  expId: string; label?: string; created?: string; nTiles: number | null; tiles: string[];
  /** What the tiles are, for headers: the experiment label, else "9 real tiles". */
  tileSet: string;
  bySpec: Map<string, Bench>;
};

const byNewest = (exps: readonly ExperimentSummary[] | null | undefined) =>
  [...(exps ?? [])].sort((a, b) => String(b.created ?? "").localeCompare(String(a.created ?? "")));
const hasSummary = (e: ExperimentSummary) => !!e.summary && Object.keys(e.summary).length > 0;
const tilesText = (n: number | null) => (n == null ? "real tiles" : `${n} real tile${n === 1 ? "" : "s"}`);
const expTiles = (e: ExperimentSummary): number | null => {
  let n: number | null = null;
  for (const agg of Object.values(e.summary ?? {})) {
    const t = (agg as { n_tiles?: number }).n_tiles;
    if (typeof t === "number") n = Math.max(n ?? 0, t);
  }
  return n ?? (e.tiles?.length || null);
};

/** The experiments the Combiners benchmark can be scored from (those with a
 *  summary), newest first: "NEXUS core · 1 tile · 2026-09-26". */
export function benchmarkChoices(exps: readonly ExperimentSummary[] | null | undefined): { value: string; label: string; production: boolean }[] {
  return byNewest(exps).filter(hasSummary).map((e) => {
    const n = expTiles(e);
    const tiles = n == null ? null : `${n} tile${n === 1 ? "" : "s"}`;
    const parts = e.label ? [e.label, tiles] : [tilesText(n)];
    const date = e.created ? formatDate(e.created, { fallback: "" }) : "";   // local, like the note under the table
    return { value: e.id, label: [...parts, date].filter(Boolean).join(" · "), production: "production" in (e.summary ?? {}) };
  });
}

/** The real-data benchmark of the Combiners table, from ONE experiment so
 *  every variant is scored on the same tiles: the one the user picked
 *  (`chosen`), else the newest experiment that ran production (else the
 *  newest one). Variants it did not run stay blank. */
export function benchmarkExperiment(exps: readonly ExperimentSummary[] | null | undefined, chosen?: string | null): Benchmark | null {
  const sorted = byNewest(exps);
  const e = (chosen ? sorted.find((x) => x.id === chosen && hasSummary(x)) : undefined)
    ?? sorted.find((x) => x.summary && "production" in x.summary) ?? sorted[0];
  if (!e) return null;
  const bySpec = new Map<string, Bench>();
  for (const [spec, agg] of Object.entries(e.summary ?? {})) {
    const a = agg as { n_tiles?: number; summary?: Record<string, number | null>; per_band?: Record<string, { hole_pct?: number | null }> };
    const bands: BandHole[] = a.per_band
      ? GATE_BANDS.filter(({ band }) => a.per_band?.[band]).map(({ band, short }) => {
        const v = a.per_band?.[band]?.hole_pct;
        return { band, short, pct: v != null && Number.isFinite(v) ? v : null };
      })
      : [];
    const worst = bands.reduce<BandHole | null>((w, b) => (b.pct != null && (w?.pct == null || b.pct > w.pct) ? b : w), null);
    bySpec.set(spec, { holeMean: a.summary?.hole_pct_mean ?? null, holeMax: a.summary?.hole_pct_max ?? null,
      medianR: a.summary?.median_R ?? null, rLt08: a.summary?.pct_R_lt_0p8 ?? null, nTiles: a.n_tiles ?? null, bands, worst });
  }
  const nTiles = expTiles(e);
  return { expId: e.id, label: e.label, created: e.created, nTiles, tiles: e.tiles ?? [], tileSet: e.label || tilesText(nTiles), bySpec };
}

/** A variant's hole % per band, "19 · 13 · 19 · 30" (VIS Y J H); the mean
 *  when the experiment has no per-band values. */
export function holesText(b: Bench): string {
  if (b.bands.length) return b.bands.map((x) => (x.pct == null ? "—" : x.pct.toFixed(0))).join(" · ");
  return b.holeMean != null ? `${b.holeMean.toFixed(1)} % (mean)` : "—";
}

/* ── member counts ─────────────────────────────────────────────────────── */

/** How many members a combiner variant reads: "6 of 20 members" for a
 *  pruned gate (it reads 6 of the 20 it was fitted with), else "26 members". */
export function readsText(v: { n_reads: number; n_members: number; pruned: boolean }): string {
  const noun = (n: number) => `member${n === 1 ? "" : "s"}`;
  return v.pruned && v.n_reads < v.n_members ? `${v.n_reads} of ${v.n_members} ${noun(v.n_members)}` : `${v.n_reads} ${noun(v.n_reads)}`;
}

/** The Disagreement member menu's button: which members the movie shows
 *  ("Members: 196, 195 +2 · Change"), or "Members: none · Pick". */
export function membersButtonText(sel: readonly string[], shown = 2): string {
  if (!sel.length) return "Members: none · Pick";
  const head = sel.slice(0, shown).join(", ");
  return `Members: ${head}${sel.length > shown ? ` +${sel.length - shown}` : ""} · Change`;
}
