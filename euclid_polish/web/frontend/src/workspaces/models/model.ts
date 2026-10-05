/* Pure logic of the Models workspace (unit-tested in model.test.ts and
   loop.test.ts): member names, knee descriptions, facets for colouring, the
   knee-PSNR leaderboard over a selectable integration range, member-list
   parsing and the headline formatting; member search, gate usage over bands,
   held-out loss comparability, the one-experiment real-data benchmark, the
   shared e⁻ number format; and the regrouping's loop helpers (the
   Leaderboard status line and real benchmark, members waiting on FASRC, the
   gate share, the combiner variant scope, the synthetic stamps, Δm vs LR, the
   running batch, the Train submit label). No React, no DOM. */
import type { SlurmJob } from "../../api/jobs";
import { formatCount, formatDate } from "../../format";
import type {
  Check, CurveFacetRow, EvalRow, ExperimentSummary, Headline, KneeInfo, KneeModel, KneePayload, MemberRow, SrSplit, SrStatus, TrainingJob, Variant,
} from "./api";

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

/* ── the Leaderboard comparison ───────────────────────────────────────── */

export type ComparisonRow = { id: "gate" | "mean" | "best"; label: string; integrated: number | null; bands: (number | null)[] };
export type Comparison = { rows: ComparisonRow[]; source: "knee" | "headline"; bestNumber: string | null };

const finiteOr = (v: number | null | undefined): number | null => (v != null && Number.isFinite(v) ? v : null);
const bandMeanOf = (vs: readonly (number | null | undefined)[] | null | undefined) => {
  const f = (vs ?? []).filter((v): v is number => v != null && Number.isFinite(v));
  return f.length ? f.reduce((a, b) => a + b, 0) / f.length : null;
};
const N_BANDS = 4;

/** The Leaderboard's gate / plain mean / best member rows on ONE metric — the
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

/** A `combiner:` inspector id → its variant dir. Older ids led with the star
 *  regime (`starfull/spatial_gate_linear`); a variant dir has no slash, so it
 *  is the last segment either way. */
export function combinerVariant(id: string): string {
  return id.slice(id.lastIndexOf("/") + 1);
}

/** Combiner method / variant id → a short display name. */
export function variantLabel(name: string): string {
  if (name === "mean") return "mean";
  if (name === "rbf" || name.startsWith("raw_incremental")) return "RBF";
  const dir = name.replace(/^gate:/, "");
  if (dir === "spatial_gate_combiner") return "production";
  return dir.replace(/^spatial_gate_/, "");
}

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

/** An e⁻ (or any physical) value at 3 significant figures, in exponent form
 *  at the extremes (≥ 1000 or < 0.01): the back-trace stamps and the
 *  diagnostics cell labels. */
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
 *  the mean over the bands that have a value (the "mean" of the Gate share
 *  readings, and what the bar draws when there is no peak) and the largest
 *  band. The VIS weight alone hid members the gate uses for NISP
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

/** A gate share as a percentage, ONE rule everywhere: two significant
 *  figures ("48%", "4.6%", "0.48%"), "<0.1%" below a tenth of a percent. */
export const share = (v: number | null | undefined) => {
  if (v == null || !Number.isFinite(v)) return "—";
  const p = 100 * v;
  if (p === 0) return "0%";
  if (Math.abs(p) < 0.1) return "<0.1%";
  return `${Number(p.toPrecision(2))}%`;
};

/** "0.0% mean · 48% peak (cores)": the all-pixel mean over the bands hides a
 *  member the gate leans on in a few bright pixels (#195: 0.0% mean, 48% of
 *  the core weight). Just the mean when the payload has no peak. */
export function gateUseText(usage: GateUsage, peak: GatePeak): string {
  if (peak.v == null) return share(usage.mean);
  // A member with no weight anywhere: one "0%", no band tag (it means nothing at zero).
  if (peak.v === 0 && !usage.mean) return share(0);
  const p = share(peak.v);
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

/* ── the regrouping's loop helpers (console phases 2–6, Team M) ─────────── */

/** One failing check of the Leaderboard status line; `fix` is the action to
 *  offer on it (null: already offered on an earlier line). */
export type StatusCheck = Check & { fix: string | null };

const numberList = (names: readonly string[], max = 8) => {
  const nums = names.map((n) => memberNumber(n) ?? n);
  return nums.length > max ? `${nums.slice(0, max).join(", ")} and ${nums.length - max} more` : nums.join(", ");
};

/** The failing staleness checks, each fix offered once (on the first line
 *  that needs it), plus "Member PSNR" when active members have no test PSNR
 *  yet. Empty means "All current". */
export function statusChecks(checks: readonly Check[], members: readonly MemberRow[] | null | undefined): StatusCheck[] {
  const offered = new Set<string>();
  const out: StatusCheck[] = [];
  const add = (c: Check) => {
    const fix = c.action && !offered.has(c.action) ? c.action : null;
    if (fix) offered.add(fix);
    out.push({ ...c, fix });
  };
  checks.filter((c) => !c.ok).forEach(add);
  const unscored = (members ?? []).filter((m) => m.psnr == null).map((m) => m.name);
  if (unscored.length) {
    add({ id: "member-psnr", ok: false, tone: "warn", title: "Member PSNR", action: "member-psnr",
      detail: `${unscored.length} member${unscored.length === 1 ? " has" : "s have"} no test PSNR yet: ${numberList(unscored)}.` });
  }
  return out;
}

/** A Sky › Compare run's page. */
export const compareRunPath = (expId: string) => `/sky/compare?${new URLSearchParams({ exp: expId }).toString()}`;

export type RealFacts = { bands: BandHole[]; worst: BandHole | null; medianR: number | null };
type ExpDetail = { id: string; fingerprints?: Record<string, string | null> | null };
type Catalog = { models: readonly { spec: string; fingerprint?: string | null }[] };
export type LeaderBenchmark =
  | { state: "loading" }
  | { state: "none"; reason: string; last?: { expId: string; created?: string; label?: string } }
  | { state: "current"; expId: string; created?: string; label?: string; tileSet: string;
      /** The real facts of one spec ("production", "mean", "member:member_196"),
       *  or null when the run did not score it for the current model. */
      facts: (spec: string) => RealFacts | null };

/** The newest Sky › Compare run that scored production (the Leaderboard's
 *  real benchmark candidate), or null. */
export function productionRun(exps: readonly ExperimentSummary[] | null | undefined): ExperimentSummary | null {
  return byNewest(exps).find((e) => hasSummary(e) && "production" in (e.summary ?? {})) ?? null;
}

/** The Leaderboard's real columns: the newest Sky › Compare run that scored
 *  production, when it scored THIS production (its spec fingerprint equals
 *  the catalogue's); each other row only when its own fingerprint matches.
 *  `detail` is that run's record (GET /api/experiments/<id>), `catalog` the
 *  current model catalogue (GET /api/models). */
export function leaderboardBenchmark(exps: readonly ExperimentSummary[] | null | undefined, detail: ExpDetail | null | undefined,
  catalog: Catalog | null | undefined): LeaderBenchmark {
  const run = productionRun(exps);
  if (!run) return { state: "none", reason: "no real benchmark yet: no Sky › Compare run has scored production" };
  if (!detail || detail.id !== run.id || !catalog) return { state: "loading" };
  const current = new Map(catalog.models.map((m) => [m.spec, m.fingerprint ?? null]));
  const fps = detail.fingerprints ?? {};
  const fresh = (spec: string) => !!fps[spec] && fps[spec] === current.get(spec);
  const last = { expId: run.id, created: run.created, label: run.label };
  if (!fresh("production")) {
    return { state: "none", last, reason: "no real benchmark for this membership: the last Sky › Compare run of production scored an earlier one" };
  }
  const bench = benchmarkExperiment([run], run.id);
  return {
    state: "current", expId: run.id, created: run.created, label: run.label, tileSet: bench?.tileSet ?? "real tiles",
    facts: (spec) => {
      const b = fresh(spec) ? bench?.bySpec.get(spec) : undefined;
      return b ? { bands: b.bands, worst: b.worst, medianR: b.medianR } : null;
    },
  };
}

/** Whether a Sky › Compare run's score of `spec` is of the CURRENT model:
 *  "current" when the run's recorded fingerprint equals the catalogue's,
 *  "earlier" when it scored another (or unrecorded) fit of that spec,
 *  "loading" while the run record or the catalogue is not in yet. */
export type BenchFreshness = "current" | "earlier" | "loading";
export function benchFreshness(expId: string | null | undefined, detail: ExpDetail | null | undefined,
  catalog: Catalog | null | undefined): (spec: string) => BenchFreshness {
  if (!expId || !detail || detail.id !== expId || !catalog) return () => "loading";
  const current = new Map(catalog.models.map((m) => [m.spec, m.fingerprint ?? null]));
  const fps = detail.fingerprints ?? {};
  return (spec) => (fps[spec] && fps[spec] === current.get(spec) ? "current" : "earlier");
}

const TERMINAL_WITH_CHECKPOINT = new Set(["COMPLETED", "TIMEOUT"]);

/** How long a finished batch counts as new for the Pull banner. */
export const WAITING_DAYS = 14;

/** What finished on FASRC and is not local yet (the Members "Pull" banner):
 *  new members of add/fork batches that ended in the last `days` (default
 *  14) and are neither active here nor archived — never those of a legacy
 *  starless batch (isStarlessJob), which the roster does not list; and
 *  continued members whose recent finished job's target lies past their
 *  local step. A job without an end time counts as recent. */
export function waitingOnFasrc(jobs: readonly TrainingJob[], members: readonly MemberRow[], archived: readonly string[],
  opts: { now?: number; days?: number } = {}): { members: string[]; continued: string[] } {
  const local = new Map(members.map((m) => [m.name, m]));
  const gone = new Set(archived);
  const cutoff = (opts.now ?? Date.now()) - (opts.days ?? WAITING_DAYS) * 86_400_000;
  const recent = (j: TrainingJob) => {
    const t = Date.parse(String(j.ended_at ?? j.submitted_at ?? ""));
    return !Number.isFinite(t) || t >= cutoff;
  };
  const fresh = new Set<string>();
  const continued = new Set<string>();
  for (const j of jobs) {
    if (!TERMINAL_WITH_CHECKPOINT.has(String(j.state ?? "").toUpperCase()) || !recent(j)) continue;
    if (j.mode === "continue") {
      for (const n of j.member_names) {
        const m = local.get(n);
        if (m && j.target_steps != null && (m.step ?? 0) < j.target_steps) continued.add(n);
      }
      continue;
    }
    if (isStarlessJob(j)) continue;
    for (const n of j.member_names) if (!local.has(n) && !gone.has(n)) fresh.add(n);
  }
  const byNum = (a: string, b: string) => Number(memberNumber(a)) - Number(memberNumber(b));
  return { members: [...fresh].sort(byNum), continued: [...continued].sort(byNum) };
}

const TRUTHY = ["1", "true", "yes", "on"];
/** Whether a past training job trained starless members (the run-wide flag
 *  or a per-member one): legacy batches only, the Train tab never sends it. */
export function isStarlessJob(job: Pick<TrainingJob, "params">): boolean {
  const p = job.params ?? {};
  if (TRUTHY.includes(String(p.starless ?? "").toLowerCase())) return true;
  let spec: unknown;
  try { spec = typeof p.member_spec === "string" ? JSON.parse(p.member_spec) : p.member_spec; } catch { spec = null; }
  return Array.isArray(spec) && spec.some((o) => !!o && typeof o === "object" && (o as Record<string, unknown>).starless === true);
}

/** A member's share of the production gate's weight as the Members and
 *  Combiner bars draw it: the peak (the largest share in any band and
 *  brightness bin — what decides whether the gate reads it) when the payload
 *  has one, else the mean over the bands. */
export function gateShare(row: Pick<MemberRow, "gate_usage" | "gate_usage_peak">): {
  value: number | null; mean: number | null; peak: number | null;
  /** The full reading, "0.0% mean · 27% peak (J cores)" (tooltips, notes). */
  text: string;
  /** The one value the bar draws, for a narrow cell: "27% J cores" (the peak
   *  and where), else "0.4% mean". */
  short: string;
} {
  const usage = gateUsage(row.gate_usage);
  const peak = gatePeak(row);
  const short = peak.v != null
    ? `${share(peak.v)}${peak.where && peak.v !== 0 ? ` ${peak.where}` : ""}`
    : usage.mean != null ? `${share(usage.mean)} mean` : "—";
  return { value: peak.v ?? usage.mean, mean: usage.mean, peak: peak.v, text: gateUseText(usage, peak), short };
}

/** The Combiner table's rows: production and the variants fitted for the
 *  current membership (every member they read is active and none joined
 *  after the fit); the rest (earlier memberships, promotion backups) behind
 *  the history chip. The legacy RBF is never listed. */
export function variantScope<V extends Pick<Variant, "kind" | "production" | "backup" | "membership">>(variants: readonly V[], history: boolean): { shown: V[]; hidden: number } {
  const gates = variants.filter((v) => v.kind !== "rbf");
  const inScope = (v: V) => v.production || (!v.backup && v.membership.current && !v.membership.extra.length);
  return { shown: history ? gates : gates.filter(inScope), hidden: gates.filter((v) => !inScope(v)).length };
}

/** One held-out curve of the Combiner: its style slot (-1 = production, the
 *  combiner colour; else one of `slots` categorical colours, dashed once they
 *  run out, so every curve is unique) and its points. */
export type HeldOutCurve = { name: string; production: boolean; slot: number; dash: boolean; x: number[]; y: number[] };
export type HeldOutMetric = "loss" | "vis" | "int";
const CAT_SLOTS = 8;

/** The held-out checkpoint curves of the fits, production first. The loss
 *  is drawn only for fits on production's loss definition (another loss is
 *  on another scale: `offScale` names those); PSNR is one scale for all. */
export function heldOutCurves<V extends Pick<Variant, "name" | "production" | "history" | "fit">>(variants: readonly V[], metric: HeldOutMetric,
  slots = CAT_SLOTS): { curves: HeldOutCurve[]; offScale: string[] } {
  const production = variants.find((v) => v.production) ?? null;
  const withHistory = [...variants].filter((v) => v.history.length)
    .sort((a, b) => Number(b.production) - Number(a.production));
  const offScale: string[] = [];
  const curves: HeldOutCurve[] = [];
  let slot = 0;
  for (const v of withHistory) {
    if (metric === "loss" && !v.production && !heldOutComparable(v, production)) { offScale.push(v.name); continue; }
    const pts = v.history.map((h) => [h.step, metric === "loss" ? h.loss : metric === "vis" ? h.vis_psnr : finiteMean(h.integrated_psnr)] as const)
      .filter((p): p is readonly [number, number] => p[1] != null && Number.isFinite(p[1]));
    if (!pts.length) continue;
    const s = v.production ? -1 : slot++;
    curves.push({ name: v.name, production: v.production, slot: s < 0 ? -1 : s % slots, dash: s >= slots,
      x: pts.map((p) => p[0]), y: pts.map((p) => p[1]) });
  }
  return { curves, offScale };
}

/** The mean of the finite values (null when there is none). */
function finiteMean(vs: readonly (number | null | undefined)[] | null | undefined): number | null {
  const f = (vs ?? []).filter((v): v is number => v != null && Number.isFinite(v));
  return f.length ? f.reduce((a, b) => a + b, 0) / f.length : null;
}

/** One member's share of a gate's weight (the Combiner's gate-share list). */
export type ShareRow = { name: string; num: string; share: number | null; text: string; read: boolean | null };
const byShare = (a: ShareRow, b: ShareRow) => (b.share ?? -1) - (a.share ?? -1)
  || Number(memberNumber(a.name)) - Number(memberNumber(b.name));

/** The production gate's share per member (members.json: the peak share when
 *  recorded, else the mean over bands), largest first, unknown last. */
export function productionShares(members: readonly MemberRow[]): ShareRow[] {
  return members.map((m): ShareRow => {
    const s = gateShare(m);
    return { name: m.name, num: memberNumber(m.name) ?? m.name, share: s.value, text: s.value == null ? "—" : s.text, read: usedByGate(m) };
  }).sort(byShare);
}

/** A variant's share per member from a compare report's `usage` block (the
 *  all-pixel weight per member and band, averaged over the bands), largest
 *  first; null when the report has none for it. */
export function reportShares(usage: { labels: string[]; all_pixels: number[][]; source_pixels?: number[][] } | null | undefined): ShareRow[] | null {
  const labels = usage?.labels ?? [];
  const m = usage?.all_pixels ?? [];
  if (!labels.length || !m.length) return null;
  const perMember = m.length === labels.length ? m
    : m[0]?.length === labels.length ? labels.map((_, i) => m.map((band) => band[i])) : null;
  if (!perMember) return null;
  return labels.map((label, i): ShareRow => {
    const v = finiteMean(perMember[i]);
    const num = memberNumber(label) ?? label;
    return { name: `member_${num}`, num, share: v, text: share(v), read: null };
  }).sort(byShare);
}

/* ── synthetic stamps (Images ?set=stamps) and SR → HR recovery ────────── */

export const STAMP_GROUPS: readonly { grade: string; label: string }[] = [
  { grade: "syn-lens", label: "Syn lens" }, { grade: "syn-gal", label: "Syn gal" },
];

const fnum = (v: unknown): number | null => {
  const n = typeof v === "number" ? v : typeof v === "string" && v.trim() !== "" ? Number(v) : NaN;
  return Number.isFinite(n) ? n : null;
};
const median = (vs: number[]): number | null => {
  if (!vs.length) return null;
  const s = [...vs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]);
};
const isOk = (r: { ok?: unknown }) => String(r.ok).toLowerCase() === "true";

export type StampPoint = { id: string; x: number; y: number; flux: number | null };
export type StampSet = {
  grade: string; label: string; n: number; points: StampPoint[];
  medianLr: number | null; medianSr: number | null; gain: number | null; improved: number;
  stale: number; madeBy: number | null;
};

/** The synthetic stamps with HR truth, per group: PSNR vs HR of LR (x) and SR
 *  (y) per object, the medians, the median paired gain, how many SR brought
 *  closer to the truth, how many predate the current model and how many
 *  members made them (the most common count). */
export function stampSets(rows: readonly EvalRow[]): StampSet[] {
  return STAMP_GROUPS.map(({ grade, label }) => {
    const mine = rows.filter((r) => r.grade === grade && isOk(r));
    const points = mine.map((r): StampPoint | null => {
      const x = fnum(r.psnr_lr_hr), y = fnum(r.psnr_sr_hr);
      return x == null || y == null ? null : { id: String(r.viewer_id || r.out_subdir || r.id), x, y, flux: fnum(r.flux_ratio_sr_over_lr) };
    }).filter((p): p is StampPoint => !!p);
    const counts = new Map<number, number>();
    for (const r of mine) { const k = fnum(r.n_members); if (k != null) counts.set(k, (counts.get(k) ?? 0) + 1); }
    const madeBy = [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? null;
    return {
      grade, label, n: mine.length, points,
      medianLr: median(points.map((p) => p.x)), medianSr: median(points.map((p) => p.y)),
      gain: median(points.map((p) => p.y - p.x)), improved: points.filter((p) => p.y > p.x).length,
      stale: mine.filter((r) => r.state && r.state !== "current").length, madeBy,
    };
  }).filter((s) => s.n > 0);
}

/** SR's total-flux change against LR as a magnitude: "Δm +0.11 (flux ×0.90)",
 *  `warn` beyond 0.1 mag. Null without both fluxes. */
export function deltaMagText(lrE: number | null | undefined, srE: number | null | undefined): { text: string; warn: boolean } | null {
  if (lrE == null || srE == null || !(lrE > 0) || !(srE > 0)) return null;
  const ratio = srE / lrE;
  const dm = -2.5 * Math.log10(ratio);
  const s = dm.toFixed(2);
  const signed = dm > 0 ? `+${s}` : dm < 0 ? s.replace("-", "−") : s;
  return { text: `Δm ${signed} (flux ×${ratio.toFixed(2)})`, warn: Math.abs(dm) > 0.1 };
}

/** The per-set caption of Images › Stamps: "30 stamps · PSNR vs HR: LR 31.20
 *  → SR 33.40 dB, median gain +2.20 dB · SR is closer to the truth on 28 of 30". */
export function stampCaption(s: Pick<StampSet, "n" | "points" | "medianLr" | "medianSr" | "gain" | "improved">): string {
  // The stamp count is on the group chip: the caption does not repeat it.
  if (s.medianLr == null || s.medianSr == null || !s.points.length) return "No PSNR vs HR recorded";
  const m = s.points.length;
  const closer = s.improved === m ? (m === 1 ? "the stamp" : "every stamp")
    : s.improved === 0 ? "no stamp" : `${formatCount(s.improved)} of ${formatCount(m)}`;
  return `Median PSNR vs HR: LR ${db(s.medianLr)} → SR ${db(s.medianSr)} dB, a gain of ${dbDelta(s.gain)} dB`
    + ` · SR is closer to the truth on ${closer}`;
}

/** The band a cube plane is summed in: the viewer's band, or VIS (plane 0)
 *  for a colour composite. */
export function bandIndex(bands: readonly string[] | null | undefined, color: string): number {
  const i = (bands ?? []).indexOf(color);
  return i >= 0 ? i : 0;
}

/** The finite sum of one band plane of an (H, W, C) Float32 cube. */
export function planeSum(data: Float32Array, c: number, band: number): number {
  let sum = 0;
  const step = Math.max(1, c);
  for (let i = Math.min(band, step - 1); i < data.length; i += step) {
    const v = data[i];
    if (Number.isFinite(v)) sum += v;
  }
  return sum;
}

type SummableCube = { data: Float32Array; c: number; bands?: string[] };
/** SR's total-flux change against LR in the viewer's band, from the two
 *  cubes the viewer shows: { band: "VIS", text: "Δm +0.11 (flux ×0.90)", warn }. */
export function cubeDeltaMag(lr: SummableCube, sr: SummableCube, color: string): { band: string; text: string; warn: boolean } | null {
  const bl = bandIndex(lr.bands, color), bs = bandIndex(sr.bands, color);
  const d = deltaMagText(planeSum(lr.data, lr.c, bl), planeSum(sr.data, sr.c, bs));
  if (!d) return null;
  const name = sr.bands?.[bs] ?? lr.bands?.[bl] ?? "VIS";
  return { band: BAND_SHORT_NAMES[name] ?? name, ...d };
}
const BAND_SHORT_NAMES: Record<string, string> = { VIS: "VIS", Y_E: "Y", J_E: "J", H_E: "H" };

/* ── diagnostics ───────────────────────────────────────────────────────── */

type CoherenceScore = { id: string; label: string; overall: number | null; sr: number | null };
/** The coherence dot plot's rows, highest overall score first (missing last). */
export function coherenceOrder<T extends CoherenceScore>(rows: readonly T[]): T[] {
  return [...rows].sort((a, b) => (b.overall ?? -Infinity) - (a.overall ?? -Infinity) || (b.sr ?? -Infinity) - (a.sr ?? -Infinity));
}

/** The coherence dot plot's row names, as the other tabs name the models. */
export function coherenceLabel(r: { id: string; label: string }): string {
  if (r.id === "ensemble_mean") return "plain mean";
  if (r.id === "spatial_gate_combiner" || r.id === "spatial_gate") return "production gate";
  if (r.id === "lr_baseline") return "LR (bicubic)";
  if (r.id === "model_agreement") return "member agreement";
  if (r.id.startsWith("member_")) return `#${memberNumber(r.label) ?? r.label}`;
  return r.label || r.id;
}

/** SR scales: the r(k) / T(k) axis is focused on θ below this [arcsec]. */
export const SR_SCALE_MAX_ARCSEC = 0.5;

/** The spectrum's x domain [arcsec]: from the smallest measured scale to
 *  0.5″ (the SR scales), or to the largest measured scale. */
export function spectrumDomain(theta: readonly (number | null)[], thetaMin: number | null | undefined, focus: boolean): [number, number] {
  const finite = theta.filter((v): v is number => v != null && Number.isFinite(v) && v > 0);
  const lo = thetaMin ?? 0.05;
  const hi = finite.length ? Math.max(...finite) : 1;
  return [lo, focus ? Math.min(SR_SCALE_MAX_ARCSEC, hi) : hi];
}

/** A band name as the page shows it (Y_E → Y). */
export const bandLabel = (band: string) => band.replace(/_E$/, "");

/** The LR-pixel and band-FWHM guides in the domain; the smaller scale's label
 *  is drawn left of its line and the larger's right, so they never overprint. */
export function spectrumGuides(g: { lr_scale?: number | null; vis_fwhm?: number | null; psf_fwhm?: number | null; band?: string | null },
  domain: [number, number]): { kind: "lr" | "fwhm"; v: number; label: string; side: "before" | "after" }[] {
  const all = [
    { kind: "lr" as const, v: g.lr_scale ?? 0.1, label: "LR pixel" },
    { kind: "fwhm" as const, v: g.psf_fwhm ?? g.vis_fwhm ?? 0.16, label: `${bandLabel(g.band ?? "VIS")} FWHM` },
  ].filter((x) => x.v >= domain[0] && x.v <= domain[1]).sort((a, b) => a.v - b.v);
  return all.map((x, i) => ({ ...x, side: all.length > 1 && i === 0 ? "before" : "after" }));
}

/** The spread section's answer, from the median per-field RMSE / σ. */
export function spreadVerdict(ratio: number | null | undefined): { text: string; warn: boolean } | null {
  if (ratio == null || !Number.isFinite(ratio)) return null;
  if (ratio > 1.25) return { text: "Cross-member σ is not an error bar", warn: true };
  if (ratio < 0.8) return { text: "Cross-member σ over-states the error", warn: true };
  return { text: "Cross-member σ tracks the error", warn: false };
}

export const GAUSS_COVER = [0.683, 0.954, 0.997] as const;
/** |z| < 1, 2, 3: observed coverage against the Gaussian's. */
export function coverageRows(stats: { cover1?: number | null; cover2?: number | null; cover3?: number | null }):
  { k: number; observed: number | null; gaussian: number }[] {
  return [stats.cover1, stats.cover2, stats.cover3].map((observed, i) => ({ k: i + 1, observed: observed ?? null, gaussian: GAUSS_COVER[i] }));
}

/** The z-pdf's y domain: up to its tallest curve (+10%), never below the
 *  unit Gaussian's peak. */
export function pdfYDomain(...curves: readonly (readonly (number | null)[])[]): [number, number] {
  let hi = 0;
  for (const c of curves) for (const v of c) if (v != null && Number.isFinite(v)) hi = Math.max(hi, v);
  return [0, Math.max(0.45, hi * 1.1)];
}

const LOSS_ORDER = ["l1", "l2", "l3", "mse", "berhu"];
/** The training curves split by loss type (L1 and L2 losses are on different
 *  scales, so each gets its own plot), in loss order. */
export function lossFacets<T extends CurveFacetRow>(curves: readonly T[]): { loss: string; curves: T[] }[] {
  const by = new Map<string, T[]>();
  for (const c of curves) {
    const l = (c.loss_norm || "l1").toLowerCase();
    by.set(l, [...(by.get(l) ?? []), c]);
  }
  const rank = (l: string) => (LOSS_ORDER.indexOf(l) + 1 || 99);
  return [...by.entries()].sort((a, b) => rank(a[0]) - rank(b[0]) || a[0].localeCompare(b[0])).map(([loss, cs]) => ({ loss, curves: cs }));
}

/* ── train ─────────────────────────────────────────────────────────────── */

/** "Submit 4 members to SLURM" — the count repeated on the button, so a
 *  wrong batch size is caught before the confirm. */
export function submitLabel(mode: "add" | "continue" | "fork", count: number): string {
  if (mode === "continue") return count > 0 ? `Continue ${count} member${count === 1 ? "" : "s"} on SLURM` : "Continue members on SLURM";
  const noun = mode === "fork" ? "fork" : "member";
  return `Submit ${count} ${noun}${count === 1 ? "" : "s"} to SLURM`;
}

/** "members 199–202" for a contiguous run, else the numbers. */
function membersText(names: readonly string[]): string {
  const nums = names.map((n) => Number(memberNumber(n))).filter(Number.isFinite).sort((a, b) => a - b);
  if (!nums.length) return "";
  const contiguous = nums.every((v, i) => i === 0 || v === nums[i - 1] + 1);
  const pad = (v: number) => String(v).padStart(2, "0");
  if (nums.length > 2 && contiguous) return `members ${pad(nums[0])}–${pad(nums[nums.length - 1])}`;
  return `member${nums.length === 1 ? "" : "s"} ${nums.map(pad).join(", ")}`;
}

/** The Train tab's "Running batch" strip: the live SLURM ensemble_train jobs
 *  (the feed Runs › Live reads), with their members from the job log. */
export function runningBatches(slurm: readonly SlurmJob[], jobs: readonly Pick<TrainingJob, "jobid" | "member_names" | "mode">[]):
  { jobid: string; state: string; members: string[]; text: string }[] {
  const byId = new Map(jobs.map((j) => [String(j.jobid), j]));
  return slurm.filter((j) => j.step_id === "ensemble_train").map((j) => {
    const state = String(j.state ?? "").toUpperCase();
    const members = byId.get(String(j.jobid))?.member_names ?? [];
    const parts = [`Job ${j.jobid}`, membersText(members)];
    if (state === "PENDING") parts.push(`pending${j.reason ? ` (${j.reason})` : ""}`);
    else {
      parts.push(state.toLowerCase() || "running");
      const step = j.progress_step, total = j.progress_total;
      if (step != null && total) parts.push(`step ${formatCount(step)} / ${formatCount(total)} (${Math.round((100 * step) / total)}%)`);
      if (j.time) parts.push(j.time_limit ? `${j.time} of ${j.time_limit}` : String(j.time));
    }
    return { jobid: String(j.jobid), state, members: [...members], text: parts.filter(Boolean).join(" · ") };
  });
}

/** The forward-model values a training batch takes from System › Config
 *  (job_config FASRC_STEP_PARAMS for ensemble_train; the Train form never
 *  sends them, so the preview and the submit use Config's). */
export const CONFIG_FORWARD_KEYS: readonly { key: string; label: string; unit?: string }[] = [
  { key: "psf_warp_prob", label: "PSF warp probability" },
  { key: "psf_warp_alpha_max", label: "PSF warp α max" },
  { key: "psf_warp_sigma", label: "PSF warp σ", unit: "HR px" },
  { key: "saturation_mask_prob", label: "Saturation mask probability" },
];

export function forwardModelFacts(config: Record<string, unknown> | null | undefined): { label: string; value: string; unit?: string }[] {
  return CONFIG_FORWARD_KEYS.filter(({ key }) => config?.[key] != null && config[key] !== "")
    .map(({ key, label, unit }) => ({ label, value: String(Number(Number(config![key]).toPrecision(4))), ...(unit ? { unit } : {}) }));
}

/** The System › Config knobs a FASRC step reads (`used_by`) whose value
 *  differs from its default — the "N knobs changed · Edit" back-link of a
 *  tab judged by a Config group. `only` narrows to a group's keys; `except`
 *  leaves a group's keys out. */
export function changedConfigKnobs(
  payload: { config?: Record<string, unknown>; defaults?: Record<string, unknown>; used_by?: Record<string, string[]> } | null | undefined,
  step: string, group: { only?: readonly string[]; except?: readonly string[] } = {},
): string[] {
  const cfg = payload?.config, defs = payload?.defaults, used = payload?.used_by;
  if (!cfg || !defs || !used) return [];
  const same = (a: unknown, b: unknown) => {
    const x = Number(a), y = Number(b);
    if (a !== "" && b !== "" && a != null && b != null && Number.isFinite(x) && Number.isFinite(y)) return Math.abs(x - y) <= 1e-12 * Math.max(1, Math.abs(y));
    return String(a ?? "") === String(b ?? "");
  };
  return Object.keys(used).filter((k) => used[k]?.includes(step)
    && (!group.only || group.only.includes(k)) && !group.except?.includes(k)
    && k in defs && !same(cfg[k], defs[k])).sort();
}

/** "2 knobs changed" (or null when every knob is at its default). */
export function knobsChangedText(keys: readonly string[]): string | null {
  return keys.length ? `${keys.length} knob${keys.length === 1 ? "" : "s"} changed` : null;
}

const sci = (v: number) => (Math.abs(v) < 0.01 && v !== 0 ? v.toExponential().replace(/\.?0+e/, "e") : String(Number(v.toPrecision(3))));

/** The LR schedule the batch takes from System › Config, in one line. */
export function lrScheduleText(config: Record<string, unknown> | null | undefined): string | null {
  const peak = fnum(config?.lr_peak), final = fnum(config?.lr_final), warm = fnum(config?.lr_warmup_steps);
  if (peak == null) return null;
  const guard = config?.plateau_lr_enabled;
  const on = guard === true || TRUTHY.includes(String(guard).toLowerCase());
  return [`LR ${warm ? `warmup ${formatCount(warm)} steps to ` : ""}${sci(peak)}${final != null ? `, then cosine to ${sci(final)}` : ""}`,
    `plateau guard ${on ? "on" : "off"}`].join("; ");
}

/* ── Images › Records ───────────────────────────────────────────────────── */

const SR_SPLITS: readonly SrSplit[] = ["test", "validate", "train"];

/** The splits whose records carry a generated SR, with their counts. */
export function recordSrSplits(s: SrStatus | null | undefined): { split: SrSplit; n: number; records: number; state: string; reasons: string[] }[] {
  if (!s) return [];
  return SR_SPLITS.flatMap((split) => {
    const info = s.splits[split];
    const n = info?.sr.count ?? s.sr[split] ?? 0;
    return info?.present && n > 0 ? [{ split, n, records: info.count, state: info.sr.state, reasons: info.sr.reasons ?? [] }] : [];
  });
}
