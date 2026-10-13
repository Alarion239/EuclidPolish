/* Pure logic of Models › Universal (unit-tested in universal.test.ts): the
   toolbar's choices (x axis, metric, combiner weighting), the universal
   models' rows (status, the chosen metric, hardware, Δ vs production and vs
   the time-matched combiner), the combiners' training hours under a
   weighting, the two plots' points and per-depth lines with their depth
   colour slots, the nearest point to a click, the real-field facts of each
   spec from the newest Sky › Compare run that scored THIS model, the
   Sky › Compare deep link, what to score and what to pull, the summary and
   the best universal model the Leaderboard names. No React, no DOM.

   Training time is the payload's: Σ duration_s of a model's training-log
   windows (rolled-back ones included) in hours; a combiner's is Σ over the
   members it reads of hours × that member's peak gate share ("max"), its
   mean share ("mean"), or plain hours ("sum") — a lower bound ("partial")
   when a member it reads has no training time.

   A Δ is only ever taken between scores on the same fields, records and
   target: Δ vs production when the payload is `comparable`, Δ vs the
   time-matched combiner when THAT combiner is (`combinerComparable`) and its
   score is current; a partial combiner is never time-matched. A stale model
   score sits where it was made (`scored_hours` / `scored_steps`). */
import type {
  CombinerMemberHours, ExperimentSummary, UniversalCombiner, UniversalFasrcRow, UniversalHardware, UniversalPayload, UniversalReference, UniversalRow,
} from "./api";
import { benchmarkExperiment, kneeNum, memberNumber, variantLabel, type Bench } from "./model";

/* ── toolbar choices (URL state) ───────────────────────────────────────── */

export type XAxis = "time" | "steps";
export const X_AXES: readonly { value: XAxis; label: string }[] = [
  { value: "time", label: "Training time" }, { value: "steps", label: "Steps" },
];

export type Metric = "mean" | "VIS" | "Y_E" | "J_E" | "H_E";
export const METRICS: readonly { value: Metric; label: string; axis: string }[] = [
  { value: "mean", label: "∫PSNR mean", axis: "∫PSNR [dB]" },
  { value: "VIS", label: "VIS", axis: "∫PSNR VIS [dB]" },
  { value: "Y_E", label: "Y", axis: "∫PSNR Y [dB]" },
  { value: "J_E", label: "J", axis: "∫PSNR J [dB]" },
  { value: "H_E", label: "H", axis: "∫PSNR H [dB]" },
];

export type Weighting = "max" | "mean" | "sum";
export const WEIGHTINGS: readonly { value: Weighting; label: string; text: string }[] = [
  { value: "max", label: "Max weight", text: "max-weighted" },
  { value: "mean", label: "Mean weight", text: "mean-weighted" },
  { value: "sum", label: "Every member", text: "summed over every member" },
];

const oneOf = <T extends string>(options: readonly { value: T }[]) =>
  (raw: string): T | undefined => (options.some((o) => o.value === raw) ? raw as T : undefined);
export const parseXAxis = oneOf(X_AXES);
export const parseMetric = oneOf(METRICS);
export const parseWeighting = oneOf(WEIGHTINGS);

/* ── values ────────────────────────────────────────────────────────────── */

export const DEFAULT_BANDS = ["VIS", "Y_E", "J_E", "H_E"] as const;
export const DEFAULT_FIELDS = 100;

const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const positive = (v: unknown): v is number => finite(v) && v > 0;
const bandsOf = (p: Pick<UniversalPayload, "bands"> | null | undefined): readonly string[] =>
  (p?.bands?.length ? p.bands : DEFAULT_BANDS);

/** A score's value under the chosen metric: the mean of the bands (the
 *  payload's `mean`, else the mean of four finite bands) or one band. */
export function metricValue(s: { integrated?: readonly (number | null)[] | null; mean?: number | null } | null | undefined,
  metric: Metric, bands: readonly string[] = DEFAULT_BANDS): number | null {
  if (!s) return null;
  if (metric === "mean") {
    if (finite(s.mean)) return s.mean;
    const vs = s.integrated ?? [];
    return vs.length && vs.every(finite) ? (vs as number[]).reduce((a, b) => a + b, 0) / vs.length : null;
  }
  const i = bands.indexOf(metric);
  const v = i >= 0 ? s.integrated?.[i] : null;
  return finite(v) ? v : null;
}

/** Σ over the members a combiner reads of hours × weight (the peak share for
 *  "max", the mean share for "mean", 1 for "sum"); null when a member has no
 *  hours or no weight (the sum would understate the combiner). */
export function weightedHours(members: readonly CombinerMemberHours[], w: Weighting): number | null {
  if (!members.length) return null;
  let sum = 0;
  for (const m of members) {
    const k = w === "sum" ? 1 : w === "max" ? m.max_weight : m.mean_weight;
    if (!finite(m.hours) || !finite(k)) return null;
    sum += m.hours * k;
  }
  return sum;
}

/** A combiner's training hours under a weighting: the payload's, else the
 *  sum over its members (a lower bound when isPartial). */
export function combinerHours(c: Pick<UniversalCombiner, "hours" | "members">, w: Weighting): number | null {
  const v = c.hours?.[w];
  return finite(v) ? v : weightedHours(c.members ?? [], w);
}

/** A member it reads has no training time: its hours are a lower bound. */
export const isPartial = (c: Pick<UniversalCombiner, "missing_hours">): boolean => (c.missing_hours?.length ?? 0) > 0;

/** A combiner's hours as text: "2.00 h", "≥ 2.00 h" when partial, "—". */
export function combinerHoursText(c: Pick<UniversalCombiner, "hours" | "members" | "missing_hours">, w: Weighting, unit = true): string {
  const v = combinerHours(c, w);
  if (v == null) return "—";
  const t = unit ? hoursText(v) : hoursText(v).replace(/ h$/, "");
  return isPartial(c) ? `≥ ${t}` : t;
}

/** Whether a combiner's score is on the universal scores' fields, records
 *  and target: the server's flag; without one a knee-PSNR score follows the
 *  payload (`payloadComparable`) and a compare report's is not (it recorded
 *  no identity). */
export function combinerComparable(c: Pick<UniversalCombiner, "comparable" | "score_source">, payloadComparable: boolean): boolean {
  if (typeof c.comparable === "boolean") return c.comparable;
  return c.score_source === "knee-psnr" ? payloadComparable : false;
}

/** "production gate", else "gate p20s1". */
export function combinerName(c: Pick<UniversalCombiner, "production" | "name" | "id">): string {
  return c.production ? "production gate" : `gate ${variantLabel(c.name || c.id)}`;
}

/** Trained on a MIG slice (gpu_test): drawn as a ring, not comparable compute. */
export function isMig(hw: Partial<UniversalHardware> | null | undefined): boolean {
  return hw?.partition === "gpu_test" || /\bMIG\b/i.test(hw?.label ?? "");
}

export const hardwareText = (hw: Partial<UniversalHardware> | null | undefined): string => hw?.label || hw?.partition || "—";

export type UniversalStatus = "scored" | "stale" | "unscored" | "no-checkpoint";
export const STATUS_TEXT: Record<UniversalStatus, string> = {
  scored: "scored", stale: "stale score", unscored: "not scored", "no-checkpoint": "no checkpoint",
};

/** Whether a model can be (re)scored and its score is current. */
export function universalStatus(r: Pick<UniversalRow, "fingerprint" | "score">): UniversalStatus {
  if (!r.fingerprint) return "no-checkpoint";
  if (!r.score) return "unscored";
  return r.score.stale ? "stale" : "scored";
}

/* ── combiners on the time axis ────────────────────────────────────────── */

/** A scored combiner at its weighted training hours: `partial` (a lower
 *  bound: a member it reads has no hours), `stale` (its score predates its
 *  fit or its members' checkpoints), `comparable` (combinerComparable). */
export type CombinerPoint = {
  id: string; name: string; production: boolean; hours: number; value: number;
  partial: boolean; stale: boolean; comparable: boolean; combiner: UniversalCombiner;
};

export function combinerPoints(combiners: readonly UniversalCombiner[], w: Weighting, metric: Metric,
  bands: readonly string[] = DEFAULT_BANDS, payloadComparable = true): CombinerPoint[] {
  const out: CombinerPoint[] = [];
  for (const c of combiners) {
    const hours = combinerHours(c, w);
    const value = metricValue({ integrated: c.integrated, mean: c.mean }, metric, bands);
    if (positive(hours) && value != null) {
      out.push({ id: c.id, name: combinerName(c), production: c.production, hours, value, partial: isPartial(c), stale: !!c.stale,
        comparable: combinerComparable(c, payloadComparable), combiner: c });
    }
  }
  return out;
}

/** The scored combiner nearest in log hours to `hours` (production on a
 *  tie); a partial one never (its hours are only a lower bound). */
export function timeMatch(hours: number | null | undefined, points: readonly CombinerPoint[]): CombinerPoint | null {
  if (!positive(hours) || !points.length) return null;
  const lh = Math.log10(hours);
  let best: CombinerPoint | null = null;
  let bestD = Infinity;
  for (const p of points) {
    if (p.partial) continue;
    const d = Math.abs(Math.log10(p.hours) - lh);
    if (d < bestD - 1e-12 || (Math.abs(d - bestD) <= 1e-12 && p.production && !best?.production)) { best = p; bestD = d; }
  }
  return best;
}

/* ── the models' rows ──────────────────────────────────────────────────── */

export type ShapedRow = UniversalRow & {
  num: string; depth: number | null;
  /** Training hours (null: none recorded); training length in steps. */
  hours: number | null; stepsDone: number | null;
  /** Where its score sits: the current hours and steps, or, for a stale
   *  score, those recorded when it was scored (else the current ones). */
  scoreHours: number | null; scoreSteps: number | null;
  /** The chosen metric, the band mean, and each band's ∫PSNR. */
  value: number | null; mean: number | null; bandValues: (number | null)[];
  status: UniversalStatus; mig: boolean;
  /** Its score is of an earlier checkpoint or other fields (status "stale"):
   *  drawn faded, its Δs marked. */
  stale: boolean;
  /** The chosen metric minus production's (null when not comparable). */
  vsProduction: number | null;
  /** The scored combiner nearest in training hours (at `scoreHours`), and
   *  the metric minus its — null when that combiner's score is on other
   *  fields (`matchBlock` "other-fields") or stale ("stale"). */
  match: CombinerPoint | null; vsMatch: number | null;
  matchBlock: "other-fields" | "stale" | null;
};

/** Whether the knee-PSNR payload (production, the plain mean, the production
 *  gate's ∫PSNR) was scored on the same fields, records and target as the
 *  universal scores. The payload's `comparable` when it says; else read off
 *  the rows: the server gives `vs_production_db` only when comparable, so a
 *  scored model without one while production has a score means "not". */
export function isComparable(p: Pick<UniversalPayload, "comparable" | "production" | "combiners" | "models"> | null | undefined): boolean {
  if (typeof p?.comparable === "boolean") return p.comparable;
  const models = p?.models ?? [];
  if (models.some((r) => finite(r.vs_production_db))) return true;
  const prod = metricValue(productionReference(p), "mean");
  return !(prod != null && models.some((r) => finite(r.score?.mean)));
}

/** The production reference: the knee-PSNR status, else the production combiner's score. */
export function productionReference(p: Pick<UniversalPayload, "production" | "combiners"> | null | undefined): UniversalReference | null {
  if (p?.production) return p.production;
  const c = p?.combiners?.find((x) => x.production);
  return c && c.integrated ? { integrated: c.integrated, mean: c.mean, stale: c.stale } : null;
}

export function shapeRows(p: UniversalPayload, opts: { weighting: Weighting; metric: Metric }): ShapedRow[] {
  const bands = bandsOf(p);
  const comparable = isComparable(p);
  const points = combinerPoints(p.combiners ?? [], opts.weighting, opts.metric, bands, comparable);
  const prod = metricValue(productionReference(p), opts.metric, bands);
  return (p.models ?? []).map((r): ShapedRow => {
    const value = metricValue(r.score, opts.metric, bands);
    const status = universalStatus(r);
    const stale = status === "stale";
    const hours = positive(r.train_hours) ? r.train_hours : null;
    const stepsDone = positive(r.steps) ? r.steps : null;
    const scoreHours = stale && positive(r.scored_hours) ? r.scored_hours : hours;
    const scoreSteps = !stale ? stepsDone : positive(r.scored_steps) ? r.scored_steps : positive(r.served_step) ? r.served_step : stepsDone;
    const match = timeMatch(scoreHours, points);
    const matchBlock = !match || value == null ? null : !match.comparable ? "other-fields" : match.stale ? "stale" : null;
    return {
      ...r,
      num: memberNumber(r.name) ?? r.name,
      depth: finite(r.num_res_blocks) ? r.num_res_blocks : null,
      hours, stepsDone, scoreHours, scoreSteps,
      value,
      mean: metricValue(r.score, "mean", bands),
      bandValues: bands.map((_, i) => { const v = r.score?.integrated?.[i]; return finite(v) ? v : null; }),
      status, stale,
      mig: isMig(r.hardware),
      vsProduction: !comparable ? null : value != null && prod != null ? value - prod
        : opts.metric === "mean" && finite(r.vs_production_db) ? r.vs_production_db : null,
      match,
      vsMatch: match && value != null && !matchBlock ? value - match.value : null,
      matchBlock,
    };
  });
}

/* ── depth colours ─────────────────────────────────────────────────────── */

/** The categorical slots a depth takes, skipping --cat-0 (the plain mean's
 *  blue, its y guide) and --cat-2 (the combiner's orange) in both themes, so
 *  no depth reads as the mean or a combiner. */
export const DEPTH_SLOTS: readonly number[] = [1, 3, 4, 5, 6, 7];

/** The depths present, shallow to deep; unknown last. */
export function depthOrder(rows: readonly { depth: number | null }[]): (number | null)[] {
  const ds = [...new Set(rows.map((r) => r.depth))];
  return ds.sort((a, b) => (a ?? Infinity) - (b ?? Infinity));
}
export const depthKey = (d: number | null) => `depth:${d ?? "?"}`;
export const depthLabel = (d: number | null) => (d == null ? "unknown depth" : `depth ${d}`);
export function depthSlot(order: readonly (number | null)[], d: number | null): number {
  return DEPTH_SLOTS[Math.max(0, order.indexOf(d)) % DEPTH_SLOTS.length];
}

/* ── plots ─────────────────────────────────────────────────────────────── */

/** One model's point: x = hours or steps, y = the chosen metric; `stale`
 *  (a stale score, drawn faded where it was made). */
export type ModelPoint = {
  name: string; num: string; key: string; depth: number | null; x: number; y: number; mig: boolean; stale: boolean; label: string;
};

/** "#211 · depth 16 · 1.0M steps · 12.7 h · MIG slice" (the tooltip name),
 *  "… (stale score)" for a stale one; the steps and hours where it sits. */
export function pointLabel(r: Pick<ShapedRow, "num" | "depth" | "mig"> & { stale?: boolean; steps: number | null; hours: number | null }): string {
  const text = [`#${r.num}`, r.depth != null ? `depth ${r.depth}` : null, r.steps ? `${stepsShort(r.steps)} steps` : null,
    r.hours ? hoursText(r.hours) : null, r.mig ? "MIG slice" : null].filter(Boolean).join(" · ");
  return r.stale ? `${text} (stale score)` : text;
}

const pointOf = (r: ShapedRow, x: number): ModelPoint => ({
  name: r.name, num: r.num, key: depthKey(r.depth), depth: r.depth, x, y: r.value as number, mig: r.mig, stale: r.stale,
  label: pointLabel({ ...r, steps: r.scoreSteps, hours: r.scoreHours }),
});

/** The "by training time" points (at `scoreHours`), and the models it
 *  cannot draw: `unscored` (no current or stale score under the metric) and
 *  `untimed` (no training time recorded). */
export function timePoints(rows: readonly ShapedRow[]): { points: ModelPoint[]; unscored: string[]; untimed: string[] } {
  const points: ModelPoint[] = [];
  const unscored: string[] = [];
  const untimed: string[] = [];
  for (const r of rows) {
    if (r.value == null) unscored.push(r.num);
    else if (r.scoreHours == null) untimed.push(r.num);
    else points.push(pointOf(r, r.scoreHours));
  }
  return { points, unscored, untimed };
}

/** The "by training length" lines: per depth (shallow first), the models
 *  with a score and a step count, by steps (at `scoreSteps`). */
export function lengthLines(rows: readonly ShapedRow[]): { depth: number | null; key: string; points: ModelPoint[] }[] {
  const by = new Map<number | null, ModelPoint[]>();
  for (const r of rows) {
    if (r.value == null || r.scoreSteps == null) continue;
    by.set(r.depth, [...(by.get(r.depth) ?? []), pointOf(r, r.scoreSteps)]);
  }
  return depthOrder([...by.keys()].map((depth) => ({ depth })))
    .map((depth) => ({ depth, key: depthKey(depth), points: [...(by.get(depth) ?? [])].sort((a, b) => a.x - b.x || a.y - b.y) }));
}

/** A log-axis domain around the positive values (×/÷ `pad`), or `fallback`. */
export function logDomain(values: Iterable<number | null | undefined>, fallback: [number, number], pad = 1.6): [number, number] {
  let lo = Infinity, hi = -Infinity;
  for (const v of values) if (positive(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (!(hi >= lo)) return fallback;
  return [lo / pad, hi * pad];
}

/** The drawn point nearest a click (in the plot's own proportions: log x when
 *  `xLog`), within `reach` of the plot's size; null when none is that close. */
export function nearestPoint<T extends { x: number; y: number }>(points: readonly T[], at: { x: number; y: number },
  xDomain: [number, number], yDomain: [number, number], xLog = true, reach = 0.06): T | null {
  const fx = (v: number) => (xLog ? Math.log10(Math.max(v, 1e-300)) : v);
  const sx = fx(xDomain[1]) - fx(xDomain[0]) || 1;
  const sy = yDomain[1] - yDomain[0] || 1;
  let best: T | null = null;
  let bestD = reach;
  for (const p of points) {
    const d = Math.hypot((fx(p.x) - fx(at.x)) / sx, (p.y - at.y) / sy);
    if (d <= bestD) { best = p; bestD = d; }
  }
  return best;
}

/* ── jobs ──────────────────────────────────────────────────────────────── */

/** The models a score job should run: a local checkpoint and no current score. */
export function toScore(rows: readonly Pick<UniversalRow, "name" | "fingerprint" | "score">[]): string[] {
  return rows.filter((r) => { const s = universalStatus(r); return s === "unscored" || s === "stale"; }).map((r) => r.name);
}

/** The SLURM states that leave a checkpoint (as Members › Pull counts them). */
const PULLABLE_STATES = new Set(["COMPLETED", "TIMEOUT"]);

/** What a FASRC row offers: "pull" (finished with a checkpoint), "again" (a
 *  registered universal model whose pull left no checkpoint here), "try"
 *  (ended FAILED / CANCELLED / OUT_OF_MEMORY …: maybe no checkpoint), or
 *  "training" (nothing yet). The server's `pullable`, else its state. */
export type FasrcKind = "pull" | "again" | "try" | "training";
export function fasrcKind(r: UniversalFasrcRow): FasrcKind {
  if (!r.finished) return "training";
  if (r.registered) return "again";
  const ok = typeof r.pullable === "boolean" ? r.pullable : PULLABLE_STATES.has(String(r.state ?? "").toUpperCase());
  return ok ? "pull" : "try";
}

/** FASRC's universal models not here, by what they offer (fasrcKind):
 *  `finished` (pull them), `again` (registered, pull again), `failed`
 *  (ended without COMPLETED / TIMEOUT: try one by one) and `training`. */
export function fasrcSplit(rows: readonly UniversalFasrcRow[]): { finished: string[]; again: string[]; failed: string[]; training: string[] } {
  const byNum = (a: string, b: string) => Number(memberNumber(a)) - Number(memberNumber(b));
  const of = (k: FasrcKind) => rows.filter((r) => fasrcKind(r) === k).map((r) => r.name).sort(byNum);
  return { finished: of("pull"), again: of("again"), failed: of("try"), training: of("training") };
}

/** How many test fields the scores use (the store's identity, else 100). */
export function scoredFields(p: Pick<UniversalPayload, "identity" | "models"> | null | undefined): number {
  const n = p?.identity?.indices?.length;
  if (n) return n;
  const seen = (p?.models ?? []).map((r) => r.score?.n_fields).filter(positive);
  return seen.length ? Math.max(...seen) : DEFAULT_FIELDS;
}

/** The gates whose weights can be measured now (the payload's
 *  `weights_pending`), named as the combiners table names them. */
export function pendingWeights(p: Pick<UniversalPayload, "weights_pending" | "combiners"> | null | undefined): { id: string; name: string }[] {
  return (p?.weights_pending ?? []).map((id) => {
    const c = p?.combiners?.find((x) => x.id === id);
    return { id, name: c ? combinerName(c) : id === "production" ? "production gate" : `gate ${variantLabel(id)}` };
  });
}

/** "212, 213 and 2 more". */
export function numbersText(names: readonly string[], max = 6): string {
  const nums = names.map((n) => memberNumber(n) ?? n);
  return nums.length > max ? `${nums.slice(0, max).join(", ")} and ${nums.length - max} more` : nums.join(", ");
}

/* ── real fields (Sky › Compare) ───────────────────────────────────────── */

type ExpDetail = { id: string; fingerprints?: Record<string, string | null> | null };
type Catalog = { models: readonly { spec: string; fingerprint?: string | null }[] };

/** One spec's real facts: from the newest Sky › Compare run that scored the
 *  CURRENT model (its fingerprint equals the catalogue's), else the run that
 *  scored an earlier one, else none; "loading" while a record is not in. */
export type RealPick =
  | { state: "current"; expId: string; created?: string; label?: string; tileSet: string; bench: Bench }
  | { state: "earlier"; expId: string; created?: string; label?: string }
  | { state: "loading" }
  | { state: "none" };

/** How many of a spec's newest runs are checked for a current score. */
export const REAL_WALK = 3;

/** Each spec's real facts (`picks`) and the run records the walk reads
 *  (`ids`: every record it needs, loaded or not, so keep fetching all of
 *  them). Per spec, its runs newest first, at most `walk`: the first whose
 *  recorded fingerprint is the catalogue's wins; a record not in yet stops
 *  the walk ("loading"); one that could not be read (`failed`) is skipped. */
export function realPicks(specs: readonly string[], exps: readonly ExperimentSummary[] | null | undefined,
  details: ReadonlyMap<string, ExpDetail>, catalog: Catalog | null | undefined, walk = REAL_WALK,
  failed: ReadonlySet<string> = new Set()): { picks: Map<string, RealPick>; ids: string[] } {
  const sorted = [...(exps ?? [])].filter((e) => e.summary && Object.keys(e.summary).length > 0)
    .sort((a, b) => String(b.created ?? "").localeCompare(String(a.created ?? "")));
  const current = catalog ? new Map(catalog.models.map((m) => [m.spec, m.fingerprint ?? null])) : null;
  const picks = new Map<string, RealPick>();
  const ids = new Set<string>();
  for (const spec of specs) {
    const runs = sorted.filter((e) => spec in (e.summary ?? {})).slice(0, Math.max(1, walk));
    let pick: RealPick | null = null;
    let earlier: RealPick | null = null;
    for (const e of runs) {
      ids.add(e.id);
      if (failed.has(e.id)) continue;
      const d = details.get(e.id);
      if (!d || !current) { pick = { state: "loading" }; break; }
      const fp = d.fingerprints?.[spec];
      const bench = fp && fp === current.get(spec) ? benchmarkExperiment([e], e.id) : null;
      const b = bench?.bySpec.get(spec);
      if (bench && b) { pick = { state: "current", expId: e.id, created: e.created, label: e.label, tileSet: bench.tileSet, bench: b }; break; }
      earlier ??= { state: "earlier", expId: e.id, created: e.created, label: e.label };
    }
    picks.set(spec, pick ?? earlier ?? { state: "none" });
  }
  return { picks, ids: [...ids] };
}

/** The Sky › Compare "New comparison" drawer with production and these specs
 *  picked: `/sky/compare?new=1&models=production,universal:member_211`. */
export function compareHref(specs: readonly string[]): string {
  const models = [...new Set(["production", ...specs.filter(Boolean)])];
  const enc = (s: string) => encodeURIComponent(s).replace(/%3A/gi, ":");
  return `/sky/compare?new=1&models=${models.map(enc).join(",")}`;
}

/* ── summary ───────────────────────────────────────────────────────────── */

/** The best universal model by the band mean (current scores; a stale one
 *  only when none is current), with the production gate it is set against:
 *  its band mean and its training hours under `weighting`. */
export function universalSummary(rows: readonly ShapedRow[], p: Pick<UniversalPayload, "production" | "combiners" | "models" | "comparable">,
  weighting: Weighting): {
  n: number; nScored: number; best: ShapedRow | null; bestStale: boolean;
  /** `partial`: its hours are a lower bound (a member it reads has none). */
  production: { mean: number | null; hours: number | null; partial: boolean; stale: boolean } | null;
  /** The best's band mean minus production's (null when not comparable). */
  gain: number | null;
} {
  const pick = (status: UniversalStatus) => rows.filter((r) => r.status === status && r.mean != null)
    .reduce<ShapedRow | null>((b, r) => (b == null || (r.mean as number) > (b.mean as number) ? r : b), null);
  const current = pick("scored");
  const best = current ?? pick("stale");
  const ref = productionReference(p);
  const prodComb = p.combiners?.find((c) => c.production) ?? null;
  const prodMean = metricValue(ref, "mean");
  return {
    n: rows.length,
    nScored: rows.filter((r) => r.status === "scored").length,
    best, bestStale: !current && !!best,
    production: ref || prodComb ? {
      mean: prodMean, hours: prodComb ? combinerHours(prodComb, weighting) : null, partial: !!prodComb && isPartial(prodComb),
      stale: !!ref?.stale,
    } : null,
    gain: best?.mean != null && prodMean != null && isComparable(p) ? best.mean - prodMean : null,
  };
}

/** The Leaderboard's "best universal model" row: the best CURRENT score,
 *  when the scores are on the Leaderboard's own fields (isComparable). */
export function bestUniversal(p: UniversalPayload | null | undefined): { row: UniversalRow; mean: number; bands: (number | null)[] } | null {
  if (!p || !isComparable(p)) return null;
  let best: { row: UniversalRow; mean: number; bands: (number | null)[] } | null = null;
  const bands = bandsOf(p);
  for (const row of p?.models ?? []) {
    if (universalStatus(row) !== "scored") continue;
    const mean = metricValue(row.score, "mean", bands);
    if (mean != null && (!best || mean > best.mean)) {
      best = { row, mean, bands: bands.map((_, i) => { const v = row.score?.integrated?.[i]; return finite(v) ? v : null; }) };
    }
  }
  return best;
}

/* ── formatting ────────────────────────────────────────────────────────── */

/** Steps as the axes read them: 950, 1.5k, 29k, 977k, 1.0M. */
export function stepsShort(v: number | null | undefined): string {
  if (!finite(v)) return "—";
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (v >= 1e4) return `${Math.round(v / 1e3)}k`;
  if (v >= 1e3) return `${+(v / 1e3).toFixed(1)}k`;
  return String(Math.round(v));
}

/** Hours: "0.29 h", "2.58 h", "12.7 h". */
export function hoursText(h: number | null | undefined): string {
  if (!finite(h)) return "—";
  return `${h < 10 ? h.toFixed(2) : h.toFixed(1)} h`;
}

/** "learned", or "fixed 10 e⁻". */
export function kneeWords(r: Pick<UniversalRow, "knee" | "output_knee">): string {
  if (r.knee === "learned") return "learned";
  return finite(r.output_knee) ? `fixed ${kneeNum(r.output_knee)} e⁻` : "fixed";
}

/** The learned output knee per band: "VIS 8.1 · Y 12 · J 15 · H 20 e⁻". */
export function learnedKneesText(r: Pick<UniversalRow, "learned_knees">, bands: readonly string[] = DEFAULT_BANDS): string | null {
  const ks = r.learned_knees ?? [];
  if (!ks.some(finite)) return null;
  const short = (b: string) => b.replace(/_E$/, "");
  return `${ks.map((k, i) => `${short(bands[i] ?? `band ${i + 1}`)} ${finite(k) ? kneeNum(k) : "—"}`).join(" · ")} e⁻`;
}

/** How the training step ran: "XLA + bf16", "XLA", "bf16" or "plain". */
export function stepImplText(r: Pick<UniversalRow, "jit_compile" | "mixed_precision">): string {
  if (r.jit_compile && r.mixed_precision) return "XLA + bf16";
  if (r.jit_compile) return "XLA";
  if (r.mixed_precision) return "bf16";
  return "plain";
}

export const classText = (c: UniversalRow["class"]) => (c === "member" ? "ensemble member" : "universal");
