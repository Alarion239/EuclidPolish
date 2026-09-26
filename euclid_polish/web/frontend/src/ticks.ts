/* Axis ticks and domains for the canvas <Plot> (and any SVG chart).
 *
 * Replaces the per-page `niceTicks` / `ticks` / `logDecadeTicks` /
 * `magnitudeTicks` / `levelTicks` / `paddedDomain` helpers. Every generator
 * returns `Tick[]` = `{v, label}` (the same shape `charts/Plot.tsx` takes).
 *
 * Log axes come in two flavours, selected with `space`:
 *   "value" (default) — tick positions are raw values (Plot xScale="log");
 *   "log10"          — the plot is linear in log10 units (the series were
 *                       pre-transformed), so positions are exponents while
 *                       labels still read in physical values.
 */
import { formatPow10 } from "./format";

export type Tick = { v: number; label: string };
export type Domain = [number, number];
export type LogSpace = "value" | "log10";

const EPS = 1e-9;

/** Hard cap on the ticks of one axis. A nice step yields ~count ticks, so
 *  reaching it means the float arithmetic broke down (overflowing, subnormal
 *  or > 2^53-step spans); the generators then fall back instead of looping. */
export const MAX_TICKS = 1000;

/** `step` multiples inside [a, b] (ascending, finite, de-duplicated), or null
 *  when the step is unusable or would produce more than MAX_TICKS. Iterates a
 *  bounded COUNTER, never `i++` on a float that may not advance. */
function stepMultiples(a: number, b: number, step: number): number[] | null {
  if (!(step > 0) || !Number.isFinite(step)) return null;
  const start = Math.ceil(a / step - EPS);
  const end = Math.floor(b / step + EPS);
  if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
  const n = end - start;
  if (!(n >= 0) || n > MAX_TICKS - 1) return null;
  const out: number[] = [];
  for (let k = 0; k <= n; k++) {
    const v = clean((start + k) * step);
    if (Number.isFinite(v) && (out.length === 0 || v > out[out.length - 1])) out.push(v);
  }
  return out;
}

/** The fallback tick set of an unusable range: its (finite) end points. */
function endpoints(a: number, b: number): number[] {
  const out = [clean(a)];
  if (clean(b) > out[0]) out.push(clean(b));
  return out;
}

/** A "nice" 1-2-5 × 10ⁿ step close to span / count (0 for an empty span). */
export function niceStep(span: number, count = 5): number {
  const raw = Math.abs(span) / Math.max(1, count);
  if (!Number.isFinite(raw) || raw === 0) return 0;
  const power = 10 ** Math.floor(Math.log10(raw));
  const e = raw / power;
  const m = e >= Math.sqrt(50) ? 10 : e >= Math.sqrt(10) ? 5 : e >= Math.sqrt(2) ? 2 : 1;
  return clean(m * power);
}

/** Decimals needed to write `step` exactly (0.25 → 2, 5 → 0, 1e-5 → 5). */
export function stepDecimals(step: number): number {
  if (!Number.isFinite(step) || step === 0) return 0;
  for (let d = 0; d <= 12; d++) {
    if (Math.abs(Math.round(step * 10 ** d) - step * 10 ** d) < 1e-6) return d;
  }
  return 12;
}

/** Strip float noise (0.6000000000000001 → 0.6). */
function clean(v: number): number {
  return Number.isFinite(v) ? Number(v.toPrecision(12)) : v;
}

const GROUPED = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

/** Label one tick consistently with its step: decimals from the step,
 *  grouping for big integers, compact exponents at the extremes. */
export function formatTick(v: number, step: number): string {
  if (!Number.isFinite(v)) return "";
  const s = Math.abs(step);
  if (Math.abs(v) < s * EPS || Math.abs(v) < 1e-12) v = 0;
  if (v !== 0 && (Math.abs(v) >= 1e6 || (s > 0 && s < 1e-3))) {
    return Number(v.toPrecision(6)).toExponential().replace("e+", "e");
  }
  const d = stepDecimals(s);
  if (d === 0) return GROUPED.format(Math.round(v));
  return v.toFixed(d).replace(/^-(0\.?0*)$/, "$1");
}

/** Round values from lo..hi (either order) on a nice step. */
export function linearTickValues(lo: number, hi: number, count = 5): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [];
  const [a, b] = lo <= hi ? [lo, hi] : [hi, lo];
  if (a === b) return [clean(a)];
  return stepMultiples(a, b, niceStep(b - a, count)) ?? endpoints(a, b);
}

export type LinearTickOpts = { count?: number; format?: (v: number, step: number) => string };

/** Linear ticks with labels: `linearTicks([0, 1])` → 0.0, 0.2, …, 1.0. */
export function linearTicks([lo, hi]: Domain, opts: LinearTickOpts = {}): Tick[] {
  const count = opts.count ?? 5;
  const vals = linearTickValues(lo, hi, count);
  const gap = vals.length > 1 ? clean(vals[1] - vals[0]) : 0;
  const step = Number.isFinite(gap) && gap > 0 ? gap : niceStep(Math.abs(hi - lo), count);
  const fmt = opts.format ?? formatTick;
  return vals.map((v) => ({ v, label: fmt(v, step) }));
}

/** Tick values on a positive log range: 1-2-5 mantissas over ≤ ~3 decades
 *  (every mantissa below one decade), else decades thinned to `maxTicks`. */
export function logTickValues(lo: number, hi: number, opts: { maxTicks?: number } = {}): number[] {
  const maxTicks = opts.maxTicks ?? 10;
  const [a, b] = lo <= hi ? [lo, hi] : [hi, lo];
  if (!(a > 0) || !Number.isFinite(b)) return [];
  const la = Math.log10(a), lb = Math.log10(b);
  const within = (v: number) => {
    const lv = Math.log10(v);
    return lv >= la - EPS && lv <= lb + EPS;
  };
  const withMantissas = (ms: number[]) => {
    const out: number[] = [];
    for (let e = Math.floor(la) - 1; e <= Math.ceil(lb); e++) {
      for (const m of ms) { const v = clean(m * 10 ** e); if (within(v)) out.push(v); }
    }
    return out;
  };
  const coarse = withMantissas([1, 2, 5]);
  if (coarse.length >= 3 && coarse.length <= maxTicks) return coarse;
  if (coarse.length < 3) {
    const fine = withMantissas([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    return fine.length <= maxTicks ? fine : coarse;
  }
  return decadeExponents(la, lb, maxTicks).map((e) => clean(10 ** e));
}

function decadeExponents(la: number, lb: number, maxTicks: number, step?: number): number[] {
  const lo = Math.ceil(la - EPS), hi = Math.floor(lb + EPS);
  const pick = (s: number) => stepMultiples(lo, hi, s) ?? [];
  // A usable explicit step is a whole number of decades (≥ 1); anything
  // else (≤ 0, NaN, fractional < 1) falls back to the automatic choice.
  if (step != null && Number.isFinite(step) && step >= 1) return pick(Math.round(step));
  for (const s of [1, 2, 3, 5, 10, 20, 50, 100]) {
    const out = pick(s);
    if (out.length <= maxTicks) return out;
  }
  return pick(100);
}

const isDecade = (v: number) => Math.abs(Math.log10(v) - Math.round(Math.log10(v))) < 1e-9;

export type LogTickOpts = { maxTicks?: number; space?: LogSpace; format?: (v: number) => string };

/** Labelled log ticks. Decade-only sets read as powers (10⁻², 1, 10, 10²);
 *  1-2-5 sets as plain numbers. With `space: "log10"` the domain and the
 *  positions are exponents. */
export function logTicks([lo, hi]: Domain, opts: LogTickOpts = {}): Tick[] {
  const space = opts.space ?? "value";
  const [a, b] = space === "log10" ? [10 ** lo, 10 ** hi] : [lo, hi];
  const vals = logTickValues(a, b, { maxTicks: opts.maxTicks });
  const decadesOnly = vals.length > 0 && vals.every(isDecade) && vals.length > 1
    && Math.log10(vals[1] / vals[0]) >= 1 - 1e-9;
  const label = opts.format ?? ((v: number) => (decadesOnly
    ? formatPow10(Math.round(Math.log10(v)))
    : plain(v)));
  return vals.map((v) => ({ v: space === "log10" ? Math.log10(v) : v, label: label(v) }));
}

function plain(v: number): string {
  if (v >= 1e6 || v < 1e-4) return formatPow10(Math.round(Math.log10(v)));
  if (v >= 1) return GROUPED.format(v);
  return String(clean(v));
}

/** Integer decades only (10ⁿ labels), optionally every `step`-th decade. */
export function decadeTicks(
  [lo, hi]: Domain,
  opts: { space?: LogSpace; step?: number; maxTicks?: number } = {},
): Tick[] {
  const space = opts.space ?? "value";
  const [la, lb] = space === "log10" ? [Math.min(lo, hi), Math.max(lo, hi)]
    : [Math.log10(Math.min(lo, hi)), Math.log10(Math.max(lo, hi))];
  if (!Number.isFinite(la) || !Number.isFinite(lb)) return [];
  const exps = decadeExponents(la, lb, opts.maxTicks ?? 12, opts.step);
  return exps.map((e) => ({ v: space === "log10" ? e : clean(10 ** e), label: formatPow10(e) }));
}

const MAG_STEPS = [0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10];

/** Magnitude-axis ticks on whole / half / tenth magnitudes. `invert: true`
 *  negates the positions (bright-at-top axes drawn as −mag). */
export function magnitudeTicks(
  [lo, hi]: Domain,
  opts: { count?: number; invert?: boolean } = {},
): Tick[] {
  const count = opts.count ?? 6;
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [];
  const [a, b] = lo <= hi ? [lo, hi] : [hi, lo];
  const span = b - a;
  // Beyond 10-mag steps (spans > 10·count mag) use a plain nice step so the
  // tick count stays ~count instead of growing with the span.
  const step = MAG_STEPS.find((s) => span / s <= count) ?? niceStep(span, count);
  const d = stepDecimals(step) || 1;
  const ms = stepMultiples(a, b, step) ?? endpoints(a, b);
  return ms.map((m) => ({ v: opts.invert ? -m : m, label: m.toFixed(d) }));
}

/* ── domains ─────────────────────────────────────────────────────────────── */

/** [min, max] of the finite values, or null. */
export function extent(values: Iterable<number | null | undefined>): Domain | null {
  let lo = Infinity, hi = -Infinity;
  for (const v of values) {
    if (typeof v !== "number" || !Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return lo <= hi ? [lo, hi] : null;
}

export type PadOpts = {
  /** Fraction of the span added on each side (default 0.05). */
  pad?: number;
  /** Minimum span (the domain grows symmetrically to reach it). */
  minSpan?: number;
  /** Anchor the domain at zero (no padding below/above zero). */
  includeZero?: boolean;
  /** Returned when there is no finite value (default [0, 1]). */
  fallback?: Domain;
};

/** A padded domain around the finite values; a single value gets ±0.5. */
export function paddedDomain(values: Iterable<number | null | undefined>, opts: PadOpts = {}): Domain {
  const { pad = 0.05, minSpan = 0, includeZero = false, fallback = [0, 1] } = opts;
  const ext = extent(values);
  if (!ext) return fallback;
  let [lo, hi] = ext;
  if (includeZero) { lo = Math.min(0, lo); hi = Math.max(0, hi); }
  if (hi - lo < minSpan) {
    const mid = (lo + hi) / 2;
    [lo, hi] = [mid - minSpan / 2, mid + minSpan / 2];
  }
  if (hi === lo) return [lo - 0.5, hi + 0.5];
  const p = (hi - lo) * pad;
  if (!Number.isFinite(p)) return [lo, hi];                      // the span overflowed
  const out: Domain = [includeZero && lo === 0 ? 0 : lo - p, includeZero && hi === 0 ? 0 : hi + p];
  return out.every(Number.isFinite) ? out : [lo, hi];
}

/** The smallest domain containing every given domain (nulls ignored). */
export function unionDomain(...domains: (Domain | null | undefined)[]): Domain | null {
  return extent(domains.flatMap((d) => (d ? [d[0], d[1]] : [])));
}
