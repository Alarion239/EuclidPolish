/* Plot model: the pure half of Plot v2 — axis transforms (linear/log), zoom
   and pan maths, nearest-point hit testing, the per-x readout, tick
   selection for zoomed views, CSV export and the "did the inputs change"
   comparison that gates redraws. Unit-tested in plotModel.test.ts. */
import { formatNumber } from "../format";
import { linearTicks, logTicks, type Tick } from "../ticks";
import { csvCell } from "../ui/tableModel";
import type { AxisScale, Series } from "./types";

/* ─── axes ────────────────────────────────────────────────────────────────── */

export type Axis = {
  scale: AxisScale;
  domain: [number, number];
  /** data → px (NaN for values a log axis cannot show). */
  toPx: (v: number) => number;
  /** px → data. */
  fromPx: (px: number) => number;
  /** Can this value be placed on the axis? */
  ok: (v: number) => boolean;
  /** data → [0, 1] along the domain (in log space for log axes). */
  frac: (v: number) => number;
};

const log10 = Math.log10;

/** Positive, ordered, non-degenerate domain for a scale. */
export function safeDomain([a, b]: [number, number], scale: AxisScale): [number, number] {
  let lo = Math.min(a, b), hi = Math.max(a, b);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) { lo = scale === "log" ? 1 : 0; hi = scale === "log" ? 10 : 1; }
  if (scale === "log") {
    if (!(hi > 0)) { lo = 1; hi = 10; }
    if (!(lo > 0)) lo = hi / 1e6;
    if (hi / lo < 1 + 1e-12) { lo /= 2; hi *= 2; }
  } else if (hi - lo <= Math.abs(hi) * 1e-12 || hi === lo) {
    const d = Math.abs(hi) > 0 ? Math.abs(hi) * 0.05 : 0.5;
    lo -= d; hi += d;
  }
  return [lo, hi];
}

/** An axis mapping `domain` onto pixels `p0 → p1` (p1 < p0 for y). */
export function axis(domain: [number, number], scale: AxisScale, p0: number, p1: number): Axis {
  const [lo, hi] = safeDomain(domain, scale);
  const log = scale === "log";
  const a = log ? log10(lo) : lo, b = log ? log10(hi) : hi;
  const frac = (v: number) => ((log ? log10(v) : v) - a) / (b - a);
  return {
    scale, domain: [lo, hi],
    toPx: (v) => (log && !(v > 0) ? NaN : p0 + frac(v) * (p1 - p0)),
    fromPx: (px) => {
      const t = a + ((px - p0) / (p1 - p0)) * (b - a);
      return log ? 10 ** t : t;
    },
    ok: (v) => Number.isFinite(v) && (!log || v > 0),
    frac,
  };
}

export type PlotGeometry = {
  m: { l: number; r: number; t: number; b: number };
  iw: number;
  ih: number;
  x: Axis;
  y: Axis;
};

/* ─── zoom / pan ──────────────────────────────────────────────────────────── */

/** Scale a domain by `factor` (<1 zooms in) about `anchor` ∈ [0, 1]. */
export function zoomDomain(domain: [number, number], scale: AxisScale, anchor: number, factor: number): [number, number] {
  const log = scale === "log";
  const a = log ? log10(domain[0]) : domain[0], b = log ? log10(domain[1]) : domain[1];
  const c = a + anchor * (b - a);
  const na = c - (c - a) * factor, nb = c + (b - c) * factor;
  return log ? [10 ** na, 10 ** nb] : [na, nb];
}

/** Shift a domain by `frac` of its span (positive = towards larger values). */
export function panDomain(domain: [number, number], scale: AxisScale, frac: number): [number, number] {
  const log = scale === "log";
  const a = log ? log10(domain[0]) : domain[0], b = log ? log10(domain[1]) : domain[1];
  const d = (b - a) * frac;
  return log ? [10 ** (a + d), 10 ** (b + d)] : [a + d, b + d];
}

/** Ordered view domain, or null when it is empty / invalid for the scale. */
export function clampView(d: [number, number], scale: AxisScale): [number, number] | null {
  const lo = Math.min(d[0], d[1]), hi = Math.max(d[0], d[1]);
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || !(hi > lo)) return null;
  if (scale === "log" && !(lo > 0)) return null;
  const span = scale === "log" ? log10(hi) - log10(lo) : hi - lo;
  if (!(span > 1e-12 * Math.max(1, Math.abs(scale === "log" ? log10(hi) : hi)))) return null;
  return [lo, hi];
}

/* ─── hit testing ─────────────────────────────────────────────────────────── */

export function seriesKey(s: Series, i: number): string {
  return s.key ?? s.name ?? s.label ?? `#${i}`;
}

export function seriesName(s: Series, i: number): string {
  return s.name ?? s.label ?? s.key ?? `series ${i + 1}`;
}

const SORTED = new WeakMap<number[], boolean>();
function isSorted(x: number[]): boolean {
  let v = SORTED.get(x);
  if (v === undefined) {
    v = true;
    for (let i = 1; i < x.length; i++) if (!(x[i] >= x[i - 1])) { v = false; break; }
    SORTED.set(x, v);
  }
  return v;
}

/** Index whose x is nearest `xv` (binary search when x is ascending). */
export function nearestIndex(x: number[], xv: number): number {
  const n = x.length;
  if (!n) return -1;
  if (isSorted(x)) {
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (x[mid] <= xv) lo = mid; else hi = mid; }
    return Math.abs(x[lo] - xv) <= Math.abs(x[hi] - xv) ? lo : hi;
  }
  let best = -1, bd = Infinity;
  for (let i = 0; i < n; i++) {
    const d = Math.abs(x[i] - xv);
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

export type Hit = { series: number; index: number; x: number; y: number; px: number; py: number; dist: number };

const validY = (y: number | null | undefined): y is number => y != null && Number.isFinite(y);

/** Can point `i` of `s` be drawn on these axes (finite, and > 0 on a log axis)? */
export function drawable(s: Series, i: number, g: Pick<PlotGeometry, "x" | "y">): boolean {
  if (!(i >= 0 && i < s.x.length)) return false;
  const y = s.y[i];
  return validY(y) && g.x.ok(s.x[i]) && g.y.ok(y);
}

/** Keyboard readout: the next drawable point of `s` after (`dir` 1) or before
 *  (-1) index `from`, staying on `from` at the end of the series; `dir` 0 is
 *  the drawable point nearest `from`. -1 when the series has none. Gaps (null,
 *  or ≤ 0 on a log axis) are stepped over. */
export function stepDrawable(s: Series, from: number, dir: -1 | 0 | 1, g: Pick<PlotGeometry, "x" | "y">): number {
  const n = s.x.length;
  if (!n) return -1;
  const at = Math.max(0, Math.min(n - 1, Math.round(from)));
  if (dir !== 0) {
    for (let i = at + dir; i >= 0 && i < n; i += dir) if (drawable(s, i, g)) return i;
  }
  for (let d = 0; d < n; d++) {
    if (drawable(s, at - d, g)) return at - d;
    if (drawable(s, at + d, g)) return at + d;
  }
  return -1;
}

/** Nearest visible data point to the cursor (px), within `maxDist` px. Lines
 *  and histograms check the points around the nearest x; scatters scan all. */
export function nearestPoint(
  series: Series[], hidden: ReadonlySet<string>, cursor: { x: number; y: number }, g: PlotGeometry,
  maxDist = 80,
): Hit | null {
  let best: Hit | null = null;
  const xv = g.x.fromPx(cursor.x);
  const consider = (si: number, i: number) => {
    const s = series[si];
    const y = s.y[i], x = s.x[i];
    if (!validY(y) || !Number.isFinite(x) || !g.x.ok(x) || !g.y.ok(y)) return;
    const px = g.x.toPx(x), py = g.y.toPx(y);
    const dist = Math.hypot(px - cursor.x, py - cursor.y);
    if (dist <= maxDist && (!best || dist < best.dist)) best = { series: si, index: i, x, y, px, py, dist };
  };
  series.forEach((s, si) => {
    if (hidden.has(seriesKey(s, si)) || !s.x.length) return;
    if (s.mode === "scatter") {
      for (let i = 0; i < s.x.length; i++) consider(si, i);
      return;
    }
    const c = nearestIndex(s.x, xv);
    if (c < 0) return;
    // the nearest-x point and the nearest valid neighbours on each side
    consider(si, c);
    for (const dir of [-1, 1]) {
      let seen = 0;
      for (let i = c + dir; i >= 0 && i < s.x.length && seen < 2; i += dir) {
        if (validY(s.y[i])) { consider(si, i); seen++; }
      }
    }
  });
  return best;
}

export type Readout = { series: number; index: number; x: number; y: number };

/** Each visible line/histogram series' value at the x nearest `xv` (the
 *  first `limit`, in series order). */
export function readoutAt(series: Series[], hidden: ReadonlySet<string>, xv: number, limit = 12): Readout[] {
  const out: Readout[] = [];
  series.forEach((s, si) => {
    if (out.length >= limit || s.mode === "scatter" || hidden.has(seriesKey(s, si))) return;
    const i = nearestIndex(s.x, xv);
    if (i < 0 || !validY(s.y[i])) return;
    out.push({ series: si, index: i, x: s.x[i], y: s.y[i] as number });
  });
  return out;
}

/** The tooltip's rows at `xv`: every visible line when they fit in `limit`;
 *  otherwise the series `first` (the hovered one) plus the lines whose value
 *  lies nearest the cursor in pixels (`cursorPy` on axis `y`). Rows stay in
 *  series order; `more` counts the lines left out. */
export function tooltipReadout(
  series: Series[], hidden: ReadonlySet<string>, xv: number,
  opts: { limit?: number; cursorPy: number; y: Axis; first?: number | null },
): { rows: Readout[]; more: number } {
  const limit = Math.max(1, opts.limit ?? 8);
  const all = readoutAt(series, hidden, xv, Infinity);
  if (all.length <= limit) return { rows: all, more: 0 };
  const dist = (r: Readout) => {
    const py = opts.y.toPx(r.y);
    return Number.isFinite(py) ? Math.abs(py - opts.cursorPy) : Infinity;
  };
  const keep = new Set<number>();
  if (opts.first != null && all.some((r) => r.series === opts.first)) keep.add(opts.first);
  // nearest first; ties keep series order (Array.prototype.sort is stable)
  const ranked = all.map((r) => ({ r, d: dist(r) })).sort((a, b) => a.d - b.d);
  for (const { r } of ranked) {
    if (keep.size >= limit) break;
    keep.add(r.series);
  }
  return { rows: all.filter((r) => keep.has(r.series)), more: all.length - keep.size };
}

/* ─── ticks for a zoomed view ─────────────────────────────────────────────── */

/** Caller ticks inside `domain` when at least `min` remain, else generated
 *  ticks (linear or log) labelled with `format` (default: shared tick format). */
export function viewTicks(
  ticks: Tick[] | undefined, domain: [number, number], scale: AxisScale,
  format?: (v: number) => string, min = 3,
): Tick[] {
  const [lo, hi] = [Math.min(...domain), Math.max(...domain)];
  const inside = (ticks ?? []).filter((t) => t.v >= lo && t.v <= hi);
  if (inside.length >= min) return inside;
  let out: Tick[] = [];
  if (scale === "log" && lo > 0) {
    out = logTicks([lo, hi], { maxTicks: 8, format });
  }
  if (out.filter((t) => t.v >= lo && t.v <= hi).length < 2) {
    out = linearTicks([lo, hi], { count: 5, format: format ? (v) => format(v) : undefined });
  }
  return out.filter((t) => t.v >= lo && t.v <= hi);
}

/* ─── readout formatting / CSV ────────────────────────────────────────────── */

export const formatValue = (v: number): string => formatNumber(v, { sig: 4 });

/** Long-format CSV: `series,x,y` (+ `low,high`, `errorLow,errorHigh` when
 *  any series carries them). Cells go through the DataTable encoder
 *  (RFC 4180 quoting, spreadsheet formulas in series names defused). */
export function seriesToCSV(series: Series[]): string {
  const band = series.some((s) => s.low || s.high);
  const err = series.some((s) => s.errorLow || s.errorHigh);
  const head = ["series", "x", "y", ...(band ? ["low", "high"] : []), ...(err ? ["errorLow", "errorHigh"] : [])];
  const lines = [head.join(",")];
  series.forEach((s, si) => {
    const name = seriesName(s, si);
    for (let i = 0; i < s.x.length; i++) {
      const row: (string | number | null | undefined)[] = [name, s.x[i], s.y[i]];
      if (band) row.push(s.low?.[i], s.high?.[i]);
      if (err) row.push(s.errorLow?.[i], s.errorHigh?.[i]);
      lines.push(row.map(csvCell).join(","));
    }
  });
  return lines.join("\r\n") + "\r\n";
}

/* ─── redraw gate ─────────────────────────────────────────────────────────── */

/* Structural equality; a function is equal only to itself (a nested
   function such as `heat.color` is a drawn input). */
function eq(a: unknown, b: unknown, depth: number): boolean {
  if (Object.is(a, b)) return true;
  if (depth <= 0 || a == null || b == null || typeof a !== "object" || typeof b !== "object") return false;
  if (Array.isArray(a)) {
    if (!Array.isArray(b) || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (!eq(a[i], b[i], depth - 1)) return false;
    return true;
  }
  if (Array.isArray(b)) return false;
  const A = a as Record<string, unknown>, B = b as Record<string, unknown>;
  for (const k of new Set([...Object.keys(A), ...Object.keys(B)])) {
    if (A[k] === undefined && B[k] === undefined) continue;
    if (!eq(A[k], B[k], depth - 1)) return false;
  }
  return true;
}

/** Structural equality of two values (arrays / plain objects, 4 levels deep;
 *  functions by identity). */
export function sameValue(a: unknown, b: unknown): boolean {
  return eq(a, b, 4);
}

/** True when two Plot prop objects draw the same picture: values compared
 *  structurally (so `series={[{x: data.x, …}]}` rebuilt on every render does
 *  not redraw). Top-level functions are skipped: the handlers never draw, and
 *  `xFormat`/`yFormat` only label zoomed ticks, which Plot compares by their
 *  output. A nested drawn function (`heat.color`) is compared by identity, so
 *  a new colour closure redraws; memoise it to avoid needless repaints. */
export function sameInputs(a: object, b: object): boolean {
  const A = a as Record<string, unknown>, B = b as Record<string, unknown>;
  for (const k of new Set([...Object.keys(A), ...Object.keys(B)])) {
    const x = A[k], y = B[k];
    const skip = (v: unknown) => v === undefined || typeof v === "function";
    if (skip(x) && skip(y)) continue;
    if (!eq(x, y, 4)) return false;
  }
  return true;
}
