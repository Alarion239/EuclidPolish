/* The Synthetic chart kit: ONE palette and one set of series builders for every
 * Synthetic figure (the galaxy marginals and relations, the three joint galaxy
 * views, the stellar plots, the noise and field-statistics plots). Colours
 * are read from theme tokens at call time (colors.ts), so a figure rebuilt on
 * a theme flip picks up the new values; ticks and domains come from ticks.ts.
 * Everything here is pure except the token readers. */
import { C, bandColor, categorical } from "../../colors";
import type { Series } from "../../charts/Plot";
import { formatNumber } from "../../format";
import { extent, linearTicks, logTicks, paddedDomain, type Domain, type Tick } from "../../ticks";

/* ─── numbers ───────────────────────────────────────────────────────────── */

/** Every ASCII hyphen before a digit as the typographic minus ("−0.15"), as
 *  the console writes negatives (ranges use an en dash, so they are safe). */
export const withMinus = (text: string): string => text.replace(/-(?=[\d.])/g, "\u2212");

/** formatNumber with the typographic minus. */
export const formatSigned = (...args: Parameters<typeof formatNumber>): string => withMinus(formatNumber(...args));

/* ─── palette ───────────────────────────────────────────────────────────── */

export type SourceKey = "euclid" | "synthetic" | "cosmos" | "fit";
export type Survey = SourceKey | "generation";

/** Survey colours: Q1/Euclid blue, generated amber, fitted model red,
 *  COSMOS green, the generation law teal (categorical tokens). */
export function surveyColor(survey: Survey): string {
  switch (survey) {
    case "euclid": return categorical(0);
    case "synthetic": return categorical(2);
    case "cosmos": return categorical(1);
    case "fit": return categorical(6);
    default: return categorical(5);
  }
}

export const SOURCE_META: Record<SourceKey, { label: string; kicker: string }> = {
  euclid: { label: "Euclid MER + PHZ", kicker: "Euclid Q1" },
  synthetic: { label: "Generated source catalogues", kicker: "Generated fields" },
  cosmos: { label: "COSMOS2025", kicker: "Diagnostic only" },
  fit: { label: "Euclid joint fit", kicker: "Fitted model" },
};

/** Joint galaxy views. Corner + explorer: Q1 blue, model red. The
 *  magnitude × radius maps: Q1 gray (the reference density), generated blue
 *  dashed, model red solid. */
export type JointSource = "q1" | "synthetic" | "model";
export function jointColor(source: JointSource, view: "pairs" | "maps" = "pairs"): string {
  if (source === "model") return categorical(6);
  if (view === "maps") return source === "q1" ? C.cross : categorical(0);
  return source === "q1" ? categorical(0) : categorical(2);
}

/** Colour of a colour index by its redder band (VIS−Y → Y, Y−J → J, J−H → H). */
export const colorIndexColor = (redderBand: "Y" | "J" | "H") => bandColor(redderBand);

/* ─── numbers and axes ──────────────────────────────────────────────────── */

export const finite = (values: readonly (number | null | undefined)[]): number[] =>
  values.filter((v): v is number => typeof v === "number" && Number.isFinite(v));

/** Positive values or null (gaps on a log axis). */
export const positiveOrNull = (values: readonly (number | null | undefined)[]): (number | null)[] =>
  values.map((v) => (typeof v === "number" && Number.isFinite(v) && v > 0 ? v : null));

/** A log-axis domain over the positive values, snapped outward to half
 *  decades (at least one decade wide). */
export function logDomain(values: readonly (number | null | undefined)[], fallback: Domain = [1e-3, 1]): Domain {
  // extent(), not Math.min(...spread): a spread throws past ~120k values.
  const span = extent(finite(values).filter((v) => v > 0).map(Math.log10));
  if (!span) return fallback;
  const lo = Math.floor(span[0] * 2) / 2;
  const hiRaw = Math.ceil(span[1] * 2) / 2;
  const hi = hiRaw <= lo ? lo + 1 : hiRaw;
  return [10 ** lo, 10 ** hi];
}

/** A labelled axis in log10 coordinates ("log₁₀ Rₑ (arcsec)") read as
 *  physical values: "Rₑ (arcsec, log scale)". */
export function physicalLogAxisLabel(label: string): string {
  const physical = label.replace(/^log₁₀\s*/, "");
  return physical.endsWith(")") ? `${physical.slice(0, -1)}, log scale)` : `${physical} (log scale)`;
}

export const isLog10Label = (label: string) => label.startsWith("log₁₀");

/** 10^v for log10 coordinates. */
export const fromLog10 = (values: readonly number[]) => values.map((v) => 10 ** v);

export const ticksFor = (domain: Domain, scale: "linear" | "log", count = 6): Tick[] =>
  scale === "log" ? logTicks(domain, { maxTicks: count + 2 }) : linearTicks(domain, { count });

/** Percent label of an enclosed-mass contour: 0.1 → "10%", 0.995 → "99.5%". */
export function contourMassLabel(fraction: number): string {
  const pct = 100 * fraction;
  const rounded = Math.round(pct * 10) / 10;
  return `${Number.isInteger(rounded) ? rounded.toFixed(0) : rounded.toFixed(1)}%`;
}

/** Inner contours heaviest, the outer tail faintest (every joint view). */
export function contourStyle(fraction: number): { width: number; opacity: number } {
  if (fraction <= 0.2) return { width: 1.8, opacity: 1 };
  if (fraction <= 0.5) return { width: 1.5, opacity: 0.95 };
  if (fraction <= 0.8) return { width: 1.15, opacity: 0.8 };
  if (fraction <= 0.95) return { width: 0.85, opacity: 0.6 };
  return { width: 0.65, opacity: 0.4 };
}

export type ContourPath = { x: number[]; y: number[] };
export type Contour = { mass_fraction: number; paths: ContourPath[] };

/** Contour lines as Plot series, labelled once (on the longest path) by the
 *  enclosed mass. `mapY` maps the path's y (e.g. log10 → physical). */
export function contourSeries(
  contours: readonly Contour[],
  opts: {
    color: string; dash?: number[]; name?: string; key?: string;
    levels?: readonly number[]; labelAt?: (contourIndex: number) => number;
    widthScale?: number; mapY?: (y: number) => number;
  },
): Series[] {
  const out: Series[] = [];
  contours.forEach((contour, ci) => {
    if (opts.levels && !opts.levels.some((l) => Math.abs(l - contour.mass_fraction) < 1e-9)) return;
    if (!contour.paths.length) return;
    const longest = contour.paths.reduce((best, p) => (p.x.length > best.x.length ? p : best), contour.paths[0]);
    const style = contourStyle(contour.mass_fraction);
    for (const path of contour.paths) {
      const main = path === longest;
      out.push({
        x: path.x, y: opts.mapY ? path.y.map(opts.mapY) : path.y,
        color: opts.color, dash: opts.dash, width: style.width * (opts.widthScale ?? 1.3), alpha: style.opacity,
        label: main ? contourMassLabel(contour.mass_fraction) : undefined,
        labelAt: main && opts.labelAt ? opts.labelAt(ci) : undefined,
        name: opts.name, key: opts.key ?? opts.name,
      });
    }
  });
  return out;
}

/* ─── histograms and steps ──────────────────────────────────────────────── */

/** Step outline of a binned density: each bin drawn flat across its edges. */
export function stepSeries(edges: readonly number[], density: readonly number[]): { x: number[]; y: number[] } {
  const x: number[] = [];
  const y: number[] = [];
  density.forEach((value, bin) => { x.push(edges[bin], edges[bin + 1]); y.push(value, value); });
  return { x, y };
}

/** ∫ density dx over [lo, hi] for a binned density given at bin centres `x` (ascending). Each bin spans
 *  the midpoints to its neighbours (the edge bins mirror their one neighbour) and contributes the part
 *  of it inside the window; missing values contribute nothing. Null when no bin overlaps the window. */
export function integrateDensity(
  x: readonly number[], density: readonly (number | null | undefined)[], lo: number, hi: number,
): number | null {
  if (!(hi > lo) || x.length < 2) return null;
  let total = 0;
  let overlapped = false;
  for (let i = 0; i < x.length; i++) {
    const left = i > 0 ? (x[i - 1] + x[i]) / 2 : x[0] - (x[1] - x[0]) / 2;
    const right = i < x.length - 1 ? (x[i] + x[i + 1]) / 2 : x[i] + (x[i] - x[i - 1]) / 2;
    const overlap = Math.min(hi, right) - Math.max(lo, left);
    const v = density[i];
    if (overlap <= 0 || v == null || !Number.isFinite(v)) continue;
    total += v * overlap;
    overlapped = true;
  }
  return overlapped ? total : null;
}

export const binCenters = (edges: readonly number[]) =>
  edges.slice(0, -1).map((lo, i) => (lo + edges[i + 1]) / 2);

/** Index of the bin holding `value` (edges ascending), -1 outside. */
export function binIndex(edges: readonly number[], value: number): number {
  if (!(value >= edges[0] && value <= edges[edges.length - 1])) return -1;
  let index = 0;
  while (index < edges.length - 2 && edges[index + 1] <= value) index++;
  return index;
}

/** Smallest enclosed-mass region containing cell (i, j): the share of the
 *  layer's mass in cells at least as dense as it (null when empty). */
export function enclosingFraction(density: readonly (readonly number[])[] | undefined, i: number, j: number): number | null {
  const value = density?.[i]?.[j];
  if (!density || value == null || !(value > 0)) return null;
  let total = 0;
  let denser = 0;
  for (const column of density) {
    for (const cell of column) {
      total += cell;
      if (cell >= value) denser += cell;
    }
  }
  return total > 0 ? denser / total : null;
}

/** Cumulative stack (tallest first, so every layer stays visible): layer k is
 *  the sum of groups 0..k. Returns tops in draw order (largest first). */
export function stackedTops(groups: readonly (readonly number[])[], bins: number): number[][] {
  const tops: number[][] = [];
  let running = new Array<number>(bins).fill(0);
  for (const counts of groups) {
    running = running.map((v, b) => v + (counts[b] ?? 0));
    tops.push(running);
  }
  return tops.reverse();
}

/** "How many fields step at least this much" from a histogram of steps: the
 *  percentage of `fields` at or above each lower edge (last edge → 0). */
export function exceedancePercent(counts: readonly number[], fields: number): number[] {
  let remaining = counts.reduce((a, b) => a + b, 0);
  const out = counts.map((count) => {
    const above = remaining;
    remaining -= count;
    return fields ? (100 * above) / fields : 0;
  });
  return [...out, 0];
}

/** Angular scale (1/k) ascending, with the permutation that orders a
 *  per-frequency array the same way (non-positive frequencies dropped). */
export function angularScaleAxis(frequencies: readonly number[]): { x: number[]; order: number[] } {
  const points = frequencies
    .map((frequency, index) => ({ frequency, index }))
    .filter(({ frequency }) => frequency > 0 && Number.isFinite(frequency))
    .map(({ frequency, index }) => ({ scale: 1 / frequency, index }))
    .sort((a, b) => a.scale - b.scale);
  return { x: points.map((p) => p.scale), order: points.map((p) => p.index) };
}

export const ordered = <T,>(values: readonly T[], order: readonly number[]): T[] => order.map((i) => values[i]);

/** A histogram with one bin (the exact-zero bin) blanked. */
export const omitBin = (values: readonly number[], index: number | null | undefined): (number | null)[] =>
  values.map((v, i) => (i === index ? null : v));

/* ─── galaxy relations ──────────────────────────────────────────────────── */

export type ConditionalRadius = {
  magnitude: number[];
  observed_mean_log10_arcsec: (number | null)[];
  model_mean_log10_arcsec: number[];
  model_core_low_log10_arcsec?: number[];
  model_core_high_log10_arcsec?: number[];
  model_low_log10_arcsec?: number[];
  model_high_log10_arcsec?: number[];
};

/** The model band of the brightness–radius relation: the core (one-scatter)
 *  interval when the fit provides it, else the full low/high interval. */
export function radiusBand(relation: ConditionalRadius): { low: number[]; high: number[]; kind: "core" | "full" | "none" } {
  if (relation.model_core_low_log10_arcsec && relation.model_core_high_log10_arcsec) {
    return { low: relation.model_core_low_log10_arcsec, high: relation.model_core_high_log10_arcsec, kind: "core" };
  }
  if (relation.model_low_log10_arcsec && relation.model_high_log10_arcsec) {
    return { low: relation.model_low_log10_arcsec, high: relation.model_high_log10_arcsec, kind: "full" };
  }
  return { low: [], high: [], kind: "none" };
}

/* ─── picking ───────────────────────────────────────────────────────────── */

/** Index of the point nearest `target`, distances measured in units of each
 *  axis span (so both axes weigh alike); -1 when none is finite or the best
 *  is farther than `maxDistance` spans. Log axes compare log10 values. */
export function nearestIndex2d(
  xs: readonly number[], ys: readonly number[], target: { x: number; y: number },
  spans: { x: number; y: number }, opts: { xLog?: boolean; yLog?: boolean; maxDistance?: number } = {},
): number {
  const fx = (v: number) => (opts.xLog ? Math.log10(v) : v);
  const fy = (v: number) => (opts.yLog ? Math.log10(v) : v);
  const tx = fx(target.x), ty = fy(target.y);
  let best = -1;
  let bestD = Infinity;
  for (let i = 0; i < xs.length; i++) {
    const x = fx(xs[i]), y = fy(ys[i]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const d = Math.hypot((x - tx) / (spans.x || 1), (y - ty) / (spans.y || 1));
    if (d < bestD) { bestD = d; best = i; }
  }
  return best >= 0 && bestD <= (opts.maxDistance ?? 0.05) ? best : -1;
}

/** A padded linear domain (min span) — the kit's one domain helper. */
export const domainOf = (values: readonly (number | null | undefined)[], minSpan = 0, includeZero = false): Domain =>
  paddedDomain(values, { pad: 0.045, minSpan, includeZero });

export const fmt = (v: number | null | undefined, digits = 2) => formatNumber(v, { digits });

/* ─── colour mixing (heat shading from tokens) ──────────────────────────── */

export type Rgb = [number, number, number];

/** A token's resolved colour (`#rgb`, `#rrggbb`, `rgb()`/`rgba()`) as RGB;
 *  `fallback` for anything else (e.g. a named colour). */
export function parseRgb(color: string, fallback: Rgb = [128, 128, 128]): Rgb {
  const text = color.trim();
  const hex = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(text);
  if (hex) {
    const h = hex[1].length === 3 ? hex[1].split("").map((c) => c + c).join("") : hex[1];
    const n = parseInt(h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  const rgb = /^rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)/i.exec(text);
  if (rgb) return [Number(rgb[1]), Number(rgb[2]), Number(rgb[3])].map((v) => Math.round(Math.min(255, v))) as Rgb;
  return fallback;
}

/** `tint` laid over `base` with `weight` ∈ [0, 1], as an opaque `rgb()` (heat
 *  cells overlap their neighbours slightly: translucent ones double up in the
 *  seams, so shading is an opaque blend). */
export function mixRgb(tint: Rgb, base: Rgb, weight: number): string {
  const w = Math.max(0, Math.min(1, weight));
  return `rgb(${base.map((channel, k) => Math.round(channel + (tint[k] - channel) * w)).join(", ")})`;
}

const RAMPS = new Map<string, (t: number) => string>();

/** Heat colour ramp of one sample's density: `tint` over the card surface,
 *  from a faint wash (t = 0) to 65 % (t = 1). Memoised per colour pair, so a
 *  re-render hands Plot the same function (its `heat.color` is compared by
 *  identity) and a theme flip — new token values — gets a new one. */
export function tintRamp(tint: string, surface: string): (t: number) => string {
  const key = `${tint}|${surface}`;
  let ramp = RAMPS.get(key);
  if (!ramp) {
    const a = parseRgb(tint);
    const b = parseRgb(surface, [255, 255, 255]);
    ramp = (t: number) => mixRgb(a, b, 0.05 + 0.6 * t);
    RAMPS.set(key, ramp);
  }
  return ramp;
}

/** The card surface token (`--surface-1`) as the current theme resolves it
 *  ("" before the stylesheet applies: `parseRgb` then falls back to white). */
export function surfaceColor(): string {
  if (typeof document === "undefined") return "";
  return getComputedStyle(document.documentElement).getPropertyValue("--surface-1").trim();
}

/* ─── axes in log10 coordinates ─────────────────────────────────────────── */

/** Ticks of an axis whose data are log10 values: positions stay exponents,
 *  labels read physical values (the pair explorer's log₁₀ SFR / Rₑ axes). */
export function log10AxisTicks(domain: Domain, maxTicks = 7): Tick[] {
  return logTicks(domain, { space: "log10", maxTicks });
}

/** A number in physical units from its log10 (tooltips of log10 axes). */
export const physicalFromLog10 = (v: number, sig = 3) => formatNumber(10 ** v, { sig });

/* ─── robust summaries (per-field counts) ───────────────────────────────── */

/** Linear-interpolated quantile of finite values (null when empty). */
export function quantile(values: readonly number[], q: number): number | null {
  const sorted = finite(values).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const pos = Math.max(0, Math.min(1, q)) * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(sorted.length - 1, lo + 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** Integer-count histogram: unit bins from 0 to the largest count (capped
 *  at `maxBins`, wider bins beyond), as fractions of the fields. */
export function countHistogram(values: readonly number[], maxBins = 40): { edges: number[]; fraction: number[] } {
  const good = finite(values).filter((v) => v >= 0);
  if (!good.length) return { edges: [0, 1], fraction: [0] };
  const top = extent(good)?.[1] ?? 0;
  const width = Math.max(1, Math.ceil((top + 1) / maxBins));
  const bins = Math.max(1, Math.ceil((top + 1) / width));
  const counts = new Array<number>(bins).fill(0);
  for (const v of good) counts[Math.min(bins - 1, Math.floor(v / width))] += 1;
  return {
    edges: Array.from({ length: bins + 1 }, (_, i) => i * width),
    fraction: counts.map((c) => c / good.length),
  };
}
