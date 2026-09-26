/* Pixel statistics of the image viewer (pure): robust statistics and zscale
 * for the auto stretches, histograms of the visible region, line and radial
 * profiles. Coordinates are CONTINUOUS image coordinates: pixel (i, j) covers
 * [i, i+1) × [j, j+1) and its centre is (i + 0.5, j + 0.5). */

export type BandCube = { data: Float32Array; w: number; h: number; c: number };

/** Linear-interpolated percentile (q in 0–100) of an ascending sample. */
export function percentile(sorted: ArrayLike<number>, q: number): number {
  const n = sorted.length;
  if (!n) return NaN;
  const pos = Math.min(n - 1, Math.max(0, ((n - 1) * q) / 100));
  const lo = Math.floor(pos), hi = Math.min(n - 1, lo + 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** The finite values of `values` (a strided subsample above `maxN`), sorted. */
export function finiteSorted(values: ArrayLike<number>, maxN = 65536): Float64Array {
  const n = values.length;
  const stride = Math.max(1, Math.ceil(n / maxN));
  const out: number[] = [];
  for (let i = 0; i < n; i += stride) {
    const v = values[i];
    if (Number.isFinite(v)) out.push(v);
  }
  return Float64Array.from(out).sort();
}

export type RobustStats = {
  n: number; min: number; max: number; median: number;
  /** 1.4826 · MAD (a Gaussian σ robust to sources). */
  sigma: number;
  p005: number; p995: number; p998: number;
};

export function robustStats(values: ArrayLike<number>, maxN = 65536): RobustStats {
  const s = finiteSorted(values, maxN);
  const n = s.length;
  if (!n) return { n: 0, min: NaN, max: NaN, median: NaN, sigma: NaN, p005: NaN, p995: NaN, p998: NaN };
  const median = percentile(s, 50);
  const dev = Float64Array.from(s, (v) => Math.abs(v - median)).sort();
  return {
    n, min: s[0], max: s[n - 1], median, sigma: 1.4826 * percentile(dev, 50),
    p005: percentile(s, 0.5), p995: percentile(s, 99.5), p998: percentile(s, 99.8),
  };
}

/** IRAF zscale display limits: fit a line to the sorted sample with
 *  iterative sigma rejection, scale its slope by 1/contrast about the median. */
export function zscale(values: ArrayLike<number>, opts: { samples?: number; contrast?: number; krej?: number; maxIter?: number } = {}): [number, number] {
  const { samples = 1000, contrast = 0.25, krej = 2.5, maxIter = 5 } = opts;
  const s = finiteSorted(values, samples);
  const n = s.length;
  if (!n) return [NaN, NaN];
  const zmin = s[0], zmax = s[n - 1];
  if (n < 5) return [zmin, zmax];
  const median = percentile(s, 50);
  const centre = (n - 1) / 2;
  const minpix = Math.max(5, Math.floor(n * 0.5));
  const ngrow = Math.max(1, Math.round(n * 0.01));
  let good = new Uint8Array(n).fill(1);
  let slope = 0, ngood = n;
  for (let iter = 0; iter < maxIter; iter++) {
    let sx = 0, sy = 0, sxx = 0, sxy = 0, m = 0;
    for (let i = 0; i < n; i++) if (good[i]) { sx += i; sy += s[i]; sxx += i * i; sxy += i * s[i]; m++; }
    const den = m * sxx - sx * sx;
    if (m < 2 || den === 0) break;
    slope = (m * sxy - sx * sy) / den;
    const intercept = (sy - slope * sx) / m;
    let ss = 0;
    for (let i = 0; i < n; i++) if (good[i]) { const r = s[i] - (intercept + slope * i); ss += r * r; }
    const sigma = Math.sqrt(ss / m);
    const next = new Uint8Array(n).fill(1);
    let rejected = 0;
    for (let i = 0; i < n; i++) {
      const r = s[i] - (intercept + slope * i);
      if (Math.abs(r) > krej * sigma) {
        for (let j = Math.max(0, i - ngrow); j <= Math.min(n - 1, i + ngrow); j++) next[j] = 0;
      }
    }
    for (let i = 0; i < n; i++) if (!next[i]) rejected++;
    ngood = n - rejected;
    const changed = next.some((v, i) => v !== good[i]);
    good = next;
    if (!changed || ngood < minpix) break;
  }
  if (ngood < minpix) return [zmin, zmax];
  if (contrast > 0) slope /= contrast;
  const z1 = Math.max(zmin, median - centre * slope);
  const z2 = Math.min(zmax, median + (n - 1 - centre) * slope);
  return z2 > z1 ? [z1, z2] : [zmin, zmax];
}

export type Histogram = { edges: number[]; counts: number[]; outside: number };

/** Equal-width histogram over [lo, hi] (hi falls in the last bin). */
export function histogram(values: ArrayLike<number>, lo: number, hi: number, bins: number): Histogram {
  const counts = new Array<number>(bins).fill(0);
  const edges = Array.from({ length: bins + 1 }, (_, i) => lo + ((hi - lo) * i) / bins);
  let outside = 0;
  const span = hi - lo;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    if (v < lo || v > hi || !(span > 0)) { outside++; continue; }
    counts[Math.min(bins - 1, Math.floor(((v - lo) / span) * bins))]++;
  }
  return { edges, counts, outside };
}

/** Band values inside an integer pixel rectangle (clipped to the image). */
export function regionValues(rec: BandCube, band: number, r: { x: number; y: number; w: number; h: number }): Float32Array {
  const x0 = Math.max(0, Math.floor(r.x)), y0 = Math.max(0, Math.floor(r.y));
  const x1 = Math.min(rec.w, Math.ceil(r.x + r.w)), y1 = Math.min(rec.h, Math.ceil(r.y + r.h));
  if (x1 <= x0 || y1 <= y0) return new Float32Array(0);
  const out = new Float32Array((x1 - x0) * (y1 - y0));
  let k = 0;
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) out[k++] = rec.data[(y * rec.w + x) * rec.c + band];
  return out;
}

const valueAt = (rec: BandCube, band: number, x: number, y: number): number | null => {
  const i = Math.floor(x), j = Math.floor(y);
  if (i < 0 || j < 0 || i >= rec.w || j >= rec.h) return null;
  const v = rec.data[(j * rec.w + i) * rec.c + band];
  return Number.isFinite(v) ? v : null;
};

export type Profile = { d: number[]; v: (number | null)[] };

/** Nearest-pixel samples every pixel step along p0 → p1; `d` in pixels. */
export function lineProfile(rec: BandCube, band: number, p0: { x: number; y: number }, p1: { x: number; y: number }): Profile {
  const L = Math.hypot(p1.x - p0.x, p1.y - p0.y);
  const n = Math.floor(L + 1e-9) + 1;
  const ux = L > 0 ? (p1.x - p0.x) / L : 0, uy = L > 0 ? (p1.y - p0.y) / L : 0;
  const d: number[] = [], v: (number | null)[] = [];
  for (let k = 0; k < n; k++) {
    d.push(k);
    v.push(valueAt(rec, band, p0.x + ux * k, p0.y + uy * k));
  }
  return { d, v };
}

export type RadialProfile = { r: number[]; v: (number | null)[]; n: number[] };

/** Azimuthal mean in annuli of width `dr` (px) around `c`, out to `rmax`. */
export function radialProfile(rec: BandCube, band: number, c: { x: number; y: number }, rmax: number, dr = 1): RadialProfile {
  const bins = Math.max(1, Math.ceil(rmax / dr));
  const sum = new Array<number>(bins).fill(0), n = new Array<number>(bins).fill(0);
  const x0 = Math.max(0, Math.floor(c.x - rmax)), x1 = Math.min(rec.w - 1, Math.ceil(c.x + rmax));
  const y0 = Math.max(0, Math.floor(c.y - rmax)), y1 = Math.min(rec.h - 1, Math.ceil(c.y + rmax));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const r = Math.hypot(x + 0.5 - c.x, y + 0.5 - c.y);
      if (r >= rmax) continue;
      const v = rec.data[(y * rec.w + x) * rec.c + band];
      if (!Number.isFinite(v)) continue;
      const b = Math.min(bins - 1, Math.floor(r / dr));
      sum[b] += v; n[b]++;
    }
  }
  return {
    r: sum.map((_, b) => (b + 0.5) * dr),
    v: sum.map((s, b) => (n[b] ? s / n[b] : null)),
    n,
  };
}
