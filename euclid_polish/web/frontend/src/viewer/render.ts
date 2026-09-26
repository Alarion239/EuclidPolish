/* The display pipeline on top of the verbatim colour core (color.ts).
 *
 * With the locked defaults (absolute asinh, black 0, gray colormap, no
 * invert) `renderPrepared` IS `transferCore`, byte for byte, plus one new
 * pass: NaN pixels are painted with the Display panel's NaN colour (the old
 * engine rendered them black, indistinguishable from empty sky). Every other
 * setting is an opt-in, separate code path:
 *   stretch   linear / sqrt / log — anchored at the same black and white as
 *             asinh-abs (white = 30·K0·factor / gain); asinh-auto / zscale —
 *             limits from the frame's own robust statistics (knee-free);
 *   black     subtracted before the transfer (knee units, e⁻ VIS-equivalent);
 *   colormap  single-band frames only (composites keep their colours);
 *   invert    flips the luminance (gray/colormap) or the RGB bytes.
 * `renderSigned` draws residual tiers with a diverging colormap about zero. */
import type { Colormap, Stretch } from "../state/display";
import { asinhTransfer, srgbGamma, transferCore, type Prepared } from "./color";
import { colormapLut } from "./colormaps";
import { robustStats, zscale } from "./stats";

export type DisplayParams = {
  stretch: Stretch;
  /** The frame's transfer group: knee (e⁻), gain (×), black point (e⁻). */
  knee: number;
  gain: number;
  black: number;
  /** meta.color.default_asinh — fixes the white reference (30·K0). */
  K0: number;
  colormap: Colormap;
  invert: boolean;
  nanColor: [number, number, number];
};

const COLOUR_MODES = new Set(["lupton", "direct-rgb", "temp"]);

/** true when the frame renders exactly as the locked default transfer. */
export function isDefaultDisplay(p: DisplayParams, mode = "gray"): boolean {
  const cmapOk = p.colormap === "gray" || COLOUR_MODES.has(mode);
  if (mode === "gray-log") return !p.invert && cmapOk;
  return p.stretch === "asinh-abs" && p.black === 0 && !p.invert && cmapOk;
}

/** The prepareCore colour for a Display colour mode: "rgb" is a Lupton
 *  composite over the chosen bands; every other mode passes through
 *  ("native" is unknown to prepareCore, which then shows channel 0). */
export function prepareColor(color: string, rgb: [string, string, string]): { color: string; scheme: string[] | undefined } {
  if (color === "rgb") return { color: "lupton", scheme: [...rgb] };
  return { color, scheme: undefined };
}

export type FrameAutoStats = { lo: number; hi: number; knee: number; z1: number; z2: number };

const STATS = new WeakMap<Prepared, FrameAutoStats>();

/** The limits the auto stretches use, in the prepared intensity's units
 *  (÷ prep.factor → the transfer's e⁻): asinh-auto black = lo (median − 3σ,
 *  ≥ p0.5), white = hi (p99.8), knee = 3σ; zscale [z1, z2]. Cached per frame. */
export function frameAutoStats(prep: Prepared): FrameAutoStats {
  return frameStats(prep);
}

function frameStats(prep: Prepared): FrameAutoStats {
  let s = STATS.get(prep);
  if (!s) {
    const st = robustStats(prep.I);
    const [z1, z2] = zscale(prep.I);
    const lo = Math.max(st.p005, st.median - 3 * st.sigma);
    const hi = st.p998;
    const knee = Math.max(3 * st.sigma, Math.abs(hi - lo) * 1e-3, 1e-30);
    s = { lo, hi, knee, z1, z2 };
    STATS.set(prep, s);
  }
  return s;
}

/** Scalar luminance t ∈ [0, 1] per pixel for the chosen stretch. */
function luminance(prep: Prepared, p: DisplayParams): (I: number) => number {
  const { factor } = prep;
  const g = p.gain;
  const clip = (t: number) => (t < 0 ? 0 : t > 1 ? 1 : t);
  if (prep.mode === "gray-log") {
    const lo = prep.logLo as number, span = Math.max((prep.logHi as number) - lo, 1e-6);
    return (I) => Math.min(1, clip((I - lo) / span) * g);
  }
  const Kc = Math.max(p.knee * factor, 1e-30);
  const norm = Math.max(Math.asinh((30.0 * p.K0 * factor) / Kc), 1e-6);
  const W = Math.max(30.0 * p.K0 * factor, 1e-30);
  const b = p.black * factor;
  switch (p.stretch) {
    case "linear": return (I) => clip(((I - b) * g) / W);
    case "sqrt": return (I) => Math.sqrt(clip(((I - b) * g) / W));
    case "log": return (I) => Math.log10(1 + 1000 * clip(((I - b) * g) / W)) / Math.log10(1001);
    case "asinh-auto": {
      const s = frameStats(prep);
      // white at the 99.8th percentile, black at the sky (median − 3σ, ≥ p0.5)
      const den = Math.asinh((s.hi > s.lo ? s.hi - s.lo : s.knee) / s.knee);
      return (I) => clip(Math.asinh((Math.max(I - s.lo, 0) * g) / s.knee) / den);
    }
    case "zscale": {
      const s = frameStats(prep);
      const span = s.z2 > s.z1 ? s.z2 - s.z1 : 1;
      return (I) => clip(((I - s.z1) * g) / span);
    }
    default: return (I) => asinhTransfer(I - b, g, Kc, norm);
  }
}

function paintNaN(img: ImageData, I: Float32Array, nan: [number, number, number]): void {
  const out = img.data;
  for (let p = 0; p < I.length; p++) {
    if (I[p] !== I[p]) { const o = p * 4; out[o] = nan[0]; out[o + 1] = nan[1]; out[o + 2] = nan[2]; out[o + 3] = 255; }
  }
}

/** A prepared frame → ImageData with the Display settings. */
export function renderPrepared(prep: Prepared, p: DisplayParams): ImageData {
  if (isDefaultDisplay(p, prep.mode)) {
    const img = transferCore(prep, p.knee, p.gain, p.K0);
    paintNaN(img, prep.I, p.nanColor);
    return img;
  }
  const { w, h, npx, I } = prep;
  const lum = luminance(prep, p);
  const img = new ImageData(w, h);
  const out = img.data;
  if (prep.mode === "gray" || prep.mode === "gray-log") {
    const lut = colormapLut(p.colormap);
    for (let q = 0; q < npx; q++) {
      let k = (lum(I[q]) * 255) | 0;
      if (p.invert) k = 255 - k;
      const o = q * 4;
      out[o] = lut[k * 3]; out[o + 1] = lut[k * 3 + 1]; out[o + 2] = lut[k * 3 + 2]; out[o + 3] = 255;
    }
  } else if (prep.mode === "temp") {
    const hueR = prep.hueR as Float32Array, hueG = prep.hueG as Float32Array, hueB = prep.hueB as Float32Array;
    for (let q = 0; q < npx; q++) {
      const t = lum(I[q]);
      const o = q * 4;
      out[o] = srgbGamma(hueR[q] * t) * 255;
      out[o + 1] = srgbGamma(hueG[q] * t) * 255;
      out[o + 2] = srgbGamma(hueB[q] * t) * 255;
      out[o + 3] = 255;
    }
  } else {
    const R = prep.R as Float32Array, G = prep.G as Float32Array, B = prep.B as Float32Array;
    for (let q = 0; q < npx; q++) {
      const t = lum(I[q]);
      const rescale = I[q] > 1e-30 ? t / I[q] : 0;
      const o = q * 4;
      out[o] = Math.min(Math.max(R[q] * rescale, 0), 1) * 255;
      out[o + 1] = Math.min(Math.max(G[q] * rescale, 0), 1) * 255;
      out[o + 2] = Math.min(Math.max(B[q] * rescale, 0), 1) * 255;
      out[o + 3] = 255;
    }
  }
  if (p.invert && prep.mode !== "gray" && prep.mode !== "gray-log") {
    for (let q = 0; q < npx; q++) { const o = q * 4; out[o] = 255 - out[o]; out[o + 1] = 255 - out[o + 1]; out[o + 2] = 255 - out[o + 2]; }
  }
  paintNaN(img, I, p.nanColor);
  return img;
}

export type SignedParams = DisplayParams & {
  /** "asinh": asinh(v/knee) about zero, saturating at ±range (default
   *  30·K0 like the image transfer); "linear": ±range (log₂ ratios, σ). */
  scale: "asinh" | "linear";
  range?: number;
};

/** A signed map (residual) → ImageData, zero at the colormap centre. */
export function renderSigned(values: Float32Array, w: number, h: number, p: SignedParams): ImageData {
  const img = new ImageData(w, h);
  const out = img.data;
  const lut = colormapLut(p.colormap);
  const g = p.gain;
  const Kc = Math.max(p.knee, 1e-30);
  const norm = Math.max(Math.asinh((p.range && p.range > 0 ? p.range : 30.0 * p.K0) / Kc), 1e-6);
  const range = p.range && p.range > 0 ? p.range : 1;
  for (let q = 0; q < w * h; q++) {
    const v = values[q];
    const o = q * 4;
    if (v !== v) { out[o] = p.nanColor[0]; out[o + 1] = p.nanColor[1]; out[o + 2] = p.nanColor[2]; out[o + 3] = 255; continue; }
    let s = p.scale === "asinh" ? Math.asinh((v * g) / Kc) / norm : (v * g) / range;
    s = s < -1 ? -1 : s > 1 ? 1 : s;
    let k = Math.round((0.5 + 0.5 * s) * 255);
    if (p.invert) k = 255 - k;
    out[o] = lut[k * 3]; out[o + 1] = lut[k * 3 + 1]; out[o + 2] = lut[k * 3 + 2]; out[o + 3] = 255;
  }
  return img;
}
