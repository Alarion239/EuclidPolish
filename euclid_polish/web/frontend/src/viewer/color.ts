/* Colour core of the image viewer — ported verbatim from the pre-rework
 * static/cutout_viewer.js, which ported it from visualization/color.py.
 *
 *   prepareCore  : cube + colour mode → Prepared   (EXPENSIVE: the per-pixel
 *                  temperature fit for "temp", the band composite for "lupton")
 *   transferCore : Prepared + knee/gain/K0 → ImageData  (CHEAP: the locked
 *                  absolute asinh transfer, re-run on every slider tick)
 *
 * The locked default (spec 2026-06-24 unified cutout viewer): absolute asinh,
 * t = clip(asinh(I·gain/Kc) / asinh(30·K0·factor/Kc), 0, 1) with Kc = knee·factor,
 * K0 = meta.color.default_asinh (white at 30·K0 e⁻, VIS-equivalent).
 *
 * This module has NO imports so Node can load it directly
 * (scripts/check_viewer_parity.mjs). Held to the old engine's outputs and to
 * color.py by color.test.ts (golden fixture from scripts/_viewer_parity_ref.py).
 * New stretches, colormaps, black point and NaN colour are separate code paths
 * in render.ts; with the defaults render.ts calls transferCore unchanged.
 */

export const EYE_T_MIN = 1667.0;
export const EYE_T_MAX = 25000.0;

/** One band's calibration constants (the served meta's `color.bands[name]`). */
export type BandConstants = {
  zeropoint_ab_e_total: number;
  solar_ab_mag?: number;
  pivot_um?: number;
  t_total_s?: number;
  zeropoint_ab?: number;
  asinh_scale_e?: number;
  /** A display-only band (JWST F###): no AB zero point, no magnitude. */
  display_only?: boolean;
};

/** The served meta's `color` block. */
export type ColorMeta = {
  band_names: string[];
  bands: Record<string, BandConstants>;
  rgb_scheme: string[];
  default_asinh?: number;
  render_mode?: string;
};

/** The minimum a cube needs to be rendered. */
export type CubeLike = {
  key?: string;
  data: Float32Array;
  h: number;
  w: number;
  c: number;
  /** Channel names from `X-Cube-Bands` (the authority for this frame). */
  bands?: string[];
  /** Display-only contrast factor (`X-Cube-Display-Scale`). */
  displayScale?: number;
  /** JWST colour composite: channels 0..2 are R, G, B (`X-Cube-Direct-RGB`). */
  directRgb?: boolean;
};

export type PreparedMode = "gray" | "gray-log" | "lupton" | "direct-rgb" | "temp";

export type Prepared = {
  mode: PreparedMode;
  /** Converts the e⁻ sliders into the unit of `I`. */
  factor: number;
  /** Per-pixel linear intensity. */
  I: Float32Array;
  R?: Float32Array;
  G?: Float32Array;
  B?: Float32Array;
  hueR?: Float32Array;
  hueG?: Float32Array;
  hueB?: Float32Array;
  logLo?: number;
  logHi?: number;
  h: number;
  w: number;
  npx: number;
};

/** AB-flux normalisation: e⁻-over-stack → proportional AB flux density,
 *  anchored on the served zeropoint_ab_e_total (= BandConfig.sim_zeropoint_e). */
export function abFluxNorm(band: BandConstants): number {
  return 1.0 / Math.pow(10, 0.4 * band.zeropoint_ab_e_total);
}

/** Solar-balance factor (whitens a G2V SED on top of abFluxNorm). */
export function solarBalance(band: BandConstants): number {
  return 1.0 / Math.pow(10, -0.4 * (band.solar_ab_mag as number));
}

/** Blackbody f_ν (arbitrary norm) at wavelengths `lam` (μm) for temperature T. */
export function planckFnu(lam: ArrayLike<number>, T: number): Float64Array {
  const out = new Float64Array(lam.length);
  for (let i = 0; i < lam.length; i++) {
    const x = 14387.77 / (lam[i] * T);
    out[i] = Math.pow(1.0 / lam[i], 3) / Math.expm1(x);
  }
  return out;
}

/** CIE 1931 (x, y) chromaticity of a blackbody at T (K), clamped to the locus fit. */
export function planckianXY(T: number): [number, number] {
  T = Math.min(Math.max(T, EYE_T_MIN), EYE_T_MAX);
  const u = 1e3 / T;
  let x: number;
  if (T <= 4000.0) {
    x = -0.2661239 * u ** 3 - 0.2343589 * u ** 2 + 0.8776956 * u + 0.179910;
  } else {
    x = -3.0258469 * u ** 3 + 2.1070379 * u ** 2 + 0.2226347 * u + 0.240390;
  }
  let y: number;
  if (T <= 2222.0) {
    y = -1.1063814 * x ** 3 - 1.34811020 * x ** 2 + 2.18555832 * x - 0.20219683;
  } else if (T <= 4000.0) {
    y = -0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867;
  } else {
    y = 3.0817580 * x ** 3 - 5.87338670 * x ** 2 + 3.75112997 * x - 0.37001483;
  }
  return [x, y];
}

/** (x, y) chromaticity → linear sRGB (D65), normalised so the max channel = 1. */
export function xyToLinearSrgb(x: number, y: number): [number, number, number] {
  const Y = 1.0;
  const X = x / y;
  const Z = (1.0 - x - y) / y;
  let r = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z;
  let g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z;
  let b = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z;
  r = Math.max(r, 0); g = Math.max(g, 0); b = Math.max(b, 0);
  const peak = Math.max(r, g, b, 1e-12);
  return [r / peak, g / peak, b / peak];
}

/** Linear-light → sRGB-encoded (standard piecewise transfer). */
export function srgbGamma(c: number): number {
  c = Math.min(Math.max(c, 0), 1);
  return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1.0 / 2.4) - 0.055;
}

/** Log-spaced colour-temperature grid over the locus fit's validity. */
export function eyeTGrid(n: number): Float64Array {
  const out = new Float64Array(n);
  const lo = Math.log(EYE_T_MIN), hi = Math.log(EYE_T_MAX);
  for (let i = 0; i < n; i++) out[i] = Math.exp(lo + (hi - lo) * (i / (n - 1)));
  return out;
}

/** asinh contrast transfer: arcsinh(I·G / Kc) / arcsinh(Wref/Kc), clipped. */
export function asinhTransfer(I: number, gain: number, Kc: number, norm: number): number {
  const t = Math.asinh((I * gain) / Kc) / norm;
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

/** The channel names of a cube: its `X-Cube-Bands` when they match its
 *  channel count, else the collection's band order. */
export function cubeBandNames(rec: Pick<CubeLike, "bands" | "c">, colorMeta: Pick<ColorMeta, "band_names">): string[] {
  return Array.isArray(rec.bands) && rec.bands.length === rec.c
    ? rec.bands : colorMeta.band_names.slice(0, rec.c);
}

/** Prepare a cube for display in colour mode `color` (a band name, "lupton",
 *  "temp"). `rgbScheme` overrides the Lupton R, G, B bands (the "rgb" custom
 *  mapping); by default it is the served `rgb_scheme` ([H_E, J_E, VIS]). */
export function prepareCore(rec: CubeLike, colorMeta: ColorMeta, color: string, rgbScheme?: string[]): Prepared {
  // A collection can mix cameras: e.g. Euclid's four calibrated channels
  // beside a one-filter JWST image. The response header, not the collection
  // default, is the authority for a particular frame's channel order.
  const names = cubeBandNames(rec, colorMeta);
  const bandOf = (n: string) => colorMeta.bands[n];
  const npx = rec.h * rec.w;
  // A heterogeneous archive panel may ask for a display-only contrast scale.
  // It never modifies the served Float32/FITS science values.
  const displayScale = Number.isFinite(rec.displayScale) && (rec.displayScale as number) > 0
    ? rec.displayScale as number : 1.0;
  const data = rec.data;
  const c = rec.c;
  const at = (p: number, k: number) => data[p * c + k] * displayScale; // band k at pixel p
  const calibrated = names.every((n) => !!bandOf(n));
  const scheme = rgbScheme && rgbScheme.length === 3 ? rgbScheme : colorMeta.rgb_scheme;
  const canLupton = calibrated && scheme.every((n) => names.includes(n));
  const canTemp = calibrated && names.length >= 2;

  let prepared: Omit<Prepared, "h" | "w" | "npx">;
  if (rec.directRgb && rec.c >= 3) {
    const R = new Float32Array(npx), G = new Float32Array(npx), B = new Float32Array(npx);
    const I = new Float32Array(npx);
    for (let p = 0; p < npx; p++) {
      R[p] = Math.max(at(p, 0), 0); G[p] = Math.max(at(p, 1), 0); B[p] = Math.max(at(p, 2), 0);
      I[p] = (R[p] + G[p] + B[p]) / 3.0;
    }
    prepared = { mode: "direct-rgb", R, G, B, I, factor: 1.0 };
  } else if ((color === "lupton" && canLupton) || (color === "temp" && canTemp)) {
    const useSolar = color === "lupton";
    const calib = names.map((n) => {
      let f = abFluxNorm(bandOf(n));
      if (useSolar) f *= solarBalance(bandOf(n));
      return f;
    });
    if (color === "lupton") {
      const sel = scheme.map((n) => names.indexOf(n)); // [H_E, J_E, VIS]
      const R = new Float32Array(npx), G = new Float32Array(npx), B = new Float32Array(npx);
      const I = new Float32Array(npx);
      for (let p = 0; p < npx; p++) {
        const r = at(p, sel[0]) * calib[sel[0]];
        const g = at(p, sel[1]) * calib[sel[1]];
        const b = at(p, sel[2]) * calib[sel[2]];
        R[p] = r; G[p] = g; B[p] = b; I[p] = (r + g + b) / 3.0;
      }
      const visB = bandOf("VIS");
      prepared = { mode: "lupton", R, G, B, I, factor: abFluxNorm(visB) * solarBalance(visB) };
    } else {
      const lam = names.map((n) => bandOf(n).pivot_um as number);
      const ts = eyeTGrid(96);
      const pvecs: number[][] = [];
      for (const T of ts) {
        const p = planckFnu(lam, T);
        let nrm = 0; for (const v of p) nrm += v * v; nrm = Math.sqrt(nrm);
        pvecs.push(Array.from(p, (v) => v / nrm));
      }
      const hueR = new Float32Array(npx), hueG = new Float32Array(npx), hueB = new Float32Array(npx);
      const I = new Float32Array(npx);
      const cal = new Float64Array(names.length);
      for (let p = 0; p < npx; p++) {
        let sum = 0;
        for (let k = 0; k < names.length; k++) { cal[k] = at(p, k) * calib[k]; sum += cal[k]; }
        I[p] = Math.max(sum / names.length, 0);
        let bestScore = -Infinity, bestT = 6500.0;
        for (let ti = 0; ti < ts.length; ti++) {
          const pv = pvecs[ti];
          let s = 0; for (let k = 0; k < names.length; k++) s += cal[k] * pv[k];
          if (s > bestScore) { bestScore = s; bestT = ts[ti]; }
        }
        const T = bestScore > 0 ? bestT : 6500.0;
        const [x, y] = planckianXY(T);
        const [hr, hg, hb] = xyToLinearSrgb(x, y);
        hueR[p] = hr; hueG[p] = hg; hueB[p] = hb;
      }
      const visB = bandOf("VIS");
      // JWST's F### approximation carries no invented AB zero point; retain
      // its display-normalised native scale instead of borrowing Euclid VIS.
      const factor = names.some((n) => bandOf(n).display_only) ? 1.0 : abFluxNorm(visB);
      prepared = { mode: "temp", hueR, hueG, hueB, I, factor };
    }
  } else {
    // A colour chosen for a different camera is never applied to this image.
    // For a one-filter JWST frame this resolves to its sole native channel.
    const k = Math.max(0, names.indexOf(color));
    const I = new Float32Array(npx);
    if (colorMeta.render_mode === "log") {
      let hi = -Infinity;
      for (let p = 0; p < npx; p++) {
        I[p] = Math.log10(Math.max(at(p, k), 1e-12));
        hi = Math.max(hi, I[p]);
      }
      prepared = { mode: "gray-log", I, factor: 1.0, logLo: Math.max(hi - 6.0, -12.0), logHi: hi };
    } else {
      for (let p = 0; p < npx; p++) I[p] = at(p, k);
      prepared = { mode: "gray", I, factor: 1.0 };
    }
  }
  return { ...prepared, h: rec.h, w: rec.w, npx };
}

/** The locked absolute asinh transfer of a prepared frame → ImageData. */
export function transferCore(prep: Prepared, knee: number, gain: number, K0: number): ImageData {
  const { h, w, npx, I, factor } = prep;
  // Kc = knee·factor; white reference Wref = 30·K0·factor (e⁻). norm cancels
  // factor → asinh(30·K0/knee); at knee=K0, gain=1 this matches eye_rgb.
  const Kc = Math.max(knee * factor, 1e-30);
  const norm = Math.max(Math.asinh((30.0 * K0 * factor) / Kc), 1e-6);
  const G = gain;
  const img = new ImageData(w, h);
  const out = img.data;

  if (prep.mode === "gray-log") {
    const lo = prep.logLo as number, span = Math.max((prep.logHi as number) - lo, 1e-6);
    for (let p = 0; p < npx; p++) {
      const t = Math.min(Math.max((I[p] - lo) / span, 0), 1);
      const v = Math.min(1, t * G) * 255;
      const o = p * 4; out[o] = v; out[o + 1] = v; out[o + 2] = v; out[o + 3] = 255;
    }
  } else if (prep.mode === "gray") {
    for (let p = 0; p < npx; p++) {
      const v = (asinhTransfer(I[p], G, Kc, norm) * 255) | 0;
      const o = p * 4; out[o] = v; out[o + 1] = v; out[o + 2] = v; out[o + 3] = 255;
    }
  } else if (prep.mode === "lupton" || prep.mode === "direct-rgb") {
    const R = prep.R as Float32Array, GG = prep.G as Float32Array, B = prep.B as Float32Array;
    for (let p = 0; p < npx; p++) {
      const t = asinhTransfer(I[p], G, Kc, norm);
      const rescale = I[p] > 1e-30 ? t / I[p] : 0;
      const o = p * 4;
      out[o] = Math.min(Math.max(R[p] * rescale, 0), 1) * 255;
      out[o + 1] = Math.min(Math.max(GG[p] * rescale, 0), 1) * 255;
      out[o + 2] = Math.min(Math.max(B[p] * rescale, 0), 1) * 255;
      out[o + 3] = 255;
    }
  } else { // temp
    const hueR = prep.hueR as Float32Array, hueG = prep.hueG as Float32Array, hueB = prep.hueB as Float32Array;
    for (let p = 0; p < npx; p++) {
      const lum = asinhTransfer(I[p], G, Kc, norm);
      const o = p * 4;
      out[o] = srgbGamma(hueR[p] * lum) * 255;
      out[o + 1] = srgbGamma(hueG[p] * lum) * 255;
      out[o + 2] = srgbGamma(hueB[p] * lum) * 255;
      out[o + 3] = 255;
    }
  }
  return img;
}

export type RenderOpts = { color?: string; knee?: number; gain?: number; K0?: number };

/** Render one N-band cube to ImageData with the viewer's exact default
 *  pipeline (the ensemble back-trace stamps use it for viewer-parity colour). */
export function renderCubeImageData(rec: CubeLike, colorMeta: ColorMeta, opts: RenderOpts = {}): ImageData {
  const prep = prepareCore(rec, colorMeta, opts.color || "VIS");
  const knee = opts.knee || 100;
  return transferCore(prep, knee, opts.gain != null ? opts.gain : 1.0, opts.K0 || knee);
}
