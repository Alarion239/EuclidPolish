/* Slider scale mapping (linear | log) — pure, unit-tested. A log slider runs
   on an internal 0…LOG_STEPS track and maps to the value through log10; a
   range log cannot map falls back to linear (see effectiveScale). */

export const LOG_STEPS = 1000;
export type SliderScale = "linear" | "log";

export type ScaleOpts = { min: number; max: number; scale?: SliderScale; step?: number };

/** Round to `sig` significant digits (log sliders snap to readable values). */
export function roundSig(v: number, sig = 3): number {
  if (!Number.isFinite(v) || v === 0) return v;
  const p = Math.pow(10, sig - 1 - Math.floor(Math.log10(Math.abs(v))));
  return Math.round(v * p) / p;
}

/** Decimals implied by a step (0.25 → 2, 1e-3 → 3, 5 → 0). */
function stepDecimals(step: number): number {
  if (!Number.isFinite(step) || step <= 0) return 0;
  const s = step.toString();
  if (s.includes("e-")) return Number(s.split("e-")[1]);
  const dot = s.indexOf(".");
  return dot < 0 ? 0 : s.length - dot - 1;
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

const warned = new Set<string>();

/** The scale actually used: "log" needs finite min, max > 0 with min ≠ max.
 *  Any other log range degrades to "linear" with one console warning per
 *  range (a bad range from data must not crash the page during render). */
export function effectiveScale(o: ScaleOpts): SliderScale {
  if (o.scale !== "log") return "linear";
  const { min, max } = o;
  if (Number.isFinite(min) && Number.isFinite(max) && min > 0 && max > 0 && min !== max) return "log";
  const key = `${min}:${max}`;
  if (!warned.has(key)) {
    warned.add(key);
    console.warn(`Slider: a log scale needs min > 0, max > 0 and min ≠ max (got ${min}…${max}); using linear.`);
  }
  return "linear";
}

/** Value → slider position (the value itself on a linear scale). */
export function toSliderPos(value: number, o: ScaleOpts): number {
  const { min, max } = o;
  if (effectiveScale(o) !== "log") return clamp(value, Math.min(min, max), Math.max(min, max));
  const v = clamp(value > 0 ? value : min, min, max);
  const f = (Math.log10(v) - Math.log10(min)) / (Math.log10(max) - Math.log10(min));
  return Math.round(clamp(f, 0, 1) * LOG_STEPS);
}

/** Slider position → value: snapped to `step` (linear) or to 3 significant
 *  digits (log), and clamped to [min, max]. */
export function fromSliderPos(pos: number, o: ScaleOpts): number {
  const { min, max } = o;
  if (effectiveScale(o) !== "log") {
    const step = o.step && o.step > 0 ? o.step : null;
    let v = step ? min + Math.round((pos - min) / step) * step : pos;
    if (step) v = Number(v.toFixed(Math.max(stepDecimals(step), stepDecimals(min))));
    return clamp(v, Math.min(min, max), Math.max(min, max));
  }
  const f = clamp(pos / LOG_STEPS, 0, 1);
  const v = Math.pow(10, Math.log10(min) + f * (Math.log10(max) - Math.log10(min)));
  return clamp(roundSig(v, 3), min, max);
}
