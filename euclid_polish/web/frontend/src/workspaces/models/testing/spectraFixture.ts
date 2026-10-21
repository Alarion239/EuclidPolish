/* Fixtures of Models › Spectra (GET /ensemble/universal/spectra.json and
 * /ensemble/universal/spectra/<name>.json), imported by tests only:
 * spectra.test.ts and spectraTab.test.tsx. Seven universal models:
 * - 218 (no isotropy rule, 2M steps) and 227 (no rule, the best ∫PSNR):
 *   current spectra with a pixel-axis grid (axis / diagonal 2.0 and 1.6 at
 *   2–3 px, 1.5 and 1.25 at 3–6 px, in every band);
 * - 230: no spectra yet (status missing, in `missing`);
 * - 233 (isotropy 0.05) and 238 (isotropy 0.5, the newest): current and
 *   round (axis / diagonal ≤ 1.1), except 238 at 2–3 px in H (1.3);
 * - 235 (isotropy 0.05): stale spectra (drawn when picked, in `missing`);
 * - 239: current, but every spectrum null, with the reason.
 * Radial rows (63 rings of a 128-px window, period 128 / ring): the truth's
 * power is (band + 1)·10⁻³·period²; a model's transfer (amplitude)
 * t = 1 − a·(2/p), its power truth × t², its coherence 1 − c·(2/p)^0.7 (so
 * it falls to ½ near 4.6 px for 218, 3.2 px for 227, 5 px for 233, 3.9 px
 * for 235 and never for 238); the finest ring is null in every model row and
 * 218's coherence also misses the coarsest. Angular rows (36 bins of 5°):
 * 1 + g·cos⁸(2θ) (g = 2 for 218, 1.5 for 227, 0 for the round ones); the
 * 12–21 px band's bins 1, 4 and 8 are empty (null); no 21–42 px profile.
 * The map (64 × 64, k −32…31): 1 + g on the kx = 0 / ky = 0 axes, else
 * 1 − 0.05·g; null at the DC bin and beyond |k| = 32. The summary's note
 * lists both models without current spectra. */
import type { SpectraModel, SpectraPayload, SpectraRecord } from "../spectraApi";

export const BANDS = ["VIS", "Y_E", "J_E", "H_E"];
export const PERIODS = Array.from({ length: 63 }, (_, i) => +(128 / (i + 1)).toFixed(3));
export const ANGLES = Array.from({ length: 36 }, (_, i) => 2.5 + 5 * i);
export const K = Array.from({ length: 64 }, (_, i) => i - 32);
export const PERIOD_BANDS: [number, number][] = [[2, 3], [3, 6], [6, 12], [12, 21], [21, 42]];
export const EMPTY_BINS = [1, 4, 8];
export const REASON = "the test fields (64 px) are smaller than the 128-px windows";

const round = (v: number) => +v.toPrecision(6);
export const truthPower = (b: number, p: number) => round((b + 1) * 1e-3 * p * p);
export const transferAt = (a: number, p: number) => round(1 - a * (2 / p));
export const coherenceAt = (c: number, p: number) => round(1 - c * (2 / p) ** 0.7);
export const angularAt = (g: number, deg: number) => round(1 + g * Math.cos((2 * deg * Math.PI) / 180) ** 8);

type Spec = {
  n: number; status: "current" | "stale" | "missing"; psnr: number; steps: number; iso: number;
  a?: number; c?: number; g?: number; grid?: [number, number]; reason?: string;
};

/** The models: `a` transfer loss, `c` coherence loss, `g` the grid's angular
 *  and map excess, `grid` the axis values at 2–3 and 3–6 px (diagonal 1). */
export const SPECS: Spec[] = [
  { n: 218, status: "current", psnr: 62.1543, steps: 2_000_000, iso: 0, a: 0.3, c: 0.9, g: 2, grid: [2.0, 1.5] },
  { n: 227, status: "current", psnr: 62.8106, steps: 15_500_000, iso: 0, a: 0.25, c: 0.7, g: 1.5, grid: [1.6, 1.25] },
  { n: 230, status: "missing", psnr: 60.8259, steps: 200_000, iso: 0 },
  { n: 233, status: "current", psnr: 60.7288, steps: 200_000, iso: 0.05, a: 0.35, c: 0.95, g: 0 },
  { n: 235, status: "stale", psnr: 60.8765, steps: 200_000, iso: 0.05, a: 0.33, c: 0.8, g: 0 },
  { n: 238, status: "current", psnr: 62.1267, steps: 1_900_000, iso: 0.5, a: 0.28, c: 0.45, g: 0 },
  { n: 239, status: "current", psnr: 60.1, steps: 200_000, iso: 0.5, reason: REASON },
];

const lastNull = (row: number[]) => row.map((v, i) => (i === row.length - 1 ? null : v));

function axisDiag(s: Spec): SpectraModel["axis_diag"] {
  const [g23, g36] = s.grid ?? [1.02, 1.02];
  const round23 = s.n === 233 ? 1.1 : g23;
  const pairs = (axis: number, b: number, key: string): [number | null, number | null] | null => {
    if (s.n === 233 && key === "21-42" && b === 0) return [null, null];
    if (s.n === 238 && key === "2-3" && b === 3) return [1.3, 1.0];
    return [axis, 1.0];
  };
  const axes: Record<string, number> = { "2-3": round23, "3-6": g36, "6-12": s.grid ? 1.1 : 1.02, "12-21": 1.0, "21-42": 1.0 };
  return Object.fromEntries(Object.entries(axes).map(([key, axis]) => [key, BANDS.map((_, b) => pairs(axis, b, key))]));
}

function angular(s: Spec): SpectraModel["angular"] {
  const row = (key: string) => ANGLES.map((deg, i) => (key === "12-21" && EMPTY_BINS.includes(i) ? null : angularAt(s.g ?? 0, deg)));
  return Object.fromEntries(["2-3", "3-6", "6-12", "12-21"].map((key) => [key, BANDS.map(() => row(key))]));
}

/** One summary row. */
export function row(s: Spec): SpectraModel {
  const withData = s.a != null;
  return {
    name: `member_${s.n}`, number: s.n, label: `${s.n}·psnr`, status: s.status,
    computed_at: s.status === "missing" ? null : `2026-10-21T13:${String(s.n % 60).padStart(2, "0")}:00+00:00`,
    n_fields: s.status === "missing" ? null : 100, mean_psnr: s.psnr, steps: s.steps, served_step: s.status === "missing" ? null : s.steps,
    depth: 16, knees: [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], isotropy_penalty: s.iso, reason: s.reason ?? null,
    radial: withData ? {
      power_sr: BANDS.map((_, b) => lastNull(PERIODS.map((p) => round(truthPower(b, p) * transferAt(s.a!, p) ** 2)))),
      coherence: BANDS.map(() => lastNull(PERIODS.map((p) => coherenceAt(s.c!, p))).map((v, i) => (s.n === 218 && i === 0 ? null : v))),
      transfer: BANDS.map(() => lastNull(PERIODS.map((p) => transferAt(s.a!, p)))),
    } : null,
    axis_diag: withData ? axisDiag(s) : null,
    angular: withData ? angular(s) : null,
  };
}

export const payload = (patch: Partial<SpectraPayload> = {}): SpectraPayload => ({
  schema: 1, records_local: true, bands: [...BANDS], period_bands: PERIOD_BANDS.map(([lo, hi]): [number, number] => [lo, hi]),
  angle_deg: [...ANGLES], period_px: [...PERIODS],
  truth_power: BANDS.map((_, b) => PERIODS.map((p) => truthPower(b, p))),
  pixel_scale_arcsec: 0.05,
  radial: { window_px: 128, stretch_knee_e: 1 },
  direction: { window_px: 64, long_window_px: 128, stretch_knee_e: 0.01 },
  models: SPECS.map(row),
  missing: ["member_230", "member_235"],
  notes: ["2 model(s) without spectra: member_230, member_235 — compute them (the score job with spectra)"],
  ...patch,
});

/** The map's values in every band: `g` on the axes. */
export function mapValues(g: number): (number | null)[][][] {
  return BANDS.map(() => K.map((ky) => K.map((kx) => {
    const r = Math.hypot(kx, ky);
    if (r === 0 || r > 32) return null;
    return round(kx === 0 || ky === 0 ? 1 + g : 1 - 0.05 * g);
  })));
}

/** One model's record (GET /ensemble/universal/spectra/<name>.json). */
export function record(n: number): SpectraRecord {
  const s = SPECS.find((x) => x.n === n)!;
  return { ...row(s), bands: [...BANDS], map: { k: [...K], values: mapValues(s.g ?? 0), period_lines_px: [21, 3], n_windows: 3600 } };
}
