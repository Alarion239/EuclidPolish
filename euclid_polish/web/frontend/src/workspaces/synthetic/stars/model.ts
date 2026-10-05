/* Stars tab: pure series builders on the chart kit. The VIS density is a
   physical value on a log y axis; the colour panels are unit-area PDFs. */
import type { Band as PlotBand, Guide, Series } from "../../../charts/Plot";
import { C, categorical } from "../../../colors";
import type { ModelNoise, StarDensityKey, StarDensityParameter, StarDistribution } from "../api";
import { formatCount } from "../../../format";
import { extent } from "../../../ticks";
import { positiveOrNull } from "../chartKit";

/** The VIS magnitude panel, then the six Euclid colour panels. */
export const DENSITY_ORDER: StarDensityKey[] = ["vis", "vis_y", "vis_j", "vis_h", "y_j", "y_h", "j_h"];

/** The density curves, one legend key each (toggled together across the
 *  seven density panels). */
export const DENSITY_KEYS = {
  pointSources: "point sources",
  q1: "Q1 PHZ",
  /** The colour panels' Euclid curve: the four-band colours of the Gaia-matched fixed-field stars. */
  fourBand: "Euclid four-band",
  model: "model",
  synthetic: "generated",
  gaia: "Gaia G",
  gaiaFit: "Gaia fit",
} as const;

export const densityColor = {
  pointSources: () => categorical(1),
  q1: () => C.mean,
  // Its own hue: the colour panels' four-band curve is a separate legend toggle from the VIS Q1 PHZ curve.
  fourBand: () => categorical(7),
  model: () => categorical(3),
  synthetic: () => categorical(4),
  gaia: () => categorical(2),
  gaiaFit: () => categorical(2),
};

/** One panel's curves. On the VIS panel the Euclid curve is the Q1 PHZ counts; on a colour panel it
 *  is the matched fixed-field stars' four-band colours. The VIS panel also draws the native Gaia G_AB
 *  counts and their shared-slope fit. */
export function densitySeries(parameter: StarDensityParameter, trainingIncluded: boolean, panel: StarDensityKey = "vis"): Series[] {
  const out: Series[] = [];
  if (parameter.point_sources) {
    out.push({ x: parameter.x, y: positiveOrNull(parameter.point_sources), color: densityColor.pointSources(), width: 2.2,
      name: "Q1 point sources (VIS)", key: DENSITY_KEYS.pointSources });
  }
  if (panel === "vis" && parameter.gaia_x && parameter.gaia) {
    out.push({ x: parameter.gaia_x, y: positiveOrNull(parameter.gaia), color: densityColor.gaia(), width: 2, mode: "scatter",
      marker: "ring", name: "native Gaia G (AB) counts", key: DENSITY_KEYS.gaia });
    if (parameter.gaia_fit) {
      out.push({ x: parameter.gaia_x, y: positiveOrNull(parameter.gaia_fit), color: densityColor.gaiaFit(), width: 1.8, dash: [4, 3],
        name: "Gaia-intercept shared-slope fit", key: DENSITY_KEYS.gaiaFit });
    }
  }
  out.push(panel === "vis"
    ? { x: parameter.x, y: positiveOrNull(parameter.euclid), color: densityColor.q1(), width: 2.2, name: "Q1 PHZ (VIS)", key: DENSITY_KEYS.q1 }
    : { x: parameter.x, y: positiveOrNull(parameter.euclid), color: densityColor.fourBand(), width: 2.2, name: "Euclid four-band",
      key: DENSITY_KEYS.fourBand });
  out.push({ x: parameter.x, y: positiveOrNull(parameter.model), color: densityColor.model(), width: 2.2, dash: [6, 4],
    name: "Q1-normalized law / colour draw", key: DENSITY_KEYS.model });
  out.push({ x: parameter.x, y: positiveOrNull(parameter.synthetic), color: densityColor.synthetic(), width: 1.8,
    marker: "filled", dots: true, markerEvery: 2,
    name: `generated ${trainingIncluded ? "train + test + validation" : "test + validation"} stars`, key: DENSITY_KEYS.synthetic });
  return out;
}

/** Density y domain: the positive values, at most 5 decades below the peak. */
export function densityDomain(parameter: StarDensityParameter): [number, number] {
  const positive = [parameter.euclid, parameter.model, parameter.synthetic, parameter.point_sources ?? [], parameter.gaia ?? []].flat().filter((v) => Number.isFinite(v) && v > 0);
  const span = extent(positive);
  if (!span) return [1e-4, 1];
  const high = Math.ceil(Math.log10(span[1]));
  const low = Math.max(Math.floor(Math.log10(span[0])), high - 5);
  return [10 ** low, 10 ** Math.max(low + 1, high)];
}

/** The fitted-range guides of the VIS panel (the Q1 fit window). */
export function fitGuides(parameter: StarDensityParameter): Guide[] {
  return Object.entries(parameter.fit_ranges ?? {}).flatMap(([which, interval]): Guide[] => {
    if (!interval || which !== "q1") return [];
    const ends = interval.filter((v): v is number => v != null && Number.isFinite(v));
    if (ends.length !== 2) return [];
    return ends.map((v, i) => ({ axis: "x", v, color: C.muted, dash: [3, 4], width: 1, alpha: 0.55,
      label: i === 0 ? "Q1 fit" : undefined }));
  });
}

/** The VIS window the Q1 counts were fitted over (the Q1 fit guides), or null. */
export function trustedWindow(parameter: StarDensityParameter): [number, number] | null {
  const [lo, hi] = parameter.fit_ranges?.q1 ?? [];
  return lo != null && hi != null && Number.isFinite(lo) && Number.isFinite(hi) ? [lo, hi] : null;
}

type DensityComparison = NonNullable<StarDistribution["density_comparison"]>;

/** Stars per arcmin² in the generated catalogues, or null without a rendered area. */
export function generatedDensity(c: DensityComparison): number | null {
  const n = c.synthetic_star_count, area = c.synthetic_area_arcmin2;
  return n != null && area != null && area > 0 ? n / area : null;
}

/** Generated ÷ prior − 1 (a fraction), or null when either side is missing. */
export function densityDelta(c: DensityComparison): number | null {
  const generated = generatedDensity(c);
  return generated != null && c.model_density_arcmin2 > 0 ? generated / c.model_density_arcmin2 - 1 : null;
}

/* ─── Synthetic › Stars: the colour panels as normalised PDFs ──────────── */

/** `y` scaled to unit area over `x` (trapezoids on the bin centres), so the
 *  colour distributions compare in shape whatever their normalisation
 *  (Q1-matched stars, model draws and generated stars differ in area and
 *  count); null where the input is not finite; all-null without area. */
export function normalisedPdf(x: readonly number[], y: readonly (number | null | undefined)[]): (number | null)[] {
  let area = 0;
  for (let i = 1; i < x.length; i++) {
    const a = y[i - 1], b = y[i];
    if (a == null || b == null || !Number.isFinite(a) || !Number.isFinite(b)) continue;
    area += 0.5 * (a + b) * (x[i] - x[i - 1]);
  }
  return y.map((v) => (v == null || !Number.isFinite(v) || !(area > 0) ? null : v / area));
}

/** The legend names of the Stars tab (one per curve key; the colour panels
 *  share the model and generated keys with the VIS panel). */
export const STAR_LEGEND = {
  pointSources: "Q1 point sources",
  q1: "Q1 PHZ stars",
  fourBand: "Gaia-matched Q1 stars",
  synthetic: "generated stars",
  gaia: "Gaia G (AB) counts",
  gaiaFit: "Gaia-intercept fit",
} as const;

/** The model curve's legend name: the VIS law, and whether its colour draws
 *  carry Q1 measurement noise (forward noised) or are intrinsic. */
export function starModelLabel(noise?: ModelNoise | null): string {
  return noise?.applied ? "model (VIS law · colour draws with Q1 noise)" : "model (VIS law · intrinsic colour draws)";
}

/** The caption clause on the colour panels' model draws and generated stars. */
export function modelNoiseNote(noise?: ModelNoise | null): string {
  if (noise?.applied) {
    const window = noise.vis_window ? ` over the Q1 colour sample's VIS ${noise.vis_window[0].toFixed(1)}–${noise.vis_window[1].toFixed(1)}` : "";
    return `the model's colour draws and the generated stars are compared${window} with Q1 measurement noise `
      + `(flux errors borrowed from ${noise.donors != null ? `${formatCount(noise.donors)} ` : ""}Q1 stars at the same VIS)`;
  }
  return `the model's colour draws carry no measurement noise${noise?.detail ? ` (${noise.detail})` : ""}`;
}

/** One colour panel's curves as unit-area PDFs: the Gaia-matched Q1 stars'
 *  four-band colours, the model's colour draws and the generated stars. */
export function colourPdfSeries(parameter: StarDensityParameter, noise?: ModelNoise | null): Series[] {
  const x = parameter.x;
  return [
    { x, y: normalisedPdf(x, parameter.euclid), color: densityColor.fourBand(), width: 2.2,
      name: STAR_LEGEND.fourBand, key: DENSITY_KEYS.fourBand },
    { x, y: normalisedPdf(x, parameter.model), color: densityColor.model(), width: 2.2, dash: [6, 4],
      name: starModelLabel(noise), key: DENSITY_KEYS.model },
    { x, y: normalisedPdf(x, parameter.synthetic), color: densityColor.synthetic(), width: 1.8,
      marker: "filled", dots: true, markerEvery: 2, name: STAR_LEGEND.synthetic, key: DENSITY_KEYS.synthetic },
  ];
}

/** A PDF panel's y domain: 0 to 1.1 × the highest value. */
export function pdfDomain(series: readonly Series[]): [number, number] {
  const values = series.flatMap((s) => s.y).filter((v): v is number => v != null && Number.isFinite(v));
  const top = values.length ? Math.max(...values) : 1;
  return [0, top > 0 ? 1.1 * top : 1];
}

/** The trusted VIS fit window as a shaded band on the VIS panel. */
export function trustedBand(parameter: StarDensityParameter): PlotBand[] {
  const w = trustedWindow(parameter);
  return w ? [{ axis: "x", from: w[0], to: w[1], color: C.mean, alpha: 0.08, label: "trusted window" }] : [];
}
