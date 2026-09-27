/* Stars tab: pure series builders on the Realism chart kit. Densities are
   physical values on a log y axis. */
import type { Guide, Series } from "../../../charts/Plot";
import { C, categorical } from "../../../colors";
import type { StarDensityKey, StarDensityParameter, StarDistribution } from "../api";
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
  gaia: "Gaia G_AB",
  gaiaFit: "Gaia fit",
  model: "model",
  synthetic: "generated",
} as const;

export const densityColor = {
  pointSources: () => categorical(1),
  q1: () => C.mean,
  // Its own hue: the colour panels' four-band curve is a separate legend toggle from the VIS Q1 PHZ curve.
  fourBand: () => categorical(7),
  gaia: () => C.comb,
  model: () => categorical(3),
  synthetic: () => categorical(4),
};

/** One panel's curves. The native Gaia G_AB counts and their shared-slope fit exist only on the VIS
 *  panel: they set the slope of the fitted magnitude law (Q1 PHZ sets its level). On a colour panel
 *  the Euclid curve is the matched fixed-field stars' four-band colours, not Q1 PHZ. */
export function densitySeries(parameter: StarDensityParameter, trainingIncluded: boolean, panel: StarDensityKey = "vis"): Series[] {
  const gx = parameter.gaia_x ?? parameter.x;
  const out: Series[] = [];
  if (parameter.point_sources) {
    out.push({ x: parameter.x, y: positiveOrNull(parameter.point_sources), color: densityColor.pointSources(), width: 2.2,
      name: "Q1 point sources (VIS)", key: DENSITY_KEYS.pointSources });
  }
  out.push(panel === "vis"
    ? { x: parameter.x, y: positiveOrNull(parameter.euclid), color: densityColor.q1(), width: 2.2, name: "Q1 PHZ (VIS)", key: DENSITY_KEYS.q1 }
    : { x: parameter.x, y: positiveOrNull(parameter.euclid), color: densityColor.fourBand(), width: 2.2, name: "Euclid four-band",
      key: DENSITY_KEYS.fourBand });
  if (parameter.gaia) {
    out.push({ x: gx, y: positiveOrNull(parameter.gaia), color: densityColor.gaia(), width: 2.2, name: "native Gaia G_AB", key: DENSITY_KEYS.gaia });
  }
  if (parameter.gaia_fit) {
    out.push({ x: gx, y: positiveOrNull(parameter.gaia_fit), color: densityColor.gaia(), width: 2.2, dash: [4, 3],
      name: "Gaia shared-slope fit", key: DENSITY_KEYS.gaiaFit });
  }
  out.push({ x: parameter.x, y: positiveOrNull(parameter.model), color: densityColor.model(), width: 2.2, dash: [6, 4],
    name: "Q1-normalized law / colour draw", key: DENSITY_KEYS.model });
  out.push({ x: parameter.x, y: positiveOrNull(parameter.synthetic), color: densityColor.synthetic(), width: 1.8,
    marker: "filled", dots: true, markerEvery: 2,
    name: `generated ${trainingIncluded ? "train + test + validation" : "test + validation"} stars`, key: DENSITY_KEYS.synthetic });
  return out;
}

/** Density y domain: the positive values, at most 5 decades below the peak. */
export function densityDomain(parameter: StarDensityParameter): [number, number] {
  const positive = [parameter.euclid, parameter.gaia ?? [], parameter.model, parameter.synthetic,
    parameter.gaia_fit ?? [], parameter.point_sources ?? []].flat().filter((v) => Number.isFinite(v) && v > 0);
  const span = extent(positive);
  if (!span) return [1e-4, 1];
  const high = Math.ceil(Math.log10(span[1]));
  const low = Math.max(Math.floor(Math.log10(span[0])), high - 5);
  return [10 ** low, 10 ** Math.max(low + 1, high)];
}

/** The fitted-range guides of the VIS panel (Q1 and Gaia fit windows). */
export function fitGuides(parameter: StarDensityParameter): Guide[] {
  return Object.entries(parameter.fit_ranges ?? {}).flatMap(([which, interval]): Guide[] => {
    if (!interval) return [];
    const ends = interval.filter((v): v is number => v != null && Number.isFinite(v));
    if (ends.length !== 2) return [];
    return ends.map((v, i) => ({ axis: "x", v, color: C.muted, dash: [3, 4], width: 1, alpha: 0.55,
      label: i === 0 ? `${which === "q1" ? "Q1" : "Gaia"} fit` : undefined }));
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
