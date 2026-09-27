/* Stars tab: pure series builders on the Realism chart kit. Densities are
   physical values on a log y axis; magnitudes are drawn bright-at-top as
   −mag with magnitudeTicks({invert}). */
import type { Guide, Series } from "../../../charts/Plot";
import { C, categorical } from "../../../colors";
import type { StarColorKey, StarDensityKey, StarDensityParameter, StarDistribution } from "../api";
import { extent } from "../../../ticks";
import { positiveOrNull } from "../chartKit";

export const COLOR_ORDER: StarColorKey[] = ["vis_y", "vis_j", "vis_h", "y_j", "y_h", "j_h"];
export const PROJECTION_ORDER = ["vis_y", "vis_j", "vis_h"] as const;
export const DENSITY_ORDER: StarDensityKey[] = ["vis", ...COLOR_ORDER];

/** The density curves, one legend key each (toggled together across the
 *  seven density panels). */
export const DENSITY_KEYS = {
  pointSources: "point sources",
  q1: "Q1 PHZ",
  gaia: "Gaia G_AB",
  gaiaFit: "Gaia fit",
  model: "model",
  synthetic: "generated",
} as const;

export const densityColor = {
  pointSources: () => categorical(1),
  q1: () => C.mean,
  gaia: () => C.comb,
  model: () => categorical(3),
  synthetic: () => categorical(4),
};

export function clamp(values: readonly number[], [lo, hi]: [number, number]): number[] {
  return values.map((v) => Math.max(lo, Math.min(hi, v)));
}

/** A negated magnitude list (bright at the top), clamped to `domain`. */
export const brightUp = (values: readonly number[], domain: [number, number]) => clamp(values, domain).map((v) => -v);

export function densitySeries(parameter: StarDensityParameter, trainingIncluded: boolean): Series[] {
  const gx = parameter.gaia_x ?? parameter.x;
  const out: Series[] = [];
  if (parameter.point_sources) {
    out.push({ x: parameter.x, y: positiveOrNull(parameter.point_sources), color: densityColor.pointSources(), width: 2.2,
      name: "Q1 point sources (VIS)", key: DENSITY_KEYS.pointSources });
  }
  out.push({ x: parameter.x, y: positiveOrNull(parameter.euclid), color: densityColor.q1(), width: 2.2, name: "Q1 PHZ (VIS)", key: DENSITY_KEYS.q1 });
  out.push({ x: gx, y: positiveOrNull(parameter.gaia), color: densityColor.gaia(), width: 2.2, name: "native Gaia G_AB", key: DENSITY_KEYS.gaia });
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
  const positive = [parameter.euclid, parameter.gaia, parameter.model, parameter.synthetic,
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

export function correlationSeries(distribution: StarDistribution, key: StarColorKey): Series[] {
  const item = distribution.colors[key];
  const x = clamp(distribution.bp_rp, distribution.x_domain);
  const y = clamp(item.values, item.y_domain);
  const fit = item.fit;
  return [
    ...(fit ? [
      { x: fit.x, y: fit.center, low: fit.two_sigma_low, high: fit.two_sigma_high, color: C.comb, fillAlpha: 0.07, alpha: 0, width: 0, name: "2σ intrinsic", key: "2σ" },
      { x: fit.x, y: fit.center, low: fit.one_sigma_low, high: fit.one_sigma_high, color: C.comb, fillAlpha: 0.16, alpha: 0, width: 0, name: "1σ intrinsic", key: "1σ" },
    ] : []),
    { x, y, color: C.mean, mode: "scatter", width: 0.7, alpha: 0.28, name: "catalogue stars", key: "stars" },
    fit
      ? { x: fit.x, y: fit.center, color: C.comb, width: 2.4, name: "fitted locus", key: "locus" }
      : { x: item.trend.x, y: item.trend.y, color: C.comb, width: 2.4, name: "trend", key: "locus" },
  ];
}

/** The Gaia colour–magnitude diagram (G bright-at-top). */
export function cmdSeries(distribution: StarDistribution): Series[] {
  const cmd = distribution.gaia_cmd;
  return [
    { x: clamp(cmd.unmatched.bp_rp, cmd.x_domain), y: brightUp(cmd.unmatched.g_mag, cmd.g_domain), color: C.muted,
      mode: "scatter", width: 0.45, alpha: 0.25, name: `unmatched · ${cmd.unmatched.bp_rp.length.toLocaleString("en")}`, key: "unmatched" },
    { x: clamp(cmd.matched.bp_rp, cmd.x_domain), y: brightUp(cmd.matched.g_mag, cmd.g_domain), color: C.comb,
      mode: "scatter", marker: "ring", width: 0.75, alpha: 0.62, name: `Euclid counterpart · ${cmd.matched.bp_rp.length.toLocaleString("en")}`, key: "matched" },
  ];
}

/** VIS vs one colour for the Gaia population projected into Euclid. */
export function projectionSeries(distribution: StarDistribution, key: (typeof PROJECTION_ORDER)[number]): Series[] {
  const p = distribution.euclid_projection!;
  const color = p.colors[key];
  const observed = p.euclid_observed[key];
  return [
    { x: clamp(p.unmatched.colors[key], color.x_domain), y: brightUp(p.unmatched.vis_mag, p.vis_domain), color: C.muted,
      mode: "scatter", width: 0.4, alpha: 0.22, name: "unmatched Gaia", key: "unmatched" },
    { x: clamp(p.matched.colors[key], color.x_domain), y: brightUp(p.matched.vis_mag, p.vis_domain), color: C.comb,
      mode: "scatter", marker: "ring", width: 0.7, alpha: 0.58, name: "Euclid counterpart", key: "matched" },
    { x: clamp(observed.color, color.x_domain), y: brightUp(observed.vis_mag, p.vis_domain), color: C.mean,
      mode: "scatter", width: 0.55, alpha: 0.34, name: "measured fixed-Q1 star", key: "measured" },
  ];
}
