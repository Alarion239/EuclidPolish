/* Galaxies tab: pure series builders (on the Realism chart kit). Densities are
   drawn on a log y axis in physical units; x axes whose parameter is in log10
   coordinates are converted to physical values on a log x axis, so the
   tooltips and ticks read physical values ("… (log scale)"). */
import type { Band as PlotBand, Guide, Series } from "../../../charts/Plot";
import { extent } from "../../../ticks";
import type {
  BrightnessCurve, GalaxyCandidate, JointMaps, Parameter, RadiusCurve,
} from "../api";
import {
  colorIndexColor, contourSeries, fromLog10, isLog10Label, jointColor, logDomain, physicalLogAxisLabel,
  positiveOrNull, surveyColor, type SourceKey,
} from "../chartKit";

export const PARAMETER_ORDER = ["magnitude", "radius", "color_vis_y", "color_y_j", "color_j_h"] as const;
export const MARGINAL_ORDER: SourceKey[] = ["euclid", "synthetic", "fit"];
export const USEFUL_BRIGHTNESS_KEYS = new Set(["q1_vis_f2", "synthetic_vis_2fwhm", "generator_vis_f2"]);
export const USEFUL_RADIUS_KEYS = new Set(["euclid_sersic_re", "synthetic_requested_re", "synthetic_clean_half_light", "fit_re"]);
export const USEFUL_SHAPE_KEYS = new Set(["euclid_sersic_re_shape", "fit_re_q1_weighted_shape", "fit_re_full_generation_shape"]);
export const CONTOUR_LEVELS = [0.1, 0.5, 0.8, 0.95, 0.99, 0.995, 0.999];

export const SURVEY_GROUP: Record<BrightnessCurve["survey"], { title: string; sub: string }> = {
  euclid: { title: "Euclid MER", sub: "VIS · solid measurements" },
  synthetic: { title: "Generated fields", sub: "selected source catalogues · point markers" },
  cosmos: { title: "COSMOS2025", sub: "HST/ACS F814W · long dashes" },
  fit: { title: "Q1 curve fits", sub: "VIS · short-dashed local fits" },
  generation: { title: "Generation law", sub: "VIS · three-segment bright bridge/main/flat law" },
};

type XAxis = { x: (xs: number[]) => number[]; scale: "linear" | "log"; label: string };

/** A parameter's x axis: physical + log scale when its label is log10. */
export function xAxisOf(parameter: Parameter): XAxis {
  if (isLog10Label(parameter.x_label)) {
    return { x: fromLog10, scale: "log", label: physicalLogAxisLabel(parameter.x_label) };
  }
  return { x: (xs) => xs, scale: "linear", label: parameter.x_label };
}

/** x domain: `parameter.x_domain` when given, else the data's extent. */
export function xDomainOf(parameter: Parameter, xs: number[]): [number, number] {
  const axis = xAxisOf(parameter);
  if (parameter.x_domain) return axis.scale === "log" ? [10 ** parameter.x_domain[0], 10 ** parameter.x_domain[1]] : parameter.x_domain;
  return extent(axis.x(xs)) ?? [0, 1];
}

const markerEvery = (n: number) => Math.max(1, Math.ceil(n / 18));

/** The three-source marginal (colours): Euclid rings, generated dots, fit line. */
export function densitySeries(parameter: Parameter): Series[] {
  const axis = xAxisOf(parameter);
  return MARGINAL_ORDER.flatMap((key) => {
    const curve = parameter.series[key];
    if (!curve) return [];
    return [{
      x: axis.x(curve.x), y: positiveOrNull(curve.density), color: surveyColor(key),
      width: key === "fit" ? 2.7 : 1.8, marker: key === "euclid" ? "ring" as const : key === "synthetic" ? "filled" as const : undefined,
      dots: key === "euclid" || key === "synthetic", markerEvery: markerEvery(curve.x.length),
      name: key === "euclid" ? "Q1" : key === "synthetic" ? "generated" : "model",
    }];
  });
}

export function brightnessEntries(parameter: Parameter): [string, BrightnessCurve][] {
  return Object.entries(parameter.photometry_series ?? {}).filter(([k]) => USEFUL_BRIGHTNESS_KEYS.has(k));
}

export function brightnessSeries(entries: [string, BrightnessCurve][]): Series[] {
  return entries.map(([key, curve]) => ({
    x: curve.x, y: positiveOrNull(curve.density), color: surveyColor(curve.survey),
    width: curve.survey === "generation" ? 3.2 : curve.survey === "fit" ? 2.7 : 2.0,
    dash: curve.survey === "cosmos" ? [7, 4] : curve.survey === "fit" ? [3, 3] : undefined,
    marker: curve.survey === "synthetic" ? "filled" : undefined, dots: curve.survey === "synthetic",
    markerEvery: markerEvery(curve.x.length), name: curve.label, key,
  }));
}

/** The Q1 support overlays of the brightness plot (trust boundary, turnover,
 *  beyond the MER nσ range). */
export function brightnessOverlays(entries: [string, BrightnessCurve][], xDomain: [number, number], yDomain: [number, number]) {
  const q1 = entries.find(([k]) => k === "q1_vis_f2")?.[1];
  const gen = entries.find(([k]) => k === "generator_vis_f2")?.[1];
  const trust = q1?.trust_boundary ?? gen?.trust_boundary;
  const peak = q1?.observed_density_cap_arcmin2_mag ?? gen?.observed_density_cap_arcmin2_mag;
  const peakMag = q1?.observed_density_cap_magnitude ?? gen?.observed_density_cap_magnitude;
  const clamp = (v: number) => Math.max(xDomain[0], Math.min(xDomain[1], v));
  const guides: Guide[] = [];
  const bands: PlotBand[] = [];
  const q1Color = surveyColor("euclid");
  if (trust && trust.magnitude >= xDomain[0] && trust.magnitude <= xDomain[1]) {
    guides.push({ axis: "x", v: trust.magnitude, color: q1Color, width: 2.2 });
  }
  if (peak && peak > 0 && peak >= yDomain[0] && peak <= yDomain[1]) {
    guides.push({ axis: "y", v: peak, color: q1Color, dash: [6, 3], width: 1.2, alpha: 0.85,
      label: `Q1 observed max ${peak.toFixed(1)}`, labelSide: "before" });
  }
  if (trust) {
    const lower = clamp(trust.lower_magnitude), upper = clamp(trust.upper_magnitude);
    const turnover = clamp(peakMag ?? trust.magnitude);
    if (turnover > xDomain[0]) bands.push({ axis: "x", from: xDomain[0], to: turnover, color: q1Color, alpha: 0.07, label: "Q1 count support to turnover" });
    if (upper > lower) bands.push({ axis: "x", from: lower, to: upper, color: q1Color, alpha: 0.16 });
    if (upper < xDomain[1]) bands.push({ axis: "x", from: upper, to: xDomain[1], color: surveyColor("fit"), alpha: 0.05, hatch: true, label: `beyond the MER ${trust.snr}σ range` });
  } else if (peakMag != null) {
    const turnover = clamp(peakMag);
    if (turnover > xDomain[0]) bands.push({ axis: "x", from: xDomain[0], to: turnover, color: q1Color, alpha: 0.07, label: "Q1 count-supported" });
    if (turnover < xDomain[1]) bands.push({ axis: "x", from: turnover, to: xDomain[1], color: surveyColor("fit"), alpha: 0.05, hatch: true, label: "beyond the Q1 turnover" });
  }
  return { guides, bands, trust, peak, peakMag, generationCap: gen?.generation_density_cap_arcmin2_mag,
    cumulativeToBoundary: q1?.observed_cumulative_density_to_boundary_arcmin2 ?? gen?.observed_cumulative_density_to_boundary_arcmin2,
    cumulativeAll: q1?.observed_cumulative_density_all_queried_bins_arcmin2 ?? gen?.observed_cumulative_density_all_queried_bins_arcmin2 };
}

/** The generation law's one-line disclosure (three-segment bright bridge,
 *  main slope, flat faint plateau). */
export function brightnessDisclosure(curve: BrightnessCurve): string {
  if (curve.fit_interval) {
    if (curve.generation_bright_join_magnitudes?.length) {
      return `fixed joins VIS ${curve.generation_bright_join_magnitudes.map((v) => v.toFixed(2)).join(" / ")}`
        + ` · bridge slopes ${curve.generation_bright_slopes?.map((v) => v.toFixed(3)).join(" / ") ?? "—"}`
        + `; main ${curve.generation_main_slope?.toFixed(3) ?? "—"} dex mag⁻¹`
        + ` · flat from VIS ${curve.generation_break_magnitude?.toFixed(2) ?? "—"} to ${curve.generation_interval?.[1].toFixed(0) ?? "—"}`;
    }
    return `fit ${curve.fit_interval[0].toFixed(2)}–${curve.fit_interval[1].toFixed(2)}`
      + ` · law ${curve.sampling_interval?.[0].toFixed(0) ?? "—"}–${curve.sampling_interval?.[1].toFixed(0) ?? "—"}`;
  }
  return `${Math.round(curve.weighted_count).toLocaleString("en")} weighted objects`;
}

const radiusColor = (key: string, curve: RadiusCurve) =>
  key === "fit_re_full_generation_shape" ? surveyColor("generation") : surveyColor(curve.source);
const radiusDash = (key: string, curve: RadiusCurve): number[] | undefined =>
  key === "fit_re_full_generation_shape" ? [4, 4]
    : key === "synthetic_clean_half_light" ? [6, 3]
      : curve.source === "cosmos" ? [7, 4] : undefined;

export const normalizationOf = (curve?: RadiusCurve) => curve?.normalization ?? "surface_density";

export function radiusEntries(parameter: Parameter, keys: Set<string>): [string, RadiusCurve][] {
  return Object.entries(parameter.radius_series ?? {}).filter(([k]) => keys.has(k));
}

export function radiusSeries(parameter: Parameter, entries: [string, RadiusCurve][]): Series[] {
  const axis = xAxisOf(parameter);
  return entries.map(([key, curve]) => ({
    x: axis.x(curve.x), y: positiveOrNull(curve.density), color: radiusColor(key, curve),
    width: curve.source === "fit" ? 2.7 : 1.9, dash: radiusDash(key, curve),
    marker: curve.source === "euclid" ? "ring" : curve.source === "synthetic" ? "filled" : undefined,
    dots: curve.source === "euclid" || curve.source === "synthetic", markerEvery: markerEvery(curve.x.length),
    name: curve.label, key,
  }));
}

/** Radius y label: a unit-integral panel reads "normalized probability / dex". */
export function radiusYLabel(parameter: Parameter, entries: [string, RadiusCurve][]): string {
  const probability = entries.length > 0 && entries.every(([, c]) => normalizationOf(c) === "probability_density");
  return probability ? "normalized probability / dex (log scale)" : `${parameter.density_unit} (log scale)`;
}

/** Toggling a radius curve keeps the selection in ONE normalization (surface
 *  density and unit-integral shapes never share an axis). */
export function toggleRadius(selected: string[], key: string, byKey: Record<string, RadiusCurve>): string[] {
  if (selected.includes(key)) return selected.filter((k) => k !== key);
  const norm = normalizationOf(byKey[key]);
  return [...selected.filter((k) => normalizationOf(byKey[k]) === norm), key];
}

export const densityDomain = (series: Series[]) => logDomain(series.flatMap((s) => s.y));

/* ─── relations ─────────────────────────────────────────────────────────── */

export const COLOR_TREND: { key: "model_mean_vis_minus_y" | "model_mean_y_j" | "model_mean_j_h"; label: string; band: "Y" | "J" | "H" }[] = [
  { key: "model_mean_vis_minus_y", label: "VIS − Y", band: "Y" },
  { key: "model_mean_y_j", label: "Y − J", band: "J" },
  { key: "model_mean_j_h", label: "J − H", band: "H" },
];

type Colors = NonNullable<NonNullable<GalaxyCandidate["plots"]>["conditional_colors"]>;

export function colorTrendSeries(colors: Colors): Series[] {
  return COLOR_TREND.map(({ key, label, band }) => ({
    x: colors.magnitude, y: colors[key], color: colorIndexColor(band), width: 2.2, name: label,
  }));
}

/** Observed (solid) vs reported-noise (dashed) flux-ratio variance per band. */
export function colorVarianceSeries(colors: Colors): Series[] {
  const edges = colors.magnitude_edges ?? [];
  const centers = edges.slice(0, -1).map((e, i) => (e + edges[i + 1]) / 2);
  if (!centers.length) return [];
  const column = (rows: (number | null)[][] | undefined, band: number) =>
    positiveOrNull((rows ?? []).map((row) => row?.[band] ?? null));
  return (["Y", "J", "H"] as const).flatMap((band, i) => [
    { x: centers, y: column(colors.observed_ratio_variance_by_magnitude, i), color: colorIndexColor(band), width: 2,
      name: `${band}/VIS observed`, key: `${band}-observed` },
    { x: centers, y: column(colors.noise_ratio_variance_by_magnitude, i), color: colorIndexColor(band), width: 1.5,
      dash: [5, 4], name: `${band}/VIS noise`, key: `${band}-noise` },
  ]);
}

/* ─── joint magnitude × radius maps ─────────────────────────────────────── */

/** Q1 gray, generated blue dashed, model red solid; each contour labelled by
 *  its enclosed mass; radius on a log axis in arcsec. */
export function jointMapSeries(data: JointMaps): Series[] {
  const q1 = data.maps.find((m) => m.key === "q1");
  const overlays = data.maps.filter((m) => m.key === "synthetic" || m.key === "model");
  const maps = q1 ? [q1, ...overlays] : overlays;
  return maps.flatMap((map, mapIndex) => contourSeries(map.contours, {
    color: jointColor(map.key, "maps"), dash: map.key === "synthetic" ? [7, 4] : undefined,
    name: map.label, key: map.key, mapY: (y) => 10 ** y, widthScale: 1,
    labelAt: (ci) => Math.min(0.84, 0.22 + 0.27 * mapIndex + 0.035 * (ci % 3)),
  }));
}

/** The progressive Q1 query phases (+0.0 … +0.4 mag offsets). */
export function queryPhases(completed: number, count: number, busy: boolean) {
  return Array.from({ length: count }, (_, i) => ({
    index: i, offset: i / 10,
    state: i < completed ? "cached" as const : busy && i === completed ? "querying" as const : "waiting" as const,
  }));
}
