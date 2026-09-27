/* Pixels (field statistics) tab: pure series builders on the Realism chart
   kit. Every figure draws the four bands in their band colours
   (--band-vis/-y/-j/-h); synthetic LR is solid with filled circles, real
   Euclid LR dashed with diamonds / hatched outlines. `visible` filters the
   bands and the two samples (the toolbar's toggles). */
import type { LegendItem, Series } from "../../../charts/Plot";
import { C, bandColor } from "../../../colors";
import { formatCount, formatNumber } from "../../../format";
import type {
  Availability, Band, Comparison, DetectionSide, FieldComparison, GalaxyPayload, Relation, SourceDetection, StarPayload,
} from "../api";
import { angularScaleAxis, countHistogram, domainOf, finite, integrateDensity, omitBin, ordered, positiveOrNull, quantile } from "../chartKit";
import { generatedDensity, trustedWindow } from "../stars/model";

export const BANDS: Band[] = ["VIS", "Y_E", "J_E", "H_E"];
export type Sample = "synthetic" | "real";
export const SAMPLES: Sample[] = ["synthetic", "real"];
export const SAMPLE_LABEL: Record<Sample, string> = { synthetic: "synthetic LR", real: "real Euclid LR" };
export const bandLabel = (band: string) => band.replace("_E", "");

/** A sample chip's label with its size: "synthetic LR · 200 fields", "real Euclid LR · 176 fields / 44 pointings".
 *  The built comparison's samples when there is one, else what the collections hold. */
export function sampleChipLabel(sample: Sample, comparison: Comparison | null | undefined, availability: Availability | undefined): string {
  if (!availability && !comparison) return SAMPLE_LABEL[sample];
  const n = (v: number | undefined) => (v == null ? null : formatCount(v));
  if (sample === "synthetic") {
    const fields = n(comparison?.samples.synthetic.fields ?? availability?.synthetic.fields);
    return fields ? `${SAMPLE_LABEL.synthetic} · ${fields} fields` : SAMPLE_LABEL.synthetic;
  }
  const fields = n(comparison?.samples.real.fields ?? availability?.real.fields);
  const pointings = n(comparison?.samples.real.independent_parents ?? availability?.real.independent_parents);
  return fields ? `${SAMPLE_LABEL.real} · ${fields} fields${pointings ? ` / ${pointings} pointings` : ""}` : SAMPLE_LABEL.real;
}

/** One row of the generated-vs-prior-vs-Q1 census: surface densities in arcmin⁻² over one VIS window.
 *  Each kind has a `q1` row (the window where Q1 is complete, all three columns integrated over the same
 *  magnitudes) and a `prior` row (the full prior range, where Q1 is incomplete, so its q1 is null). */
export type CensusRow = {
  kind: "galaxies" | "stars"; window: "q1" | "prior"; range: [number, number];
  generated: number | null; prior: number | null; q1: number | null;
};

const perArea = (n?: number | null, area?: number | null) => (n != null && area != null && area > 0 ? n / area : null);
const within = (range: [number, number]) => (curve: { x: readonly number[]; density: readonly (number | null)[] } | undefined) =>
  curve ? integrateDensity(curve.x, curve.density, range[0], range[1]) : null;

/** The census from the galaxy and star payloads (independent of the pixel cache). Galaxies compare over
 *  the prior's bright end to the Q1 5σ limit (VIS 2FWHM: the generated magnitudes, the generator law and
 *  the PHZ-weighted Q1 counts); stars over the Q1 trusted window of the VIS panel. The full prior range
 *  follows with generated and prior only. */
export function censusRows(galaxies: GalaxyPayload | null | undefined, stars: StarPayload | null | undefined): CensusRow[] {
  const rows: CensusRow[] = [];
  const generation = galaxies?.calibration.candidate?.generation;
  if (galaxies && generation) {
    const synthetic = galaxies.sources.synthetic;
    const curves = galaxies.parameters.magnitude?.photometry_series;
    const q1Curve = curves?.q1_vis_f2;
    const limit = q1Curve?.trust_boundary?.magnitude ?? curves?.generator_vis_f2?.trust_boundary?.magnitude;
    const full: [number, number] = [generation.vis_magnitude_min, generation.vis_magnitude_max];
    const lo = Math.max(full[0], galaxies.q1_counts?.bright ?? full[0]);
    const hi = Math.min(full[1], limit ?? Number.NaN, galaxies.q1_counts?.faint ?? full[1]);
    if (q1Curve && hi > lo) {
      const window: [number, number] = [lo, hi];
      const over = within(window);
      rows.push({
        kind: "galaxies", window: "q1", range: window,
        generated: synthetic?.available ? over(curves?.synthetic_vis_2fwhm) : null,
        prior: over(curves?.generator_vis_f2), q1: over(q1Curve),
      });
    }
    rows.push({
      kind: "galaxies", window: "prior", range: full,
      generated: synthetic?.available ? perArea(synthetic.rows, synthetic.area_arcmin2) : null,
      prior: generation.surface_density_arcmin2 ?? null, q1: null,
    });
  }
  const c = stars?.distribution?.density_comparison;
  if (c) {
    const vis = c.parameters.vis;
    const generated = generatedDensity(c);
    const trusted = trustedWindow(vis);
    const lo = trusted ? Math.max(trusted[0], vis.x_domain[0]) : Number.NaN;
    const hi = trusted ? Math.min(trusted[1], vis.x_domain[1]) : Number.NaN;
    if (hi > lo) {
      const window: [number, number] = [lo, hi];
      rows.push({
        kind: "stars", window: "q1", range: window,
        generated: generated != null ? integrateDensity(vis.x, vis.synthetic, lo, hi) : null,
        prior: integrateDensity(vis.x, vis.model, lo, hi), q1: integrateDensity(vis.x, vis.euclid, lo, hi),
      });
    }
    rows.push({ kind: "stars", window: "prior", range: vis.x_domain, generated, prior: c.model_density_arcmin2, q1: null });
  }
  return rows;
}

/** Which bands and samples are drawn (the hidden keys of the toolbar). */
export type Visible = { bands: Band[]; samples: Sample[] };
export function visibleFrom(hidden: readonly string[]): Visible {
  return {
    bands: BANDS.filter((b) => !hidden.includes(b)),
    samples: SAMPLES.filter((s) => !hidden.includes(s)),
  };
}

const style = {
  synthetic: { dash: undefined, marker: "filled" as const, width: 2.6 },
  real: { dash: [10, 5], marker: "diamond" as const, width: 2.3 },
};

export function histogramSeries(fields: FieldComparison, v: Visible): Series[] {
  return v.bands.flatMap((band) => {
    const h = fields.histograms[band];
    return v.samples.map((s): Series => ({
      x: h.x, y: omitBin(h[s], h.zero_bin), color: bandColor(band), mode: "histogram",
      ...(s === "synthetic" ? { width: 1.5, alpha: 0.92, fillAlpha: 0.22 } : { width: 1.9, dash: [8, 4], hatch: true, alpha: 1, fillAlpha: 0.025 }),
      name: `${bandLabel(band)} · ${SAMPLE_LABEL[s]}`, key: `${band}:${s}`,
    }));
  });
}

export function quantileSeries(fields: FieldComparison, v: Visible): Series[] {
  return v.bands.flatMap((band) => {
    const q = fields.quantiles[band];
    return v.samples.map((s): Series => ({
      x: q.q, y: q[s], color: bandColor(band), width: style[s].width, dash: style[s].dash, dots: true,
      marker: style[s].marker, markerEvery: 4, name: `${bandLabel(band)} · ${SAMPLE_LABEL[s]}`, key: `${band}:${s}`,
    }));
  });
}

/** Median mean-subtracted power against angular scale (1/k), log–log. */
export function powerSeries(fields: FieldComparison, v: Visible): Series[] {
  return v.bands.flatMap((band) => {
    const power = fields.power[band];
    const axis = angularScaleAxis(power.k);
    const markerEvery = Math.max(1, Math.round(axis.x.length / 9));
    return v.samples.map((s): Series => ({
      x: axis.x, y: positiveOrNull(ordered(power[s].median, axis.order)), color: bandColor(band),
      width: style[s].width, dash: style[s].dash, dots: true, marker: style[s].marker, markerEvery,
      name: `${bandLabel(band)} · ${SAMPLE_LABEL[s]}`, key: `${band}:${s}`,
    }));
  });
}

export function similaritySeries(fields: FieldComparison, v: Visible): Series[] {
  return v.bands.map((band) => {
    const sim = fields.scale_similarity[band];
    const axis = angularScaleAxis(sim.k);
    return {
      x: axis.x, y: ordered(sim.log_shape_ratio.median, axis.order),
      low: ordered(sim.log_shape_ratio.p16, axis.order), high: ordered(sim.log_shape_ratio.p84, axis.order),
      color: bandColor(band), width: 2.4, fillAlpha: 0.08, name: bandLabel(band), key: band,
    };
  });
}

export type RelationKey = keyof FieldComparison["relations"];

export function relationSeries(fields: FieldComparison, key: RelationKey, v: Visible): Series[] {
  return v.bands.flatMap((band) => {
    const r: Relation = fields.relations[key][band];
    return v.samples.map((s): Series => ({
      x: r[s].x, y: r[s].y, color: bandColor(band), mode: "scatter", marker: style[s].marker,
      width: s === "synthetic" ? 1.7 : 1.8, alpha: s === "synthetic" ? 0.52 : 0.92,
      name: `${bandLabel(band)} · ${SAMPLE_LABEL[s]}`, key: `${band}:${s}`,
    }));
  });
}

export function relationDomains(fields: FieldComparison, key: RelationKey): { x: [number, number]; y: [number, number] } {
  const rel = BANDS.map((b) => fields.relations[key][b]);
  return {
    x: domainOf(rel.flatMap((r) => [...r.synthetic.x, ...r.real.x])),
    y: domainOf(rel.flatMap((r) => [...r.synthetic.y, ...r.real.y]), 0, true),
  };
}

/** The field a click on a relation plot is nearest to, per visible series
 *  (the real sample's parent pointing, to open an archive field). */
export function relationPoints(fields: FieldComparison, key: RelationKey, v: Visible) {
  return v.bands.flatMap((band) => v.samples.flatMap((s) => {
    const r = fields.relations[key][band][s];
    return r.x.map((x, i) => ({ x, y: r.y[i], sample: s, band, parent: r.parent_ids?.[i] ?? null }));
  }));
}

export function correlationSeries(fields: FieldComparison, v: Visible): Series[] {
  const c = fields.band_correlation;
  const xs = c.pairs.map((_, i) => i);
  return v.samples.map((s): Series => ({
    x: xs, y: c[s].median, low: c[s].p16, high: c[s].p84, color: s === "synthetic" ? C.comb : C.mean,
    width: style[s].width, dash: style[s].dash, dots: true, marker: style[s].marker,
    fillAlpha: s === "synthetic" ? 0.11 : 0.055, name: SAMPLE_LABEL[s], key: s,
  }));
}

/** Band-colour legend + sample-style legend (external, toggling the
 *  toolbar's hidden keys). */
export function bandLegend(histogram = false): LegendItem[] {
  return BANDS.map((band) => ({ label: bandLabel(band), key: band, color: bandColor(band), histogram }));
}
export function sampleLegend(kind: "line" | "histogram" | "scatter"): LegendItem[] {
  if (kind === "histogram") {
    return [
      { label: "synthetic LR", key: "synthetic", color: C.cross, histogram: true, filled: true },
      { label: "real Euclid LR · hatched", key: "real", color: C.cross, histogram: true, hatch: true, dash: true },
    ];
  }
  if (kind === "scatter") {
    return [
      { label: "synthetic LR · filled circles", key: "synthetic", color: C.cross, marker: "filled" },
      { label: "real Euclid LR · diamonds", key: "real", color: C.cross, marker: "diamond" },
    ];
  }
  return [
    { label: "synthetic LR · solid + circles", key: "synthetic", color: C.cross, line: true, marker: "filled" },
    { label: "real Euclid LR · dashed + diamonds", key: "real", color: C.cross, line: true, dash: true, marker: "diamond" },
  ];
}

/* ─── source detection (VIS segmentation per field) ─────────────────────── */

export type DetectionStats = {
  fields: number;
  positive: { median: number | null; p16: number | null; p84: number | null };
  negative: { median: number | null; p16: number | null; p84: number | null };
  /** Σ negative islands ÷ Σ positive detections: the false-positive rate of
   *  the threshold (a symmetric noise field gives as many of each). */
  spurious: number | null;
  /** Σ matched galaxies ÷ Σ truth galaxies (synthetic only; null without truth). */
  completeness: number | null;
  matchedStars: number;
};

const band3 = (values: readonly number[]) => ({ median: quantile(values, 0.5), p16: quantile(values, 0.16), p84: quantile(values, 0.84) });
const sum = (values: readonly number[]) => finite(values).reduce((a, b) => a + b, 0);

export function detectionStats(side: DetectionSide): DetectionStats {
  const positive = sum(side.positive);
  const truth = sum(side.truth_galaxies);
  return {
    fields: side.positive.length,
    positive: band3(side.positive),
    negative: band3(side.negative),
    spurious: positive > 0 ? sum(side.negative) / positive : null,
    completeness: truth > 0 ? sum(side.matched_galaxies) / truth : null,
    matchedStars: sum(side.matched_stars),
  };
}

/** Per-field completeness (matched ÷ truth galaxies), fields with truth only. */
export function perFieldCompleteness(side: DetectionSide): number[] {
  return side.truth_galaxies.flatMap((t, i) => (t > 0 ? [side.matched_galaxies[i] / t] : []));
}

/** Histograms of per-field detection counts for both samples on shared bins. */
export function detectionHistogram(detection: SourceDetection, which: "positive" | "negative", v: Visible): Series[] {
  const all = [...detection.synthetic[which], ...detection.real[which]];
  const shared = countHistogram(all);
  const width = shared.edges[1] - shared.edges[0];
  const centers = shared.edges.slice(0, -1).map((e) => e + width / 2);
  return v.samples.map((s): Series => {
    const counts = new Array<number>(centers.length).fill(0);
    const values = finite(detection[s][which]);
    for (const value of values) counts[Math.min(centers.length - 1, Math.floor(value / width))] += 1;
    return {
      x: centers, y: counts.map((c) => (values.length ? c / values.length : 0)), mode: "histogram",
      color: s === "synthetic" ? C.comb : C.mean,
      ...(s === "synthetic" ? { width: 1.5, fillAlpha: 0.24 } : { width: 1.9, dash: [8, 4], hatch: true, fillAlpha: 0.03 }),
      name: SAMPLE_LABEL[s], key: s,
    };
  });
}

/** "Generated: 5,489 galaxies and 175 stars over 36.4 arcmin²" — the shared area stated once; each
 *  sample keeps its own area when they differ. */
export function generatedSample(
  galaxies: { rows?: number | null; area_arcmin2?: number | null } | undefined,
  stars: { synthetic_star_count?: number | null; synthetic_area_arcmin2?: number | null } | null | undefined,
): string | null {
  const parts: { text: string; area: number }[] = [];
  if (galaxies?.rows != null && galaxies.area_arcmin2) parts.push({ text: `${formatCount(galaxies.rows)} galaxies`, area: galaxies.area_arcmin2 });
  if (stars?.synthetic_star_count != null && stars.synthetic_area_arcmin2) {
    parts.push({ text: `${formatCount(stars.synthetic_star_count)} stars`, area: stars.synthetic_area_arcmin2 });
  }
  if (!parts.length) return null;
  const areaText = (a: number) => `${formatNumber(a)} arcmin²`;
  const shared = parts.every((p) => areaText(p.area) === areaText(parts[0].area));
  return shared
    ? `Generated: ${parts.map((p) => p.text).join(" and ")} over ${areaText(parts[0].area)}`
    : `Generated: ${parts.map((p) => `${p.text} over ${areaText(p.area)}`).join(", ")}`;
}
