/* Pixels (field statistics) tab: pure series builders on the Realism chart
   kit. Every figure draws the four bands in their band colours
   (--band-vis/-y/-j/-h); synthetic LR is solid with filled circles, real
   Euclid LR dashed with diamonds / hatched outlines. `visible` filters the
   bands and the two samples (the toolbar's toggles). */
import type { LegendItem, Series } from "../../../charts/Plot";
import { C, bandColor } from "../../../colors";
import type { Band, DetectionSide, FieldComparison, Relation, SourceDetection } from "../api";
import { angularScaleAxis, countHistogram, domainOf, finite, omitBin, ordered, positiveOrNull, quantile } from "../chartKit";

export const BANDS: Band[] = ["VIS", "Y_E", "J_E", "H_E"];
export type Sample = "synthetic" | "real";
export const SAMPLES: Sample[] = ["synthetic", "real"];
export const SAMPLE_LABEL: Record<Sample, string> = { synthetic: "synthetic LR", real: "real Euclid LR" };
export const bandLabel = (band: string) => band.replace("_E", "");

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
