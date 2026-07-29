import { useEffect, useState, type CSSProperties } from "react";
import { NavLink } from "react-router-dom";
import Plot, { Legend, type PlotProps, type Series, type Tick } from "../charts/Plot";
import { C, categorical } from "../colors";
import { useResource } from "../hooks";
import { JobProgressView, useJob } from "../jobs";
import {
  Badge, Button, Card, CardBody, CardHead, Checkbox, Empty, Field, Input, Page,
  PageHead, Spinner, Stat,
} from "../ui";
import "./population-comparison.css";

type Band = "VIS" | "Y_E" | "J_E" | "H_E";
type Curve = { p16: (number | null)[]; median: (number | null)[]; p84: (number | null)[] };
type Interval = { median: number; p16: number; p84: number };
type PointCloud = { x: number[]; y: number[] };
type Relation = {
  synthetic: PointCloud; real: PointCloud;
  x_label: string; y_label: string;
};
type FieldComparison = {
  bands: Band[];
  histograms: Record<Band, {
    x: number[]; synthetic: number[]; real: number[];
    zero_bin: number | null;
    x_label: string; y_label: string;
  }>;
  quantiles: Record<Band, {
    q: number[]; synthetic: number[]; real: number[];
    x_label: string; y_label: string;
  }>;
  power: Record<Band, {
    k: number[]; synthetic: Curve; real: Curve;
    x_label: string; y_label: string;
  }>;
  scale_similarity: Record<Band, {
    k: number[];
    log_shape_ratio: Curve;
    overlap: Interval;
    variance_ratio: Interval;
    x_label: string;
    y_label: string;
  }>;
  relations: Record<"mean_std" | "median_robust_std", Record<Band, Relation>>;
  band_correlation: {
    pairs: string[];
    synthetic: Curve;
    real: Curve;
    x_label: string;
    y_label: string;
  };
  summary: Record<"synthetic" | "real", Record<Band, Record<
    "mean" | "median" | "std" | "robust_std" | "p01" | "p99"
    | "zero_fraction" | "negative_fraction",
    Interval
  >>>;
};
type Population = {
  objects: number;
  counts: Record<string, number>;
  density_arcmin2: Record<string, number | null>;
  area_arcmin2: number;
};
type PriorInterval = { median: number; p16: number; p84: number };
type CatalogPrior = {
  method: string;
  current_prior_arcmin2: number;
  fitted_prior_arcmin2: number;
  interval_arcmin2: PriorInterval;
  turnover_limit_mag: number;
  selected_bin_count: number;
  synthetic_selected_count: number;
  euclid_selected_count: number;
  reduced_poisson_deviance: number;
  log10_prior_slope_per_mag: number;
  single_scalar_adequate: boolean;
  curve: {
    mag: number[];
    prior_arcmin2: number[];
    synthetic_density: number[];
    euclid_density: number[];
  };
  uncertainty_note: string;
};
type VisiblePrior = {
  method: string;
  current_prior_arcmin2: number;
  fitted_prior_arcmin2: number;
  interval_arcmin2: PriorInterval;
  synthetic_detected_density_arcmin2: number;
  real_detected_density_arcmin2: number;
  synthetic_retained_truth_density_arcmin2: number;
  matched_truth_fraction: number;
  synthetic_fields: number;
  real_fields: number;
  caveat: string;
};
type TngPrior = {
  catalog: CatalogPrior | null;
  visible: VisiblePrior | null;
  dataset_prior_arcmin2?: number;
  configured_prior_arcmin2?: number;
  configured_mf_alpha?: number;
  single_scalar_adequate: boolean | null;
  calibration_scope?: string;
  pilot_grid_arcmin2: number[];
  recommendation: string;
};
type Comparison = {
  version: number;
  geometry: {
    tile_size: number; analysis_size: number;
    pixel_scale_arcsec: number; field_area_arcmin2: number;
  };
  samples: {
    synthetic: { fields: number; area_arcmin2: number; splits: string[] };
    real: {
      fields: number; area_arcmin2: number;
      inference_fields: number; jwst_overlap_fields: number;
    };
  };
  fields: FieldComparison;
  population: {
    synthetic: Population;
    synthetic_field_count: number;
    euclid: Population | null;
    shared: null;
    tng_prior?: TngPrior | null;
    synthetic_splits?: string[];
    training_included?: boolean;
    calibration_splits?: string[];
    euclid_meta: {
      ra: number; dec: number; radius_arcmin: number; area_arcmin2: number;
      rows: number; limit: number; limit_reached: boolean; classification: string;
      catalog_version: number;
      counts: Record<string, number>;
      classification_note: string;
      photometry: string;
    } | null;
  };
};
type Availability = {
  synthetic: {
    fields: number; area_arcmin2: number; record_files: number;
    source_catalogs: number; train_source_catalog: boolean;
    population_fields: number; population_area_arcmin2: number;
    population_fields_with_training: number;
    population_area_arcmin2_with_training: number;
  };
  real: {
    fields: number; area_arcmin2: number;
    inference_fields: number; jwst_overlap_fields: number;
  };
  euclid_catalog: { cached: boolean; meta: Comparison["population"]["euclid_meta"] };
  field_area_arcmin2: number;
  default_cone: { ra: number; dec: number; radius_arcmin: number; area_arcmin2: number };
};
type ApiPayload = {
  comparison: Comparison | null;
  availability: Availability;
  authenticated: boolean;
};

const BANDS: Band[] = ["VIS", "Y_E", "J_E", "H_E"];
const TYPE_ORDER = ["galaxy", "star", "unknown"];
const BAND_COLOR = (band: Band) => categorical(BANDS.indexOf(band));
const bandLabel = (band: Band) => band.replace("_E", "");

function finite(values: readonly (number | null)[]): number[] {
  return values.filter((value): value is number => value != null && Number.isFinite(value));
}

function euclidMetaMatches(
  cached: Comparison["population"]["euclid_meta"],
  displayed: Comparison["population"]["euclid_meta"],
): boolean {
  if (cached == null || displayed == null) return cached === displayed;
  return cached.rows === displayed.rows
    && cached.ra === displayed.ra
    && cached.dec === displayed.dec
    && cached.radius_arcmin === displayed.radius_arcmin
    && cached.area_arcmin2 === displayed.area_arcmin2;
}

function domain(values: readonly (number | null)[], includeZero = false): [number, number] {
  const good = finite(values);
  if (!good.length) return [0, 1];
  let lo = Math.min(...good), hi = Math.max(...good);
  if (includeZero) lo = Math.min(0, lo);
  if (lo === hi) return [lo - 0.5, hi + 0.5];
  const pad = (hi - lo) * 0.06;
  return [includeZero ? 0 : lo - pad, hi + pad];
}

function ticks([lo, hi]: [number, number], count = 5): Tick[] {
  return Array.from({ length: count }, (_, index) => {
    const value = lo + (hi - lo) * index / (count - 1);
    const abs = Math.abs(value);
    const label = abs >= 1000 || (abs > 0 && abs < 0.01)
      ? value.toExponential(1) : Number(value.toPrecision(3)).toString();
    return { v: value, label };
  });
}

function logarithmicTicks([lo, hi]: [number, number], count = 5): Tick[] {
  if (!(lo > 0) || !(hi > lo)) return [];
  const a = Math.log10(lo), b = Math.log10(hi);
  return Array.from({ length: count }, (_, index) => {
    const value = 10 ** (a + (b - a) * index / (count - 1));
    return { v: value, label: Number(value.toPrecision(3)).toString() };
  });
}

type AxisDomain = [number, number];
type DomainTicks = (value: AxisDomain) => Tick[];
type AdjustablePlotProps = Omit<PlotProps, "xTicks" | "yTicks"> & {
  boundsLabel: string;
  xTicksForDomain?: DomainTicks;
  yTicksForDomain?: DomainTicks;
};
type DualRangeProps = {
  axis: "x" | "y";
  boundsLabel: string;
  domain: AxisDomain;
  value: AxisDomain;
  scale?: PlotProps["xScale"];
  onChange: (value: AxisDomain) => void;
};

const SLIDER_STEPS = 10_000;

function boundText(value: number): string {
  return Number(value.toPrecision(8)).toString();
}

function sliderPosition(
  value: number,
  [lo, hi]: AxisDomain,
  scale: PlotProps["xScale"],
): number {
  const fraction = scale === "log"
    ? (Math.log(value) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))
    : (value - lo) / (hi - lo);
  return Math.round(Math.max(0, Math.min(1, fraction)) * SLIDER_STEPS);
}

function sliderValue(
  position: number,
  [lo, hi]: AxisDomain,
  scale: PlotProps["xScale"],
): number {
  if (position <= 0) return lo;
  if (position >= SLIDER_STEPS) return hi;
  const fraction = position / SLIDER_STEPS;
  return scale === "log"
    ? Math.exp(Math.log(lo) + fraction * (Math.log(hi) - Math.log(lo)))
    : lo + fraction * (hi - lo);
}

function sameDomain(a: AxisDomain, b: AxisDomain): boolean {
  return a[0] === b[0] && a[1] === b[1];
}

function DualRange({
  axis,
  boundsLabel,
  domain,
  value,
  scale,
  onChange,
}: DualRangeProps) {
  const lower = sliderPosition(value[0], domain, scale);
  const upper = sliderPosition(value[1], domain, scale);
  const start = `${100 * lower / SLIDER_STEPS}%`;
  const end = `${100 * upper / SLIDER_STEPS}%`;
  const setLower = (position: number) => {
    const next = Math.min(position, upper - 1);
    onChange([sliderValue(next, domain, scale), value[1]]);
  };
  const setUpper = (position: number) => {
    const next = Math.max(position, lower + 1);
    onChange([value[0], sliderValue(next, domain, scale)]);
  };

  return (
    <div className="dual-range">
      <div className="dual-range__head">
        <span>{axis} range</span>
        <output>
          <b>{boundText(value[0])}</b>
          <i>to</i>
          <b>{boundText(value[1])}</b>
        </output>
      </div>
      <div className="dual-range__control"
        style={{ "--range-start": start, "--range-end": end } as CSSProperties}>
        <div className="dual-range__rail" />
        <div className="dual-range__selection" />
        <input type="range" min={0} max={SLIDER_STEPS} step={1}
          value={lower}
          aria-label={`${boundsLabel} ${axis} lower bound`}
          aria-valuetext={boundText(value[0])}
          onInput={(event) => setLower(Number(event.currentTarget.value))}
          onChange={(event) => setLower(Number(event.target.value))} />
        <input type="range" min={0} max={SLIDER_STEPS} step={1}
          value={upper}
          aria-label={`${boundsLabel} ${axis} upper bound`}
          aria-valuetext={boundText(value[1])}
          onInput={(event) => setUpper(Number(event.currentTarget.value))}
          onChange={(event) => setUpper(Number(event.target.value))} />
      </div>
    </div>
  );
}

function AdjustablePlot({
  boundsLabel,
  xDomain,
  yDomain,
  xScale,
  xTicksForDomain = ticks,
  yTicksForDomain = ticks,
  ...plotProps
}: AdjustablePlotProps) {
  const [view, setView] = useState<{ x: AxisDomain; y: AxisDomain }>({
    x: xDomain, y: yDomain,
  });

  useEffect(() => {
    setView({ x: xDomain, y: yDomain });
  }, [xDomain[0], xDomain[1], yDomain[0], yDomain[1]]);

  const custom = !sameDomain(view.x, xDomain) || !sameDomain(view.y, yDomain);
  const reset = () => {
    setView({ x: xDomain, y: yDomain });
  };

  return (
    <div className="adjustable-plot">
      <Plot {...plotProps}
        xDomain={view.x} yDomain={view.y} xScale={xScale}
        xTicks={xTicksForDomain(view.x)}
        yTicks={yTicksForDomain(view.y)} />
      <details className={`plot-bounds${custom ? " plot-bounds--custom" : ""}`}>
        <summary>
          <span>axis bounds</span>
          <small>{custom ? "custom view" : "full data range"}</small>
        </summary>
        <div className="plot-bounds__controls">
          <DualRange axis="x" boundsLabel={boundsLabel}
            domain={xDomain} value={view.x} scale={xScale}
            onChange={(value) => setView((current) => ({ ...current, x: value }))} />
          <DualRange axis="y" boundsLabel={boundsLabel}
            domain={yDomain} value={view.y}
            onChange={(value) => setView((current) => ({ ...current, y: value }))} />
          <button type="button" className="plot-bounds__reset"
            disabled={!custom} onClick={reset}>
            reset
          </button>
        </div>
      </details>
    </div>
  );
}

function intervalLabel(interval: Interval, digits: number): string {
  return `${interval.median.toFixed(digits)} · ${interval.p16.toFixed(digits)}–${interval.p84.toFixed(digits)}`;
}

function log10(values: readonly (number | null)[]): (number | null)[] {
  return values.map((value) =>
    value != null && value > 0 && Number.isFinite(value) ? Math.log10(value) : null);
}

function angularScaleAxis(frequencies: readonly number[]): {
  x: number[];
  order: number[];
} {
  const points = frequencies
    .map((frequency, index) => ({ frequency, index }))
    .filter(({ frequency }) => frequency > 0 && Number.isFinite(frequency))
    .map(({ frequency, index }) => ({ scale: 1 / frequency, index }))
    .sort((left, right) => left.scale - right.scale);
  return {
    x: points.map(({ scale }) => scale),
    order: points.map(({ index }) => index),
  };
}

function ordered<T>(values: readonly T[], order: readonly number[]): T[] {
  return order.map((index) => values[index]);
}

function omitBin(values: readonly number[], index: number | null): (number | null)[] {
  return values.map((value, bin) => bin === index ? null : value);
}

function DatasetRuler({ availability, comparison }: {
  availability: Availability; comparison: Comparison | null;
}) {
  const synthetic = comparison?.samples.synthetic.fields ?? availability.synthetic.fields;
  const real = comparison?.samples.real.fields ?? availability.real.fields;
  const max = Math.max(synthetic, real, 1);
  const inference = comparison?.samples.real.inference_fields ?? availability.real.inference_fields;
  const overlap = comparison?.samples.real.jwst_overlap_fields ?? availability.real.jwst_overlap_fields;
  return (
    <section className="survey-ruler" aria-label="Compared field census">
      <div className="survey-ruler__axis">
        <span>same footprint</span>
        <strong>{comparison?.geometry.tile_size ?? 256} px</strong>
        <span>× {comparison?.geometry.pixel_scale_arcsec ?? 0.1}″ px⁻¹</span>
        <i />
        <span>{availability.field_area_arcmin2.toFixed(3)} arcmin² / field</span>
      </div>
      <div className="survey-ruler__row survey-ruler__row--synthetic">
        <div className="survey-ruler__label">
          <strong>Synthetic LR</strong>
          <span>test + validate</span>
        </div>
        <div className="survey-ruler__track">
          <span style={{ width: `${100 * synthetic / max}%` }} />
        </div>
        <div className="survey-ruler__value">
          <strong>{synthetic}</strong><span>fields</span>
        </div>
      </div>
      <div className="survey-ruler__row survey-ruler__row--real">
        <div className="survey-ruler__label">
          <strong>Real Euclid LR</strong>
          <span>{inference} inference + {overlap} JWST-overlap</span>
        </div>
        <div className="survey-ruler__track">
          <span style={{ width: `${100 * real / max}%` }} />
        </div>
        <div className="survey-ruler__value">
          <strong>{real}</strong><span>fields</span>
        </div>
      </div>
      <div className="survey-ruler__scale mono">
        <span>0</span><span>{Math.round(max / 2)}</span><span>{max} fields</span>
      </div>
    </section>
  );
}

function FieldPlots({ comparison }: { comparison: Comparison }) {
  const histograms = BANDS.map((band) => [band, comparison.fields.histograms[band]] as const);
  const quantiles = BANDS.map((band) => [band, comparison.fields.quantiles[band]] as const);
  const powers = BANDS.map((band) => [band, comparison.fields.power[band]] as const);
  const similarities = BANDS.map((band) => [band, comparison.fields.scale_similarity[band]] as const);
  const meanStd = BANDS.map((band) =>
    [band, comparison.fields.relations.mean_std[band]] as const);
  const medianRobust = BANDS.map((band) =>
    [band, comparison.fields.relations.median_robust_std[band]] as const);
  const histogramX = domain(histograms.flatMap(([, histogram]) => histogram.x));
  const histogramY = domain(histograms.flatMap(([, histogram]) => [
    ...omitBin(histogram.synthetic, histogram.zero_bin),
    ...omitBin(histogram.real, histogram.zero_bin),
  ]), true);
  const powerY = domain(powers.flatMap(([, power]) => [
    ...log10(power.synthetic.median), ...log10(power.real.median),
  ]));
  const quantileY = domain(quantiles.flatMap(([, item]) => [
    ...item.synthetic, ...item.real,
  ]));
  const powerAxes = new Map(powers.map(([band, power]) => [
    band, angularScaleAxis(power.k),
  ]));
  const allPowerScales = [...powerAxes.values()].flatMap((axis) => axis.x);
  const powerX: [number, number] = [
    Math.min(...allPowerScales), Math.max(...allPowerScales),
  ];
  const similarityAxes = new Map(similarities.map(([band, similarity]) => [
    band, angularScaleAxis(similarity.k),
  ]));
  const allSimilarityScales = [...similarityAxes.values()].flatMap((axis) => axis.x);
  const similarityX: [number, number] = [
    Math.min(...allSimilarityScales), Math.max(...allSimilarityScales),
  ];
  const ratioValues = similarities.flatMap(([, similarity]) => [
    ...similarity.log_shape_ratio.p16,
    ...similarity.log_shape_ratio.median,
    ...similarity.log_shape_ratio.p84,
  ]);
  const finiteRatios = finite(ratioValues);
  const ratioLow = Math.min(0, ...finiteRatios);
  const ratioHigh = Math.max(0, ...finiteRatios);
  const ratioPad = Math.max(0.05, (ratioHigh - ratioLow) * 0.06);
  const ratioY: [number, number] = [ratioLow - ratioPad, ratioHigh + ratioPad];
  const histogramSeries: Series[] = histograms.flatMap(([band, histogram]) => [
    {
      x: histogram.x, y: omitBin(histogram.synthetic, histogram.zero_bin),
      color: BAND_COLOR(band), mode: "histogram",
      width: 1.5, alpha: 0.92, fillAlpha: 0.22,
    },
    {
      x: histogram.x, y: omitBin(histogram.real, histogram.zero_bin),
      color: BAND_COLOR(band), mode: "histogram",
      width: 1.9, dash: [8, 4], hatch: true, alpha: 1, fillAlpha: 0.025,
    },
  ]);
  const quantileSeries: Series[] = quantiles.flatMap(([band, item]) => [
    {
      x: item.q, y: item.synthetic, color: BAND_COLOR(band),
      width: 2.7, dots: true, marker: "filled", markerEvery: 4,
    },
    {
      x: item.q, y: item.real, color: BAND_COLOR(band),
      width: 2.4, dash: [10, 5], dots: true, marker: "diamond", markerEvery: 4,
    },
  ]);
  const powerSeries: Series[] = powers.flatMap(([band, power]) => {
    const axis = powerAxes.get(band)!;
    const markerEvery = Math.max(1, Math.round(axis.x.length / 9));
    return [
      {
        x: axis.x, y: ordered(log10(power.synthetic.median), axis.order),
        color: BAND_COLOR(band), width: 2.7,
        dots: true, marker: "filled", markerEvery,
      },
      {
        x: axis.x, y: ordered(log10(power.real.median), axis.order),
        color: BAND_COLOR(band), width: 2.4, dash: [10, 5],
        dots: true, marker: "diamond", markerEvery,
      },
    ];
  });
  const similaritySeries: Series[] = similarities.map(([band, similarity]) => {
    const axis = similarityAxes.get(band)!;
    return {
      x: axis.x,
      y: ordered(similarity.log_shape_ratio.median, axis.order),
      low: ordered(similarity.log_shape_ratio.p16, axis.order),
      high: ordered(similarity.log_shape_ratio.p84, axis.order),
      color: BAND_COLOR(band),
      width: 2.4,
      fillAlpha: 0.08,
    };
  });
  const relationSeries = (
    entries: readonly (readonly [Band, Relation])[],
  ): Series[] => entries.flatMap(([band, relation]) => [
    {
      x: relation.synthetic.x, y: relation.synthetic.y,
      color: BAND_COLOR(band), mode: "scatter", marker: "filled",
      width: 1.7, alpha: 0.52,
    },
    {
      x: relation.real.x, y: relation.real.y,
      color: BAND_COLOR(band), mode: "scatter", marker: "diamond",
      width: 1.8, alpha: 0.92,
    },
  ]);
  const relationDomains = (
    entries: readonly (readonly [Band, Relation])[],
  ): { x: [number, number]; y: [number, number] } => ({
    x: domain(entries.flatMap(([, relation]) => [
      ...relation.synthetic.x, ...relation.real.x,
    ])),
    y: domain(entries.flatMap(([, relation]) => [
      ...relation.synthetic.y, ...relation.real.y,
    ]), true),
  });
  const meanStdDomain = relationDomains(meanStd);
  const medianRobustDomain = relationDomains(medianRobust);
  const correlations = comparison.fields.band_correlation;
  const correlationX: [number, number] = [-0.35, correlations.pairs.length - 0.65];
  const correlationValues = [
    ...correlations.synthetic.p16, ...correlations.synthetic.p84,
    ...correlations.real.p16, ...correlations.real.p84,
  ];
  const correlationY = domain(correlationValues);
  const correlationSeries: Series[] = [
    {
      x: correlations.pairs.map((_, index) => index),
      y: correlations.synthetic.median,
      low: correlations.synthetic.p16,
      high: correlations.synthetic.p84,
      color: C.comb, width: 2.8, dots: true, marker: "filled",
      fillAlpha: 0.11,
    },
    {
      x: correlations.pairs.map((_, index) => index),
      y: correlations.real.median,
      low: correlations.real.p16,
      high: correlations.real.p84,
      color: C.mean, width: 2.5, dash: [10, 5],
      dots: true, marker: "diamond", fillAlpha: 0.055,
    },
  ];
  const bandLegend = BANDS.map((band) => ({
    color: BAND_COLOR(band), label: bandLabel(band),
  }));
  const sourceLegend = [
    {
      color: C.cross, label: "synthetic LR · solid + circles",
      line: true, dash: false, marker: "filled" as const,
    },
    {
      color: C.cross, label: "real Euclid LR · dashed + diamonds",
      line: true, dash: true, marker: "diamond" as const,
    },
  ];
  const histogramBandLegend = BANDS.map((band) => ({
    color: BAND_COLOR(band), label: bandLabel(band), histogram: true,
  }));
  const histogramSourceLegend = [
    {
      color: C.cross, label: "synthetic LR",
      histogram: true, filled: true, dash: false,
    },
    {
      color: C.cross, label: "real Euclid LR · hatched outline",
      histogram: true, filled: false, hatch: true, dash: true,
    },
  ];
  const scatterSourceLegend = [
    { color: C.cross, label: "synthetic LR · filled circles", marker: "filled" as const },
    { color: C.cross, label: "real Euclid LR · hollow diamonds", marker: "diamond" as const },
  ];

  return (
    <section className="comparison-section">
      <header className="comparison-section__head">
        <div>
          <div className="eyebrow">image domain · one field at a time</div>
          <h2>Pixel statistics</h2>
          <p>VIS, Y, J and H share each plot; every histogram and curve uses the same 255×255 center crop and native 0.1″ pixel scale.</p>
        </div>
      </header>
      <div className="field-plot-grid">
        <Card className="comparison-plot">
          <CardHead title="Brightness distribution"
            sub="Filled bars are synthetic; hatched outlines are real Euclid. The 0 e⁻ bin is hidden to expose the signal wings." />
          <CardBody>
            <AdjustablePlot boundsLabel="Brightness distribution"
              xDomain={histogramX} yDomain={histogramY}
              xLabel="pixel brightness (e⁻ / stack)"
              yLabel="fraction of sampled pixels / bin"
              series={histogramSeries} />
            <Legend items={histogramBandLegend} />
            <Legend items={histogramSourceLegend} />
          </CardBody>
        </Card>

        <Card className="comparison-plot">
          <CardHead title="Pixel quantile profile"
            sub="A tail-stable view from the 0.1st to 99.9th percentile. Circles are synthetic; diamonds are real Euclid." />
          <CardBody>
            <AdjustablePlot boundsLabel="Pixel quantile profile"
              xDomain={[0.1, 99.9]} yDomain={quantileY}
              xTicksForDomain={(value) => ticks(value).map((tick) => ({
                ...tick, label: `${tick.label}%`,
              }))}
              xLabel="pixel percentile"
              yLabel="pixel brightness (e⁻ / stack)"
              series={quantileSeries} />
            <Legend items={bandLegend} />
            <Legend items={sourceLegend} />
          </CardBody>
        </Card>

        <Card className="comparison-plot">
          <CardHead title="Angular-scale power"
            sub="Median mean-subtracted field power. Solid circles are synthetic; dashed diamonds are real Euclid." />
          <CardBody>
            <AdjustablePlot boundsLabel="Angular-scale power"
              xDomain={powerX} yDomain={powerY} xScale="log"
              xTicksForDomain={logarithmicTicks}
              yTicksForDomain={(value) => ticks(value).map((tick) => ({
                ...tick, label: `10^${tick.label}`,
              }))}
              xLabel="angular scale (arcsec / cycle)"
              yLabel="log₁₀ mean-subtracted power (e⁻²)"
              series={powerSeries} />
            <Legend items={bandLegend} />
            <Legend items={sourceLegend} />
          </CardBody>
        </Card>

        <Card className="comparison-plot">
          <CardHead title="Mean brightness vs field variation"
            sub="Each marker is one 255×255 field. Filled circles are synthetic; hollow diamonds are real Euclid." />
          <CardBody>
            <AdjustablePlot boundsLabel="Mean brightness vs field variation"
              xDomain={meanStdDomain.x} yDomain={meanStdDomain.y}
              xLabel="field mean (e⁻ / pixel)"
              yLabel="field standard deviation (e⁻ / pixel)"
              series={relationSeries(meanStd)} />
            <Legend items={bandLegend} />
            <Legend items={scatterSourceLegend} />
          </CardBody>
        </Card>

        <Card className="comparison-plot">
          <CardHead title="Background vs robust noise"
            sub="Median and MAD suppress bright objects. Filled circles are synthetic; hollow diamonds are real Euclid." />
          <CardBody>
            <AdjustablePlot boundsLabel="Background vs robust noise"
              xDomain={medianRobustDomain.x} yDomain={medianRobustDomain.y}
              xLabel="field median (e⁻ / pixel)"
              yLabel="robust noise · 1.4826 × MAD (e⁻ / pixel)"
              series={relationSeries(medianRobust)} />
            <Legend items={bandLegend} />
            <Legend items={scatterSourceLegend} />
          </CardBody>
        </Card>

        <Card className="comparison-plot">
          <CardHead title="Inter-band pixel correlation"
            sub="Median within-field Pearson correlation; ribbons span the field-to-field 16–84% range." />
          <CardBody>
            <AdjustablePlot boundsLabel="Inter-band pixel correlation"
              xDomain={correlationX} yDomain={correlationY}
              xTicksForDomain={(value) => correlations.pairs
                .map((label, index) => ({ v: index, label }))
                .filter((tick) => tick.v >= value[0] && tick.v <= value[1])}
              xLabel="band pair"
              yLabel="within-field pixel correlation"
              series={correlationSeries} />
            <Legend items={[
              {
                color: C.comb, label: "synthetic LR · solid + circles",
                line: true, marker: "filled",
              },
              {
                color: C.mean, label: "real Euclid LR · dashed + diamonds",
                line: true, dash: true, marker: "diamond",
              },
            ]} />
          </CardBody>
        </Card>

        <Card className="comparison-plot comparison-plot--scale">
          <CardHead title="Scale-spectrum similarity"
            sub="Phase-free comparison of where each population places its fluctuation power. Intervals are bootstrap 16–84%." />
          <CardBody>
            <div className="scale-score-grid">
              {similarities.map(([band, similarity]) => (
                <div className="scale-score" key={band}
                  style={{ "--band-color": BAND_COLOR(band) } as CSSProperties}>
                  <div className="scale-score__band">{bandLabel(band)}</div>
                  <div>
                    <span>shape overlap</span>
                    <strong>{intervalLabel(similarity.overlap, 3)}</strong>
                  </div>
                  <div>
                    <span>fluctuation power · syn / real</span>
                    <strong>{intervalLabel(similarity.variance_ratio, 1)}</strong>
                  </div>
                </div>
              ))}
            </div>
            <AdjustablePlot boundsLabel="Scale-spectrum similarity"
              xDomain={similarityX}
              yDomain={ratioY} xScale="log"
              xTicksForDomain={logarithmicTicks}
              guides={[{ axis: "y", v: 0, dash: [4, 4] }]}
              xLabel="angular scale (arcsec / cycle)"
              yLabel="log₁₀ scale-share ratio"
              series={similaritySeries} />
            <Legend items={bandLegend} />
            <p className="scale-score__note">
              0 on the plot means equal scale share. Positive values mean the synthetic fields place more of their variance at that scale; negative values mean Euclid does.
            </p>
          </CardBody>
        </Card>
      </div>

      <div className="band-ledger">
        <div className="band-ledger__head">
          median field metrics · synthetic / real · e⁻ per pixel unless marked
        </div>
        {BANDS.map((item) => {
          const synthetic = comparison.fields.summary.synthetic[item];
          const real = comparison.fields.summary.real[item];
          return (
            <div className="band-ledger__row" key={item}>
              <strong>{item}</strong>
              <span><b>mean</b> {synthetic.mean.median.toPrecision(4)} / {real.mean.median.toPrecision(4)}</span>
              <span><b>std</b> {synthetic.std.median.toPrecision(4)} / {real.std.median.toPrecision(4)}</span>
              <span><b>robust σ</b> {synthetic.robust_std.median.toPrecision(4)} / {real.robust_std.median.toPrecision(4)}</span>
              <span><b>zero</b> {(100 * synthetic.zero_fraction.median).toFixed(2)}% / {(100 * real.zero_fraction.median).toFixed(2)}%</span>
              <span><b>negative</b> {(100 * synthetic.negative_fraction.median).toFixed(2)}% / {(100 * real.negative_fraction.median).toFixed(2)}%</span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function PopulationSummary({ title, eyebrow, population, tone }: {
  title: string; eyebrow: string; population: Population; tone: "synthetic" | "real";
}) {
  return (
    <Card className={`population-summary population-summary--${tone}`}>
      <CardHead title={title} sub={`${population.area_arcmin2.toFixed(2)} arcmin² sampled`} />
      <CardBody>
        <div className="population-summary__eyebrow eyebrow">{eyebrow}</div>
        <div className="population-summary__stats">
          <Stat k="catalog objects" v={population.objects.toLocaleString()} />
          {TYPE_ORDER.filter((kind) => kind in population.counts).map((kind) => (
            <Stat key={kind} k={`${
              tone === "real" && kind === "unknown"
                ? "non-stellar candidates"
                : kind === "galaxy" ? "galaxies" : `${kind}s`
            } / arcmin²`}
              v={(population.density_arcmin2[kind] ?? 0).toFixed(2)} />
          ))}
        </div>
      </CardBody>
    </Card>
  );
}

function priorValue(value: number): string {
  return Number.isFinite(value) ? value.toFixed(0) : "—";
}

function TngPriorPanel({ prior }: { prior: TngPrior }) {
  const catalog = prior.catalog;
  const visible = prior.visible;
  const current = catalog?.current_prior_arcmin2
    ?? visible?.current_prior_arcmin2
    ?? prior.dataset_prior_arcmin2
    ?? 60;
  const configured = prior.configured_prior_arcmin2 ?? current;
  const alphaLabel = Number.isFinite(prior.configured_mf_alpha)
    ? ` (α=${prior.configured_mf_alpha!.toFixed(2)})`
    : "";
  return (
    <section className="tng-prior">
      <header className="tng-prior__head">
        <div>
          <div className="eyebrow">normalization inference · common VIS detector</div>
          <h3>TNG draw-prior calibration</h3>
          <p>
            Displayed fields used {current.toFixed(0)} raw draws / arcmin².
            {" "}The generator is now configured for {configured.toFixed(0)} with one
            smooth mass prior{alphaLabel}.
          </p>
        </div>
        <Badge tone="good">selection-matched density</Badge>
      </header>

      <div className="tng-prior__readouts">
        {visible && (
          <article className="tng-prior__readout tng-prior__readout--visible">
            <span>common VIS detections</span>
            <strong>{priorValue(visible.fitted_prior_arcmin2)}</strong>
            <small>
              {priorValue(visible.interval_arcmin2.p16)}–
              {priorValue(visible.interval_arcmin2.p84)} / arcmin² ·
              {" "}{visible.synthetic_fields} / {visible.real_fields} fields
            </small>
            <p>{visible.synthetic_detected_density_arcmin2.toFixed(1)} synthetic /
              {" "}{visible.real_detected_density_arcmin2.toFixed(1)} Euclid
              detections / arcmin².</p>
          </article>
        )}
      </div>

      <div className="tng-prior__detail-grid">
        <Card className="tng-prior__decision">
          <CardHead title="What this constrains"
            sub="Matched detections constrain normalization, not the intrinsic luminosity-function shape." />
          <CardBody>
            <p>{prior.recommendation}</p>
            {!!prior.pilot_grid_arcmin2.length && (
              <div className="tng-prior__pilot">
                <span className="eyebrow">matched-seed pilot grid</span>
                <strong>{prior.pilot_grid_arcmin2.join(" · ")}</strong>
                <small>raw TNG draws / arcmin²</small>
              </div>
            )}
            {visible && (
              <dl className="tng-prior__diagnostics">
                <div><dt>truth recovered by detector</dt>
                  <dd>{(100 * visible.matched_truth_fraction).toFixed(1)}%</dd></div>
                <div><dt>selection</dt><dd>same VIS detector</dd></div>
              </dl>
            )}
          </CardBody>
        </Card>
      </div>
    </section>
  );
}

function ConeQuery({ api, onQueried }: { api: ApiPayload; onQueried: () => void }) {
  const defaults = api.availability.default_cone;
  const [ra, setRa] = useState(String(defaults.ra));
  const [dec, setDec] = useState(String(defaults.dec));
  const [radius, setRadius] = useState(defaults.radius_arcmin.toFixed(3));
  const query = useJob();
  const radiusNumber = Number(radius);
  const area = Number.isFinite(radiusNumber) ? Math.PI * radiusNumber ** 2 : 0;
  return (
    <Card className="cone-query">
      <CardHead title="Euclid population cone"
        sub="Query clean MER sources with four-band aperture photometry, classifier probabilities, morphology, blending and Gaia-match fields."
        right={<Badge tone={api.authenticated ? "good" : "warn"}>
          {api.authenticated ? "archive session ready" : "archive login required"}
        </Badge>} />
      <CardBody>
        <div className="cone-query__form">
          <Field label="RA · deg"><Input value={ra} type="number" onChange={setRa} step={0.0001} /></Field>
          <Field label="Dec · deg"><Input value={dec} type="number" onChange={setDec} step={0.0001} /></Field>
          <Field label="Radius · arcmin"><Input value={radius} type="number"
            onChange={setRadius} min={0.01} max={30} step={0.01} /></Field>
          <div className="cone-query__area">
            <span className="eyebrow">cone area</span>
            <strong>{area.toFixed(2)} arcmin²</strong>
            <small>{Math.abs(area - defaults.area_arcmin2) < 0.1
              ? "matches the 200-field synthetic footprint"
              : `${(area / defaults.area_arcmin2).toFixed(2)}× synthetic footprint`}</small>
          </div>
          <Button variant="primary" disabled={!api.authenticated || query.busy}
            onClick={() => query.run("/api/population-comparison/query-euclid", {
              ra, dec, radius_arcmin: radius,
            }, { onDone: (job) => { if (job.status !== "failed") onQueried(); } })}>
            {query.busy ? "Querying…" : "Query Euclid cone"}
          </Button>
        </div>
        {!api.authenticated && (
          <p className="cone-query__login">
            Open <NavLink to="/catalog">Catalog</NavLink> and log in to the Euclid archive first.
          </p>
        )}
        <JobProgressView job={query.job} error={query.error} />
      </CardBody>
    </Card>
  );
}

export default function PopulationComparisonPage() {
  const [includeTraining, setIncludeTraining] = useState(false);
  const resource = useResource<ApiPayload>(
    `/api/population-comparison?include_training=${includeTraining ? "1" : "0"}`,
    [includeTraining],
    { ttl: 10_000 },
  );
  const build = useJob();
  const trainingCatalog = useJob();
  const api = resource.data;
  const comparison = api?.comparison ?? null;
  const stale = !!comparison && !!api && (
    comparison.samples.synthetic.fields !== api.availability.synthetic.fields
    || comparison.samples.real.fields !== api.availability.real.fields
    || comparison.population.synthetic_field_count !== (
      includeTraining
        ? api.availability.synthetic.population_fields_with_training
        : api.availability.synthetic.population_fields
    )
    || !euclidMetaMatches(
      api.availability.euclid_catalog.meta,
      comparison.population.euclid_meta,
    )
  );

  const rebuild = () => build.run("/api/population-comparison/build", {}, {
    onDone: (job) => { if (job.status !== "failed") resource.reload(); },
  });
  const syncTrainingCatalog = () => trainingCatalog.run(
    "/api/population-comparison/sync-training-catalog",
    {},
    { onDone: (job) => {
      if (job.status !== "failed") {
        resource.reload();
      }
    } },
  );
  if (resource.loading && !api) {
    return <Page><Empty><Spinner /> reading local field census…</Empty></Page>;
  }
  if (!api) {
    return <Page><Empty>Population-comparison status is unavailable.</Empty></Page>;
  }

  return (
    <Page>
      <PageHead eyebrow="survey validation · pixels + sources"
        title="Field statistics"
        sub="Compare the image domain and the source populations that train the model against cached Euclid sky."
        right={<div className="comparison-actions">
          {stale && <Badge tone="warn">cache is behind local data</Badge>}
          <Button variant={comparison ? "default" : "primary"} disabled={build.busy}
            onClick={rebuild}>
            {build.busy ? "Measuring…" : comparison ? "Rebuild statistics" : "Measure local fields"}
          </Button>
        </div>} />

      <DatasetRuler availability={api.availability} comparison={comparison} />
      <JobProgressView job={build.job} error={build.error} />

      {!comparison ? (
        <Empty>
          <div className="comparison-empty">
            <strong>The local fields are ready; the compact statistics cache has not been built.</strong>
            <span>The pass streams {api.availability.synthetic.fields + api.availability.real.fields} fields one at a time and leaves the source FITS/TFRecords unchanged.</span>
            <Button variant="primary" onClick={rebuild} disabled={build.busy}>Measure local fields</Button>
          </div>
        </Empty>
      ) : (
        <>
          <FieldPlots comparison={comparison} />

          <section className="comparison-section comparison-section--population">
            <header className="comparison-section__head">
              <div>
                <div className="eyebrow">source domain · area-normalized</div>
                <h2>Population census</h2>
                <p>Synthetic truth and Euclid MER detections are shown as separate censuses because their selection functions differ.</p>
              </div>
              <div className="comparison-actions">
                <Checkbox checked={includeTraining}
                  disabled={!api.availability.synthetic.train_source_catalog}
                  onChange={setIncludeTraining}>
                  include training catalog
                </Checkbox>
                <Badge tone={includeTraining ? "warn" : "good"}>
                  {includeTraining
                    ? `train + test + validate · ${comparison.population.synthetic_field_count.toLocaleString()} fields`
                    : `test + validate · ${comparison.population.synthetic_field_count.toLocaleString()} fields`}
                </Badge>
                {!api.availability.synthetic.train_source_catalog && (
                  <Button size="sm" disabled={trainingCatalog.busy}
                    onClick={syncTrainingCatalog}>
                    {trainingCatalog.busy ? "Syncing…" : "Sync 6,400-field source catalog"}
                  </Button>
                )}
              </div>
            </header>
            <JobProgressView job={trainingCatalog.job} error={trainingCatalog.error} />
            {includeTraining && (
              <p className="catalog-classification-note">
                Training truth uses the legacy generator prior. It is included in
                the population histograms only; TNG calibration remains restricted
                to the regenerated 200-field test + validation catalogs.
              </p>
            )}
            <div className="population-summary-grid">
              <PopulationSummary title="Synthetic source truth" eyebrow="complete generated population"
                population={comparison.population.synthetic} tone="synthetic" />
              {comparison.population.euclid ? (
                <PopulationSummary title="Euclid MER catalog" eyebrow="detection-selected population"
                  population={comparison.population.euclid} tone="real" />
              ) : (
                <Card className="population-summary population-summary--empty">
                  <CardHead title="Euclid MER catalog" sub="No population cone is cached yet." />
                  <CardBody><Empty>Choose a cone below to add the real-source population.</Empty></CardBody>
                </Card>
              )}
            </div>
            {comparison.population.euclid_meta && (
              <p className="catalog-classification-note">
                <strong>Selection:</strong> synthetic truth contains all generated
                sources, while MER contains detected and deblended sources only.
                Their raw counts and parameter distributions are not direct
                population-shape comparisons.{" "}
                <strong>Classification:</strong> {comparison.population.euclid_meta.classification}.{" "}
                Unknown rows are plotted as non-stellar candidates alongside synthetic galaxies,
                not as confirmed galaxies. {comparison.population.euclid_meta.classification_note}
              </p>
            )}

            {comparison.population.tng_prior ? (
              <TngPriorPanel prior={comparison.population.tng_prior} />
            ) : comparison.population.euclid ? (
              <Card className="tng-prior tng-prior--empty">
                <CardHead title="TNG draw-prior calibration"
                  sub="Rebuild statistics to run the common VIS detector normalization." />
              </Card>
            ) : null}

            <ConeQuery api={api} onQueried={resource.reload} />
          </section>
        </>
      )}
    </Page>
  );
}
