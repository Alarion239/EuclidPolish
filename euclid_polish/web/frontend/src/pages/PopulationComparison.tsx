import { useEffect, useState, type CSSProperties } from "react";
import { NavLink } from "react-router-dom";
import Plot, { Legend, type PlotProps, type Series, type Tick } from "../charts/Plot";
import { C, categorical } from "../colors";
import { StepById } from "../fasrc";
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
type Histogram = {
  x: number[]; density: number[]; count: number; range?: [number, number];
};
type ComparisonClass = "nonstellar" | "star";
type SharedParameter = {
  label: string;
  unit: string;
  classes: Partial<Record<ComparisonClass, {
    synthetic: Histogram;
    euclid: Histogram;
  }>>;
};
type SharedPopulation = {
  parameters: Record<string, SharedParameter>;
  class_labels: Record<ComparisonClass, string>;
  density_unit: string;
};
type PhzProbabilityRelation = {
  mer_galaxy_probability: number[];
  mean_phz_galaxy_probability: (number | null)[];
  objects: number[];
  correlation: number | null;
  interpretation: string;
};
type TngColourConditioning = {
  available: boolean;
  note: string;
  colours: Record<string, {
    label: string; x: number[];
    series: Record<"current" | "phz_conditioned", {
      probability: number[]; count: number;
    }>;
  }>;
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
  detection_residual_arcmin2?: number;
  actionable?: boolean;
  transfer_compatibility?: {
    compatible: boolean; reason: string; source_fingerprints: string[];
    active_fingerprint?: string | null;
  };
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
  density_calibration?: CalibrationState;
  photometric_transfer?: CalibrationState;
  historical_incompatible_points?: Array<{
    density_arcmin2: number; job_id: string; offset_mag: number;
    magnitude_slope: number; scatter_mag: number;
  }>;
};
type CalibrationArtifact = {
  version?: number;
  valid?: boolean;
  validated?: boolean;
  warnings?: string[];
  coverage_notes?: string[];
  fingerprint?: string;
  generation?: {
    surface_density_arcmin2: number;
    vis_magnitude_min: number;
    vis_magnitude_max: number;
    morphology_assignment: string;
    position_process: string;
  };
  magnitude_plot?: {
    label: string;
    law: { x: number[]; density: number[] };
    observed?: { x: number[]; density: number[] };
    fit_interval: [number, number];
    sampling_interval: [number, number];
    extrapolated_interval: [number, number];
  };
  calibration_fingerprint?: string;
  recommended_density_arcmin2?: number | null;
  interval_arcmin2?: PriorInterval | null;
  response_points?: Array<{
    density_arcmin2: number;
    detected_density_arcmin2: number;
    isotonic_density_arcmin2: number;
  }>;
  transfer_fingerprint?: string;
  population?: {
    density_arcmin2: number; bright_gaia_density_arcmin2: number;
    magnitude_slope: number; mag_bright: number; mag_faint: number;
  };
  euclid_mapping?: { matched_stars: number };
  gaia?: { rows: number };
  diagnostics?: {
    star_density_per_cone: {
      x: number[]; observed: number[]; fitted: number[]; label: string; unit: string;
      x_label?: string;
      gaia_observed?: number[];
      gaia_fitted?: number[];
      fit_ranges?: { q1?: [number, number]; gaia?: [number, number] };
      statistics?: { mean?: number | null; std?: number | null;
        p16?: number | null; p50?: number | null; p84?: number | null };
    };
    parameters: Record<string, {
      x: number[]; observed: (number | null)[]; fitted: (number | null)[];
      label: string; unit: string; density_unit: string;
      observed_count: number; fitted_count: number;
      observed_label?: string;
      gaia_bright?: (number | null)[];
      gaia_bright_label?: string;
      euclid_weighted?: (number | null)[];
      euclid_weighted_label?: string;
      dirty_observed?: (number | null)[];
      dirty_observed_label?: string;
      posterior_predictive?: (number | null)[];
      posterior_predictive_label?: string;
      statistics?: {
        expected_count?: number | null; density_arcmin2?: number | null;
        mean?: number | null; std?: number | null;
        p16?: number | null; p50?: number | null; p84?: number | null;
        effective_n?: number | null;
        classification_sigma_count?: number | null;
        classification_sigma_density_arcmin2?: number | null;
      };
      dirty_statistics?: {
        mean?: number | null; std?: number | null;
        p16?: number | null; p50?: number | null; p84?: number | null;
        effective_n?: number | null;
      };
      posterior_predictive_statistics?: {
        mean?: number | null; std?: number | null;
        p16?: number | null; p50?: number | null; p84?: number | null;
        effective_n?: number | null;
      };
      observed_limit_mag?: number;
      extrapolation_note?: string;
    }>;
  };
  active?: boolean;
  coefficients?: {
    offset_mag: number; magnitude_slope: number; scatter_mag: number;
  };
  observation_model?: {
    completeness_m50: number; completeness_width_mag: number;
  };
  retained_detection_fraction?: number;
  euclid_detected_density_arcmin2?: number;
  classification_weighting?: {
    star_weight?: string;
    galaxy_weight?: string;
    missing_probability_rows?: number;
    invalid_probability_rows?: number;
  };
  euclid_cones?: unknown[];
  cosmos_generator_rows?: number;
  local_draws?: number;
  interval_arcmin2?: PriorInterval;
};
type CalibrationState = {
  candidate: CalibrationArtifact | null;
  active: CalibrationArtifact | null;
  is_active: boolean;
};
type GalaxyRecommendation = {
  recommendation_available: boolean;
  validated: boolean;
  warnings: string[];
  generator_parameters: {
    galaxy_density_arcmin2: number | null;
    cosmos_vis_offset_mag: number | null;
    cosmos_vis_magnitude_slope: number | null;
    cosmos_vis_scatter_mag: number | null;
  };
  observation_model_diagnostics: {
    completeness_m50?: number;
    completeness_width_mag?: number;
  };
  density_interval_arcmin2?: PriorInterval | null;
};
type JointFitSeries = {
  x: number[]; observed: (number | null)[]; model: (number | null)[];
  unit: string; label: string;
};
type JointFitDensity = {
  x: number[]; density: (number | null)[]; unit: string; label?: string;
};
type TngDrawMarginals = {
  redshift: JointFitDensity;
  magnitude: JointFitDensity;
  angular_radius: JointFitDensity;
  surface_density_arcmin2: number;
};
type PhzCoverage = {
  classification_fraction: number;
  valid_pdf_fraction: number;
  valid_physical_fraction: number;
  pathological_ssfr_weight: number;
  qso_overlap_fraction: number;
};
type CosmosEuclidFit = {
  version: 2 | 3;
  fingerprint: string;
  interpretation: string;
  inputs: {
    cosmos_population_rows: number;
    cosmos_measured_size_rows: number;
    euclid_cone_count: number;
    euclid_expected_galaxies_with_sizes: number;
    missing_probability_rows: number;
    missing_size_rows: number;
  };
  fit_quality: {
    valid: boolean;
    phz_valid?: boolean;
    phz_quality_gates?: Record<string, boolean>;
    cosmos_reduced_poisson_deviance: number;
    cosmos_reduced_negative_binomial_deviance?: number;
    euclid_reduced_poisson_deviance: number;
    warnings: string[];
  };
  phz_redshift_correction?: {
    z_edges: number[];
    vis_magnitude_edges: number[];
    observed_weighted_counts: number[][];
    baseline_weighted_counts: number[][];
    corrected_weighted_counts: number[][];
    median_absolute_fractional_residual: number;
    density_change_fraction: number;
    cross_validation: {
      mean_improvement_fraction: number;
      folds: Array<{
        test_cones: number[];
        baseline_deviance: number;
        corrected_deviance: number;
        improvement_fraction: number;
      }>;
    };
  };
  physical_conditionals?: {
    z_edges: number[];
    vis_magnitude_edges: number[];
    phz_rows: number;
    cosmos_rows: number;
    quenched_fraction: number[][];
    classes: Record<"quenched" | "star_forming", {
      mass_mean: number[][]; mass_sigma: number[][];
      ssfr_mean: number[][]; ssfr_sigma: number[][];
      effective_weight: number[][];
    }>;
  };
  phz_inputs?: {
    coverage: PhzCoverage;
    pdf_rows: number;
  };
  phz_quality_gates?: Record<string, boolean>;
  model: {
    luminosity_function: Record<string, number | number[]>;
    size_relation: Record<string, number | number[]>;
    euclid_response: Record<string, number | number[]>;
  };
  parameters: Array<{
    group: string; key: string; label: string;
    value: number; standard_error: number | null; unit: string;
  }>;
  diagnostics: {
    magnitude_counts: { cosmos: JointFitSeries; euclid: JointFitSeries };
    redshift: JointFitSeries;
    angular_radius: { cosmos: JointFitSeries; euclid: JointFitSeries };
    tng_draw: {
      full: TngDrawMarginals;
      comparison_window: TngDrawMarginals & {
        vis_magnitude_min: number;
        vis_magnitude_max: number;
      };
      definition: string;
    };
    median_radius_by_magnitude: {
      x: number[];
      cosmos_observed: (number | null)[]; cosmos_model: (number | null)[];
      euclid_observed: (number | null)[]; euclid_model: (number | null)[];
      unit: string;
    };
    surface_brightness: {
      x: number[];
      cosmos_observed: number[]; cosmos_model: number[];
      euclid_observed: number[]; euclid_model: number[];
      unit: string;
    };
    completeness: {
      magnitude: number[];
      by_radius_arcsec: Record<string, number[]>;
    };
  };
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
    shared: SharedPopulation | null;
    phz_probability_relation?: PhzProbabilityRelation | null;
    tng_colour_conditioning?: TngColourConditioning | null;
    tng_prior?: TngPrior | null;
    cosmos_euclid_fit?: CosmosEuclidFit | null;
    synthetic_splits?: string[];
    training_included?: boolean;
    calibration_splits?: string[];
    euclid_meta: {
      ra?: number; dec?: number; radius_arcmin: number; area_arcmin2: number;
      cone_count?: number; cones?: Array<{ star_id: string; ra: number; dec: number; rows: number }>;
      rows: number; limit?: number; limit_reached?: boolean; classification: string;
      catalog_version: number;
      counts: Record<string, number>;
      classification_note: string;
      photometry: string;
      phz_coverage?: PhzCoverage;
      phz_quality?: Record<string, boolean>;
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
  euclid_catalog: {
    cached: boolean;
    meta: Comparison["population"]["euclid_meta"];
    phz_pdf_current?: boolean;
  };
  field_area_arcmin2: number;
  default_cone: { ra: number; dec: number; radius_arcmin: number; area_arcmin2: number };
};
type ApiPayload = {
  comparison: Comparison | null;
  availability: Availability;
  authenticated: boolean;
  calibrations?: {
    brightness_transfer: CalibrationState;
    galaxy_density: CalibrationState;
    joint_galaxy: CalibrationState;
    stars: CalibrationState;
    galaxy_recommendation: GalaxyRecommendation;
  };
};

const EMPTY_CALIBRATION: CalibrationState = {
  candidate: null, active: null, is_active: false,
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

function PhzCalibrationPanel({
  fit, probabilityRelation, colourConditioning,
}: {
  fit: CosmosEuclidFit;
  probabilityRelation?: PhzProbabilityRelation | null;
  colourConditioning?: TngColourConditioning | null;
}) {
  const correction = fit.phz_redshift_correction;
  const physical = fit.physical_conditionals;
  const coverage = fit.phz_inputs?.coverage;
  if (!correction || !physical || !coverage) return null;
  const z = correction.z_edges.slice(0, -1).map(
    (lower, index) => 0.5 * (lower + correction.z_edges[index + 1]),
  );
  const sumMagnitude = (plane: number[][]) => plane.map(
    (row) => row.reduce((total, value) => total + value, 0),
  );
  const observed = sumMagnitude(correction.observed_weighted_counts);
  const baseline = sumMagnitude(correction.baseline_weighted_counts);
  const corrected = sumMagnitude(correction.corrected_weighted_counts);
  const physicalZ = physical.z_edges.slice(0, -1).map(
    (lower, index) => 0.5 * (lower + physical.z_edges[index + 1]),
  );
  const brightColumns = physical.vis_magnitude_edges.slice(0, -1)
    .map((lower, index) => ({ lower, upper: physical.vis_magnitude_edges[index + 1], index }))
    .filter(({ upper }) => upper <= 24.5);
  const averageColumns = (plane: number[][]) => plane.map((row) => {
    const values = brightColumns.map(({ index }) => row[index]).filter(Number.isFinite);
    return values.length
      ? values.reduce((total, value) => total + value, 0) / values.length
      : 0;
  });
  const gateEntries = Object.entries(fit.phz_quality_gates ?? {});
  const colour = colourConditioning?.colours.vis_y_color;
  return (
    <section className="atlas-section">
      <div className="section-head">
        <div><span className="eyebrow">Euclid PHZ</span>
          <h2>Redshift and physical-population calibration</h2></div>
        <p>PHZ changes conditional galaxy properties; MER still fixes total density and measured-size response.</p>
      </div>
      <div className="calibration-status-grid">
        <div><span className="eyebrow">classification</span>
          <strong>{(100 * coverage.classification_fraction).toFixed(1)}%</strong>
          <small>eligible MER galaxy weight with PHZ probabilities</small></div>
        <div><span className="eyebrow">redshift PDFs</span>
          <strong>{(100 * coverage.valid_pdf_fraction).toFixed(1)}%</strong>
          <small>{fit.phz_inputs?.pdf_rows.toLocaleString()} compacted PDFs</small></div>
        <div><span className="eyebrow">physical properties</span>
          <strong>{(100 * coverage.valid_physical_fraction).toFixed(1)}%</strong>
          <small>{physical.phz_rows.toLocaleString()} PHZ · {physical.cosmos_rows.toLocaleString()} COSMOS rows</small></div>
        <div><span className="eyebrow">activation gates</span>
          <strong>{fit.fit_quality.phz_valid ? "passed" : "blocked"}</strong>
          <small>{gateEntries.filter(([, passed]) => passed).length}/{gateEntries.length} PHZ gates passed</small></div>
      </div>
      <div className="parameter-atlas">
        <Card className="parameter-card">
          <CardHead title="PHZ redshift constraint"
            sub="probability-weighted Q1 PDFs · baseline and density-preserving correction" />
          <CardBody>
            <AdjustablePlot boundsLabel="PHZ redshift calibration"
              xDomain={domain(z)} yDomain={domain([...observed, ...baseline, ...corrected], true)}
              xLabel="photometric redshift" yLabel="weighted objects"
              series={[
                { x: z, y: observed, color: "#1267d6", mode: "scatter", marker: "ring", width: 2 },
                { x: z, y: baseline, color: "#242424", width: 2, dash: [6, 4] },
                { x: z, y: corrected, color: "#cf3d2e", width: 2.8 },
              ]} aspect={0.62} />
            <Legend items={[
              { color: "#1267d6", label: "Euclid PHZ PDFs", marker: "ring" },
              { color: "#242424", label: "uncorrected model", line: true, dash: true },
              { color: "#cf3d2e", label: "PHZ-corrected model", line: true },
            ]} />
          </CardBody>
        </Card>
        <Card className="parameter-card">
          <CardHead title="Galaxy activity and mass"
            sub="VIS<24.5 PHZ conditionals; COSMOS carries the faint continuation" />
          <CardBody>
            <AdjustablePlot boundsLabel="PHZ physical conditionals"
              xDomain={domain(physicalZ)} yDomain={[0, 1]}
              xLabel="redshift" yLabel="quenched fraction"
              series={[{
                x: physicalZ, y: averageColumns(physical.quenched_fraction),
                color: "#7a3db8", width: 2.8,
              }]} aspect={0.62} />
            <p className="scale-score__note">
              Pathological high-sSFR weight excluded: {coverage.pathological_ssfr_weight.toFixed(1)}.
              QSO overlap is diagnostic only: {(100 * coverage.qso_overlap_fraction).toFixed(1)}%.
            </p>
          </CardBody>
        </Card>
        {probabilityRelation && <Card className="parameter-card">
          <CardHead title="MER versus PHZ classification"
            sub="PHZ is diagnostic here; MER retains the generation density normalization" />
          <CardBody>
            <AdjustablePlot boundsLabel="MER and PHZ galaxy probabilities"
              xDomain={[0, 1]} yDomain={[0, 1]}
              xLabel="MER galaxy probability · 1 - POINT_LIKE_PROB"
              yLabel="mean PHZ_GAL_PROB"
              series={[{
                x: probabilityRelation.mer_galaxy_probability,
                y: probabilityRelation.mean_phz_galaxy_probability,
                color: "#008c68", width: 2.6, dots: true, marker: "filled",
              }]} aspect={0.62} />
            <p className="scale-score__note">
              Correlation: {probabilityRelation.correlation == null
                ? "not estimable" : probabilityRelation.correlation.toFixed(3)}.
            </p>
          </CardBody>
        </Card>}
        {colour && <Card className="parameter-card">
          <CardHead title="Rendered TNG colour response"
            sub="current donor assignment versus PHZ-conditioned activity and mass transport" />
          <CardBody>
            <AdjustablePlot boundsLabel="TNG colour conditioning"
              xDomain={domain(colour.x)}
              yDomain={domain([
                ...colour.series.current.probability,
                ...colour.series.phz_conditioned.probability,
              ], true)}
              xLabel={`${colour.label} rendered colour (AB mag)`}
              yLabel="fraction of TNG galaxies / bin"
              series={[
                { x: colour.x, y: colour.series.current.probability,
                  color: "#242424", width: 2, dash: [6, 4] },
                { x: colour.x, y: colour.series.phz_conditioned.probability,
                  color: "#7a3db8", width: 2.8 },
              ]} aspect={0.62} />
            <Legend items={[
              { color: "#242424", label: `current · ${colour.series.current.count}`, line: true, dash: true },
              { color: "#7a3db8", label: `PHZ-conditioned · ${colour.series.phz_conditioned.count}`, line: true },
            ]} />
            <p className="scale-score__note">{colourConditioning?.note}</p>
          </CardBody>
        </Card>}
      </div>
    </section>
  );
}

function CosmosEuclidDensityPanel({ fit, calibration }: {
  fit: CosmosEuclidFit; calibration: CalibrationArtifact | null;
}) {
  const diagnostics = fit.diagnostics;
  const tngFull = diagnostics.tng_draw.full;
  const magnitude = calibration?.magnitude_plot;
  const magnitudeValues = [
    ...(magnitude?.observed?.density ?? []),
    ...(magnitude?.law.density ?? []),
  ];
  const magnitudeLogValues = log10(magnitudeValues);
  const median = diagnostics.median_radius_by_magnitude;
  const surface = diagnostics.surface_brightness;
  return (
    <>
      <div className="publication-atlas-export">
        <div>
          <div className="eyebrow">presentation figure</div>
          <strong>Q1 brightness × staged TNG50 geometry atlas</strong>
          <span>Independent VIS 2FWHM brightness, redshift, and half-light-radius density.</span>
        </div>
        <div className="publication-atlas-export__actions">
          <a className="ui-btn ui-btn--primary"
            href="/view/population-atlas?format=png&dpi=300" download>
            Download PNG
          </a>
          <a className="ui-btn" href="/view/population-atlas?format=pdf" download>
            PDF
          </a>
          <a className="ui-btn" href="/view/population-atlas?format=svg" download>
            SVG
          </a>
        </div>
      </div>
      <div className="parameter-atlas">
      <Card className="parameter-card">
        <CardHead title="Independent goal 2FWHM brightness"
          sub="Q1 MER + PHZ VIS · straight in log density from 14 to 29; 28–29 is extrapolated" />
        <CardBody>
          {magnitude ? <>
            <AdjustablePlot boundsLabel="Q1 2FWHM brightness law"
              xDomain={magnitude.sampling_interval}
              yDomain={domain(magnitudeLogValues)}
              yTicksForDomain={(value) => ticks(value).map((tick) => ({
                ...tick, label: `10^${tick.label}`,
              }))}
              xLabel="VIS 2FWHM aperture magnitude [AB]"
              yLabel="objects / arcmin² / mag (log scale)"
              guides={[
                ...magnitude.fit_interval.map((v) => ({
                  axis: "x" as const, v, color: "#1267d6",
                  dash: [3, 4], width: 1.2, alpha: 0.7,
                })),
                { axis: "x", v: magnitude.extrapolated_interval[0],
                  color: "#7a3db8", dash: [7, 4], width: 1.2, alpha: 0.8 } as const,
              ]}
              series={[
                ...(magnitude.observed ? [{
                  x: magnitude.observed.x,
                  y: log10(magnitude.observed.density),
                  color: "#1267d6", mode: "scatter" as const,
                  marker: "ring" as const, width: 2,
                }] : []),
                { x: magnitude.law.x, y: log10(magnitude.law.density),
                  color: "#7a3db8", width: 3 },
              ]} aspect={0.62} />
            <Legend items={[
              ...(magnitude.observed ? [{
                color: "#1267d6", label: "Q1 MER + PHZ 2FWHM counts",
                marker: "ring" as const,
              }] : []),
              { color: "#7a3db8", label: "Q1-normalized straight law", line: true },
            ]} />
          </> : <Empty>Fit the cached Q1 VIS 2FWHM counts to draw the brightness law.</Empty>}
        </CardBody>
      </Card>
      <Card className="parameter-card">
        <CardHead title="Redshift distribution"
          sub={fit.phz_redshift_correction
            ? "COSMOS latent fit; Euclid PHZ correction is shown below"
            : "COSMOS constrains z; refresh the Euclid cache to add PHZ"} />
        <CardBody>
          <AdjustablePlot boundsLabel="COSMOS redshift distribution"
            xDomain={domain(diagnostics.redshift.x)}
            yDomain={domain([
              ...diagnostics.redshift.observed, ...diagnostics.redshift.model,
              ...tngFull.redshift.density,
            ], true)} xLabel="photometric redshift"
            yLabel={diagnostics.redshift.unit}
            series={[
              { x: diagnostics.redshift.x, y: diagnostics.redshift.observed,
                color: "#242424", mode: "scatter", marker: "ring", width: 2 },
              { x: diagnostics.redshift.x, y: diagnostics.redshift.model,
                color: "#008c68", width: 2.5 },
              { x: tngFull.redshift.x,
                y: tngFull.redshift.density,
                color: "#7a3db8", width: 3 },
            ]} aspect={0.62} />
          <Legend items={[
            { color: "#242424", label: "COSMOS observed", marker: "ring" },
            { color: "#008c68", label: "shared intrinsic fit", line: true },
            { color: "#7a3db8", label: "brightness-marginalized geometry", line: true },
          ]} />
        </CardBody>
      </Card>
      <Card className="parameter-card">
        <CardHead title="Angular-radius density"
          sub="COSMOS fitted Re · Euclid MER proxy · brightness-marginalized staged geometry" />
        <CardBody>
          <AdjustablePlot boundsLabel="Joint angular-size distribution"
            xDomain={domain([
              ...diagnostics.angular_radius.cosmos.x,
              ...diagnostics.angular_radius.euclid.x,
            ])}
            yDomain={domain([
              ...diagnostics.angular_radius.cosmos.observed,
              ...diagnostics.angular_radius.cosmos.model,
              ...diagnostics.angular_radius.euclid.observed,
              ...diagnostics.angular_radius.euclid.model,
              ...tngFull.angular_radius.density,
            ], true)}
            xLabel="log₁₀ angular radius / arcsec"
            yLabel="objects / arcmin² / dex"
            series={[
              { x: diagnostics.angular_radius.cosmos.x,
                y: diagnostics.angular_radius.cosmos.observed,
                color: "#008c68", mode: "scatter", marker: "ring", width: 2 },
              { x: diagnostics.angular_radius.cosmos.x,
                y: diagnostics.angular_radius.cosmos.model,
                color: "#008c68", width: 2.5 },
              { x: diagnostics.angular_radius.euclid.x,
                y: diagnostics.angular_radius.euclid.observed,
                color: "#1267d6", mode: "scatter", marker: "ring", width: 2 },
              { x: diagnostics.angular_radius.euclid.x,
                y: diagnostics.angular_radius.euclid.model,
                color: "#1267d6", width: 2.5 },
              { x: tngFull.angular_radius.x, y: tngFull.angular_radius.density,
                color: "#7a3db8", width: 3 },
            ]} aspect={0.62} />
          <Legend items={[
            { color: "#008c68", label: "COSMOS measured + response", line: true, marker: "ring" },
            { color: "#1267d6", label: "Euclid MER + response", line: true, marker: "ring" },
            { color: "#7a3db8", label: "brightness-marginalized geometry", line: true },
          ]} />
        </CardBody>
      </Card>
      <Card className="parameter-card">
        <CardHead title="Magnitude-conditioned radius"
          sub="the comparison that exposes size-selection and resolution floors" />
        <CardBody>
          <AdjustablePlot boundsLabel="Median radius by magnitude"
            xDomain={domain(median.x)}
            yDomain={domain([
              ...median.cosmos_observed, ...median.cosmos_model,
              ...median.euclid_observed, ...median.euclid_model,
            ], true)} xLabel="survey AB magnitude" yLabel="median radius / arcsec"
            series={[
              { x: median.x, y: median.cosmos_observed, color: "#242424",
                mode: "scatter", marker: "ring", width: 2 },
              { x: median.x, y: median.cosmos_model, color: "#008c68", width: 2.5 },
              { x: median.x, y: median.euclid_observed, color: "#1267d6",
                mode: "scatter", marker: "ring", width: 2 },
              { x: median.x, y: median.euclid_model, color: "#cf3d2e", width: 2.5 },
            ]} aspect={0.62} />
        </CardBody>
      </Card>
      <Card className="parameter-card">
        <CardHead title="Derived mean surface brightness"
          sub="computed from each survey magnitude and angular-size estimator" />
        <CardBody>
          <AdjustablePlot boundsLabel="Mean surface-brightness distribution"
            xDomain={domain(surface.x)} yDomain={domain([
              ...surface.cosmos_observed, ...surface.cosmos_model,
              ...surface.euclid_observed, ...surface.euclid_model,
            ], true)} xLabel="mag / arcsec²" yLabel="survey-specific density"
            series={[
              { x: surface.x, y: surface.cosmos_observed, color: "#242424",
                mode: "scatter", marker: "ring", width: 2 },
              { x: surface.x, y: surface.cosmos_model, color: "#008c68", width: 2.5 },
              { x: surface.x, y: surface.euclid_observed, color: "#1267d6",
                mode: "scatter", marker: "ring", width: 2 },
              { x: surface.x, y: surface.euclid_model, color: "#cf3d2e", width: 2.5 },
            ]} aspect={0.62} />
        </CardBody>
      </Card>
      <Card className="parameter-card">
        <CardHead title="Euclid completeness surface"
          sub="magnitude dependence changes continuously with angular size" />
        <CardBody>
          <AdjustablePlot boundsLabel="Euclid completeness by angular size"
            xDomain={domain(diagnostics.completeness.magnitude)} yDomain={[0, 1]}
            xLabel="VIS AB magnitude" yLabel="detection probability"
            series={Object.entries(diagnostics.completeness.by_radius_arcsec)
              .map(([radius, values], index) => ({
                x: diagnostics.completeness.magnitude, y: values,
                color: categorical(index), width: 2.5,
                label: `${radius} arcsec`,
              }))} aspect={0.62} />
          <Legend items={Object.keys(diagnostics.completeness.by_radius_arcsec)
            .map((radius, index) => ({
              color: categorical(index), label: `${radius} arcsec`, line: true,
            }))} />
        </CardBody>
      </Card>
      </div>
    </>
  );
}

function GalaxyCalibrationControls({ api, onChanged }: {
  api: ApiPayload; onChanged: () => void;
}) {
  const localFit = useJob();
  const activate = useJob();
  const fit = api.comparison?.population.cosmos_euclid_fit;
  const state = api.calibrations?.joint_galaxy ?? EMPTY_CALIBRATION;
  const starsActive = api.calibrations?.stars?.is_active ?? false;
  const candidate = state.candidate;
  const refresh = (job: { status: string }) => {
    if (job.status !== "failed") onChanged();
  };
  return (
    <Card className="calibration-workflow">
      <CardHead title="Joint analytical galaxy population"
        sub="One smooth Schechter × lognormal distribution, observed through separate COSMOS and Euclid response models." />
      <CardBody>
        <div className="population-flow" aria-label="Galaxy calibration flow">
          <div className="population-flow__step">
            <span>1 · latent distribution</span>
            <strong>Evolving Schechter LF</strong>
            <small>redshift × absolute-like F814W magnitude</small>
          </div>
          <div className="population-flow__step">
            <span>2 · intrinsic sizes</span>
            <strong>Lognormal R<sub>e</sub></strong>
            <small>smooth luminosity and redshift evolution</small>
          </div>
          <div className="population-flow__step">
            <span>3 · two surveys</span>
            <strong>COSMOS + {fit?.inputs.euclid_cone_count ?? "—"} Euclid cones</strong>
            <small>different bandpass, resolution and selection responses</small>
          </div>
          <div className="population-flow__step population-flow__step--result">
            <span>4 · current result</span>
            <strong>{fit ? (fit.fit_quality.valid ? "quality gate passed" : "diagnostic warnings") : "not fitted"}</strong>
            <small>{state.is_active
              ? "active for future synthetic jobs"
              : "review and activate before generation"}</small>
          </div>
        </div>
        <div className="calibration-explainer">
          <p>
            COSMOS constrains the redshift, luminosity and physical-size evolution.
            Euclid constrains the projected magnitude–size response and a
            surface-brightness-dependent completeness function used to recover
            geometry. Final brightness is drawn independently from the Q1 VIS
            2FWHM straight law; there is no row-by-row donor matching.
          </p>
          {fit && (
            <p className="fit-caution">
              <strong>Classification coverage:</strong>{" "}
              {fit.inputs.missing_probability_rows} Euclid rows without
              POINT_LIKE_PROB and {fit.inputs.missing_size_rows} without a usable
              size proxy were excluded; galaxy weight = 1 − POINT_LIKE_PROB.
            </p>
          )}
        </div>
        {fit && <div className="calibration-table-wrap">
          <table className="calibration-table">
            <thead><tr><th>Group</th><th>Parameter</th><th>Fit ± local SE</th><th>Unit</th></tr></thead>
            <tbody>
              {fit.parameters.map((parameter, index) => <tr key={parameter.key}>
                <td>{index === 0 || fit.parameters[index - 1].group !== parameter.group
                  ? parameter.group : ""}</td>
                <td>{parameter.label}</td>
                <td className="mono">{parameter.value.toPrecision(5)}{parameter.standard_error != null
                  ? ` ± ${parameter.standard_error.toPrecision(3)}` : ""}</td>
                <td>{parameter.unit || "—"}</td>
              </tr>)}
            </tbody>
          </table>
        </div>}
        {fit?.fit_quality.warnings.map((warning) => (
          <p className="fit-caution" key={warning}><strong>Fit note:</strong> {warning}</p>
        ))}
        <div className="calibration-actions">
          <Button disabled={localFit.busy}
            onClick={() => localFit.run(
              "/api/population-comparison/run-local-galaxy-calibration",
              {}, { onDone: refresh },
            )}>{localFit.busy ? "Fitting locally…" : "Refit joint distribution + plots"}</Button>
          <Button variant="primary"
            disabled={!candidate?.valid
              || (candidate.version === 2 && !candidate.validated)
              || activate.busy || localFit.busy}
            onClick={() => activate.run(
              "/api/population-comparison/activate-joint-galaxy",
              {}, { onDone: refresh },
            )}>{activate.busy
              ? "Activating…"
              : state.is_active ? "Re-activate TNG model" : "Use this TNG model"}</Button>
          <Badge tone={state.is_active ? "good" : "warn"}>
            {state.is_active ? "generation-ready" : "not active"}
          </Badge>
          {state.is_active && (
            <NavLink to="/sky" className="ui-btn ui-btn--primary">
              Open Sky job submission
            </NavLink>
          )}
        </div>
        {candidate?.generation && (
          <p className="calibration-plain-note">
            Submission draws {candidate.generation.surface_density_arcmin2.toFixed(2)} galaxies / arcmin²
            by first sampling brightness-marginalized R<sub>e</sub> and z, then
            drawing an independent Q1 2FWHM goal magnitude over VIS {candidate.generation.vis_magnitude_min.toFixed(0)}–{candidate.generation.vis_magnitude_max.toFixed(0)}.
            TNG morphology is conditioned on the staged geometry and PHZ/COSMOS
            physical state before one shared four-band aperture scale is applied.
          </p>
        )}
        <JobProgressView job={localFit.job} error={localFit.error} />
        <JobProgressView job={activate.job} error={activate.error} />
        {state.is_active && (
          <div className="fasrc-submit-boundary">
            <div className="fasrc-step-inline__head">
              <div>
                <span className="eyebrow">FASRC · synthetic generation</span>
                <h3>Submit fields with this TNG population</h3>
                <small>
                  The job embeds fingerprint {state.active?.fingerprint?.slice(0, 12)}…
                  and rechecks every population artifact before allocation.
                </small>
              </div>
            </div>
            {!starsActive && (
              <p className="fit-caution">
                <strong>Before submission:</strong> re-activate the current Gaia +
                Euclid stellar prior on the <NavLink to="/star-distribution">Star
                distribution page</NavLink>. No job will be queued while that
                calibration is stale.
              </p>
            )}
            <StepById stepId="synthetic_generate" embedded showHistory />
          </div>
        )}
      </CardBody>
    </Card>
  );
}

function StarCalibrationControls({ api, onChanged }: {
  api: ApiPayload; onChanged: () => void;
}) {
  const query = useJob();
  const activate = useJob();
  const state = api.calibrations?.stars ?? EMPTY_CALIBRATION;
  const candidate = state.candidate;
  const diagnostics = candidate?.diagnostics;
  const refresh = (job: { status: string }) => {
    if (job.status !== "failed") onChanged();
  };
  return (
    <Card className="calibration-workflow">
      <CardHead title="Gaia + Euclid stellar prior"
        sub="Query Gaia DR3 on the same footprints, exclude each deliberately selected centre, and fit an error-aware latent stellar locus." />
      <CardBody>
        <div className="calibration-status-grid">
          <div><span className="eyebrow">candidate</span>
            <strong>{candidate?.valid ? "valid" : candidate ? "quality warning" : "not fitted"}</strong>
            <small>{candidate?.gaia?.rows?.toLocaleString() ?? 0} Gaia sources · {candidate?.euclid_mapping?.matched_stars ?? 0} Euclid matches</small></div>
          <div><span className="eyebrow">generation state</span>
            <strong>{state.is_active ? "active" : "inactive"}</strong>
            <small>{candidate?.population
              ? `${candidate.population.density_arcmin2.toFixed(2)} stars / arcmin² · slope ${candidate.population.magnitude_slope.toFixed(3)}`
              : "legacy scalar fallback remains in use"}</small></div>
        </div>
        {candidate?.warnings?.[0] && (
          <p className="fit-caution"><strong>Fit note:</strong> {candidate.warnings[0]}</p>
        )}
        {candidate?.coverage_notes?.[0] && (
          <p className="fit-caution"><strong>Coverage:</strong> {candidate.coverage_notes[0]}</p>
        )}
        <div className="row" style={{ gap: "var(--s2)" }}>
          <Button variant="primary" disabled={query.busy || !api.availability.euclid_catalog.cached}
            onClick={() => query.run(
              "/api/population-comparison/query-gaia-stars", {}, { onDone: refresh },
            )}>{query.busy ? "Querying + fitting…" : "Query Q1 counts + fit star colours"}</Button>
          <Button disabled={!candidate?.valid || activate.busy}
            onClick={() => activate.run(
              "/api/population-comparison/activate-star-prior", {}, { onDone: refresh },
            )}>{state.is_active ? "Re-activate calibration" : "Activate calibration"}</Button>
        </div>
        <p className="calibration-plain-note">
          Q1 objects with POINT_LIKE_PROB ≥ 0.9 supply the 0.1-mag VIS count distribution, weighted by PHZ_STAR_PROB and normalized over the 63.1 deg² footprint. Matched Gaia sources supply only the latent colour/temperature relation.
        </p>
        {diagnostics && (
          <div className="publication-atlas-export">
            <div>
              <div className="eyebrow">presentation figure</div>
              <strong>Q1 PHZ × Gaia × Euclid stellar calibration</strong>
              <span>Footprint-normalized VIS counts and fitted, inferred, simulated-noise, and catalogue stellar colours.</span>
            </div>
            <div className="publication-atlas-export__actions">
              <a className="ui-btn ui-btn--primary"
                href="/view/star-population-calibration?format=png&dpi=300" download>
                Download PNG
              </a>
              <a className="ui-btn"
                href="/view/star-population-calibration?format=pdf" download>PDF</a>
              <a className="ui-btn"
                href="/view/star-population-calibration?format=svg" download>SVG</a>
            </div>
          </div>
        )}
        <JobProgressView job={query.job} error={query.error} />
        <JobProgressView job={activate.job} error={activate.error} />
        {diagnostics && (
          <div className="parameter-atlas" style={{ marginTop: "var(--s4)" }}>
            <Card className="parameter-card">
              <CardHead title={diagnostics.star_density_per_cone.label}
                sub="shared slope from native Gaia G_AB and Q1 PHZ VIS; separate intercepts, with Q1 normalizing the 12–25 generator" />
              <CardBody>
                <AdjustablePlot boundsLabel="Q1 and Gaia stellar straight laws"
                  xDomain={[12, 25]}
                  yDomain={domain([
                    ...log10(diagnostics.star_density_per_cone.observed),
                    ...log10(diagnostics.star_density_per_cone.fitted),
                    ...log10(diagnostics.star_density_per_cone.gaia_observed ?? []),
                    ...log10(diagnostics.star_density_per_cone.gaia_fitted ?? []),
                  ])}
                  yTicksForDomain={(value) => ticks(value).map((tick) => ({
                    ...tick, label: `10^${tick.label}`,
                  }))}
                  xLabel={diagnostics.star_density_per_cone.x_label ?? "native survey magnitude [AB]"}
                  yLabel={`${diagnostics.star_density_per_cone.unit} (log scale)`}
                  guides={Object.values(diagnostics.star_density_per_cone.fit_ranges ?? {})
                    .flatMap((interval) => interval.map((v) => ({
                      axis: "x" as const, v, color: "#7a3db8",
                      dash: [3, 4], width: 1, alpha: 0.55,
                    })))}
                  series={[
                    { x: diagnostics.star_density_per_cone.x,
                      y: log10(diagnostics.star_density_per_cone.observed),
                      color: categorical(2), mode: "scatter", marker: "ring", width: 2 },
                    { x: diagnostics.star_density_per_cone.x,
                      y: log10(diagnostics.star_density_per_cone.fitted),
                      color: categorical(0), width: 2.5 },
                    ...(diagnostics.star_density_per_cone.gaia_observed ? [{
                      x: diagnostics.star_density_per_cone.x,
                      y: log10(diagnostics.star_density_per_cone.gaia_observed),
                      color: categorical(4), mode: "scatter" as const,
                      marker: "diamond" as const, width: 1.7,
                    }] : []),
                    ...(diagnostics.star_density_per_cone.gaia_fitted ? [{
                      x: diagnostics.star_density_per_cone.x,
                      y: log10(diagnostics.star_density_per_cone.gaia_fitted),
                      color: categorical(4), width: 2.2, dash: [6, 3],
                    }] : []),
                  ]} aspect={0.62} />
                <Legend items={[
                  { color: categorical(2), label: "Q1 PHZ VIS counts", marker: "ring" },
                  { color: categorical(0), label: "Q1-normalized straight law", line: true },
                  { color: categorical(4), label: "native Gaia G_AB counts", marker: "diamond" },
                  { color: categorical(4), label: "Gaia-intercept shared-slope fit", line: true, dash: true },
                ]} />
              </CardBody>
            </Card>
            {Object.entries(diagnostics.parameters)
              .filter(([key]) => ["mag_vis", "vis_y", "y_j", "j_h",
                "bp_rp", "temperature_k"].includes(key))
              .map(([key, item]) => {
                const magnitudeCounts = key === "mag_vis";
                const gaiaBright = item.gaia_bright ?? [];
                const euclidWeighted = item.euclid_weighted ?? [];
                const dirtyObserved = item.dirty_observed ?? [];
                const posteriorPredictive = item.posterior_predictive ?? [];
                const comparisonValues = [
                  ...item.observed, ...item.fitted, ...gaiaBright,
                  ...euclidWeighted, ...dirtyObserved, ...posteriorPredictive,
                ];
                const stats = item.statistics;
                const dirtyStats = item.dirty_statistics;
                const format = (value: number | null | undefined) =>
                  value == null || !Number.isFinite(value) ? "—" : value.toFixed(3);
                return (
              <Card className="parameter-card" key={key}>
                  <CardHead title={item.label}
                    sub={magnitudeCounts
                      ? "fitted prior × Gaia bright × probability-weighted Euclid faint"
                    : "Fitted, inferred, noise-simulated, and catalogue stellar colours"} />
                <CardBody>
                  <AdjustablePlot boundsLabel={item.label}
                    xDomain={domain(item.x)}
                    yDomain={domain(comparisonValues, true)}
                    xLabel={item.unit} yLabel={item.density_unit}
                    series={[
                      { x: item.x, y: item.fitted, color: categorical(0),
                        mode: "histogram", fillAlpha: 0.26, width: 1.8 },
                      { x: item.x, y: item.observed, color: categorical(2),
                        mode: "histogram", hatch: true, dash: [8, 4],
                        fillAlpha: 0.02, width: 2.1 },
                      ...(magnitudeCounts && gaiaBright.length ? [{
                        x: item.x, y: gaiaBright, color: categorical(4),
                        mode: "line" as const, dash: [5, 4], width: 1.5,
                      }] : []),
                      ...(magnitudeCounts && euclidWeighted.length ? [{
                        x: item.x, y: euclidWeighted, color: categorical(5),
                        mode: "line" as const, width: 1.8,
                      }] : []),
                      ...(!magnitudeCounts && dirtyObserved.length ? [{
                        x: item.x, y: dirtyObserved, color: categorical(5),
                        mode: "histogram" as const, hatch: true, dash: [4, 4],
                        fillAlpha: 0.03, width: 1.8,
                      }] : []),
                      ...(!magnitudeCounts && posteriorPredictive.length ? [{
                        x: item.x, y: posteriorPredictive, color: categorical(4),
                        mode: "histogram" as const, fillAlpha: 0.16, width: 1.8,
                      }] : []),
                    ]}
                    guides={item.observed_limit_mag != null ? [{
                      axis: "x", v: item.observed_limit_mag,
                      color: categorical(2), dash: [5, 5], width: 1.4,
                    }] : undefined}
                    aspect={0.62} />
                  <Legend items={[
                    { color: categorical(0), label: "Fitted true-colour population", histogram: true, filled: true },
                    { color: categorical(2),
                      label: "Estimated true colours of observed stars",
                      histogram: true, hatch: true, dash: true },
                    ...(magnitudeCounts && gaiaBright.length ? [{
                      color: categorical(4),
                      label: item.gaia_bright_label ?? "Gaia bright component",
                      line: true, dash: true,
                    }] : []),
                    ...(magnitudeCounts && euclidWeighted.length ? [{
                      color: categorical(5),
                      label: item.euclid_weighted_label ?? "Euclid weighted point sources",
                      line: true,
                    }] : []),
                    ...(!magnitudeCounts && dirtyObserved.length ? [{
                      color: categorical(5),
                      label: "Raw Euclid catalogue colours",
                      histogram: true, hatch: true, dash: true,
                    }] : []),
                    ...(!magnitudeCounts && posteriorPredictive.length ? [{
                      color: categorical(4),
                      label: "Estimated colours with simulated Euclid noise",
                      histogram: true, filled: true,
                    }] : []),
                  ]} />
                  {stats && (
                    <div className="calibration-status-grid" style={{ marginTop: "var(--s3)" }}>
                      <div><span className="eyebrow">weighted mean ± sd</span>
                        <strong>{format(stats.mean)} ± {format(stats.std)}</strong></div>
                      <div><span className="eyebrow">p16 / p50 / p84</span>
                        <strong>{format(stats.p16)} / {format(stats.p50)} / {format(stats.p84)}</strong></div>
                      <div><span className="eyebrow">expected count / effective N</span>
                        <strong>{format(stats.expected_count)} / {format(stats.effective_n)}</strong></div>
                      {magnitudeCounts && <div><span className="eyebrow">density ± class. sigma</span>
                        <strong>{format(stats.density_arcmin2)} ± {format(stats.classification_sigma_density_arcmin2)}</strong></div>}
                    </div>
                  )}
                  {dirtyStats && (
                    <div className="calibration-status-grid" style={{ marginTop: "var(--s2)" }}>
                      <div><span className="eyebrow">simulated-noise mean ± sd</span>
                        <strong>{format(dirtyStats.mean)} ± {format(dirtyStats.std)}</strong></div>
                      <div><span className="eyebrow">observed p16 / p50 / p84</span>
                        <strong>{format(dirtyStats.p16)} / {format(dirtyStats.p50)} / {format(dirtyStats.p84)}</strong></div>
                      <div><span className="eyebrow">observed effective N</span>
                        <strong>{format(dirtyStats.effective_n)}</strong></div>
                    </div>
                  )}
                  {item.posterior_predictive_statistics && (
                    <div className="calibration-status-grid" style={{ marginTop: "var(--s2)" }}>
                      <div><span className="eyebrow">prediction mean ± sd</span>
                        <strong>{format(item.posterior_predictive_statistics.mean)} ± {format(item.posterior_predictive_statistics.std)}</strong></div>
                      <div><span className="eyebrow">prediction p16 / p50 / p84</span>
                        <strong>{format(item.posterior_predictive_statistics.p16)} / {format(item.posterior_predictive_statistics.p50)} / {format(item.posterior_predictive_statistics.p84)}</strong></div>
                    </div>
                  )}
                  {magnitudeCounts && item.extrapolation_note && (
                    <p className="fit-caution stellar-count-note">
                      <strong>Dashed boundary:</strong> {item.extrapolation_note}
                    </p>
                  )}
                </CardBody>
              </Card>
                );
              })}
          </div>
        )}
      </CardBody>
    </Card>
  );
}

const MAX_POPULATION_CONES = 24;

function ConeQuery({ api, onQueried }: { api: ApiPayload; onQueried: () => void }) {
  const defaults = api.availability.default_cone;
  const [count, setCount] = useState("6");
  const [radius, setRadius] = useState(defaults.radius_arcmin.toFixed(3));
  const query = useJob();
  const refreshSame = useJob();
  const analysis = useJob();
  const countNumber = Number(count);
  const radiusNumber = Number(radius);
  const valid = Number.isInteger(countNumber) && countNumber >= 1
    && countNumber <= MAX_POPULATION_CONES
    && Number.isFinite(radiusNumber) && radiusNumber > 0 && radiusNumber <= 30;
  const area = valid ? countNumber * Math.PI * radiusNumber ** 2 : 0;
  const cachedConeCount = api.availability.euclid_catalog.meta?.cone_count ?? 0;
  const fitButtonLabel = cachedConeCount
    ? `Fit + evaluate ${cachedConeCount} cached cone${cachedConeCount === 1 ? "" : "s"}`
    : "Fit + evaluate cached cones";
  return (
    <Card className="cone-query">
      <CardHead title="Random Euclid population cones"
        sub="Randomly select non-overlapping saved-star positions, then query clean MER sources around each one."
        right={<Badge tone={api.authenticated ? "good" : "warn"}>
          {api.authenticated ? "archive session ready" : "archive login required"}
        </Badge>} />
      <CardBody>
        <div className="cone-query__form">
          <Field label="Number of cones"><Input value={count} type="number"
            onChange={setCount} min={1} max={MAX_POPULATION_CONES} step={1} /></Field>
          <Field label="Radius · arcmin"><Input value={radius} type="number"
            onChange={setRadius} min={0.01} max={30} step={0.01} /></Field>
          <div className="cone-query__area">
            <span className="eyebrow">total cone area</span>
            <strong>{area.toFixed(2)} arcmin²</strong>
            <small>{Math.abs(area - defaults.area_arcmin2) < 0.1
              ? "matches the 200-field synthetic footprint"
              : `${(area / defaults.area_arcmin2).toFixed(2)}× synthetic footprint`}</small>
          </div>
          <Button variant="primary"
            disabled={!api.authenticated || query.busy || refreshSame.busy || analysis.busy || !valid}
            onClick={() => query.run(
              "/api/population-comparison/query-euclid-multi",
              { count, radius_arcmin: radius },
              { onDone: (job) => { if (job.status !== "failed") onQueried(); } },
            )}>
            {query.busy ? "Querying…" : `Query ${count || "0"} random cone${countNumber === 1 ? "" : "s"}`}
          </Button>
          <Button disabled={!api.availability.euclid_catalog.cached
              || query.busy || refreshSame.busy || analysis.busy}
            onClick={() => analysis.run(
              "/api/population-comparison/fit-euclid",
              {},
              { onDone: (job) => { if (job.status !== "failed") onQueried(); } },
            )}>
            {analysis.busy
              ? "Fitting…"
              : fitButtonLabel}
          </Button>
          <Button disabled={!api.authenticated || !cachedConeCount
              || query.busy || refreshSame.busy || analysis.busy}
            onClick={() => refreshSame.run(
              "/api/population-comparison/refresh-euclid-multi",
              {},
              { onDone: (job) => { if (job.status !== "failed") onQueried(); } },
            )}>
            {refreshSame.busy ? "Refreshing saved cones…" : "Refresh same saved cones"}
          </Button>
        </div>
        <p className="cone-query__login">
          Each run draws a fresh set of saved stars. Cones are kept at least
          two radii apart; the selected centers and random seed are saved with
          the catalog metadata. Up to {MAX_POPULATION_CONES} cones may be queried
          in one run.
        </p>
        {cachedConeCount > 0 && (
          <p className="cone-query__login">
            Use <strong>Refresh same saved cones</strong> after a catalog-schema update to
            retain the exact footprint while replacing raw flux/error measurements atomically.
          </p>
        )}
        {!api.authenticated && (
          <p className="cone-query__login">
            Open <NavLink to="/catalog">Catalog</NavLink> and log in to the Euclid archive first.
          </p>
        )}
        <JobProgressView job={query.job} error={query.error} />
        <JobProgressView job={analysis.job} error={analysis.error} />
        <JobProgressView job={refreshSame.job} error={refreshSame.error} />
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
                <p>What is in the generated fields, what Euclid detects, and the fit that connects them.</p>
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
              <p className="calibration-plain-note">
                Training truth is included in the census only. Calibration still uses
                the regenerated test + validation catalogs.
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
              <p className="population-census-note">
                <strong>Do not compare the two totals directly.</strong> Synthetic is
                complete source truth; Euclid is a detection/deblending catalog.
                “Non-stellar” is a detection-selected label; the active galaxy
                calibration instead uses the fractional extended-source weight
                1 − POINT_LIKE_PROB and is not a confirmed-galaxy count.
              </p>
            )}

            <GalaxyCalibrationControls api={api} onChanged={resource.reload} />

            {comparison.population.cosmos_euclid_fit && (
              <>
                <PhzCalibrationPanel
                  fit={comparison.population.cosmos_euclid_fit}
                  probabilityRelation={comparison.population.phz_probability_relation}
                  colourConditioning={comparison.population.tng_colour_conditioning}
                />
                <CosmosEuclidDensityPanel
                  fit={comparison.population.cosmos_euclid_fit}
                  calibration={api.calibrations?.joint_galaxy.candidate ?? null} />
              </>
            )}

            <ConeQuery api={api} onQueried={resource.reload} />

          </section>
        </>
      )}
    </Page>
  );
}
