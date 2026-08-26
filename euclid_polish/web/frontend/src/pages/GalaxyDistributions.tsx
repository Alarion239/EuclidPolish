import { useEffect, useMemo, useState } from "react";
import { NavLink } from "react-router-dom";
import Plot, { type Band, type Guide, type Series, type Tick } from "../charts/Plot";
import { useResource } from "../hooks";
import { JobProgressView, useJob } from "../jobs";
import {
  Badge, Button, Card, CardBody, CardHead, Checkbox, Chip, Empty,
  Page, PageHead, Spinner, Stat,
} from "../ui";
import "./galaxy-distributions.css";

type Curve = {
  x: number[];
  density: number[];
  weighted_count: number;
  definition: string;
};
type BrightnessCurve = Curve & {
  label: string;
  survey: "euclid" | "synthetic" | "cosmos" | "fit" | "generation";
  band: string;
  estimator: string;
  selection: string;
  default_on?: boolean;
  fit_interval?: [number, number];
  sampling_interval?: [number, number];
  extrapolated_interval?: [number, number];
  generation_interval?: [number, number];
  generation_bright_join_magnitudes?: [number, number, number];
  generation_bright_slopes?: [number, number, number];
  generation_main_slope?: number;
  generation_break_magnitude?: number;
  generation_density_cap_arcmin2_mag?: number;
  trust_boundary?: {
    kind: "empirical_5sigma";
    magnitude: number;
    lower_magnitude: number;
    upper_magnitude: number;
    snr: number;
    sample_size: number;
    estimator: string;
    selection: string;
    caveat: string;
  };
  observed_density_cap_arcmin2_mag?: number;
  observed_density_cap_magnitude?: number;
  observed_cumulative_density_to_boundary_arcmin2?: number;
  observed_cumulative_density_all_queried_bins_arcmin2?: number;
};
type RadiusCurve = Curve & {
  label: string;
  source: SourceKey;
  radius_type: "detection" | "kron" | "half_light" | "rendered_half_light" | "half_light_shape";
  units: string;
  normalization?: "surface_density" | "probability_density";
  default_on?: boolean;
};
type Parameter = {
  label: string;
  x_label: string;
  x_domain?: [number, number];
  density_unit: string;
  note: string;
  series: Partial<Record<SourceKey, Curve>>;
  photometry_series?: Record<string, BrightnessCurve>;
  photometry_missing?: string[];
  radius_series?: Record<string, RadiusCurve>;
  radius_missing?: string[];
};
type Source = {
  available?: boolean;
  detail?: string;
  rows?: number;
  area_arcmin2?: number;
  schema_version?: number;
  phz_pdf_rows?: number;
  physical_rows?: number;
  phz_pdf_source?: string;
  fingerprint?: string;
  active_fingerprint?: string;
  is_active?: boolean;
  validated?: boolean;
  version?: number;
  fields?: number;
  measured_radius_rows?: number;
  measured_radius_fraction?: number;
};
type FluxKey = "f1" | "f2" | "f3" | "f4";
type Payload = {
  version: number;
  stale: boolean;
  authenticated?: boolean;
  sources: Partial<Record<SourceKey, Source>>;
  q1_counts?: Q1Counts | null;
  q1_radius?: null | {
    complete: boolean;
    completed_queries: number;
    total_queries: number;
    magnitude_bins: unknown[];
    radius_bins: unknown[];
  };
  calibration: {
    candidate: null | {
      valid?: boolean;
      version: number;
      fingerprint: string;
      magnitude_law: {
        bright_join_magnitudes: [number, number, number];
        bright_slopes: [number, number, number];
        break_magnitude: number;
        straight_law: { slope: number };
      };
      radius_law: {
        slope_log10_arcsec_per_mag: number;
        scatter_dex: number;
        fitted_rows: number;
      };
      generation: {
        surface_density_arcmin2: number;
        differential_density_cap_arcmin2_mag: number;
        break_magnitude: number;
        fitted_surface_density_arcmin2: number;
        vis_magnitude_min: number;
        vis_magnitude_max: number;
        fitted_vis_magnitude_max: number;
        faint_end_policy: string;
      };
      plots?: {
        conditional_radius?: {
          magnitude: number[];
          observed_mean_log10_arcsec: Array<number | null>;
          model_mean_log10_arcsec: number[];
          model_core_low_log10_arcsec?: number[];
          model_core_high_log10_arcsec?: number[];
          model_low_log10_arcsec?: number[];
          model_high_log10_arcsec?: number[];
        };
      };
      provenance?: {
        object_catalog_used?: boolean;
      };
    };
    is_active: boolean;
    active?: null | { fingerprint?: string };
  };
  parameters: Record<string, Parameter>;
  joint_maps?: JointMaps;
  training_included?: boolean;
  training_variant_available?: boolean;
  availability?: {
    synthetic: {
      train_source_catalog: boolean;
      population_fields: number;
      population_fields_with_training: number;
    };
  };
};
type JointContour = {
  mass_fraction: number;
  level: number;
  paths: Array<{ x: number[]; y: number[] }>;
};
type JointMap = {
  key: "q1" | "synthetic" | "model";
  label: string;
  detail: string;
  color: string;
  density: number[][];
  surface_density_arcmin2: number;
  rows?: number | null;
  contours: JointContour[];
};
type JointMaps = {
  available: boolean;
  detail?: string;
  magnitude_edges: number[];
  log_radius_edges: number[];
  density_unit: string;
  contour_mass_fractions: number[];
  shared_density_max: number;
  maps: JointMap[];
};
type Q1Counts = {
  footprint_area_deg2: number;
  bright: number;
  faint: number;
  bin_width: number;
  query_count: number;
  completed_queries?: number;
  total_queries?: number;
  complete?: boolean;
  phases_completed?: number;
  phase_count?: number;
  fit_available?: boolean;
  fit_ready?: boolean;
  selection: string;
  apertures: Record<FluxKey, {
    label: string;
    selected_galaxies: number;
    expected_galaxies?: number;
    queried_bins?: number;
  }>;
};
type SourceKey = "euclid" | "synthetic" | "cosmos" | "fit";

const SOURCE: Record<SourceKey, { label: string; kicker: string; color: string }> = {
  euclid: { label: "Euclid MER + PHZ", kicker: "observed layer", color: "#2478d4" },
  synthetic: { label: "Generated source catalogues", kicker: "actual rendered draws", color: "#d39b32" },
  cosmos: { label: "COSMOS2025", kicker: "diagnostic only", color: "#00a078" },
  fit: { label: "Euclid joint fit", kicker: "generator candidate", color: "#e25543" },
};
const ORDER: SourceKey[] = ["euclid", "synthetic", "fit"];
const PARAMETER_ORDER = ["magnitude", "radius"];
const USEFUL_BRIGHTNESS_KEYS = new Set([
  "q1_vis_f2", "synthetic_vis_2fwhm", "generator_vis_f2",
]);
const USEFUL_RADIUS_KEYS = new Set([
  "euclid_sersic_re", "synthetic_requested_re",
  "synthetic_clean_half_light", "fit_re",
]);
const USEFUL_SHAPE_KEYS = new Set([
  "euclid_sersic_re_shape", "fit_re_q1_weighted_shape",
  "fit_re_full_generation_shape",
]);
const JOINT_DENSITY_COLOR = (t: number): string => {
  const value = Math.round(34 + 190 * Math.max(0, Math.min(1, t)));
  return `rgb(${value}, ${value}, ${value})`;
};
const BRIGHTNESS_COLORS = {
  euclid: ["#2478d4", "#31a7d8", "#33c4c9", "#786fd4", "#ad6bd8", "#e268a7"],
  synthetic: ["#d39b32", "#e1ad45", "#b98028", "#9d7134", "#7d6445"],
  cosmos: ["#00a078", "#35b45f", "#83b93d", "#c2a82f", "#dd843c", "#df5f51", "#c95285", "#9e68c7", "#697fd0", "#3c9eb9", "#4b8c70", "#8c8751", "#ad6d69", "#88738e", "#5f8290"],
  fit: ["#e25543", "#ef754b", "#c94d68", "#9f518d"],
  generation: ["#168f65"],
};
const RADIUS_COLORS: Record<string, string> = {
  euclid_detection: "#2478d4",
  euclid_kron: "#786fd4",
  euclid_sersic_re: "#31a7d8",
  cosmos_re: "#00a078",
  synthetic_requested_re: "#d39b32",
  synthetic_clean_half_light: "#f0b45b",
  fit_re: "#e25543",
  euclid_sersic_re_shape: "#2478d4",
  fit_re_q1_weighted_shape: "#e25543",
  fit_re_full_generation_shape: "#168f65",
};
const RADIUS_TYPE_LABEL: Record<RadiusCurve["radius_type"], string> = {
  detection: "Detection / deblender",
  kron: "Kron photometry",
  half_light: "Half-light radius",
  rendered_half_light: "Rendered image half-light",
  half_light_shape: "Normalized half-light shape",
};

const compact = (value?: number) => value == null ? "—" : new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value);
const ticks = ([a, b]: [number, number], count = 5): Tick[] => Array.from({ length: count }, (_, i) => {
  const v = a + (b - a) * i / (count - 1);
  return { v, label: Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(Math.abs(b - a) < 2 ? 2 : 1) };
});

const physicalTickLabel = (value: number): string => {
  const magnitude = Math.abs(value);
  if (magnitude === 0) return "0";
  if (magnitude >= 100_000 || magnitude < 0.0001) {
    return value.toExponential(1).replace(".0e", "e").replace("e+", "e");
  }
  const decimals = magnitude >= 100 ? 0
    : magnitude >= 10 ? 1
      : magnitude >= 0.1 ? 2
        : magnitude >= 0.01 ? 3
          : 4;
  const fixed = value.toFixed(decimals);
  return decimals ? fixed.replace(/\.?0+$/, "") : fixed;
};

// Curves stay linear in their already-transformed log10 coordinates. Only the
// tick text is inverted, so readers see physical values without changing plot
// geometry or interpolation.
const physicalLogTicks = (domain: [number, number], count = 5): Tick[] => ticks(domain, count)
  .map(({ v }) => ({ v, label: physicalTickLabel(10 ** v) }));

const physicalLogAxisLabel = (label: string): string => {
  const physical = label.replace(/^log₁₀\s*/, "");
  return physical.endsWith(")")
    ? `${physical.slice(0, -1)}, log scale)`
    : `${physical} (log scale)`;
};

const paddedDomain = (values: number[], minimumSpan: number): [number, number] => {
  const finite = values.filter(Number.isFinite);
  if (!finite.length) return [0, minimumSpan];
  const lo = Math.min(...finite), hi = Math.max(...finite);
  const span = Math.max(hi - lo, minimumSpan);
  const pad = 0.045 * span;
  return [lo - pad, hi + pad];
};

function SourceLedger({
  sources, includeTraining,
}: {
  sources: Payload["sources"];
  includeTraining: boolean;
}) {
  return <section className="galaxy-ledger" aria-label="Galaxy distribution data layers">
    {ORDER.map((key, index) => {
      const source = sources[key] ?? {};
      const fitted = key === "fit";
      return <article className="galaxy-ledger__source" key={key} style={{ "--source": SOURCE[key].color } as React.CSSProperties}>
        <div className="galaxy-ledger__number">0{index + 1}</div>
        <div>
          <div className="galaxy-ledger__kicker">{SOURCE[key].kicker}</div>
          <h2>{key === "synthetic"
            ? includeTraining ? "Generated train + test + validation" : "Generated test + validation"
            : SOURCE[key].label}</h2>
          <p>{source.detail ?? "No cached product yet."}</p>
        </div>
        <div className="galaxy-ledger__metrics">
          {source.available ? <Badge tone={fitted && !source.validated ? "warn" : "good"}>{fitted ? (source.is_active ? "active" : "candidate") : "cached"}</Badge> : <Badge tone="warn">missing</Badge>}
          {!fitted && <span><b>{compact(source.rows)}</b> objects</span>}
          {!fitted && <span><b>{source.area_arcmin2?.toFixed(1) ?? "—"}</b> arcmin²</span>}
          {key === "synthetic" && <span><b>{compact(source.measured_radius_rows)}</b> image-measured radii</span>}
          {key === "euclid" && <span><b>{compact(source.phz_pdf_rows)}</b> {source.phz_pdf_source === "summary_reconstruction" ? "reconstructed PDFs" : "PHZ PDFs"}</span>}
          {fitted && source.fingerprint && <span title={source.fingerprint}><b>{source.fingerprint.slice(0, 10)}</b> fingerprint</span>}
        </div>
      </article>;
    })}
  </section>;
}

function DensityPlot({ parameter }: { parameter: Parameter }) {
  const curves = ORDER.flatMap((key) => parameter.series[key] ? [[key, parameter.series[key]!] as const] : []);
  const logValues = curves.flatMap(([, curve]) => curve.density.filter((v) => v > 0).map(Math.log10));
  if (!curves.length || !logValues.length) return <Empty>No source has produced this marginal yet.</Empty>;
  const xValues = curves.flatMap(([, curve]) => curve.x);
  const xDomain: [number, number] = [Math.min(...xValues), Math.max(...xValues)];
  const lo = Math.floor(Math.min(...logValues) * 2) / 2;
  const hi = Math.ceil(Math.max(...logValues) * 2) / 2;
  const yDomain: [number, number] = [lo, hi <= lo ? lo + 1 : hi];
  const logarithmicX = parameter.x_label.startsWith("log₁₀");
  const series: Series[] = curves.map(([key, curve]) => ({
    x: curve.x,
    y: curve.density.map((v) => v > 0 ? Math.log10(v) : null),
    color: SOURCE[key].color,
    width: key === "fit" ? 2.7 : 1.8,
    dash: key === "cosmos" ? [7, 4] : undefined,
    marker: key === "euclid" ? "ring" : key === "synthetic" ? "filled" : undefined,
    dots: key === "euclid" || key === "synthetic",
    markerEvery: Math.max(1, Math.ceil(curve.x.length / 18)),
  }));
  return <>
    <Plot
      xDomain={xDomain} yDomain={yDomain}
      xTicks={logarithmicX ? physicalLogTicks(xDomain) : ticks(xDomain)}
      yTicks={physicalLogTicks(yDomain)}
      xLabel={logarithmicX ? physicalLogAxisLabel(parameter.x_label) : parameter.x_label}
      yLabel={`${parameter.density_unit} (log scale)`}
      series={series} aspect={0.62}
    />
    <div className="galaxy-plot__definitions">
      {curves.map(([key, curve]) => <span key={key}><i style={{ background: SOURCE[key].color }} />{SOURCE[key].label}: {curve.definition}</span>)}
    </div>
  </>;
}

function ApparentBrightnessPlot({ parameter }: { parameter: Parameter }) {
  const entries = Object.entries(parameter.photometry_series ?? {})
    .filter(([key]) => USEFUL_BRIGHTNESS_KEYS.has(key));
  const [selected, setSelected] = useState<string[]>(() => entries
    .filter(([, curve]) => curve.default_on)
    .map(([key]) => key));
  const colorByKey = useMemo(() => {
    const indices = { euclid: 0, synthetic: 0, cosmos: 0, fit: 0, generation: 0 };
    return Object.fromEntries(entries.map(([key, curve]) => {
      const palette = BRIGHTNESS_COLORS[curve.survey];
      const color = palette[indices[curve.survey] % palette.length];
      indices[curve.survey] += 1;
      return [key, color];
    }));
  }, [parameter.photometry_series]);
  const visible = entries.filter(([key]) => selected.includes(key));
  const logValues = visible.flatMap(([, curve]) => curve.density
    .filter((value) => value > 0).map(Math.log10));
  const xValues = visible.flatMap(([, curve]) => curve.x);
  const toggle = (key: string) => setSelected((current) => (
    current.includes(key) ? current.filter((item) => item !== key) : [...current, key]
  ));
  const surveyEntries = (survey: BrightnessCurve["survey"]) => entries
    .filter(([, curve]) => curve.survey === survey);
  const q1Curve = entries.find(([key]) => key === "q1_vis_f2")?.[1];
  const generationCurve = entries.find(([key]) => key === "generator_vis_f2")?.[1];
  const trustBoundary = q1Curve?.trust_boundary ?? generationCurve?.trust_boundary;
  const observedPeak = q1Curve?.observed_density_cap_arcmin2_mag
    ?? generationCurve?.observed_density_cap_arcmin2_mag;
  const observedPeakMagnitude = q1Curve?.observed_density_cap_magnitude
    ?? generationCurve?.observed_density_cap_magnitude;
  const observedCumulativeToBoundary = q1Curve?.observed_cumulative_density_to_boundary_arcmin2
    ?? generationCurve?.observed_cumulative_density_to_boundary_arcmin2;
  const observedCumulativeAll = q1Curve?.observed_cumulative_density_all_queried_bins_arcmin2
    ?? generationCurve?.observed_cumulative_density_all_queried_bins_arcmin2;
  const generationCap = generationCurve?.generation_density_cap_arcmin2_mag;

  let plot = <Empty>Select at least one catalogue measurement to draw.</Empty>;
  if (visible.length && logValues.length && xValues.length) {
    const xDomain: [number, number] = [Math.min(...xValues), Math.max(...xValues)];
    const lo = Math.floor(Math.min(...logValues) * 2) / 2;
    const hi = Math.ceil(Math.max(...logValues) * 2) / 2;
    const yDomain: [number, number] = [lo, hi <= lo ? lo + 1 : hi];
    const guides: Guide[] = visible.flatMap(([, curve]) => [
      ...(curve.fit_interval ?? []).map((value) => ({
        axis: "x" as const, v: value, color: "#2478d4",
        dash: [3, 4], width: 1, alpha: 0.65,
      })),
      ...(curve.extrapolated_interval ? [{
        axis: "x" as const, v: curve.extrapolated_interval[0],
        color: "#e25543", dash: [7, 4], width: 1.2, alpha: 0.75,
      }] : []),
      ...(curve.generation_bright_join_magnitudes ?? []).map((join) => ({
        axis: "x" as const, v: join,
        color: "#d39b32", dash: [5, 3], width: 1.5, alpha: 0.9,
      })),
      ...(curve.generation_break_magnitude != null ? [{
        axis: "x" as const, v: curve.generation_break_magnitude,
        color: "#168f65", dash: [2, 3], width: 1.8, alpha: 0.95,
      }] : []),
    ]);
    if (trustBoundary && trustBoundary.magnitude >= xDomain[0]
      && trustBoundary.magnitude <= xDomain[1]) {
      guides.push({
        axis: "x", v: trustBoundary.magnitude,
        color: "#31a7d8", width: 2.2, alpha: 1,
        label: `MER ${trustBoundary.snr}σ median ${trustBoundary.magnitude.toFixed(2)}`,
        labelSide: "after",
      });
    }
    if (observedPeak && observedPeak > 0) {
      const peakLog = Math.log10(observedPeak);
      if (peakLog >= yDomain[0] && peakLog <= yDomain[1]) guides.push({
        axis: "y", v: peakLog,
        color: "#2478d4", dash: [6, 3], width: 1.2, alpha: 0.85,
        label: `Q1 observed max ${observedPeak.toFixed(1)}`,
        labelSide: "before",
      });
    }
    const bands: Band[] = [];
    if (trustBoundary) {
      const lower = Math.max(xDomain[0], Math.min(xDomain[1], trustBoundary.lower_magnitude));
      const upper = Math.max(xDomain[0], Math.min(xDomain[1], trustBoundary.upper_magnitude));
      const turnover = Math.max(xDomain[0], Math.min(
        xDomain[1], observedPeakMagnitude ?? trustBoundary.magnitude,
      ));
      if (turnover > xDomain[0]) bands.push({
        axis: "x", from: xDomain[0], to: turnover,
        color: "#2478d4", alpha: 0.07, label: "Q1 COUNT SUPPORT TO TURNOVER",
      });
      if (upper > lower) bands.push({
        axis: "x", from: lower, to: upper,
        color: "#31a7d8", alpha: 0.16, label: "MER 5σ P16–P84",
      });
      if (upper < xDomain[1]) bands.push({
        axis: "x", from: upper, to: xDomain[1],
        color: "#e25543", alpha: 0.05, hatch: true, label: `BEYOND MER ${trustBoundary.snr}σ RANGE`,
      });
    } else if (observedPeakMagnitude != null) {
      const turnover = Math.max(xDomain[0], Math.min(xDomain[1], observedPeakMagnitude));
      if (turnover > xDomain[0]) bands.push({
        axis: "x", from: xDomain[0], to: turnover,
        color: "#2478d4", alpha: 0.07, label: "Q1 COUNT-SUPPORTED",
      });
      if (turnover < xDomain[1]) bands.push({
        axis: "x", from: turnover, to: xDomain[1],
        color: "#e25543", alpha: 0.05, hatch: true, label: "BEYOND Q1 TURNOVER",
      });
    }
    plot = <>
      <Plot
        xDomain={xDomain} yDomain={yDomain}
        xTicks={ticks(xDomain, 6)} yTicks={physicalLogTicks(yDomain, 6)}
        xLabel={parameter.x_label}
        yLabel={`${parameter.density_unit} (log scale)`}
        bands={bands}
        guides={guides}
        series={visible.map(([key, curve]) => ({
          x: curve.x,
          y: curve.density.map((value) => value > 0 ? Math.log10(value) : null),
          color: colorByKey[key],
          width: curve.survey === "generation" ? 3.2
            : curve.survey === "fit" ? 2.7 : 2.0,
          dash: curve.survey === "cosmos" ? [7, 4]
            : curve.survey === "fit" ? [3, 3] : undefined,
          marker: curve.survey === "synthetic" ? "filled" : undefined,
          dots: curve.survey === "synthetic",
          markerEvery: Math.max(1, Math.ceil(curve.x.length / 18)),
        }))}
        aspect={0.36}
      />
      <div className="brightness-disclosure">
        {visible.map(([key, curve]) => <div key={key}>
          <i style={{ background: colorByKey[key] }} />
          <span><b>{curve.label}</b><small>{curve.band} · {curve.estimator}</small></span>
          <em>{curve.fit_interval
            ? curve.generation_bright_join_magnitudes?.length
              ? `fixed joins VIS ${curve.generation_bright_join_magnitudes.map((value) => value.toFixed(2)).join(" / ")} · bridge slopes ${curve.generation_bright_slopes?.map((value) => value.toFixed(3)).join(" / ") ?? "—"}; main ${curve.generation_main_slope?.toFixed(3) ?? "—"} dex mag⁻¹ · flat from VIS ${curve.generation_break_magnitude?.toFixed(2)} to ${curve.generation_interval?.[1].toFixed(0)}`
              : `fit ${curve.fit_interval[0].toFixed(2)}–${curve.fit_interval[1].toFixed(2)} · law ${curve.sampling_interval?.[0].toFixed(0)}–${curve.sampling_interval?.[1].toFixed(0)}`
            : `${compact(curve.weighted_count)} weighted objects`}</em>
        </div>)}
      </div>
    </>;
  }

  return <div className="brightness-comparison">
    <div className="brightness-warning">
      <strong>One fitted brightness coordinate.</strong>
      <span>Only VIS 2FWHM is shown: the PHZ-weighted Q1 aggregate, the actual galaxies in the current test and validation fields, and the active generation law. Total-flux, Kron, other-aperture, and F814W diagnostics are intentionally omitted here.</span>
    </div>
    {trustBoundary && observedPeak != null && observedPeakMagnitude != null && <div className="brightness-trust" aria-label="Euclid magnitude support and five-sigma boundary">
      <section className="brightness-trust__observed">
        <small>Q1 count turnover</small>
        <b>VIS {observedPeakMagnitude.toFixed(2)}</b>
        <span>The observed differential density peaks at {observedPeak.toFixed(1)} galaxies / arcmin² / mag, then turns over.</span>
        {observedCumulativeToBoundary != null && <span><strong>Integrated:</strong> {observedCumulativeToBoundary.toFixed(1)} galaxies / arcmin² through the median 5σ boundary{observedCumulativeAll != null ? `; ${observedCumulativeAll.toFixed(1)} across all queried bins` : ""}.</span>}
      </section>
      <section className="brightness-trust__depth">
        <small>empirical MER {trustBoundary.snr}σ limit</small>
        <b>VIS {trustBoundary.magnitude.toFixed(2)} <i>({trustBoundary.lower_magnitude.toFixed(2)}–{trustBoundary.upper_magnitude.toFixed(2)})</i></b>
        <span>Median and 16th–84th percentiles from {compact(trustBoundary.sample_size)} selected rows using {trustBoundary.estimator}. {trustBoundary.caveat}</span>
        <span><strong>Selection:</strong> {trustBoundary.selection}.</span>
      </section>
      <section className="brightness-trust__ceiling">
        <small>generation-law ceiling</small>
        <b>{generationCap?.toFixed(1) ?? "—"} galaxies / arcmin² / mag</b>
        <span>{generationCap != null && generationCap > observedPeak
          ? `The current law reaches ${(generationCap / observedPeak).toFixed(1)}× the observed Q1 peak; the trust overlay does not change the sampler.`
          : "The ceiling matches the observed Q1 peak; its plateau beyond the MER 5σ range remains an explicit unsupported extrapolation."}</span>
      </section>
    </div>}
    <div className="brightness-controls">
      {(["euclid", "synthetic", "cosmos", "fit", "generation"] as const).map((survey) => {
        const group = surveyEntries(survey);
        if (!group.length) return null;
        return <section key={survey}>
          <header>
            <div><span>{survey === "euclid" ? "Euclid MER" : survey === "synthetic" ? "Generated fields" : survey === "cosmos" ? "COSMOS2025" : survey === "fit" ? "Q1 curve fits" : "Generation law"}</span><small>{survey === "euclid" ? "VIS · solid measurements" : survey === "synthetic" ? "selected source catalogues · point markers · curve-specific coverage" : survey === "cosmos" ? "HST/ACS F814W · long dashes" : survey === "fit" ? "VIS · short-dashed local fits" : "VIS · three-segment bright bridge/main/flat law"}</small></div>
            <div>
              <Button size="sm" variant="ghost" onClick={() => setSelected((current) => Array.from(new Set([...current, ...group.map(([key]) => key)])))}>all</Button>
              <Button size="sm" variant="ghost" onClick={() => setSelected((current) => current.filter((key) => !group.some(([candidate]) => candidate === key)))}>none</Button>
            </div>
          </header>
          <div className="brightness-controls__chips">{group.map(([key, curve]) => <Chip
            key={key} on={selected.includes(key)} onClick={() => toggle(key)}
            dot={colorByKey[key]}
            title={`${curve.band}; ${curve.estimator}. Selection: ${curve.selection}`}
          >{curve.label}</Chip>)}</div>
          {survey === "cosmos" && parameter.photometry_missing?.map((message) =>
            <p className="brightness-controls__missing" key={message}>{message}</p>)}
        </section>;
      })}
    </div>
    <div className="brightness-plot">{plot}</div>
  </div>;
}

function RadiusPlot({ parameter }: { parameter: Parameter }) {
  const entries = Object.entries(parameter.radius_series ?? {})
    .filter(([key]) => USEFUL_RADIUS_KEYS.has(key));
  const curveByKey = Object.fromEntries(entries) as Record<string, RadiusCurve>;
  const normalizationOf = (curve?: RadiusCurve) => curve?.normalization ?? "surface_density";
  const [selected, setSelected] = useState<string[]>(() => (
    entries.map(([key]) => key)
  ));
  const visible = entries.filter(([key]) => selected.includes(key));
  const toggle = (key: string) => setSelected((current) => {
    if (current.includes(key)) return current.filter((item) => item !== key);
    const normalization = normalizationOf(curveByKey[key]);
    return [
      ...current.filter(
        (item) => normalizationOf(curveByKey[item]) === normalization,
      ),
      key,
    ];
  });
  const grouped = ([
    "half_light", "rendered_half_light",
  ] as RadiusCurve["radius_type"][])
    .map((radiusType) => [radiusType, entries.filter(([, curve]) => curve.radius_type === radiusType)] as const)
    .filter(([, group]) => group.length);
  const logValues = visible.flatMap(([, curve]) => curve.density
    .filter((value) => value > 0).map(Math.log10));
  const xValues = visible.flatMap(([, curve]) => curve.x);
  const probabilityDensity = visible.length > 0 && visible.every(
    ([, curve]) => normalizationOf(curve) === "probability_density",
  );
  const selectGroup = (group: [string, RadiusCurve][]) => {
    const normalization = normalizationOf(group[0]?.[1]);
    setSelected((current) => Array.from(new Set([
      ...current.filter(
        (key) => normalizationOf(curveByKey[key]) === normalization,
      ),
      ...group.map(([key]) => key),
    ])));
  };

  let plot = <Empty>Select at least one radius observable to draw.</Empty>;
  if (visible.length && logValues.length && xValues.length) {
    const xDomain: [number, number] = parameter.x_domain
      ? [parameter.x_domain[0], parameter.x_domain[1]]
      : [Math.min(...xValues), Math.max(...xValues)];
    const lo = Math.floor(Math.min(...logValues) * 2) / 2;
    const hi = Math.ceil(Math.max(...logValues) * 2) / 2;
    const yDomain: [number, number] = [lo, hi <= lo ? lo + 1 : hi];
    plot = <>
      <Plot
        xDomain={xDomain} yDomain={yDomain}
        xTicks={physicalLogTicks(xDomain, 6)} yTicks={physicalLogTicks(yDomain, 6)}
        xLabel={physicalLogAxisLabel(parameter.x_label)}
        yLabel={probabilityDensity
          ? "normalized probability / dex (log scale)"
          : `${parameter.density_unit} (log scale)`}
        series={visible.map(([key, curve]) => ({
          x: curve.x,
          y: curve.density.map((value) => value > 0 ? Math.log10(value) : null),
          color: RADIUS_COLORS[key] ?? SOURCE[curve.source].color,
          width: curve.source === "fit" ? 2.7 : 1.9,
          dash: key === "fit_re_full_generation_shape" ? [3, 3]
            : curve.source === "cosmos" ? [7, 4] : undefined,
          marker: curve.source === "euclid" ? "ring"
            : curve.source === "synthetic" ? "filled" : undefined,
          dots: curve.source === "euclid" || curve.source === "synthetic",
          markerEvery: Math.max(1, Math.ceil(curve.x.length / 18)),
        }))}
        aspect={0.38}
      />
      <div className="radius-disclosure">
        {visible.map(([key, curve]) => <div key={key}>
          <i style={{ background: RADIUS_COLORS[key] ?? SOURCE[curve.source].color }} />
          <span><b>{curve.label}</b><small>{curve.definition}</small></span>
          <em>{normalizationOf(curve) === "probability_density"
            ? "unit integral"
            : `${compact(curve.weighted_count)} weighted objects`}</em>
        </div>)}
      </div>
    </>;
  }

  return <div className="radius-comparison">
    <div className="radius-warning">
      <strong>Surface-density radii only.</strong>
      <span>The Q1 circularized Sérsic Rₑ, requested generated Sérsic Rₑ, clean-image curve-of-growth half-light radius, and active model marginal stay distinct. Detection, Kron, COSMOS, and normalized shapes are excluded from this panel.</span>
    </div>
    <div className="radius-controls">
      {grouped.map(([radiusType, group]) => <section key={radiusType}>
        <header>
          <div><span>{RADIUS_TYPE_LABEL[radiusType]}</span><small>{radiusType === "detection" ? "diagnostic only" : radiusType === "kron" ? "diagnostic only" : radiusType === "rendered_half_light" ? "measured on clean generated images · isolated subset" : radiusType === "half_light_shape" ? "normalized Q1 and candidate magnitude mixes" : "catalogue and requested geometry"}</small></div>
          <div>
            <Button size="sm" variant="ghost" onClick={() => selectGroup([...group])}>all</Button>
            <Button size="sm" variant="ghost" onClick={() => setSelected((current) => current.filter((key) => !group.some(([candidate]) => candidate === key)))}>none</Button>
          </div>
        </header>
        <div className="radius-controls__chips">{group.map(([key, curve]) => <Chip
          key={key} on={selected.includes(key)} onClick={() => toggle(key)}
          dot={RADIUS_COLORS[key] ?? SOURCE[curve.source].color}
          title={`${curve.units}; ${curve.definition}`}
        >{curve.label}</Chip>)}</div>
        {radiusType === "half_light" && parameter.radius_missing?.map((message) =>
          <p className="brightness-controls__missing" key={message}>{message}</p>)}
      </section>)}
    </div>
    <div className="radius-plot">{plot}</div>
  </div>;
}

function RadiusShapePlot({ parameter }: { parameter: Parameter }) {
  const entries = Object.entries(parameter.radius_series ?? {})
    .filter(([key]) => USEFUL_SHAPE_KEYS.has(key));
  const logValues = entries.flatMap(([, curve]) => curve.density
    .filter((value) => value > 0).map(Math.log10));
  if (!entries.length || !logValues.length) {
    return <Empty>Build the Q1 radius aggregate and candidate model first.</Empty>;
  }
  const xValues = entries.flatMap(([, curve]) => curve.x);
  const xDomain: [number, number] = parameter.x_domain
    ? [parameter.x_domain[0], parameter.x_domain[1]]
    : [Math.min(...xValues), Math.max(...xValues)];
  const lo = Math.floor(Math.min(...logValues) * 2) / 2;
  const hi = Math.ceil(Math.max(...logValues) * 2) / 2;
  const yDomain: [number, number] = [lo, hi <= lo ? lo + 1 : hi];
  return <div className="shape-comparison">
    <div className="radius-warning">
      <strong>Shape comparison, normalized separately.</strong>
      <span>Each curve integrates to one over log-radius. The solid model uses the observed Q1 magnitude mix; the dashed model includes the full faint extension used for generation.</span>
    </div>
    <div className="radius-plot">
      <Plot
        xDomain={xDomain} yDomain={yDomain}
        xTicks={physicalLogTicks(xDomain, 6)} yTicks={physicalLogTicks(yDomain, 6)}
        xLabel={physicalLogAxisLabel(parameter.x_label)}
        yLabel="normalized probability / dex (log scale)"
        series={entries.map(([key, curve]) => ({
          x: curve.x,
          y: curve.density.map((value) => value > 0 ? Math.log10(value) : null),
          color: RADIUS_COLORS[key] ?? SOURCE[curve.source].color,
          width: curve.source === "fit" ? 2.7 : 1.9,
          dash: key === "fit_re_full_generation_shape" ? [4, 4] : undefined,
          marker: curve.source === "euclid" ? "ring" : undefined,
          dots: curve.source === "euclid",
          markerEvery: Math.max(1, Math.ceil(curve.x.length / 18)),
        }))}
        aspect={0.36}
      />
      <div className="radius-disclosure">
        {entries.map(([key, curve]) => <div key={key}>
          <i style={{ background: RADIUS_COLORS[key] ?? SOURCE[curve.source].color }} />
          <span><b>{curve.label}</b><small>{curve.definition}</small></span>
          <em>unit integral</em>
        </div>)}
      </div>
    </div>
  </div>;
}

function JointDensityMaps({ data }: { data?: JointMaps }) {
  if (!data?.available || !data.maps?.length) {
    return <Empty>Rebuild the galaxy statistics to make the joint maps.</Empty>;
  }
  const q1 = data.maps.find((map) => map.key === "q1");
  const overlays = data.maps.filter(
    (map) => map.key === "synthetic" || map.key === "model",
  );
  const synthetic = overlays.find((map) => map.key === "synthetic");
  if (!q1 || !overlays.length) {
    return <Empty>The Q1 density or generated/model contour layers are unavailable.</Empty>;
  }
  const xDomain: [number, number] = [
    data.magnitude_edges[0], data.magnitude_edges[data.magnitude_edges.length - 1],
  ];
  const yDomain: [number, number] = [
    data.log_radius_edges[0], data.log_radius_edges[data.log_radius_edges.length - 1],
  ];
  const maximum = Math.max(
    ...q1.density.flat().filter((value) => Number.isFinite(value)), 10,
  );
  const maximumPower = Math.max(1, Math.floor(Math.log10(maximum)));
  const colorTicks = Array.from({ length: maximumPower + 1 }, (_, power) => ({
    v: 10 ** power,
    label: power === 0 ? "1" : `1e${power}`,
  }));
  const contourSeries: Series[] = overlays.flatMap((map) => (
    map.contours.flatMap((contour, index) => contour.paths.map((path) => ({
      x: path.x,
      y: path.y,
      color: map.color,
      width: 1.15 + index * 0.20,
      dash: map.key === "synthetic" ? [7, 4] : undefined,
    })))
  ));
  return <section className="joint-atlas" aria-labelledby="joint-atlas-title">
    <header className="joint-atlas__head">
      <div>
        <div className="eyebrow">joint population comparison</div>
        <h2 id="joint-atlas-title">Q1 MER + PHZ density with generated/model contours</h2>
        <p>The image is the PHZ-weighted Q1 magnitude–radius density in neutral grayscale. Blue dashed contours show {synthetic?.label.toLowerCase() ?? "the generated galaxies"}; vermillion solid contours show the active generation law.</p>
      </div>
      <Badge tone="good">one shared Q1 plot</Badge>
    </header>
    <div className="joint-atlas__maps">
      <article className="joint-map" style={{ "--map-source": q1.color } as React.CSSProperties}>
        <header>
          <div><span>{q1.label}</span><small>{q1.detail}</small></div>
          <div>
            <b>{q1.surface_density_arcmin2.toFixed(1)}</b>
            <small>Q1 objects arcmin⁻² in map</small>
          </div>
        </header>
        <Plot
          xDomain={xDomain} yDomain={yDomain}
          xTicks={ticks(xDomain, 8)} yTicks={physicalLogTicks(yDomain, 6)}
          xLabel="VIS 2FWHM AB magnitude"
          yLabel="Circularized Sérsic Rₑ (arcsec, log scale)"
          heat={{
            z: q1.density,
            xEdges: data.magnitude_edges,
            yEdges: data.log_radius_edges,
            max: maximum,
            scale: "log",
            color: JOINT_DENSITY_COLOR,
            colorTicks,
            colorLabel: data.density_unit,
          }}
          series={contourSeries}
          aspect={0.43}
        />
        <footer>
          <span><i style={{ background: q1.color }} />Q1 MER + PHZ density image</span>
          {overlays.map((map) => <span key={map.key}>
            <i style={{ background: map.color }} />
            {map.label} · {map.key === "synthetic" ? "dashed" : "solid"} 10 / 25 / 50 / 80 / 95 / 99 / 99.5% contours
          </span>)}
          {synthetic?.rows != null && <span>
            {synthetic.rows.toLocaleString()} generated galaxies
          </span>}
        </footer>
      </article>
    </div>
  </section>;
}

function PublicationPlate({
  version, includeTraining,
}: {
  version: number;
  includeTraining: boolean;
}) {
  const endpoint = `/view/galaxy-distribution-plate?include_training=${includeTraining ? "1" : "0"}`;
  return <section className="publication-plate" aria-labelledby="publication-plate-title">
    <header className="publication-plate__head">
      <div>
        <div className="eyebrow">paper figure · fixed layout</div>
        <h2 id="publication-plate-title">Galaxy population diagnostics · 2 × 2</h2>
        <p>Rendered from the cached numerical arrays at publication resolution—not from a browser screenshot.</p>
      </div>
      <div className="publication-plate__downloads">
        {(["svg", "pdf", "png"] as const).map((format) => <a
          className="ui-btn" key={format}
          href={`${endpoint}&format=${format}&dpi=300`}
          download={`euclidpolish_galaxy_distributions_2x2.${format}`}
        >Download {format.toUpperCase()}</a>)}
      </div>
    </header>
    <div className="publication-plate__preview">
      <img
        src={`${endpoint}&format=svg&inline=1&v=${version}`}
        alt="Four-panel paper figure comparing Q1, current generated galaxies, and the active galaxy population model"
      />
    </div>
  </section>;
}

export default function GalaxyDistributionsPage() {
  const [includeTraining, setIncludeTraining] = useState(false);
  const resource = useResource<Payload>(
    `/api/galaxy-distributions?include_training=${includeTraining ? "1" : "0"}`,
    [includeTraining],
    { ttl: 10_000 },
  );
  const q1Query = useJob();
  const activate = useJob();
  const plotBuild = useJob();
  const trainingCatalog = useJob();
  const api = resource.data;
  const refresh = (job: { status: string }) => { if (job.status !== "failed") resource.reload(); };
  const rebuildPlots = () => plotBuild.run(
    "/api/galaxy-distributions/build", {}, { onDone: refresh },
  );
  const syncTrainingCatalog = () => trainingCatalog.run(
    "/api/population-comparison/sync-training-catalog",
    {},
    { onDone: (job) => {
      if (job.status !== "failed") rebuildPlots();
    } },
  );
  useEffect(() => {
    if (!q1Query.busy) return;
    resource.reload();
    const timer = window.setInterval(resource.reload, 1500);
    return () => window.clearInterval(timer);
  }, [q1Query.busy, resource.reload]);
  const parameters = useMemo(() => api ? PARAMETER_ORDER.flatMap((key) => (
    api.parameters[key] ? [[key, api.parameters[key]] as const] : []
  )) : [], [api]);

  if (resource.loading && !api) return <Page><Empty><Spinner /> reading galaxy distributions…</Empty></Page>;
  if (!api) return <Page><Empty>Galaxy-distribution status is unavailable.</Empty></Page>;

  return <Page>
    <PageHead
      eyebrow="population laboratory · fitted observables only"
      title="Galaxy distributions"
      sub={`Compare Q1, galaxies in ${api.training_included ? "the training + test + validation source catalogues" : "the current test + validation fields"}, and the active VIS 2FWHM × circularized-size model.`}
      right={<div className="galaxy-actions__buttons">
        <Checkbox
          checked={includeTraining}
          disabled={!api.availability?.synthetic.train_source_catalog}
          onChange={setIncludeTraining}
        >include training catalog</Checkbox>
        <Badge tone={api.training_included ? "warn" : api.stale ? "warn" : "good"}>
          {api.training_included ? "catalog-only all splits" : api.stale ? "plots need rebuild" : "test + validation"}
        </Badge>
        {!api.availability?.synthetic.train_source_catalog && <Button
          size="sm" disabled={trainingCatalog.busy || plotBuild.busy}
          onClick={syncTrainingCatalog}
        >{trainingCatalog.busy ? "Syncing catalog…" : "Sync training catalog only"}</Button>}
      </div>}
    />

    <SourceLedger sources={api.sources} includeTraining={!!api.training_included} />

    {api.training_included && <p className="galaxy-training-note">
      Training contributes from <code>sources_train.csv</code> only. No training
      TFRecords are downloaded or read. Exact VIS 2FWHM and clean-image radius
      curves retain their test + validation area because those measurements are
      absent from the legacy training catalogue.
    </p>}
    <JobProgressView job={trainingCatalog.job} error={trainingCatalog.error} />
    <JobProgressView job={plotBuild.job} error={plotBuild.error} />

    <Card className="galaxy-q1-counts galaxy-actions">
      <CardHead
        title="Q1 MER + PHZ galaxy workflow"
        sub="One run queries galaxy-brightness and Sérsic-radius brackets, fits the galaxy model, and rebuilds the galaxy plots."
      />
      <CardBody>
        <div className="galaxy-q1-counts__content">
          <div className="galaxy-q1-counts__stats">
            <Stat k="Q1 footprint" v={`${(api.q1_counts?.footprint_area_deg2 ?? 63.1).toFixed(1)} deg²`} />
            <Stat k="VIS range" v={api.q1_counts ? `${api.q1_counts.bright.toFixed(1)}–${api.q1_counts.faint.toFixed(1)}` : "14.0–28.0"} />
            <Stat k="bin width" v={`${(api.q1_counts?.bin_width ?? 0.1).toFixed(1)} mag`} />
            <Stat k="checkpoints" v={`${api.q1_counts?.completed_queries ?? api.q1_counts?.query_count ?? 0}/${api.q1_counts?.total_queries ?? 560}`} />
            <Stat k="Rₑ brackets" v={`${api.q1_radius?.completed_queries ?? 0}/${api.q1_radius?.total_queries ?? 170}`} />
            <Stat k="passes" v={`${api.q1_counts?.phases_completed ?? 0}/${api.q1_counts?.phase_count ?? 5}`} />
            <Stat k="F₁ PHZ weight" v={compact(api.q1_counts?.apertures.f1.expected_galaxies)} />
            <Stat k="F₄ PHZ weight" v={compact(api.q1_counts?.apertures.f4.expected_galaxies)} />
          </div>
        </div>
        <div className="galaxy-q1-phases" aria-label="Progressive magnitude-bin sampling phases">
          {Array.from({ length: api.q1_counts?.phase_count ?? 5 }, (_, index) => {
            const completed = index < (api.q1_counts?.phases_completed ?? 0);
            const active = q1Query.busy && index === (api.q1_counts?.phases_completed ?? 0);
            return <div
              className={`galaxy-q1-phases__step${completed ? " is-complete" : ""}${active ? " is-active" : ""}`}
              key={index}
            >
              <span>phase {index + 1}</span>
              <b>+{(index / 10).toFixed(1)} mag</b>
              <small>{completed ? "cached" : active ? "querying" : "waiting"}</small>
            </div>;
          })}
        </div>
        <div className="galaxy-actions__row">
          <div className="galaxy-actions__buttons">
            <Button
              variant="primary"
              disabled={!api.authenticated || q1Query.busy}
              onClick={() => q1Query.run(
                "/api/galaxy-distributions/query-q1-counts",
                {},
                { onDone: refresh },
              )}
            >
              {q1Query.busy ? "Querying + fitting galaxies…" : "Query MER + PHZ"}
            </Button>
            <Button variant="ghost" onClick={resource.reload}>Refresh view</Button>
            <Button variant="ghost" disabled={plotBuild.busy || q1Query.busy}
              onClick={rebuildPlots}>
              {plotBuild.busy ? "Rebuilding plots…" : "Rebuild cached plots"}
            </Button>
            {!api.authenticated && <NavLink className="ui-btn" to="/catalog">Log in to Euclid archive</NavLink>}
          </div>
        </div>
        <p className="galaxy-q1-counts__note">
          <strong>Single acquisition path:</strong> exact 0.1-mag bins are queried at 0.5-mag spacing first,
          then revisited at offsets of 0.1, 0.2, 0.3, and 0.4 mag. Each F₁–F₄ result
          and aggregate Sérsic-R<sub>e</sub> result is stored immediately and skipped on later runs.
          This action applies only the galaxy selection: POINT_LIKE_FLAG IS NULL and
          PHZ_GAL_PROB ≥ 0.5. It never refreshes star caches.
          Stellar counts and Gaia–Euclid colours have their own query action on Star
          distribution. The galaxy density fit uses aggregate brackets rather than a
          downloaded object catalogue.
        </p>
        <JobProgressView job={q1Query.job} error={q1Query.error} />
      </CardBody>
    </Card>

    <Card className="calibration-workflow">
      <CardHead title="Euclid VIS 2FWHM × circularized Sérsic Rₑ model"
        sub="Continuous three-segment bright bridge/main/flat counts plus one straight truncated-Gaussian conditional log-radius law, with no radius tail or break."
        right={<Badge tone={api.calibration.is_active ? "good" : api.calibration.candidate?.valid ? "warn" : undefined}>
          {api.calibration.is_active ? "active for generation" : api.calibration.candidate?.valid ? "candidate ready" : "not fitted"}
        </Badge>} />
      <CardBody>
        <div className="galaxy-q1-counts__stats">
          <Stat k="brightness" v="Q1 VIS 2FWHM · 14–29" />
          <Stat k="integrated density" v={api.calibration.candidate
            ? `${api.calibration.candidate.generation.surface_density_arcmin2.toFixed(0)} arcmin⁻²`
            : "—"} />
          <Stat k="faint plateau" v={api.calibration.candidate
            ? `${api.calibration.candidate.generation.differential_density_cap_arcmin2_mag.toFixed(0)} arcmin⁻² mag⁻¹`
            : "100 arcmin⁻² mag⁻¹"} />
          <Stat k="bright joins" v={api.calibration.candidate
            ? `VIS ${api.calibration.candidate.magnitude_law.bright_join_magnitudes.map((value) => value.toFixed(2)).join(" / ")}`
            : "—"} />
          <Stat k="bridge slopes" v={api.calibration.candidate
            ? `${api.calibration.candidate.magnitude_law.bright_slopes.map((value) => value.toFixed(3)).join(" / ")} dex/mag`
            : "—"} />
          <Stat k="faint plateau starts" v={api.calibration.candidate
            ? `VIS ${api.calibration.candidate.generation.break_magnitude.toFixed(2)}`
            : "—"} />
          <Stat k="radius source" v="Q1 cleaned circularized Sérsic Rₑ" />
          <Stat k="COSMOS in fit" v="no" />
          <Stat k="slope" v={api.calibration.candidate
            ? `${api.calibration.candidate.radius_law.slope_log10_arcsec_per_mag.toFixed(4)} dex/mag`
            : "—"} />
          <Stat k="scatter" v={api.calibration.candidate
            ? `${api.calibration.candidate.radius_law.scatter_dex.toFixed(4)} dex`
            : "—"} />
        </div>
        <div className="galaxy-actions__row">
          <div className="galaxy-actions__buttons">
            <Button variant="primary"
              disabled={!api.calibration.candidate?.valid || activate.busy || q1Query.busy}
              onClick={() => activate.run(
                "/api/galaxy-distributions/activate", {}, { onDone: refresh },
              )}>
              {activate.busy ? "Activating…" : api.calibration.is_active
                ? "Re-activate this TNG model" : "Use this TNG model"}
            </Button>
            {api.calibration.is_active && <NavLink className="ui-btn" to="/sky">Open Sky jobs</NavLink>}
          </div>
        </div>
        {api.calibration.candidate && <p className="galaxy-q1-counts__note">
          The candidate contains {api.calibration.candidate.radius_law.fitted_rows.toLocaleString()} clean
          aggregate-weighted radii. Three fitted bright-bridge slopes meet continuously at fixed VIS joins {api.calibration.candidate.magnitude_law.bright_join_magnitudes.map((value) => value.toFixed(2)).join(", ")}, then join the main Q1 line continuously; counts stay flat at {api.calibration.candidate.generation.differential_density_cap_arcmin2_mag.toFixed(0)} galaxies / arcmin² / mag through VIS {api.calibration.candidate.generation.vis_magnitude_max.toFixed(0)}.
          {" "}All magnitudes use the same straight truncated-Gaussian conditional radius law; there is no separate broad tail or radius break.
          {" "}Its integral is {api.calibration.candidate.generation.surface_density_arcmin2.toFixed(2)} galaxies / arcmin².
          {" "}Its fingerprint is <code>{api.calibration.candidate.fingerprint.slice(0, 12)}…</code>.
        </p>}
        <JobProgressView job={activate.job} error={activate.error} />
      </CardBody>
    </Card>

    {api.calibration.candidate?.plots?.conditional_radius && (() => {
      const relation = api.calibration.candidate.plots!.conditional_radius!;
      const modelLow = relation.model_core_low_log10_arcsec
        ?? relation.model_low_log10_arcsec
        ?? [];
      const modelHigh = relation.model_core_high_log10_arcsec
        ?? relation.model_high_log10_arcsec
        ?? [];
      const observed = relation.observed_mean_log10_arcsec.filter(
        (value): value is number => value != null && Number.isFinite(value),
      );
      const yDomain = paddedDomain([
        ...observed,
        ...modelLow,
        ...modelHigh,
      ], 0.5);
      const xDomain = paddedDomain(relation.magnitude, 1.0);
      return <Card className="parameter-card">
        <CardHead title="Joint brightness–radius relation"
          sub="Aggregate Q1 circularized Sérsic-Rₑ moments in each VIS 2FWHM magnitude bracket and one fitted straight truncated-Gaussian conditional law." />
        <CardBody>
          <Plot
            xDomain={xDomain} yDomain={yDomain}
            xTicks={ticks(xDomain, 7)} yTicks={physicalLogTicks(yDomain, 6)}
            xLabel="VIS 2FWHM AB magnitude"
            yLabel="Circularized Sérsic Rₑ (arcsec, log scale)"
            series={[
              {
                x: relation.magnitude,
                y: relation.model_mean_log10_arcsec,
                low: modelLow,
                high: modelHigh,
                color: "#e25543", fillAlpha: 0.12, alpha: 0, width: 0,
              },
              {
                x: relation.magnitude,
                y: relation.observed_mean_log10_arcsec,
                color: "#31a7d8", mode: "scatter", marker: "ring", width: 1.7,
              },
              {
                x: relation.magnitude,
                y: relation.model_mean_log10_arcsec,
                color: "#e25543", width: 2.6,
              },
            ]}
            aspect={0.36}
          />
          <p className="galaxy-q1-counts__note">
            Blue points are bracket-level Euclid measurements; the red line and band are
            the single straight conditional mean and its one-scatter truncated-Gaussian
            interval. There is no magnitude break or broad radius tail; COSMOS and
            object-level samples are absent from this fit.
          </p>
        </CardBody>
      </Card>;
    })()}

    <section className="galaxy-density-section">
      <header className="galaxy-density-section__head">
        <div>
          <div className="eyebrow">fitted one-dimensional observables</div>
          <h2>Q1 aggregates, {api.training_included ? "all catalogued generated fields" : "current generated galaxies"}, and the active law</h2>
          <p>Only VIS 2FWHM brightness and circularized half-light size are retained. Every diagnostic occupies its own full-width row.</p>
        </div>
        <div className="galaxy-key">
          {ORDER.map((key) => <span key={key}><i style={{ background: SOURCE[key].color }} />{SOURCE[key].label}</span>)}
        </div>
      </header>
      <div className="galaxy-plot-grid">
        {parameters.map(([key, parameter]) => <article className="galaxy-plot" key={key}>
          <header><span>{parameter.label}</span><small>{parameter.note}</small></header>
          {key === "magnitude"
            ? <ApparentBrightnessPlot
                key={Object.keys(parameter.photometry_series ?? {}).join("|")}
                parameter={parameter}
              />
            : key === "radius"
              ? <RadiusPlot
                  key={Object.keys(parameter.radius_series ?? {}).join("|")}
                  parameter={parameter}
                />
            : <DensityPlot parameter={parameter} />}
        </article>)}
        {api.parameters.radius && <article className="galaxy-plot galaxy-plot--shape">
          <header>
            <span>Normalized half-light shape</span>
            <small>Unit-integral Q1 and model radius densities are kept off the surface-density axis.</small>
          </header>
          <RadiusShapePlot
            key={Object.keys(api.parameters.radius.radius_series ?? {}).join("|")}
            parameter={api.parameters.radius}
          />
        </article>}
      </div>
    </section>

    <JointDensityMaps data={api.joint_maps} />

    <PublicationPlate version={api.version} includeTraining={!!api.training_included} />
  </Page>;
}
