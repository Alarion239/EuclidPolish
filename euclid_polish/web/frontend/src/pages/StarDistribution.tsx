import { useEffect, useState } from "react";
import Plot, { type Tick } from "../charts/Plot";
import { C, categorical } from "../colors";
import { useResource } from "../hooks";
import { JobProgressView, useJob } from "../jobs";
import {
  Badge, Button, Card, CardBody, CardHead, Checkbox, Empty, Page, PageHead, Spinner, Stat,
} from "../ui";
import "./star-distribution.css";

type ColorKey = "vis_y" | "vis_j" | "vis_h" | "y_j" | "y_h" | "j_h";
type ProjectionColorKey = "vis_y" | "vis_j" | "vis_h";
type DensityKey = "vis" | ColorKey;
type DensityParameter = {
  label: string;
  x_label: string;
  x: number[];
  x_domain: [number, number];
  euclid: number[];
  gaia_x?: number[];
  gaia: number[];
  model: number[];
  synthetic: number[];
  point_sources?: number[] | null;
  gaia_fit?: number[] | null;
  fit_ranges?: {
    q1?: [number | null, number | null];
    gaia?: [number | null, number | null];
  };
};
type ColorPlot = {
  label: string;
  values: number[];
  pearson_r: number | null;
  y_domain: [number, number];
  trend: { x: number[]; y: number[] };
  fit: null | {
    x: number[];
    center: number[];
    sigma: number;
    one_sigma_low: number[];
    one_sigma_high: number[];
    two_sigma_low: number[];
    two_sigma_high: number[];
  };
};
type Distribution = {
  matched_stars: number;
  high_quality_stars: number;
  pointlike_over_0_9: number;
  bp_rp: number[];
  x_domain: [number, number];
  colors: Record<ColorKey, ColorPlot>;
  axis_note: string;
  fit_note: string | null;
  training_included?: boolean;
  training_catalog_only?: boolean;
  synthetic_splits?: string[];
  gaia_cmd: {
    cached_stars: number;
    plotted_stars: number;
    without_color: number;
    x_domain: [number, number];
    g_domain: [number, number];
    matched: { bp_rp: number[]; g_mag: number[] };
    unmatched: { bp_rp: number[]; g_mag: number[] };
    note: string;
  };
  euclid_projection: null | {
    vis_domain: [number, number];
    matched: {
      vis_mag: number[];
      colors: Record<ProjectionColorKey, number[]>;
    };
    unmatched: {
      vis_mag: number[];
      colors: Record<ProjectionColorKey, number[]>;
    };
    euclid_observed: Record<ProjectionColorKey, {
      vis_mag: number[];
      color: number[];
    }>;
    colors: Record<ProjectionColorKey, {
      label: string;
      x_domain: [number, number];
      sigma: number;
    }>;
    note: string;
  };
  density_comparison: null | {
    area_arcmin2: number;
    gaia_area_arcmin2?: number;
    model_density_arcmin2: number;
    model_sample_count: number;
    euclid_vis_count: number;
    q1_phz_expected_stars: number | null;
    q1_expected_point_sources: number | null;
    q1_selected_point_sources: number | null;
    q1_area_arcmin2: number | null;
    euclid_color_count: number;
    gaia_count: number;
    gaia_native_g_count?: number;
    synthetic_area_arcmin2?: number | null;
    synthetic_star_count?: number;
    synthetic_color_count?: number;
    parameters: Record<DensityKey, DensityParameter>;
    note: string;
  };
  gaia_sampling?: null | {
    sampling_kind?: string;
    field_count?: number;
    radius_deg?: number;
    radius_arcmin: number;
    tap_provider?: string;
    query_mode?: string;
    area_deg2?: number;
    area_arcmin2: number;
    fields?: Array<{ name?: string; ra: number; dec: number; rows?: number }>;
  };
};
type Calibration = {
  candidate: null | {
    valid?: boolean;
    warnings?: string[];
    coverage_notes?: string[];
    gaia?: { rows?: number };
    euclid_mapping?: { matched_stars?: number };
  };
  is_active: boolean;
};
type Q1Counts = {
  footprint_area_deg2: number;
  selected_point_sources: number;
  expected_point_sources: number;
  expected_stars: number;
  bins: Array<{
    mag_lo: number;
    mag_hi: number;
    selected_point_sources: number;
    expected_point_sources: number;
    classified_rows: number;
    expected_stars: number;
  }>;
  edges: number[];
  selection: string;
};
type ApiPayload = {
  authenticated: boolean;
  color_sample: {
    cached: boolean;
    euclid: null | { rows?: number; field_count?: number };
    gaia: null | { rows?: number; field_count?: number };
  };
  calibration: Calibration;
  distribution: Distribution | null;
  q1_counts: Q1Counts | null;
  availability?: {
    synthetic: {
      train_source_catalog: boolean;
      population_fields: number;
      population_fields_with_training: number;
    };
  };
};

const COLOR_ORDER: ColorKey[] = [
  "vis_y", "vis_j", "vis_h", "y_j", "y_h", "j_h",
];
const PROJECTION_COLOR_ORDER: ProjectionColorKey[] = ["vis_y", "vis_j", "vis_h"];
const DENSITY_ORDER: DensityKey[] = ["vis", ...COLOR_ORDER];

function ticks([lo, hi]: [number, number], count = 5): Tick[] {
  return Array.from({ length: count }, (_, index) => {
    const value = lo + (hi - lo) * index / (count - 1);
    return { v: value, label: Number(value.toPrecision(3)).toString() };
  });
}

function clamp(values: number[], [lo, hi]: [number, number]): number[] {
  return values.map((value) => Math.max(lo, Math.min(hi, value)));
}

function magnitudeTicks([lo, hi]: [number, number], count = 6): Tick[] {
  return Array.from({ length: count }, (_, index) => {
    const magnitude = lo + (hi - lo) * index / (count - 1);
    return { v: -magnitude, label: Number(magnitude.toPrecision(3)).toString() };
  });
}

function logDensity(values: number[]): (number | null)[] {
  return values.map((value) => value > 0 ? Math.log10(value) : null);
}

function densityDomain(parameter: DensityParameter): [number, number] {
  const positive = [
    parameter.euclid, parameter.gaia, parameter.model, parameter.synthetic,
    parameter.gaia_fit ?? [], parameter.point_sources ?? [],
  ]
    .flat().filter((value) => Number.isFinite(value) && value > 0);
  if (!positive.length) return [-4, 0];
  const high = Math.ceil(Math.log10(Math.max(...positive)));
  const low = Math.max(
    Math.floor(Math.log10(Math.min(...positive))),
    high - 5,
  );
  return [low, Math.max(low + 1, high)];
}

function densityTicks(domain: [number, number]): Tick[] {
  const start = Math.ceil(domain[0]);
  const end = Math.floor(domain[1]);
  return Array.from({ length: end - start + 1 }, (_, index) => {
    const exponent = start + index;
    return { v: exponent, label: `10^${exponent}` };
  });
}

function GaiaColourMagnitudePlot({ distribution }: { distribution: Distribution }) {
  const cmd = distribution.gaia_cmd;
  const yDomain: [number, number] = [-cmd.g_domain[1], -cmd.g_domain[0]];
  const matchedCount = cmd.matched.bp_rp.length;
  const unmatchedCount = cmd.unmatched.bp_rp.length;
  return (
    <section className="star-cmd-section">
      <header className="star-colour-section__head">
        <div>
          <div className="eyebrow">all cached Gaia sources · apparent magnitudes</div>
          <h2>Gaia colour–magnitude diagram</h2>
          <p>{cmd.note}</p>
          {cmd.without_color > 0 && (
            <p>{cmd.without_color.toLocaleString()} cached sources without BP−RP remain stored but cannot be plotted.</p>
          )}
        </div>
        <div className="star-colour-key" aria-label="Colour–magnitude plot legend">
          <span><i className="star-colour-key__unmatched" />unmatched · {unmatchedCount.toLocaleString()}</span>
          <span><i className="star-colour-key__matched" />Euclid counterpart · {matchedCount.toLocaleString()}</span>
        </div>
      </header>
      <div className="star-cmd-plot">
        <Plot
          xDomain={cmd.x_domain}
          yDomain={yDomain}
          xTicks={ticks(cmd.x_domain, 6)}
          yTicks={magnitudeTicks(cmd.g_domain)}
          xLabel="Gaia BP − RP [mag]"
          yLabel="Gaia G [mag] · brighter ↑"
          series={[
            {
              x: clamp(cmd.unmatched.bp_rp, cmd.x_domain),
              y: clamp(cmd.unmatched.g_mag, cmd.g_domain).map((value) => -value),
              color: C.muted, mode: "scatter", width: 0.45, alpha: 0.25,
            },
            {
              x: clamp(cmd.matched.bp_rp, cmd.x_domain),
              y: clamp(cmd.matched.g_mag, cmd.g_domain).map((value) => -value),
              color: C.comb, mode: "scatter", marker: "ring", width: 0.75, alpha: 0.62,
            },
          ]}
          aspect={0.46}
        />
      </div>
    </section>
  );
}

function EuclidProjectionPlots({ distribution }: { distribution: Distribution }) {
  const projection = distribution.euclid_projection;
  if (!projection) return null;
  const yDomain: [number, number] = [
    -projection.vis_domain[1], -projection.vis_domain[0],
  ];
  return (
    <section className="star-euclid-projection">
      <header className="star-colour-section__head">
        <div>
          <div className="eyebrow">all Gaia sources · transformed by the fitted locus</div>
          <h2>Gaia population projected into Euclid</h2>
          <p>{projection.note}</p>
        </div>
        <div className="star-colour-key" aria-label="Projected population legend">
          <span><i className="star-colour-key__unmatched" />unmatched Gaia</span>
          <span><i className="star-colour-key__matched" />Euclid counterpart</span>
          <span><i className="star-colour-key__euclid" />measured fixed-Q1 star</span>
        </div>
      </header>
      <div className="star-euclid-projection__grid">
        {PROJECTION_COLOR_ORDER.map((key) => {
          const color = projection.colors[key];
          const observed = projection.euclid_observed[key];
          return (
            <section className="star-colour-plot" key={key}>
              <header>
                <h3>VIS vs {color.label}</h3>
                <span>
                  intrinsic σ = {color.sigma.toFixed(3)} mag
                  {" · "}Euclid N = {observed.vis_mag.length.toLocaleString()}
                </span>
              </header>
              <Plot
                xDomain={color.x_domain}
                yDomain={yDomain}
                xTicks={ticks(color.x_domain)}
                yTicks={magnitudeTicks(projection.vis_domain)}
                xLabel={`${color.label} [AB mag]`}
                yLabel="VIS [AB mag] · brighter ↑"
                series={[
                  {
                    x: clamp(projection.unmatched.colors[key], color.x_domain),
                    y: clamp(projection.unmatched.vis_mag, projection.vis_domain)
                      .map((value) => -value),
                    color: C.muted, mode: "scatter", width: 0.4, alpha: 0.22,
                  },
                  {
                    x: clamp(projection.matched.colors[key], color.x_domain),
                    y: clamp(projection.matched.vis_mag, projection.vis_domain)
                      .map((value) => -value),
                    color: C.comb, mode: "scatter", marker: "ring",
                    width: 0.7, alpha: 0.58,
                  },
                  {
                    x: clamp(observed.color, color.x_domain),
                    y: clamp(observed.vis_mag, projection.vis_domain)
                      .map((value) => -value),
                    color: C.mean, mode: "scatter", width: 0.55, alpha: 0.34,
                  },
                ]}
                aspect={0.78}
              />
            </section>
          );
        })}
      </div>
    </section>
  );
}

function StellarDensityComparison({ distribution }: { distribution: Distribution }) {
  const comparison = distribution.density_comparison;
  if (!comparison) return null;
  return (
    <section className="star-density-comparison">
      <header className="star-colour-section__head">
        <div>
          <div className="eyebrow">Q1 footprint normalization · probability-weighted counts</div>
          <h2>Stellar density in Euclid magnitude and colour</h2>
          <p>{comparison.note}</p>
        </div>
        <div className="star-colour-key" aria-label="Stellar density legend">
          <span><i className="star-density-key__point-source" />Q1 point sources (VIS)</span>
          <span><i className="star-density-key__euclid" />Q1 PHZ (VIS)</span>
          <span><i className="star-density-key__gaia" />native Gaia G<sub>AB</sub> + shared-slope fit</span>
          <span><i className="star-density-key__model" />Q1-normalized straight law / colour draw</span>
          <span><i className="star-density-key__synthetic" />actual generated {distribution.training_included ? "train + test + validation" : "test + validation"} stars</span>
        </div>
      </header>
      <div className="star-density-summary" aria-label="Density comparison sample sizes">
        <span>Q1 area <b>{(comparison.q1_area_arcmin2 ?? comparison.area_arcmin2).toLocaleString(undefined, { maximumFractionDigits: 0 })}</b> arcmin²</span>
        <span>Gaia field area <b>{(comparison.gaia_area_arcmin2 ?? comparison.area_arcmin2).toLocaleString(undefined, { maximumFractionDigits: 0 })}</b> arcmin²</span>
        <span>PHZ expected stars <b>{comparison.q1_phz_expected_stars?.toLocaleString(undefined, { maximumFractionDigits: 1 }) ?? "—"}</b></span>
        <span>expected point sources <b>{comparison.q1_expected_point_sources?.toLocaleString(undefined, { maximumFractionDigits: 1 }) ?? "—"}</b></span>
        <span>Euclid four-band <b>{comparison.euclid_color_count.toLocaleString()}</b></span>
        <span>native Gaia G<sub>AB</sub> <b>{comparison.gaia_native_g_count?.toLocaleString() ?? "—"}</b></span>
        <span>Gaia colour projection <b>{comparison.gaia_count.toLocaleString()}</b></span>
        <span>generated stars <b>{comparison.synthetic_star_count?.toLocaleString() ?? "—"}</b> / <b>{comparison.synthetic_area_arcmin2?.toFixed(1) ?? "—"}</b> arcmin²</span>
        <span>model density <b>{comparison.model_density_arcmin2.toFixed(3)}</b> arcmin⁻²</span>
      </div>
      <div className="star-density-grid">
        {DENSITY_ORDER.map((key) => {
          const parameter = comparison.parameters[key];
          const yDomain = densityDomain(parameter);
          const fitGuides = key === "vis"
            ? Object.values(parameter.fit_ranges ?? {}).flatMap((interval) => (
              interval && interval.every((value): value is number =>
                value != null && Number.isFinite(value))
                ? interval.map((value) => ({
                  axis: "x" as const, v: value, color: C.muted,
                  dash: [3, 4], width: 1, alpha: 0.55,
                }))
                : []
            ))
            : [];
          return (
            <section className={`star-colour-plot ${key === "vis" ? "star-density-plot--vis" : ""}`} key={key}>
              <header>
                <h3>{parameter.label}</h3>
                <span>{key === "vis"
                  ? "Q1 0.1 mag · Gaia fit 0.5 mag · guides mark fitted regions"
                  : "log density · no simulated noise"}</span>
              </header>
              <Plot
                xDomain={parameter.x_domain}
                yDomain={yDomain}
                xTicks={ticks(parameter.x_domain, key === "vis" ? 7 : 5)}
                yTicks={densityTicks(yDomain)}
                xLabel={parameter.x_label}
                yLabel="stars / arcmin² / mag"
                series={[
                  ...(parameter.point_sources ? [{
                    x: parameter.x, y: logDensity(parameter.point_sources),
                    color: categorical(1), width: 2.2,
                  }] : []),
                  { x: parameter.x, y: logDensity(parameter.euclid), color: C.mean, width: 2.2 },
                  { x: parameter.gaia_x ?? parameter.x, y: logDensity(parameter.gaia), color: C.comb, width: 2.2 },
                  ...(parameter.gaia_fit ? [{
                    x: parameter.gaia_x ?? parameter.x, y: logDensity(parameter.gaia_fit),
                    color: C.comb, width: 2.2, dash: [4, 3],
                  }] : []),
                  { x: parameter.x, y: logDensity(parameter.model), color: categorical(3), width: 2.2, dash: [6, 4] },
                  { x: parameter.x, y: logDensity(parameter.synthetic), color: categorical(4), width: 1.8, marker: "filled", dots: true, markerEvery: 2 },
                ]}
                guides={fitGuides}
                aspect={key === "vis" ? 0.34 : 0.62}
              />
            </section>
          );
        })}
      </div>
    </section>
  );
}

function CorrelationPlot({ distribution, colorKey }: {
  distribution: Distribution; colorKey: ColorKey;
}) {
  const item = distribution.colors[colorKey];
  const x = clamp(distribution.bp_rp, distribution.x_domain);
  const y = clamp(item.values, item.y_domain);
  const fittedSeries = item.fit ? [
    {
      x: item.fit.x, y: item.fit.center,
      low: item.fit.two_sigma_low, high: item.fit.two_sigma_high,
      color: C.comb, fillAlpha: 0.07, alpha: 0, width: 0,
    },
    {
      x: item.fit.x, y: item.fit.center,
      low: item.fit.one_sigma_low, high: item.fit.one_sigma_high,
      color: C.comb, fillAlpha: 0.16, alpha: 0, width: 0,
    },
  ] : [];
  return (
    <section className="star-colour-plot">
      <header>
        <h3>{item.label}</h3>
        <span>
          Pearson r = {item.pearson_r == null ? "—" : item.pearson_r.toFixed(3)}
          {item.fit ? ` · σ = ${item.fit.sigma.toFixed(3)}` : ""}
          {" · "}N = {distribution.matched_stars.toLocaleString()}
        </span>
      </header>
      <Plot
        xDomain={distribution.x_domain}
        yDomain={item.y_domain}
        xTicks={ticks(distribution.x_domain)}
        yTicks={ticks(item.y_domain)}
        xLabel="Gaia BP − RP [mag]"
        yLabel={`${item.label} [AB mag]`}
        series={[
          ...fittedSeries,
          { x, y, color: C.mean, mode: "scatter", width: 0.7, alpha: 0.28 },
          item.fit
            ? { x: item.fit.x, y: item.fit.center, color: C.comb, width: 2.4 }
            : { x: item.trend.x, y: item.trend.y, color: C.comb, width: 2.4 },
        ]}
        aspect={0.67}
      />
    </section>
  );
}

export default function StarDistributionPage() {
  const [includeTraining, setIncludeTraining] = useState(false);
  const resource = useResource<ApiPayload>(
    `/api/star-distribution?include_training=${includeTraining ? "1" : "0"}`,
    [includeTraining],
    { ttl: 10_000 },
  );
  const query = useJob();
  const fit = useJob();
  const activate = useJob();
  const trainingCatalog = useJob();
  const api = resource.data;
  const refresh = (job: { status: string }) => {
    if (job.status !== "failed") resource.reload();
  };
  const syncTrainingCatalog = () => trainingCatalog.run(
    "/api/population-comparison/sync-training-catalog",
    {},
    { onDone: refresh },
  );
  useEffect(() => {
    if (!query.busy) return;
    resource.reload();
    const timer = window.setInterval(resource.reload, 1500);
    return () => window.clearInterval(timer);
  }, [query.busy, resource.reload]);

  if (resource.loading && !api) {
    return <Page><Empty><Spinner /> reading matched stellar colours…</Empty></Page>;
  }
  if (!api) {
    return <Page><Empty>Star-distribution status is unavailable.</Empty></Page>;
  }

  const candidate = api.calibration.candidate;
  const distribution = api.distribution;
  const q1Counts = api.q1_counts;
  return (
    <Page>
      <PageHead
        eyebrow="stellar calibration · Q1 PHZ × Gaia × Euclid"
        title="Star distribution"
        sub="Select POINT_LIKE_PROB ≥ 0.9, show both point-source and PHZ-weighted 0.1-mag VIS counts over the 63.1 deg² Q1 footprint, and use matched Gaia only for the colour/temperature locus."
        right={<div className="star-query-strip__actions">
          <Checkbox
            checked={includeTraining}
            disabled={!api.availability?.synthetic.train_source_catalog}
            onChange={setIncludeTraining}
          >include training catalog</Checkbox>
          <Badge tone={distribution?.training_included ? "warn" : api.calibration.is_active ? "good" : candidate ? "warn" : undefined}>
            {distribution?.training_included ? "catalog-only all splits" : api.calibration.is_active ? "active stellar prior" : candidate ? "candidate ready" : "not fitted"}
          </Badge>
          {!api.availability?.synthetic.train_source_catalog && <Button
            size="sm" disabled={trainingCatalog.busy}
            onClick={syncTrainingCatalog}
          >{trainingCatalog.busy ? "Syncing catalog…" : "Sync training catalog only"}</Button>}
        </div>}
      />

      {distribution?.training_included && <p className="star-training-note">
        Training stars come only from <code>sources_train.csv</code>; no training
        TFRecords are downloaded or read. The 6,400-field legacy catalogue is
        shown as an optional historical census, separate from the active prior.
      </p>}
      <JobProgressView job={trainingCatalog.job} error={trainingCatalog.error} />

      <Card className="star-query-strip">
        <CardHead title="Q1 point-source and stellar counts"
          sub="Independent stellar MER + PHZ brackets and fixed-field Gaia colours" />
        <CardBody>
          <div className="star-query-strip__content">
            <div className="star-query-strip__stats">
              <Stat k="Q1 footprint" v={`${(q1Counts?.footprint_area_deg2 ?? 63.1).toFixed(1)} deg²`} />
              <Stat k="VIS bin width" v="0.1 mag" />
              <Stat k="expected point sources" v={q1Counts ? q1Counts.expected_point_sources.toLocaleString(undefined, { maximumFractionDigits: 1 }) : "not queried"} />
              <Stat k="PHZ expected stars" v={q1Counts ? q1Counts.expected_stars.toLocaleString(undefined, { maximumFractionDigits: 1 }) : "not queried"} />
              <Stat k="selected objects" v={(q1Counts?.selected_point_sources ?? 0).toLocaleString()} />
              <Stat k="bins" v={(q1Counts?.bins.length ?? 0).toLocaleString()} />
            </div>
            <div className="star-query-strip__actions">
              <Button variant="primary" disabled={!api.authenticated || query.busy}
                onClick={() => query.run(
                  "/api/star-distribution/query", {}, { onDone: refresh },
                )}>
                {query.busy ? "Querying stars…" : "Query stars · MER + PHZ + Gaia"}
              </Button>
            </div>
          </div>
          <p className="star-query-strip__note">
            <strong>Selection:</strong> 0.1-mag VIS PSF bins, POINT_LIKE_PROB ≥ 0.9,
            showing both Σ POINT_LIKE_PROB and Σ PHZ_STAR_PROB divided by the 63.1 deg² Q1 footprint.
            The fixed-field colour query uses the same stellar point-like threshold and Gaia
            matches. No galaxy selection is used by this action.
          </p>
          {!api.authenticated && <p className="star-query-strip__note">
            Log in to the Euclid archive on Catalog before running this query.
          </p>}
          <JobProgressView job={query.job} error={query.error} />
        </CardBody>
      </Card>

      <Card className="star-query-strip">
        <CardHead title="Stellar fit and activation"
          sub="Q1 brackets set the magnitude density; a cached matched Gaia-Euclid sample supplies colours only." />
        <CardBody>
          <div className="star-query-strip__content">
            <div className="star-query-strip__stats">
              <Stat k="matched stars" v={(distribution?.matched_stars ?? candidate?.euclid_mapping?.matched_stars ?? 0).toLocaleString()} />
              <Stat k="all-band S/N ≥ 5" v={(distribution?.high_quality_stars ?? 0).toLocaleString()} />
              <Stat k="POINT_LIKE_PROB ≥ 0.9" v={(distribution?.pointlike_over_0_9 ?? 0).toLocaleString()} />
              <Stat k="colour sample" v={api.color_sample.cached
                ? `${(api.color_sample.euclid?.rows ?? 0).toLocaleString()} Euclid candidates`
                : "not cached"} />
              <Stat k="Gaia rows" v={(api.color_sample.gaia?.rows ?? 0).toLocaleString()} />
              <Stat k="fixed Q1 fields" v={(api.color_sample.gaia?.field_count ?? 0).toLocaleString()} />
            </div>
            <div className="star-query-strip__actions">
              <Button disabled={!q1Counts || !api.color_sample.cached || fit.busy}
                onClick={() => fit.run("/api/star-distribution/fit", {}, { onDone: refresh })}>
                {fit.busy ? "Fitting stellar prior…" : "Fit stellar prior from cached data"}
              </Button>
              <Button disabled={!candidate?.valid || activate.busy}
                onClick={() => activate.run("/api/star-distribution/activate", {}, { onDone: refresh })}>
                {api.calibration.is_active ? "Re-activate stellar prior" : "Activate stellar prior"}
              </Button>
            </div>
          </div>
          <p className="star-query-strip__note">
            The fixed Q1 fields supply a magnitude-stratified colour/temperature sample only.
            They do not set the stellar surface density or magnitude law. The query above is
            independent of the galaxy workflow. The straight count fit keeps Q1 at 0.1-mag
            resolution and bins the smaller Gaia shape sample at 0.5 mag; after the query
            completes, fit these stellar caches.
          </p>
          {candidate?.warnings?.[0] && <p className="star-query-strip__note"><strong>Fit note:</strong> {candidate.warnings[0]}</p>}
          {candidate?.coverage_notes?.[0] && <p className="star-query-strip__note"><strong>Coverage:</strong> {candidate.coverage_notes[0]}</p>}
          <JobProgressView job={fit.job} error={fit.error} />
          <JobProgressView job={activate.job} error={activate.error} />
        </CardBody>
      </Card>

      {distribution && <GaiaColourMagnitudePlot distribution={distribution} />}
      {distribution && <EuclidProjectionPlots distribution={distribution} />}
      {distribution && <StellarDensityComparison distribution={distribution} />}

      {distribution ? (
        <section className="star-colour-section">
          <header className="star-colour-section__head">
            <div>
              <div className="eyebrow">measured colours · all matched stars</div>
              <h2>Gaia colour versus fitted Euclid distributions</h2>
              <p>{distribution.axis_note}</p>
              {distribution.fit_note && <p>{distribution.fit_note}</p>}
            </div>
            <div className="star-colour-key" aria-label="Plot legend">
              <span><i className="star-colour-key__points" />catalogue stars</span>
              <span><i className="star-colour-key__two-sigma" />2σ intrinsic</span>
              <span><i className="star-colour-key__one-sigma" />1σ intrinsic</span>
              <span><i className="star-colour-key__trend" />fitted locus</span>
            </div>
          </header>
          <div className="star-colour-grid">
            {COLOR_ORDER.map((key) => (
              <CorrelationPlot key={key} distribution={distribution} colorKey={key} />
            ))}
          </div>
        </section>
      ) : (
        <Empty>
          Query the stellar MER + PHZ and Gaia data above, then fit the cached colours
          to create the six matched-colour plots.
        </Empty>
      )}
    </Page>
  );
}
