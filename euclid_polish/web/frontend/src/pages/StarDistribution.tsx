import { NavLink } from "react-router-dom";
import Plot, { type Tick } from "../charts/Plot";
import { C, categorical } from "../colors";
import { useResource } from "../hooks";
import { JobProgressView, useJob } from "../jobs";
import {
  Badge, Button, Card, CardBody, CardHead, Empty, Page, PageHead, Spinner, Stat,
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
  gaia: number[];
  model: number[];
  point_sources?: number[] | null;
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
    parameters: Record<DensityKey, DensityParameter>;
    note: string;
  };
  gaia_sampling?: null | {
    sampling_kind?: string;
    field_count?: number;
    cone_count: number;
    radius_deg?: number;
    radius_arcmin: number;
    tap_provider?: string;
    query_mode?: string;
    area_deg2?: number;
    area_arcmin2: number;
    cones?: Array<{ ra: number; dec: number; member_stars?: number }>;
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
  availability: { euclid_catalog: { cached: boolean; meta?: { cone_count?: number } | null } };
  calibration: Calibration;
  distribution: Distribution | null;
  q1_counts: Q1Counts | null;
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
  const positive = [parameter.euclid, parameter.gaia, parameter.model, parameter.point_sources ?? []]
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
          <span><i className="star-colour-key__euclid" />measured Euclid cone star</span>
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
          <span><i className="star-density-key__gaia" />native Gaia G<sub>AB</sub> (magnitude) / fitted projection (colours)</span>
          <span><i className="star-density-key__model" />model draw</span>
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
        <span>model density <b>{comparison.model_density_arcmin2.toFixed(3)}</b> arcmin⁻²</span>
      </div>
      <div className="star-density-grid">
        {DENSITY_ORDER.map((key) => {
          const parameter = comparison.parameters[key];
          const yDomain = densityDomain(parameter);
          return (
            <section className={`star-colour-plot ${key === "vis" ? "star-density-plot--vis" : ""}`} key={key}>
              <header>
                <h3>{parameter.label}</h3>
                <span>log density · no simulated noise</span>
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
                  { x: parameter.x, y: logDensity(parameter.gaia), color: C.comb, width: 2.2 },
                  { x: parameter.x, y: logDensity(parameter.model), color: categorical(3), width: 2.2, dash: [6, 4] },
                ]}
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
  const resource = useResource<ApiPayload>("/api/star-distribution", [], { ttl: 10_000 });
  const q1Query = useJob();
  const query = useJob();
  const activate = useJob();
  const api = resource.data;
  const refresh = (job: { status: string }) => {
    if (job.status !== "failed") resource.reload();
  };

  if (resource.loading && !api) {
    return <Page><Empty><Spinner /> reading matched stellar colours…</Empty></Page>;
  }
  if (!api) {
    return <Page><Empty>Star-distribution status is unavailable.</Empty></Page>;
  }

  const candidate = api.calibration.candidate;
  const distribution = api.distribution;
  const q1Counts = api.q1_counts;
  const coneCount = api.availability.euclid_catalog.meta?.cone_count ?? 0;
  const gaiaSampling = distribution?.gaia_sampling;
  return (
    <Page>
      <PageHead
        eyebrow="stellar calibration · Q1 PHZ × Gaia × Euclid"
        title="Star distribution"
        sub="Select POINT_LIKE_PROB ≥ 0.9, show both point-source and PHZ-weighted 0.1-mag VIS counts over the 63.1 deg² Q1 footprint, and use matched Gaia only for the colour/temperature locus."
        right={<Badge tone={api.calibration.is_active ? "good" : candidate ? "warn" : undefined}>
          {api.calibration.is_active ? "active stellar prior" : candidate ? "candidate ready" : "not fitted"}
        </Badge>}
      />

      <Card className="star-query-strip">
        <CardHead title="Q1 point-source and stellar counts"
          sub="Direct Euclid archive query · no random cones and no Gaia normalization" />
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
              <Button variant="primary" disabled={!api.authenticated || q1Query.busy}
                onClick={() => q1Query.run("/api/star-distribution/query-q1-counts", {}, { onDone: refresh })}>
                {q1Query.busy ? "Querying Q1 counts…" : q1Counts ? "Re-query Q1 counts" : "Query Q1 counts"}
              </Button>
              {!api.authenticated && <NavLink className="ui-btn" to="/catalog">Log in to Euclid archive</NavLink>}
            </div>
          </div>
          <p className="star-query-strip__note">
            <strong>Selection:</strong> 0.1-mag VIS PSF bins, POINT_LIKE_PROB ≥ 0.9,
            showing both Σ POINT_LIKE_PROB and Σ PHZ_STAR_PROB divided by the 63.1 deg² Q1 footprint.
          </p>
          <JobProgressView job={q1Query.job} error={q1Query.error} />
        </CardBody>
      </Card>

      <Card className="star-query-strip">
        <CardHead title="Matched-star sample"
          sub={gaiaSampling?.sampling_kind === "spherical_kmeans_euclid_point_sources"
            ? `${gaiaSampling.cone_count} spherical k-means field centres · ${gaiaSampling.radius_deg?.toFixed(2) ?? "0.35"}° Gaia DR3 cones · ${gaiaSampling.tap_provider ?? "ARI Gaia TAP"} ${gaiaSampling.query_mode ?? "sync"}`
            : coneCount
            ? `${coneCount} cached Euclid cones · next query uses 12 field centres, 0.35° cones, and ARI synchronous TAP`
            : "Cache Euclid cones on Field statistics before querying Gaia"} />
        <CardBody>
          <div className="star-query-strip__content">
            <div className="star-query-strip__stats">
              <Stat k="matched stars" v={(distribution?.matched_stars ?? candidate?.euclid_mapping?.matched_stars ?? 0).toLocaleString()} />
              <Stat k="all-band S/N ≥ 5" v={(distribution?.high_quality_stars ?? 0).toLocaleString()} />
              <Stat k="POINT_LIKE_PROB ≥ 0.9" v={(distribution?.pointlike_over_0_9 ?? 0).toLocaleString()} />
              <Stat k="Gaia sampling" v={gaiaSampling
                ? `${gaiaSampling.cone_count} × ${(gaiaSampling.radius_deg ?? gaiaSampling.radius_arcmin / 60).toFixed(2)}°`
                : "12 × 0.35° on next query"} />
            </div>
            <div className="star-query-strip__actions">
              <Button disabled={!q1Counts || !api.availability.euclid_catalog.cached || query.busy}
                onClick={() => query.run("/api/star-distribution/query", {}, { onDone: refresh })}>
                {query.busy ? "Querying ARI field cones + fitting…" : "Query 12 × 0.35° Gaia cones via ARI sync + fit colours"}
              </Button>
              <Button disabled={!candidate?.valid || activate.busy}
                onClick={() => activate.run("/api/star-distribution/activate", {}, { onDone: refresh })}>
                {api.calibration.is_active ? "Re-activate stellar prior" : "Activate stellar prior"}
              </Button>
              <NavLink className="ui-btn" to="/population-comparison">Euclid cone queries</NavLink>
            </div>
          </div>
          {candidate?.warnings?.[0] && <p className="star-query-strip__note"><strong>Fit note:</strong> {candidate.warnings[0]}</p>}
          {candidate?.coverage_notes?.[0] && <p className="star-query-strip__note"><strong>Coverage:</strong> {candidate.coverage_notes[0]}</p>}
          <JobProgressView job={query.job} error={query.error} />
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
          Query Gaia and fit the stellar distribution to create the six matched-colour plots.
        </Empty>
      )}
    </Page>
  );
}
