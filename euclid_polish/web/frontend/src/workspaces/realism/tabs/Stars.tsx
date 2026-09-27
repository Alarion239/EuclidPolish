/* realism/stars (spec §8.3): the stellar prior — Q1 PHZ point-source counts
   set the magnitude density, a cached fixed-field Gaia–Euclid sample supplies
   the colours only. Views (?view=): density (Euclid magnitude + colour
   densities), colours (Gaia BP−RP vs the six Euclid colours), gaia (the CMD
   and the projection into Euclid) and prior (query → fit → activate). No
   galaxy selection is used anywhere here; the include-training toggle is the
   shared header's. GET /api/star-distribution never writes. */
import type { ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import Plot, { Legend, type LegendItem } from "../../../charts/Plot";
import { usePageActions } from "../../../app/palette";
import { formatCount, formatNumber } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { linearTicks, magnitudeTicks } from "../../../ticks";
import {
  Badge, Button, Card, CardBody, CardHead, EmptyState, IconButton, JobProgress, Menu, Page, Segmented, Stat, Table,
  Tooltip, type Column,
} from "../../../ui";
import { Link } from "react-router-dom";
import { useStars, type StarDistribution, type StarPayload } from "../api";
import { ticksFor } from "../chartKit";
import { BarGroup, BarSpacer, Info, LoadState, RealismBar, SkyLink, StatStrip, atlasHref, downloadUrl, useUrlLegend } from "../common";
import { useIncludeTraining } from "../header";
import { JOB, useRealismJob } from "../jobs";
import { activateStars, fitStars, queryStars, starFigureUrl } from "../stars/actions";
import {
  COLOR_ORDER, DENSITY_KEYS, DENSITY_ORDER, PROJECTION_ORDER, cmdSeries, correlationSeries, densityColor,
  densityDomain, densitySeries, fitGuides, projectionSeries,
} from "../stars/model";
import { C } from "../../../colors";

export const STAR_VIEWS = [
  { value: "density", label: "Density" },
  { value: "colours", label: "Colours" },
  { value: "gaia", label: "Gaia" },
  { value: "prior", label: "Prior" },
] as const;
type View = (typeof STAR_VIEWS)[number]["value"];
const VIEW_SET = new Set<string>(STAR_VIEWS.map((v) => v.value));
const GAIA_LAYERS = ["q1-tiles:0.2", "gaia-fields"];

function NoDistribution({ onPrior }: { onPrior: () => void }) {
  return (
    <EmptyState icon="activity" title="No stellar distribution yet"
      action={<Button size="sm" onClick={onPrior}>Open the prior workflow</Button>}>
      Query the stellar MER + PHZ and Gaia data, then fit the cached colours.
    </EmptyState>
  );
}

function Panel({ title, sub, children, wide }: { title: ReactNode; sub?: ReactNode; children: ReactNode; wide?: boolean }) {
  return (
    <article className={`rl-panel${wide ? " rl-wide" : ""}`}>
      <header className="rl-panel__head"><strong>{title}</strong>{sub && <small>{sub}</small>}</header>
      {children}
    </article>
  );
}

function Density({ distribution }: { distribution: StarDistribution }) {
  const comparison = distribution.density_comparison;
  const legend = useUrlLegend("shide");
  if (!comparison) return <EmptyState icon="activity" title="The density comparison needs the Q1 counts and a fit" />;
  const items: LegendItem[] = [
    ...(comparison.parameters.vis.point_sources ? [{ label: "Q1 point sources (VIS)", key: DENSITY_KEYS.pointSources, color: densityColor.pointSources() }] : []),
    { label: "Q1 PHZ (VIS)", key: DENSITY_KEYS.q1, color: densityColor.q1() },
    { label: "native Gaia G_AB", key: DENSITY_KEYS.gaia, color: densityColor.gaia() },
    ...(comparison.parameters.vis.gaia_fit ? [{ label: "Gaia shared-slope fit", key: DENSITY_KEYS.gaiaFit, color: densityColor.gaia(), dash: true }] : []),
    { label: "Q1-normalized law / colour draw", key: DENSITY_KEYS.model, color: densityColor.model(), dash: true },
    { label: `generated ${distribution.training_included ? "train + test + validation" : "test + validation"} stars`,
      key: DENSITY_KEYS.synthetic, color: densityColor.synthetic(), marker: "filled" },
  ];
  const area = (v: number | null | undefined) => formatNumber(v, { digits: 0 });
  return (
    <div className="rl-stack">
      <StatStrip label="Density comparison sample sizes">
        <Stat k="Q1 area" v={`${area(comparison.q1_area_arcmin2 ?? comparison.area_arcmin2)} arcmin²`} />
        <Stat k="Gaia field area" v={`${area(comparison.gaia_area_arcmin2 ?? comparison.area_arcmin2)} arcmin²`} />
        <Stat k="PHZ expected stars" v={formatNumber(comparison.q1_phz_expected_stars, { digits: 1 })} />
        <Stat k="expected point sources" v={formatNumber(comparison.q1_expected_point_sources, { digits: 1 })} />
        <Stat k="Euclid four-band" v={formatCount(comparison.euclid_color_count)} />
        <Stat k="native Gaia G_AB" v={formatCount(comparison.gaia_native_g_count)} />
        <Stat k="Gaia colour projection" v={formatCount(comparison.gaia_count)} />
        <Stat k="generated stars" v={`${formatCount(comparison.synthetic_star_count)} / ${formatNumber(comparison.synthetic_area_arcmin2, { digits: 1 })} arcmin²`} />
        <Stat k="model density" v={`${comparison.model_density_arcmin2.toFixed(3)} arcmin⁻²`} />
      </StatStrip>
      <Card>
        <CardHead title="Stellar density in Euclid magnitude and colour" sub="Q1 footprint normalization · probability-weighted counts"
          right={<Info label="About the stellar density">{comparison.note}</Info>} />
        <CardBody>
          <Legend items={items} {...legend.legendProps} />
          <div className="rl-plot-grid rl-plot-grid--3">
            {DENSITY_ORDER.map((key) => {
              const parameter = comparison.parameters[key];
              const yDomain = densityDomain(parameter);
              const vis = key === "vis";
              return (
                <Panel key={key} wide={vis} title={parameter.label}
                  sub={vis ? "Q1 at 0.1 mag · Gaia shape sample at 0.5 mag · guides mark the fitted regions" : "no simulated noise"}>
                  <Plot xDomain={parameter.x_domain} yDomain={yDomain} yScale="log"
                    xTicks={linearTicks(parameter.x_domain, { count: vis ? 8 : 5 })} yTicks={ticksFor(yDomain, "log")}
                    xLabel={parameter.x_label} yLabel="stars / arcmin² / mag (log scale)"
                    series={densitySeries(parameter, !!distribution.training_included)}
                    guides={vis ? fitGuides(parameter) : []} aspect={vis ? 0.34 : 0.66}
                    syncKey={vis ? undefined : "star-colour-density"} exportName={`star-density-${key}`}
                    aria-label={`Stellar density vs ${parameter.label}`} {...legend.plotProps} />
                </Panel>
              );
            })}
          </div>
        </CardBody>
      </Card>
    </div>
  );
}

function Colours({ distribution }: { distribution: StarDistribution }) {
  const legend = useUrlLegend("chide");
  const items: LegendItem[] = [
    { label: "catalogue stars", key: "stars", color: C.mean, marker: "filled" },
    { label: "2σ intrinsic", key: "2σ", color: C.comb, filled: true },
    { label: "1σ intrinsic", key: "1σ", color: C.comb, filled: true },
    { label: "fitted locus", key: "locus", color: C.comb, line: true },
  ];
  return (
    <Card>
      <CardHead title="Gaia colour versus fitted Euclid distributions" sub={`measured colours · ${formatCount(distribution.matched_stars)} matched stars`}
        right={<Info label="About the colour fits"><p>{distribution.axis_note}</p>{distribution.fit_note && <p>{distribution.fit_note}</p>}</Info>} />
      <CardBody>
        <Legend items={items} {...legend.legendProps} />
        <div className="rl-plot-grid rl-plot-grid--3">
          {COLOR_ORDER.map((key) => {
            const item = distribution.colors[key];
            return (
              <Panel key={key} title={item.label}
                sub={`r = ${item.pearson_r == null ? "—" : item.pearson_r.toFixed(3)}${item.fit ? ` · σ = ${item.fit.sigma.toFixed(3)}` : ""}`}>
                <Plot xDomain={distribution.x_domain} yDomain={item.y_domain}
                  xTicks={linearTicks(distribution.x_domain, { count: 5 })} yTicks={linearTicks(item.y_domain, { count: 5 })}
                  xLabel="Gaia BP − RP [mag]" yLabel={`${item.label} [AB mag]`} series={correlationSeries(distribution, key)}
                  aspect={0.7} syncKey="star-colours" exportName={`star-colour-${key}`}
                  aria-label={`Gaia BP − RP versus ${item.label}`} {...legend.plotProps} />
              </Panel>
            );
          })}
        </div>
      </CardBody>
    </Card>
  );
}

function Gaia({ distribution }: { distribution: StarDistribution }) {
  const cmd = distribution.gaia_cmd;
  const projection = distribution.euclid_projection;
  const gDomain: [number, number] = [-cmd.g_domain[1], -cmd.g_domain[0]];
  return (
    <div className="rl-stack">
      <Card>
        <CardHead title="Gaia colour–magnitude diagram" sub="all cached Gaia sources · apparent magnitudes"
          right={<Info label="About the Gaia CMD"><p>{cmd.note}</p>
            {cmd.without_color > 0 && <p>{formatCount(cmd.without_color)} cached sources without BP−RP stay stored but cannot be plotted.</p>}</Info>} />
        <CardBody>
          <Plot xDomain={cmd.x_domain} yDomain={gDomain} xTicks={linearTicks(cmd.x_domain, { count: 6 })}
            yTicks={magnitudeTicks(cmd.g_domain, { invert: true })} xLabel="Gaia BP − RP [mag]" yLabel="Gaia G [mag] · brighter ↑"
            series={cmdSeries(distribution)} legend="auto" aspect={0.46} exportName="gaia-cmd"
            yFormat={(v) => (-v).toFixed(2)} aria-label="Gaia colour–magnitude diagram" />
        </CardBody>
      </Card>
      {projection && (
        <Card>
          <CardHead title="Gaia population projected into Euclid" sub="all Gaia sources, transformed by the fitted locus"
            right={<Info label="About the projection">{projection.note}</Info>} />
          <CardBody>
            <div className="rl-plot-grid rl-plot-grid--3">
              {PROJECTION_ORDER.map((key) => {
                const color = projection.colors[key];
                const visDomain: [number, number] = [-projection.vis_domain[1], -projection.vis_domain[0]];
                return (
                  <Panel key={key} title={`VIS vs ${color.label}`}
                    sub={`intrinsic σ = ${color.sigma.toFixed(3)} mag · Euclid N = ${formatCount(projection.euclid_observed[key].vis_mag.length)}`}>
                    <Plot xDomain={color.x_domain} yDomain={visDomain} xTicks={linearTicks(color.x_domain, { count: 5 })}
                      yTicks={magnitudeTicks(projection.vis_domain, { invert: true })} xLabel={`${color.label} [AB mag]`}
                      yLabel="VIS [AB mag] · brighter ↑" series={projectionSeries(distribution, key)} legend="auto"
                      aspect={0.8} syncKey="star-projection" yFormat={(v) => (-v).toFixed(2)}
                      exportName={`star-projection-${key}`} aria-label={`VIS versus ${color.label} (projected Gaia)`} />
                  </Panel>
                );
              })}
            </div>
          </CardBody>
        </Card>
      )}
    </div>
  );
}

type GaiaField = { name?: string; ra: number; dec: number; rows?: number };

function Prior({ api }: { api: StarPayload }) {
  const query = useRealismJob(JOB.starQuery);
  const fit = useRealismJob(JOB.starFit);
  const activate = useRealismJob(JOB.starActivate);
  const q1 = api.q1_counts;
  const d = api.distribution;
  const candidate = api.calibration.candidate;
  const login = !api.authenticated;
  const sampling = d?.gaia_sampling;
  const fieldColumns: Column<GaiaField>[] = [
    { header: "field", cell: (f) => f.name ?? "—" },
    { header: "RA, Dec", align: "right", cell: (f) => <span className="rl-num">{f.ra.toFixed(3)}, {f.dec.toFixed(3)}</span> },
    { header: "rows", align: "right", cell: (f) => formatCount(f.rows) },
    { header: "", cell: (f) => <SkyLink layers={GAIA_LAYERS} ra={f.ra} dec={f.dec} fov={1} inspect={f.name ? `source:gaia-fields/${f.name}` : undefined} /> },
  ];
  return (
    <div className="rl-stack">
      <Card>
        <CardHead title="Q1 point-source and stellar counts" sub="stellar MER + PHZ brackets and fixed-field Gaia colours"
          right={<Info label="About the stellar query">
            <p>0.1-mag VIS PSF bins, POINT_LIKE_PROB ≥ 0.9, as Σ POINT_LIKE_PROB and Σ PHZ_STAR_PROB over the Q1
              footprint. The fixed-field colour query uses the same point-like threshold and Gaia matches.</p>
          </Info>} />
        <CardBody>
          <StatStrip label="Q1 stellar counts">
            <Stat k="Q1 footprint" v={q1 ? `${q1.footprint_area_deg2.toFixed(1)} deg²` : "—"} />
            <Stat k="VIS bin width" v="0.1 mag" />
            <Stat k="expected point sources" v={q1 ? formatNumber(q1.expected_point_sources, { digits: 1 }) : "not queried"} />
            <Stat k="PHZ expected stars" v={q1 ? formatNumber(q1.expected_stars, { digits: 1 }) : "not queried"} />
            <Stat k="selected objects" v={q1 ? formatCount(q1.selected_point_sources) : "not queried"} />
            <Stat k="bins" v={q1 ? formatCount(q1.bins.length) : "not queried"} />
          </StatStrip>
          <p className="rl-note">No galaxy selection is used by this action.</p>
          <div className="rl-row">
            <Tooltip content={login ? "Log in to the Euclid archive (Settings › Connections) first" : "Queries the Euclid archive"}>
              <span tabIndex={login ? 0 : -1}>
                <Button variant="primary" icon="activity" loading={query.busy} disabled={login} onClick={() => void queryStars()}>
                  {query.busy ? "Querying stars…" : "Query stars · MER + PHZ + Gaia"}
                </Button>
              </span>
            </Tooltip>
            {login && <Button asChild size="sm" variant="ghost"><Link to="/settings/connections">Log in to Euclid archive</Link></Button>}
          </div>
          <JobProgress job={query.job} error={query.error} />
        </CardBody>
      </Card>
      <Card>
        <CardHead title="Stellar fit and activation" sub="Q1 brackets set the magnitude density; the Gaia–Euclid sample supplies colours only"
          right={<Badge tone={api.calibration.is_active ? "good" : candidate?.valid ? "warn" : undefined}>
            {api.calibration.is_active ? "active stellar prior" : candidate?.valid ? "candidate ready" : candidate ? "needs a refit" : "not fitted"}
          </Badge>} />
        <CardBody>
          <StatStrip label="Stellar colour sample">
            <Stat k="matched stars" v={formatCount(d?.matched_stars ?? candidate?.euclid_mapping?.matched_stars ?? 0)} />
            <Stat k="all-band S/N ≥ 5" v={formatCount(d?.high_quality_stars ?? 0)} />
            <Stat k="POINT_LIKE_PROB ≥ 0.9" v={formatCount(d?.pointlike_over_0_9 ?? 0)} />
            <Stat k="colour sample" v={api.color_sample.cached ? `${formatCount(api.color_sample.euclid?.rows ?? 0)} Euclid candidates` : "not cached"} />
            <Stat k="Gaia rows" v={formatCount(api.color_sample.gaia?.rows ?? 0)} />
            <Stat k="fixed Q1 fields" v={formatCount(api.color_sample.gaia?.field_count ?? 0)} />
          </StatStrip>
          <p className="rl-note">The straight count fit keeps Q1 at 0.1-mag resolution and bins the smaller Gaia shape sample at 0.5 mag.</p>
          {candidate?.warnings?.[0] && <p className="rl-note"><strong>Fit note:</strong> {candidate.warnings[0]}</p>}
          {candidate?.coverage_notes?.[0] && <p className="rl-note"><strong>Coverage:</strong> {candidate.coverage_notes[0]}</p>}
          <div className="rl-row">
            <Button loading={fit.busy} disabled={!q1 || !api.color_sample.cached || query.busy} onClick={() => void fitStars()}>
              Fit stellar prior from cached data
            </Button>
            <Button variant="primary" loading={activate.busy} disabled={!candidate?.valid || fit.busy}
              onClick={() => void activateStars(api.calibration.is_active)}>
              {api.calibration.is_active ? "Re-activate stellar prior" : "Activate stellar prior"}
            </Button>
          </div>
          <JobProgress job={fit.job} error={fit.error} />
          <JobProgress job={activate.job} error={activate.error} />
        </CardBody>
      </Card>
      {sampling?.fields?.length ? (
        <Card>
          <CardHead title="Gaia colour fields" sub={`${sampling.fields.length} fixed Q1 fields · r = ${sampling.radius_arcmin.toFixed(0)}′ · ${formatNumber(sampling.area_arcmin2, { digits: 0 })} arcmin²`}
            right={<SkyLink layers={GAIA_LAYERS} hint="The Gaia colour fields on the sky atlas" />} />
          <CardBody><Table columns={fieldColumns} rows={sampling.fields} rowKey={(f) => f.name ?? `${f.ra},${f.dec}`} /></CardBody>
        </Card>
      ) : null}
    </div>
  );
}

export default function StarsTab() {
  const [training] = useIncludeTraining();
  const [rawView, setView] = useUrlState("view", "density");
  const view: View = (VIEW_SET.has(rawView) ? rawView : "density") as View;
  const query = useRealismJob(JOB.starQuery);
  const resource = useStars(training, query.busy ? 1500 : undefined);
  const api = resource.data;
  const navigate = useNavigate();
  const candidate = api?.calibration.candidate;
  usePageActions([
    ...STAR_VIEWS.map((v) => ({ id: `stars-view-${v.value}`, label: `Stars: ${v.label}`, group: "Stars", run: () => setView(v.value) })),
    { id: "stars-query", label: "Query stars · MER + PHZ + Gaia", group: "Stars", keywords: ["euclid", "gaia"],
      disabled: !api?.authenticated, run: () => { void queryStars(); } },
    { id: "stars-fit", label: "Fit the stellar prior from cached data", group: "Stars",
      disabled: !api?.q1_counts || !api?.color_sample.cached, run: () => { void fitStars(); } },
    { id: "stars-activate", label: "Activate the stellar prior", group: "Stars", disabled: !candidate?.valid,
      run: () => { void activateStars(!!api?.calibration.is_active); } },
    { id: "stars-gaia-sky", label: "Show the Gaia colour fields on the sky", group: "Stars", keywords: ["atlas"],
      run: () => navigate(atlasHref({ layers: GAIA_LAYERS })) },
  ]);
  const distribution = api?.distribution ?? null;
  const toPrior = () => setView("prior");
  return (
    <Page className="rl-page">
      <RealismBar label="Star controls">
        <BarGroup>
          <Segmented size="sm" aria-label="Star view" value={view} onChange={setView} options={[...STAR_VIEWS]} />
        </BarGroup>
        {api && (
          <BarGroup>
            <Badge size="sm" dot tone={api.calibration.is_active ? "good" : candidate?.valid ? "warn" : "bad"}>
              {api.calibration.is_active ? "prior active" : candidate?.valid ? "candidate not active" : candidate ? "needs a refit" : "not fitted"}
            </Badge>
          </BarGroup>
        )}
        <BarSpacer />
        <SkyLink layers={GAIA_LAYERS} hint="The Gaia colour fields on the sky atlas">Gaia fields</SkyLink>
        {candidate && (
          <Menu label="Download the stellar calibration figure"
            trigger={<IconButton size="sm" icon="download" label="Download the stellar calibration figure" />}
            items={(["png", "pdf", "svg"] as const).map((f) => ({
              label: `Calibration figure · ${f.toUpperCase()}`,
              onSelect: () => downloadUrl(starFigureUrl(f), `euclidpolish_star_population_calibration.${f}`),
            }))} />
        )}
        <IconButton size="sm" icon="reset" label="Refresh" loading={resource.fetching} onClick={() => resource.reload()} />
      </RealismBar>
      <LoadState loading={resource.loading && !api} error={resource.error} onRetry={resource.reload}>
        {api && (view === "prior" ? <Prior api={api} />
          : !distribution ? <NoDistribution onPrior={toPrior} />
            : view === "density" ? <Density distribution={distribution} />
              : view === "colours" ? <Colours distribution={distribution} />
                : <Gaia distribution={distribution} />)}
      </LoadState>
    </Page>
  );
}
