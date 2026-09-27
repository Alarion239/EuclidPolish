/* realism/stars (spec §8.3): the stellar prior — Q1 PHZ point-source counts
   set the magnitude density (the native Gaia G_AB counts share its slope), a
   cached fixed-field Gaia–Euclid sample supplies the colours only. Views
   (?view=): density (Euclid magnitude + colour densities) and prior (query →
   fit → activate); the Gaia colour and CMD views were deleted, and an old
   ?view= falls back to density. No galaxy selection is used anywhere here;
   the include-training toggle is the shared header's. GET
   /api/star-distribution never writes. */
import type { ReactNode } from "react";
import Plot, { Legend, type LegendItem } from "../../../charts/Plot";
import { usePageActions } from "../../../app/palette";
import { DASH, formatApprox, formatCount, formatNumber } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { linearTicks } from "../../../ticks";
import {
  Badge, Button, Caption, Card, CardBody, CardHead, Details, EmptyState, FactsList, IconButton, JobProgress, Menu, Num, Page,
  Segmented, SummaryLine, Table, Tooltip, type Column,
} from "../../../ui";
import { Link } from "react-router-dom";
import { useStars, type StarDistribution, type StarPayload } from "../api";
import { ticksFor } from "../chartKit";
import { BarGroup, BarSpacer, Info, LoadState, RealismBar, SkyLink, downloadUrl, useUrlLegend } from "../common";
import { useIncludeTraining } from "../header";
import { JOB, useRealismJob } from "../jobs";
import { activateStars, fitStars, queryStars, starFigureUrl } from "../stars/actions";
import {
  DENSITY_KEYS, DENSITY_ORDER, densityColor, densityDelta, densityDomain, densitySeries, fitGuides, generatedDensity, trustedWindow,
} from "../stars/model";

export const STAR_VIEWS = [
  { value: "density", label: "Density" },
  { value: "prior", label: "Prior" },
] as const;
type View = (typeof STAR_VIEWS)[number]["value"];
const VIEW_SET = new Set<string>(STAR_VIEWS.map((v) => v.value));

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

/** A legend label with the sample it draws ("Q1 PHZ (VIS) · ≈403k"); the bare label when the size is unknown. */
const withSize = (label: string, size: string | null) => (size && size !== DASH ? `${label} · ${size}` : label);
const density3 = (v: number) => formatNumber(v, { sig: 3 });
/** Fitted magnitude limits share one precision (2 decimals), so a range never mixes "18.15–21".  */
const mag = (v: number) => v.toFixed(2);

function Density({ distribution }: { distribution: StarDistribution }) {
  const comparison = distribution.density_comparison;
  const legend = useUrlLegend("shide");
  if (!comparison) return <EmptyState icon="activity" title="The density comparison needs the Q1 counts and a fit" />;
  const vis = comparison.parameters.vis;
  const fourBand = DENSITY_ORDER.some((k) => k !== "vis" && comparison.parameters[k]?.euclid.some((v) => v > 0));
  const generatedArea = comparison.synthetic_area_arcmin2;
  const items: LegendItem[] = [
    ...(vis.point_sources ? [{ label: withSize("Q1 point sources (VIS)", formatApprox(comparison.q1_expected_point_sources)),
      key: DENSITY_KEYS.pointSources, color: densityColor.pointSources() }] : []),
    { label: withSize("Q1 PHZ (VIS)", formatApprox(comparison.q1_phz_expected_stars)), key: DENSITY_KEYS.q1, color: densityColor.q1() },
    ...(fourBand ? [{ label: withSize("Euclid four-band", formatCount(comparison.euclid_color_count)), key: DENSITY_KEYS.fourBand,
      color: densityColor.fourBand() }] : []),
    // Native Gaia G_AB counts share the magnitude law's slope, so they stay (VIS panel only).
    ...(vis.gaia ? [{ label: "native Gaia G_AB", key: DENSITY_KEYS.gaia, color: densityColor.gaia() }] : []),
    ...(vis.gaia_fit ? [{ label: "Gaia shared-slope fit", key: DENSITY_KEYS.gaiaFit, color: densityColor.gaia(), dash: true }] : []),
    { label: "Q1-normalized law / colour draw", key: DENSITY_KEYS.model, color: densityColor.model(), dash: true },
    { label: withSize(`generated ${distribution.training_included ? "train + test + validation" : "test + validation"} stars`,
      comparison.synthetic_star_count != null && generatedArea
        ? `${formatCount(comparison.synthetic_star_count)} in ${formatNumber(generatedArea)} arcmin²` : null),
    key: DENSITY_KEYS.synthetic, color: densityColor.synthetic(), marker: "filled" },
  ];
  const generated = generatedDensity(comparison);
  const delta = densityDelta(comparison);
  const window = trustedWindow(vis);
  const q1Area = comparison.q1_area_arcmin2;
  const fields = distribution.gaia_sampling?.field_count ?? distribution.gaia_sampling?.fields?.length;
  const caption = [
    q1Area ? `Q1 footprint ${formatNumber(q1Area / 3600, { digits: 1 })} deg²` : null,
    // The matched count sits on the "Euclid four-band" legend label when that series is drawn; else here.
    `colours from ${fourBand ? "the" : `${formatCount(comparison.euclid_color_count)}`} Gaia-matched stars in ${fields ? `${fields} fixed Q1 fields` : "the fixed Q1 fields"}`,
  ].filter(Boolean).join(" · ");
  return (
    <div className="rl-stack">
      <SummaryLine>
        {generated != null ? <>Generated <Num>{density3(generated)}</Num> vs prior </> : "Prior "}
        <Num unit="arcmin⁻²">{density3(comparison.model_density_arcmin2)}</Num>
        {delta != null && <> (<Num tone={Math.abs(delta) > 0.05 ? "warn" : undefined}>
          {formatNumber(100 * delta, { digits: Math.abs(delta) < 0.1 ? 1 : 0, signed: true })}%</Num>)</>}
        {window && <>, trusted window VIS <Num>{mag(window[0])}–{mag(window[1])}</Num></>}
      </SummaryLine>
      <Card>
        <CardHead title="Stellar density in Euclid magnitude and colour" sub="Q1 footprint normalization · probability-weighted counts"
          right={<Info label="About the stellar density">
            <p>{comparison.note}</p>
            <p>Q1 is binned at 0.1 mag in VIS; the smaller Gaia shape sample at 0.5 mag.</p>
          </Info>} />
        <CardBody>
          <Legend items={items} {...legend.legendProps} />
          <div className="rl-plot-grid rl-plot-grid--3">
            {DENSITY_ORDER.map((key) => {
              const parameter = comparison.parameters[key];
              const yDomain = densityDomain(parameter);
              const isVis = key === "vis";
              return (
                <Panel key={key} wide={isVis} title={parameter.label}
                  sub={isVis ? "guides mark the fitted regions" : "no simulated noise"}>
                  <Plot xDomain={parameter.x_domain} yDomain={yDomain} yScale="log"
                    xTicks={linearTicks(parameter.x_domain, { count: isVis ? 8 : 5 })} yTicks={ticksFor(yDomain, "log")}
                    xLabel={parameter.x_label} yLabel="stars / arcmin² / mag (log scale)"
                    series={densitySeries(parameter, !!distribution.training_included, key)}
                    guides={isVis ? fitGuides(parameter) : []} aspect={isVis ? 0.34 : 0.66}
                    syncKey={isVis ? undefined : "star-colour-density"} exportName={`star-density-${key}`}
                    aria-label={`Stellar density vs ${parameter.label}`} {...legend.plotProps} />
                </Panel>
              );
            })}
          </div>
          <Caption>{caption}</Caption>
        </CardBody>
      </Card>
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
    { header: "", cell: (f) => <SkyLink layers={["q1-tiles:0.2"]} ra={f.ra} dec={f.dec} fov={1} hint="The Q1 tiles around this field" /> },
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
          {q1 ? (
            <FactsList title="Query result" facts={[
              { label: "Objects selected", value: formatApprox(q1.selected_point_sources), hint: q1.selection },
              { label: "Point sources", value: formatApprox(q1.expected_point_sources), hint: "Σ POINT_LIKE_PROB over the selected objects" },
              { label: "PHZ stars", value: formatApprox(q1.expected_stars), hint: "Σ PHZ_STAR_PROB over the selected objects" },
              { label: "Footprint", value: formatNumber(q1.footprint_area_deg2, { digits: 1 }), unit: "deg²" },
            ]} />
          ) : <p className="rl-note">The Q1 stellar counts have not been queried yet.</p>}
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
          right={<>
            {!api.calibration.is_active && (  // the OK state is the bar's quiet "prior active" alone
              <Badge tone={candidate?.valid ? "warn" : undefined}>{candidate?.valid ? "candidate ready" : candidate ? "needs a refit" : "not fitted"}</Badge>
            )}
            <Info label="About the stellar fit">
              <p>The straight count fit keeps Q1 at 0.1-mag resolution and bins the smaller Gaia shape sample at 0.5 mag.</p>
              {candidate?.coverage_notes?.map((n) => <p key={n}>{n}</p>)}
            </Info>
          </>} />
        <CardBody>
          {d ? (
            <SummaryLine>
              Colours fitted on <Num>{formatCount(d.high_quality_stars)}</Num> stars with S/N ≥ 5 in all bands,
              of <Num>{formatCount(d.matched_stars)}</Num> matched
            </SummaryLine>
          ) : (
            <p className="rl-note">{api.color_sample.cached
              ? "The Gaia–Euclid colour sample is cached; fit the prior to derive the colours."
              : "No Gaia–Euclid colour sample is cached yet: run the stellar query first."}</p>
          )}
          {candidate?.warnings?.[0] && <p className="rl-note"><strong>Fit note:</strong> {candidate.warnings[0]}</p>}
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
          {sampling?.fields?.length ? (
            <Details summary="Gaia colour fields" className="rl-details">
              <Caption>{`r = ${sampling.radius_arcmin.toFixed(0)}′ each`}</Caption>
              <Table columns={fieldColumns} rows={sampling.fields} rowKey={(f) => f.name ?? `${f.ra},${f.dec}`} />
            </Details>
          ) : null}
        </CardBody>
      </Card>
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
  const candidate = api?.calibration.candidate;
  usePageActions([
    ...STAR_VIEWS.map((v) => ({ id: `stars-view-${v.value}`, label: `Stars: ${v.label}`, group: "Stars", run: () => setView(v.value) })),
    { id: "stars-query", label: "Query stars · MER + PHZ + Gaia", group: "Stars", keywords: ["euclid", "gaia"],
      disabled: !api?.authenticated, run: () => { void queryStars(); } },
    { id: "stars-fit", label: "Fit the stellar prior from cached data", group: "Stars",
      disabled: !api?.q1_counts || !api?.color_sample.cached, run: () => { void fitStars(); } },
    { id: "stars-activate", label: "Activate the stellar prior", group: "Stars", disabled: !candidate?.valid,
      run: () => { void activateStars(!!api?.calibration.is_active); } },
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
            {api.calibration.is_active ? <span className="rl-quiet">prior active</span> : (  // a badge only on a problem
              <Badge size="sm" dot tone={candidate?.valid ? "warn" : "bad"}>
                {candidate?.valid ? "candidate not active" : candidate ? "needs a refit" : "not fitted"}
              </Badge>
            )}
          </BarGroup>
        )}
        <BarSpacer />
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
            : <Density distribution={distribution} />)}
      </LoadState>
    </Page>
  );
}
