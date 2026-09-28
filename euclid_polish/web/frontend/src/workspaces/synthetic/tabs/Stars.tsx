/* Synthetic › Stars: does the stellar prior match Euclid Q1, and do the
   generated stars follow it? One view (GET /api/star-distribution, read-only):
   the verdict line (generated vs prior density, the trusted fit window),
   the legend with the sample sizes, the wide VIS density panel with the
   trusted window shaded (Q1 point sources, Q1 PHZ stars, the model law, the
   generated stars; the native Gaia counts are not drawn), then
   the six colour panels as unit-area PDFs (Gaia-matched Q1 stars, the
   model's colour draws, the generated stars) and one caption (the Q1 area,
   the matched stars and their fields). Then the drawers: Prior (`?prior=1`:
   the fit sample in one sentence, Fit and Activate, both confirmed, the Gaia
   colour-field inputs collapsed) and How this is produced (`?how=1`: the
   confirmed "Query stars · MER + PHZ + Gaia", its job, the query result as
   facts). The Gaia colour and CMD views were deleted: an old `?view=` shows
   this view (`?view=prior` opens the Prior drawer). The shared header's
   training toggle changes only the generated curve (a cue says so). */
import { useEffect, type ReactNode } from "react";
import { Link } from "react-router-dom";
import Plot, { Legend, type LegendItem } from "../../../charts/Plot";
import { usePageActions } from "../../../app/palette";
import { DASH, formatApprox, formatCount, formatNumber } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { linearTicks } from "../../../ticks";
import {
  Badge, Button, Caption, Details, EmptyState, FactsList, IconButton, JobProgress, Menu, Num, Page, SummaryLine, Table,
  Toolbar, ToolbarGroup, ToolbarSpacer, Tooltip, type Column,
} from "../../../ui";
import { useStars, type ModelNoise, type StarDistribution, type StarPayload } from "../api";
import { formatSigned, ticksFor } from "../chartKit";
import { Drawer, DrawerButton, Info, LoadState, SkyLink, downloadUrl, useDrawer, useUrlLegend } from "../common";
import { useIncludeTraining } from "../header";
import { JOB, useRealismJob } from "../jobs";
import { activateStars, fitStars, queryStars, starFigureUrl } from "../stars/actions";
import {
  DENSITY_KEYS, STAR_LEGEND, colourPdfSeries, densityColor, densityDelta, densityDomain, densitySeries, fitGuides,
  generatedDensity, modelNoiseNote, pdfDomain, starModelLabel, trustedBand, trustedWindow,
} from "../stars/model";
import "../synthetic.css";

const COLOUR_KEYS = ["vis_y", "vis_j", "vis_h", "y_j", "y_h", "j_h"] as const;

/** A legend label with the sample it draws ("Q1 PHZ stars · ≈403k"); the bare label when the size is unknown. */
const withSize = (label: string, size: string | null) => (size && size !== DASH ? `${label} · ${size}` : label);
const density3 = (v: number) => formatNumber(v, { sig: 3 });
/** A fitted magnitude range at one shared precision ("17.0–23.5", "18.15–21.00"), never mixed. */
function magRange(lo: number, hi: number): string {
  const tenths = (v: number) => Math.abs(v * 10 - Math.round(v * 10)) < 1e-6;
  const digits = tenths(lo) && tenths(hi) ? 1 : 2;
  return `${lo.toFixed(digits)}–${hi.toFixed(digits)}`;
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
  const vis = comparison.parameters.vis;
  const generatedArea = comparison.synthetic_area_arcmin2;
  const hasColours = COLOUR_KEYS.some((k) => comparison.parameters[k]?.euclid.some((v) => v > 0));
  const items: LegendItem[] = [
    ...(vis.point_sources ? [{ label: withSize(STAR_LEGEND.pointSources, formatApprox(comparison.q1_expected_point_sources)),
      key: DENSITY_KEYS.pointSources, color: densityColor.pointSources() }] : []),
    { label: withSize(STAR_LEGEND.q1, formatApprox(comparison.q1_phz_expected_stars)), key: DENSITY_KEYS.q1, color: densityColor.q1() },
    ...(hasColours ? [{ label: withSize(STAR_LEGEND.fourBand, formatCount(comparison.euclid_color_count)), key: DENSITY_KEYS.fourBand,
      color: densityColor.fourBand() }] : []),
    { label: starModelLabel(comparison.model_color_noise), key: DENSITY_KEYS.model, color: densityColor.model(), dash: true },
    { label: withSize(`${STAR_LEGEND.synthetic} (${distribution.training_included ? "train + test + validate" : "test + validate"})`,
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
    `colours from ${hasColours ? "the" : formatCount(comparison.euclid_color_count)} Gaia-matched stars in ${fields ? `${fields} fixed Q1 fields` : "the fixed Q1 fields"}`,
    `the colour panels are unit-area PDFs; ${modelNoiseNote(comparison.model_color_noise)}`,
  ].filter(Boolean).join(" · ");
  const yDomain = densityDomain(vis);
  return (
    <div className="rl-stack">
      <SummaryLine>
        {generated != null ? <>Generated <Num>{density3(generated)}</Num> vs prior </> : "Prior "}
        <Num unit="stars arcmin⁻²">{density3(comparison.model_density_arcmin2)}</Num>
        {delta != null && <> (<Num tone={Math.abs(delta) > 0.05 ? "warn" : undefined}>
          {formatSigned(100 * delta, { digits: Math.abs(delta) < 0.1 ? 1 : 0, signed: true })}%</Num>)</>}
        {window && <>, trusted window VIS <Num>{magRange(window[0], window[1])}</Num></>}
      </SummaryLine>
      <Legend items={items} {...legend.legendProps} />
      <Caption>The training toggle adds the train split to the generated stars only; the Q1 and model curves do not change.</Caption>
      <div className="rl-plot-grid rl-plot-grid--3">
        <Panel wide title="Stellar density in VIS" sub="Q1 footprint normalisation · the shaded window is where the Q1 counts were fitted">
          <Plot xDomain={vis.x_domain} yDomain={yDomain} yScale="log"
            xTicks={linearTicks(vis.x_domain, { count: 8 })} yTicks={ticksFor(yDomain, "log")}
            xLabel={vis.x_label} yLabel="stars / arcmin² / mag (log scale)"
            series={densitySeries(vis, !!distribution.training_included, "vis")
              .map((s) => ({ ...s, name: legendName(s.key, comparison.model_color_noise) ?? s.name }))}
            guides={fitGuides(vis)} bands={trustedBand(vis)} aspect={0.34} exportName="star-density-vis"
            aria-label="Stellar density against VIS magnitude" {...legend.plotProps} />
        </Panel>
        {COLOUR_KEYS.map((key) => {
          const parameter = comparison.parameters[key];
          if (!parameter) return null;
          const series = colourPdfSeries(parameter, comparison.model_color_noise);
          const y = pdfDomain(series);
          return (
            <Panel key={key} title={parameter.label}>
              <Plot xDomain={parameter.x_domain} yDomain={y}
                xTicks={linearTicks(parameter.x_domain, { count: 5 })} yTicks={linearTicks(y, { count: 4 })}
                xLabel={parameter.x_label} yLabel="probability density (per mag)" series={series}
                aspect={0.66} syncKey="star-colour-pdf" exportName={`star-colour-${key}`}
                aria-label={`Colour distribution ${parameter.label}`} {...legend.plotProps} />
            </Panel>
          );
        })}
      </div>
      <Caption>{caption}</Caption>
    </div>
  );
}

/** The VIS panel's series named as the legend names them. */
function legendName(key: string | undefined, noise?: ModelNoise | null): string | undefined {
  switch (key) {
    case DENSITY_KEYS.pointSources: return STAR_LEGEND.pointSources;
    case DENSITY_KEYS.q1: return STAR_LEGEND.q1;
    case DENSITY_KEYS.model: return starModelLabel(noise);
    case DENSITY_KEYS.synthetic: return STAR_LEGEND.synthetic;
    default: return undefined;
  }
}

type GaiaField = { name?: string; ra: number; dec: number; rows?: number };

const FIELD_COLUMNS: Column<GaiaField>[] = [
  { header: "Field", cell: (f) => f.name ?? "—" },
  { header: "RA, Dec", align: "right", cell: (f) => <span className="rl-num">{f.ra.toFixed(3)}, {f.dec.toFixed(3)}</span> },
  { header: "Stars", align: "right", cell: (f) => formatCount(f.rows) },
  { header: "", cell: (f) => <SkyLink layers={["q1-tiles:0.2"]} ra={f.ra} dec={f.dec} fov={1} hint="The Q1 tiles around this field" /> },
];

function PriorDrawer({ api }: { api: StarPayload }) {
  const fit = useRealismJob(JOB.starFit);
  const activate = useRealismJob(JOB.starActivate);
  const query = useRealismJob(JOB.starQuery);
  const d = api.distribution;
  const candidate = api.calibration.candidate;
  const sampling = d?.gaia_sampling;
  return (
    <Drawer flag="prior" title="Prior" sub="Q1 counts set the magnitude law; the Gaia–Euclid sample supplies the colours">
      {d ? (
        <p className="syn-sentence">
          Colours fitted on <Num>{formatCount(d.high_quality_stars)}</Num> stars with S/N ≥ 5 in all bands,
          of <Num>{formatCount(d.matched_stars)}</Num> matched
        </p>
      ) : (
        <p className="rl-note">{api.color_sample.cached
          ? "The Gaia–Euclid colour sample is cached; fit the prior to derive the colours."
          : "No Gaia–Euclid colour sample is cached yet: run the stellar query (How this is produced) first."}</p>
      )}
      {candidate?.warnings?.[0] && <p className="rl-note"><strong>Fit note:</strong> {candidate.warnings[0]}</p>}
      <div className="rl-row">
        <Button loading={fit.busy} disabled={!api.q1_counts || !api.color_sample.cached || query.busy} onClick={() => void fitStars()}>
          Fit stellar prior from cached data
        </Button>
        <Button variant="primary" loading={activate.busy} disabled={!candidate?.valid || fit.busy}
          onClick={() => void activateStars(api.calibration.is_active)}>
          {api.calibration.is_active ? "Re-activate stellar prior" : "Activate stellar prior"}
        </Button>
        <Info label="About the stellar fit">
          <p>The straight count fit keeps Q1 at 0.1-mag resolution and bins the smaller Gaia shape sample at 0.5 mag.</p>
          {candidate?.coverage_notes?.map((n) => <p key={n}>{n}</p>)}
        </Info>
      </div>
      <JobProgress job={fit.job} error={fit.error} />
      <JobProgress job={activate.job} error={activate.error} />
      {sampling?.fields?.length ? (
        <Details summary="Inputs: the Gaia colour fields" className="rl-details">
          <Caption>{`${formatCount(sampling.fields.length)} fixed Q1 fields, r = ${sampling.radius_arcmin.toFixed(0)}′ each`}</Caption>
          <Table columns={FIELD_COLUMNS} rows={sampling.fields} rowKey={(f) => f.name ?? `${f.ra},${f.dec}`} />
        </Details>
      ) : null}
    </Drawer>
  );
}

function HowDrawer({ api }: { api: StarPayload }) {
  const query = useRealismJob(JOB.starQuery);
  const q1 = api.q1_counts;
  const login = !api.authenticated;
  return (
    <Drawer flag="how" title="How this is produced" sub="the Q1 stellar counts and the Gaia–Euclid colour sample">
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
        <Tooltip content={login ? "Log in to the Euclid archive (System › Connections) first" : "Queries the Euclid archive (confirmed)"}>
          <span tabIndex={login ? 0 : -1}>
            <Button variant="primary" icon="activity" loading={query.busy} disabled={login} onClick={() => void queryStars()}>
              {query.busy ? "Querying stars…" : "Query stars · MER + PHZ + Gaia"}
            </Button>
          </span>
        </Tooltip>
        {login && <Button asChild size="sm" variant="ghost"><Link to="/system/connections">Log in to the Euclid archive</Link></Button>}
        <Info label="About the stellar query">
          <p>0.1-mag VIS PSF bins of point-like objects, summed as Σ POINT_LIKE_PROB and Σ PHZ_STAR_PROB over the Q1
            footprint. The fixed-field colour query uses the same point-like threshold and Gaia matches.</p>
        </Info>
      </div>
      <JobProgress job={query.job} error={query.error} />
    </Drawer>
  );
}

export default function StarsTab() {
  const [training] = useIncludeTraining();
  const query = useRealismJob(JOB.starQuery);
  const resource = useStars(training, query.busy ? 1500 : undefined);
  const api = resource.data;
  const candidate = api?.calibration.candidate;
  const prior = useDrawer("prior");
  const how = useDrawer("how");
  const openPrior = prior.setOpen;
  // The deleted views fall back to this one; the old prior view opens its drawer.
  const [legacyView, setLegacyView] = useUrlState("view", "");
  useEffect(() => {
    if (!legacyView) return;
    if (legacyView === "prior") openPrior(true);
    setLegacyView("");
  }, [legacyView, openPrior, setLegacyView]);
  usePageActions([
    { id: "stars-prior", label: "Stars: open the prior (fit and activate)", group: "Stars", run: prior.reveal },
    { id: "stars-query", label: "Query stars · MER + PHZ + Gaia…", group: "Stars", keywords: ["euclid", "gaia"],
      disabled: !api?.authenticated, run: () => { how.reveal(); void queryStars(); } },
    { id: "stars-fit", label: "Fit the stellar prior from cached data…", group: "Stars",
      disabled: !api?.q1_counts || !api?.color_sample.cached, run: () => { void fitStars(); } },
    { id: "stars-activate", label: "Activate the stellar prior…", group: "Stars", disabled: !candidate?.valid,
      run: () => { void activateStars(!!api?.calibration.is_active); } },
  ]);
  const distribution = api?.distribution ?? null;
  return (
    <Page className="rl-page syn-page">
      <Toolbar label="Star controls">
        {api && !api.calibration.is_active && (
          <ToolbarGroup label="Prior" hideLabel>
            <Badge size="sm" dot tone={candidate?.valid ? "warn" : "bad"}>
              {candidate?.valid ? "candidate not active" : candidate ? "needs a refit" : "not fitted"}
            </Badge>
          </ToolbarGroup>
        )}
        <ToolbarSpacer />
        <DrawerButton flag="prior" icon="activity" hint="Fit and activate the stellar prior">Prior</DrawerButton>
        <DrawerButton flag="how" icon="database" hint="The Q1 + Gaia query behind the prior">How this is produced</DrawerButton>
        {candidate && (
          <Menu label="Download the stellar calibration figure"
            trigger={<IconButton size="sm" icon="download" label="Download the stellar calibration figure" />}
            items={(["png", "pdf", "svg"] as const).map((f) => ({
              label: `Calibration figure · ${f.toUpperCase()}`,
              onSelect: () => downloadUrl(starFigureUrl(f), `euclidpolish_star_population_calibration.${f}`),
            }))} />
        )}
        <IconButton size="sm" icon="reset" label="Refresh" loading={resource.fetching} onClick={() => resource.reload()} />
      </Toolbar>
      <LoadState loading={resource.loading && !api} error={resource.error} onRetry={resource.reload}>
        {api && (
          <>
            {distribution ? <Density distribution={distribution} /> : (
              <EmptyState icon="activity" title="No stellar distribution yet"
                action={<Button size="sm" onClick={how.reveal}>How this is produced</Button>}>
                Query the stellar MER + PHZ and Gaia data, then fit the cached colours.
              </EmptyState>
            )}
            <PriorDrawer api={api} />
            <HowDrawer api={api} />
          </>
        )}
      </LoadState>
    </Page>
  );
}
