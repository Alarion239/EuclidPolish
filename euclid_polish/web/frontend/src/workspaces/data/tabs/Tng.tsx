/* Data › TNG (spec §8.4): the TNG50-1 SKIRT atlas.
 *
 * Toolbar: the API token on FASRC (managed in Settings › Connections), the
 * radius-manifest validation, and the explicit remote actions — refresh the
 * galaxy properties from the TNG API, pull the grid / stack job results (all
 * background jobs). Body: the interactive property explorer over the local
 * calibration CSVs (any two properties, log axes, colour by a third, click a
 * galaxy → the `tng` inspector), the distribution of one property, the galaxy
 * table, then the measured radii (validated asynchronously, cached) and the
 * atlas / grid / stack FASRC steps with their pulled results. Axes, colour,
 * histogram property and open sections are in the URL. */
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useJob, useJobsStore } from "../../../api/jobs";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import Plot from "../../../charts/Plot";
import { C, viridis } from "../../../colors";
import { StepById } from "../../../fasrc";
import { formatCount, formatNumber } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { linearTicks, logTicks } from "../../../ticks";
import {
  Badge, Button, Callout, Card, CardBody, CardHead, DataTable, EmptyState, Field, JobProgress, Page, Section, Select,
  Switch, Tooltip, type DataColumn,
} from "../../../ui";
import { URLS, useTngProperties, type TngAuth, type TngRadius, type TngResults } from "../api";
import { DataBar, Freshness, JobStrip, LoadState, OFFLINE_HINT, Spacer, startDataJob, useFasrcOnline } from "../common";
import {
  axisDomain, decodeTng, nearestPoint, propertyHistogram, scatterGroups, summaryStats, TNG_PROPS, tngPropMeta,
  tngValue, type TngRow,
} from "../model";
import "../register";
import "../data.css";

/** Shared with Realism (its readiness checklist starts the same validation). */
const RADII_JOB_KEY = "realism:tng-radii";

const fmtSci = (v: number | null | undefined) => (v == null ? "—" : Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-2 && v !== 0)
  ? v.toExponential(2) : formatNumber(v, { digits: 3 }));
const PROP_OPTIONS = TNG_PROPS.map((p) => ({ value: p.key, label: `${p.label} (${p.unit})` }));
const axisLabel = (key: string) => { const m = tngPropMeta(key); return `${m.label} [${m.unit}]`; };
const ticksFor = (domain: [number, number], log: boolean) => (log ? logTicks(domain, { maxTicks: 7 }) : linearTicks(domain));

const COLUMNS: DataColumn<TngRow>[] = [
  { id: "id", header: "Subhalo", numeric: true, width: 80 },
  { id: "sfr", header: "SFR M☉/yr", numeric: true, width: 96, cell: (r) => fmtSci(r.sfr) },
  { id: "mass_stars", header: "M★ M☉", numeric: true, width: 92, cell: (r) => fmtSci(r.mass_stars) },
  { id: "m_halo", header: "M_halo M☉", numeric: true, width: 96, cell: (r) => fmtSci(r.m_halo) },
  { id: "reff", header: "r½ kpc", numeric: true, width: 72, cell: (r) => formatNumber(r.reff, { digits: 2 }) },
  { id: "re_kpc", header: "Rₑ kpc", numeric: true, width: 72, cell: (r) => formatNumber(r.re_kpc, { digits: 2 }) },
  { id: "re_range", header: "Rₑ range", width: 96, accessor: (r) => (r.re_kpc_max != null && r.re_kpc_min != null ? r.re_kpc_max - r.re_kpc_min : null),
    cell: (r) => (r.re_kpc_min != null ? `${formatNumber(r.re_kpc_min, { digits: 2 })}–${formatNumber(r.re_kpc_max, { digits: 2 })}` : "—") },
  { id: "n_orient", header: "Views", numeric: true, width: 60 },
  { id: "local", header: "Local", numeric: true, width: 60, cell: (r) => (r.local ? <Badge size="sm" tone="info">{r.local}</Badge> : "") },
];

function Explorer({ rows }: { rows: TngRow[] }) {
  const [x, setX] = useUrlState("x", "mass_stars");
  const [y, setY] = useUrlState("y", "sfr");
  const [c, setC] = useUrlState("c", "re_kpc");
  const [xlog, setXlog] = useUrlState("xlog", true);
  const [ylog, setYlog] = useUrlState("ylog", true);
  const { groups, hidden } = useMemo(
    () => scatterGroups(rows, x, y, c, { xlog, ylog, groups: 5, format: (v) => fmtSci(v) }), [rows, x, y, c, xlog, ylog]);
  const xs = groups.flatMap((g) => g.x), ys = groups.flatMap((g) => g.y);
  const xDom = axisDomain(xs, xlog), yDom = axisDomain(ys, ylog);
  const span = (d: [number, number], log: boolean) => (log ? Math.log10(d[1]) - Math.log10(d[0]) : d[1] - d[0]);
  const series = groups.map((g) => ({
    x: g.x, y: g.y, mode: "scatter" as const, color: g.t < 0 ? C.muted : viridis(g.t), name: g.label, key: g.key, width: 1,
  }));
  return (
    <Card>
      <CardHead title="Property explorer" sub={`${formatCount(xs.length)} galaxies${hidden ? ` · ${hidden} not on these axes` : ""}`} />
      <CardBody>
        <div className="dt-axes">
          <Field label="X"><Select size="sm" value={x} onChange={setX} options={PROP_OPTIONS} /></Field>
          <Switch size="sm" checked={xlog} onChange={setXlog}>log</Switch>
          <Field label="Y"><Select size="sm" value={y} onChange={setY} options={PROP_OPTIONS} /></Field>
          <Switch size="sm" checked={ylog} onChange={setYlog}>log</Switch>
          <Field label="Colour"><Select size="sm" value={c} onChange={setC} options={PROP_OPTIONS} /></Field>
        </div>
        {xs.length ? (
          <Plot xDomain={xDom} yDomain={yDom} xScale={xlog ? "log" : "linear"} yScale={ylog ? "log" : "linear"}
            xTicks={ticksFor(xDom, xlog)} yTicks={ticksFor(yDom, ylog)} xLabel={axisLabel(x)} yLabel={axisLabel(y)}
            series={series} height={320} legend="auto" exportName={`tng_${y}_vs_${x}`}
            xFormat={(v) => fmtSci(v)} yFormat={(v) => fmtSci(v)}
            aria-label={`${tngPropMeta(y).label} against ${tngPropMeta(x).label}, coloured by ${tngPropMeta(c).label}`}
            onPlotClick={(pt) => {
              const id = nearestPoint(groups, pt, { xlog, ylog, xSpan: span(xDom, xlog), ySpan: span(yDom, ylog) });
              if (id != null) openInspector({ kind: "tng", id: String(id) });
            }} />
        ) : <EmptyState compact icon="activity" title="No galaxy has both values" />}
        <p className="dt-note">Legend = {tngPropMeta(c).label} quintiles. Click a point to inspect the galaxy.</p>
      </CardBody>
    </Card>
  );
}

function Distribution({ rows }: { rows: TngRow[] }) {
  const [h, setH] = useUrlState("h", "re_kpc");
  const [hlog, setHlog] = useUrlState("hlog", true);
  const values = useMemo(() => rows.map((r) => tngValue(r, h)), [rows, h]);
  const hist = useMemo(() => propertyHistogram(values, hlog, 30), [values, hlog]);
  const stats = useMemo(() => summaryStats(values.filter((v) => v != null && (!hlog || v > 0))), [values, hlog]);
  const top = Math.max(1, ...hist.counts);
  const dom: [number, number] = hist.edges.length ? [hist.edges[0], hist.edges[hist.edges.length - 1]] : [0, 1];
  return (
    <Card>
      <CardHead title="Distribution" sub={stats ? `median ${fmtSci(stats.median)} · 16–84 % ${fmtSci(stats.p16)}–${fmtSci(stats.p84)} · n ${stats.n}` : undefined} />
      <CardBody>
        <div className="dt-axes">
          <Field label="Property"><Select size="sm" value={h} onChange={setH} options={PROP_OPTIONS} /></Field>
          <Switch size="sm" checked={hlog} onChange={setHlog}>log</Switch>
        </div>
        {hist.counts.length ? (
          <Plot xDomain={dom} yDomain={[0, top * 1.08]} xScale={hlog ? "log" : "linear"} xTicks={ticksFor(dom, hlog)}
            yTicks={linearTicks([0, top])} xLabel={axisLabel(h)} yLabel="galaxies" height={260}
            series={[{ x: hist.centers, y: hist.counts, mode: "histogram", color: C.mean, name: tngPropMeta(h).label, fillAlpha: 0.35 }]}
            exportName={`tng_${h}_hist`} xFormat={(v) => fmtSci(v)} aria-label={`Distribution of ${tngPropMeta(h).label}`} />
        ) : <EmptyState compact icon="activity" title="No values" />}
      </CardBody>
    </Card>
  );
}

function RadiiCard() {
  const [poll, setPoll] = useState(false);
  const radius = useResource<TngRadius>(URLS.tngRadii, [], { poll: poll ? 3_000 : undefined });
  const r = radius.data;
  const refresh = useJob(RADII_JOB_KEY);
  const runningId = r?.refresh_job ?? null;
  const running = useJobsStore((st) => (runningId ? st.jobs[runningId] ?? null : null));
  const [steps, setSteps] = useUrlState("rsteps", false);
  const reload = radius.reload;
  // The GET answers from the cache and never starts anything: re-validating
  // runs a script on FASRC, so it only happens from the button (confirmed).
  const validate = () => void startDataJob(refresh, URLS.tngRadiiRefresh, {}, {
    label: "TNG radius validation",
    question: {
      title: "Validate the TNG radius manifest on FASRC?",
      message: "Runs scripts/validate_tng_radius_manifest.py on the cluster (up to a few minutes) and caches the result.",
      confirmLabel: "Validate",
    },
    onDone: () => void reload(),
  });
  const canValidate = !!r?.connected && !runningId && !refresh.busy;
  useEffect(() => { setPoll(!!runningId && !refresh.busy); }, [runningId, refresh.busy]);
  return (
    <Card>
      <CardHead title="Measured effective radii" sub="required for COSMOS-conditioned population generation"
        right={radius.loading && !r ? <Badge size="sm">…</Badge>
          : r?.valid ? <Badge size="sm" tone="good">valid</Badge> : <Badge size="sm" tone="warn">recalculation required</Badge>} />
      <CardBody>
        {r && (
          <p className="mono dt-note">
            {r.valid_count ?? 0}/{r.expected_count ?? 0} frames valid{r.failed_count ? ` · ${r.failed_count} failed` : ""}
            {r.checked_at ? <> · <Freshness at={r.checked_at} label="checked" stale={24 * 3600} /></> : null}
          </p>
        )}
        {!r?.valid && r?.reasons?.length ? <Callout tone="bad" title="Not valid"><span className="dt-pre">{r.reasons.join("\n")}</span></Callout> : null}
        {r?.stale && !refresh.job && !runningId && (
          <p className="dt-note">
            {r.connected ? "This validation is out of date; validate it on FASRC to re-check it."
              : "This validation is out of date; connect to FASRC to re-check it."}
          </p>
        )}
        <JobProgress job={refresh.job ?? running} error={refresh.error} />
        <div className="dt-chips">
          <Tooltip content={r?.connected ? (runningId || refresh.busy ? "A validation is already running" : "Re-check the measured radii on FASRC") : OFFLINE_HINT}>
            <span><Button size="sm" icon="check" disabled={!canValidate} loading={refresh.busy} onClick={validate}>Validate on FASRC</Button></span>
          </Tooltip>
          <Button size="sm" variant="ghost" icon="reset" onClick={() => void reload()}>Refresh status</Button>
          <Button size="sm" variant="ghost" onClick={() => setSteps(!steps)}>{steps ? "Hide" : "Measure"} (FASRC step)</Button>
        </div>
        {steps && <StepById stepId="measure_tng_radii" embedded />}
      </CardBody>
    </Card>
  );
}

function ResultsCard() {
  const results = useResource<TngResults>(URLS.tngResults, [], { ttl: 30_000 });
  const pull = useJob("data:tng-pull");
  const { online } = useFasrcOnline();
  const [open, setOpen] = useUrlState("grid", false);
  const res = results.data;
  const run = (kind: "grid" | "stack" | "all") => void startDataJob(pull, URLS.tngPull, { kind }, {
    label: "Pull the TNG results",
    question: { title: `Pull the latest ${kind === "all" ? "grid and stack" : kind} from FASRC?`,
      message: kind === "grid" ? "A small PNG." : "The stacked FITS is ~51 MB.", confirmLabel: "Pull" },
  });
  return (
    <Section title="Grid and stack (FASRC jobs)" collapsible open={open} onOpenChange={setOpen}
      sub={res?.grid.present ? <Freshness at={res.grid.pulled_at} label="grid pulled" /> : undefined}>
      <div className="dt-steps">
        <JobStrip job={pull} />
        <div className="dt-chips">
          <Tooltip content={online ? "Pull the latest tng_grid image" : OFFLINE_HINT}>
            <span><Button size="sm" icon="download" disabled={!online} loading={pull.busy} onClick={() => run("grid")}>Pull grid</Button></span>
          </Tooltip>
          <Tooltip content={online ? "Pull the latest tng_stack FITS (~51 MB)" : OFFLINE_HINT}>
            <span><Button size="sm" icon="download" disabled={!online} loading={pull.busy} onClick={() => run("stack")}>Pull stack</Button></span>
          </Tooltip>
          {res?.stack.present && <Button size="sm" variant="ghost" icon="download" href={URLS.tngStack} download>Stacked FITS</Button>}
        </div>
        {res?.grid.present
          ? <img className="dt-grid-img" src={`${URLS.tngGrid}?t=${res.grid.pulled_at ?? 0}`} alt="5 × 5 TNG galaxy grid" loading="lazy" />
          : <EmptyState compact icon="image" title="No grid pulled yet">Submit the grid job, then pull its result.</EmptyState>}
        <StepById stepId="tng_grid" />
        <StepById stepId="tng_stack" />
      </div>
    </Section>
  );
}

export default function Tng() {
  const props = useTngProperties();
  const rows = useMemo(() => decodeTng(props.data), [props.data]);
  const auth = useResource<TngAuth>(URLS.tngAuth, [], { ttl: 60_000 });
  const refreshProps = useJob("data:tng-properties");
  const { online } = useFasrcOnline();
  const [atlas, setAtlas] = useUrlState("atlas", false);
  const summary = props.data?.summary;
  const refresh = () => void startDataJob(refreshProps, URLS.tngPropertiesRefresh, {}, {
    label: "Refresh the TNG properties",
    question: { title: "Refresh the galaxy properties from the TNG API?",
      message: "Lists the downloaded galaxies on FASRC and queries the TNG API for any missing from the local cache (~0.4 s each).",
      confirmLabel: "Refresh" },
  });
  usePageActions([
    { id: "tng-refresh-props", label: "Refresh the TNG galaxy properties (TNG API)…", group: "TNG", disabled: !online, run: refresh },
    { id: "tng-atlas", label: "Download the TNG50 SKIRT atlas (FASRC)…", group: "TNG", run: () => setAtlas(true) },
  ]);
  const a = auth.data;
  return (
    <Page className="dt-page">
      <DataBar label="TNG">
        <Tooltip content="The IllustrisTNG API token lives in ~/.tng_api_key on FASRC (Settings › Connections)">
          <Link to="/settings/connections" className="dt-fresh">
            <Badge size="sm" tone={!a ? "neutral" : !a.connected ? "neutral" : a.present ? "good" : "warn"} dot>
              {!a ? "token …" : !a.connected ? "token: FASRC offline" : a.present ? "token saved" : "no token"}
            </Badge>
          </Link>
        </Tooltip>
        {summary && (
          <Badge size="sm">{formatCount(summary.n)} galaxies · {formatCount(summary.n_in_atlas)} measured · {summary.n_local} local</Badge>
        )}
        {props.data?.files.properties.present && <Freshness at={props.data.files.properties.mtime} label="properties" stale={90 * 24 * 3600} />}
        <Spacer />
        <Tooltip content={online ? "Query the TNG API for galaxies missing from the property cache" : OFFLINE_HINT}>
          <span><Button size="sm" icon="reset" loading={refreshProps.busy} disabled={!online} onClick={refresh}>Refresh properties</Button></span>
        </Tooltip>
      </DataBar>
      <JobStrip job={refreshProps} />
      <LoadState loading={props.loading && !props.data} error={props.error} onRetry={() => void props.reload()}>
        {!props.data?.present ? (
          <EmptyState icon="database" title="No TNG property cache on this machine">
            data/_tng_infographics/tng_properties.csv and tng_atlas_parameters.csv are missing — refresh the properties
            {online ? "" : ` (${OFFLINE_HINT})`}.
          </EmptyState>
        ) : (
          <>
            <div className="dt-two">
              <Explorer rows={rows} />
              <Distribution rows={rows} />
            </div>
            <Card>
              <CardBody>
                <DataTable rows={rows} columns={COLUMNS} rowKey={(r) => String(r.id)} urlKey="tg" height={420} dense
                  aria-label="TNG galaxies" exportName="tng_galaxies"
                  inspect={(r) => ({ kind: "tng", id: String(r.id) })}
                  empty="No galaxies" />
              </CardBody>
            </Card>
          </>
        )}
      </LoadState>
      <div className="dt-two">
        <RadiiCard />
        <Card>
          <CardHead title="Atlas download" sub="TNG50-1 SKIRT: ~1,150 galaxies × 5 views × 4 bands" />
          <CardBody>
            <Button size="sm" variant="ghost" onClick={() => setAtlas(!atlas)}>{atlas ? "Hide" : "Show"} the download step</Button>
            {atlas && <StepById stepId="download_tng_skirt" embedded />}
          </CardBody>
        </Card>
      </div>
      <ResultsCard />
    </Page>
  );
}
