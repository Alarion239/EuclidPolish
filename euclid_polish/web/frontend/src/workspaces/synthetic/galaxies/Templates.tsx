/* Synthetic › Galaxies › templates: the TNG50-1 SKIRT atlas the generator
   draws galaxy morphologies from (SFR-rank matched), from the local property
   and atlas CSVs (read-only; nothing starts on a visit).

   - Templates: the caption (galaxies, those with a measured Rₑ), a badge
     only when something matters (no API token, properties older than 90
     days); the property explorer (any two properties, log axes, colour by a
     third; the SFR = 0 galaxies on a labelled floor strip instead of
     dropped; a click opens the `tng` inspector) with the histogram of its x
     property underneath as its marginal (median and 16–84% drawn, at 2
     significant figures); the galaxy table; the template thumbnails of the
     pulled tng_grid; and the radius manifest as ONE line (its fix,
     "Validate TNG radii on FASRC", is on Status).
   - TngSteps (the How-this-is-produced drawer): Refresh properties (TNG
     API), the grid / stack pulls and the download_tng_skirt,
     measure_tng_radii, tng_grid and tng_stack step cards, each in a closed
     section (a closed section renders nothing). */
import { useMemo } from "react";
import { Link } from "react-router-dom";
import { useJob } from "../../../api/jobs";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import Plot, { type Band as PlotBand, type Series } from "../../../charts/Plot";
import { C, viridis } from "../../../colors";
import { StepById } from "../../../fasrc";
import { formatRelative } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { linearTicks, logTicks } from "../../../ticks";
import {
  Badge, Button, Caption, DataTable, EmptyState, Field, Section, Select, Switch, Tooltip, type DataColumn,
} from "../../../ui";
import { StateDot } from "../common";
import { URLS, useTngProperties, type TngAuth, type TngRadius, type TngResults } from "../dataApi";
import { Freshness, JobStrip, LoadState, OFFLINE_HINT, startDataJob, useFasrcOnline } from "../dataCommon";
import {
  axisDomain, decodeTng, nearestPoint, propertyHistogram, scatterGroups, summaryStats, TNG_PROPS, tngPropMeta,
  tngValue, type ScatterGroup, type TngRow,
} from "../dataModel";
import { floorDomain, marginalGuides, radiusManifestLine, sig2, templatesCaption, zeroStrip } from "./templatesModel";

const STALE_PROPERTIES_S = 90 * 24 * 3600;
const PROP_OPTIONS = TNG_PROPS.map((p) => ({ value: p.key, label: `${p.label} (${p.unit})` }));
const axisLabel = (key: string) => { const m = tngPropMeta(key); return `${m.label} [${m.unit}]`; };
const ticksFor = (domain: [number, number], log: boolean) => (log ? logTicks(domain, { maxTicks: 7 }) : linearTicks(domain));

const COLUMNS: DataColumn<TngRow>[] = [
  { id: "id", header: "Subhalo", numeric: true, width: 80 },
  { id: "sfr", header: "SFR (M☉/yr)", numeric: true, width: 104, cell: (r) => (r.sfr == null ? "—" : sig2(r.sfr)) },
  { id: "mass_stars", header: "M★ (M☉)", numeric: true, width: 92, cell: (r) => (r.mass_stars == null ? "—" : sig2(r.mass_stars)) },
  { id: "m_halo", header: "M_halo (M☉)", numeric: true, width: 104, hidden: true, cell: (r) => (r.m_halo == null ? "—" : sig2(r.m_halo)) },
  { id: "reff", header: "r½ (kpc)", numeric: true, width: 80, hidden: true, cell: (r) => (r.reff == null ? "—" : sig2(r.reff)) },
  { id: "re_kpc", header: "Rₑ (kpc)", headerText: "Measured VIS Rₑ (kpc)", numeric: true, width: 84, cell: (r) => (r.re_kpc == null ? "—" : sig2(r.re_kpc)) },
  { id: "re_range", header: "Rₑ over views", width: 116, accessor: (r) => (r.re_kpc_max != null && r.re_kpc_min != null ? r.re_kpc_max - r.re_kpc_min : null),
    cell: (r) => (r.re_kpc_min != null && r.re_kpc_max != null ? `${sig2(r.re_kpc_min)}–${sig2(r.re_kpc_max)}` : "—") },
  { id: "n_orient", header: "Views", numeric: true, width: 64 },
  { id: "local", header: "Local frames", numeric: true, width: 96, hidden: true, cell: (r) => (r.local ? <Badge size="sm" tone="info">{r.local}</Badge> : "") },
];

/** The explorer and, under it, its x property's histogram (the marginal). */
function Explorer({ rows }: { rows: TngRow[] }) {
  const [x, setX] = useUrlState("x", "mass_stars");
  const [y, setY] = useUrlState("y", "sfr");
  const [c, setC] = useUrlState("c", "re_kpc");
  const [xlog, setXlog] = useUrlState("xlog", true);
  const [ylog, setYlog] = useUrlState("ylog", true);
  const { groups, hidden } = useMemo(
    () => scatterGroups(rows, x, y, c, { xlog, ylog, groups: 5, format: sig2 }), [rows, x, y, c, xlog, ylog]);
  const strip = useMemo(() => zeroStrip(rows, x, y, { xlog, ylog }), [rows, x, y, xlog, ylog]);
  const xs = groups.flatMap((g) => g.x), ys = groups.flatMap((g) => g.y);
  let xDom = axisDomain([...xs, ...(strip?.axis === "y" ? strip.values : [])], xlog);
  let yDom = axisDomain([...ys, ...(strip?.axis === "x" ? strip.values : [])], ylog);
  const bands: PlotBand[] = [];
  const series: Series[] = groups.map((g) => ({
    x: g.x, y: g.y, mode: "scatter" as const, color: g.t < 0 ? C.muted : viridis(g.t), name: g.label, key: g.key, width: 1,
  }));
  const clickable: ScatterGroup[] = [...groups];
  if (strip) {
    const f = floorDomain(strip.axis === "y" ? yDom : xDom, true);
    if (strip.axis === "y") yDom = f.domain; else xDom = f.domain;
    bands.push({ axis: strip.axis, from: f.strip[0], to: f.strip[1], color: C.muted, alpha: 0.1, label: strip.label });
    const floor = strip.values.map(() => f.at);
    const pts = strip.axis === "y" ? { x: strip.values, y: floor } : { x: floor, y: strip.values };
    series.push({ ...pts, mode: "scatter", color: C.muted, name: strip.label, key: "zero-floor", width: 1 });
    clickable.push({ key: "zero-floor", label: strip.label, t: -1, ...pts, ids: strip.ids });
  }
  const off = hidden - (strip?.count ?? 0);
  const span = (d: [number, number], log: boolean) => (log ? Math.log10(d[1]) - Math.log10(d[0]) : d[1] - d[0]);
  const values = useMemo(() => rows.map((r) => tngValue(r, x)), [rows, x]);
  const hist = useMemo(() => propertyHistogram(values, xlog, 30), [values, xlog]);
  const stats = useMemo(() => summaryStats(values.filter((v) => v != null && (!xlog || v > 0))), [values, xlog]);
  const top = Math.max(1, ...hist.counts);
  const meta = tngPropMeta(x);
  return (
    <div className="rl-stack">
      <div className="dt-axes" role="group" aria-label="Explorer axes">
        <Field label="X"><Select size="sm" value={x} onChange={setX} options={PROP_OPTIONS} /></Field>
        <Switch size="sm" checked={xlog} onChange={setXlog}>log</Switch>
        <Field label="Y"><Select size="sm" value={y} onChange={setY} options={PROP_OPTIONS} /></Field>
        <Switch size="sm" checked={ylog} onChange={setYlog}>log</Switch>
        <Field label="Colour"><Select size="sm" value={c} onChange={setC} options={PROP_OPTIONS} /></Field>
      </div>
      {xs.length || strip ? (
        <Plot xDomain={xDom} yDomain={yDom} xScale={xlog ? "log" : "linear"} yScale={ylog ? "log" : "linear"}
          xTicks={ticksFor(xDom, xlog)} yTicks={ticksFor(yDom, ylog)} xLabel={axisLabel(x)} yLabel={axisLabel(y)}
          series={series} bands={bands} aspect={0.5} legend="auto" exportName={`tng_${y}_vs_${x}`}
          xFormat={sig2} yFormat={sig2}
          aria-label={`${tngPropMeta(y).label} against ${meta.label}, coloured by ${tngPropMeta(c).label}`}
          onPlotClick={(pt) => {
            const id = nearestPoint(clickable, pt, { xlog, ylog, xSpan: span(xDom, xlog), ySpan: span(yDom, ylog) });
            if (id != null) openInspector({ kind: "tng", id: String(id) });
          }} />
      ) : <EmptyState compact icon="activity" title="No galaxy has both values" />}
      <Caption>
        {`Colour: ${tngPropMeta(c).label} quintiles · click a galaxy to inspect it`
          + (off > 0 ? ` · ${off} galaxies lack a value on these axes` : "")}
      </Caption>
      {hist.counts.length ? (
        <Plot xDomain={hist.edges.length ? [hist.edges[0], hist.edges[hist.edges.length - 1]] : [0, 1]} yDomain={[0, top * 1.12]}
          xScale={xlog ? "log" : "linear"} yTicks={linearTicks([0, top])} xLabel={axisLabel(x)} yLabel="galaxies" aspect={0.22}
          series={[{ x: hist.centers, y: hist.counts, mode: "histogram", color: C.mean, name: meta.label, fillAlpha: 0.35 }]}
          guides={marginalGuides(stats, meta.unit)} exportName={`tng_${x}_hist`} xFormat={sig2}
          aria-label={`Distribution of ${meta.label}`} />
      ) : null}
    </div>
  );
}

function RadiusManifest() {
  const radius = useResource<TngRadius>(URLS.tngRadii, [], { ttl: 60_000 });
  if (radius.loading && !radius.data) return null;
  const line = radiusManifestLine(radius.data);
  return (
    <p className="syn-line" role="status">
      <StateDot state={line.state === "ok" ? "ok" : line.state === "bad" ? "bad" : "unknown"} label={line.state} />
      <span>{line.text}</span>
      {line.fix && (
        <Button asChild size="sm" variant="ghost" iconRight="chevronRight">
          <Link to="/synthetic/status">Validate on Status</Link>
        </Button>
      )}
    </p>
  );
}

function TemplateGrid() {
  const results = useResource<TngResults>(URLS.tngResults, [], { ttl: 30_000 });
  const grid = results.data?.grid;
  if (!grid?.present) {
    return (
      <EmptyState compact icon="image" title="No template grid pulled yet">
        Submit tng_grid on FASRC and pull its image (How this is produced).
      </EmptyState>
    );
  }
  const src = `${URLS.tngGrid}?t=${grid.pulled_at ?? 0}`;
  return (
    <figure className="rl-fig">
      <a href={src} target="_blank" rel="noreferrer" className="dt-grid-link" title="Open the grid at full size">
        <img className="dt-grid-img" src={src} alt="Grid of TNG template galaxies from the last tng_grid run" loading="lazy" />
      </a>
      <Caption>{`The last tng_grid run's template thumbnails${grid.pulled_at ? ` · pulled ${formatRelative(grid.pulled_at)}` : ""}`}</Caption>
    </figure>
  );
}

export function Templates() {
  const props = useTngProperties();
  const rows = useMemo(() => decodeTng(props.data), [props.data]);
  const auth = useResource<TngAuth>(URLS.tngAuth, [], { ttl: 60_000 });
  const a = auth.data;
  const mtime = props.data?.files.properties.mtime ?? null;
  const oldProperties = mtime != null && Date.now() / 1000 - mtime > STALE_PROPERTIES_S;
  const caption = templatesCaption(props.data?.summary);
  return (
    <LoadState loading={props.loading && !props.data} error={props.error} onRetry={() => void props.reload()}>
      {!props.data?.present ? (
        <EmptyState icon="database" title="No TNG property cache on this machine">
          data/_tng_infographics/tng_properties.csv and tng_atlas_parameters.csv are missing: refresh the properties
          (How this is produced).
        </EmptyState>
      ) : (
        <div className="rl-stack">
          <div className="rl-row">
            {caption && <span className="rl-note">{caption}</span>}
            {/* Badges only when they matter: a missing token, properties past 90 days. */}
            {a?.connected && !a.present && (
              <Tooltip content="The IllustrisTNG API token lives in ~/.tng_api_key on FASRC">
                <Link to="/system/connections"><Badge size="sm" tone="warn" dot>No TNG API token</Badge></Link>
              </Tooltip>
            )}
            {oldProperties && <Freshness at={mtime} label="properties" stale={STALE_PROPERTIES_S} />}
          </div>
          <Explorer rows={rows} />
          <DataTable rows={rows} columns={COLUMNS} rowKey={(r) => String(r.id)} urlKey="tg" height={380} dense
            aria-label="TNG galaxies" exportName="tng_galaxies" inspect={(r) => ({ kind: "tng", id: String(r.id) })}
            empty="No galaxies" />
          <h3 className="syn-subtitle">Template thumbnails</h3>
          <TemplateGrid />
          <RadiusManifest />
        </div>
      )}
    </LoadState>
  );
}

/** The TNG part of the How-this-is-produced drawer. */
export function TngSteps() {
  const refreshProps = useJob("data:tng-properties");
  const pull = useJob("data:tng-pull");
  const results = useResource<TngResults>(URLS.tngResults, [], { ttl: 30_000 });
  const { online } = useFasrcOnline();
  const res = results.data;
  const refresh = () => void startDataJob(refreshProps, URLS.tngPropertiesRefresh, {}, {
    label: "Refresh the TNG properties",
    question: { title: "Refresh the galaxy properties from the TNG API?",
      message: "Lists the downloaded galaxies on FASRC and queries the TNG API for any missing from the local cache (~0.4 s each).",
      confirmLabel: "Refresh" },
  });
  const runPull = (kind: "grid" | "stack") => void startDataJob(pull, URLS.tngPull, { kind }, {
    label: "Pull the TNG results",
    question: { title: `Pull the latest ${kind} from FASRC?`,
      message: kind === "grid" ? "A small PNG." : "The stacked FITS is ~51 MB.", confirmLabel: "Pull" },
  });
  return (
    <div className="rl-stack">
      <h3 className="syn-subtitle">TNG50-1 templates (FASRC)</h3>
      <div className="rl-row">
        <Tooltip content={online ? "Query the TNG API for galaxies missing from the property cache (confirmed)" : OFFLINE_HINT}>
          <span><Button size="sm" icon="reset" loading={refreshProps.busy} disabled={!online} onClick={refresh}>Refresh properties</Button></span>
        </Tooltip>
        <Tooltip content={online ? "Pull the latest tng_grid image (confirmed)" : OFFLINE_HINT}>
          <span><Button size="sm" icon="download" disabled={!online} loading={pull.busy} onClick={() => runPull("grid")}>Pull grid</Button></span>
        </Tooltip>
        <Tooltip content={online ? "Pull the latest tng_stack FITS, ~51 MB (confirmed)" : OFFLINE_HINT}>
          <span><Button size="sm" icon="download" disabled={!online} loading={pull.busy} onClick={() => runPull("stack")}>Pull stack</Button></span>
        </Tooltip>
        {res?.stack.present && <Button size="sm" variant="ghost" icon="download" href={URLS.tngStack} download>Stacked FITS</Button>}
      </div>
      <JobStrip job={refreshProps} />
      <JobStrip job={pull} />
      <Section title="Atlas download" sub="download_tng_skirt · ~1,150 galaxies × 5 views × 4 bands" collapsible defaultOpen={false}>
        <StepById stepId="download_tng_skirt" embedded />
      </Section>
      <Section title="Measure the radii" sub="measure_tng_radii" collapsible defaultOpen={false}>
        <StepById stepId="measure_tng_radii" embedded />
      </Section>
      <Section title="Template grid and stack" sub="tng_grid · tng_stack" collapsible defaultOpen={false}>
        <div className="syn-steps">
          <StepById stepId="tng_grid" />
          <StepById stepId="tng_stack" />
        </div>
      </Section>
    </div>
  );
}
