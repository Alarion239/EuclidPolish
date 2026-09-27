/* One experiment (GET /api/experiments/<id>, spec §7.3), image first: the
 * head (state, label, log to tracking, re-run, open the tiles) and the tile
 * picker, then the comparison viewer (LR + one tier per model + JWST;
 * residuals from the viewer bar), then the per-band metrics of every model —
 * pooled over the tiles or of the picked tile — as a band chart and a
 * DataTable (CSV), the gate core weights and the run details. */
import { useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import Plot from "../../../charts/Plot";
import { categorical } from "../../../colors";
import { openInspector } from "../../../app/inspector";
import { formatDateTime, formatDuration } from "../../../format";
import { linearTicks } from "../../../ticks";
import { useSelection } from "../../../state/selection";
import {
  Badge, Callout, DataTable, DefList, Section, Select, Tooltip, type DataColumn, type MenuItem,
} from "../../../ui";
import { MoreMenu } from "../atlas/inspectors/common";
import { ImageViewer } from "../../../viewer";
import { runModels } from "./actions";
import { BANDS, splitRef, type ExperimentRecord } from "./api";
import { CoreWeights, StateBadge } from "./common";
import { useFollowViewer } from "./follow";
import { FitBox } from "./FitBox";
import { LogToTrackingButton } from "./LogToTracking";
import {
  bandLabel, bandSeries, CHART_METRICS, experimentMarkdown, formatMetric, METRIC_BY_KEY, METRICS,
  metricHeader, metricRows, recordSpecs, seriesDomain, specShort, type MetricKey, type MetricRow,
} from "./model";

const METRIC_COLUMNS: DataColumn<MetricRow>[] = [
  { id: "spec", header: "Model", cell: (r) => <Tooltip content={r.label}><code className="mono" tabIndex={0}>{specShort(r.spec)}</code></Tooltip>,
    filterText: (r) => `${r.spec} ${r.label}`, csv: (r) => r.spec },
  { id: "band", header: "Band", cell: (r) => bandLabel(r.band), sortFn: (a, b) => BANDS.indexOf(a.band as never) - BANDS.indexOf(b.band as never) },
  ...METRICS.map((m): DataColumn<MetricRow> => ({
    id: m.key, header: <Tooltip content={m.hint}><span tabIndex={0} className="res-case">{metricHeader(m.short)}</span></Tooltip>, headerText: m.label,
    numeric: true, cell: (r) => formatMetric(m.key, r[m.key]),
    accessor: (r) => (typeof r[m.key] === "number" ? r[m.key] : null),
    hidden: m.key === "min_R" || m.key === "n_artifacts",
  })),
];

const BAND_TICKS = BANDS.map((b, i) => ({ v: i, label: bandLabel(b) }));

export function ExperimentMetrics({ record, scope, metric, onMetric, urlKey }: {
  record: ExperimentRecord; scope: string; metric: MetricKey; onMetric: (m: MetricKey) => void; urlKey?: string;
}) {
  const rows = useMemo(() => metricRows(record, scope), [record, scope]);
  const series = useMemo(() => bandSeries(record, scope, metric), [record, scope, metric]);
  const domain = seriesDomain(series, metric);
  const yTicks = useMemo(() => linearTicks(domain, { count: 4 }), [domain[0], domain[1]]);   // eslint-disable-line react-hooks/exhaustive-deps
  const def = METRIC_BY_KEY[metric];
  const guides = def?.better === "one" ? [{ axis: "y" as const, v: 1, dash: [3, 3] }] : [];
  if (!rows.length) {
    return <p className="muted res-note">No metrics {scope === "pooled" ? "yet" : "for this tile"}{record.status === "running" ? " — the experiment is still running." : "."}</p>;
  }
  return (
    <div className="res-metrics">
      <div className="res-metrics__chart">
        <div className="res-bar res-bar--inline">
          <label className="res-inline">
            <span className="res-inline__label">Chart</span>
            <Select size="sm" value={metric} onChange={(v) => onMetric(v as MetricKey)} aria-label="Chart metric"
              options={CHART_METRICS.map((k) => ({ value: k, label: METRIC_BY_KEY[k].label }))} />
          </label>
          <span className="muted res-note">{def?.hint}</span>
        </div>
        <Plot xDomain={[-0.5, BANDS.length - 0.5]} yDomain={domain} xTicks={BAND_TICKS} yTicks={yTicks}
          yLabel={def?.label} height={240} zoomAxes="y" guides={guides}
          series={series.map((s, i) => ({
            x: s.x, y: s.y, color: categorical(i), mode: "line", dots: true, width: 1.2,
            name: specShort(s.spec), key: s.spec,
          }))}
          legend="auto" legendToggle exportName={`experiment-${record.id}-${metric}`}
          yFormat={(v) => formatMetric(metric, v)} aria-label={`${def?.label} per band and model`} />
      </div>
      <DataTable rows={rows} columns={METRIC_COLUMNS} rowKey={(r) => r.key} aria-label="Metrics per model and band"
        dense height={rows.length > 16 ? 420 : "auto"} exportName={`experiment-${record.id}-${scope.replace("/", "_")}`}
        urlKey={urlKey} filterPlaceholder="Filter: band:VIS  hole_pct>5  spec:gate" />
    </div>
  );
}

/** The tiers a URL carried for the comparison viewer (`v.<key>.t`), read once
 *  at mount: a shared link's tier choice wins over the page's default. */
function urlTiers(search: string, key: string): string[] | null {
  const raw = new URLSearchParams(search).get(`v.${key}.t`);
  const list = raw ? raw.split(",").map((t) => t.trim()).filter(Boolean) : [];
  return list.length ? list : null;
}

export function ExperimentDetail({ record, scope, onScope, metric, onMetric, viewer = true }: {
  record: ExperimentRecord; scope: string; onScope: (scope: string) => void;
  metric: MetricKey; onMetric: (m: MetricKey) => void; viewer?: boolean;
}) {
  const navigate = useNavigate();
  const location = useLocation();
  const specs = recordSpecs(record);
  const tiles = record.tiles ?? [];
  const viewTile = scope !== "pooled" && tiles.includes(scope) ? scope : tiles[0];
  const [source, tileId] = viewTile ? splitRef(viewTile) : ["", ""];
  const params = useMemo(() => ({ source, models: specs.join(",") }), [source, specs]);
  const tiers = useMemo(
    () => ["lr", ...specs.slice(0, 4).map((s) => `m:${s}`), ...(source === "nexus" || source === "pair" ? ["jwst"] : [])],
    [specs, source],
  );
  const [mountTiers] = useState(() => urlTiers(location.search, "exp"));
  // nav off: the collection walks every tile of the source; the Scope select
  // is this experiment's navigation (the follower keeps the images on it).
  const follow = useFollowViewer(tileId, mountTiers ?? tiers);
  const errors = Object.entries(record.errors ?? {});
  const skipped = Object.entries(record.skipped ?? {});
  const gateSpecs = specs.filter((s) => s === "production" || s.startsWith("gate:"));
  const tileResults = scope !== "pooled" ? record.results?.[scope] ?? {} : {};
  const counts = record.counts ?? {};
  const openTiles = () => {
    useSelection.getState().select("tile", tiles);
    navigate("/sky/results");
  };
  const pooledLabel = tiles.length === 1 ? "Pooled metrics (1 tile)" : `Pooled metrics (all ${tiles.length} tiles)`;
  const more: MenuItem[] = [
    { label: "Re-run this experiment…", disabled: record.status === "running" || !specs.length,
      onSelect: () => { void runModels(tiles, specs, { label: record.label ? `${record.label} (re-run)` : undefined }); } },
    { label: "Select its tiles in Real results", onSelect: openTiles },
    ...(scope !== "pooled" ? [{ label: "Open the tile card", onSelect: () => openInspector({ kind: "tile", id: scope }) }] : []),
  ];
  // One row: what (state, label, which tile) then the actions; it wraps in a narrow pane.
  return (
    <div className="res-exp">
      <div className="res-exp__head">
        <StateBadge state={record.status} />
        <strong className="res-exp__title" title={record.id}>{record.label || record.id}</strong>
        <Select size="sm" value={scope} onChange={onScope} aria-label="Metric scope" className="res-exp__scope"
          options={[{ value: "pooled", label: pooledLabel }, ...tiles.map((t) => ({ value: t, label: t }))]} />
        <span className="res-exp__actions">
          <LogToTrackingButton note={() => experimentMarkdown(record)} disabled={record.status === "running"} />
          <MoreMenu items={more} label="More experiment actions" />
        </span>
      </div>
      {viewer && viewTile && tileId && (
        <FitBox className="res-exp__viewer" label={`Comparison on ${viewTile}`}>
          <ImageViewer key={`${record.id}:${source}:${specs.join(",")}`} collection="real" params={params} initialId={tileId}
            nav={false} onReady={follow.onReady} onState={follow.onState} tiers={tiers}
            id={`experiment-${record.id}`} urlKey="exp" toolbar="full" />
        </FitBox>
      )}
      {!!skipped.length && (
        <Callout tone="warn" title={`${skipped.length} model${skipped.length === 1 ? "" : "s"} skipped`}>
          {skipped.map(([s, why]) => <div key={s}><code className="mono">{s}</code>: {why}</div>)}
        </Callout>
      )}
      {!!errors.length && (
        <Callout tone="bad" title={`${errors.length} error${errors.length === 1 ? "" : "s"}`}>
          {errors.slice(0, 5).map(([k, v]) => <div key={k} className="res-note"><code className="mono">{k}</code>: {v}</div>)}
          {errors.length > 5 && <div className="muted">… {errors.length - 5} more</div>}
        </Callout>
      )}
      <Section title="Metrics" sub={scope === "pooled" ? "pooled over the tiles" : `of ${scope}`} collapsible defaultOpen>
        <ExperimentMetrics record={record} scope={scope} metric={metric} onMetric={onMetric} urlKey={viewer ? "em" : undefined} />
      </Section>
      {scope !== "pooled" && gateSpecs.some((s) => tileResults[s]?.metrics?.gate_core_weights) && (
        <Section title="Gate core weights" sub="top members over the brightest 1 % of pixels" collapsible defaultOpen>
          {gateSpecs.map((s) => tileResults[s]?.metrics?.gate_core_weights ? (
            <div key={s} className="res-weights__block">
              <code className="mono">{specShort(s)}</code>
              <CoreWeights weights={tileResults[s]?.metrics?.gate_core_weights} />
            </div>
          ) : null)}
        </Section>
      )}
      <Section title="Run details" collapsible defaultOpen={!viewer}>
        <DefList dense items={[
          ["id", <code className="mono">{record.id}</code>],
          ["created", record.created ? formatDateTime(record.created) : "—"],
          record.duration_s != null ? ["duration", formatDuration(record.duration_s)] : null,
          ["tiles", tiles.length],
          ["models", <span className="res-chips">{specs.map((s) => <Badge key={s} size="sm">{specShort(s)}</Badge>)}</span>],
          ["members", `${counts.members_computed ?? 0} computed, ${counts.members_reused ?? 0} reused${counts.members_not_cached ? `, ${counts.members_not_cached} not cached` : ""}`],
          ["outputs", `${counts.outputs_computed ?? 0} computed, ${counts.outputs_reused ?? 0} reused`],
        ]} />
      </Section>
    </div>
  );
}
