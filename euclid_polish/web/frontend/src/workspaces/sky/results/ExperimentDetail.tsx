/* One experiment (GET /api/experiments/<id>, spec §7.3): comparison viewer
 * (LR + one tier per model + JWST; residuals from the viewer toolbar), the
 * per-band metrics of every model — pooled over the tiles or of one tile —
 * as a DataTable (CSV) and a band chart, the gate core weights, and the
 * actions (log to tracking, re-run, open the tiles). */
import { useCallback, useEffect, useMemo, useRef } from "react";
import { useNavigate } from "react-router-dom";
import Plot from "../../../charts/Plot";
import { categorical } from "../../../colors";
import { openInspector } from "../../../app/inspector";
import { formatDateTime, formatDuration } from "../../../format";
import { useSelection } from "../../../state/selection";
import {
  Badge, Button, Callout, DataTable, DefList, Field, Section, Select, Tooltip, type DataColumn,
} from "../../../ui";
import { ImageViewer, type ViewerApi } from "../../../viewer";
import { runModels } from "./actions";
import { BANDS, splitRef, type ExperimentRecord } from "./api";
import { CoreWeights, StateBadge } from "./common";
import { LogToTrackingButton } from "./LogToTracking";
import {
  bandLabel, bandSeries, CHART_METRICS, experimentMarkdown, formatMetric, METRIC_BY_KEY, METRICS,
  metricRows, recordSpecs, seriesDomain, specShort, type MetricKey, type MetricRow,
} from "./model";

const METRIC_COLUMNS: DataColumn<MetricRow>[] = [
  { id: "spec", header: "Model", cell: (r) => <Tooltip content={r.label}><code className="mono" tabIndex={0}>{specShort(r.spec)}</code></Tooltip>,
    filterText: (r) => `${r.spec} ${r.label}`, csv: (r) => r.spec },
  { id: "band", header: "Band", cell: (r) => bandLabel(r.band), sortFn: (a, b) => BANDS.indexOf(a.band as never) - BANDS.indexOf(b.band as never) },
  ...METRICS.map((m): DataColumn<MetricRow> => ({
    id: m.key, header: <Tooltip content={m.hint}><span tabIndex={0}>{m.short}</span></Tooltip>, headerText: m.label,
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
  const def = METRIC_BY_KEY[metric];
  const guides = def?.better === "one" ? [{ axis: "y" as const, v: 1, dash: [3, 3] }] : [];
  if (!rows.length) {
    return <p className="muted res-note">No metrics {scope === "pooled" ? "yet" : "for this tile"}{record.status === "running" ? " — the experiment is still running." : "."}</p>;
  }
  return (
    <div className="res-metrics">
      <div className="res-metrics__chart">
        <div className="res-bar res-bar--inline">
          <Field label="Chart" inline>
            <Select size="sm" value={metric} onChange={(v) => onMetric(v as MetricKey)}
              options={CHART_METRICS.map((k) => ({ value: k, label: METRIC_BY_KEY[k].label }))} />
          </Field>
          <span className="muted res-note">{def?.hint}</span>
        </div>
        <Plot xDomain={[-0.5, BANDS.length - 0.5]} yDomain={domain} xTicks={BAND_TICKS}
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

/** Keeps a nav-less viewer on the object `want` (the Scope select drives the
 *  comparison): re-armed whenever `want` changes or the viewer mounts, it
 *  moves the viewer once its meta is in — so neither the mount-time id nor a
 *  URL-carried `v.exp.id` can leave the images on another tile than the
 *  metrics beside them. Returns the viewer's onReady / onState handlers. */
export function useFollowId(want: string) {
  const api = useRef<ViewerApi | null>(null);
  const wantRef = useRef(want);
  wantRef.current = want;
  const armed = useRef(true);
  const busy = useRef(false);
  const ensure = useCallback(() => {
    const a = api.current;
    const w = wantRef.current;
    if (!a || !w || !armed.current || busy.current) return;
    const cur = a.getState().id;
    if (cur == null) return;                          // meta not in yet: onState re-tries
    if (cur === w) { armed.current = false; return; }
    busy.current = true;
    void a.goToId(w).then(
      () => {
        busy.current = false;
        if (wantRef.current !== w) ensure();   // the scope moved on meanwhile
        else armed.current = false;            // reached, or an unknown id: stop either way
      },
      () => { busy.current = false; armed.current = false; },
    );
  }, []);
  useEffect(() => { armed.current = true; ensure(); }, [want, ensure]);
  const onReady = useCallback((a: ViewerApi | null) => { api.current = a; armed.current = true; busy.current = false; ensure(); }, [ensure]);
  const onState = useCallback(() => ensure(), [ensure]);
  return { onReady, onState };
}

export function ExperimentDetail({ record, scope, onScope, metric, onMetric, viewer = true }: {
  record: ExperimentRecord; scope: string; onScope: (scope: string) => void;
  metric: MetricKey; onMetric: (m: MetricKey) => void; viewer?: boolean;
}) {
  const navigate = useNavigate();
  const specs = recordSpecs(record);
  const tiles = record.tiles ?? [];
  const viewTile = scope !== "pooled" && tiles.includes(scope) ? scope : tiles[0];
  const [source, tileId] = viewTile ? splitRef(viewTile) : ["", ""];
  const params = useMemo(() => ({ source, models: specs.join(",") }), [source, specs]);
  const follow = useFollowId(tileId);
  const errors = Object.entries(record.errors ?? {});
  const skipped = Object.entries(record.skipped ?? {});
  const gateSpecs = specs.filter((s) => s === "production" || s.startsWith("gate:"));
  const tileResults = scope !== "pooled" ? record.results?.[scope] ?? {} : {};
  const counts = record.counts ?? {};
  const openTiles = () => {
    useSelection.getState().select("tile", tiles);
    navigate("/sky/results");
  };
  return (
    <div className="res-exp">
      <div className="res-exp__head">
        <StateBadge state={record.status} />
        <strong className="res-exp__title">{record.label || record.id}</strong>
        {record.label && <code className="mono muted">{record.id}</code>}
        <span className="res-exp__spacer" />
        <LogToTrackingButton note={() => experimentMarkdown(record)} disabled={record.status === "running"} />
        <Button size="sm" icon="reset" disabled={record.status === "running" || !specs.length}
          onClick={() => { void runModels(tiles, specs, { label: record.label ? `${record.label} (re-run)` : undefined }); }}>
          Re-run
        </Button>
        <Button size="sm" icon="table" onClick={openTiles}>Tiles in results</Button>
      </div>
      <DefList dense items={[
        ["created", record.created ? formatDateTime(record.created) : "—"],
        record.duration_s != null ? ["duration", formatDuration(record.duration_s)] : null,
        ["tiles", tiles.length],
        ["models", <span className="res-chips">{specs.map((s) => <Badge key={s} size="sm">{specShort(s)}</Badge>)}</span>],
        ["members", `${counts.members_computed ?? 0} computed · ${counts.members_reused ?? 0} reused${counts.members_not_cached ? ` · ${counts.members_not_cached} not cached` : ""}`],
        ["outputs", `${counts.outputs_computed ?? 0} computed · ${counts.outputs_reused ?? 0} reused`],
      ]} />
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
      <div className="res-bar res-bar--inline">
        <Field label="Scope" inline>
          <Select size="sm" value={scope} onChange={onScope} aria-label="Metric scope"
            options={[{ value: "pooled", label: `All ${tiles.length} tiles (pooled)` }, ...tiles.map((t) => ({ value: t, label: t }))]} />
        </Field>
        {scope !== "pooled" && (
          <Button size="sm" variant="ghost" onClick={() => openInspector({ kind: "realtile", id: scope })}>Inspect tile</Button>
        )}
      </div>
      {viewer && viewTile && tileId && (
        <Section title="Comparison" sub={viewTile} collapsible defaultOpen>
          <div className="res-exp__viewer">
            {/* nav off: the collection walks every tile of the source, the Scope
                select is this experiment's navigation (useFollowId). */}
            <ImageViewer key={`${record.id}:${source}:${specs.join(",")}`} collection="real" params={params} initialId={tileId}
              nav={false} onReady={follow.onReady} onState={follow.onState}
              tiers={["lr", ...specs.slice(0, 4).map((s) => `m:${s}`), ...(source === "nexus" || source === "pair" ? ["jwst"] : [])]}
              id={`experiment-${record.id}`}
              urlKey="exp" toolbar="full" />
          </div>
        </Section>
      )}
      <Section title="Metrics" sub={scope === "pooled" ? "pooled over tiles" : scope} collapsible defaultOpen>
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
    </div>
  );
}
