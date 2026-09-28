/* Sky › Compare (console regrouping): which model is safest on real bright
 * objects? Models on real tiles, no truth: holes, enclosed flux and total
 * flux against the LR, with JWST where a tile has it. The only producer of
 * the real-holes numbers the Leaderboard, the Combiner and Home read.
 *
 * It opens on the newest comparison (`?exp=` picks another): the head
 * (comparison picker, scope pooled / one tile, the ONE metric-definitions
 * popover — `?defs=1` opens it, Targets links there —, Log to notebook, New
 * comparison, more), the page's one sentence, the viewer (LR, the models,
 * JWST) with a Δm-per-model footer, the metric select with the band strip
 * (a dot per model per band) and the pivot table (models × VIS / Y / J / H,
 * the long form as CSV), the gate core weights of one tile, the run details
 * (collapsed), the history (with its headline result) and the New
 * comparison drawer (`?new=1`). Opening the page runs nothing. */
import { useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import { formatDateTime, formatDuration } from "../../../format";
import { useMediaQuery } from "../../../hooks/useMediaQuery";
import { useUrlState } from "../../../hooks/useUrlState";
import { useSelection } from "../../../state/selection";
import {
  Badge, Button, Callout, Caption, DataTable, DefList, Details, EmptyState, IconButton, Menu, Page, Popover, Section,
  Select, Skeleton, Toolbar, ToolbarGroup, ToolbarSpacer, downloadText, type MenuItem,
} from "../../../ui";
import { ImageViewer } from "../../../viewer";
import { BandStrip, CompareSentence, DeltaFooter, PivotTable, historyColumns, useSpecColors } from "../compare/parts";
import { comparisonLabel, longCsv } from "../compare/model";
import { NewComparison } from "../compare/NewComparison";
import { refreshResults, runModels } from "../results/actions";
import { splitRef, URLS, type ExperimentRecord, type ExperimentsPayload } from "../results/api";
import { CoreWeights, MetricDefinitions } from "../results/common";
import { useFollowViewer } from "../results/follow";
import { LogToNotebookButton } from "../../shared/LogToNotebook";
import {
  CHART_METRICS, defaultExperimentId, experimentMarkdown, METRIC_BY_KEY, recordSpecs, specShort, specWords, type MetricKey,
} from "../results/model";
import "../compare/compare.css";
import "../results/register";
import "../results/results.css";

/** The metrics the pivot offers (the chart metrics; counts stay in the CSV). */
const PIVOT_METRICS: MetricKey[] = CHART_METRICS;
const NEW_ID = "cmp-new";
const HISTORY = historyColumns();

/** The tiers a URL carried for the comparison viewer (`v.cmp.t`), read once
 *  at mount: a shared link's tier choice wins over the page's default. */
function urlTiers(search: string): string[] | null {
  const raw = new URLSearchParams(search).get("v.cmp.t");
  const list = raw ? raw.split(",").map((t) => t.trim()).filter(Boolean) : [];
  return list.length ? list : null;
}

function Comparison({ record, scope, setScope, metric, setMetric, onDefs }: {
  record: ExperimentRecord; scope: string; setScope: (s: string) => void; metric: MetricKey; setMetric: (m: MetricKey) => void;
  /** Opens the one metric-definitions popover (the head's). */
  onDefs: () => void;
}) {
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
  const [mountTiers] = useState(() => urlTiers(location.search));
  // nav off: the collection walks every tile of the source; the scope select
  // is this comparison's navigation (the follower keeps the images on it).
  const follow = useFollowViewer(tileId, mountTiers ?? tiers);
  const colors = useSpecColors(record);
  const errors = Object.entries(record.errors ?? {});
  const skipped = Object.entries(record.skipped ?? {});
  const gateSpecs = specs.filter((s) => s === "production" || s.startsWith("gate:"));
  const tileResults = scope !== "pooled" ? record.results?.[scope] ?? {} : {};
  const counts = record.counts ?? {};
  const def = METRIC_BY_KEY[metric];
  const exportCsv = () => downloadText(`comparison-${record.id}-${scope.replace("/", "_")}.csv`, longCsv(record, scope), "text/csv;charset=utf-8");
  return (
    <>
      <CompareSentence record={record} scope={scope} />
      {record.status === "running" && (
        <Callout tone="info" title="Running">
          {record.duration_s != null ? `${formatDuration(record.duration_s)} so far. ` : ""}Results appear here as each tile finishes.
        </Callout>
      )}
      {viewTile && tileId && (
        <div className="cmp-viewer" role="region" aria-label={`Comparison on ${viewTile}`}>
          <ImageViewer key={`${record.id}:${source}:${specs.join(",")}`} collection="real" params={params} initialId={tileId}
            nav={false} onReady={follow.onReady} onState={follow.onState} tiers={tiers}
            id={`comparison-${record.id}`} urlKey="cmp" toolbar="full" />
          <DeltaFooter record={record} tile={viewTile} />
        </div>
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
      <section className="cmp-metric" aria-label="Metric per model and band">
        <Toolbar label="Metric" plain>
          <ToolbarGroup label="Metric" hideLabel>
            <Select size="sm" value={metric} onChange={(v) => setMetric(v as MetricKey)} aria-label="Metric"
              options={PIVOT_METRICS.map((k) => ({ value: k, label: METRIC_BY_KEY[k].label }))} />
          </ToolbarGroup>
          {def?.better && <span className="cmp-metric__hint">{def.better === "lower" ? "Lower is better." : def.better === "higher" ? "Higher is better." : "1 is ideal."}</span>}
          <Button size="sm" variant="ghost" icon="help" onClick={onDefs}>What it measures</Button>
          <ToolbarSpacer />
          <Button size="sm" variant="ghost" icon="download" onClick={exportCsv}>CSV, every metric</Button>
        </Toolbar>
        <BandStrip record={record} scope={scope} metric={metric} colors={colors} />
        <PivotTable record={record} scope={scope} metric={metric} colors={colors} />
        <Caption>
          {scope === "pooled" ? `Pooled over the ${tiles.length === 1 ? "tile" : `${tiles.length} tiles`}` : `On ${scope}`}; the best value per band is bold.
          {scope === "pooled" && tiles.length > 1 ? " Pick one tile in the scope to see it alone." : ""}
        </Caption>
      </section>
      {scope !== "pooled" && gateSpecs.some((s) => tileResults[s]?.metrics?.gate_core_weights) && (
        <Section title="Gate core weights" sub="top members over the brightest 1 % of pixels" collapsible defaultOpen>
          {gateSpecs.map((s) => tileResults[s]?.metrics?.gate_core_weights ? (
            <div key={s} className="res-weights__block">
              <span>{specWords(s)}</span>
              <CoreWeights weights={tileResults[s]?.metrics?.gate_core_weights} />
            </div>
          ) : null)}
        </Section>
      )}
      <Details summary="Run details">
        <DefList dense items={[
          ["id", <code className="mono">{record.id}</code>],
          ["created", record.created ? formatDateTime(record.created) : "—"],
          record.duration_s != null ? ["duration", formatDuration(record.duration_s)] : null,
          ["tiles", <span className="res-chips">{tiles.map((t) => (
            <Button key={t} size="sm" variant="ghost" onClick={() => setScope(t)}><span className="mono">{t}</span></Button>
          ))}</span>],
          ["models", <span className="res-chips">{specs.map((s) => <Badge key={s} size="sm" title={record.model_labels?.[s]}>{specShort(s)}</Badge>)}</span>],
          ["member SRs", `${counts.members_computed ?? 0} computed, ${counts.members_reused ?? 0} reused${counts.members_not_cached ? `, ${counts.members_not_cached} not cached` : ""}`],
          ["outputs", `${counts.outputs_computed ?? 0} computed, ${counts.outputs_reused ?? 0} reused`],
        ]} />
      </Details>
    </>
  );
}

export default function Compare() {
  const navigate = useNavigate();
  const reduce = useMediaQuery("(prefers-reduced-motion: reduce)");
  const [tiles, setTiles] = useUrlState<string[]>("tiles", []);
  const [models, setModels] = useUrlState<string[]>("models", ["production", "mean"]);
  const [exp, setExp] = useUrlState("exp", "");
  const [scope, setScope] = useUrlState("scope", "pooled");
  const [metricRaw, setMetric] = useUrlState("metric", "hole_pct");
  const [defs, setDefs] = useUrlState("defs", false);
  const [newFlag, setNewFlag] = useUrlState("new", false);
  const metric = (PIVOT_METRICS.includes(metricRaw as MetricKey) ? metricRaw : "hole_pct") as MetricKey;
  const top = useRef<HTMLDivElement>(null);

  const [pollList, setPollList] = useState(false);
  const list = useResource<ExperimentsPayload>(URLS.experiments, [], { ttl: 10_000, poll: pollList ? 4_000 : undefined });
  const history = useMemo(() => list.data?.experiments ?? [], [list.data]);
  const anyRunning = history.some((e) => e.status === "running");
  useEffect(() => { setPollList(anyRunning); }, [anyRunning]);
  // A plain visit opens the newest comparison without writing it to the
  // URL; handed-over tiles (`?tiles=`) open the New-comparison drawer.
  const current = defaultExperimentId(history, exp, tiles);
  const summary = history.find((e) => e.id === current);
  const running = summary?.status === "running";
  const record = useResource<ExperimentRecord>(current ? URLS.experiment(current) : null, [], { ttl: 5_000, poll: running ? 3_000 : undefined });
  const newOpen = newFlag || (!list.loading && !history.length) || !!tiles.length;

  const reveal = (id: string | null) => {
    requestAnimationFrame(() => requestAnimationFrame(() => {
      const el = id ? document.getElementById(id) : top.current;
      el?.scrollIntoView?.({ block: "start", behavior: reduce ? "auto" : "smooth" });
    }));
  };
  // Opened from a link (`?new=1`, `?tiles=`): bring the drawer into view
  // once, after what sits above it has loaded (the comparison and its viewer).
  const revealed = useRef(false);
  const settled = !list.loading && !(current && record.loading);
  useEffect(() => {
    if (revealed.current || !(newFlag || tiles.length) || !settled) return;
    const timer = setTimeout(() => { revealed.current = true; reveal(NEW_ID); }, 300);
    return () => clearTimeout(timer);
  }, [newFlag, tiles.length, settled]);   // eslint-disable-line react-hooks/exhaustive-deps
  const openNew = () => { setNewFlag(true); reveal(NEW_ID); };
  const setOpen = (open: boolean) => {
    setNewFlag(open);
    if (!open && tiles.length) setTiles([]);
  };
  const onStarted = (id: string | null) => {
    if (id) { setExp(id); setScope("pooled"); }
    setNewFlag(false);
    setTiles([]);
    void list.reload();
    reveal(null);
  };
  const pick = (id: string) => {
    setExp(id);
    setScope("pooled");
    reveal(null);
  };
  const reload = () => { refreshResults(); void list.reload(); void record.reload(); };

  usePageActions([
    { id: "cmp-new", label: "New comparison…", group: "Compare", run: openNew },
    { id: "cmp-defs", label: "What the real-data metrics measure", group: "Compare", keywords: ["holes", "flux", "definitions"], run: () => setDefs(true) },
    { id: "cmp-refresh", label: "Refresh comparisons", group: "Compare", run: reload },
  ]);

  const r = record.data;
  const tilesOf = r?.tiles ?? [];
  const more: MenuItem[] = r ? [
    { label: "Re-run this comparison…", disabled: r.status === "running" || !recordSpecs(r).length,
      onSelect: () => { void runModels(tilesOf, recordSpecs(r), { label: r.label ? `${r.label} (re-run)` : undefined }); } },
    { label: "Select its tiles in Targets", onSelect: () => { useSelection.getState().select("tile", tilesOf); navigate("/sky/targets"); } },
    ...(scope !== "pooled" ? [{ label: "Open the tile card", onSelect: () => openInspector({ kind: "tile", id: scope }) }] : []),
    { type: "separator" },
    { label: "Refresh", onSelect: reload },
  ] : [{ label: "Refresh", onSelect: reload }];
  const pooledLabel = tilesOf.length === 1 ? "Pooled (1 tile)" : `Pooled (${tilesOf.length} tiles)`;
  const choices = history.map((e) => ({ value: e.id, label: comparisonLabel(e) }));

  return (
    <Page className="res-page cmp-page">
      <div ref={top} className="cmp-top">
        <Toolbar label="Comparison">
          {history.length > 0 && (
            <ToolbarGroup label="Comparison" hideLabel>
              <Select size="sm" value={current} onChange={pick} aria-label="Comparison" className="cmp-pick" searchable={history.length > 8}
                options={choices} />
            </ToolbarGroup>
          )}
          {r && (
            <ToolbarGroup label="Scope" hideLabel>
              <Select size="sm" value={scope} onChange={setScope} aria-label="Scope" className="cmp-scope"
                options={[{ value: "pooled", label: pooledLabel }, ...tilesOf.map((t) => ({ value: t, label: t }))]} />
            </ToolbarGroup>
          )}
          <ToolbarSpacer />
          <Popover label="What the metrics measure" width={440} align="end" open={defs} onOpenChange={setDefs}
            trigger={<IconButton icon="help" label="What the metrics measure" size="sm" pressed={defs} />}>
            <div className="res-form">
              <strong>What the metrics measure</strong>
              <p className="res-note">Real tiles have no truth: every metric compares the SR with its own LR, per band.</p>
              <MetricDefinitions />
            </div>
          </Popover>
          {r && <LogToNotebookButton from="Sky › Compare" note={() => experimentMarkdown(r)} disabled={r.status === "running"} />}
          <Button size="sm" icon="plus" onClick={openNew}>New comparison</Button>
          <Menu label="More comparison actions" items={more} trigger={<IconButton icon="more" size="sm" label="More comparison actions" />} />
        </Toolbar>
        <p className="cmp-sub">Models on real tiles, no truth.</p>

        {list.loading ? <Skeleton lines={8} /> : list.error && !history.length ? (
          <Callout tone="bad" title="Could not load the comparisons" action={<Button size="sm" onClick={() => { void list.reload(); }}>Retry</Button>}>
            {list.error.message}
          </Callout>
        ) : !current ? (
          <EmptyState icon="table" title="No comparison yet"
            action={<Button size="sm" onClick={openNew}>New comparison</Button>}>
            Run production, the mean, gate variants or single members on the poster galaxy, NEXUS tiles or the Q1 targets.
          </EmptyState>
        ) : record.loading && !r ? <Skeleton lines={8} /> : !r ? (
          <Callout tone="bad" title={record.error?.status === 404 ? `Unknown comparison ${current}` : "Could not load the comparison"}
            action={<Button size="sm" onClick={() => setExp("")}>Show the newest</Button>}>
            {record.error?.message ?? "No data."}
          </Callout>
        ) : (
          <Comparison record={r} scope={scope !== "pooled" && !tilesOf.includes(scope) ? "pooled" : scope} setScope={setScope}
            metric={metric} setMetric={setMetric} onDefs={() => { setDefs(true); reveal(null); }} />
        )}
      </div>

      {history.length > 0 && (
        <Section title="History" collapsible defaultOpen>
          <DataTable rows={history} columns={HISTORY} rowKey={(e) => e.id} aria-label="Comparisons"
            activeKey={current || null} onRowClick={(e) => pick(e.id)}
            dense height={history.length > 8 ? 300 : "auto"} hideToolbar={history.length <= 8} exportName="comparisons" urlKey="ch" />
        </Section>
      )}

      <Section id={NEW_ID} className="cmp-drawer" title="New comparison"
        sub={tiles.length ? `${tiles.length} tile${tiles.length === 1 ? "" : "s"}, ${models.length} model${models.length === 1 ? "" : "s"}` : undefined}
        collapsible open={newOpen} onOpenChange={setOpen}>
        <NewComparison tiles={tiles} setTiles={setTiles} models={models} setModels={setModels} onStarted={onStarted} />
      </Section>
    </Page>
  );
}
