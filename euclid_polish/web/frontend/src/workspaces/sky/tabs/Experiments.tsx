/* Sky › Experiments (spec §7.3): model comparison on real tiles, image first.
 * The picked experiment (`?exp=`, else the newest) is the top of the page: its comparison
 * viewer, then the per-band metrics (chart + table, CSV), gate core weights,
 * log to tracking. Below: the history (a row opens that experiment at the
 * top), the new-experiment form — tiles from the atlas / Real-results
 * selection, a `?tiles=` link or pasted refs; models from the catalogue; the
 * cost stated before the Run confirm — and the metric definitions, readable
 * before any experiment exists. One cancellable local job caches every
 * (tile, model) SR with its real-data metrics. Selection, scope and metric are
 * in the URL. */
import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useJob } from "../../../api/jobs";
import { useResource } from "../../../api/query";
import { usePageActions } from "../../../app/palette";
import { formatDuration, formatRelative } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { useSelected } from "../../../state/selection";
import {
  Badge, Button, Callout, Card, CardBody, Chip, DataTable, EmptyState, IconButton, Input, JobProgress,
  Page, Section, Skeleton, type DataColumn,
} from "../../../ui";
import { refreshResults, runModels } from "../results/actions";
import { URLS, type ExperimentRecord, type ExperimentSummary, type ExperimentsPayload } from "../results/api";
import { MetricDefinitions, StateBadge } from "../results/common";
import { ExperimentDetail } from "../results/ExperimentDetail";
import { ModelPicker, useModels } from "../results/ModelPicker";
import {
  defaultExperimentId, experimentCost, experimentCostText, METRIC_BY_KEY, parseRefs, specShort, type MetricKey,
} from "../results/model";
import "../results/register";
import "../results/results.css";


const HISTORY: DataColumn<ExperimentSummary>[] = [
  { id: "created", header: "Created", accessor: (e) => e.created ?? "", cell: (e) => (e.created ? formatRelative(e.created) : "—"), width: 110 },
  { id: "status", header: "Status", accessor: (e) => e.status ?? "", cell: (e) => <StateBadge state={e.status} />, width: 96 },
  { id: "label", header: "Label", accessor: (e) => e.label || "", cell: (e) => e.label || <span className="muted">—</span>, width: 180 },
  { id: "id", header: "Id", cell: (e) => <code className="mono">{e.id}</code>, hidden: true },
  { id: "tiles", header: "Tiles", numeric: true, accessor: (e) => e.tiles?.length ?? 0, filterText: (e) => (e.tiles ?? []).join(" "), width: 64 },
  { id: "models", header: "Models", accessor: (e) => (e.models ?? []).join(" "),
    cell: (e) => (
      <span className="res-chips res-chips--tight">
        {(e.models ?? []).slice(0, 4).map((m) => <Badge key={m} size="sm" tone={e.skipped?.[m] ? "warn" : undefined}>{specShort(m)}</Badge>)}
        {(e.models?.length ?? 0) > 4 && <span className="muted">+{(e.models?.length ?? 0) - 4}</span>}
      </span>
    ), width: 240 },
  { id: "outputs", header: "Outputs", numeric: true, accessor: (e) => (e.counts?.outputs_computed ?? 0) + (e.counts?.outputs_reused ?? 0),
    cell: (e) => <span title={`${e.counts?.outputs_computed ?? 0} computed · ${e.counts?.outputs_reused ?? 0} reused`}>{(e.counts?.outputs_computed ?? 0) + (e.counts?.outputs_reused ?? 0)}</span>, width: 76 },
  { id: "errors", header: "Errors", numeric: true, accessor: (e) => Object.keys(e.errors ?? {}).length,
    cell: (e) => { const n = Object.keys(e.errors ?? {}).length; return n ? <Badge size="sm" tone="bad">{n}</Badge> : "0"; }, width: 64 },
];

function NewExperiment({ tiles, setTiles, models, setModels, onStarted }: {
  tiles: string[]; setTiles: (t: string[]) => void; models: string[]; setModels: (m: string[]) => void;
  onStarted: (id: string | null) => void;
}) {
  const selection = useSelected("tile");
  const job = useJob("sky:experiment");
  const catalogue = useModels();
  const [paste, setPaste] = useState("");
  const [label, setLabel] = useState("");
  const [busy, setBusy] = useState(false);
  const fromSelection = selection.filter((s) => !tiles.includes(s));
  const cost = catalogue.data && tiles.length && models.length
    ? experimentCostText(experimentCost(models, catalogue.data.models, tiles.length)) : "";
  const addPasted = () => {
    const refs = parseRefs(paste);
    if (refs.length) setTiles([...tiles, ...refs.filter((r) => !tiles.includes(r))]);
    setPaste("");
  };
  const run = async () => {
    setBusy(true);
    try {
      const r = await runModels(tiles, models, { label });
      if (r) { onStarted(r.experimentId); setLabel(""); }
    } finally { setBusy(false); }
  };
  return (
    <div className="res-new">
      <div className="res-new__tiles">
        <div className="res-bar res-bar--inline">
          <strong>Tiles</strong>
          <span className="muted">{tiles.length}</span>
          <span className="res-bar__spacer" />
          {!!fromSelection.length && (
            <Button size="sm" variant="subtle" onClick={() => setTiles([...tiles, ...fromSelection])}>
              Add the selection ({fromSelection.length})
            </Button>
          )}
          <Button size="sm" variant="ghost" asChild><Link to="/sky/results">Pick in Real results</Link></Button>
          {!!tiles.length && <Button size="sm" variant="ghost" onClick={() => setTiles([])}>Clear</Button>}
        </div>
        {tiles.length ? (
          <div className="res-tilelist" aria-label="Experiment tiles">
            {tiles.map((t) => <Chip key={t} onRemove={() => setTiles(tiles.filter((x) => x !== t))}><span className="mono">{t}</span></Chip>)}
          </div>
        ) : <p className="muted res-note">Select tiles on the atlas or in Real results, or paste refs below.</p>}
        <Input size="sm" value={paste} onChange={setPaste} onEnter={addPasted} icon="search"
          placeholder="Paste refs: nexus/f200w-0040 poster/… (Enter)" aria-label="Add tiles by ref" />
      </div>
      <div>
        <ModelPicker value={models} onChange={setModels} />
      </div>
      <div className="res-new__foot">
        <Input size="sm" value={label} onChange={setLabel} placeholder="Label (optional)" aria-label="Experiment label" />
        <Button variant="primary" icon="activity" loading={busy || job.busy} disabled={!tiles.length || !models.length}
          onClick={run}>
          Run {plural(models.length, "model")} on {plural(tiles.length, "tile")}
        </Button>
        {cost && <span className="res-note res-new__cost">{cost}</span>}
        <JobProgress job={job.job} error={job.error} />
      </div>
    </div>
  );
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

export default function Experiments() {
  const [tiles, setTiles] = useUrlState<string[]>("tiles", []);
  const [models, setModels] = useUrlState<string[]>("models", ["production", "mean"]);
  const [exp, setExp] = useUrlState("exp", "");
  const [scope, setScope] = useUrlState("scope", "pooled");
  const [metricRaw, setMetric] = useUrlState("metric", "hole_pct");
  const metric = (METRIC_BY_KEY[metricRaw] ? metricRaw : "hole_pct") as MetricKey;
  const [newOpen, setNewOpen] = useState<boolean | null>(null);
  const [defsOpen, setDefsOpen] = useState<boolean | null>(null);
  const top = useRef<HTMLDivElement>(null);
  const newRef = useRef<HTMLDivElement>(null);

  const [pollList, setPollList] = useState(false);
  const list = useResource<ExperimentsPayload>(URLS.experiments, [], { ttl: 10_000, poll: pollList ? 4_000 : undefined });
  const history = useMemo(() => list.data?.experiments ?? [], [list.data]);
  const anyRunning = history.some((e) => e.status === "running");
  useEffect(() => { setPollList(anyRunning); }, [anyRunning]);
  // A plain visit (the tab strip's link) opens the newest experiment — the
  // comparison first — without writing it to the URL; handed-over tiles
  // (`?tiles=`) open the form instead.
  const current = defaultExperimentId(history, exp, tiles);
  const summary = history.find((e) => e.id === current);
  const running = summary?.status === "running";
  const record = useResource<ExperimentRecord>(current ? URLS.experiment(current) : null, [], { ttl: 5_000, poll: running ? 3_000 : undefined });

  // The form is open when there is nothing to look at, or tiles were handed over.
  const open = newOpen ?? (!history.length || !!tiles.length || !current);
  const onStarted = (id: string | null) => {
    if (id) { setExp(id); setScope("pooled"); }
    setNewOpen(false);
    void list.reload();
  };
  const reveal = (el: HTMLElement | null) => {
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    el?.scrollIntoView?.({ block: "start", behavior: reduce ? "auto" : "smooth" });
  };
  const pick = (id: string) => {
    setExp(id);
    setScope("pooled");
    setTimeout(() => reveal(top.current), 0);   // after the render: the comparison is at the top
  };
  const openNew = () => { setNewOpen(true); setTimeout(() => reveal(newRef.current), 0); };
  usePageActions([
    { id: "exp-run", label: "Run the new experiment", group: "Experiments", disabled: !tiles.length || !models.length,
      run: () => { void runModels(tiles, models).then((r) => r && onStarted(r.experimentId)); } },
    { id: "exp-new", label: "New experiment…", group: "Experiments", run: openNew },
    { id: "exp-refresh", label: "Refresh experiments", group: "Experiments", run: () => { refreshResults(); void list.reload(); } },
  ]);

  return (
    <Page className="res-page">
      <div ref={top} className="res-exp-top">
        {current && (
          record.loading && !record.data ? <Skeleton lines={8} />
            : !record.data ? (
              <Callout tone="bad" title={record.error?.status === 404 ? `Unknown experiment ${current}` : "Could not load the experiment"}
                action={<Button size="sm" onClick={() => setExp("")}>Close</Button>}>
                {record.error?.message ?? "No data."}
              </Callout>
            ) : (
              <section aria-label="Experiment detail" className="res-exp-detail">
                {record.data.status === "running" && (
                  <Callout tone="info" title="Running">
                    {record.data.duration_s != null ? formatDuration(record.data.duration_s) : "Results appear here as each tile finishes."}
                  </Callout>
                )}
                <ExperimentDetail record={record.data} scope={scope} onScope={setScope} metric={metric}
                  onMetric={setMetric} />
              </section>
            )
        )}
      </div>

      <Card><CardBody>
        <Section title="History" sub={history.length ? String(history.length) : undefined}
          right={<>
            {current && !open && <Button size="sm" icon="plus" onClick={openNew}>New experiment</Button>}
            <IconButton icon="reset" label="Refresh" size="sm" onClick={() => { void list.reload(); void record.reload(); }} />
          </>}>
          {list.loading ? <Skeleton lines={4} />
            : list.error && !history.length ? (
              <Callout tone="bad" title="Could not load experiments" action={<Button size="sm" onClick={list.reload}>Retry</Button>}>
                {list.error.message}
              </Callout>
            ) : !history.length ? (
              <EmptyState icon="table" title="No experiments yet"
                action={<Button size="sm" asChild><Link to="/sky/results">Pick tiles in Real results</Link></Button>}>
                Compare production, the mean, gate variants and single members on real bright objects.
              </EmptyState>
            ) : (
              <DataTable rows={history} columns={HISTORY} rowKey={(e) => e.id} aria-label="Experiments"
                activeKey={current || null} onRowClick={(e) => pick(e.id)}
                dense height={history.length > 8 ? 300 : "auto"} exportName="experiments" urlKey="eh" />
            )}
        </Section>
      </CardBody></Card>

      <div ref={newRef}>
        <Card><CardBody>
          <Section title="New experiment" sub={tiles.length ? `${plural(tiles.length, "tile")}, ${plural(models.length, "model")}` : undefined}
            collapsible open={open} onOpenChange={setNewOpen}>
            <NewExperiment tiles={tiles} setTiles={setTiles} models={models} setModels={setModels} onStarted={onStarted} />
          </Section>
        </CardBody></Card>
      </div>

      <Card><CardBody>
        <Section title="What the metrics measure" sub="computed per band for every tile and model"
          collapsible open={defsOpen ?? (!list.loading && !history.length)} onOpenChange={setDefsOpen}>
          <MetricDefinitions />
        </Section>
      </CardBody></Card>
    </Page>
  );
}
