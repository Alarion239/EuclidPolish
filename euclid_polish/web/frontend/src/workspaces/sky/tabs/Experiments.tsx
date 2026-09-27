/* Sky › Experiments (spec §7.3): model comparison on real tiles. Pick tiles
 * (the atlas / Real-results selection, a `?tiles=` link, or pasted refs) and
 * models from the catalogue → one cancellable local job caches every
 * (tile, model) SR with its real-data metrics → the history table and the
 * detail: comparison viewer, per-band metrics table + chart, gate core
 * weights, CSV, log to tracking. Selection, scope and metric are in the URL. */
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useJob } from "../../../api/jobs";
import { useResource } from "../../../api/query";
import { usePageActions } from "../../../app/palette";
import { formatDuration, formatRelative } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { useSelected } from "../../../state/selection";
import {
  Badge, Button, Callout, Card, CardBody, Chip, DataTable, EmptyState, IconButton, Input, JobProgress,
  Page, Section, Skeleton, Tooltip, type DataColumn,
} from "../../../ui";
import { refreshResults, runModels } from "../results/actions";
import { URLS, type ExperimentRecord, type ExperimentSummary, type ExperimentsPayload } from "../results/api";
import { StateBadge } from "../results/common";
import { ExperimentDetail } from "../results/ExperimentDetail";
import { ModelPicker } from "../results/ModelPicker";
import { METRIC_BY_KEY, parseRefs, specShort, type MetricKey } from "../results/model";
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
  const [paste, setPaste] = useState("");
  const [label, setLabel] = useState("");
  const [busy, setBusy] = useState(false);
  const fromSelection = selection.filter((s) => !tiles.includes(s));
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
              + selection ({fromSelection.length})
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
          Run {models.length} × {tiles.length}
        </Button>
        <JobProgress job={job.job} error={job.error} />
      </div>
    </div>
  );
}

export default function Experiments() {
  const [tiles, setTiles] = useUrlState<string[]>("tiles", []);
  const [models, setModels] = useUrlState<string[]>("models", ["production", "mean"]);
  const [exp, setExp] = useUrlState("exp", "");
  const [scope, setScope] = useUrlState("scope", "pooled");
  const [metricRaw, setMetric] = useUrlState("metric", "hole_pct");
  const metric = (METRIC_BY_KEY[metricRaw] ? metricRaw : "hole_pct") as MetricKey;
  const [newOpen, setNewOpen] = useState<boolean | null>(null);

  const [pollList, setPollList] = useState(false);
  const list = useResource<ExperimentsPayload>(URLS.experiments, [], { ttl: 10_000, poll: pollList ? 4_000 : undefined });
  const history = useMemo(() => list.data?.experiments ?? [], [list.data]);
  const anyRunning = history.some((e) => e.status === "running");
  useEffect(() => { setPollList(anyRunning); }, [anyRunning]);
  const current = exp || "";
  const summary = history.find((e) => e.id === current);
  const running = summary?.status === "running";
  const record = useResource<ExperimentRecord>(current ? URLS.experiment(current) : null, [], { ttl: 5_000, poll: running ? 3_000 : undefined });

  const open = newOpen ?? (!history.length || !!tiles.length || !current);
  const onStarted = (id: string | null) => {
    if (id) { setExp(id); setScope("pooled"); }
    setNewOpen(false);
    void list.reload();
  };
  usePageActions([
    { id: "exp-run", label: "Run the new experiment", group: "Experiments", disabled: !tiles.length || !models.length,
      run: () => { void runModels(tiles, models).then((r) => r && onStarted(r.experimentId)); } },
    { id: "exp-new", label: "New experiment…", group: "Experiments", run: () => setNewOpen(true) },
    { id: "exp-refresh", label: "Refresh experiments", group: "Experiments", run: () => { refreshResults(); void list.reload(); } },
  ]);

  return (
    <Page className="res-page">
      <Card><CardBody>
        <Section title="New experiment" sub={tiles.length ? `${tiles.length} tile${tiles.length === 1 ? "" : "s"} × ${models.length} model${models.length === 1 ? "" : "s"}` : undefined}
          collapsible open={open} onOpenChange={setNewOpen}
          right={<Tooltip content="Holes, enclosed-flux R and flux ratios per band are computed for every (tile, model); cached members are reused.">
            <IconButton icon="help" label="About experiments" size="sm" />
          </Tooltip>}>
          <NewExperiment tiles={tiles} setTiles={setTiles} models={models} setModels={setModels} onStarted={onStarted} />
        </Section>
      </CardBody></Card>

      <Card><CardBody>
        <Section title="History" sub={history.length ? String(history.length) : undefined}
          right={<IconButton icon="reset" label="Refresh" size="sm" onClick={() => { void list.reload(); void record.reload(); }} />}>
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
                activeKey={current || null} onRowClick={(e) => { setExp(e.id); setScope("pooled"); }}
                dense height={history.length > 8 ? 300 : "auto"} exportName="experiments" urlKey="eh" />
            )}
        </Section>
      </CardBody></Card>

      {current && (
        <Card aria-label="Experiment detail">
          <CardBody>
            {record.loading && !record.data ? <Skeleton lines={8} />
              : !record.data ? (
                <Callout tone="bad" title={record.error?.status === 404 ? `Unknown experiment ${current}` : "Could not load the experiment"}
                  action={<Button size="sm" onClick={() => setExp("")}>Close</Button>}>
                  {record.error?.message ?? "No data."}
                </Callout>
              ) : (
                <>
                  {record.data.status === "running" && (
                    <Callout tone="info" title="Running">
                      {record.data.duration_s != null ? formatDuration(record.data.duration_s) : "Results appear here as each tile finishes."}
                    </Callout>
                  )}
                  <ExperimentDetail record={record.data} scope={scope} onScope={setScope} metric={metric}
                    onMetric={setMetric} />
                </>
              )}
          </CardBody>
        </Card>
      )}
    </Page>
  );
}
