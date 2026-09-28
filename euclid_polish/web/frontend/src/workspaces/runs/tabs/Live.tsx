/* Runs › Live (`/runs/live`): everything running or queued, locally and on
 * FASRC, in ONE list — the running local jobs of this server and the live
 * SLURM jobs of $USER (the console's submissions, whose labels name the
 * members they train, plus the squeue jobs submitted from a shell, marked
 * CLI) — with the selected job beside it: a SLURM job's live monitor (one
 * card per array task: "step 10,650 / 70,000 (15%)", GPU / CPU %, cancel),
 * a local job's progress and log. Then the fail-stop submission queue.
 * The rail badge, the job tray and Home all land here.
 * URL: `scope` (all | local | slurm), `job` (a SLURM job id, or `local:<id>`)
 * and the table's `l.q` / `l.sort`. Nothing claims "offline" or "no job"
 * before the FASRC status and the SLURM feed have answered. */
import { useMemo } from "react";
import { Link } from "react-router-dom";
import { JOBS_FEED_KEY, cancelSlurmJob, useJobsFeed } from "../../../api/jobs";
import { queryClient, useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { pagePath } from "../../../app/nav";
import { usePageActions } from "../../../app/palette";
import { useFasrcStatus } from "../../../app/status";
import { formatDateTime, formatRelative } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Badge, Button, Callout, Card, CardBody, CardHead, DataTable, EmptyState, FactsList, IconButton, Page, Segmented,
  Skeleton, Toolbar, ToolbarGroup, ToolbarSpacer, Tooltip, confirm, toast, type DataColumn,
} from "../../../ui";
import { ConnectionBar } from "../Connection";
import { LocalJobCard, askCancel } from "../LocalJob";
import { QueuePanel } from "../Queue";
import {
  jobStateTone, liveItems, mergeCliJobs, scopeCounts, type LiveItem, type LiveRow, type Scope, type SqueueRow,
} from "../model";
import { SlurmMonitor } from "../steps/SlurmMonitor";
import "../runs.css";

const SCOPES: Scope[] = ["all", "local", "slurm"];
const SCOPE_LABEL: Record<Scope, string> = { all: "All", local: "This laptop", slurm: "SLURM" };
const parseScope = (raw: string): Scope | undefined => ((SCOPES as string[]).includes(raw) ? raw as Scope : undefined);

/** Whether the shared SLURM feed has answered once (the feed itself reports
 *  an empty list while its first poll is in flight). */
function useSlurmAnswered(): boolean {
  return queryClient.getQueryState([...JOBS_FEED_KEY, "slurm"])?.data !== undefined;
}

async function cancelSlurm(item: LiveItem) {
  if (!(await confirm({ title: `Cancel SLURM job ${item.id}?`, message: `${item.label}: scancel on FASRC; the job stops at once.`,
    tone: "danger", confirmLabel: "Cancel job" }))) return;
  const r = await cancelSlurmJob(item.id);
  if (r.ok) toast.success(`Cancelled ${item.id}`); else toast.error(r.error ?? "cancel refused");
}

/** What squeue says about a job submitted outside the console. */
function CliJobCard({ item }: { item: LiveItem }) {
  const r = item.slurm;
  return (
    <Card>
      <CardHead title={<span>Job <code className="mono">{item.id}</code></span>} sub="Submitted outside the console (squeue only)"
        right={<Badge size="sm" tone={jobStateTone(item.state)} dot={item.state === "RUNNING"}>{item.state}</Badge>} />
      <CardBody>
        <FactsList facts={[
          { label: "Name", value: item.label || "—" },
          item.elapsed ? { label: "Elapsed / limit", value: item.elapsed } : null,
          r?.nodes ? { label: "Nodes", value: String(r.nodes) } : null,
          r?.reason ? { label: "Reason", value: String(r.reason) } : null,
          r?.start_time ? { label: "Expected start", value: String(r.start_time) } : null,
        ]} />
      </CardBody>
    </Card>
  );
}

type Live = ReturnType<typeof useLiveItems>;

/** The one list: the running local jobs, the console's live SLURM jobs and
 *  the squeue jobs submitted from a shell (read only while connected). */
function useLiveItems() {
  const feed = useJobsFeed();
  const fasrc = useFasrcStatus();
  const slurmAnswered = useSlurmAnswered();
  const squeue = useResource<{ ok?: boolean; rows?: SqueueRow[] }>(
    feed.fasrcOffline || !fasrc.data?.ssh_connected ? null : "/api/fasrc/queue", [], { ttl: 10_000, poll: 20_000 });
  const all = useMemo(() => liveItems(feed.jobs, mergeCliJobs(feed.slurm as LiveRow[], squeue.data?.rows)),
    [feed.jobs, feed.slurm, squeue.data]);
  return { feed, fasrc, slurmAnswered, squeue, all };
}

/** The live list and the selected job's monitor (also the fasrc.tsx
 *  `CurrentSubmission` panel). */
export function LiveJobs({ scope = "all" }: { scope?: Scope }) {
  return <LiveList live={useLiveItems()} scope={scope} />;
}

function LiveList({ live, scope }: { live: Live; scope: Scope }) {
  const { feed, fasrc, slurmAnswered, squeue, all } = live;
  const [job, setJob] = useUrlState("job", "");
  const rows = useMemo(() => (scope === "all" ? all : all.filter((i) => i.source === scope)), [all, scope]);
  const selectedKey = job ? (job.startsWith("local:") ? job : `slurm:${job}`) : rows[0]?.key ?? "";
  const selected = all.find((i) => i.key === selectedKey) ?? null;
  // A job named in the URL that has left the list (finished) still opens.
  const slurmId = !selected && job && !job.startsWith("local:") ? job : selected?.source === "slurm" && !selected.cli ? selected.id : "";
  const localId = !selected && job.startsWith("local:") ? job.slice("local:".length) : selected?.source === "local" ? selected.id : "";

  usePageActions([
    { id: "live-refresh", label: "Refresh the live jobs", group: "Runs", run: feed.refresh },
    { id: "live-cancel", label: "Cancel the selected job", group: "Runs", disabled: !selected?.cancellable,
      run: () => { if (!selected) return; if (selected.source === "local" && selected.job) void askCancel(selected.job); else void cancelSlurm(selected); } },
  ]);

  const columns = useMemo<DataColumn<LiveItem>[]>(() => [
    { id: "state", header: "State", width: 108,
      cell: (i) => <Badge size="sm" tone={jobStateTone(i.state)} dot={i.state === "RUNNING"}>{i.job?.cancel_requested ? "CANCELLING" : i.state}</Badge> },
    { id: "label", header: "Job", width: 260, accessor: (i) => `${i.label} ${i.sub}`,
      cell: (i) => (
        <span className="runs-cell2">
          <span title={i.label}>{i.label}</span>
          <span className="runs-dim runs-small">
            {i.cli ? <Tooltip content="Submitted outside the console: squeue only, no event stream, read-only here.">
              <span><Badge size="sm" tone="neutral">CLI</Badge></span></Tooltip>
              : <><span className="mono">{i.source === "slurm" ? `#${i.id}` : i.sub}</span>{i.source === "slurm" && i.sub ? ` · ${i.sub}` : ""}</>}
          </span>
        </span>
      ) },
    { id: "source", header: "Where", width: 120, accessor: (i) => i.where,
      cell: (i) => <span className="mono runs-small runs-ellipsis" title={i.where}>{i.where || "—"}</span> },
    { id: "progress", header: "Progress", width: 190, priority: 2,
      cell: (i) => <span className="mono runs-small">{i.progress || "—"}</span> },
    { id: "elapsed", header: "Elapsed / limit", width: 120, priority: 3,
      cell: (i) => <span className="mono runs-small">{i.elapsed || "—"}</span> },
    { id: "startedAt", header: "Started", width: 96, priority: 1, accessor: (i) => i.startedAt ?? 0,
      cell: (i) => <span className="runs-dim runs-small" title={formatDateTime(i.startedAt)}>{i.startedAt ? formatRelative(i.startedAt) : "—"}</span> },
    { id: "actions", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 72,
      cell: (i) => i.cli ? null : (
        <span className="runs-row-actions">
          <IconButton size="sm" icon="panelRight" label={`Open ${i.label} in the inspector`}
            onClick={() => openInspector({ kind: "job", id: i.source === "slurm" ? `slurm/${i.id}` : `local/${i.id}` })} />
          {i.cancellable && <IconButton size="sm" icon="stop" label={`Cancel ${i.label}`}
            onClick={() => { if (i.source === "local" && i.job) void askCancel(i.job); else void cancelSlurm(i); }} />}
        </span>
      ) },
  ], []);

  const slurmScope = scope !== "local";
  const statusKnown = !!fasrc.data;
  const offline = statusKnown && (feed.fasrcOffline || !fasrc.data?.ssh_connected);
  // Nothing is "empty" (or "offline") until the local feed and, for a SLURM
  // scope, the FASRC status and the SLURM feed have answered.
  const slurmReady = !slurmScope || (statusKnown && (offline || slurmAnswered));
  const settled = !feed.loading && slurmReady;
  return (
    <>
      {slurmScope && offline && (
        <Callout tone="warn" title="FASRC offline" action={<ConnectionBar />}>
          SLURM jobs appear here once the SSH session is up. Local jobs and the run history work offline.
        </Callout>
      )}
      {slurmScope && feed.slurmStale && <Callout tone="info">The login node was slow: the SLURM rows may be a poll old.</Callout>}
      {slurmScope && feed.slurmError && !feed.slurm.length && (
        <Callout tone="bad" title="Could not read the SLURM queue">{feed.slurmError.message}</Callout>
      )}
      {slurmScope && squeue.error && <Callout tone="warn" title="squeue failed: CLI-submitted jobs are not listed">{squeue.error.message}</Callout>}
      <div className="runs-split">
        <div className="runs-split__main">
          <DataTable rows={rows} columns={columns} rowKey={(i) => i.key} aria-label="Running and queued jobs"
            activeKey={selected?.key ?? null} urlKey="l" exportName="live-jobs" dense height="auto"
            onRowClick={(i) => setJob(i.source === "local" ? i.key : i.id)}
            loading={!settled && !rows.length}
            empty={settled ? (
              <EmptyState compact icon="activity" title={scope === "local" ? "No local job is running" : scope === "slurm" ? "No SLURM job is running or queued" : "Nothing is running"}
                action={<Button asChild size="sm"><Link to={pagePath("runs", { tab: "steps" })}>Run a FASRC step</Link></Button>}>
                Finished runs are in the history.
              </EmptyState>
            ) : <Skeleton lines={3} />} />
        </div>
        {(selected || slurmId || localId) && (
          <aside className="runs-split__side" aria-label={`Job ${selected?.label ?? job}`}>
            {selected?.cli ? <CliJobCard item={selected} />
              : localId ? <LocalJobCard key={localId} id={localId} stored={selected?.job} />
              : slurmId ? <SlurmMonitor key={slurmId} jobid={slurmId} /> : null}
          </aside>
        )}
      </div>
    </>
  );
}

export default function Live() {
  const live = useLiveItems();
  const { feed } = live;
  const [scope, setScope] = useUrlState<Scope>("scope", "all", { parse: parseScope });
  const counts = scopeCounts(live.all);
  return (
    <Page className="runs-page">
      <Toolbar label="Live jobs">
        <ToolbarGroup label="Scope" hideLabel>
          <Segmented<Scope> size="sm" value={scope} onChange={setScope} aria-label="Scope"
            options={SCOPES.map((s) => ({ value: s, label: counts[s] ? `${SCOPE_LABEL[s]} · ${counts[s]}` : SCOPE_LABEL[s] }))} />
        </ToolbarGroup>
        <ToolbarSpacer />
        <ConnectionBar />
        <IconButton size="sm" icon="reset" label="Refresh" onClick={feed.refresh} />
      </Toolbar>
      <LiveList live={live} scope={scope} />
      <QueuePanel />
    </Page>
  );
}
