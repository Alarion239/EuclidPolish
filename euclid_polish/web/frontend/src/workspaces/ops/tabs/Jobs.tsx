/* ops/jobs (spec §8.7): the local job centre — every background job of this
 * server (C2: newest 200 finished are kept), filtered by status, with the
 * selected job's live progress, full searchable log, cancel and its JSON
 * result. URL: `status`, `job`, and the table's `j.q` / `j.sort`. */
import { useMemo, useState } from "react";
import { cancelJob, useJobsFeed, type Job } from "../../../api/jobs";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { orderJobs } from "../../../app/JobTray";
import { usePageActions } from "../../../app/palette";
import { formatDateTime, formatDuration, formatRelative } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Badge, Button, Callout, Card, CardBody, CardHead, DataTable, DefList, EmptyState, IconButton, JobProgress,
  JsonTree, LogView, Page, PageHead, ProgressBar, Section, Segmented, Skeleton, confirm, toast, type DataColumn,
} from "../../../ui";
import { localJobTone } from "../model";
import "../ops.css";

const STATUSES = ["all", "running", "failed", "done", "cancelled"] as const;
type StatusFilter = typeof STATUSES[number];
const parseStatus = (raw: string): StatusFilter | undefined =>
  (STATUSES as readonly string[]).includes(raw) ? raw as StatusFilter : undefined;

async function askCancel(job: Job): Promise<void> {
  if (!(await confirm({ title: `Cancel “${job.label}”?`, message: "The job stops at its next progress tick; partial output stays.",
    tone: "danger", confirmLabel: "Cancel job" }))) return;
  const r = await cancelJob(job.job_id);
  if (r.ok) toast.info("Cancel requested"); else toast.error(r.error ?? "cancel refused");
}

function JobDetail({ id, stored }: { id: string; stored: Job | undefined }) {
  const [poll, setPoll] = useState(() => !stored || stored.status === "running");
  const detail = useResource<Job>(`/api/jobs/${encodeURIComponent(id)}`, [id], { ttl: poll ? 1_000 : 60_000, poll: poll ? 2_000 : undefined });
  const job = detail.data ?? stored ?? null;
  const running = (detail.data?.status ?? stored?.status) === "running";
  if (running !== poll && !(detail.error?.status === 404)) setPoll(running);
  if (!job) {
    return detail.loading ? <Skeleton lines={6} /> : (
      <Callout tone="warn" title="Job not found">
        {detail.error?.status === 404 ? "The server no longer knows this job (restarted or evicted)." : detail.error?.message}
      </Callout>
    );
  }
  return (
    <Card className="ops-jobdetail">
      <CardHead title={job.label} sub={<code className="mono">{job.job_id}{job.kind ? ` · ${job.kind}` : ""}</code>}
        right={<div className="ops-row">
          <Badge tone={localJobTone(job.status)} dot={job.status === "running"}>{job.status}</Badge>
          <IconButton size="sm" icon="panelRight" label="Open in the inspector" onClick={() => openInspector({ kind: "job", id: `local/${job.job_id}` })} />
          {job.status === "running" && job.cancellable !== false && !job.cancel_requested && (
            <Button size="sm" variant="ghost" icon="stop" onClick={() => void askCancel(job)}>Cancel</Button>
          )}
        </div>} />
      <CardBody>
        <DefList dense items={[
          ["started", job.started ? formatDateTime(job.started) : "—"],
          job.finished ? ["finished", formatDateTime(job.finished)] : null,
          ["duration", formatDuration(job.duration)],
        ]} />
        <JobProgress job={{ ...job, log: null }} cancel={false} />
        <Section title="Log" sub={job.log_truncated ? "tail" : undefined}>
          <LogView text={job.log ?? ""} title={job.label} exportName={`job-${job.job_id}`} maxHeight="min(56vh, 620px)" />
        </Section>
        {job.result != null && (
          <Section title="Result" collapsible defaultOpen>
            <JsonTree data={job.result} expandDepth={1} />
          </Section>
        )}
      </CardBody>
    </Card>
  );
}

export default function Jobs() {
  const feed = useJobsFeed({ slurm: false });
  const [status, setStatus] = useUrlState<StatusFilter>("status", "all", { parse: parseStatus });
  const [selected, setSelected] = useUrlState("job", "");
  const counts = useMemo(() => {
    const c: Record<string, number> = { all: feed.jobs.length };
    for (const j of feed.jobs) c[j.status] = (c[j.status] ?? 0) + 1;
    return c;
  }, [feed.jobs]);
  // Running jobs first, then newest first (the table's own sort can override).
  const rows = useMemo(() => orderJobs(status === "all" ? feed.jobs : feed.jobs.filter((j) => j.status === status)),
    [feed.jobs, status]);
  const current = feed.jobs.find((j) => j.job_id === selected);

  usePageActions([
    { id: "jobs-refresh", label: "Refresh local jobs", group: "Jobs", run: feed.refresh },
    { id: "jobs-running", label: "Show running jobs", group: "Jobs", run: () => setStatus("running") },
    { id: "jobs-failed", label: "Show failed jobs", group: "Jobs", run: () => setStatus("failed") },
    { id: "jobs-cancel", label: "Cancel the selected job", group: "Jobs",
      disabled: !current || current.status !== "running", run: () => { if (current) void askCancel(current); } },
  ]);

  const columns = useMemo<DataColumn<Job>[]>(() => [
    { id: "status", header: "Status", width: 100,
      cell: (j) => <Badge size="sm" tone={localJobTone(j.status)} dot={j.status === "running"}>{j.cancel_requested && j.status === "running" ? "cancelling" : j.status}</Badge> },
    { id: "label", header: "Job", cell: (j) => (
      <span className="ops-cell2"><span>{j.label}</span>{j.kind && <span className="ops-dim mono ops-small">{j.kind}</span>}</span>
    ) },
    { id: "progress", header: "Progress", width: 150, accessor: (j) => j.progress?.pct ?? null, sortable: true,
      cell: (j) => (j.status === "running" && j.progress && j.progress.total > 0
        ? <ProgressBar value={j.progress.current} max={j.progress.total} label={`${Math.round(j.progress.pct)}%`} />
        : j.error ? <span className="ops-bad ops-small ops-ellipsis" title={j.error}>{j.error.split("\n")[0]}</span>
        : <span className="ops-dim ops-small">{j.progress?.label || "—"}</span>) },
    { id: "started", header: "Started", width: 104, accessor: (j) => j.started ?? 0,
      cell: (j) => <span className="ops-dim ops-small" title={formatDateTime(j.started)}>{formatRelative(j.started)}</span> },
    { id: "duration", header: "Duration", numeric: true, width: 90, cell: (j) => formatDuration(j.duration) },
    { id: "kind", header: "Kind", hidden: true },
    { id: "job_id", header: "Id", hidden: true, cell: (j) => <code className="mono">{j.job_id}</code> },
  ], []);

  return (
    <Page className="ops-page">
      <PageHead eyebrow="ops · jobs" title="Local jobs" />
      <div className="ops-bar" role="toolbar" aria-label="Job filters">
        <Segmented<StatusFilter> value={status} onChange={setStatus} aria-label="Status"
          options={STATUSES.map((s) => ({ value: s, label: `${s}${counts[s] ? ` · ${counts[s]}` : ""}` }))} />
        <span className="ops-spacer" />
        {feed.running.length > 0 && <Badge tone="info" dot>{feed.running.length} running</Badge>}
        <IconButton size="sm" icon="reset" label="Refresh" onClick={feed.refresh} />
      </div>
      {feed.error && !feed.jobs.length && <Callout tone="bad" title="Could not list the jobs">{feed.error.message}</Callout>}
      <div className="ops-split">
        <div className="ops-split__main">
          <DataTable rows={rows} columns={columns} rowKey={(j) => j.job_id} aria-label="Local jobs" urlKey="j"
            activeKey={selected || null} onRowClick={(j) => setSelected(j.job_id)} loading={feed.loading}
            height={640} exportName="local-jobs"
            empty={<EmptyState compact icon="activity" title={status === "all" ? "No local jobs since the server started" : `No ${status} jobs`} />} />
        </div>
        {selected && (
          <aside className="ops-split__side" aria-label="Selected job">
            <JobDetail key={selected} id={selected} stored={current} />
          </aside>
        )}
      </div>
    </Page>
  );
}
