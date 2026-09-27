/* Ops › FASRC › Live: every PENDING/RUNNING SLURM job of $USER — the
 * console's submissions (`live` of the shared jobs feed, C5) plus, read-only
 * and marked CLI, the squeue jobs the local job DB does not know (runs
 * submitted from a shell) — with the selected one's live monitor.
 * `?job=` selects. */
import { useMemo } from "react";
import { cancelSlurmJob, useJobsFeed } from "../../../api/jobs";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { formatRelative } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Badge, Button, Callout, Card, CardBody, CardHead, DataTable, DefList, EmptyState, IconButton, Tooltip, confirm, toast,
  type DataColumn,
} from "../../../ui";
import { jobStateTone, mergeCliJobs, type LiveRow, type SqueueRow } from "../model";
import { SlurmMonitor } from "../steps/SlurmMonitor";
import { ConnectionBar } from "./Connection";
import "../ops.css";

export function LiveJobs({ onView }: { onView?: (view: string) => void }) {
  const feed = useJobsFeed();
  const [job, setJob] = useUrlState("job", "");
  const squeue = useResource<{ ok?: boolean; rows?: SqueueRow[] }>(feed.fasrcOffline ? null : "/api/fasrc/queue", [],
    { ttl: 10_000, poll: 20_000 });
  const rows = useMemo(() => mergeCliJobs(feed.slurm, squeue.data?.rows), [feed.slurm, squeue.data]);
  const selected = job || rows[0]?.jobid || "";
  const selectedRow = rows.find((r) => r.jobid === selected);

  async function cancel(j: LiveRow) {
    if (!(await confirm({ title: `Cancel SLURM job ${j.jobid}?`, message: `${j.label ?? j.step_id ?? ""} — scancel on FASRC.`,
      tone: "danger", confirmLabel: "Cancel job" }))) return;
    const r = await cancelSlurmJob(j.jobid);
    if (r.ok) toast.success(`Cancelled ${j.jobid}`); else toast.error(r.error ?? "cancel refused");
  }

  const columns = useMemo<DataColumn<LiveRow>[]>(() => [
    { id: "state", header: "State", width: 104,
      cell: (j) => <Badge size="sm" tone={jobStateTone(j.state)} dot={j.state === "RUNNING"}>{j.state}</Badge> },
    { id: "jobid", header: "Job", width: 110, cell: (j) => <code className="mono">{j.jobid}</code> },
    { id: "label", header: "Run", accessor: (j) => j.label ?? j.step_id ?? "",
      cell: (j) => (
        <span className="ops-cell2">
          <span>{j.label ?? "—"}</span>
          {j.cli ? (
            <Tooltip content="Submitted outside the console (squeue only): no event stream, read-only here.">
              <span><Badge size="sm" tone="neutral">CLI</Badge></span>
            </Tooltip>
          ) : <span className="ops-dim mono ops-small">{j.step_id ?? ""}</span>}
        </span>
      ) },
    { id: "source", header: "Source", hidden: true, accessor: (j) => (j.cli ? "cli" : "console") },
    { id: "time", header: "Elapsed / limit", width: 132, accessor: (j) => j.time ?? "",
      cell: (j) => <span className="mono ops-small">{j.time || "—"}{j.time_limit ? ` / ${j.time_limit}` : ""}</span> },
    { id: "where", header: "Node / reason", accessor: (j) => (j.state === "PENDING" ? j.reason : j.nodes) ?? "",
      cell: (j) => <span className="mono ops-small">{(j.state === "PENDING" ? j.reason : j.nodes) || "—"}</span> },
    { id: "submitted_at", header: "Submitted", width: 100, accessor: (j) => j.submitted_at ?? 0,
      cell: (j) => <span className="ops-dim ops-small">{j.submitted_at ? formatRelative(j.submitted_at) : "—"}</span> },
    { id: "actions", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 72,
      cell: (j) => j.cli ? null : (
        <span className="ops-row-actions">
          <IconButton size="sm" icon="panelRight" label={`Inspect ${j.jobid}`}
            onClick={() => openInspector({ kind: "job", id: `slurm/${j.jobid}` })} />
          <IconButton size="sm" icon="stop" label={`Cancel ${j.jobid}`} onClick={() => void cancel(j)} />
        </span>
      ) },
  ], []);

  if (feed.fasrcOffline) {
    return (
      <Callout tone="warn" title="FASRC offline" action={<ConnectionBar />}>
        Live SLURM jobs appear once the SSH session is up. The run history works offline.
      </Callout>
    );
  }
  return (
    <div className="ops-split">
      <div className="ops-split__main">
        {feed.slurmStale && <Callout tone="info">The login node was slow: rows may be a poll old.</Callout>}
        {feed.slurmError && !rows.length && <Callout tone="bad" title="Could not read the SLURM queue">{feed.slurmError.message}</Callout>}
        {squeue.error && <Callout tone="warn" title="squeue failed: CLI-submitted jobs are not listed">{squeue.error.message}</Callout>}
        <DataTable rows={rows} columns={columns} rowKey={(j) => j.jobid} aria-label="Live SLURM jobs"
          activeKey={selected || null} onRowClick={(j) => setJob(j.jobid)} dense height="auto"
          loading={feed.loading && !rows.length} exportName="slurm-live"
          empty={<EmptyState compact icon="server" title="No SLURM job is running"
            action={onView && <Button size="sm" onClick={() => onView("steps")}>Run a step</Button>}>
            Finished runs are in the history.
          </EmptyState>} />
      </div>
      {selected && (
        <aside className="ops-split__side" aria-label={`Job ${selected}`}>
          {selectedRow?.cli ? <CliJobCard job={selectedRow} /> : <SlurmMonitor key={selected} jobid={selected} />}
        </aside>
      )}
    </div>
  );
}

/** A CLI-submitted job: what squeue says (the console has no event stream). */
function CliJobCard({ job }: { job: LiveRow }) {
  return (
    <Card>
      <CardHead title={<span>Job <code className="mono">{job.jobid}</code></span>} sub="Submitted outside the console"
        right={<Badge size="sm" tone={jobStateTone(job.state)} dot={job.state === "RUNNING"}>{job.state}</Badge>} />
      <CardBody>
        <DefList dense items={[
          ["Name", job.label || "—"],
          ["Elapsed / limit", <span className="mono" key="t">{job.time || "—"}{job.time_limit ? ` / ${job.time_limit}` : ""}</span>],
          job.nodes ? ["Nodes", <span className="mono" key="n">{job.nodes}</span>] : null,
          job.reason ? ["Reason", <span className="mono" key="r">{job.reason}</span>] : null,
          job.start_time ? ["Start", <span className="mono" key="s">{job.start_time}</span>] : null,
        ]} />
      </CardBody>
    </Card>
  );
}
