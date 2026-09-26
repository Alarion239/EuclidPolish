/* Job tray (spec §4): the top-bar button with the running count and a popover
 * listing local jobs (`/api/jobs?summary=1`) and live SLURM jobs
 * (`/api/fasrc/current-submission`) from the ONE shared jobs feed
 * (`useJobsFeed`). Rows show progress and ETA, cancel (SLURM cancels ask
 * first), and open the full log in the inspector (`job:local/<id>`,
 * `job:slurm/<jobid>`). FASRC offline (the C4 503) is shown as "offline",
 * never as an error. `useJobToasts` (mounted by the shell) toasts every job
 * this session saw running when it finishes, fails or is cancelled.
 */
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  cancelJob, cancelSlurmJob, isTerminal, refreshJobsFeed, useJobsFeed, useJobsStore,
  type Job, type SlurmJob,
} from "../api/jobs";
import { formatDuration, formatRelative } from "../format";
import {
  Badge, Button, Icon, IconButton, Popover, ProgressBar, confirm, toast, type Tone,
} from "../ui";
import { openInspector } from "./inspector";
import { useShellUi } from "./shellStore";

const STATUS_TONE: Record<string, Tone> = {
  running: "info", done: "good", failed: "bad", cancelled: "warn",
};

const SLURM_TONE: Record<string, Tone> = {
  RUNNING: "info", PENDING: "warn", COMPLETED: "good", FAILED: "bad", CANCELLED: "warn", TIMEOUT: "bad",
};

/** Seconds since the epoch → a relative time ("3 min ago"). */
const ago = (epochSeconds: number | null | undefined) =>
  (epochSeconds ? formatRelative(epochSeconds * 1000) : "—");

export function JobRow({ job, onOpen }: { job: Job; onOpen?: () => void }) {
  const [busy, setBusy] = useState(false);
  const running = job.status === "running";
  const p = job.progress;
  const determinate = !!p && p.total > 0;
  const canCancel = running && job.cancellable !== false && !job.cancel_requested;
  const cancel = async () => {
    setBusy(true);
    const r = await cancelJob(job.job_id);
    setBusy(false);
    if (!r.ok) toast.error(`Could not cancel “${job.label}”`, { description: r.error ?? "refused" });
  };
  const open = () => {
    openInspector({ kind: "job", id: `local/${job.job_id}` });
    onOpen?.();
  };
  return (
    <li className="jobrow" data-status={job.status}>
      <div className="jobrow__head">
        <Badge tone={STATUS_TONE[job.status]} dot size="sm">
          {running && job.cancel_requested ? "cancelling…" : job.status}
        </Badge>
        <button type="button" className="jobrow__label" onClick={open} title="Open the log in the inspector">
          {job.label}
        </button>
        {canCancel && (
          <IconButton icon="stop" size="sm" label={`Cancel ${job.label}`} loading={busy} onClick={cancel} />
        )}
      </div>
      <div className="jobrow__meta mono">
        {job.kind && <span>{job.kind}</span>}
        <span>{running ? `started ${ago(job.started)}` : `${job.status} ${ago(job.finished ?? job.started)}`}</span>
        <span>{formatDuration(job.duration)}</span>
        {running && p?.eta_seconds != null && <span>ETA {formatDuration(p.eta_seconds)}</span>}
      </div>
      {running && (
        <ProgressBar value={determinate ? p.pct : null} max={100}
          aria-label={`${job.label} progress`}
          label={determinate ? `${p.label ? `${p.label} · ` : ""}${p.current}/${p.total}` : undefined} />
      )}
      {job.status === "failed" && job.error && (
        <div className="jobrow__err mono">{job.error.split("\n")[0]}</div>
      )}
    </li>
  );
}

export function SlurmRow({ job, onOpen }: { job: SlurmJob; onOpen?: () => void }) {
  const [busy, setBusy] = useState(false);
  const live = job.state === "RUNNING" || job.state === "PENDING";
  const total = Number(job.progress_total ?? 0);
  const step = Number(job.progress_step ?? 0);
  const cancel = async () => {
    const ok = await confirm({
      title: `Cancel SLURM job ${job.jobid}?`,
      message: `${job.label ?? job.step_id ?? "This job"} will be stopped with scancel on FASRC.`,
      tone: "danger", confirmLabel: "Cancel job", cancelLabel: "Keep running",
    });
    if (!ok) return;
    setBusy(true);
    const r = await cancelSlurmJob(job.jobid);
    setBusy(false);
    if (!r.ok) toast.error(`Could not cancel SLURM job ${job.jobid}`, { description: r.error ?? "refused" });
    else toast.info(`Cancel sent to SLURM job ${job.jobid}`);
  };
  const open = () => {
    openInspector({ kind: "job", id: `slurm/${job.jobid}` });
    onOpen?.();
  };
  return (
    <li className="jobrow" data-status={job.state}>
      <div className="jobrow__head">
        <Badge tone={SLURM_TONE[job.state]} dot size="sm">{job.state}</Badge>
        <button type="button" className="jobrow__label" onClick={open} title="Open in the inspector">
          {job.label ?? job.step_id ?? `job ${job.jobid}`}
        </button>
        {live && <IconButton icon="stop" size="sm" label={`Cancel SLURM job ${job.jobid}`} loading={busy} onClick={cancel} />}
      </div>
      <div className="jobrow__meta mono">
        <span>#{job.jobid}</span>
        {job.step_id && <span>{job.step_id}</span>}
        {job.time && <span>{job.time}{job.time_limit ? ` / ${job.time_limit}` : ""}</span>}
        {job.state === "PENDING" && job.reason && <span>{job.reason}</span>}
      </div>
      {job.state === "RUNNING" && total > 0 && (
        <ProgressBar value={(100 * step) / total} max={100} label={`${step}/${total}`}
          aria-label={`SLURM job ${job.jobid} progress`} />
      )}
    </li>
  );
}

/** Local jobs, running first then the newest finished (`limit` rows). */
export function orderJobs(jobs: Job[], limit = Infinity): Job[] {
  const running = jobs.filter((j) => j.status === "running");
  const done = jobs.filter((j) => j.status !== "running");
  return [...running, ...done].slice(0, limit);
}

export function JobList({ jobs, limit, onOpen, empty }: {
  jobs: Job[]; limit?: number; onOpen?: () => void; empty?: string;
}) {
  const shown = orderJobs(jobs, limit);
  if (!shown.length) return <p className="jobtray__empty">{empty ?? "No local jobs yet."}</p>;
  return <ul className="joblist">{shown.map((j) => <JobRow key={j.job_id} job={j} onOpen={onOpen} />)}</ul>;
}

export function JobTrayPanel({ onClose }: { onClose?: () => void }) {
  const feed = useJobsFeed();
  const LIMIT = 8;
  const hidden = Math.max(0, feed.jobs.length - LIMIT);
  return (
    <div className="jobtray" aria-label="Jobs">
      <section className="jobtray__sec">
        <header className="jobtray__head">
          <h3>Local jobs</h3>
          {feed.running.length > 0 && <Badge tone="info" size="sm">{feed.running.length} running</Badge>}
          <span className="jobtray__spacer" />
          <IconButton icon="reset" size="sm" label="Refresh jobs" onClick={() => { void refreshJobsFeed(); }} />
        </header>
        {feed.error && !feed.jobs.length
          ? <p className="jobtray__empty">Could not list jobs: {feed.error.message}</p>
          : <JobList jobs={feed.jobs} limit={LIMIT} onOpen={onClose} />}
        {hidden > 0 && <p className="jobtray__more">{hidden} older job{hidden === 1 ? "" : "s"} in Ops › Jobs</p>}
      </section>
      <section className="jobtray__sec">
        <header className="jobtray__head">
          <h3>SLURM</h3>
          {feed.fasrcOffline
            ? <Badge size="sm" tone="neutral" dot>FASRC offline</Badge>
            : feed.slurmStale ? <Badge size="sm" tone="warn">stale</Badge> : null}
        </header>
        {feed.fasrcOffline ? (
          <p className="jobtray__empty">Connect to FASRC to see cluster jobs.
            {" "}<Link to="/settings/connections" onClick={onClose}>Connections</Link></p>
        ) : feed.slurmError ? (
          <p className="jobtray__empty">Could not read SLURM jobs: {feed.slurmError.message}</p>
        ) : feed.slurm.length ? (
          <ul className="joblist">{feed.slurm.map((j) => <SlurmRow key={j.jobid} job={j} onOpen={onClose} />)}</ul>
        ) : <p className="jobtray__empty">No live SLURM jobs.</p>}
      </section>
      <footer className="jobtray__foot">
        <Button asChild size="sm" variant="ghost"><Link to="/ops/jobs" onClick={onClose}>All jobs</Link></Button>
        <Button asChild size="sm" variant="ghost"><Link to="/ops/fasrc" onClick={onClose}>FASRC console</Link></Button>
      </footer>
    </div>
  );
}

/** The top-bar button (running count) + its popover. */
export function JobTray() {
  const open = useShellUi((s) => s.tray);
  const setOpen = (v: boolean) => useShellUi.getState().setOpen("tray", v);
  const feed = useJobsFeed();
  const n = feed.runningCount;
  return (
    <Popover open={open} onOpenChange={setOpen} align="end" label="Jobs" width={400}
      className="jobtray__pop"
      trigger={(
        <button type="button" className="topbar__btn" data-active={n > 0 || undefined}
          aria-label={n ? `Jobs: ${n} running` : "Jobs"} title="Jobs">
          <Icon name="activity" />
          {n > 0 && <span className="topbar__count" aria-hidden="true">{n}</span>}
        </button>
      )}>
      <JobTrayPanel onClose={() => setOpen(false)} />
    </Popover>
  );
}

/** Toast every local job this session saw running when it ends. */
export function useJobToasts(): void {
  const jobs = useJobsStore((s) => s.jobs);
  const seen = useRef<Record<string, string>>({});
  useEffect(() => {
    const before = seen.current;
    const now: Record<string, string> = {};
    for (const j of Object.values(jobs)) {
      now[j.job_id] = j.status;
      if (before[j.job_id] !== "running" || !isTerminal(j.status)) continue;
      const action = {
        label: "Log",
        onClick: () => openInspector({ kind: "job", id: `local/${j.job_id}` }),
      };
      if (j.status === "done") {
        toast.success(j.label, { description: `finished in ${formatDuration(j.duration)}`, action });
      } else if (j.status === "failed") {
        toast.error(j.label, { description: (j.error ?? "failed").split("\n")[0], action, duration: 10_000 });
      } else {
        toast.warning(`${j.label} cancelled`, { action });
      }
    }
    seen.current = now;
  }, [jobs]);
}
