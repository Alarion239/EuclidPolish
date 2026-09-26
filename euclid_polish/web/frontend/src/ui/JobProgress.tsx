/* JobProgress: the live panel for one local job (status badge, stage,
   progress bar, telemetry, error, log). Running cancellable jobs get a Cancel
   button (C2 cooperative cancel). `JobProgressView` is the compat name. */
import { useState } from "react";
import { cancelJob, type Job } from "../api/jobs";
import { formatCount, formatDuration } from "../format";
import { Button } from "./Button";
import { Badge, ProgressBar } from "./display";
import { LogTail } from "./LogView";

const formatRate = (rate: number | null | undefined, label: string) => {
  if (rate == null || !Number.isFinite(rate) || rate <= 0) return null;
  const value = rate < 0.01 ? rate.toFixed(3) : rate < 1 ? rate.toFixed(2) : rate.toFixed(1);
  return `${value} ${/field/i.test(label) ? "fields" : "items"}/s`;
};

/** Renders nothing until a job is spawned or an error occurs. `onCancel`
 *  overrides the default `cancelJob(job.job_id)`; `cancel={false}` hides it. */
export function JobProgress(
  { job, error, cancel = true, onCancel }: {
    job: Job | null; error?: string | null; cancel?: boolean; onCancel?: () => void | Promise<unknown>;
  },
) {
  const [cancelling, setCancelling] = useState(false);
  const [cancelError, setCancelError] = useState<string | null>(null);
  if (error) return <div className="job-panel job-panel--err" role="alert"><pre className="job-panel__err">{error}</pre></div>;
  if (!job) return null;
  const p = job.progress;
  const tone = job.status === "done" ? "good" : job.status === "failed" ? "bad"
    : job.status === "cancelled" ? "warn" : undefined;
  const determinate = !!p && p.total > 0;
  const rate = formatRate(p?.rate_per_second, p?.label ?? "");
  const eta = p?.eta_seconds == null ? null : formatDuration(p.eta_seconds);
  const calibrating = job.status === "running" && determinate && p!.current < p!.total && eta == null;
  const requested = job.status === "running" && (job.cancel_requested || cancelling);
  const canCancel = cancel && job.status === "running" && job.cancellable !== false && !job.cancel_requested;
  const doCancel = async () => {
    setCancelling(true);
    setCancelError(null);
    try {
      if (onCancel) await onCancel();
      else {
        const r = await cancelJob(job.job_id);
        if (!r.ok) { setCancelError(r.error ?? "cancel refused"); setCancelling(false); }
      }
    } catch (e) {
      setCancelError(e instanceof Error ? e.message : String(e));
      setCancelling(false);
    }
  };
  return (
    <div className={`job-panel job-panel--${job.status}`} aria-busy={job.status === "running" || undefined}>
      <div className="job-panel__head">
        <Badge tone={tone}>{requested ? "cancelling…" : job.status}</Badge>
        <span className="job-panel__label">{job.label}</span>
        <span className="job-panel__dur mono">elapsed {formatDuration(job.duration)}</span>
        {canCancel && (
          <Button size="sm" variant="ghost" loading={cancelling} onClick={doCancel}>Cancel</Button>
        )}
      </div>
      {determinate ? (
        <>
          <div className="job-panel__stage">
            <span>{p!.label || "working"}</span>
            <span className="mono">{formatCount(p!.current)}/{formatCount(p!.total)} · {p!.pct.toFixed(0)}%</span>
          </div>
          <ProgressBar value={p!.pct} max={100} tone={tone} aria-label={`${job.label} progress`} />
          {job.status === "running" && (
            <div className="job-panel__telemetry mono" aria-live="polite">
              <span>stage {formatDuration(p!.stage_elapsed)}</span>
              <span>{rate ?? "measuring throughput…"}</span>
              <span className="job-panel__eta">
                {calibrating ? "ETA calibrating…" : `stage ETA ${eta ?? "—"}`}
              </span>
              {(p!.updated_ago_seconds ?? 0) >= 4
                && <span>updated {formatDuration(p!.updated_ago_seconds)} ago</span>}
            </div>
          )}
        </>
      ) : job.status === "running" ? (
        <ProgressBar value={null} aria-label={`${job.label} running`} />
      ) : null}
      {cancelError && <pre className="job-panel__err">cancel failed: {cancelError}</pre>}
      {job.error && <pre className="job-panel__err">{job.error}</pre>}
      {job.log && (
        <details open={job.status !== "running"}>
          <summary>log{job.log_truncated ? " (tail)" : ""}</summary>
          <LogTail text={job.log} className="job-panel__log" />
        </details>
      )}
    </div>
  );
}

/** Compat name used by the pre-rework pages. */
export const JobProgressView = JobProgress;
