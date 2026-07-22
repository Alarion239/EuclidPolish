/* Local background jobs (the `/api/jobs` registry: POST spawns → {job_id},
   poll GET /api/jobs/<id>). One hook + one component drive every job form in
   the SPA — the React-native replacement for the classic `polishUI.run`. */
import { useCallback, useEffect, useRef, useState } from "react";
import { postForm, type FormValue } from "./api";
import { getJSON } from "./api";
import { Badge } from "./ui";

export type JobProgress = {
  current: number;
  total: number;
  pct: number;
  label: string;
  stage_elapsed?: number | null;
  rate_per_second?: number | null;
  eta_seconds?: number | null;
  updated_ago_seconds?: number | null;
};

export type Job = {
  job_id: string;
  label: string;
  status: "running" | "done" | "failed" | string;
  duration: number;
  error: string | null;
  log: string;
  log_truncated: boolean;
  progress: JobProgress;
};

export type RunOpts = { onDone?: (job: Job) => void };

const TERMINAL = (s: string) => s !== "running";

const formatSpan = (seconds: number | null | undefined) => {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 10) return `${seconds.toFixed(1)}s`;
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60);
    return `${minutes}m ${String(Math.round(seconds % 60)).padStart(2, "0")}s`;
  }
  const hours = Math.floor(seconds / 3600);
  return `${hours}h ${String(Math.floor((seconds % 3600) / 60)).padStart(2, "0")}m`;
};

const formatRate = (rate: number | null | undefined, label: string) => {
  if (rate == null || !Number.isFinite(rate) || rate <= 0) return null;
  const value = rate < 0.01 ? rate.toFixed(3) : rate < 1 ? rate.toFixed(2) : rate.toFixed(1);
  return `${value} ${/field/i.test(label) ? "fields" : "items"}/s`;
};

/** Spawn a local job by POSTing a form, then poll `/api/jobs/<id>` with 0.5→2s
 *  backoff, exposing the live Job. `run` resolves the job_id from the POST
 *  response; a non-job JSON response calls `onDone` immediately. */
export function useJob() {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const cancelled = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => () => { cancelled.current = true; clearTimeout(timer.current); }, []);

  const poll = useCallback((id: string, onDone?: (j: Job) => void) => {
    let interval = 500;
    const tick = async () => {
      if (cancelled.current) return;
      const j = await getJSON<Job>(`/api/jobs/${id}`);
      if (cancelled.current) return;
      if (!j) {
        setError("job is no longer available — the local server may have restarted; run it again");
        setBusy(false);
        return;
      }
      setJob(j);
      if (TERMINAL(j.status)) { setBusy(false); onDone?.(j); return; }
      interval = Math.min(2000, interval * 1.4);
      timer.current = setTimeout(tick, interval);
    };
    tick();
  }, []);

  const run = useCallback(
    async (url: string, data?: Record<string, FormValue> | FormData, opts?: RunOpts) => {
      setError(null);
      setJob(null);
      setBusy(true);
      let res: { job_id?: string; error?: string } & Record<string, unknown>;
      try {
        res = await postForm(url, data);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setBusy(false);
        return;
      }
      if (res?.error) { setError(res.error); setBusy(false); return; }
      if (res?.job_id) { poll(res.job_id, opts?.onDone); return; }
      // Non-job response (e.g. a synchronous {ok:true}) — done immediately.
      setBusy(false);
      opts?.onDone?.(res as unknown as Job);
    },
    [poll],
  );

  const reset = useCallback(() => { setJob(null); setError(null); setBusy(false); }, []);
  return { job, error, busy, run, reset };
}

/** Discover a matching running job even when it was submitted from another
 *  tab or before this page loaded. Once attached, keep its terminal snapshot. */
export function useTrackedJob(labelNeedle: string) {
  const [job, setJob] = useState<Job | null>(null);
  const trackedId = useRef<string | null>(null);

  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    trackedId.current = null;
    setJob(null);
    const tick = async () => {
      if (!active) return;
      let candidate: Job | null = null;
      if (trackedId.current) {
        candidate = await getJSON<Job>(`/api/jobs/${trackedId.current}`);
      } else {
        const jobs = await getJSON<Job[]>("/api/jobs");
        candidate = jobs?.find((item) =>
          item.status === "running" && item.label.includes(labelNeedle)) ?? null;
        if (candidate) trackedId.current = candidate.job_id;
      }
      if (!active) return;
      if (candidate) setJob(candidate);
      if (!candidate || !TERMINAL(candidate.status)) {
        timer = setTimeout(tick, 2000);
      }
    };
    tick();
    return () => { active = false; clearTimeout(timer); };
  }, [labelNeedle]);

  return job;
}

/** Live progress panel for a Job (status badge, progress bar, error, log tail).
 *  Renders nothing until a job is spawned or an error occurs. */
export function JobProgressView({ job, error }: { job: Job | null; error?: string | null }) {
  if (error) return <div className="job-panel job-panel--err"><pre>{error}</pre></div>;
  if (!job) return null;
  const p = job.progress;
  const tone = job.status === "done" ? "good" : job.status === "failed" ? "bad" : undefined;
  const determinate = p && p.total > 0;
  const rate = formatRate(p?.rate_per_second, p?.label ?? "");
  const eta = p?.eta_seconds == null ? null : formatSpan(p.eta_seconds);
  const calibrating = job.status === "running" && determinate
    && p.current < p.total && eta == null;
  return (
    <div className={`job-panel job-panel--${job.status}`}>
      <div className="job-panel__head">
        <Badge tone={tone}>{job.status}</Badge>
        <span className="job-panel__label">{job.label}</span>
        <span className="job-panel__dur mono">elapsed {formatSpan(job.duration)}</span>
      </div>
      {determinate ? (
        <>
          <div className="job-panel__stage">
            <span>{p.label || "working"}</span>
            <span className="mono">{p.current.toLocaleString()}/{p.total.toLocaleString()} · {p.pct.toFixed(0)}%</span>
          </div>
          <div className="job-panel__bar">
            <div className="job-bar">
              <div className="job-bar__fill" style={{ width: `${p.pct}%` }} />
            </div>
          </div>
          {job.status === "running" && (
            <div className="job-panel__telemetry mono" aria-live="polite">
              <span>stage {formatSpan(p.stage_elapsed)}</span>
              <span>{rate ?? "measuring throughput…"}</span>
              <span className="job-panel__eta">
                {calibrating ? "ETA calibrating…" : `stage ETA ${eta ?? "—"}`}
              </span>
              {(p.updated_ago_seconds ?? 0) >= 4
                && <span>updated {formatSpan(p.updated_ago_seconds)} ago</span>}
            </div>
          )}
        </>
      ) : job.status === "running" ? (
        <div className="job-panel__bar">
          <div className="job-bar job-bar--indeterminate"><div className="job-bar__fill" /></div>
        </div>
      ) : null}
      {job.error && <pre className="job-panel__err">{job.error}</pre>}
      {job.log && (
        <details open={job.status !== "running"}>
          <summary>log{job.log_truncated ? " (tail)" : ""}</summary>
          <pre className="job-panel__log">{job.log}</pre>
        </details>
      )}
    </div>
  );
}
