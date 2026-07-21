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

/** Live progress panel for a Job (status badge, progress bar, error, log tail).
 *  Renders nothing until a job is spawned or an error occurs. */
export function JobProgressView({ job, error }: { job: Job | null; error?: string | null }) {
  if (error) return <div className="job-panel job-panel--err"><pre>{error}</pre></div>;
  if (!job) return null;
  const p = job.progress;
  const tone = job.status === "done" ? "good" : job.status === "failed" ? "bad" : undefined;
  const determinate = p && p.total > 0;
  return (
    <div className={`job-panel job-panel--${job.status}`}>
      <div className="job-panel__head">
        <Badge tone={tone}>{job.status}</Badge>
        <span className="job-panel__label">{job.label}</span>
        <span className="job-panel__dur mono">{job.duration.toFixed(1)}s</span>
      </div>
      <div className="job-panel__bar">
        {determinate ? (
          <>
            <div className="job-bar">
              <div className="job-bar__fill" style={{ width: `${p.pct}%` }} />
            </div>
            <span className="job-panel__pct mono">
              {p.label ? `${p.label} ` : ""}{p.current}/{p.total} ({p.pct.toFixed(0)}%)
            </span>
          </>
        ) : job.status === "running" ? (
          <div className="job-bar job-bar--indeterminate"><div className="job-bar__fill" /></div>
        ) : null}
      </div>
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
