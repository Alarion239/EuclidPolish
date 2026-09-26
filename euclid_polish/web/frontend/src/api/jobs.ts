/* Jobs data layer (contracts C2 + C8): local background jobs and SLURM jobs.
 *
 *   useJob(key?)       — compat `{job, error, busy, run(url, data, {onDone}),
 *                        reset}` (+ `cancel`). `run` POSTs a form; a
 *                        `{job_id}` reply is polled at /api/jobs/<id> with a
 *                        0.5 → 2 s backoff until it ends. Every started job
 *                        is registered in the global jobs store, so it stays
 *                        visible in the job tray after navigating away; with a
 *                        `key` the hook re-attaches to the same job on return.
 *   useJobsFeed()      — every local job (`/api/jobs?summary=1`) + live SLURM
 *                        jobs (`/api/fasrc/current-submission`). ONE shared
 *                        poll per source (TanStack Query dedupe): local 2 s
 *                        while anything runs / 15 s idle; SLURM 10 s while a
 *                        job is live / 30 s idle / 60 s while FASRC is
 *                        offline; both paused while the tab is hidden.
 *   useTrackedJob(s)   — re-attach to a running job by label substring,
 *                        discovered through the shared feed (no own poller
 *                        while idle).
 *   cancelJob(id)      — POST /api/jobs/<id>/cancel (cooperative, C2).
 *   cancelSlurmJob(id) — POST /api/fasrc/cancel.
 */
import { focusManager, useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { create } from "zustand";
import { ApiError, apiGet, apiPost, isAbortError, isFasrcOffline, type FormRecord } from "./client";
import { queryClient } from "./query";

/* ── types (C2) ──────────────────────────────────────────────────────────── */

export type JobStatus = "running" | "done" | "failed" | "cancelled";

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
  kind?: string | null;
  status: JobStatus | string;
  /** Epoch seconds. */
  started?: number;
  finished?: number | null;
  duration: number;
  error: string | null;
  /** Log tail; null in summary listings (`?summary=1`). */
  log: string | null;
  log_truncated: boolean;
  cancellable?: boolean;
  cancel_requested?: boolean;
  /** Small JSON-safe return value of a finished job (≤ 64 KB), else null. */
  result?: unknown;
  progress: JobProgress;
};

/** One SLURM job row (FASRC job DB row + live squeue fields). */
export type SlurmJob = {
  jobid: string;
  state: string;
  label?: string | null;
  step_id?: string | null;
  submitted_at?: number | null;
  started_at?: number | null;
  ended_at?: number | null;
  progress_step?: number | null;
  progress_total?: number | null;
  reason?: string | null;
  nodes?: string | null;
  time?: string | null;
  time_limit?: string | null;
  [key: string]: unknown;
};

export type SlurmQueue = {
  count: number;
  names?: string[];
  halted?: boolean;
  halted_reason?: string | null;
  [key: string]: unknown;
};

export type SlurmFeed = {
  /** FASRC not connected (the C4 gate) — `jobs` is then empty. */
  offline: boolean;
  /** The login node was slow; rows may be a tick old. */
  stale: boolean;
  jobs: SlurmJob[];
  queue: SlurmQueue | null;
  /** The raw `current` block (status/array/accounting of the newest job). */
  current: unknown;
};

export const LOCAL_FEED_URL = "/api/jobs?summary=1";
export const SLURM_FEED_URL = "/api/fasrc/current-submission";

/** Poll cadences in ms (mutable so tests can shorten them). */
export const JOBS_FEED_TIMING = {
  localActiveMs: 2_000,
  localIdleMs: 15_000,
  slurmActiveMs: 10_000,
  slurmIdleMs: 30_000,
  slurmOfflineMs: 60_000,
  /** /api/jobs/<id> detail poll: backoff from min to max. */
  detailMinMs: 500,
  detailMaxMs: 2_000,
};

export const isTerminal = (status: string): boolean => status !== "running";

const GONE = "job is no longer available — the local server may have restarted; run it again";

const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

/* ── global jobs store ───────────────────────────────────────────────────── */

type JobsState = {
  /** Latest known snapshot per job id (full or summary). */
  jobs: Record<string, Job>;
  /** Ids started from this SPA session, newest first. */
  started: string[];
  /** `useJob(key)` → the key's current job id. */
  keyed: Record<string, string>;
  /** Ids whose terminal snapshot came from the detail endpoint (full log). */
  final: Record<string, true>;
};

type JobsActions = {
  /** Merge one snapshot (a summary never erases a known log; a late
   *  "running" snapshot never regresses a finished job). */
  upsert: (job: Job, opts?: { detail?: boolean }) => void;
  /** Merge a `/api/jobs?summary=1` listing; jobs the server no longer lists
   *  are dropped unless they were started here and have finished. */
  mergeFeed: (list: Job[]) => void;
  register: (id: string, key?: string) => void;
  forgetKey: (key: string) => void;
  reset: () => void;
};

export type JobsStore = JobsState & JobsActions;

function mergeJob(prev: Job | undefined, next: Job): Job {
  if (!prev) return next;
  if (isTerminal(prev.status) && !isTerminal(next.status)) return prev;
  if (next.log != null) return next;
  return { ...next, log: prev.log, log_truncated: prev.log != null ? prev.log_truncated : next.log_truncated };
}

const EMPTY_STATE: JobsState = { jobs: {}, started: [], keyed: {}, final: {} };

export const useJobsStore = create<JobsStore>()((set, get) => ({
  ...EMPTY_STATE,
  upsert: (job, opts = {}) => {
    if (!job || typeof job.job_id !== "string") return;
    const { jobs, final } = get();
    const merged = mergeJob(jobs[job.job_id], job);
    const isFinal = !!opts.detail && isTerminal(merged.status) && merged === job;
    if (merged === jobs[job.job_id] && !isFinal) return;
    set({
      jobs: { ...jobs, [job.job_id]: merged },
      final: isFinal ? { ...final, [job.job_id]: true } : final,
    });
  },
  mergeFeed: (list) => {
    const { jobs, started } = get();
    const next: Record<string, Job> = {};
    for (const j of list) if (j && typeof j.job_id === "string") next[j.job_id] = mergeJob(jobs[j.job_id], j);
    // A finished job started here stays visible after the server evicts it; a
    // RUNNING one the server no longer lists is gone (server restart) — keeping
    // it would pin the feed at the fast cadence forever.
    for (const id of started) if (!next[id] && jobs[id] && isTerminal(jobs[id].status)) next[id] = jobs[id];
    set({ jobs: next });
  },
  register: (id, key) => {
    const { started, keyed } = get();
    set({
      started: [id, ...started.filter((x) => x !== id)],
      keyed: key ? { ...keyed, [key]: id } : keyed,
    });
  },
  forgetKey: (key) => {
    const { keyed } = get();
    if (!(key in keyed)) return;
    const next = { ...keyed };
    delete next[key];
    set({ keyed: next });
  },
  reset: () => set({ ...EMPTY_STATE }),
}));

/* ── feed ────────────────────────────────────────────────────────────────── */

export const JOBS_FEED_KEY = ["jobs-feed"] as const;
const LOCAL_KEY = [...JOBS_FEED_KEY, "local"] as const;
const SLURM_KEY = [...JOBS_FEED_KEY, "slurm"] as const;

async function fetchLocal({ signal }: { signal: AbortSignal }): Promise<Job[]> {
  const list = await apiGet<Job[]>(LOCAL_FEED_URL, { signal });
  const jobs = Array.isArray(list) ? list : [];
  useJobsStore.getState().mergeFeed(jobs);
  return jobs;
}

function anyRunningLocal(list: Job[] | undefined): boolean {
  if ((list ?? []).some((j) => j.status === "running")) return true;
  return Object.values(useJobsStore.getState().jobs).some((j) => j.status === "running");
}

type CurrentSubmissionResp = {
  ok?: boolean;
  stale?: boolean;
  current?: { job?: SlurmJob | null } | null;
  live?: SlurmJob[];
  queue?: SlurmQueue | null;
};

/** The C4 gate (503 fasrc_offline) or the pre-gate "not connected" reply. */
function isOffline(e: unknown): boolean {
  if (isFasrcOffline(e)) return true;
  return e instanceof ApiError && (e.status === 400 || e.status === 503) && /not connected/i.test(e.message);
}

export function parseSlurmFeed(r: CurrentSubmissionResp | null | undefined): SlurmFeed {
  const current = r?.current ?? null;
  const jobs = Array.isArray(r?.live)
    ? r!.live!.filter((j) => j && j.jobid != null).map((j) => ({ ...j, jobid: String(j.jobid) }))
    : current?.job && current.job.jobid != null ? [{ ...current.job, jobid: String(current.job.jobid) }] : [];
  return { offline: false, stale: !!r?.stale, jobs, queue: r?.queue ?? null, current };
}

async function fetchSlurm({ signal }: { signal: AbortSignal }): Promise<SlurmFeed> {
  try {
    return parseSlurmFeed(await apiGet<CurrentSubmissionResp>(SLURM_FEED_URL, { signal }));
  } catch (e) {
    if (isOffline(e)) return { offline: true, stale: false, jobs: [], queue: null, current: null };
    throw e;
  }
}

/** Refetch both feeds now (e.g. right after starting or cancelling a job). */
export function refreshJobsFeed(): Promise<void> {
  return queryClient.invalidateQueries({ queryKey: JOBS_FEED_KEY });
}

export type JobsFeed = {
  /** Every known local job, newest first. */
  jobs: Job[];
  running: Job[];
  slurm: SlurmJob[];
  slurmQueue: SlurmQueue | null;
  slurmStale: boolean;
  fasrcOffline: boolean;
  /** Running local jobs + live SLURM jobs (the tray badge). */
  runningCount: number;
  /** Ids started from this SPA session, newest first. */
  started: string[];
  loading: boolean;
  error: ApiError | null;
  slurmError: ApiError | null;
  refresh: () => void;
};

const toApiError = (e: unknown): ApiError | null =>
  e == null ? null : e instanceof ApiError ? e : new ApiError({ status: 0, message: message(e) });

/** Subscribe to the shared jobs feed. `slurm: false` skips the FASRC poll. */
export function useJobsFeed(opts: { slurm?: boolean; enabled?: boolean } = {}): JobsFeed {
  const enabled = opts.enabled ?? true;
  const withSlurm = enabled && (opts.slurm ?? true);
  const local = useQuery<Job[], ApiError>({
    queryKey: LOCAL_KEY,
    queryFn: fetchLocal,
    enabled,
    staleTime: 1_000,
    refetchInterval: (q) => (anyRunningLocal(q.state.data)
      ? JOBS_FEED_TIMING.localActiveMs : JOBS_FEED_TIMING.localIdleMs),
    refetchIntervalInBackground: false,
    refetchOnWindowFocus: true,
  }, queryClient);
  const slurm = useQuery<SlurmFeed, ApiError>({
    queryKey: SLURM_KEY,
    queryFn: fetchSlurm,
    enabled: withSlurm,
    staleTime: 5_000,
    retry: false,
    refetchInterval: (q) => {
      const d = q.state.data;
      if (d?.offline) return JOBS_FEED_TIMING.slurmOfflineMs;
      return d?.jobs.length ? JOBS_FEED_TIMING.slurmActiveMs : JOBS_FEED_TIMING.slurmIdleMs;
    },
    refetchIntervalInBackground: false,
    refetchOnWindowFocus: true,
  }, queryClient);

  const jobsById = useJobsStore((s) => s.jobs);
  const started = useJobsStore((s) => s.started);
  const jobs = useMemo(
    () => Object.values(jobsById).sort((a, b) => (b.started ?? 0) - (a.started ?? 0)),
    [jobsById],
  );
  const running = useMemo(() => jobs.filter((j) => j.status === "running"), [jobs]);
  const sd = withSlurm ? slurm.data : undefined;
  const refresh = useCallback(() => { void refreshJobsFeed(); }, []);
  return {
    jobs,
    running,
    slurm: sd?.jobs ?? [],
    slurmQueue: sd?.queue ?? null,
    slurmStale: !!sd?.stale,
    fasrcOffline: !!sd?.offline,
    runningCount: running.length + (sd?.jobs.filter((j) => j.state === "RUNNING" || j.state === "PENDING").length ?? 0),
    started,
    loading: enabled && local.isPending,
    error: enabled ? toApiError(local.error) : null,
    slurmError: withSlurm ? toApiError(slurm.error) : null,
    refresh,
  };
}

/* ── one job's detail poll ───────────────────────────────────────────────── */

/** Poll `/api/jobs/<id>` (backoff) into the store until a TERMINAL detail
 *  snapshot arrives (so the final log is complete). Skips ticks while the
 *  tab is hidden. `onGone` gets a message when the job cannot be read. */
function useJobDetail(id: string | null, enabled: boolean, onGone?: (msg: string) => void): void {
  const gone = useRef(onGone);
  useEffect(() => { gone.current = onGone; });
  useEffect(() => {
    if (!id || !enabled) return;
    if (useJobsStore.getState().final[id]) return;
    let alive = true;
    let failures = 0;
    let interval = JOBS_FEED_TIMING.detailMinMs;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const ctl = new AbortController();
    const schedule = (ms: number) => { timer = setTimeout(tick, ms); };
    async function tick() {
      if (!alive) return;
      if (!focusManager.isFocused()) { schedule(JOBS_FEED_TIMING.detailMaxMs); return; }
      try {
        const j = await apiGet<Job>(`/api/jobs/${encodeURIComponent(id!)}`, { signal: ctl.signal });
        if (!alive) return;
        failures = 0;
        useJobsStore.getState().upsert(j, { detail: true });
        if (isTerminal(j.status)) return;
      } catch (e) {
        if (!alive || isAbortError(e)) return;
        if (e instanceof ApiError && e.status === 404) { gone.current?.(GONE); return; }
        failures += 1;
        if (failures >= 5) { gone.current?.(`${GONE} (${message(e)})`); return; }
      }
      interval = Math.min(JOBS_FEED_TIMING.detailMaxMs, interval * 1.4);
      schedule(interval);
    }
    void tick();
    return () => { alive = false; clearTimeout(timer); ctl.abort(); };
  }, [id, enabled]);
}

/* ── useJob (compat) ─────────────────────────────────────────────────────── */

export type RunOpts = { onDone?: (job: Job) => void };

export type UseJob = {
  job: Job | null;
  error: string | null;
  busy: boolean;
  run: (url: string, data?: FormRecord | FormData, opts?: RunOpts) => Promise<void>;
  reset: () => void;
  /** Request a cooperative cancel of the current job. */
  cancel: () => Promise<{ ok: boolean; error?: string }>;
};

/**
 * Spawn a local job by POSTing a form and follow it. `key` (optional) names
 * the job slot, e.g. "ensemble:evaluate": a component remounted with the
 * same key shows the same job again (running or finished).
 */
export function useJob(key?: string): UseJob {
  const [ownId, setOwnId] = useState<string | null>(null);
  const keyedId = useJobsStore((s) => (key ? s.keyed[key] ?? null : null));
  const jobId = key ? keyedId : ownId;
  const job = useJobsStore((s) => (jobId ? s.jobs[jobId] ?? null : null));
  const [error, setError] = useState<string | null>(null);
  const [posting, setPosting] = useState(false);
  const pending = useRef<{ id: string; onDone?: (j: Job) => void } | null>(null);
  const mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);

  useJobDetail(jobId, error == null, setError);

  useEffect(() => {
    const p = pending.current;
    if (!p || !job || p.id !== job.job_id || !isTerminal(job.status)) return;
    pending.current = null;
    p.onDone?.(job);
  }, [job]);

  const run = useCallback(async (url: string, data?: FormRecord | FormData, opts?: RunOpts) => {
    setError(null);
    setPosting(true);
    pending.current = null;
    if (key) useJobsStore.getState().forgetKey(key);
    else setOwnId(null);
    let res: { job_id?: unknown; error?: unknown } & Record<string, unknown>;
    try {
      res = (await apiPost<typeof res>(url, data ?? {})) ?? {};
    } catch (e) {
      if (mounted.current) { setError(message(e)); setPosting(false); }
      return;
    }
    const id = res.job_id != null ? String(res.job_id) : null;
    if (id) {
      // Registered even if the page was left meanwhile: the tray still shows it.
      useJobsStore.getState().register(id, key);
      void refreshJobsFeed();
    }
    if (!mounted.current) return;
    setPosting(false);
    if (res.error) { setError(String(res.error)); return; }
    if (id) {
      pending.current = { id, onDone: opts?.onDone };
      if (!key) setOwnId(id);
      return;
    }
    // Non-job response (e.g. a synchronous {ok:true}) — done immediately.
    opts?.onDone?.(res as unknown as Job);
  }, [key]);

  const reset = useCallback(() => {
    pending.current = null;
    setError(null);
    setPosting(false);
    if (key) useJobsStore.getState().forgetKey(key);
    else setOwnId(null);
  }, [key]);

  const cancel = useCallback(
    () => (jobId ? cancelJob(jobId) : Promise.resolve({ ok: false, error: "no job" })),
    [jobId],
  );

  const busy = posting || (jobId != null && error == null && (job == null || job.status === "running"));
  return { job, error, busy, run, reset, cancel };
}

/* ── useTrackedJob (compat) ──────────────────────────────────────────────── */

/**
 * Discover a matching RUNNING job even when it was started from another tab
 * or before this page loaded (label substring match on the shared feed).
 * Once attached, follow it (full log) and keep its terminal snapshot.
 */
export function useTrackedJob(labelNeedle: string): Job | null {
  const { jobs } = useJobsFeed({ slurm: false });
  const [tracked, setTracked] = useState<{ needle: string; id: string } | null>(null);
  const id = tracked && tracked.needle === labelNeedle ? tracked.id : null;
  useEffect(() => {
    if (id) return;
    const hit = jobs.find((j) => j.status === "running" && j.label.includes(labelNeedle));
    if (hit) setTracked({ needle: labelNeedle, id: hit.job_id });
  }, [jobs, id, labelNeedle]);
  useJobDetail(id, true);
  return useJobsStore((s) => (id ? s.jobs[id] ?? null : null));
}

/* ── cancel ──────────────────────────────────────────────────────────────── */

/** Request a cooperative cancel (C2): the job ends as `cancelled` at its next
 *  progress tick. Resolves `{ok:false, error}` when refused (404/409). */
export async function cancelJob(id: string): Promise<{ ok: boolean; error?: string }> {
  try {
    await apiPost(`/api/jobs/${encodeURIComponent(id)}/cancel`);
  } catch (e) {
    return { ok: false, error: message(e) };
  }
  const cur = useJobsStore.getState().jobs[id];
  if (cur && cur.status === "running") {
    useJobsStore.getState().upsert({ ...cur, cancel_requested: true, cancellable: false });
  }
  void refreshJobsFeed();
  return { ok: true };
}

/** Cancel a SLURM job (scancel on FASRC). */
export async function cancelSlurmJob(jobid: string): Promise<{ ok: boolean; error?: string }> {
  try {
    const r = await apiPost<{ ok?: boolean; error?: string }>("/api/fasrc/cancel", { jobid });
    if (r && r.ok === false) return { ok: false, error: r.error ?? "cancel refused" };
  } catch (e) {
    return { ok: false, error: message(e) };
  }
  void refreshJobsFeed();
  return { ok: true };
}
