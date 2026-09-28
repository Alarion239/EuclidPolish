/* "Run job X" from anywhere: the local jobs a user starts most often
 * (evaluate the STARFULL ensemble, PSNR vs knee, member PSNR, disk usage,
 * the health checks) as palette actions under "Run a job", plus shortcuts
 * to the pages that own the knob-heavy runs (fit a gate variant, compare
 * models on real tiles, train members). The shell mounts <RunActions/> once;
 * Home's quick actions reuse `useRunJobs()`.
 *
 * TensorFlow-heavy runs ask first (confirm()). A started job is registered in
 * the jobs store under its `run:*` key, so it shows in the job tray, toasts
 * when it ends (useJobToasts) and a second start while it runs is refused. */
import { useNavigate } from "react-router-dom";
import { ApiError, apiGet, apiPost } from "../api/client";
import { refreshJobsFeed, useJobsStore } from "../api/jobs";
import { invalidate } from "../api/query";
import { confirm, toast } from "../ui";
import { openInspector } from "./inspector";
import { pagePath } from "./nav";
import { usePageActions, type PageAction } from "./palette";
import { SYSTEM_ALERTS_URL } from "./status";

/** The palette group of these actions (listed after the page's own actions). */
export const RUN_GROUP = "Run a job";

export type Runner = {
  id: string;
  label: string;
  busy: boolean;
  run: () => Promise<void>;
};

type Question = { title: string; message: string; confirmLabel: string };

type RunSpec = {
  key: string;
  label: string;
  url: string;
  data?: Record<string, string>;
  question?: Question;
};

const SPECS = {
  evaluate: {
    key: "run:evaluate", label: "Evaluate the STARFULL ensemble", url: "/ensemble/evaluate",
    data: { mode: "starfull" },
    question: {
      title: "Evaluate the STARFULL ensemble?",
      message: "Loads every active member (TensorFlow) and scores the local test records. It takes several minutes.",
      confirmLabel: "Evaluate",
    },
  },
  knee: {
    key: "run:knee", label: "Compute PSNR vs knee (STARFULL)", url: "/ensemble/knee-psnr",
    data: { mode: "starfull" },
    question: {
      title: "Recompute PSNR vs knee?",
      message: "Scores every member, the mean and each combiner at every asinh knee from the cached test cubes.",
      confirmLabel: "Compute",
    },
  },
  memberPsnr: {
    key: "run:member-psnr", label: "Re-score the members' test PSNR", url: "/ensemble/member-psnr",
    question: {
      title: "Re-score the members' test PSNR?",
      message: "Only changed or unscored members are evaluated (TensorFlow).",
      confirmLabel: "Re-score",
    },
  },
  diskUsage: {
    key: "run:disk-usage", label: "Measure disk usage per data root", url: "/api/system/disk-usage/refresh",
  },
} satisfies Record<string, RunSpec>;

type JobKey = keyof typeof SPECS;

const openLog = (id: string) => openInspector({ kind: "job", id: `local/${id}` });

function runningId(key: string): string | null {
  const { keyed, jobs } = useJobsStore.getState();
  const id = keyed[key];
  return id && jobs[id]?.status === "running" ? id : null;
}

const KEY_BY_URL = new Map<string, string>(Object.values(SPECS).map((s) => [s.url, s.key]));

/** The one-at-a-time key of a run: a spec POSTing a shared runner's endpoint
 *  (e.g. a Home health-check fix hitting /ensemble/knee-psnr) shares that
 *  runner's key, so the palette, Home's quick actions and the check rows can
 *  never start the same job twice. */
export function jobKeyFor(spec: Pick<RunSpec, "key" | "url">): string {
  return KEY_BY_URL.get(spec.url) ?? spec.key;
}

/** POST a job endpoint (after `question`), register the job, toast the start
 *  (`quiet`: no start toast — for runs the page starts by itself; a failure
 *  still toasts, and the tray/useJobToasts still report the end). */
export async function startJob(spec: RunSpec, opts: { quiet?: boolean } = {}): Promise<string | null> {
  const key = jobKeyFor(spec);
  const already = runningId(key);
  if (already) {
    if (!opts.quiet) toast.info(`${spec.label} is already running`, { action: { label: "Log", onClick: () => openLog(already) } });
    return already;
  }
  if (spec.question && !(await confirm(spec.question))) return null;
  try {
    const res = await apiPost<{ ok?: boolean; job_id?: unknown; error?: string }>(spec.url, spec.data ?? {});
    if (res?.ok === false || res?.error) throw new Error(res.error ?? "refused");
    const id = res?.job_id != null ? String(res.job_id) : null;
    if (!id) throw new Error("the server did not return a job");
    useJobsStore.getState().register(id, key);
    void refreshJobsFeed();
    if (!opts.quiet) toast.info(`${spec.label}: started`, { action: { label: "Log", onClick: () => openLog(id) } });
    return id;
  } catch (e) {
    toast.error(`${spec.label}: did not start`, { description: e instanceof ApiError || e instanceof Error ? e.message : String(e) });
    return null;
  }
}

/** Recompute the health checks now (bypassing the server's 30 s memo). */
export async function refreshHealth(): Promise<void> {
  try {
    await apiGet(`${SYSTEM_ALERTS_URL}?fresh=1`);
    await invalidate(SYSTEM_ALERTS_URL);
    toast.success("Health checks refreshed");
  } catch (e) {
    toast.error("Health checks failed", { description: e instanceof Error ? e.message : String(e) });
  }
}

/** The job runners Home and the palette share (busy = a run is in flight). */
export function useRunJobs(): Record<JobKey | "health", Runner> {
  const busyKeys = useJobsStore((s) => Object.keys(SPECS).filter((k) => {
    const id = s.keyed[SPECS[k as JobKey].key];
    return !!id && s.jobs[id]?.status === "running";
  }).join(","));
  const busy = new Set(busyKeys ? busyKeys.split(",") : []);
  const runner = (k: JobKey): Runner => ({
    id: k, label: SPECS[k].label, busy: busy.has(k),
    run: async () => { await startJob(SPECS[k]); },
  });
  return {
    evaluate: runner("evaluate"),
    knee: runner("knee"),
    memberPsnr: runner("memberPsnr"),
    diskUsage: runner("diskUsage"),
    health: { id: "health", label: "Re-run the health checks", busy: false, run: refreshHealth },
  };
}

/** Registers the "Run a job" palette actions while the shell is mounted. */
export function RunActions() {
  const navigate = useNavigate();
  const jobs = useRunJobs();
  const K = ["run job", "job", "run"];
  const actions: PageAction[] = [
    { id: "run:evaluate", label: jobs.evaluate.label, group: RUN_GROUP, keywords: [...K, "evaluate", "psnr", "test"],
      disabled: jobs.evaluate.busy, run: () => { void jobs.evaluate.run(); } },
    { id: "run:knee", label: jobs.knee.label, group: RUN_GROUP, keywords: [...K, "knee", "integrated", "leaderboard"],
      disabled: jobs.knee.busy, run: () => { void jobs.knee.run(); } },
    { id: "run:member-psnr", label: jobs.memberPsnr.label, group: RUN_GROUP, keywords: [...K, "members", "psnr", "score"],
      disabled: jobs.memberPsnr.busy, run: () => { void jobs.memberPsnr.run(); } },
    { id: "run:disk-usage", label: jobs.diskUsage.label, group: RUN_GROUP, keywords: [...K, "disk", "space", "storage", "du"],
      disabled: jobs.diskUsage.busy, run: () => { void jobs.diskUsage.run(); } },
    { id: "run:health", label: jobs.health.label, group: RUN_GROUP, keywords: [...K, "health", "alerts", "stale", "staleness"],
      run: () => { void jobs.health.run(); } },
    { id: "run:fit-gate", label: "Fit a spatial-gate variant…", group: RUN_GROUP, keywords: [...K, "combiner", "gate", "fit"],
      run: () => navigate(pagePath("models", { tab: "combiner" })) },
    { id: "run:experiment", label: "Compare models on real tiles…", group: RUN_GROUP, keywords: [...K, "experiment", "real", "holes"],
      run: () => navigate(pagePath("sky", { tab: "compare" })) },
    { id: "run:train", label: "Train or continue members on FASRC…", group: RUN_GROUP, keywords: [...K, "train", "members", "slurm"],
      run: () => navigate(pagePath("models", { tab: "train" })) },
  ];
  usePageActions(actions);
  return null;
}
