/* Typed endpoints of the Runs workspace (euclid_polish/web/API.md: FASRC
 * jobs, history, logs, queue, steps). URL builders + response shapes only. */
import type { Recommendation } from "../shared/resourceAdviceModel";
import type { HistoryRow } from "./model";
import type { Step } from "./steps/stepForm";

export type { HistoryRow, Recommendation };

const qs = (params: Record<string, string | number | boolean | null | undefined>): string => {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === "" || v === false) continue;
    sp.set(k, String(v));
  }
  const s = sp.toString();
  return s ? `?${s}` : "";
};

/* ── FASRC ────────────────────────────────────────────────────────────────── */

export type StepsStatus = {
  ssh_connected: boolean;
  steps: Step[];
  artifacts: Record<string, boolean | null>;
  remote_paths: Record<string, string>;
};

export const STEPS_STATUS_URL = "/api/fasrc/steps/status";
export const stepHistoryUrl = (stepId: string) => `/api/fasrc/steps/${encodeURIComponent(stepId)}/history`;

export type StepHistoryResp = { ok: boolean; step_id: string; history: HistoryRow[]; match: HistoryRow | null };

export type HistoryResp = {
  ok: boolean; total: number; offset: number; limit: number; rows: HistoryRow[];
  facets: { steps: Record<string, number>; states: Record<string, number> }; unresolved: number;
};
export const historyUrl = (opts: { step?: string; state?: string; q?: string; limit?: number; offset?: number } = {}) =>
  `/api/fasrc/history${qs({ limit: 2000, ...opts })}`;

export type QueueItem = { id: string; label: string; step?: string | null; queued_at?: number | null; position: number };
export type QueueState = {
  count: number; names: string[]; items: QueueItem[]; active_jobid?: string | null;
  halted: boolean; halted_reason?: string | null;
};
export const QUEUE_STATE_URL = "/api/fasrc/queue/state";

export type SubmitResp = {
  ok?: boolean; jobid?: string; slurm_id?: string; queued?: boolean; label?: string;
  queue?: QueueState; error?: string;
};

export type RunTask = {
  index: number; member: string; jobid: string; name: string; state?: string | null;
  out_path?: string | null; err_path?: string | null; out_size?: number; err_size?: number; missing?: boolean;
};
export type RunRow = {
  name: string; jobid?: string | null; label?: string | null; state?: string | null;
  submitted_at?: number; started_at?: number | null; ended_at?: number | null;
  out_size?: number; err_size?: number; missing?: boolean; mtime?: number;
  out_path?: string | null; err_path?: string | null; array_count?: number; tasks?: RunTask[];
  params?: Record<string, unknown>;
};
export type RunsResp = {
  ok: boolean; log_dir?: string; runs: RunRow[]; total_runs: number; page: number; page_size: number;
  has_older?: boolean; has_newer?: boolean;
};
export const runsUrl = (page: number, pageSize = 100) => `/api/fasrc/runs${qs({ page, page_size: pageSize })}`;

export type LogPage = {
  ok: boolean; path: string; page: number; page_size: number; total_lines: number;
  start_line: number; end_line: number; has_older: boolean; has_newer: boolean; content: string;
};
export type LogGrep = { ok: boolean; path: string; grep: string; matches: { line: number; text: string }[]; truncated: boolean };
export const logGrepUrl = (path: string, needle: string) => `/api/fasrc/runs/log${qs({ path, grep: needle })}`;

/* ── resource usage (Runs › Resources; the resource advisor) ─────────────── */

/** Runs per state over all of a step's ledger rows. */
export type StateCounts = { completed: number; oom: number; timeout: number; failed: number; cancelled: number; running: number };

/** One step over its ledger rows (`GET /api/fasrc/resources`): medians
 *  (`cpu_efficiency` 0–1, `gpu_util` %, `mem_ratio` / `time_ratio` used ÷
 *  requested), the p90 peak memory and the allocated vs used hours of the
 *  finished runs (an array job counts one task's allocation). `null` = unknown.
 *  `registered`: the console still submits the step (false for a historical
 *  one, which only has its ledger rows). */
export type StepSummary = {
  step_id: string; label: string; needs_gpu: boolean; registered: boolean; runs: number; states: StateCounts;
  success_rate: number | null; last_submitted_at: string | null;
  cpu_efficiency: number | null; gpu_util: number | null; mem_ratio: number | null; time_ratio: number | null;
  peak_mem_p90_mb: number | null;
  cpu_hours_alloc: number | null; cpu_hours_used: number | null;
  gpu_hours_alloc: number | null; gpu_hours_used: number | null;
  mem_gb_hours_alloc: number | null; mem_gb_hours_used: number | null;
};
export type ResourcesResp = { ok: boolean; steps: StepSummary[] };
export const RESOURCES_URL = "/api/fasrc/resources";

/** One run's requested vs used resources (per array task). */
export type RunUsage = {
  jobid: string; submitted_at: string | null; state: string; partition?: string | null;
  cpus: number | null; gpus: number | null;
  req_memory: string | null; req_memory_mb: number | null; req_time_limit: string | null; req_time_s: number | null;
  elapsed_s: number | null; cpu_efficiency: number | null; cores_used: number | null;
  peak_mem_mb: number | null; mem_ratio: number | null; time_ratio: number | null;
  gpu_util: number | null; gpu_mem_used_mb: number | null;
  units: number | null; units_label: string | null; key_label: string | null; label: string | null;
};
/** One step: its summary, its runs (newest first, ≤ 200) and the advisor's
 *  recommendation for a run like its latest counted one. */
export type StepResourcesResp = {
  ok: boolean; step_id: string; summary: StepSummary; runs: RunUsage[]; recommendation: Recommendation | null;
};
export const stepResourcesUrl = (stepId: string) => `${RESOURCES_URL}/${encodeURIComponent(stepId)}`;

/* ── campaigns (the History campaign filter) ──────────────────────────────── */

/** The FASRC job ids a campaign logged (`current`, `unassigned` or an
 *  archived campaign's dir): `GET /api/tracking/jobs?ids=1`. */
export const campaignJobIdsUrl = (campaign: string) => `/api/tracking/jobs${qs({ campaign, ids: 1 })}`;
export type CampaignJobIds = { ok: boolean; campaign: string; total: number; jobids: string[] };

/** The campaign names for the filter (the tracking store's state). */
export const TRACKING_STATE_URL = "/api/tracking/state";
export type CampaignChoices = {
  active: { title: string } | null; archived: { title: string; _dir: string }[];
  jobs_count: number; unassigned_count: number;
};

/** Per-member training series (the wall time per 1000 steps of a run). */
export const TRAINING_CURVES_URL = "/ensemble/training-curves.json";
