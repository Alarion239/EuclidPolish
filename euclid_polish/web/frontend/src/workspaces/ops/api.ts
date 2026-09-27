/* Typed endpoints of the Ops workspace (euclid_polish/web/API.md: FASRC,
 * Local git, Tracking, Provenance). URL builders + response shapes only. */
import type { HistoryRow } from "./model";
import type { Step } from "./steps/stepForm";

export type { HistoryRow };

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

export type DataListing = {
  ok: boolean; error?: string; data_dir?: string; ckpt_dir?: string;
  du?: [string, string][]; tfrecords?: { path: string; size: number }[];
  checkpoints?: { path: string; size: number; mtime?: string }[];
};

export type RemoteEntry = {
  name: string; path: string; type: "dir" | "file" | "link" | "other";
  size: number | null; mtime: number | null; inspectable: boolean;
};
export type FilesResp = {
  ok: boolean; dir: string | null; crumbs: { name: string; path: string }[]; entries: RemoteEntry[]; truncated: boolean;
};
export const filesUrl = (dir: string) => `/api/fasrc/files${qs({ dir })}`;
export const remoteInspectHref = (path: string) => `/fasrc/file/inspect${qs({ remote_path: path })}`;
export const remoteDownloadHref = (path: string) => `/fasrc/file/download${qs({ remote_path: path })}`;

export type RemoteGit = {
  ok: boolean; error?: string; repo?: string; branch?: string; ahead?: number; behind?: number;
  head?: string; local_head?: string;
  relation?: { relation: string; ahead: number | null; behind: number | null };
  dirty?: boolean; dirty_files?: string[];
  last?: { hash?: string; subject?: string; relative?: string };
};
export type GitPullResp = { ok: boolean; stdout?: string; changed_files?: string[]; env_update_needed?: boolean; error?: string };

export type MirrorStatus = {
  last_run_at: number | null; last_rc: number | null; last_error: string; last_stdout: string;
  remote_dir: string; local_dir: string; job_id: string | null;
};

/* ── tracking ─────────────────────────────────────────────────────────────── */

export type Commit = { short?: string; hash?: string; branch?: string; dirty?: boolean } | null;
export type Campaign = {
  title: string; description?: string; slug: string; status?: string;
  created_at?: string; created_commit?: Commit; saved_at?: string | null; saved_commit?: Commit;
};
export type BackupRec = {
  name: string; kind: string; comment?: string; size_bytes: number; created_at?: string; commit?: Commit;
  source_path?: string; files?: string[];
};
export type Archived = Campaign & { _dir: string; models: BackupRec[] };
export type Sandbox = {
  short: string; commit?: string; created_at?: string; source?: Record<string, unknown> | null;
  source_label?: string | null; running: boolean; url?: string | null; port?: number | null;
  remote?: { ok?: boolean; error?: string; worktree?: string } | null; worktree?: string;
};
export type Backups = { models: BackupRec[]; fits: BackupRec[]; images: BackupRec[] };
export type TrackingState = {
  active: Campaign | null; archived: Archived[]; backups: Backups;
  jobs_count: number; unassigned_count: number; log_md: string;
  tracking_dir?: string; remote_dir?: string; ssh_connected: boolean; sandboxes: Sandbox[];
};
export const TRACKING_STATE_URL = "/api/tracking/state";

export type TrackedJob = {
  jobid: string; label?: string; step_id?: string; logged_at?: string; commit?: Commit;
  log_path?: string; err_path?: string; params: Record<string, unknown>; params_omitted: Record<string, number>;
};
export type TrackingJobsResp = { ok: boolean; campaign: string; total: number; offset: number; limit: number; jobs: TrackedJob[] };
export const trackingJobsUrl = (campaign: string, offset: number, limit: number, q: string) =>
  `/api/tracking/jobs${qs({ campaign, offset, limit, q })}`;

export type CampaignResp = {
  ok: boolean; dir: string; active: boolean; metadata: Campaign; backups: Backups; log_md: string; jobs_count: number;
};
export const campaignUrl = (name: string) => `/api/tracking/campaign/${encodeURIComponent(name)}`;

export type RestoreResp = {
  ok: boolean; short?: string; url?: string | null; error?: string; warning?: string | null;
  remote?: { ok?: boolean; error?: string } | null;
};

/* ── local git ────────────────────────────────────────────────────────────── */

export type GitFile = {
  xy: string; path: string; orig: string | null; staged: boolean; unstaged: boolean; untracked: boolean;
  size: number | null; guard: string | null;
};
export type GitStatus = {
  in_repo: boolean; root?: string; branch?: string; upstream?: string | null; ahead?: number; behind?: number;
  files?: GitFile[]; last?: { hash?: string; subject?: string; relative?: string } | null; clean?: boolean;
};
export type GitStatusResp = { status: GitStatus; log: GitCommit[] };
export type GitCommit = { hash: string; full?: string; author: string; date?: string; relative: string; subject: string };
export type GitLogResp = { commits: GitCommit[]; total: number; skip: number; limit: number; has_more: boolean };
export const gitLogUrl = (skip: number, limit: number) => `/api/git/log${qs({ skip, limit })}`;
export const gitDiffUrl = (path: string | null, staged: boolean) => `/api/git/diff${qs({ path, staged: staged ? 1 : null })}`;
export type GitDiffResp = { diff: string; staged: boolean; path: string | null };
export type GitShow = {
  ok: boolean; error?: string; full: string; hash: string; author: string; email: string; date: string;
  subject: string; body: string; stat: string; patch: string; truncated: boolean;
};
export const gitShowUrl = (rev: string) => `/api/git/commit/${encodeURIComponent(rev)}`;
export type GitActionResp = { ok: boolean; error?: string; code?: string; stdout?: string; committed?: string[];
  staged?: string[]; unstaged?: string[]; refused?: { path: string; size?: number; reason?: string }[] };

/* ── provenance ───────────────────────────────────────────────────────────── */

export type ProvVerdict = "current" | "stale" | "unknown";
export type ProvModel = { id: string; member: string | null };
export type ProvRow = {
  id: string; kind: string; category: "process" | "artifact"; source: "prov" | "sidecar" | "checkpoint";
  file: string; created_at: string | null; status: string | null; path: string | null; format: string | null;
  label: string; git: string | null; dirty: boolean | null; config_type: string | null; seed: number | null;
  produced_by: string | null; parents: string[]; inputs: string[]; outputs: string[];
  ra: number | null; dec: number | null; member: string | null;
  verdict: ProvVerdict | null; models: ProvModel[]; n_upstream: number; n_downstream: number;
};
export type CurrentModel = { id: string; member: string; regime: string; dir: string };
export type ProvSummary = {
  ok: boolean; total: number; counts: { kinds: Record<string, number>; verdicts: Record<ProvVerdict, number> };
  roots: { path: string; role: string; records: number }[]; current_models: CurrentModel[];
  truncated: boolean; duplicates: number; built_at: number; build_seconds: number;
};
export const PROV_SUMMARY_URL = "/api/provenance/summary";
export type ProvRecordsResp = { ok: boolean; total: number; offset: number; limit: number; records: ProvRow[] };
export const provRecordsUrl = (opts: { q?: string; kind?: string; verdict?: string; source?: string; limit?: number; offset?: number }) =>
  `/api/provenance/records${qs({ limit: 1000, ...opts })}`;
export type ProvRef = { id: string; role: string; exists: boolean; kind: string | null; label: string | null;
  member: string | null; depth?: number };
export type ProvRecordResp = {
  ok: boolean; entry: ProvRow; record: Record<string, unknown> | null; upstream: ProvRef[]; downstream: ProvRef[];
  ancestors: { total: number; items: ProvRef[] }; descendants: { total: number; items: ProvRef[] };
  models: ProvModel[]; current_models: CurrentModel[]; inspect_path: string | null;
};
export const provRecordUrl = (id: string) => `/api/provenance/record/${encodeURIComponent(id)}`;
