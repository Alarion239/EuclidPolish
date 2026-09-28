/* Typed endpoints of the System workspace (euclid_polish/web/API.md: Local
 * git, FASRC checkout and storage, Provenance, /api/system). URL builders +
 * response shapes only. */

const qs = (params: Record<string, string | number | boolean | null | undefined>): string => {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === "" || v === false) continue;
    sp.set(k, String(v));
  }
  const s = sp.toString();
  return s ? `?${s}` : "";
};

/* ── FASRC checkout and storage ───────────────────────────────────────────── */

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
/** The staleness service (routes/system.py → helpers/system_alerts.py):
 *  one verdict per Loop stage, the same Home's Loop strip reads. */
export const LOOP_URL = "/api/system/loop";
export type LoopResp = {
  computed_at: string; ttl_s: number; counts: Record<string, number>; errors: Record<string, string>;
  stages: { id: string; label: string; state: string; reason: string; detail?: string | null; to: string }[];
};
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

/* ── this server: runtime, disk, data roots (GET /api/system) ────────────── */

export type DiskLevel = "ok" | "warn" | "bad" | "unknown";
export type RootRow = { id: string; label: string; path: string; group: string; bytes: number; files: number; exists: boolean };
export type SystemInfo = {
  python: { version: string; implementation: string; executable: string };
  platform: { system: string; release: string; machine: string; platform: string };
  packages: Record<string, string | null>;
  node: string | null;
  pid: number;
  data_dir: string;
  noise_model: string;
  disk: {
    path: string; total_bytes: number; free_bytes: number; used_bytes: number; used_fraction: number | null;
    level: DiskLevel; warn_below_bytes: number; bad_below_bytes: number; warn_used_fraction: number;
  };
  roots: { items: RootRow[]; computed_at: string | null; total_bytes: number | null; stale: boolean; refresh_job: string | null };
  experiments: { cache_budget_bytes: number; min_free_bytes: number; cache_bytes: number | null; outputs_bytes: number | null };
};
export const SYSTEM_URL = "/api/system";
export const GIT_STATUS_URL = "/api/git/status";
export const FASRC_GIT_URL = "/api/fasrc/git-status";
