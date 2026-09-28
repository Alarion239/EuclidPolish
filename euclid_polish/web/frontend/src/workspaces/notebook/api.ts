/* Typed endpoints of the Notebook workspace (euclid_polish/web/API.md,
 * Tracking): the tracking store's state, a campaign, its logged jobs, time
 * travel. URL builders + response shapes only. */

const qs = (params: Record<string, string | number | boolean | null | undefined>): string => {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === "" || v === false) continue;
    sp.set(k, String(v));
  }
  const s = sp.toString();
  return s ? `?${s}` : "";
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

