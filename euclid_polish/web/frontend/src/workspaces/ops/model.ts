/* Pure helpers of the Ops workspace (no React): SLURM state tones, the
 * resource-usage figures of a history row, the params that vary across runs,
 * log paging, FASRC-vs-local git relations and remote paths. */
import type { Tone } from "../../ui";

/* ── SLURM states ─────────────────────────────────────────────────────────── */

const GOOD = new Set(["COMPLETED", "DONE"]);
const BAD = new Set(["FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE",
  "PREEMPTED", "NO_SACCT"]);
export const LIVE_STATES = new Set(["PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "COMPLETING",
  "CONFIGURING"]);

/** Badge tone of a SLURM / job-ledger state (RUNNING is the accent-less
 *  default; PENDING, UNKNOWN and anything unrecognised warn). */
export function jobStateTone(state: string | null | undefined): "good" | "warn" | "bad" | "info" | undefined {
  const s = String(state ?? "").trim().toUpperCase().split(" ")[0];
  if (GOOD.has(s)) return "good";
  if (BAD.has(s)) return "bad";
  if (s === "RUNNING") return "info";
  return s ? "warn" : undefined;
}

export const isLiveState = (state: string | null | undefined): boolean =>
  LIVE_STATES.has(String(state ?? "").trim().toUpperCase());

/** Local job statuses (C2) → tone. */
export function localJobTone(status: string): Tone {
  return status === "done" ? "good" : status === "failed" ? "bad" : status === "cancelled" ? "warn" : "info";
}

/* ── numbers of a history row ─────────────────────────────────────────────── */

export const finiteNumber = (value: unknown): number | null => {
  if (value === "" || value === null || value === undefined || typeof value === "boolean") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
};

/** "8G" / "512M" / "8000Mc" / 2048 (MB) → megabytes. */
export function memoryMegabytes(value: unknown): number | null {
  const n = finiteNumber(value);
  if (n != null) return n;
  const m = String(value ?? "").trim().match(/^([\d.]+)\s*([KMGT])?i?B?c?n?$/i);
  if (!m) return null;
  const scale = { k: 1 / 1024, m: 1, g: 1024, t: 1024 * 1024 }[(m[2] || "m").toLowerCase() as "k" | "m" | "g" | "t"];
  return Number(m[1]) * scale;
}

export const formatMemory = (mb: number | null): string =>
  mb == null ? "—" : mb >= 1024 ? `${(mb / 1024).toFixed(mb >= 10240 ? 0 : 1)} GB` : `${Math.round(mb)} MB`;

export type HistoryRow = {
  jobid: string;
  step_id?: string;
  label?: string;
  submitted_at?: string;
  state?: string;
  state_display?: string;
  db_state?: string | null;
  params?: Record<string, unknown>;
  params_omitted?: Record<string, number>;
  params_json?: string;
  partition?: string;
  req_cpus?: string | number;
  req_gpus?: string | number;
  req_memory?: string;
  req_time_limit?: string;
  elapsed_seconds?: string;
  exit_code?: string;
  alloc_cpus?: string | number;
  alloc_gpus?: string | number;
  alloc_memory_mb?: string;
  cpu_efficiency?: string;
  max_rss_mb?: string;
  cpu_util_mean?: string;
  cpu_util_peak?: string;
  gpu_util_mean?: string;
  gpu_util_peak?: string;
  gpu_mem_peak?: string;
  gpu_mem_peak_mb?: string;
  gpu_mem_util_peak?: string;
  accounting_source?: string;
  jobstats_cpu_util?: string;
  jobstats_cpu_memory_util?: string;
  jobstats_cpu_memory_used_mb?: string;
  jobstats_cpu_memory_alloc_mb?: string;
  jobstats_gpu_util?: string;
  jobstats_gpu_memory_util?: string;
  jobstats_gpu_memory_used_mb?: string;
  jobstats_gpu_memory_total_mb?: string;
  jobstats_notes_json?: string;
  log_path?: string;
  err_path?: string;
  [key: string]: unknown;
};

/** The state a history row shows: sacct's verdict, else the DB state. */
export const rowState = (row: HistoryRow): string =>
  String(row.state_display || row.state || row.db_state || "PENDING").toUpperCase();

export type Usage = { used: number | null; requested: number | null; pct: number | null; mean?: number | null };

/** CPU cores used at peak / requested, peak % and mean %. */
export function cpuUsage(row: HistoryRow): Usage {
  const requested = finiteNumber(row.req_cpus) ?? finiteNumber(row.alloc_cpus);
  const eff = finiteNumber(row.cpu_efficiency);
  const peak = finiteNumber(row.cpu_util_peak) ?? finiteNumber(row.jobstats_cpu_util) ?? (eff == null ? null : eff * 100);
  const mean = finiteNumber(row.jobstats_cpu_util) ?? finiteNumber(row.cpu_util_mean);
  return { used: requested != null && peak != null ? (requested * peak) / 100 : null, requested, pct: peak, mean };
}

/** CPU memory max used / requested (MB) and the fraction. */
export function memoryUsage(row: HistoryRow): Usage {
  const used = finiteNumber(row.jobstats_cpu_memory_used_mb) ?? finiteNumber(row.max_rss_mb);
  const requested = finiteNumber(row.jobstats_cpu_memory_alloc_mb) ?? memoryMegabytes(row.alloc_memory_mb)
    ?? memoryMegabytes(row.req_memory);
  const pct = used != null && requested != null && requested > 0 ? (100 * used) / requested : null;
  return { used, requested, pct };
}

/** GPUs used at peak / requested; `mean` utilisation; `memPct` GPU memory. */
export function gpuUsage(row: HistoryRow): Usage & { memPct: number | null; memUsed: number | null; memTotal: number | null } {
  const requested = finiteNumber(row.req_gpus) ?? finiteNumber(row.alloc_gpus);
  const peak = finiteNumber(row.gpu_util_peak);
  const mean = finiteNumber(row.jobstats_gpu_util) ?? finiteNumber(row.gpu_util_mean);
  // `gpu_mem_peak` is a legacy live-sampler percentage; a row that also
  // carries `gpu_mem_peak_mb` has an absolute MB there, never a percentage.
  const memPct = finiteNumber(row.jobstats_gpu_memory_util) ?? finiteNumber(row.gpu_mem_util_peak)
    ?? (finiteNumber(row.gpu_mem_peak_mb) == null ? finiteNumber(row.gpu_mem_peak) : null);
  return {
    used: requested != null && peak != null ? (requested * peak) / 100 : null, requested, pct: peak, mean, memPct,
    memUsed: finiteNumber(row.jobstats_gpu_memory_used_mb), memTotal: finiteNumber(row.jobstats_gpu_memory_total_mb),
  };
}

export const hasGpu = (row: HistoryRow): boolean => {
  const g = gpuUsage(row);
  return (g.requested ?? 0) > 0 || g.mean != null || g.pct != null || g.memPct != null;
};

/** Advisory notes of Jobstats (JSON list of strings). */
export function accountingNotes(row: HistoryRow): string {
  try {
    const notes = JSON.parse(row.jobstats_notes_json || "[]");
    return Array.isArray(notes) ? notes.filter((n) => typeof n === "string").join(" ") : "";
  } catch { return ""; }
}

/** The params of a history row (compact `params`, else parsed `params_json`). */
export function rowParams(row: HistoryRow): Record<string, unknown> {
  if (row.params && typeof row.params === "object") return row.params;
  try {
    const p = JSON.parse(row.params_json || "{}");
    return p && typeof p === "object" && !Array.isArray(p) ? p : {};
  } catch { return {}; }
}

/** One row of `GET /api/fasrc/queue` (the user's `squeue`, `-r`: array tasks
 *  one per row; `reason` is the nodelist of a running job, `%R`). */
export type SqueueRow = {
  jobid: string; name?: string; state?: string; time?: string; time_limit?: string; nodes?: string; reason?: string;
  start_time?: string;
};

/** A Live-table row: a console-submitted job of the jobs feed, or (`cli`) a
 *  job of `$USER` that squeue lists but the local job DB does not know. */
export type LiveRow = {
  jobid: string; state: string; label?: string | null; step_id?: string | null; submitted_at?: number | null;
  reason?: string | null; nodes?: string | null; time?: string | null; time_limit?: string | null;
  start_time?: string | null; cli?: boolean;
  [key: string]: unknown;
};

const parentJobid = (jobid: string): string => jobid.split("_")[0];

/** The feed's jobs, then every squeue row whose job (or array parent) the
 *  feed lacks — the jobs submitted from the CLI, marked `cli`. */
export function mergeCliJobs(feed: readonly LiveRow[], squeue: readonly SqueueRow[] | null | undefined): LiveRow[] {
  const known = new Set(feed.map((j) => parentJobid(String(j.jobid))));
  const cli: LiveRow[] = [];
  for (const r of squeue ?? []) {
    if (!r.jobid || known.has(parentJobid(r.jobid))) continue;
    const state = (r.state || "UNKNOWN").toUpperCase();
    cli.push({
      jobid: r.jobid, state, label: r.name || null, step_id: null, cli: true,
      time: r.time || null, time_limit: r.time_limit || null,
      reason: state === "PENDING" ? r.reason || null : null,
      nodes: state === "PENDING" ? null : r.reason || null,
      start_time: r.start_time && r.start_time !== "N/A" ? r.start_time : null,
    });
  }
  return [...feed, ...cli];
}

const blank = (v: unknown) => v === null || v === undefined || (typeof v === "string" && v.trim() === "");

/** Up to `max` of `names` whose values differ across `rows` (the task params
 *  worth a column in a run history), in schema order. */
export function varyingParams(rows: readonly HistoryRow[], names: readonly string[], max = 4): string[] {
  const out: string[] = [];
  if (max <= 0) return out;
  for (const name of names) {
    const seen = new Set<string>();
    for (const r of rows) {
      const v = rowParams(r)[name];
      seen.add(blank(v) ? "" : typeof v === "object" ? JSON.stringify(v) : String(v));
      if (seen.size > 1) break;
    }
    if (seen.size > 1) out.push(name);
    if (out.length >= max) break;
  }
  return out;
}

export const paramText = (v: unknown): string =>
  blank(v) ? "—" : typeof v === "object" ? JSON.stringify(v) : String(v);

/** Submission bookkeeping, not task params worth a summary. */
const SUMMARY_SKIP = new Set(["confirm", "partition", "n_cpus", "n_gpus", "memory", "time_limit", "step_id", "label",
  "preset"]);

/** "k=v · k=v" of the first `max` user-facing params: blanks, SLURM resources
 *  and internal `_`-prefixed keys (star-prior paths / hashes) are skipped —
 *  they stay in the inspector / JSON. */
export function paramsSummary(params: Record<string, unknown> | null | undefined, max = 4): string {
  return Object.entries(params ?? {})
    .filter(([k, v]) => !k.startsWith("_") && !SUMMARY_SKIP.has(k) && !blank(v))
    .slice(0, max).map(([k, v]) => `${k}=${paramText(v)}`).join(" · ");
}

/* ── logs ─────────────────────────────────────────────────────────────────── */

/** The page (counted from the END of the file, page 0 = newest lines) that
 *  holds 1-based line `line` of a `total`-line file with `pageSize` lines. */
export function pageForLine(line: number, total: number, pageSize: number): number {
  if (total <= 0 || pageSize <= 0 || line < 1) return 0;
  return Math.max(0, Math.floor((total - Math.min(line, total)) / pageSize));
}

/* ── git ──────────────────────────────────────────────────────────────────── */

export type Relation = "same" | "remote_behind" | "remote_ahead" | "diverged" | "unknown";

export function relationText(rel: { relation: Relation | string; ahead?: number | null; behind?: number | null } | null | undefined):
  { label: string; tone: Tone; hint: string } {
  const r = rel?.relation ?? "unknown";
  const n = (k: number | null | undefined) => `${k ?? "?"} commit${k === 1 ? "" : "s"}`;
  switch (r) {
    case "same": return { label: "in sync with this laptop", tone: "good", hint: "FASRC runs the same commit as the local checkout." };
    case "remote_behind": return { label: `FASRC ${n(rel?.ahead)} behind`, tone: "warn", hint: "Push the local commits, then git pull on FASRC." };
    case "remote_ahead": return { label: `FASRC ${n(rel?.behind)} ahead`, tone: "info", hint: "FASRC has commits this laptop has not pulled." };
    case "diverged": return { label: "diverged", tone: "bad", hint: `Local ${n(rel?.ahead)} ahead, FASRC ${n(rel?.behind)} ahead.` };
    default: return { label: "unknown", tone: "neutral", hint: "The FASRC commit is not in the local repo (fetch first)." };
  }
}

/** Porcelain XY → a short human status. */
export function gitStatusText(xy: string): string {
  if (xy === "??") return "untracked";
  if (xy.includes("U") || xy === "AA" || xy === "DD") return "conflict";
  const pick = (c: string) => ({ M: "modified", A: "added", D: "deleted", R: "renamed", C: "copied", T: "type" }[c]);
  return pick(xy[0]) ?? pick(xy[1]) ?? xy.trim();
}

/** Append one history page to the commits already loaded, dropping repeats
 *  (a new commit shifts the pages by one while the history is open). */
export function mergeCommitPages<C extends { hash: string; full?: string }>(older: readonly C[], page: readonly C[]): C[] {
  const seen = new Set(older.map((c) => c.full ?? c.hash));
  return [...older, ...page.filter((c) => !seen.has(c.full ?? c.hash))];
}

/* ── paths ────────────────────────────────────────────────────────────────── */

export const basename = (path: string): string => path.replace(/\/+$/, "").split("/").pop() || path;

export function parentDir(path: string): string {
  const trimmed = path.replace(/\/+$/, "");
  const i = trimmed.lastIndexOf("/");
  return i > 0 ? trimmed.slice(0, i) : "/";
}

/** A local project path the Inspect workspace can open (`tracking/…`). */
export const inspectHrefFor = (path: string): string => `/inspect?fits=${encodeURIComponent(path)}`;
