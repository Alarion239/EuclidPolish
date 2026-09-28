/* Pure helpers of the Runs workspace (no React): SLURM state tones, the
 * resource-usage figures of a ledger row, the params that vary across runs,
 * log paging, the one Live list (local + SLURM), the one History ledger, the
 * member names a job trains, progress text, wall time per 1000 steps and the
 * FASRC steps by pipeline stage. Unit-tested in model.test.ts. */
import type { Job } from "../../api/jobs";
import { pagePath } from "../../app/nav";
import { formatCount } from "../../format";
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

/** Jobstats' sign-off: seff's closing line, not a finding. */
const NOTE_BOILERPLATE = /^have a nice day!?$/i;
/** Sentences of a Jobstats note that give advice or a link, not the finding. */
const NOTE_ADVICE = /^(please |for more info|for instance|if there are|for future jobs|this will lower)/i;

/** The findings of Jobstats (JSON list of strings), one line: the seff
 *  sign-off is dropped, and each note keeps its finding sentences (the
 *  advice, the rhetorical question and the docs link are cut). */
export function accountingNotes(row: HistoryRow): string {
  let notes: unknown;
  try { notes = JSON.parse(row.jobstats_notes_json || "[]"); } catch { return ""; }
  if (!Array.isArray(notes)) return "";
  const out: string[] = [];
  for (const n of notes) {
    if (typeof n !== "string" || NOTE_BOILERPLATE.test(n.trim())) continue;
    const kept = n.trim().split(/(?<=[.!?])\s+(?=[A-Z])/).filter((sentence) => !NOTE_ADVICE.test(sentence));
    if (kept.length) out.push(kept.join(" "));
  }
  return out.join(" ");
}

/** The exit code worth showing: only a non-zero one ("1:0", "0:125"); a
 *  "0:0" says nothing the state does not (TIMEOUT and CANCELLED read 0:0). */
export function exitCodeText(row: HistoryRow): string | null {
  const code = String(row.exit_code ?? "").trim();
  return code && !/^0(:0)?$/.test(code) ? code : null;
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

/** Submission bookkeeping, not task params worth a summary: SLURM
 *  resources, the array layout, seeds, and the member list (the run's label
 *  already names the members). */
const SUMMARY_SKIP = new Set(["confirm", "partition", "n_cpus", "n_gpus", "memory", "time_limit", "step_id", "label",
  "preset", "array_count", "array_max_parallel", "base_seed", "count", "member_names", "members", "n_members",
  "member_spec", "fork_track", "executor", "workers"]);
/** The knobs that define what a run made, shown first (the rest follow in
 *  schema order): a member's loss, knee and regime before its batch size. */
const SUMMARY_FIRST = ["loss", "knee_loss", "mode", "starless", "steps", "target_steps", "extra_steps", "fork_from",
  "icnr", "noise_aug", "bootstrap", "lr_peak", "n_train", "n_valid", "n_test"];

/** "k=v · k=v" of the first `max` user-facing params, the defining knobs
 *  first: blanks, bookkeeping and internal `_`-prefixed keys (star-prior
 *  paths / hashes) are skipped — they stay in the inspector / JSON. */
export function paramsSummary(params: Record<string, unknown> | null | undefined, max = 4): string {
  const rank = (k: string) => { const i = SUMMARY_FIRST.indexOf(k); return i < 0 ? SUMMARY_FIRST.length : i; };
  return Object.entries(params ?? {})
    .filter(([k, v]) => !k.startsWith("_") && !SUMMARY_SKIP.has(k) && !blank(v))
    .map((e, i) => ({ e, i })).sort((a, b) => (rank(a.e[0]) - rank(b.e[0])) || (a.i - b.i))
    .slice(0, max).map(({ e: [k, v] }) => `${k}=${paramText(v)}`).join(" · ");
}

/* ── logs ─────────────────────────────────────────────────────────────────── */

/** The page (counted from the END of the file, page 0 = newest lines) that
 *  holds 1-based line `line` of a `total`-line file with `pageSize` lines. */
export function pageForLine(line: number, total: number, pageSize: number): number {
  if (total <= 0 || pageSize <= 0 || line < 1) return 0;
  return Math.max(0, Math.floor((total - Math.min(line, total)) / pageSize));
}


/* ── the members a job trains ─────────────────────────────────────────────── */

type ParamsCarrier = { params?: Record<string, unknown> | null; params_json?: string | null; [key: string]: unknown };

function paramsOf(row: ParamsCarrier): Record<string, unknown> {
  if (row.params && typeof row.params === "object") return row.params;
  try {
    const p = JSON.parse(String(row.params_json || "{}"));
    return p && typeof p === "object" && !Array.isArray(p) ? p : {};
  } catch { return {}; }
}

/** The member names of an ensemble_train submission: `members` of a
 *  continue, else `member_names` of a new batch. */
export function memberNames(params: Record<string, unknown> | null | undefined): string[] {
  const p = params ?? {};
  const raw = p.mode === "continue" ? p.members : p.member_names;
  return String(raw ?? "").split(",").map((s) => s.trim()).filter(Boolean);
}

/** "members 199–202" for a contiguous run of numbers, "members 190, 196"
 *  otherwise, "member 7" for one; names that are not member_<n> as written. */
export function membersText(names: readonly string[]): string {
  if (!names.length) return "";
  const nums = names.map((n) => /^member_0*(\d+)$/.exec(n)?.[1]).map((d) => (d == null ? null : Number(d)));
  if (nums.some((n) => n == null)) return names.join(", ");
  const sorted = [...new Set(nums as number[])].sort((a, b) => a - b);
  if (sorted.length === 1) return `member ${sorted[0]}`;
  const contiguous = sorted.every((n, i) => i === 0 || n === sorted[i - 1] + 1);
  return contiguous ? `members ${sorted[0]}–${sorted[sorted.length - 1]}` : `members ${sorted.join(", ")}`;
}

/** A SLURM job's label with the members it trains ("Train ensemble ·
 *  members 199–202"; the step's generic "(N members, distinct seeds)" gives
 *  way to the names), unless the label already names them. */
export function jobLabel(row: ParamsCarrier & { label?: string | null; step_id?: string | null }): string {
  const raw = String(row.label || row.step_id || "");
  const members = membersText(memberNames(paramsOf(row)));
  if (!members || raw.includes(members)) return raw;
  const base = raw.replace(/\s*\([^)]*\)\s*$/, "");
  return base ? `${base} · ${members}` : members;
}

/* ── progress ─────────────────────────────────────────────────────────────── */

/** "step 10,650 / 70,000 (15%)": the unit word once (a Reporter label like
 *  "step 10650" keeps only its word), the counts grouped, the share rounded. */
export function progressText(step: { current: number; total: number; label?: string | null } | null | undefined): string {
  if (!step || !(step.total > 0)) return "";
  const word = String(step.label ?? "").trim().split(/\s+/)[0]?.replace(/[^\p{L}_-]+/gu, "") || "step";
  const pct = Math.round((100 * Math.max(0, Math.min(step.current, step.total))) / step.total);
  return `${word} ${formatCount(step.current)} / ${formatCount(step.total)} (${pct}%)`;
}

/* ── Live: one list ───────────────────────────────────────────────────────── */

export type LiveItem = {
  key: string; source: "local" | "slurm"; id: string; state: string; label: string; sub: string;
  progress: string; startedAt: number | null; elapsed: string; where: string; cli: boolean;
  cancellable: boolean; job?: Job; slurm?: LiveRow;
};

const LIVE_ORDER: Record<string, number> = { RUNNING: 0, COMPLETING: 1, PENDING: 2 };

/** The running local jobs and the live SLURM jobs (with the CLI squeue
 *  rows already merged in), running first, then newest first. */
export function liveItems(local: readonly Job[], slurm: readonly LiveRow[]): LiveItem[] {
  const items: LiveItem[] = [];
  for (const j of local) {
    if (j.status !== "running") continue;
    items.push({
      key: `local:${j.job_id}`, source: "local", id: j.job_id, state: "RUNNING", label: j.label, sub: j.kind ?? "",
      progress: j.progress && j.progress.total > 0 ? progressText(j.progress) : j.progress?.label ?? "",
      startedAt: j.started ?? null, elapsed: "", where: "this laptop", cli: false,
      cancellable: j.cancellable !== false && !j.cancel_requested, job: j,
    });
  }
  for (const r of slurm) {
    const state = String(r.state || "UNKNOWN").toUpperCase();
    const current = typeof r.progress_step === "number" ? r.progress_step : null;
    const total = typeof r.progress_total === "number" ? r.progress_total : null;
    items.push({
      key: `slurm:${r.jobid}`, source: "slurm", id: String(r.jobid), state, label: jobLabel(r) || String(r.jobid),
      sub: r.cli ? "submitted outside the console" : String(r.step_id ?? ""),
      progress: current != null && total ? progressText({ current, total }) : "",
      startedAt: typeof r.submitted_at === "number" ? r.submitted_at : null,
      elapsed: r.time ? `${r.time}${r.time_limit ? ` / ${r.time_limit}` : ""}` : "",
      where: String((state === "PENDING" ? r.reason : r.nodes) ?? ""), cli: !!r.cli, cancellable: !r.cli, slurm: r,
    });
  }
  return items
    .map((it, i) => ({ it, i }))
    .sort((a, b) => (LIVE_ORDER[a.it.state] ?? 3) - (LIVE_ORDER[b.it.state] ?? 3)
      || (b.it.startedAt ?? 0) - (a.it.startedAt ?? 0) || a.i - b.i)
    .map((x) => x.it);
}

export type Scope = "all" | "local" | "slurm";

export function scopeCounts(items: readonly { source: "local" | "slurm" }[]): Record<Scope, number> {
  const local = items.filter((i) => i.source === "local").length;
  return { all: items.length, local, slurm: items.length - local };
}

/* ── History: one ledger ──────────────────────────────────────────────────── */

export type HistoryItem = {
  key: string; source: "local" | "slurm"; id: string; submitted: number | null; step: string; state: string;
  label: string; elapsed: number | null; row?: HistoryRow; job?: Job;
};

const epoch = (v: unknown): number | null => {
  if (typeof v === "number" && Number.isFinite(v)) return v > 1e12 ? v / 1000 : v;
  if (typeof v === "string" && v.trim()) {
    const t = Date.parse(v);
    return Number.isFinite(t) ? t / 1000 : null;
  }
  return null;
};

/** The SLURM ledger rows and the FINISHED local jobs (running ones are on
 *  Live), newest first. */
export function historyItems(ledger: readonly HistoryRow[], local: readonly Job[]): HistoryItem[] {
  const items: HistoryItem[] = ledger.map((r) => ({
    key: String(r.jobid), source: "slurm" as const, id: String(r.jobid), submitted: epoch(r.submitted_at),
    step: String(r.step_id ?? ""), state: rowState(r), label: jobLabel(r) || String(r.jobid),
    elapsed: finiteNumber(r.elapsed_seconds), row: r,
  }));
  for (const j of local) {
    if (j.status === "running") continue;
    items.push({
      key: `local:${j.job_id}`, source: "local", id: j.job_id, submitted: j.started ?? null, step: j.kind ?? "",
      state: j.status.toUpperCase(), label: j.label, elapsed: finiteNumber(j.duration), job: j,
    });
  }
  return items.map((it, i) => ({ it, i }))
    .sort((a, b) => (b.it.submitted ?? 0) - (a.it.submitted ?? 0) || a.i - b.i).map((x) => x.it);
}

/* ── wall time per 1000 steps (ensemble_train runs) ───────────────────────── */

export type StepTimeCurve = { name: string; step_time?: [number, number][] | null };

/** The run's members' seconds per 1000 steps (from training-curves.json) and
 *  the median over all their samples. */
export function wallPer1k(curves: readonly StepTimeCurve[], members: readonly string[]):
  { members: { name: string; points: [number, number][] }[]; median: number | null } {
  const want = new Set(members);
  const picked = curves.filter((c) => want.has(c.name)).map((c) => ({ name: c.name, points: c.step_time ?? [] }));
  const values = picked.flatMap((m) => m.points.map((p) => p[1])).filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  const median = values.length ? (values.length % 2 ? values[(values.length - 1) / 2]
    : (values[values.length / 2 - 1] + values[values.length / 2]) / 2) : null;
  return { members: picked, median };
}

/* ── FASRC steps by pipeline stage ────────────────────────────────────────── */

export type StageId = "reference" | "noise-fields" | "generation" | "training" | "figures" | "other";

export const STEP_STAGES: { id: Exclude<StageId, "other">; label: string }[] = [
  { id: "reference", label: "Reference data" },
  { id: "noise-fields", label: "Noise and fields" },
  { id: "generation", label: "Generation" },
  { id: "training", label: "Training" },
  { id: "figures", label: "Figures" },
];

const STAGE_OF: Record<string, StageId> = {
  euclid_query: "reference", euclid_verify_photometry: "reference", download_euclid_cutouts: "reference",
  extract_euclid_psf: "reference", psf_rotation_pool: "reference", download_tng_skirt: "reference",
  measure_tng_radii: "reference",
  vis_noise_sample: "noise-fields", archive_field_sample: "noise-fields",
  synthetic_generate: "generation",
  ensemble_train: "training",
  tng_grid: "figures", tng_stack: "figures", poster_cutout: "figures",
};

export const stageOfStep = (stepId: string): StageId => STAGE_OF[stepId] ?? "other";

/** The steps in stage order (registry order within a stage); an unknown step
 *  lands in a trailing "Other" group. Empty stages are left out. */
export function stepsByStage<S extends { step_id: string }>(steps: readonly S[]):
  { id: StageId; label: string; steps: S[] }[] {
  const groups = [...STEP_STAGES, { id: "other" as const, label: "Other" }]
    .map((g) => ({ ...g, steps: steps.filter((s) => stageOfStep(s.step_id) === g.id) }));
  return groups.filter((g) => g.steps.length);
}

const withQuery = (path: string, query: string) => (query ? `${path}?${query}` : path);

/** The tab whose drawer embeds a step (its "home"), or null for an unknown step. */
export function stepHome(stepId: string): { label: string; to: string } | null {
  const synthetic = (tab: string, query: string, label: string) =>
    ({ label: `Synthetic › ${label}`, to: withQuery(pagePath("synthetic", { tab }), query) });
  switch (stepId) {
    case "euclid_query": case "euclid_verify_photometry": return synthetic("psf", "how=1", "PSF");
    case "download_euclid_cutouts": return synthetic("psf", "view=cutouts&how=1", "PSF");
    case "extract_euclid_psf": case "psf_rotation_pool": return synthetic("psf", "view=epsf&how=1", "PSF");
    case "download_tng_skirt": case "measure_tng_radii": case "tng_grid": case "tng_stack":
      return synthetic("galaxies", "view=templates&how=1", "Galaxies");
    case "vis_noise_sample": return synthetic("noise", "how=1", "Noise");
    case "archive_field_sample": return synthetic("fields", "ref=1", "Fields");
    case "synthetic_generate": return synthetic("records", "gen=1", "Records");
    case "ensemble_train": return { label: "Models › Train", to: pagePath("models", { tab: "train", params: { mode: "starfull" } }) };
    case "poster_cutout": return { label: "Figures › Plates", to: pagePath("figures", { tab: "plates" }) };
    default: return null;
  }
}

/* ── layout ───────────────────────────────────────────────────────────────── */

/** Bring a panel that opened below the fold into view (a narrow page stacks
 *  the selected run / step under its list); a panel already visible stays. */
export function revealBelowFold(el: HTMLElement | null | undefined): void {
  if (!el || typeof window === "undefined") return;
  const top = el.getBoundingClientRect().top;
  if (top < window.innerHeight * 0.75) return;
  const reduce = typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  el.scrollIntoView?.({ block: "start", behavior: reduce ? "auto" : "smooth" });
}

/* ── paths ────────────────────────────────────────────────────────────────── */

export const basename = (path: string): string => path.replace(/\/+$/, "").split("/").pop() || path;

export function parentDir(path: string): string {
  const trimmed = path.replace(/\/+$/, "");
  const i = trimmed.lastIndexOf("/");
  return i > 0 ? trimmed.slice(0, i) : "/";
}
