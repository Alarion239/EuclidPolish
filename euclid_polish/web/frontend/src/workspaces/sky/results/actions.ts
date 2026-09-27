/* Mutating actions of the Real results / Experiments / Catalog-eval tabs:
 * confirm → POST → a local job in the global tray (it survives navigation;
 * the shell toasts its end) → refresh the affected resources. Every server
 * refusal is shown with the server's own error text. */
import { apiGet, apiPost, ApiError } from "../../../api/client";
import { isTerminal, refreshJobsFeed, useJobsStore, type Job } from "../../../api/jobs";
import { getResourceData, invalidate } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { formatBytes, formatDec, formatRA } from "../../../format";
import { confirm, toast } from "../../../ui";
import { URLS, type ExperimentStart, type ModelSpecRow, type ModelsPayload } from "./api";
import { experimentCost, experimentCostText, specShort } from "./model";

type Form = Record<string, string | number | boolean | null | undefined>;

export type Started = { ok: true; jobId: string | null; body: Record<string, unknown> } | { ok: false; error: string; code?: string };

export const errorText = (e: unknown): string =>
  e instanceof ApiError || e instanceof Error ? e.message : String(e);

const openLog = (id: string) => openInspector({ kind: "job", id: `local/${id}` });

/** Refresh every view of real tiles, experiments and the atlas layers. */
export function refreshResults(): void {
  void invalidate("/api/real/");
  void invalidate("/api/experiments");
  void invalidate("/api/sky/");
}

/** Call `onDone` once the job (tracked by the shared jobs feed) ends. */
export function whenJobEnds(id: string, onDone: (job: Job) => void): void {
  const check = (): boolean => {
    const job = useJobsStore.getState().jobs[id];
    if (job && isTerminal(job.status)) { onDone(job); return true; }
    return false;
  };
  if (check()) return;
  const unsubscribe = useJobsStore.subscribe(() => { if (check()) unsubscribe(); });
}

/** POST a job-starting endpoint; register the job in the tray. */
export async function startJob(url: string, data: Form, opts: { key?: string; onDone?: (job: Job) => void } = {}): Promise<Started> {
  try {
    const body = (await apiPost<Record<string, unknown>>(url, data)) ?? {};
    if (body.error || body.ok === false) {
      return { ok: false, error: String(body.error ?? "refused"), code: body.code ? String(body.code) : undefined };
    }
    const jobId = body.job_id != null ? String(body.job_id) : null;
    if (jobId) {
      useJobsStore.getState().register(jobId, opts.key);
      void refreshJobsFeed();
      if (opts.onDone) whenJobEnds(jobId, opts.onDone);
    }
    return { ok: true, jobId, body };
  } catch (e) {
    return { ok: false, error: errorText(e), code: e instanceof ApiError ? e.code : undefined };
  }
}

function announce(r: Started, what: string): void {
  if (!r.ok) { toast.error(`${what}: did not start`, { description: r.error }); return; }
  const id = r.jobId;
  toast.info(`${what}: started`, id ? { action: { label: "Log", onClick: () => openLog(id) } } : undefined);
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

/* ── experiments ───────────────────────────────────────────────────────── */

export type RunResult = { jobId: string | null; experimentId: string | null } | null;

/** The model catalogue for a cost estimate (the query cache, else one GET;
 *  null when it cannot be read — the confirm then names no numbers). */
async function modelCatalogue(): Promise<ModelSpecRow[] | null> {
  const cached = getResourceData<ModelsPayload>(URLS.models);
  if (cached?.models) return cached.models;
  try {
    return (await apiGet<ModelsPayload>(URLS.models))?.models ?? null;
  } catch {
    return null;
  }
}

/** The Run confirm's body: what runs, what it costs, what is reused. */
export function runMessage(refs: readonly string[], specs: readonly string[], catalogue: readonly ModelSpecRow[] | null): string {
  const what = `${specs.map(specShort).join(", ")} on ${refs.length === 1 ? refs[0] : plural(refs.length, "tile")}.`;
  const cost = catalogue ? experimentCostText(experimentCost(specs, catalogue, refs.length)) : "";
  return [
    what,
    cost || "Runs the needed ensemble members on this machine; cached ones are reused.",
    "A local TensorFlow job; every SR is scored with the real-data metrics.",
  ].join(" ");
}

/** Run model specs on real tiles as an experiment (TensorFlow, local job). */
export async function runModels(
  refs: readonly string[], specs: readonly string[],
  opts: { label?: string; ask?: boolean; onDone?: (job: Job, experimentId: string | null) => void } = {},
): Promise<RunResult> {
  if (!refs.length || !specs.length) return null;
  if (opts.ask !== false) {
    const ok = await confirm({
      title: `Run ${plural(specs.length, "model")} on ${plural(refs.length, "tile")}?`,
      message: runMessage(refs, specs, await modelCatalogue()),
      confirmLabel: "Run",
    });
    if (!ok) return null;
  }
  let experimentId: string | null = null;
  const r = await startJob(URLS.experiments, { tiles: refs.join(","), models: specs.join(","), label: opts.label || undefined }, {
    key: "sky:experiment",
    onDone: (job) => { refreshResults(); opts.onDone?.(job, experimentId); },
  });
  if (!r.ok) {
    toast.error("Experiment: did not start", { description: r.error });
    return null;
  }
  const body = r.body as ExperimentStart;
  experimentId = body.experiment_id ?? null;
  const skipped = Object.entries(body.skipped ?? {});
  announce(r, "Experiment");
  if (skipped.length) {
    toast.warning(`Skipped ${plural(skipped.length, "model")}`, {
      description: skipped.map(([s, why]) => `${specShort(s)}: ${why}`).join(" · "),
    });
  }
  void invalidate("/api/experiments");
  return { jobId: r.jobId, experimentId };
}

/** Score the current outputs that have no metrics yet (`metricsPlan`): one
 *  experiment per group of tiles — the SRs are reused, only the real-data
 *  metrics are computed. One confirm for the whole plan. */
export async function computeMetrics(plan: readonly { specs: string[]; refs: string[] }[]): Promise<number> {
  if (!plan.length) return 0;
  const outputs = plan.reduce((n, g) => n + g.specs.length * g.refs.length, 0);
  const tiles = plan.reduce((n, g) => n + g.refs.length, 0);
  const ok = await confirm({
    title: `Compute the metrics of ${plural(outputs, "output")} on ${plural(tiles, "tile")}?`,
    message: `Scores the cached SRs (${[...new Set(plan.flatMap((g) => g.specs))].map(specShort).join(", ")}) with the real-data metrics as ${plural(plan.length, "experiment")}; no SR is recomputed while it is current.`,
    confirmLabel: "Compute metrics",
  });
  if (!ok) return 0;
  let started = 0;
  for (const g of plan) {
    if (await runModels(g.refs, g.specs, { ask: false, label: "metrics" })) started += 1;
  }
  return started;
}

/* ── delete outputs ────────────────────────────────────────────────────── */

/** Delete the cached model outputs + member-SR cache of tiles (never the LR). */
export async function deleteOutputs(refs: readonly string[]): Promise<number> {
  if (!refs.length) return 0;
  const ok = await confirm({
    title: `Delete the model outputs of ${plural(refs.length, "tile")}?`,
    message: `Removes every cached SR, its metrics and the member-SR cache of ${refs.length <= 3 ? refs.join(", ") : `${refs.slice(0, 3).join(", ")} …`}. The LR tiles and legacy SR files stay.`,
    tone: "danger", confirmLabel: "Delete outputs",
  });
  if (!ok) return 0;
  let removed = 0, freed = 0;
  const failures: string[] = [];
  for (const ref of refs) {
    try {
      const r = await apiPost<{ ok?: boolean; removed_count?: number; cache_bytes_freed?: number; error?: string }>(URLS.deleteOutputs(ref), {});
      if (r?.ok === false || r?.error) throw new Error(r.error ?? "refused");
      removed += r?.removed_count ?? 0;
      freed += r?.cache_bytes_freed ?? 0;
    } catch (e) {
      failures.push(`${ref}: ${errorText(e)}`);
    }
  }
  refreshResults();
  if (failures.length) toast.error(`Could not delete ${plural(failures.length, "tile")}' outputs`, { description: failures.slice(0, 3).join(" · ") });
  if (refs.length > failures.length) toast.success(`Deleted ${plural(removed, "file")}`, { description: freed ? `${formatBytes(freed)} of member cache freed` : undefined });
  return removed;
}

/* ── cache a 25.6″ tile ────────────────────────────────────────────────── */

type AtResponse = { q1_verdict?: "observed" | "unobserved" | "outside"; best_tile?: string | null };
const where = (ra: number, dec: number) => `${formatRA(ra)} ${formatDec(dec)}`;

/** Cache a four-band 25.6″ tile at (ra, dec), optionally + production & mean. */
export async function cacheTile(ra: number, dec: number, opts: { run?: boolean } = {}): Promise<Started | null> {
  let at: AtResponse;
  try {
    at = await apiGet<AtResponse>(`/api/sky/at?ra=${ra.toFixed(6)}&dec=${dec.toFixed(6)}`);
  } catch (e) {
    toast.error("Could not check the Q1 coverage", { description: errorText(e) });
    return null;
  }
  if (at.q1_verdict === "outside") {
    toast.error(`${where(ra, dec)} is outside Euclid Q1`, { description: "No 25.6″ tile can be cached there." });
    return null;
  }
  const force = at.q1_verdict === "unobserved";
  const ok = await confirm({
    title: force ? "Cache a tile on an unobserved Q1 tile?" : "Cache a 25.6″ tile?",
    message: force
      ? `Every Q1 tile containing ${where(ra, dec)} was measured unobserved; the cutout is probably blank.`
      : `Downloads VIS + NISP Y/J/H at ${where(ra, dec)} from the public Euclid archive${at.best_tile ? ` (Q1 tile ${at.best_tile})` : ""}${opts.run ? ", then runs production and the mean" : ""}.`,
    confirmLabel: force ? "Cache anyway" : "Cache tile",
    tone: force ? "danger" : undefined,
  });
  if (!ok) return null;
  const r = await startJob(URLS.cacheTile, {
    ra: ra.toFixed(6), dec: dec.toFixed(6), run: opts.run ? "production,mean" : undefined, force: force ? 1 : undefined,
  }, { key: "sky:cache-tile", onDone: refreshResults });
  announce(r, "Tile cache");
  return r;
}

/* ── NEXUS field inference (the legacy per-tile SR the NEXUS viewer shows) ─ */

export async function runNexusField(fieldId: string, tiles: readonly string[], spec = "production"): Promise<Started | null> {
  const ok = await confirm({
    title: `Run ${specShort(spec)} on ${tiles.length ? plural(tiles.length, "NEXUS tile") : "every stale NEXUS tile"}?`,
    message: `NEXUS field inference (TensorFlow, local job): ${spec === "production" ? "replaces the stale per-tile SRs the NEXUS viewer and plates use" : "writes the model output store"}.`,
    confirmLabel: "Run",
  });
  if (!ok) return null;
  const r = await startJob("/api/jwst-euclid/nexus/infer", {
    field_id: fieldId, tiles: tiles.length ? tiles.join(",") : undefined, spec,
  }, { key: "sky:nexus-inference", onDone: refreshResults });
  announce(r, "NEXUS inference");
  return r;
}

/* ── the legacy 10×10 real field (its diagnostics) ─────────────────────── */

/** Re-run the newest STARFULL combiners on the cached real field (rebuilding
 *  stale member SRs) — rewrites its model–model / σ / occupancy diagnostics. */
export async function refreshFieldDiagnostics(fieldId: string, onDone?: () => void): Promise<Started | null> {
  const ok = await confirm({
    title: "Recompute the real-field diagnostics?",
    message: `Applies the newest starfull combiners to field ${fieldId} (TensorFlow, local job; stale member SRs are rebuilt, the 100 sub-tiles re-downloaded only if the member cache is stale).`,
    confirmLabel: "Recompute",
  });
  if (!ok) return null;
  const r = await startJob(URLS.fieldRefresh, {}, {
    key: "sky:field-diagnostics",
    onDone: () => { void invalidate("/api/inference/"); refreshResults(); onDone?.(); },
  });
  announce(r, "Field diagnostics");
  return r;
}

