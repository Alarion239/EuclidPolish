/* Sky › Targets actions: "Run production on stale" (one confirm for the
 * whole plan, then local jobs in the tray) and the Sources (grouped
 * analysis, Q1 lens catalogue, real-galaxy query, the FASRC sync of the
 * evaluation results), each confirmed first.
 * Every server refusal is shown with the server's own error text. */
import { apiPost } from "../../../api/client";
import { invalidate } from "../../../api/query";
import { formatCount } from "../../../format";
import { confirm, toast } from "../../../ui";
import { errorText, refreshResults, startJob } from "../results/actions";
import { URLS } from "../results/api";
import type { ProductionPlan } from "./model";

const plural = (n: number, one: string, many = `${one}s`) => `${formatCount(n)} ${n === 1 ? one : many}`;

/** After a catalogue job: the evaluation rows, the eval tiles and the atlas. */
export function refreshTargets(): void {
  refreshResults();
  void invalidate(URLS.evalRuns);
  void invalidate("/viewer/meta/evaluation");
}

/** The confirm body of a plan: what runs, on how many targets, how. */
export function planLines(plan: ProductionPlan): string[] {
  const out: string[] = [];
  if (plan.nexus) out.push(`${plural(plan.nexus.tiles.length, "NEXUS × JWST tile")}: NEXUS field inference, the per-tile SR the NEXUS viewer and plates read.`);
  if (plan.tiles.length) out.push(`${plural(plan.tiles.length, "other tile")}: one production comparison (every SR is scored).`);
  if (plan.catalogue) {
    out.push(`${plural(plan.catalogue.stale, "lens candidate or Q1 galaxy", "lens candidates and Q1 galaxies")}: the grouped analysis re-makes the stale reconstructions (${plan.catalogue.n} per lens grade, ${3 * plan.catalogue.n} galaxies; current ones are reused).`);
  }
  return out;
}

const announce = (ok: boolean, what: string, error?: string) => {
  if (ok) toast.info(`${what}: started`, { description: "Follow it in the job tray." });
  else toast.error(`${what}: did not start`, { description: error });
};

/** Start every job of the plan after ONE confirm; resolves to the jobs started. */
export async function runProductionPlan(plan: ProductionPlan): Promise<number> {
  if (!plan.stale) return 0;
  const yes = await confirm({
    title: `Run production on ${plural(plan.stale, "stale target")}?`,
    message: [...planLines(plan), "Each runs locally (TensorFlow) as a job in the tray."].join(" "),
    confirmLabel: "Run production",
  });
  if (!yes) return 0;
  let started = 0;
  if (plan.nexus) {
    const r = await startJob("/api/jwst-euclid/nexus/infer", {
      field_id: plan.nexus.field, tiles: plan.nexus.tiles.join(","), spec: "production",
    }, { key: "sky:nexus-inference", onDone: refreshTargets });
    announce(r.ok, "NEXUS production", r.ok ? undefined : r.error);
    if (r.ok) started += 1;
  }
  if (plan.tiles.length) {
    const r = await startJob(URLS.experiments, { tiles: plan.tiles.join(","), models: "production", label: "production on stale" },
      { key: "sky:experiment", onDone: refreshTargets });
    announce(r.ok, "Production comparison", r.ok ? undefined : r.error);
    if (r.ok) started += 1;
  }
  if (plan.catalogue) {
    const r = await startJob("/api/evaluation/run-grouped", { n: plan.catalogue.n, synthetic: 0 },
      { key: "catalog-eval:grouped", onDone: refreshTargets });
    announce(r.ok, "Grouped analysis", r.ok ? undefined : r.error);
    if (r.ok) started += 1;
  }
  return started;
}

/** The grouped analysis over the lens candidates and Q1 galaxies. */
export async function runGrouped(n: number, synthetic: boolean): Promise<boolean> {
  const yes = await confirm({
    title: "Run the grouped analysis?",
    message: `Reconstructs up to ${n} lens candidates per grade and ${3 * n} Q1 galaxies with the production model${synthetic ? `, plus ${3 * n} synthetic lens and ${3 * n} synthetic galaxy stamps (Models › Images)` : ""}. A local TensorFlow job; current reconstructions are reused.`,
    confirmLabel: "Run",
  });
  if (!yes) return false;
  const r = await startJob("/api/evaluation/run-grouped", { n, synthetic: synthetic ? 1 : 0 }, { key: "catalog-eval:grouped", onDone: refreshTargets });
  announce(r.ok, "Grouped analysis", r.ok ? undefined : r.error);
  return r.ok;
}

/** Query real galaxies from the Euclid archive (needs the archive login). */
export async function queryGalaxies(n: number, regenerate: boolean): Promise<boolean> {
  const yes = await confirm({
    title: `Query ${plural(n, "Q1 galaxy", "Q1 galaxies")}?`,
    message: `Draws galaxies from the lens fields with MER + PHZ cone queries on the Euclid archive${regenerate ? ", discarding the cached draw" : " (the cached draw is topped up)"}; the grouped analysis then reconstructs them. A local job.`,
    confirmLabel: "Query",
    tone: regenerate ? "danger" : undefined,
  });
  if (!yes) return false;
  const r = await startJob("/api/evaluation/query-galaxies", { n_galaxies: n, regenerate: regenerate ? 1 : 0 },
    { key: "catalog-eval:galaxies", onDone: refreshTargets });
  announce(r.ok, "Galaxy query", r.ok ? undefined : r.error);
  return r.ok;
}

/** Download the Euclid Q1 strong-lens catalogue (synchronous). */
export async function fetchLensCatalogue(): Promise<boolean> {
  const yes = await confirm({
    title: "Fetch the Q1 strong-lens catalogue?",
    message: "Downloads the Euclid Q1 discovery-engine lens catalogue (≈ 0.4 MB, Zenodo) and rewrites lens_catalog/lenses.csv.",
    confirmLabel: "Fetch",
  });
  if (!yes) return false;
  try {
    const r = (await apiPost<{ ok?: boolean; error?: string; rows?: number }>("/api/evaluation/fetch-catalog", {})) ?? {};
    if (r.ok === false || r.error) throw new Error(String(r.error ?? "refused"));
    toast.success(`Lens catalogue: ${formatCount(r.rows)} rows`);
    void invalidate("/api/sky/");
    return true;
  } catch (e) {
    toast.error("Lens catalogue: failed", { description: errorText(e) });
    return false;
  }
}

/** Pull the catalogue evaluation's results from FASRC. The server refuses it
 *  without confirm=1 (it runs `rsync --delete-after`), so ask first. */
export async function syncEvaluation(): Promise<boolean> {
  const yes = await confirm({
    title: "Sync the evaluation results from FASRC?",
    message: "rsync --delete-after will delete local-only results in data/eval_results that FASRC does not have.",
    tone: "danger", confirmLabel: "Sync and delete local-only",
  });
  if (!yes) return false;
  try {
    const r = (await apiPost<{ ok?: boolean; error?: string; n_ok?: number; n?: number }>("/api/evaluation/sync", { confirm: "1" })) ?? {};
    if (r.ok === false || r.error) throw new Error(String(r.error ?? "refused"));
    toast.success(`Synced: ${formatCount(r.n_ok)} of ${formatCount(r.n)} objects reconstructed`);
    refreshTargets();
    return true;
  } catch (e) {
    toast.error("FASRC sync: failed", { description: errorText(e) });
    return false;
  }
}

/** Drop the evaluation run's cached eye/solar PNGs, so the catalogue objects'
 *  images re-render from their FITS (older gallery renders). Local files only,
 *  no cluster round-trip; the renders come back on the next view. */
export async function dropCachedPngs(): Promise<boolean> {
  const yes = await confirm({
    title: "Drop the cached eye/solar PNGs?",
    message: "Deletes the evaluation run's cached eye and solar PNG renders in data/eval_results. The FITS stay; each image re-renders from them the next time it is shown.",
    confirmLabel: "Drop cached PNGs",
  });
  if (!yes) return false;
  try {
    const r = (await apiPost<{ ok?: boolean; error?: string; removed?: number }>("/api/evaluation/rerender", {})) ?? {};
    if (r.ok === false || r.error) throw new Error(String(r.error ?? "refused"));
    toast.success(`Dropped ${plural(r.removed ?? 0, "cached PNG")}`);
    refreshTargets();
    return true;
  } catch (e) {
    toast.error("Cached PNGs: failed", { description: errorText(e) });
    return false;
  }
}
