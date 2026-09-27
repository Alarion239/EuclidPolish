/* Atlas actions: external links (pure) and the remote / mutating actions
 * (confirm → POST → local job in the tray → refresh the layers when it ends).
 * Jobs are registered in the global jobs store, so they survive closing the
 * menu, the inspector or the page; the shell toasts their end. */
import { apiGet, apiPost, ApiError } from "../../../api/client";
import { isTerminal, refreshJobsFeed, useJobsStore, type Job } from "../../../api/jobs";
import { invalidate } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { formatDec, formatRA } from "../../../format";
import { confirm, toast } from "../../../ui";
import { fmtCoord } from "./urlState";

/* ── pure helpers ────────────────────────────────────────────────────── */

export function esaskyUrl(ra: number, dec: number, fovDeg = 0.1): string {
  const p = new URLSearchParams({
    target: `${fmtCoord(ra)} ${fmtCoord(dec)}`, fov: String(Number(Math.max(fovDeg, 0.003).toPrecision(3))),
    cooframe: "J2000", sci: "true", lang: "en",
  });
  return `https://sky.esa.int/esasky/?${p.toString()}`;
}

export function simbadUrl(ra: number, dec: number, radiusArcmin = 1): string {
  const p = new URLSearchParams({
    Coord: `${fmtCoord(ra)} ${dec >= 0 ? "+" : ""}${fmtCoord(dec)}`,
    Radius: String(radiusArcmin), "Radius.unit": "arcmin", submit: "submit query",
  });
  return `https://simbad.cds.unistra.fr/simbad/sim-coo?${p.toString()}`;
}

export function coordText(ra: number, dec: number, mode: "deg" | "sex" = "deg"): string {
  if (mode === "sex") return `${formatRA(ra, { style: "colon" })} ${formatDec(dec, { style: "colon" })}`;
  return `${fmtCoord(ra)} ${dec >= 0 ? "+" : ""}${fmtCoord(dec)}`;
}

/** The discovery region of a view: `ra,dec,radius_deg` (radius ≤ 10°). */
export function viewRegion(view: { ra: number; dec: number; fov: number }): string {
  const r = Math.min(10, Math.max(0.01, view.fov / 2));
  return `${fmtCoord(view.ra)},${fmtCoord(view.dec)},${Number(r.toPrecision(3))}`;
}

const where = (ra: number, dec: number) => `${formatRA(ra)} ${formatDec(dec)}`;

/* ── jobs ────────────────────────────────────────────────────────────── */

export type StartResult =
  | { ok: true; jobId: string | null; body: Record<string, unknown> }
  | { ok: false; error: string; code?: string; status?: number };

function watchJob(id: string, onDone: (j: Job) => void): void {
  const check = (): boolean => {
    const j = useJobsStore.getState().jobs[id];
    if (j && isTerminal(j.status)) { onDone(j); return true; }
    return false;
  };
  if (check()) return;
  const unsub = useJobsStore.subscribe(() => { if (check()) unsub(); });
}

/** POST a job-starting endpoint; the job joins the tray (`key` re-attaches it). */
export async function startJob(
  url: string, data: Record<string, string | number | boolean | null | undefined>,
  opts: { key?: string; onDone?: (j: Job) => void } = {},
): Promise<StartResult> {
  try {
    const body = (await apiPost<Record<string, unknown>>(url, data)) ?? {};
    if (body.error || body.ok === false) {
      return { ok: false, error: String(body.error ?? "request failed"), code: body.code ? String(body.code) : undefined };
    }
    const id = body.job_id != null ? String(body.job_id) : null;
    if (id) {
      useJobsStore.getState().register(id, opts.key);
      void refreshJobsFeed();
      if (opts.onDone) watchJob(id, opts.onDone);
    }
    return { ok: true, jobId: id, body };
  } catch (e) {
    if (e instanceof ApiError) return { ok: false, error: e.message, code: e.code, status: e.status };
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
}

const refreshSky = () => { void invalidate("/api/sky/"); void invalidate("/api/real/"); };

function reportStart(r: StartResult, what: string): boolean {
  if (!r.ok) { toast.error(`${what}: ${r.error}`); return false; }
  toast.info(`${what} started`, { description: "Follow it in the job tray." });
  return true;
}

/* ── the actions ─────────────────────────────────────────────────────── */

type AtResponse = { q1_verdict?: "observed" | "unobserved" | "outside"; best_tile?: string | null; field?: string | null };

/** Cache a 25.6″ four-band tile at (ra, dec) — checks Q1 coverage first. */
export async function cacheTileAt(ra: number, dec: number, opts: { run?: boolean } = {}): Promise<StartResult | null> {
  let verdict: AtResponse["q1_verdict"];
  let tile: string | null | undefined;
  try {
    const at = await apiGet<AtResponse>(`/api/sky/at?ra=${fmtCoord(ra)}&dec=${fmtCoord(dec)}`);
    verdict = at.q1_verdict;
    tile = at.best_tile;
  } catch (e) {
    toast.error(`Could not check Q1 coverage: ${e instanceof Error ? e.message : String(e)}`);
    return null;
  }
  if (verdict === "outside") {
    toast.error(`${where(ra, dec)} is outside Euclid Q1: no tile can be cached there.`);
    return null;
  }
  const force = verdict === "unobserved";
  const ok = await confirm({
    title: force ? "Cache a tile on an unobserved Q1 tile?" : "Cache a 25.6″ tile here?",
    message: force
      ? `Every Q1 tile containing ${where(ra, dec)} was measured unobserved by the noise campaign; the cutout is probably blank.`
      : `Downloads VIS + NISP Y/J/H at ${where(ra, dec)} from the Euclid archive${tile ? ` (Q1 tile ${tile})` : ""}${opts.run ? ", then runs the production model and the mean" : ""}.`,
    confirmLabel: force ? "Cache anyway" : "Cache tile",
    tone: force ? "danger" : undefined,
  });
  if (!ok) return null;
  const r = await startJob("/api/real/tiles", {
    ra: fmtCoord(ra), dec: fmtCoord(dec), run: opts.run ? "production,mean" : undefined, force: force ? 1 : undefined,
  });
  if (reportStart(r, "Tile cache") && r.ok && r.jobId) {
    // When it lands, open the new tile's card: its LR beside the SR of the
    // models it ran (production + mean with "run"), ready to compare or overlay.
    const ref = typeof r.body?.ref === "string" ? r.body.ref : null;
    watchJob(r.jobId, (j) => {
      refreshSky();
      const id = (j.result as { id?: unknown } | null | undefined)?.id;
      const target = ref ?? (typeof id === "string" ? `tile/${id}` : null);
      if (j.status !== "done" || !target) return;
      openInspector({ kind: "tile", id: target });
      toast.success("Tile cached", { description: opts.run ? "LR, production and mean are open in the inspector." : "Its LR is open in the inspector." });
    });
  }
  return r;
}

/** Download + align a JWST × Euclid pair at a position or for a MAST observation. */
export async function downloadPair(at: { ra: number; dec: number } | { obs_id: string }, sizeArcsec = 30): Promise<StartResult | null> {
  const label = "obs_id" in at ? `observation ${at.obs_id}` : where(at.ra, at.dec);
  const ok = await confirm({
    title: "Download a JWST × Euclid pair?",
    message: `Fetches the JWST imaging and the Euclid tile at ${label} (${sizeArcsec}″) and aligns them (a local job).`,
    confirmLabel: "Download pair",
  });
  if (!ok) return null;
  const data = "obs_id" in at
    ? { obs_id: at.obs_id, size_arcsec: sizeArcsec }
    : { ra: fmtCoord(at.ra), dec: fmtCoord(at.dec), size_arcsec: sizeArcsec };
  const r = await startJob("/api/sky/jwst/pair", data, { onDone: refreshSky });
  reportStart(r, "JWST × Euclid pair");
  return r;
}

/** Discover JWST imaging overlapping Euclid Q1 (MAST) in a region or fields. */
export async function discoverJwst(scope: { region?: string; fields?: string; label: string; refresh?: boolean }): Promise<StartResult | null> {
  const ok = await confirm({
    title: "Discover JWST observations?",
    message: `Queries MAST for JWST imaging in ${scope.label} and intersects it with the Q1 tiles (a local job; results join the "JWST MAST footprints" layer).`,
    confirmLabel: "Discover",
  });
  if (!ok) return null;
  const r = await startJob("/api/sky/jwst/discover", {
    region: scope.region, fields: scope.fields, refresh: scope.refresh ? 1 : undefined,
  }, { key: "sky:jwst-discover", onDone: refreshSky });
  reportStart(r, "JWST discovery");
  return r;
}

/** Run model specs on real tiles (an experiment job). */
export async function runModels(refs: readonly string[], specs: readonly string[]): Promise<StartResult | null> {
  if (!refs.length || !specs.length) return null;
  const ok = await confirm({
    title: `Run ${specs.join(", ")} on ${refs.length} tile${refs.length === 1 ? "" : "s"}?`,
    message: "Runs the ensemble locally (TensorFlow) and caches every SR with its real-data metrics.",
    confirmLabel: "Run",
  });
  if (!ok) return null;
  const r = await startJob("/api/experiments", { tiles: refs.join(","), models: specs.join(",") }, { onDone: refreshSky });
  reportStart(r, "Experiment");
  return r;
}

export async function cacheNexusMosaic(filter: "F200W" | "F444W"): Promise<StartResult | null> {
  const ok = await confirm({
    title: `Cache the NEXUS ${filter} mosaic?`,
    message: `Downloads the public NEXUS Deep Epoch 05 ${filter} mosaic (${filter === "F200W" ? "≈ 1 GB" : "≈ 250 MB"}) and the four-band Euclid coverage of every tile. Cached tiles are reused.`,
    confirmLabel: "Download",
  });
  if (!ok) return null;
  const r = await startJob("/api/jwst-euclid/nexus/download-field", { filter }, { key: "sky:nexus-mosaic", onDone: refreshSky });
  reportStart(r, "NEXUS mosaic");
  return r;
}

export async function downloadAllPairs(): Promise<StartResult | null> {
  const ok = await confirm({
    title: "Download every discovered pair?",
    message: "Downloads every discovered JWST × Euclid location not saved yet (30″, one local job).",
    confirmLabel: "Download all",
  });
  if (!ok) return null;
  const r = await startJob("/api/jwst-euclid/download-all", { size_arcsec: 30 }, { key: "sky:pairs-all", onDone: refreshSky });
  reportStart(r, "Pair downloads");
  return r;
}

/** A layer's own fill action (catalogue downloads, syncs…). */
export async function runFillAction(action: { url: string; label: string; requires_fasrc?: boolean }): Promise<StartResult | null> {
  const ok = await confirm({
    title: `${action.label}?`,
    message: action.requires_fasrc ? "Needs the FASRC connection." : "Runs as a local job.",
    confirmLabel: "Start",
  });
  if (!ok) return null;
  const r = await startJob(action.url, {}, { onDone: refreshSky });
  reportStart(r, action.label);
  return r;
}

type NexusField = { field_id: string; target_name?: string; count?: number; stale_sr_count?: number; current_sr_count?: number; sr_count?: number };

/** Run the production model on the stale tiles of the cached NEXUS field
 *  (`/api/jwst-euclid/nexus/infer`; the legacy JWST × Euclid page's action). */
export async function runNexusProduction(fieldId?: string): Promise<StartResult | null> {
  let fields: NexusField[];
  try {
    fields = (await apiGet<{ fields?: NexusField[] }>("/api/jwst-euclid/nexus/fields")).fields ?? [];
  } catch (e) {
    toast.error(`Could not list the NEXUS fields: ${e instanceof Error ? e.message : String(e)}`);
    return null;
  }
  const field = fieldId ? fields.find((f) => f.field_id === fieldId) : fields[0];
  if (!field) { toast.error("No NEXUS field is cached yet: cache the NEXUS mosaic first (JWST tools)."); return null; }
  const stale = field.stale_sr_count ?? field.count ?? 0;
  if (!stale) { toast.success(`Every ${field.target_name ?? "NEXUS"} tile already has a current production SR.`); return null; }
  const ok = await confirm({
    title: `Run production on ${stale} NEXUS tile${stale === 1 ? "" : "s"}?`,
    message: `Runs the production spatial gate (STARFULL) locally on the stale tiles of ${field.target_name ?? field.field_id} (${field.count ?? "?"} tiles; TensorFlow, a local job).`,
    confirmLabel: "Run production",
  });
  if (!ok) return null;
  const r = await startJob("/api/jwst-euclid/nexus/infer", { field_id: field.field_id }, { key: "sky:nexus-infer", onDone: refreshSky });
  reportStart(r, "NEXUS production");
  return r;
}

/** Build a saved pair's four-band LR input and run production on it
 *  (`/api/jwst-euclid/infer`; pairs saved before C9 are VIS-only until then). */
export async function buildPairInput(pairId: string, label = pairId): Promise<StartResult | null> {
  const ok = await confirm({
    title: "Build the four-band LR and run production?",
    message: `Cuts VIS + NISP Y/J/H for ${label} from the containing Q1 tile, then runs the production model on it (a local job).`,
    confirmLabel: "Build + run",
  });
  if (!ok) return null;
  const r = await startJob("/api/jwst-euclid/infer", { field_id: pairId }, { onDone: refreshSky });
  reportStart(r, "Pair production");
  return r;
}

/** A layer row's fill action: the ones with their own flow (discovery,
 *  NEXUS mosaic) go through it, the others are a confirmed POST. */
export function runLayerFill(action: { url: string; label: string; requires_fasrc?: boolean }): Promise<StartResult | null> {
  if (action.url === "/api/sky/jwst/discover") return discoverJwst({ label: "all of Euclid Q1" });
  if (action.url === "/api/jwst-euclid/nexus/download-field") return cacheNexusMosaic("F200W");
  return runFillAction(action);
}
