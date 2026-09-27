/* Figures workspace endpoints (euclid_polish/web/API.md: Viewer › Saved viewer
 * results, Poster cutout, Figures). Types, URLs and the mutations; every
 * mutation invalidates what it changed. */
import { apiPost } from "../../api/client";
import { invalidate } from "../../api/query";

export type FigureTier = "dirty" | "sr" | "hr" | "bhr" | "jwst";
export type FigureMode = "VIS" | "H_E" | "VIS_H" | "native";
export type FigureRegime = "real" | "synthetic";
export type RecipeKey = `${FigureTier}:${FigureMode}`;

export type SavedObject = {
  label?: string; id?: string; ref?: string; field?: string; ra?: number; dec?: number;
  grade?: string | number | null; subdir?: string | null; tiers?: string[];
};

export type SavedSource = {
  collection?: string;
  regime?: FigureRegime;
  index?: number;
  params?: Record<string, string>;
  object?: SavedObject | null;
  viewer_tiers?: string[];
};

export type SavedFile = {
  filename?: string; shape_hwc?: number[]; bands?: string[]; pixscale_arcsec?: number;
  source_tier?: string; source_label?: string; display_scale?: number | null;
  direct_rgb?: boolean; transfer_group?: string; wcs?: boolean;
};

export type RecipeOption = { tier: string; mode: string; key: string; label: string };

export type SavedSelection = {
  u?: number; v?: number; mode?: "angular" | "relative";
  angular_side_arcsec?: number | null; relative_side?: number | null; source_tier?: string;
};

export type SavedResult = {
  id: string;
  created_utc?: string | null;
  label: string;
  default_label?: string;
  regime?: FigureRegime;
  source?: SavedSource;
  selection?: SavedSelection | null;
  logical_tiers: string[];
  bands: Record<string, string[]>;
  pixscale_arcsec: Record<string, number>;
  recipes: RecipeKey[];
  recipe_options?: RecipeOption[];
  thumbnail?: string | null;
  files?: Record<string, SavedFile>;
  bytes?: number;
  inspect_paths?: Record<string, string>;
  display?: Record<string, unknown>;
  wcs_preserved: boolean;
  wcs_tiers?: string[];
  center?: { ra: number; dec: number } | null;
};

export type ResultsIndex = {
  schema_version?: number;
  limits?: { max_results?: number; max_rows?: number };
  supported?: { logical_tiers?: string[]; modes?: string[]; dpi?: { preview?: number; default?: number; min?: number; max?: number } };
  results: SavedResult[];
};

export type GridLayout = {
  id: string; name: string; results: string[]; rows: string[];
  regime?: FigureRegime | null; created_utc?: string; updated_utc?: string;
};

export type PlateTileRecord = {
  index: number; id?: string | null; ref?: string | null; file: string;
  ra_deg?: number | null; dec_deg?: number | null; model_state?: string | null;
  legacy?: boolean; sr_label?: string | null;
};

export type PlateRender = {
  band: string; model: string | null; model_label?: string | null; model_short?: string | null;
  model_fingerprint?: string | null; model_available?: boolean; legacy?: boolean;
  field_id?: string | null; filter?: string | null; created?: string | null;
  sheet: string | null; tiles: PlateTileRecord[];
};

export type PlateFile = {
  name: string; size: number; kind: "tile" | "sheet"; band: string;
  tile_index: number | null; model_slug: string | null;
};

export type PlateRun = { tag: string; updated: string | null; renders: PlateRender[]; files: PlateFile[] };

export type PlatesIndex = {
  root: string; runs: PlateRun[]; bands: string[];
  defaults: { tiles: number[]; band: string; max_tiles: number };
};

export type PosterFile = { size: number; mtime: number; pulled_at: string } | null;
export type PosterStatus = { ok: boolean; available: boolean; png: PosterFile; fits: PosterFile; archive_dir?: string };

export const URLS = {
  results: "/viewer/results",
  result: (id: string) => `/viewer/results/${encodeURIComponent(id)}`,
  layouts: "/viewer/grid-layouts",
  plates: "/api/figures/nexus-plates",
  poster: "/poster/result/status",
  models: "/api/models",
  nexusTiles: "/api/real/nexus",
};

/** `panel.png` of a saved result: a recipe (`tier:mode`) or the result's own
 *  thumbnail recipe, optionally downsampled to `size` px. */
export function panelUrl(id: string, recipe?: string | null, size?: number): string {
  const q = new URLSearchParams();
  if (recipe) {
    const [tier, mode] = recipe.split(":");
    q.set("tier", tier);
    q.set("mode", mode);
  }
  if (size) q.set("size", String(Math.round(size)));
  const s = q.toString();
  return `/viewer/results/${encodeURIComponent(id)}/panel.png${s ? `?${s}` : ""}`;
}

export function fitsUrl(id: string, tier: string): string {
  return `/viewer/results/${encodeURIComponent(id)}/${encodeURIComponent(tier)}.fits`;
}

export function gridUrl(ids: readonly string[], rows: readonly string[], format: "png" | "pdf", dpi: number, inline = false): string {
  const q = new URLSearchParams();
  for (const id of ids) q.append("result", id);
  for (const row of rows) q.append("row", row);
  q.set("dpi", String(dpi));
  if (inline) q.set("inline", "1");
  return `/viewer/results/grid.${format}?${q.toString()}`;
}

export function plateFileUrl(tag: string, name: string, opts: { thumb?: number; download?: boolean } = {}): string {
  const q = new URLSearchParams();
  if (opts.thumb) q.set("thumb", String(opts.thumb));
  if (opts.download) q.set("download", "1");
  const s = q.toString();
  return `/api/figures/nexus-plates/${encodeURIComponent(tag)}/${encodeURIComponent(name)}${s ? `?${s}` : ""}`;
}

/* ─── mutations ───────────────────────────────────────────────────────────── */

type Ok<T> = { ok?: boolean; error?: string } & T;

function refreshResults() {
  invalidate(URLS.results);
  invalidate(URLS.layouts);
}

export async function renameResult(id: string, label: string): Promise<SavedResult> {
  const r = await apiPost<Ok<{ result: SavedResult }>>(`${URLS.result(id)}/rename`, { label }, { json: true });
  if (r.ok === false || !r.result) throw new Error(r.error || "rename failed");
  refreshResults();
  return r.result;
}

/** Delete several results; resolves with the ids deleted and the failures. */
export async function deleteResults(ids: readonly string[]): Promise<{ deleted: string[]; failed: [string, string][] }> {
  const deleted: string[] = [];
  const failed: [string, string][] = [];
  for (const id of ids) {
    try {
      const r = await apiPost<Ok<{ id?: string }>>(`${URLS.result(id)}/delete`, {});
      if (r.ok === false) failed.push([id, r.error || "delete failed"]);
      else deleted.push(id);
    } catch (e) {
      failed.push([id, e instanceof Error ? e.message : String(e)]);
    }
  }
  refreshResults();
  return { deleted, failed };
}

export async function saveLayout(layout: { name: string; results: readonly string[]; rows: readonly string[]; regime?: FigureRegime; id?: string }):
  Promise<{ layout: GridLayout; created: boolean }> {
  const r = await apiPost<Ok<{ layout: GridLayout; created: boolean }>>(URLS.layouts, layout, { json: true });
  if (r.ok === false || !r.layout) throw new Error(r.error || "could not save the layout");
  invalidate(URLS.layouts);
  return { layout: r.layout, created: r.created };
}

export async function deleteLayout(id: string): Promise<void> {
  const r = await apiPost<Ok<object>>(`${URLS.layouts}/${encodeURIComponent(id)}/delete`, {});
  if (r.ok === false) throw new Error(r.error || "could not delete the layout");
  invalidate(URLS.layouts);
}

export async function deletePlateRun(tag: string): Promise<void> {
  const r = await apiPost<Ok<object>>(`${URLS.plates}/${encodeURIComponent(tag)}/delete`, {});
  if (r.ok === false) throw new Error(r.error || "could not delete the plate run");
  invalidate(URLS.plates);
}

export async function pullPoster(): Promise<{ archived?: string | null; errors?: Record<string, string> }> {
  const r = await apiPost<Ok<{ archived?: string | null; errors?: Record<string, string> }>>("/poster/result/pull", {});
  if (r.ok === false) throw new Error(r.error || "pull failed");
  invalidate(URLS.poster);
  return r;
}
