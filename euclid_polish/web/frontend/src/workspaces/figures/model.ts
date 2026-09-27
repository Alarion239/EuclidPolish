/* Pure logic of the Figures workspace: the saved-results index, recipes and
 * presets of the grid, links back to each result's source viewer, and the
 * NEXUS plate form (tile lists, model coverage). Unit-tested in model.test.ts. */
import type {
  FigureMode, FigureRegime, FigureTier, GridLayout, PlateRender, PlateRun, RecipeKey,
  ResultsIndex, SavedResult,
} from "./api";
import { MAX_GRID_RESULTS, MAX_GRID_ROWS } from "./grid/limits";

/* ─── recipes and presets ─────────────────────────────────────────────────── */

export const TIERS: { value: FigureTier; label: string }[] = [
  { value: "dirty", label: "Dirty / LR" },
  { value: "sr", label: "SR" },
  { value: "hr", label: "HR truth" },
  { value: "bhr", label: "Blurred HR" },
  { value: "jwst", label: "JWST reference" },
];

export const MODES: { value: FigureMode; label: string }[] = [
  { value: "VIS", label: "VIS" },
  { value: "H_E", label: "H_E" },
  { value: "VIS_H", label: "VIS + H_E" },
  { value: "native", label: "native band" },
];

const TIER_SET = new Set<string>(TIERS.map((t) => t.value));
const MODE_SET = new Set<string>(MODES.map((m) => m.value));

export type Preset = { id: string; label: string; regime: FigureRegime; rows: RecipeKey[] };

export const PRESETS: Preset[] = [
  { id: "real-vis-h", label: "Real · VIS/H composite", regime: "real",
    rows: ["dirty:VIS", "dirty:H_E", "sr:VIS_H", "jwst:native"] },
  { id: "real-bandwise", label: "Real · bandwise SR", regime: "real",
    rows: ["dirty:VIS", "sr:VIS", "dirty:H_E", "sr:H_E", "jwst:native"] },
  { id: "real-input-composite", label: "Real · input composite", regime: "real",
    rows: ["dirty:VIS", "dirty:VIS_H", "sr:VIS_H", "jwst:native"] },
  { id: "synthetic-bandwise", label: "Synthetic · bandwise", regime: "synthetic",
    rows: ["dirty:VIS", "sr:VIS", "hr:VIS", "dirty:H_E", "sr:H_E", "hr:H_E"] },
  { id: "synthetic-composite", label: "Synthetic · VIS/H composite", regime: "synthetic",
    rows: ["dirty:VIS_H", "sr:VIS_H", "hr:VIS_H"] },
];

export const DEFAULT_PRESET = PRESETS[0];

export function isRecipeKey(value: unknown): value is RecipeKey {
  if (typeof value !== "string") return false;
  const parts = value.split(":");
  return parts.length === 2 && TIER_SET.has(parts[0]) && MODE_SET.has(parts[1]);
}

export function splitRecipe(key: RecipeKey): [FigureTier, FigureMode] {
  const [tier, mode] = key.split(":");
  return [tier as FigureTier, mode as FigureMode];
}

/** The row title the grid prints (the backend's `_recipe_label`). */
export function recipeLabel(key: string): string {
  if (!isRecipeKey(key)) return key;
  const [tier, mode] = splitRecipe(key);
  if (tier === "jwst" && mode === "native") return "NEXUS F200W";
  const band = mode === "VIS_H" ? "VIS + H_E" : mode === "native" ? "Native" : mode;
  const product = tier === "dirty" ? "Dirty" : tier === "bhr" ? "BHR" : tier === "jwst" ? "JWST" : tier.toUpperCase();
  return `${band} ${product}`;
}

/** Colour family of a recipe row (the grid's spectrum rule). */
export function modeTone(mode: string): "vis" | "h" | "vis-h" | "native" {
  return mode === "VIS_H" ? "vis-h" : mode === "VIS" ? "vis" : mode === "H_E" ? "h" : "native";
}

/* ─── the saved-results index ─────────────────────────────────────────────── */

const SYNTHETIC_COLLECTIONS = new Set(["sky", "evaluation", "ensemble", "psfs"]);
const REAL_COLLECTIONS = new Set(["archive-fields", "real-field", "jwst-euclid", "nexus-field", "real", "cutouts"]);

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function strings(v: unknown): string[] {
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : [];
}

function stringMap(v: unknown): Record<string, string[]> {
  if (!isRecord(v)) return {};
  return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, strings(x)]));
}

function numberMap(v: unknown): Record<string, number> {
  if (!isRecord(v)) return {};
  return Object.fromEntries(Object.entries(v).filter((e): e is [string, number] => typeof e[1] === "number" && Number.isFinite(e[1])));
}

function positiveInt(v: unknown): number | undefined {
  return typeof v === "number" && Number.isSafeInteger(v) && v > 0 ? v : undefined;
}

function normalizeResult(v: unknown): SavedResult | null {
  if (!isRecord(v) || typeof v.id !== "string" || !v.id.trim()) return null;
  const source = isRecord(v.source) ? (v.source as SavedResult["source"]) : undefined;
  const objLabel = isRecord(source?.object) && typeof source?.object?.label === "string" ? source.object.label : undefined;
  const label = typeof v.label === "string" && v.label.trim() ? v.label : objLabel || v.id;
  const center = isRecord(v.center) && typeof v.center.ra === "number" && typeof v.center.dec === "number"
    ? { ra: v.center.ra, dec: v.center.dec } : null;
  return {
    ...(v as Partial<SavedResult>),
    id: v.id,
    label,
    default_label: typeof v.default_label === "string" ? v.default_label : label,
    regime: v.regime === "real" || v.regime === "synthetic" ? v.regime : undefined,
    source,
    logical_tiers: strings(v.logical_tiers),
    bands: stringMap(v.bands),
    pixscale_arcsec: numberMap(v.pixscale_arcsec),
    recipes: strings(v.recipes).filter(isRecipeKey),
    wcs_preserved: v.wcs_preserved === true,
    wcs_tiers: strings(v.wcs_tiers),
    center,
  };
}

export type NormalizedIndex = {
  results: SavedResult[];
  maxResults: number;
  maxRows: number;
  malformed: boolean;
  dropped: number;
  tiers: FigureTier[];
  modes: FigureMode[];
};

/** A defensive read of `GET /viewer/results` (malformed rows are dropped
 *  and counted, never crash the page). */
export function normalizeIndex(value: unknown): NormalizedIndex {
  const base = { maxResults: MAX_GRID_RESULTS, maxRows: MAX_GRID_ROWS, tiers: TIERS.map((t) => t.value), modes: MODES.map((m) => m.value) };
  if (value == null) return { ...base, results: [], malformed: false, dropped: 0 };
  if (!isRecord(value) || !Array.isArray(value.results)) return { ...base, results: [], malformed: true, dropped: 0 };
  const all = value.results.map(normalizeResult);
  const results = all.filter((r): r is SavedResult => r !== null);
  const index = value as Partial<ResultsIndex>;
  const tiers = strings(index.supported?.logical_tiers).filter((t) => TIER_SET.has(t)) as FigureTier[];
  const modes = strings(index.supported?.modes).filter((m) => MODE_SET.has(m)) as FigureMode[];
  return {
    results,
    maxResults: positiveInt(index.limits?.max_results) ?? MAX_GRID_RESULTS,
    maxRows: positiveInt(index.limits?.max_rows) ?? MAX_GRID_ROWS,
    malformed: false,
    dropped: all.length - results.length,
    tiers: tiers.length ? TIERS.map((t) => t.value).filter((t) => tiers.includes(t)) : base.tiers,
    modes: modes.length ? MODES.map((m) => m.value).filter((m) => modes.includes(m)) : base.modes,
  };
}

export function resultRegime(r: SavedResult): FigureRegime | null {
  if (r.regime) return r.regime;
  const c = r.source?.collection;
  if (c && SYNTHETIC_COLLECTIONS.has(c)) return "synthetic";
  if (c && REAL_COLLECTIONS.has(c)) return "real";
  if (r.recipes.some((k) => k.startsWith("jwst:"))) return "real";
  if (r.recipes.some((k) => k.startsWith("hr:"))) return "synthetic";
  return null;
}

export function missingRecipes(r: SavedResult, rows: readonly string[]): string[] {
  return rows.filter((row) => !r.recipes.includes(row as RecipeKey));
}

/** The recipes every result supports (the grid rows that can render). */
export function commonRecipes(results: readonly SavedResult[]): RecipeKey[] {
  if (!results.length) return [];
  const [first, ...rest] = results;
  return first.recipes.filter((k) => rest.every((r) => r.recipes.includes(k)));
}

/** The grid's state for the status line and whether it can render.
 *  `missing`: some cells are not available — the sheet renders anyway, those
 *  cells grey "Not available" in place (`gridUrl(…, missing)`). */
export type GridStatus = { canRender: boolean; tone: "good" | "warn" | "bad"; text: string; unsupported: number; missing: boolean };

export function gridStatus(opts: {
  loading: boolean; error: boolean; results: readonly SavedResult[]; columns: readonly string[];
  rows: readonly string[]; maxResults: number; maxRows: number;
}): GridStatus {
  const byId = new Map(opts.results.map((r) => [r.id, r]));
  const cols = opts.columns.map((id) => byId.get(id)).filter((r): r is SavedResult => !!r);
  const unsupported = cols.reduce((n, r) => n + missingRecipes(r, opts.rows).length, 0);
  const done = (canRender: boolean, tone: GridStatus["tone"], text: string): GridStatus =>
    ({ canRender, tone, text, unsupported, missing: canRender && unsupported > 0 });
  if (opts.loading && !opts.results.length) return done(false, "warn", "Loading saved crops…");
  if (opts.error) return done(false, "bad", "Saved crops unavailable");
  if (opts.columns.length > opts.maxResults) return done(false, "bad", `At most ${opts.maxResults} columns`);
  if (opts.rows.length > opts.maxRows) return done(false, "bad", `At most ${opts.maxRows} rows`);
  if (cols.length !== opts.columns.length) return done(false, "bad", "Some columns are no longer saved");
  if (!cols.length) return done(false, "warn", "Pick at least one saved crop");
  if (!opts.rows.length) return done(false, "warn", "Add at least one row");
  if (unsupported >= cols.length * opts.rows.length) return done(false, "bad", "No cell is available: no column has these rows");
  if (unsupported) return done(true, "warn", `${unsupported} cell${unsupported === 1 ? "" : "s"} not available (grey in the sheet)`);
  return done(true, "good", `${opts.rows.length} × ${cols.length} ready`);
}

/** Keep only saved columns of the regime, at most `max`. */
export function sanitizeColumns(columns: readonly string[], results: readonly SavedResult[], regime: FigureRegime, max: number): string[] {
  const byId = new Map(results.map((r) => [r.id, r]));
  const out: string[] = [];
  for (const id of columns) {
    const r = byId.get(id);
    if (r && resultRegime(r) === regime && !out.includes(id) && out.length < max) out.push(id);
  }
  return out;
}

export function moveItem<T>(list: readonly T[], index: number, offset: -1 | 1): T[] {
  const target = index + offset;
  if (index < 0 || target < 0 || target >= list.length) return [...list];
  const next = [...list];
  [next[index], next[target]] = [next[target], next[index]];
  return next;
}

/** The grid tab opened on these columns (`/figures/grid?regime&cols`). */
export function gridHref(ids: readonly string[], regime: string): string {
  const q = new URLSearchParams({ regime, cols: ids.join(",") });
  return `/figures/grid?${q.toString()}`;
}

/** The template picker value of a layout (`layout:<id>`). */
export const layoutValue = (l: Pick<GridLayout, "id">) => `layout:${l.id}`;

/* ─── links back to the source ────────────────────────────────────────────── */

export type SourceLink = { to: string; label: string };

function q(params: Record<string, string | undefined>): string {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v != null && v !== "") p.set(k, v);
  const s = p.toString();
  return s ? `?${s}` : "";
}

/** Where a saved result's source viewer lives now (the viewers' URL keys:
 *  `v.<key>.id` / `.i`, or the Sky results `realtile` inspector). */
export function viewerLink(r: SavedResult): SourceLink | null {
  const src = r.source;
  if (!src?.collection) return null;
  const obj = src.object ?? {};
  const params = src.params ?? {};
  const id = typeof obj.id === "string" && obj.id ? obj.id : undefined;
  const index = typeof src.index === "number" ? src.index : undefined;
  const pos = (key: string) => (id ? { [`v.${key}.id`]: id } : index != null ? { [`v.${key}.i`]: String(index) } : {});
  const realtile = (ref: string) => ({ to: `/sky/results${q({ inspect: `realtile:${ref}` })}`, label: "Open the real tile" });
  switch (src.collection) {
    case "real": {
      const ref = obj.ref ?? (id && params.source ? `${params.source}/${id}` : undefined);
      return ref ? realtile(ref) : null;
    }
    case "nexus-field": {
      const n = id?.match(/(\d+)$/)?.[1] ?? (index != null ? String(index) : undefined);
      return n != null ? realtile(`nexus/f200w-${n.padStart(4, "0").slice(-4)}`) : null;
    }
    case "real-field": {
      const m = id?.match(/^(.+)\/(\d+)$/);
      return m ? realtile(`field/${m[1]}-${m[2].padStart(3, "0")}`) : null;
    }
    case "jwst-euclid":
      return id ? realtile(`pair/${id}`) : null;
    case "evaluation":
      return { to: `/sky/catalog-eval${q(pos("cev"))}`, label: "Open in Catalog eval" };
    case "ensemble": {
      const mode = params.mode === "starless" ? "starless" : "starfull";
      return { to: `/ensemble/${mode}/disagreement${q(pos("ens"))}`, label: "Open in Disagreement" };
    }
    case "sky":
      return { to: `/data/records${q({ ...pos("sky"), subset: params.subset })}`, label: "Open in Records" };
    case "cutouts":
      return { to: `/data/cutouts${q(pos("cutouts"))}`, label: "Open in Cutouts" };
    case "psfs":
      return { to: `/data/psfs${q(pos("psfs"))}`, label: "Open in PSFs" };
    case "archive-fields":
      return { to: `/realism/visual${q(pos("real"))}`, label: "Open in Visual" };
    default:
      return null;
  }
}

/** The source shown on badges: the C9 real-tile source (`nexus`, `tile`, …)
 *  for the `real` collection, else the viewer collection. */
export function sourceLabel(r: SavedResult): string {
  const c = r.source?.collection;
  if (c === "real") return r.source?.params?.source || "real";
  return c || "viewer";
}

/** The Sky atlas at the result's centre (saved position, else the object's). */
export function skyLink(r: SavedResult): string | null {
  const ra = r.center?.ra ?? r.source?.object?.ra;
  const dec = r.center?.dec ?? r.source?.object?.dec;
  if (typeof ra !== "number" || typeof dec !== "number" || !Number.isFinite(ra) || !Number.isFinite(dec)) return null;
  const p = new URLSearchParams({ ra: (((ra % 360) + 360) % 360).toFixed(6), dec: dec.toFixed(6), fov: "0.01" });
  return `/sky/atlas?${p.toString()}`;
}

export function inspectLink(r: SavedResult, tier: string): string | null {
  const path = r.inspect_paths?.[tier];
  return path ? `/inspect?path=${encodeURIComponent(path)}` : null;
}

/** Crop side in arcsec (the saved selection, else the first tier's size). */
export function cropSideArcsec(r: SavedResult): number | null {
  const a = r.selection?.angular_side_arcsec;
  if (typeof a === "number" && a > 0) return a;
  for (const [tier, f] of Object.entries(r.files ?? {})) {
    const scale = r.pixscale_arcsec[tier] ?? f.pixscale_arcsec;
    const side = f.shape_hwc?.[0];
    if (typeof scale === "number" && typeof side === "number") return side * scale;
  }
  return null;
}

export type WcsState = "all" | "partial" | "none";
export function wcsState(r: SavedResult): WcsState {
  if (r.wcs_preserved) return "all";
  return (r.wcs_tiers ?? []).length ? "partial" : "none";
}

/* ─── NEXUS plates ────────────────────────────────────────────────────────── */

/** Tile references from free text: `40, 42 70` → ["40","42","70"]; ids and
 *  refs (`f200w-0040`, `nexus/f200w-0040`) pass through; duplicates dropped. */
export function parseTileList(text: string): string[] {
  const out: string[] = [];
  for (const raw of text.split(/[\s,;]+/)) {
    const item = raw.trim().replace(/^nexus\//, "");
    if (!item) continue;
    const key = /^\d+$/.test(item) ? String(Number(item)) : item;
    if (!out.includes(key)) out.push(key);
  }
  return out;
}

export type NexusTile = { id: string; ref: string; models?: Record<string, { state?: string; legacy?: boolean }> };

/** The NEXUS tile a typed token names (number → `…-NNNN`, else the id). */
export function matchTile(token: string, tiles: readonly NexusTile[]): NexusTile | null {
  if (/^\d+$/.test(token)) {
    const n = Number(token);
    return tiles.find((t) => Number(t.id.match(/(\d+)$/)?.[1]) === n) ?? null;
  }
  return tiles.find((t) => t.id === token) ?? null;
}

export type Coverage = { tiles: NexusTile[]; unknown: string[]; missing: string[]; stale: string[] };

/** Which picked tiles hold an output of `spec` (and whether it is current). */
export function plateCoverage(tokens: readonly string[], tiles: readonly NexusTile[], spec: string): Coverage {
  const out: Coverage = { tiles: [], unknown: [], missing: [], stale: [] };
  for (const token of tokens) {
    const tile = matchTile(token, tiles);
    if (!tile) { out.unknown.push(token); continue; }
    out.tiles.push(tile);
    const m = tile.models?.[spec];
    if (!m) out.missing.push(tile.id);
    else if (m.state && m.state !== "current") out.stale.push(tile.id);
  }
  return out;
}

/** Specs with an output on every picked tile, best first. */
export function specsCoveringAll(tiles: readonly NexusTile[], specs: readonly string[]): string[] {
  return specs.filter((s) => tiles.length > 0 && tiles.every((t) => !!t.models?.[s]));
}

export function tileNumberLabel(r: { index: number }): string {
  return String(r.index).padStart(3, "0");
}

/** Human title of one render of a run. */
export function renderTitle(r: PlateRender): string {
  const band = r.band === "temp" ? "temperature" : r.band;
  return `${band} · ${r.legacy ? "legacy SR" : r.model_short || r.model || "model"}`;
}

/** Stable key of one render in a run (`band~model`). */
export function renderKey(r: PlateRender): string {
  return `${r.band}~${r.model ?? "legacy"}`;
}

export function findRender(run: PlateRun | undefined, key: string): PlateRender | undefined {
  if (!run) return undefined;
  return run.renders.find((r) => renderKey(r) === key) ?? run.renders[0];
}

/* ─── full-size view vs the in-page preview ───────────────────────────────── */

/** "6 rows × 1 column". */
export function gridSizeText(rows: number, columns: number): string {
  return `${rows} row${rows === 1 ? "" : "s"} × ${columns} column${columns === 1 ? "" : "s"}`;
}

/** The full-size view's geometry (css px), fed to figures.css as custom
 *  properties so the CSS and this arithmetic cannot drift: the dialog is the
 *  window minus `inset` on every side; one header row (title, description,
 *  scale, actions, close) of `head`; a crop's panel choices add `panels`. */
export const LIGHTBOX = { inset: 12, head: 44, panels: 36 } as const;

/** The grid preview's cap (css px), also fed to figures.css: beside the
 *  editors it sits `stickyTop` below the app top bar under a `head` row;
 *  stacked above them it keeps `stackedChrome` for the tab strip and the bar
 *  (never below `stackedMin`); the paper pads the image by `pad` a side. */
export const PREVIEW = { stickyTop: 8, head: 48, stackedChrome: 200, stackedMin: 320, pad: 8 } as const;

/** The full-size stage height for a window `vh` tall. */
export function lightboxStageHeight(vh: number, opts: { panels?: boolean } = {}): number {
  return vh - 2 * LIGHTBOX.inset - LIGHTBOX.head - (opts.panels ? LIGHTBOX.panels : 0);
}

/** The tallest the in-page grid preview image gets in a window `vh` tall. */
export function previewPaperCap(vh: number, topbar: number, stacked: boolean): number {
  const box = stacked ? Math.max(PREVIEW.stackedMin, vh - topbar - PREVIEW.stackedChrome) : vh - topbar - PREVIEW.stickyTop - PREVIEW.head;
  return box - 2 * PREVIEW.pad;
}

/** The Results gallery's text filter: every word must appear in the label,
 *  id, source object, source or tiers (case-insensitive). */
export function galleryMatches(r: SavedResult, query: string): boolean {
  const words = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return true;
  const o = r.source?.object;
  const hay = [r.label, r.id, o?.id, o?.label, sourceLabel(r), r.source?.collection, ...r.logical_tiers]
    .filter((v) => v != null && v !== "").join(" ").toLowerCase();
  return words.every((w) => hay.includes(w));
}
