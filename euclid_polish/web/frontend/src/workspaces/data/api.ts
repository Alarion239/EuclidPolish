/* Data workspace — endpoint URLs, response types and shared resources
 * (backend: routes/{views,cutouts,psfs,tng}.py; documented in web/API.md). */
import { useResource } from "../../api/query";

/* ── records (routes/views.py) ─────────────────────────────────────────── */

export const SPLITS = ["test", "validate", "train"] as const;
export type Split = (typeof SPLITS)[number];
export const RECORD_KINDS = ["dirty", "hr", "clean"] as const;
export const SYNC_KINDS = ["dirty", "hr", "clean", "sources"] as const;

export type RecordFile = { name: string; size_bytes: number; mtime: number; count?: number | null };
export type SrState = "current" | "stale" | "partial" | "missing" | "unknown";
export type ModelIdentity = { member_labels: string[]; combiner_kind: string | null; combiner_fingerprint: string | null };
export type SrManifest = {
  subset: string; count: number; model_label: string | null; generated_at: string;
  identity: ModelIdentity | null; records: { name: string; size: number; mtime_ns: number } | null;
};
export type SplitInfo = {
  files: { dirty: RecordFile | null; hr: RecordFile | null; clean: RecordFile | null; sources: RecordFile | null };
  count: number;
  present: boolean;
  sr: { state: SrState; reasons: string[]; count: number; records_count: number | null; manifest: SrManifest | null };
};
export type SrStatus = {
  records: boolean; checkpoint: boolean; can_generate: boolean; subsets: Split[];
  sr: Record<Split, number>; records_dir: string;
  splits: Record<Split, SplitInfo>;
  model: ModelIdentity | null;
  sync_job: string | null; generate_job: string | null;
};

export type Grid = { height: number; width: number; pixscale: number };
export type Geometry = { hr: Grid | null; lr: Grid | null };
export type SourceType = "galaxy" | "star" | "lens" | "other";
export type TruthSource = {
  row: number; type: SourceType | string; render: string | null;
  x_pix: number | null; y_pix: number | null; off_field: boolean;
  flux_vis_e: number | null; flux_y_e: number | null; flux_j_e: number | null; flux_h_e: number | null;
  mag_vis: number | null; mag_y_e: number | null; mag_j_e: number | null; mag_h_e: number | null;
  target_vis_mag: number | null; z: number | null; re_arcsec: number | null; theta_E_arcsec: number | null;
  orientation: number | null; temperature_k: number | null;
  subhalo_id: string | null; source_subhalo_id: string | null; sfr_class: string | null;
};
export type SourceCounts = { galaxy: number; star: number; lens: number; other: number; off_field: number };
export type RecordSources = {
  subset: Split; field_index: number; present: boolean; sources: TruthSource[]; counts: SourceCounts; geometry: Geometry;
};
export type FieldCensus = SourceCounts & {
  field_index: number; n: number; brightest_star_mag: number | null; brightest_galaxy_mag: number | null; total_vis_e: number;
};
export type SourcesCensus = { subset: Split; present: boolean; fields: FieldCensus[]; geometry: Geometry };
export type SourceDetail = {
  subset: Split; field_index: number; row: number; source: TruthSource; values: Record<string, unknown>;
};

export const URLS = {
  srStatus: "/api/sky/sr-status",
  sync: "/api/sky/sync",
  generateSr: "/api/sky/generate-sr",
  sources: (split: string, index?: number) =>
    `/api/sky/records/sources?subset=${encodeURIComponent(split)}${index != null ? `&index=${index}` : ""}`,
  source: (split: string, index: number, row: number) =>
    `/api/sky/records/source?subset=${encodeURIComponent(split)}&index=${index}&row=${row}`,
  stars: "/api/catalog/stars",
  refreshCatalog: "/api/status/refresh-catalog",
  totals: "/api/star-cutouts/totals",
  gallery: (band: string, page: number, perPage = 48) =>
    `/api/cutouts/${encodeURIComponent(band)}/list.json?page=${page}&per_page=${perPage}`,
  cutoutImage: (band: string, file: string, size: number, outputDir: string) =>
    `/cutout-image/${encodeURIComponent(band)}/${encodeURIComponent(file)}?size=${size}&output_dir=${encodeURIComponent(outputDir)}`,
  psfInventory: "/api/euclid-psf/inventory",
  psfSync: "/api/euclid-psf/sync",
  psfSyncMeta: "/api/euclid-psf/sync-meta",
  config: "/api/config",
  tngProperties: "/api/tng/properties",
  tngPropertiesRefresh: "/api/tng/properties/refresh",
  tngResults: "/api/tng/results",
  tngPull: "/api/tng/result/pull",
  tngGrid: "/tng/result/grid.png",
  tngStack: "/tng/result/stack.fits",
  tngRadii: "/api/tng/radii/status",
  tngRadiiRefresh: "/api/tng/radii/refresh",
  tngAuth: "/tng-auth/status",
  alerts: "/api/system/alerts",
} as const;

/* ── star catalogue (routes/cutouts.py) ────────────────────────────────── */

export type StarsPayload = {
  present: boolean; source: string; path: string | null; local_path: string | null;
  size_bytes: number | null; mtime: number | null; age_s: number | null;
  columns: string[]; rows: (number | string | null)[][]; bands: string[]; sizes: number[];
  bits?: { valid: number; corrupted: number; failed: number; size_shift: number };
  summary: {
    total: number; valid: number; corrupted: number; failed: number; pending: number; valid_all4: number;
    navigator: { size: number | null; count: number }; mag_min: number | null; mag_max: number | null;
  } | null;
  band_stats: { band: string; valid: number; corrupted: number; failed: number; pending: number; by_size: Record<string, number> }[];
};
export type Totals = {
  count: number; size: number | null; cached: boolean;
  catalog: { present: boolean; path: string | null; mtime: number | null; age_s: number | null };
};
export type GalleryItemRow = { file: string; id: number | null; size: number | null; ra: number | null; dec: number | null; mag: number | null };
export type GalleryPage = {
  band: string; files: string[]; items: GalleryItemRow[]; total: number; page: number; n_pages: number;
  per_page: number; output_dir: string;
};

/* ── PSFs (routes/psfs.py) ─────────────────────────────────────────────── */

export type PsfState = "empirical" | "no_empirical" | "not_cached";
export type PsfBand = {
  name: string; fwhm: number; oversampling: number; epsf_pixel_scale: number; state: PsfState; empirical: boolean;
  path?: string; size_bytes?: number; synced_at?: number; n_psf?: number; shape?: [number, number];
  pixel_scale?: number | null; measured_fwhm?: number | null; error?: string | null;
  last_sync: { ok: boolean; error?: string | null; missing_remote?: boolean; checked_at?: number } | null;
};
export type PsfCluster = {
  index: number; id: string; ra: number | null; dec: number | null; n_stars: number | null;
  fwhm_by_band: Record<string, number | null>;
};
export type PsfInventory = {
  bands: PsfBand[]; clusters: PsfCluster[]; clusters_source: "metadata" | "vis_headers" | null;
  clusters_meta: { present: boolean; synced_at: number | null }; last_sync: number | null;
};

/* ── TNG (routes/tng.py) ───────────────────────────────────────────────── */

export type TngFile = { present: boolean; name: string; rows: number | null; mtime: number | null; size_bytes?: number };
export type TngPayload = {
  present: boolean; files: { properties: TngFile; atlas: TngFile }; atlas_meta: Record<string, unknown> | null;
  columns: string[]; rows: (number | string | null)[][];
  orientations: Record<string, [number | null, number | null, number | null][]>;
  summary: { n: number; n_quenched: number; n_missing_sfr: number; n_in_atlas: number; n_local: number };
};
export type TngResults = {
  grid: { present: boolean; pulled_at: number | null; size_bytes: number | null };
  stack: { present: boolean; pulled_at: number | null; size_bytes: number | null };
  pull_job: string | null;
};
export type TngRadius = {
  valid: boolean; connected?: boolean; expected_count?: number; valid_count?: number; failed_count?: number;
  reasons?: string[]; stale?: boolean; cached?: boolean; refresh_job?: string | null; checked_at?: number;
};
export type TngAuth = { present: boolean; connected: boolean; chars?: number };

/* ── shared resources (one cache entry each) ───────────────────────────── */

const MIN = 60_000;

export const useSrStatus = (poll?: number) => useResource<SrStatus>(URLS.srStatus, [], { ttl: 30_000, poll });
export const useStars = () => useResource<StarsPayload>(URLS.stars, [], { ttl: 10 * MIN });
export const usePsfInventory = () => useResource<PsfInventory>(URLS.psfInventory, [], { ttl: 5 * MIN });
export const useTngProperties = () => useResource<TngPayload>(URLS.tngProperties, [], { ttl: 10 * MIN });

/** Everything a Data job may have changed. */
export const DATA_PREFIXES = ["/api/sky/", "/api/catalog/", "/api/star-cutouts/", "/api/cutouts/", "/api/euclid-psf/",
  "/api/tng/", "/api/status"] as const;
