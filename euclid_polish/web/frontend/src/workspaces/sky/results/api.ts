/* Typed endpoints of the Sky › Real results / Experiments / Catalog-eval tabs
 * (contract C9 + /api/evaluation, euclid_polish/web/API.md). Pure types and
 * URL builders: the tabs GET through useResource and POST through the
 * actions in ./actions. */

export const SOURCES = ["nexus", "tile", "field", "archive", "eval", "poster", "pair"] as const;
export type SourceId = (typeof SOURCES)[number];
export const BANDS = ["VIS", "Y_E", "J_E", "H_E"] as const;
export type Band = (typeof BANDS)[number];
export type ModelState = "current" | "stale" | "unavailable" | "missing";

/* ── GET /api/real/sources ─────────────────────────────────────────────── */

export type SourceInfo = {
  id: string; label: string; description?: string; count: number;
  model_ready?: boolean; has_jwst?: boolean; ready?: boolean; reason?: string | null;
};
export type SourcesPayload = { sources: SourceInfo[] };

/* ── metrics (real_metrics, version 1) ─────────────────────────────────── */

export type BandMetrics = {
  hole_pct?: number | null; n_bright_px?: number | null;
  hole_pct_100sigma?: number | null; n_bright_100sigma_px?: number | null;
  flux_ratio?: number | null; lr_flux_e?: number | null; sr_flux_e?: number | null;
  background_e?: number | null; sigma_e?: number | null;
  n_peaks?: number | null; n_artifacts?: number | null; n_edge?: number | null;
  pct_R_lt_0p8?: number | null; pct_R_lt_0p5?: number | null;
  median_R?: number | null; min_R?: number | null;
};
export type MetricsSummary = {
  n_peaks?: number | null; hole_pct_max?: number | null; hole_pct_mean?: number | null;
  pct_R_lt_0p8?: number | null; pct_R_lt_0p5?: number | null; median_R?: number | null;
};
/** `{band: [[member label, mean weight], …]}` (top 3 over the brightest 1 %). */
export type GateCoreWeights = Record<string, [string, number][]>;
export type Metrics = {
  version?: number; n_tiles?: number; bands?: string[];
  per_band?: Record<string, BandMetrics>; summary?: MetricsSummary;
  gate_core_weights?: GateCoreWeights;
};

/* ── GET /api/real/<source> (rows) and /api/real/<source>/<id> (card) ─── */

export type TileModelRow = {
  state?: ModelState | string; legacy?: boolean; label?: string | null; fingerprint?: string | null;
  created?: string | null; experiment_id?: string | null; file?: string | null; origin?: string | null;
  summary?: MetricsSummary | null;
};
export type TileRow = {
  source: string; id: string; ref: string; label: string;
  ra: number | null; dec: number | null; field?: string | null;
  shape?: [number, number] | null; pixscale?: number | null; bands?: string[];
  model_ready?: boolean; tiers?: string[]; has_jwst?: boolean;
  polygon?: [number, number][];
  extras?: Record<string, unknown>;
  models?: Record<string, TileModelRow>;
  production_state?: "current" | "stale" | "missing" | string;
};
export type TileList = { source: string; label: string; description?: string; count: number; tiles: TileRow[] };

export type CardModel = TileModelRow & {
  kind?: string | null; member_labels?: string[] | null; combiner_kind?: string | null;
  lr_sha?: string | null; shape?: number[] | null; metrics?: Metrics | null; image_url?: string;
};
export type TileCard = Omit<TileRow, "models"> & {
  models?: Record<string, CardModel>;
  legacy?: Record<string, unknown> | null;
  runnable_models?: string[];
  experiments?: string[];
  disk?: { tile_bytes?: number; output_bytes?: number; cache_bytes?: number; legacy_bytes?: number; total_bytes?: number };
  q1_tile?: { tile?: string; field?: string | null; levels_e?: number[] | null; rejected?: string | null } | null;
  image_urls?: Record<string, string>;
  viewer?: { collection: string; params: Record<string, string>; id: string };
};

/* ── GET /api/models ───────────────────────────────────────────────────── */

export type ModelKind = "production" | "mean" | "rbf" | "member" | "gate";
export type ModelSpecRow = {
  spec: string; kind: ModelKind | string; label: string; slug?: string;
  members?: string[]; member_names?: string[]; reads?: string[]; n_members?: number;
  available: boolean; reason?: string | null; fingerprint?: string | null;
  combiner_kind?: string | null; combiner_fingerprint?: string | null;
  details?: Record<string, unknown>;
};
export type ModelsPayload = { regime: string; production_kind?: string; members?: string[]; models: ModelSpecRow[] };

/* ── /api/experiments ──────────────────────────────────────────────────── */

export type ExperimentStatus = "running" | "done" | "failed" | "cancelled";
export type ExperimentCounts = {
  members_computed?: number; members_reused?: number; members_not_cached?: number;
  members_evicted?: number; outputs_computed?: number; outputs_reused?: number;
};
export type ExperimentSummary = {
  id: string; label?: string | null; created?: string | null; finished?: string | null;
  status?: ExperimentStatus | string; job_id?: string | null; tiles?: string[]; models?: string[];
  skipped?: Record<string, string>; summary?: Record<string, Metrics>;
  errors?: Record<string, string>; counts?: ExperimentCounts;
};
export type ExperimentResult = {
  state?: "computed" | "reused" | string; fingerprint?: string | null; file?: string | null; metrics?: Metrics | null;
};
export type ExperimentRecord = ExperimentSummary & {
  version?: number; duration_s?: number | null;
  fingerprints?: Record<string, string | null>; model_labels?: Record<string, string>;
  definitions?: Record<string, unknown>;
  results?: Record<string, Record<string, ExperimentResult>>;
};
export type ExperimentsPayload = { experiments: ExperimentSummary[] };
export type ExperimentStart = {
  ok: boolean; job_id?: string; experiment_id?: string; tiles?: string[]; models?: string[];
  skipped?: Record<string, string>; error?: string;
};

/* ── /api/evaluation ───────────────────────────────────────────────────── */

export type EvalIdentity = {
  n_members: number; member_labels?: string[]; combiner_kind?: string | null; combiner_fingerprint?: string | null;
};
export type EvalRow = Record<string, string | number | null | string[] | undefined> & {
  id?: string; ra?: string; dec?: string; grade?: string; ok?: string; error?: string; out_subdir?: string;
  lr_total_e?: string; sr_total_e?: string; flux_ratio_sr_over_lr?: string; psnr_lr_hr?: string; psnr_sr_hr?: string;
  kind?: "lens" | "galaxy" | "synthetic" | string; field?: string | null; viewer_id?: string | null;
  realtile?: string | null; tiers?: string[];
  state?: "current" | "stale" | "unknown" | null; state_reason?: string | null;
  n_members?: number | null; combiner_kind?: string | null;
};
export type EvalRuns = {
  name: string; run: string; n: number; n_ok: number; mtime?: number; columns?: string[];
  rows: EvalRow[]; current?: EvalIdentity; counts?: { current: number; stale: number; unknown: number };
  groups?: Record<string, number>;
};
export type EvalObjectCard = EvalRow & {
  row?: Record<string, string> | null;
  members?: { member_labels?: string[]; combiner_kind?: string | null; combiner_fingerprint?: string | null } | null;
  current?: EvalIdentity;
  disagreement?: { pca_n?: number; pca_amps?: number[]; pca_var?: number[] } | null;
  files?: { name: string; bytes: number; mtime: number }[];
  provenance?: { id?: string; created_at?: string; produced_by?: string; git?: string; dirty?: boolean; file?: string }[];
  sr_header?: Record<string, string | number | boolean>;
  downloads?: Record<string, string>;
  viewer?: { collection: string; id: string };
};

/* ── URLs ──────────────────────────────────────────────────────────────── */

const enc = encodeURIComponent;

export const URLS = {
  sources: "/api/real/sources",
  list: (source: string) => `/api/real/${enc(source)}`,
  card: (ref: string) => {
    const [source, id] = splitRef(ref);
    return `/api/real/${enc(source)}/${enc(id)}`;
  },
  image: (ref: string, tier: string, band: string) => {
    const [source, id] = splitRef(ref);
    return `/api/real/${enc(source)}/${enc(id)}/image.fits?tier=${enc(tier)}&band=${enc(band)}`;
  },
  deleteOutputs: (ref: string) => {
    const [source, id] = splitRef(ref);
    return `/api/real/${enc(source)}/${enc(id)}/delete-outputs`;
  },
  cacheTile: "/api/real/tiles",
  models: "/api/models",
  experiments: "/api/experiments",
  experiment: (id: string) => `/api/experiments/${enc(id)}`,
  evalRuns: "/api/evaluation/runs",
  evalObject: (id: string) => `/api/evaluation/objects/${enc(id)}`,
  authStatus: "/auth/status",
  trackingLog: "/api/tracking/log",
  fieldStatus: "/api/inference/field.json",
  fieldDiagnostics: "/api/inference/diagnostics.json",
  fieldRefresh: "/inference/refresh-combiners",
  syntheticEvals: "/ensemble/evals.json?mode=starfull",
} as const;

/** `"nexus/f200w-0001"` → `["nexus", "f200w-0001"]` (ids never contain `/`). */
export function splitRef(ref: string): [string, string] {
  const i = ref.indexOf("/");
  return i < 0 ? [ref, ""] : [ref.slice(0, i), ref.slice(i + 1)];
}

/** The atlas URL centred on a position, inspecting the real tile. */
export function atlasHref(ra: number, dec: number, ref?: string): string {
  const q = new URLSearchParams({ ra: ra.toFixed(6), dec: dec.toFixed(6) });
  if (ref) q.set("inspect", `realtile:${ref}`);
  return `/sky/atlas?${q.toString()}`;
}

/** `/sky/experiments` preselecting tiles (the atlas builds the same link). */
export function experimentsHref(refs: readonly string[], models?: readonly string[]): string {
  const q = new URLSearchParams();
  if (refs.length) q.set("tiles", refs.join(","));
  if (models?.length) q.set("models", models.join(","));
  const s = q.toString();
  return s ? `/sky/experiments?${s}` : "/sky/experiments";
}
