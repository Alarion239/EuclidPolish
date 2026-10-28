/* Typed endpoints of Models › Knees (knee attribution; spec
   docs/superpowers/specs/2026-10-15-knee-attribution-design.md › API):
   the runs list, the multi-knee model catalogue, one run record, the field
   images and the start of a run; the pixel types (§ Pixel types): the
   analyses and comparisons, their figures and the start of either. Every payload is read through
   `normalise*` in ./knees.ts (the server's record may be partial while a run
   is still writing), so a missing array never breaks the view. The real-tile
   choices reuse Sky's tile catalogue client (../sky/results/api). */
import { useResource } from "../../api/query";
import { URLS as SKY_URLS, type SourcesPayload, type TileList } from "../sky/results/api";

/* ── the contract (what the server sends) ──────────────────────────────── */

export type KneeRunStatus = "done" | "failed" | "running" | "cancelled";

/** One row of GET /api/knee-attribution (newest first). */
export type KneeRunSummary = {
  id: string; created?: string | null; finished?: string | null; status: KneeRunStatus | string;
  models: string[]; synthetic: number; tiles: string[]; per_class?: number | null;
  /** How the run sampled pixels (absent before version 4: per class). */
  sampling?: KneeSampling | string | null; per_band?: number | null;
  /** Runs with one seed sample the same uniform pixels of a field. */
  seed?: number | null; version?: number | null;
};

/** Per class (`per_class` of each class present), uniform (`per_band`
 *  pixels per band) or both pooled (weights halved). */
export type KneeSampling = "class" | "uniform" | "mixed";

/** One multi-knee catalogue model (GET /api/knee-attribution/models). */
export type KneeModelChoice = {
  spec: string; label: string; knees: number[]; depth: number | null;
  kind: "universal" | "member" | string; available: boolean; reason?: string | null;
};

/** One class's mean shares over the run's fields (one output band). The
 *  gradient shares are noise-weighted (record version 2): each input band's
 *  chain-rule terms times its noise σ, so a share is the output's response to
 *  a 1σ change of the input. They sum over the receptive field; the
 *  `*_local` ones (version 3) only over the 3×3 LR pixels under the sample. */
export type ClassShares = {
  knee_share: number[]; ablation_share: number[]; band_share: number[];
  /** C × K (rows: input bands VIS, Y, J, H; columns: the knees): each input
   *  band's own split over the knees, each row summing to 1 (σ cancels);
   *  knee_share = Σ_c band_share_c · row_c. Null before version 2. */
  knee_share_by_input?: number[][] | null;
  /** The local (3×3) knee split, band split and C × K by-input split; null
   *  before version 3. */
  knee_share_local?: number[] | null;
  band_share_local?: number[] | null;
  knee_share_by_input_local?: number[][] | null;
  n: number;
};

/** One output band's brightness bins (null: no sample in the bin). */
export type BrightnessBins = {
  /** Per bin the mean knee share. */
  knee_share: (number[] | null)[];
  /** Per bin the mean C × K by-input split; the whole list null before version 2. */
  knee_share_by_input?: (number[][] | null)[] | null;
  /** Per bin the local (3×3) knee split and C × K by-input split; null before version 3. */
  knee_share_local?: (number[] | null)[] | null;
  knee_share_by_input_local?: (number[][] | null)[] | null;
  n: number[];
};

/** The run's aggregate over all fields of one set (synthetic or real). */
export type KneeAggregate = {
  /** band index ("0".."3") → class name → its mean shares. */
  by_class: Record<string, Record<string, ClassShares>>;
  by_brightness: {
    /** Half-decade bin edges in e⁻ (21 edges, 1e-3 … 1e7). */
    edges_e: number[];
    /** band index → its bins. */
    bands: Record<string, BrightnessBins>;
  };
  /** Mean band share: rows = output band, cols = input band. */
  cross_band: (number | null)[][];
  /** The same from the local (3×3) band shares; null before version 3. */
  cross_band_local?: (number | null)[][] | null;
  n_samples: number;
};

/** One field's sampled pixels, as parallel arrays (one entry per sample). */
export type KneeSamples = {
  band: number[]; y: number[]; x: number[]; class: string[]; value_e: number[];
  knee_share: number[][]; ablation_share: number[][]; band_share: number[][]; sensitivity: number[];
  /** The local (3×3) knee split per sample (absent before version 3). */
  knee_share_local?: number[][];
};

export type KneeField = {
  /** `test/17` (a synthetic test field) or a real ref `source/id`. */
  ref: string; kind: "synthetic" | "real";
  /** The HR (SR) shape [H, W]: the samples' (y, x) coordinate system. */
  shape: [number, number];
  /** The field PNG's URL (`?band=` and `?layer=` are appended here). */
  image?: string | null;
  /** The field's noise σ per input band [VIS, Y, J, H] in e⁻: the weights of
   *  its shares (null before version 2). */
  noise_sigma_e?: number[] | null;
  samples: KneeSamples;
};

export type KneeModelResult = {
  knees: number[]; bands: string[]; depth?: number | null;
  aggregate: { synthetic?: KneeAggregate | null; real?: KneeAggregate | null };
  fields: KneeField[];
};

export type KneeRunParams = { models: string[]; synthetic: number; tiles: string[]; per_class: number; seed: number };

/** GET /api/knee-attribution/<id>. */
export type KneeRunRecord = {
  /** 3: adds the local (3×3) splits; 2: noise-weighted shares with the
   *  per-input split; absent (or 1): per input electron, which leans every
   *  share toward NISP. */
  version?: number | null;
  id: string; created?: string | null; finished?: string | null; status: KneeRunStatus | string;
  params: KneeRunParams;
  knee_grid_note?: string | null;
  models: Record<string, KneeModelResult>;
  /** `<model>` or `<model>|<field>` → the reason it failed. */
  errors: Record<string, string>;
};

/** POST /api/knee-attribution's reply. */
export type KneeRunStart = { ok: boolean; job_id?: string; run_id?: string; error?: string };

/** One finished pixel-types analysis (GET /api/knee-attribution/pixel-types):
 *  one model of one run; `files` its figures (`knee_map_synthetic.png`,
 *  `groups_learned_real_VIS.png`, …) and `summary.json`. */
export type PixelAnalysis = {
  /** `<run id>/<model slug>`. */
  folder: string; run_id: string; model: string; created: string;
  kinds: ("synthetic" | "real")[]; files: string[];
  params: Partial<{ sampling: string; per_band: number; per_class: number; seed: number; synthetic: number; tiles: string[] }>;
};

/** One finished comparison: model B against model A on the same pixels. */
export type PixelComparison = {
  /** `compare_<A>_vs_<B>[_<tag>]`. */
  folder: string;
  /** name → spec, A first. */
  models: Record<string, string>;
  /** name → run id. */
  runs: Record<string, string>;
  created: string; tag: string; kinds: ("synthetic" | "real")[]; files: string[];
};

export type PixelReports = {
  analyses: PixelAnalysis[]; comparisons: PixelComparison[];
  /** Several models' knee curves (e.g. one run's milestones), same shape as a comparison. */
  overviews: PixelComparison[];
};

/* ── URLs ──────────────────────────────────────────────────────────────── */

const enc = encodeURIComponent;

export const kneeUrl = {
  runs: () => "/api/knee-attribution",
  models: () => "/api/knee-attribution/models",
  run: (id: string) => `/api/knee-attribution/${enc(id)}`,
  /** POST: start a run (form: models, synthetic, tiles, per_class, seed). */
  start: () => "/api/knee-attribution",
  /** A field image by slugs (when the record carries no `image` URL). */
  field: (id: string, modelSlug: string, refSlug: string) =>
    `/api/knee-attribution/${enc(id)}/field/${enc(modelSlug)}/${enc(refSlug)}.png`,
  /** The pixel-types analyses and comparisons. */
  pixelTypes: () => "/api/knee-attribution/pixel-types",
  /** One figure or summary of an analysis/comparison folder; `v` (the
   *  folder's time) makes a re-analysis a new URL. */
  pixelFile: (folder: string, name: string, v?: string) =>
    `/api/knee-attribution/pixel-types/file/${folder.split("/").map(enc).join("/")}/${enc(name)}${v ? `?v=${enc(v)}` : ""}`,
  /** POST: analyse a run (form: run_id, models?). */
  pixelAnalyse: () => "/api/knee-attribution/pixel-types/analyse",
  /** POST: compare two models (form: a, b = `<run id>|<spec>`, tag?). */
  pixelCompare: () => "/api/knee-attribution/pixel-types/compare",
  /** Sky's real-tile catalogue: the sources and one source's tiles. */
  sources: () => SKY_URLS.sources,
  tiles: (source: string) => SKY_URLS.list(source),
};

/** The real-tile sources the run panel offers (the poster galaxy, the NEXUS
 *  tiles and the cached Q1 tiles). */
export const KNEE_TILE_SOURCES = ["poster", "nexus", "tile"] as const;

/* ── reads (GET only: opening the tab never starts a job) ──────────────── */

const MIN = 60_000;
/** The runs, newest first; `poll` (ms) while a run is being written. */
export const useKneeRuns = (poll?: number) =>
  useResource<{ runs?: unknown }>(kneeUrl.runs(), [], { ttl: 30_000, poll });
export const useKneeModels = (enabled = true) =>
  useResource<{ models?: unknown }>(enabled ? kneeUrl.models() : null, [], { ttl: MIN });
/** One run record (null id: idle); `poll` (ms) while it runs. */
export const useKneeRun = (id: string | null | undefined, poll?: number) =>
  useResource<unknown>(id ? kneeUrl.run(id) : null, [id], { ttl: 5 * MIN, poll });
export const useTileSources = (enabled: boolean) =>
  useResource<SourcesPayload>(enabled ? kneeUrl.sources() : null, [], { ttl: MIN });
/** One source's tiles (null: idle, e.g. the source has none). */
export const useTileList = (source: string | null) =>
  useResource<TileList>(source ? kneeUrl.tiles(source) : null, [source], { ttl: MIN });

/** The pixel-types analyses and comparisons; `poll` (ms) while a job runs. */
export const usePixelReports = (enabled: boolean, poll?: number) =>
  useResource<unknown>(enabled ? kneeUrl.pixelTypes() : null, [], { ttl: 30_000, poll });
/** One file (a native view, stamps, pairs) of an analysis or comparison
 *  folder (null: idle). */
export const usePixelFile = (folder: string | null, name: string | null, v?: string) =>
  useResource<unknown>(folder && name ? kneeUrl.pixelFile(folder, name, v) : null, [folder, name, v], { ttl: 10 * MIN });
/** One analysis' or comparison's summary.json (null folder: idle). */
export const usePixelSummary = (folder: string | null, v?: string) =>
  useResource<unknown>(folder ? kneeUrl.pixelFile(folder, "summary.json", v) : null, [folder, v], { ttl: 5 * MIN });

export type { SourcesPayload, TileList };
