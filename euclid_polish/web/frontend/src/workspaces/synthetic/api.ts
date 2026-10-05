/* Synthetic workspace data (the realism checks, the priors, Status):
   payload types and the typed resources of every tab (all local, read-only
   GETs; the jobs are in ./jobs.ts). Records, the star catalogue, the PSFs
   and TNG are in ./dataApi.ts. */
import { useResource } from "../../api/query";
import type { ArchiveCollectionMeta } from "./archiveFields";
import type { Contour } from "./chartKit";

export const ENDPOINT = {
  overview: "/api/realism/overview",
  noise: "/api/noise",
  noisePosition: (tile: string) => `/api/noise/positions/${encodeURIComponent(tile)}`,
  galaxies: (training: boolean) => `/api/galaxy-distributions?include_training=${training ? "1" : "0"}`,
  jointPair: (x: string, y: string, revision: string) =>
    `/api/galaxy-distributions/joint-pair?x=${encodeURIComponent(x)}&y=${encodeURIComponent(y)}&r=${encodeURIComponent(revision)}`,
  stars: (training: boolean) => `/api/star-distribution?include_training=${training ? "1" : "0"}`,
  pixels: (training: boolean) => `/api/population-comparison?include_training=${training ? "1" : "0"}`,
  archiveMeta: "/viewer/meta/archive-fields",
  skyMeta: (subset: string) => `/viewer/meta/sky?subset=${encodeURIComponent(subset)}`,
  plate: (training: boolean) => `/view/galaxy-distribution-plate?include_training=${training ? "1" : "0"}`,
} as const;

/* ─── overview ──────────────────────────────────────────────────────────── */

export type CheckState = "ok" | "warn" | "bad" | "unknown";
export type ItemAction = {
  label: string; method: "POST"; url: string; params: Record<string, string>;
  confirm: string | null; requires_fasrc: boolean; self_connects?: boolean; requires_login: boolean;
};
/** Whether the local records were built with an ingredient (file times or
 *  the records' provenance; `unknown` when nothing can tell). */
export type RecordsTick = {
  state: "current" | "predates" | "unknown"; detail: string; records_at: string | null; prior_at: string | null;
};
export type OverviewItem = {
  id: string; label: string; state: CheckState; title: string; detail: string | null;
  to: string | null; action: ItemAction | null; facts: Record<string, unknown>;
  /** Synthetic › Status group: what generation reads, or a diagnostic cache. */
  group?: "generation" | "diagnostic";
  records?: RecordsTick | null;
};
export type OverviewPayload = {
  computed_at: string;
  gate: { step: string; ready: boolean; blockers: { id: string; message: string }[]; message: string | null; to: string };
  items: OverviewItem[];
  counts: Record<CheckState, number>;
  /** When the local test + validate records were generated (oldest shard). */
  records?: { generated_at: string | null; splits: string[] };
  authenticated: boolean;
  training: {
    available: boolean; population_fields: number | null; population_fields_with_training: number | null;
    sync: ItemAction;
  };
};

/* ─── noise ─────────────────────────────────────────────────────────────── */

export type Quantiles = { count: number; min: number; p5: number; p16: number; median: number; p84: number; p95: number; max: number };
export type NoisePayload = {
  bands: string[];
  source: {
    release: string; archive: string; description: string; units: string;
    retrieved_first: string; retrieved_last: string;
    tiles_attempted: number; position_count: number; unobserved_tiles: number; table_path: string;
  };
  generator: {
    noise_model: string; draws_measured_levels: boolean; scene_scale: [number, number] | null;
    region: null | { probability: number; fraction: [number, number]; step: [number, number] };
  };
  summary: Record<string, Quantiles & { pixel_scatter_ratio: number }>;
  fields: { name: string; positions: number; bands: Record<string, Quantiles> }[];
  histograms: Record<string, { log10_edges: number[]; counts_by_field: Record<string, number[]>; jittered_counts: number[] }>;
  within_field: null | {
    cutout_arcsec: number; sub_tile_arcsec: number; grid_side: number; step_edges: number[];
    step_threshold: number; uniformity_threshold: number;
    bands: Record<string, { fields: number; seam_count: number; seam_rate: number; counts: number[];
      steps: null | { p50: number; p90: number; max: number } }>;
  };
  log_correlation: number[][];
  positions: { field: string; tile: string; ra: number; dec: number; levels_e: number[] }[];
};
export type NoisePosition = {
  tile: string; field: string; ra: number; dec: number; bands: string[];
  levels_e: Record<string, number>; sub_levels_e: Record<string, (number | null)[]> | null;
  grid_side: number | null; steps: Record<string, null | { step: number; scatter: number; seam: boolean }>;
  step_threshold: number; uniformity_threshold: number; noise_model: string;
};

/* ─── galaxies ──────────────────────────────────────────────────────────── */

export type Curve = { x: number[]; density: number[]; weighted_count: number; definition: string };
export type BrightnessCurve = Curve & {
  label: string; survey: "euclid" | "synthetic" | "cosmos" | "fit" | "generation";
  band: string; estimator: string; selection: string; default_on?: boolean;
  fit_interval?: [number, number]; sampling_interval?: [number, number];
  generation_interval?: [number, number];
  generation_bright_join_magnitudes?: [number, number, number];
  generation_bright_slopes?: [number, number, number];
  generation_main_slope?: number; generation_break_magnitude?: number;
  generation_density_cap_arcmin2_mag?: number;
  trust_boundary?: {
    kind: "empirical_5sigma"; magnitude: number; lower_magnitude: number; upper_magnitude: number;
    snr: number; sample_size: number; estimator: string; selection: string; caveat: string;
  };
  observed_density_cap_arcmin2_mag?: number; observed_density_cap_magnitude?: number;
  observed_cumulative_density_to_boundary_arcmin2?: number;
  observed_cumulative_density_all_queried_bins_arcmin2?: number;
};
export type GalaxySourceKey = "euclid" | "synthetic" | "cosmos" | "fit";
export type RadiusCurve = Curve & {
  label: string; source: GalaxySourceKey;
  radius_type: "detection" | "kron" | "half_light" | "rendered_half_light" | "half_light_shape";
  units: string; normalization?: "surface_density" | "probability_density"; default_on?: boolean;
};
export type Parameter = {
  label: string; x_label: string; x_domain?: [number, number]; density_unit: string; note: string;
  series: Partial<Record<GalaxySourceKey, Curve>>;
  photometry_series?: Record<string, BrightnessCurve>; photometry_missing?: string[];
  radius_series?: Record<string, RadiusCurve>; radius_missing?: string[];
};
export type GalaxySource = {
  available?: boolean; detail?: string; rows?: number; area_arcmin2?: number; phz_pdf_rows?: number;
  phz_pdf_source?: string; fingerprint?: string; is_active?: boolean; validated?: boolean;
  measured_radius_rows?: number;
  /** Q1 (euclid) cache schema, and its population-cone count when the payload carries it. */
  schema_version?: number; cone_count?: number;
  /** Generated (synthetic) catalogue fields. */
  fields?: number;
};
export type Q1Counts = {
  footprint_area_deg2: number; bright: number; faint: number; bin_width: number; query_count: number;
  completed_queries?: number; total_queries?: number; complete?: boolean;
  phases_completed?: number; phase_count?: number; selection: string;
  apertures: Record<"f1" | "f2" | "f3" | "f4", { label: string; selected_galaxies: number; expected_galaxies?: number; queried_bins?: number }>;
};
export type GalaxyCandidate = {
  valid?: boolean; version: number; fingerprint: string;
  magnitude_law: { bright_join_magnitudes: [number, number, number]; bright_slopes: [number, number, number];
    break_magnitude: number; straight_law: { slope: number } };
  radius_law: { slope_log10_arcsec_per_mag: number; scatter_dex: number; fitted_rows: number };
  generation: {
    surface_density_arcmin2: number; differential_density_cap_arcmin2_mag: number; break_magnitude: number;
    fitted_surface_density_arcmin2: number; vis_magnitude_min: number; vis_magnitude_max: number;
    fitted_vis_magnitude_max: number; faint_end_policy: string;
  };
  aperture_fwhm_distribution?: { magnitude_edges: number[]; fwhm_edges_arcsec: number[]; probability: number[][];
    source_magnitude_bin: number[]; out_of_support_policy: string };
  color_sfr_model?: { row_count: number; tree_count: number; catalog_version: number; vis_snr_floor?: number;
    min_leaf_weight?: number; calibration_fingerprint: string };
  plots?: {
    conditional_radius?: {
      magnitude: number[]; observed_mean_log10_arcsec: (number | null)[]; model_mean_log10_arcsec: number[];
      model_core_low_log10_arcsec?: number[]; model_core_high_log10_arcsec?: number[];
      model_low_log10_arcsec?: number[]; model_high_log10_arcsec?: number[];
    };
    conditional_aperture_fwhm?: { magnitude: number[]; observed_mean_arcsec: (number | null)[];
      model_mean_arcsec: number[]; model_kind: string; out_of_support_policy: string };
    conditional_colors?: {
      magnitude: number[]; model_mean_vis_minus_y: number[]; model_mean_y_j: number[]; model_mean_j_h: number[];
      magnitude_edges?: number[]; observed_ratio_variance_by_magnitude?: (number | null)[][];
      noise_ratio_variance_by_magnitude?: (number | null)[][];
    };
  };
  provenance?: { color_sfr_valid_weight_fraction?: number; color_resolved_radius_weight_fraction?: number };
};
export type JointMap = {
  key: "q1" | "synthetic" | "model"; label: string; detail: string; color?: string;
  density: number[][]; surface_density_arcmin2: number; rows?: number | null; contours: Contour[];
};
export type JointMaps = {
  available: boolean; detail?: string; magnitude_edges: number[]; log_radius_edges: number[];
  density_unit: string; contour_mass_fractions: number[]; shared_density_max: number; maps: JointMap[];
};
export type CornerVariable = { key: string; label: string; unit: string; domain: [number, number];
  outside_fraction?: { q1: number; model: number } };
export type CornerCell = { row: number; col: number; source: "q1" | "model"; rows: number; contours: Contour[] };
export type CornerDiagonal = { edges: number[]; q1: number[]; model: number[] };
/** Whether the model's colour draws carry Q1 measurement noise (forward
 *  noising): the galaxy corner names the noised variables; the stellar
 *  comparison its donor count and the VIS window its colour panels share
 *  (model draws and generated stars both windowed and noised), or why they
 *  stayed intrinsic. Absent on a
 *  cache built before forward noising. */
export type ModelNoise = { applied: boolean; variables?: string[]; donors?: number; vis_window?: [number, number]; detail?: string };
export type CornerData = {
  available: boolean; detail?: string; variables?: CornerVariable[]; contour_mass_fractions?: number[];
  diagonal?: CornerDiagonal[]; cells?: CornerCell[]; q1_rows?: number; model_draws?: number; vis_range?: [number, number];
  model_noise?: ModelNoise | null;
};
export type PairLayer = { rows: number; density: number[][]; contours: Contour[] };
export type PairView = {
  available: boolean; detail?: string; kind?: "joint" | "marginal"; x?: CornerVariable; y?: CornerVariable;
  x_edges?: number[]; y_edges?: number[]; q1?: PairLayer; model?: PairLayer; diagonal?: CornerDiagonal;
  contour_mass_fractions?: number[]; vis_range?: [number, number]; model_noise?: ModelNoise | null;
};
export type Availability = {
  synthetic: {
    fields: number; area_arcmin2: number; record_files: number; source_catalogs: number;
    train_source_catalog: boolean; population_fields: number; population_area_arcmin2: number;
    population_fields_with_training: number; population_area_arcmin2_with_training: number;
  };
  real: {
    /** Every stored archive sample; `compared_fields` the offset tiles the statistics compare. */
    fields: number; compared_fields?: number; area_arcmin2: number; independent_parents: number; available: boolean; valid: boolean;
    ready: boolean; complete: boolean; current: boolean; unavailable_reason?: string | null;
    collection_fingerprint?: string | null;
  };
  field_area_arcmin2: number;
  input_fingerprint: string;
  comparison_cache: { present: boolean; schema_current: boolean; fresh: boolean; reason?: string | null };
};
export type GalaxyPayload = {
  version: number; stale: boolean; authenticated?: boolean;
  sources: Partial<Record<GalaxySourceKey, GalaxySource>>;
  q1_counts?: Q1Counts | null;
  q1_radius?: null | { complete: boolean; completed_queries: number; total_queries: number };
  calibration: { candidate: GalaxyCandidate | null; is_active: boolean; active?: null | { fingerprint?: string } };
  parameters: Record<string, Parameter>;
  joint_maps?: JointMaps;
  corner?: CornerData;
  training_included?: boolean;
  training_variant_available?: boolean;
  availability?: Availability;
};

/* ─── stars ─────────────────────────────────────────────────────────────── */

export type StarColorKey = "vis_y" | "vis_j" | "vis_h" | "y_j" | "y_h" | "j_h";
export type StarDensityKey = "vis" | StarColorKey;
/** One density panel. Only the VIS panel carries the Q1 point sources, the Q1 fit window and the
 *  native Gaia G_AB counts (`gaia` on the coarser `gaia_x` bins) with their shared-slope fit. */
export type StarDensityParameter = {
  label: string; x_label: string; x: number[]; x_domain: [number, number];
  euclid: number[]; model: number[]; synthetic: number[];
  point_sources?: number[] | null;
  gaia_x?: number[]; gaia?: number[] | null; gaia_fit?: number[] | null;
  fit_ranges?: { q1?: [number | null, number | null] };
};
export type StarDistribution = {
  /** The Gaia–Euclid colour sample the prior fits colours on. */
  matched_stars: number; high_quality_stars: number; pointlike_over_0_9: number;
  training_included?: boolean;
  density_comparison: null | {
    area_arcmin2: number; gaia_area_arcmin2?: number; model_density_arcmin2: number; model_sample_count: number;
    model_color_noise?: ModelNoise; euclid_vis_count: number; q1_phz_expected_stars: number | null; q1_expected_point_sources: number | null;
    q1_selected_point_sources: number | null; q1_area_arcmin2: number | null; euclid_color_count: number;
    synthetic_area_arcmin2?: number | null;
    synthetic_star_count?: number; parameters: Record<StarDensityKey, StarDensityParameter>; note: string;
  };
  gaia_sampling?: null | {
    field_count?: number; radius_arcmin: number; area_arcmin2: number;
    fields?: { name?: string; ra: number; dec: number; rows?: number }[];
  };
};
export type StarPayload = {
  authenticated: boolean;
  color_sample: { cached: boolean; euclid: null | { rows?: number; field_count?: number }; gaia: null | { rows?: number; field_count?: number } };
  calibration: {
    candidate: null | { valid?: boolean; warnings?: string[]; coverage_notes?: string[]; gaia?: { rows?: number };
      euclid_mapping?: { matched_stars?: number } };
    is_active: boolean;
  };
  distribution: StarDistribution | null;
  q1_counts: null | {
    footprint_area_deg2: number; selected_point_sources: number; expected_point_sources: number;
    expected_stars: number; bins: unknown[]; edges: number[]; selection: string;
  };
  availability?: Availability;
};

/* ─── pixels (field statistics) ─────────────────────────────────────────── */

export type Band = "VIS" | "Y_E" | "J_E" | "H_E";
export type Pctl = { p16: (number | null)[]; median: (number | null)[]; p84: (number | null)[] };
export type Interval = { median: number; p16: number; p84: number };
export type PointCloud = { x: number[]; y: number[]; parent_ids?: string[] };
export type Relation = { synthetic: PointCloud; real: PointCloud; x_label: string; y_label: string };
export type DetectionSide = {
  positive: number[]; negative: number[]; matched_galaxies: number[]; matched_stars: number[]; truth_galaxies: number[];
};
export type SourceDetection = {
  settings: { band: string; threshold_sigma: number; minimum_connected_pixels: number; background_box_pixels: number;
    deblend_levels: number; deblend_contrast: number; negative_image_correction: boolean;
    truth_match_radius_lr_pixels: number };
  synthetic: DetectionSide;
  real: DetectionSide;
};
export type FieldComparison = {
  bands: Band[];
  histograms: Record<Band, { x: number[]; synthetic: number[]; real: number[]; zero_bin: number | null; x_label: string; y_label: string }>;
  quantiles: Record<Band, { q: number[]; synthetic: number[]; real: number[]; x_label: string; y_label: string }>;
  power: Record<Band, { k: number[]; synthetic: Pctl; real: Pctl; x_label: string; y_label: string }>;
  scale_similarity: Record<Band, { k: number[]; log_shape_ratio: Pctl; overlap: Interval; variance_ratio: Interval; x_label: string; y_label: string }>;
  relations: Record<"mean_std" | "median_robust_std", Record<Band, Relation>>;
  band_correlation: { pairs: string[]; synthetic: Pctl; real: Pctl; x_label: string; y_label: string };
  summary: Record<"synthetic" | "real", Record<Band, Record<
    "mean" | "median" | "std" | "robust_std" | "p01" | "p99" | "zero_fraction" | "negative_fraction", Interval>>>;
  source_detection?: SourceDetection;
  sampling?: { bootstrap_draws?: number; synthetic_independent_parents?: number; real_independent_parents?: number };
};
export type Population = { objects: number; counts: Record<string, number>; density_arcmin2: Record<string, number | null>; area_arcmin2: number };
export type Comparison = {
  version: number;
  provenance?: { generated_at?: string };
  geometry: { tile_size: number; analysis_size: number; pixel_scale_arcsec: number; field_area_arcmin2: number };
  samples: {
    synthetic: { fields: number; area_arcmin2: number; splits: string[] };
    real: { fields: number; area_arcmin2: number; independent_parents: number };
  };
  fields: FieldComparison;
  population: { synthetic: Population; synthetic_field_count: number; synthetic_splits?: string[]; training_included?: boolean };
};
/** `comparison` is the cache at the current schema; `previous` the last one
 *  built at an older schema (shown with a stale badge), else null. */
export type PixelsPayload = {
  comparison: Comparison | null; previous?: Comparison | null; availability: Availability; authenticated: boolean;
};

/* ─── visual ────────────────────────────────────────────────────────────── */

export type SkyMeta = { count: number; tier_counts?: Record<string, number> };

/* ─── hooks ─────────────────────────────────────────────────────────────── */

export const useOverview = () => useResource<OverviewPayload>(ENDPOINT.overview, [], { ttl: 15_000 });
export const useNoise = () => useResource<NoisePayload>(ENDPOINT.noise, [], { ttl: 600_000 });
export const useNoisePosition = (tile: string | null) =>
  useResource<NoisePosition>(tile ? ENDPOINT.noisePosition(tile) : null, [tile], { ttl: 600_000 });
export const useGalaxies = (training: boolean, poll?: number) =>
  useResource<GalaxyPayload>(ENDPOINT.galaxies(training), [training], { ttl: 10_000, poll });
export const useJointPair = (x: string | null, y: string, revision: string) =>
  useResource<PairView>(x ? ENDPOINT.jointPair(x, y, revision) : null, [x, y, revision], { ttl: 60_000 });
export const useStars = (training: boolean, poll?: number) =>
  useResource<StarPayload>(ENDPOINT.stars(training), [training], { ttl: 10_000, poll });
export const usePixels = (training: boolean) =>
  useResource<PixelsPayload>(ENDPOINT.pixels(training), [training], { ttl: 10_000 });
export const useArchiveMeta = () => useResource<ArchiveCollectionMeta>(ENDPOINT.archiveMeta, [], { ttl: 30_000 });
export const useSkyMeta = (subset: string) => useResource<SkyMeta>(ENDPOINT.skyMeta(subset), [subset], { ttl: 30_000 });
