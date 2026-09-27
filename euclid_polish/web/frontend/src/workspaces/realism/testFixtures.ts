/* Small hand-made payloads for the Realism tests, shaped like the real
   /api/realism/overview, /api/noise, /api/galaxy-distributions,
   /api/star-distribution, /api/population-comparison and the archive /
   sky viewer metas (checked against captured responses). Test-only. */
import type {
  Band, CornerData, FieldComparison, GalaxyPayload, JointMaps, NoisePayload, NoisePosition, OverviewPayload,
  PairView, PixelsPayload, StarDistribution, StarPayload,
} from "./api";
import type { ArchiveCollectionMeta } from "./archiveFields";

const BANDS: Band[] = ["VIS", "Y_E", "J_E", "H_E"];
const per = <T,>(f: (band: Band, i: number) => T) => Object.fromEntries(BANDS.map((b, i) => [b, f(b, i)])) as Record<Band, T>;
const q = (m: number) => ({ count: 3, min: m * 0.5, p5: m * 0.6, p16: m * 0.8, median: m, p84: m * 1.2, p95: m * 1.4, max: m * 2 });
const action = (label: string, url: string, patch: Record<string, unknown> = {}) => ({
  label, method: "POST" as const, url, params: {}, confirm: null, requires_fasrc: false, requires_login: false, ...patch,
});

export const OVERVIEW: OverviewPayload = {
  computed_at: "2026-09-26T12:00:00+00:00",
  gate: {
    step: "synthetic_generate", ready: false, message: "activate a valid Gaia+Euclid stellar calibration before generating fields",
    blockers: [{ id: "star-prior", message: "activate a valid Gaia+Euclid stellar calibration before generating fields" }],
    to: "/data/records",
  },
  items: [
    { id: "galaxy-model", label: "Galaxy joint model", state: "warn", title: "Galaxy candidate ready, not active",
      detail: "candidate gggggggggggg…", to: "/realism/galaxies",
      action: action("Activate model", "/api/galaxy-distributions/activate", { confirm: "Activate this galaxy candidate?" }),
      facts: { is_active: false, aperture_checkpoints: [560, 560] } },
    { id: "star-prior", label: "Stellar prior", state: "bad", title: "Stellar candidate needs a refit",
      detail: "refit required: stellar counts", to: "/realism/stars", action: null, facts: { is_active: false } },
    { id: "tng-radii", label: "TNG radius manifest", state: "unknown", title: "TNG radius manifest not validated", detail: null,
      to: "/data/tng", action: action("Validate on FASRC", "/api/tng/radii/refresh", { requires_fasrc: true }), facts: { cached: false } },
    { id: "noise-model", label: "Noise model", state: "ok", title: "Noise model v5", detail: "294 measured Q1 positions",
      to: "/realism/noise", action: null, facts: { noise_model: "mer-noise-v5", positions: 294 } },
    { id: "comparison-cache", label: "Field-statistics cache", state: "warn", title: "Field statistics are stale",
      detail: "comparison cache uses an older schema", to: "/realism/pixels",
      action: action("Rebuild statistics", "/api/population-comparison/build", { confirm: "Stream 376 fields?" }), facts: {} },
  ],
  counts: { ok: 1, warn: 2, bad: 1, unknown: 1 },
  authenticated: true,
  training: {
    available: false, population_fields: 200, population_fields_with_training: 200,
    sync: action("Sync training catalogue", "/api/population-comparison/sync-training-catalog", {
      confirm: "Pull sources_train.csv from FASRC and rebuild the galaxy plots?", self_connects: true, params: { rebuild: "1" },
    }),
  },
};

export const NOISE: NoisePayload = {
  bands: BANDS,
  source: {
    release: "Q1_R1", archive: "IRSA", description: "MER noise maps", units: "e⁻ per 0.1″ pixel",
    retrieved_first: "2026-09-19T18:22:46+00:00", retrieved_last: "2026-09-19T18:37:40+00:00",
    tiles_attempted: 300, position_count: 3, unobserved_tiles: 6, table_path: "euclid_polish/sky/observation/mer_noise_levels.json",
  },
  generator: { noise_model: "euclid-q1-mer-noise-levels-dithered-bilinear-v5", draws_measured_levels: true, scene_scale: [0.99, 1.01],
    region: { probability: 0.1, fraction: [0.2, 0.5], step: [1.1, 1.45] } },
  summary: per((_b, i) => ({ ...q(10 + i), pixel_scatter_ratio: 1.2 })),
  fields: [
    { name: "EDF-N", positions: 2, bands: per((_b, i) => q(10 + i)) },
    { name: "EDF-S", positions: 1, bands: per((_b, i) => q(12 + i)) },
  ],
  histograms: per(() => ({ log10_edges: [0.5, 1, 1.5, 2], counts_by_field: { "EDF-N": [1, 1, 0], "EDF-S": [0, 1, 0] }, jittered_counts: [0.9, 2.1, 0] })),
  within_field: {
    cutout_arcsec: 25.6, sub_tile_arcsec: 6.4, grid_side: 4, step_edges: [1, 1.1, 1.2, 1.5], step_threshold: 1.1, uniformity_threshold: 1.05,
    bands: per(() => ({ fields: 3, seam_count: 1, seam_rate: 1 / 3, counts: [2, 1, 0], steps: { p50: 1.02, p90: 1.15, max: 1.21 } })),
  },
  log_correlation: BANDS.map((_r, i) => BANDS.map((_c, j) => (i === j ? 1 : 0.4))),
  positions: [
    { field: "EDF-N", tile: "102018211", ra: 269.7, dec: 66.0, levels_e: [27.1, 11.2, 12.3, 13.4] },
    { field: "EDF-N", tile: "102018212", ra: 270.1, dec: 66.2, levels_e: [28.0, 11.9, 12.8, 13.9] },
    { field: "EDF-S", tile: "102021990", ra: 61.2, dec: -48.4, levels_e: [27.0, 10.9, 12.0, 13.1] },
  ],
};

export const NOISE_POSITION: NoisePosition = {
  tile: "102021990", field: "EDF-S", ra: 61.2, dec: -48.4, bands: BANDS,
  levels_e: { VIS: 27.05, Y_E: 10.9, J_E: 12.0, H_E: 13.1 },
  sub_levels_e: per((_b, i) => Array.from({ length: 16 }, (_, k) => 10 + i + (k < 4 ? 2 : 0))), grid_side: 4,
  steps: per((_b, i) => (i === 0 ? null : { step: 1.2, scatter: 1.01, seam: true })),
  step_threshold: 1.1, uniformity_threshold: 1.05, noise_model: "mer-noise-v5",
};

/* ─── galaxies ──────────────────────────────────────────────────────────── */

const mags = [18, 20, 22, 24, 26];
const curve = (definition: string, scale = 1) => ({
  x: mags, density: mags.map((m) => scale * 10 ** ((m - 18) / 4)), weighted_count: 1234, definition,
});
const logRe = [-1.5, -1, -0.5, 0, 0.5];
const reCurve = (definition: string, scale = 1) => ({
  x: logRe, density: logRe.map((l) => scale * Math.exp(-((l + 0.5) ** 2))), weighted_count: 800, definition,
});
const colorCurve = () => ({ x: [-0.5, 0, 0.5, 1], density: [0.1, 1, 0.8, 0.05], weighted_count: 500, definition: "2FWHM colours" });
const contour = (fraction: number, r = 1) => ({
  mass_fraction: fraction,
  paths: [{ x: [0, r, 0, -r, 0].map((v) => 21 + v), y: [r, 0, -r, 0, r].map((v) => -0.5 + 0.2 * v) }],
});

const TRUST = {
  kind: "empirical_5sigma" as const, magnitude: 25.3, lower_magnitude: 24.9, upper_magnitude: 25.7, snr: 5, sample_size: 12000,
  estimator: "median S/N per bracket", selection: "POINT_LIKE_FLAG IS NULL", caveat: "Depth varies across Q1.",
};

export const JOINT_MAPS: JointMaps = {
  available: true, magnitude_edges: [18, 20, 22, 24, 26], log_radius_edges: [-1.5, -1, -0.5, 0, 0.5],
  density_unit: "objects / arcmin² / mag / dex", contour_mass_fractions: [0.1, 0.5, 0.8, 0.95, 0.99, 0.995, 0.999],
  shared_density_max: 12,
  maps: [
    { key: "q1", label: "Q1 MER + PHZ", detail: "PHZ-weighted", density: [[1]], surface_density_arcmin2: 88.4, rows: null,
      contours: [0.1, 0.5, 0.8, 0.95, 0.99, 0.995, 0.999].map((f, i) => contour(f, 0.4 + i * 0.2)) },
    { key: "synthetic", label: "Current generated galaxies", detail: "test + validation", density: [[1]], surface_density_arcmin2: 80, rows: 5321,
      contours: [contour(0.5), contour(0.95, 1.6)] },
    { key: "model", label: "Active generation law", detail: "model", density: [[1]], surface_density_arcmin2: 90, rows: null,
      contours: [contour(0.5, 0.9), contour(0.95, 1.5)] },
  ],
};

const cornerContour = (f: number) => ({ mass_fraction: f, paths: [{ x: [19, 21, 23, 21, 19], y: [0, 0.5, 0, -0.5, 0] }] });

export const CORNER: CornerData = {
  available: true,
  variables: [
    { key: "vis", label: "VIS 2FWHM", unit: "AB mag", domain: [18, 26] },
    { key: "log_sfr", label: "log₁₀ SFR", unit: "M☉ yr⁻¹", domain: [-3, 2], outside_fraction: { q1: 0.03, model: 0.001 } },
    { key: "log_re", label: "log₁₀ Rₑ", unit: "arcsec", domain: [-1.5, 0.5] },
  ],
  contour_mass_fractions: [0.99, 0.95, 0.8, 0.5, 0.2],
  diagonal: [0, 1, 2].map(() => ({ edges: [0, 1, 2, 3].map((e) => e), q1: [0.2, 0.5, 0.3], model: [0.25, 0.45, 0.3] })),
  cells: [
    { row: 1, col: 0, source: "q1", rows: 4000, contours: [cornerContour(0.5)] },
    { row: 2, col: 0, source: "q1", rows: 3900, contours: [cornerContour(0.5)] },
    { row: 2, col: 1, source: "q1", rows: 3800, contours: [] },
    { row: 0, col: 1, source: "model", rows: 6000, contours: [cornerContour(0.5)] },
    { row: 0, col: 2, source: "model", rows: 6000, contours: [cornerContour(0.5)] },
    { row: 1, col: 2, source: "model", rows: 6000, contours: [cornerContour(0.8)] },
  ],
  q1_rows: 41234, model_draws: 6000, vis_range: [18, 26],
};

export const PAIR: PairView = {
  available: true, kind: "joint",
  x: CORNER.variables![0], y: CORNER.variables![2],
  x_edges: [18, 20, 22, 24, 26], y_edges: [-1.5, -1, -0.5, 0, 0.5],
  q1: { rows: 4000, density: [[0.1, 0.5, 1, 0.2], [0.1, 0.3, 0.6, 0.1], [0, 0.1, 0.2, 0], [0, 0, 0, 0]],
    contours: [{ mass_fraction: 0.5, paths: [{ x: [19, 21, 19], y: [-1, -0.5, -1] }] }, { mass_fraction: 0.95, paths: [{ x: [18.5, 23, 18.5], y: [-1.4, 0, -1.4] }] }] },
  model: { rows: 6000, density: [[0.2, 0.5, 1, 0.1], [0.1, 0.4, 0.5, 0.1], [0, 0.1, 0.1, 0], [0, 0, 0, 0]],
    contours: [{ mass_fraction: 0.5, paths: [{ x: [19.5, 21.5, 19.5], y: [-1, -0.4, -1] }] }] },
  contour_mass_fractions: [0.2, 0.5, 0.8, 0.95, 0.99], vis_range: [18, 26],
};

export function galaxyPayload(patch: Partial<GalaxyPayload> = {}): GalaxyPayload {
  return {
    version: 25, stale: false, authenticated: true,
    sources: {
      euclid: { available: true, rows: 140085, area_arcmin2: 1884.96, phz_pdf_rows: 132147, phz_pdf_source: "phz", schema_version: 7,
        detail: "MER + PHZ cache, schema v7" },
      synthetic: { available: true, rows: 5489, area_arcmin2: 36.41, measured_radius_rows: 4802, fields: 200, detail: "test + validation source catalogues" },
      fit: { available: true, fingerprint: "f".repeat(64), is_active: false, validated: true, detail: "fitted candidate" },
    },
    q1_counts: {
      footprint_area_deg2: 63.1, bright: 14, faint: 28, bin_width: 0.1, query_count: 400, completed_queries: 400, total_queries: 560,
      complete: false, phases_completed: 3, phase_count: 5, selection: "POINT_LIKE_FLAG IS NULL",
      apertures: { f1: { label: "F1", selected_galaxies: 1, expected_galaxies: 1.2e6 }, f2: { label: "F2", selected_galaxies: 1 },
        f3: { label: "F3", selected_galaxies: 1 }, f4: { label: "F4", selected_galaxies: 1, expected_galaxies: 9.8e5 } },
    },
    q1_radius: { complete: true, completed_queries: 170, total_queries: 170 },
    calibration: {
      is_active: false, active: null,
      candidate: {
        valid: true, version: 15, fingerprint: "c".repeat(64),
        magnitude_law: { bright_join_magnitudes: [19.5, 20.5, 21.5], bright_slopes: [0.5, 0.42, 0.37], break_magnitude: 25.3, straight_law: { slope: 0.31 } },
        radius_law: { slope_log10_arcsec_per_mag: -0.1497, scatter_dex: 0.2290, fitted_rows: 7528938 },
        generation: { surface_density_arcmin2: 151.5, differential_density_cap_arcmin2_mag: 30.16, break_magnitude: 25.06,
          fitted_surface_density_arcmin2: 120, vis_magnitude_min: 14, vis_magnitude_max: 29, fitted_vis_magnitude_max: 26, faint_end_policy: "flat" },
        aperture_fwhm_distribution: { magnitude_edges: [17, 19, 21, 23, 25, 27], fwhm_edges_arcsec: [0, 0.5, 1, 1.5],
          probability: [[0.2, 0.6, 0.2], [0.3, 0.5, 0.2], [0.4, 0.4, 0.2], [0.6, 0.3, 0.1], [0.7, 0.2, 0.1]],
          source_magnitude_bin: [0, 1, 2, 3, 3], out_of_support_policy: "nearest populated bin" },
        color_sfr_model: { row_count: 83583, tree_count: 50, catalog_version: 7, vis_snr_floor: 5, calibration_fingerprint: "a".repeat(64) },
        plots: {
          conditional_radius: { magnitude: mags, observed_mean_log10_arcsec: [-0.1, -0.3, null, -0.6, -0.8], model_mean_log10_arcsec: [-0.1, -0.25, -0.4, -0.55, -0.7],
            model_core_low_log10_arcsec: [-0.3, -0.45, -0.6, -0.75, -0.9], model_core_high_log10_arcsec: [0.1, -0.05, -0.2, -0.35, -0.5] },
          conditional_aperture_fwhm: { magnitude: mags, observed_mean_arcsec: [0.9, 0.7, 0.6, null, null], model_mean_arcsec: [0.9, 0.72, 0.6, 0.5, 0.45],
            model_kind: "empirical histogram", out_of_support_policy: "nearest populated bin" },
          conditional_colors: { magnitude: mags, model_mean_vis_minus_y: [0.1, 0.2, 0.3, 0.3, 0.35], model_mean_y_j: [0.05, 0.1, 0.1, 0.1, 0.12],
            model_mean_j_h: [0, 0.02, 0.05, 0.06, 0.06], magnitude_edges: [17, 19, 21, 23, 25, 27],
            observed_ratio_variance_by_magnitude: [[0.01, 0.02, 0.03], [0.02, 0.03, 0.04], [0.05, 0.06, 0.08], [0.1, 0.12, 0.2], [null, null, null]],
            noise_ratio_variance_by_magnitude: [[0.001, 0.002, 0.003], [0.004, 0.006, 0.01], [0.03, 0.04, 0.05], [0.09, 0.1, 0.18], [null, null, null]] },
        },
        provenance: { color_sfr_valid_weight_fraction: 0.330, color_resolved_radius_weight_fraction: 0.975 },
      },
    },
    parameters: {
      magnitude: {
        label: "Apparent brightness", x_label: "VIS 2FWHM AB magnitude", density_unit: "galaxies / arcmin² / mag", note: "PHZ-weighted brackets",
        series: {},
        photometry_series: {
          q1_vis_f2: { ...curve("Q1 PHZ-weighted"), label: "Q1 MER + PHZ galaxies · VIS · 2 FWHM", survey: "euclid", band: "VIS", estimator: "2FWHM aperture",
            selection: "PHZ_GAL_PROB ≥ 0.5", default_on: true, trust_boundary: TRUST, observed_density_cap_arcmin2_mag: 30.5,
            observed_density_cap_magnitude: 24.8, observed_cumulative_density_to_boundary_arcmin2: 88.1,
            observed_cumulative_density_all_queried_bins_arcmin2: 74.4 },
          synthetic_vis_2fwhm: { ...curve("generated", 0.9), label: "Generated fields · VIS 2FWHM", survey: "synthetic", band: "VIS",
            estimator: "2FWHM aperture", selection: "all", default_on: true },
          generator_vis_f2: { ...curve("law", 1.1), label: "Generator · three-segment bright bridge + main + flat", survey: "generation", band: "VIS",
            estimator: "law", selection: "law", default_on: true, fit_interval: [19.5, 25.3], generation_interval: [14, 29],
            generation_bright_join_magnitudes: [19.5, 20.5, 21.5], generation_bright_slopes: [0.5, 0.42, 0.37], generation_main_slope: 0.31,
            generation_break_magnitude: 25.3, generation_density_cap_arcmin2_mag: 61.2 },
          mer_vis_kron: { ...curve("kron"), label: "Euclid · Kron", survey: "euclid", band: "VIS", estimator: "kron", selection: "all" },
        },
      },
      radius: {
        label: "Angular size", x_label: "log₁₀ radius (arcsec)", x_domain: [-2.4, 1], density_unit: "objects / arcmin² / dex", note: "circularized",
        series: {},
        radius_series: {
          euclid_sersic_re: { ...reCurve("Q1 Sérsic"), label: "Q1 PHZ/MER · circularized VIS Sersic R_e", source: "euclid", radius_type: "half_light", units: "arcsec" },
          synthetic_requested_re: { ...reCurve("requested", 0.8), label: "Generated fields · requested Sérsic Rₑ", source: "synthetic", radius_type: "half_light", units: "arcsec" },
          synthetic_clean_half_light: { ...reCurve("clean image", 0.7), label: "Generated fields · clean-image half-light radius", source: "synthetic", radius_type: "rendered_half_light", units: "arcsec" },
          fit_re: { ...reCurve("model"), label: "Generator · circularized Euclid Sérsic Rₑ", source: "fit", radius_type: "half_light", units: "arcsec" },
          euclid_sersic_re_shape: { ...reCurve("shape"), label: "Q1 clean · normalized circularized Sérsic Rₑ shape", source: "euclid",
            radius_type: "half_light_shape", units: "arcsec", normalization: "probability_density" },
          fit_re_q1_weighted_shape: { ...reCurve("shape"), label: "Candidate · Q1-magnitude-weighted circularized Sérsic Rₑ", source: "fit",
            radius_type: "half_light_shape", units: "arcsec", normalization: "probability_density" },
          fit_re_full_generation_shape: { ...reCurve("shape", 0.9), label: "Candidate · full-generation circularized Sérsic Rₑ (faint extension)", source: "fit",
            radius_type: "half_light_shape", units: "arcsec", normalization: "probability_density" },
          euclid_kron: { ...reCurve("kron"), label: "Euclid · Kron radius", source: "euclid", radius_type: "kron", units: "arcsec" },
        },
      },
      color_vis_y: { label: "VIS − Y colour", x_label: "VIS − Y colour (AB mag, 2FWHM apertures)", density_unit: "galaxies / arcmin² / mag", note: "",
        series: { euclid: colorCurve(), synthetic: colorCurve(), fit: colorCurve() } },
      color_y_j: { label: "Y − J colour", x_label: "Y − J colour (AB mag)", density_unit: "galaxies / arcmin² / mag", note: "",
        series: { euclid: colorCurve(), synthetic: colorCurve() } },
      color_j_h: { label: "J − H colour", x_label: "J − H colour (AB mag)", density_unit: "galaxies / arcmin² / mag", note: "",
        series: { euclid: colorCurve(), synthetic: colorCurve() } },
    },
    joint_maps: JOINT_MAPS,
    corner: CORNER,
    training_included: false,
    training_variant_available: true,
    ...patch,
  };
}

/* ─── stars ─────────────────────────────────────────────────────────────── */

/* The VIS panel carries the Q1 point sources and the native Gaia G_AB counts + shared-slope fit (the
   magnitude-law fit inputs); a colour panel carries only Q1, the model and the generated stars. */
const starDensity = (label: string, x: number[]) => ({
  label, x_label: `${label} [AB mag]`, x, x_domain: [x[0], x[x.length - 1]] as [number, number],
  euclid: x.map((_, i) => 0.01 * (i + 1)), model: x.map((_, i) => 0.011 * (i + 1)), synthetic: x.map((_, i) => 0.009 * (i + 1)),
});
const starVisDensity = (x: number[]) => ({
  ...starDensity("VIS", x),
  gaia: x.map((_, i) => 0.012 * (i + 1)), point_sources: x.map((_, i) => 0.02 * (i + 1)), gaia_fit: x.map((_, i) => 0.012 * (i + 1)),
  fit_ranges: { q1: [18, 23] as [number, number], gaia: [16, 20] as [number, number] },
});
const COLOR_KEYS = ["vis_y", "vis_j", "vis_h", "y_j", "y_h", "j_h"] as const;

export function starPayload(patch: Partial<StarPayload> = {}): StarPayload {
  const mag = [16, 18, 20, 22];
  return {
    authenticated: true,
    color_sample: { cached: true, euclid: { rows: 3462, field_count: 3 }, gaia: { rows: 5963, field_count: 3 } },
    calibration: { candidate: { valid: true, warnings: ["refit after the Q1 counts changed"], coverage_notes: ["three fixed fields"],
      euclid_mapping: { matched_stars: 3456 } }, is_active: false },
    q1_counts: { footprint_area_deg2: 63.1, selected_point_sources: 536780, expected_point_sources: 519611.8, expected_stars: 403069.7,
      bins: [1, 2, 3], edges: [14, 14.1, 14.2, 14.3], selection: "POINT_LIKE_PROB ≥ 0.9" },
    distribution: {
      matched_stars: 3456, high_quality_stars: 2398, pointlike_over_0_9: 3456, training_included: false,
      density_comparison: {
        area_arcmin2: 4156.3, gaia_area_arcmin2: 4156.3, model_density_arcmin2: 5.084, model_sample_count: 50000, euclid_vis_count: 3462,
        q1_phz_expected_stars: 403069.7, q1_expected_point_sources: 519611.8, q1_selected_point_sources: 536780, q1_area_arcmin2: 227160,
        euclid_color_count: 3456, gaia_native_g_count: 5963, synthetic_area_arcmin2: 1201.49, synthetic_star_count: 6040,
        parameters: { vis: starVisDensity(mag), ...Object.fromEntries(COLOR_KEYS.map((k) => [k, starDensity(k, [-0.5, 0, 0.5, 1])])) } as
          NonNullable<StarDistribution["density_comparison"]>["parameters"],
        note: "Q1 0.1-mag brackets normalize the population.",
      },
      gaia_sampling: { field_count: 3, radius_arcmin: 21, area_arcmin2: 4156,
        fields: [{ name: "EDF-N", ra: 269.733, dec: 66.018, rows: 2950 }, { name: "EDF-S", ra: 61.241, dec: -48.423, rows: 1637 },
          { name: "EDF-F", ra: 52.932, dec: -28.088, rows: 1376 }] },
    },
    availability: undefined,
    ...patch,
  };
}

/* ─── pixels ────────────────────────────────────────────────────────────── */

const pctl = (v: number) => ({ p16: [v * 0.8, v * 0.7], median: [v, v * 0.9], p84: [v * 1.2, v * 1.1] });
const interval = (v: number) => ({ median: v, p16: v * 0.9, p84: v * 1.1 });
const metrics = (v: number) => ({ mean: interval(v), median: interval(v), std: interval(2 * v), robust_std: interval(1.5 * v),
  p01: interval(-v), p99: interval(5 * v), zero_fraction: interval(0.01), negative_fraction: interval(0.3) });

export const FIELDS: FieldComparison = {
  bands: BANDS,
  histograms: per(() => ({ x: [-10, 0, 10, 20], synthetic: [0.2, 0.5, 0.2, 0.1], real: [0.25, 0.45, 0.2, 0.1], zero_bin: 1,
    x_label: "pixel brightness (e⁻ / stack)", y_label: "fraction" })),
  quantiles: per(() => ({ q: [0.1, 50, 99.9], synthetic: [-5, 1, 40], real: [-6, 1, 38], x_label: "percentile", y_label: "e⁻" })),
  power: per(() => ({ k: [0, 0.5, 0.1, 1], synthetic: { p16: [1, 2, 3, 4], median: [2, 3, 4, 5], p84: [3, 4, 5, 6] }, real: { p16: [1, 2, 3, 4], median: [2, 3, 5, 5], p84: [3, 4, 5, 6] },
    x_label: "k", y_label: "power" })),
  scale_similarity: per(() => ({ k: [0.5, 0.1, 1], log_shape_ratio: { p16: [-0.1, -0.05, 0], median: [0, 0.02, 0.05], p84: [0.1, 0.1, 0.1] },
    overlap: interval(0.97), variance_ratio: interval(1.1), x_label: "scale", y_label: "ratio" })),
  relations: {
    mean_std: per(() => ({ synthetic: { x: [1, 2], y: [3, 4], parent_ids: ["dirty_test:0", "dirty_test:1"] },
      real: { x: [1.5, 2.5], y: [3.5, 4.5], parent_ids: ["parent-3", "parent-4"] }, x_label: "field mean (e⁻ / pixel)", y_label: "field standard deviation (e⁻ / pixel)" })),
    median_robust_std: per(() => ({ synthetic: { x: [1, 2], y: [2, 3] }, real: { x: [1.5, 2.5], y: [2.5, 3.5] },
      x_label: "field median (e⁻ / pixel)", y_label: "robust noise, 1.4826 × MAD (e⁻ / pixel)" })),
  },
  band_correlation: { pairs: ["VIS·Y", "Y·J"], synthetic: pctl(0.5), real: pctl(0.45), x_label: "pair", y_label: "r" },
  summary: { synthetic: per((_b, i) => metrics(1 + i)), real: per((_b, i) => metrics(1.1 + i)) },
  source_detection: {
    settings: { band: "VIS", threshold_sigma: 1.5, minimum_connected_pixels: 5, background_box_pixels: 64, deblend_levels: 32,
      deblend_contrast: 0.001, negative_image_correction: true, truth_match_radius_lr_pixels: 2 },
    synthetic: { positive: [20, 24, 30], negative: [2, 1, 3], matched_galaxies: [15, 18, 20], matched_stars: [1, 0, 2], truth_galaxies: [20, 20, 25] },
    real: { positive: [28, 32], negative: [3, 5], matched_galaxies: [0, 0], matched_stars: [0, 0], truth_galaxies: [0, 0] },
  },
};

export function pixelsPayload(patch: Partial<PixelsPayload> = {}): PixelsPayload {
  return {
    authenticated: false,
    availability: {
      synthetic: { fields: 200, area_arcmin2: 36.4, record_files: 2, source_catalogs: 2, train_source_catalog: false, population_fields: 200,
        population_area_arcmin2: 36.4, population_fields_with_training: 200, population_area_arcmin2_with_training: 36.4 },
      real: { fields: 176, area_arcmin2: 32, independent_parents: 44, available: true, valid: true, ready: true, complete: true, current: true,
        unavailable_reason: null, collection_fingerprint: "c".repeat(64) },
      field_area_arcmin2: 0.182, input_fingerprint: "i".repeat(64),
      comparison_cache: { present: true, schema_current: true, fresh: true, reason: null },
    },
    comparison: {
      version: 9, provenance: { generated_at: "2026-09-25T10:00:00Z" },
      geometry: { tile_size: 256, analysis_size: 255, pixel_scale_arcsec: 0.1, field_area_arcmin2: 0.182 },
      samples: { synthetic: { fields: 200, area_arcmin2: 36.4, splits: ["test", "validate"] }, real: { fields: 176, area_arcmin2: 32, independent_parents: 44 } },
      fields: FIELDS,
      population: { synthetic: { objects: 12345, counts: { galaxy: 12000, star: 345 }, density_arcmin2: { galaxy: 329.7, star: 9.5 }, area_arcmin2: 36.4 },
        synthetic_field_count: 200, training_included: false },
    },
    ...patch,
  };
}

/* ─── viewer metas ──────────────────────────────────────────────────────── */

export const ARCHIVE_META: ArchiveCollectionMeta = {
  count: 2,
  archive: {
    available: true, valid: true, ready: true, complete: true, current: true, reasons: [], sample_count: 220, planned_sample_count: 220,
    parent_count: 44, fields: { "EDF-N": 80, "EDF-S": 95, "EDF-F": 45 }, comparison_sample_count: 176,
    comparison_fields: { "EDF-N": 64, "EDF-S": 76, "EDF-F": 36 }, bands: BANDS, tile_size: 256, manifest_fingerprint: "m".repeat(64),
    collection_fingerprint: "c".repeat(64), source_release: "Q1_R1", source_plan_fingerprint: "p".repeat(64), source_manifest_sha256: "s".repeat(64),
  },
  objects: [
    { id: "17", label: "sample 17", tiers: ["lr"], sample_id: 17, source_sample_id: 3, parent_id: "parent-3", field: "EDF-N", ra: 269.1, dec: 65.9, position_name: "northeast" },
    { id: "18", label: "sample 18", tiers: ["lr"], sample_id: 18, source_sample_id: 4, parent_id: "parent-4", field: "EDF-S", ra: 61.1, dec: -48.3, position_name: "south" },
  ],
};

export const SKY_META = { count: 3, tier_counts: { dirty: 3, clean: 3 } };
