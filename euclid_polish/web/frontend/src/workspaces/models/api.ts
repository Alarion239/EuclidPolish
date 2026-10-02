/* Typed endpoints of the Models workspace (routes/ensemble.py; shapes in
   euclid_polish/web/API.md › Ensemble), plus the few reads it shares with
   other workspaces (the Sky › Compare runs, the model catalogue, the
   catalogue evaluation's synthetic stamps, the SR over the local records,
   System › Config), typed here to the fields this workspace uses. */
import { useResource } from "../../api/query";

/** The `?mode=` of every ensemble read and job. The models are starfull only
 *  (the training still injects stars into starless scenes); the endpoints
 *  default to it, but the URLs name it so they equal Home's and share its cache. */
export const REGIME = "starfull";
export const BANDS = ["VIS", "Y_E", "J_E", "H_E"] as const;
export type Band = (typeof BANDS)[number];
export const BAND_SHORT: Record<string, string> = { VIS: "VIS", Y_E: "Y", J_E: "J", H_E: "H" };

export type Series2 = [number, number][];
export type KneeInfo = { asinh_knee?: number | null; asinh_knees?: number[] | null; output_knee?: number | null; knee_loss?: string | null };

/* ── overview.json ─────────────────────────────────────────────────────── */
export type Check = { id: string; ok: boolean; tone: "good" | "warn" | "bad" | "info"; title: string; detail: string; action?: string | null };
export type Headline = {
  metric?: string | null; knee_e?: number | null; n_scored?: number | null;
  production: { psnr?: number | null; vs_mean_db?: number | null; vs_best_member_db?: number | null };
  mean: { psnr?: number | null; vs_mean_member_db?: number | null };
  best_member: { psnr?: number | null; label?: string | null; mean_member_psnr?: number | null };
  knee: {
    available: boolean; stale: boolean; n_fields?: number | null;
    integration?: { from_e: number; to_e: number; weighting?: string } | null;
    production?: number | null; production_bands?: number[] | null; mean?: number | null;
    best_member?: number | null; best_member_label?: string | null;
  };
};
export type Overview = {
  active_members: string[]; n_members: number; records_dir?: string | null;
  eval_subset?: string; test_present: boolean; evaluated_at?: string | null;
  summary: Record<string, unknown> | null; headline: Headline; checks: Check[];
  production_gate: { available: boolean; n_members: number; mix_space?: string | null; fitted_at?: string | null; promoted_from?: string | null };
};

/* ── members.json / member/<name>.json ─────────────────────────────────── */
export type MemberJob = {
  jobid: string; state?: string | null; submitted_at?: string | null; ended_at?: string | null;
  elapsed_seconds?: number | null; req_time_limit?: string | null; gpu_util_mean?: number | null; mode?: string;
};
export type MemberStatus = "complete" | "timeout" | "running" | "unknown";
export type MemberRow = KneeInfo & {
  name: string; label: string;
  origin: Record<string, unknown> | null; op?: string | null; forked_from?: string | null;
  loss: string; blocks?: number | null;
  noise_aug?: number | null; bootstrap?: number | null; icnr?: boolean | null; seed?: number | null;
  commit?: string | null; created_at?: string | null; noise_model?: string | null;
  step?: number | null; target_steps?: number | null; fraction?: number | null;
  status: MemberStatus; timeout: boolean; job: MemberJob | null;
  /** member-PSNR cache: joint 4-band asinh psnr_stretched over `psnr_fields` test fields */
  psnr?: number | null; psnr_rank?: number | null;
  /** eval_summary headline metric (VIS asinh, the Overview "Best member" tile) */
  vis_psnr?: number | null;
  knee_integrated?: Record<string, number | null> | null; knee_rank?: number | null;
  gate_usage?: Record<string, number | null> | null; gate_usage_source?: Record<string, number | null> | null;
  /** The member's largest share of the production gate's weight over the
   *  bands and brightness bins (a bare share, or where it is). */
  gate_usage_peak?: number | { value?: number | null; band?: string | null; bin?: string | null } | null;
  /** Whether production SR runs this member (the gate reads it). */
  used_by_gate?: boolean | null;
  coherence?: { overall?: number | null; sr?: number | null } | null;
  has_loss_best?: boolean; size_mb?: number;
};
export type Tombstone = {
  name: string; archived_at?: string; zip?: string; commit?: string | null;
  zip_found: boolean; zip_path?: string | null; campaign?: string | null; size_bytes?: number | null;
};
export type MembersPayload = {
  members: MemberRow[]; archived: Tombstone[];
  knee: { available: boolean; stale: boolean; n_fields?: number | null };
  gate: { available: boolean; stale: boolean; n_members: number };
  psnr_fields: number; eval_subset: string;
  vis_psnr?: { metric: string; knee_e?: number | null; n_scored?: number | null } | null;
};
export type MemberCurves = {
  psnr: Series2; band_psnr: Record<string, Series2>; loss_series: Series2; train_loss: Series2;
  gnorm: Series2; gnorm_max: Series2; step_time: Series2;
};
export type KneeModel = KneeInfo & {
  id: string; kind: "member" | "mean" | "combiner"; label: string;
  loss?: string | null; blocks?: number | null;
  psnr: number[][]; integrated: number[];
};
export type MemberDetail = {
  name: string; label: string; active: boolean; archived: Tombstone | null;
  row: MemberRow | null; curves: MemberCurves | null;
  knee: { knees: number[]; bands: string[]; stale: boolean; models: KneeModel[] } | null;
  gate: {
    stale: boolean; bands: string[]; brightness_names: string[];
    usage: Record<string, number | null>; usage_source: Record<string, number | null>;
    by_brightness: Record<string, (number | null)[]>; uniform: number;
  } | null;
};

/* ── training-curves.json ──────────────────────────────────────────────── */
export type Curve = MemberCurves & KneeInfo & {
  /** training-curves.json lists every active member, a legacy starless one too (Curves drops it). */
  name: string; label: string; starless: boolean; loss_norm: string; blocks?: number | null;
  target_steps?: number | null; test_psnr?: number | null;
};

/* ── knee-psnr.json ────────────────────────────────────────────────────── */
export type KneePayload = {
  available?: boolean; stale?: boolean; reason?: string; n_fields?: number;
  knees?: number[]; bands?: string[]; models?: KneeModel[];
  integration?: { from_e: number; to_e: number; weighting?: string };
};

/* ── combiners.json / compare.json ─────────────────────────────────────── */
export type HistoryRow = {
  step: number; loss?: number | null; train_loss?: number | null; vis_psnr?: number | null;
  band_psnr?: number[] | null; integrated_psnr?: number[] | null; vis_integrated_psnr?: number | null;
};
export type Variant = {
  name: string; kind: "gate" | "rbf"; spec: string; production: boolean; backup: boolean;
  member_labels: string[]; reads: string[]; n_members: number; n_reads: number; pruned: boolean;
  mix_space?: string | null; use_lr: boolean; width?: number | null; fitted_at?: string | null;
  fingerprint?: string | null;
  /** current: every member it reads is active; extra: members that joined after the fit. */
  membership: { current: boolean; missing: string[]; extra: string[]; missing_reads?: string[] };
  /** Whether it can be promoted now, and why not (a read member left, an unfinished fit). */
  promotion?: { ok: boolean; reason: string | null } | null;
  applies_to_test_cubes: boolean;
  fit: Record<string, unknown>;
  selected?: HistoryRow | null; baseline?: HistoryRow | null; history: HistoryRow[];
  test?: { source: string; report?: string | null; band_psnr?: NumArr | null; blackout_band_psnr?: NumArr | null } | null;
  knee?: { source: "knee" | "compare"; stale?: boolean; report?: string | null; integrated?: number[] | null; psnr?: number[][] | null } | null;
  eval?: { psnr?: number | null; vs_mean_db?: number | null; vs_best_member_db?: number | null };
};
export type ReportRef = { id: string; created?: string | null; methods?: string[] | null; n_fields?: Record<string, number> | null; gates_requested?: string[] | null };
export type CombinersPayload = {
  production: string; active_members: string[]; cube_members: string[];
  variants: Variant[]; compare: ReportRef | null; reports: ReportRef[];
};
/** One method's scores on a field group (null: not scored, e.g. a report's
 *  empty blackout group when it was run without blackout fields). */
export type GroupScores = { band_psnr: NumArr; bin_mse: NumArr; halo_mse: NumArr; hole_mse: NumArr };
export type CompareReport = {
  id?: string; created?: string; members: string[]; bands: string[]; brightness_names: string[];
  methods: string[]; method_labels?: Record<string, string>; method_members?: Record<string, string[]>;
  n_fields?: Record<string, number>;
  groups: Record<string, Record<string, GroupScores>>;
  usage?: Record<string, { labels: string[]; all_pixels: number[][]; source_pixels: number[][] }>;
  knee?: { knees: number[]; n_fields: number; methods: Record<string, { psnr: number[][]; integrated: number[] }> };
  timing_s?: Record<string, number | null>; members_needed?: Record<string, number>;
};

/* ── experiments (Sky › Compare; the real-data benchmark) ─────────────── */
export type ExperimentSummary = {
  id: string; label?: string; created?: string; status?: string; tiles?: string[]; models?: string[];
  summary?: Record<string, Record<string, unknown>>;
};

/* ── training-jobs.json ────────────────────────────────────────────────── */
export type TrainingJob = {
  jobid: string; state?: string | null; submitted_at?: string | null; started_at?: string | null;
  ended_at?: string | null; elapsed_seconds?: number | null; req_time_limit?: string | null;
  req_memory?: string | null; req_cpus?: number | null; req_gpus?: number | null; partition?: string | null;
  gpu_util_mean?: number | null; mode: string; member_names: string[]; steps?: number | null;
  continue_basis?: string | null; target_steps?: number | null; extra_steps?: number | null;
  params: Record<string, unknown>;
};

/* ── evals.json (Diagnostics) ──────────────────────────────────────────── */
export type NumArr = (number | null)[];
/** The bands the evaluation diagnostics are measured in (VIS is
 *  ensemble_evals.json; the others are computed from the same cached cubes). */
export const EVAL_BANDS = ["VIS", "Y_E", "J_E", "H_E"] as const;
export type EvalBand = (typeof EVAL_BANDS)[number];
export type Evals = {
  subset?: string; n_fields?: number; n_members?: number;
  /** The band this payload measures (absent on payloads older than the band switch: VIS). */
  band?: EvalBand;
  /** A Y/J/H payload computed for an earlier evaluation (other members, fields or gate). */
  stale?: boolean;
  members?: ({ label: string; loss?: string; blocks?: number } & KneeInfo)[];
  guides?: { theta_min?: number; lr_scale?: number; vis_fwhm?: number; psf_fwhm?: number; band?: string };
  ps?: {
    theta: NumArr; r: NumArr; r_lr?: NumArr; r_members?: NumArr[]; r_pairs?: NumArr[]; r_cross?: NumArr;
    T?: NumArr; T_members?: NumArr[];
    model_combiners?: Record<string, { r?: NumArr; T?: NumArr }>;
  } | null;
  coherence?: {
    domains?: Record<string, { theta_min?: number; theta_max?: number }>;
    scores?: { id: string; label: string; overall: number | null; sr: number | null;
      overall_lo?: number | null; overall_hi?: number | null; sr_lo?: number | null; sr_hi?: number | null }[];
  } | null;
  std_err?: (StdErrModel & { models?: Record<string, StdErrModel> }) | null;
  bright_std?: { bright_edges: NumArr; std_edges: NumArr; hist: number[][]; bright: NumArr; lo: NumArr; med: NumArr; hi: NumArr; stretch: number } | null;
  calibration?: {
    z_edges: NumArr; pdf: NumArr; field_std: NumArr; field_rmse: NumArr;
    stats: { cover1?: number; cover2?: number; cover3?: number; sigma_z?: number };
  } | null;
  model_combiners?: Record<string, { available?: boolean; psnr?: number | null; asinh_l1?: number | null;
    ensemble_mean_psnr?: number | null; best_member_psnr?: number | null; best_member_label?: string | null;
    coherence_overall?: number | null; coherence_sr?: number | null } | null>;
};
/** The per-band HR-vs-SR angular power spectrum of the synced validation
 *  records (GET /api/evaluation/angular-power-spectrum.json). */
export type ApsCurves = { theta: NumArr; T: NumArr; T_lo: NumArr; T_hi: NumArr; r: NumArr; r_lo: NumArr; r_hi: NumArr; count?: NumArr };
export type Aps = {
  subset?: string; n_fields: number; field_n?: number; pixel_scale?: number; lr_scale?: number; theta_max?: number;
  band_names?: string[];
  bands: Record<string, { psf_fwhm?: number; linear?: ApsCurves; asinh?: ApsCurves }>;
};
export type StdErrModel = { edges: NumArr; hist: number[][]; med_std: NumArr; med_err: NumArr; n_fields?: number };

/* ── shared reads (other workspaces' endpoints, the fields used here) ──── */

/** One row of the catalogue evaluation (GET /api/evaluation/runs): the
 *  synthetic stamps (grade syn-lens / syn-gal) carry an HR truth. CSV
 *  values arrive as strings. */
export type EvalRow = {
  id?: string | null; grade?: string | null; ok?: string | boolean | null; kind?: string | null;
  psnr_lr_hr?: string | number | null; psnr_sr_hr?: string | number | null;
  flux_ratio_sr_over_lr?: string | number | null; lr_total_e?: string | number | null; sr_total_e?: string | number | null;
  n_members?: number | string | null; state?: string | null; state_reason?: string | null;
  viewer_id?: string | null; out_subdir?: string | null;
};
export type EvalRuns = { rows: EvalRow[]; n?: number; n_ok?: number; groups?: Record<string, number> };

/** A curve row the loss facets split (training-curves.json). */
export type CurveFacetRow = { loss_norm: string; name: string };

/** The model catalogue (GET /api/models): each spec's fingerprint. */
export type ModelCatalog = { models: { spec: string; label?: string; fingerprint?: string | null; available?: boolean }[] };
/** One Sky › Compare run's record (GET /api/experiments/<id>): the fields read here. */
export type ExperimentDetail = { id: string; label?: string; created?: string; fingerprints?: Record<string, string | null> | null };

/** The production SR over the local records (GET /api/sky/sr-status). */
export type SrSplit = "test" | "validate" | "train";
export type SrStatus = {
  records: boolean; checkpoint: boolean; can_generate: boolean; subsets: SrSplit[];
  sr: Partial<Record<SrSplit, number>>;
  splits: Partial<Record<SrSplit, { count: number; present: boolean; sr: { state: string; reasons: string[]; count: number } }>>;
};

/* ── URLs ──────────────────────────────────────────────────────────────── */
export const url = {
  overview: () => `/ensemble/overview.json?mode=${REGIME}`,
  members: () => `/ensemble/members.json?mode=${REGIME}`,
  member: (name: string) => `/ensemble/member/${encodeURIComponent(name)}.json`,
  curves: () => "/ensemble/training-curves.json",
  knee: () => `/ensemble/knee-psnr.json?mode=${REGIME}`,
  evals: (band: EvalBand = "VIS") => `/ensemble/evals.json?mode=${REGIME}${band === "VIS" ? "" : `&band=${band}`}`,
  evalBands: () => "/ensemble/evals/bands",
  combiners: () => `/ensemble/combiners.json?mode=${REGIME}`,
  report: (id?: string | null) => `/ensemble/combiners/compare.json?mode=${REGIME}${id ? `&report=${encodeURIComponent(id)}` : ""}`,
  trainingJobs: () => "/ensemble/training-jobs.json",
  experiments: () => "/api/experiments",
  experiment: (id: string) => `/api/experiments/${encodeURIComponent(id)}`,
  models: () => "/api/models",
  viewerMeta: () => `/viewer/meta/ensemble?mode=${REGIME}`,
  evalRuns: () => "/api/evaluation/runs",
  srStatus: () => "/api/sky/sr-status",
  generateSr: () => "/api/sky/generate-sr",
  config: () => "/api/config",
  mirrorStatus: () => "/api/fasrc/mirror/status",
  fieldStatus: () => "/api/inference/field.json",
  fieldDiagnostics: () => "/api/inference/diagnostics.json",
  fieldRefresh: () => "/inference/refresh-combiners",
  recoveryFigure: () => "/api/evaluation/angular-power-spectrum",
  recoverySpectrum: () => "/api/evaluation/angular-power-spectrum.json",
};

const MIN = 60_000;
export const useOverview = () => useResource<Overview>(url.overview(), [], { ttl: 30_000 });
export const useMembers = () => useResource<MembersPayload>(url.members(), [], { ttl: 30_000 });
export const useCurves = () => useResource<{ members: Curve[] }>(url.curves(), [], { ttl: 2 * MIN });
export const useKnee = () => useResource<KneePayload>(url.knee(), [], { ttl: MIN });
export const useCombiners = () => useResource<CombinersPayload>(url.combiners(), [], { ttl: 30_000 });
export const useTrainingJobs = () => useResource<{ jobs: TrainingJob[] }>(url.trainingJobs(), [], { ttl: MIN });
export const useExperiments = () => useResource<{ experiments: ExperimentSummary[] }>(url.experiments(), [], { ttl: MIN });
export const useExperiment = (id: string | null | undefined) =>
  useResource<ExperimentDetail>(id ? url.experiment(id) : null, [id], { ttl: 5 * MIN });
export const useModelCatalog = () => useResource<ModelCatalog>(url.models(), [], { ttl: MIN });
export const useEvalRuns = () => useResource<EvalRuns>(url.evalRuns(), [], { ttl: 30_000 });
export const useSrStatus = (poll?: number) => useResource<SrStatus>(url.srStatus(), [], { ttl: 30_000, poll });
