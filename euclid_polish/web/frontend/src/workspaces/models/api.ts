/* Typed endpoints of the Models workspace (routes/ensemble.py; shapes in
   euclid_polish/web/API.md › Ensemble), plus the few reads it shares with
   other workspaces (the Sky › Compare runs, the model catalogue, the
   catalogue evaluation's synthetic stamps, the SR over the local records,
   System › Config), typed here to the fields this workspace uses. */
import { useQueries, type UseQueryResult } from "@tanstack/react-query";
import { useMemo } from "react";
import { apiGet } from "../../api/client";
import { queryClient, resourceKey, useResource } from "../../api/query";

/** The `?mode=` of every ensemble read and job. The models are starfull only
 *  (the training still injects stars into starless scenes); the endpoints
 *  default to it, but the URLs name it so they equal Home's and share its cache. */
export const REGIME = "starfull";
export const BANDS = ["VIS", "Y_E", "J_E", "H_E"] as const;
export type Band = (typeof BANDS)[number];
export const BAND_SHORT: Record<string, string> = { VIS: "VIS", Y_E: "Y", J_E: "J", H_E: "H" };

export type Series2 = [number, number][];
export type KneeInfo = {
  asinh_knee?: number | null; asinh_knees?: number[] | null; output_knee?: number | null; knee_loss?: string | null;
  /** One output image whose knee is learned per band (no fixed `output_knee`). */
  learned_output_knee?: boolean | null;
};

/* ── members.json / member/<name>.json ─────────────────────────────────── */
export type MemberJob = {
  jobid: string; state?: string | null; submitted_at?: string | null; ended_at?: string | null;
  elapsed_seconds?: number | null; req_time_limit?: string | null; gpu_util_mean?: number | null; mode?: string;
};
export type MemberStatus = "complete" | "timeout" | "running" | "unknown";
export type MemberRow = KneeInfo & {
  name: string; label: string;
  /** `universal`: a registered universal model's row (member/<name>.json),
   *  not an ensemble member; `member`: an active member that is universal. */
  class?: "universal" | "member" | null;
  origin: Record<string, unknown> | null; op?: string | null; forked_from?: string | null;
  loss: string; blocks?: number | null;
  noise_aug?: number | null; bootstrap?: number | null; icnr?: boolean | null; seed?: number | null;
  commit?: string | null; created_at?: string | null; noise_model?: string | null;
  step?: number | null; target_steps?: number | null; fraction?: number | null;
  status: MemberStatus; timeout: boolean; job: MemberJob | null;
  /** member-PSNR cache: joint 4-band asinh psnr_stretched over `psnr_fields` test fields */
  psnr?: number | null; psnr_rank?: number | null;
  /** eval_summary headline metric (VIS asinh, the "Test VIS" column) */
  vis_psnr?: number | null;
  knee_integrated?: Record<string, number | null> | null; knee_rank?: number | null;
  /** A universal model's row: its ∫PSNR (`knee_integrated`) is from an earlier
   *  checkpoint or other test fields (null: not scored). */
  score_stale?: boolean | null;
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
  /** What a restore brings it back as: an ensemble member (absent on old
   *  tombstones) or a standalone universal model. */
  class?: "member" | "universal" | null;
};
export type MembersPayload = {
  members: MemberRow[]; archived: Tombstone[];
  /** The registered universal models (standalone, not ensemble members): local
   *  for the Pull banner, listed in the Universal tab. */
  universal?: string[] | null;
  knee: { available: boolean; stale: boolean; n_fields?: number | null };
  gate: { available: boolean; stale: boolean; n_members: number };
  psnr_fields: number; eval_subset: string;
  vis_psnr?: { metric: string; knee_e?: number | null; n_scored?: number | null } | null;
};
export type MemberCurves = {
  psnr: Series2; band_psnr: Record<string, Series2>; loss_series: Series2; train_loss: Series2;
  gnorm: Series2; gnorm_max: Series2; step_time: Series2;
};
export type MemberDetail = {
  name: string; label: string; active: boolean; archived: Tombstone | null;
  /** A registered universal model (standalone; Models › Universal). */
  universal?: boolean | null;
  /** `universal` for a registered universal model and for an archived one
   *  (from its tombstone); `member` otherwise. */
  class?: "member" | "universal" | null;
  row: MemberRow | null; curves: MemberCurves | null;
};

/* ── universal.json (the Universal tab) ─────────────────────────────────── */
/** A score on the test fields: ∫PSNR per band (VIS Y J H) and their mean. */
export type UniversalScore = {
  integrated: (number | null)[]; mean: number | null; n_fields?: number | null; stale: boolean; scored_at?: string | null;
};
/** Where a model trained: the partition and a hardware label ("A100",
 *  "A100 MIG slice", "mixed"), with the jobs that trained it. */
export type UniversalHardware = { partition: string | null; label: string | null; jobs: string[] };
/** One universal model (a single-image multi-knee member): a standalone
 *  universal model (`class` universal, spec `universal:member_NNN`) or an
 *  active ensemble member that happens to be one (`class` member). */
export type UniversalRow = {
  name: string; label: string; spec: string; class: "universal" | "member";
  num_res_blocks?: number | null; steps?: number | null; served_step?: number | null; target_steps?: number | null;
  batch_size?: number | null; crops_per_field?: number | null;
  knee: "learned" | "fixed"; output_knee?: number | null;
  /** The learned output knee per band (e⁻), recorded when it was scored. */
  learned_knees?: (number | null)[] | null;
  jit_compile?: boolean | null; mixed_precision?: boolean | null; bootstrap?: number | null; seed?: number | null;
  /** Σ duration_s of its training_log.csv windows (rolled-back ones included), in hours. */
  train_hours: number | null; train_rows: number;
  /** The training hours and steps when its score was made (null: an older
   *  score that did not record them); a stale score is placed there. */
  scored_hours?: number | null; scored_steps?: number | null;
  hardware: UniversalHardware;
  /** The checkpoint identity (null: no local checkpoint yet). */
  fingerprint: string | null;
  score: UniversalScore | null;
  vs_production_db: number | null;
  created_at?: string | null; commit?: string | null;
};
/** A universal model trained on FASRC that is not here (not active, not
 *  archived; registered only when its pull left no checkpoint here). */
export type UniversalFasrcRow = {
  name: string; jobids: string[]; state?: string | null; num_res_blocks?: number | null; steps?: number | null;
  knee?: "learned" | "fixed" | null; submitted_at?: string | null; ended_at?: string | null; partition?: string | null;
  /** Its newest job is in a final state (COMPLETED, TIMEOUT, FAILED, …). */
  finished: boolean;
  /** Its newest job ended COMPLETED or TIMEOUT (it left a checkpoint). */
  pullable?: boolean;
  /** Registered as a universal model but with no checkpoint here: pull it again. */
  registered?: boolean;
};
/** The production model's own score (`model`: its name). */
export type UniversalReference = { integrated: (number | null)[]; mean: number | null; stale: boolean; model?: string | null };
export type UniversalPayload = {
  available: boolean; knees: number[]; bands: string[];
  integration: { from_e: number; to_e: number } | null;
  /** The scores store's identity: the test fields (`indices`) and the truth. */
  identity: { indices?: number[] | null; subset?: string | null; [key: string]: unknown } | null;
  production: UniversalReference | null;
  /** The universal model that is production (`member_257`): `production` is
   *  its own score. */
  production_model?: string | null;
  /** Whether `production` is on the scores' own fields, so the rows'
   *  `vs_production_db` is meaningful. */
  production_comparable?: boolean | null;
  models: UniversalRow[];
  fasrc: UniversalFasrcRow[];
  notes: string[];
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

/* ── shared reads (other workspaces' endpoints, the fields used here) ──── */

/** What production is (GET /api/models `production`): one registered
 *  universal model. */
export type CatalogProduction = {
  kind: "universal" | "gate"; spec: string; name: string | null; label: string; available?: boolean; reason?: string | null;
};
/** The model catalogue (GET /api/models): each spec's fingerprint, and what
 *  production is (absent on older servers: the gate). */
export type ModelCatalog = {
  models: { spec: string; label?: string; fingerprint?: string | null; available?: boolean }[];
  production?: CatalogProduction | null; production_kind?: string | null;
};
/** One Sky › Compare run's record (GET /api/experiments/<id>): the fields read here. */
export type ExperimentDetail = { id: string; label?: string; created?: string; fingerprints?: Record<string, string | null> | null };

/* ── URLs ──────────────────────────────────────────────────────────────── */
export const url = {
  members: () => `/ensemble/members.json?mode=${REGIME}`,
  member: (name: string) => `/ensemble/member/${encodeURIComponent(name)}.json`,
  /** POST `model` = universal:member_NNN (instant, no job). */
  production: () => "/ensemble/production",
  trainingJobs: () => "/ensemble/training-jobs.json",
  universal: () => `/ensemble/universal.json?mode=${REGIME}`,
  universalScore: () => "/ensemble/universal/score",
  universalPull: () => "/ensemble/universal/pull",
  experiments: () => "/api/experiments",
  experiment: (id: string) => `/api/experiments/${encodeURIComponent(id)}`,
  models: () => "/api/models",
  config: () => "/api/config",
};

const MIN = 60_000;
export const useMembers = () => useResource<MembersPayload>(url.members(), [], { ttl: 30_000 });
export const useTrainingJobs = () => useResource<{ jobs: TrainingJob[] }>(url.trainingJobs(), [], { ttl: MIN });
export const useExperiments = () => useResource<{ experiments: ExperimentSummary[] }>(url.experiments(), [], { ttl: MIN });
export const useExperiment = (id: string | null | undefined) =>
  useResource<ExperimentDetail>(id ? url.experiment(id) : null, [id], { ttl: 5 * MIN });
export const useModelCatalog = () => useResource<ModelCatalog>(url.models(), [], { ttl: MIN });
export const useUniversal = () => useResource<UniversalPayload>(url.universal(), [], { ttl: 30_000 });

type DetailSlim = { data: ExperimentDetail | undefined; at: number; failed: boolean };
/* Stable (module-level) so TanStack keeps the combined array referentially
   stable while no query result changes. */
const combineDetails = (results: UseQueryResult<ExperimentDetail>[]): DetailSlim[] =>
  results.map((r) => ({ data: r.data, at: r.dataUpdatedAt, failed: r.isError && !r.data }));

/** Several Sky › Compare run records at once (the Universal tab checks each
 *  model's newest run against the catalogue), keyed by id; the shared GET
 *  cache of `useExperiment`. `details`: the records in; `failed`: the ids that
 *  could not be read (the rest are still loading). */
export function useExperimentDetails(ids: readonly string[]): { details: Map<string, ExperimentDetail>; failed: Set<string> } {
  const key = ids.join(",");
  const wanted = useMemo(() => (key ? key.split(",") : []), [key]);
  const results = useQueries({
    queries: wanted.map((id) => ({
      queryKey: resourceKey(url.experiment(id)),
      queryFn: ({ signal }: { signal: AbortSignal }) => apiGet<ExperimentDetail>(url.experiment(id), { signal }),
      staleTime: 5 * MIN,
    })),
    combine: combineDetails,
  }, queryClient);
  return useMemo(() => {
    const details = new Map<string, ExperimentDetail>();
    const failed = new Set<string>();
    wanted.forEach((id, i) => {
      const r = results[i];
      if (r?.data) details.set(id, r.data);
      else if (r?.failed) failed.add(id);
    });
    return { details, failed };
  }, [wanted, results]);
}
