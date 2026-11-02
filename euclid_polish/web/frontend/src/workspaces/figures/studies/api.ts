/* Figures › Studies: the typed `/api/studies*` endpoints (API.md "Model
 * studies"). Every read here is a GET; the writes (note, selections, delete,
 * resume, fetch) are explicit POSTs from buttons. */
import { useQuery } from "@tanstack/react-query";
import { ApiError, apiGetText } from "../../../api/client";
import { queryClient, useResource } from "../../../api/query";
import { parseCsv, type CsvRow } from "./model";

export type StudySummary = {
  id: string;
  name: string;
  created: string | null;
  completed: string | null;
  regime: string | null;
  members: number;
  gate: string | null;
  fields: number;
  field_ids: string[];
  note: string;
  commit: { short?: string; hash?: string; dirty?: boolean } | string | null;
  state: "complete" | "incomplete";
  reason: string | null;
  numbers_bytes: number;
  fields_bytes: number;
  /** "ensemble" (the membership) or "battery" (chosen models, studies/battery.py). */
  kind?: "ensemble" | "battery" | string;
  /** A battery study's model names, in table order (the first is the baseline). */
  models?: string[];
};

export type StudiesList = {
  ok: boolean;
  studies: StudySummary[];
  root: string;
  max_fields: number;
  freezing: { job_id: string; study_id: string } | null;
};

export type StudyMember = {
  label: string;
  name?: string;
  loss?: string | null;
  asinh_knee?: number | null;
  asinh_knees?: number[] | null;
  output_knee?: number | null;
  knee_loss?: string | null;
  blocks?: number | null;
  bootstrap?: unknown;
  noise_aug?: unknown;
  icnr?: unknown;
  seed?: number | null;
  status?: string | null;
  timeout?: boolean;
  step?: number | null;
  target_steps?: number | null;
  op?: string | null;
  fingerprint?: string | null;
  [k: string]: unknown;
};

export type StudyManifest = {
  id: string;
  name: string;
  note?: string;
  created?: string;
  completed?: string | null;
  commit?: StudySummary["commit"];
  regime?: string;
  complete?: boolean;
  error?: string | null;
  ensemble?: { members?: StudyMember[]; labels?: string[]; n_members?: number };
  gate?: { name?: string | null; promoted_from?: string | null; kind?: string | null; member_labels?: string[]; reads?: string[]; mix_space?: string | null; fitted_at?: string | null; fingerprint?: string | null };
  records?: { records_fp?: string | null; subset?: string | null; indices?: number[]; noise_models?: unknown };
  evaluation?: { evaluated_at?: string | null };
  warnings?: string[];
  kind?: string;
  battery?: BatteryManifest;
  knees?: number[];
  bands?: string[];
  knee_fields?: number[];
  [k: string]: unknown;
};

/** One model of a battery study (the battery's ModelInfo, without paths). */
export type BatteryModel = { spec: string; name: string; kind: string; member: string; step: number | null; fingerprint: string };

export type BatteryManifest = {
  models: BatteryModel[];
  tests: string[];
  created?: string | null;
  records_fp?: string | null;
  records_dir?: string | null;
  seconds?: number | null;
  source?: string | null;
  /** Numbers-file names of the figures (`figure-<name>.png`). */
  figures: string[];
};

/** One table row: [section, metric, one value per model (paired: per model after the baseline)]. */
export type BatteryRow = [string, string, string[]];

export type BatteryNumbers = { meta?: Record<string, unknown>; table: BatteryRow[]; paired: BatteryRow[] };

export type StudyViewer = { collection: string; params: Record<string, string>; id: string };

export type StudyField = {
  fid: string;
  kind: "test" | "blackout" | "real" | string;
  ref: string;
  label: string;
  state: "pending" | "uploaded" | string;
  /** Compressed bytes of every product. */
  bytes: number | null;
  estimated_bytes: number | null;
  core_bytes: number;
  member_bytes: Record<string, number>;
  gate: unknown;
  fetched: boolean;
  cached_products: string[];
  members_fetched: number;
  products: string[];
  thumb_url: string | null;
  viewer: StudyViewer;
};

export type SavedSelection = { name: string; members?: string[] | null; group?: string | null; note?: string | null };

export type GateNumbers = {
  diagnostic?: { available?: boolean; labels?: string[]; bands?: string[]; brightness_names?: string[] } | null;
  compare?: unknown;
  compare_note?: string | null;
};

export type RealNumbers = {
  experiments?: { id: string; label?: string | null; created?: string | null; tiles?: unknown }[];
  note?: string | null;
};

export type KneeNumbers = { knees?: number[]; bands?: string[]; fields?: number[] };

export type StudyDetail = {
  ok: boolean;
  study: StudySummary;
  manifest: StudyManifest;
  manifest_sha256: string;
  note: string;
  selections: SavedSelection[];
  fields: StudyField[];
  charts: string[];
  group_fields: string[];
  citation: string;
  numbers?: { knee_psnr?: KneeNumbers; training_curves?: unknown; gate?: GateNumbers; real?: RealNumbers; battery?: BatteryNumbers };
};

export const STUDIES_URL = "/api/studies";
export const studyUrl = (id: string) => `/api/studies/${encodeURIComponent(id)}`;

export const useStudies = () => useResource<StudiesList>(STUDIES_URL, [], { ttl: 10_000 });
export const useStudy = (id: string | null) => useResource<StudyDetail>(id ? studyUrl(id) : null, [id], { ttl: 60_000 });

/** A chart table from the backend CSV (exactly the numbers the export draws). */
export function useStudyCsv(url: string | null): { rows: CsvRow[] | null; loading: boolean; error: ApiError | null; reload: () => void } {
  const q = useQuery<CsvRow[], ApiError>({
    queryKey: ["GET-text", url],
    queryFn: async ({ signal }) => parseCsv(await apiGetText(url as string, { signal })),
    enabled: !!url,
    staleTime: 10 * 60_000,
    retry: false,
  }, queryClient);
  return { rows: q.data ?? null, loading: !!url && q.isPending, error: q.error ?? null, reload: () => { void q.refetch(); } };
}
