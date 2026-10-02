/* The freeze-study dialog's data and pure rules (shared: Models › Leaderboard
 * and Figures › Studies both open it). `GET /api/studies/candidates?mode=` is
 * read-only; `POST /api/studies` is the only write (the dialog's "Freeze").
 * API.md "Model studies". */
import { ApiError, isFasrcOffline } from "../../api/client";
import { pagePath } from "../../app/nav";
import { formatBytes } from "../../format";

/** A study freezes the starfull ensemble (no other regime is trained). */
export const STUDY_MODE = "starfull";
export type BlockState = "current" | "stale" | "missing";

export type CandidateBlock = {
  id: "members" | "test_cubes" | "gate" | "gate_diagnostic" | "compare" | "training_curves" | "real" | string;
  title: string;
  state: BlockState | string;
  detail: string;
};

export type FieldKind = "test" | "blackout" | "real";

export type CandidateField = {
  fid: string;
  kind: FieldKind | string;
  ref: string;
  label: string;
  available: boolean;
  reason: string | null;
  /** Upper bound (uncompressed) of every product of the field. */
  bytes: number;
  core_bytes: number;
  largest_product_bytes: number;
  bytes_upper_bound?: boolean;
  thumb_url: string | null;
};

export type Candidates = {
  ok: boolean;
  regime: string;
  ensemble: {
    members: string[];
    n_members: number;
    gate: { available: boolean; state: string; name: string | null; reads?: string[] | null; mix_space?: string | null; fitted_at?: string | null };
    evaluated_at: string | null;
    blocks: CandidateBlock[];
    stale: string[];
    numbers_bytes: number;
  };
  fields: CandidateField[];
  max_fields: number;
  can_freeze: boolean;
  blocking: string | null;
  fasrc_connected: boolean;
  fields_note: string | null;
};

export type FreezeReply = { ok: boolean; job_id?: string; study_id?: string | null; fields?: string[]; upload_bytes?: number; error?: string };

export const MAX_FIELDS = 10;
export const FREEZE_JOB_KEY = "study:freeze";

export const candidatesUrl = () => `/api/studies/candidates?mode=${STUDY_MODE}`;
export const studyPath = (id: string) => `/figures/studies?study=${encodeURIComponent(id)}`;

export const KIND_ORDER: readonly FieldKind[] = ["test", "blackout", "real"];
export const KIND_TITLE: Record<FieldKind, string> = {
  test: "Synthetic test fields",
  blackout: "Blackout test fields",
  real: "Real tiles",
};
export const KIND_HINT: Record<FieldKind, string> = {
  test: "HR truth: every metric can be recomputed at any knee",
  blackout: "Saturated cores blanked: the gate's holes",
  real: "Cached member SR of real Euclid tiles (no truth)",
};

export type FieldGroup = {
  kind: FieldKind;
  fields: CandidateField[];
  available: number;
  /** The one reason every unavailable field of the group shares (said once). */
  sharedReason: string | null;
};

/** The gallery's groups in kind order; empty kinds are dropped. */
export function groupFields(fields: readonly CandidateField[]): FieldGroup[] {
  return KIND_ORDER.map((kind) => {
    const list = fields.filter((f) => f.kind === kind);
    const off = list.filter((f) => !f.available);
    const reasons = new Set(off.map((f) => f.reason ?? ""));
    return {
      kind, fields: list, available: list.length - off.length,
      sharedReason: off.length > 1 && reasons.size === 1 ? [...reasons][0] || null : null,
    };
  }).filter((g) => g.fields.length > 0);
}

/** Sum of the picked fields' upper-bound sizes. */
export function uploadBytes(fields: readonly CandidateField[], picked: readonly string[]): number {
  const set = new Set(picked);
  return fields.filter((f) => set.has(f.fid)).reduce((s, f) => s + (Number(f.bytes) || 0), 0);
}

/** "3 of 10 · ≤ 480 MiB to upload" (sizes are upper bounds). */
export function counterText(n: number, max: number, bytes: number): string {
  return `${n} of ${max}${n ? ` · ≤ ${formatBytes(bytes)} to upload` : ""}`;
}

/** Toggle `fid` in `picked`, never past `max`. */
export function togglePick(picked: readonly string[], fid: string, max: number): string[] {
  if (picked.includes(fid)) return picked.filter((x) => x !== fid);
  return picked.length >= max ? [...picked] : [...picked, fid];
}

/** Where each numbers block is fixed, when it is not current. */
export function blockFix(block: CandidateBlock): { label: string; to: string } | null {
  if (block.state === "current") return null;
  const models = (tab: string) => pagePath("models", { tab });
  switch (block.id) {
    case "test_cubes": return { label: "Re-evaluate", to: models("leaderboard") };
    case "gate": case "gate_diagnostic": case "compare": return { label: "Combiner", to: models("combiner") };
    case "training_curves": return { label: "Members", to: models("members") };
    case "real": return { label: "Sky › Compare", to: "/sky/compare" };
    default: return null;
  }
}

/** A freeze refusal in plain words (400 / 409 / 503 / 507 / network). */
export function refusalMessage(err: unknown): { title: string; text: string; retry: "candidates" | "fields" | null } {
  if (!(err instanceof ApiError)) {
    return { title: "The freeze did not start", text: err instanceof Error ? err.message : String(err), retry: null };
  }
  const body = (err.body && typeof err.body === "object" ? err.body : {}) as Record<string, unknown>;
  if (isFasrcOffline(err)) {
    return { title: "FASRC is not connected", retry: "fields",
      text: "Attached fields are stored on holylabs, so they need the connection. Connect in System › Connections, or go back and freeze without fields." };
  }
  if (err.status === 507) {
    const needed = Number(body.needed_bytes), free = Number(body.free_bytes);
    const sizes = Number.isFinite(needed) && Number.isFinite(free) ? ` It needs ${formatBytes(needed)}; ${formatBytes(free)} is free.` : "";
    return { title: "Not enough free disk", retry: null,
      text: `The freeze would leave less than the 5 GiB the console keeps free.${sizes} Free some space (System › Storage) and try again.` };
  }
  if (err.status === 409 && err.code === "busy") {
    return { title: "Another study is being frozen", retry: null,
      text: `${err.message} It shows in the job tray; freeze this one when it finishes.` };
  }
  if (err.status === 409) {
    return { title: "The ensemble cannot be frozen as it is", retry: "candidates",
      text: `${err.message.replace(/[.\s]+$/, "")}. Fix it (re-evaluate, refit), then check again.` };
  }
  if (err.status === 400) return { title: "The freeze was refused", text: err.message, retry: null };
  if (err.status === 0) return { title: "The server did not answer", text: err.message, retry: null };
  return { title: `The freeze failed (HTTP ${err.status})`, text: err.message, retry: null };
}
