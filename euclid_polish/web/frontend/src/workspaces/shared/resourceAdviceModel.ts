/* The resource advisor's recommendation (spec 2026-09-30, "Resource
 * advisor"): the shape of `POST /api/fasrc/resources/<step>/recommend` and of
 * the `recommendation` of `GET /api/fasrc/resources/<step>`, and the pure
 * helpers the "Recommended from N past runs" callout (ResourceAdvice.tsx: the
 * step card, Models › Train) and Runs › Resources' recommendation card share:
 * field labels, the level and confidence words, the changes a host can apply
 * and the resources an Apply hands back. No React. Unit-tested in
 * resourceAdviceModel.test.ts. */
import { pagePath } from "../../app/nav";
import { formatCount, formatDuration } from "../../format";

export type AdviceField = "n_cpus" | "n_gpus" | "memory" | "time_limit";
/** Resources as the forms post them: `n_cpus` / `n_gpus` strings, memory
 *  "36G", time limit SLURM text ("3:15:00", "1-02:00:00"). */
export type AdviceResources = Record<AdviceField, string>;

export const ADVICE_FIELDS: readonly AdviceField[] = ["n_cpus", "n_gpus", "memory", "time_limit"];

export type AdviceChange = {
  field: string; current: string | number | null; recommended: string | number | null; reason: string;
};

export type AdviceBasis = {
  /** "exact" (same plan and CPU count), "similar" (same plan), "step" (every run of the step). */
  level: string | null;
  level_label?: string | null;
  n_runs: number;
  jobids?: string[];
  /** The planned amount of work (images, training steps) and its unit word. */
  units?: number | null;
  units_label?: string | null;
  /** Seconds per unit of work over the runs used (their p90). */
  rate_s_per_unit?: number | null;
};

export type Recommendation = {
  ok: boolean;
  step_id: string;
  available: boolean;
  confidence: string | null;
  resources: Partial<Record<AdviceField, string | number | null>>;
  current: Partial<Record<AdviceField, string | number | null>>;
  changes: AdviceChange[];
  basis: AdviceBasis | null;
  notes: string[];
  warnings: string[];
  error?: string;
};

export const recommendUrl = (stepId: string) => `/api/fasrc/resources/${encodeURIComponent(stepId)}/recommend`;

/** Runs › Resources opened on one step (the callout's "Open usage"). */
export const usageHref = (stepId: string) =>
  `${pagePath("runs", { tab: "resources" })}?${new URLSearchParams({ step: stepId }).toString()}`;

const FIELD_LABEL: Record<AdviceField, string> = {
  n_cpus: "CPUs", n_gpus: "GPUs", memory: "Memory", time_limit: "Time limit",
};

export const isAdviceField = (f: string): f is AdviceField => (ADVICE_FIELDS as readonly string[]).includes(f);

/** "Memory", or "Memory / member" for a per-array-task form. */
export function fieldLabel(field: string, perTask?: string): string {
  const base = isAdviceField(field) ? FIELD_LABEL[field] : field.replace(/_/g, " ");
  return perTask ? `${base} / ${perTask}` : base;
}

const LEVEL_WORDS: Record<string, string> = {
  exact: "same plan and CPU count", similar: "similar plan", step: "every run of the step",
};

/** The match level in words: the backend's `level_label`, else ours. */
export function levelText(basis: AdviceBasis | null | undefined): string {
  if (!basis) return "";
  return String(basis.level_label || (basis.level ? LEVEL_WORDS[basis.level] ?? basis.level : "")).trim();
}

/** "high confidence" · "low confidence" · "" (unknown). */
export function confidenceText(confidence: string | null | undefined): string {
  const c = String(confidence ?? "").trim().toLowerCase();
  return c ? `${c} confidence` : "";
}

/** "Recommended from 8 past runs (similar plan) · high confidence"; a level
 *  that carries its own parentheses ("same settings (batch 4 · …)") follows
 *  a " · " instead, never nested brackets. */
export function adviceHeadline(rec: Recommendation): string {
  const n = rec.basis?.n_runs ?? 0;
  const level = levelText(rec.basis);
  const conf = confidenceText(rec.confidence);
  const lvl = !level ? "" : /[()]/.test(level) ? ` · ${level}` : ` (${level})`;
  return `Recommended from ${n} past run${n === 1 ? "" : "s"}${lvl}${conf ? ` · ${conf}` : ""}`;
}

export type AdviceRow = { field: AdviceField; label: string; current: string; recommended: string; reason: string };

const text = (v: unknown): string => (v === null || v === undefined ? "" : String(v).trim());

/** The changes the host can apply (its editable `fields`, in form order):
 *  current → recommended with the reason. A change whose recommendation is
 *  blank, or that names a field the host does not edit, is left out. */
export function adviceRows(rec: Recommendation | null | undefined, fields: readonly AdviceField[] = ADVICE_FIELDS,
  perTask?: string): AdviceRow[] {
  if (!rec?.available) return [];
  const wanted = new Set(fields);
  const out: AdviceRow[] = [];
  for (const field of ADVICE_FIELDS) {
    if (!wanted.has(field)) continue;
    const change = rec.changes.find((c) => c.field === field);
    const recommended = text(change?.recommended ?? rec.resources[field]);
    if (!change || !recommended) continue;
    out.push({ field, label: fieldLabel(field, perTask), current: text(change.current) || "—", recommended, reason: change.reason });
  }
  return out;
}

/** What Apply hands back: the current resources with every applicable
 *  change's recommended value in place (the rest as they are). */
export function appliedResources(current: Partial<AdviceResources>, rec: Recommendation,
  fields: readonly AdviceField[] = ADVICE_FIELDS): AdviceResources {
  const out: AdviceResources = {
    n_cpus: text(current.n_cpus), n_gpus: text(current.n_gpus), memory: text(current.memory), time_limit: text(current.time_limit),
  };
  for (const row of adviceRows(rec, fields)) out[row.field] = row.recommended;
  return out;
}

/** "3m 05s per 1,000 steps" for the basis' rate: the unit count scaled by
 *  powers of 1000 until one lot takes at least a second. */
export function rateText(rate: number | null | undefined, unitsLabel: string | null | undefined): string {
  if (rate == null || !Number.isFinite(rate) || rate <= 0) return "";
  let lot = 1;
  while (rate * lot < 1 && lot < 1e9) lot *= 1000;
  const unit = String(unitsLabel || "unit").trim();
  return `${formatDuration(rate * lot)} per ${lot === 1 ? unit.replace(/s$/, "") : `${formatCount(lot)} ${unit}`}`;
}
