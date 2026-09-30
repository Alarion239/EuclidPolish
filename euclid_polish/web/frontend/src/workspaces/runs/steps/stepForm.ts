/* The pure model of the schema-driven FASRC step form (spec §8.7, contract C5).
 *
 * `/api/fasrc/steps/status` publishes each step's `task_params` (name, type,
 * default, min/max, choices, help, required) and `last_params` (the typed task
 * params of its newest successful run). The form keeps every value as the
 * string a form would post; this module turns the schema into that state,
 * validates it exactly like the server's `TaskParam.parse`, and builds the
 * submit body. No React here: StepCard.tsx renders it. */

export type TaskParamType = "int" | "float" | "str" | "bool" | "choice" | "json";

export type TaskParam = {
  name: string;
  type: TaskParamType;
  default: unknown;
  help: string;
  min?: number;
  max?: number;
  choices?: string[];
  required?: boolean;
};

export type StepDefaults = {
  partition: string; n_cpus: number; n_gpus: number; memory: string; time_limit: string;
};

export type StepOutput = { key: string; path: string; exists: boolean | null };

export type Step = {
  step_id: string;
  label: string;
  needs_gpu: boolean;
  fixed_cpus?: number | null;
  fixed_gpus?: number | null;
  defaults: StepDefaults;
  task_params?: TaskParam[];
  last_params?: Record<string, unknown> | null;
  outputs?: StepOutput[];
};

export type FormValues = Record<string, string>;
export type Resources = { n_cpus: string; n_gpus: string; memory: string; time_limit: string };
export type PrefillSource = "defaults" | "last" | "clone";

const TRUE_WORDS = new Set(["1", "true", "yes", "on"]);
const FALSE_WORDS = new Set(["", "0", "false", "no", "off"]);

/** A typed value as the string a form posts (bool → "1"/"0", json → compact). */
export function toFormValue(param: TaskParam, value: unknown): string {
  if (value === null || value === undefined) return "";
  if (param.type === "bool") {
    if (typeof value === "boolean") return value ? "1" : "0";
    return TRUE_WORDS.has(String(value).trim().toLowerCase()) ? "1" : "0";
  }
  if (param.type === "json") return typeof value === "string" ? value : JSON.stringify(value);
  if (param.type === "float" && typeof value === "number" && Number.isInteger(value)) return String(value);
  return String(value);
}

export function defaultValues(params: readonly TaskParam[]): FormValues {
  const out: FormValues = {};
  for (const p of params) out[p.name] = toFormValue(p, p.default);
  return out;
}

export function defaultResources(step: Step): Resources {
  const d = step.defaults;
  return {
    n_cpus: String(step.fixed_cpus ?? d.n_cpus),
    n_gpus: String(step.fixed_gpus ?? (step.needs_gpu ? d.n_gpus : 0)),
    memory: d.memory,
    time_limit: d.time_limit,
  };
}

/** The params a history row / clone source carries (strings as posted). */
export type ParamSource = Record<string, unknown>;

/** One-shot destructive flags (a forced redownload / regeneration): the
 *  danger confirmation asks about them, and the last-run prefill never
 *  carries them over — only an explicit clone does. */
export const DESTRUCTIVE_FLAGS: ReadonlySet<string> = new Set(["force", "force_redownload", "regenerate_catalog"]);

/** Schema defaults, overlaid by the last successful run's params (a `null`
 *  there — a blank or a fresh-entropy seed — keeps the default; the
 *  `DESTRUCTIVE_FLAGS` keep theirs too), then by an explicit clone source.
 *  Returns the values and where they came from. */
export function initialValues(
  step: Step, opts: { clone?: ParamSource | null } = {},
): { values: FormValues; source: PrefillSource } {
  const params = step.task_params ?? [];
  const values = defaultValues(params);
  let source: PrefillSource = "defaults";
  const last = step.last_params;
  if (last && typeof last === "object") {
    for (const p of params) {
      const v = last[p.name];
      if (v === null || v === undefined || DESTRUCTIVE_FLAGS.has(p.name)) continue;
      const s = toFormValue(p, v);
      if (s !== values[p.name]) source = "last";
      values[p.name] = s;
    }
  }
  if (opts.clone) {
    for (const p of params) {
      if (!(p.name in opts.clone)) continue;
      values[p.name] = toFormValue(p, opts.clone[p.name]);
    }
    source = "clone";
  }
  return { values, source };
}

/** Mirrors `TaskParam.parse` + the blank handling of `fill_task_params`:
 *  the error text for one raw value, or null when the server would accept it. */
export function validateParam(param: TaskParam, raw: string): string | null {
  const text = raw.trim();
  if (text === "") return param.required ? "required" : null;
  switch (param.type) {
    case "bool":
      return TRUE_WORDS.has(text.toLowerCase()) || FALSE_WORDS.has(text.toLowerCase()) ? null : "must be on or off";
    case "int":
    case "float": {
      const n = Number(text);
      if (!Number.isFinite(n)) return param.type === "int" ? "must be an integer" : "must be a number";
      if (param.type === "int" && !Number.isInteger(n)) return "must be an integer";
      if (param.min != null && n < param.min) return `must be ≥ ${param.min}`;
      if (param.max != null && n > param.max) return `must be ≤ ${param.max}`;
      return null;
    }
    case "choice":
      return param.choices?.includes(text) ? null : `must be one of ${(param.choices ?? []).join(", ")}`;
    case "json":
      try { JSON.parse(text); return null; } catch { return "is not valid JSON"; }
    default:
      return null;
  }
}

export function validateValues(params: readonly TaskParam[], values: FormValues,
  skip: ReadonlySet<string> = new Set()): Record<string, string> {
  const errors: Record<string, string> = {};
  for (const p of params) {
    if (skip.has(p.name)) continue;
    const e = validateParam(p, values[p.name] ?? "");
    if (e) errors[p.name] = e;
  }
  return errors;
}

export function validateResources(step: Step, r: Resources): Partial<Record<keyof Resources, string>> {
  const out: Partial<Record<keyof Resources, string>> = {};
  const int = (v: string) => /^\d+$/.test(v.trim());
  if (!int(r.n_cpus) || Number(r.n_cpus) < 1) out.n_cpus = "a whole number ≥ 1";
  if (!int(r.n_gpus)) out.n_gpus = "a whole number ≥ 0";
  if (!/^\d+(\.\d+)?\s*[KMGT]?i?B?$/i.test(r.memory.trim())) out.memory = "e.g. 16G or 512M";
  if (!/^(\d+-)?\d{1,3}(:\d{2}){0,2}$/.test(r.time_limit.trim())) out.time_limit = "e.g. 2:00:00 or 1-00:00:00";
  if (step.fixed_cpus != null) delete out.n_cpus;
  if (step.fixed_gpus != null) delete out.n_gpus;
  return out;
}

/** Whether a raw value differs from the schema default (blank vs default
 *  compared as posted: "18" vs 18.0 are the same). */
export function isChanged(param: TaskParam, raw: string): boolean {
  const def = toFormValue(param, param.default);
  if (raw === def) return false;
  if (param.type === "int" || param.type === "float") {
    if (raw.trim() === "" || def === "") return raw.trim() !== def;
    return Number(raw) !== Number(def);
  }
  if (param.type === "bool") return (TRUE_WORDS.has(raw.toLowerCase())) !== (TRUE_WORDS.has(def.toLowerCase()));
  return raw.trim() !== def;
}

export function changedParams(params: readonly TaskParam[], values: FormValues,
  skip: ReadonlySet<string> = new Set()): TaskParam[] {
  return params.filter((p) => !skip.has(p.name) && isChanged(p, values[p.name] ?? ""));
}

/** The params the form shows: all when `expanded`, else the first `limit`
 *  plus every changed or invalid one (a value that matters is never hidden). */
export function visibleParams(params: readonly TaskParam[], values: FormValues, opts: {
  hidden?: ReadonlySet<string>; expanded?: boolean; limit?: number; errors?: Record<string, string>;
} = {}): { shown: TaskParam[]; more: number } {
  const hidden = opts.hidden ?? new Set<string>();
  const pool = params.filter((p) => !hidden.has(p.name));
  if (opts.expanded || pool.length <= (opts.limit ?? 8)) return { shown: pool, more: 0 };
  const limit = opts.limit ?? 8;
  const shown = pool.filter((p, i) => i < limit || isChanged(p, values[p.name] ?? "") || !!opts.errors?.[p.name]);
  return { shown, more: pool.length - shown.length };
}

/** The task params a submit posts: every form-controlled task param (blank
 *  = the server's default/unset), then the host page's `extraParams` (they
 *  win). The resource advisor is asked about exactly these. */
export function submitParams(step: Step, values: FormValues,
  extraParams: Record<string, string | number> = {}, hidden: ReadonlySet<string> = new Set()): Record<string, string> {
  const out: Record<string, string> = {};
  for (const p of step.task_params ?? []) {
    if (hidden.has(p.name)) continue;
    out[p.name] = (values[p.name] ?? "").trim();
  }
  for (const [k, v] of Object.entries(extraParams)) out[k] = String(v);
  return out;
}

/** The POST body for `/api/fasrc/steps/<id>/submit`: resources, the task
 *  params (`submitParams`) and `confirm=yes`. */
export function submitBody(step: Step, values: FormValues, resources: Resources,
  extraParams: Record<string, string | number> = {}, hidden: ReadonlySet<string> = new Set()): Record<string, string> {
  return {
    n_cpus: String(step.fixed_cpus ?? resources.n_cpus).trim(),
    n_gpus: String(step.fixed_gpus ?? resources.n_gpus).trim(),
    memory: resources.memory.trim(),
    time_limit: resources.time_limit.trim(),
    ...submitParams(step, values, extraParams, hidden),
    confirm: "yes",
  };
}

/** The resource fields a user can edit on a step's card: CPUs unless the
 *  step fixes them, GPUs only on a GPU step that does not fix them, memory
 *  and time always (the partition is fixed per step). */
export function editableResources(step: Step): (keyof Resources)[] {
  return (["n_cpus", "n_gpus", "memory", "time_limit"] as (keyof Resources)[]).filter((k) =>
    k === "n_cpus" ? step.fixed_cpus == null : k === "n_gpus" ? step.needs_gpu && step.fixed_gpus == null : true);
}

/** The form's resources after applying advised ones: only the editable
 *  fields change, and only to a non-blank value. */
export function applyAdvisedResources(step: Step, current: Resources, advised: Partial<Resources>): Resources {
  const next = { ...current };
  for (const k of editableResources(step)) {
    const v = String(advised[k] ?? "").trim();
    if (v) next[k] = v;
  }
  return next;
}

/** Why a submit deserves a danger confirmation (destructive flags set). */
export function dangerReasons(step: Step, values: FormValues, hidden: ReadonlySet<string> = new Set()): string[] {
  const out: string[] = [];
  for (const p of step.task_params ?? []) {
    if (p.type !== "bool" || hidden.has(p.name)) continue;
    if (!DESTRUCTIVE_FLAGS.has(p.name)) continue;
    if (TRUE_WORDS.has((values[p.name] ?? "").toLowerCase())) out.push(p.help || p.name);
  }
  return out;
}

/** `num_stars` → "num stars". */
export const humanName = (name: string): string => name.replace(/_/g, " ");

/** "default 10000 · ≥ 1" style hint line for a param. */
export function paramFacts(p: TaskParam): string {
  const parts: string[] = [];
  const def = toFormValue(p, p.default);
  parts.push(def === "" ? "default: unset" : `default: ${p.type === "bool" ? (def === "1" ? "on" : "off") : def}`);
  if (p.min != null && p.max != null) parts.push(`${p.min} – ${p.max}`);
  else if (p.min != null) parts.push(`≥ ${p.min}`);
  else if (p.max != null) parts.push(`≤ ${p.max}`);
  if (p.required) parts.push("required");
  return parts.join(" · ");
}

/** Resources of a past run (history row `req_*`), for "clone this run". */
export function resourcesFromRow(step: Step, row: Record<string, unknown>): Resources {
  const base = defaultResources(step);
  const pick = (v: unknown, fallback: string) => (v === undefined || v === null || String(v).trim() === "" ? fallback : String(v));
  return {
    n_cpus: pick(row.req_cpus, base.n_cpus),
    n_gpus: pick(row.req_gpus, base.n_gpus),
    memory: pick(row.req_memory, base.memory),
    time_limit: pick(row.req_time_limit, base.time_limit),
  };
}
