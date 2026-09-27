/* The job-config editor's logic, pure (configModel.test.ts): form state as
 * strings, dirty-only saves with `base_version` (the lost-update guard of
 * POST /api/config/save), rebasing on a 409, defaults, validation from the
 * field metadata, and the toolbar filters. */
import { FIELDS, fieldMeta, type FieldMeta, type GroupId } from "./configFields";

export type ConfigValues = Record<string, number | string | null | undefined>;
export type FormState = Record<string, string>;
export type Conflict = {
  fields: Record<string, { base: unknown; current: unknown }>;
  config: ConfigValues;
  version: string | null;
};

export function toForm(config: ConfigValues): FormState {
  const out: FormState = {};
  for (const [k, v] of Object.entries(config)) out[k] = v == null ? "" : String(v);
  return out;
}

export function dirtyFields(form: FormState, loaded: FormState): string[] {
  return Object.keys(form).filter((k) => form[k] !== loaded[k]);
}

/** Only the edited fields, plus the version they were edited against. */
export function saveBody(form: FormState, loaded: FormState, version: string | null): Record<string, string> {
  const body: Record<string, string> = {};
  for (const k of dirtyFields(form, loaded)) body[k] = form[k];
  if (version) body.base_version = version;
  return body;
}

/** After a 409: take the server's values for the conflicting fields, keep
 *  every other edit, and continue from the server's version. */
export function rebase(form: FormState, loaded: FormState, conflict: Conflict): { form: FormState; loaded: FormState; version: string | null } {
  const server = toForm(conflict.config);
  const next: FormState = { ...server };
  for (const k of Object.keys(form)) {
    if (!(k in conflict.fields) && form[k] !== loaded[k]) next[k] = form[k];
  }
  return { form: next, loaded: server, version: conflict.version };
}

const num = (v: unknown): number | null => {
  if (typeof v === "number") return Number.isFinite(v) ? v : null;
  if (typeof v !== "string" || v.trim() === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};

/** Whether the form value equals the field's default (numbers by value). */
export function isDefault(name: string, raw: string, defaults: ConfigValues): boolean {
  if (!(name in defaults)) return true;
  const d = defaults[name];
  if (typeof d === "number") {
    const n = num(raw);
    return n != null && Math.abs(n - d) <= 1e-12 * Math.max(1, Math.abs(d));
  }
  return raw === String(d ?? "");
}

/** A validation message for the form value, or null when it can be saved. */
export function fieldError(name: string, raw: string, type?: string): string | null {
  const meta = fieldMeta(name, type);
  if (meta.kind === "choice" || meta.kind === "text") return null;
  if (raw.trim() === "") return "required";
  const n = num(raw);
  if (n == null) return "must be a number";
  if (meta.kind === "int" && !Number.isInteger(n)) return "must be a whole number";
  if (meta.min != null && n < meta.min) return `must be ≥ ${meta.min}`;
  if (meta.max != null && n > meta.max) return `must be ≤ ${meta.max}`;
  return null;
}

const ORDER = new Map(FIELDS.map((f, i) => [f.name, i]));

/** The server's fields with their metadata, in the editor's order. */
export function orderedFields(names: string[], types: Record<string, string> = {}): FieldMeta[] {
  return names
    .map((n) => fieldMeta(n, types[n]))
    .sort((a, b) => (ORDER.get(a.name) ?? 1e6) - (ORDER.get(b.name) ?? 1e6) || a.name.localeCompare(b.name));
}

export type FieldFilter = { q?: string; group?: GroupId | "all"; changed?: boolean; edited?: boolean };

export function filterFields(fields: FieldMeta[], f: FieldFilter, form: FormState, loaded: FormState,
  defaults: ConfigValues): FieldMeta[] {
  const q = (f.q ?? "").trim().toLowerCase();
  return fields.filter((m) => {
    if (f.group && f.group !== "all" && m.group !== f.group) return false;
    if (f.changed && isDefault(m.name, form[m.name] ?? "", defaults)) return false;
    if (f.edited && form[m.name] === loaded[m.name]) return false;
    if (q && ![m.name, m.label, m.hint, m.unit ?? ""].some((s) => s.toLowerCase().includes(q))) return false;
    return true;
  });
}

/** The FASRC steps EVERY field of a group is injected into, in the first
 *  field's order — the group shows them once; each field lists only its
 *  extra steps. Empty when a field feeds no step (or the group is empty). */
export function commonSteps(fields: FieldMeta[], usedBy: Record<string, string[] | undefined>): string[] {
  if (!fields.length) return [];
  const [first, ...rest] = fields.map((f) => usedBy[f.name] ?? []);
  return first.filter((step) => rest.every((steps) => steps.includes(step)));
}
