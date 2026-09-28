/* The Home page's numbers, as pure functions (homeModel.test.ts; the Loop
 * strip, the running line and the thumbnails are in loop.ts).
 *
 * Definitions (the production combiner is the spatial gate,
 * `eval/combiner.py` ACTIVE_COMBINER_KINDS[0]):
 *  - knee-integrated PSNR (the metric to compare models by): the mean over the
 *    four bands of each model's PSNR integrated over log asinh knee
 *    0.1–1e4 e⁻ (`/ensemble/knee-psnr.json`), for the production gate, the
 *    plain mean and the best member by the same number.
 *  - production test PSNR, the fallback without knee curves:
 *    `eval_summary.json`'s `spatial_gate_combiner_psnr` (served by the light
 *    `/api/system/production`), with its gain over the plain mean
 *    (`…_vs_mean_db`) and over the best member (`…_vs_best_member_db`). The
 *    bare `combiner_psnr` keys are the RBF combiner and are never used.
 *    Without a gate score the plain mean is shown, labelled, with its gain
 *    over the MEAN member derived here.
 *  - real holes: the newest Sky › Compare run that scored production
 *    (`/api/experiments`), its worst band's hole % (the share of bright LR
 *    pixels the SR blanks), shown only when that run scored THIS production
 *    fit (its recorded fingerprint, `/api/experiments/<id>`, equals the
 *    catalogue's, `/api/models`); else "no real benchmark for this
 *    membership".
 *  - STARFULL members: the regime labels of the active STARFULL members
 *    (`/api/models` `members`, or `/api/system/production`), never all
 *    active members.
 *  - production model: the `production` spec of `/api/models` (its combiner,
 *    mix space and fit time; `available` = fitted for the current members).
 *  - tracking catch-up: the `tracking` health check's `facts.unlogged`
 *    (results written after the newest `## <ISO>` heading of log.md), one
 *    line each with the matching headline number, for the "Log to notebook" entry
 *    (workspaces/shared/LogToNotebook: Notebook › Log, prefilled). */
import { formatPow10, formatRelative } from "../../format";
import { utcText } from "../shared/noteText";

export type EvalSummary = {
  ensemble_psnr?: number | null;
  mean_member_psnr?: number | null;
  ensemble_gain_db?: number | null;
  spatial_gate_combiner_psnr?: number | null;
  spatial_gate_combiner_vs_mean_db?: number | null;
  spatial_gate_combiner_vs_best_member_db?: number | null;
  n_scored?: number | null;
  member_labels?: string[];
  [key: string]: unknown;
};

export type ProductionHeadline =
  | { kind: "gate"; psnr: number; vsMean: number | null; vsBest: number | null; meanPsnr: number | null; stale: boolean }
  | { kind: "mean"; psnr: number; vsMeanMember: number | null; stale: boolean };

const finite = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

export function productionHeadline(s: EvalSummary | null | undefined, stale: boolean): ProductionHeadline | null {
  if (!s) return null;
  const gate = finite(s.spatial_gate_combiner_psnr);
  const mean = finite(s.ensemble_psnr);
  if (gate != null) {
    return {
      kind: "gate", psnr: gate, vsMean: finite(s.spatial_gate_combiner_vs_mean_db),
      vsBest: finite(s.spatial_gate_combiner_vs_best_member_db), meanPsnr: mean, stale,
    };
  }
  if (mean == null) return null;
  const meanMember = finite(s.mean_member_psnr);
  return { kind: "mean", psnr: mean, vsMeanMember: meanMember != null ? mean - meanMember : null, stale };
}

export type KneeModel = {
  id: string;
  kind: "member" | "mean" | "combiner" | string;
  label?: string;
  integrated?: (number | null)[];
};

export type KneePayload = {
  available?: boolean;
  stale?: boolean;
  reason?: string;
  n_fields?: number;
  bands?: string[];
  models?: KneeModel[];
  /** The integration range in e⁻ (0.1–1e4). */
  integration?: { from_e?: number; to_e?: number } | null;
};

export type KneeHeadline = {
  /** Production gate, band-mean integrated PSNR (null when not baked into the cubes). */
  gate: number | null;
  mean: number | null;
  best: { name: string; label: string; value: number } | null;
  vsBest: number | null;
  vsMean: number | null;
  perBand: { band: string; value: number | null }[];
  nFields: number | null;
  stale: boolean;
};

/** Mean of the finite values (null when none). */
export function bandMean(values: readonly (number | null | undefined)[] | null | undefined): number | null {
  const ok = (values ?? []).filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  return ok.length ? ok.reduce((a, b) => a + b, 0) / ok.length : null;
}

/** "196·psnr" → "member_196" (the inspector / model-spec name). */
export function memberName(label: string): string {
  const head = String(label).split("·")[0];
  return head.startsWith("member_") ? head : `member_${head}`;
}

const PRODUCTION_KNEE_ID = "spatial_gate";

export function kneeHeadline(p: KneePayload | null | undefined): KneeHeadline | null {
  if (!p?.available || !p.models?.length) return null;
  const gateModel = p.models.find((m) => m.kind === "combiner" && m.id === PRODUCTION_KNEE_ID) ?? null;
  const meanModel = p.models.find((m) => m.kind === "mean") ?? null;
  const gate = bandMean(gateModel?.integrated);
  const mean = bandMean(meanModel?.integrated);
  let best: KneeHeadline["best"] = null;
  for (const m of p.models) {
    if (m.kind !== "member") continue;
    const v = bandMean(m.integrated);
    if (v != null && (best == null || v > best.value)) {
      const label = m.label ?? m.id;
      best = { name: memberName(label), label, value: v };
    }
  }
  const head = gate ?? mean;
  const bands = p.bands ?? [];
  const source = gateModel ?? meanModel;
  return {
    gate, mean, best,
    vsBest: head != null && best != null ? head - best.value : null,
    vsMean: gate != null && mean != null ? gate - mean : null,
    perBand: bands.map((band, i) => ({ band, value: finite(source?.integrated?.[i]) })),
    nFields: finite(p.n_fields),
    stale: !!p.stale,
  };
}

/** `GET /api/system/production` (routes/system.py): the STARFULL eval
 *  summary's scalar keys, its staleness and the member counts. */
export type ProductionPayload = {
  eval_summary?: EvalSummary | null;
  evaluated_at?: string | null;
  stale?: boolean;
  stale_reason?: string | null;
  members?: number;
  starless_members?: number;
};

/** The slice of the heavy `GET /ensemble/status.json` the fallback reads. */
export type EnsembleStatusSlice = {
  eval_summary?: Record<string, unknown> | null;
  eval_summary_stale?: boolean;
  members?: { name: string; starless?: boolean }[];
};

/** Fallback for a server that predates `/api/system/production` (404): the
 *  same payload derived from `/ensemble/status.json?mode=starfull` (its
 *  eval summary's scalar keys; members counted per regime). */
export function productionFromStatus(status: EnsembleStatusSlice | null | undefined): ProductionPayload | null {
  if (!status) return null;
  const raw = status.eval_summary;
  const summary = raw && typeof raw === "object"
    ? Object.fromEntries(Object.entries(raw).filter(([, v]) => v === null || ["boolean", "number", "string"].includes(typeof v)))
    : null;
  const members = Array.isArray(status.members) ? status.members : [];
  const stale = !!status.eval_summary_stale;
  return {
    eval_summary: summary as EvalSummary | null,
    stale,
    stale_reason: stale ? "Membership changed since the last evaluation" : null,
    members: members.filter((m) => !m.starless).length,
    starless_members: members.filter((m) => m.starless).length,
  };
}

type ModelsPayload = { members?: string[] } | null | undefined;

/** Active STARFULL member count: `/api/models` regime labels first, else
 *  `/api/system/production`'s count (the same regime labels); plus the
 *  starless (opt-in) count. */
export function starfullMembers(models: ModelsPayload, production: ProductionPayload | null | undefined):
  { count: number; source: "models" | "production"; starless: number | null } | null {
  const starless = typeof production?.starless_members === "number" ? production.starless_members : null;
  if (Array.isArray(models?.members)) return { count: models!.members!.length, source: "models", starless };
  if (typeof production?.members === "number") return { count: production.members, source: "production", starless };
  return null;
}

export type ModelSpecRow = {
  spec: string;
  available: boolean;
  /** The spec's identity (a Sky › Compare run records the one it scored). */
  fingerprint?: string | null;
  reason?: string | null;
  label?: string;
  combiner_kind?: string | null;
  details?: { mix_space?: string | null; fitted_at?: string | null; [key: string]: unknown } | null;
  [key: string]: unknown;
};

export type ModelsCatalog = { production_kind?: string; members?: string[]; models?: ModelSpecRow[] };

export type ProductionModel = {
  /** "spatial gate" — the combiner's name without the "Production ·" prefix or its qualifiers. */
  label: string;
  mix: string | null;
  fittedAt: string | null;
  available: boolean;
  reason: string | null;
};

/** The production combiner as `/api/models` describes it (null without one). */
export function productionModel(catalog: ModelsCatalog | null | undefined): ProductionModel | null {
  const spec = catalog?.models?.find((m) => m.spec === "production");
  if (!spec) return null;
  const fromLabel = (spec.label ?? "").replace(/^production\s*·\s*/i, "").replace(/\s*\(.*\)\s*$/, "").trim();
  const kind = spec.combiner_kind ?? catalog?.production_kind ?? "";
  const label = fromLabel || kind.replace(/_/g, " ") || "production";
  const details = spec.details ?? null;
  return {
    label,
    mix: typeof details?.mix_space === "string" ? details.mix_space : null,
    fittedAt: typeof details?.fitted_at === "string" ? details.fitted_at : null,
    available: !!spec.available,
    reason: spec.reason ?? null,
  };
}

/* ── members ────────────────────────────────────────────────────────────── */

/** ["member_199", …, "member_202"] → "members 199–202" (runs of consecutive
 *  numbers become ranges); null for an empty list. */
export function memberRange(names: readonly string[]): string | null {
  const nums = names.map((n) => String(n).trim().replace(/^member_/, "")).filter((n) => /^\d+$/.test(n));
  if (!nums.length) return null;
  const sorted = [...new Set(nums)].sort((a, b) => Number(a) - Number(b));
  const parts: string[] = [];
  for (let i = 0; i < sorted.length;) {
    let j = i;
    while (j + 1 < sorted.length && Number(sorted[j + 1]) === Number(sorted[j]) + 1) j += 1;
    // Two in a row read better as a list ("195, 196") than as a range.
    if (j - i >= 2) parts.push(`${sorted[i]}–${sorted[j]}`);
    else for (let k = i; k <= j; k += 1) parts.push(sorted[k]);
    i = j + 1;
  }
  return `${sorted.length === 1 ? "member" : "members"} ${parts.join(", ")}`;
}

/* ── tracking catch-up note ────────────────────────────────────────────── */

export type Unlogged = { at: string; label: string };

/** The `tracking` check's unlogged results (as the server lists them, newest first). */
export function unloggedItems(check: { facts?: Record<string, unknown> } | null | undefined): Unlogged[] {
  const raw = check?.facts?.unlogged;
  if (!Array.isArray(raw)) return [];
  return raw.filter((u): u is Unlogged => !!u && typeof u === "object" && typeof (u as Unlogged).label === "string")
    .map((u) => ({ at: String(u.at ?? ""), label: u.label }));
}

type CatchUpFacts = { knee: KneeHeadline | null; prod: ProductionHeadline | null; members: number | null; production: ProductionModel | null };

const fmt = (v: number) => v.toFixed(2);
const signed = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`;

function kneeLine(k: KneeHeadline | null): string | null {
  const head = k?.gate ?? k?.mean;
  if (!k || head == null) return null;
  const what = k.gate != null ? "production gate" : "plain mean";
  const vs = [k.best && k.vsBest != null ? `${signed(k.vsBest)} dB vs ${k.best.name.replace("member_", "member ")}` : null,
    k.vsMean != null ? `${signed(k.vsMean)} dB vs plain mean` : null, k.nFields != null ? `${k.nFields} fields` : null].filter(Boolean).join(", ");
  return `∫PSNR ${what} ${fmt(head)} dB${vs ? ` (${vs})` : ""}${k.stale ? ", stale" : ""}`;
}

function testLine(p: ProductionHeadline | null): string | null {
  if (!p) return null;
  if (p.kind === "mean") return `test PSNR plain mean ${fmt(p.psnr)} dB${p.vsMeanMember != null ? ` (${signed(p.vsMeanMember)} dB vs mean member)` : ""}`;
  const vs = [p.vsBest != null ? `${signed(p.vsBest)} dB vs best member` : null, p.vsMean != null ? `${signed(p.vsMean)} dB vs plain mean` : null].filter(Boolean).join(", ");
  return `test PSNR production gate ${fmt(p.psnr)} dB${vs ? ` (${vs})` : ""}${p.stale ? ", summary stale" : ""}`;
}

const gateLine = (m: ProductionModel | null): string | null => (m
  ? `${m.label}${m.mix ? `, ${m.mix} mix` : ""}, ${m.available ? "fitted for the current members" : `out of date${m.reason ? ` (${m.reason})` : ""}`}`
  : null);

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

/** The Home "Log to notebook" entry: one line per result the notebook has not
 *  logged yet (the `tracking` check), else the production model as it is. */
export function trackingCatchUpNote(check: { facts?: Record<string, unknown> } | null | undefined, f: CatchUpFacts): string {
  const items = unloggedItems(check);
  const members = f.members != null ? `${f.members} STARFULL members` : null;
  if (!items.length) {
    return [`**Production model**${members ? ` — ${members}` : ""}`, "",
      ...[kneeLine(f.knee), testLine(f.prod), gateLine(f.production)].filter((l): l is string => !!l).map((l) => `- ${cap(l)}`)].join("\n");
  }
  const last = check?.facts?.last_entry;
  const detail = (label: string): string | null => {
    const l = label.toLowerCase();
    if (l.includes("knee")) return kneeLine(f.knee);
    if (l.includes("evaluation")) return [testLine(f.prod), members].filter(Boolean).join(", ") || null;
    if (l.includes("gate")) return gateLine(f.production);
    if (l.includes("experiment")) return "see Sky › Compare";
    return null;
  };
  return [
    `**Catch-up** — results since the last tracking entry${typeof last === "string" ? ` (${utcText(last)})` : ""}`, "",
    ...items.map((u) => { const d = detail(u.label); return `- ${u.label} — ${utcText(u.at)}${d ? `: ${d}` : ""}`; }),
  ].join("\n");
}

/* ── the real-data benchmark (Sky › Compare) ───────────────────────────── */

/** One Sky › Compare run of `GET /api/experiments` (the fields read here). */
export type ExperimentSummary = {
  id: string; label?: string; created?: string;
  summary?: Record<string, { n_tiles?: number; per_band?: Record<string, { hole_pct?: number | null }>; summary?: { hole_pct_mean?: number | null } }> | null;
};
/** `GET /api/experiments/<id>`: the fingerprint each spec was scored with. */
export type ExperimentDetail = { id: string; fingerprints?: Record<string, string | null> | null };
export type CatalogFingerprints = { models?: readonly { spec: string; fingerprint?: string | null }[] | null };

export type RealHoles =
  | { state: "loading" }
  | { state: "none"; why: "no-run" | "earlier"; expId?: string; created?: string; label?: string }
  | { state: "current"; expId: string; created?: string; label?: string; nTiles: number | null;
      /** The band with the most holes (null without per-band values). */
      worst: { band: string; pct: number } | null; mean: number | null };

const BAND_SHORT: Record<string, string> = { VIS: "VIS", Y_E: "Y", J_E: "J", H_E: "H" };

/** The newest Sky › Compare run that scored production, or null. */
export function productionRun(exps: readonly ExperimentSummary[] | null | undefined): ExperimentSummary | null {
  return [...(exps ?? [])].sort((a, b) => String(b.created ?? "").localeCompare(String(a.created ?? "")))
    .find((e) => !!e.summary && "production" in e.summary) ?? null;
}

/** Production's real holes from its newest Sky › Compare run, when that run
 *  scored the CURRENT production fit (see the definitions above). */
export function productionRealHoles(exps: readonly ExperimentSummary[] | null | undefined,
  detail: ExperimentDetail | null | undefined, catalog: CatalogFingerprints | null | undefined): RealHoles {
  if (exps == null) return { state: "loading" };
  const run = productionRun(exps);
  if (!run) return { state: "none", why: "no-run" };
  if (!detail || detail.id !== run.id || !catalog) return { state: "loading" };
  const current = catalog.models?.find((m) => m.spec === "production")?.fingerprint ?? null;
  const scored = detail.fingerprints?.production ?? null;
  if (!current || !scored || current !== scored) return { state: "none", why: "earlier", expId: run.id, created: run.created, label: run.label };
  const agg = run.summary?.production;
  let worst: { band: string; pct: number } | null = null;
  for (const [band, v] of Object.entries(agg?.per_band ?? {})) {
    const pct = finite(v?.hole_pct);
    if (pct != null && (worst == null || pct > worst.pct)) worst = { band: BAND_SHORT[band] ?? band, pct };
  }
  return { state: "current", expId: run.id, created: run.created, label: run.label, nTiles: finite(agg?.n_tiles),
    worst, mean: finite(agg?.summary?.hole_pct_mean) };
}

/** A Sky › Compare run's page. */
export const compareRunPath = (expId: string) => `/sky/compare?${new URLSearchParams({ exp: expId }).toString()}`;

/* ── the caption under the verdict ─────────────────────────────────────── */

/** 0.1 → "0.1", 10000 → "10⁴" (a knee in e⁻). */
export function kneeText(v: number): string {
  const exp = Math.log10(v);
  return v >= 1000 && Number.isInteger(exp) ? formatPow10(exp) : String(v);
}

/** "30 members · gate fitted 4 h ago · 100 test fields · knees 0.1–10⁴ e⁻":
 *  what the verdict's numbers are made of (parts without data are left out). */
export function homeCaption({ members, production, knee, nFields, now = Date.now() }: {
  members: number | null; production: ProductionModel | null; knee: KneePayload | null | undefined; nFields: number | null; now?: number;
}): string[] {
  const range = knee?.available ? knee.integration : null;
  return [
    members != null ? `${members} members` : null,
    production ? (production.available ? (production.fittedAt ? `gate fitted ${formatRelative(production.fittedAt, now)}` : "gate fitted") : "gate out of date") : null,
    nFields != null ? `${nFields} test fields` : null,
    range?.from_e != null && range.to_e != null ? `knees ${kneeText(range.from_e)}–${kneeText(range.to_e)} e⁻` : null,
  ].filter((p): p is string => !!p);
}
