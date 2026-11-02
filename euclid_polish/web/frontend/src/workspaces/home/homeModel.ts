/* The Home page's numbers, as pure functions (homeModel.test.ts; the Loop
 * strip, the running line and the thumbnails are in loop.ts).
 *
 * Definitions:
 *  - production model: ONE registered universal model, `/api/models`
 *    `production` ("Universal 257"; `available` = it can run).
 *  - its score: the knee-integrated PSNR — the mean over the four bands of
 *    its PSNR integrated over log asinh knee 0.1–1e4 e⁻ on the test fields —
 *    from `/ensemble/universal.json` (`production`, and the production row's
 *    `scored_at`), the number Models › Universal shows.
 *  - real holes: the newest Sky › Compare run that scored production
 *    (`/api/experiments`), its worst band's hole % (the share of bright LR
 *    pixels the SR blanks), shown only when that run scored THIS production
 *    model (its recorded fingerprint, `/api/experiments/<id>`, equals the
 *    catalogue's, `/api/models`); else "no real benchmark for" it.
 *  - tracking catch-up: the `tracking` health check's `facts.unlogged`
 *    (results written after the newest `## <ISO>` heading of log.md), one
 *    line each, for the "Log to notebook" entry
 *    (workspaces/shared/LogToNotebook: Notebook › Log, prefilled). */
import { formatRelative } from "../../format";
import { utcText } from "../shared/noteText";

const finite = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

/** Mean of the finite values (null when none). */
export function bandMean(values: readonly (number | null | undefined)[] | null | undefined): number | null {
  const ok = (values ?? []).filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  return ok.length ? ok.reduce((a, b) => a + b, 0) / ok.length : null;
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

/** `/api/models` `production`: what production is. */
export type CatalogProduction = {
  kind?: string; spec?: string; name?: string | null; label?: string; available?: boolean; reason?: string | null;
};

export type ModelsCatalog = { production?: CatalogProduction | null; models?: ModelSpecRow[] };

export type ProductionModel = {
  /** The universal model's label ("Universal 257") and name ("member_257"). */
  label: string;
  name: string | null;
  available: boolean;
  reason: string | null;
};

/** The production model as `/api/models` describes it (null without one). */
export function productionModel(catalog: ModelsCatalog | null | undefined): ProductionModel | null {
  const spec = catalog?.models?.find((m) => m.spec === "production");
  const summary = catalog?.production ?? null;
  if (!spec && !summary) return null;
  const fromLabel = (spec?.label ?? "").replace(/^production\s*·\s*/i, "").replace(/\s*\(.*\)\s*$/, "").trim();
  return {
    label: summary?.label || fromLabel || "a universal model",
    name: summary?.name ?? (typeof spec?.details?.member === "string" ? spec.details.member : null),
    available: !!(spec?.available ?? summary?.available),
    reason: spec?.reason ?? summary?.reason ?? null,
  };
}

/** The production model's score (UniversalScore) from GET
 *  /ensemble/universal.json: its ∫PSNR per band and their mean. */
export type UniversalScore = { mean: number | null; bands: (number | null)[]; stale: boolean; scoredAt: string | null };
/** The slice of GET /ensemble/universal.json read here. */
export type UniversalSlice = {
  production?: { integrated?: (number | null)[] | null; mean?: number | null; stale?: boolean | null } | null;
  production_model?: string | null;
  models?: { name: string; score?: { scored_at?: string | null } | null }[] | null;
};

export function universalScore(u: UniversalSlice | null | undefined): UniversalScore | null {
  const p = u?.production;
  if (!p) return null;
  const bands = (p.integrated ?? []).map((v) => finite(v));
  const mean = finite(p.mean) ?? (bands.length === 4 && bands.every((v) => v != null) ? bandMean(bands) : null);
  const row = u?.models?.find((m) => m.name === u.production_model);
  return { mean, bands, stale: !!p.stale, scoredAt: row?.score?.scored_at ?? null };
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

type CatchUpFacts = { production: ProductionModel | null; score?: UniversalScore | null };

/** The production model in one line, with its own ∫PSNR. */
function productionLine(m: ProductionModel | null, score?: UniversalScore | null): string | null {
  if (!m) return null;
  const value = score?.mean != null ? `, ∫PSNR ${score.mean.toFixed(2)} dB${score.stale ? " (stale score)" : ""}` : "";
  return `${m.label} is production${value}${m.available ? "" : `, cannot run${m.reason ? ` (${m.reason})` : ""}`}`;
}

/** The Home "Log to notebook" entry: one line per result the notebook has not
 *  logged yet (the `tracking` check), else the production model as it is. */
export function trackingCatchUpNote(check: { facts?: Record<string, unknown> } | null | undefined, f: CatchUpFacts): string {
  const items = unloggedItems(check);
  const line = productionLine(f.production, f.score);
  if (!items.length) return ["**Production model**", "", ...(line ? [`- ${line}`] : [])].join("\n");
  const last = check?.facts?.last_entry;
  const detail = (label: string): string | null => {
    const l = label.toLowerCase();
    if (l.includes("production") || l.includes("universal")) return line;
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
 *  scored the CURRENT production model (see the definitions above). */
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

/** "scored 3 h ago": what the verdict's number is made of ("cannot run"
 *  when the model is unavailable; nothing before it is scored). */
export function homeCaption({ production, score, now = Date.now() }: {
  production: ProductionModel | null; score?: UniversalScore | null; now?: number;
}): string[] {
  if (!production) return [];
  return [!production.available ? "cannot run" : score?.scoredAt ? `scored ${formatRelative(score.scoredAt, now)}` : null]
    .filter((p): p is string => !!p);
}
