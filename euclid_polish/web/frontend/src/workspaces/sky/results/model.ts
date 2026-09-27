/* Pure logic of the Sky › Real results / Experiments / Catalog-eval tabs:
 * row flattening, model grouping, metric tables and chart series, the
 * tracking-log summary. No React, no fetch (unit-tested in model.test.ts). */
import { formatNumber } from "../../../format";
import { extent } from "../../../ticks";
import type { Tone } from "../../../ui";
import {
  BANDS, SOURCES, type BandMetrics, type ExperimentRecord, type Metrics, type ModelSpecRow,
  type TileList, type TileRow,
} from "./api";

/* ── small value helpers ───────────────────────────────────────────────── */

/** A finite number from a JSON number or a CSV string, else null. */
export function num(v: unknown): number | null {
  if (typeof v === "number") return Number.isFinite(v) ? v : null;
  if (typeof v === "string" && v.trim() !== "") {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

export const STATE_TONE: Record<string, Tone> = {
  current: "good", stale: "warn", missing: "neutral", unavailable: "neutral", unknown: "neutral",
  computed: "good", reused: "info", done: "good", running: "info", failed: "bad", cancelled: "neutral",
};

export const BAND_LABEL: Record<string, string> = { VIS: "VIS", Y_E: "Y", J_E: "J", H_E: "H" };
export const bandLabel = (band: string): string => BAND_LABEL[band] ?? band;

/* ── model specs ───────────────────────────────────────────────────────── */

/** Sort key: production, mean, rbf, gate variants, members (then natural). */
export function specRank(spec: string): number {
  if (spec === "production") return 0;
  if (spec === "mean") return 1;
  if (spec === "rbf") return 2;
  if (spec.startsWith("gate:")) return 3;
  if (spec.startsWith("member:")) return 4;
  return 5;
}

const COLLATOR = new Intl.Collator("en", { numeric: true, sensitivity: "base" });

export function sortSpecs(specs: Iterable<string>): string[] {
  return [...specs].sort((a, b) => specRank(a) - specRank(b) || COLLATOR.compare(a, b));
}

/** Compact label: `member:member_196` → `m196`, `gate:26m` → `gate 26m`. */
export function specShort(spec: string): string {
  if (spec.startsWith("member:")) return `m${spec.slice("member:".length).replace(/^member_/, "")}`;
  if (spec.startsWith("gate:")) return `gate ${spec.slice("gate:".length)}`;
  return spec;
}

export type ModelGroup = { id: string; label: string; items: ModelSpecRow[] };

/** The model catalogue grouped for pickers (spec §7.3): production, mean,
 *  rbf, gate variants, members — each group in catalogue order. */
export function groupModels(models: readonly ModelSpecRow[]): ModelGroup[] {
  const groups: ModelGroup[] = [
    { id: "production", label: "Production", items: [] },
    { id: "mean", label: "Mean", items: [] },
    { id: "rbf", label: "RBF", items: [] },
    { id: "gate", label: "Gate variants", items: [] },
    { id: "member", label: "Members", items: [] },
  ];
  const byId = new Map(groups.map((g) => [g.id, g]));
  for (const m of models) (byId.get(String(m.kind)) ?? byId.get("member"))!.items.push(m);
  return groups.filter((g) => g.items.length);
}

/** How many members a model reads: "6 of 20 members" when a pruned gate
 *  reads fewer members than it was fitted on (`reads` vs `n_fitted`; older
 *  servers sent the fitted total as `n_members`). */
export function membersText(m: Pick<ModelSpecRow, "n_members" | "n_fitted" | "reads" | "members">): string {
  const total = m.n_fitted ?? m.members?.length ?? m.n_members ?? 0;
  const reads = m.reads?.length ?? total;
  if (!total && !reads) return "";
  const noun = (n: number) => `member${n === 1 ? "" : "s"}`;
  return reads < total ? `${reads} of ${total} ${noun(total)}` : `${reads || total} ${noun(reads || total)}`;
}

/** Specs to keep after the catalogue refreshed: known and available ones. */
export function runnableSelection(selected: readonly string[], models: readonly ModelSpecRow[]): string[] {
  const ok = new Set(models.filter((m) => m.available).map((m) => m.spec));
  return selected.filter((s) => ok.has(s));
}

/* ── real tiles ────────────────────────────────────────────────────────── */

/** Every source's rows in one list (a ref appears once). */
export function flattenTiles(lists: readonly (TileList | null | undefined)[]): TileRow[] {
  const seen = new Set<string>();
  const out: TileRow[] = [];
  for (const list of lists) {
    for (const row of list?.tiles ?? []) {
      if (seen.has(row.ref)) continue;
      seen.add(row.ref);
      out.push(row);
    }
  }
  return out;
}

export type StateCounts = { total: number; current: number; stale: number; missing: number };

export function productionCounts(rows: readonly TileRow[]): StateCounts {
  const out: StateCounts = { total: rows.length, current: 0, stale: 0, missing: 0 };
  for (const r of rows) {
    const s = r.production_state;
    if (s === "current" || s === "stale") out[s] += 1;
    else out.missing += 1;
  }
  return out;
}

/** A tile's model outputs, ordered, with their state. */
export function tileModels(row: Pick<TileRow, "models">): { spec: string; state: string; legacy: boolean }[] {
  const models = row.models ?? {};
  return sortSpecs(Object.keys(models)).map((spec) => ({
    spec, state: String(models[spec]?.state ?? "unknown"), legacy: !!models[spec]?.legacy,
  }));
}

/** "Compute metrics" as experiments: the tiles grouped by the set of CURRENT
 *  outputs that have no metrics yet. An experiment reuses a current SR and
 *  only scores it, but runs every spec on every tile it is given — so tiles
 *  are grouped, never handed a spec whose SR they do not have. */
export function metricsPlan(rows: readonly Pick<TileRow, "ref" | "models">[]): { specs: string[]; refs: string[] }[] {
  const groups = new Map<string, { specs: string[]; refs: string[] }>();
  for (const r of rows) {
    const models = r.models ?? {};
    const specs = sortSpecs(Object.keys(models).filter((s) => models[s]?.state === "current" && !models[s]?.summary));
    if (!specs.length) continue;
    const key = specs.join(",");
    const g = groups.get(key) ?? { specs, refs: [] };
    g.refs.push(r.ref);
    groups.set(key, g);
  }
  return [...groups.values()];
}

/** Filter by production state (`all` keeps every row). */
export function filterByState(rows: readonly TileRow[], state: string): TileRow[] {
  if (!state || state === "all") return [...rows];
  return rows.filter((r) => (r.production_state ?? "missing") === state);
}

/** Refs typed or pasted (comma / whitespace separated), `source/id` with a
 *  known source, de-duplicated in order. */
export function parseRefs(text: string): string[] {
  const out: string[] = [];
  for (const raw of text.split(/[\s,;]+/)) {
    const ref = raw.trim();
    const i = ref.indexOf("/");
    if (i <= 0 || i === ref.length - 1) continue;
    if (!(SOURCES as readonly string[]).includes(ref.slice(0, i))) continue;
    if (ref.slice(i + 1).includes("/")) continue;
    if (!out.includes(ref)) out.push(ref);
  }
  return out;
}

/** The first spec with metrics on a tile: production, else any. */
export function headlineSpec(row: Pick<TileRow, "models">): string | null {
  const models = row.models ?? {};
  const specs = sortSpecs(Object.keys(models));
  return specs.find((s) => models[s]?.summary) ?? null;
}

/* ── metrics ───────────────────────────────────────────────────────────── */

export type MetricKey = keyof BandMetrics;
export type MetricDef = {
  key: MetricKey; label: string; short: string; digits: number; unit?: string;
  /** "lower" / "higher" is better, "one" = closest to 1. */
  better?: "lower" | "higher" | "one"; hint: string;
};

export const METRICS: MetricDef[] = [
  { key: "hole_pct", label: "Hole %", short: "holes", digits: 1, unit: "%", better: "lower",
    hint: "SR pixels under the brightest 1 % of LR pixels with SR < 0.5 × LR/4." },
  { key: "hole_pct_100sigma", label: "Hole % (> 100σ)", short: "holes >100σ", digits: 1, unit: "%", better: "lower",
    hint: "Hole % over the bright-1 % pixels that are also above 100σ (faint tiles: the top 1 % is noise)." },
  { key: "pct_R_lt_0p8", label: "% R < 0.8", short: "R<0.8", digits: 1, unit: "%", better: "lower",
    hint: "Share of bright, locally dominant peaks whose enclosed-flux ratio R = F_SR/F_LR (boxes 0.3–1.7″) drops below 0.8." },
  { key: "pct_R_lt_0p5", label: "% R < 0.5", short: "R<0.5", digits: 1, unit: "%", better: "lower",
    hint: "Share of those peaks with R below 0.5." },
  { key: "median_R", label: "Median R", short: "R̃", digits: 3, better: "one",
    hint: "Median enclosed-flux ratio over the peaks (1 = flux conserved)." },
  { key: "min_R", label: "Min R", short: "R min", digits: 3, better: "one", hint: "Lowest enclosed-flux ratio." },
  { key: "flux_ratio", label: "Flux SR/LR", short: "flux", digits: 4, better: "one",
    hint: "Total SR flux over total LR flux of the band." },
  { key: "n_peaks", label: "Peaks", short: "peaks", digits: 0,
    hint: "Bright (> 100σ), locally dominant LR peaks that passed the central-pixel-fraction cut." },
  { key: "n_edge", label: "Edge", short: "edge", digits: 0, hint: "Peaks too close to the edge for the 1.7″ box." },
  { key: "n_artifacts", label: "Artifacts", short: "art.", digits: 0,
    hint: "Peaks rejected by the central-pixel-fraction cut (VIS > 0.25, NISP > 0.14)." },
];
export const METRIC_BY_KEY: Record<string, MetricDef> = Object.fromEntries(METRICS.map((m) => [m.key, m]));
/** Metrics worth a chart (the rest are counts). */
export const CHART_METRICS: MetricKey[] = ["hole_pct", "hole_pct_100sigma", "pct_R_lt_0p8", "pct_R_lt_0p5", "median_R", "flux_ratio"];

export function formatMetric(key: string, v: unknown): string {
  const def = METRIC_BY_KEY[key];
  const n = num(v);
  if (n == null) return "—";
  return formatNumber(n, { digits: def?.digits ?? 3 });
}

/** One metrics table row: a (model, band) pair. */
export type MetricRow = { key: string; spec: string; label: string; band: string } & Record<string, unknown>;

/** The per-band metrics of an experiment record: pooled over every tile
 *  (`scope = "pooled"`, record.summary) or of one tile (`scope` = its ref). */
export function metricsFor(record: ExperimentRecord, scope: string, spec: string): Metrics | null {
  if (scope === "pooled") return record.summary?.[spec] ?? null;
  return record.results?.[scope]?.[spec]?.metrics ?? null;
}

export function recordSpecs(record: ExperimentRecord): string[] {
  const specs = new Set<string>();
  for (const spec of Object.keys(record.summary ?? {})) specs.add(spec);
  for (const byTile of Object.values(record.results ?? {})) for (const spec of Object.keys(byTile)) specs.add(spec);
  for (const spec of record.models ?? []) if (!record.skipped?.[spec]) specs.add(spec);
  return sortSpecs(specs);
}

export function metricRows(record: ExperimentRecord, scope: string): MetricRow[] {
  const rows: MetricRow[] = [];
  for (const spec of recordSpecs(record)) {
    const metrics = metricsFor(record, scope, spec);
    const perBand = metrics?.per_band ?? {};
    const bands = metrics?.bands?.length ? metrics.bands : BANDS.filter((b) => perBand[b]);
    for (const band of bands) {
      const m = perBand[band];
      if (!m) continue;
      rows.push({
        key: `${spec}|${band}`, spec, label: record.model_labels?.[spec] ?? spec, band, ...m,
      });
    }
  }
  return rows;
}

export type BandSeries = { spec: string; label: string; x: number[]; y: (number | null)[] };

/** One series per model: `metric` per band (x = band index + a small
 *  per-model offset so overlapping dots stay readable). */
export function bandSeries(record: ExperimentRecord, scope: string, metric: MetricKey): BandSeries[] {
  const specs = recordSpecs(record);
  const n = specs.length;
  const spread = n > 1 ? Math.min(0.36, 0.06 * (n - 1)) : 0;
  return specs.map((spec, i) => {
    const perBand = metricsFor(record, scope, spec)?.per_band ?? {};
    const offset = n > 1 ? -spread / 2 + (spread * i) / (n - 1) : 0;
    return {
      spec, label: record.model_labels?.[spec] ?? spec,
      x: BANDS.map((_, b) => b + offset),
      y: BANDS.map((band) => num(perBand[band]?.[metric])),
    };
  });
}

/** The value domain of a set of series (padded; `[0, 1]` when empty). */
export function seriesDomain(series: readonly BandSeries[], metric: MetricKey): [number, number] {
  const values = series.flatMap((s) => s.y).filter((v): v is number => v != null);
  const def = METRIC_BY_KEY[metric];
  const span = extent(values);
  if (!span) return [0, 1];
  let [lo, hi] = span;
  if (def?.unit === "%") { lo = 0; hi = Math.max(hi, 1); }
  if (def?.better === "one") { lo = Math.min(lo, 1); hi = Math.max(hi, 1); }
  if (hi === lo) { hi = lo + 1; }
  const pad = (hi - lo) * 0.08;
  return [def?.unit === "%" ? 0 : lo - pad, hi + pad];
}

/* ── what an experiment costs (the Run confirm, the New-experiment form) ─ */

export type ExperimentCost = {
  tiles: number;
  /** Runnable models (the server skips unavailable ones). */
  models: number;
  /** Distinct member SRs the models need per tile (the union of what they read). */
  members: number;
  /** members × tiles: the member inferences at most (cached member SRs are reused). */
  inferences: number;
  /** models × tiles: the (tile, model) outputs written or re-scored. */
  outputs: number;
  skipped: string[];
  unknown: string[];
};

/** The upper bound of an experiment's work from the model catalogue
 *  (`GET /api/models`: each spec's `reads`, else its `members`). The member
 *  cache is not visible to the client, so this is "at most". */
export function experimentCost(specs: readonly string[], catalogue: readonly ModelSpecRow[], nTiles: number): ExperimentCost {
  const bySpec = new Map(catalogue.map((m) => [m.spec, m]));
  const members = new Set<string>();
  const skipped: string[] = [], unknown: string[] = [];
  let models = 0;
  for (const spec of specs) {
    const m = bySpec.get(spec);
    if (!m) { unknown.push(spec); continue; }
    if (!m.available) { skipped.push(spec); continue; }
    models += 1;
    for (const label of m.reads ?? m.members ?? []) members.add(label);
  }
  const tiles = Math.max(0, nTiles);
  return { tiles, models, members: members.size, inferences: members.size * tiles, outputs: models * tiles, skipped, unknown };
}

const count = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

/** The cost as a sentence: "4 outputs (2 models on 2 tiles). Needs 4 member
 *  SRs per tile: at most 8 member inferences on this machine; cached ones are
 *  reused." (plain words, no "A · B" label string). */
export function experimentCostText(c: ExperimentCost | null): string {
  if (!c) return "";
  const work = `${count(c.outputs, "output")} (${count(c.models, "model")} on ${count(c.tiles, "tile")}).`;
  if (!c.members) return work;
  return `${work} Needs ${count(c.members, "member SR")} per tile: at most ${count(c.inferences, "member inference")} on this machine; cached ones are reused.`;
}

/** The experiment a visit to Sky › Experiments opens: the `?exp=` one, else
 *  the newest (so the plain tab link opens on a comparison, not a form) —
 *  unless tiles were handed over (`?tiles=`): then the new-experiment form
 *  is the point. Ids start with their timestamp (`20260927-024251-…`), which
 *  orders them when `created` is missing. */
export function defaultExperimentId(
  history: readonly { id: string; created?: string | null }[], exp: string, tiles: readonly string[],
): string {
  if (exp) return exp;
  if (tiles.length || !history.length) return "";
  const newer = (a: { id: string; created?: string | null }, b: { id: string; created?: string | null }) =>
    (a.created && b.created ? b.created > a.created : b.id > a.id);
  return history.reduce((a, b) => (newer(a, b) ? b : a)).id;
}

/** A metric column header in sentence case, as the kit's headers read
 *  ("holes >100σ" → "Holes >100σ"; σ stays σ — no CSS text-transform, which
 *  would have turned it into Σ, a sum sign). */
export function metricHeader(short: string): string {
  return short ? short[0].toUpperCase() + short.slice(1) : short;
}

/* ── the real-tile card ────────────────────────────────────────────────── */

/** The `real` viewer's params for one tile: exactly its own model outputs,
 *  so the tier picker offers only tiers that exist for it. Without `models`
 *  the server would list every spec any tile of the source has; "," is the
 *  explicit empty list (LR and JWST only). */
export function realTileViewerParams(source: string, specs: readonly string[]): { source: string; models: string } {
  return { source, models: sortSpecs(specs).join(",") || "," };
}

/** The card's first frames — two, so they are large in the ~380 px
 *  inspector (three stacked at ~170 px were too small to judge): LR and the
 *  first model output, else LR and the JWST truth. Every other tier is one
 *  chip away in the viewer's bar. */
export function cardViewerTiers(specs: readonly string[], hasJwst: boolean): string[] {
  const first = sortSpecs(specs)[0];
  if (first) return ["lr", `m:${first}`];
  return ["lr", ...(hasJwst ? ["jwst"] : [])];
}

/** Where an output came from, in words ("legacy nexus-field · 2 members · spatial gate"). */
export function outputOrigin(m: {
  legacy?: boolean; origin?: string | null; member_labels?: readonly string[] | null; combiner_kind?: string | null;
}): string {
  const parts: string[] = [];
  if (m.legacy) parts.push(`legacy ${m.origin ?? "record"}`);
  if (m.member_labels?.length) parts.push(`${m.member_labels.length} member${m.member_labels.length === 1 ? "" : "s"}`);
  if (m.combiner_kind) parts.push(m.combiner_kind.replace(/_/g, " "));
  return parts.join(" · ");
}

/** "Run models…" preselection: production and the mean when missing or
 *  stale; both when both are current (a re-score). */
export function defaultRunSpecs(models: Record<string, { state?: string | null } | undefined>): string[] {
  const missing = ["production", "mean"].filter((s) => models[s]?.state !== "current");
  return missing.length ? missing : ["production", "mean"];
}

/* ── the tracking-log summary ──────────────────────────────────────────── */

const cell = (v: unknown, digits: number) => {
  const n = num(v);
  return n == null ? "—" : formatNumber(n, { digits });
};

/** A concise markdown summary of an experiment for the tracking notebook
 *  (the store adds the `## <ISO>` heading itself). */
export function experimentMarkdown(record: ExperimentRecord): string {
  const specs = recordSpecs(record);
  const tiles = record.tiles ?? [];
  const lines: string[] = [];
  lines.push(`**Real-data experiment \`${record.id}\`**${record.label ? ` — ${record.label}` : ""} (${record.status ?? "?"})`);
  lines.push("");
  const tileList = tiles.length <= 6 ? tiles.map((t) => `\`${t}\``).join(", ") : `${tiles.slice(0, 5).map((t) => `\`${t}\``).join(", ")} … (+${tiles.length - 5})`;
  lines.push(`- tiles (${tiles.length}): ${tileList || "—"}`);
  lines.push(`- models: ${specs.map((s) => `\`${s}\``).join(", ") || "—"}`);
  const skipped = Object.entries(record.skipped ?? {});
  if (skipped.length) lines.push(`- skipped: ${skipped.map(([s, why]) => `\`${s}\` (${why})`).join("; ")}`);
  lines.push("");
  lines.push("Pooled over tiles — hole % per band (VIS/Y/J/H), % R<0.8, median R, flux SR/LR (VIS):");
  lines.push("");
  lines.push("| model | hole % V/Y/J/H | % R<0.8 | R̃ | flux VIS |");
  lines.push("|---|---|---|---|---|");
  for (const spec of specs) {
    const m = record.summary?.[spec];
    const pb = m?.per_band ?? {};
    const holes = BANDS.map((b) => cell(pb[b]?.hole_pct, 1)).join(" / ");
    lines.push(`| \`${spec}\` | ${holes} | ${cell(m?.summary?.pct_R_lt_0p8, 1)} | ${cell(m?.summary?.median_R, 3)} | ${cell(pb.VIS?.flux_ratio, 3)} |`);
  }
  const errors = Object.entries(record.errors ?? {});
  if (errors.length) {
    lines.push("");
    lines.push(`Errors (${errors.length}): ${errors.slice(0, 3).map(([k, v]) => `\`${k}\`: ${v}`).join("; ")}${errors.length > 3 ? " …" : ""}`);
  }
  return lines.join("\n");
}

/* ── catalogue evaluation ──────────────────────────────────────────────── */

export const EVAL_GROUPS: { id: string; label: string }[] = [
  { id: "A", label: "Lens A" }, { id: "B", label: "Lens B" }, { id: "C", label: "Lens C" },
  { id: "gal", label: "Galaxies" }, { id: "syn-lens", label: "Syn lens" }, { id: "syn-gal", label: "Syn gal" },
];

/** Keep the rows of the chosen groups (none chosen = all) and state. */
export function filterEvalRows<T extends { grade?: string; state?: string | null; ok?: string }>(
  rows: readonly T[], groups: readonly string[], state: string, okOnly: boolean,
): T[] {
  const g = new Set(groups);
  return rows.filter((r) => (!okOnly || String(r.ok).toLowerCase() === "true")
    && (!g.size || g.has(String(r.grade ?? "")))
    && (!state || state === "all" || (r.state ?? "unknown") === state));
}
