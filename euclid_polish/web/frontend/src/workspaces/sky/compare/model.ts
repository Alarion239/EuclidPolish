/* Sky › Compare, pure logic (unit-tested in model.test.ts): the pivot of one
 * metric (models × VIS/Y/J/H), the Δm of each model against the LR, the
 * history's headline result, the long-form CSV and the New-comparison
 * target sets. No React, no fetch. */
import { formatCount, formatDate } from "../../../format";
import { BANDS, type ExperimentRecord, type ExperimentSummary, type Metrics, type TileRow } from "../results/api";
import {
  bandLabel, deltaMagWarn, fluxDeltaMag, formatMetric, METRIC_BY_KEY, METRICS, metricRows, metricsFor, num, recordSpecs,
  sortSpecs, specWords, worstHoles, type MetricKey,
} from "../results/model";

export type PivotRow = { spec: string; label: string } & Record<(typeof BANDS)[number], number | null>;

/** One metric as models × bands (models in catalogue order: production — the
 *  reference — first, then the mean, gate variants, members). */
export function pivotRows(record: ExperimentRecord, scope: string, metric: MetricKey): PivotRow[] {
  const out: PivotRow[] = [];
  for (const spec of recordSpecs(record)) {
    const perBand = metricsFor(record, scope, spec)?.per_band;
    if (!perBand) continue;
    const row = { spec, label: record.model_labels?.[spec] ?? spec } as PivotRow;
    for (const b of BANDS) row[b] = num(perBand[b]?.[metric]);
    out.push(row);
  }
  return out;
}

/** The best model per band for the metric's direction (none for counts). */
export function pivotBest(rows: readonly PivotRow[], metric: MetricKey): Partial<Record<(typeof BANDS)[number], string>> {
  const better = METRIC_BY_KEY[metric]?.better;
  const out: Partial<Record<(typeof BANDS)[number], string>> = {};
  if (!better) return out;
  const score = (v: number) => (better === "lower" ? v : better === "higher" ? -v : Math.abs(v - 1));
  for (const b of BANDS) {
    let best: { spec: string; s: number } | null = null;
    for (const r of rows) {
      const v = r[b];
      if (v == null) continue;
      const s = score(v);
      if (!best || s < best.s) best = { spec: r.spec, s };
    }
    if (best) out[b] = best.spec;
  }
  return out;
}

export type DeltaMag = { spec: string; label: string; ratio: number; dm: number | null; warn: boolean };

/** Each model's total flux against the LR in one band (pooled or one tile):
 *  Δm = −2.5 log10(SR/LR), warned beyond 0.1 mag. Models without the band are left out. */
export function deltaMags(record: ExperimentRecord, scope: string, band: string): DeltaMag[] {
  const out: DeltaMag[] = [];
  for (const spec of recordSpecs(record)) {
    const ratio = num(metricsFor(record, scope, spec)?.per_band?.[band]?.flux_ratio);
    if (ratio == null) continue;
    const dm = fluxDeltaMag(ratio);
    out.push({ spec, label: record.model_labels?.[spec] ?? spec, ratio, dm, warn: deltaMagWarn(dm) });
  }
  return out;
}

export type Headline = { spec: string; label: string; holes: number; band: string; best?: boolean };

const worstOf = (spec: string, m: Metrics | null | undefined): Headline | null => {
  const w = worstHoles(m);
  return w ? { spec, label: specWords(spec), holes: w.holes, band: bandLabel(w.band || "VIS") } : null;
};

/** The history's headline result: the worst-band hole % of production and
 *  of the plain mean (those that ran); when neither ran, the model with the
 *  fewest holes (`best`). */
export function headline(record: Pick<ExperimentSummary, "summary">): Headline[] {
  const summary = record.summary ?? {};
  const out = ["production", "mean"].map((spec) => worstOf(spec, summary[spec])).filter((h): h is Headline => !!h);
  if (out.length) return out;
  let best: Headline | null = null;
  for (const spec of sortSpecs(Object.keys(summary))) {
    const h = worstOf(spec, summary[spec]);
    if (h && (!best || h.holes < best.holes)) best = h;
  }
  return best ? [{ ...best, best: true }] : [];
}

/** The headline as one cell: "production 20.2 % (J) · member mean 32.6 % (VIS)". */
export function headlineText(record: Pick<ExperimentSummary, "summary">): string {
  return headline(record).map((h) => `${h.label} ${formatMetric("hole_pct", h.holes)} % (${h.band})${h.best ? ", the fewest" : ""}`).join(" · ");
}

export type SentencePiece = { text: string; num?: boolean; warn?: boolean };

/** The page's one sentence for the scope (pooled over the tiles, or one
 *  tile): production's worst-band holes against the plain mean's, else
 *  against the best other model; without production, the model with the
 *  fewest holes. Null when nothing is scored in the scope. */
export function compareSentence(record: ExperimentRecord, scope: string): SentencePiece[] | null {
  const scored = recordSpecs(record)
    .map((spec) => worstOf(spec, metricsFor(record, scope, spec)))
    .filter((h): h is Headline => !!h);
  if (!scored.length) return null;
  const prod = scored.find((h) => h.spec === "production");
  const subject = prod ?? scored.reduce((a, b) => (b.holes < a.holes ? b : a));
  const others = scored.filter((h) => h.spec !== subject.spec);
  const mean = prod ? others.find((h) => h.spec === "mean") : undefined;
  const against = prod ? (mean ?? (others.length ? others.reduce((a, b) => (b.holes < a.holes ? b : a)) : undefined)) : undefined;
  const pct = (v: number) => `${formatMetric("hole_pct", v)} %`;
  const t = (text: string): SentencePiece => ({ text });
  const out: SentencePiece[] = [];
  const nTiles = record.tiles?.length ?? 0;
  if (scope === "pooled") out.push(t("On "), { text: formatCount(nTiles), num: true }, t(` tile${nTiles === 1 ? "" : "s"}, `));
  else out.push(t(`On ${scope}, `));
  out.push(t(`${subject.label} leaves holes in `), { text: pct(subject.holes), num: true, warn: !!against && subject.holes > against.holes });
  out.push(t(` of the bright pixels of its worst band (${subject.band})`));
  if (against) {
    out.push(t(", against "), { text: pct(against.holes), num: true }, t(` for ${against.spec === "mean" ? "the " : ""}${against.label} (${against.band})`));
    if (!mean) out.push(t(`, the best of ${others.length} other model${others.length === 1 ? "" : "s"}`));
  }
  out.push(t("."));
  return out;
}

/** "seed vs pruning · 2026-09-27" (the id when unlabelled). */
export function comparisonLabel(e: Pick<ExperimentSummary, "id" | "label" | "created">): string {
  if (!e.label) return e.id;
  return e.created ? `${e.label} · ${formatDate(e.created, { utc: true })}` : e.label;
}

/* ── the long-form CSV (every metric of every model × band) ────────────── */

/** RFC 4180 cells; a text cell that a spreadsheet would run as a formula
 *  (= + - @) is prefixed with ' (the kit DataTable's rule). */
export function csvText(rows: readonly (readonly (string | number | null | undefined)[])[]): string {
  const cell = (v: string | number | null | undefined): string => {
    if (v == null) return "";
    if (typeof v === "number") return Number.isFinite(v) ? String(v) : "";
    let s = v;
    if (/^[=+\-@]/.test(s)) s = `'${s}`;
    return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return rows.map((r) => r.map(cell).join(",")).join("\n");
}

export function longCsv(record: ExperimentRecord, scope: string): string {
  const head = ["model", "label", "band", ...METRICS.map((m) => m.label)];
  const body = metricRows(record, scope).map((r) => [
    r.spec, r.label, r.band, ...METRICS.map((m) => num(r[m.key])),
  ]);
  return csvText([head, ...body]);
}

/* ── New comparison: target sets ───────────────────────────────────────── */

export type CompareSetId = "poster" | "nexus" | "lenses" | "galaxies";

/** The poster file to run on: every poster file holds the same galaxy LR, so
 *  the one with the most model runs (the current poster runs) stands for it. */
export function posterRef(posters: readonly TileRow[]): string | null {
  let best: TileRow | null = null;
  for (const p of posters) if (!best || Object.keys(p.models ?? {}).length > Object.keys(best.models ?? {}).length) best = p;
  return best?.ref ?? null;
}

/** A set's tile refs: the poster galaxy, the NEXUS tiles of the shared tile
 *  selection, every lens candidate or Q1 galaxy of the evaluation store. */
export function setRefs(set: CompareSetId, from: {
  selection: readonly string[]; evals: readonly TileRow[]; posters: readonly TileRow[];
}): string[] {
  if (set === "poster") { const ref = posterRef(from.posters); return ref ? [ref] : []; }
  if (set === "nexus") return from.selection.filter((r) => r.startsWith("nexus/"));
  const kind = set === "lenses" ? "lens" : "galaxy";
  return from.evals.filter((t) => t.extras?.kind === kind).map((t) => t.ref);
}
