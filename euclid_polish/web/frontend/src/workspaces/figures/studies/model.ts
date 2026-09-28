/* Figures › Studies — pure shaping (unit-tested in studies.test.tsx).
 *
 * The charts draw the backend's chart tables (`GET …/figure/<chart>.csv`),
 * the same rows the PDF / PNG / SVG exports draw, so an on-screen number is
 * always the exported number (render.py `table`). Nothing here recomputes a
 * statistic: the paired bootstrap, the group medians and bands and the
 * knee integration all come from the backend. This module only parses the
 * CSV, builds the chart / export URLs from the selection, and arranges rows
 * into plot series. */

/** One CSV row, by column name (all values as text; "" = empty). */
export type CsvRow = Record<string, string>;

/** RFC 4180 CSV → rows keyed by the header (quoted fields, "" escapes, CRLF). */
export function parseCsv(text: string): CsvRow[] {
  const records: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === "\"") {
        if (text[i + 1] === "\"") { field += "\""; i++; } else quoted = false;
      } else field += ch;
    } else if (ch === "\"") quoted = true;
    else if (ch === ",") { row.push(field); field = ""; }
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      records.push(row); row = [];
    } else field += ch;
  }
  if (field !== "" || row.length) { row.push(field); records.push(row); }
  const [header, ...body] = records.filter((r) => !(r.length === 1 && r[0] === ""));
  if (!header) return [];
  return body.map((r) => Object.fromEntries(header.map((h, j) => [h, r[j] ?? ""])));
}

/** A CSV cell as a number (empty / non-numeric → null). */
export function num(v: string | undefined | null): number | null {
  if (v == null || v.trim() === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

/* ── selection → URLs ─────────────────────────────────────────────────── */

export const CHARTS = ["knee", "integrated", "paired", "gate", "training", "real"] as const;
export type Chart = (typeof CHARTS)[number];

export const CHART_TITLE: Record<Chart, string> = {
  knee: "PSNR vs knee",
  integrated: "Integrated PSNR by loss × knee",
  paired: "Paired differences",
  gate: "Gate weights",
  training: "Training curves",
  real: "Real tiles",
};

/** The chart strip's short labels (the titles above name them in full). */
export const CHART_TAB: Record<Chart, string> = {
  knee: "Knee curves", integrated: "Integrated", paired: "Paired Δ", gate: "Gate weights", training: "Training", real: "Real tiles",
};

export type ChartSelection = {
  /** A member subset (labels); null or empty = every member. */
  members?: readonly string[] | null;
  /** Recipe field to group by (render.GROUP_FIELDS); null = per member. */
  group?: string | null;
  reference?: string | null;
  source?: string | null;
  metric?: string | null;
  experiment?: string | null;
};

/** The query every chart request carries: the selection plus the options of
 *  that chart only (so the export link of one chart names what it draws). */
export function chartQuery(chart: Chart, sel: ChartSelection): URLSearchParams {
  const q = new URLSearchParams();
  if (sel.members && sel.members.length) q.set("members", sel.members.join(","));
  if (sel.group) q.set("group", sel.group);
  if (chart === "paired" && sel.reference && sel.reference !== "mean") q.set("reference", sel.reference);
  if (chart === "gate" && sel.source && sel.source !== "all") q.set("source", sel.source);
  if (chart === "training" && sel.metric && sel.metric !== "psnr") q.set("metric", sel.metric);
  if (chart === "real" && sel.experiment) q.set("experiment", sel.experiment);
  return q;
}

const base = (id: string) => `/api/studies/${encodeURIComponent(id)}/figure`;
const withQuery = (path: string, q: URLSearchParams) => { const s = q.toString(); return s ? `${path}?${s}` : path; };

/** The chart's numbers as the page reads them (and as "CSV" downloads them). */
export function csvUrl(id: string, chart: Chart, sel: ChartSelection, download = false): string {
  const q = chartQuery(chart, sel);
  if (download) q.set("download", "1");
  return withQuery(`${base(id)}/${chart}.csv`, q);
}

export type FigureFormat = "pdf" | "png" | "svg";

/** The publication figure in the plate style, rendered by the backend. */
export function figureUrl(id: string, chart: Chart, format: FigureFormat, sel: ChartSelection, dpi = 300): string {
  const q = new URLSearchParams({ format, dpi: String(dpi) });
  for (const [k, v] of chartQuery(chart, sel)) q.set(k, v);
  q.set("download", "1");
  return `${base(id)}/${chart}?${q.toString()}`;
}

/* ── recipe keys (colour and the group reference options) ─────────────── */

export const MISSING = "—";

const g = (v: number) => {
  if (Number.isInteger(v)) return String(v);
  return String(Number(v.toPrecision(6)));
};

/** A recipe value as the backend names its group (stats.group_key). */
export function groupKey(value: unknown): string {
  if (value == null || value === "") return MISSING;
  if (Array.isArray(value)) {
    if (!value.length) return MISSING;
    return value.map((v) => (typeof v === "number" ? g(v) : String(v))).join("+");
  }
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "number") return g(value);
  return String(value);
}

type Recipe = { asinh_knee?: unknown; asinh_knees?: unknown; [k: string]: unknown };

/** The group a member falls in for a recipe field (training_knee = the
 *  knee list of a multi-knee member, else its one knee). */
export function memberKey(m: Recipe, field: string): string {
  if (field === "training_knee") {
    const many = Array.isArray(m.asinh_knees) && m.asinh_knees.length ? m.asinh_knees : null;
    return groupKey(many ?? m.asinh_knee);
  }
  return groupKey(m[field]);
}

/** Group names in first-seen order over `members`. */
export function groupNames(members: readonly Recipe[], field: string): string[] {
  return [...new Set(members.map((m) => memberKey(m, field)))];
}

export const GROUP_LABEL: Record<string, string> = {
  loss: "Loss", training_knee: "Training knee", asinh_knee: "Asinh knee", output_knee: "Output knee",
  knee_loss: "Knee loss", blocks: "Depth (blocks)", bootstrap: "Bootstrap", noise_aug: "Noise augmentation",
  icnr: "ICNR", status: "Status", op: "Origin (op)",
};

/** "170·psnr" → "170" (the member number the console names members by). */
export const memberNumber = (label: string) => label.split("·")[0];
export const memberName = (label: string) => `member ${memberNumber(label)}`;

/** A training-knee category as a tick: a multi-knee list reads "6 knees". */
export function kneeTick(key: string): string {
  if (key === MISSING) return "default";
  return key.includes("+") ? `${key.split("+").length} knees` : key;
}

/** Order categories numerically where they are numbers (the backend's order). */
export function sortKneeKeys(keys: Iterable<string>): string[] {
  return [...new Set(keys)].sort((a, b) => {
    const x = Number(a), y = Number(b);
    const fx = Number.isFinite(x), fy = Number.isFinite(y);
    if (fx && fy) return x - y;
    if (fx !== fy) return fx ? -1 : 1;
    return a.localeCompare(b);
  });
}

/* ── chart tables → series ────────────────────────────────────────────── */

export type Curve = { name: string; kind: string; n: number; x: number[]; y: (number | null)[]; lo: (number | null)[] | null; hi: (number | null)[] | null };

/** Knee table → per band, one curve per series (members / groups, then mean
 *  and gate), in the table's order. */
export function kneeCurves(rows: readonly CsvRow[]): { bands: string[]; curves: Record<string, Curve[]> } {
  const bands: string[] = [];
  const out: Record<string, Map<string, Curve>> = {};
  for (const r of rows) {
    const band = r.band;
    if (!out[band]) { out[band] = new Map(); bands.push(band); }
    let c = out[band].get(r.series);
    if (!c) {
      c = { name: r.series, kind: r.kind, n: num(r.n_members) ?? 1, x: [], y: [], lo: r.lo === "" ? null : [], hi: r.hi === "" ? null : [] };
      out[band].set(r.series, c);
    }
    c.x.push(num(r.knee_e) ?? NaN);
    c.y.push(num(r.psnr));
    c.lo?.push(num(r.lo));
    c.hi?.push(num(r.hi));
  }
  return { bands, curves: Object.fromEntries(bands.map((b) => [b, [...out[b].values()]])) };
}

/** `curve` minus `ref` knee by knee (the "vs mean" view); null where either is. */
export function minus(y: readonly (number | null)[], ref: readonly (number | null)[] | undefined): (number | null)[] {
  if (!ref) return [...y];
  return y.map((v, i) => (v == null || ref[i] == null ? null : v - (ref[i] as number)));
}

export type PairedRow = { target: string; n: number; reference: string; band: string; mean: number | null; lo: number | null; hi: number | null; nFields: number | null; resamples: number | null; seed: number | null };

export function pairedRows(rows: readonly CsvRow[]): PairedRow[] {
  return rows.map((r) => ({
    target: r.target, n: num(r.n_members) ?? 1, reference: r.reference, band: r.band,
    mean: num(r.mean_delta), lo: num(r.lo), hi: num(r.hi), nFields: num(r.n_fields), resamples: num(r.n_resamples), seed: num(r.seed),
  }));
}

/** Whether a 95 % interval excludes zero ("better" / "worse" than the reference). */
export function intervalVerdict(p: Pick<PairedRow, "lo" | "hi">): "better" | "worse" | "unresolved" {
  if (p.lo == null || p.hi == null) return "unresolved";
  if (p.lo > 0) return "better";
  if (p.hi < 0) return "worse";
  return "unresolved";
}

/** "+0.12 dB [+0.05, +0.19]". */
export function deltaText(p: Pick<PairedRow, "mean" | "lo" | "hi">): string {
  // Round first, so a −0.004 reads "0.00", never "−0.00".
  const s = (v: number | null) => {
    if (v == null) return "—";
    const r = Math.round(v * 100) / 100;
    return r === 0 ? "0.00" : `${r > 0 ? "+" : "−"}${Math.abs(r).toFixed(2)}`;
  };
  return `${s(p.mean)} [${s(p.lo)}, ${s(p.hi)}]`;
}

/** Indices of categorical ticks to label so at most `limit` labels show. */
export function sparseTicks(n: number, limit = 16): number[] {
  const step = Math.max(1, Math.ceil(n / limit));
  const out: number[] = [];
  for (let i = 0; i < n; i += step) out.push(i);
  return out;
}

/** "3 of 37 members", "all 37 members". */
export function selectionText(selected: number, total: number): string {
  return selected && selected < total ? `${selected} of ${total} members` : `all ${total} members`;
}
