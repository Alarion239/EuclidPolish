/* Pure shaping behind Models › Spectra (unit-tested in spectra.test.ts). No
   React, no DOM:
   - normalise the server's payloads (the summary and one model's record):
     any number may be null and a row may carry no spectra at all;
   - the toolbar's choices (section, band, period band) and the model
     picker: the models with spectra, the default pick (the best ∫PSNR, the
     newest trained without the isotropy rule, then the newest), the URL's
     pick capped at eight, each shown model's colour slot (its place in the
     pick); the server notes the page does not already say;
   - the curves by period (ascending, a null kept as a gap) and by angle,
     their axes, and where the coherence falls to ½;
   - the axis / diagonal rows with the grid verdict (≥ 1.2), the 2-D error
     map as a heat layer (log₂ of the ratio, in the plot's x-major cells)
     with its period rings and a clicked cell's readout;
   - the compute confirm's estimate. */
import { formatNumber } from "../../format";
import type { Tick } from "../../ticks";
import { colormapLut } from "../../viewer/colormaps";
import { BANDS, BAND_SHORT, type Band } from "./api";
import { memberNumber } from "./model";
import type { AxisDiag, BandRows, SpectraMap, SpectraModel, SpectraPayload, SpectraRadial, SpectraRecord } from "./spectraApi";
import { stepsShort } from "./universal";

/* ── toolbar choices (URL state) ───────────────────────────────────────── */

export type Section = "power" | "coherence" | "direction";
export const SECTIONS: readonly { value: Section; label: string; title: string }[] = [
  { value: "power", label: "Power", title: "Radial power and transfer by period, against the truth" },
  { value: "coherence", label: "Coherence", title: "How closely each output follows the truth at each period" },
  { value: "direction", label: "Direction", title: "Whether the error prefers the pixel axes (a grid) or is round" },
];
export const parseSection = (raw: string): Section | undefined =>
  (SECTIONS.some((s) => s.value === raw) ? raw as Section : undefined);
export const parseBand = (raw: string): Band | undefined => ((BANDS as readonly string[]).includes(raw) ? raw as Band : undefined);
export const bandText = (band: string) => BAND_SHORT[band] ?? band;

/** The direction diagnostics' period bands (px) when the payload names none. */
export const DEFAULT_PERIOD_BANDS: readonly [number, number][] = [[2, 3], [3, 6], [6, 12], [12, 21], [21, 42]];
/** The angular profiles' default band: it holds the LR pixel's Nyquist period (4 px). */
export const DEFAULT_PB = "3-6";
export const pbKey = ([lo, hi]: readonly [number, number]) => `${lo}-${hi}`;
/** "3-6" → "3–6 px". */
export const pbLabel = (key: string) => `${key.replace("-", "–")} px`;

/** The models shown at once, and picked by default. */
export const MAX_SHOWN = 8;
export const DEFAULT_PICK = 5;
/** Axis / diagonal at or above this is a pixel-axis grid. */
export const GRID_RATIO = 1.2;
/** The periods (HR px) of the HR Nyquist and of the LR pixel's Nyquist. */
export const HR_NYQUIST_PX = 2;
export const LR_NYQUIST_PX = 4;
/** The rings drawn on the error map when the record names none (px). */
export const DEFAULT_PERIOD_LINES = [21, 3];
/** The compute confirm's cost model: the spectra of one model on the test fields. */
export const SECONDS_PER_MODEL = 25;

/* ── normalisation ─────────────────────────────────────────────────────── */

type Obj = Record<string, unknown>;
const isObj = (v: unknown): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);
const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const numOrNull = (v: unknown): number | null => (finite(v) ? v : null);
const strOrNull = (v: unknown): string | null => (typeof v === "string" && v ? v : null);
const strs = (v: unknown): string[] => (Array.isArray(v) ? v.filter((x) => x != null).map((x) => String(x)) : []);
const numbers = (v: unknown): number[] => (Array.isArray(v) ? v.filter(finite) : []);
/** A value list: a non-finite entry reads null (a gap, never a 0). */
const values = (v: unknown): (number | null)[] => (Array.isArray(v) ? v.map(numOrNull) : []);
const bandRows = (v: unknown): BandRows | null => (Array.isArray(v) ? v.map(values) : null);
const pair = (v: unknown): [number | null, number | null] | null => (Array.isArray(v) ? [numOrNull(v[0]), numOrNull(v[1])] : null);

function radial(v: unknown): SpectraRadial | null {
  if (!isObj(v)) return null;
  return { power_sr: bandRows(v.power_sr) ?? [], coherence: bandRows(v.coherence) ?? [], transfer: bandRows(v.transfer) ?? [] };
}

/** A per-period-band object (`axis_diag`, `angular`): each key's value read by `read`. */
function byPeriodBand<T>(v: unknown, read: (x: unknown) => T): Record<string, T> | null {
  if (!isObj(v)) return null;
  const out: Record<string, T> = {};
  for (const [k, x] of Object.entries(v)) out[k] = read(x);
  return out;
}

const axisDiag = (v: unknown): AxisDiag => (Array.isArray(v) ? v.map(pair) : []);

function normaliseModel(raw: Obj): SpectraModel {
  const name = String(raw.name);
  const num = memberNumber(name);
  return {
    name, number: numOrNull(raw.number) ?? (num == null ? null : Number(num)), label: strOrNull(raw.label),
    status: typeof raw.status === "string" ? raw.status : "missing",
    computed_at: strOrNull(raw.computed_at), n_fields: numOrNull(raw.n_fields), mean_psnr: numOrNull(raw.mean_psnr),
    steps: numOrNull(raw.steps), served_step: numOrNull(raw.served_step), depth: numOrNull(raw.depth),
    knees: Array.isArray(raw.knees) ? numbers(raw.knees) : null, isotropy_penalty: numOrNull(raw.isotropy_penalty),
    reason: strOrNull(raw.reason),
    radial: radial(raw.radial),
    axis_diag: byPeriodBand(raw.axis_diag, axisDiag),
    angular: byPeriodBand(raw.angular, (x) => bandRows(x) ?? []),
  };
}

function periodBands(v: unknown): [number, number][] {
  const out = (Array.isArray(v) ? v : []).flatMap((b): [number, number][] =>
    (Array.isArray(b) && finite(b[0]) && finite(b[1]) && b[0] < b[1] ? [[b[0], b[1]]] : []));
  return out.length ? out : DEFAULT_PERIOD_BANDS.map(([lo, hi]): [number, number] => [lo, hi]);
}

/** The measurements' windows and knees when the payload does not say. */
export const DEFAULT_RADIAL = { window_px: 128, stretch_knee_e: 1 };
export const DEFAULT_DIRECTION = { window_px: 64, long_window_px: 128, stretch_knee_e: 0.01 };
const positiveOr = (v: unknown, d: number) => (finite(v) && v > 0 ? v : d);

/** The summary payload with every collection present; null when it is not
 *  an object. `records_local` reads true unless the server says false. */
export function normaliseSpectra(raw: unknown): SpectraPayload | null {
  if (!isObj(raw)) return null;
  const bands = strs(raw.bands);
  const rad = isObj(raw.radial) ? raw.radial : {};
  const dir = isObj(raw.direction) ? raw.direction : {};
  return {
    schema: numOrNull(raw.schema),
    records_local: raw.records_local !== false,
    bands: bands.length ? bands : [...BANDS],
    period_bands: periodBands(raw.period_bands),
    angle_deg: numbers(raw.angle_deg),
    period_px: values(raw.period_px).map((v) => v ?? NaN),
    truth_power: bandRows(raw.truth_power),
    pixel_scale_arcsec: finite(raw.pixel_scale_arcsec) && raw.pixel_scale_arcsec > 0 ? raw.pixel_scale_arcsec : null,
    radial: {
      window_px: positiveOr(rad.window_px, DEFAULT_RADIAL.window_px),
      stretch_knee_e: positiveOr(rad.stretch_knee_e, DEFAULT_RADIAL.stretch_knee_e),
    },
    direction: {
      window_px: positiveOr(dir.window_px, DEFAULT_DIRECTION.window_px),
      long_window_px: positiveOr(dir.long_window_px, DEFAULT_DIRECTION.long_window_px),
      stretch_knee_e: positiveOr(dir.stretch_knee_e, DEFAULT_DIRECTION.stretch_knee_e),
    },
    models: (Array.isArray(raw.models) ? raw.models : []).filter(isObj).filter((m) => m.name != null).map(normaliseModel),
    missing: strs(raw.missing),
    notes: strs(raw.notes),
  };
}

function normaliseMap(v: unknown): SpectraMap | null {
  if (!isObj(v)) return null;
  const lines = numbers(v.period_lines_px).filter((p) => p > 0);
  return {
    k: numbers(v.k),
    values: Array.isArray(v.values) ? v.values.map((band) => (Array.isArray(band) ? band.map(values) : [])) : [],
    period_lines_px: Array.isArray(v.period_lines_px) ? lines : [...DEFAULT_PERIOD_LINES],
    n_windows: numOrNull(v.n_windows),
  };
}

/** One model's record (its row plus the map); null when it is not a record. */
export function normaliseRecord(raw: unknown): SpectraRecord | null {
  if (!isObj(raw) || raw.name == null) return null;
  const bands = strs(raw.bands);
  return { ...normaliseModel(raw), bands: bands.length ? bands : [...BANDS], map: normaliseMap(raw.map) };
}

/* ── models ────────────────────────────────────────────────────────────── */

const anyValue = (rows: BandRows | null | undefined) => !!rows?.some((r) => r.some((v) => v != null));
export const hasRadial = (m: SpectraModel) =>
  !!m.radial && (anyValue(m.radial.power_sr) || anyValue(m.radial.coherence) || anyValue(m.radial.transfer));
export const hasAngular = (m: SpectraModel) => !!m.angular && Object.values(m.angular).some(anyValue);
export const hasAxisDiag = (m: SpectraModel) =>
  !!m.axis_diag && Object.values(m.axis_diag).some((rows) => rows.some((p) => !!p && (p[0] != null || p[1] != null)));
export const hasDirection = (m: SpectraModel) => hasAngular(m) || hasAxisDiag(m);
export const hasSpectra = (m: SpectraModel) => hasRadial(m) || hasDirection(m);

/** "#238", else the name. */
export const modelLabel = (m: Pick<SpectraModel, "name" | "number">) => (m.number != null ? `#${m.number}` : m.name);
const byNumber = (a: SpectraModel, b: SpectraModel) =>
  (a.number ?? Infinity) - (b.number ?? Infinity) || a.name.localeCompare(b.name);
const trainedWithoutRule = (m: SpectraModel) => !(m.isotropy_penalty != null && m.isotropy_penalty > 0);

/** The models with any spectra (radial or direction), by number: the picker's list. */
export function pickable(models: readonly SpectraModel[]): SpectraModel[] {
  return models.filter(hasSpectra).sort(byNumber);
}

/** The default pick (at most `n`, by number): among the models with current
 *  spectra (any with spectra when none is current), the best ∫PSNR, the
 *  newest trained without the isotropy rule (the reference a grid is judged
 *  against), then the newest. */
export function defaultPick(models: readonly SpectraModel[], n = DEFAULT_PICK): string[] {
  const list = pickable(models);
  const current = list.filter((m) => m.status === "current");
  const pool = current.length ? current : list;
  const newest = [...pool].sort((a, b) => byNumber(b, a));
  const best = pool.reduce<SpectraModel | null>((b, m) => (m.mean_psnr != null && (b?.mean_psnr == null || m.mean_psnr > b.mean_psnr) ? m : b), null);
  const out: SpectraModel[] = [];
  const add = (m: SpectraModel | null | undefined) => { if (m && out.length < n && !out.includes(m)) out.push(m); };
  add(best);
  add(newest.find(trainedWithoutRule));
  newest.forEach(add);
  return out.sort(byNumber).map((m) => m.name);
}

/** The models shown, in pick order: the URL's pick (those with spectra, the
 *  first `max`), else the default pick (by number). */
export function shownModels(models: readonly SpectraModel[], picked: readonly string[], max = MAX_SHOWN): string[] {
  const names = new Set(pickable(models).map((m) => m.name));
  const mine = [...new Set(picked)].filter((p) => names.has(p)).slice(0, max);
  return mine.length ? mine : defaultPick(models);
}

/** The shown models by number (the order they are drawn and listed in). */
export function byNumberOf(models: readonly SpectraModel[], names: readonly string[]): SpectraModel[] {
  return pickable(models).filter((m) => names.includes(m.name));
}

/** A new pick (the picker's order: the last picked last) capped at `max`,
 *  the earliest dropping first; an empty pick means the default one. */
export function limitPick(next: readonly string[], max = MAX_SHOWN): string[] {
  return next.length > max ? next.slice(next.length - max) : [...next];
}

/** Each shown model's slot on the categorical palette: its place in the pick
 *  (eight slots for at most eight models, so no two share a colour). A
 *  model keeps its colour in every figure and when another is picked; one
 *  colour per model over all of them would repeat, the palette having eight. */
export function colourSlots(shown: readonly string[]): Map<string, number> {
  return new Map(shown.map((n, i) => [n, i % 8]));
}

export const isotropyText = (w: number | null | undefined) =>
  (w != null && w > 0 ? `isotropy ${formatNumber(w, { sig: 3 })}` : "no isotropy rule");

/** "depth 16 · 1.9M steps · ∫PSNR 62.13 dB · isotropy 0.5" (the steps the
 *  spectra were computed at, else the trained ones). */
function modelFacts(m: SpectraModel): string[] {
  const steps = m.served_step ?? m.steps;
  return [
    m.depth != null ? `depth ${m.depth}` : null, steps != null ? `${stepsShort(steps)} steps` : null,
    m.mean_psnr != null ? `∫PSNR ${m.mean_psnr.toFixed(2)} dB` : null, isotropyText(m.isotropy_penalty),
  ].filter((s): s is string => !!s);
}

/** "#238 · depth 16 · … · isotropy 0.5", and why a stale one is stale. */
export function modelText(m: SpectraModel): string {
  const stale = m.status === "stale" ? ["stale spectra (an earlier checkpoint or other test records)"] : [];
  return [modelLabel(m), ...modelFacts(m), ...stale].join(" · ");
}

/** The models in `models` a figure cannot draw (`has` false): "#239 (why)". */
export function notDrawn(models: readonly SpectraModel[], has: (m: SpectraModel) => boolean): string[] {
  return models.filter((m) => !has(m)).map((m) => (m.reason ? `${modelLabel(m)} (${m.reason})` : modelLabel(m)));
}

/** The models without any spectra that say why, grouped by the reason. */
export function reasonGroups(models: readonly SpectraModel[]): { reason: string; labels: string[] }[] {
  const by = new Map<string, string[]>();
  for (const m of [...models].sort(byNumber)) {
    if (!m.reason || hasSpectra(m)) continue;
    by.set(m.reason, [...(by.get(m.reason) ?? []), modelLabel(m)]);
  }
  return [...by].map(([reason, labels]) => ({ reason, labels }));
}

/** The server's notes, less one that only lists every model without
 *  spectra (the page's own callout names them). */
export function freshNotes(notes: readonly string[], missing: readonly string[]): string[] {
  return notes.filter((n) => !(missing.length > 0 && missing.every((m) => n.includes(m))));
}

/** The picker's hint: the facts, and "stale". */
export function modelHint(m: SpectraModel): string {
  return [...modelFacts(m), ...(m.status === "stale" ? ["stale"] : [])].join(" · ");
}

/** How many test fields the spectra were measured on (the largest count), or null. */
export function fieldsOf(p: Pick<SpectraPayload, "models"> | null | undefined): number | null {
  const ns = (p?.models ?? []).map((m) => m.n_fields).filter((n): n is number => n != null && n > 0);
  return ns.length ? Math.max(...ns) : null;
}

/* ── curves ────────────────────────────────────────────────────────────── */

export type Curve = { x: number[]; y: (number | null)[] };
export type RadialKind = keyof SpectraRadial;

/** A band's row in a payload: its index in the payload's bands (the default order else). */
export function bandIndex(bands: readonly string[], band: string): number {
  const i = bands.indexOf(band);
  return i >= 0 ? i : Math.max(0, (BANDS as readonly string[]).indexOf(band));
}

/** The rings' periods: the payload's, else 128 / ring (ring 1 … n). */
export function ringPeriods(p: Pick<SpectraPayload, "period_px">, n: number): number[] {
  return p.period_px.length ? p.period_px : Array.from({ length: n }, (_, i) => 128 / (i + 1));
}

/** A radial row against the rings' periods, ascending in period (the plot's
 *  x order): a ring without a finite period is dropped, a missing value is a
 *  null (a gap in the line, never a 0). */
export function periodCurve(p: Pick<SpectraPayload, "period_px">, row: readonly (number | null)[] | null | undefined): Curve {
  const vals = row ?? [];
  const pts: [number, number | null][] = [];
  ringPeriods(p, vals.length).forEach((x, i) => { if (Number.isFinite(x) && x > 0) pts.push([x, vals[i] ?? null]); });
  pts.sort((a, b) => a[0] - b[0]);
  return { x: pts.map((q) => q[0]), y: pts.map((q) => q[1]) };
}

export const hasValues = (c: Curve) => c.y.some((v) => v != null);

export const radialRow = (p: Pick<SpectraPayload, "bands">, m: SpectraModel, kind: RadialKind, band: string) =>
  m.radial?.[kind]?.[bandIndex(p.bands, band)] ?? null;
export const truthRow = (p: Pick<SpectraPayload, "bands" | "truth_power">, band: string) =>
  p.truth_power?.[bandIndex(p.bands, band)] ?? null;

/** A row of the angular profile at period band `key` against the angle bins
 *  (the payload's centres, else n bins over 0–180°). */
export function angularCurve(p: Pick<SpectraPayload, "bands" | "angle_deg">, m: SpectraModel, key: string, band: string): Curve {
  const row = m.angular?.[key]?.[bandIndex(p.bands, band)] ?? [];
  const x = p.angle_deg.length === row.length ? p.angle_deg : row.map((_, i) => ((i + 0.5) * 180) / row.length);
  return { x: [...x], y: [...row] };
}

/** The period axis (px, log): fine scales left, coarse right. */
export const PERIOD_DOMAIN: [number, number] = [1.85, 140];
export const PERIOD_TICKS: Tick[] = [2, 4, 8, 16, 32, 64, 128].map((v) => ({ v, label: String(v) }));
/** "4 px", or "4 px (0.2″)" with the HR pixel's size. */
export const periodText = (v: number, arcsecPerPx?: number | null) =>
  `${+v.toPrecision(3)} px${arcsecPerPx ? ` (${+(v * arcsecPerPx).toPrecision(2)}″)` : ""}`;
export const ANGLE_DOMAIN: [number, number] = [0, 180];
export const ANGLE_TICKS: Tick[] = [0, 45, 90, 135, 180].map((v) => ({ v, label: `${v}°` }));

/** A log domain symmetric about 1 for a ratio (½ and 2 equally far): the
 *  largest factor away from 1, padded, at least `min` and at most `max`. */
export function ratioDomain(vals: Iterable<number | null | undefined>, min = 2, max = 100): [number, number] {
  let f = 1;
  for (const v of vals) if (v != null && Number.isFinite(v) && v > 0) f = Math.max(f, v, 1 / v);
  const r = Math.min(max, Math.max(min, f * 1.15));
  return [1 / r, r];
}

/** The coherence axis: 0 to 1 with a little headroom, down to the lowest
 *  value when it goes negative (never below −1). */
export function coherenceDomain(vals: Iterable<number | null | undefined>): [number, number] {
  let lo = 0;
  for (const v of vals) if (v != null && Number.isFinite(v)) lo = Math.min(lo, v);
  return [lo < 0 ? Math.max(-1.02, Math.round(Math.floor(lo * 10) * 10 - 2) / 100) : 0, 1.02];
}

/** Quarter ticks over a coherence domain. */
export function coherenceTicks([lo, hi]: readonly [number, number]): Tick[] {
  const out: Tick[] = [];
  for (let v = Math.ceil(lo * 4) / 4; v <= hi + 1e-9; v += 0.25) {
    const r = Math.round(v * 100) / 100 || 0;
    out.push({ v: r, label: String(r) });
  }
  return out;
}

/** Where the coherence falls to `level`: walking from the coarsest ring to
 *  the finest, the first one under it, interpolated in log period with the
 *  ring before; `holds` when it never falls under (the period is then the
 *  finest ring's). Null when the coarsest ring is already under or no ring
 *  has a value. */
export function coherencePeriod(c: Curve, level = 0.5): { period: number; holds: boolean } | null {
  const pts = c.x.flatMap((x, i) => { const y = c.y[i]; return y != null && Number.isFinite(y) ? [[x, y] as const] : []; });
  if (!pts.length) return null;
  for (let i = pts.length - 1; i >= 0; i--) {
    const [x, y] = pts[i];
    if (y >= level) continue;
    if (i === pts.length - 1) return null;
    const [px, py] = pts[i + 1];
    const t = (py - level) / (py - y);
    return { period: 10 ** (Math.log10(px) + t * (Math.log10(x) - Math.log10(px))), holds: false };
  }
  return { period: pts[0][0], holds: true };
}

/* ── direction ─────────────────────────────────────────────────────────── */

/** The period bands in order: the payload's, then any other key a model
 *  carries (by its shorter period). */
export function periodBandKeys(p: Pick<SpectraPayload, "period_bands" | "models">): string[] {
  const keys = p.period_bands.map(pbKey);
  const extra = new Set<string>();
  for (const m of p.models) for (const k of [...Object.keys(m.axis_diag ?? {}), ...Object.keys(m.angular ?? {})]) if (!keys.includes(k)) extra.add(k);
  return [...keys, ...[...extra].sort((a, b) => parseFloat(a) - parseFloat(b))];
}

/** The period bands with an angular profile in some model (the default four
 *  when none has one: 21–42 px has no profile). */
export function angularKeys(p: Pick<SpectraPayload, "period_bands" | "models">): string[] {
  const keys = periodBandKeys(p).filter((k) => p.models.some((m) => anyValue(m.angular?.[k])));
  return keys.length ? keys : DEFAULT_PERIOD_BANDS.slice(0, 4).map(pbKey);
}

/** The period band drawn: the URL's when offered, else the default, else the first. */
export function effectivePb(picked: string, keys: readonly string[]): string {
  if (keys.includes(picked)) return picked;
  return keys.includes(DEFAULT_PB) ? DEFAULT_PB : keys[0] ?? DEFAULT_PB;
}

export type AxisDiagCell = { axis: number | null; diag: number | null; ratio: number | null };

/** An [axis, diagonal] pair with its ratio (null when either is missing or
 *  the diagonal is not positive); null when the pair has no value. */
export function axisDiagCell(p: [number | null, number | null] | null | undefined): AxisDiagCell | null {
  if (!p || (p[0] == null && p[1] == null)) return null;
  const [axis, diag] = p;
  return { axis, diag, ratio: axis != null && diag != null && diag > 0 ? axis / diag : null };
}

export const isGrid = (ratio: number | null | undefined) => ratio != null && ratio >= GRID_RATIO;

export type DirectionRow = {
  name: string; num: number | null; label: string; status: string;
  steps: number | null; served_step: number | null; mean_psnr: number | null; isotropy: number | null; depth: number | null;
  /** Per period band key, in `band`. */
  cells: Record<string, AxisDiagCell | null>;
};

/** Every model with axis / diagonal values, newest first, in `band`. */
export function directionRows(p: Pick<SpectraPayload, "bands" | "period_bands" | "models">, band: string): DirectionRow[] {
  const keys = periodBandKeys(p);
  const b = bandIndex(p.bands, band);
  return p.models.filter(hasAxisDiag).sort((a, z) => byNumber(z, a)).map((m) => {
    const cells: Record<string, AxisDiagCell | null> = {};
    for (const k of keys) cells[k] = axisDiagCell(m.axis_diag?.[k]?.[b]);
    return {
      name: m.name, num: m.number, label: modelLabel(m), status: m.status,
      steps: m.steps, served_step: m.served_step, mean_psnr: m.mean_psnr, isotropy: m.isotropy_penalty, depth: m.depth, cells,
    };
  });
}

/** The grid verdict at one period band: the rows measured there, those at
 *  a grid (axis / diagonal ≥ 1.2) and the round ones, newest first. One band
 *  at a time: the finest (2–3 px) can exceed it for every model. */
export function gridVerdict(rows: readonly DirectionRow[], key: string): { measured: DirectionRow[]; grid: DirectionRow[]; round: DirectionRow[] } {
  const measured = rows.filter((r) => r.cells[key]?.ratio != null);
  const grid = measured.filter((r) => isGrid(r.cells[key]?.ratio));
  return { measured, grid, round: measured.filter((r) => !grid.includes(r)) };
}

/* ── the 2-D error map ─────────────────────────────────────────────────── */

/** The map in one band as a heat layer: x-major cells (z[i][j] at kx[i],
 *  ky[j] — the transpose of the server's rows), each log₂ of its ratio
 *  (null or ≤ 0 → NaN: left blank), unit cells centred on the k values;
 *  `size`: the window's width in px (period = size / |k|); `extent`: the
 *  half-width of the square domain around the k grid. */
export type MapLayer = {
  z: number[][]; xEdges: number[]; yEdges: number[]; kx: number[]; ky: number[]; size: number; extent: number;
  periodLines: number[]; nWindows: number | null;
};

const centred = (n: number) => Array.from({ length: n }, (_, i) => i - Math.floor(n / 2));
const ascending = (k: readonly number[]) => k.every((v, i) => i === 0 || v > k[i - 1]);
const edgesOf = (k: readonly number[]) => [...k.map((v) => v - 0.5), k[k.length - 1] + 0.5];
const log2Ratio = (v: number | null | undefined) => (v != null && v > 0 ? Math.log2(v) : NaN);

export function mapLayer(rec: SpectraRecord | null | undefined, band: string): MapLayer | null {
  const m = rec?.map;
  const grid = m?.values[bandIndex(rec?.bands ?? [], band)];
  if (!m || !grid?.length) return null;
  const ny = grid.length;
  const nx = Math.max(...grid.map((r) => r.length));
  if (!nx) return null;
  const axisK = (n: number) => (m.k.length === n && ascending(m.k) ? m.k : centred(n));
  const kx = axisK(nx), ky = axisK(ny);
  const xEdges = edgesOf(kx), yEdges = edgesOf(ky);
  return {
    z: kx.map((_, i) => ky.map((__, j) => log2Ratio(grid[j]?.[i]))),
    xEdges, yEdges, kx, ky, size: nx,
    extent: Math.max(...[...xEdges, ...yEdges].map(Math.abs)),
    periodLines: m.period_lines_px, nWindows: m.n_windows,
  };
}

/** The map's colour range: a ratio from ½ to 2 on a log scale (log₂ −1 … 1), white at 1. */
export const MAP_LOG2_RANGE: [number, number] = [-1, 1];
export const MAP_COLOR_TICKS: Tick[] = [0.5, 1, 2].map((v) => ({ v: Math.log2(v), label: String(v) }));
/** Map axis ticks: every 16 cycles. */
export function mapTicks(extent: number): Tick[] {
  const out: Tick[] = [];
  for (let v = -Math.floor(extent / 16) * 16; v <= extent; v += 16) out.push({ v, label: String(v) });
  return out;
}

const RDBU = colormapLut("rdbu");
/** The map's diverging colour (the viewer's residual colormap, RdBu_r):
 *  t = 0 blue, ½ white, 1 red. Module-level, so the plot sees one identity. */
export function divergingColor(t: number): string {
  const i = Math.round(Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0.5)) * 255) * 3;
  return `rgb(${RDBU[i]}, ${RDBU[i + 1]}, ${RDBU[i + 2]})`;
}

/** Circles at the given periods (|k| = size / period), those inside the map. */
export function ringCurves(periods: readonly number[], size: number, extent: number, steps = 96): { period: number; x: number[]; y: number[] }[] {
  return periods.filter((p) => p > 0 && size / p <= extent).map((p) => {
    const r = size / p;
    const x: number[] = [], y: number[] = [];
    for (let s = 0; s <= steps; s++) {
      const t = (2 * Math.PI * s) / steps;
      x.push(r * Math.cos(t));
      y.push(r * Math.sin(t));
    }
    return { period: p, x, y };
  });
}

/** A clicked cell: its wave vector, the ratio (null: outside the measured
 *  disc), the period (size / |k|) and the direction (0–180°, 0° and 90° the
 *  pixel axes); null when the cell is off the map. */
export type MapCellInfo = { kx: number; ky: number; ratio: number | null; period: number | null; angle: number | null };
export function mapCell(layer: MapLayer, cell: { i: number; j: number }): MapCellInfo | null {
  const kx = layer.kx[cell.i], ky = layer.ky[cell.j];
  if (kx == null || ky == null) return null;
  const z = layer.z[cell.i]?.[cell.j];
  const r = Math.hypot(kx, ky);
  return {
    kx, ky, ratio: z != null && Number.isFinite(z) ? 2 ** z : null,
    period: r > 0 ? layer.size / r : null,
    angle: r > 0 ? ((Math.atan2(ky, kx) * 180) / Math.PI + 360) % 180 : null,
  };
}

/** "kx 0, ky 16 · period 4 px at 90° · 3.20× the round expectation". */
export function cellText(c: MapCellInfo): string {
  const where = `kx ${c.kx}, ky ${c.ky}`;
  if (c.period == null || c.angle == null) return `${where} · the mean (no direction)`;
  const at = `period ${periodText(c.period)} at ${Math.round(c.angle)}°`;
  return c.ratio == null ? `${where} · ${at} · outside the measured disc` : `${where} · ${at} · ${c.ratio.toFixed(2)}× the round expectation`;
}

/* ── the compute job ───────────────────────────────────────────────────── */

const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/** "local job: 3 models, ~80 s of spectra" (~25 s per model). */
export function spectraEstimate(nModels: number): string {
  const s = SECONDS_PER_MODEL * Math.max(0, nModels);
  const time = s < 120 ? `~${Math.max(10, Math.round(s / 10) * 10)} s` : `~${Math.round(s / 60)} min`;
  return `local job: ${plural(nModels, "model")}, ${time} of spectra`;
}
