/* Data workspace — pure logic (unit-tested in model.test.ts): the star
 * catalogue decoding + filters, histograms, the records' source-map geometry,
 * ids of the inspector kinds, PSF state wording and the TNG explorer's
 * grouping. No React, no fetch. */
import { formatBytes } from "../../format";
import { extent } from "../../ticks";
import type {
  Grid, PsfState, SplitInfo, SrState, StarsPayload, TngPayload, TruthSource,
} from "./api";

export type Tone = "neutral" | "good" | "warn" | "bad" | "info" | "accent";

/* ── ids of the inspector kinds (ids.ts; re-exported) ──────────────────── */

export { clusterObjectId, parseClusterId, parseTruthId, recordObjectId, truthId } from "./ids";

/* ── sky atlas links (W-SkyAtlas URL contract: atlas/urlState.ts) ──────── */

export function atlasHref(opts: { ra?: number | null; dec?: number | null; fov?: number; layers?: string[]; inspect?: string }): string {
  const q: string[] = [];
  if (opts.ra != null && opts.dec != null && Number.isFinite(opts.ra) && Number.isFinite(opts.dec)) {
    q.push(`ra=${Number(opts.ra.toFixed(6))}`, `dec=${Number(opts.dec.toFixed(6))}`);
  }
  if (opts.fov != null) q.push(`fov=${opts.fov}`);
  if (opts.layers?.length) q.push(`layers=${opts.layers.join(",")}`);
  if (opts.inspect) q.push(`inspect=${encodeURIComponent(opts.inspect).replace(/%3A/g, ":").replace(/%2F/g, "/")}`);
  return q.length ? `/sky/atlas?${q.join("&")}` : "/sky/atlas";
}

/* ── the star catalogue ────────────────────────────────────────────────── */

export const BANDS = ["VIS", "Y_E", "J_E", "H_E"] as const;
export const bandShort = (band: string) => band.replace(/_E$/, "");

export type BandState = "valid" | "corrupted" | "failed" | "pending";
export type StarBand = { valid: boolean; corrupted: boolean; failed: boolean; sizes: number[] };
export type Star = {
  id: number; ra: number | null; dec: number | null; mag: number | null;
  flux: number | null; fluxErr: number | null; field: string;
  bands: Record<string, StarBand>;
  /** In the cutouts navigator (valid in all four bands at its size). */
  nav: boolean;
  /** Bands with a valid cutout (0–4). */
  nValid: number;
};

const DEFAULT_BITS = { valid: 1, corrupted: 2, failed: 4, size_shift: 3 };

export function decodeBand(code: number, sizes: readonly number[], bits = DEFAULT_BITS): StarBand {
  return {
    valid: (code & bits.valid) !== 0,
    corrupted: (code & bits.corrupted) !== 0,
    failed: (code & bits.failed) !== 0,
    sizes: sizes.filter((_s, i) => (code & (1 << (bits.size_shift + i))) !== 0),
  };
}

/** One word per band: a valid cutout wins, then corrupted, then failed. */
export function bandState(b: StarBand | undefined): BandState {
  if (!b) return "pending";
  if (b.valid) return "valid";
  if (b.corrupted) return "corrupted";
  if (b.failed) return "failed";
  return "pending";
}

/** A star's overall state: its best band (valid in any band wins, then corrupted, …).
 *  The backend's summary counts use the same rule, so the KPIs match the "Overall" filter. */
export function starState(s: Star): BandState {
  const states = new Set(Object.values(s.bands).map(bandState));
  return (["valid", "corrupted", "failed"] as const).find((st) => states.has(st)) ?? "pending";
}

export const BAND_STATE_TONE: Record<BandState, Tone> = { valid: "good", corrupted: "warn", failed: "bad", pending: "neutral" };

export const BAND_STATE_HELP: Record<BandState, string> = {
  valid: "A cutout at this size was downloaded and passed validation.",
  corrupted: "Downloaded but rejected: NaN/Inf, all-zero or constant pixels, or an unopenable file.",
  failed: "The download failed: no mosaic tile matched the position, or its coordinates were invalid.",
  pending: "Never attempted for this band.",
};

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

export function decodeStars(p: StarsPayload | null | undefined): Star[] {
  if (!p?.present || !p.rows?.length) return [];
  const col = new Map(p.columns.map((c, i) => [c, i]));
  const at = (name: string) => col.get(name) ?? -1;
  const bandCols = p.bands.map((b) => [b, at(`b_${b}`)] as const);
  const bits = p.bits ?? DEFAULT_BITS;
  const iId = at("id"), iRa = at("ra"), iDec = at("dec"), iMag = at("mag"), iFlux = at("flux_uJy");
  const iErr = at("fluxerr_uJy"), iField = at("field"), iNav = at("nav");
  const out: Star[] = [];
  for (const row of p.rows) {
    const id = num(row[iId]);
    if (id == null) continue;
    const bands: Record<string, StarBand> = {};
    let nValid = 0;
    for (const [band, i] of bandCols) {
      const b = decodeBand(Number(row[i] ?? 0), p.sizes, bits);
      bands[band] = b;
      if (b.valid) nValid += 1;
    }
    out.push({
      id, ra: num(row[iRa]), dec: num(row[iDec]), mag: num(row[iMag]), flux: num(row[iFlux]),
      fluxErr: num(row[iErr]), field: typeof row[iField] === "string" ? (row[iField] as string) : "",
      bands, nav: row[iNav] === 1, nValid,
    });
  }
  return out;
}

export type CutoutFilter = "any" | "nav" | "all4" | "some" | "none";
export type StarFilter = {
  field: string;               // "all" | "EDF-N" | … | "none" (outside the deep fields)
  cutouts: CutoutFilter;
  band: string;                // "any" (the star's overall, best-band state) or a band name
  bandState: BandState | "any";
  mag: [number, number] | null;
};
export const DEFAULT_STAR_FILTER: StarFilter = { field: "all", cutouts: "any", band: "any", bandState: "any", mag: null };

export function starMatches(s: Star, f: StarFilter): boolean {
  if (f.field !== "all" && (f.field === "none" ? s.field !== "" : s.field !== f.field)) return false;
  switch (f.cutouts) {
    case "nav": if (!s.nav) return false; break;
    case "all4": if (s.nValid < 4) return false; break;
    case "some": if (s.nValid === 0 || s.nValid === 4) return false; break;
    case "none": if (s.nValid > 0) return false; break;
    default: break;
  }
  if (f.bandState !== "any") {
    const state = f.band === "any" ? starState(s) : bandState(s.bands[f.band]);
    if (state !== f.bandState) return false;
  }
  if (f.mag) {
    if (s.mag == null || s.mag < f.mag[0] || s.mag > f.mag[1]) return false;
  }
  return true;
}

export const filterStars = (stars: readonly Star[], f: StarFilter): Star[] => stars.filter((s) => starMatches(s, f));

export function fieldCounts(stars: readonly Star[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const s of stars) out[s.field || "none"] = (out[s.field || "none"] ?? 0) + 1;
  return out;
}

/** `"17.2-18.5"` ↔ `[17.2, 18.5]` (the URL's `mag`). */
export function parseRange(raw: string): [number, number] | null {
  const m = /^\s*(-?\d+(?:\.\d+)?)\s*[-:,]\s*(-?\d+(?:\.\d+)?)\s*$/.exec(raw);
  if (!m) return null;
  const a = Number(m[1]), b = Number(m[2]);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return a <= b ? [a, b] : [b, a];
}
export const serializeRange = (r: [number, number] | null): string =>
  r ? `${Number(r[0].toFixed(2))}-${Number(r[1].toFixed(2))}` : "";

/* ── histograms ────────────────────────────────────────────────────────── */

export type Histogram = { centers: number[]; counts: number[]; edges: number[] };

/** Counts of `values` in `bins` equal bins over [lo, hi] (the top edge closed). */
export function histogram(values: readonly (number | null | undefined)[], lo: number, hi: number, bins: number): Histogram {
  const n = Math.max(1, Math.floor(bins));
  const width = (hi - lo) / n || 1;
  const counts = new Array<number>(n).fill(0);
  for (const v of values) {
    if (v == null || !Number.isFinite(v) || v < lo || v > hi) continue;
    const i = Math.min(n - 1, Math.floor((v - lo) / width));
    counts[i] += 1;
  }
  const edges = Array.from({ length: n + 1 }, (_x, i) => lo + i * width);
  return { centers: counts.map((_c, i) => lo + (i + 0.5) * width), counts, edges };
}

/** A nice bin range for magnitudes: whole 0.1-mag bins covering [min, max]. */
export function magBins(min: number, max: number, step = 0.05): { lo: number; hi: number; bins: number } {
  const lo = Math.floor(min / step) * step;
  const hi = Math.max(lo + step, Math.ceil(max / step) * step);
  return { lo, hi, bins: Math.max(1, Math.round((hi - lo) / step)) };
}

export type Stats = { n: number; min: number; max: number; median: number; p16: number; p84: number } | null;

export function quantile(sorted: readonly number[], q: number): number {
  if (!sorted.length) return Number.NaN;
  const pos = (sorted.length - 1) * Math.min(1, Math.max(0, q));
  const i = Math.floor(pos), f = pos - i;
  return i + 1 < sorted.length ? sorted[i] + (sorted[i + 1] - sorted[i]) * f : sorted[i];
}

export function summaryStats(values: readonly (number | null | undefined)[]): Stats {
  const v = values.filter((x): x is number => x != null && Number.isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return null;
  return { n: v.length, min: v[0], max: v[v.length - 1], median: quantile(v, 0.5), p16: quantile(v, 0.16), p84: quantile(v, 0.84) };
}

/* ── records: SR tier + the truth-source map ───────────────────────────── */

export const SR_STATE_TONE: Record<SrState, Tone> = {
  current: "good", stale: "warn", partial: "warn", missing: "neutral", unknown: "info",
};
export const SR_STATE_LABEL: Record<SrState, string> = {
  current: "SR current", stale: "SR stale", partial: "SR partial", missing: "No SR", unknown: "SR unverified",
};

/** One local file of a split (LR, HR, Clean, Sources) for the Files badge's tip. */
export type RecordFileRow = { key: string; label: string; state: "ok" | "corrupt" | "missing"; detail: string };

const RECORD_FILES: [keyof SplitInfo["files"], string][] = [["dirty", "LR"], ["hr", "HR"], ["clean", "Clean"], ["sources", "Sources"]];

/** The split's four local files as ONE toolbar badge (label + tone) and the
 *  per-file rows for its tip: ok, truncated / corrupt (a TFRecord whose count
 *  is null), or not synced. */
export function recordFiles(files: SplitInfo["files"], split: string): { label: string; tone: Tone; rows: RecordFileRow[] } {
  const rows = RECORD_FILES.map(([key, label]): RecordFileRow => {
    const f = files[key];
    if (!f) return { key, label, state: "missing", detail: `${key}_${split} is not synced` };
    const count = "count" in f ? f.count : undefined;
    const parts = [f.name, formatBytes(f.size_bytes)];
    if (count === null) parts.push("truncated or corrupt");
    else if (count != null) parts.push(`${count} records`);
    return { key, label, state: count === null ? "corrupt" : "ok", detail: parts.join(", ") };
  });
  const corrupt = rows.filter((r) => r.state === "corrupt").length;
  const present = rows.filter((r) => r.state !== "missing").length;
  if (corrupt) return { label: `${corrupt} corrupt file${corrupt > 1 ? "s" : ""}`, tone: "bad", rows };
  if (present === rows.length) return { label: "Files ok", tone: "good", rows };
  return { label: `${present} of ${rows.length} files`, tone: "neutral", rows };
}

/** The records-noise system check (ok | warn | bad | unknown) in plain words. */
export function noiseBadge(state: string): { label: string; tone: Tone } {
  if (state === "ok") return { label: "Noise ok", tone: "good" };
  if (state === "bad") return { label: "Old noise model", tone: "bad" };
  if (state === "warn") return { label: "Check noise", tone: "warn" };
  return { label: "Noise unverified", tone: "neutral" };
}

/* ── truth-source types (Records' overlay chips) ── */

export const SOURCE_TYPES = ["galaxy", "star", "lens", "other"] as const;
export type SourceType = (typeof SOURCE_TYPES)[number];
const isSourceType = (v: string): v is SourceType => (SOURCE_TYPES as readonly string[]).includes(v);
/** A source's chip type (anything unknown is "other"). */
export const sourceType = (type: string): SourceType => (isSourceType(type) ? type : "other");

/** Hide a shown type / show a hidden one (the URL keeps the hidden types, so
 *  every type starts shown and a chip is a plain on / off toggle). */
export function toggleHiddenType(hidden: readonly string[], type: SourceType): SourceType[] {
  const clean = hidden.filter(isSourceType);
  return clean.includes(type) ? clean.filter((t) => t !== type) : [...clean, type];
}

/** Whether a source type is drawn / listed. */
export const shownTypes = (hidden: readonly string[]) => (type: string): boolean => !hidden.includes(sourceType(type));

/** One chip per type the record has: its count and whether it is shown. */
export function sourceTypeChips(counts: Partial<Record<SourceType, number>>, hidden: readonly string[]): { type: SourceType; count: number; shown: boolean }[] {
  const show = shownTypes(hidden);
  return SOURCE_TYPES.filter((t) => (counts[t] ?? 0) > 0).map((t) => ({ type: t, count: counts[t] ?? 0, shown: show(t) }));
}

/* ── the Cutouts gallery ── */

type CutoutFile = { file: string; id: number | null; size: number | null; mag: number | null };
export type CutoutTile = CutoutFile & { key: string; sizes: number[] };

/** One gallery tile per star (the cache holds a file per cutout size): the
 *  navigator's size when cached, else the largest; `sizes` lists them all.
 *  Files without a star id stay tiles of their own; first-seen order. */
export function cutoutTiles(items: readonly CutoutFile[], navSize: number | null): CutoutTile[] {
  const tiles: CutoutTile[] = [];
  const byId = new Map<number, CutoutTile>();
  const better = (a: CutoutFile, b: CutoutTile) => {
    if (navSize != null && (a.size === navSize) !== (b.size === navSize)) return a.size === navSize;
    return (a.size ?? 0) > (b.size ?? 0);
  };
  for (const it of items) {
    if (it.id == null) { tiles.push({ ...it, key: `file:${it.file}`, sizes: it.size != null ? [it.size] : [] }); continue; }
    const had = byId.get(it.id);
    if (!had) {
      const tile = { ...it, key: `star:${it.id}`, sizes: it.size != null ? [it.size] : [] };
      byId.set(it.id, tile);
      tiles.push(tile);
      continue;
    }
    if (it.size != null && !had.sizes.includes(it.size)) had.sizes = [...had.sizes, it.size].sort((a, b) => a - b);
    if (better(it, had)) Object.assign(had, { file: it.file, size: it.size, mag: it.mag ?? had.mag });
  }
  return tiles;
}

export type Marker = { row: number; kind: string; cx: number; cy: number; r: number; off: boolean; title: string };

/** SVG marker of one source on the HR grid: a galaxy's circle is its
 *  half-light radius, a lens ring its Einstein radius, a star a cross sized
 *  by brightness. */
export function sourceMarker(s: TruthSource, grid: Grid | null): Marker | null {
  if (s.x_pix == null || s.y_pix == null) return null;
  const scale = grid?.pixscale && grid.pixscale > 0 ? grid.pixscale : 0.05;
  let r: number;
  if (s.type === "lens") r = Math.max(4, (s.theta_E_arcsec ?? 0.5) / scale);
  else if (s.type === "galaxy") r = Math.max(2, Math.min(80, (s.re_arcsec ?? 0.1) / scale));
  else if (s.type === "star") r = s.mag_vis != null ? Math.max(2.5, Math.min(9, 2.5 + (22 - s.mag_vis) * 1.1)) : 3;
  else r = 3;
  const parts = [`${s.type} at (${s.x_pix.toFixed(1)}, ${s.y_pix.toFixed(1)}) px`];
  if (s.mag_vis != null) parts.push(`VIS ${s.mag_vis.toFixed(2)}`);
  if (s.flux_vis_e != null) parts.push(`${formatCompact(s.flux_vis_e)} e⁻`);
  if (s.off_field) parts.push("off-field");
  return { row: s.row, kind: s.type, cx: s.x_pix, cy: s.y_pix, r, off: s.off_field, title: parts.join(", ") };
}

export function formatCompact(v: number): string {
  const a = Math.abs(v);
  if (a >= 1e6) return `${(v / 1e6).toFixed(a >= 1e7 ? 0 : 1)}M`;
  if (a >= 1e3) return `${(v / 1e3).toFixed(a >= 1e4 ? 0 : 1)}k`;
  return a >= 10 ? v.toFixed(0) : v.toPrecision(2);
}

/* ── PSFs ───────────────────────────────────────────────────────────────── */

export const PSF_STATE: Record<PsfState, { label: string; tone: Tone; hint: string }> = {
  empirical: { label: "empirical", tone: "good", hint: "The FASRC ePSF is synchronised locally; generation uses it." },
  no_empirical: {
    label: "no empirical PSF", tone: "warn",
    hint: "The last sync found no ePSF for this band on FASRC: generation uses the Gaussian fallback (the FWHM in the config).",
  },
  not_cached: {
    label: "not cached", tone: "neutral",
    hint: "Not synchronised to this machine yet — FASRC may well have one. Sync the ePSFs.",
  },
};

/** The PSFs toolbar's band badges: one per state, naming its bands
 *  ("J, H: not cached"; "All bands: …" when they all share it), band order. */
export function psfBandGroups(bands: readonly { name: string; state: PsfState }[]): { state: PsfState; bands: string[]; label: string }[] {
  const groups: { state: PsfState; bands: string[]; label: string }[] = [];
  for (const band of bands) {
    const g = groups.find((x) => x.state === band.state);
    if (g) g.bands.push(band.name);
    else groups.push({ state: band.state, bands: [band.name], label: "" });
  }
  for (const g of groups) {
    const who = groups.length === 1 && bands.length > 1 ? "All bands" : g.bands.map(bandShort).join(", ");
    g.label = `${who}: ${PSF_STATE[g.state].label}`;
  }
  return groups;
}

/* ── TNG explorer ──────────────────────────────────────────────────────── */

export type TngRow = {
  id: number | string; sfr: number | null; mass_stars: number | null; m_halo: number | null; reff: number | null;
  re_kpc: number | null; re_kpc_min: number | null; re_kpc_max: number | null; n_orient: number; local: number;
};

export function decodeTng(p: TngPayload | null | undefined): TngRow[] {
  if (!p?.rows?.length) return [];
  const col = new Map(p.columns.map((c, i) => [c, i]));
  const get = (row: (number | string | null)[], name: string) => {
    const i = col.get(name);
    return i == null ? null : row[i];
  };
  return p.rows.map((row) => ({
    id: (get(row, "id") ?? "") as number | string,
    sfr: num(get(row, "sfr")), mass_stars: num(get(row, "mass_stars")), m_halo: num(get(row, "m_halo")),
    reff: num(get(row, "reff")), re_kpc: num(get(row, "re_kpc")), re_kpc_min: num(get(row, "re_kpc_min")),
    re_kpc_max: num(get(row, "re_kpc_max")), n_orient: num(get(row, "n_orient")) ?? 0, local: num(get(row, "local")) ?? 0,
  }));
}

export type TngProp = "sfr" | "mass_stars" | "m_halo" | "reff" | "re_kpc" | "ssfr";
export const TNG_PROPS: { key: TngProp; label: string; unit: string; log: boolean }[] = [
  { key: "mass_stars", label: "Stellar mass", unit: "M☉", log: true },
  { key: "sfr", label: "SFR", unit: "M☉/yr", log: true },
  { key: "ssfr", label: "sSFR", unit: "1/yr", log: true },
  { key: "m_halo", label: "Halo (bound) mass", unit: "M☉", log: true },
  { key: "reff", label: "Half-mass radius (catalogue)", unit: "kpc", log: true },
  { key: "re_kpc", label: "Measured VIS Rₑ", unit: "kpc", log: true },
];
export const tngPropMeta = (key: string) => TNG_PROPS.find((p) => p.key === key) ?? TNG_PROPS[0];

export function tngValue(row: TngRow, key: TngProp | string): number | null {
  if (key === "ssfr") return row.sfr != null && row.mass_stars ? row.sfr / row.mass_stars : null;
  const v = (row as unknown as Record<string, unknown>)[key];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** Values usable on an axis (> 0 on a log axis). */
export const axisOk = (v: number | null, log: boolean): v is number => v != null && Number.isFinite(v) && (!log || v > 0);

/** `n` quantile edges of `values` (n + 1 numbers, ascending). */
export function quantileEdges(values: readonly number[], n: number): number[] {
  const v = [...values].filter(Number.isFinite).sort((a, b) => a - b);
  if (!v.length) return [];
  return Array.from({ length: n + 1 }, (_x, i) => quantile(v, i / n));
}

export type ScatterGroup = { key: string; label: string; t: number; x: number[]; y: number[]; ids: (number | string)[] };

/** Scatter points grouped into colour quantiles of `colorBy` (+ a "no value"
 *  group), skipping points an axis cannot show. `t` ∈ [0, 1] for a colormap. */
export function scatterGroups(
  rows: readonly TngRow[], x: string, y: string, colorBy: string,
  opts: { xlog: boolean; ylog: boolean; groups?: number; format?: (v: number) => string },
): { groups: ScatterGroup[]; hidden: number } {
  const n = Math.max(1, opts.groups ?? 5);
  const fmt = opts.format ?? ((v: number) => v.toPrecision(2));
  const colors = rows.map((r) => tngValue(r, colorBy));
  const edges = quantileEdges(colors.filter((c): c is number => c != null), n);
  const groups: ScatterGroup[] = edges.length
    ? Array.from({ length: n }, (_x, i) => ({
      key: `q${i}`, label: `${fmt(edges[i])} – ${fmt(edges[i + 1])}`, t: n === 1 ? 0.5 : i / (n - 1), x: [], y: [], ids: [],
    }))
    : [];
  const none: ScatterGroup = { key: "none", label: "no value", t: -1, x: [], y: [], ids: [] };
  let hidden = 0;
  rows.forEach((r, i) => {
    const xv = tngValue(r, x), yv = tngValue(r, y);
    if (!axisOk(xv, opts.xlog) || !axisOk(yv, opts.ylog)) { hidden += 1; return; }
    const c = colors[i];
    let g = none;
    if (c != null && groups.length) {
      let k = groups.length - 1;
      for (let j = 1; j < edges.length - 1; j++) if (c < edges[j]) { k = j - 1; break; }
      g = groups[k];
    }
    g.x.push(xv); g.y.push(yv); g.ids.push(r.id);
  });
  return { groups: [...groups.filter((g) => g.x.length), ...(none.x.length ? [none] : [])], hidden };
}

/** The row nearest a click in axis space (log axes compared in log10). */
export function nearestPoint(
  groups: readonly ScatterGroup[], at: { x: number; y: number },
  opts: { xlog: boolean; ylog: boolean; xSpan: number; ySpan: number },
): (number | string) | null {
  const tx = (v: number) => (opts.xlog ? Math.log10(v) : v);
  const ty = (v: number) => (opts.ylog ? Math.log10(v) : v);
  const cx = tx(at.x), cy = ty(at.y);
  let best: (number | string) | null = null;
  let bestD = Infinity;
  for (const g of groups) {
    for (let i = 0; i < g.x.length; i++) {
      const dx = (tx(g.x[i]) - cx) / (opts.xSpan || 1), dy = (ty(g.y[i]) - cy) / (opts.ySpan || 1);
      const d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = g.ids[i]; }
    }
  }
  return bestD <= 0.0025 ? best : null;    // within 5 % of the plot diagonal
}

/** A padded [lo, hi] domain (log-aware) for positive / any values. */
export function axisDomain(values: readonly number[], log: boolean): [number, number] {
  // extent(), never Math.min(...v): an argument spread overflows the stack past ~120k values.
  const range = extent(log ? values.filter((x) => x > 0) : values);
  if (!range) return log ? [1, 10] : [0, 1];
  let [lo, hi] = range;
  if (log) {
    const a = Math.log10(lo), b = Math.log10(hi);
    const pad = Math.max(0.05, (b - a) * 0.05);
    return [10 ** (a - pad), 10 ** (b + pad)];
  }
  if (lo === hi) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.05;
  return [lo - pad, hi + pad];
}

/** Histogram bins of a property (log10 bins on a log axis). */
export function propertyHistogram(values: readonly (number | null)[], log: boolean, bins = 30): Histogram & { log: boolean } {
  const v = values.filter((x): x is number => x != null && Number.isFinite(x) && (!log || x > 0));
  if (!v.length) return { centers: [], counts: [], edges: [], log };
  const t = log ? v.map(Math.log10) : v;
  let [lo, hi] = extent(t) ?? [0, 1];
  if (lo === hi) { lo -= 0.5; hi += 0.5; }
  const h = histogram(t, lo, hi, bins);
  return log
    ? { centers: h.centers.map((c) => 10 ** c), counts: h.counts, edges: h.edges.map((e) => 10 ** e), log }
    : { ...h, log };
}

/* ── synthetic_generate: a resume must not inherit a rebuild ───────────── */

const REBUILD_FLAG = /^--regenerate-splits(?:=.*)?$/;
const FORCE_FLAG = /^--force(?:=.*)?$/;

/** Split `--regenerate-splits[=| ]<splits>` and `--force` out of free-form extra
 *  flags: the step card prefills from the last run, and those tokens would turn
 *  what this section calls a resume into deleting and rebuilding splits. */
export function stripRebuildFlags(flags: string): { rest: string; dropped: string[] } {
  const tokens = flags.trim().split(/\s+/).filter(Boolean);
  const rest: string[] = [];
  const dropped: string[] = [];
  for (let i = 0; i < tokens.length; i += 1) {
    const t = tokens[i];
    if (t === "--regenerate-splits" && i + 1 < tokens.length && !tokens[i + 1].startsWith("--")) {
      dropped.push(`${t} ${tokens[i + 1]}`);
      i += 1;
    } else if (REBUILD_FLAG.test(t) || FORCE_FLAG.test(t)) {
      dropped.push(t);
    } else {
      rest.push(t);
    }
  }
  return { rest: rest.join(" "), dropped };
}

/** The step with its last-run `extra_flags` stripped of rebuild tokens (same
 *  object when there is nothing to strip), and what was dropped. */
export function resumeSafeStep<S extends { last_params?: Record<string, unknown> | null }>(
  step: S,
): { step: S; dropped: string[] } {
  const flags = step.last_params?.extra_flags;
  if (typeof flags !== "string") return { step, dropped: [] };
  const { rest, dropped } = stripRebuildFlags(flags);
  if (!dropped.length) return { step, dropped };
  return { step: { ...step, last_params: { ...step.last_params, extra_flags: rest } }, dropped };
}
