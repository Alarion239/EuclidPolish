/* Files workspace — pure helpers (paths, planes, HDU selection, header
   rows, sky links, table paging, recent files, the per-frame statistics and
   the roots by stage). Unit-tested in model.test.ts. */
import { formatBytes, formatCount, formatNumber, formatRaDec } from "../../format";
import { readStorage, writeStorage } from "../../state/storage";
import type { SortState } from "../../ui";
import type {
  BandGroup, BrowseResponse, Crumb, HduSummary, Histogram, ImageStats, InspectResponse, InspectRoot, TableColumn,
  WcsSummary,
} from "./api";

export function basename(path: string): string {
  const parts = path.split("/").filter(Boolean);
  return parts[parts.length - 1] ?? path;
}

export function dirname(path: string): string {
  const i = path.replace(/\/+$/, "").lastIndexOf("/");
  return i > 0 ? path.slice(0, i) : "";
}

/** Breadcrumbs of a file's directory: the root label, then each folder below it. */
export function fileCrumbs(rel: string, root: InspectRoot | null): Crumb[] {
  const dir = dirname(rel);
  if (!dir) return [];
  if (!root || !(dir === root.rel || dir.startsWith(`${root.rel}/`))) {
    const parts = dir.split("/").filter(Boolean);
    return parts.map((name, i) => ({ name, rel: parts.slice(0, i + 1).join("/") }));
  }
  const crumbs: Crumb[] = [{ name: root.label, rel: root.rel }];
  const tail = dir.slice(root.rel.length).split("/").filter(Boolean);
  let current = root.rel;
  for (const part of tail) {
    current = `${current}/${part}`;
    crumbs.push({ name: part, rel: current });
  }
  return crumbs;
}

// ---- planes ------------------------------------------------------------------

/** Row-major multi-index (numpy order) of flat plane `k` over `axes`. */
export function planeIndex(axes: readonly number[], k: number): number[] {
  const out = new Array<number>(axes.length).fill(0);
  let rest = Math.max(0, Math.floor(k));
  for (let a = axes.length - 1; a >= 0; a--) {
    const n = Math.max(1, axes[a]);
    out[a] = rest % n;
    rest = Math.floor(rest / n);
  }
  return out;
}

/** Flat plane of a multi-index (each entry clamped to its axis). */
export function flatIndex(axes: readonly number[], index: readonly number[]): number {
  let k = 0;
  for (let a = 0; a < axes.length; a++) {
    const n = Math.max(1, axes[a]);
    const i = Math.min(n - 1, Math.max(0, Math.floor(index[a] ?? 0)));
    k = k * n + i;
  }
  return k;
}

/** A numpy-order shape in FITS order: NAXIS1 × NAXIS2 × … */
export function shapeText(shape: readonly number[] | null | undefined): string {
  if (!shape || !shape.length) return "—";
  return [...shape].reverse().join(" × ");
}

/** FITS axis name of leading (numpy) axis `k` of an `ndim`-D image. */
export function axisName(ndim: number, k: number): string {
  return `NAXIS${ndim - k}`;
}

export function planeAxesLabel(axes: readonly number[]): string {
  const ndim = axes.length + 2;
  return axes.map((_, k) => axisName(ndim, k)).join(" × ");
}

// ---- payload normalisation ---------------------------------------------------------

const TABLE_KINDS = new Set(["BinTableHDU", "TableHDU"]);

/** Fill the fields a pre-rework server (or a partial payload) leaves out, so a
 *  server that was not restarted after an update degrades instead of crashing:
 *  the old `/api/inspect` sent only `{hdu_index, name, kind, shape, dtype, cards}`. */
export function normalizeSummary(raw: InspectResponse): InspectResponse {
  const hdus = (raw.hdus ?? []).map((h): HduSummary => {
    const index = h.index ?? h.hdu_index ?? 0;
    const shape = h.shape ?? null;
    const ndim = h.ndim ?? shape?.length ?? 0;
    const type = h.type ?? (TABLE_KINDS.has(h.kind) ? "table" : !shape || !shape.length ? "empty" : ndim === 1 ? "vector" : "image");
    const planeAxes = h.plane_axes ?? (ndim > 2 && shape ? shape.slice(0, ndim - 2) : []);
    return {
      ...h, index, hdu_index: index, type, shape, ndim,
      plane_axes: planeAxes,
      planes: h.planes ?? (type === "image" ? planeAxes.reduce((a, b) => a * b, 1) : 0),
      viewable: h.viewable ?? type === "image",
      bands: h.bands ?? null, band: h.band ?? null, wcs: h.wcs ?? null, bunit: h.bunit ?? "",
    };
  });
  return { ...raw, hdus, band_groups: raw.band_groups ?? [], scan_truncated: !!raw.scan_truncated, stamp: raw.stamp ?? null,
    root: raw.root ?? null, roots: raw.roots ?? [] };
}

// ---- HDU selection -----------------------------------------------------------------

export type Selected = { key: string; hdu: HduSummary | null; group: BandGroup | null };

export const isGroupKey = (key: string) => key.startsWith("b:");

export function hduKeyOf(h: HduSummary | BandGroup): string {
  return "hdus" in h ? h.id : String(h.index);
}

export function hduByKey(s: InspectResponse, key: string): Selected | null {
  if (isGroupKey(key)) {
    const group = s.band_groups.find((g) => g.id === key);
    return group ? { key, hdu: null, group } : null;
  }
  if (!/^\d+$/.test(key)) return null;
  const hdu = s.hdus.find((h) => h.index === Number(key));
  return hdu ? { key, hdu, group: null } : null;
}

/** The first 4-band colour group (a file of band HDUs opens as its colour
 *  composite), else the first viewable image, else the first table / vector,
 *  else HDU 0. */
export function defaultHduKey(s: InspectResponse): string {
  if (s.band_groups.length) return s.band_groups[0].id;
  const image = s.hdus.find((h) => h.type === "image" && h.viewable);
  if (image) return String(image.index);
  const other = s.hdus.find((h) => h.type === "table" || h.type === "vector");
  return String(other?.index ?? 0);
}

/** The first view of a file with two or more colour groups (a results FITS:
 *  LR and SR): those groups side by side, when one of them is selected. An
 *  HDU picked by hand, or a lone group, keeps the viewer's own default. */
export function compareTiers(s: InspectResponse, sel: Selected): string[] | undefined {
  if (!sel.group || s.band_groups.length < 2) return undefined;
  return s.band_groups.slice(0, 2).map((g) => g.id);
}

/** The per-viewer display of that first comparison: the chips, frame labels
 *  and readout say "colour", so the two composites are drawn in colour
 *  (Lupton) rather than the console's default VIS greyscale. A per-viewer
 *  override (the Display panel is untouched); the bar's band group changes it. */
export function compareDisplay(tiers: readonly string[] | undefined): { color: "lupton" } | undefined {
  return tiers && tiers.length ? { color: "lupton" } : undefined;
}

/** A per-viewer knee for a bright target (the poster galaxy's core at VIS
 *  12 AB): when the plane's 99.9th percentile is above the default stretch's
 *  white (30·K0 e⁻), the knee goes where the default look puts it relative to
 *  that percentile, p99.9 / 30. The white point is the file's own — the
 *  server moves it to the plane's 99.99th percentile (`meta.color.
 *  default_asinh`, viewer_data `_fits_white`) — so the core keeps its
 *  structure and the disk stays visible (brightness 1×). Null otherwise. */
export function brightExposure(p999: number | null | undefined, K0: number): { knee: number } | null {
  if (p999 == null || !Number.isFinite(p999) || !(K0 > 0)) return null;
  if (!(p999 > 30 * K0)) return null;
  return { knee: p999 / 30 };
}

/** "LR_VIS" → "LR VIS", "SR_Y_E" → "SR Y" (the NISP suffix dropped), as the viewer's chips read. */
export function hduWords(name: string | null | undefined): string {
  return String(name ?? "").trim().replace(/(^|_)([YJH])_E$/i, "$1$2").replace(/_/g, " ").trim();
}

/** The page's HDU picker entry: the HDU's name first, its index second
 *  ("LR VIS (HDU 1)"; a non-image says its kind: "CAT (HDU 2, table)"). */
export function hduOptionLabel(h: Pick<HduSummary, "index" | "name" | "type">): string {
  const name = hduWords(h.name);
  const kind = h.type === "image" ? "" : `, ${h.type}`;
  return name ? `${name} (HDU ${h.index}${kind})` : `HDU ${h.index}${kind}`;
}

/** A 4-band group in the picker: "LR colour (HDUs 1–4)". */
export function groupOptionLabel(g: Pick<BandGroup, "prefix" | "hdus">): string {
  const head = String(g.prefix ?? "").replace(/[_\- ]+$/, "");
  const span = g.hdus.length ? ` (HDUs ${Math.min(...g.hdus)}–${Math.max(...g.hdus)})` : "";
  return `${head ? `${head} colour` : "Colour"}${span}`;
}

/** A label split for a middle ellipsis: the head (which gives way, CSS
 *  ellipsis) and the last `tail` characters (always shown), so a squeezed
 *  crumb reads "Post…(repo)" rather than "P…". Short labels stay whole. */
export function middleSplit(text: string, tail = 6): [string, string] {
  if (text.length <= tail * 2) return [text, ""];
  return [text.slice(0, -tail), text.slice(-tail)];
}

/** Band names without the `_E` suffix: "VIS Y J H". */
export function bandsText(bands: readonly string[] | null | undefined): string {
  return (bands ?? []).map((b) => b.replace(/_E$/, "")).join(" ");
}

/** One line about the selection: size · dtype · unit · bands (tables: rows × columns). */
export function hduFacts(sel: Selected): string {
  if (sel.group) {
    const g = sel.group;
    return [shapeText(g.shape), `${g.bands.length} bands`, g.bunit].filter(Boolean).join(" · ");
  }
  const h = sel.hdu;
  if (!h) return "";
  if (h.type === "table") return `${formatCount(h.nrows ?? 0)} rows × ${formatCount(h.ncols ?? 0)} columns`;
  if (!h.shape?.length) return "no data";
  return [
    shapeText(h.shape), (h.dtype ?? "").replace(/^[<>|=]/, "") + (h.scaling ? " (scaled)" : ""), h.bunit ?? "",
    bandsText(h.bands ?? (h.band ? [h.band] : null)),
  ].filter(Boolean).join(" · ");
}

export type View = "image" | "plot" | "table" | "header" | "provenance";

export const VIEW_LABELS: Record<View, string> = {
  image: "Image", plot: "Plot", table: "Table", header: "Header", provenance: "Provenance",
};

/** The views the selected HDU (or band group) supports, main view first. */
export function viewsFor(hdu: HduSummary | null, group?: BandGroup | null): View[] {
  if (group) return ["image", "provenance"];
  if (!hdu) return ["header", "provenance"];
  const main: View[] = hdu.type === "image" && hdu.viewable ? ["image"]
    : hdu.type === "table" ? ["table"] : hdu.type === "vector" ? ["plot"] : [];
  return [...main, "header", "provenance"];
}

export function defaultView(hdu: HduSummary | null, group?: BandGroup | null): View {
  return viewsFor(hdu, group)[0];
}

/** The viewer's `fits` collection params for a selection (the defaults —
 *  colour stacking, auto bin, the Display stretch — are left out). */
export function viewerParams(fits: string, sel: Selected, o: { stack: string; bin: string; render: string }): Record<string, string> {
  const p: Record<string, string> = { path: fits, hdu: sel.key };
  if (sel.hdu?.ndim === 3 && sel.hdu.bands && o.stack === "planes") p.stack = "planes";
  if (o.bin !== "auto") p.bin = o.bin;
  if (o.render === "log") p.render = "log";
  return p;
}

/** A `?slice=` link (spec §3: `/files?path=&hdu=&slice=`) → the viewer
 *  object of that plane (`p<k>`) and whether a band cube must switch to one
 *  plane at a time. `raw` is a flat plane index, a band name (`J_E`, `j`, `H`)
 *  or a multi-index (`1,2` / `[1, 2]`, numpy order); indices are clamped.
 *  Null when the selection has no planes to pick or `raw` means nothing. */
export function sliceTarget(sel: Selected, raw: string): { id: string; planes: boolean } | null {
  const hdu = sel.hdu;
  const text = raw.trim().replace(/^\[|\]$/g, "").trim();
  if (!hdu || !text || hdu.type !== "image") return null;
  const planes = hdu.planes ?? 1;
  const bandCube = hdu.ndim === 3 && !!hdu.bands?.length;
  if (planes <= 1) return null;
  let k: number | null = null;
  if (/^\d+$/.test(text)) {
    k = Number(text);
  } else if (/^\d+(\s*,\s*\d+)+$/.test(text)) {
    k = flatIndex(hdu.plane_axes ?? [], text.split(",").map((v) => Number(v.trim())));
  } else if (bandCube) {
    const want = text.toUpperCase();
    const i = hdu.bands!.findIndex((b) => b.toUpperCase() === want || b.toUpperCase().replace(/_E$/, "") === want);
    k = i >= 0 ? i : null;
  }
  if (k == null) return null;
  return { id: `p${Math.min(planes - 1, Math.max(0, k))}`, planes: bandCube };
}

// ---- header cards ---------------------------------------------------------------

export type CardRow = { i: number; key: string; value: string; comment: string };

export function cardRows(cards: readonly [string, string, string][] | undefined): CardRow[] {
  return (cards ?? []).map(([key, value, comment], i) => ({ i, key, value, comment }));
}

// ---- sky ----------------------------------------------------------------------------

export function fovArcsec(wcs: WcsSummary): number {
  return Math.max(wcs.width_arcsec, wcs.height_arcsec);
}

/** The Sky atlas centred on the image, with ~3× its extent for context
 *  (`fov` in degrees, at least 0.01°). */
export function skyHref(wcs: WcsSummary): string {
  const fov = Math.max(0.01, Math.min(180, 3 * wcs.fov_deg));
  const p = new URLSearchParams({ ra: wcs.ra.toFixed(6), dec: wcs.dec.toFixed(6), fov: fov.toPrecision(3) });
  return `/sky/atlas?${p.toString()}`;
}

/** The Sky atlas at one position (a table row); null for an invalid one. */
export function skyAt(ra: number, dec: number, fovDeg = 0.02): string | null {
  if (!Number.isFinite(ra) || !Number.isFinite(dec) || Math.abs(dec) > 90) return null;
  const p = new URLSearchParams({
    ra: (((ra % 360) + 360) % 360).toFixed(6), dec: dec.toFixed(6),
    fov: Math.max(0.001, Math.min(180, fovDeg)).toPrecision(3),
  });
  return `/sky/atlas?${p.toString()}`;
}

const RA_NAMES = ["ra", "ra_deg", "raj2000", "ra_j2000", "alpha_j2000", "alpha", "right_ascension", "ra_obj", "ra_icrs", "ra_mean"];
const DEC_NAMES = ["dec", "dec_deg", "dej2000", "dec_j2000", "delta_j2000", "delta", "declination", "dec_obj", "de_icrs", "dec_icrs", "dec_mean"];

/** A table's numeric RA / Dec columns (degrees assumed), by the usual names. */
export function skyColumns(columns: readonly TableColumn[]): { ra: string; dec: string } | null {
  const numeric = columns.filter((c) => (c.kind ?? "numeric") === "numeric");
  const find = (names: string[]) => {
    for (const n of names) {
      const hit = numeric.find((c) => c.name.toLowerCase() === n);
      if (hit) return hit.name;
    }
    return null;
  };
  const ra = find(RA_NAMES), dec = find(DEC_NAMES);
  return ra && dec ? { ra, dec } : null;
}

// ---- tables -------------------------------------------------------------------------

/** The DataTable row-number column id (never sent to the server). */
export const ROW_COLUMN = "#";

export function sortToServer(sort: SortState): { sort: string | null; desc: boolean } {
  const first = sort[0];
  if (!first || first.id === ROW_COLUMN) return { sort: null, desc: false };
  return { sort: first.id, desc: first.desc };
}

export function pageLabel(offset: number, limit: number, total: number): string {
  if (!total) return "0 rows";
  const lo = Math.min(total, offset + 1), hi = Math.min(total, offset + limit);
  return `${formatCount(lo)}–${formatCount(hi)} of ${formatCount(total)}`;
}

export function histogramSeries(h: Histogram): { x: number[]; y: number[] } {
  const x: number[] = [];
  for (let i = 0; i < h.counts.length; i++) x.push((h.edges[i] + h.edges[i + 1]) / 2);
  return { x, y: [...h.counts] };
}

// ---- recent files (a per-viewer convenience) ------------------------------------------

const RECENT_KEY = "ep-inspect-recent";
export const RECENT_MAX = 8;

export function readRecent(): string[] {
  try {
    const raw = JSON.parse(readStorage(RECENT_KEY) ?? "[]") as unknown;
    return Array.isArray(raw) ? raw.filter((v): v is string => typeof v === "string").slice(0, RECENT_MAX) : [];
  } catch {
    return [];
  }
}

export function pushRecent(rel: string): string[] {
  const next = [rel, ...readRecent().filter((r) => r !== rel)].slice(0, RECENT_MAX);
  writeStorage(RECENT_KEY, JSON.stringify(next));
  return next;
}

// ---- layout + labels -------------------------------------------------------------------

/** The page's two-column breakpoint (the `.insp-host` container query in
 *  files.css uses the same number). */
export const INSPECT_WIDE_PX = 1000;

/** Narrow = the host is below the breakpoint; an unmeasured host (0: tests,
 *  before layout) falls back to the viewport guess. */
export function isNarrowWidth(width: number, fallback: boolean): boolean {
  return width > 0 ? width < INSPECT_WIDE_PX : fallback;
}

const COMPACT = [{ exp: 12, p: "T" }, { exp: 9, p: "G" }, { exp: 6, p: "M" }, { exp: 3, p: "k" }];

/** A short axis / readout label: 25000 → "25k", 2.5e6 → "2.5M"; smaller
 *  values as `formatNumber` (3 significant digits). */
export function axisLabel(v: number): string {
  const a = Math.abs(v);
  if (!Number.isFinite(v) || a < 1e3) return formatNumber(v);
  let i = COMPACT.findIndex((s) => a >= 10 ** s.exp);
  if (i < 0) i = COMPACT.length - 1;
  let mant = Number((v / 10 ** COMPACT[i].exp).toPrecision(3));
  if (Math.abs(mant) >= 1000 && i > 0) { i -= 1; mant = Number((v / 10 ** COMPACT[i].exp).toPrecision(3)); }
  return `${mant}${COMPACT[i].p}`;
}

/** "under <where>" of the browser's deep search: the loaded folder's last
 *  crumb, else (still loading, or a stale listing) the root's label or the
 *  folder's own name. */
export function searchScope(
  dir: string, data: Pick<BrowseResponse, "dir" | "crumbs"> | null, roots: readonly InspectRoot[] = [],
): string {
  if (!dir) return "all roots";
  if (data && data.dir === dir && data.crumbs.length) return data.crumbs[data.crumbs.length - 1].name;
  return roots.find((r) => r.rel === dir)?.label ?? basename(dir);
}

// ---- per-frame statistics ------------------------------------------------------------

/** One frame the viewer shows and the plane its statistics describe. */
export type FrameTarget = { key: string; label: string; hdu: number; plane: number };

const bandWord = (band: string | null | undefined) => String(band ?? "").replace(/_E$/, "");

/** The frames of the statistics table, in the viewer's order: one row per
 *  shown tier (`h<index>` or a band group id), and one per band when the tier
 *  is a colour frame (a band group, or a band cube drawn in colour), so VIS,
 *  Y, J and H each get their own row. A cube shown one plane at a time reads
 *  the plane shown. Unknown keys are skipped. Row keys are unique: a band row
 *  is `<tier>#<band index>`. */
export function frameTargets(
  s: InspectResponse, sel: Selected, shown: readonly string[],
  o: { stacked: boolean; plane: number },
): FrameTarget[] {
  const out: FrameTarget[] = [];
  const seen = new Set<string>();
  for (const key of shown) {
    if (seen.has(key)) continue;
    const group = s.band_groups.find((g) => g.id === key);
    if (group) {
      seen.add(key);
      const prefix = groupOptionLabel({ prefix: group.prefix, hdus: [] });
      group.hdus.forEach((hdu, i) => {
        out.push({ key: `${key}#${i}`, label: `${prefix} · ${bandWord(group.bands[i])}`, hdu, plane: 0 });
      });
      continue;
    }
    const m = /^h(\d+)$/.exec(key);
    const h = m ? s.hdus.find((x) => x.index === Number(m[1])) : undefined;
    if (!h) continue;
    seen.add(key);
    const name = hduWords(h.name) || `HDU ${h.index}`;
    const cube = h.ndim === 3 && !!h.bands?.length;
    const selected = sel.hdu?.index === h.index;
    if (cube && (o.stacked || !selected)) {
      (h.bands ?? []).forEach((b, i) => {
        out.push({ key: `${key}#${i}`, label: `${name} · ${bandWord(b)}`, hdu: h.index, plane: i });
      });
    } else if (cube) {
      const bands = h.bands ?? [];
      const plane = Math.min(Math.max(0, o.plane), bands.length - 1);
      out.push({ key, label: `${name} · ${bandWord(bands[plane])}`, hdu: h.index, plane });
    } else if ((h.planes ?? 1) > 1 && selected && !o.stacked) {
      out.push({ key, label: `${name} · plane ${formatCount(o.plane)}`, hdu: h.index, plane: o.plane });
    } else {
      out.push({ key, label: name, hdu: h.index, plane: 0 });
    }
  }
  return out;
}

export type FrameStats = { median: number | null; sigma: number | null; p99: number | null; sum: number | null; nonFinite: number };

/** The numbers a frame's row shows: median, σ from the MAD, the 99th
 *  percentile, the total flux and the non-finite pixel count. */
export function frameStats(st: ImageStats): FrameStats {
  const fin = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);
  return {
    median: fin(st.median), sigma: fin(st.mad_std), p99: fin(st.percentiles?.["99"]), sum: fin(st.sum),
    nonFinite: (st.n_nan ?? 0) + (st.n_posinf ?? 0) + (st.n_neginf ?? 0),
  };
}

/** The histogram's clipping in words (its tooltip), or null when nothing fell outside. */
export function histClipText(h: Histogram): string | null {
  if (!(h.below > 0 || h.above > 0)) return null;
  return `${formatCount(h.below)} pixels below and ${formatCount(h.above)} above the plotted range`;
}

/** The sky footprint as one caption line. */
export function wcsCaption(wcs: WcsSummary | null | undefined): string {
  if (!wcs) return "No celestial WCS: the viewer assumes 0.1″ pixels.";
  return [
    `Centre ${formatRaDec(wcs.ra, wcs.dec)}`,
    `${formatNumber(wcs.pixscale_arcsec, { sig: 4 })}″ pixels`,
    `${formatNumber(wcs.width_arcsec, { sig: 4 })}″ × ${formatNumber(wcs.height_arcsec, { sig: 4 })}″`,
    wcs.ctype.join(" "),
    wcs.constructed ? "WCS built from RA/DEC" : "",
  ].filter(Boolean).join(" · ");
}

/** The file bar's one muted facts line: size, HDUs, compression, age, PROVID. */
export function fileFactsLine(s: InspectResponse, modified: string): string {
  const n = s.hdus.length;
  const g = s.band_groups.length;
  return [
    formatBytes(s.file.size),
    `${formatCount(n)} HDU${n === 1 ? "" : "s"}${g ? ` + ${g} colour group${g === 1 ? "" : "s"}` : ""}`,
    s.file.compressed ? "compressed" : "",
    `modified ${modified}`,
    s.stamp ? `PROVID ${s.stamp.id}` : "",
  ].filter(Boolean).join(" · ");
}

// ---- roots by stage ------------------------------------------------------------------

/** The pipeline stages the browser groups its roots by, in order. */
export const ROOT_STAGES = ["Reference data", "Synthetic", "Real sky", "Evaluation", "Figures", "Bookkeeping", "Other"] as const;
export type RootStage = typeof ROOT_STAGES[number];

/** Which stage each inspectable root (helpers/paths.py `_root_specs`) belongs to. */
const STAGE_OF: Record<string, RootStage> = {
  stars: "Reference data", psf: "Reference data", tng: "Reference data", sky: "Reference data",
  population: "Reference data",
  records: "Synthetic",
  inference: "Real sky", jwst: "Real sky", "viewer-results": "Real sky",
  eval: "Evaluation",
  vis: "Figures", poster: "Figures", output: "Figures",
  tracking: "Bookkeeping", "fasrc-cache": "Bookkeeping",
};

export const rootStage = (id: string | null | undefined): RootStage => STAGE_OF[String(id ?? "")] ?? "Other";

/** Root entries in stage order (the browse order kept within a stage). */
export function sortRootsByStage<T extends { root_id?: string }>(entries: readonly T[]): T[] {
  const rank = (e: T) => ROOT_STAGES.indexOf(rootStage(e.root_id));
  return entries.map((e, i) => ({ e, i })).sort((a, b) => rank(a.e) - rank(b.e) || a.i - b.i).map((x) => x.e);
}
