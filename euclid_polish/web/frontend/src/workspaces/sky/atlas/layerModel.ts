/* The atlas's layer model (pure): the backend catalogue (`GET /api/sky/layers`,
 * contract C9), the client-side coverage MOCs, and one normalised feature
 * shape for every layer payload (`GET /api/sky/layer/<id>`: points rows,
 * polygon features, circle features). */
import { centroid, polygonDiameter, type RaDec } from "../../../sky/geometry";
import { COVERAGE_MOCS } from "../../../sky/surveys";
import type { InspectTarget } from "../../../state/inspector";

export type LayerGroup = "coverage" | "results" | "catalogues";

export type LayerStyle = {
  color?: string;
  color_by?: string;
  colors?: Record<string, string>;
  shape?: string;
  size?: number;
  opacity?: number;
  rejected_color?: string;
};

export type FillAction = { method: string; url: string; label: string; requires_fasrc?: boolean };

export type LayerInfo = {
  id: string;
  label: string;
  group: LayerGroup;
  kind: "points" | "polygons" | "circles" | "moc";
  count: number;
  bbox: { ra_min: number; ra_max: number; dec_min: number; dec_max: number } | null;
  style: LayerStyle;
  ready: boolean;
  reason: string | null;
  fill_action: FillAction | null;
  description: string;
  url: string | null;
  /** Client-side layers (coverage MOCs): drawn from `mocUrl`, no backend payload. */
  client?: boolean;
  mocUrl?: string;
};

export type LayersResponse = { groups?: string[]; layers: LayerInfo[] };

export type RawInspect = { kind: string; id: string };

export type PointsPayload = {
  kind: "points";
  columns: string[];
  rows: unknown[][];
  inspect?: { kind: string; prefix: string; id_column: string | null };
  flag_bits?: Record<string, number>;
  polygons_url?: string;
};

export type ShapeFeatureRaw = {
  id: string;
  polygon?: [number, number][];
  ra?: number;
  dec?: number;
  radius_deg?: number;
  props?: Record<string, unknown>;
  inspect?: RawInspect | null;
};

export type ShapesPayload = { kind: "polygons" | "circles"; features: ShapeFeatureRaw[] };

export type LayerPayload = (PointsPayload | ShapesPayload) & {
  id: string; label: string; count: number; group?: string;
};

export type SkyFeature = {
  layer: string;
  /** Unique within the layer (feature id, id column or row index). */
  key: string;
  label: string;
  ra: number;
  dec: number;
  polygon?: RaDec[];
  /** Extra footprints (JWST MAST rows carry their observation polygons). */
  footprints?: RaDec[][];
  radius?: number;
  /** Angular diameter (deg); 0 for points. */
  sizeDeg: number;
  props: Record<string, unknown>;
  inspect: InspectTarget | null;
};

export const GROUP_LABEL: Record<LayerGroup, string> = {
  coverage: "Coverage",
  results: "Real results",
  catalogues: "Catalogues",
};
const GROUP_ORDER: LayerGroup[] = ["coverage", "results", "catalogues"];

/** Layers shown on a first visit (the `layers` URL param overrides). */
export const DEFAULT_LAYERS: readonly string[] = [
  "moc-q1", "q1-fields", "nexus-tiles", "real-tiles", "poster", "pairs",
];

export const CLIENT_LAYERS: readonly LayerInfo[] = COVERAGE_MOCS.map((m) => ({
  id: m.id, label: m.label, group: "coverage" as const, kind: "moc" as const, count: 0, bbox: null,
  style: { opacity: 0.3 }, ready: true, reason: null, fill_action: null, description: m.description,
  url: null, client: true, mocUrl: m.url,
}));

/** The server catalogue with the client MOCs first in "coverage". */
export function withClientLayers(server: readonly LayerInfo[]): LayerInfo[] {
  const known = new Set(server.map((l) => l.id));
  return [...CLIENT_LAYERS.filter((l) => !known.has(l.id)), ...server];
}

/** A stand-in catalogue row for a layer whose payload arrived before the
 *  catalogue (a cold `/api/sky/layers` can take seconds): the payload names
 *  its kind, group and label; the style waits for the catalogue. */
export function stubLayerInfo(id: string, payload?: Pick<LayerPayload, "label" | "group" | "kind" | "count"> | null): LayerInfo {
  const group = (["coverage", "results", "catalogues"] as const).find((g) => g === payload?.group) ?? "results";
  return {
    id, label: payload?.label ?? id, group, kind: payload?.kind ?? "points", count: payload?.count ?? 0, bbox: null,
    style: {}, ready: true, reason: null, fill_action: null, description: "", url: null,
  };
}

/** The catalogue plus stand-ins for enabled layers it does not list yet. */
export function withStubLayers(
  known: readonly LayerInfo[], enabled: readonly string[],
  payloads: Readonly<Record<string, { payload: LayerPayload | null } | undefined>>,
): LayerInfo[] {
  const have = new Set(known.map((l) => l.id));
  const extra = enabled.filter((id) => !have.has(id) && payloads[id]?.payload).map((id) => stubLayerInfo(id, payloads[id]!.payload));
  return extra.length ? [...known, ...extra] : [...known];
}

export function groupLayers(layers: readonly LayerInfo[]): { group: LayerGroup; label: string; layers: LayerInfo[] }[] {
  return GROUP_ORDER
    .map((group) => ({ group, label: GROUP_LABEL[group], layers: layers.filter((l) => l.group === group) }))
    .filter((g) => g.layers.length > 0);
}

/* ── inspector targets ───────────────────────────────────────────────── */

/** The atlas opens real tiles in its own `tile` card (sky actions); the
 *  backend links them as `realtile:` (the Real-results card). */
export function atlasTarget(t: RawInspect | null | undefined): InspectTarget | null {
  if (!t || !t.kind || t.id == null) return null;
  return t.kind === "realtile" ? { kind: "tile", id: String(t.id) } : { kind: t.kind, id: String(t.id) };
}

/** `tile:` ids → a real-tile ref: `nexus/12`, `nexus/0012` and `12` are the
 *  palette's NEXUS tile numbers (→ `nexus/f200w-0012`); refs pass through. */
export function tileTargetId(id: string): string {
  const t = id.trim();
  const bare = /^(?:nexus\/)?(\d{1,4})$/i.exec(t);
  if (bare) return `nexus/f200w-${bare[1].padStart(4, "0")}`;
  return t;
}

/** `source:<layer>/<id>` → parts (the id may contain further slashes). */
export function sourceTargetParts(id: string): { layer: string; id: string } | null {
  const i = id.indexOf("/");
  if (i <= 0 || i === id.length - 1) return null;
  return { layer: id.slice(0, i), id: id.slice(i + 1) };
}

/** The real-tile ref (`source/id`) of a feature that is a real tile. */
export function tileRefOf(f: Pick<SkyFeature, "inspect">): string | null {
  return f.inspect?.kind === "tile" ? f.inspect.id : null;
}

/* ── normalisation ───────────────────────────────────────────────────── */

const num = (v: unknown): number | null => {
  const n = typeof v === "number" ? v : typeof v === "string" && v.trim() !== "" ? Number(v) : NaN;
  return Number.isFinite(n) ? n : null;
};

const isPolygon = (p: unknown): p is [number, number][] =>
  Array.isArray(p) && p.length >= 3 && p.every((v) => Array.isArray(v) && num(v[0]) != null && num(v[1]) != null);

function pointLabel(layerLabel: string, layerId: string, key: string, props: Record<string, unknown>): string {
  if (typeof props.label === "string" && props.label) return props.label;
  if (layerId === "stars") return `Star ${key}${props.mag != null ? ` · mag ${props.mag}` : ""}`;
  if (typeof props.target === "string" && props.target) return String(props.target);
  if (typeof props.id === "string" && props.id) return props.id;
  if (typeof props.obs_id === "string" && props.obs_id) return props.obs_id;
  if (typeof props.tile === "string" && props.tile) return `Tile ${props.tile}`;
  return `${layerLabel} · ${key}`;
}

function normalisePoints(p: LayerPayload & PointsPayload): SkyFeature[] {
  const cols = p.columns ?? [];
  const ira = cols.indexOf("ra"), idec = cols.indexOf("dec");
  const iRa = ira >= 0 ? ira : 0, iDec = idec >= 0 ? idec : 1;
  const idCol = p.inspect?.id_column ?? null;
  const iid = idCol ? cols.indexOf(idCol) : -1;
  const out: SkyFeature[] = [];
  (p.rows ?? []).forEach((row, index) => {
    if (!Array.isArray(row)) return;
    const ra = num(row[iRa]), dec = num(row[iDec]);
    if (ra == null || dec == null) return;
    const props: Record<string, unknown> = {};
    cols.forEach((c, i) => { if (c !== "polygons") props[c] = row[i]; });
    const key = iid >= 0 && row[iid] != null && row[iid] !== "" ? String(row[iid]) : String(index);
    const polys = cols.includes("polygons") ? row[cols.indexOf("polygons")] : null;
    const footprints = Array.isArray(polys) ? polys.filter(isPolygon).map((q) => q.map(([a, b]) => [Number(a), Number(b)] as RaDec)) : undefined;
    out.push({
      layer: p.id, key, label: pointLabel(p.label, p.id, key, props), ra, dec, sizeDeg: 0, props,
      footprints: footprints?.length ? footprints : undefined,
      inspect: p.inspect ? atlasTarget({ kind: p.inspect.kind, id: `${p.inspect.prefix ?? ""}${key}` }) : null,
    });
  });
  return out;
}

function normaliseShapes(p: LayerPayload & ShapesPayload): SkyFeature[] {
  const out: SkyFeature[] = [];
  for (const f of p.features ?? []) {
    const props = f.props ?? {};
    const label = typeof props.label === "string" && props.label ? props.label
      : typeof props.name === "string" && props.name ? props.name
        : typeof props.target === "string" && props.target ? props.target
          : `${p.label} ${f.id}`;
    if (p.kind === "circles") {
      const ra = num(f.ra), dec = num(f.dec), r = num(f.radius_deg);
      if (ra == null || dec == null || r == null) continue;
      out.push({ layer: p.id, key: String(f.id), label, ra, dec, radius: r, sizeDeg: 2 * r, props, inspect: atlasTarget(f.inspect) });
      continue;
    }
    if (!isPolygon(f.polygon)) continue;
    const polygon = f.polygon.map(([a, b]) => [Number(a), Number(b)] as RaDec);
    const c = centroid(polygon);
    const ra = num(props.ra) ?? c[0], dec = num(props.dec) ?? c[1];
    out.push({
      layer: p.id, key: String(f.id), label, ra, dec, polygon, sizeDeg: polygonDiameter(polygon), props,
      inspect: atlasTarget(f.inspect),
    });
  }
  return out;
}

export function normalisePayload(p: LayerPayload): SkyFeature[] {
  if (p.kind === "points") return normalisePoints(p as LayerPayload & PointsPayload);
  return normaliseShapes(p as LayerPayload & ShapesPayload);
}

/* ── JWST footprints near the view (`GET /api/sky/jwst/footprints`) ──── */

export type FootprintsResponse = {
  ra: number; dec: number; r: number; ready?: boolean; count?: number; truncated?: boolean;
  footprints: {
    obs_id: string; instrument?: string; filters?: string; target?: string; proposal_id?: string;
    exptime_s?: number; ra?: number; dec?: number; polygons?: [number, number][][]; status?: string;
  }[];
};

/** One polygon feature per observation polygon; they inspect as the MAST row. */
export function footprintFeatures(r: FootprintsResponse | null | undefined): SkyFeature[] {
  const out: SkyFeature[] = [];
  for (const fp of r?.footprints ?? []) {
    (fp.polygons ?? []).forEach((poly, i) => {
      if (!isPolygon(poly)) return;
      const polygon = poly.map(([a, b]) => [Number(a), Number(b)] as RaDec);
      const c = centroid(polygon);
      out.push({
        layer: "jwst-footprints", key: `${fp.obs_id}#${i}`, label: fp.target || fp.obs_id,
        ra: num(fp.ra) ?? c[0], dec: num(fp.dec) ?? c[1], polygon, sizeDeg: polygonDiameter(polygon),
        props: { obs_id: fp.obs_id, instrument: fp.instrument, filters: fp.filters, target: fp.target, exptime_s: fp.exptime_s, status: fp.status },
        inspect: { kind: "source", id: `jwst-mast/${fp.obs_id}` },
      });
    });
  }
  return out;
}

/** The loaded feature an inspector target points at (tile refs normalised). */
export function findFeatureByTarget(
  target: InspectTarget | null | undefined,
  byLayer: Readonly<Record<string, readonly SkyFeature[]>>,
): SkyFeature | null {
  if (!target || (target.kind !== "tile" && target.kind !== "source")) return null;
  const id = target.kind === "tile" ? tileTargetId(target.id) : target.id;
  if (target.kind === "source" && id.startsWith("at/")) return null;
  for (const feats of Object.values(byLayer)) {
    for (const f of feats) if (f.inspect && f.inspect.kind === target.kind && f.inspect.id === id) return f;
  }
  return null;
}

/* ── human facts ─────────────────────────────────────────────────────── */

const FACT_KEYS: [string, string][] = [
  ["state", "state"], ["field", "field"], ["grade", "grade"], ["kind", "kind"],
  ["flux_ratio_sr_over_lr", "SR/LR flux"], ["mag", "mag"], ["fwhm_arcsec", "FWHM ″"],
  ["vis_level_e", "VIS sky e⁻"], ["VIS", "VIS e⁻"], ["instrument", "instrument"], ["filters", "filters"],
  ["target", "target"], ["rows", "rows"], ["rejected", "rejected"], ["subset", "subset"],
];

function factValue(v: unknown): string | null {
  if (v == null || v === "") return null;
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(4)));
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "string") return v.length > 48 ? `${v.slice(0, 47)}…` : v;
  return null;
}

/** Up to `max` [label, value] facts for a hover tooltip. */
export function featureFacts(f: SkyFeature, max = 4): [string, string][] {
  const out: [string, string][] = [];
  for (const [k, label] of FACT_KEYS) {
    if (out.length >= max) break;
    const v = factValue(f.props[k]);
    if (v != null && !(k === "target" && v === f.label)) out.push([label, v]);
  }
  return out;
}
