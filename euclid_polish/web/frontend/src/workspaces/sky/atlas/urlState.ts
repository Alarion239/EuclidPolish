/* URL state of the atlas (pure codecs; the hook is useAtlasUrl.ts).
 *
 *   /sky/atlas?ra=268.46&dec=65.2&fov=0.45&proj=SIN&base=q1-vis
 *             &ov=jwst-nircam:0.6&layers=moc-q1,nexus-tiles:0.5,stars::by.mag
 *             &sel=c:268.46,65.2,0.1&inspect=tile:nexus/f200w-0012
 *
 * `layers` entries are `id[:opacity[:colour]]` (colour = a sky token such as
 * `cat-3` / `good`, or `by.<prop>`); "-" means every layer off (absent means
 * the defaults). `ov` entries are `id[:opacity]`. `sel` is a region:
 * `c:ra,dec,r` or `p:ra,dec;ra,dec;…` (degrees). `goto` (palette) is a name or
 * coordinates, resolved once and removed. */
import { parseSkyCoord } from "../../../format";
import type { Region } from "../../../sky/geometry";
import { PROJECTIONS, type Projection } from "../../../sky/surveys";

export type LayerSetting = { id: string; opacity?: number; color?: string };
export type OverlaySetting = { id: string; opacity?: number };

type Codec<T> = { parse: (raw: string) => T | undefined; serialize: (v: T) => string | null };

const ID_RE = /^[a-z0-9][a-z0-9._-]*$/i;
const COLOR_RE = /^(?:by\.[\w.-]+|cat-\d|accent|good|warn|bad|info)$/;

const trimNum = (v: number, digits: number) => {
  const s = v.toFixed(digits);
  return String(Number(s) === 0 ? 0 : Number(s));
};

function parseOpacity(raw: string | undefined): number | undefined {
  if (raw == null || raw === "") return undefined;
  const n = Number(raw);
  if (!Number.isFinite(n)) return undefined;
  return Math.max(0, Math.min(1, n));
}

function parseEntries(raw: string): { id: string; opacity?: number; color?: string }[] {
  const byId = new Map<string, { id: string; opacity?: number; color?: string }>();
  for (const part of raw.split(",")) {
    const [id, op, color] = part.split(":");
    if (!id || !ID_RE.test(id)) continue;
    const e: { id: string; opacity?: number; color?: string } = { id };
    const o = parseOpacity(op);
    if (o != null) e.opacity = o;
    if (color && COLOR_RE.test(color)) e.color = color;
    byId.set(id, e); // a duplicate's settings win; it keeps its first position
  }
  return [...byId.values()];
}

function entry(e: { id: string; opacity?: number; color?: string }): string {
  const op = e.opacity != null ? trimNum(e.opacity, 2) : "";
  if (e.color) return `${e.id}:${op}:${e.color}`;
  return op ? `${e.id}:${op}` : e.id;
}

export const LAYERS_CODEC: Codec<LayerSetting[]> = {
  parse: (raw) => (raw.trim() === "-" ? [] : parseEntries(raw)),
  serialize: (v) => (v.length ? v.map(entry).join(",") : "-"),
};

export const OVERLAYS_CODEC: Codec<OverlaySetting[]> = {
  parse: (raw) => parseEntries(raw).map(({ id, opacity }) => (opacity != null ? { id, opacity } : { id })),
  serialize: (v) => v.map(({ id, opacity }) => entry({ id, opacity })).join(","),
};

export const REGION_CODEC: Codec<Region | null> = {
  parse: (raw) => {
    const m = /^([cp]):(.+)$/.exec(raw.trim());
    if (!m) return undefined;
    if (m[1] === "c") {
      const n = m[2].split(",").map(Number);
      if (n.length !== 3 || !n.every(Number.isFinite) || n[2] <= 0) return undefined;
      return { type: "circle", ra: n[0], dec: n[1], r: n[2] };
    }
    const pts = m[2].split(";").map((p) => p.split(",").map(Number));
    if (pts.length < 3 || !pts.every((p) => p.length === 2 && p.every(Number.isFinite))) return undefined;
    return { type: "polygon", points: pts.map(([a, b]) => [a, b] as [number, number]) };
  },
  serialize: (r) => {
    if (!r) return null;
    if (r.type === "circle") return `c:${fmtCoord(r.ra)},${fmtCoord(r.dec)},${fmtCoord(r.r)}`;
    return `p:${r.points.map(([a, b]) => `${fmtCoord(a)},${fmtCoord(b)}`).join(";")}`;
  },
};

/** RA/Dec for the URL: 5 decimals (≈ 0.04″), trailing zeros trimmed. */
export function fmtCoord(v: number): string {
  return trimNum(v, 5);
}

/** Field of view for the URL: 4 significant digits. */
export function fmtFovParam(v: number): string {
  if (!Number.isFinite(v)) return "360";
  return String(Number(v.toPrecision(4)));
}

export function parseProjection(raw: string): Projection | undefined {
  const p = raw.trim().toUpperCase();
  return (PROJECTIONS as readonly string[]).includes(p) ? (p as Projection) : undefined;
}

export type GotoTarget = { kind: "coord"; ra: number; dec: number } | { kind: "name"; name: string };

export function parseGoto(raw: string): GotoTarget | null {
  const text = raw.trim();
  if (!text) return null;
  const c = parseSkyCoord(text);
  if (c) return { kind: "coord", ra: c.ra, dec: c.dec };
  return { kind: "name", name: text };
}

export function toggleLayer(list: readonly LayerSetting[], id: string): LayerSetting[] {
  return list.some((l) => l.id === id) ? list.filter((l) => l.id !== id) : [...list, { id }];
}

export function updateLayer<T extends { id: string }>(list: readonly T[], id: string, patch: Partial<T>): T[] {
  if (!list.some((l) => l.id === id)) return list as T[];
  return list.map((l) => {
    if (l.id !== id) return l;
    const next = { ...l, ...patch } as T & Record<string, unknown>;
    for (const k of Object.keys(next)) if (next[k] === undefined) delete next[k];
    return next as T;
  });
}

/** The inspector id of a sky point card (`source:at/<ra>,<dec>`). */
export function pointTargetId(ra: number, dec: number): string {
  return `at/${fmtCoord(ra)},${fmtCoord(dec)}`;
}

/** `at/<ra>,<dec>` (or `<ra>,<dec>`) → degrees. */
export function parsePointId(id: string): { ra: number; dec: number } | null {
  const m = /^(?:at\/)?(-?[\d.]+),(-?[\d.]+)$/.exec(id.trim());
  if (!m) return null;
  const ra = Number(m[1]), dec = Number(m[2]);
  if (!Number.isFinite(ra) || !Number.isFinite(dec) || ra < 0 || ra >= 360 || Math.abs(dec) > 90) return null;
  return { ra, dec };
}

/** The Experiments tab with tiles preselected (it also reads the `tile` selection scope). */
export function experimentsHref(refs: readonly string[]): string {
  return refs.length ? `/sky/experiments?${new URLSearchParams({ tiles: refs.join(",") }).toString()}` : "/sky/experiments";
}

/** Set / delete query params, keeping every other param's exact spelling
 *  (`layers=a,b`, `inspect=tile:nexus/12` are never re-encoded). A new
 *  value is appended; an existing key keeps its position. */
export function patchSearch(search: string, patch: Readonly<Record<string, string | null>>): string {
  const segs = search.replace(/^\?/, "").split("&").filter(Boolean);
  const keyOf = (seg: string) => {
    const raw = seg.split("=")[0];
    try { return decodeURIComponent(raw.replace(/\+/g, " ")); } catch { return raw; }
  };
  const enc = (v: string) => encodeURIComponent(v).replace(/%2C/gi, ",").replace(/%3A/gi, ":").replace(/%2F/gi, "/");
  const done = new Set<string>();
  const out: string[] = [];
  for (const seg of segs) {
    const k = keyOf(seg);
    if (!(k in patch)) { out.push(seg); continue; }
    if (done.has(k)) continue;
    done.add(k);
    const v = patch[k];
    if (v != null) out.push(`${enc(k)}=${enc(v)}`);
  }
  for (const [k, v] of Object.entries(patch)) {
    if (!done.has(k) && v != null) out.push(`${enc(k)}=${enc(v)}`);
  }
  return out.length ? `?${out.join("&")}` : "";
}

/** The view that frames a feature: centred on it, the field of view a few
 *  times its size (a NEXUS tile, 25.6″ → ~2.6′), at least 1.2′ and at most
 *  30°. A deep link that names only the inspected feature (`?inspect=` with no
 *  `ra`/`dec`) opens here instead of on the whole sky. */
export function featureView(f: { ra: number; dec: number; sizeDeg: number }): { ra: number; dec: number; fov: number } {
  const size = Number.isFinite(f.sizeDeg) && f.sizeDeg > 0 ? f.sizeDeg : 0;
  return { ra: f.ra, dec: f.dec, fov: Math.min(30, Math.max(0.02, size * 6)) };
}
