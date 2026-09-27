/* A tiny all-sky Mollweide projection for Home's static sky overview
 * (no Aladin on Home). Equatorial coordinates, north up, RA increasing to
 * the LEFT with RA 180° at the centre — the sky seen from inside, as the
 * atlas shows it. Pure; skyProjection.test.ts. */

const DEG = Math.PI / 180;
const R2 = Math.SQRT2;

/** SVG viewBox of the overview (2:1, the Mollweide ellipse fills it). */
export const SKY_W = 360;
export const SKY_H = 180;
const PAD = 2;

export type XY = { x: number; y: number };

/** Mollweide x ∈ [−2√2, 2√2], y ∈ [−√2, √2] for (ra, dec) in degrees. */
export function mollweide(ra: number, dec: number): XY {
  const phi = Math.max(-90, Math.min(90, dec)) * DEG;
  let lambda = -(((ra % 360) + 360) % 360 - 180) * DEG;       // RA grows leftwards
  if (lambda > Math.PI) lambda -= 2 * Math.PI;
  let theta = phi;
  if (Math.abs(Math.abs(phi) - Math.PI / 2) < 1e-9) theta = phi;
  else {
    for (let i = 0; i < 50; i += 1) {
      const f = 2 * theta + Math.sin(2 * theta) - Math.PI * Math.sin(phi);
      const d = f / (2 + 2 * Math.cos(2 * theta));
      theta -= d;
      if (Math.abs(d) < 1e-10) break;
    }
  }
  return { x: (2 * R2 / Math.PI) * lambda * Math.cos(theta), y: R2 * Math.sin(theta) };
}

/** Projected → SVG user units (north up). */
export function toSvg(p: XY): XY {
  const sx = (SKY_W / 2 - PAD) / (2 * R2);
  const sy = (SKY_H / 2 - PAD) / R2;
  return { x: SKY_W / 2 + p.x * sx, y: SKY_H / 2 - p.y * sy };
}

const fmt = (v: number) => (Math.round(v * 100) / 100).toString();
const at = (ra: number, dec: number) => {
  const q = toSvg(mollweide(ra, dec));
  return `${fmt(q.x)},${fmt(q.y)}`;
};

/** Split a sky polyline into SVG sub-paths wherever two consecutive points
 *  land on opposite sides of the map's RA seam (RA 0/360 at the edges). */
export function linePath(points: readonly (readonly [number, number])[]): string {
  let d = "";
  let prev: XY | null = null;
  for (const [ra, dec] of points) {
    if (!Number.isFinite(ra) || !Number.isFinite(dec)) { prev = null; continue; }
    const q = toSvg(mollweide(ra, dec));
    d += `${prev == null || Math.abs(q.x - prev.x) > SKY_W / 2 ? "M" : "L"}${fmt(q.x)},${fmt(q.y)}`;
    prev = q;
  }
  return d;
}

/* J2000 galactic frame: the north galactic pole and the galactic longitude
   of the north celestial pole. */
const RA_NGP = 192.85948;
const DEC_NGP = 27.12825;
const L_NCP = 122.93192;

/** Galactic (l, b) → equatorial J2000 [ra, dec], degrees. */
export function galacticToEquatorial(l: number, b: number): [number, number] {
  const bR = b * DEG, dNgp = DEC_NGP * DEG, dl = (L_NCP - l) * DEG;
  const sinDec = Math.sin(bR) * Math.sin(dNgp) + Math.cos(bR) * Math.cos(dNgp) * Math.cos(dl);
  const dec = Math.asin(Math.max(-1, Math.min(1, sinDec)));
  const y = Math.cos(bR) * Math.sin(dl);
  const x = Math.sin(bR) * Math.cos(dNgp) - Math.cos(bR) * Math.sin(dNgp) * Math.cos(dl);
  const ra = (((RA_NGP + Math.atan2(y, x) / DEG) % 360) + 360) % 360;
  return [ra, dec / DEG];
}

/** The galactic plane (b = 0) as an SVG path, broken at the RA seam. */
export function galacticPlane(step = 2): string {
  const pts: [number, number][] = [];
  for (let l = 0; l <= 360; l += step) pts.push(galacticToEquatorial(l, 0));
  return linePath(pts);
}

/** A small circle of `radiusDeg` around (ra, dec) as a closed SVG path. */
export function circlePath(ra: number, dec: number, radiusDeg: number, steps = 48): string {
  const a0 = ra * DEG, d0 = dec * DEG, r = radiusDeg * DEG;
  const pts: string[] = [];
  for (let i = 0; i < steps; i += 1) {
    const b = (2 * Math.PI * i) / steps;             // position angle
    const sinD = Math.sin(d0) * Math.cos(r) + Math.cos(d0) * Math.sin(r) * Math.cos(b);
    const d = Math.asin(Math.max(-1, Math.min(1, sinD)));
    const a = a0 + Math.atan2(Math.sin(b) * Math.sin(r) * Math.cos(d0), Math.cos(r) - Math.sin(d0) * sinD);
    pts.push(at(a / DEG, d / DEG));
  }
  return `M${pts.join("L")}Z`;
}

/** Where to write a field's name: beside its cone, on the map's outer side
 *  (right of it on the right half, left of it on the left half), vertically
 *  centred on it — clear of the cone and of a neighbouring field above or
 *  below. `y` is a text baseline. */
export function fieldLabel(ra: number, dec: number, radiusDeg: number, gap = 4): { x: number; y: number; anchor: "start" | "end" } {
  const centre = toSvg(mollweide(ra, dec));
  const xs = [...circlePath(ra, dec, radiusDeg, 24).matchAll(/(-?\d+(?:\.\d+)?),/g)].map((m) => Number(m[1]));
  const right = centre.x >= SKY_W / 2;
  const edge = right ? Math.max(centre.x, ...xs) : Math.min(centre.x, ...xs);
  return { x: right ? edge + gap : edge - gap, y: centre.y + 4, anchor: right ? "start" : "end" };
}

/** A sky polygon `[[ra, dec], …]` as a closed SVG path ("" when < 3 vertices). */
export function polygonPath(polygon: readonly (readonly number[])[]): string {
  const pts = polygon.filter((p) => p.length >= 2 && Number.isFinite(p[0]) && Number.isFinite(p[1]));
  if (pts.length < 3) return "";
  return `M${pts.map((p) => at(p[0], p[1])).join("L")}Z`;
}

/** Meridians every 60° of RA, parallels every 30° of Dec, and the outline. */
export function graticule(): { meridians: string[]; parallels: string[]; outline: string } {
  const line = (pts: [number, number][]) => `M${pts.map(([ra, dec]) => at(ra, dec)).join("L")}`;
  const meridians: string[] = [];
  for (let ra = 0; ra < 360; ra += 60) {
    if (ra === 0) continue;
    const pts: [number, number][] = [];
    for (let dec = -90; dec <= 90; dec += 5) pts.push([ra, dec]);
    meridians.push(line(pts));
  }
  meridians.push(line(Array.from({ length: 37 }, (_, i) => [180, -90 + 5 * i] as [number, number])));
  const parallels: string[] = [];
  for (let dec = -60; dec <= 60; dec += 30) {
    const pts: [number, number][] = [];
    for (let ra = 0.001; ra <= 359.999; ra += 5) pts.push([ra, dec]);
    pts.push([359.999, dec]);
    parallels.push(line(pts));
  }
  const edge: [number, number][] = [];
  for (let dec = -90; dec <= 90; dec += 3) edge.push([0.0001, dec]);
  for (let dec = 90; dec >= -90; dec -= 3) edge.push([359.9999, dec]);
  return { meridians, parallels, outline: `${line(edge)}Z` };
}
