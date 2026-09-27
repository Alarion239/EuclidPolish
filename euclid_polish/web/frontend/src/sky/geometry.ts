/* Spherical geometry for the Sky atlas (pure; degrees everywhere).
 *
 * Positions are ICRS (RA, Dec) in degrees. Polygons are vertex lists
 * `[[ra, dec], …]` (open: the first vertex is not repeated). Small regions
 * (tiles, selections) are handled in the gnomonic (TAN) plane about their
 * centroid, which is exact for great-circle edges. */

export type RaDec = [number, number];

export type Region =
  | { type: "circle"; ra: number; dec: number; r: number }
  | { type: "polygon"; points: RaDec[] };

const D2R = Math.PI / 180;
const R2D = 180 / Math.PI;

export function normRa(ra: number): number {
  const r = ra % 360;
  return r < 0 ? r + 360 : r + 0; // + 0 turns -0 into 0
}

type Vec = [number, number, number];

function toVec(ra: number, dec: number): Vec {
  const a = ra * D2R, d = dec * D2R;
  const c = Math.cos(d);
  return [c * Math.cos(a), c * Math.sin(a), Math.sin(d)];
}

function fromVec([x, y, z]: Vec): RaDec {
  const n = Math.hypot(x, y, z);
  return [normRa(Math.atan2(y, x) * R2D), Math.asin(Math.max(-1, Math.min(1, z / n))) * R2D];
}

/** Great-circle separation (haversine; accurate at tiny separations). */
export function angularDistance(ra1: number, dec1: number, ra2: number, dec2: number): number {
  const d1 = dec1 * D2R, d2 = dec2 * D2R;
  const s = Math.sin((d2 - d1) / 2) ** 2 + Math.cos(d1) * Math.cos(d2) * Math.sin(((ra2 - ra1) * D2R) / 2) ** 2;
  return 2 * Math.asin(Math.min(1, Math.sqrt(s))) * R2D;
}

/** Mean direction of the points (undefined for antipodal sets: returns the
 *  first point then). */
export function centroid(points: readonly RaDec[]): RaDec {
  if (!points.length) return [0, 0];
  let x = 0, y = 0, z = 0;
  for (const [ra, dec] of points) {
    const v = toVec(ra, dec);
    x += v[0]; y += v[1]; z += v[2];
  }
  if (Math.hypot(x, y, z) < 1e-9 * points.length) return [normRa(points[0][0]), points[0][1]];
  return fromVec([x, y, z]);
}

/** Degenerate-safe centroid: null when the points cancel out (all-sky sets). */
function centroidOrNull(points: readonly RaDec[]): RaDec | null {
  let x = 0, y = 0, z = 0;
  for (const [ra, dec] of points) {
    const v = toVec(ra, dec);
    x += v[0]; y += v[1]; z += v[2];
  }
  const n = Math.hypot(x, y, z);
  return n < 1e-6 * Math.max(1, points.length) ? null : fromVec([x, y, z]);
}

/** Largest vertex-to-vertex separation (the polygon's angular diameter). */
export function polygonDiameter(points: readonly RaDec[]): number {
  let best = 0;
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      best = Math.max(best, angularDistance(points[i][0], points[i][1], points[j][0], points[j][1]));
    }
  }
  return best;
}

/** Gnomonic projection of (ra, dec) about (ra0, dec0); null behind the tangent plane. */
export function gnomonic(ra: number, dec: number, ra0: number, dec0: number): [number, number] | null {
  const a = ra * D2R, d = dec * D2R, a0 = ra0 * D2R, d0 = dec0 * D2R;
  const cosc = Math.sin(d0) * Math.sin(d) + Math.cos(d0) * Math.cos(d) * Math.cos(a - a0);
  if (cosc <= 1e-12) return null;
  const x = (Math.cos(d) * Math.sin(a - a0)) / cosc;
  const y = (Math.cos(d0) * Math.sin(d) - Math.sin(d0) * Math.cos(d) * Math.cos(a - a0)) / cosc;
  return [x, y];
}

function planarInside(x: number, y: number, poly: [number, number][]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** Is (ra, dec) inside the spherical polygon? (Polygons smaller than a hemisphere.) */
export function pointInPolygon(ra: number, dec: number, polygon: readonly RaDec[]): boolean {
  if (polygon.length < 3) return false;
  const c = centroidOrNull(polygon);
  if (!c) return false;
  const p = gnomonic(ra, dec, c[0], c[1]);
  if (!p) return false;
  const plane: [number, number][] = [];
  for (const [vr, vd] of polygon) {
    const q = gnomonic(vr, vd, c[0], c[1]);
    if (!q) return false;
    plane.push(q);
  }
  return planarInside(p[0], p[1], plane);
}

export function regionContains(region: Region, ra: number, dec: number): boolean {
  if (region.type === "circle") return angularDistance(ra, dec, region.ra, region.dec) <= region.r;
  return pointInPolygon(ra, dec, region.points);
}

export function regionCenter(region: Region): RaDec {
  return region.type === "circle" ? [region.ra, region.dec] : centroid(region.points);
}

/* ── galactic coordinates ─────────────────────────────────────────────── */

/* ICRS → galactic rotation (Hipparcos / IAU, J2000). */
const GAL: [Vec, Vec, Vec] = [
  [-0.0548755604162154, -0.873437090234885, -0.4838350155487132],
  [0.4941094278755837, -0.4448296299600112, 0.746982244497219],
  [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669],
];

/** ICRS (ra, dec) → galactic (l, b), degrees. */
export function icrsToGalactic(ra: number, dec: number): RaDec {
  const v = toVec(ra, dec);
  const g: Vec = [0, 1, 2].map((i) => GAL[i][0] * v[0] + GAL[i][1] * v[1] + GAL[i][2] * v[2]) as Vec;
  return fromVec(g);
}

/* ── views ────────────────────────────────────────────────────────────── */

/** A view (centre + field of view) framing the points with a margin. */
export function fitView(
  points: readonly RaDec[],
  { minFov = 0.005, margin = 1.3 }: { minFov?: number; margin?: number } = {},
): { ra: number; dec: number; fov: number } {
  if (!points.length) return { ra: 0, dec: 0, fov: 360 };
  const c = centroidOrNull(points);
  if (!c) return { ra: normRa(points[0][0]), dec: points[0][1], fov: 360 };
  let r = 0;
  for (const [ra, dec] of points) r = Math.max(r, angularDistance(ra, dec, c[0], c[1]));
  if (r >= 90) return { ra: c[0], dec: c[1], fov: 360 };
  return { ra: c[0], dec: c[1], fov: Math.min(360, Math.max(minFov, 2 * r * margin)) };
}

/** 360°, 12.3°, 30.0′, 25.6″. */
export function formatFov(deg: number): string {
  if (!Number.isFinite(deg)) return "—";
  if (deg >= 100) return `${Math.round(deg)}°`;
  if (deg >= 1) return `${deg.toFixed(1)}°`;
  if (deg * 60 >= 1) return `${(deg * 60).toFixed(1)}′`;
  return `${(deg * 3600).toFixed(1)}″`;
}

/* ── STC-S ────────────────────────────────────────────────────────────── */

/** `POLYGON [frame] ra dec ra dec …` → vertices (null for other shapes). */
export function stcsPolygon(stcs: string): RaDec[] | null {
  const tokens = stcs.trim().split(/\s+/);
  if (!tokens.length || tokens[0].toUpperCase() !== "POLYGON") return null;
  const nums: number[] = [];
  for (const t of tokens.slice(1)) {
    if (/^[a-z]/i.test(t)) { if (nums.length) break; continue; } // frame / flavour words
    const n = Number(t);
    if (!Number.isFinite(n)) return null;
    nums.push(n);
  }
  if (nums.length < 6 || nums.length % 2) return null;
  const out: RaDec[] = [];
  for (let i = 0; i < nums.length; i += 2) out.push([nums[i], nums[i + 1]]);
  return out;
}

export function polygonToStcs(points: readonly RaDec[]): string {
  return `POLYGON ICRS ${points.map(([ra, dec]) => `${ra} ${dec}`).join(" ")}`;
}
