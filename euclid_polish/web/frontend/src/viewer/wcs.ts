/* Celestial WCS for the image viewer: pixel ↔ sky for the zenithal TAN
 * (gnomonic) and SIN (orthographic, no PV terms) projections, from FITS header
 * keywords (Calabretta & Greisen 2002, "Paper II").
 *
 * Input: the compact JSON of the `X-Cube-WCS` cube header (contract C6) or an
 * object of the same keys: CTYPE1/2, CRVAL1/2, CRPIX1/2 and either
 * CD1_1..CD2_2, or PCi_j with CDELTi, or CDELTi alone; optional LONPOLE.
 * Conventions: FITS pixels are 1-based and axis 1 is the column (x); a cube
 * pixel (x, y) is 0-based with row 0 = FITS y = 1 — so FITS p = cube + 1.
 * Every function here takes and returns CUBE (0-based) pixel coordinates.
 * Checked against astropy in wcs.test.ts (≤ 1e-9 deg). */

export type Projection = "TAN" | "SIN";

export type Wcs = {
  proj: Projection;
  /** Reference pixel (FITS, 1-based). */
  crpix: [number, number];
  /** CRVAL of the longitude and latitude axes (deg). */
  crval: [number, number];
  /** Linear transform (deg per pixel), rows = world axis 1/2, cols = pixel axis 1/2. */
  cd: [[number, number], [number, number]];
  /** Its inverse. */
  inv: [[number, number], [number, number]];
  /** true when axis 1 is latitude (DEC first). */
  swapped: boolean;
  /** Native longitude of the celestial pole (deg). */
  lonpole: number;
};

export type Sky = { ra: number; dec: number };

const D2R = Math.PI / 180;
const R2D = 180 / Math.PI;

type Header = Record<string, unknown>;

const num = (h: Header, k: string): number | null => {
  const v = h[k];
  const n = typeof v === "number" ? v : typeof v === "string" && v.trim() !== "" ? Number(v) : NaN;
  return Number.isFinite(n) ? n : null;
};

function readHeader(input: unknown): Header | null {
  if (input == null || input === "") return null;
  if (typeof input === "string") {
    try {
      const parsed = JSON.parse(input);
      return parsed && typeof parsed === "object" ? parsed as Header : null;
    } catch {
      return null;
    }
  }
  return typeof input === "object" ? input as Header : null;
}

function axisKind(ctype: unknown): { kind: "lng" | "lat"; proj: string } | null {
  if (typeof ctype !== "string" || ctype.length < 8) return null;
  const head = ctype.slice(0, 4).replace(/-+$/, "").toUpperCase();
  const proj = ctype.slice(5, 8).toUpperCase();
  if (["RA", "GLON", "ELON"].includes(head)) return { kind: "lng", proj };
  if (["DEC", "GLAT", "ELAT"].includes(head)) return { kind: "lat", proj };
  return null;
}

/** Parse a WCS header (object or JSON string). Null when absent, malformed,
 *  non-celestial, singular or in a projection other than TAN/SIN. */
export function parseWcs(input: unknown): Wcs | null {
  const h = readHeader(input);
  if (!h) return null;
  const a1 = axisKind(h.CTYPE1), a2 = axisKind(h.CTYPE2);
  if (!a1 || !a2 || a1.kind === a2.kind || a1.proj !== a2.proj) return null;
  if (a1.proj !== "TAN" && a1.proj !== "SIN") return null;
  const crpix1 = num(h, "CRPIX1"), crpix2 = num(h, "CRPIX2");
  const crval1 = num(h, "CRVAL1"), crval2 = num(h, "CRVAL2");
  if (crpix1 == null || crpix2 == null || crval1 == null || crval2 == null) return null;

  let cd: [[number, number], [number, number]];
  const cd11 = num(h, "CD1_1"), cd12 = num(h, "CD1_2"), cd21 = num(h, "CD2_1"), cd22 = num(h, "CD2_2");
  if (cd11 != null || cd12 != null || cd21 != null || cd22 != null) {
    cd = [[cd11 ?? 0, cd12 ?? 0], [cd21 ?? 0, cd22 ?? 0]];
  } else {
    const c1 = num(h, "CDELT1"), c2 = num(h, "CDELT2");
    if (c1 == null || c2 == null) return null;
    const pc = [[num(h, "PC1_1") ?? 1, num(h, "PC1_2") ?? 0], [num(h, "PC2_1") ?? 0, num(h, "PC2_2") ?? 1]];
    // (+ 0 turns a -0 from a zero PC term into +0.)
    cd = [[c1 * pc[0][0] + 0, c1 * pc[0][1] + 0], [c2 * pc[1][0] + 0, c2 * pc[1][1] + 0]];
  }
  const det = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0];
  if (!(Math.abs(det) > 0) || !Number.isFinite(det)) return null;
  const inv: [[number, number], [number, number]] = [
    [cd[1][1] / det, -cd[0][1] / det],
    [-cd[1][0] / det, cd[0][0] / det],
  ];
  const swapped = a1.kind === "lat";
  const crval: [number, number] = swapped ? [crval2, crval1] : [crval1, crval2];
  const lonpoleKey = num(h, "LONPOLE");
  // Zenithal: the reference point is the native pole (θ0 = 90°); the default
  // LONPOLE is 180° unless δ0 ≥ θ0.
  const lonpole = lonpoleKey ?? (crval[1] >= 90 ? 0 : 180);
  return { proj: a1.proj as Projection, crpix: [crpix1, crpix2], crval, cd, inv, swapped, lonpole };
}

const wrap360 = (deg: number) => {
  const v = deg % 360;
  return v < 0 ? v + 360 : v;
};

/** Cube pixel (0-based x = column, y = row) → sky (deg, RA in [0, 360)). */
export function pixToSky(w: Wcs, x: number, y: number): Sky | null {
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  const dx = x + 1 - w.crpix[0], dy = y + 1 - w.crpix[1];
  const w1 = w.cd[0][0] * dx + w.cd[0][1] * dy;
  const w2 = w.cd[1][0] * dx + w.cd[1][1] * dy;
  const px = w.swapped ? w2 : w1;   // intermediate longitude (deg)
  const py = w.swapped ? w1 : w2;   // intermediate latitude (deg)
  const r = Math.hypot(px, py);
  const phi = r === 0 ? 0 : Math.atan2(px, -py);
  let theta: number;
  if (w.proj === "TAN") {
    theta = r === 0 ? Math.PI / 2 : Math.atan2(R2D, r);
  } else {
    const s = r * D2R;
    if (s > 1) return null;
    theta = Math.acos(s);
  }
  const a0 = w.crval[0] * D2R, d0 = w.crval[1] * D2R, pp = w.lonpole * D2R;
  const sinT = Math.sin(theta), cosT = Math.cos(theta);
  const dphi = phi - pp;
  // Rotate the native unit vector; atan2 on its components stays accurate
  // near the poles, where asin loses precision.
  const X = -cosT * Math.sin(dphi);
  const Y = sinT * Math.cos(d0) - cosT * Math.sin(d0) * Math.cos(dphi);
  const Z = sinT * Math.sin(d0) + cosT * Math.cos(d0) * Math.cos(dphi);
  const ra = a0 + Math.atan2(X, Y);
  const dec = Math.atan2(Z, Math.hypot(X, Y));
  return { ra: wrap360(ra * R2D), dec: dec * R2D };
}

/** Sky (deg) → cube pixel (0-based); null when the point is not on the
 *  projection's visible hemisphere. */
export function skyToPix(w: Wcs, ra: number, dec: number): { x: number; y: number } | null {
  if (!Number.isFinite(ra) || !Number.isFinite(dec)) return null;
  const a = ra * D2R, d = dec * D2R;
  const a0 = w.crval[0] * D2R, d0 = w.crval[1] * D2R, pp = w.lonpole * D2R;
  const da = a - a0;
  // The native unit vector: (X, Y) horizontal, Z = sin θ (no asin/acos, so
  // pixels near the reference point keep full precision).
  const X = -Math.cos(d) * Math.sin(da);
  const Y = Math.sin(d) * Math.cos(d0) - Math.cos(d) * Math.sin(d0) * Math.cos(da);
  const Z = Math.sin(d) * Math.sin(d0) + Math.cos(d) * Math.cos(d0) * Math.cos(da);
  if (!(Z > 0)) return null;
  const cosT = Math.hypot(X, Y);
  const r = w.proj === "TAN" ? R2D * cosT / Z : R2D * cosT;
  const phi = pp + Math.atan2(X, Y);
  const px = r * Math.sin(phi);
  const py = -r * Math.cos(phi);
  const w1 = w.swapped ? py : px;
  const w2 = w.swapped ? px : py;
  const dx = w.inv[0][0] * w1 + w.inv[0][1] * w2;
  const dy = w.inv[1][0] * w1 + w.inv[1][1] * w2;
  return { x: dx + w.crpix[0] - 1, y: dy + w.crpix[1] - 1 };
}

/** Mean pixel side (arcsec) from the linear transform. */
export function pixelScaleArcsec(w: Wcs): number {
  const det = w.cd[0][0] * w.cd[1][1] - w.cd[0][1] * w.cd[1][0];
  return Math.sqrt(Math.abs(det)) * 3600;
}

/** Great-circle separation (deg), haversine form (stable at small angles). */
export function angularSeparationDeg(ra1: number, dec1: number, ra2: number, dec2: number): number {
  const p1 = dec1 * D2R, p2 = dec2 * D2R;
  const dp = p2 - p1, dl = (ra2 - ra1) * D2R;
  const h = Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
  return 2 * Math.asin(Math.min(1, Math.sqrt(h))) * R2D;
}
