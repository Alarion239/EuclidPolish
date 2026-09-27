/* Level of detail (pure). Aladin skips polygons smaller than their line
 * width (`isTooSmall`), so a 25″ tile vanishes in an all-sky view: a layer
 * whose typical feature is below LOD_MIN_PX on screen is drawn as centroid
 * markers, and as polygons / circles once zoomed in. */

/** Features smaller than this on screen (px) are drawn as markers. */
export const LOD_MIN_PX = 4;

/** Below this field of view (deg) the atlas fetches JWST footprints per view. */
export const FOOTPRINT_FETCH_FOV = 1.5;

/** Screen pixels per degree for a horizontal field of view `fovDeg` over `widthPx`. */
export function pixelsPerDegree(fovDeg: number, widthPx: number): number {
  if (!(fovDeg > 0) || !(widthPx > 0)) return 0;
  return widthPx / fovDeg;
}

/** Median of the finite feature diameters (deg); 0 when none. */
export function typicalSizeDeg(sizes: readonly number[]): number {
  const s = sizes.filter((v) => Number.isFinite(v) && v > 0).sort((a, b) => a - b);
  if (!s.length) return 0;
  return s[Math.floor((s.length - 1) / 2)];
}

export type Lod = "shapes" | "markers";

/** Shapes when a typical feature spans ≥ `minPx` on screen, else markers. */
export function lodFor(sizeDeg: number, fovDeg: number, widthPx: number, minPx = LOD_MIN_PX): Lod {
  if (!(sizeDeg > 0)) return "shapes";
  const ppd = pixelsPerDegree(fovDeg, widthPx);
  if (!(ppd > 0)) return "shapes";
  return sizeDeg * ppd >= minPx ? "shapes" : "markers";
}

export function wantsFootprints(fovDeg: number): boolean {
  return fovDeg > 0 && fovDeg <= FOOTPRINT_FETCH_FOV;
}

/** The per-view JWST footprint query (`/api/sky/jwst/footprints`): a cone
 *  of radius = the field of view (it covers the corners), its centre snapped to a grid of a third of
 *  the radius so panning reuses the cached answer instead of refetching on
 *  every move. Null when zoomed out past FOOTPRINT_FETCH_FOV. */
export function footprintsQuery(view: { ra: number; dec: number; fov: number } | null): string | null {
  if (!view || !wantsFootprints(view.fov)) return null;
  const r = Math.min(5, Math.max(0.3, Number(view.fov.toPrecision(2))));
  const step = r / 3;
  const snap = (v: number) => Number((Math.round(v / step) * step).toFixed(3));
  const ra = ((snap(view.ra) % 360) + 360) % 360;
  const dec = Math.max(-90, Math.min(90, snap(view.dec)));
  return `/api/sky/jwst/footprints?ra=${Number(ra.toFixed(3))}&dec=${dec}&r=${r}`;
}
