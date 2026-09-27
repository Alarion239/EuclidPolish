/* Hit testing on the sky (pure). Aladin 3.8.2 reports a click / hover on a
 * polygon or circle only when the pointer is on its OUTLINE (`isInStroke`),
 * so a click inside a 25″ tile hits nothing. The atlas therefore tests the
 * clicked position against the drawn shapes itself: every visible layer drawn
 * as shapes, the smallest containing feature wins (a NEXUS tile over the Q1
 * tile over the deep-field circle). Marker layers keep Aladin's own picking. */
import { angularDistance, pointInPolygon } from "../../../sky/geometry";
import type { SkyFeature } from "./layerModel";
import type { RenderSpec } from "./render";

function contains(f: SkyFeature, ra: number, dec: number): boolean {
  if (f.polygon) {
    // Cheap reject: no interior point is farther from the centroid than the diameter.
    if (angularDistance(ra, dec, f.ra, f.dec) > f.sizeDeg) return false;
    return pointInPolygon(ra, dec, f.polygon);
  }
  if (f.radius != null && f.radius > 0) return angularDistance(ra, dec, f.ra, f.dec) <= f.radius;
  return false;
}

function inFootprints(f: SkyFeature, ra: number, dec: number): number | null {
  let best: number | null = null;
  for (const poly of f.footprints ?? []) {
    if (pointInPolygon(ra, dec, poly)) {
      const d = Math.max(...poly.map(([a, b]) => angularDistance(a, b, ra, dec)));
      best = best == null ? d : Math.min(best, d);
    }
  }
  return best;
}

/** Every drawn shape containing (ra, dec), smallest first. */
export function featuresAt(ra: number, dec: number, specs: readonly RenderSpec[]): SkyFeature[] {
  if (!Number.isFinite(ra) || !Number.isFinite(dec)) return [];
  const hits: { f: SkyFeature; size: number }[] = [];
  for (const s of specs) {
    if (!s.visible || s.info.kind === "moc" || s.lod !== "shapes") continue;
    for (const f of s.features) {
      if (s.info.kind === "points") {
        const size = f.footprints?.length ? inFootprints(f, ra, dec) : null;
        if (size != null) hits.push({ f, size });
        continue;
      }
      if (contains(f, ra, dec)) hits.push({ f, size: f.sizeDeg });
    }
  }
  return hits.sort((a, b) => a.size - b.size).map((h) => h.f);
}

/** The feature a click at (ra, dec) means (only features with an inspector). */
export function featureAt(ra: number, dec: number, specs: readonly RenderSpec[]): SkyFeature | null {
  return featuresAt(ra, dec, specs).find((f) => f.inspect) ?? null;
}

/** Aladin's own (outline / marker) hit vs ours: the more specific wins —
 *  a marker (size 0) or the smaller shape. */
export function preferHit(native: SkyFeature | null, inside: SkyFeature | null): SkyFeature | null {
  if (!native) return inside;
  if (!inside) return native;
  return inside.sizeDeg < native.sizeDeg ? inside : native;
}
