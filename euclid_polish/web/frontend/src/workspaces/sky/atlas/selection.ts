/* Region selection (pure): Aladin's `al.select` hands back a screen-pixel
 * shape; it becomes a sky region (URL `sel`) and the selection is every
 * visible feature whose centre lies inside — recomputed from the region, so
 * a shared link reproduces it. */
import { angularDistance, regionContains, type RaDec, type Region } from "../../../sky/geometry";
import type { AladinSelectionShape } from "../../../sky/types";
import { tileRefOf, type SkyFeature } from "./layerModel";

type Pix2World = (x: number, y: number) => [number, number] | null;

export function shapeToRegion(shape: AladinSelectionShape, pix2world: Pix2World): Region | null {
  const label = shape.label ?? "";
  if (label === "circle" && shape.x != null && shape.y != null && shape.r != null) {
    if (!(shape.r > 0)) return null;
    const c = pix2world(shape.x, shape.y);
    const edge = pix2world(shape.x + shape.r, shape.y);
    if (!c || !edge) return null;
    const r = angularDistance(c[0], c[1], edge[0], edge[1]);
    return r > 0 ? { type: "circle", ra: c[0], dec: c[1], r } : null;
  }
  let px: { x: number; y: number }[] = [];
  if (label === "polygon" && shape.vertices?.length) px = shape.vertices;
  else if (shape.x != null && shape.y != null && shape.w != null && shape.h != null) {
    const { x, y, w, h } = shape as { x: number; y: number; w: number; h: number };
    px = [{ x, y }, { x: x + w, y }, { x: x + w, y: y + h }, { x, y: y + h }];
  }
  if (px.length < 3) return null;
  const points: RaDec[] = [];
  for (const p of px) {
    const w = pix2world(p.x, p.y);
    if (!w) return null;
    points.push([w[0], w[1]]);
  }
  return { type: "polygon", points };
}

export type SelectionGroup = { layer: string; features: SkyFeature[]; total: number };

/** Features of the `visible` layers inside `region` (≤ `cap` listed per layer). */
export function featuresInRegion(
  region: Region | null,
  byLayer: Readonly<Record<string, readonly SkyFeature[]>>,
  visible: readonly string[],
  cap = 5000,
): SelectionGroup[] {
  if (!region) return [];
  const out: SelectionGroup[] = [];
  for (const layer of visible) {
    const feats = byLayer[layer];
    if (!feats?.length) continue;
    const inside: SkyFeature[] = [];
    let total = 0;
    for (const f of feats) {
      if (!regionContains(region, f.ra, f.dec)) continue;
      total++;
      if (inside.length < cap) inside.push(f);
    }
    if (total) out.push({ layer, features: inside, total });
  }
  return out;
}

/** The real-tile refs (`source/id`) in a selection, de-duplicated. */
export function tileRefsOf(groups: readonly SelectionGroup[]): string[] {
  const refs = new Set<string>();
  for (const g of groups) for (const f of g.features) { const r = tileRefOf(f); if (r) refs.add(r); }
  return [...refs];
}
