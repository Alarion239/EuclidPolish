/* What to draw (pure): catalogue × URL layer settings × loaded data × view →
 * one RenderSpec per layer (visibility, opacity, colour scale, LOD). */
import { lodFor } from "../../../sky/lod";
import { scaleFor, type ColorScale, type SkyPalette } from "./colorScale";
import type { LayerData } from "./layerData";
import type { LayerInfo, SkyFeature } from "./layerModel";
import type { RenderSpec } from "./render";
import type { LayerSetting } from "./urlState";

const DEFAULT_OPACITY: Record<LayerInfo["kind"], number> = { points: 0.9, polygons: 0.7, circles: 0.45, moc: 0.3 };

/* A scale scans the features (domains, categories: O(n log n) for the 43k
   stars), and the specs are rebuilt on every zoom step: cache it per
   feature array (stable per payload) × layer × override × palette. */
const scaleCache = new WeakMap<readonly SkyFeature[], Map<string, ColorScale>>();

function cachedScale(info: LayerInfo, features: readonly SkyFeature[], palette: SkyPalette, override?: string): ColorScale {
  const key = [info.id, info.style.color_by ?? "", override ?? "", palette.accent, palette.muted, palette.bad, ...palette.cat].join("|");
  let byKey = scaleCache.get(features);
  if (!byKey) { byKey = new Map(); scaleCache.set(features, byKey); }
  let hit = byKey.get(key);
  if (!hit) { hit = scaleFor(info, features, palette, override); byKey.set(key, hit); }
  return hit;
}

export function layerOpacity(info: LayerInfo, setting?: LayerSetting): number {
  return setting?.opacity ?? info.style.opacity ?? DEFAULT_OPACITY[info.kind];
}

/** The per-view JWST footprints pseudo-layer (drawn with "jwst-mast"). */
export const FOOTPRINTS_INFO: LayerInfo = {
  id: "jwst-footprints", label: "JWST footprints in view", group: "catalogues", kind: "polygons", count: 0,
  bbox: null, style: {}, ready: true, reason: null, fill_action: null,
  description: "MAST observation polygons near the view centre", url: null,
};

export function buildSpecs(a: {
  layers: readonly LayerInfo[];
  settings: readonly LayerSetting[];
  data: Readonly<Record<string, LayerData>>;
  palette: SkyPalette;
  fov: number;
  width: number;
  markerScale: number;
  footprints?: { features: SkyFeature[]; version: number } | null;
}): RenderSpec[] {
  const out: RenderSpec[] = [];
  for (const info of a.layers) {
    const setting = a.settings.find((s) => s.id === info.id);
    const d = a.data[info.id];
    const features = d?.features ?? [];
    out.push({
      info, features, dataVersion: d?.version ?? 0, visible: !!setting,
      opacity: layerOpacity(info, setting),
      scale: cachedScale(info, features, a.palette, setting?.color),
      lod: info.kind === "moc" ? "shapes" : lodFor(d?.typicalSize ?? 0, a.fov, a.width),
      markerScale: a.markerScale,
    });
  }
  const jwst = a.settings.find((s) => s.id === "jwst-mast");
  if (a.footprints) {
    const info = FOOTPRINTS_INFO;
    const mast = a.layers.find((l) => l.id === "jwst-mast");
    out.push({
      info, features: a.footprints.features, dataVersion: a.footprints.version, visible: !!jwst,
      opacity: mast ? layerOpacity(mast, jwst) : 0.7,
      scale: cachedScale(mast ?? info, a.footprints.features, a.palette, jwst?.color),
      lod: "shapes", markerScale: a.markerScale,
    });
  }
  return out;
}
