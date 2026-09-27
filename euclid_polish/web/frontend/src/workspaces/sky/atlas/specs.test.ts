import { describe, expect, it } from "vitest";
import type { SkyPalette } from "./colorScale";
import type { LayerData } from "./layerData";
import { CLIENT_LAYERS, type LayerInfo, type SkyFeature } from "./layerModel";
import { buildSpecs, layerOpacity } from "./specs";

const P: SkyPalette = {
  accent: "#4c9ffe", good: "#56d68a", warn: "#ffcc66", bad: "#ff7a7a", info: "#6fb2ff",
  muted: "#8a97ab", ink: "#e5edf7", select: "#00e5ff", hover: "#ffffff",
  cat: ["#111111", "#222222", "#333333", "#444444", "#555555", "#666666", "#777777", "#888888"],
};
const TILES: LayerInfo = {
  id: "nexus-tiles", label: "NEXUS", group: "results", kind: "polygons", count: 445, bbox: null,
  style: { color_by: "state", opacity: 0.5 }, ready: true, reason: null, fill_action: null, description: "", url: "/api/sky/layer/nexus-tiles",
};
const data = (typicalSize: number, features: SkyFeature[] = []): LayerData => ({
  features, version: 7, typicalSize, loading: false, fetching: false, error: null, payload: null,
});

describe("render specs", () => {
  it("visibility, opacity and colour come from the URL settings", () => {
    const specs = buildSpecs({
      layers: [CLIENT_LAYERS[0], TILES], settings: [{ id: "nexus-tiles", opacity: 0.2, color: "bad" }],
      data: { "nexus-tiles": data(0.01) }, palette: P, fov: 0.5, width: 900, markerScale: 1,
    });
    const [moc, tiles] = specs;
    expect(moc.visible).toBe(false);
    expect(tiles.visible).toBe(true);
    expect(tiles.opacity).toBe(0.2);
    expect(tiles.scale).toMatchObject({ type: "fixed", color: P.bad });
    expect(tiles.dataVersion).toBe(7);
    expect(tiles.lod).toBe("shapes");
  });

  it("switches polygon layers to markers when zoomed out", () => {
    const [tiles] = buildSpecs({
      layers: [TILES], settings: [{ id: "nexus-tiles" }], data: { "nexus-tiles": data(0.01) },
      palette: P, fov: 360, width: 900, markerScale: 1,
    });
    expect(tiles.lod).toBe("markers");
    expect(tiles.opacity).toBe(0.5); // the layer's style default
  });

  it("default opacities by kind", () => {
    expect(layerOpacity({ ...TILES, style: {} })).toBe(0.7);
    expect(layerOpacity({ ...TILES, kind: "points", style: {} })).toBe(0.9);
    expect(layerOpacity(CLIENT_LAYERS[0])).toBe(0.3);
  });

  it("adds the per-view JWST footprints with the MAST layer's visibility", () => {
    const fp = { features: [], version: 3 };
    const mast: LayerInfo = { ...TILES, id: "jwst-mast", kind: "points", style: {} };
    const specs = buildSpecs({ layers: [mast], settings: [{ id: "jwst-mast" }], data: {}, palette: P, fov: 1, width: 900, markerScale: 1, footprints: fp });
    expect(specs.map((s) => s.info.id)).toEqual(["jwst-mast", "jwst-footprints"]);
    expect(specs[1].visible).toBe(true);
    expect(specs[1].dataVersion).toBe(3);
  });
});
