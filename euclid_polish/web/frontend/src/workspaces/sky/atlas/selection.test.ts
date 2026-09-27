import { describe, expect, it } from "vitest";
import type { SkyFeature } from "./layerModel";
import { featuresInRegion, shapeToRegion, tileRefsOf } from "./selection";

const f = (layer: string, key: string, ra: number, dec: number, tile = false): SkyFeature => ({
  layer, key, label: key, ra, dec, sizeDeg: 0, props: {},
  inspect: tile ? { kind: "tile", id: `${layer === "nexus-tiles" ? "nexus" : "archive"}/${key}` } : { kind: "source", id: `${layer}/${key}` },
});

describe("region selection", () => {
  const pix2world = (x: number, y: number): [number, number] | null => (x < 0 ? null : [x / 10, y / 10]);

  it("converts Aladin's screen shapes to sky regions", () => {
    const rect = shapeToRegion({ label: "rect", x: 10, y: 20, w: 30, h: 40, contains: () => true, bbox: () => ({ x: 10, y: 20, w: 30, h: 40 }) }, pix2world);
    expect(rect).toEqual({ type: "polygon", points: [[1, 2], [4, 2], [4, 6], [1, 6]] });
    const circle = shapeToRegion({ label: "circle", x: 100, y: 0, r: 10, contains: () => true, bbox: () => ({ x: 90, y: -10, w: 20, h: 20 }) }, pix2world);
    expect(circle?.type).toBe("circle");
    if (circle?.type === "circle") {
      expect(circle.ra).toBe(10);
      expect(circle.r).toBeCloseTo(1, 6);
    }
    const poly = shapeToRegion({ label: "polygon", vertices: [{ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 10, y: 10 }], contains: () => true, bbox: () => ({ x: 0, y: 0, w: 10, h: 10 }) }, pix2world);
    expect(poly).toEqual({ type: "polygon", points: [[0, 0], [1, 0], [1, 1]] });
    // Off the sky → no region.
    expect(shapeToRegion({ label: "rect", x: -5, y: 0, w: 1, h: 1, contains: () => true, bbox: () => ({ x: 0, y: 0, w: 0, h: 0 }) }, pix2world)).toBeNull();
    expect(shapeToRegion({ label: "circle", x: 5, y: 5, r: 0, contains: () => true, bbox: () => ({ x: 0, y: 0, w: 0, h: 0 }) }, pix2world)).toBeNull();
  });

  it("collects the visible features whose centre lies inside, grouped by layer", () => {
    const layers = {
      "nexus-tiles": [f("nexus-tiles", "f200w-0000", 268.40, 65.20, true), f("nexus-tiles", "f200w-0001", 268.60, 65.20, true)],
      stars: [f("stars", "0", 268.41, 65.19), f("stars", "1", 10, 10)],
      galaxies: [f("galaxies", "g", 268.4, 65.2)],
    };
    const region = { type: "circle" as const, ra: 268.4, dec: 65.2, r: 0.05 };
    const groups = featuresInRegion(region, layers, ["nexus-tiles", "stars"]);
    expect(groups.map((g) => [g.layer, g.features.map((x) => x.key)])).toEqual([
      ["nexus-tiles", ["f200w-0000"]], ["stars", ["0"]],
    ]);
    expect(tileRefsOf(groups)).toEqual(["nexus/f200w-0000"]);
    expect(featuresInRegion(null, layers, ["stars"])).toEqual([]);
  });

  it("caps huge selections per layer but reports the true count", () => {
    const many = Array.from({ length: 50 }, (_, i) => f("stars", String(i), 1 + i * 1e-4, 1));
    const [g] = featuresInRegion({ type: "circle", ra: 1, dec: 1, r: 1 }, { stars: many }, ["stars"], 10);
    expect(g.features).toHaveLength(10);
    expect(g.total).toBe(50);
  });
});
