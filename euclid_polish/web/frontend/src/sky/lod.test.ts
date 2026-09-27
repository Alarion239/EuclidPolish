import { describe, expect, it } from "vitest";
import { LOD_MIN_PX, footprintsQuery, lodFor, pixelsPerDegree, typicalSizeDeg, wantsFootprints } from "./lod";

describe("level of detail", () => {
  it("pixels per degree from the horizontal field of view", () => {
    expect(pixelsPerDegree(360, 720)).toBe(2);
    expect(pixelsPerDegree(0.5, 800)).toBe(1600);
    expect(pixelsPerDegree(0, 800)).toBe(0);
    expect(pixelsPerDegree(10, 0)).toBe(0);
  });

  it("typical feature size is the median diameter", () => {
    expect(typicalSizeDeg([])).toBe(0);
    expect(typicalSizeDeg([1, 3, 2])).toBe(2);
    expect(typicalSizeDeg([0.01, 0.01, 5])).toBe(0.01);
    expect(typicalSizeDeg([Number.NaN, 0.5])).toBe(0.5);
  });

  it("25″ tiles are markers all-sky and polygons when zoomed in", () => {
    const tile = 36 / 3600;                        // NEXUS tile diagonal
    expect(lodFor(tile, 360, 900)).toBe("markers"); // sub-pixel
    expect(lodFor(tile, 20, 900)).toBe("markers");  // 0.45 px
    expect(lodFor(tile, 0.5, 900)).toBe("shapes");  // 18 px
    // The threshold itself.
    const fovAtThreshold = (tile * 900) / LOD_MIN_PX;
    expect(lodFor(tile, fovAtThreshold * 0.99, 900)).toBe("shapes");
    expect(lodFor(tile, fovAtThreshold * 1.01, 900)).toBe("markers");
    // Unknown sizes draw shapes (never hide data behind a marker by accident).
    expect(lodFor(0, 360, 900)).toBe("shapes");
  });

  it("Q1 MER tiles (0.53°) are polygons from a ~100° view", () => {
    expect(lodFor(0.75, 360, 900)).toBe("markers");
    expect(lodFor(0.75, 60, 900)).toBe("shapes");
  });

  it("JWST footprints are fetched per view only when zoomed in", () => {
    expect(wantsFootprints(360)).toBe(false);
    expect(wantsFootprints(0.8)).toBe(true);
  });

  it("footprint queries snap the centre so small pans reuse the cached answer", () => {
    expect(footprintsQuery(null)).toBeNull();
    expect(footprintsQuery({ ra: 268.46, dec: 65.2, fov: 20 })).toBeNull();
    const a = footprintsQuery({ ra: 268.4615, dec: 65.1964, fov: 0.45 });
    const b = footprintsQuery({ ra: 268.47, dec: 65.19, fov: 0.45 });
    expect(a).toBe(b);
    expect(a).toMatch(/^\/api\/sky\/jwst\/footprints\?ra=268\.\d+&dec=65\.\d+&r=0\.45$/);
    // The cone always covers the view: |snap error| ≤ step/2 = r/6, r = fov covers the corner (≤ 0.71·fov) plus the snap (≤ 0.24·fov).
    expect(footprintsQuery({ ra: 359.99, dec: 0, fov: 1 })).toMatch(/ra=0&/);
    expect(footprintsQuery({ ra: 10, dec: 89.99, fov: 1 })).toMatch(/dec=90&/);
  });
});

