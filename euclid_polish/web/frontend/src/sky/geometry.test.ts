import { describe, expect, it } from "vitest";
import {
  angularDistance, centroid, fitView, formatFov, icrsToGalactic, normRa, pointInPolygon,
  polygonDiameter, polygonToStcs, regionCenter, regionContains, stcsPolygon, type RaDec,
} from "./geometry";

const NEXUS_TILE_0: RaDec[] = [
  [268.3855965, 65.09367503], [268.36877695, 65.09368528],
  [268.36879904, 65.10076859], [268.38562306, 65.10075833],
];

describe("sky geometry", () => {
  it("normalises RA into [0, 360)", () => {
    expect(normRa(-10)).toBe(350);
    expect(normRa(360)).toBe(0);
    expect(normRa(725)).toBe(5);
  });

  it("angular distance: poles, equator, tiny separations", () => {
    expect(angularDistance(0, 90, 123, 90)).toBeCloseTo(0, 9);
    expect(angularDistance(0, 0, 90, 0)).toBeCloseTo(90, 9);
    expect(angularDistance(10, -30, 10, 60)).toBeCloseTo(90, 9);
    // 1″ at Dec +65 along RA: Δra = 1/3600/cos(65°)
    const dra = 1 / 3600 / Math.cos(65 * Math.PI / 180);
    expect(angularDistance(268.4, 65, 268.4 + dra, 65) * 3600).toBeCloseTo(1, 4);
  });

  it("centroid of a tile and across the RA wrap", () => {
    const [ra, dec] = centroid(NEXUS_TILE_0);
    expect(ra).toBeCloseTo(268.3772, 3);
    expect(dec).toBeCloseTo(65.0972, 3);
    const [wra, wdec] = centroid([[359, 0], [1, 0]]);
    expect(Math.min(wra, 360 - wra)).toBeCloseTo(0, 6);
    expect(wdec).toBeCloseTo(0, 6);
  });

  it("polygon diameter of a 25.5″ NEXUS tile", () => {
    const d = polygonDiameter(NEXUS_TILE_0) * 3600;
    expect(d).toBeGreaterThan(35);      // diagonal ≈ 25.5·√2 ≈ 36″
    expect(d).toBeLessThan(37.5);
  });

  it("point in spherical polygon (gnomonic about the centroid)", () => {
    expect(pointInPolygon(268.3772, 65.0972, NEXUS_TILE_0)).toBe(true);
    expect(pointInPolygon(268.40, 65.0972, NEXUS_TILE_0)).toBe(false);
    // A polygon straddling RA 0.
    const wrap: RaDec[] = [[359, -1], [1, -1], [1, 1], [359, 1]];
    expect(pointInPolygon(0.2, 0.3, wrap)).toBe(true);
    expect(pointInPolygon(180, 0, wrap)).toBe(false);
    expect(pointInPolygon(2, 0, wrap)).toBe(false);
  });

  it("regions: circle and polygon containment + centre", () => {
    const c = { type: "circle" as const, ra: 269.733, dec: 66.018, r: 6 };
    expect(regionContains(c, 268.46, 65.2)).toBe(true);
    expect(regionContains(c, 61.2, -48.4)).toBe(false);
    expect(regionCenter(c)).toEqual([269.733, 66.018]);
    const p = { type: "polygon" as const, points: NEXUS_TILE_0 };
    expect(regionContains(p, 268.3772, 65.0972)).toBe(true);
    expect(regionCenter(p)[1]).toBeCloseTo(65.0972, 3);
  });

  it("ICRS → galactic (known anchors)", () => {
    // The galactic centre and the north galactic pole (J2000).
    const [l0, b0] = icrsToGalactic(266.40499, -28.93617);
    expect(Math.min(l0, 360 - l0)).toBeLessThan(0.01);
    expect(b0).toBeCloseTo(0, 2);
    const [, bp] = icrsToGalactic(192.85948, 27.12825);
    expect(bp).toBeCloseTo(90, 2);
  });

  it("fits a view around points (zoom to)", () => {
    const v = fitView(NEXUS_TILE_0);
    expect(v.ra).toBeCloseTo(268.3772, 3);
    expect(v.dec).toBeCloseTo(65.0972, 3);
    expect(v.fov).toBeGreaterThan(0.01);
    expect(v.fov).toBeLessThan(0.05);
    expect(fitView([]).fov).toBe(360);
    expect(fitView([[10, 10]], { minFov: 0.1 }).fov).toBe(0.1);
    expect(fitView([[0, -80], [180, 80]]).fov).toBe(360);
  });

  it("formats a field of view", () => {
    expect(formatFov(360)).toBe("360°");
    expect(formatFov(12.345)).toBe("12.3°");
    expect(formatFov(0.5)).toBe("30.0′");
    expect(formatFov(25.6 / 3600)).toBe("25.6″");
    expect(formatFov(Number.NaN)).toBe("—");
  });

  it("parses and writes STC-S polygons", () => {
    expect(stcsPolygon("POLYGON ICRS 67.75 -47.73 66.91 -47.73 66.91 -48.26")).toEqual([
      [67.75, -47.73], [66.91, -47.73], [66.91, -48.26],
    ]);
    expect(stcsPolygon("POLYGON 1 2 3 4 5 6")).toEqual([[1, 2], [3, 4], [5, 6]]);
    expect(stcsPolygon("CIRCLE ICRS 1 2 3")).toBeNull();
    expect(stcsPolygon("POLYGON ICRS 1 2 3")).toBeNull();
    expect(polygonToStcs([[1, 2], [3.5, -4]])).toBe("POLYGON ICRS 1 2 3.5 -4");
  });
});
