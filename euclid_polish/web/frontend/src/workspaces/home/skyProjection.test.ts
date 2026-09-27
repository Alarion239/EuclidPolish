import { describe, expect, it } from "vitest";
import {
  circlePath, fieldLabel, galacticPlane, galacticToEquatorial, graticule, linePath, mollweide, polygonPath, SKY_H,
  SKY_W, toSvg,
} from "./skyProjection";

describe("mollweide (RA increasing to the left, RA 180° at the centre)", () => {
  it("maps the centre, the poles and the edges", () => {
    expect(mollweide(180, 0).x).toBeCloseTo(0, 9);
    expect(mollweide(180, 0).y).toBeCloseTo(0, 9);
    expect(mollweide(42, 90).y).toBeCloseTo(Math.SQRT2, 6);
    expect(mollweide(42, -90).y).toBeCloseTo(-Math.SQRT2, 6);
    expect(Math.abs(mollweide(0.0001, 0).x)).toBeCloseTo(2 * Math.SQRT2, 3);
  });

  it("puts larger RA to the left (sky as seen from inside)", () => {
    expect(mollweide(190, 0).x).toBeLessThan(0);
    expect(mollweide(170, 0).x).toBeGreaterThan(0);
  });

  it("is equal-area: the latitude map is monotonic", () => {
    let prev = -Infinity;
    for (let dec = -90; dec <= 90; dec += 5) {
      const { y } = mollweide(180, dec);
      expect(y).toBeGreaterThan(prev);
      prev = y;
    }
  });
});

describe("svg helpers", () => {
  it("maps projected coordinates into the viewBox (north up)", () => {
    const c = toSvg(mollweide(180, 0));
    expect(c.x).toBeCloseTo(SKY_W / 2, 6);
    expect(c.y).toBeCloseTo(SKY_H / 2, 6);
    expect(toSvg(mollweide(0, 90)).y).toBeLessThan(1e-6 + 2);
  });

  it("draws a small circle as a closed path around its centre", () => {
    const d = circlePath(269.7, 66, 6);
    expect(d.startsWith("M")).toBe(true);
    expect(d.endsWith("Z")).toBe(true);
    const centre = toSvg(mollweide(269.7, 66));
    const xs = [...d.matchAll(/(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)/g)].map((m) => Number(m[1]));
    expect(Math.min(...xs)).toBeLessThan(centre.x);
    expect(Math.max(...xs)).toBeGreaterThan(centre.x);
  });

  it("draws a polygon path and skips invalid vertices", () => {
    expect(polygonPath([[268.2, 65.1], [268.7, 65.1], [268.7, 65.3]])).toMatch(/^M.+L.+L.+Z$/);
    expect(polygonPath([[1, 2]])).toBe("");
  });

  it("builds a graticule of meridians and parallels", () => {
    const g = graticule();
    expect(g.meridians.length).toBe(6);
    expect(g.parallels.length).toBe(5);
    expect(g.outline.endsWith("Z")).toBe(true);
  });
});

describe("galactic plane", () => {
  it("converts galactic to equatorial J2000 coordinates", () => {
    const [ra0, dec0] = galacticToEquatorial(0, 0);          // the Galactic centre
    expect(ra0).toBeCloseTo(266.405, 2);
    expect(dec0).toBeCloseTo(-28.936, 2);
    const [, decPole] = galacticToEquatorial(0, 90);         // the north galactic pole
    expect(decPole).toBeCloseTo(27.128, 2);
    const [raAnti, decAnti] = galacticToEquatorial(180, 0);  // the anticentre
    expect(raAnti).toBeCloseTo(86.405, 2);
    expect(decAnti).toBeCloseTo(28.936, 2);
  });

  it("draws the plane as a line broken at the map's RA seam, never across it", () => {
    const d = galacticPlane();
    const parts = d.split("M").filter(Boolean);
    expect(parts.length).toBeGreaterThanOrEqual(2);
    for (const part of parts) {
      const xs = [...part.matchAll(/(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)/g)].map((m) => Number(m[1]));
      for (let i = 1; i < xs.length; i += 1) expect(Math.abs(xs[i] - xs[i - 1])).toBeLessThan(SKY_W / 2);
    }
  });

  it("starts a new sub-path where consecutive points jump across the seam", () => {
    expect(linePath([[170, 0], [175, 0], [185, 0]]).split("M").filter(Boolean)).toHaveLength(1);
    expect(linePath([[5, 0], [355, 0], [350, 0]]).split("M").filter(Boolean)).toHaveLength(2);
    expect(linePath([])).toBe("");
  });
});

describe("fieldLabel", () => {
  it("puts the label outside the cone, on the map's outer side", () => {
    const east = fieldLabel(52.9, -28.1, 6);          // EDF-F: right half → label to the right
    const centreF = toSvg(mollweide(52.9, -28.1));
    expect(east.anchor).toBe("start");
    expect(east.x).toBeGreaterThan(centreF.x + 3);
    const west = fieldLabel(269.7, 66, 6);            // EDF-N: left half → label to the left
    const centreN = toSvg(mollweide(269.7, 66));
    expect(west.anchor).toBe("end");
    expect(west.x).toBeLessThan(centreN.x - 3);
    expect(Math.abs(west.y - centreN.y)).toBeLessThan(6);
  });

  it("keeps two neighbouring southern fields' labels apart", () => {
    const f = fieldLabel(52.9, -28.1, 6);
    const s = fieldLabel(61.24, -48.42, 6);
    expect(Math.abs(f.y - s.y)).toBeGreaterThan(12);
  });
});
