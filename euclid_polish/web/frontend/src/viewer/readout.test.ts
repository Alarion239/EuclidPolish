import { describe, expect, it } from "vitest";
import golden from "./__fixtures__/color_golden.json";
import type { ColorMeta } from "./color";
import { cubeIsEmpty, formatValue, readoutLineWidth, magInfo, magLabel, pixelAt, pixelValues, sigmaMagnitude, unitLabel } from "./readout";

type Mag = { band: string; index: number; h: number; w: number; c: number; bands: string[];
  data: number[]; std: number[]; tot: number; mag: number; std_tot: number; dm: number };
const G = golden as unknown as { color_meta: ColorMeta; magnitudes: Mag[] };

describe("the magnitude overlay (photometry.electrons_to_ab_mag parity)", () => {
  for (const m of G.magnitudes) {
    it(m.band, () => {
      const rec = { h: m.h, w: m.w, c: m.c, bands: m.bands, data: Float32Array.from(m.data) };
      const std = { h: m.h, w: m.w, c: m.c, bands: m.bands, data: Float32Array.from(m.std) };
      const mi = magInfo(rec, G.color_meta, m.band)!;
      expect(mi.name).toBe(m.band);
      expect(mi.tot / m.tot).toBeCloseTo(1, 9);
      expect(mi.mag).toBeCloseTo(m.mag, 9);
      expect(sigmaMagnitude(mi, magInfo(std, G.color_meta, m.band)!)).toBeCloseTo(m.dm, 9);
    });
  }
  it("composite colours fall back to band 0; log mode, display-only and non-positive sums have none", () => {
    const rec = { h: 1, w: 1, c: 4, bands: ["VIS", "Y_E", "J_E", "H_E"], data: Float32Array.from([100, 1, 1, 1]) };
    expect(magInfo(rec, G.color_meta, "lupton")?.name).toBe("VIS");
    expect(magInfo(rec, { ...G.color_meta, render_mode: "log" }, "VIS")).toBeNull();
    const jw = { h: 1, w: 1, c: 1, bands: ["F200W"], data: Float32Array.from([5]) };
    expect(magInfo(jw, { ...G.color_meta, bands: { ...G.color_meta.bands, F200W: { zeropoint_ab_e_total: 1, display_only: true } } }, "VIS")).toBeNull();
    const dark = { h: 1, w: 1, c: 4, bands: rec.bands, data: Float32Array.from([-1, 0, 0, 0]) };   // (sums are cached per cube)
    expect(magInfo(dark, G.color_meta, "VIS")?.mag).toBeNull();
    expect(magLabel(magInfo(rec, G.color_meta, "VIS"))).toMatch(/^ · VIS \d+\.\d\d AB$/);
    // NISP bands read as the bar's chips do ("Y", not "Y_E")
    expect(magLabel(magInfo(rec, G.color_meta, "Y_E"))).toMatch(/^ · Y \d+\.\d\d AB$/);
  });
  it("only electron tiers get a magnitude (ADU/s, MJy/sr, arb. do not)", () => {
    const mk = (unit: string) => ({ h: 1, w: 1, c: 4, bands: ["VIS", "Y_E", "J_E", "H_E"], unit, data: Float32Array.from([100, 1, 1, 1]) });
    expect(magInfo(mk("e-"), G.color_meta, "VIS")?.mag).not.toBeNull();
    expect(magInfo(mk(""), G.color_meta, "VIS")?.mag).not.toBeNull();
    for (const u of ["ADU/s", "MJy/sr", "arb"]) expect(magInfo(mk(u), G.color_meta, "VIS")).toBeNull();
  });
});

describe("pixel readout", () => {
  const rec = { h: 2, w: 3, c: 2, data: Float32Array.from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) };
  it("values of every band at a pixel; null outside", () => {
    expect(pixelValues(rec, 1, 1)).toEqual([8, 9]);
    expect(pixelValues(rec, 3, 0)).toBeNull();
    expect(pixelValues(rec, -1, 0)).toBeNull();
  });
  it("continuous image coordinates → integer pixel", () => {
    expect(pixelAt(rec, 2.99, 0.01)).toEqual({ x: 2, y: 0 });
    expect(pixelAt(rec, 3, 0)).toBeNull();
  });
  it("units and value formatting", () => {
    expect(unitLabel("e-")).toBe("e⁻");
    expect(unitLabel("")).toBe("");
    expect(unitLabel("MJy/sr")).toBe("MJy/sr");
    expect(unitLabel("arb")).toBe("arb.");
    expect(formatValue(1234.5678)).toBe("1234.6");
    expect(formatValue(0.000123)).toBe("1.23e-4");
    expect(formatValue(NaN)).toBe("NaN");
    expect(formatValue(-42)).toBe("−42.0");
  });
});

describe("cubeIsEmpty", () => {
  it("is true only when no pixel of any band is finite", () => {
    expect(cubeIsEmpty({ data: Float32Array.from([NaN, NaN, Infinity, -Infinity]) })).toBe(true);
    expect(cubeIsEmpty({ data: Float32Array.from([NaN, NaN, 0, NaN]) })).toBe(false);   // partial NaNs keep the NaN colour
    expect(cubeIsEmpty({ data: new Float32Array(0) })).toBe(false);
    expect(cubeIsEmpty(null)).toBe(false);
  });
  it("is cached per data array", () => {
    const data = Float32Array.from([NaN, NaN]);
    expect(cubeIsEmpty({ data })).toBe(true);
    data[0] = 1;   // (the cache keeps the first answer: served cubes are immutable)
    expect(cubeIsEmpty({ data })).toBe(true);
  });
});

describe("readoutLineWidth (two reserved lines when one would cut values off)", () => {
  const w7 = (t: string) => t.length * 7;
  it("grows with every tier and fits three short tiers in a wide viewer", () => {
    const one = readoutLineWidth([{ name: "LR", unit: "e-" }], w7);
    const three = readoutLineWidth([{ name: "LR", unit: "e-" }, { name: "JWST", unit: "MJy/sr" }, { name: "RBF", unit: "e-" }], w7);
    expect(three).toBeGreaterThan(one);
    expect(three).toBeLessThan(760);                   // one line in a 1024-px viewer
    expect(three).toBeGreaterThan(330);                // two lines in the ~350 px inspector
  });
  it("leaves the sky part out when the collection has no positions", () => {
    const withSky = readoutLineWidth([{ name: "LR", unit: "e-" }], w7);
    expect(readoutLineWidth([{ name: "LR", unit: "e-" }], w7, { hasSky: false })).toBeLessThan(withSky);
  });
});
