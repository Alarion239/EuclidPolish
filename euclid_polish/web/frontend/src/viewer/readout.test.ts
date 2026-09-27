import { describe, expect, it } from "vitest";
import golden from "./__fixtures__/color_golden.json";
import type { ColorMeta } from "./color";
import { EMPTY_MAX_FINITE, MIN_COVERAGE, READOUT_WRAP_WIDTH, cubeCoverage, cubeIsEmpty, cubeIsSparse, formatValue, readoutLineWidth, readoutLines, magInfo, magLabel, pixelAt, pixelValues, sigmaMagnitude, unitLabel } from "./readout";

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
  it("counts the finite values (cubeCoverage)", () => {
    expect(cubeCoverage({ data: Float32Array.from([NaN, 1, 2, NaN]) })).toEqual({ finite: 2, total: 4, fraction: 0.5 });
    expect(cubeCoverage(null)).toEqual({ finite: 0, total: 0, fraction: 1 });
  });
  it("is true when (almost) no pixel has data: a JWST cutout outside the mosaic", () => {
    expect(MIN_COVERAGE).toBe(0.01);
    // the NEXUS tile f200w-0000: 58 finite of 850 × 850 JWST pixels
    const d = new Float32Array(850 * 850).fill(NaN);
    for (let i = 0; i < 58; i++) d[i * 997] = 1;
    expect(cubeIsEmpty({ data: d })).toBe(true);
    expect(cubeCoverage({ data: d }).finite).toBe(58);
    // a partly covered cutout keeps its image (and the NaN colour for the rest)
    const half = new Float32Array(100).fill(NaN).fill(3, 0, 50);
    expect(cubeIsEmpty({ data: half })).toBe(false);
  });
  it("a corner of real data (≥ EMPTY_MAX_FINITE values) is sparse, not empty: it is painted", () => {
    expect(EMPTY_MAX_FINITE).toBe(1000);
    // an 850 × 850 JWST cutout overlapping the mosaic in an 80 × 80 corner: 0.9 %
    const d = new Float32Array(850 * 850).fill(NaN);
    for (let y = 0; y < 80; y++) for (let x = 0; x < 80; x++) d[y * 850 + x] = 1;
    expect(cubeCoverage({ data: d }).fraction).toBeLessThan(MIN_COVERAGE);
    expect(cubeIsEmpty({ data: d })).toBe(false);
    expect(cubeIsSparse({ data: d })).toBe(true);
    // 58 scattered values are empty, not sparse; a half-covered cutout is neither
    const few = new Float32Array(850 * 850).fill(NaN);
    for (let i = 0; i < 58; i++) few[i * 997] = 1;
    expect(cubeIsSparse({ data: few })).toBe(false);
    expect(cubeIsSparse({ data: new Float32Array(100).fill(NaN).fill(3, 0, 50) })).toBe(false);
    expect(cubeIsSparse(null)).toBe(false);
  });
  it("is true only when no pixel of any band is finite (or below 1 %)", () => {
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

describe("readoutLines (the lines the readout reserves, before any hover)", () => {
  const w7 = (t: string) => t.length * 7;
  const three = [{ name: "LR", unit: "e-" }, { name: "JWST", unit: "MJy/sr" }, { name: "RBF", unit: "e-" }];
  it("one line in a wide viewer that fits it", () => {
    expect(readoutLines([{ name: "LR", unit: "e-" }], w7, { width: 900 })).toBe(1);
    expect(readoutLines(three, w7, { width: 900 })).toBe(1);
  });
  it("two lines below ~560 px of viewer width even when one would fit", () => {
    expect(READOUT_WRAP_WIDTH).toBe(560);
    expect(readoutLines([{ name: "LR", unit: "e-" }], w7, { width: 540 })).toBe(2);
  });
  it("wraps between tiers, never inside one, and reserves every line it needs (the 258 px tile inspector)", () => {
    // the review's case: 581 px of readout in 258 px → position, then the values over two lines
    const n = readoutLines(three, w7, { width: 258 });
    expect(n).toBeGreaterThanOrEqual(3);
    // each line holds whole parts: the position alone on a narrow line, the values packed after it
    expect(readoutLines(three, w7, { width: 400 })).toBe(3);
    expect(readoutLines(three, w7, { width: 520 })).toBe(2);
  });
  it("a collection without positions saves the sky part", () => {
    expect(readoutLines([{ name: "LR", unit: "e-" }], w7, { width: 300, hasSky: false })).toBe(2);
  });
  it("SR's magnitude ± σ (with a std tier) is counted", () => {
    const sr = [{ name: "LR", unit: "e-" }, { name: "SR", unit: "e-", sigma: true }];
    expect(readoutLineWidth(sr, w7)).toBeGreaterThan(readoutLineWidth([{ name: "LR", unit: "e-" }, { name: "SR", unit: "e-" }], w7));
  });
  it("never more than four lines", () => {
    const many = Array.from({ length: 12 }, (_, i) => ({ name: `M${i}`, unit: "e-" }));
    expect(readoutLines(many, w7, { width: 200 })).toBe(4);
  });
});
