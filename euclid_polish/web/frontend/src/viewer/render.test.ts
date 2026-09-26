import { describe, expect, it } from "vitest";
import golden from "./__fixtures__/color_golden.json";
import { prepareCore, transferCore, type ColorMeta, type CubeLike } from "./color";
import { COLORMAP_LUTS, colormapLut, parseCssColor } from "./colormaps";
import { isDefaultDisplay, prepareColor, renderPrepared, renderSigned, type DisplayParams } from "./render";

const meta = (golden as unknown as { color_meta: ColorMeta }).color_meta;
const BANDS = ["VIS", "Y_E", "J_E", "H_E"];

const cube = (values: number[], c = 1, bands = c === 1 ? ["VIS"] : BANDS): CubeLike => ({
  data: Float32Array.from(values), h: 1, w: values.length / c, c, bands,
});

const base: DisplayParams = {
  stretch: "asinh-abs", knee: 100, gain: 1, black: 0, K0: 100, colormap: "gray", invert: false, nanColor: [255, 0, 255],
};

describe("the default display is the locked transfer, bit for bit", () => {
  it("renderPrepared = transferCore when nothing is changed", () => {
    for (const color of ["VIS", "lupton", "temp"]) {
      const rec = cube([10, 100, 1000, 5, 20, 200, 2000, 4, 30, 300, 3000, 1, 40, 400, 4000, 0], 4);
      const prep = prepareCore(rec, meta, color);
      expect(isDefaultDisplay(base)).toBe(true);
      expect(Array.from(renderPrepared(prep, base).data)).toEqual(Array.from(transferCore(prep, 100, 1, 100).data));
    }
  });
  it("NaN pixels get the NaN colour (the only difference from transferCore)", () => {
    const prep = prepareCore(cube([NaN, 50, 5000]), meta, "VIS");
    const img = renderPrepared(prep, base).data;
    expect(Array.from(img.slice(0, 4))).toEqual([255, 0, 255, 255]);
    expect(Array.from(img.slice(4))).toEqual(Array.from(transferCore(prep, 100, 1, 100).data.slice(4)));
  });
});

describe("opt-in stretches, colormaps, black point, invert", () => {
  const prep = prepareCore(cube([0, 300, 1500, 3000, 6000]), meta, "VIS");
  const gray = (p: Partial<DisplayParams>) => Array.from(renderPrepared(prep, { ...base, ...p }).data).filter((_, i) => i % 4 === 0);
  it("linear / sqrt / log are anchored at the same white (30·K0 / gain)", () => {
    expect(gray({ stretch: "linear" })).toEqual([0, 25, 127, 255, 255]);
    expect(gray({ stretch: "sqrt" })[1]).toBe(Math.floor(Math.sqrt(0.1) * 255));
    const log = gray({ stretch: "log" });
    expect(log[0]).toBe(0);
    expect(log[3]).toBe(255);
    expect(log[1]).toBeGreaterThan(gray({ stretch: "linear" })[1]);
  });
  it("the black point is subtracted before the transfer", () => {
    expect(gray({ stretch: "linear", black: 300 })).toEqual([0, 0, 102, 229, 255]);
  });
  it("invert flips the luminance", () => {
    const a = gray({ stretch: "linear" }), b = gray({ stretch: "linear", invert: true });
    a.forEach((v, i) => expect(b[i]).toBe(255 - v));
  });
  it("auto stretches do not depend on the knee and span the data", () => {
    const auto = gray({ stretch: "zscale" });
    expect(auto[0]).toBe(0);
    expect(auto[4]).toBe(255);
    expect(gray({ stretch: "zscale", knee: 7 })).toEqual(auto);
    expect(gray({ stretch: "asinh-auto" })[4]).toBe(255);
  });
  it("a colormap maps the gray luminance through its LUT", () => {
    const img = renderPrepared(prep, { ...base, colormap: "viridis", stretch: "linear" }).data;
    const lut = colormapLut("viridis");
    expect(Array.from(img.slice(0, 3))).toEqual(Array.from(lut.slice(0, 3)));          // t = 0
    expect(Array.from(img.slice(12, 15))).toEqual(Array.from(lut.slice(255 * 3, 255 * 3 + 3))); // t = 1
  });
  it("colour composites ignore the colormap but follow the stretch", () => {
    const rec = cube([30, 30, 30, 30, 200, 150, 120, 100], 4);   // unsaturated in both stretches
    const lup = prepareCore(rec, meta, "lupton");
    const a = renderPrepared(lup, { ...base, colormap: "magma" }).data;
    expect(Array.from(a)).toEqual(Array.from(transferCore(lup, 100, 1, 100).data));
    const lin = renderPrepared(lup, { ...base, stretch: "linear" }).data;
    expect(Array.from(lin)).not.toEqual(Array.from(a));
  });
});

describe("signed (residual) rendering", () => {
  it("zero is the colormap centre, sign picks the side, NaN the NaN colour", () => {
    const img = renderSigned(Float32Array.from([0, 1e6, -1e6, NaN]), 4, 1, { ...base, colormap: "rdbu", scale: "asinh" }).data;
    const lut = colormapLut("rdbu");
    const at = (k: number) => Array.from(lut.slice(k * 3, k * 3 + 3));
    expect(Array.from(img.slice(0, 3))).toEqual(at(128));
    expect(Array.from(img.slice(4, 7))).toEqual(at(255));
    expect(Array.from(img.slice(8, 11))).toEqual(at(0));
    expect(Array.from(img.slice(12, 15))).toEqual([255, 0, 255]);
  });
  it("a linear scale saturates at ±range", () => {
    const img = renderSigned(Float32Array.from([5, -2.5]), 2, 1, { ...base, colormap: "gray", scale: "linear", range: 5 }).data;
    expect(img[0]).toBe(255);
    expect(img[4]).toBe(Math.round(0.25 * 255));   // signed maps round, so 0 is exactly the centre (128)
  });
});

describe("colour mode resolution", () => {
  it("custom RGB is a Lupton composite over the chosen bands; native is the first channel", () => {
    expect(prepareColor("rgb", ["J_E", "Y_E", "VIS"])).toEqual({ color: "lupton", scheme: ["J_E", "Y_E", "VIS"] });
    expect(prepareColor("native", ["H_E", "J_E", "VIS"])).toEqual({ color: "native", scheme: undefined });
    expect(prepareColor("temp", ["H_E", "J_E", "VIS"])).toEqual({ color: "temp", scheme: undefined });
    const rec = cube([1, 2, 3, 4], 4);
    const p = prepareCore(rec, meta, "lupton", ["J_E", "Y_E", "VIS"]);
    const q = prepareCore(rec, meta, "lupton");
    expect(Array.from(p.R!)).not.toEqual(Array.from(q.R!));
  });
});

describe("colormaps", () => {
  it("every LUT has 256 RGB entries and the matplotlib end points", () => {
    for (const name of Object.keys(COLORMAP_LUTS)) expect(colormapLut(name as never).length).toBe(768);
    expect(Array.from(colormapLut("viridis").slice(0, 3))).toEqual([0x44, 0x01, 0x54]);
    expect(Array.from(colormapLut("viridis").slice(765))).toEqual([0xfd, 0xe7, 0x25]);
    expect(Array.from(colormapLut("gray").slice(381, 384))).toEqual([127, 127, 127]);
  });
  it("parses CSS hex colours (fallback magenta)", () => {
    expect(parseCssColor("#ff00ff")).toEqual([255, 0, 255]);
    expect(parseCssColor("#0a0")).toEqual([0, 170, 0]);
    expect(parseCssColor("nonsense")).toEqual([255, 0, 255]);
  });
});
