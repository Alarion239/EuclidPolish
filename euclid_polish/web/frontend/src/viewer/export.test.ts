import { describe, expect, it } from "vitest";
import { colormapLut } from "./colormaps";
import {
  heatbarModel, heatbarStops, niceAngularScale, publicationElectronLabel, publicationLayout, publicationPanelName,
  publicationUnitLabel, exportStem,
} from "./export";

describe("publication heat bar (unit-aware, WP-B1b handoff)", () => {
  it("unit labels: e⁻ by default, arb. units, native units verbatim", () => {
    expect(publicationUnitLabel("")).toBe("e⁻");
    expect(publicationUnitLabel("e-")).toBe("e⁻");
    expect(publicationUnitLabel("electron")).toBe("e⁻");
    expect(publicationUnitLabel("arb")).toBe("arb. units");
    expect(publicationUnitLabel("MJy/sr")).toBe("MJy/sr");
  });
  it("tick labels: k above 1000, exponent notation for small values", () => {
    expect(publicationElectronLabel(12345)).toBe("12k");
    expect(publicationElectronLabel(1500)).toBe("1.5k");
    expect(publicationElectronLabel(42.4)).toBe("42");
    expect(publicationElectronLabel(3.14)).toBe("3.1");
    expect(publicationElectronLabel(0.5)).toBe("0.50");
    expect(publicationElectronLabel(0.0012)).toBe("1.2e-3");
    expect(publicationElectronLabel(0)).toBe("0.00");
  });
  it("an electron panel reads the knee in e⁻ and ticks sinh(f·norm)·knee/gain", () => {
    const m = heatbarModel({ band: "VIS", knee: 100, gain: 1, log: false, unit: "e⁻", scale: 1 }, 100);
    expect(m.parameterText).toBe("Band: VIS  ·  asinh knee: 100 e⁻");
    expect(m.signalLabel).toBe("Pixel signal (e⁻)");
    expect(m.ticks.map((t) => t.fraction)).toEqual([0, 0.25, 0.5, 0.75, 1]);
    expect(m.ticks[4].label).toBe("3.0k");     // white = 30·K0
    expect(m.ticks[0].label).toBe("0.00");
  });
  it("a JWST panel divides the knee and ticks by the display scale and names MJy/sr", () => {
    const scale = 17743.00471101362;
    const m = heatbarModel({ band: "F200W", knee: 100, gain: 1, log: false, unit: "MJy/sr", scale }, 100);
    expect(m.parameterText).toBe(`Band: F200W  ·  asinh knee: ${publicationElectronLabel(100 / scale)} MJy/sr`);
    expect(m.signalLabel).toBe("Pixel signal (MJy/sr)");
    expect(m.ticks[4].label).toBe(publicationElectronLabel(3000 / scale));
    expect(m.ticks[4].label).toBe("0.17");
  });
  it("a log (PSF) panel is a relative intensity bar", () => {
    const m = heatbarModel({ band: "VIS", knee: 100, gain: 1, log: true, unit: "arb. units", scale: 1 }, 100);
    expect(m.parameterText).toBe("Band: VIS  ·  logarithmic display");
    expect(m.signalLabel).toBe("relative display intensity");
    expect(m.ticks[2].label).toBe("0.50");
  });
});

describe("heat bar follows the display settings", () => {
  const base = { band: "VIS", knee: 100, gain: 1, log: false, unit: "e⁻", scale: 1 } as const;
  const labels = (m: { ticks: { label: string }[] }) => m.ticks.map((t) => t.label);

  it("the locked default (asinh-abs, black 0) is unchanged", () => {
    const m = heatbarModel({ ...base, stretch: "asinh-abs", black: 0 }, 100);
    expect(m).toEqual(heatbarModel(base, 100));
  });
  it("linear, sqrt and log run from black to black + 30·K0/gain", () => {
    const lin = heatbarModel({ ...base, stretch: "linear" }, 100);
    expect(labels(lin)).toEqual(["0.00", "750", "1.5k", "2.3k", "3.0k"]);
    expect(lin.parameterText).toBe("Band: VIS  ·  linear stretch: 0.00 – 3.0k e⁻");
    const sq = heatbarModel({ ...base, stretch: "sqrt" }, 100);
    expect(labels(sq)).toEqual(["0.00", "188", "750", "1.7k", "3.0k"]);
    expect(sq.parameterText).toBe("Band: VIS  ·  sqrt stretch: 0.00 – 3.0k e⁻");
    const lg = heatbarModel({ ...base, stretch: "log" }, 100);
    const x = (f: number) => ((Math.pow(1001, f) - 1) / 1000) * 3000;
    expect(labels(lg)).toEqual([0, 0.25, 0.5, 0.75, 1].map((f) => publicationElectronLabel(x(f))));
    expect(lg.parameterText).toBe("Band: VIS  ·  log stretch: 0.00 – 3.0k e⁻");
    // black point and gain move both ends: [50, 50 + 3000/2]
    const moved = heatbarModel({ ...base, stretch: "linear", black: 50, gain: 2 }, 100);
    expect(labels(moved)[0]).toBe("50");
    expect(labels(moved)[4]).toBe("1.6k");
  });
  it("asinh with a black point offsets every tick and names it", () => {
    const m = heatbarModel({ ...base, black: 50 }, 100);
    const plain = heatbarModel(base, 100);
    const norm = Math.asinh(30);
    expect(labels(m)).toEqual([0, 0.25, 0.5, 0.75, 1].map((f) => publicationElectronLabel(50 + Math.sinh(f * norm) * 100)));
    expect(labels(m)).not.toEqual(labels(plain));
    expect(m.parameterText).toBe("Band: VIS  ·  asinh knee: 100 e⁻  ·  black 50 e⁻");
  });
  it("asinh-auto and zscale use the frame's own limits", () => {
    const auto = heatbarModel({ ...base, stretch: "asinh-auto", auto: { lo: 10, hi: 1000, knee: 30 } }, 100);
    const den = Math.asinh(990 / 30);
    expect(labels(auto)).toEqual([0, 0.25, 0.5, 0.75, 1].map((f) => publicationElectronLabel(10 + Math.sinh(f * den) * 30)));
    expect(labels(auto)[4]).toBe("1.0k");
    expect(auto.parameterText).toBe("Band: VIS  ·  asinh (auto limits): 10 – 1.0k e⁻");
    const z = heatbarModel({ ...base, stretch: "zscale", auto: { lo: -5, hi: 45 } }, 100);
    expect(labels(z)).toEqual(["−5.0", "7.5", "20", "33", "45"]);
    expect(z.parameterText).toBe("Band: VIS  ·  zscale: −5.0 – 45 e⁻");
  });
  it("a JWST panel divides every stretch's ticks by the display scale", () => {
    const m = heatbarModel({ ...base, band: "F200W", unit: "MJy/sr", scale: 10000, stretch: "linear" }, 100);
    expect(labels(m)).toEqual(["0.00", "0.07", "0.15", "0.23", "0.30"]);   // 0.075 is 0.07499… in binary
    expect(m.signalLabel).toBe("Pixel signal (MJy/sr)");
  });
  it("the bar's gradient is the frame's colormap (reversed when inverted); composites stay luminance", () => {
    const lut = colormapLut("viridis");
    const rgb = (k: number) => `rgb(${lut[k * 3]}, ${lut[k * 3 + 1]}, ${lut[k * 3 + 2]})`;
    const v = heatbarStops("viridis", false, "gray");
    expect(v[0]).toBe(rgb(0));
    expect(v[v.length - 1]).toBe(rgb(255));
    expect(heatbarStops("viridis", true, "gray")).toEqual([...v].reverse());
    expect(heatbarStops("magma", false, "gray-log")[0]).toBe(heatbarStops("magma", false, "gray")[0]);
    const lup = heatbarStops("viridis", false, "lupton");
    expect([lup[0], lup[lup.length - 1]]).toEqual(["rgb(0, 0, 0)", "rgb(255, 255, 255)"]);
    const inv = heatbarStops("gray", true, "temp");
    expect([inv[0], inv[inv.length - 1]]).toEqual(["rgb(255, 255, 255)", "rgb(0, 0, 0)"]);
  });
});

describe("residual (signed) heat bar", () => {
  it("ticks run −range … 0 … +range with the residual's own scale and unit", () => {
    const lin = heatbarModel({ band: "VIS", knee: 100, gain: 1, log: false, unit: "σ", scale: 1,
      signed: { scale: "linear", knee: 1, range: 5, label: "(SR − HR) / σ", stops: ["#00f", "#fff", "#f00"] } }, 100);
    expect(lin.ticks.map((t) => t.label)).toEqual(["−5.0", "−2.5", "0.00", "2.5", "5.0"]);
    expect(lin.parameterText).toBe("Band: VIS  ·  (SR − HR) / σ  ·  linear ±5.0 σ");
    expect(lin.signalLabel).toBe("Residual (σ)");
    const as = heatbarModel({ band: "J_E", knee: 100, gain: 1, log: false, unit: "e⁻", scale: 1,
      signed: { scale: "asinh", knee: 2, range: 40, label: "LR − HR", stops: [] } }, 100);
    expect(as.ticks[2].label).toBe("0.00");
    expect(as.ticks[4].label).toBe("40");
    expect(as.ticks[0].label).toBe("−40");
    expect(as.ticks[3].label).toBe(publicationElectronLabel(2 * Math.sinh(0.5 * Math.asinh(20))));
    expect(as.parameterText).toBe("Band: J_E  ·  LR − HR  ·  asinh, knee 2.0 e⁻");
    expect(as.signalLabel).toBe("Residual (e⁻)");
  });
});

describe("figure layout and names", () => {
  it("panel titles", () => {
    expect(publicationPanelName("lr", "LR · Euclid")).toBe("Euclid Image");
    expect(publicationPanelName("dirty", "LR")).toBe("Euclid Image");
    expect(publicationPanelName("sr", "SR · production gate")).toBe("Super-resolved Image");
    expect(publicationPanelName("jwst", "NEXUS F200W · native")).toBe("NEXUS F200W · native");
  });
  it("scale bar: the largest nice value ≤ 20% of the side", () => {
    expect(niceAngularScale(25.6)).toBe(5);
    expect(niceAngularScale(2.1)).toBe(0.2);
    expect(niceAngularScale(0)).toBeNull();
  });
  it("plate geometry: one row, or ceil(n/2) columns in two rows", () => {
    const one = publicationLayout(3, "one-row");
    expect(one).toMatchObject({ columns: 3, rows: 1, side: 1260 });
    expect(one.width).toBe(2 * 28 + 3 * 1260 + 2 * 14);
    const two = publicationLayout(3, "two-rows");
    expect(two).toMatchObject({ columns: 2, rows: 2 });
    expect(publicationLayout(8, "one-row").side).toBe(Math.max(640, Math.floor((4800 - 56 - 14 * 7) / 8)));
  });
  it("file stem", () => {
    expect(exportStem("nexus-field", 12, ["lr", "sr"], "VIS")).toBe("nexus-field_idx12_lr-sr_VIS");
  });
});
