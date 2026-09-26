/* The TS colour core against the golden values of the pre-rework engine
 * (static/cutout_viewer.js, run in Node) and visualization/color.py.
 * Regenerate the fixture with scripts/_viewer_parity_ref.py --write-golden. */
import { describe, expect, it } from "vitest";
import golden from "./__fixtures__/color_golden.json";
import {
  abFluxNorm, asinhTransfer, eyeTGrid, planckFnu, planckianXY, prepareCore, renderCubeImageData,
  solarBalance, srgbGamma, transferCore, xyToLinearSrgb, type ColorMeta, type CubeLike,
} from "./color";

type Special = number | string;
const decode = (v: Special): number =>
  v === "NaN" ? NaN : v === "Infinity" ? Infinity : v === "-Infinity" ? -Infinity : Number(v);

type GoldenCase = {
  name: string;
  rec: { h: number; w: number; c: number; bands?: string[]; displayScale?: number; directRgb?: boolean };
  data: Special[];
  color: string;
  render_mode: string | null;
  transfers: [number, number, number][];
};
type EngineCase = {
  name: string;
  prepared: Record<string, unknown> & { mode: string; factor: number };
  renders: { knee: number; gain: number; K0: number; rgba: number[] }[];
};

const G = golden as unknown as {
  color_meta: ColorMeta;
  cases: GoldenCase[];
  engine: {
    primitives: {
      abFluxNorm: Record<string, number>; solarBalance: Record<string, number>;
      planck: { T: number; xy: number[]; lin: number[]; srgb: number[] }[];
      srgbGamma: [number, number][]; eyeTGrid96: number[];
      planckFnu: { T: number; lam: number[]; f: number[] }[];
      asinhTransfer: [number, number, number, number, number][];
    };
    cases: EngineCase[];
  };
  python: {
    bands: Record<string, { ab_flux_norm: number; solar_balance: number }>;
    planck_chain: { T: number; srgb: number[] }[];
    eye_rgb: { asinh_scale_e: number; cube: number[][]; rgb: number[][] };
  };
};

const recOf = (c: GoldenCase): CubeLike => ({ key: c.name, ...c.rec, data: Float32Array.from(c.data.map(decode)) });
const close = (got: number, exp: number, rel = 1e-12) =>
  got === exp || (Number.isNaN(exp) ? Number.isNaN(got) : Math.abs(got - exp) <= rel * Math.max(1, Math.abs(exp)));

describe("colour primitives match the old engine", () => {
  const P = G.engine.primitives;
  const meta = G.color_meta;
  it("AB-flux normalisation and solar balance", () => {
    for (const [n, v] of Object.entries(P.abFluxNorm)) expect(close(abFluxNorm(meta.bands[n]), v)).toBe(true);
    for (const [n, v] of Object.entries(P.solarBalance)) expect(close(solarBalance(meta.bands[n]), v)).toBe(true);
  });
  it("Planckian locus → linear sRGB → gamma", () => {
    for (const row of P.planck) {
      const xy = planckianXY(row.T);
      const lin = xyToLinearSrgb(xy[0], xy[1]);
      xy.forEach((v, k) => expect(close(v, row.xy[k])).toBe(true));
      lin.forEach((v, k) => expect(close(v, row.lin[k])).toBe(true));
      lin.map(srgbGamma).forEach((v, k) => expect(close(v, row.srgb[k])).toBe(true));
    }
    for (const [c, v] of P.srgbGamma) expect(close(srgbGamma(c), v)).toBe(true);
  });
  it("temperature grid, Planck f_ν and the asinh transfer", () => {
    Array.from(eyeTGrid(96)).forEach((v, i) => expect(close(v, P.eyeTGrid96[i])).toBe(true));
    for (const row of P.planckFnu) Array.from(planckFnu(row.lam, row.T)).forEach((v, i) => expect(close(v, row.f[i])).toBe(true));
    for (const [I, g, k, n, v] of P.asinhTransfer) expect(close(asinhTransfer(I, g, k, n), v)).toBe(true);
  });
});

describe("prepareCore + transferCore match the old engine in every colour mode", () => {
  const byName = new Map(G.engine.cases.map((c) => [c.name, c]));
  for (const c of G.cases) {
    it(c.name, () => {
      const exp = byName.get(c.name)!;
      const meta = c.render_mode ? { ...G.color_meta, render_mode: c.render_mode } : G.color_meta;
      const prep = prepareCore(recOf(c), meta, c.color);
      expect(prep.mode).toBe(exp.prepared.mode);
      expect(close(prep.factor, exp.prepared.factor)).toBe(true);
      for (const k of ["I", "R", "G", "B", "hueR", "hueG", "hueB"] as const) {
        const want = exp.prepared[k] as Special[] | undefined;
        const got = (prep as unknown as Record<string, Float32Array | undefined>)[k];
        if (!want) { expect(got).toBeUndefined(); continue; }
        expect(got!.length).toBe(want.length);
        // Float32 planes: bit-identical to the old engine's.
        want.forEach((v, i) => expect(close(got![i], decode(v), 0)).toBe(true));
      }
      if (prep.mode === "gray-log") {
        expect(close(prep.logLo!, exp.prepared.logLo as number)).toBe(true);
        expect(close(prep.logHi!, exp.prepared.logHi as number)).toBe(true);
      }
      for (const r of exp.renders) {
        const img = transferCore(prep, r.knee, r.gain, r.K0);
        expect(Array.from(img.data)).toEqual(r.rgba);
      }
    });
  }
});

describe("parity with visualization/color.py", () => {
  it("calibration constants", () => {
    for (const [name, v] of Object.entries(G.python.bands)) {
      expect(abFluxNorm(G.color_meta.bands[name]) / v.ab_flux_norm).toBeCloseTo(1, 6);
      expect(solarBalance(G.color_meta.bands[name]) / v.solar_balance).toBeCloseTo(1, 6);
    }
  });
  it("the Planckian hue chain", () => {
    for (const row of G.python.planck_chain) {
      const [x, y] = planckianXY(row.T);
      xyToLinearSrgb(x, y).map(srgbGamma).forEach((v, k) => expect(Math.abs(v - row.srgb[k])).toBeLessThan(1e-4));
    }
  });
  it("Temp mode at the default knee is eye_rgb", () => {
    const e = G.python.eye_rgb;
    const rec: CubeLike = { h: 1, w: e.cube.length, c: 4, bands: ["VIS", "Y_E", "J_E", "H_E"], data: Float32Array.from(e.cube.flat()) };
    const img = transferCore(prepareCore(rec, G.color_meta, "temp"), e.asinh_scale_e, 1, e.asinh_scale_e);
    e.rgb.forEach((px, i) => px.forEach((v, k) => expect(Math.abs(img.data[i * 4 + k] / 255 - v)).toBeLessThan(2e-3 + 0.5 / 255)));
  });
});

describe("renderCubeImageData (the back-trace stamps' entry point)", () => {
  it("equals prepare + transfer with the viewer defaults", () => {
    const c = G.cases.find((x) => x.name === "lupton")!;
    const exp = G.engine.cases.find((x) => x.name === "lupton")!.renders[0];
    const img = renderCubeImageData(recOf(c), G.color_meta, { color: "lupton", knee: 100, gain: 1, K0: 100 });
    expect(Array.from(img.data)).toEqual(exp.rgba);
  });
  it("falls back to knee 100, gain 1 and K0 = knee", () => {
    const c = G.cases.find((x) => x.name === "gray-VIS")!;
    const a = renderCubeImageData(recOf(c), G.color_meta, {});
    const b = transferCore(prepareCore(recOf(c), G.color_meta, "VIS"), 100, 1, 100);
    expect(Array.from(a.data)).toEqual(Array.from(b.data));
  });
});
