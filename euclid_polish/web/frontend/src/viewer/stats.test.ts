import { describe, expect, it } from "vitest";
import { histogram, lineProfile, percentile, radialProfile, regionValues, robustStats, zscale } from "./stats";

describe("robust statistics", () => {
  it("percentiles interpolate a sorted sample", () => {
    const s = Float64Array.from([0, 1, 2, 3, 4]);
    expect(percentile(s, 0)).toBe(0);
    expect(percentile(s, 50)).toBe(2);
    expect(percentile(s, 100)).toBe(4);
    expect(percentile(s, 12.5)).toBe(0.5);
  });
  it("median and MAD σ ignore NaN and ±∞", () => {
    const st = robustStats(Float32Array.from([1, 2, 3, 4, 100, NaN, Infinity, -Infinity]));
    expect(st.n).toBe(5);
    expect(st.median).toBe(3);
    expect(st.sigma).toBeCloseTo(1.4826 * 1, 6);
    expect(st.min).toBe(1);
    expect(st.max).toBe(100);
  });
  it("an empty sample has no statistics", () => {
    expect(robustStats(Float32Array.from([NaN])).n).toBe(0);
  });
  it("zscale brackets the bulk of a noisy flat image and ignores bright outliers", () => {
    const v = new Float32Array(2000);
    for (let i = 0; i < v.length; i++) v[i] = 100 + ((i * 7919) % 97) / 97 - 0.5;
    v[3] = 1e6; v[9] = 5e5;
    const [z1, z2] = zscale(v);
    expect(z1).toBeGreaterThan(99);
    expect(z2).toBeLessThan(101);
    expect(z2).toBeGreaterThan(z1);
  });
});

describe("histogram", () => {
  it("counts finite values into equal bins, clamping the upper edge", () => {
    const h = histogram(Float32Array.from([0, 0.1, 0.5, 0.99, 1, 2, NaN]), 0, 1, 4);
    expect(h.counts).toEqual([2, 0, 1, 2]);
    expect(h.edges).toEqual([0, 0.25, 0.5, 0.75, 1]);
    expect(h.outside).toBe(1);
  });
});

describe("region and profiles on a cube band", () => {
  // 4×4 two-band cube: band 0 = x + 10·y, band 1 = 1
  const w = 4, h = 4, c = 2;
  const data = new Float32Array(w * h * c);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { data[(y * w + x) * c] = x + 10 * y; data[(y * w + x) * c + 1] = 1; }
  const rec = { data, w, h, c };
  it("region values of a crop", () => {
    expect(Array.from(regionValues(rec, 0, { x: 1, y: 2, w: 2, h: 1 }))).toEqual([21, 22]);
    expect(Array.from(regionValues(rec, 1, { x: -5, y: -5, w: 1, h: 1 }))).toEqual([]);
  });
  it("a line profile samples every pixel step along the segment", () => {
    const p = lineProfile(rec, 0, { x: 0.5, y: 0.5 }, { x: 3.5, y: 0.5 });
    expect(p.d).toEqual([0, 1, 2, 3]);
    expect(p.v).toEqual([0, 1, 2, 3]);
    const q = lineProfile(rec, 0, { x: 0.5, y: 0.5 }, { x: 0.5, y: 3.5 });
    expect(q.v).toEqual([0, 10, 20, 30]);
  });
  it("a radial profile averages annuli around the centre", () => {
    const p = radialProfile(rec, 1, { x: 2, y: 2 }, 2, 1);
    expect(p.r).toEqual([0.5, 1.5]);
    expect(p.v).toEqual([1, 1]);
    expect(p.n[0]).toBe(4);   // the four pixels whose centres are 0.707 px away
  });
});
