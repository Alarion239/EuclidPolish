import { describe, expect, it } from "vitest";
import {
  MORPH_FRAMES, MovieStore, morphBaseTier, morphCoefficients, movieBytes, movieLabel, pcaCount, slotAt,
  synthesizeMorphFrame,
} from "./movie";

describe("the PCA morph", () => {
  it("coefficients: amp_k · morphAmp · sin(2π f_k φ + φ_k), f = 1,2,3, φ_k = 0, π/2, π/3", () => {
    const c = morphCoefficients([2, 3, 4, 5], 1.5, 0.125);
    expect(c[0]).toBeCloseTo(2 * 1.5 * Math.sin(2 * Math.PI * 0.125), 12);
    expect(c[1]).toBeCloseTo(3 * 1.5 * Math.sin(2 * Math.PI * 2 * 0.125 + Math.PI / 2), 12);
    expect(c[2]).toBeCloseTo(4 * 1.5 * Math.sin(2 * Math.PI * 3 * 0.125 + Math.PI / 3), 12);
    expect(c[3]).toBeCloseTo(5 * 1.5 * Math.sin(2 * Math.PI * 0.125), 12);   // k % 3
    expect(morphCoefficients([0, NaN as unknown as number], 1, 0.3)).toEqual([0, 0]);
  });
  it("a frame is the centre plus the weighted components (period 1)", () => {
    const base = Float32Array.from([1, 2, 3]);
    const comps = [Float32Array.from([1, 0, -1]), Float32Array.from([0, 1, 0])];
    const out = new Float32Array(3);
    synthesizeMorphFrame(base, comps, [0.5, -2], out);
    expect(Array.from(out)).toEqual([1.5, 0, 2.5]);
    const a = morphCoefficients([1, 1, 1], 1, 0.3), b = morphCoefficients([1, 1, 1], 1, 1.3);
    a.forEach((v, k) => expect(v).toBeCloseTo(b[k], 12));
  });
  it("slots cover [0, 1) with MORPH_FRAMES frames", () => {
    expect(slotAt(0)).toBe(0);
    expect(slotAt(0.999999)).toBe(MORPH_FRAMES - 1);
    expect(slotAt(3.5)).toBe(MORPH_FRAMES / 2);
    expect(slotAt(-0.25)).toBe(MORPH_FRAMES * 0.75);
  });
  it("PC count: all PCs of the full ensemble, members − 1 (≤ pca_max) for a subset", () => {
    expect(pcaCount({ pca_n: 3 }, null)).toBe(3);
    expect(pcaCount({ pca_n: 3, pca_max: 3 }, "0,1")).toBe(1);
    expect(pcaCount({ pca_max: 3 }, "0,1,2,3,4,5")).toBe(3);
    expect(pcaCount({}, null)).toBe(0);
  });
  it("centres on meta.morph_base_tier, falling back to sr", () => {
    expect(morphBaseTier({ morph_base_tier: "mean" })).toBe("mean");
    expect(morphBaseTier({})).toBe("sr");
    expect(morphBaseTier(null)).toBe("sr");
  });
  it("labels the subset and the variance", () => {
    expect(movieLabel(null, [{ varexp: 0.4 }, { varexp: 0.2 }])).toBe("disagreement movie · 2 PCs ≈ 60% of variance");
    expect(movieLabel("1,4,7", [])).toBe("disagreement movie · 3 members");
  });
});

describe("MovieStore (LRU with a byte budget)", () => {
  it("estimates bytes: gray keeps one plane, colour modes four", () => {
    expect(movieBytes(48, 10, 10, "gray")).toBe(48 * 100 * 4);
    expect(movieBytes(48, 10, 10, "temp")).toBe(48 * 100 * 16);
  });
  it("evicts the oldest entries but never the one playing", () => {
    const s = new MovieStore<{ bytes: number }>(100);
    s.set("a", { bytes: 60 });
    s.set("b", { bytes: 30 });
    s.playing = "a";
    s.set("c", { bytes: 30 });
    s.evict();
    expect(s.has("a")).toBe(true);        // pinned
    expect(s.has("b")).toBe(false);
    expect(s.has("c")).toBe(true);
    expect(s.totalBytes()).toBe(90);
    s.get("a");
    s.playing = null;
    s.set("d", { bytes: 50 });
    s.evict();
    expect(s.has("c")).toBe(false);       // LRU after the touch of a
    expect(s.has("a")).toBe(false);       // still over budget → a goes too
    expect(s.has("d")).toBe(true);
  });
});
