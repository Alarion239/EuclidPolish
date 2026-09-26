import { describe, expect, it } from "vitest";
import {
  decadeTicks,
  extent,
  formatTick,
  linearTicks,
  linearTickValues,
  logTicks,
  logTickValues,
  magnitudeTicks,
  niceStep,
  paddedDomain,
  unionDomain,
} from "./ticks";

const values = (t: { v: number }[]) => t.map((x) => x.v);
const labels = (t: { label: string }[]) => t.map((x) => x.label);

describe("niceStep", () => {
  it("picks 1-2-5 steps near span/count", () => {
    expect(niceStep(10, 5)).toBe(2);
    expect(niceStep(1, 5)).toBe(0.2);
    expect(niceStep(100, 4)).toBe(20);
    expect(niceStep(7, 5)).toBe(1);
    expect(niceStep(0.03, 5)).toBe(0.005);
    expect(niceStep(0, 5)).toBe(0);
  });
});

describe("linear ticks", () => {
  it("returns clean round values inside the domain", () => {
    expect(linearTickValues(0, 10)).toEqual([0, 2, 4, 6, 8, 10]);
    expect(linearTickValues(0.1, 0.9, 4)).toEqual([0.2, 0.4, 0.6, 0.8]);
    expect(linearTickValues(-1.3, 1.3)).toEqual([-1, -0.5, 0, 0.5, 1]);
    expect(linearTickValues(38.2, 44.9)).toEqual([39, 40, 41, 42, 43, 44]);
  });

  it("works on reversed domains and degenerate input", () => {
    expect(linearTickValues(10, 0)).toEqual([0, 2, 4, 6, 8, 10]);
    expect(linearTickValues(5, 5)).toEqual([5]);
    expect(linearTickValues(NaN, 1)).toEqual([]);
  });

  it("labels with the step's precision", () => {
    expect(labels(linearTicks([0, 1], { count: 4 }))).toEqual(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]);
    expect(labels(linearTicks([0, 0.05]))).toEqual(["0.00", "0.01", "0.02", "0.03", "0.04", "0.05"]);
    expect(labels(linearTicks([0, 10]))).toEqual(["0", "2", "4", "6", "8", "10"]);
    expect(labels(linearTicks([0, 30000], { count: 3 }))).toEqual(["0", "10,000", "20,000", "30,000"]);
    expect(labels(linearTicks([0, 1], { count: 4, format: (v) => `${Math.round(v * 100)}%` })))
      .toEqual(["0%", "20%", "40%", "60%", "80%", "100%"]);
  });

  it("formats individual ticks", () => {
    expect(formatTick(0.25, 0.25)).toBe("0.25");
    expect(formatTick(3, 0.5)).toBe("3.0");
    expect(formatTick(2e-5, 1e-5)).toBe("2e-5");
    expect(formatTick(3e7, 1e7)).toBe("3e7");
    expect(formatTick(-0.0000000001, 0.5)).toBe("0.0");
  });
});

describe("log ticks", () => {
  it("uses 1-2-5 mantissas over a few decades", () => {
    expect(logTickValues(1, 100)).toEqual([1, 2, 5, 10, 20, 50, 100]);
  });

  it("uses decades only over a wide range, thinning to fit", () => {
    expect(logTickValues(1, 1e6)).toEqual([1, 10, 100, 1e3, 1e4, 1e5, 1e6]);
    expect(logTickValues(1e-3, 1e9, { maxTicks: 7 })).toEqual([1e-2, 1, 1e2, 1e4, 1e6, 1e8]);
    expect(logTickValues(2, 8)).toEqual([2, 3, 4, 5, 6, 7, 8]);
  });

  it("returns nothing for non-positive domains", () => {
    expect(logTickValues(0, 10)).toEqual([]);
    expect(logTickValues(-5, -1)).toEqual([]);
  });

  it("labels decades as powers of ten, other ticks as plain numbers", () => {
    expect(labels(logTicks([1, 100]))).toEqual(["1", "2", "5", "10", "20", "50", "100"]);
    expect(labels(logTicks([0.01, 1e5]))).toEqual(["10⁻²", "10⁻¹", "1", "10", "10²", "10³", "10⁴", "10⁵"]);
  });

  it("can place ticks on an axis already drawn in log10 units", () => {
    const t = logTicks([0, 2], { space: "log10" });
    expect(values(t)).toEqual([0, Math.log10(2), Math.log10(5), 1, Math.log10(20), Math.log10(50), 2]);
    expect(labels(t)).toEqual(["1", "2", "5", "10", "20", "50", "100"]);
  });
});

describe("decade ticks", () => {
  it("emits integer decades with 10ⁿ labels", () => {
    expect(values(decadeTicks([-1.5, 3.2], { space: "log10" }))).toEqual([-1, 0, 1, 2, 3]);
    expect(labels(decadeTicks([-1.5, 3.2], { space: "log10" }))).toEqual(["10⁻¹", "1", "10", "10²", "10³"]);
    expect(values(decadeTicks([1e-2, 1e2]))).toEqual([0.01, 0.1, 1, 10, 100]);
    expect(values(decadeTicks([-6, 6], { space: "log10", step: 3 }))).toEqual([-6, -3, 0, 3, 6]);
  });
});

describe("magnitude ticks", () => {
  it("prefers whole / half magnitudes", () => {
    expect(values(magnitudeTicks([18, 24]))).toEqual([18, 19, 20, 21, 22, 23, 24]);
    expect(values(magnitudeTicks([18.2, 19.9]))).toEqual([18.5, 19, 19.5]);
    expect(labels(magnitudeTicks([18.2, 19.9]))).toEqual(["18.5", "19.0", "19.5"]);
  });

  it("can negate positions for bright-at-top axes", () => {
    const t = magnitudeTicks([18, 20], { invert: true });
    expect(values(t)).toEqual([-18, -18.5, -19, -19.5, -20]);
    expect(labels(t)).toEqual(["18.0", "18.5", "19.0", "19.5", "20.0"]);
  });
});

describe("domains", () => {
  it("computes the finite extent", () => {
    expect(extent([3, null, 1, NaN, 7, undefined])).toEqual([1, 7]);
    expect(extent([])).toBeNull();
    expect(extent([null, NaN])).toBeNull();
  });

  it("pads domains and keeps degenerate spans visible", () => {
    expect(paddedDomain([0, 10], { pad: 0.1 })).toEqual([-1, 11]);
    expect(paddedDomain([5, 5])).toEqual([4.5, 5.5]);
    expect(paddedDomain([], { fallback: [0, 1] })).toEqual([0, 1]);
    expect(paddedDomain([2, 10], { pad: 0.1, includeZero: true })).toEqual([0, 11]);
    expect(paddedDomain([0, 1], { pad: 0, minSpan: 4 })).toEqual([-1.5, 2.5]);
  });

  it("unions domains", () => {
    expect(unionDomain([0, 1], null, [-2, 0.5])).toEqual([-2, 1]);
    expect(unionDomain(null)).toBeNull();
  });
});

describe("pathological domains never hang and never emit NaN / ∞ ticks", () => {
  const MAX = 1000;
  const sane = (vs: number[]) => {
    expect(vs.length).toBeLessThanOrEqual(MAX);
    expect(vs.every(Number.isFinite)).toBe(true);
    for (let i = 1; i < vs.length; i++) expect(vs[i]).toBeGreaterThan(vs[i - 1]);
  };

  it("linear: a span that overflows to Infinity", () => {
    expect(niceStep(Infinity, 5)).toBe(0);
    const vs = linearTickValues(-1e308, 1e308);
    sane(vs);
    expect(vs.length).toBeGreaterThanOrEqual(2);
  });

  it("linear: a subnormal span (niceStep underflows to 0)", () => {
    expect(niceStep(5e-324, 5)).toBe(0);
    const vs = linearTickValues(5e-324, 1e-323);
    sane(vs);
    expect(vs.length).toBeGreaterThanOrEqual(1);
  });

  it("linear: a step that overflows to Infinity", () => {
    sane(linearTickValues(0, 1.7e308, 1));
    expect(linearTicks([0, 1.7e308], { count: 1 }).every((t) => t.label !== "")).toBe(true);
  });

  it("linear: offsets beyond 2^53 steps (i++ cannot advance)", () => {
    sane(linearTickValues(1e17, 1e17 + 64));
    sane(linearTickValues(-1e300, -1e300 + 1e285));
  });

  it("magnitude: huge and overflowing spans stay bounded", () => {
    const big = values(magnitudeTicks([0, 1e9]));
    sane(big);
    expect(big.length).toBeLessThanOrEqual(20);
    sane(values(magnitudeTicks([-1e308, 1e308])));
    sane(values(magnitudeTicks([1e17, 1e17 + 64])));
  });

  it("decades: a non-positive, fractional or huge step never loops", () => {
    sane(values(decadeTicks([1, 1e6], { step: -1 })));
    sane(values(decadeTicks([1, 1e6], { step: 1e-20 })));
    sane(values(decadeTicks([1, 1e6], { step: NaN })));
    expect(values(decadeTicks([1, 1e6], { step: 1e9 }))).toEqual([1]);
    expect(values(decadeTicks([-6, 6], { space: "log10", step: 3 }))).toEqual([-6, -3, 0, 3, 6]);
  });

  it("paddedDomain stays finite when the span overflows", () => {
    const d = paddedDomain([-1e308, 1e308]);
    expect(d.every(Number.isFinite)).toBe(true);
    expect(d[0]).toBeLessThanOrEqual(-1e308);
    expect(d[1]).toBeGreaterThanOrEqual(1e308);
  });
});
