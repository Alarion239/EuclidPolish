import { describe, expect, it } from "vitest";
import { autoTicks, unitTicks } from "./plotTicks";

describe("autoTicks (a chart whose caller passes no ticks still has labelled axes)", () => {
  it("gives ~5 nice linear ticks inside the domain", () => {
    const t = autoTicks([0, 12000]);
    expect(t.length).toBeGreaterThanOrEqual(4);
    expect(t.length).toBeLessThanOrEqual(8);
    expect(t.every((x) => x.v >= 0 && x.v <= 12000)).toBe(true);
    expect(t.every((x) => x.label.length > 0)).toBe(true);
  });
  it("PSNR in dB: a narrow domain far from zero", () => {
    const t = autoTicks([43.1, 44.6]);
    expect(t.length).toBeGreaterThanOrEqual(3);
    expect(t[0].v).toBeGreaterThanOrEqual(43.1);
  });
  it("decades on a log axis", () => {
    expect(autoTicks([0.01, 100], "log").map((x) => x.v)).toEqual([0.01, 0.1, 1, 10, 100]);
  });
  it("labels with the caller's format", () => {
    expect(autoTicks([0, 1], "linear", (v) => `${v * 100}%`).some((x) => x.label.endsWith("%"))).toBe(true);
  });
  it("a log axis over non-positive values falls back to linear; a degenerate domain has none", () => {
    expect(autoTicks([-1, 1], "log").length).toBeGreaterThan(0);
    expect(autoTicks([3, 3])).toEqual([]);
    expect(autoTicks([NaN, 1])).toEqual([]);
  });
});

describe("unitTicks", () => {
  it("quarters from 0 up to the top (r(k))", () => {
    expect(unitTicks(1.05).map((t) => t.v)).toEqual([0, 0.25, 0.5, 0.75, 1]);
    expect(unitTicks(1.25).map((t) => t.label)).toEqual(["0", "0.25", "0.5", "0.75", "1", "1.25"]);
  });
});
