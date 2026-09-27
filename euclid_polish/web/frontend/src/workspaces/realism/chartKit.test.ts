import { describe, expect, it } from "vitest";
import {
  angularScaleAxis, binIndex, contourMassLabel, contourSeries, contourStyle, countHistogram, enclosingFraction,
  exceedancePercent, jointColor, log10AxisTicks, logDomain, mixRgb, nearestIndex2d, omitBin, ordered, parseRgb,
  physicalFromLog10, physicalLogAxisLabel, positiveOrNull, quantile, radiusBand, stackedTops, stepSeries,
  surveyColor, tintRamp,
} from "./chartKit";

describe("chart kit", () => {
  it("labels log10 axes in physical units with '(log scale)'", () => {
    expect(physicalLogAxisLabel("log₁₀ Rₑ (arcsec)")).toBe("Rₑ (arcsec, log scale)");
    expect(physicalLogAxisLabel("log₁₀ density")).toBe("density (log scale)");
  });

  it("snaps log domains to half decades and blanks non-positive values", () => {
    expect(logDomain([0.02, 0, 3])).toEqual([10 ** -2, 10 ** 0.5]);
    expect(logDomain([5, 5])).toEqual([10 ** 0.5, 10]);
    expect(logDomain([10])).toEqual([10, 100]);
    expect(logDomain([], [1, 10])).toEqual([1, 10]);
    expect(positiveOrNull([1, 0, -2, NaN, 3])).toEqual([1, null, null, null, 3]);
  });

  it("labels contours by enclosed mass, 10 → 99.9%", () => {
    expect([0.1, 0.5, 0.8, 0.95, 0.99, 0.995, 0.999].map(contourMassLabel))
      .toEqual(["10%", "50%", "80%", "95%", "99%", "99.5%", "99.9%"]);
    expect(contourStyle(0.1).width).toBeGreaterThan(contourStyle(0.999).width);
  });

  it("builds contour series labelled once, on the longest path, with the source dash", () => {
    const contours = [
      { mass_fraction: 0.5, paths: [{ x: [0, 1], y: [0, 1] }, { x: [0, 1, 2], y: [0, 1, 2] }] },
      { mass_fraction: 0.95, paths: [{ x: [0, 1], y: [1, 0] }] },
    ];
    const series = contourSeries(contours, { color: "red", dash: [7, 4], name: "generated", mapY: (y) => 10 ** y });
    expect(series).toHaveLength(3);
    expect(series.map((s) => s.label)).toEqual([undefined, "50%", "95%"]);
    expect(series.every((s) => s.dash?.[0] === 7 && s.key === "generated")).toBe(true);
    expect(series[1].y).toEqual([1, 10, 100]);
    expect(contourSeries(contours, { color: "red", levels: [0.95] })).toHaveLength(1);
  });

  it("finds bins and the enclosing mass fraction of a cell", () => {
    expect(binIndex([0, 1, 2, 3], 1.5)).toBe(1);
    expect(binIndex([0, 1, 2, 3], 3)).toBe(2);
    expect(binIndex([0, 1, 2, 3], -1)).toBe(-1);
    const density = [[4, 1], [3, 0]];
    expect(enclosingFraction(density, 0, 0)).toBeCloseTo(0.5);
    expect(enclosingFraction(density, 0, 1)).toBeCloseTo(1);
    expect(enclosingFraction(density, 1, 1)).toBeNull();
  });

  it("steps, stacks and exceedance curves", () => {
    expect(stepSeries([0, 1, 2], [3, 5])).toEqual({ x: [0, 1, 1, 2], y: [3, 3, 5, 5] });
    expect(stackedTops([[1, 2], [3, 4]], 2)).toEqual([[4, 6], [1, 2]]);
    expect(exceedancePercent([6, 3, 1], 10)).toEqual([100, 40, 10, 0]);
    expect(omitBin([1, 2, 3], 1)).toEqual([1, null, 3]);
  });

  it("orders power spectra by angular scale", () => {
    const axis = angularScaleAxis([0, 0.5, 0.25, 1]);
    expect(axis.x).toEqual([1, 2, 4]);
    expect(ordered(["a", "b", "c", "d"], axis.order)).toEqual(["d", "b", "c"]);
  });

  it("uses the core model band, falling back to the full low/high interval", () => {
    const base = { magnitude: [20], observed_mean_log10_arcsec: [null], model_mean_log10_arcsec: [-0.5] };
    expect(radiusBand({ ...base, model_core_low_log10_arcsec: [-0.6], model_core_high_log10_arcsec: [-0.4],
      model_low_log10_arcsec: [-0.9], model_high_log10_arcsec: [-0.1] }).kind).toBe("core");
    const full = radiusBand({ ...base, model_low_log10_arcsec: [-0.9], model_high_log10_arcsec: [-0.1] });
    expect(full).toEqual({ low: [-0.9], high: [-0.1], kind: "full" });
    expect(radiusBand(base).kind).toBe("none");
  });

  it("picks the nearest point in span units", () => {
    const xs = [0, 10, 5];
    const ys = [0, 10, 5];
    expect(nearestIndex2d(xs, ys, { x: 5.1, y: 4.9 }, { x: 10, y: 10 })).toBe(2);
    expect(nearestIndex2d(xs, ys, { x: 50, y: 50 }, { x: 10, y: 10 })).toBe(-1);
    expect(nearestIndex2d([1, 100], [1, 1], { x: 90, y: 1 }, { x: 2, y: 1 }, { xLog: true })).toBe(1);
  });

  it("parses token colours and blends heat tints opaquely over the surface", () => {
    expect(parseRgb("#2563eb")).toEqual([37, 99, 235]);
    expect(parseRgb("#fff")).toEqual([255, 255, 255]);
    expect(parseRgb("rgb(10, 20, 30)")).toEqual([10, 20, 30]);
    expect(parseRgb("rgba(10 20 30 / 0.5)")).toEqual([10, 20, 30]);
    expect(parseRgb("tomato", [1, 2, 3])).toEqual([1, 2, 3]);
    expect(mixRgb([255, 0, 0], [0, 0, 255], 0.5)).toBe("rgb(128, 0, 128)");
    expect(mixRgb([255, 0, 0], [0, 0, 0], 2)).toBe("rgb(255, 0, 0)");
    const ramp = tintRamp("#ff0000", "#ffffff");
    expect(ramp(0)).toBe("rgb(255, 242, 242)");
    expect(ramp(1)).toBe("rgb(255, 89, 89)");
  });

  it("labels log10 axes with physical ticks", () => {
    const ticks = log10AxisTicks([-1, 1]);
    expect(ticks.map((t) => t.v)).toContain(0);
    expect(ticks.find((t) => t.v === 0)?.label).toBe("1");
    expect(physicalFromLog10(-1)).toBe("0.1");
  });

  it("summarises per-field counts", () => {
    expect(quantile([3, 1, 2], 0.5)).toBe(2);
    expect(quantile([1, 2, 3, 4], 0.5)).toBe(2.5);
    expect(quantile([], 0.5)).toBeNull();
    const h = countHistogram([0, 1, 1, 3]);
    expect(h.edges).toEqual([0, 1, 2, 3, 4]);
    expect(h.fraction).toEqual([0.25, 0.5, 0, 0.25]);
    const wide = countHistogram([0, 100], 10);
    expect(wide.edges[1]).toBe(11);
    expect(wide.fraction.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
  });

  it("reads every colour from tokens (no literals)", () => {
    for (const s of ["euclid", "synthetic", "cosmos", "fit", "generation"] as const) {
      expect(typeof surveyColor(s)).toBe("string");
    }
    expect(jointColor("q1", "maps")).not.toBe(jointColor("q1", "pairs"));
    expect(jointColor("model", "maps")).toBe(jointColor("model", "pairs"));
  });
});
