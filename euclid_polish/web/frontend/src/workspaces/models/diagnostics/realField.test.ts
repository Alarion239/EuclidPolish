import { describe, expect, it } from "vitest";
import {
  brightnessPair, crossCurves, displayEdges, extendTo, heatPair, membersCaption, rebinHeat, sharedDomain,
  transformedSeries, type FieldDiagnostics,
} from "./realField";

describe("real-field diagnostics shaping", () => {
  it("transforms, drops non-positive / missing points and sorts", () => {
    expect(transformedSeries([0.5, null, 0, 0.1, 0.25], [1, 2, 3, null, 0.5], (k) => 0.5 / k))
      .toEqual([{ x: 1, y: 1 }, { x: 2, y: 0.5 }]);
  });

  it("holds the last value out to the common endpoint", () => {
    expect(extendTo([{ x: 1, y: 0.4 }], 10)).toEqual([{ x: 1, y: 0.4 }, { x: 10, y: 0.4 }]);
    expect(extendTo([{ x: 12, y: 0.4 }], 10)).toEqual([{ x: 12, y: 0.4 }]);
    expect(extendTo([], 10)).toEqual([]);
  });

  it("puts the real field (frequency) and the synthetic evals (θ) on the same d axis", () => {
    const real = crossCurves([0.5, 0.05], [[0.9, 0.2]], [0.8, 0.3], true)!;
    expect(real.median.x).toEqual([1, 10]);                 // d = 0.5 / k, already at 10″
    expect(real.median.y).toEqual([0.8, 0.3]);
    const syn = crossCurves([1, 2], [[0.9, 0.8]], [0.7, 0.6], false)!;
    expect(syn.median).toEqual({ x: [1, 2, 10], y: [0.7, 0.6, 0.6] });
    expect(crossCurves([1], [], [null], false)).toBeNull();
  });

  it("shares a domain across arrays with gaps", () => {
    expect(sharedDomain([1, null, 3], [NaN, 5])).toEqual([1, 5]);
    expect(sharedDomain([2, 2])).toEqual([2, 3]);
    expect(sharedDomain([null])).toEqual([0, 1]);
    expect(displayEdges([0, 1], 4)).toEqual([0, 0.25, 0.5, 0.75, 1]);
  });

  it("re-bins a histogram by cell centre, conserving the counts inside the target", () => {
    const z = [[1, 2], [3, 4]];                              // x-bins [0,1],[1,2]; y-bins [0,1],[1,2]
    expect(rebinHeat(z, [0, 1, 2], [0, 1, 2], [0, 2], [0, 2])).toEqual([[10]]);
    expect(rebinHeat(z, [0, 1, 2], [0, 1, 2], [0, 1], [0, 2])).toEqual([[3]]);   // x-centre 1.5 dropped
  });

  it("pairs real and synthetic heat maps on shared axes and the coarser binning", () => {
    const p = heatPair({ z: [[1], [1], [1], [1]], xEdges: [0, 1, 2, 3, 4], yEdges: [0, 1] },
      { z: [[5], [5]], xEdges: [0, 2, null, 4].filter((v) => v !== null), yEdges: [0, 1] }, { x: "b", y: "s" });
    expect(p.xDomain).toEqual([0, 4]);
    expect(p.xEdges).toEqual([0, 2, 4]);                      // min(4, 2) bins
    expect(p.real).toEqual([[2], [2]]);
    expect(p.synthetic).toEqual([[5], [5]]);
    const alone = heatPair({ z: [[1]], xEdges: [0, 1], yEdges: [0, 1] }, null, { x: "b", y: "s" });
    expect(alone.synthetic).toBeNull();
  });

  const DIAG: FieldDiagnostics = {
    version: 2, member_labels: ["1·psnr", "2·psnr"],
    model_power: { k: [0.5], r_pairs: [[0.9]], r_cross: [0.9], pixel_scale_arcsec: 0.05 },
    std_brightness: { x_edges: [0, 1], y_edges: [0, 1], counts: [[4]], x_label: "brightness", y_label: "σ" },
  };

  it("builds the brightness pair on shared axes", () => {
    const b = brightnessPair(DIAG, { bright_std: { bright_edges: [0, 1], std_edges: [0, 1], hist: [[7]], bright: [], lo: [], med: [], hi: [], stretch: 1 } });
    expect(b?.real).toEqual([[4]]);
    expect(b?.synthetic).toEqual([[7]]);
    expect(brightnessPair({ ...DIAG, std_brightness: { ...DIAG.std_brightness, counts: [] } }, null)).toBeNull();
  });
});

describe("real vs synthetic member counts", () => {
  it("names how many members made each side (the real field ran the gate's members)", () => {
    const f = { member_labels: ["1", "2", "3"], run_member_labels: ["1", "2"], member_scope: "gate" as const };
    expect(membersCaption(f, 30)).toBe("2 real vs 30 synthetic members (the real field ran the production gate's)");
    expect(membersCaption({ ...f, run_member_labels: f.member_labels }, 30)).toBe("3 real vs 30 synthetic members");
    expect(membersCaption({ member_labels: ["1"] }, 1)).toBe("1 real vs 1 synthetic member");
    expect(membersCaption({ member_labels: ["1", "2"] }, null)).toBe("2 real members; the synthetic evaluation is not cached");
  });
});
