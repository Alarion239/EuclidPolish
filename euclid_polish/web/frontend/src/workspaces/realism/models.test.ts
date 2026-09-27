/* Pure series builders of the Stars and Pixels tabs. */
import { describe, expect, it } from "vitest";
import { bandColor } from "../../colors";
import {
  correlationSeries as pixelCorrelation, detectionHistogram, detectionStats, perFieldCompleteness, powerSeries,
  quantileSeries, relationDomains, similaritySeries, visibleFrom,
} from "./pixels/model";
import { brightUp, cmdSeries, correlationSeries, densityDomain, densitySeries, fitGuides, projectionSeries } from "./stars/model";
import { offlinePolicy } from "./jobs";
import { FIELDS, starPayload } from "./testFixtures";

const distribution = starPayload().distribution!;

describe("stars model", () => {
  it("draws the density curves on a log axis with one legend key each", () => {
    const vis = distribution.density_comparison!.parameters.vis;
    const series = densitySeries(vis, false);
    expect(series.map((s) => s.key)).toEqual(["point sources", "Q1 PHZ", "Gaia G_AB", "Gaia fit", "model", "generated"]);
    expect(series.at(-1)!.name).toBe("generated test + validation stars");
    expect(densitySeries(vis, true).at(-1)!.name).toBe("generated train + test + validation stars");
    const [lo, hi] = densityDomain(vis);
    expect(lo).toBeCloseTo(1e-3);
    expect(hi).toBeCloseTo(0.1);            // decades around 0.009–0.08
    expect(densityDomain({ ...vis, euclid: [], gaia: [], model: [], synthetic: [], point_sources: null, gaia_fit: null })).toEqual([1e-4, 1]);
  });

  it("takes the density domain of 200k values (no spread into Math.max)", () => {
    const vis = distribution.density_comparison!.parameters.vis;
    const big = Array.from({ length: 200_000 }, (_, i) => 0.01 + (i % 10) / 100);
    const [lo, hi] = densityDomain({ ...vis, euclid: big, gaia: [], model: [], synthetic: [], point_sources: null, gaia_fit: null });
    expect(lo).toBeCloseTo(1e-2);
    expect(hi).toBeCloseTo(0.1);            // values 0.01–0.1
  });

  it("marks the Q1 and Gaia fit windows on the VIS panel", () => {
    const guides = fitGuides(distribution.density_comparison!.parameters.vis);
    expect(guides.map((g) => g.v)).toEqual([18, 23, 16, 20]);
    expect(guides.filter((g) => g.label).map((g) => g.label)).toEqual(["Q1 fit", "Gaia fit"]);
    expect(fitGuides({ ...distribution.density_comparison!.parameters.vis, fit_ranges: { q1: [18, null] } })).toEqual([]);
  });

  it("draws magnitudes bright-at-top and colour fits with their 1σ/2σ bands", () => {
    expect(brightUp([15, 30], [12, 21])).toEqual([-15, -21]);
    const cmd = cmdSeries(distribution);
    expect(cmd.map((s) => s.key)).toEqual(["unmatched", "matched"]);
    expect(cmd[1].y).toEqual([-15, -17]);
    expect(correlationSeries(distribution, "vis_y").map((s) => s.key)).toEqual(["2σ", "1σ", "stars", "locus"]);
    expect(projectionSeries(distribution, "vis_j").map((s) => s.key)).toEqual(["unmatched", "matched", "measured"]);
  });
});

describe("pixels model", () => {
  it("draws every band in its band colour, synthetic solid and real dashed", () => {
    const series = quantileSeries(FIELDS, visibleFrom([]));
    expect(series).toHaveLength(8);
    expect(series[0]).toMatchObject({ key: "VIS:synthetic", color: bandColor("VIS") });
    expect(series[0].dash).toBeUndefined();
    expect(series[1].dash).toEqual([10, 5]);
    expect(similaritySeries(FIELDS, visibleFrom(["J_E"])).map((s) => s.key)).toEqual(["VIS", "Y_E", "H_E"]);
    expect(pixelCorrelation(FIELDS, visibleFrom(["synthetic"])).map((s) => s.key)).toEqual(["real"]);
  });

  it("orders power by angular scale, drops k = 0 and keeps positive values for the log axis", () => {
    const [vis] = powerSeries(FIELDS, visibleFrom(["real"]));
    expect(vis.x).toEqual([1, 2, 10]);
    expect(vis.y).toEqual([5, 3, 4]);
  });

  it("pads relation domains and starts y at zero", () => {
    const d = relationDomains(FIELDS, "mean_std");
    expect(d.x[0]).toBeLessThan(1);
    expect(d.y[0]).toBe(0);
  });

  it("summarises the per-field detections: counts, negative ÷ positive and completeness", () => {
    const s = detectionStats(FIELDS.source_detection!.synthetic);
    expect(s.fields).toBe(3);
    expect(s.positive.median).toBe(24);
    expect(s.spurious).toBeCloseTo(6 / 74);
    expect(s.completeness).toBeCloseTo(53 / 65);
    expect(s.matchedStars).toBe(3);
    expect(detectionStats(FIELDS.source_detection!.real).completeness).toBeNull();
    expect(perFieldCompleteness(FIELDS.source_detection!.synthetic)).toEqual([0.75, 0.9, 0.8]);
  });

  it("bins both samples' detection counts on shared bins, as fractions of their fields", () => {
    const series = detectionHistogram(FIELDS.source_detection!, "positive", visibleFrom([]));
    expect(series.map((s) => s.key)).toEqual(["synthetic", "real"]);
    expect(series[0].x).toEqual(series[1].x);
    for (const s of series) expect(s.y.reduce((a, b) => (a ?? 0) + (b ?? 0), 0)).toBeCloseTo(1);
  });
});

describe("FASRC offline policy", () => {
  it("disables only gated endpoints offline; self-connecting jobs stay enabled and say so", () => {
    expect(offlinePolicy({ requires_fasrc: true }, false)).toEqual({ disabled: false, hint: null });
    expect(offlinePolicy({ requires_fasrc: true }, true)).toEqual({ disabled: true, hint: "FASRC is offline" });
    expect(offlinePolicy({ self_connects: true }, true)).toEqual({ disabled: false, hint: "FASRC is offline: the job connects first" });
    expect(offlinePolicy({}, true)).toEqual({ disabled: false, hint: null });
  });
});
