/* Pure series builders of the Stars and Pixels tabs. */
import { describe, expect, it } from "vitest";
import { bandColor } from "../../colors";
import {
  censusRows, correlationSeries as pixelCorrelation, detectionHistogram, detectionStats, generatedSample, perFieldCompleteness, powerSeries,
  quantileSeries, relationDomains, similaritySeries, visibleFrom,
} from "./pixels/model";
import * as starModel from "./stars/model";
import { densityDelta, densityDomain, densitySeries, fitGuides, generatedDensity, trustedWindow } from "./stars/model";
import { brightnessEntries, brightnessOverlays } from "./galaxies/model";
import { offlinePolicy } from "./jobs";
import { FIELDS, galaxyPayload, starPayload } from "./testFixtures";

const distribution = starPayload().distribution!;

describe("stars model", () => {
  it("gives the colour panels' four-band curve its own hue, apart from the VIS Q1 PHZ curve", () => {
    expect(starModel.densityColor.fourBand()).not.toBe(starModel.densityColor.q1());
  });

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

  it("draws a colour panel without the deleted Gaia projection", () => {
    // The native Gaia G_AB counts feed the shared-slope magnitude-law fit, so they stay on the
    // VIS panel; the colour panels carry no Gaia series (the projection is a deterministic locus).
    const colour = distribution.density_comparison!.parameters.vis_j;
    expect(colour.gaia).toBeUndefined();
    // Its Euclid curve is the matched stars' four-band colours (its own legend key), not Q1 PHZ.
    expect(densitySeries(colour, false, "vis_j").map((s) => s.key)).toEqual(["Euclid four-band", "model", "generated"]);
    const [lo, hi] = densityDomain(colour);
    expect(lo).toBeCloseTo(1e-3);
    expect(hi).toBeCloseTo(0.1);
  });

  it("reads the trusted Q1 window and the generated-vs-prior density for the summary line", () => {
    const comparison = distribution.density_comparison!;
    expect(trustedWindow(comparison.parameters.vis)).toEqual([18, 23]);
    expect(trustedWindow({ ...comparison.parameters.vis, fit_ranges: { q1: [18, null] } })).toBeNull();
    expect(trustedWindow(comparison.parameters.vis_j)).toBeNull();
    expect(generatedDensity(comparison)).toBeCloseTo(6040 / 1201.49);
    expect(densityDelta(comparison)).toBeCloseTo(6040 / 1201.49 / 5.084 - 1);
    expect(generatedDensity({ ...comparison, synthetic_area_arcmin2: null })).toBeNull();
    expect(densityDelta({ ...comparison, synthetic_area_arcmin2: 0 })).toBeNull();
  });

  it("no longer exports the Gaia colour and CMD view builders", () => {
    for (const name of ["COLOR_ORDER", "PROJECTION_ORDER", "correlationSeries", "cmdSeries", "projectionSeries", "brightUp", "clamp"]) {
      expect(Object.keys(starModel)).not.toContain(name);
    }
    expect(starModel.DENSITY_ORDER).toEqual(["vis", "vis_y", "vis_j", "vis_h", "y_j", "y_h", "j_h"]);
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

describe("galaxy brightness overlays", () => {
  it("labels the Q1 support bands in sentence case (no ALL-CAPS)", () => {
    const entries = brightnessEntries(galaxyPayload().parameters.magnitude);
    const labels = brightnessOverlays(entries, [14, 29], [1e-3, 100]).bands.flatMap((b) => (b.label ? [b.label] : []));
    expect(labels).toEqual(["Q1 count support to turnover", "beyond the MER 5σ range"]);
    const noTrust = entries.map(([k, c]) => [k, { ...c, trust_boundary: undefined }] as typeof entries[number]);
    expect(brightnessOverlays(noTrust, [14, 29], [1e-3, 100]).bands.map((b) => b.label)).toEqual(["Q1 count-supported", "beyond the Q1 turnover"]);
  });

  it("census: Q1 compared only over its complete window, every column over the same magnitudes", () => {
    const rows = censusRows(galaxyPayload(), starPayload());
    expect(rows.map((r) => [r.kind, r.window, r.range])).toEqual([
      ["galaxies", "q1", [14, 25.3]], ["galaxies", "prior", [14, 29]], ["stars", "q1", [18, 22]], ["stars", "prior", [16, 22]],
    ]);
    // The full prior range carries no Q1 value: Q1 is incomplete there.
    expect(rows.filter((r) => r.window === "prior").map((r) => r.q1)).toEqual([null, null]);
    const stars = rows[2];
    expect(stars.q1).toBeCloseTo(0.12);
    expect(stars.prior).toBeCloseTo(0.132);
    expect(stars.generated).toBeCloseTo(0.108);
    // Without a Q1 5σ limit there is no complete window, so only the full-range row remains.
    const g = galaxyPayload();
    const q1 = g.parameters.magnitude!.photometry_series!.q1_vis_f2;
    const noLimit = { ...g, parameters: { ...g.parameters, magnitude: { ...g.parameters.magnitude!,
      photometry_series: { ...g.parameters.magnitude!.photometry_series!, q1_vis_f2: { ...q1, trust_boundary: undefined } } } } };
    expect(censusRows(noLimit, null).map((r) => r.window)).toEqual(["prior"]);
  });
});

describe("census generated sample", () => {
  it("states a shared area once", () => {
    expect(generatedSample({ rows: 5489, area_arcmin2: 36.41 }, { synthetic_star_count: 175, synthetic_area_arcmin2: 36.4 }))
      .toBe("Generated: 5,489 galaxies and 175 stars over 36.4 arcmin²");
  });
  it("keeps each area when they differ, and drops a missing sample", () => {
    expect(generatedSample({ rows: 5489, area_arcmin2: 36.4 }, { synthetic_star_count: 6040, synthetic_area_arcmin2: 1201.49 }))
      .toBe("Generated: 5,489 galaxies over 36.4 arcmin², 6,040 stars over 1,201 arcmin²");
    expect(generatedSample(undefined, { synthetic_star_count: 175, synthetic_area_arcmin2: 36.4 })).toBe("Generated: 175 stars over 36.4 arcmin²");
    expect(generatedSample(undefined, null)).toBeNull();
  });
});
