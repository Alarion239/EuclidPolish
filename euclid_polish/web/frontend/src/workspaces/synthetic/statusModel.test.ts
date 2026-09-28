/* Synthetic › Status verdicts, the gate words, the records tick and the
   last field statistics (statusModel.ts). */
import { describe, expect, it } from "vitest";
import type { OverviewItem } from "./api";
import {
  backgroundNoise, galaxyDensities, gateSummary, lastComparison, listText, minus, noiseOff, noiseRatiosText, noiseVerdict,
  powerOff, recordsTickText, rowVerdict, scaleVerdict, statusGroups,
} from "./statusModel";
import { FIELDS, OVERVIEW, galaxyPayload, pixelsPayload, starPayload } from "./testFixtures";

const item = (id: string): OverviewItem => {
  const found = OVERVIEW.items.find((i) => i.id === id);
  if (!found) throw new Error(`no fixture row ${id}`);
  return found;
};

describe("row verdicts", () => {
  it("galaxies: generated vs prior density once the galaxy payload is here, the prior alone before", () => {
    expect(rowVerdict(item("galaxy-model"))).toEqual({ text: "prior 152 galaxies arcmin⁻²" });
    // 5,489 galaxies over 36.41 arcmin² = 151 vs 151.5
    expect(rowVerdict(item("galaxy-model"), { galaxies: galaxyPayload() })).toEqual({
      text: "generated 151 vs prior 152 galaxies arcmin⁻²", tone: undefined,
    });
    expect(galaxyDensities(galaxyPayload()).generated).toBeCloseTo(5489 / 36.41, 6);
  });

  it("stars: generated vs prior, warn-toned beyond 5%", () => {
    expect(rowVerdict(item("star-prior"))).toEqual({ text: "prior 5.08 stars arcmin⁻²" });
    // 6,040 stars in 1,201.49 arcmin² = 5.03 vs 5.084 (−1%)
    expect(rowVerdict(item("star-prior"), { stars: starPayload() })).toEqual({
      text: "generated 5.03 vs prior 5.08 stars arcmin⁻²", tone: undefined,
    });
    const far = starPayload();
    far.distribution!.density_comparison!.synthetic_star_count = 4000;
    expect(rowVerdict(item("star-prior"), { stars: far })?.tone).toBe("warn");
  });

  it("noise: the realised background σ ratio over every band, worst band named, else the measured positions", () => {
    expect(rowVerdict(item("noise-model"))).toEqual({ text: "294 measured Q1 positions" });
    // fixture robust σ ratios VIS 0.91 · Y 0.95 · J 0.97 · H 0.98 (all inside 10%)
    expect(rowVerdict(item("noise-model"), { pixels: pixelsPayload() })).toEqual({
      text: "background σ syn/real 0.91–0.98 in every band", tone: undefined,
    });
    // A stale statistics cache still gives its number, said to be the last result.
    const stale = pixelsPayload({ comparison: null, previous: pixelsPayload().comparison });
    expect(rowVerdict(item("noise-model"), { pixels: stale })?.text).toBe("background σ syn/real 0.91–0.98 in every band (last result)");
  });

  it("noise: one band outside the tolerance warns and is named, even when VIS is fine", () => {
    const rows = backgroundNoise(FIELDS).map((r) => (r.band === "H_E" ? { ...r, ratio: 0.86 } : r.band === "VIS" ? { ...r, ratio: 1 } : r));
    expect(noiseVerdict(rows)).toEqual({ text: "0.86–1.00, H worst", tone: "warn" });
    expect(noiseVerdict([rows[0]])).toEqual({ text: "1.00 (VIS)", tone: undefined });
    expect(noiseVerdict([])).toBeNull();
  });

  it("field statistics: the power ratio of every band, the off bands warn-toned and named", () => {
    // fixture: power syn/real 1.1 in every band (inside ±25%), VIS overlap 0.97
    expect(rowVerdict(item("comparison-cache"), { pixels: pixelsPayload() })).toEqual({
      text: "VIS overlap 0.97, power within 25% in every band", tone: undefined,
    });
    const nisp = { ...FIELDS, scale_similarity: { ...FIELDS.scale_similarity,
      Y_E: { ...FIELDS.scale_similarity.Y_E, variance_ratio: { median: 0.38, p16: 0.3, p84: 0.45 } },
      J_E: { ...FIELDS.scale_similarity.J_E, variance_ratio: { median: 0.36, p16: 0.3, p84: 0.45 } },
      H_E: { ...FIELDS.scale_similarity.H_E, variance_ratio: { median: 0.4, p16: 0.3, p84: 0.45 } } } };
    const payload = pixelsPayload();
    const off = pixelsPayload({ comparison: { ...payload.comparison!, fields: nisp } });
    expect(rowVerdict(item("comparison-cache"), { pixels: off })).toEqual({ text: "power syn/real 0.36–0.40 in Y, J and H", tone: "warn" });
    const v = scaleVerdict(nisp)!;
    expect(v.vis?.powerOff).toBe(false);
    expect(v.off.map((s) => s.band)).toEqual(["Y_E", "J_E", "H_E"]);
    expect(powerOff(1.24)).toBe(false);
    expect(powerOff(0.74)).toBe(true);
  });

  it("writes negatives with the typographic minus", () => {
    expect(minus("-0.15")).toBe("\u22120.15");
    expect(minus("0.15")).toBe("0.15");
  });

  it("the other rows read their own facts", () => {
    expect(rowVerdict(item("psf"))?.text).toBe("not synced here");
    // the records' provenance says what the last generation run used
    expect(rowVerdict({ ...item("psf"), facts: { not_cached: ["VIS"], records_psf_kinds: { VIS: "empirical", Y: "empirical", J: "empirical", H: "empirical" } } }))
      .toEqual({ text: "records used empirical ePSFs in 4 of 4 bands" });
    expect(rowVerdict({ ...item("psf"), facts: { records_psf_kinds: { VIS: "empirical", H: "gaussian" } } }))
      .toEqual({ text: "records used the Gaussian fallback in H", tone: "warn" });
    expect(rowVerdict({ ...item("psf"), facts: { empirical: ["VIS", "Y"], gaussian_fallback: ["J", "H"], not_cached: [] } }))
      .toEqual({ text: "Gaussian fallback in J, H", tone: "warn" });
    expect(rowVerdict({ ...item("psf"), facts: { empirical: ["VIS", "Y", "J", "H"], gaussian_fallback: [], not_cached: [] } })?.text)
      .toBe("empirical in 4 of 4 bands");
    expect(rowVerdict({ ...item("tng-radii"), facts: { valid_count: 5770, expected_count: 5770 } })?.text)
      .toBe("5,770 of 5,770 radii valid");
    expect(rowVerdict(item("saturation"))?.text).toBe("blackout 20% → 90% of cores from 5× to 20× the well");
    expect(rowVerdict(item("training-catalog"))).toBeNull();
    expect(rowVerdict({ ...item("training-catalog"), facts: { cached: true, population_fields: 200, population_fields_with_training: 6600 } })?.text)
      .toBe("6,400 training fields");
    expect(rowVerdict(item("galaxy-plots"), {}, Date.parse("2026-09-26T13:00:00Z"))?.text).toBe("built 3 h ago");
  });
});

describe("last field statistics", () => {
  it("keeps a stale result visible and flags it", () => {
    const current = pixelsPayload();
    expect(lastComparison(current)).toEqual({ comparison: current.comparison, stale: false });
    const changed = pixelsPayload({ availability: { ...current.availability,
      comparison_cache: { present: true, schema_current: true, fresh: false, reason: "comparison inputs changed" } } });
    expect(lastComparison(changed).stale).toBe(true);
    const old = pixelsPayload({ comparison: null, previous: current.comparison });
    expect(lastComparison(old)).toEqual({ comparison: current.comparison, stale: true });
    expect(lastComparison(pixelsPayload({ comparison: null }))).toEqual({ comparison: null, stale: false });
    expect(rowVerdict(item("comparison-cache"), { pixels: old })?.text).toBe("VIS overlap 0.97, power within 25% in every band (last result)");
  });

  it("realised background noise per band, and the ratios in one line", () => {
    const rows = backgroundNoise(FIELDS);
    expect(rows.map((r) => r.band)).toEqual(["VIS", "Y_E", "J_E", "H_E"]);
    expect(rows[0].synthetic).toBe(1.5);
    expect(rows[0].real).toBeCloseTo(1.65, 9);
    expect(noiseRatiosText(rows)).toBe("VIS 0.91 · Y 0.95 · J 0.97 · H 0.98");
    expect(noiseOff(rows)).toBe(false);
    expect(noiseOff([{ ...rows[0], ratio: 0.85 }])).toBe(true);
  });
});

describe("groups, gate and records", () => {
  it("splits what generation reads from the diagnostic caches", () => {
    const g = statusGroups(OVERVIEW.items);
    expect(g.generation.map((i) => i.id)).toEqual(["galaxy-model", "star-prior", "noise-model", "psf", "tng-radii", "saturation", "training-catalog"]);
    expect(g.diagnostic.map((i) => i.id)).toEqual(["galaxy-plots", "comparison-cache"]);
  });

  it("states the gate and names the ingredients the records predate", () => {
    expect(gateSummary(OVERVIEW)).toEqual({
      ready: false, headline: "Blocked by 1",
      blockers: ["activate a valid Gaia+Euclid stellar calibration before generating fields"], predates: ["Stars"],
    });
    const ready = gateSummary({ ...OVERVIEW, gate: { ...OVERVIEW.gate, ready: true, blockers: [], message: null } });
    expect(ready.headline).toBe("Ready to generate");
    expect(listText(["Stars"])).toBe("Stars");
    expect(listText(["Galaxies", "Stars", "Noise"])).toBe("Galaxies, Stars and Noise");
  });

  it("words the records tick", () => {
    expect(recordsTickText(item("galaxy-model").records)?.label).toBe("records built with it");
    expect(recordsTickText(item("star-prior").records)).toMatchObject({ label: "records predate it", tone: "warn" });
    expect(recordsTickText(item("noise-model").records)?.tone).toBe("neutral");
    expect(recordsTickText(null)).toBeNull();
  });
});
