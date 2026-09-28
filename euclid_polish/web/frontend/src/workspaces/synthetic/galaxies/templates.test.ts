import { describe, expect, it } from "vitest";
import type { TngRow } from "../dataModel";
import { floorDomain, marginalGuides, radiusManifestLine, sig2, templatesCaption, zeroStrip } from "./templatesModel";

const row = (id: number, sfr: number | null, mass: number | null, re: number | null = 5): TngRow => ({
  id, sfr, mass_stars: mass, m_halo: 1e12, reff: 6, re_kpc: re, re_kpc_min: null, re_kpc_max: null, n_orient: 5, local: 0,
});

describe("TNG templates: SFR = 0 on a floor strip", () => {
  const rows = [row(1, 0, 1e10), row(2, 0.5, 2e10), row(3, 0, 3e10), row(4, 2, null), row(5, 0, null)];

  it("collects the galaxies a log axis cannot show because the value is exactly zero", () => {
    const strip = zeroStrip(rows, "mass_stars", "sfr", { xlog: true, ylog: true })!;
    expect(strip.axis).toBe("y");
    expect(strip.count).toBe(2);           // row 5 has no mass, so it is on neither axis
    expect(strip.ids).toEqual([1, 3]);
    expect(strip.values).toEqual([1e10, 3e10]);
    expect(strip.label).toBe("SFR = 0: 2");
  });

  it("needs a log axis and a zero: a linear axis shows zeros itself", () => {
    expect(zeroStrip(rows, "mass_stars", "sfr", { xlog: true, ylog: false })).toBeNull();
    expect(zeroStrip([row(2, 0.5, 2e10)], "mass_stars", "sfr", { xlog: true, ylog: true })).toBeNull();
  });

  it("works on the x axis too, and names sSFR for the ratio", () => {
    const strip = zeroStrip(rows, "ssfr", "mass_stars", { xlog: true, ylog: true })!;
    expect(strip.axis).toBe("x");
    expect(strip.label).toBe("sSFR = 0: 2");
  });

  it("extends a log domain downwards by a strip below its positive range", () => {
    const f = floorDomain([1e-2, 1e2], true);
    expect(f.domain[1]).toBe(1e2);
    expect(f.strip[1]).toBe(1e-2);
    expect(f.domain[0]).toBe(f.strip[0]);
    expect(f.domain[0]).toBeLessThan(1e-2);
    expect(f.at).toBeGreaterThan(f.strip[0]);
    expect(f.at).toBeLessThan(f.strip[1]);
  });
});

describe("TNG templates: the histogram is the explorer's marginal", () => {
  it("draws the median and the 16–84% range at 2 significant figures in the unit", () => {
    const g = marginalGuides({ n: 5, min: 1, max: 9, median: 5.0253, p16: 2.0501, p84: 9.4531 }, "kpc");
    expect(g.map((x) => x.label)).toEqual(["p16 2.1 kpc", "median 5.0 kpc", "p84 9.5 kpc"]);
    expect(g[1].v).toBeCloseTo(5.0253);
    expect(marginalGuides(null, "kpc")).toEqual([]);
    expect([sig2(5), sig2(0.0503), sig2(151.5), sig2(3.9e11), sig2(0.004)]).toEqual(["5.0", "0.050", "150", "3.9e11", "4.0e-3"]);
  });
});

describe("TNG templates: words", () => {
  it("captions the atlas once: galaxies and those with a measured Rₑ", () => {
    expect(templatesCaption({ n: 1154, n_in_atlas: 1140, n_local: 3, n_missing_sfr: 0, n_quenched: 144 }))
      .toBe("TNG50-1 atlas: 1,154 galaxies (1,140 with measured Rₑ)");
    expect(templatesCaption(undefined)).toBeNull();
  });

  it("states the radius manifest ONCE: valid with its check time, or what is wrong", () => {
    const now = 1_790_475_333_000 + 2 * 86_400_000;
    const valid = radiusManifestLine({ valid: true, valid_count: 5770, expected_count: 5770, checked_at: 1_790_475_333, stale: true, reasons: [] }, now);
    expect(valid).toEqual({ state: "ok", text: "5,770 of 5,770 measured radii valid · checked 2 d ago", fix: false });
    const bad = radiusManifestLine({ valid: false, valid_count: 5000, expected_count: 5770, failed_count: 12, reasons: ["hash mismatch"] }, now);
    expect(bad).toEqual({ state: "bad", text: "Radius manifest invalid: hash mismatch", fix: true });
    expect(radiusManifestLine({ cached: false } as never, now)).toEqual({ state: "unknown", text: "Radius manifest not validated yet", fix: true });
  });
});
