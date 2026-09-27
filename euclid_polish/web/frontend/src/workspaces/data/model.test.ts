import { describe, expect, it } from "vitest";
import type { StarsPayload, TngPayload, TruthSource } from "./api";
import {
  atlasHref, axisDomain, bandState, starState, clusterObjectId, decodeBand, decodeStars, decodeTng, DEFAULT_STAR_FILTER,
  fieldCounts, filterStars, histogram, magBins, mapViewBox, nearestPoint, parseClusterId, parseRange,
  parseTruthId, propertyHistogram, resumeSafeStep, stripRebuildFlags, scatterGroups, serializeRange, sourceMarker, summaryStats, tngValue, truthId,
} from "./model";

const BITS = { valid: 1, corrupted: 2, failed: 4, size_shift: 3 };

const payload = (rows: (number | string | null)[][]): StarsPayload => ({
  present: true, source: "fasrc-mirror", path: "/r/stars.csv", local_path: "/l/stars.csv", size_bytes: 1, mtime: 1,
  age_s: 10, bands: ["VIS", "Y_E", "J_E", "H_E"], sizes: [255, 511], bits: BITS,
  columns: ["id", "ra", "dec", "mag", "flux_uJy", "fluxerr_uJy", "field", "b_VIS", "b_Y_E", "b_J_E", "b_H_E", "nav"],
  rows, summary: null, band_stats: [],
});

const V511 = 1 | (1 << 4);          // valid at 511
const V_BOTH = 1 | (1 << 3) | (1 << 4);

describe("star catalogue", () => {
  const stars = decodeStars(payload([
    [1, 269.7, 66.0, 17.5, 275, 0.4, "EDF-N", V511, V511, V511, V511, 1],
    [2, 61.2, -48.4, 18.2, 180, 0.5, "EDF-S", V_BOTH, 0, 0, 2, 0],
    [3, 10, 10, 18.9, 100, 0.6, "", 0, 0, 4, 0, 0],
    [null, 1, 1, 1, 1, 1, "", 0, 0, 0, 0, 0],             // no id: dropped
  ]));

  it("decodes rows and band codes", () => {
    expect(stars.map((s) => s.id)).toEqual([1, 2, 3]);
    expect(stars[0].nav).toBe(true);
    expect(stars[0].nValid).toBe(4);
    expect(stars[1].bands.VIS).toEqual({ valid: true, corrupted: false, failed: false, sizes: [255, 511] });
    expect(bandState(stars[1].bands.H_E)).toBe("corrupted");
    expect(bandState(stars[2].bands.J_E)).toBe("failed");
    expect(bandState(stars[2].bands.VIS)).toBe("pending");
    expect(decodeBand(V511 | 2, [255, 511])).toEqual({ valid: true, corrupted: true, failed: false, sizes: [511] });
  });

  it("filters by field, cutout coverage, band state and magnitude", () => {
    const f = DEFAULT_STAR_FILTER;
    expect(filterStars(stars, f)).toHaveLength(3);
    expect(filterStars(stars, { ...f, field: "EDF-S" }).map((s) => s.id)).toEqual([2]);
    expect(filterStars(stars, { ...f, field: "none" }).map((s) => s.id)).toEqual([3]);
    expect(filterStars(stars, { ...f, cutouts: "nav" }).map((s) => s.id)).toEqual([1]);
    expect(filterStars(stars, { ...f, cutouts: "some" }).map((s) => s.id)).toEqual([2]);
    expect(filterStars(stars, { ...f, cutouts: "none" }).map((s) => s.id)).toEqual([3]);
    expect(filterStars(stars, { ...f, band: "H_E", bandState: "corrupted" }).map((s) => s.id)).toEqual([2]);
    expect(filterStars(stars, { ...f, band: "any", bandState: "failed" }).map((s) => s.id)).toEqual([3]);
    // "any" = the star's overall (best-band) state, the same rule as the KPI strip:
    // star 2 has a corrupted H but a valid VIS, so it is overall valid, not corrupted
    expect(filterStars(stars, { ...f, band: "any", bandState: "corrupted" })).toEqual([]);
    expect(filterStars(stars, { ...f, band: "any", bandState: "valid" }).map((s) => s.id)).toEqual([1, 2]);
    expect(stars.map(starState)).toEqual(["valid", "valid", "failed"]);
    expect(filterStars(stars, { ...f, mag: [18, 18.5] }).map((s) => s.id)).toEqual([2]);
    expect(fieldCounts(stars)).toEqual({ "EDF-N": 1, "EDF-S": 1, none: 1 });
  });

  it("is empty for an absent mirror", () => {
    expect(decodeStars({ ...payload([]), present: false })).toEqual([]);
    expect(decodeStars(null)).toEqual([]);
  });

  it("round-trips the magnitude range of the URL", () => {
    expect(parseRange("17.2-18.5")).toEqual([17.2, 18.5]);
    expect(parseRange("19,17")).toEqual([17, 19]);
    expect(parseRange("x")).toBeNull();
    expect(serializeRange([17.2, 18.5])).toBe("17.2-18.5");
    expect(serializeRange(null)).toBe("");
  });
});

describe("histograms and stats", () => {
  it("bins with a closed top edge and ignores the rest", () => {
    const h = histogram([0, 0.5, 1, 1.5, 2, 3, null, Number.NaN], 0, 2, 2);
    expect(h.counts).toEqual([2, 3]);
    expect(h.centers).toEqual([0.5, 1.5]);
    expect(h.edges).toEqual([0, 1, 2]);
  });

  it("chooses whole magnitude bins", () => {
    const b = magBins(16.02, 18.97, 0.05);
    expect(b.lo).toBeCloseTo(16.0);
    expect(b.hi).toBeCloseTo(19.0);
    expect(b.bins).toBe(60);
  });

  it("summarises finite values", () => {
    const s = summaryStats([3, 1, 2, null])!;
    expect({ n: s.n, min: s.min, max: s.max, median: s.median }).toEqual({ n: 3, min: 1, max: 3, median: 2 });
    expect(s.p16).toBeCloseTo(1.32);
    expect(s.p84).toBeCloseTo(2.68);
    expect(summaryStats([])).toBeNull();
  });
});

describe("ids and links", () => {
  it("builds and parses inspector ids", () => {
    expect(truthId("test", 5, 2)).toBe("test/5/2");
    expect(parseTruthId("validate/12/0")).toEqual({ split: "validate", index: 12, row: 0 });
    expect(parseTruthId("nope/1/2")).toBeNull();
    expect(clusterObjectId(7)).toBe("cluster-007");
    expect(parseClusterId("cluster-012")).toBe(12);
    expect(parseClusterId("3")).toBe(3);
    expect(parseClusterId("x")).toBeNull();
  });

  it("links to the atlas with readable params", () => {
    expect(atlasHref({ ra: 269.7123456, dec: 66.1, fov: 0.05, layers: ["stars", "q1-tiles:0.3"], inspect: "star:12" }))
      .toBe("/sky/atlas?ra=269.712346&dec=66.1&fov=0.05&layers=stars,q1-tiles:0.3&inspect=star:12");
    expect(atlasHref({})).toBe("/sky/atlas");
  });
});

describe("truth-source map", () => {
  const src = (over: Partial<TruthSource>): TruthSource => ({
    row: 0, type: "galaxy", render: "tng", x_pix: 10, y_pix: 20, off_field: false, flux_vis_e: 1000,
    flux_y_e: null, flux_j_e: null, flux_h_e: null, mag_vis: 24, mag_y_e: null, mag_j_e: null, mag_h_e: null,
    target_vis_mag: null, z: null, re_arcsec: 0.2, theta_E_arcsec: null, orientation: null, temperature_k: null,
    subhalo_id: null, source_subhalo_id: null, sfr_class: null, ...over,
  });
  const grid = { width: 510, height: 510, pixscale: 0.05 };

  it("sizes markers by physics", () => {
    expect(sourceMarker(src({}), grid)!.r).toBeCloseTo(4);                    // 0.2″ / 0.05″
    expect(sourceMarker(src({ type: "lens", theta_E_arcsec: 1.0 }), grid)!.r).toBeCloseTo(20);
    const bright = sourceMarker(src({ type: "star", mag_vis: 16 }), grid)!.r;
    const faint = sourceMarker(src({ type: "star", mag_vis: 21 }), grid)!.r;
    expect(bright).toBeGreaterThan(faint);
    expect(sourceMarker(src({ x_pix: null }), grid)).toBeNull();
    expect(sourceMarker(src({ off_field: true }), grid)!.title).toContain("off-field");
  });

  it("frames every marker, off-field ones too", () => {
    const m = [sourceMarker(src({ x_pix: -30, y_pix: 5 }), grid)!, sourceMarker(src({ x_pix: 500, y_pix: 530 }), grid)!];
    const [x, y, w, h] = mapViewBox(m, 510, 510);
    expect(x).toBeLessThan(-30);
    expect(y).toBeLessThan(0);
    expect(x + w).toBeGreaterThan(510);
    expect(y + h).toBeGreaterThan(530);
  });
});

describe("TNG explorer", () => {
  const tng: TngPayload = {
    present: true, files: { properties: { present: true, name: "p", rows: 3, mtime: 1 }, atlas: { present: true, name: "a", rows: 3, mtime: 1 } },
    atlas_meta: null,
    columns: ["id", "sfr", "mass_stars", "m_halo", "reff", "re_kpc", "re_kpc_min", "re_kpc_max", "n_orient", "local"],
    rows: [
      [1, 0, 3e11, 2e12, 8.4, 6, 5, 7, 2, 0],
      [2, 0.5, 1e10, null, 3, null, null, null, 0, 0],
      [9, 2, 1e9, 1e11, 1.5, 1, 1, 1, 1, 12],
      [10, 4, 2e9, 1e11, 1.5, 2, 2, 2, 1, 0],
    ],
    orientations: {}, summary: { n: 4, n_quenched: 1, n_missing_sfr: 0, n_in_atlas: 3, n_local: 1 },
  };
  const rows = decodeTng(tng);

  it("decodes rows and derives sSFR", () => {
    expect(rows.map((r) => r.id)).toEqual([1, 2, 9, 10]);
    expect(rows[1].m_halo).toBeNull();
    expect(tngValue(rows[2], "ssfr")).toBeCloseTo(2e-9);
    expect(tngValue(rows[0], "re_kpc")).toBe(6);
  });

  it("groups points by colour quantile and hides what a log axis cannot show", () => {
    const { groups, hidden } = scatterGroups(rows, "mass_stars", "sfr", "re_kpc", { xlog: true, ylog: true, groups: 2 });
    expect(hidden).toBe(1);                               // SFR = 0 on a log axis
    const all = groups.flatMap((g) => g.ids);
    expect(all.sort()).toEqual([10, 2, 9].sort());
    expect(groups.find((g) => g.key === "none")?.ids).toEqual([2]);   // no measured Rₑ
    expect(groups.every((g) => g.t <= 1)).toBe(true);
  });

  it("finds the clicked point", () => {
    const { groups } = scatterGroups(rows, "mass_stars", "sfr", "re_kpc", { xlog: true, ylog: true, groups: 2 });
    expect(nearestPoint(groups, { x: 1.05e9, y: 2.05 }, { xlog: true, ylog: true, xSpan: 3, ySpan: 1 })).toBe(9);
    expect(nearestPoint(groups, { x: 1e12, y: 1e-3 }, { xlog: true, ylog: true, xSpan: 3, ySpan: 1 })).toBeNull();
  });

  it("builds log-aware domains and histograms", () => {
    const [lo, hi] = axisDomain([1e9, 1e11], true);
    expect(lo).toBeLessThan(1e9);
    expect(hi).toBeGreaterThan(1e11);
    expect(axisDomain([], false)).toEqual([0, 1]);
    const h = propertyHistogram([1e9, 1e10, 1e11, 0, null], true, 2);
    expect(h.counts).toEqual([1, 2]);
    expect(h.log).toBe(true);
  });
});

describe("synthetic_generate resume prefill", () => {
  it("drops split rebuild / force tokens from extra flags, keeping the rest", () => {
    expect(stripRebuildFlags("--regenerate-splits=train")).toEqual({ rest: "", dropped: ["--regenerate-splits=train"] });
    expect(stripRebuildFlags("--seed 3 --regenerate-splits validate,test --force --n-jobs=4")).toEqual({
      rest: "--seed 3 --n-jobs=4", dropped: ["--regenerate-splits validate,test", "--force"],
    });
    expect(stripRebuildFlags("  --seed 3  ")).toEqual({ rest: "--seed 3", dropped: [] });
    expect(stripRebuildFlags("")).toEqual({ rest: "", dropped: [] });
    expect(stripRebuildFlags("--forced-photometry")).toEqual({ rest: "--forced-photometry", dropped: [] });
  });

  it("sanitises the last run's params so the form really resumes", () => {
    const step = {
      step_id: "synthetic_generate", label: "Gen", needs_gpu: false,
      defaults: { partition: "shared", n_cpus: 16, n_gpus: 0, memory: "64G", time_limit: "6:00:00" },
      last_params: { extra_flags: "--regenerate-splits=train --seed 2", n_train: 100 },
    };
    const { step: safe, dropped } = resumeSafeStep(step);
    expect(dropped).toEqual(["--regenerate-splits=train"]);
    expect(safe.last_params).toEqual({ extra_flags: "--seed 2", n_train: 100 });
    expect(step.last_params.extra_flags).toBe("--regenerate-splits=train --seed 2");   // not mutated
    const clean = { ...step, last_params: { extra_flags: "--seed 2" } };
    expect(resumeSafeStep(clean)).toEqual({ step: clean, dropped: [] });
    expect(resumeSafeStep({ ...step, last_params: null }).dropped).toEqual([]);
  });
});
