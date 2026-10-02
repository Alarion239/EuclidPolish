// @vitest-environment node
import { describe, expect, it } from "vitest";
import type { PlateRun, SavedResult } from "./api";
import {
  LIGHTBOX, capText, commonRecipes, cropSideArcsec, findRender, galleryMatches, gridSizeText, gridStatus, lightboxStageHeight, previewPaperCap, inspectLink, isRecipeKey, matchTile,
  missingRecipes, modeTone, moveItem, normalizeIndex, parseTileList, plateCoverage, recipeLabel,
  renderKey, renderTitle, resultRegime, sanitizeColumns, sheetLegend, skyLink, sourceLabel, specsCoveringAll,
  viewerLink, wcsState, type NexusTile,
} from "./model";

const result = (over: Partial<SavedResult> = {}): SavedResult => ({
  id: "vr-1", label: "one", logical_tiers: ["dirty", "sr"], bands: {}, pixscale_arcsec: {},
  recipes: ["dirty:VIS", "sr:VIS", "sr:VIS_H"], wcs_preserved: false, ...over,
});

describe("recipes", () => {
  it("validates keys and labels rows like the backend", () => {
    expect(isRecipeKey("sr:VIS_H")).toBe(true);
    expect(isRecipeKey("sr:F200W")).toBe(false);
    expect(isRecipeKey("sr")).toBe(false);
    expect(recipeLabel("jwst:native")).toBe("NEXUS F200W");
    expect(recipeLabel("sr:VIS_H")).toBe("VIS + H_E SR");
    expect(recipeLabel("dirty:H_E")).toBe("H_E Dirty");
    expect(recipeLabel("bhr:VIS")).toBe("VIS BHR");
  });

  it("knows the Y and J bands (shown once the backend lists them)", () => {
    expect(isRecipeKey("sr:Y_E")).toBe(true);
    expect(isRecipeKey("dirty:J_E")).toBe(true);
    expect(recipeLabel("sr:Y_E")).toBe("Y_E SR");
    expect(recipeLabel("dirty:J_E")).toBe("J_E Dirty");
    // a backend that lists only VIS / H_E keeps offering only those
    expect(normalizeIndex({ supported: { modes: ["VIS", "H_E", "VIS_H", "native"] }, results: [] }).modes)
      .toEqual(["VIS", "H_E", "VIS_H", "native"]);
    expect(normalizeIndex({ supported: { modes: ["VIS", "Y_E", "J_E", "H_E", "VIS_H"] }, results: [] }).modes)
      .toEqual(["VIS", "Y_E", "J_E", "H_E", "VIS_H"]);
  });

  it("colours rows the way the sheet draws them: one band grey, the composite VIS azure + H_E amber", () => {
    expect(["VIS", "Y_E", "J_E", "H_E"].map(modeTone)).toEqual(["grey", "grey", "grey", "grey"]);
    expect(modeTone("VIS_H")).toBe("vis-h");
    expect(modeTone("native")).toBe("grey");                          // NEXUS F200W: one band
    expect(sheetLegend(["dirty:VIS", "sr:VIS_H", "jwst:native"]).map((e) => e.id)).toEqual(["grey", "vis-h"]);
    expect(sheetLegend(["sr:VIS_H"]).map((e) => e.id)).toEqual(["vis-h"]);
    expect(sheetLegend([]).map((e) => e.id)).toEqual([]);
    const composite = sheetLegend(["sr:VIS_H"])[0];
    expect(composite.swatches.map((s) => [s.tone, s.label])).toEqual([["vis", "VIS"], ["h", "H_E"]]);
  });

  it("names a cap only once it is reached", () => {
    expect(capText(3, 12, "columns")).toBeNull();
    expect(capText(12, 12, "columns")).toBe("12 columns: the most a sheet holds");
    expect(capText(16, 16, "rows")).toBe("16 rows: the most a sheet holds");
  });

  it("finds missing and common recipes", () => {
    const a = result({ id: "a" });
    const b = result({ id: "b", recipes: ["dirty:VIS", "hr:VIS"] });
    expect(missingRecipes(a, ["dirty:VIS", "hr:VIS"])).toEqual(["hr:VIS"]);
    expect(commonRecipes([a, b])).toEqual(["dirty:VIS"]);
    expect(commonRecipes([])).toEqual([]);
  });

});

describe("normalizeIndex", () => {
  it("keeps valid results, counts malformed rows and reads the limits", () => {
    const idx = normalizeIndex({
      limits: { max_results: 6, max_rows: 4 },
      supported: { logical_tiers: ["dirty", "sr", "jwst"], modes: ["VIS", "native"] },
      results: [
        { id: "vr-a", recipes: ["dirty:VIS", "nope:VIS"], source: { object: { label: "obj" } }, center: { ra: 1, dec: 2 } },
        { nope: true }, "x",
      ],
    });
    expect(idx.results.map((r) => r.id)).toEqual(["vr-a"]);
    expect(idx.results[0].label).toBe("obj");
    expect(idx.results[0].recipes).toEqual(["dirty:VIS"]);
    expect(idx.results[0].center).toEqual({ ra: 1, dec: 2 });
    expect(idx.dropped).toBe(2);
    expect([idx.maxResults, idx.maxRows]).toEqual([6, 4]);
    expect(idx.tiers).toEqual(["dirty", "sr", "jwst"]);
    expect(idx.modes).toEqual(["VIS", "native"]);
  });

  it("treats null as empty and a wrong shape as malformed", () => {
    expect(normalizeIndex(null)).toMatchObject({ results: [], malformed: false });
    expect(normalizeIndex({ results: 3 })).toMatchObject({ results: [], malformed: true });
  });
});

describe("regime and grid status", () => {
  it("derives the regime from the collection or the recipes", () => {
    expect(resultRegime(result({ regime: "real" }))).toBe("real");
    expect(resultRegime(result({ source: { collection: "evaluation" } }))).toBe("synthetic");
    expect(resultRegime(result({ source: { collection: "real" } }))).toBe("real");
    expect(resultRegime(result({ recipes: ["jwst:native"] }))).toBe("real");
    expect(resultRegime(result({ recipes: ["hr:VIS"] }))).toBe("synthetic");
    expect(resultRegime(result({ recipes: [] }))).toBeNull();
  });

  it("reports why a grid cannot render", () => {
    const results = [result({ id: "a" }), result({ id: "b", recipes: ["dirty:VIS"] })];
    const base = { loading: false, error: false, results, maxResults: 2, maxRows: 3 };
    expect(gridStatus({ ...base, columns: [], rows: ["dirty:VIS"] }).text).toMatch(/at least one saved crop/);
    expect(gridStatus({ ...base, columns: ["a"], rows: [] }).text).toMatch(/at least one row/);
    expect(gridStatus({ ...base, columns: ["a", "b", "c"], rows: ["dirty:VIS"] }).text).toMatch(/At most 2 columns/);
    expect(gridStatus({ ...base, columns: ["zz"], rows: ["dirty:VIS"] }).text).toMatch(/no longer saved/);
    // a missing cell no longer blocks the sheet: it renders, grey "Not available" in place
    const partial = gridStatus({ ...base, columns: ["a", "b"], rows: ["dirty:VIS", "sr:VIS"] });
    expect(partial).toMatchObject({ canRender: true, missing: true, unsupported: 1, tone: "warn", text: "1 cell not available (grey in the sheet)" });
    expect(gridStatus({ ...base, columns: ["a", "b"], rows: ["dirty:VIS", "sr:VIS", "hr:VIS"] }).text).toBe("3 cells not available (grey in the sheet)");
    // nothing to draw at all: refused
    expect(gridStatus({ ...base, columns: ["b"], rows: ["sr:VIS"] })).toMatchObject({ canRender: false, missing: false, tone: "bad",
      text: "No cell is available: no column has these rows" });
    expect(gridStatus({ ...base, columns: ["a", "b"], rows: ["dirty:VIS"] })).toMatchObject({ canRender: true, missing: false, text: "1 × 2 ready" });
    expect(gridStatus({ ...base, error: true, columns: ["a"], rows: ["dirty:VIS"] }).canRender).toBe(false);
  });

  it("sanitizes columns to saved results of the regime and the limit", () => {
    const results = [result({ id: "a", regime: "real" }), result({ id: "b", regime: "synthetic" }), result({ id: "c", regime: "real" })];
    expect(sanitizeColumns(["a", "b", "x", "a", "c"], results, "real", 5)).toEqual(["a", "c"]);
    expect(sanitizeColumns(["a", "c"], results, "real", 1)).toEqual(["a"]);
  });

  it("moves items within bounds", () => {
    expect(moveItem(["a", "b", "c"], 0, 1)).toEqual(["b", "a", "c"]);
    expect(moveItem(["a", "b"], 0, -1)).toEqual(["a", "b"]);
  });
});

describe("links back to the source", () => {
  it("opens real tiles in the realtile inspector", () => {
    expect(viewerLink(result({ source: { collection: "real", params: { source: "nexus" }, object: { id: "f200w-0040" } } }))?.to)
      .toBe("/sky/targets?inspect=realtile%3Anexus%2Ff200w-0040");
    expect(viewerLink(result({ source: { collection: "real", object: { ref: "tile/ra1_dec2" } } }))?.to)
      .toBe("/sky/targets?inspect=realtile%3Atile%2Fra1_dec2");
    expect(viewerLink(result({ source: { collection: "nexus-field", index: 12, object: {} } }))?.to)
      .toBe("/sky/targets?inspect=realtile%3Anexus%2Ff200w-0012");
    expect(viewerLink(result({ source: { collection: "real-field", object: { id: "rf1/007" } } }))?.to)
      .toBe("/sky/targets?inspect=realtile%3Afield%2Frf1-007");
    expect(viewerLink(result({ source: { collection: "jwst-euclid", object: { id: "pairX" } } }))?.to)
      .toBe("/sky/targets?inspect=realtile%3Apair%2FpairX");
  });

  it("opens a real catalogue object's tile card in its Sky › Targets set", () => {
    expect(viewerLink(result({ source: { collection: "evaluation", object: { id: "102018666_NEG57", grade: "A" } } })))
      .toEqual({ to: "/sky/targets?set=lenses&inspect=tile%3Aeval%2F102018666_NEG57", label: "Open in Sky › Targets" });
    expect(viewerLink(result({ source: { collection: "evaluation", object: { id: "g-9", grade: "gal" } } }))?.to)
      .toBe("/sky/targets?set=galaxies&inspect=tile%3Aeval%2Fg-9");
    expect(viewerLink(result({ source: { collection: "evaluation", object: { id: "x-1" } } }))?.to)
      .toBe("/sky/targets?set=lenses%2Cgalaxies&inspect=tile%3Aeval%2Fx-1");
  });

  it("opens a synthetic stamp in Models › Images (it has HR truth, so it is synthetic validation)", () => {
    expect(viewerLink(result({ source: { collection: "evaluation", object: { id: "syn-lens_0007", grade: "syn-lens" } } })))
      .toEqual({ to: "/models/images?set=stamps&g=syn-lens&id=syn-lens_0007", label: "Open in Models › Images" });
    // the grade is read from the id when the save did not record it
    expect(viewerLink(result({ source: { collection: "evaluation", object: { id: "syn-gal_0100" } } }))?.to)
      .toBe("/models/images?set=stamps&g=syn-gal&id=syn-gal_0100");
    // a stamp saved from an old starless view opens in Models too (one regime now)
    expect(viewerLink(result({ source: { collection: "evaluation", params: { mode: "starless" }, object: { id: "syn-gal_0100" } } }))?.to)
      .toBe("/models/images?set=stamps&g=syn-gal&id=syn-gal_0100");
  });

  it("opens viewer collections on their page with the object id or index", () => {
    expect(viewerLink(result({ source: { collection: "ensemble", index: 3, params: { mode: "starless" } } }))?.to)
      .toBe("/models/images?v.ens.i=3");
    expect(viewerLink(result({ source: { collection: "sky", object: { id: "test:5" }, params: { subset: "test" } } }))?.to)
      .toBe("/synthetic/records?v.sky.id=test%3A5&subset=test");
    expect(viewerLink(result({ source: { collection: "psfs", index: 0 } }))?.to).toBe("/synthetic/psf?view=epsf&v.psfs.i=0");
    expect(viewerLink(result({ source: {} }))).toBeNull();
    expect(viewerLink(result({ source: { collection: "unknown" } }))).toBeNull();
  });

  it("links to the atlas and the inspector", () => {
    expect(skyLink(result({ center: { ra: -1, dec: 65.1 } }))).toBe("/sky/atlas?ra=359.000000&dec=65.100000&fov=0.01");
    expect(skyLink(result({ source: { object: { ra: 10, dec: -5 } } }))).toBe("/sky/atlas?ra=10.000000&dec=-5.000000&fov=0.01");
    expect(skyLink(result())).toBeNull();
    expect(inspectLink(result({ inspect_paths: { sr: "data/viewer_results/vr-1/sr.fits" } }), "sr"))
      .toBe("/files?path=data%2Fviewer_results%2Fvr-1%2Fsr.fits");
    expect(inspectLink(result(), "sr")).toBeNull();
  });

  it("labels the source by the real-tile source for the real collection", () => {
    expect(sourceLabel(result({ source: { collection: "real", params: { source: "nexus" } } }))).toBe("nexus");
    expect(sourceLabel(result({ source: { collection: "evaluation" } }))).toBe("evaluation");
    expect(sourceLabel(result())).toBe("viewer");
  });

  it("reads the crop side and the WCS state", () => {
    expect(cropSideArcsec(result({ selection: { angular_side_arcsec: 1.5 } }))).toBe(1.5);
    expect(cropSideArcsec(result({ pixscale_arcsec: { dirty: 0.1 }, files: { dirty: { shape_hwc: [12, 12, 4] } } }))).toBeCloseTo(1.2);
    expect(cropSideArcsec(result())).toBeNull();
    expect(wcsState(result({ wcs_preserved: true }))).toBe("all");
    expect(wcsState(result({ wcs_tiers: ["dirty"] }))).toBe("partial");
    expect(wcsState(result())).toBe("none");
  });
});

describe("NEXUS plate form", () => {
  const tiles: NexusTile[] = [
    { id: "f200w-0040", ref: "nexus/f200w-0040", models: { rbf: { state: "current" }, production: { state: "stale" } } },
    { id: "f200w-0042", ref: "nexus/f200w-0042", models: { rbf: { state: "current" } } },
  ];

  it("parses tile lists", () => {
    expect(parseTileList("40, 042 ;nexus/f200w-0042,, 40")).toEqual(["40", "42", "f200w-0042"]);
    expect(parseTileList("  ")).toEqual([]);
  });

  it("matches tokens and reports coverage per spec", () => {
    expect(matchTile("40", tiles)?.id).toBe("f200w-0040");
    expect(matchTile("f200w-0042", tiles)?.id).toBe("f200w-0042");
    expect(matchTile("7", tiles)).toBeNull();
    expect(plateCoverage(["40", "42", "7"], tiles, "production")).toEqual({
      tiles: [tiles[0], tiles[1]], unknown: ["7"], missing: ["f200w-0042"], stale: ["f200w-0040"],
    });
    expect(specsCoveringAll(tiles, ["production", "rbf", "mean"])).toEqual(["rbf"]);
    expect(specsCoveringAll([], ["rbf"])).toEqual([]);
  });

  it("names renders", () => {
    const run: PlateRun = { tag: "t", updated: null, files: [], renders: [
      { band: "temp", model: null, legacy: true, sheet: "s.png", tiles: [] },
      { band: "VIS", model: "rbf", model_short: "RBF", sheet: null, tiles: [] },
    ] };
    expect(renderTitle(run.renders[0])).toBe("temperature · legacy SR");
    expect(renderTitle(run.renders[1])).toBe("VIS · RBF");
    expect(renderKey(run.renders[1])).toBe("VIS~rbf");
    expect(findRender(run, "VIS~rbf")).toBe(run.renders[1]);
    expect(findRender(run, "nope")).toBe(run.renders[0]);
    expect(findRender(undefined, "x")).toBeUndefined();
  });
});

describe("full-size view", () => {
  it("pluralises the grid size", () => {
    expect(gridSizeText(6, 1)).toBe("6 rows × 1 column");
    expect(gridSizeText(1, 3)).toBe("1 row × 3 columns");
  });

  it("gives the image more height than the in-page preview it opens from", () => {
    // the app top bar (tokens.css --topbar-h)
    const topbar = 48;
    for (const vh of [560, 640, 720, 768, 800, 900, 1080, 1440]) {
      for (const stacked of [false, true]) {
        const preview = previewPaperCap(vh, topbar, stacked);
        expect(lightboxStageHeight(vh)).toBeGreaterThan(preview);
        // a crop's panel row costs one more line and still wins
        expect(lightboxStageHeight(vh, { panels: true })).toBeGreaterThan(preview);
      }
    }
    // the dialog is the window minus a thin inset, the stage minus one header row
    expect(lightboxStageHeight(768)).toBe(768 - 2 * LIGHTBOX.inset - LIGHTBOX.head);
  });
});

describe("gallery filter", () => {
  it("matches every word against label, id, source and tiers", () => {
    const r = result({ label: "Tile 42 core", logical_tiers: ["dirty", "sr", "jwst"] });
    expect(galleryMatches(r, "")).toBe(true);
    expect(galleryMatches(r, "  tile  JWST ")).toBe(true);
    expect(galleryMatches(r, "vr-1")).toBe(true);
    expect(galleryMatches(r, "tile hr")).toBe(false);
  });
});
