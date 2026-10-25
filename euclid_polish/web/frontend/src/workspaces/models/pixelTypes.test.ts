/* The pixel types' pure model: the listing normalised, a figure's file per
 * field set × band × grouping, the labels, the models a comparison can take
 * and whether two of them pair their pixels, and an analysis' band facts. */
import { describe, expect, it } from "vitest";
import type { KneeRunSummary } from "./kneesApi";
import {
  analyseEstimate, analysisLabel, bandFacts, bandsWithFile, compareFigure, comparisonLabel, comparisonSpecs, modelRefs,
  normalisePixelReports, pairCheck, parseBasis, parseCompareFigure, parsePixelFigure, pixelFigure,
} from "./pixelTypes";

const RUN = "20261025-070221-7a9c27";

describe("pixel types model", () => {
  it("normalises the listing, dropping rows without a folder and unknown kinds", () => {
    const r = normalisePixelReports({
      analyses: [{ folder: `${RUN}/universal__member_249`, run_id: RUN, model: "universal:member_249", created: "2026-10-25T07:40:00Z",
        kinds: ["synthetic", "real", "weird"], files: ["knee_map_synthetic.png"], params: { synthetic: 30, sampling: "mixed" } },
      { run_id: RUN }],
      comparisons: [{ folder: "compare_member_249_vs_member_238", models: { member_249: "universal:member_249", member_238: "universal:member_238" },
        runs: { member_249: RUN }, created: "x", kinds: ["synthetic"], files: [], tag: null }],
    });
    expect(r.analyses).toHaveLength(1);
    expect(r.analyses[0].kinds).toEqual(["synthetic", "real"]);
    expect(r.comparisons[0].tag).toBe("");
    expect(comparisonSpecs(r.comparisons[0])).toEqual(["universal:member_249", "universal:member_238"]);
    expect(normalisePixelReports(null)).toEqual({ analyses: [], comparisons: [] });
  });

  it("names each figure's file by field set, band and grouping", () => {
    expect(pixelFigure("groups").file("synthetic", "VIS", "learned")).toBe("groups_learned_synthetic_VIS.png");
    expect(pixelFigure("groups").file("real", "H_E", "raw")).toBe("groups_real_H_E.png");
    expect(pixelFigure("features").file("synthetic", "VIS", "raw")).toBe("group_features_synthetic.png");
    expect(pixelFigure("map").file("real", "Y_E", "learned")).toBe("knee_map_real.png");
    expect(pixelFigure("types").file("synthetic", "J_E", "learned")).toBe("types_synthetic_J_E.png");
    expect(compareFigure("pairs").file("synthetic", "H_E", "learned")).toBe("compare_pairs_synthetic_H_E.png");
    expect(compareFigure("diff").file("real", "VIS", "learned")).toBe("compare_share_diff_real.png");
    expect(parsePixelFigure("look")).toBe("look");
    expect(parsePixelFigure("nope")).toBe("groups");
    expect(parseCompareFigure("nope")).toBe("curves");
    expect(parseBasis("raw")).toBe("raw");
    expect(parseBasis("x")).toBe("learned");
    const files = ["types_synthetic_VIS.png", "types_synthetic_H_E.png"];
    expect(bandsWithFile(files, (b) => `types_synthetic_${b}.png`, ["VIS", "Y_E", "J_E", "H_E"])).toEqual(["VIS", "H_E"]);
  });

  it("labels analyses and comparisons", () => {
    expect(analysisLabel({ folder: "f", run_id: RUN, model: "universal:member_249", created: "", kinds: [], files: [],
      params: { synthetic: 30, tiles: ["a", "b"], sampling: "mixed" } }, "25 Oct 07:40")).toBe("#249 · 25 Oct 07:40 · 30 synthetic + 2 real · mixed");
    expect(analysisLabel({ folder: "f", run_id: RUN, model: "universal:member_249", created: "", kinds: [], files: [],
      params: { synthetic: 3, sampling: "class" } }, "now")).toBe("#249 · now · 3 synthetic");
    expect(comparisonLabel({ folder: "c", models: { member_245: "universal:member_245", member_244: "universal:member_244" }, runs: {},
      created: "", tag: "floor", kinds: [], files: [] }, "now")).toBe("#244 vs #245 · now · floor");
    expect(analyseEstimate(1)).toBe("local job: 1 model, ~2 min");
    expect(analyseEstimate(4)).toBe("local job: 4 models, ~8 min");
  });

  it("lists every model of the finished runs as a comparison side, and checks a pair", () => {
    const runs: KneeRunSummary[] = [
      { id: "r3", status: "running", models: ["universal:member_250"], synthetic: 1, tiles: [], seed: 0 },
      { id: "r2", status: "done", models: ["universal:member_244", "universal:member_245"], synthetic: 15, tiles: [], seed: 0,
        sampling: "mixed", version: 5 },
      { id: "r1", status: "done", models: ["universal:member_238"], synthetic: 30, tiles: [], seed: 7, sampling: "mixed", version: 5 },
      { id: "r0", status: "failed", models: ["universal:member_1"], synthetic: 1, tiles: [] },
    ];
    const refs = modelRefs(runs, (r) => r.id);
    expect(refs.map((r) => r.value)).toEqual(["r2|universal:member_244", "r2|universal:member_245", "r1|universal:member_238"]);
    expect(refs[0].label).toBe("#244 · r2 · seed 0");
    expect(pairCheck(refs[0], refs[1])).toEqual({ ok: true, notes: [] });
    expect(pairCheck(refs[0], refs[0]).error).toMatch(/two different/);
    expect(pairCheck(refs[0], refs[2]).error).toMatch(/different seeds \(0 and 7\)/);
    expect(pairCheck(null, refs[0]).ok).toBe(false);
    const classRun = { ...refs[1], runId: "r9", value: "r9|universal:member_245", sampling: "class", version: 4, seed: 0 };
    const notes = pairCheck(refs[0], classRun).notes.join(" ");
    expect(notes).toMatch(/class-sampled/);
    expect(notes).toMatch(/version 5/);
  });

  it("reads an analysis' facts for one field set and band", () => {
    const summary = { synthetic: { VIS: { n: 36921, noise_sigma_e: 20.31, blank_fraction: 0, types: [{}, {}, {}] } } };
    expect(bandFacts(summary, "synthetic", "VIS")).toEqual({ n: 36921, sigma: 20.31, blank: 0, types: 3 });
    expect(bandFacts(summary, "real", "VIS")).toBeNull();
    expect(bandFacts(null, "synthetic", "H_E")).toBeNull();
  });
});
