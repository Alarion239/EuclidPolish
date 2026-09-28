import { describe, expect, it } from "vitest";
import type { ExperimentRecord, TileRow } from "../results/api";
import {
  comparisonLabel, compareSentence, csvText, deltaMags, headline, headlineText, longCsv, pivotRows, pivotBest, posterRef, setRefs,
} from "./model";

const RECORD: ExperimentRecord = {
  id: "20260927-203856-6cb6ad", label: "seed vs pruning", status: "done", created: "2026-09-27T20:38:56Z",
  tiles: ["poster/new4", "nexus/f200w-0000"], models: ["gate:p20", "production", "mean"],
  model_labels: { production: "Production · spatial gate", mean: "Mean of 30", "gate:p20": "Gate variant · p20" },
  summary: {
    production: { bands: ["VIS", "Y_E", "J_E", "H_E"], per_band: {
      VIS: { hole_pct: 15.23, flux_ratio: 0.967 }, Y_E: { hole_pct: 18.68, flux_ratio: 1.077 },
      J_E: { hole_pct: 20.19, flux_ratio: 1.303 }, H_E: { hole_pct: 17.5, flux_ratio: 1.429 },
    }, summary: { hole_pct_max: 20.19 } },
    mean: { bands: ["VIS"], per_band: { VIS: { hole_pct: 32.55, flux_ratio: 0.31 } }, summary: { hole_pct_max: 32.55 } },
    "gate:p20": { bands: ["VIS"], per_band: { VIS: { hole_pct: 14.0, flux_ratio: 1.0 } }, summary: {} },
  },
  results: {
    "poster/new4": { production: { state: "computed", metrics: { per_band: { VIS: { hole_pct: 9, flux_ratio: 0.5 } } } } },
  },
};

describe("pivot: models × bands for one metric", () => {
  it("puts the reference (production) first, then the mean, then variants; one column per band", () => {
    const rows = pivotRows(RECORD, "pooled", "hole_pct");
    expect(rows.map((r) => [r.spec, r.VIS, r.Y_E, r.J_E, r.H_E])).toEqual([
      ["production", 15.23, 18.68, 20.19, 17.5],
      ["mean", 32.55, null, null, null],
      ["gate:p20", 14.0, null, null, null],
    ]);
    expect(rows[0].label).toBe("Production · spatial gate");
    expect(pivotRows(RECORD, "poster/new4", "hole_pct").map((r) => [r.spec, r.VIS])).toEqual([["production", 9]]);
  });

  it("marks the best value per band by the metric's direction", () => {
    expect(pivotBest(pivotRows(RECORD, "pooled", "hole_pct"), "hole_pct")).toEqual({ VIS: "gate:p20", Y_E: "production", J_E: "production", H_E: "production" });
    // flux: closest to 1
    expect(pivotBest(pivotRows(RECORD, "pooled", "flux_ratio"), "flux_ratio").VIS).toBe("gate:p20");
  });
});

describe("Δm per model vs LR", () => {
  it("reads each model's flux ratio of one band (pooled or one tile) as a magnitude, warning beyond 0.1", () => {
    const d = deltaMags(RECORD, "pooled", "VIS");
    expect(d.map((m) => [m.spec, m.dm != null ? Number(m.dm.toFixed(2)) : null, m.warn])).toEqual([
      ["production", 0.04, false], ["mean", 1.27, true], ["gate:p20", 0, false],
    ]);
    expect(deltaMags(RECORD, "poster/new4", "VIS").map((m) => [m.spec, Number(m.dm?.toFixed(2))])).toEqual([["production", 0.75]]);
    expect(deltaMags(RECORD, "pooled", "J_E").map((m) => m.spec)).toEqual(["production"]);
  });
});

describe("history headline", () => {
  it("names the worst-band holes of production and of the mean", () => {
    expect(headline(RECORD)).toEqual([
      { spec: "production", label: "production", holes: 20.19, band: "J" },
      { spec: "mean", label: "member mean", holes: 32.55, band: "VIS" },
    ]);
  });

  it("falls back to the model with the fewest holes when neither production nor the mean ran", () => {
    expect(headline({ summary: { "gate:p20": RECORD.summary!["gate:p20"], "gate:x": { per_band: { VIS: { hole_pct: 30 } } } } }))
      .toEqual([{ spec: "gate:p20", label: "gate p20", holes: 14, band: "VIS", best: true }]);
    expect(headlineText(RECORD)).toBe("production 20.2 % (J) · member mean 32.6 % (VIS)");
    expect(headlineText({ summary: {} })).toBe("");
  });

  it("labels a comparison by its label and date", () => {
    expect(comparisonLabel(RECORD)).toBe("seed vs pruning · 2026-09-27");
    expect(comparisonLabel({ id: "20260927-024251-0fb0c1" })).toBe("20260927-024251-0fb0c1");
  });
});

describe("the page's one sentence", () => {
  it("compares production's worst-band holes with the plain mean's", () => {
    expect(compareSentence(RECORD, "pooled")?.map((p) => p.text).join("")).toBe(
      "On 2 tiles, production leaves holes in 20.2 % of the bright pixels of its worst band (J), against 32.6 % for the member mean (VIS).");
    const pieces = compareSentence(RECORD, "pooled")!;
    expect(pieces.filter((p) => p.num).map((p) => [p.text, !!p.warn])).toEqual([["2", false], ["20.2 %", false], ["32.6 %", false]]);
  });

  it("without the mean, compares production with the best other model; per tile, names the tile", () => {
    const noMean: ExperimentRecord = { ...RECORD, summary: { production: RECORD.summary!.production, "gate:p20": RECORD.summary!["gate:p20"] } };
    const pieces = compareSentence(noMean, "pooled")!;
    expect(pieces.map((p) => p.text).join("")).toBe(
      "On 2 tiles, production leaves holes in 20.2 % of the bright pixels of its worst band (J), against 14.0 % for gate p20 (VIS), the best of 1 other model.");
    expect(pieces.find((p) => p.text === "20.2 %")?.warn).toBe(true);        // production is worse than the alternative
    expect(compareSentence(RECORD, "poster/new4")?.map((p) => p.text).join("")).toBe(
      "On poster/new4, production leaves holes in 9.0 % of the bright pixels of its worst band (VIS).");
    const onlyGate: ExperimentRecord = { ...RECORD, summary: { "gate:p20": RECORD.summary!["gate:p20"] }, models: ["gate:p20"] };
    expect(compareSentence(onlyGate, "pooled")?.map((p) => p.text).join("")).toBe(
      "On 2 tiles, gate p20 leaves holes in 14.0 % of the bright pixels of its worst band (VIS).");
    expect(compareSentence({ ...RECORD, summary: {}, results: {} }, "pooled")).toBeNull();
  });
});

describe("long-form CSV", () => {
  it("writes one row per model × band with every metric (RFC 4180, formulas defused)", () => {
    const csv = longCsv(RECORD, "pooled").split("\n");
    expect(csv[0].split(",").slice(0, 4)).toEqual(["model", "label", "band", "Hole %"]);
    expect(csv).toHaveLength(1 + 4 + 1 + 1);
    expect(csv[1].startsWith("production,Production · spatial gate,VIS,15.23,")).toBe(true);
    expect(csvText([["=cmd", "a,b", 'say "hi"', null]])).toBe(`'=cmd,"a,b","say ""hi""",`);
  });
});

describe("new comparison target sets", () => {
  const tile = (ref: string, models: string[] = [], extras: Record<string, unknown> = {}): TileRow => {
    const [source, id] = ref.split("/");
    return { source, id, ref, label: ref, ra: 1, dec: 2, extras, models: Object.fromEntries(models.map((m) => [m, { state: "current" }])) };
  };
  it("picks the poster file that holds the most model runs (all hold the same LR)", () => {
    expect(posterRef([tile("poster/combiner", ["rbf"]), tile("poster/new4", ["production", "mean", "gate:p20"]), tile("poster/m169", ["rbf"])])).toBe("poster/new4");
    expect(posterRef([])).toBeNull();
  });
  it("collects the refs of a set: the NEXUS tiles of the selection, lens candidates, galaxies", () => {
    const evals = [tile("eval/l1", [], { kind: "lens", grade: "A" }), tile("eval/g1", [], { kind: "galaxy", grade: "gal" })];
    expect(setRefs("nexus", { selection: ["nexus/f200w-0001", "poster/p", "nexus/f200w-0002"], evals, posters: [] })).toEqual(["nexus/f200w-0001", "nexus/f200w-0002"]);
    expect(setRefs("lenses", { selection: [], evals, posters: [] })).toEqual(["eval/l1"]);
    expect(setRefs("galaxies", { selection: [], evals, posters: [] })).toEqual(["eval/g1"]);
    expect(setRefs("poster", { selection: [], evals, posters: [tile("poster/new4", ["production"])] })).toEqual(["poster/new4"]);
  });
});
