import { describe, expect, it } from "vitest";
import type { ExperimentRecord, ModelSpecRow, TileRow } from "./api";
import { atlasHref, experimentsHref, splitRef, URLS } from "./api";
import {
  bandSeries, cardDelta, cardFiles, evalHeadline, cardHeadline, cardViewerTiers, defaultExperimentId, defaultRunSpecs, experimentCost, experimentCostText,
  experimentMarkdown, formatMetric, groupModels, headlineSpec, isUniversalSpec, membersText, metricRows, outputOrigin, productionMembersText,
  metricsPlan, num, parseRefs, productionModelLabel, productionUniversalSpec, realTileViewerParams, recordSpecs, seriesDomain, sortSpecs, specRank,
  specShort, specWords, universalDetails, universalDetailText,
} from "./model";

const tile = (ref: string, state?: string, models: TileRow["models"] = {}): TileRow => {
  const [source, id] = splitRef(ref);
  return { source, id, ref, label: ref, ra: 1, dec: 2, production_state: state, models };
};

const RECORD: ExperimentRecord = {
  id: "20260926-101010-abcdef", label: "poster core", status: "done",
  tiles: ["poster/p1", "nexus/f200w-0040"], models: ["member:member_192", "production", "gate:26m"],
  skipped: { "gate:26m": "member 171 archived" },
  model_labels: { production: "Production · spatial gate", "member:member_192": "Member 192·psnr" },
  summary: {
    production: {
      bands: ["VIS", "Y_E"], per_band: {
        VIS: { hole_pct: 5.2, pct_R_lt_0p8: 10, median_R: 0.91, flux_ratio: 0.998 },
        Y_E: { hole_pct: 6.1, median_R: 0.95 },
      }, summary: { pct_R_lt_0p8: 12.5, median_R: 0.93 },
    },
    "member:member_192": { bands: ["VIS"], per_band: { VIS: { hole_pct: 2.0, flux_ratio: 1.0 } }, summary: {} },
  },
  results: {
    "poster/p1": { production: { state: "computed", metrics: { bands: ["VIS"], per_band: { VIS: { hole_pct: 7 } } } } },
  },
  errors: { "nexus/f200w-0040|production": "RuntimeError: boom" },
};

describe("value helpers", () => {
  it("reads numbers from JSON and CSV strings", () => {
    expect(num(3)).toBe(3);
    expect(num("0.67")).toBe(0.67);
    expect(num("")).toBeNull();
    expect(num("nan")).toBeNull();
    expect(num(Infinity)).toBeNull();
    expect(num(null)).toBeNull();
  });
  it("formats metrics with their precision", () => {
    expect(formatMetric("hole_pct", 5.234)).toBe("5.2");
    expect(formatMetric("median_R", "0.91234")).toBe("0.912");
    expect(formatMetric("n_peaks", 12)).toBe("12");
    expect(formatMetric("hole_pct", null)).toBe("—");
  });
});

describe("refs and URLs", () => {
  it("splits refs at the first slash", () => {
    expect(splitRef("nexus/f200w-0001")).toEqual(["nexus", "f200w-0001"]);
    expect(splitRef("bare")).toEqual(["bare", ""]);
  });
  it("builds C9 URLs with encoded ids", () => {
    expect(URLS.card("eval/102_NEG1")).toBe("/api/real/eval/102_NEG1");
    expect(URLS.image("tile/ra1_dec2", "m:production", "VIS")).toBe(
      "/api/real/tile/ra1_dec2/image.fits?tier=m%3Aproduction&band=VIS");
    expect(URLS.deleteOutputs("nexus/f200w-0001")).toBe("/api/real/nexus/f200w-0001/delete-outputs");
  });
  it("links to the atlas and to Compare", () => {
    // the atlas card of a real tile is the `tile:` kind (the one real-tile card, atlas-highlighted)
    expect(atlasHref(268.4, 65.2, "nexus/f200w-0001")).toBe(
      "/sky/atlas?ra=268.400000&dec=65.200000&inspect=tile%3Anexus%2Ff200w-0001");
    expect(experimentsHref(["a/1", "b/2"])).toBe("/sky/compare?tiles=a%2F1%2Cb%2F2");
    expect(experimentsHref([])).toBe("/sky/compare");
  });
  it("parses pasted refs, keeping known sources once", () => {
    expect(parseRefs("nexus/f200w-0001, tile/x\npair/p1 nexus/f200w-0001 bogus/1 nope eval/a/b"))
      .toEqual(["nexus/f200w-0001", "tile/x", "pair/p1"]);
  });
});

describe("model specs", () => {
  it("orders production, universal models, mean, rbf, gates, members (natural)", () => {
    expect(sortSpecs(["member:member_10", "gate:b", "rbf", "member:member_2", "production", "mean", "gate:a"]))
      .toEqual(["production", "mean", "rbf", "gate:a", "gate:b", "member:member_2", "member:member_10"]);
    // universal models right after production, so Compare's first viewer tiers reach them
    expect(sortSpecs(["member:member_196", "universal:member_211", "mean", "gate:a", "production", "universal:member_206", "mystery"]))
      .toEqual(["production", "universal:member_206", "universal:member_211", "mean", "gate:a", "member:member_196", "mystery"]);
    expect([specRank("production"), specRank("universal:member_211"), specRank("milestone:member_257@200000"),
      specRank("mean"), specRank("rbf"), specRank("gate:x"), specRank("member:member_1"), specRank("other")])
      .toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(sortSpecs(["milestone:member_257@2000000", "milestone:member_257@200000", "member:member_1",
      "milestone:member_257@1000000", "production"])).toEqual([
      "production", "milestone:member_257@200000", "milestone:member_257@1000000", "milestone:member_257@2000000",
      "member:member_1"]);
  });

  it("names a milestone by its member and step", () => {
    expect(specShort("milestone:member_257@200000")).toBe("257@0.2M");
    expect(specShort("milestone:member_257@15000000")).toBe("257@15M");
    expect(specWords("milestone:member_257@1000000")).toBe("milestone 257 @ 1M");
    expect(specWords("milestone:member_257@50000")).toBe("milestone 257 @ 50k");
    expect(groupModels([
      { spec: "milestone:member_257@200000", kind: "milestone", label: "Milestone 257 @ 0.2M", available: true },
      { spec: "universal:member_249", kind: "universal", label: "Universal 249", available: true },
    ]).map((g) => [g.id, g.items.map((i) => i.spec)])).toEqual([
      ["universal", ["universal:member_249"]], ["milestone", ["milestone:member_257@200000"]],
    ]);
  });
  it("shortens specs", () => {
    expect(specShort("member:member_196")).toBe("m196");
    expect(specShort("gate:26m")).toBe("gate 26m");
    expect(specShort("production")).toBe("production");
    expect(specShort("universal:member_211")).toBe("u211");
    expect(specShort("universal:211")).toBe("u211");
  });
  it("says specs in words", () => {
    expect(specWords("universal:member_211")).toBe("universal 211");
    expect(specWords("member:member_196")).toBe("member 196");
    expect(specWords("gate:p20")).toBe("gate p20");
    expect(specWords("mean")).toBe("member mean");
  });
  it("groups the catalogue and drops unavailable selections", () => {
    const models: ModelSpecRow[] = [
      { spec: "production", kind: "production", label: "P", available: true },
      { spec: "mean", kind: "mean", label: "M", available: true },
      { spec: "rbf", kind: "rbf", label: "R", available: false, reason: "stale" },
      { spec: "member:member_1", kind: "member", label: "m1", available: true },
      { spec: "gate:x", kind: "gate", label: "gx", available: false, reason: "member gone" },
    ];
    expect(groupModels(models).map((g) => [g.id, g.items.map((i) => i.spec)])).toEqual([
      ["production", ["production"]], ["mean", ["mean"]], ["rbf", ["rbf"]],
      ["gate", ["gate:x"]], ["member", ["member:member_1"]],
    ]);
  });
  it("groups universal models right after production; a universal member stays in Members too", () => {
    const models: ModelSpecRow[] = [
      { spec: "production", kind: "production", label: "P", available: true },
      { spec: "mean", kind: "mean", label: "M", available: true },
      { spec: "member:member_1", kind: "member", label: "m1", available: true },
      { spec: "member:member_196", kind: "member", label: "m196", available: true, details: { universal: true, class: "member" } },
      { spec: "gate:x", kind: "gate", label: "gx", available: true },
      { spec: "universal:member_211", kind: "universal", label: "Universal 211", available: true, details: { universal: true, class: "universal" } },
      { spec: "universal:member_209", kind: "universal", label: "Universal 209", available: false, reason: "no checkpoint" },
    ];
    const groups = groupModels(models);
    expect(groups.map((g) => [g.id, g.items.map((i) => i.spec)])).toEqual([
      ["production", ["production"]],
      ["universal", ["member:member_196", "universal:member_211", "universal:member_209"]],
      ["mean", ["mean"]], ["gate", ["gate:x"]], ["member", ["member:member_1", "member:member_196"]],
    ]);
    expect(groups.find((g) => g.id === "universal")?.label).toBe("Universal models");
    // a member without universal details is not a universal model
    expect(groupModels([{ spec: "member:member_197", kind: "member", label: "m197", available: true, details: { universal: false } }])
      .map((g) => g.id)).toEqual(["member"]);
  });
  it("describes a universal model in one line: depth, steps, knee, training time", () => {
    const d = { universal: true, class: "universal", num_res_blocks: 16, steps: 1_000_000, target_steps: 1_000_000, knee: "learned", train_hours: 12.7 };
    expect(universalDetailText({ details: d })).toBe("depth 16 · 1.0M steps · learned knee · 12.7 h");
    // unfinished: the step it stopped at against its target; a fixed knee says its value
    expect(universalDetailText({ details: { ...d, steps: 620_000, knee: "fixed", output_knee: 10, train_hours: 0.04 } }))
      .toBe("depth 16 · 620k of 1.0M steps · fixed knee 10 e⁻ · 0.04 h");
    // an active ensemble member that is universal says so
    expect(universalDetailText({ details: { ...d, class: "member", steps: 300_000, target_steps: null, train_hours: null } }))
      .toBe("depth 16 · 300k steps · learned knee · ensemble member");
    expect(universalDetailText({ kind: "universal", details: { universal: true } })).toBe("");
    expect(universalDetailText({ kind: "gate", details: { mix_space: "linear" } })).toBe("");
    expect(universalDetailText({})).toBe("");
  });
  it("names the model production runs, and the universal spec it already covers", () => {
    expect(productionModelLabel({ label: "Production · Universal 238" })).toBe("Universal 238");
    expect(productionModelLabel({ label: "Production · spatial gate (convolutional, convex)" })).toBe("spatial gate (convolutional, convex)");
    expect(productionUniversalSpec([{ kind: "mean", details: {} }, { kind: "production", details: { universal: true, model_spec: "universal:member_238" } }]))
      .toBe("universal:member_238");
    expect(productionUniversalSpec([{ kind: "production", details: { mix_space: "linear" } }])).toBeNull();
    expect(productionUniversalSpec([{ kind: "production", details: { model_spec: "gate:p20" } }])).toBeNull();
    expect(productionUniversalSpec([])).toBeNull();
  });
  it("reads typed universal details only from a universal row", () => {
    expect(universalDetails({ kind: "universal", details: { universal: true, class: "universal", num_res_blocks: 16, knee: "learned" } }))
      .toEqual({ universal: true, class: "universal", num_res_blocks: 16, knee: "learned" });
    // an unknown knee word is dropped, not passed on
    expect(universalDetails({ kind: "universal", details: { knee: "sideways", steps: 5 } })).toEqual({ universal: true, class: "universal", steps: 5 });
    // kind universal without details is still universal (an older or partial row)
    expect(universalDetails({ kind: "universal" })).toMatchObject({ universal: true, class: "universal" });
    expect(universalDetails({ kind: "member", details: { universal: true, class: "member" } })?.class).toBe("member");
    expect(universalDetails({ kind: "member", details: {} })).toBeNull();
    expect(universalDetails({ kind: "gate" })).toBeNull();
    expect(isUniversalSpec("universal:member_211")).toBe(true);
    expect(isUniversalSpec("member:member_196")).toBe(false);
  });
  it("counts the members a model reads (a pruned gate reads fewer than it was fitted on)", () => {
    const six = ["170·psnr", "171·psnr", "180·psnr", "181·psnr", "184·psnr", "187·psnr"];
    expect(membersText({ n_members: 6, n_fitted: 20, reads: six })).toBe("6 of 20 members");
    expect(membersText({ n_members: 20, reads: six })).toBe("6 of 20 members"); // older servers
    expect(membersText({ n_members: 30, reads: Array(30).fill("x") })).toBe("30 members");
    expect(membersText({ n_members: 1 })).toBe("1 member");
    expect(membersText({ members: ["a", "b"] })).toBe("2 members");
    expect(membersText({})).toBe("");
  });
  it("says how many members production runs, with the pruning rule when recorded", () => {
    const twenty = Array.from({ length: 20 }, (_, i) => `${170 + i}·psnr`);
    expect(productionMembersText({ n_fitted: 30, reads: twenty, details: { prune_threshold: 0.005 } }))
      .toBe("Runs 20 of 30 members: those with ≥ 0.5% of the gate's weight somewhere");
    expect(productionMembersText({ n_fitted: 30, reads: twenty })).toBe("Runs 20 of 30 members: the ones the gate reads");
    expect(productionMembersText({ n_members: 30, reads: Array(30).fill("x") })).toBe("Runs all 30 members");
    expect(productionMembersText({})).toBe("");
  });
});

describe("experiment cost", () => {
  const CAT: ModelSpecRow[] = [
    { spec: "production", kind: "production", label: "P", available: true, n_members: 3, reads: ["1·psnr", "2·psnr", "3·psnr"] },
    { spec: "mean", kind: "mean", label: "M", available: true, n_members: 3, members: ["1·psnr", "2·psnr", "3·psnr"] },
    { spec: "gate:pruned", kind: "gate", label: "G", available: true, n_members: 3, reads: ["2·psnr", "4·psnr"] },
    { spec: "member:member_5", kind: "member", label: "m5", available: true, n_members: 1, reads: ["5·psnr"] },
    { spec: "rbf", kind: "rbf", label: "R", available: false, reason: "stale", n_members: 2, reads: ["8·psnr", "9·psnr"] },
  ];
  it("counts every member SR the models need once per tile (union), and the outputs", () => {
    expect(experimentCost(["production", "gate:pruned"], CAT, 2)).toEqual({
      tiles: 2, models: 2, members: 4, inferences: 8, outputs: 4, skipped: [], unknown: [],
    });
    expect(experimentCost(["mean", "member:member_5"], CAT, 1)).toMatchObject({ members: 4, inferences: 4, outputs: 2 });
  });
  it("leaves out unavailable specs (the server skips them) and names unknown ones", () => {
    expect(experimentCost(["rbf", "production", "gate:new"], CAT, 1)).toMatchObject({
      models: 1, members: 3, outputs: 1, skipped: ["rbf"], unknown: ["gate:new"],
    });
  });
  it("says the cost in words", () => {
    expect(experimentCostText(experimentCost(["production", "gate:pruned"], CAT, 2)))
      .toBe("4 outputs (2 models on 2 tiles). Needs 4 member SRs per tile: at most 8 member inferences on this machine; cached ones are reused.");
    expect(experimentCostText(experimentCost(["member:member_5"], CAT, 1)))
      .toBe("1 output (1 model on 1 tile). Needs 1 member SR per tile: at most 1 member inference on this machine; cached ones are reused.");
    expect(experimentCostText({ ...experimentCost(["production"], CAT, 2), members: 0, inferences: 0 }))
      .toBe("2 outputs (1 model on 2 tiles).");
    expect(experimentCostText(null)).toBe("");
    // a sentence, not an "A · B" label string
    expect(experimentCostText(experimentCost(["production"], CAT, 2))).not.toMatch(/·|×/);
  });
});

describe("the real-tile card", () => {
  it("asks the viewer for exactly the tile's own model tiers", () => {
    expect(realTileViewerParams("nexus", ["rbf", "production"])).toEqual({ source: "nexus", models: "production,rbf" });
    // no model output: an empty list (",") — without `models` the server lists every spec of the source
    expect(realTileViewerParams("poster", [])).toEqual({ source: "poster", models: "," });
  });
  it("opens on two frames, LR and the first model output (JWST one chip away), so they are large in the inspector", () => {
    expect(cardViewerTiers(["production", "mean", "rbf"], false)).toEqual(["lr", "m:production"]);
    expect(cardViewerTiers(["rbf", "production"], true)).toEqual(["lr", "m:production"]);
    // no model output yet: LR and the JWST truth when there is one
    expect(cardViewerTiers([], true)).toEqual(["lr", "jwst"]);
    expect(cardViewerTiers([], false)).toEqual(["lr"]);
    // a universal model ranks right after production
    expect(cardViewerTiers(["mean", "universal:member_211"], false)).toEqual(["lr", "m:universal:member_211"]);
  });
  it("says where an output came from: a universal model is one standalone network, not \"1 member\"", () => {
    expect(outputOrigin({ member_labels: ["211·psnr"], kind: "universal" })).toBe("universal model");
    // the spec prefix stands in when the server sends no kind
    expect(outputOrigin({ spec: "universal:member_211", member_labels: ["211·psnr"] })).toBe("universal model");
    expect(outputOrigin({ spec: "member:member_196", kind: "member", member_labels: ["196·psnr"] })).toBe("1 member");
    expect(outputOrigin({ member_labels: ["a", "b"], combiner_kind: "spatial_gate" })).toBe("2 members · spatial gate");
  });
  it("proposes the production / mean outputs a tile is missing", () => {
    expect(defaultRunSpecs({})).toEqual(["production", "mean"]);
    expect(defaultRunSpecs({ production: { state: "current" } })).toEqual(["mean"]);
    expect(defaultRunSpecs({ production: { state: "stale" }, mean: { state: "current" } })).toEqual(["production"]);
    expect(defaultRunSpecs({ production: { state: "current" }, mean: { state: "current" } })).toEqual(["production", "mean"]);
  });
});

describe("real tiles", () => {
  it("picks the headline spec: the first scored output in catalogue order", () => {
    const row = tile("nexus/a", "stale", {
      "member:member_3": { state: "current" }, rbf: { state: "current", legacy: true, summary: { median_R: 1 } },
      production: { state: "stale" },
    });
    expect(headlineSpec(row)).toBe("rbf");
    expect(headlineSpec(tile("x/y"))).toBeNull();
  });
});

describe("experiment metrics", () => {
  it("collects every computed spec (not the skipped ones)", () => {
    expect(recordSpecs(RECORD)).toEqual(["production", "member:member_192"]);
  });
  it("tabulates model × band, pooled or per tile", () => {
    const pooled = metricRows(RECORD, "pooled");
    expect(pooled.map((r) => r.key)).toEqual(["production|VIS", "production|Y_E", "member:member_192|VIS"]);
    expect(pooled[0]).toMatchObject({ spec: "production", band: "VIS", hole_pct: 5.2, label: "Production · spatial gate" });
    const one = metricRows(RECORD, "poster/p1");
    expect(one.map((r) => [r.key, r.hole_pct])).toEqual([["production|VIS", 7]]);
    expect(metricRows(RECORD, "nope/none")).toEqual([]);
  });
  it("builds per-band series with per-model offsets", () => {
    const series = bandSeries(RECORD, "pooled", "hole_pct");
    expect(series.map((s) => s.spec)).toEqual(["production", "member:member_192"]);
    expect(series[0].y).toEqual([5.2, 6.1, null, null]);
    expect(series[1].y).toEqual([2, null, null, null]);
    expect(series[0].x[0]).toBeLessThan(0);
    expect(series[1].x[0]).toBeGreaterThan(0);
    const one = bandSeries({ ...RECORD, models: ["production"], summary: { production: RECORD.summary!.production }, results: {} }, "pooled", "hole_pct");
    expect(one[0].x).toEqual([0, 1, 2, 3]);
    // production against many universal models: the dots spread wider (still inside the band's slot)
    const many = ["production", ...Array.from({ length: 13 }, (_, i) => `universal:member_${206 + i}`)];
    const wide = bandSeries({ id: "w", models: many, summary: {} }, "pooled", "hole_pct");
    const span = wide.at(-1)!.x[0] - wide[0].x[0];
    expect(span).toBeGreaterThan(0.36);
    expect(span).toBeLessThanOrEqual(0.6);
    expect(wide.map((s) => s.spec).slice(0, 2)).toEqual(["production", "universal:member_206"]);
  });
  it("pads chart domains: % from zero, ratios around one", () => {
    expect(seriesDomain(bandSeries(RECORD, "pooled", "hole_pct"), "hole_pct")[0]).toBe(0);
    const [lo, hi] = seriesDomain(bandSeries(RECORD, "pooled", "median_R"), "median_R");
    expect(lo).toBeLessThan(0.91);
    expect(hi).toBeGreaterThan(1);
    expect(seriesDomain([], "hole_pct")).toEqual([0, 1]);
  });
  it("finds the domain of a very long series without spreading it into Math.min", () => {
    const y = Array.from({ length: 200_000 }, (_, i) => (i % 7) / 10);
    expect(seriesDomain([{ spec: "p", label: "p", x: [], y }], "flux_ratio")[1]).toBeGreaterThan(1);
  });
  it("summarises the experiment for the tracking log", () => {
    const md = experimentMarkdown(RECORD);
    expect(md).toContain("**Real-data experiment `20260926-101010-abcdef`** — poster core (done)");
    expect(md).toContain("- tiles (2): `poster/p1`, `nexus/f200w-0040`");
    expect(md).toContain("`gate:26m` (member 171 archived)");
    expect(md).toContain("| `production` | 5.2 / 6.1 / — / — | 12.5 | 0.930 | 0.998 |");
    expect(md).toContain("Errors (1)");
    expect(md.startsWith("#")).toBe(false);          // the store adds the heading
  });
});

describe("metricsPlan", () => {
  it("groups the tiles by the current outputs that still lack metrics (never an SR a tile does not have)", () => {
    const rows = [
      tile("nexus/a", "current", { production: { state: "current" }, mean: { state: "current", summary: { hole_pct_max: 1 } } }),
      tile("nexus/b", "current", { production: { state: "current" } }),
      tile("poster/p", "stale", { production: { state: "stale" }, rbf: { state: "current", legacy: true } }),
      tile("nexus/c", "current", { production: { state: "current", summary: { hole_pct_max: 2 } } }),
    ];
    expect(metricsPlan(rows)).toEqual([
      { specs: ["production"], refs: ["nexus/a", "nexus/b"] },
      { specs: ["rbf"], refs: ["poster/p"] },
    ]);
    expect(metricsPlan([rows[3]])).toEqual([]);
  });
});

describe("the experiment a visit opens", () => {
  const H = [
    { id: "20260925-101010-aaaaaa", created: "2026-09-25T10:10:10" },
    { id: "20260927-024251-0fb0c1", created: "2026-09-27T02:42:51" },
    { id: "20260926-090000-bbbbbb", created: "2026-09-26T09:00:00" },
  ];
  it("is the ?exp= one when the URL names it", () => {
    expect(defaultExperimentId(H, "20260925-101010-aaaaaa", [])).toBe("20260925-101010-aaaaaa");
  });
  it("is the newest one on a plain visit (the tab strip's link)", () => {
    expect(defaultExperimentId(H, "", [])).toBe("20260927-024251-0fb0c1");
    // no `created`: the id's timestamp orders them
    expect(defaultExperimentId(H.map(({ id }) => ({ id })), "", [])).toBe("20260927-024251-0fb0c1");
  });
  it("is none when tiles were handed over (the form is the point) or nothing exists", () => {
    expect(defaultExperimentId(H, "", ["nexus/f200w-0040"])).toBe("");
    expect(defaultExperimentId([], "", [])).toBe("");
  });
});

describe("the tile card's headline, Δm footer and files", () => {
  const CARD = {
    source: "poster", id: "new4", ref: "poster/new4", label: "Poster", ra: 273.2, dec: 68.4, extras: {},
    models: {
      production: { label: "Production · spatial gate", metrics: {
        per_band: { VIS: { hole_pct: 2.1, flux_ratio: 0.31 }, Y_E: { hole_pct: 5.8 }, J_E: { hole_pct: 6.6 }, H_E: { hole_pct: 4 } },
        summary: { hole_pct_max: 6.6, median_R: 1.23 } } },
      "gate:p20": { label: "Gate variant · p20" },
    },
    files: { lr: "poster/new4_lr.fits", "m:production": "real_outputs/poster/new4/production.fits" },
  };

  it("names two headline metrics of the model the card shows: worst-band holes (with its band) and median R", () => {
    expect(cardHeadline(CARD, "production")).toEqual({
      spec: "production", label: "production",
      facts: [
        { label: "Holes, worst band (J)", value: "6.6", unit: "%", hint: expect.stringContaining("brightest 1 %") },
        { label: "Median R", value: "1.230", hint: expect.stringContaining("1 = flux conserved") },
      ],
      note: null,
    });
    // no bright peak on the tile: no median R; the lowest NISP band's flux stands in, and a note says why
    const noPeaks = { models: { production: { metrics: {
      summary: { hole_pct_max: 77.5, median_R: null, n_peaks: 0 },
      per_band: { VIS: { hole_pct: 70, flux_ratio: 0.3 }, Y_E: { hole_pct: 77.5, flux_ratio: 0.45 }, H_E: { hole_pct: 73, flux_ratio: 0.39 } },
    } } } };
    expect(cardHeadline(noPeaks, "production")).toEqual({
      spec: "production", label: "production",
      facts: [
        { label: "Holes, worst band (Y)", value: "77.5", unit: "%", hint: expect.any(String) },
        { label: "Flux SR/LR, lowest NISP band (H)", value: "0.39", hint: expect.any(String) },
      ],
      note: expect.stringContaining("no bright peak"),
    });
    expect(cardHeadline(CARD, "gate:p20")).toBeNull();          // not scored: no numbers to show
    expect(cardHeadline(CARD, null)).toBeNull();
  });

  it("puts the VIS Δm against the LR in the footer, warned beyond 0.1 mag", () => {
    expect(cardDelta(CARD, "production")).toEqual({ label: "production", text: "Δm +1.27 (flux ×0.31)", warn: true });
    expect(cardDelta(CARD, "gate:p20")).toBeNull();
    // a catalogue object carries its SR's flux ratio itself, named by what made it
    const evalCard = { source: "eval", models: {}, extras: { flux_ratio_sr_over_lr: 0.97 } };
    expect(cardDelta(evalCard, null, "22 members"))
      .toEqual({ label: "SR (22 members)", text: "Δm +0.03 (flux ×0.97)", warn: false });
    expect(cardDelta(evalCard, null)?.label).toBe("catalogue SR");
    expect(cardDelta(evalCard, null, "22-member ensemble (combiner not recorded)")?.label).toBe("SR (22-member ensemble)");
  });

  it("gives a catalogue object its LR and SR fluxes as the headline numbers", () => {
    expect(evalHeadline({ lr_total_e: "123456", sr_total_e: "80000" })).toEqual([
      expect.objectContaining({ label: "LR", value: "123k", unit: "e⁻" }),
      expect.objectContaining({ label: "SR", value: "80k", unit: "e⁻" }),
    ]);
    expect(evalHeadline({ lr_total_e: "", sr_total_e: null })).toBeNull();
  });

  it("lists the FITS files Files can open: the LR, then each model output", () => {
    expect(cardFiles(CARD, ["production", "gate:p20"])).toEqual([
      { key: "lr", label: "LR", href: "/files?fits=poster%2Fnew4_lr.fits" },
      { key: "m:production", label: "production", href: "/files?fits=real_outputs%2Fposter%2Fnew4%2Fproduction.fits" },
    ]);
    expect(cardFiles({ files: undefined }, ["production"])).toEqual([]);
    // a catalogue object's own evaluation SR
    expect(cardFiles({ files: { lr: "data/eval_results/o/original_stack.fits", sr: "data/eval_results/o/SR.fits" } }, []).map((f) => f.label))
      .toEqual(["LR", "SR (catalogue evaluation)"]);
  });
});
