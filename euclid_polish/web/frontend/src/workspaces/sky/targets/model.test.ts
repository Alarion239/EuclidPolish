import { describe, expect, it } from "vitest";
import type { EvalRow, TileRow } from "../results/api";
import {
  TARGET_SETS, cardStatus, evalTargets, failedCutouts, fluxStrip, legacyTargetsPatch, lensGrades, madeByEval, madeByTile, parseSets, productionPlan,
  sentenceText, setCounts, setSentence, stateCounts, stripCoverage, tileTargets, withState, type TargetRow,
} from "./model";

const tile = (ref: string, state: string | undefined, models: TileRow["models"] = {}, extras: Record<string, unknown> = {}): TileRow => {
  const [source, id] = ref.split("/");
  return { source, id, ref, label: `Tile ${id}`, ra: 268.4, dec: 65.1, field: "EDF-N", production_state: state, models, extras };
};

const evalRow = (id: string, grade: string, p: Partial<EvalRow> = {}): EvalRow => ({
  id, grade, ok: "True", out_subdir: id, ra: "57.1", dec: "-49.5", field: "EDF-S", kind: grade === "gal" ? "galaxy" : "lens",
  realtile: `eval/${id}`, state: "stale", n_members: 22, combiner_kind: null, flux_ratio_sr_over_lr: "0.67",
  state_reason: "membership changed: made by 22 member(s), the production gate is fitted for 30 now", ...p,
});

describe("target sets", () => {
  it("reads ?set= (v2) and the interim ?src= store ids, keeping only known sets in panel order", () => {
    expect(parseSets("", "")).toEqual([]);
    expect(parseSets("lenses,galaxies", "")).toEqual(["lenses", "galaxies"]);
    expect(parseSets("galaxies,nexus,bogus", "")).toEqual(["nexus", "galaxies"]);
    // the interim handover wrote the old store ids (?src=tile) into links
    expect(parseSets("", "tile,field,pair")).toEqual(["cached", "legacy", "pairs"]);
    expect(parseSets("", "eval")).toEqual(["lenses", "galaxies"]);
    expect(parseSets("poster", "tile")).toEqual(["poster"]);            // ?set= wins
    // a redirect maps whole values only: old store ids inside a ?set= list
    expect(parseSets("nexus,tile", "")).toEqual(["nexus", "cached"]);
    expect(parseSets("eval,pair,archive", "")).toEqual(["lenses", "galaxies", "pairs"]);
  });

  it("rewrites an old link once: the group list ?g= narrows the catalogue sets, ?st= becomes ?state=", () => {
    const q = (p: Partial<Parameters<typeof legacyTargetsPatch>[0]>) => legacyTargetsPatch({ set: "", src: "", g: [], st: "", state: "all", ...p });
    expect(q({})).toBeNull();
    expect(q({ set: "lenses", g: ["A"] })).toBeNull();                   // a v2 link
    expect(q({ set: "nexus,tile" })).toEqual({ set: "nexus,cached" });
    // /sky/catalog-eval?g=A,B → ?g=A,B&set=lenses,galaxies: the old page showed lenses A and B only
    expect(q({ set: "lenses,galaxies", g: ["A", "B"] })).toEqual({ set: "lenses" });
    expect(q({ set: "lenses,galaxies", g: ["A", "gal"] })).toEqual({ g: ["A"] });
    expect(q({ set: "lenses,galaxies", g: ["lensB", "galaxies"] })).toEqual({ g: ["B"] });
    expect(q({ set: "lenses", g: ["B", "gal"] })).toEqual({ g: ["B"] });   // never adds a set
    expect(q({ g: ["gal"] })).toEqual({ set: "galaxies", g: [] });
    expect(q({ set: "nexus,lenses", g: ["A"] })).toBeNull();              // not the catalogue's sets: kept
    // the old state filter (unknown is stale now)
    expect(q({ set: "lenses", st: "unknown" })).toEqual({ st: "", state: "stale" });
    expect(q({ set: "lenses", st: "all" })).toEqual({ st: "" });
    expect(q({ set: "lenses", st: "stale", state: "current" })).toEqual({ st: "" });   // ?state= wins
  });

  it("labels every set in words; Legacy field and JWST pairs sit under more", () => {
    expect(TARGET_SETS.map((s) => [s.id, s.label, !!s.more])).toEqual([
      ["nexus", "NEXUS × JWST", false], ["poster", "Poster galaxy", false], ["lenses", "Lens candidates", false],
      ["galaxies", "Q1 galaxies", false], ["cached", "Cached tiles", false], ["legacy", "Legacy field", true],
      ["pairs", "JWST pairs", true],
    ]);
  });

  it("counts each set: tile stores from /api/real/sources, catalogue sets from their reconstructions", () => {
    const rows = [evalRow("a", "A"), evalRow("b", "B"), evalRow("x", "A", { ok: "False", state: null }), evalRow("g", "gal"), evalRow("s", "syn-gal")];
    const counts = setCounts(
      { sources: [{ id: "nexus", label: "", count: 445 }, { id: "poster", label: "", count: 4 }, { id: "tile", label: "", count: 2 }] },
      rows,
    );
    // a failed cutout is not a reconstruction (it has no LR to run on): the chips count 293, not 300
    expect(counts).toEqual({ nexus: 445, poster: 4, lenses: 2, galaxies: 1, cached: 2 });
    expect(failedCutouts(rows)).toEqual({ lenses: 1, galaxies: 0 });
    // under a ?g= grade filter only the failed cutouts of those grades count, as in the table
    expect(failedCutouts(rows, ["B"])).toEqual({ lenses: 0, galaxies: 0 });
    expect(failedCutouts(rows, ["A"])).toEqual({ lenses: 1, galaxies: 0 });
  });

  it("filters lens candidates by grade (a galaxy group in ?g= is not a grade)", () => {
    expect(lensGrades(["A", "gal", "C", "syn-lens", "A"])).toEqual(["A", "C"]);
    expect(lensGrades(["lensB", "galaxies"])).toEqual(["B"]);
    expect(lensGrades([])).toEqual([]);
  });
});

describe("the eval-store adapter: one current / stale / missing vocabulary", () => {
  it("maps real tiles by their production state, with 'made by' in words and the headline flux", () => {
    const rows = tileTargets("nexus", [
      tile("nexus/f200w-0000", "stale", {
        production: { state: "stale", label: "Production · spatial gate (convolutional, convex)", summary: { hole_pct_max: 20.19, median_R: 1.18 }, flux_ratio: { VIS: 0.97, J_E: 1.3 } },
        rbf: { state: "current", legacy: true, label: "minibatched convex all-asinh RBF" },
      }, { field_id: "nf" }),
      tile("nexus/f200w-0001", "stale", { rbf: { state: "current", legacy: true, label: "minibatched convex all-asinh RBF" } }),
      tile("nexus/f200w-0002", undefined),
    ]);
    expect(rows.map((r) => [r.key, r.state, r.madeBy, r.flux, r.holes, r.medR, r.scored])).toEqual([
      ["nexus/f200w-0000", "stale", "spatial gate (convolutional, convex)", 0.97, 20.19, 1.18, true],
      ["nexus/f200w-0001", "stale", "legacy minibatched convex all-asinh RBF", null, null, null, false],
      ["nexus/f200w-0002", "missing", null, null, null, null, false],
    ]);
    expect(rows[0]).toMatchObject({ set: "nexus", ref: "nexus/f200w-0000", id: "f200w-0000", field: "EDF-N", nexusField: "nf" });
    expect(rows[1].reason).toBe("only a legacy SR exists");
    expect(rows[2].reason).toBe("no production SR yet");
  });

  it("scores a tile on its production output first, else the first scored output", () => {
    const [row] = tileTargets("poster", [tile("poster/new4", "stale", {
      "gate:p20": { state: "current", label: "Gate variant · p20", summary: { hole_pct_max: 5 }, flux_ratio: { VIS: 0.9 } },
      mean: { state: "stale", legacy: true, label: "Poster SR · mean_explicit_members (4 members)" },
    })]);
    expect(row).toMatchObject({ flux: 0.9, holes: 5, scored: true, madeBy: "legacy member mean (4 members)" });
  });

  it("maps eval objects: stale / unknown → stale, failed → missing with its error, synthetic stamps left out", () => {
    const rows = evalTargets([
      evalRow("lensA", "A"),
      evalRow("lensB", "B", { state: "current", combiner_kind: "spatial_gate", n_members: 30, state_reason: null }),
      evalRow("lensC", "C", { state: "unknown", n_members: null, state_reason: "no model recorded (members.json missing)" }),
      evalRow("bad", "A", { ok: "False", state: null, error: "RuntimeError: VIS: downloaded file is empty", realtile: null, flux_ratio_sr_over_lr: "" }),
      evalRow("gal1", "gal", { flux_ratio_sr_over_lr: "0.82" }),
      evalRow("syn1", "syn-lens", { ra: "", dec: "", realtile: null }),
    ], { failed: true });
    expect(rows.map((r) => [r.key, r.set, r.state, r.flux, r.ref])).toEqual([
      ["eval/lensA", "lenses", "stale", 0.67, "eval/lensA"],
      ["eval/lensB", "lenses", "current", 0.67, "eval/lensB"],
      ["eval/lensC", "lenses", "stale", 0.67, "eval/lensC"],
      ["eval/bad", "lenses", "missing", null, null],
      ["eval/gal1", "galaxies", "stale", 0.82, "eval/gal1"],
    ]);
    expect(rows[0].madeBy).toBe("22-member ensemble (combiner not recorded)");
    expect(rows[1].madeBy).toBe("spatial gate over 30 members");
    expect(rows[2].madeBy).toBeNull();
    expect(rows[3].reason).toBe("the cutout failed: RuntimeError: VIS: downloaded file is empty");
    expect(rows[0].grade).toBe("A");
    // eval rows are never "scored" by the real-tile metrics (no holes / R̃)
    expect(rows.every((r) => !r.scored)).toBe(true);
    // failed cutouts are shown only on request (?failed=1)
    expect(evalTargets([evalRow("lensA", "A"), evalRow("bad", "A", { ok: "False", state: null })]).map((r) => r.key)).toEqual(["eval/lensA"]);
  });

  it("words the model that made an output", () => {
    expect(madeByTile({ production: { label: "Production · member mean" } })).toBe("member mean");
    expect(madeByTile({ production: {} })).toBe("production");
    expect(madeByTile({ rbf: { legacy: true, label: "Poster SR · raw_incremental_minmeanmax_rbf (10 members)" } }))
      .toBe("legacy RBF combiner (10 members)");
    expect(madeByTile({})).toBeNull();
    // no combiner kind: a current SR is the member mean; a stale one may predate the record
    expect(madeByEval({ n_members: 1, combiner_kind: null, state: "current" })).toBe("a single member");
    expect(madeByEval({ n_members: 22, combiner_kind: null, state: "current" })).toBe("mean of 22 members");
    expect(madeByEval({ n_members: 22, combiner_kind: null, state: "stale" })).toBe("22-member ensemble (combiner not recorded)");
    expect(madeByEval({ n_members: 30, combiner_kind: "raw_incremental_minmeanmax_rbf" })).toBe("RBF combiner over 30 members");
  });
});

const T = (p: Partial<TargetRow>): TargetRow => ({
  key: p.key ?? "k", set: p.set ?? "nexus", id: "x", label: "x", ra: 1, dec: 2, field: null, state: "stale", reason: null,
  madeBy: null, flux: null, holes: null, medR: null, scored: false, grade: null, hasJwst: false, ref: p.key ?? "k",
  nexusField: null, ...p,
});

describe("per-set sentence and counts", () => {
  it("counts the states once, for the state control", () => {
    const rows = [T({ state: "current" }), T({ state: "stale" }), T({ state: "stale" }), T({ state: "missing" })];
    expect(stateCounts(rows)).toEqual({ all: 4, current: 1, stale: 2, missing: 1 });
    expect(withState(rows, "stale")).toHaveLength(2);
    expect(withState(rows, "all")).toHaveLength(4);
  });

  it("says what the set's production SR is, with the median flux over the rows that have one", () => {
    const lenses = Array.from({ length: 293 }, (_, i) => T({ key: `l${i}`, set: "lenses", flux: 0.67 }));
    const s = setSentence("lenses", lenses);
    expect(s).toEqual({
      set: "lenses", label: "Lens candidates", noun: "reconstructions", total: 293,
      current: 0, stale: 293, missing: 0, medianFlux: 0.67, fluxN: 293, fluxWarn: true, medianHoles: null, scoredN: 0,
    });
    expect(sentenceText(s, "gate")).toBe("Lens candidates: all 293 reconstructions predate the current gate · median flux SR/LR 0.67");
    const nexus = [
      ...Array.from({ length: 437 }, (_, i) => T({ key: `a${i}` })),
      ...Array.from({ length: 8 }, (_, i) => T({ key: `b${i}`, flux: 0.97, scored: true, holes: 10 + i * 0.1 })),
    ];
    // the holes the table shows only for an all-scored view ride in the sentence
    expect(sentenceText(setSentence("nexus", nexus), "gate"))
      .toBe("NEXUS × JWST: all 445 tiles predate the current gate · median flux SR/LR 0.97, median worst-band holes 10.4 % over 8 scored tiles");
    expect(sentenceText(setSentence("cached", [T({ set: "cached", state: "current", scored: true, holes: 3 }), T({ set: "cached", state: "current" })]), "gate"))
      .toBe("Cached tiles: all 2 tiles are current · worst-band holes 3.0 % over 1 scored tile");
    const cached = [T({ set: "cached", state: "missing" }), T({ set: "cached", state: "stale" }), T({ set: "cached", state: "current" })];
    expect(sentenceText(setSentence("cached", cached), "production model"))
      .toBe("Cached tiles: 1 current, 1 stale, 1 without a production SR");
    expect(sentenceText(setSentence("legacy", [T({ state: "missing" }), T({ state: "missing" })]), "gate"))
      .toBe("Legacy field: none of the 2 tiles has a production SR yet");
    expect(sentenceText(setSentence("poster", [T({ state: "current", flux: 1.02 })]), "gate"))
      .toBe("Poster galaxy: the tile is current · flux SR/LR 1.02");
    expect(sentenceText(setSentence("pairs", []), "gate")).toBe("JWST pairs: none yet");
    expect(setSentence("poster", [T({ state: "current", flux: 1.02 })]).fluxWarn).toBe(false);
  });
});

describe("flux SR/LR strip", () => {
  it("draws one row per set (dots jittered deterministically) with its median", () => {
    const rows = [T({ key: "a", set: "lenses", flux: 0.6 }), T({ key: "b", set: "lenses", flux: 0.8 }), T({ key: "c", set: "galaxies", flux: 1.1 }), T({ key: "d", set: "galaxies" })];
    const strip = fluxStrip(["lenses", "galaxies"], rows);
    expect(strip.rows.map((r) => [r.set, r.label, r.n, r.total, r.median])).toEqual([["lenses", "Lens candidates", 2, 2, 0.7], ["galaxies", "Q1 galaxies", 1, 2, 1.1]]);
    // the caption names the sets whose targets are not all plotted
    expect(stripCoverage(strip.rows)).toBe("Plotted: 1 of 2 in Q1 galaxies; the other targets have no flux yet (no scored SR).");
    expect(stripCoverage(strip.rows.slice(0, 1))).toBeNull();
    expect(strip.points.map((p) => [p.key, p.x, Math.round(p.y) || 0])).toEqual([["a", 0.6, 0], ["b", 0.8, 0], ["c", 1.1, 1]]);
    expect(strip.points.every((p) => Math.abs(p.y - Math.round(p.y)) <= 0.3)).toBe(true);
    expect(fluxStrip(["lenses", "galaxies"], rows).points).toEqual(strip.points);   // stable jitter
    expect(strip.domain[0]).toBeLessThanOrEqual(0.6);
    expect(strip.domain[1]).toBeGreaterThanOrEqual(1.1);
    expect(strip.domain[0]).toBeGreaterThanOrEqual(0);
    // a negative flux ratio (a background-dominated SR) stays in view
    expect(fluxStrip(["lenses"], [T({ key: "n", set: "lenses", flux: -0.03 })]).domain[0]).toBeLessThan(-0.03);
  });
});

describe("Run production on stale", () => {
  it("plans NEXUS field inference, an experiment for other tiles and the grouped analysis for catalogue targets", () => {
    const rows = [
      T({ key: "nexus/f200w-0000", set: "nexus", nexusField: "nf" }), T({ key: "nexus/f200w-0001", set: "nexus", nexusField: "nf", state: "current" }),
      T({ key: "poster/p1", set: "poster" }), T({ key: "tile/t1", set: "cached", state: "missing" }),
      T({ key: "eval/a", set: "lenses", grade: "A" }), T({ key: "eval/b", set: "lenses", grade: "A" }), T({ key: "eval/c", set: "lenses", grade: "B" }),
      T({ key: "eval/g", set: "galaxies", grade: "gal" }), T({ key: "eval/x", set: "lenses", grade: "A", state: "missing", ref: null }),
    ];
    const plan = productionPlan(rows, { A: 3, B: 1, gal: 7 });
    expect(plan).toEqual({
      stale: 6,
      nexus: { field: "nf", tiles: ["f200w-0000"] },
      tiles: ["poster/p1"],
      catalogue: { stale: 4, n: 3 },
    });
    // n covers every lens grade and a third of the galaxies (the grouped run draws 3N)
    expect(productionPlan(rows, { A: 3, gal: 12 }).catalogue?.n).toBe(4);
    expect(productionPlan([T({ state: "current" })], {})).toEqual({ stale: 0, nexus: null, tiles: [], catalogue: null });
  });
});

describe("the tile card's status sentence", () => {
  it("says the production SR's state and who made it, in the one vocabulary", () => {
    expect(cardStatus({ source: "poster", production_state: "stale", models: { production: { label: "Production · spatial gate (convolutional, convex)" } } })?.text)
      .toBe("Production SR is stale, made by spatial gate (convolutional, convex): made before the current production model.");
    expect(cardStatus({ source: "nexus", production_state: "stale", models: { rbf: { legacy: true, label: "minibatched convex all-asinh RBF" } } })?.text)
      .toBe("Production SR is stale, made by legacy minibatched convex all-asinh RBF: only a legacy SR exists.");
    expect(cardStatus({ source: "tile", production_state: "current", models: { production: { label: "Production · spatial gate" } } })?.text)
      .toBe("Production SR is current, made by spatial gate.");
    expect(cardStatus({ source: "tile", production_state: "missing", models: {} })?.text).toBe("No production SR yet.");
    expect(cardStatus({ source: "tile", models: {} })?.state).toBe("missing");
  });

  it("reads a catalogue object's state from its evaluation record", () => {
    const s = cardStatus({ source: "eval", production_state: "stale", models: {} }, {
      ok: "True", state: "stale", state_reason: "membership changed: made by 22 member(s), the production gate is fitted for 30 now",
      members: { member_labels: Array(22).fill("m"), combiner_kind: null },
    });
    expect(s).toMatchObject({ state: "stale", madeBy: "22-member ensemble (combiner not recorded)" });
    // the evaluation's reason already names the model: said once
    expect(s?.text).toBe("Production SR is stale: membership changed: made by 22 members, the production gate is fitted for 30 now.");
    expect(cardStatus({ source: "eval", models: {} }, { ok: "True", state: "stale", state_reason: "gate refitted", members: { member_labels: Array(22).fill("m") } })?.text)
      .toBe("Production SR is stale, made by 22-member ensemble (combiner not recorded): gate refitted.");
    expect(cardStatus({ source: "eval", models: {} }, { ok: "True", state: "current", members: { member_labels: Array(30).fill("m"), combiner_kind: "spatial_gate" } })?.text)
      .toBe("Production SR is current, made by spatial gate over 30 members.");
    expect(cardStatus({ source: "eval", models: {} }, { ok: "True", state: "unknown", state_reason: "no model recorded", members: null })?.state).toBe("stale");
    expect(cardStatus({ source: "eval", models: {} }, { ok: "False", error: "VIS: downloaded file is empty" })?.text)
      .toBe("No production SR yet: the cutout failed: VIS: downloaded file is empty.");
    expect(cardStatus({ source: "eval", models: {} }, null)).toBeNull();   // waits for the record
  });
});
