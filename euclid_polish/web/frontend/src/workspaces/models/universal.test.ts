/* The pure shaping of Models › Universal: the toolbar choices, metrics, the
 * rows (a stale score placed where it was made, Δ vs production only when
 * comparable), depth colours, the plots' points and lines, the nearest point
 * to a click, what to score and pull (pullable / failed / pull again), the
 * real-field facts from the newest run that scored THIS model, the Sky ›
 * Compare deep link and the summary. */
import { describe, expect, it } from "vitest";
import type { ExperimentSummary, UniversalFasrcRow } from "./api";
import { payload, score, universalProduction, urow } from "./testing/universalFixture";
import { DEPTH_SLOTS, compareHref, isComparable, isMig, parseMetric, parseXAxis, productionLabel, productionReference, depthOrder, depthSlot, fasrcKind, fasrcSplit, hoursText, kneeWords, learnedKneesText, lengthLines, logDomain, metricValue, nearestPoint, numbersText, pointLabel, preferCurrent, realPicks, scoredFields, shapeRows, stepImplText, stepsShort, timePoints, toScore, universalStatus, universalSummary } from "./universal";

describe("values", () => {
  it("reads the band mean or one band", () => {
    const s = { integrated: [55, 64, 62, 61], mean: 60.5 };
    expect(metricValue(s, "mean")).toBe(60.5);
    expect(metricValue(s, "J_E")).toBe(62);
    expect(metricValue({ integrated: [55, 64, 62, 61] }, "mean")).toBe(60.5);
    expect(metricValue({ integrated: [55, null, 62, 61] }, "mean")).toBeNull();
    expect(metricValue(null, "VIS")).toBeNull();
  });

  it("gives each model a status: scored, stale, not scored, or no checkpoint", () => {
    expect(universalStatus(urow(1))).toBe("scored");
    expect(universalStatus(urow(1, { score: score(60, { stale: true }) }))).toBe("stale");
    expect(universalStatus(urow(1, { score: null }))).toBe("unscored");
    expect(universalStatus(urow(1, { fingerprint: null }))).toBe("no-checkpoint");
  });
});

describe("rows", () => {

  it("places a stale score at its scoring-time steps: recorded steps, else the served step, else the current ones", () => {
    const one = (patch: Parameters<typeof urow>[1]) => shapeRows(payload({ models: [urow(215, { score: score(60.8, { stale: true }), ...patch })] }),
      { metric: "mean" })[0];
    expect(one({ scored_hours: 1.2, scored_steps: 120_000 })).toMatchObject({ scoreHours: 1.2, scoreSteps: 120_000 });
    expect(one({ scored_hours: null, scored_steps: null, served_step: 150_000 })).toMatchObject({ scoreHours: 2.5, scoreSteps: 150_000 });
    expect(one({ scored_hours: undefined, scored_steps: undefined, served_step: null })).toMatchObject({ scoreHours: 2.5, scoreSteps: 200_000 });
    // a current score sits at the current hours and steps whatever was recorded
    expect(shapeRows(payload({ models: [urow(211, { scored_hours: 1, scored_steps: 1 })] }), { metric: "mean" })[0])
      .toMatchObject({ stale: false, scoreHours: 2.5, scoreSteps: 200_000 });
  });
});

describe("depth colours", () => {
  it("orders the depths shallow to deep, unknown last, and never gives one the mean's or the combiner's slot", () => {
    const order = depthOrder([{ depth: 32 }, { depth: null }, { depth: 8 }, { depth: 16 }, { depth: 32 }]);
    expect(order).toEqual([8, 16, 32, null]);
    expect(order.map((d) => depthSlot(order, d))).toEqual([1, 3, 4, 5]);
    expect(DEPTH_SLOTS).toEqual([1, 3, 4, 5, 6, 7]);
    // --cat-0 is the plain mean's blue (its y guide), --cat-2 the combiner's orange
    expect(DEPTH_SLOTS).not.toContain(0);
    expect(DEPTH_SLOTS).not.toContain(2);
  });
});

describe("plots", () => {
  const rows = shapeRows(payload(), { metric: "mean" });

  it("draws the scored, timed models and names the rest", () => {
    const t = timePoints(rows);
    expect(t.points.map((p) => p.num)).toEqual(["211", "210", "208", "215"]);
    expect(t.points[0]).toMatchObject({ x: 12.7, y: 61.6, key: "depth:16", mig: false, stale: false, label: "#211 · depth 16 · 1.0M steps · 12.7 h" });
    // the stale score: at the hours it was scored, named so
    expect(t.points[3]).toMatchObject({ x: 1.2, stale: true, label: "#215 · depth 16 · 120k steps · 1.20 h · MIG slice (stale score)" });
    expect(t.unscored).toEqual(["212"]);
    expect(t.untimed).toEqual([]);
    const untimed = timePoints(shapeRows(payload({ models: [urow(220, { train_hours: null })] }), { metric: "mean" }));
    expect(untimed.untimed).toEqual(["220"]);
  });

  it("draws one line per depth through its models by steps", () => {
    const lines = lengthLines(rows);
    expect(lines.map((l) => [l.key, l.points.map((p) => [p.num, p.x])])).toEqual([
      // same steps: lower ∫PSNR first, so the line rises through them
      ["depth:16", [["215", 120_000], ["210", 200_000], ["211", 1_000_000]]],
      ["depth:32", [["208", 200_000]]],
    ]);
  });

  it("pads a log domain and falls back without positive values", () => {
    expect(logDomain([2, 8], [1, 10], 2)).toEqual([1, 16]);
    expect(logDomain([0, null, -1], [0.1, 100])).toEqual([0.1, 100]);
  });

  it("finds the drawn point nearest a click on a log x axis, within reach", () => {
    const pts = [{ x: 1, y: 60, id: "a" }, { x: 10, y: 60, id: "b" }, { x: 100, y: 61, id: "c" }];
    expect(nearestPoint(pts, { x: 9, y: 60.02 }, [0.5, 200], [59, 62])?.id).toBe("b");
    expect(nearestPoint(pts, { x: 30, y: 60.5 }, [0.5, 200], [59, 62])).toBeNull();
    // linear x: 9 is nearer 10 than 1 in value too
    expect(nearestPoint(pts, { x: 12, y: 60 }, [0, 120], [59, 62], false)?.id).toBe("b");
  });

  it("labels a point with what it is", () => {
    expect(pointLabel({ num: "207", depth: 64, steps: 200_000, hours: 7.76, mig: false })).toBe("#207 · depth 64 · 200k steps · 7.76 h");
    expect(pointLabel({ num: "x", depth: null, steps: null, hours: null, mig: false })).toBe("#x");
    expect(pointLabel({ num: "215", depth: 16, steps: null, hours: 1.2, mig: false, stale: true })).toBe("#215 · depth 16 · 1.20 h (stale score)");
  });
});

describe("jobs", () => {
  it("scores the local models without a current score, and pulls what finished on FASRC", () => {
    expect(toScore(payload().models)).toEqual(["member_215"]);
    expect(fasrcSplit(payload().fasrc)).toEqual({ finished: ["member_213"], again: [], failed: [], training: ["member_218"] });
    expect(scoredFields(payload())).toBe(100);
    expect(scoredFields(payload({ identity: { indices: [0, 1, 2] } }))).toBe(3);
    expect(scoredFields(payload({ identity: null, models: [urow(1, { score: score(60, { n_fields: 40 }) })] }))).toBe(40);
    expect(scoredFields(null)).toBe(100);
    expect(numbersText(["member_212", "member_213"])).toBe("212, 213");
    expect(numbersText(["member_1", "member_2", "member_3"], 2)).toBe("01, 02 and 1 more");
  });
});

describe("FASRC rows", () => {
  const row = (name: string, state: string, patch: Partial<UniversalFasrcRow> = {}): UniversalFasrcRow => ({
    name, jobids: ["1"], state, finished: !["RUNNING", "PENDING"].includes(state), ...patch,
  });

  it("offers a pull only for a job that left a checkpoint; a failed one is a try, a registered one a pull again", () => {
    const rows = [
      row("member_220", "OUT_OF_MEMORY", { pullable: false }),
      row("member_213", "COMPLETED", { pullable: true }),
      row("member_214", "TIMEOUT"),                              // no flag: read off the state
      row("member_221", "CANCELLED by 123"),
      row("member_216", "COMPLETED", { pullable: true, registered: true }),
      row("member_218", "RUNNING", { pullable: false }),
    ];
    expect(rows.map(fasrcKind)).toEqual(["try", "pull", "pull", "try", "again", "training"]);
    expect(fasrcSplit(rows)).toEqual({
      finished: ["member_213", "member_214"], again: ["member_216"], failed: ["member_220", "member_221"], training: ["member_218"],
    });
    // the server's flag wins over the state
    expect(fasrcKind(row("member_222", "FAILED", { pullable: true }))).toBe("pull");
    expect(fasrcKind(row("member_223", "COMPLETED", { pullable: false }))).toBe("try");
  });
});

describe("real fields", () => {
  const agg = (vis: number, y: number, j: number, h: number, r: number) => ({
    n_tiles: 2, per_band: { VIS: { hole_pct: vis }, Y_E: { hole_pct: y }, J_E: { hole_pct: j }, H_E: { hole_pct: h } },
    summary: { hole_pct_mean: (vis + y + j + h) / 4, hole_pct_max: Math.max(vis, y, j, h), median_R: r },
  });
  const EXPS: ExperimentSummary[] = [
    { id: "e3", created: "2026-10-13T09:00:00Z", label: "poster", summary: { production: agg(17, 18, 22, 20, 1.1), "universal:member_211": agg(3, 4, 5, 6, 1.02) } },
    { id: "e2", created: "2026-10-12T09:00:00Z", label: "nexus", summary: { "universal:member_211": agg(9, 9, 9, 9, 0.9), "universal:member_210": agg(7, 8, 9, 10, 0.95) } },
    { id: "e1", created: "2026-10-11T09:00:00Z", summary: { "universal:member_210": agg(1, 1, 1, 1, 1) } },
  ];
  const CATALOG = { models: [{ spec: "universal:member_211", fingerprint: "u211" }, { spec: "universal:member_210", fingerprint: "u210" }] };

  it("waits for the newest run's record, and keeps asking for every record it walked", () => {
    const { picks, ids } = realPicks(["universal:member_211", "universal:member_210", "universal:member_999"], EXPS, new Map(), CATALOG);
    expect(picks.get("universal:member_211")).toEqual({ state: "loading" });
    expect(picks.get("universal:member_999")).toEqual({ state: "none" });
    expect(ids).toEqual(["e3", "e2"]);
  });

  it("takes the newest run that scored THIS model; an earlier fit is only named", () => {
    const details = new Map<string, { id: string; fingerprints: Record<string, string> }>([
      ["e3", { id: "e3", fingerprints: { production: "p", "universal:member_211": "old" } }],
      ["e2", { id: "e2", fingerprints: { "universal:member_211": "u211", "universal:member_210": "older" } }],
      ["e1", { id: "e1", fingerprints: { "universal:member_210": "oldest" } }],
    ]);
    const { picks, ids } = realPicks(["universal:member_211", "universal:member_210"], EXPS, details, CATALOG);
    const p211 = picks.get("universal:member_211");
    expect(p211?.state).toBe("current");
    if (p211?.state === "current") {
      expect(p211.expId).toBe("e2");
      expect(p211.bench.worst).toMatchObject({ short: "VIS", pct: 9 });
      expect(p211.bench.medianR).toBe(0.9);
    }
    expect(picks.get("universal:member_210")).toMatchObject({ state: "earlier", expId: "e2" });
    expect(ids).toEqual(["e3", "e2", "e1"]);
    // no catalogue yet: nothing is current
    expect(realPicks(["universal:member_211"], EXPS, details, null).picks.get("universal:member_211")).toEqual({ state: "loading" });
  });

  it("links Sky › Compare's new comparison with production and the specs", () => {
    expect(compareHref(["universal:member_211"])).toBe("/sky/compare?new=1&models=production,universal:member_211");
    expect(compareHref(["universal:member_211", "universal:member_212", "production"]))
      .toBe("/sky/compare?new=1&models=production,universal:member_211,universal:member_212");
    expect(compareHref([])).toBe("/sky/compare?new=1&models=production");
  });
});

describe("a universal model as production", () => {
  const p = universalProduction();

  it("names production in the summary, with its own hours; the best may be production itself", () => {
    const s = universalSummary(shapeRows(p, { metric: "mean" }), p);
    expect(s.production).toEqual({ name: "member_210", label: "Universal 210", mean: 61, hours: 2.5, stale: false });
    expect(s.best?.name).toBe("member_211");
    expect(s.gain).toBeCloseTo(0.6, 9);
    const top = universalProduction("member_211");
    const t = universalSummary(shapeRows(top, { metric: "mean" }), top);
    expect([t.best?.name, t.production?.name, t.gain]).toEqual(["member_211", "member_211", 0]);
  });

  it("never compares production with itself on real fields, and reads its real facts from either spec", () => {
    expect(compareHref(["universal:member_211", "universal:member_210"], "universal:member_210"))
      .toBe("/sky/compare?new=1&models=production,universal:member_211");
    expect(compareHref(["universal:member_210"], "universal:member_210")).toBe("/sky/compare?new=1&models=production");
    const bench = { holeMean: 4, holeMax: 6, medianR: 1, rLt08: 0, nTiles: 1, bands: [], worst: null };
    const current = (expId: string, created: string) => ({ state: "current" as const, expId, created, tileSet: "1 real tile", bench });
    const earlier = { state: "earlier" as const, expId: "e0" };
    // a run of `production` (which runs this model) when its own spec has none current
    expect(preferCurrent(earlier, current("e2", "2026-10-21"))).toMatchObject({ expId: "e2" });
    expect(preferCurrent({ state: "none" }, { state: "loading" })).toEqual({ state: "loading" });
    expect(preferCurrent(current("e1", "2026-10-20"), earlier)).toMatchObject({ expId: "e1" });
    // both current: the newer run
    expect(preferCurrent(current("e1", "2026-10-20"), current("e2", "2026-10-21"))).toMatchObject({ expId: "e2" });
    expect(preferCurrent(current("e3", "2026-10-22"), current("e2", "2026-10-21"))).toMatchObject({ expId: "e3" });
    expect(preferCurrent(undefined, undefined)).toBeUndefined();
  });
});

describe("formatting", () => {
  it("reads steps, hours, knees and the training step", () => {
    expect([950, 1500, 29_000, 977_000, 1_000_000, 2_000_000].map(stepsShort)).toEqual(["950", "1.5k", "29k", "977k", "1.0M", "2.0M"]);
    expect(stepsShort(null)).toBe("—");
    expect([0.29, 2.583, 12.73].map(hoursText)).toEqual(["0.29 h", "2.58 h", "12.7 h"]);
    expect(hoursText(null)).toBe("—");
    expect(kneeWords({ knee: "learned", output_knee: null })).toBe("learned");
    expect(kneeWords({ knee: "fixed", output_knee: 10 })).toBe("fixed 10 e⁻");
    expect(learnedKneesText({ learned_knees: [8.1, 12, null, 2000] })).toBe("VIS 8.1 · Y 12 · J — · H 2k e⁻");
    expect(learnedKneesText({ learned_knees: null })).toBeNull();
    expect(stepImplText({ jit_compile: true, mixed_precision: true })).toBe("XLA + bf16");
    expect(stepImplText({ jit_compile: true, mixed_precision: false })).toBe("XLA");
    expect(stepImplText({ jit_compile: false, mixed_precision: null })).toBe("plain");
  });
});

describe("toolbar choices and production", () => {
  it("parses only the offered x axes and metrics", () => {
    expect(parseXAxis("steps")).toBe("steps");
    expect(parseXAxis("hours")).toBeUndefined();
    expect(parseMetric("J_E")).toBe("J_E");
    expect(parseMetric("max")).toBeUndefined();
  });

  it("takes production's own score and label, comparable only when the server says so", () => {
    const p = universalProduction();
    expect(productionReference(p)).toMatchObject({ mean: 61, model: "member_210" });
    expect(productionLabel(p)).toBe("Universal 210");
    expect(isComparable(p)).toBe(true);
    expect(isComparable({ ...p, production_comparable: false })).toBe(false);
    expect(productionReference({ production: null })).toBeNull();
  });

  it("rings a model trained on a MIG slice", () => {
    expect(isMig({ partition: "gpu_test", label: null })).toBe(true);
    expect(isMig({ partition: "gpu", label: "A100" })).toBe(false);
  });
});
