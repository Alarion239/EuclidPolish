/* The pure shaping behind Models › Knees (knee attribution): normalising the
 * payloads (a partial record never breaks; the version-2 fields read null in
 * an older record), the brightness curves (geometric bin centres, empty bins
 * skipped), the class bars in the spec's order, the gradient-vs-ablation
 * agreement, the cross-band matrix, the local (3×3) split in every gradient
 * view (the receptive field's for an older record), the input band (one
 * input's own knee split, all under the ablation split or for an older
 * record), the per-electron warning and the noise σ line, the explorer's dots and their
 * dominant knee, one colour per knee value, the run form (default pick,
 * validation, the POSTed fields, the confirm's estimate) and the field image
 * URL (the backend's slug rule). */
import { describe, expect, it } from "vitest";
import { viridis } from "../../colors";
import {
  KNEE_INPUTS, KNEE_RAMP, agreement, binCentres, brightnessExtent, brightnessLines, brightnessSamples, checkRunForm, classRows, crossBandMatrix,
  defaultModelPick, dominantKnee, edgeLeads, effectiveInput, effectiveSplit, electronText, errorRows, estimateText, fieldDots, fieldImageUrl,
  hasInputSplit, hasLocalSplit, inputIndex, isPerElectron, kneeLabel, kneeRamp, kneeUnion, modelShort, noiseText, normaliseAggregate,
  normaliseModels, normaliseRecord, normaliseRuns, overlayModels, parseBand, parseInput, parseLayer, parseSet, parseShare, runFormData,
  runText, setsWithData, slug,
} from "./knees";
import type { KneeAggregate } from "./kneesApi";
import {
  BAND_SHARE, CROSS_LOCAL, EDGES, KNEES_6, KNEES_8, LOCAL_BAND_SHARE, RUN_ID, aggregate, byInput, catalogue, failedRecord, field, localSplit,
  record, runs, v1Record, v2Record,
} from "./testing/kneesFixture";

describe("normalising the payloads", () => {
  it("keeps a well-formed record as it is, models in params order", () => {
    const rec = normaliseRecord(record());
    expect(rec?.id).toBe(RUN_ID);
    expect(Object.keys(rec!.models)).toEqual(["universal:member_222", "universal:member_221"]);
    expect(rec!.models["universal:member_222"].knees).toEqual(KNEES_8);
    expect(rec!.models["universal:member_222"].fields[2]).toMatchObject({ ref: "poster/target_181255", kind: "real", shape: [128, 128] });
    expect(rec!.models["universal:member_221"].fields[0].image).toBeNull();
  });

  it("fills a partial record (a run still writing) with empty collections", () => {
    const rec = normaliseRecord({ id: "x", status: "running", models: { "universal:member_9": { knees: [1, 10], fields: [{ ref: "test/0" }] } } });
    expect(rec).toMatchObject({ id: "x", status: "running", errors: {}, params: { models: [], synthetic: 0, tiles: [], per_class: 24, seed: 0 } });
    const m = rec!.models["universal:member_9"];
    expect(m.aggregate).toEqual({ synthetic: null, real: null });
    expect(m.bands).toEqual(["VIS", "Y_E", "J_E", "H_E"]);
    expect(m.fields[0]).toMatchObject({ ref: "test/0", kind: "synthetic", shape: [0, 0], image: null });
    expect(m.fields[0].samples.band).toEqual([]);
    expect(normaliseRecord(null)).toBeNull();
    expect(normaliseRecord({ status: "done" })).toBeNull();
  });

  it("reads the version-2 fields (the by-input matrices, the noise σ, the version); an older record reads them null", () => {
    const rec = normaliseRecord(record())!;
    expect(rec.version).toBe(3);
    const m = rec.models["universal:member_222"];
    const agg = m.aggregate.synthetic!;
    expect(agg.by_class["0"]["galaxy core"].knee_share_by_input).toHaveLength(4);
    expect(agg.by_class["0"]["galaxy core"].knee_share_by_input![0]).toHaveLength(8);
    expect(agg.by_brightness.bands["0"].knee_share_by_input).toHaveLength(20);
    expect(agg.by_brightness.bands["0"].knee_share_by_input![0]).toBeNull();          // an empty bin
    expect(agg.by_brightness.bands["0"].knee_share_by_input![2]).toHaveLength(4);
    expect(m.fields[0].noise_sigma_e).toEqual([17.6, 0.87, 0.98, 0.9]);
    const old = normaliseRecord(v1Record())!;
    expect(old.version).toBeNull();
    const oldAgg = old.models["universal:member_222"].aggregate.synthetic!;
    expect(oldAgg.by_class["0"]["galaxy core"].knee_share_by_input).toBeNull();
    expect(oldAgg.by_brightness.bands["0"].knee_share_by_input).toBeNull();
    expect(old.models["universal:member_222"].fields[0].noise_sigma_e).toBeNull();
    // a malformed matrix: a non-array row reads empty, a non-finite share 0; an empty matrix null
    const odd = normaliseAggregate({ by_class: { 0: { sky: { knee_share_by_input: [[0.5, "x"], 3], n: 1 }, faint: { knee_share_by_input: [], n: 1 } } } });
    expect(odd?.by_class["0"].sky.knee_share_by_input).toEqual([[0.5, 0], []]);
    expect(odd?.by_class["0"].faint.knee_share_by_input).toBeNull();
    // the fixture keeps the identity knee_share = Σ_c band_share_c · knee_share_by_input_c
    const core = agg.by_class["0"]["galaxy core"];
    core.knee_share.forEach((v, k) => expect(BAND_SHARE.reduce((s, w, c) => s + w * core.knee_share_by_input![c][k], 0)).toBeCloseTo(v, 3));
  });

  it("reads the version-3 local (3×3) splits; a version-2 record reads them null (per sample: empty)", () => {
    const rec = normaliseRecord(record())!;
    const m = rec.models["universal:member_222"];
    const agg = m.aggregate.synthetic!;
    const core = agg.by_class["0"]["galaxy core"];
    expect(core.knee_share_local).toEqual(localSplit(KNEES_8, 300));
    expect(core.band_share_local).toEqual(LOCAL_BAND_SHARE);
    expect(core.knee_share_by_input_local).toEqual(byInput(localSplit(KNEES_8, 300)));
    expect(agg.by_brightness.bands["0"].knee_share_local).toHaveLength(20);
    expect(agg.by_brightness.bands["0"].knee_share_local![0]).toBeNull();               // an empty bin
    expect(agg.by_brightness.bands["0"].knee_share_by_input_local![2]).toHaveLength(4);
    expect(agg.cross_band_local).toEqual(CROSS_LOCAL);
    expect(m.fields[0].samples.knee_share_local).toHaveLength(6);
    const old = normaliseRecord(v2Record())!;
    expect(old.version).toBe(2);
    const oldAgg = old.models["universal:member_222"].aggregate.synthetic!;
    expect(oldAgg.by_class["0"]["galaxy core"]).toMatchObject({ knee_share_local: null, band_share_local: null, knee_share_by_input_local: null });
    expect(oldAgg.by_brightness.bands["0"]).toMatchObject({ knee_share_local: null, knee_share_by_input_local: null });
    expect(oldAgg.cross_band_local).toBeNull();
    expect(oldAgg.by_class["0"]["galaxy core"].knee_share_by_input).toHaveLength(4);     // the per-input split stays
    expect(old.models["universal:member_222"].fields[0].samples.knee_share_local).toEqual([]);
    // a null row of the local cross-band reads empty (its cells null)
    expect(normaliseAggregate({ cross_band_local: [[0.5, null], null] })?.cross_band_local).toEqual([[0.5, null], []]);
    // the fixture keeps the local identity knee_share_local = Σ_c band_share_local_c · knee_share_by_input_local_c
    core.knee_share_local!.forEach((v, k) =>
      expect(LOCAL_BAND_SHARE.reduce((s, w, c) => s + w * core.knee_share_by_input_local![c][k], 0)).toBeCloseTo(v, 3));
  });

  it("reads the runs and the model catalogue defensively", () => {
    expect(normaliseRuns(runs()).map((r) => [r.id, r.status, r.models.length])).toEqual([[RUN_ID, "done", 2], ["20261014-090000-def456", "failed", 1]]);
    expect(normaliseRuns({})).toEqual([]);
    expect(normaliseRuns({ runs: [{ status: "done" }, { id: "a" }] })).toEqual([
      { id: "a", created: null, finished: null, status: "done", models: [], synthetic: 0, tiles: [], per_class: null }]);
    const models = normaliseModels(catalogue());
    expect(models).toHaveLength(7);
    expect(models.find((m) => m.spec === "member:member_178")).toMatchObject({ available: false, reason: "a single-knee model (nothing to split over knees)" });
    expect(normaliseModels({ models: [{ spec: "universal:member_1", available: "yes" }] })[0]).toMatchObject({ available: false, knees: [], depth: null });
  });
});

describe("labels and colours", () => {
  it("labels knees as powers of ten in e⁻", () => {
    expect(KNEES_8.map(kneeLabel)).toEqual(["10⁻³ e⁻", "10⁻² e⁻", "10⁻¹ e⁻", "1 e⁻", "10 e⁻", "10² e⁻", "10³ e⁻", "10⁴ e⁻"]);
    expect(kneeLabel(3.2)).toBe("3.2 e⁻");
  });

  it("formats brightness and model names", () => {
    expect(electronText(1200)).toBe("1.2k e⁻");
    expect(electronText(0.003)).toBe("0.003 e⁻");
    expect(electronText(null)).toBe("—");
    expect(modelShort("universal:member_222")).toBe("#222");
    expect(modelShort("member:member_196")).toBe("#196 (member)");
    expect(modelShort("production")).toBe("production");
  });

  it("gives each knee value one colour on a ramp from faint to bright, the same in every panel", () => {
    const union = kneeUnion([KNEES_8, KNEES_6]);
    expect(union).toEqual(KNEES_8);
    const color = kneeRamp(union);
    expect(color(0.001)).toBe(viridis(0.04));
    expect(color(10000)).toBe(viridis(0.84));
    expect(color(0.1)).toBe(viridis(0.04 + (0.8 * 2) / 7));
    expect(new Set(KNEES_8.map(color)).size).toBe(8);
    expect(kneeRamp([5])(5)).toBe(viridis(0.44));
    // the dark theme drops viridis' dark end (invisible on a dark surface)
    const dark = kneeRamp(union, KNEE_RAMP.dark);
    expect(dark(0.001)).toBe(viridis(0.2));
    expect(dark(10000)).toBe(viridis(1));
  });
});

describe("the aggregates", () => {
  const agg = aggregate(KNEES_8, "synthetic");

  it("centres the bins at the geometric mean of their edges", () => {
    const c = binCentres(EDGES);
    expect(c).toHaveLength(20);
    expect(c[0]).toBeCloseTo(Math.sqrt(EDGES[0] * EDGES[1]), 15);
    expect(c[6]).toBeCloseTo(10 ** 0.25, 4);                          // the bin 1 … 10^0.5 e⁻
    expect(binCentres([0, 2])).toEqual([1]);
  });

  it("draws one line per knee over the bins with samples (null and n = 0 bins skipped)", () => {
    const lines = brightnessLines(agg, "VIS", KNEES_8);
    expect(lines).toHaveLength(8);
    expect(lines[0].x).toHaveLength(15);                              // bins 2..16
    expect(lines[0].x[0]).toBeCloseTo(Math.sqrt(EDGES[2] * EDGES[3]), 9);
    expect(lines[0].n[0]).toBe(12);
    expect(brightnessSamples(agg, "VIS")).toBe(Array.from({ length: 15 }, (_, i) => 12 + i).reduce((a, b) => a + b, 0));
    expect(brightnessExtent(lines)).toEqual([lines[0].x[0], lines[0].x[14]]);
    expect(brightnessLines(null, "VIS", KNEES_8)).toEqual([]);
    // the faintest bins lean on the faint knees, the brightest on the bright ones
    const leads = edgeLeads(lines);
    expect(leads.faint?.knee).toBe(0.01);
    expect(leads.bright?.knee).toBe(10000);
  });

  it("lists the sampled classes in the spec's order, the gradient or the ablation split", () => {
    const g = classRows(agg, "VIS", "synthetic", "gradient");
    expect(g.map((r) => r.cls)).toEqual(["star core", "star wing", "galaxy core", "galaxy outskirts", "faint", "sky"]); // no saturated in VIS
    expect(classRows(agg, "Y_E", "synthetic", "gradient")[0].cls).toBe("saturated");
    const a = classRows(agg, "VIS", "synthetic", "ablation");
    expect(a[0].shares).toEqual([...g[0].shares].reverse());
    expect(classRows(aggregate(KNEES_8, "real"), "VIS", "real", "gradient").map((r) => r.cls)).toEqual(["sky", "faint", "mid", "bright", "core", "blank"]);
  });

  it("measures the agreement of the two splits and the dominant knee", () => {
    expect(agreement([0.5, 0.5], [0.5, 0.5])).toBe(1);
    expect(agreement([1, 0], [0, 1])).toBe(0);
    expect(agreement([0.6, 0.4], [0.3, 0.2])).toBeCloseTo(1, 12); // renormalised
    expect(agreement([0, 0], [1, 0])).toBeNull();
    expect(dominantKnee([0.1, 0.7, 0.2])).toBe(1);
    expect(dominantKnee([])).toBe(-1);
    expect(dominantKnee([0, 0])).toBe(-1);
  });

  it("reads the 4×4 cross-band matrix, missing cells null", () => {
    expect(crossBandMatrix(agg)[0]).toEqual([0.82, 0.06, 0.06, 0.06]);
    expect(crossBandMatrix({ ...agg, cross_band: [[0.5]] })[1]).toEqual([null, null, null, null]);
  });
});

describe("the local (3×3) split", () => {
  const agg = aggregate(KNEES_8, "synthetic");
  const v2 = normaliseRecord(v2Record())!.models["universal:member_222"].aggregate.synthetic;

  it("draws the local brightness curves (and their per-input rows); the ablation draws the receptive field's", () => {
    const rf = brightnessLines(agg, "VIS", KNEES_8);
    const local = brightnessLines(agg, "VIS", KNEES_8, "all", "local");
    expect(local.map((l) => l.x)).toEqual(rf.map((l) => l.x));
    expect(local[2].y[5]).toBe(agg.by_brightness.bands["0"].knee_share_local![7]![2]);   // bins 2..16: the 6th is bin 7
    expect(local).not.toEqual(rf);
    expect(brightnessLines(agg, "VIS", KNEES_8, "Y", "local")[2].y[5]).toBe(agg.by_brightness.bands["0"].knee_share_by_input_local![7]![1][2]);
    expect(brightnessLines(agg, "VIS", KNEES_8, "all", "ablation")).toEqual(rf);
    // the faintest bins lean a knee brighter locally (the fixture's 10·b)
    expect(edgeLeads(local).faint?.knee).toBe(0.1);
    expect(edgeLeads(rf).faint?.knee).toBe(0.01);
  });

  it("splits the class bars locally; the agreement's gradient stays the receptive field's", () => {
    const rf = classRows(agg, "VIS", "synthetic", "gradient");
    const local = classRows(agg, "VIS", "synthetic", "local");
    expect(local.map((r) => r.cls)).toEqual(rf.map((r) => r.cls));
    expect(local[2].shares).toEqual(localSplit(KNEES_8, 300));
    expect(local[2].gradient).toEqual(rf[2].gradient);
    expect(local[2].ablation).toEqual(rf[2].ablation);
    expect(classRows(agg, "VIS", "synthetic", "local", "Y")[2].shares).toEqual(byInput(localSplit(KNEES_8, 300))[1]);
  });

  it("reads the local cross-band matrix and the explorer's local dominant knee", () => {
    expect(crossBandMatrix(agg, "local")[0]).toEqual(CROSS_LOCAL[0]);
    expect(crossBandMatrix(agg, "ablation")[0]).toEqual([0.82, 0.06, 0.06, 0.06]);
    const f = field("test/0", "synthetic", KNEES_8, null);
    // galaxy core (300 e⁻) → 10³ e⁻, sky (0.02) → 10⁻¹, star core (2000) → 10⁴, faint (0.3) → 1 e⁻
    expect(fieldDots(f, "VIS", "local").map((d) => KNEES_8[d.dominant])).toEqual([1000, 0.1, 10000, 1]);
    expect(fieldDots(f, "VIS", "gradient").map((d) => KNEES_8[d.dominant])).toEqual([100, 0.01, 1000, 0.1]);
  });

  it("falls back to the receptive field's split without the local data (a version-2 record)", () => {
    expect(hasLocalSplit(agg)).toBe(true);
    expect(hasLocalSplit(v2)).toBe(false);
    expect(hasLocalSplit(null)).toBe(false);
    expect(brightnessLines(v2, "VIS", KNEES_8, "Y", "local")).toEqual(brightnessLines(v2, "VIS", KNEES_8, "Y"));
    expect(classRows(v2, "VIS", "synthetic", "local", "Y")).toEqual(classRows(v2, "VIS", "synthetic", "gradient", "Y"));
    expect(crossBandMatrix(v2, "local")).toEqual(crossBandMatrix(v2));
    const f = normaliseRecord(v2Record())!.models["universal:member_222"].fields[0];
    expect(fieldDots(f, "VIS", "local")).toEqual(fieldDots(f, "VIS"));
  });

  it("resolves the split drawn: local unless an aggregate predates it; the input band stays enabled for local", () => {
    expect(effectiveSplit("local", [agg])).toEqual({ split: "local", missing: false });
    expect(effectiveSplit("local", [agg, v2])).toEqual({ split: "gradient", missing: true });
    expect(effectiveSplit("gradient", [v2])).toEqual({ split: "gradient", missing: false });
    expect(effectiveSplit("ablation", [v2])).toEqual({ split: "ablation", missing: false });
    expect(effectiveSplit("local", [null])).toEqual({ split: "local", missing: false });
    expect(effectiveInput("Y", "local", [agg])).toEqual({ input: "Y", disabled: false, missing: false });
    // local shares without their per-input rows: the input band falls back under local only
    const noLocalInputs: KneeAggregate = {
      ...agg, by_class: {},
      by_brightness: { ...agg.by_brightness,
        bands: Object.fromEntries(Object.entries(agg.by_brightness.bands).map(([k, b]) => [k, { ...b, knee_share_by_input_local: null }])) },
    };
    expect(hasInputSplit(noLocalInputs)).toBe(true);
    expect(hasInputSplit(noLocalInputs, true)).toBe(false);
    expect(effectiveInput("Y", "local", [noLocalInputs])).toEqual({ input: "all", disabled: false, missing: true });
    expect(effectiveInput("Y", "gradient", [noLocalInputs])).toEqual({ input: "Y", disabled: false, missing: false });
  });
});

describe("the input band", () => {
  const agg = aggregate(KNEES_8, "synthetic");
  const old = normaliseRecord(v1Record())!.models["universal:member_222"].aggregate.synthetic;

  it("draws one input band's own split over the same bins, the all-input split by default", () => {
    const all = brightnessLines(agg, "VIS", KNEES_8);
    expect(brightnessLines(agg, "VIS", KNEES_8, "all")).toEqual(all);
    const vis = brightnessLines(agg, "VIS", KNEES_8, "VIS");
    const y = brightnessLines(agg, "VIS", KNEES_8, "Y");
    expect(vis.map((l) => l.x)).toEqual(all.map((l) => l.x));
    expect(vis[7].y[5]).toBe(agg.by_brightness.bands["0"].knee_share_by_input![7]![0][7]);  // bins 2..16: the 6th is bin 7
    expect(y[7].y[5]).toBe(agg.by_brightness.bands["0"].knee_share_by_input![7]![1][7]);
    // VIS leans on the brightest knee more than the all-input split, Y less; J is the split itself
    expect(vis[7].y.every((v, i) => v >= all[7].y[i])).toBe(true);
    expect(y[7].y.every((v, i) => v <= all[7].y[i])).toBe(true);
    expect(vis[7].y).not.toEqual(all[7].y);
    expect(brightnessLines(agg, "VIS", KNEES_8, "J")).toEqual(all);
  });

  it("splits the class bars by one input band's gradient; the ablation and the agreement's gradient stay all-input", () => {
    const all = classRows(agg, "VIS", "synthetic", "gradient");
    const y = classRows(agg, "VIS", "synthetic", "gradient", "Y");
    expect(y.map((r) => r.cls)).toEqual(all.map((r) => r.cls));
    expect(y[2].cls).toBe("galaxy core");
    expect(y[2].shares).toEqual(agg.by_class["0"]["galaxy core"].knee_share_by_input![1]);
    expect(y[2].shares).not.toEqual(all[2].shares);
    expect(y[2].gradient).toEqual(all[2].gradient);
    expect(classRows(agg, "VIS", "synthetic", "ablation", "Y")).toEqual(classRows(agg, "VIS", "synthetic", "ablation"));
  });

  it("falls back to the all-input split for an aggregate without the per-input data (an older record)", () => {
    expect(hasInputSplit(agg)).toBe(true);
    expect(hasInputSplit(old)).toBe(false);
    expect(hasInputSplit(null)).toBe(false);
    expect(brightnessLines(old, "VIS", KNEES_8, "Y")).toEqual(brightnessLines(old, "VIS", KNEES_8));
    expect(classRows(old, "VIS", "synthetic", "gradient", "Y")).toEqual(classRows(old, "VIS", "synthetic", "gradient"));
  });

  it("resolves the input drawn: the pick, else all under the ablation split or when an aggregate predates the split", () => {
    expect(KNEE_INPUTS.map(inputIndex)).toEqual([null, 0, 1, 2, 3]);
    expect(effectiveInput("Y", "gradient", [agg])).toEqual({ input: "Y", disabled: false, missing: false });
    expect(effectiveInput("Y", "ablation", [agg])).toEqual({ input: "all", disabled: true, missing: false });
    expect(effectiveInput("Y", "ablation", [old])).toEqual({ input: "all", disabled: true, missing: false });
    expect(effectiveInput("Y", "gradient", [agg, old])).toEqual({ input: "all", disabled: false, missing: true });
    expect(effectiveInput("all", "gradient", [old])).toEqual({ input: "all", disabled: false, missing: false });
    expect(effectiveInput("H", "gradient", [null, undefined])).toEqual({ input: "H", disabled: false, missing: false });
  });

  it("flags the per-electron records (version 1) and writes the field's noise σ line", () => {
    expect(isPerElectron(normaliseRecord(record()))).toBe(false);
    expect(isPerElectron(normaliseRecord(v1Record()))).toBe(true);
    expect(isPerElectron(normaliseRecord({ ...record(), version: 1 }))).toBe(true);
    expect(isPerElectron(null)).toBe(false);
    expect(noiseText([17.6, 0.87, 0.98, 0.9])).toBe("noise σ: VIS 17.6 · Y 0.87 · J 0.98 · H 0.90 e⁻");
    expect(noiseText([18.04, NaN, 1.234])).toBe("noise σ: VIS 18.0 · Y — · J 1.23 · H — e⁻");
    expect(noiseText(null)).toBeNull();
    expect(noiseText([NaN, NaN])).toBeNull();
  });
});

describe("the field explorer", () => {
  const f = field("test/0", "synthetic", KNEES_8, null);

  it("draws the samples of one band as dots with their dominant knee", () => {
    const vis = fieldDots(f, "VIS");
    expect(vis.map((d) => [d.x, d.y, d.cls])).toEqual([[12, 10, "galaxy core"], [50, 40, "sky"], [30, 60, "star core"], [90, 100, "faint"]]);
    expect(vis.map((d) => KNEES_8[d.dominant])).toEqual([100, 0.01, 1000, 0.1]);
    expect(fieldDots(f, "Y_E")).toHaveLength(2);
    expect(fieldDots(f, "H_E")).toHaveLength(0);
    expect(fieldDots(null, "VIS")).toEqual([]);
  });

  it("builds the image URL from the record's, or from the backend's slugs", () => {
    expect(slug("universal:member_222")).toBe("universal__member_222");
    expect(slug("poster/target_181255")).toBe("poster__target_181255");
    expect(slug(":::")).toBe("x");
    expect(fieldImageUrl(RUN_ID, "universal:member_222", f, "J_E", "lr"))
      .toBe(`/api/knee-attribution/${RUN_ID}/field/universal__member_222/test__0.png?band=J_E&layer=lr`);
    const withImage = { ...f, image: "/api/knee-attribution/r/field/m/f.png?v=2" };
    expect(fieldImageUrl(RUN_ID, "x", withImage, "VIS", "sr")).toBe("/api/knee-attribution/r/field/m/f.png?v=2&band=VIS&layer=sr");
  });
});

describe("the run form", () => {
  it("preselects the newest available universal models", () => {
    expect(defaultModelPick(normaliseModels(catalogue()))).toEqual(["universal:member_223", "universal:member_222", "universal:member_221"]);
    const onlyMembers = normaliseModels(catalogue()).filter((m) => m.kind === "member");
    expect(defaultModelPick(onlyMembers)).toEqual(["member:member_196"]);
    expect(defaultModelPick([])).toEqual([]);
  });

  it("validates the numbers and needs a model and some field", () => {
    const ok = checkRunForm({ models: ["universal:member_222"], synthetic: "20", tiles: [], perClass: "24", seed: "0" });
    expect(ok).toMatchObject({ ok: true, synthetic: 20, perClass: 24, seed: 0, errors: {} });
    const bad = checkRunForm({ models: [], synthetic: "101", tiles: [], perClass: "0", seed: "-1" });
    expect(bad.ok).toBe(false);
    expect(Object.keys(bad.errors).sort()).toEqual(["models", "perClass", "seed", "synthetic"]);
    expect(checkRunForm({ models: ["m"], synthetic: "0", tiles: [], perClass: "24", seed: "0" }).errors.fields).toBeTruthy();
    expect(checkRunForm({ models: ["m"], synthetic: "0", tiles: ["poster/p"], perClass: "24", seed: "0" }).ok).toBe(true);
    expect(checkRunForm({ models: ["m"], synthetic: "2.5", tiles: [], perClass: "24", seed: "0" }).errors.synthetic).toBeTruthy();
  });

  it("posts comma lists and numbers, and estimates ~3 s per field per model", () => {
    expect(runFormData({ models: ["universal:member_222", "universal:member_221"], synthetic: 20, tiles: ["poster/a", "nexus/b"], perClass: 24, seed: 7 }))
      .toEqual({ models: "universal:member_222,universal:member_221", synthetic: "20", tiles: "poster/a,nexus/b", per_class: "24", seed: "7" });
    expect(estimateText(3, 22)).toBe("local job: 3 models × 22 fields, ~21 min");
    expect(estimateText(1, 1)).toBe("local job: 1 model × 1 field, ~1 min");
    expect(estimateText(4, 34)).toBe("local job: 4 models × 34 fields, ~43 min");
    expect(estimateText(4, 100, 8)).toBe("local job: 4 models × 100 fields, ~51 min");
    expect(estimateText(30, 100)).toBe("local job: 30 models × 100 fields, ~16 h");
  });
});

describe("runs and the URL", () => {
  it("describes a run in one line and lists errors", () => {
    expect(runText(runs().runs[0], "2026-10-15 10:15")).toBe("2026-10-15 10:15 · #222, #221 · 2 synthetic + 1 real");
    expect(runText(runs().runs[1], "d")).toBe("d · #211 · 5 synthetic · failed");
    expect(runText(normaliseRuns({ runs: [{ ...runs().runs[0], status: "cancelled" }] })[0], "d")).toBe("d · #222, #221 · 2 synthetic + 1 real · cancelled");
    expect(errorRows({ "universal:member_211": "boom", "universal:member_222|test/3": "bad field" })).toEqual([
      { where: "#211", reason: "boom" }, { where: "#222 › test/3", reason: "bad field" }]);
  });

  it("overlays the picked models in the run (at most four), else the run's first", () => {
    const rec = normaliseRecord(record());
    expect(overlayModels(rec, [])).toEqual(["universal:member_222", "universal:member_221"]);
    expect(overlayModels(rec, ["universal:member_221", "universal:member_9"])).toEqual(["universal:member_221"]);
    expect(overlayModels(null, [])).toEqual([]);
    expect(setsWithData(rec, ["universal:member_222"])).toEqual(["synthetic", "real"]);
    expect(setsWithData(normaliseRecord(failedRecord()), ["universal:member_211"])).toEqual([]);
  });

  it("parses the URL values, unknown ones read the default", () => {
    expect([parseBand("J_E"), parseBand("K")]).toEqual(["J_E", "VIS"]);
    expect([parseSet("real"), parseSet("x")]).toEqual(["real", "synthetic"]);
    expect([parseShare("ablation"), parseShare("local"), parseShare(""), parseShare("Local")]).toEqual(["ablation", "local", "gradient", "gradient"]);
    expect([parseLayer("lr"), parseLayer("hr")]).toEqual(["lr", "sr"]);
    expect([parseInput("Y"), parseInput("VIS"), parseInput("Y_E"), parseInput("")]).toEqual(["Y", "VIS", "all", "all"]);
  });
});
