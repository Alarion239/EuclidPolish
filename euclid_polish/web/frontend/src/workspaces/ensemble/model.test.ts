import { describe, expect, it } from "vitest";
import type { KneeModel, TrainingJob } from "./api";
import {
  benchmarkExperiment, dbDelta, facetOf, facetValues, formatE, gateUsage, heldOutComparable, integrateKnee, kneeLeaderboard,
  kneeText, memberLabel, memberMatches, memberName, memberNumber, movieStatus, parseMemberList, relativeTo, smooth, stepsText,
  stampKnee, variantLabel,
} from "./model";
import {
  RECIPE_RESOURCES, buildParams, buildSpec, continueTarget, defaultForm, defaultResources, formFromJob, jobRegime, lastBatch,
  newRow, recipeSummary, rowsFromSpec, validate,
} from "./trainModel";

describe("member names", () => {
  it("normalises every spelling and zero-pads like the directories", () => {
    expect(memberNumber("196·psnr")).toBe("196");
    expect(memberNumber("member_07")).toBe("07");
    expect(memberNumber("7")).toBe("07");
    expect(memberNumber("196·loss")).toBeNull();
    expect(memberName("196")).toBe("member_196");
    expect(memberLabel("member_196")).toBe("196·psnr");
  });

  it("parses picker text with ranges", () => {
    expect(parseMemberList("170, 171 180-182 member_190 x")).toEqual({
      names: ["member_170", "member_171", "member_180", "member_181", "member_182", "member_190"], bad: ["x"],
    });
    expect(parseMemberList("5-3").bad).toEqual(["5-3"]);
  });
});

describe("knee description", () => {
  it("shows multi-knee members as multi ×N → out, never 100e", () => {
    expect(kneeText({ asinh_knee: null, asinh_knees: [0.1, 1, 10, 100, 1000, 10000], output_knee: 10 }).text)
      .toBe("multi ×6 → 10");
    expect(kneeText({ asinh_knees: [0.1, 1, 10, 100, 1000, 10000], output_knee: null }).text).toBe("multi ×6 heads");
    expect(kneeText({ asinh_knee: 3000 }).text).toBe("3k e⁻");
    expect(kneeText({ asinh_knee: 10 }).kind).toBe("single");
    const d = kneeText({});
    expect(d.kind).toBe("default");
    expect(d.text).toBe("100 e⁻");
  });

  it("orders facets meaningfully", () => {
    const rows = [
      { loss: "l2", blocks: 32, asinh_knee: 1000 }, { loss: "l1", blocks: 16, asinh_knee: 10 },
      { loss: "l2", blocks: 32, asinh_knees: [1, 10], output_knee: 10 },
    ];
    expect(facetValues(rows, "knee")).toEqual(["10 e⁻", "1k e⁻", "multi ×2 → 10"]);
    expect(facetValues(rows, "depth")).toEqual(["16 blocks", "32 blocks"]);
    expect(facetValues(rows, "loss")).toEqual(["l2", "l1"].sort());
    expect(facetOf(rows[2], "multi")).toBe("multi-knee, 1 image");
    expect(facetOf(rows[0], "multi")).toBe("single knee");
  });
});

describe("knee integration", () => {
  const knees = [0.1, 1, 10, 100, 1000, 10000];
  const flat = knees.map(() => [50, 60]);
  const ramp = knees.map((_, k) => [k, 2 * k]);

  it("equals the trapezoid mean over the whole grid", () => {
    expect(integrateKnee(flat, knees, 0.1, 10000)).toEqual([50, 60]);
    expect(integrateKnee(ramp, knees, 0.1, 10000)).toEqual([2.5, 5]);
  });

  it("integrates a sub-range with interpolated ends", () => {
    const [a] = integrateKnee(ramp, knees, 1, 100);          // k = 1..3 → mean 2
    expect(a).toBeCloseTo(2, 10);
    const [b] = integrateKnee(ramp, knees, Math.sqrt(10), 10); // log10 0.5..1 → mean of 1.5..2
    expect(b).toBeCloseTo(1.75, 10);
    expect(integrateKnee(ramp, knees, 10, 10)[0]).toBeCloseTo(2, 10);   // a point
    expect(integrateKnee(ramp, knees, 100, 1)[0]).toBeCloseTo(2, 10);   // reversed
  });

  it("ranks the leaderboard and reports the rank change vs the full range", () => {
    const models: KneeModel[] = [
      { id: "member_0", kind: "member", label: "01·psnr", psnr: knees.map((_, k) => [k < 3 ? 60 : 40]), integrated: [] },
      { id: "member_1", kind: "member", label: "02·psnr", psnr: knees.map(() => [51]), integrated: [] },
      { id: "ensemble_mean", kind: "mean", label: "mean", psnr: knees.map(() => [50]), integrated: [] },
    ];
    const full = kneeLeaderboard(models, knees, [0.1, 10000]);
    expect(full.map((r) => r.rank)).toEqual([2, 1, 3]);
    const low = kneeLeaderboard(models, knees, [0.1, 10]);
    expect(low[0].rank).toBe(1);
    expect(low[0].rankDelta).toBe(1);           // climbs from 2 to 1
    expect(low[1].vsMean).toBeCloseTo(1, 10);
  });

  it("relative curves subtract the reference", () => {
    expect(relativeTo([[3, 4]], [[1, 1]])).toEqual([[2, 3]]);
  });
});

describe("formatting", () => {
  it("formats deltas with a real minus", () => {
    expect(dbDelta(0.2948)).toBe("+0.29");
    expect(dbDelta(-1.5)).toBe("−1.50");
    expect(dbDelta(null)).toBe("—");
    expect(stepsText(52000, 70000)).toBe("52k / 70k");
    expect(variantLabel("gate:spatial_gate_combiner")).toBe("production");
    expect(variantLabel("spatial_gate_linear")).toBe("linear");
    expect(smooth([1, 2, 3, 4], 2)).toEqual([1, 1.5, 2.5, 3.5]);
  });
});

const JOB: TrainingJob = {
  jobid: "48107719", state: "COMPLETED", mode: "add", member_names: ["member_195", "member_196"],
  params: {
    mode: "add", count: 2, steps: "70000", batch_size: "4", forward_onthefly: "1", saturation_mask_prob: "0.5",
    psf_warp_alpha_max: "5", array_max_parallel: 2,
    member_spec: JSON.stringify([
      { loss: "l2", bootstrap: 0.7, asinh_knees: [0.1, 1, 10, 100, 1000, 10000], knee_loss: "balanced", output_knee: 10, num_res_blocks: 32, icnr: true },
      { loss: "l2", bootstrap: 0.7, asinh_knee: 3000, num_res_blocks: 32, icnr: true },
    ]),
  },
};

describe("train form", () => {
  it("builds the multi-knee member spec the step consumes", () => {
    const spec = buildSpec([newRow({ kneeMode: "multi", outputKnee: "10" }), newRow({ knee: "100", boot: "" })], "add");
    expect(spec[0]).toEqual({ loss: "l2", bootstrap: 0.7, asinh_knees: [0.1, 1, 10, 100, 1000, 10000],
      output_knee: 10, knee_loss: "balanced", num_res_blocks: 32, icnr: true });
    expect(spec[1]).toEqual({ loss: "l2", num_res_blocks: 32, icnr: true });
    expect(buildSpec([newRow()], "fork")[0]).not.toHaveProperty("num_res_blocks");
  });

  it("round-trips a past job (clone) without seeds or names", () => {
    const f = formFromJob(JOB);
    expect(f.mode).toBe("add");
    expect(f.rows).toHaveLength(2);
    expect(f.rows[0]).toMatchObject({ kneeMode: "multi", outputKnee: "10", kneeLoss: "balanced", seed: "" });
    expect(f.rows[1]).toMatchObject({ kneeMode: "single", knee: "3000" });
    expect(f.geometry.saturation_mask_prob).toBe("0.5");
    expect(JSON.parse(buildParams(f).member_spec)).toEqual(JSON.parse(String(JOB.params.member_spec)));
    expect(rowsFromSpec("nope")).toEqual([]);
    expect(recipeSummary(JOB)).toBe("add 2 × L2 · 1 multi-knee, knee 3000 · 70k steps");
    expect(lastBatch([{ ...JOB, mode: "continue" }, JOB])?.jobid).toBe("48107719");
  });

  it("builds continue and fork bodies", () => {
    const f = { ...defaultForm(), mode: "continue" as const, members: ["member_178", "member_179"], continueBasis: "target" as const, targetSteps: "70000" };
    const p = buildParams(f);
    expect(p).toMatchObject({ mode: "continue", members: "member_178,member_179", continue_basis: "target", target_steps: "70000" });
    expect(p).not.toHaveProperty("member_spec");
    const fork = buildParams({ ...defaultForm(), mode: "fork", forkFrom: "member_196", forkTrack: "loss", baseSeed: "12" });
    expect(fork).toMatchObject({ mode: "fork", fork_from: "member_196", fork_track: "loss", base_seed: "12" });
  });

  it("takes the star regime from the workspace, never from a per-row knob", () => {
    const add = buildParams(defaultForm(), "starless");
    expect(add.starless).toBe("1");
    expect(JSON.parse(add.member_spec)[0]).not.toHaveProperty("starless");
    expect(buildParams(defaultForm(), "starfull")).not.toHaveProperty("starless");
    expect(buildParams(defaultForm())).not.toHaveProperty("starless");
    expect(buildParams({ ...defaultForm(), mode: "fork", forkFrom: "member_7" }, "starless").starless).toBe("1");
    // continue keeps each member's recorded regime (origin.json wins in train_ensemble.py)
    expect(buildParams({ ...defaultForm(), mode: "continue", members: ["member_7"] }, "starless")).not.toHaveProperty("starless");
    expect(newRow()).not.toHaveProperty("starless");
    expect(jobRegime(JOB)).toBe("starfull");
    expect(jobRegime({ ...JOB, params: { ...JOB.params, starless: "1" } })).toBe("starless");
    expect(jobRegime({ ...JOB, params: { ...JOB.params, member_spec: JSON.stringify([{ loss: "l2", starless: true }]) } })).toBe("starless");
  });

  it("validates what the submit would refuse", () => {
    expect(validate(defaultForm())).toEqual([]);
    const bad = { ...defaultForm(), rows: [newRow({ kneeMode: "multi", knees: "0,1" })] };
    expect(validate(bad)[0]).toMatch(/knees must be positive/);
    expect(validate({ ...defaultForm(), mode: "continue" })).toContain("pick at least one member to continue");
    const geo = { ...defaultForm(), geometry: { ...defaultForm().geometry, hr_crop_size: "255" } };
    expect(validate(geo)).toContain("HR example side must be a positive even number");
  });
});

/* ── image-first pass (2026-09-27) ─────────────────────────────────────── */

describe("shared number format", () => {
  it("formats e⁻ values with 3 significant figures, exponents at the extremes", () => {
    expect(formatE(0.01360404)).toBe("0.0136");
    expect(formatE(2.396e-19)).toBe("2.4e-19");
    expect(formatE(123456)).toBe("1.2e+5");
    expect(formatE(0)).toBe("0");
    expect(formatE(NaN)).toBe("—");
  });
});

describe("gate usage over bands", () => {
  it("is the mean of the bands with a value, with the max and each band", () => {
    const u = gateUsage({ VIS: 0.00018, Y_E: 0.374, J_E: 0.366, H_E: 0.329 });
    expect(u.mean).toBeCloseTo((0.00018 + 0.374 + 0.366 + 0.329) / 4, 9);
    expect(u.max).toBeCloseTo(0.374, 9);
    expect(u.bands.map((b) => b.short)).toEqual(["VIS", "Y", "J", "H"]);
    expect(u.bands[0].v).toBeCloseTo(0.00018, 9);
  });

  it("skips missing bands and is empty without usage", () => {
    const u = gateUsage({ VIS: 0.2, Y_E: null });
    expect(u.mean).toBeCloseTo(0.2, 9);
    expect(u.bands.map((b) => b.v)).toEqual([0.2, null, null, null]);
    expect(gateUsage(null)).toEqual({ mean: null, max: null, bands: [
      { band: "VIS", short: "VIS", v: null }, { band: "Y_E", short: "Y", v: null },
      { band: "J_E", short: "J", v: null }, { band: "H_E", short: "H", v: null }] });
  });
});

describe("member search", () => {
  const hay = { num: "196", loss: "l2", knee: "multi ×6 → 10", label: "196·psnr" };
  it("matches every word against the number, loss and knee (case-insensitive)", () => {
    expect(memberMatches(hay, "")).toBe(true);
    expect(memberMatches(hay, "196")).toBe(true);
    expect(memberMatches(hay, "#196")).toBe(true);
    expect(memberMatches(hay, "19")).toBe(true);
    expect(memberMatches(hay, "L2 multi")).toBe(true);
    expect(memberMatches(hay, "l1")).toBe(false);
    expect(memberMatches(hay, "196 l1")).toBe(false);
  });
});

describe("disagreement status", () => {
  it("says what the selection shows, in plain words", () => {
    expect(movieStatus([])).toBe("Pick members: one shows its SR, two or more play the disagreement movie");
    expect(movieStatus(["196"])).toBe("Showing member #196");
    expect(movieStatus(["196", "195", "190"])).toBe("Movie over 3 members: #196, #195, #190");
    expect(movieStatus(["1", "2", "3", "4", "5", "6", "7"])).toBe("Movie over 7 members: #1, #2, #3, #4, #5 and 2 more");
  });
});

describe("held-out loss comparability", () => {
  const knee = "per-field relative asinh MSE: … (1.0 = best member); at every loss knee, averaged over knees";
  it("compares a variant's held-out loss only when its loss definition is production's", () => {
    expect(heldOutComparable({ fit: { loss: knee } }, { fit: { loss: knee } })).toBe(true);
    expect(heldOutComparable({ fit: { loss: "band-weighted asinh squared error (1.0 = best member)" } }, { fit: { loss: knee } })).toBe(false);
    expect(heldOutComparable({ fit: {} }, { fit: { loss: knee } })).toBe(false);
    // no production definition to compare with: nothing to flag
    expect(heldOutComparable({ fit: { loss: knee } }, undefined)).toBe(true);
    expect(heldOutComparable({ fit: { loss: knee } }, { fit: {} })).toBe(true);
  });
});

describe("real-data benchmark", () => {
  const exp = (id: string, created: string, specs: string[], n = 4) => ({
    id, created, summary: Object.fromEntries(specs.map((s, i) => [s, { n_tiles: n, summary: { hole_pct_mean: 10 + i, hole_pct_max: 20 + i, median_R: 0.9, pct_R_lt_0p8: 5 } }])),
  });
  it("scores every variant from ONE experiment: the newest that ran production", () => {
    const b = benchmarkExperiment([
      exp("e-new", "2026-09-26", ["gate:26m"], 1),
      exp("e-prod", "2026-09-25", ["production", "gate:linear"], 9),
      exp("e-old", "2026-09-20", ["production", "gate:26m"]),
    ]);
    expect(b?.expId).toBe("e-prod");
    expect(b?.nTiles).toBe(9);
    expect([...(b?.bySpec.keys() ?? [])]).toEqual(["production", "gate:linear"]);
    expect(b?.bySpec.get("gate:26m")).toBeUndefined();
    expect(b?.bySpec.get("gate:linear")).toMatchObject({ holeMean: 11, holeMax: 21 });
  });
  it("falls back to the newest experiment, and to nothing", () => {
    expect(benchmarkExperiment([exp("a", "2026-09-01", ["gate:x"]), exp("b", "2026-09-02", ["gate:y"])])?.expId).toBe("b");
    expect(benchmarkExperiment([])).toBeNull();
    expect(benchmarkExperiment(undefined)).toBeNull();
  });
});

describe("train resources and continue targets", () => {
  it("seeds the resources from the newest finished training job, else the recipe", () => {
    expect(defaultResources([{ ...JOB, req_cpus: 16, req_memory: "48G", req_time_limit: "3:00:00" }]))
      .toEqual({ n_cpus: "16", memory: "48G", time_limit: "3:00:00", from: "48107719" });
    expect(defaultResources([{ ...JOB, state: "RUNNING", req_cpus: 2, req_time_limit: "1:00:00" }]))
      .toEqual({ ...RECIPE_RESOURCES, from: null });
    expect(defaultResources([])).toEqual({ ...RECIPE_RESOURCES, from: null });
    expect(RECIPE_RESOURCES).toEqual({ n_cpus: "16", memory: "32G", time_limit: "3:00:00" });
  });
  it("seeds a new batch from the last finished NEW batch, never a newer short continue job", () => {
    const add = { ...JOB, jobid: "100", mode: "add", req_cpus: 16, req_memory: "32G", req_time_limit: "3:00:00" };
    const cont = { ...JOB, jobid: "200", mode: "continue", req_cpus: 8, req_memory: "16G", req_time_limit: "1:00:00" };
    const jobs = [cont, add];   // newest first
    expect(defaultResources(jobs)).toEqual({ n_cpus: "16", memory: "32G", time_limit: "3:00:00", from: "100" });
    expect(defaultResources(jobs, "add").from).toBe("100");
    expect(defaultResources(jobs, "fork").from).toBe("100");
    expect(defaultResources(jobs, "continue")).toEqual({ n_cpus: "8", memory: "16G", time_limit: "1:00:00", from: "200" });
    // no job of the kind: the recipe, not another kind's resources
    expect(defaultResources([cont], "add")).toEqual({ ...RECIPE_RESOURCES, from: null });
    expect(defaultResources([add], "continue")).toEqual({ ...RECIPE_RESOURCES, from: null });
  });
  it("continues members up to their recorded target", () => {
    const rows = [{ name: "member_178", target_steps: 70000 }, { name: "member_179", target_steps: 80000 }, { name: "member_7", target_steps: null }];
    expect(continueTarget(rows, ["member_178", "member_179"])).toBe(80000);
    expect(continueTarget(rows, ["member_7"])).toBeNull();
    expect(continueTarget(rows, [])).toBeNull();
  });
});

describe("stampKnee (the pixel back-trace's per-row knee)", () => {
  it("is the traced pixel's own level: the largest of |target|, σ and |err|", () => {
    expect(stampKnee({ hr_val: 3.1, std_val: 1.2, err_val: 1.6 })).toBeCloseTo(3.1);
    expect(stampKnee({ hr_val: -11, std_val: 1.2, err_val: 1.6 })).toBeCloseTo(11);
    expect(stampKnee({ hr_val: 0.2, std_val: 1.61, err_val: 0.4 })).toBeCloseTo(1.61);
  });
  it("never below the floor, and ignores non-finite values", () => {
    expect(stampKnee({ hr_val: 0, std_val: 0, err_val: 0 })).toBe(0.25);
    expect(stampKnee({ hr_val: NaN, std_val: 2, err_val: Infinity })).toBe(2);
  });
});
