import { describe, expect, it } from "vitest";
import type { TrainingJob } from "./api";
import { benchmarkExperiment, productionRunsText } from "./model";
import {
  DEFAULT_MULTI_KNEES, RECIPE_RESOURCES, buildParams, buildSpec, continueTarget, defaultForm, defaultResources, formFromJob, lastBatch,
  newRow, recipeSummary, rowsFromSpec, validate,
} from "./trainModel";

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
    const spec = buildSpec([newRow({ kneeMode: "multi", outputKnee: "10", learnKnee: false }), newRow({ knee: "100", boot: "" })], "add");
    expect(spec[0]).toEqual({ loss: "l2", bootstrap: 0.7, asinh_knees: [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
      output_knee: 10, knee_loss: "balanced", num_res_blocks: 32 });              // ICNR off by default
    expect(spec[1]).toEqual({ loss: "l2", num_res_blocks: 32 });
    expect(buildSpec([newRow({ icnr: true })], "add")[0]).toMatchObject({ icnr: true });    // still on request
    expect(buildSpec([newRow()], "fork")[0]).not.toHaveProperty("num_res_blocks");
  });

  it("makes a new multi-knee member learn its output knee from 0.1 e⁻", () => {
    const [learned] = buildSpec([newRow({ kneeMode: "multi" })], "add");
    expect(learned).toMatchObject({ output_knee: 0.1, learn_output_knee: true });
    const [heads] = buildSpec([newRow({ kneeMode: "multi", outputKnee: "" })], "add");
    expect(heads).not.toHaveProperty("output_knee");
    expect(heads).not.toHaveProperty("learn_output_knee");
    const rows = rowsFromSpec([{ loss: "l2", asinh_knees: [0.1, 10], output_knee: 0.1, learn_output_knee: true }]);
    expect(rows[0]).toMatchObject({ kneeMode: "multi", outputKnee: "0.1", learnKnee: true });
    expect(buildSpec(rows, "add")[0]).toMatchObject({ output_knee: 0.1, learn_output_knee: true });
    const form = (outputKnee: string, learnKnee = true) => ({ ...defaultForm(), rows: [newRow({ kneeMode: "multi", outputKnee, learnKnee })] });
    expect(validate(form("0.1"))).toEqual([]);
    expect(validate(form("0.01")).join()).toMatch(/must start inside \(0.01, 10000\)/);
    expect(validate(form("0.01", false))).toEqual([]);           // a fixed knee has no bounds
  });

  it("round-trips a past job (clone) without seeds or names", () => {
    const f = formFromJob(JOB);
    expect(f.mode).toBe("add");
    expect(f.rows).toHaveLength(2);
    expect(f.rows[0]).toMatchObject({ kneeMode: "multi", outputKnee: "10", kneeLoss: "balanced", seed: "" });
    expect(f.rows[1]).toMatchObject({ kneeMode: "single", knee: "3000" });
    expect(f.geometry.batch_size).toBe("4");
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

  it("never sends the forward-model values System › Config owns (the step fills them from Config)", () => {
    const p = buildParams(formFromJob(JOB));
    for (const k of ["psf_warp_prob", "psf_warp_alpha_max", "psf_warp_sigma", "saturation_mask_prob"]) expect(p).not.toHaveProperty(k);
    expect(p).toMatchObject({ batch_size: "4", forward_onthefly: "1", hr_crop_size: "256", crops_per_field: "8", psf_subset: "64" });
  });

  it("sends the training step's arithmetic and clones it from a past job", () => {
    expect(buildParams(defaultForm())).toMatchObject({ xla: "0", bf16: "0" });
    const fast = { ...defaultForm(), geometry: { ...defaultForm().geometry, xla: true, bf16: true } };
    expect(buildParams(fast)).toMatchObject({ xla: "1", bf16: "1" });
    expect(formFromJob({ ...JOB, params: { ...JOB.params, xla: "1", bf16: "true" } }).geometry)
      .toMatchObject({ xla: true, bf16: true });
    expect(formFromJob(JOB).geometry).toMatchObject({ xla: false, bf16: false });
  });

  it("validates what the submit would refuse", () => {
    expect(validate(defaultForm())).toEqual([]);
    const bad = { ...defaultForm(), rows: [newRow({ kneeMode: "multi", knees: "0,1" })] };
    expect(validate(bad)[0]).toMatch(/knees must be positive/);
    expect(validate({ ...defaultForm(), mode: "continue" })).toContain("pick at least one member to continue");
    const geo = { ...defaultForm(), geometry: { ...defaultForm().geometry, hr_crop_size: "255" } };
    expect(validate(geo)).toContain("HR example side must be a positive even number");
    expect(validate({ ...defaultForm(), geometry: { ...defaultForm().geometry, batch_size: "0" } })).toContain("batch size must be a positive whole number");
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
  it("scores the experiment the user picked; an unknown pick falls back to the default", () => {
    const exps = [exp("e-new", "2026-09-26", ["gate:26m"], 1), exp("e-prod", "2026-09-25", ["production", "gate:linear"], 9)];
    expect(benchmarkExperiment(exps, "e-new")?.expId).toBe("e-new");
    expect(benchmarkExperiment(exps, "e-new")?.bySpec.get("production")).toBeUndefined();
    expect(benchmarkExperiment(exps, "gone")?.expId).toBe("e-prod");
    expect(benchmarkExperiment(exps, "")?.expId).toBe("e-prod");
  });
  it("names the tile set by the experiment label, else its tile count", () => {
    expect(benchmarkExperiment([exp("e", "2026-09-25", ["production"], 9)])?.tileSet).toBe("9 real tiles");
    expect(benchmarkExperiment([exp("e", "2026-09-25", ["production"], 1)])?.tileSet).toBe("1 real tile");
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
    // the faint knees 10⁻³ and 10⁻² are part of every new multi-knee member
    expect(DEFAULT_MULTI_KNEES).toBe("0.001,0.01,0.1,1,10,100,1000,10000");
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

describe("production member count", () => {
  it("says how many members production runs and why", () => {
    expect(productionRunsText(20, 30, 0.005)).toBe("Runs 20 of 30 members: those with ≥ 0.5% of the gate's weight somewhere");
    expect(productionRunsText(20, 30)).toBe("Runs 20 of 30 members: the ones the gate reads");
    expect(productionRunsText(30, 30)).toBe("Runs all 30 members");
    expect(productionRunsText(1, 1)).toBe("Runs its 1 member");
  });
});
