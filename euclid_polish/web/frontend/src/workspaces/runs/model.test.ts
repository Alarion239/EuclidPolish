import { describe, expect, it } from "vitest";
import { matchPage } from "../../app/manifest";
import type { Job } from "../../api/jobs";
import {
  STEP_STAGES, accountingNotes, cpuUsage, exitCodeText, finiteNumber, formatMemory, gpuUsage, hasGpu, historyItems, isLiveState,
  jobLabel, jobStateTone, liveItems, memberNames, membersText, memoryMegabytes, memoryUsage, mergeCliJobs,
  pageForLine, paramsSummary, progressText, rowParams, rowState, scopeCounts, stageOfStep, stepHome, stepsByStage,
  varyingParams, wallPer1k, type HistoryRow,
} from "./model";

describe("SLURM states", () => {
  it("tones", () => {
    expect(jobStateTone("COMPLETED")).toBe("good");
    expect(jobStateTone("CANCELLED by 1000")).toBe("bad");
    expect(jobStateTone("OUT_OF_MEMORY")).toBe("bad");
    expect(jobStateTone("RUNNING")).toBe("info");
    expect(jobStateTone("PENDING")).toBe("warn");
    expect(jobStateTone("UNKNOWN")).toBe("warn");
    expect(jobStateTone("")).toBeUndefined();
    expect(isLiveState("running")).toBe(true);
    expect(isLiveState("DONE")).toBe(false);
  });
  it("row state prefers sacct, then the DB", () => {
    expect(rowState({ jobid: "1", state_display: "COMPLETED" })).toBe("COMPLETED");
    expect(rowState({ jobid: "1", state: "", db_state: "UNKNOWN" })).toBe("UNKNOWN");
    expect(rowState({ jobid: "1" })).toBe("PENDING");
  });
});

describe("usage figures", () => {
  it("parses numbers and memory", () => {
    expect(finiteNumber("4")).toBe(4);
    expect(finiteNumber("")).toBeNull();
    expect(finiteNumber(true)).toBeNull();
    expect(memoryMegabytes("8G")).toBe(8192);
    expect(memoryMegabytes("512M")).toBe(512);
    expect(memoryMegabytes("8000Mc")).toBe(8000);
    expect(memoryMegabytes(2048)).toBe(2048);
    expect(memoryMegabytes("lots")).toBeNull();
    expect(formatMemory(20480)).toBe("20 GB");
    expect(formatMemory(1536)).toBe("1.5 GB");
    expect(formatMemory(300)).toBe("300 MB");
    expect(formatMemory(null)).toBe("—");
  });
  it("CPU used from peak utilisation or sacct efficiency", () => {
    expect(cpuUsage({ jobid: "1", req_cpus: "8", cpu_util_peak: "50", cpu_util_mean: "20" }))
      .toEqual({ used: 4, requested: 8, pct: 50, mean: 20 });
    expect(cpuUsage({ jobid: "1", req_cpus: "4", cpu_efficiency: "0.25" }).pct).toBe(25);
  });
  it("memory from Jobstats first, then MaxRSS over the request", () => {
    expect(memoryUsage({ jobid: "1", max_rss_mb: "2048", req_memory: "8G" }))
      .toEqual({ used: 2048, requested: 8192, pct: 25 });
    expect(memoryUsage({ jobid: "1", jobstats_cpu_memory_used_mb: "100", jobstats_cpu_memory_alloc_mb: "200" }).pct).toBe(50);
  });
  it("GPU memory: an absolute gpu_mem_peak_mb row never reads as a percentage", () => {
    expect(gpuUsage({ jobid: "1", gpu_mem_peak: "18000", gpu_mem_peak_mb: "18000" }).memPct).toBeNull();
    expect(gpuUsage({ jobid: "1", gpu_mem_peak: "40" }).memPct).toBe(40);
    expect(hasGpu({ jobid: "1", req_gpus: "1" })).toBe(true);
    expect(hasGpu({ jobid: "1", req_gpus: "0" })).toBe(false);
  });
  it("notes", () => {
    expect(accountingNotes({ jobid: "1", jobstats_notes_json: '["low GPU use", 3, "idle"]' })).toBe("low GPU use idle");
    expect(accountingNotes({ jobid: "1", jobstats_notes_json: "{bad" })).toBe("");
  });
  it("notes: the seff sign-off, the advice and the links are dropped; the finding stays", () => {
    expect(accountingNotes({ jobid: "1", jobstats_notes_json: '["Have a nice day!"]' })).toBe("");
    const long = ["The overall CPU utilization of this job is 0.5%. This value is low compared to the target range of 90% and above. "
      + "Please investigate the reason for the low efficiency. For instance, have you conducted a scaling analysis? "
      + "For more info: https://docs.rc.fas.harvard.edu/kb/job-efficiency-and-optimization-best-practices/#Cores",
    "This job failed because it exceeded the time limit. If there are no other problems then the solution is to increase the value "
      + "of the --time Slurm directive and resubmit the job. For more info: https://docs.rc.fas.harvard.edu/kb/x/#Time",
    "Have a nice day!"];
    expect(accountingNotes({ jobid: "1", jobstats_notes_json: JSON.stringify(long) })).toBe(
      "The overall CPU utilization of this job is 0.5%. This value is low compared to the target range of 90% and above. "
      + "This job failed because it exceeded the time limit.");
  });
  it("exit code: only a non-zero code is worth showing", () => {
    expect(exitCodeText({ jobid: "1", state: "COMPLETED", exit_code: "0:0" })).toBeNull();
    expect(exitCodeText({ jobid: "1", state: "TIMEOUT", exit_code: "0:0" })).toBeNull();
    expect(exitCodeText({ jobid: "1", state: "FAILED", exit_code: "1:0" })).toBe("1:0");
    expect(exitCodeText({ jobid: "1", state: "OUT_OF_MEMORY", exit_code: "0:125" })).toBe("0:125");
    expect(exitCodeText({ jobid: "1" })).toBeNull();
  });
});

describe("params across runs", () => {
  const rows: HistoryRow[] = [
    { jobid: "3", params: { num_stars: "10000", snr_min: "50", seed: "" } },
    { jobid: "2", params_json: '{"num_stars":"200","snr_min":"50"}' },
    { jobid: "1", params: { num_stars: "10000", snr_min: "50", seed: "" } },
  ];
  it("reads compact params or params_json", () => {
    expect(rowParams(rows[1])).toEqual({ num_stars: "200", snr_min: "50" });
    expect(rowParams({ jobid: "x", params_json: "[1]" })).toEqual({});
  });
  it("keeps only the params that vary, in schema order", () => {
    expect(varyingParams(rows, ["snr_min", "num_stars", "seed"])).toEqual(["num_stars"]);
    expect(varyingParams(rows, ["num_stars"], 0)).toEqual([]);
  });
});

describe("log paging", () => {
  it("finds the page holding a line (page 0 = the newest lines)", () => {
    expect(pageForLine(1000, 1000, 100)).toBe(0);
    expect(pageForLine(901, 1000, 100)).toBe(0);
    expect(pageForLine(900, 1000, 100)).toBe(1);
    expect(pageForLine(1, 1000, 100)).toBe(9);
    expect(pageForLine(5, 0, 100)).toBe(0);
  });
});

describe("paramsSummary", () => {
  it("skips internal underscore keys, resources and blanks", () => {
    expect(paramsSummary({ _star_prior_file: "logs/x.json", _star_prior_sha256: "c18f", partition: "gpu", n_cpus: 8,
      array_count: 4, loss: "l1", seed: "" })).toBe("loss=l1");
    expect(paramsSummary({ a: 1, b: 2, c: 3 }, 2)).toBe("a=1 · b=2");
    expect(paramsSummary(null)).toBe("");
  });
  it("leaves out array and seed bookkeeping and puts the member-defining knobs first", () => {
    expect(paramsSummary({
      array_count: 4, array_max_parallel: 4, base_seed: 4039766209, batch_size: 4, count: 4, member_names: ["member_199"],
      member_spec: "[{}]", crops_per_field: 2, loss: "l1", knee_loss: "multi", mode: "starfull", steps: 70000, icnr: 1,
    }, 5)).toBe("loss=l1 · knee_loss=multi · mode=starfull · steps=70000 · icnr=1");
  });
});

describe("mergeCliJobs", () => {
  it("appends squeue jobs the feed lacks, marked cli, and skips known array tasks", () => {
    const feed = [{ jobid: "100", state: "RUNNING", label: "train" }];
    const squeue = [
      { jobid: "100_0", name: "train", state: "RUNNING", reason: "holygpu1" },
      { jobid: "200", name: "smoke-single", state: "RUNNING", time: "5:00", time_limit: "1:00:00", reason: "holy7c" },
      { jobid: "300", name: "bench", state: "PENDING", reason: "(Priority)", start_time: "N/A" },
    ];
    const out = mergeCliJobs(feed, squeue);
    expect(out.map((j) => j.jobid)).toEqual(["100", "200", "300"]);
    expect(out[1]).toMatchObject({ cli: true, label: "smoke-single", nodes: "holy7c", reason: null, time: "5:00" });
    expect(out[2]).toMatchObject({ cli: true, state: "PENDING", reason: "(Priority)", nodes: null, start_time: null });
    expect(mergeCliJobs(feed, null)).toEqual(feed);
  });
});

describe("job labels name the members", () => {
  it("reads the member names of a new batch or a continue, and says them as a range", () => {
    expect(memberNames({ member_names: "member_199,member_200, member_201,member_202" }))
      .toEqual(["member_199", "member_200", "member_201", "member_202"]);
    expect(memberNames({ mode: "continue", members: "member_196,member_195", member_names: "x" }))
      .toEqual(["member_196", "member_195"]);
    expect(memberNames({ members: "member_1" })).toEqual([]);             // not a continue: not its members
    expect(membersText(["member_199", "member_200", "member_201", "member_202"])).toBe("members 199–202");
    expect(membersText(["member_196", "member_195"])).toBe("members 195–196");
    expect(membersText(["member_190", "member_196"])).toBe("members 190, 196");
    expect(membersText(["member_07"])).toBe("member 7");
    expect(membersText(["oddname"])).toBe("oddname");
    expect(membersText([])).toBe("");
  });
  it("adds the members to a SLURM job's label once", () => {
    expect(jobLabel({ label: "Train ensemble", step_id: "ensemble_train",
      params_json: JSON.stringify({ member_names: "member_199,member_200" }) })).toBe("Train ensemble · members 199–200");
    expect(jobLabel({ label: null, step_id: "ensemble_train", params: { member_names: "member_7" } }))
      .toBe("ensemble_train · member 7");
    expect(jobLabel({ label: "euclid_query", step_id: "euclid_query" })).toBe("euclid_query");
    expect(jobLabel({ label: "Train ensemble (N members, distinct seeds)",
      params: { member_names: "member_199,member_200,member_201,member_202" } })).toBe("Train ensemble · members 199–202");
    expect(jobLabel({ label: "members 199–200 continue", params: { mode: "continue", members: "member_199,member_200" } }))
      .toBe("members 199–200 continue");
  });
});

describe("progress", () => {
  it("says the step once with its share", () => {
    expect(progressText({ current: 10650, total: 70000, label: "step 10650" })).toBe("step 10,650 / 70,000 (15%)");
    expect(progressText({ current: 3, total: 12, label: "tile" })).toBe("tile 3 / 12 (25%)");
    expect(progressText({ current: 5, total: 10 })).toBe("step 5 / 10 (50%)");
    expect(progressText({ current: 5, total: 0 })).toBe("");
    expect(progressText(null)).toBe("");
  });
});

const job = (id: string, patch: Partial<Job> = {}): Job => ({
  job_id: id, label: `job ${id}`, kind: "k", status: "done", started: 1_790_000_000, finished: 1_790_000_060,
  duration: 60, error: null, log: null, log_truncated: false, cancellable: false, cancel_requested: false,
  result: null, progress: { current: 0, total: 0, pct: 0, label: "" }, ...patch,
});

describe("Runs › Live: one list of local and SLURM jobs", () => {
  it("lists the running local jobs and the live SLURM jobs, running first, with scope counts", () => {
    const local = [job("a", { status: "running", progress: { current: 2, total: 4, pct: 50, label: "tile" } }), job("b")];
    const slurm = [
      { jobid: "300", state: "PENDING", label: "Query", step_id: "euclid_query", submitted_at: 1_790_000_100, reason: "(Priority)" },
      { jobid: "200", state: "RUNNING", label: "Train", step_id: "ensemble_train", submitted_at: 1_790_000_050,
        params_json: JSON.stringify({ member_names: "member_199,member_200" }), nodes: "holygpu1", time: "1:00", time_limit: "3:00:00" },
    ];
    const items = liveItems(local, slurm);
    // running first (newest first among them), then the pending job
    expect(items.map((i) => i.key)).toEqual(["slurm:200", "local:a", "slurm:300"]);
    expect(items[1]).toMatchObject({ source: "local", state: "RUNNING", label: "job a", progress: "tile 2 / 4 (50%)" });
    expect(items[0]).toMatchObject({ source: "slurm", label: "Train · members 199–200", where: "holygpu1", elapsed: "1:00 / 3:00:00" });
    expect(items[2]).toMatchObject({ state: "PENDING", where: "(Priority)" });
    expect(scopeCounts(items)).toEqual({ all: 3, local: 1, slurm: 2 });
  });
});

describe("Runs › History: one ledger of local and SLURM runs", () => {
  it("merges the SLURM ledger with the finished local jobs, newest first", () => {
    const ledger: HistoryRow[] = [
      { jobid: "11", step_id: "euclid_query", submitted_at: "2026-09-20T10:00:00Z", state: "COMPLETED", elapsed_seconds: "120" },
      { jobid: "12", step_id: "ensemble_train", submitted_at: "2026-09-22T10:00:00Z", state: "TIMEOUT", req_gpus: "1",
        params: { member_names: "member_7,member_8" } },
    ];
    const local = [job("r", { status: "running" }), job("d", { started: Date.parse("2026-09-21T10:00:00Z") / 1000, duration: 30, status: "failed" })];
    const items = historyItems(ledger, local);
    expect(items.map((i) => i.key)).toEqual(["12", "local:d", "11"]);
    expect(items[0]).toMatchObject({ source: "slurm", state: "TIMEOUT", step: "ensemble_train", label: "ensemble_train · members 7–8" });
    expect(items[1]).toMatchObject({ source: "local", state: "FAILED", step: "k", elapsed: 30, label: "job d" });
    expect(items[2].elapsed).toBe(120);
  });
});

describe("wall time per 1000 steps", () => {
  it("keeps the run's members and their median", () => {
    const curves = [
      { name: "member_7", step_time: [[1000, 40], [2000, 44]] as [number, number][] },
      { name: "member_8", step_time: [[1000, 50]] as [number, number][] },
      { name: "member_9", step_time: [[1000, 99]] as [number, number][] },
    ];
    const w = wallPer1k(curves, ["member_7", "member_8"]);
    expect(w.members.map((m) => m.name)).toEqual(["member_7", "member_8"]);
    expect(w.median).toBe(44);
    expect(wallPer1k(curves, []).median).toBeNull();
  });
});

describe("Runs › Steps by stage", () => {
  const steps = ["archive_field_sample", "download_euclid_cutouts", "download_tng_skirt", "ensemble_train", "euclid_query",
    "euclid_verify_photometry", "extract_euclid_psf", "measure_tng_radii", "poster_cutout", "psf_rotation_pool",
    "synthetic_generate", "tng_grid", "tng_stack", "vis_noise_sample"];
  it("groups every registered step under one of the five stages, in order", () => {
    expect(STEP_STAGES.map((s) => s.label)).toEqual(["Reference data", "Noise and fields", "Generation", "Training", "Figures"]);
    expect(stageOfStep("vis_noise_sample")).toBe("noise-fields");
    expect(stageOfStep("ensemble_train")).toBe("training");
    expect(stageOfStep("brand_new")).toBe("other");
    const groups = stepsByStage(steps.map((step_id) => ({ step_id, label: step_id })));
    expect(groups.map((g) => g.id)).toEqual(["reference", "noise-fields", "generation", "training", "figures"]);
    expect(groups.flatMap((g) => g.steps).length).toBe(steps.length);
    expect(stepsByStage([{ step_id: "brand_new", label: "New" }]).map((g) => g.id)).toEqual(["other"]);
  });
  it("links every step to the tab whose drawer embeds it, a real page", () => {
    expect(stepHome("ensemble_train")).toEqual({ label: "Models › Train", to: "/models/train" });
    expect(stepHome("extract_euclid_psf")?.to).toBe("/synthetic/psf?view=epsf&how=1");
    for (const s of steps) {
      const home = stepHome(s);
      expect(home, s).toBeTruthy();
      expect(matchPage(home!.to.split("?")[0]), s).toBeTruthy();
    }
    expect(stepHome("brand_new")).toBeNull();
  });
});
