import { describe, expect, it } from "vitest";
import {
  accountingNotes, basename, cpuUsage, finiteNumber, formatMemory, gitStatusText, gpuUsage, mergeCliJobs, mergeCommitPages, paramsSummary, hasGpu, isLiveState,
  jobStateTone, memoryMegabytes, memoryUsage, pageForLine, parentDir, relationText, rowParams, rowState,
  varyingParams, type HistoryRow,
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

describe("logs, git and paths", () => {
  it("finds the page holding a line (page 0 = the newest lines)", () => {
    expect(pageForLine(1000, 1000, 100)).toBe(0);
    expect(pageForLine(901, 1000, 100)).toBe(0);
    expect(pageForLine(900, 1000, 100)).toBe(1);
    expect(pageForLine(1, 1000, 100)).toBe(9);
    expect(pageForLine(5, 0, 100)).toBe(0);
  });
  it("describes the FASRC-vs-local relation", () => {
    expect(relationText({ relation: "same" }).tone).toBe("good");
    expect(relationText({ relation: "remote_behind", ahead: 1 }).label).toBe("FASRC 1 commit behind");
    expect(relationText({ relation: "remote_ahead", behind: 3 }).label).toBe("FASRC 3 commits ahead");
    expect(relationText(null).label).toBe("unknown");
  });
  it("names porcelain states", () => {
    expect(gitStatusText("??")).toBe("untracked");
    expect(gitStatusText(" M")).toBe("modified");
    expect(gitStatusText("R ")).toBe("renamed");
    expect(gitStatusText("UU")).toBe("conflict");
  });
  it("splits paths", () => {
    expect(basename("/n/data/psf.fits")).toBe("psf.fits");
    expect(basename("/n/data/")).toBe("data");
    expect(parentDir("/n/data/psf.fits")).toBe("/n/data");
    expect(parentDir("/n")).toBe("/");
  });
});

describe("paramsSummary", () => {
  it("skips internal underscore keys, resources and blanks", () => {
    expect(paramsSummary({ _star_prior_file: "logs/x.json", _star_prior_sha256: "c18f", partition: "gpu", n_cpus: 8,
      array_count: 4, loss: "l1", seed: "" })).toBe("array_count=4 · loss=l1");
    expect(paramsSummary({ a: 1, b: 2, c: 3 }, 2)).toBe("a=1 · b=2");
    expect(paramsSummary(null)).toBe("");
  });
});

describe("mergeCommitPages", () => {
  it("appends a page and drops commits already loaded", () => {
    const c = (h: string) => ({ hash: h, full: h + "full", author: "a", relative: "now", subject: h });
    expect(mergeCommitPages([c("a"), c("b")], [c("b"), c("c")]).map((x) => x.hash)).toEqual(["a", "b", "c"]);
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
