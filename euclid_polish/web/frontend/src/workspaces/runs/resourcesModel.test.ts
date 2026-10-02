import { describe, expect, it } from "vitest";
import type { RunUsage, StepSummary } from "./api";
import {
  barTone, chronological, finishedRuns, memorySeries, outcomeClauses, pctText, runAt, runTicks, stepMeta, submitHome,
  timeSeries, usedFraction, utilText, waste, workText, yAxis, yTop,
} from "./resourcesModel";

const RUN = (p: Partial<RunUsage> & { jobid: string }): RunUsage => ({
  submitted_at: null, state: "COMPLETED", partition: "gpu", cpus: 16, gpus: 1, req_memory: "32G", req_memory_mb: 32768,
  req_time_limit: "3:00:00", req_time_s: 10800, elapsed_s: 9000, cpu_efficiency: 0.5, cores_used: 8, peak_mem_mb: 16384,
  mem_ratio: 0.5, time_ratio: 0.83, gpu_util: 70, gpu_mem_used_mb: 79000, units: 70000, units_label: "steps",
  key_label: "new · batch 16", label: "Train ensemble", ...p,
});

const SUMMARY: StepSummary = {
  step_id: "ensemble_train", label: "Train ensemble", needs_gpu: true, registered: true, runs: 58,
  states: { completed: 28, oom: 3, timeout: 16, failed: 6, cancelled: 5, running: 0 },
  success_rate: 0.5, last_submitted_at: "2026-09-27T21:25:02Z", cpu_efficiency: 0.42, gpu_util: 71.4, mem_ratio: 0.4,
  time_ratio: 0.9, peak_mem_p90_mb: 20000, cpu_hours_alloc: 2000, cpu_hours_used: 800, gpu_hours_alloc: 120,
  gpu_hours_used: 85, mem_gb_hours_alloc: 4000, mem_gb_hours_used: 1500,
};

describe("summary numbers", () => {
  it("counts the finished runs and words what did not complete, worst first", () => {
    expect(finishedRuns(SUMMARY.states)).toBe(58);
    expect(outcomeClauses(SUMMARY.states).map((c) => [c.n, c.text, c.tone])).toEqual([
      [3, "ran out of memory", "bad"], [16, "hit the time limit", "warn"], [6, "failed", undefined], [5, "were cancelled", undefined],
    ]);
    expect(outcomeClauses({ ...SUMMARY.states, oom: 0, timeout: 0, failed: 0, cancelled: 1, running: 1 }).map((c) => c.text))
      .toEqual(["was cancelled", "is still running"]);
  });
  it("formats shares and utilisations", () => {
    expect(pctText(0.536)).toBe("54%");
    expect(pctText(null)).toBe("—");
    expect(pctText(0.004)).toBe("<1%");                       // 188 MB of 32 GB is not "0%"
    expect(pctText(0)).toBe("0%");
    expect(utilText(71.4)).toBe("71%");
  });
  it("gives the idle share of the allocated hours", () => {
    expect(waste(2000, 800)).toEqual({ alloc: 2000, used: 800, idle: 1200, share: 0.6 });
    expect(waste(100, 130)?.idle).toBe(0);                     // never negative
    expect(waste(null, 5)).toBeNull();
    expect(waste(0, 0)).toBeNull();
  });
  it("writes the step list line; GPU only on a GPU step", () => {
    expect(stepMeta(SUMMARY)).toBe("58 runs · 50% completed · CPU 42% · GPU 71%");
    expect(stepMeta({ ...SUMMARY, needs_gpu: false, runs: 1, success_rate: null })).toBe("1 run · CPU 42%");
  });
  it("links each step to where it is submitted", () => {
    expect(submitHome("ensemble_train")).toEqual({ label: "Models › Train", to: "/models/train" });
    expect(submitHome("synthetic_generate").label).toBe("Synthetic › Records");
    expect(submitHome("lensfinder_train")).toEqual({ label: "Runs › Steps", to: "/runs/steps?step=lensfinder_train" });
  });
});

describe("per-run charts", () => {
  const runs = [
    RUN({ jobid: "3", submitted_at: "2026-09-20T10:00:00Z", state: "OUT_OF_MEMORY", peak_mem_mb: null, elapsed_s: 600 }),
    RUN({ jobid: "2", submitted_at: "2026-09-14T10:00:00Z", state: "TIMEOUT", elapsed_s: 10810 }),
    RUN({ jobid: "1", submitted_at: "2026-09-14T09:00:00Z" }),
  ];
  const ordered = chronological(runs);
  it("orders the runs oldest first", () => {
    expect(ordered.map((r) => r.jobid)).toEqual(["1", "2", "3"]);
    // a run without a date goes first; equal dates keep the oldest-first order of a newest-first list
    expect(chronological([RUN({ jobid: "b" }), RUN({ jobid: "a" })]).map((r) => r.jobid)).toEqual(["a", "b"]);
  });
  it("draws memory in GB and marks the OOM run at what it asked for", () => {
    const m = memorySeries(ordered);
    expect(m.x).toEqual([1, 2, 3]);
    expect(m.requested).toEqual([32, 32, 32]);
    expect(m.used).toEqual([16, 16, null]);
    expect(m.marked).toEqual({ x: [3], y: [32] });
    expect(yTop(m)).toBeCloseTo(32 * 1.08);
    expect(yAxis(m)).toEqual({ scale: "linear", domain: [0, yTop(m)] });
  });
  it("switches the y axis to log when one run dwarfs the rest", () => {
    const s = { x: [1, 2], requested: [30, 0.25], used: [0.1, 0.2], marked: { x: [], y: [] } };
    expect(yAxis(s)).toEqual({ scale: "log", domain: [0.1 / 1.5, 30 * 1.5] });
  });
  it("draws time in hours and marks the TIMEOUT run at its elapsed", () => {
    const t = timeSeries(ordered);
    expect(t.requested).toEqual([3, 3, 3]);
    expect(t.used[0]).toBeCloseTo(2.5);
    expect(t.marked.x).toEqual([2]);
    expect(t.marked.y[0]).toBeCloseTo(10810 / 3600);
  });
  it("ticks at evenly spaced runs by submission day, without repeating a day", () => {
    expect(runTicks(ordered).map((t) => t.v)).toEqual([1, 3]);          // runs 1 and 2 share 09-14
    expect(runTicks(ordered)[0].label).toMatch(/^09-1[34]$/);
    expect(runTicks([])).toEqual([]);
  });
  it("names run x for the tooltip", () => {
    expect(runAt(ordered, 1.2)).toMatch(/^#1 · 2026-09-1[34]$/);
    expect(runAt(ordered, 9)).toBe("");
  });
});

describe("the run table", () => {
  it("gives used ÷ requested only when both are known", () => {
    expect(usedFraction(8, 16)).toBe(0.5);
    expect(usedFraction(null, 16)).toBeNull();
    expect(usedFraction(3, 0)).toBeNull();
  });
  it("tones an OOM or over-asked memory bar bad and a timed-out time bar warn", () => {
    expect(barTone("memory", 0.9, "OUT_OF_MEMORY")).toBe("bad");
    expect(barTone("memory", 1.05, "COMPLETED")).toBe("bad");
    expect(barTone("memory", 0.5, "COMPLETED")).toBeUndefined();
    expect(barTone("time", 1, "TIMEOUT")).toBe("warn");
    expect(barTone("cpu", 0.1, "TIMEOUT")).toBeUndefined();
  });
  it("words the work of a run", () => {
    expect(workText({ units: 70000, units_label: "steps" })).toBe("70,000 steps");
    expect(workText({ units: null, units_label: "images" })).toBe("");
  });
});
