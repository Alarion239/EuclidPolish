import { afterEach, describe, expect, it, vi } from "vitest";
import type { SlurmJob } from "../api/jobs";

const apiGet = vi.fn();
vi.mock("../api/client", async (orig) => ({
  ...(await orig<typeof import("../api/client")>()),
  apiGet: (...args: unknown[]) => apiGet(...args),
}));
const toastFns = vi.hoisted(() => ({
  success: vi.fn(), error: vi.fn(), warning: vi.fn(), info: vi.fn(),
}));
vi.mock("../ui", async (orig) => ({
  ...(await orig<typeof import("../ui")>()),
  toast: toastFns,
}));

import { announceSlurmEnd, finishedSlurmJobs, isLiveSlurmState, normalizeSlurmState } from "./JobTray";

const job = (jobid: string, state: string, extra: Partial<SlurmJob> = {}): SlurmJob => ({ jobid, state, ...extra });
const snap = (...jobs: SlurmJob[]) => new Map(jobs.map((j) => [j.jobid, j]));

afterEach(() => {
  apiGet.mockReset();
  for (const fn of Object.values(toastFns)) fn.mockReset();
});

describe("isLiveSlurmState", () => {
  it("treats every non-terminal state as live", () => {
    for (const s of ["RUNNING", "PENDING", "COMPLETING", "CONFIGURING", "SUSPENDED", "REQUEUED",
      "RESIZING", "STOPPED", "R", "PD", "CG", "CF", "S", ""]) {
      expect(isLiveSlurmState(s)).toBe(true);
    }
  });
  it("treats terminal states (long and short, with suffixes) as finished", () => {
    for (const s of ["COMPLETED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED",
      "CANCELLED by 12345", "CANCELLED+", "cd", "F", "TO", "CA", "PREEMPTED"]) {
      expect(isLiveSlurmState(s)).toBe(false);
    }
  });
});

describe("normalizeSlurmState", () => {
  it("expands short codes and upper-cases", () => {
    expect(normalizeSlurmState("cd")).toBe("COMPLETED");
    expect(normalizeSlurmState("TO")).toBe("TIMEOUT");
    expect(normalizeSlurmState("failed")).toBe("FAILED");
    expect(normalizeSlurmState(null)).toBe("");
  });
});

describe("finishedSlurmJobs", () => {
  it("reports nothing on the first snapshot (no previous state)", () => {
    expect(finishedSlurmJobs(null, [job("1", "RUNNING")])).toEqual([]);
  });

  it("reports a job that disappeared from the list", () => {
    const prev = snap(job("1", "RUNNING"), job("2", "PENDING"));
    expect(finishedSlurmJobs(prev, [job("2", "RUNNING")]).map((j) => j.jobid)).toEqual(["1"]);
  });

  it("keeps a RUNNING→COMPLETING job live (no early toast)", () => {
    const prev = snap(job("1", "RUNNING"));
    expect(finishedSlurmJobs(prev, [job("1", "COMPLETING")])).toEqual([]);
    expect(finishedSlurmJobs(prev, [job("1", "CONFIGURING")])).toEqual([]);
    expect(finishedSlurmJobs(prev, [job("1", "SUSPENDED")])).toEqual([]);
  });

  it("reports a job still listed but in a terminal state, with that state", () => {
    const prev = snap(job("1", "COMPLETING"));
    const out = finishedSlurmJobs(prev, [job("1", "COMPLETED")]);
    expect(out).toHaveLength(1);
    expect(out[0].state).toBe("COMPLETED");
  });

  it("ignores jobs that were never seen live", () => {
    expect(finishedSlurmJobs(snap(), [job("9", "COMPLETED")])).toEqual([]);
  });
});

describe("announceSlurmEnd", () => {
  it("uses the job DB's final state", async () => {
    apiGet.mockResolvedValue({ state: "TIMEOUT" });
    await announceSlurmEnd(job("7", "RUNNING", { label: "train m3" }));
    expect(apiGet).toHaveBeenCalledWith("/api/fasrc/jobs/7/status");
    expect(toastFns.error).toHaveBeenCalledWith("SLURM 7 · train m3", expect.objectContaining({ description: "timeout" }));
  });

  it("falls back to the listed terminal state when the DB has none", async () => {
    apiGet.mockResolvedValue({ state: null });
    await announceSlurmEnd(job("8", "CD"));
    expect(toastFns.success).toHaveBeenCalledWith("SLURM 8 · job 8", expect.objectContaining({ description: "completed" }));
  });

  it("says it left the queue when the state is unknown", async () => {
    apiGet.mockRejectedValue(new Error("offline"));
    await announceSlurmEnd(job("9", "RUNNING", { step_id: "evaluate" }));
    expect(toastFns.info).toHaveBeenCalledWith("SLURM 9 · evaluate", expect.objectContaining({ description: "left the SLURM queue" }));
  });

  it("toasts cancellations as a warning", async () => {
    apiGet.mockResolvedValue({ state: "CANCELLED by 501" });
    await announceSlurmEnd(job("10", "RUNNING"));
    expect(toastFns.warning).toHaveBeenCalledWith("SLURM 10 · job 10", expect.objectContaining({ description: "cancelled" }));
  });
});
