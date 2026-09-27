/* startJob's one-at-a-time guard across entry points: a Home health-check fix
 * and the palette/quick action for the same endpoint share one key. */
import { afterEach, describe, expect, it, vi } from "vitest";
import { useJobsStore, type Job } from "../api/jobs";

const apiPost = vi.fn();
vi.mock("../api/client", async (orig) => ({
  ...(await orig<typeof import("../api/client")>()),
  apiPost: (...args: unknown[]) => apiPost(...args),
}));
vi.mock("../ui", async (orig) => ({
  ...(await orig<typeof import("../ui")>()),
  toast: { success: vi.fn(), error: vi.fn(), warning: vi.fn(), info: vi.fn() },
  confirm: vi.fn(async () => true),
}));

import { jobKeyFor, startJob } from "./RunActions";

const running = (id: string): Job => ({
  job_id: id, label: "knee", status: "running", duration: 1, error: null, log: null, log_truncated: false,
  progress: {} as Job["progress"],
});

afterEach(() => { apiPost.mockReset(); useJobsStore.getState().reset(); });

describe("jobKeyFor", () => {
  it("maps a shared runner's endpoint to that runner's key", () => {
    expect(jobKeyFor({ key: "check:knee", url: "/ensemble/knee-psnr" })).toBe("run:knee");
    expect(jobKeyFor({ key: "check:evals", url: "/ensemble/evaluate" })).toBe("run:evaluate");
  });
  it("keeps the spec's own key for other endpoints", () => {
    expect(jobKeyFor({ key: "check:x", url: "/somewhere/else" })).toBe("check:x");
  });
});

describe("startJob", () => {
  it("does not start a check's knee job while the palette's knee job runs", async () => {
    useJobsStore.getState().upsert(running("j1"));
    useJobsStore.getState().register("j1", "run:knee");
    const id = await startJob({ key: "check:knee", label: "Recompute", url: "/ensemble/knee-psnr" });
    expect(id).toBe("j1");
    expect(apiPost).not.toHaveBeenCalled();
  });

  it("registers a check-started job under the shared key", async () => {
    apiPost.mockResolvedValue({ ok: true, job_id: "j2" });
    const id = await startJob({ key: "check:evals", label: "Evaluate", url: "/ensemble/evaluate" });
    expect(id).toBe("j2");
    expect(useJobsStore.getState().keyed["run:evaluate"]).toBe("j2");
  });
});
