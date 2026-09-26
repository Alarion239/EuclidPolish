/* The local-job inspector polls while the job runs and stops for a job the
 * server no longer knows (a stale ?inspect=job:local/<id> link). */
import { act, render, screen } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useJobsStore, type Job } from "../../api/jobs";
import { queryClient } from "../../api/query";
import { JobInspector } from "./JobInspector";

const job = (id: string, patch: Partial<Job> = {}): Job => ({
  job_id: id, label: `job ${id}`, kind: null, status: "running", started: 1_790_000_000, finished: null,
  duration: 5, error: null, log: "hello", log_truncated: false, cancellable: false, cancel_requested: false,
  result: null, progress: { current: 1, total: 2, pct: 50, label: "" }, ...patch,
});

let replies: Record<string, () => { status: number; body: unknown }>;
let hits: Record<string, number>;

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  hits = {};
  replies = {};
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    hits[url] = (hits[url] ?? 0) + 1;
    const r = replies[url]?.() ?? { status: 404, body: { ok: false, error: "unknown job" } };
    return new Response(JSON.stringify(r.body), { status: r.status });
  }));
  queryClient.clear();
  useJobsStore.getState().reset();
});
afterEach(() => {
  vi.useRealTimers();
  queryClient.clear();
});

const show = (id: string) => render(
  <QueryClientProvider client={queryClient}><JobInspector id={id} /></QueryClientProvider>,
);
const wait = (ms: number) => act(() => vi.advanceTimersByTimeAsync(ms));

describe("JobInspector (local)", () => {
  it("stops polling a job the server does not know", async () => {
    show("local/gone");
    expect(await screen.findByText("Job not found")).toBeTruthy();
    const after404 = hits["/api/jobs/gone"];
    await wait(10_000);
    expect(hits["/api/jobs/gone"]).toBe(after404);
  });

  it("polls a running job and stops once it has finished", async () => {
    let status: Job["status"] = "running";
    replies["/api/jobs/r1"] = () => ({ status: 200, body: job("r1", { status }) });
    show("local/r1");
    expect(await screen.findByText(/hello/)).toBeTruthy();
    const first = hits["/api/jobs/r1"];
    await wait(4_500);
    expect(hits["/api/jobs/r1"]).toBeGreaterThan(first);
    status = "done";
    await wait(2_500);
    const done = hits["/api/jobs/r1"];
    await wait(10_000);
    expect(hits["/api/jobs/r1"]).toBe(done);
  });
});
