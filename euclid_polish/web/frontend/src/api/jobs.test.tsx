import { act, renderHook, waitFor } from "@testing-library/react";
import { focusManager, onlineManager } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  JOBS_FEED_TIMING,
  LOCAL_FEED_URL,
  SLURM_FEED_URL,
  cancelJob,
  cancelSlurmJob,
  isTerminal,
  useJob,
  useJobsFeed,
  useJobsStore,
  useTrackedJob,
  type Job,
} from "./jobs";
import { queryClient } from "./query";
import { useJob as compatUseJob, useTrackedJob as compatUseTrackedJob } from "../jobs";

/* ── fake Flask: a tiny local job registry behind a mocked fetch ─────────── */

type Route = (init: RequestInit) => { status?: number; body: unknown } | Promise<{ status?: number; body: unknown }>;

function server() {
  const routes = new Map<string, Route>();
  const calls: { url: string; method: string; body?: FormData | string }[] = [];
  const fn = vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = String(input);
    calls.push({ url, method: init.method ?? "GET", body: init.body as FormData | string | undefined });
    const route = routes.get(`${init.method ?? "GET"} ${url}`);
    if (!route) return new Response(JSON.stringify({ ok: false, error: `no route ${url}` }), { status: 404 });
    const r = await route(init);
    return new Response(JSON.stringify(r.body), { status: r.status ?? 200 });
  });
  vi.stubGlobal("fetch", fn);
  return {
    on(method: string, url: string, route: Route) { routes.set(`${method} ${url}`, route); },
    count(url: string, method = "GET") { return calls.filter((c) => c.url === url && c.method === method).length; },
    calls,
  };
}

function job(id: string, patch: Partial<Job> = {}): Job {
  return {
    job_id: id, label: `job ${id}`, kind: null, status: "running", started: 1000, finished: null,
    duration: 1, error: null, log: `log of ${id}`, log_truncated: false, cancellable: true,
    cancel_requested: false, result: null,
    progress: { current: 0, total: 0, pct: 0, label: "" },
    ...patch,
  };
}

const DEFAULT_TIMING = { ...JOBS_FEED_TIMING };

beforeEach(() => {
  queryClient.clear();
  useJobsStore.getState().reset();
  focusManager.setFocused(undefined);
  Object.assign(JOBS_FEED_TIMING, {
    localActiveMs: 25, localIdleMs: 1_000, slurmActiveMs: 40, slurmIdleMs: 1_000, slurmOfflineMs: 1_000,
    detailMinMs: 5, detailMaxMs: 20,
  });
});
afterEach(() => {
  queryClient.clear();
  vi.unstubAllGlobals();
  focusManager.setFocused(undefined);
  Object.assign(JOBS_FEED_TIMING, DEFAULT_TIMING);
});

describe("isTerminal", () => {
  it("treats every non-running status as terminal", () => {
    expect(isTerminal("running")).toBe(false);
    for (const s of ["done", "failed", "cancelled"]) expect(isTerminal(s)).toBe(true);
  });
});

describe("useJob", () => {
  it("POSTs the form, polls the job until it ends and calls onDone once", async () => {
    const srv = server();
    let polls = 0;
    srv.on("POST", "/api/ensemble/evaluate", () => ({ body: { job_id: "j1" } }));
    srv.on("GET", "/api/jobs/j1", () => {
      polls += 1;
      return { body: polls < 3 ? job("j1") : job("j1", { status: "done", log: "final log", result: { n: 3 } }) };
    });
    srv.on("GET", LOCAL_FEED_URL, () => ({ body: [] }));
    const onDone = vi.fn();
    const { result } = renderHook(() => useJob());
    expect(result.current).toMatchObject({ job: null, error: null, busy: false });
    await act(() => result.current.run("/api/ensemble/evaluate", { force: 1, skip: null }, { onDone }));
    expect(result.current.busy).toBe(true);
    await waitFor(() => expect(result.current.job?.status).toBe("done"));
    expect(result.current.busy).toBe(false);
    expect(result.current.job?.log).toBe("final log");
    expect(result.current.job?.result).toEqual({ n: 3 });
    expect(onDone).toHaveBeenCalledTimes(1);
    expect(onDone.mock.calls[0][0].status).toBe("done");
    const post = srv.calls.find((c) => c.method === "POST")!;
    expect([...(post.body as FormData).entries()]).toEqual([["force", "1"]]);
    await new Promise((r) => setTimeout(r, 40));
    expect(polls).toBe(3);                                     // stopped after the terminal snapshot
  });

  it("reports a {error} body and HTTP errors without polling", async () => {
    const srv = server();
    srv.on("POST", "/api/soft", () => ({ body: { error: "no members selected" } }));
    srv.on("POST", "/api/hard", () => ({ status: 409, body: { error: "a fit is already running" } }));
    const { result } = renderHook(() => useJob());
    await act(() => result.current.run("/api/soft"));
    expect(result.current).toMatchObject({ error: "no members selected", busy: false, job: null });
    await act(() => result.current.run("/api/hard"));
    expect(result.current).toMatchObject({ error: "a fit is already running", busy: false });
    expect(srv.calls.filter((c) => c.method === "GET")).toEqual([]);
  });

  it("calls onDone at once for a synchronous (non-job) response", async () => {
    const srv = server();
    srv.on("POST", "/api/sync", () => ({ body: { ok: true, n: 2 } }));
    const onDone = vi.fn();
    const { result } = renderHook(() => useJob());
    await act(() => result.current.run("/api/sync", {}, { onDone }));
    expect(onDone).toHaveBeenCalledWith({ ok: true, n: 2 });
    expect(result.current.busy).toBe(false);
  });

  it("says so when the job disappears (server restarted)", async () => {
    const srv = server();
    srv.on("POST", "/api/run", () => ({ body: { job_id: "gone" } }));
    srv.on("GET", "/api/jobs/gone", () => ({ status: 404, body: { ok: false, error: "unknown job gone" } }));
    srv.on("GET", LOCAL_FEED_URL, () => ({ body: [] }));
    const { result } = renderHook(() => useJob());
    await act(() => result.current.run("/api/run"));
    await waitFor(() => expect(result.current.error).toMatch(/no longer available/));
    expect(result.current.busy).toBe(false);
  });

  it("registers started jobs globally so they survive navigation (keyed re-attach)", async () => {
    const srv = server();
    let done = false;
    srv.on("POST", "/api/fit", () => ({ body: { job_id: "fit1" } }));
    srv.on("GET", "/api/jobs/fit1", () => ({ body: done ? job("fit1", { status: "done" }) : job("fit1") }));
    srv.on("GET", LOCAL_FEED_URL, () => ({ body: [] }));
    const first = renderHook(() => useJob("ensemble:fit"));
    await act(() => first.result.current.run("/api/fit"));
    await waitFor(() => expect(first.result.current.job?.status).toBe("running"));
    first.unmount();                                           // navigate away
    expect(useJobsStore.getState().started).toContain("fit1");
    done = true;
    const again = renderHook(() => useJob("ensemble:fit"));  // navigate back
    expect(again.result.current.job?.job_id).toBe("fit1");
    expect(again.result.current.busy).toBe(true);
    await waitFor(() => expect(again.result.current.job?.status).toBe("done"));
    expect(again.result.current.busy).toBe(false);
    act(() => again.result.current.reset());
    expect(again.result.current.job).toBeNull();
  });

  it("is re-exported by the old jobs.tsx module", () => {
    expect(compatUseJob).toBe(useJob);
    expect(compatUseTrackedJob).toBe(useTrackedJob);
  });
});

describe("useJobsFeed", () => {
  it("polls the local summary fast while something runs and slowly when idle", async () => {
    const srv = server();
    let list = [job("a", { log: null }), job("b", { status: "done", log: null })];
    srv.on("GET", LOCAL_FEED_URL, () => ({ body: list }));
    const { result, unmount } = renderHook(() => useJobsFeed({ slurm: false }));
    await waitFor(() => expect(result.current.jobs.map((j) => j.job_id).sort()).toEqual(["a", "b"]));
    expect(result.current.running.map((j) => j.job_id)).toEqual(["a"]);
    await waitFor(() => expect(srv.count(LOCAL_FEED_URL)).toBeGreaterThanOrEqual(3));   // 25 ms active cadence
    list = [job("a", { status: "done", log: null }), job("b", { status: "done", log: null })];
    await waitFor(() => expect(result.current.running).toEqual([]));
    const idleStart = srv.count(LOCAL_FEED_URL);
    await new Promise((r) => setTimeout(r, 120));
    expect(srv.count(LOCAL_FEED_URL)).toBeLessThanOrEqual(idleStart + 1);             // 1 s idle cadence
    unmount();
  });

  it("pauses while the tab is hidden", async () => {
    const srv = server();
    srv.on("GET", LOCAL_FEED_URL, () => ({ body: [job("a", { log: null })] }));
    const { result, unmount } = renderHook(() => useJobsFeed({ slurm: false }));
    await waitFor(() => expect(result.current.running.length).toBe(1));
    act(() => focusManager.setFocused(false));
    await new Promise((r) => setTimeout(r, 30));
    const paused = srv.count(LOCAL_FEED_URL);
    await new Promise((r) => setTimeout(r, 100));
    expect(srv.count(LOCAL_FEED_URL)).toBe(paused);
    act(() => focusManager.setFocused(true));
    await waitFor(() => expect(srv.count(LOCAL_FEED_URL)).toBeGreaterThan(paused));
    unmount();
  });

  it("shares one poll between subscribers and keeps a job's full log over summaries", async () => {
    const srv = server();
    JOBS_FEED_TIMING.localActiveMs = 10_000;                   // only the initial fetch in this test
    useJobsStore.getState().upsert(job("a", { log: "detailed log" }));
    srv.on("GET", LOCAL_FEED_URL, () => ({ body: [job("a", { log: null, progress: { current: 5, total: 10, pct: 50, label: "x" } })] }));
    const a = renderHook(() => useJobsFeed({ slurm: false }));
    const b = renderHook(() => useJobsFeed({ slurm: false }));
    await waitFor(() => expect(a.result.current.jobs[0]?.progress.pct).toBe(50));
    expect(a.result.current.jobs[0].log).toBe("detailed log");
    expect(b.result.current.jobs[0].progress.pct).toBe(50);
    expect(srv.count(LOCAL_FEED_URL)).toBe(1);
    a.unmount(); b.unmount();
  });

  it("keeps finished jobs started here but drops vanished running ones (server restart)", () => {
    const store = useJobsStore.getState();
    store.upsert(job("mine-done", { status: "done" }));
    store.upsert(job("mine-running"));
    store.upsert(job("other"));
    store.register("mine-done");
    store.register("mine-running");
    store.mergeFeed([job("new", { log: null })]);
    expect(Object.keys(useJobsStore.getState().jobs).sort()).toEqual(["mine-done", "new"]);
  });

  it("never regresses a finished job to running from a late snapshot", () => {
    const store = useJobsStore.getState();
    store.upsert(job("a", { status: "done", log: "end" }));
    store.upsert(job("a", { status: "running", log: "older" }));
    expect(useJobsStore.getState().jobs.a.status).toBe("done");
    expect(useJobsStore.getState().jobs.a.log).toBe("end");
  });

  it("lists live SLURM jobs from current-submission", async () => {
    const srv = server();
    srv.on("GET", LOCAL_FEED_URL, () => ({ body: [] }));
    srv.on("GET", SLURM_FEED_URL, () => ({
      body: {
        ok: true, stale: false,
        current: { job: { jobid: "48107719", state: "RUNNING", label: "train" }, status: null },
        live: [{ jobid: "48107719", state: "RUNNING", label: "train" }, { jobid: "48107720", state: "PENDING", label: "eval" }],
        queue: { count: 1, names: ["next"], halted: false },
      },
    }));
    const { result, unmount } = renderHook(() => useJobsFeed());
    await waitFor(() => expect(result.current.slurm.map((j) => j.jobid)).toEqual(["48107719", "48107720"]));
    expect(result.current.slurmQueue?.count).toBe(1);
    expect(result.current.fasrcOffline).toBe(false);
    expect(result.current.runningCount).toBe(2);
    unmount();
  });

  it("falls back to the single current job when `live` is absent", async () => {
    const srv = server();
    srv.on("GET", LOCAL_FEED_URL, () => ({ body: [] }));
    srv.on("GET", SLURM_FEED_URL, () => ({
      body: { ok: true, current: { job: { jobid: "7", state: "PENDING" }, status: null }, queue: null },
    }));
    const { result, unmount } = renderHook(() => useJobsFeed());
    await waitFor(() => expect(result.current.slurm.map((j) => j.jobid)).toEqual(["7"]));
    unmount();
  });

  it("keeps polling both feeds after the browser fires `offline` (Flask is local)", async () => {
    queryClient.mount();
    try {
      act(() => { window.dispatchEvent(new Event("offline")); });
      expect(onlineManager.isOnline()).toBe(false);
      const srv = server();
      srv.on("GET", LOCAL_FEED_URL, () => ({ body: [job("a", { log: null })] }));
      srv.on("GET", SLURM_FEED_URL, () => ({
        status: 503, body: { ok: false, error: "FASRC not connected", code: "fasrc_offline" },
      }));
      const { result, unmount } = renderHook(() => useJobsFeed());
      await waitFor(() => expect(result.current.running.map((j) => j.job_id)).toEqual(["a"]));
      await waitFor(() => expect(result.current.fasrcOffline).toBe(true));
      await waitFor(() => expect(srv.count(LOCAL_FEED_URL)).toBeGreaterThanOrEqual(3));   // 25 ms active cadence
      unmount();
    } finally {
      act(() => { window.dispatchEvent(new Event("online")); });
      onlineManager.setOnline(true);
      queryClient.unmount();
    }
  });

  it("reports FASRC offline (C4 503) as an empty SLURM list, not an error", async () => {
    const srv = server();
    srv.on("GET", LOCAL_FEED_URL, () => ({ body: [] }));
    srv.on("GET", SLURM_FEED_URL, () => ({
      status: 503, body: { ok: false, error: "FASRC not connected", code: "fasrc_offline" },
    }));
    const { result, unmount } = renderHook(() => useJobsFeed());
    await waitFor(() => expect(result.current.fasrcOffline).toBe(true));
    expect(result.current.slurm).toEqual([]);
    expect(result.current.slurmError).toBeNull();
    unmount();
  });
});

describe("cancelJob", () => {
  it("POSTs the cancel and marks the job as cancel-requested", async () => {
    const srv = server();
    useJobsStore.getState().upsert(job("a"));
    srv.on("POST", "/api/jobs/a/cancel", () => ({ body: { ok: true } }));
    srv.on("GET", LOCAL_FEED_URL, () => ({ body: [] }));
    await expect(cancelJob("a")).resolves.toEqual({ ok: true });
    expect(useJobsStore.getState().jobs.a).toMatchObject({ cancel_requested: true, cancellable: false });
  });

  it("returns the server's refusal", async () => {
    const srv = server();
    srv.on("POST", "/api/jobs/b/cancel", () => ({ status: 409, body: { ok: false, error: "job b is already done" } }));
    await expect(cancelJob("b")).resolves.toEqual({ ok: false, error: "job b is already done" });
  });
});

describe("useJob().cancel", () => {
  it("cancels the hook's current job and reports the server's reply", async () => {
    const srv = server();
    srv.on("POST", "/api/run", () => ({ body: { job_id: "c1" } }));
    srv.on("GET", "/api/jobs/c1", () => ({ body: job("c1") }));
    srv.on("GET", LOCAL_FEED_URL, () => ({ body: [] }));
    srv.on("POST", "/api/jobs/c1/cancel", () => ({ body: { ok: true } }));
    const { result } = renderHook(() => useJob());
    await act(() => result.current.run("/api/run"));
    await waitFor(() => expect(result.current.job?.status).toBe("running"));
    let reply: { ok: boolean; error?: string } | undefined;
    await act(async () => { reply = await result.current.cancel(); });
    expect(reply).toEqual({ ok: true });
    expect(srv.count("/api/jobs/c1/cancel", "POST")).toBe(1);
    expect(result.current.job).toMatchObject({ cancel_requested: true, cancellable: false });
  });

  it("refuses without a job (no request)", async () => {
    const srv = server();
    const { result } = renderHook(() => useJob());
    await expect(result.current.cancel()).resolves.toEqual({ ok: false, error: "no job" });
    expect(srv.calls).toEqual([]);
  });
});

describe("cancelSlurmJob", () => {
  it("POSTs the jobid to /api/fasrc/cancel and refreshes the feed", async () => {
    const srv = server();
    srv.on("POST", "/api/fasrc/cancel", () => ({ body: { ok: true } }));
    await expect(cancelSlurmJob("48107719")).resolves.toEqual({ ok: true });
    const post = srv.calls.find((c) => c.url === "/api/fasrc/cancel")!;
    expect(post.method).toBe("POST");
    expect([...(post.body as FormData).entries()]).toEqual([["jobid", "48107719"]]);
  });

  it("returns a refusal in the body ({ok:false}) and HTTP errors (FASRC offline) as {ok:false, error}", async () => {
    const srv = server();
    srv.on("POST", "/api/fasrc/cancel", () => ({ body: { ok: false, error: "scancel: invalid job id" } }));
    await expect(cancelSlurmJob("1")).resolves.toEqual({ ok: false, error: "scancel: invalid job id" });
    srv.on("POST", "/api/fasrc/cancel", () => ({ body: { ok: false } }));
    await expect(cancelSlurmJob("1")).resolves.toEqual({ ok: false, error: "cancel refused" });
    srv.on("POST", "/api/fasrc/cancel", () => ({
      status: 503, body: { ok: false, error: "FASRC not connected", code: "fasrc_offline" },
    }));
    await expect(cancelSlurmJob("1")).resolves.toEqual({ ok: false, error: "FASRC not connected" });
  });
});

describe("job detail poll", () => {
  it("skips ticks while the tab is hidden and resumes when it is visible again", async () => {
    const srv = server();
    srv.on("POST", "/api/run", () => ({ body: { job_id: "h1" } }));
    srv.on("GET", "/api/jobs/h1", () => ({ body: job("h1") }));
    srv.on("GET", LOCAL_FEED_URL, () => ({ body: [] }));
    const { result, unmount } = renderHook(() => useJob());
    await act(() => result.current.run("/api/run"));
    await waitFor(() => expect(srv.count("/api/jobs/h1")).toBeGreaterThanOrEqual(2));
    act(() => focusManager.setFocused(false));
    await new Promise((r) => setTimeout(r, 30));                 // let an in-flight tick land
    const hidden = srv.count("/api/jobs/h1");
    await new Promise((r) => setTimeout(r, 100));                // ≥ 5 skipped 20 ms ticks
    expect(srv.count("/api/jobs/h1")).toBe(hidden);
    act(() => focusManager.setFocused(true));
    await waitFor(() => expect(srv.count("/api/jobs/h1")).toBeGreaterThan(hidden));
    unmount();
  });
});

describe("useTrackedJob", () => {
  it("attaches to a running job by label from the shared feed and follows it to the end", async () => {
    const srv = server();
    let status: Job["status"] = "running";
    srv.on("GET", LOCAL_FEED_URL, () => ({
      body: [job("x", { label: "other", log: null }), job("fit", { label: "combiner: fit starfull on validate", log: null, status })],
    }));
    srv.on("GET", "/api/jobs/fit", () => ({ body: job("fit", { label: "combiner: fit starfull on validate", status }) }));
    const { result, unmount } = renderHook(() => useTrackedJob("combiner: fit starfull"));
    await waitFor(() => expect(result.current?.job_id).toBe("fit"));
    expect(result.current?.log).toBe("log of fit");
    status = "done";
    await waitFor(() => expect(result.current?.status).toBe("done"));
    unmount();
  });

  it("stays null (without its own poller) while nothing matches", async () => {
    const srv = server();
    srv.on("GET", LOCAL_FEED_URL, () => ({ body: [job("x", { label: "other", log: null, status: "done" })] }));
    const a = renderHook(() => useTrackedJob("combiner: fit"));
    const b = renderHook(() => useTrackedJob("combiner: fit starless"));
    await waitFor(() => expect(srv.count(LOCAL_FEED_URL)).toBe(1));
    await new Promise((r) => setTimeout(r, 50));
    expect(a.result.current).toBeNull();
    expect(b.result.current).toBeNull();
    expect(srv.count(LOCAL_FEED_URL)).toBe(1);                 // idle cadence (1 s), one shared query
    expect(srv.calls.every((c) => c.url === LOCAL_FEED_URL)).toBe(true);
    a.unmount(); b.unmount();
  });
});
