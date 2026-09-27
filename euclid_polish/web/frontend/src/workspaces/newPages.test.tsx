/* Behaviour of the pages WP-F3 adds (Ops › Jobs) against a mocked Flask.
 * Home and Settings moved to workspaces/home/Dashboard.test.tsx and
 * workspaces/settings/{Config,Settings}.test.tsx with their pages. */
import { render, screen, waitFor, within } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import type { ReactElement } from "react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useJobsStore, type Job } from "../api/jobs";
import { queryClient } from "../api/query";
import { useShellUi } from "../app/shellStore";
import { useDisplay } from "../state/display";
import { usePrefs } from "../state/prefs";
import Jobs from "./ops/tabs/Jobs";

type Reply = { status?: number; body: unknown };
let routes: Record<string, () => Reply>;
let calls: { url: string; method: string }[];

const VERSION = {
  boot_commit: "1111111aaaa", boot_short: "1111111", head_commit: "2222222bbbb", head_short: "2222222",
  behind: true, dirty: true, started_at: "2026-09-26T01:00:00Z", pid: 4242,
  dist: { built_at: "2026-09-26T00:00:00Z", index_hash: "deadbeef" },
};

const job = (id: string, patch: Partial<Job> = {}): Job => ({
  job_id: id, label: `job ${id}`, kind: "k", status: "done", started: 1_790_000_000, finished: 1_790_000_060,
  duration: 60, error: null, log: null, log_truncated: false, cancellable: false, cancel_requested: false,
  result: null, progress: { current: 0, total: 0, pct: 0, label: "" }, ...patch,
});

beforeEach(() => {
  calls = [];
  routes = {
    "GET /api/version": () => ({ body: VERSION }),
    "GET /api/fasrc/status": () => ({ body: { ssh_connected: false, last_error: "ssh: connect to host login.rc: timed out" } }),
    "GET /api/jobs?summary=1": () => ({ body: [job("a", { status: "running", cancellable: true }), job("b")] }),
    "GET /api/fasrc/current-submission": () => ({ status: 503, body: { ok: false, error: "FASRC not connected", code: "fasrc_offline" } }),
    // ensemble_gain_db is deliberately inconsistent: the dashboard derives
    // "vs mean member" itself (the full-eval writer stores gain vs BEST member).
    "GET /ensemble/status.json?mode=starfull": () => ({ body: { n_members: 26, eval_summary: {
      ensemble_psnr: 44.123, mean_member_psnr: 43.913, ensemble_gain_db: -0.4,
    } } }),
    "GET /api/status": () => ({ body: {
      catalog: { present: true, cached: true }, psfs: { bands: [{ name: "VIS", empirical: true }, { name: "Y_E", empirical: false }] },
      tfrecords: { dir: "x", files: [{}, {}, {}] }, checkpoints: { dir: "y", files: [{ member: "m1" }, { member: "m1" }, { member: "m2" }] },
    } }),
    "POST /api/fasrc/connect": () => ({ status: 400, body: { ok: false, error: "Permission denied (publickey)" } }),
  };
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = String(input);
    const method = init.method ?? "GET";
    calls.push({ url, method });
    const r = routes[`${method} ${url}`]?.() ?? { status: 404, body: { ok: false, error: `no route ${url}` } };
    return new Response(JSON.stringify(r.body), { status: r.status ?? 200 });
  }));
  queryClient.clear();
  useJobsStore.getState().reset();
  usePrefs.getState().reset();
  useDisplay.getState().reset();
  useShellUi.getState().closeAll();
});
afterEach(() => { queryClient.clear(); });

const show = (el: ReactElement) => render(
  <QueryClientProvider client={queryClient}>
    <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>{el}</MemoryRouter>
  </QueryClientProvider>,
);

describe("Ops › Jobs", () => {
  it("lists running jobs first", async () => {
    show(<Jobs />);
    const grid = await screen.findByRole("grid", { name: "Local jobs" });
    await waitFor(() => expect(within(grid).getAllByRole("row").length).toBe(3));   // header + 2
    expect(within(grid).getAllByRole("row")[1].textContent).toContain("job a");
  });
});
