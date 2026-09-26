/* Behaviour of the pages WP-F3 adds (Home dashboard, Settings › Appearance /
 * About / Connections, Ops › Jobs) against a mocked Flask. */
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import type { ReactElement } from "react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useJobsStore, type Job } from "../api/jobs";
import { queryClient } from "../api/query";
import { useShellUi } from "../app/shellStore";
import { useDisplay } from "../state/display";
import { usePrefs } from "../state/prefs";
import Dashboard from "./home/Dashboard";
import Jobs from "./ops/tabs/Jobs";
import About from "./settings/tabs/About";
import Appearance from "./settings/tabs/Appearance";
import Connections from "./settings/tabs/Connections";

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

describe("Home dashboard", () => {
  it("shows connection, version, jobs, the ensemble mean and local data", async () => {
    show(<Dashboard />);
    expect(await screen.findByText("44.12")).toBeTruthy();
    expect(screen.getByText("Ensemble mean")).toBeTruthy();
    expect(screen.queryByText(/Production/)).toBeNull();
    expect(screen.getByText("+0.21 dB vs mean member")).toBeTruthy();
    expect(screen.getByText(/plain mean of 26 starfull members · no production-gate score/)).toBeTruthy();
    expect(await screen.findByText("1111111")).toBeTruthy();
    expect(screen.getByText("HEAD 2222222 — restart")).toBeTruthy();
    expect(await screen.findByText(/2 member checkpoints · ePSF 1\/2 · catalogue cached/)).toBeTruthy();
    expect(await screen.findByText("job a")).toBeTruthy();
    expect(screen.getByText("offline")).toBeTruthy();
  });
});

describe("Home dashboard · production gate", () => {
  it("headlines the production spatial gate when the summary scored it", async () => {
    routes["GET /ensemble/status.json?mode=starfull"] = () => ({ body: { n_members: 26, eval_summary_stale: true, eval_summary: {
      ensemble_psnr: 44.1, mean_member_psnr: 43.9, combiner_psnr: 99, // the RBF block: not production
      spatial_gate_combiner_psnr: 44.456, spatial_gate_combiner_vs_mean_db: 0.356,
      spatial_gate_combiner_vs_best_member_db: 0.12,
    } } });
    show(<Dashboard />);
    expect(await screen.findByText("44.46")).toBeTruthy();
    expect(screen.getByText("Production gate")).toBeTruthy();
    expect(screen.getByText("+0.36 dB vs plain mean")).toBeTruthy();
    expect(screen.getByText(/spatial gate · \+0\.12 dB vs best member · plain mean 44\.10 dB · 26 starfull members · summary stale/)).toBeTruthy();
    expect(screen.queryByText("99.00")).toBeNull();
  });
});

describe("Settings › Appearance", () => {
  it("edits theme, accent, density, rail and the viewer wheel", async () => {
    show(<Appearance />);
    fireEvent.click(screen.getByRole("radio", { name: "Dark" }));
    expect(usePrefs.getState().theme).toBe("dark");
    fireEvent.click(screen.getByRole("button", { name: "Accent teal" }));
    expect(usePrefs.getState().accent).toBe("teal");
    fireEvent.click(screen.getByRole("radio", { name: "Compact" }));
    expect(usePrefs.getState().density).toBe("compact");
    fireEvent.click(screen.getByRole("switch", { name: /Collapse the navigation rail/ }));
    expect(usePrefs.getState().railCollapsed).toBe(true);
    fireEvent.change(screen.getByLabelText("Mouse wheel over a viewer"), { target: { value: "scroll" } });
    expect(useDisplay.getState().wheel).toBe("scroll");
    fireEvent.click(screen.getByText("Open the Display panel"));
    expect(useShellUi.getState().display).toBe(true);
    fireEvent.click(screen.getByText("Reset appearance"));
    expect(usePrefs.getState().theme).toBe("light");
  });
});

describe("Settings › About", () => {
  it("shows boot vs HEAD, the dirty tree and the dist build", async () => {
    show(<About />);
    expect(await screen.findByText("Restart the server")).toBeTruthy();
    expect(screen.getByText("1111111aaaa")).toBeTruthy();
    expect(screen.getByText("2222222bbbb")).toBeTruthy();
    expect(screen.getByText("uncommitted changes")).toBeTruthy();
    expect(screen.getByText("deadbeef")).toBeTruthy();
  });
});

describe("Settings › Connections", () => {
  it("shows the real last error and surfaces a failed connect", async () => {
    show(<Connections />);
    expect(await screen.findByText("ssh: connect to host login.rc: timed out")).toBeTruthy();
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Connect" })); });
    await waitFor(() => expect(calls.some((c) => c.method === "POST" && c.url === "/api/fasrc/connect")).toBe(true));
    // the status is re-read after the attempt
    await waitFor(() => expect(calls.filter((c) => c.url === "/api/fasrc/status").length).toBeGreaterThan(1));
  });
});

describe("Ops › Jobs", () => {
  it("lists running jobs first", async () => {
    show(<Jobs />);
    const list = await screen.findByRole("list");
    await waitFor(() => expect(within(list).getAllByRole("listitem")).toHaveLength(2));
    expect(within(list).getAllByRole("listitem")[0].textContent).toContain("job a");
  });
});
