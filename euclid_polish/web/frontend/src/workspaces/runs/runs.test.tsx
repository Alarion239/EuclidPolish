/* Runs workspace against a mocked Flask: the schema-driven step card (shared
 * by every page that embeds a step), the SLURM monitor, Live (one list of
 * local and SLURM jobs, the queue), History (one ledger, its filters, Clone /
 * Logs, the log side panel, wall time) and Steps (by stage, home tabs). */
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import type { ReactElement } from "react";
import { MemoryRouter, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useJobsStore } from "../../api/jobs";
import { queryClient } from "../../api/query";
import { resetConfirm } from "../../ui";
import { QueuePanel } from "./Queue";
import { SlurmMonitor } from "./steps/SlurmMonitor";
import { StepCard } from "./steps/StepCard";
import type { Step } from "./steps/stepForm";
import History from "./tabs/History";
import Live from "./tabs/Live";
import Steps from "./tabs/Steps";

type Reply = { status?: number; body: unknown };
type Call = { url: string; method: string; form: Record<string, string | string[]> };
let routes: Record<string, (form: Call["form"], url: URL) => Reply>;
let calls: Call[];
let location = "";

const formOf = (body: BodyInit | null | undefined): Call["form"] => {
  const out: Call["form"] = {};
  if (!(body instanceof FormData)) return out;
  body.forEach((v, k) => {
    const cur = out[k];
    out[k] = cur === undefined ? String(v) : Array.isArray(cur) ? [...cur, String(v)] : [cur, String(v)];
  });
  return out;
};

beforeEach(() => {
  calls = [];
  routes = {
    "GET /api/jobs": () => ({ body: [] }),
    "GET /api/fasrc/status": () => ({ body: { ssh_connected: true, connected_at: 1_790_000_000 } }),
    "GET /api/fasrc/queue/state": () => ({ body: { ok: true, queue: { count: 0, names: [], items: [], halted: false } } }),
    "GET /api/fasrc/current-submission": () => ({ body: { ok: true, current: null, live: [], queue: { count: 0, names: [], items: [], halted: false } } }),
    "GET /api/fasrc/queue": () => ({ body: { ok: true, rows: [] } }),
    "GET /api/tracking/state": () => ({ body: { active: { title: "multi-knee" }, archived: [{ title: "July", _dir: "2026-07-july" }],
      jobs_count: 2, unassigned_count: 0 } }),
  };
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = new URL(String(input), "http://localhost");
    const method = init.method ?? "GET";
    const form = formOf(init.body);
    calls.push({ url: `${url.pathname}${url.search}`, method, form });
    const handler = routes[`${method} ${url.pathname}${url.search}`] ?? routes[`${method} ${url.pathname}`];
    const r = handler?.(form, url) ?? { status: 404, body: { ok: false, error: `no route ${url.pathname}` } };
    return new Response(JSON.stringify(r.body), { status: r.status ?? 200 });
  }));
  queryClient.clear();
  useJobsStore.getState().reset();
});
afterEach(() => {
  act(() => resetConfirm());
  queryClient.clear();
  vi.unstubAllGlobals();
});

function LocationProbe() {
  const loc = useLocation();
  location = `${loc.pathname}${loc.search}`;
  return null;
}
const params = () => new URLSearchParams(location.split("?")[1] ?? "");

const show = (el: ReactElement, url = "/runs/live") => render(
  <QueryClientProvider client={queryClient}>
    <MemoryRouter initialEntries={[url]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>{el}<LocationProbe /></MemoryRouter>
  </QueryClientProvider>,
);
const posts = (path: string) => calls.filter((c) => c.method === "POST" && c.url === path);
const gets = (prefix: string) => calls.filter((c) => c.method === "GET" && c.url.startsWith(prefix));
const answer = async (title: RegExp | string, button: string) => {
  const dlg = await screen.findByRole("alertdialog", { name: title });
  fireEvent.click(within(dlg).getByRole("button", { name: button }));
  await waitFor(() => expect(screen.queryByRole("alertdialog", { name: title })).toBeNull());
  return dlg;
};

/* ── step card ───────────────────────────────────────────────────────────── */

const QUERY: Step = {
  step_id: "euclid_query", label: "Query Euclid catalog", needs_gpu: false,
  defaults: { partition: "shared", n_cpus: 1, n_gpus: 0, memory: "4G", time_limit: "30:00" },
  task_params: [
    { name: "num_stars", type: "int", default: 10000, min: 1, help: "Brightest N stars to keep." },
    { name: "magnitude_min", type: "float", default: 18, help: "Brightest VIS magnitude kept (blank = no bright cut)." },
    { name: "mode", type: "choice", default: "a", choices: ["a", "b"], help: "Mode." },
  ],
  last_params: { num_stars: 500, magnitude_min: null, mode: "a" },
  outputs: [{ key: "euclid_psf", path: "/n/data/euclid_psf/x.fits", exists: true }],
};

describe("StepCard (schema-driven)", () => {
  beforeEach(() => {
    routes["GET /api/fasrc/steps/euclid_query/history"] = () => ({ body: { ok: true, step_id: "euclid_query", match: null, history: [
      { jobid: "41", state: "COMPLETED", submitted_at: "2026-09-01T00:00:00Z", params: { num_stars: "200", mode: "b" },
        params_json: '{"num_stars":"200","mode":"b"}', req_cpus: "2", req_memory: "8G", req_time_limit: "1:00:00" },
    ] } });
  });

  it("renders the schema, prefilled from the last successful run", async () => {
    show(<StepCard step={QUERY} sshConnected />);
    expect(await screen.findByText("Prefilled from the last successful run")).toBeTruthy();
    expect((screen.getByLabelText("num stars") as HTMLInputElement).value).toBe("500");
    expect((screen.getByLabelText("magnitude min") as HTMLInputElement).value).toBe("18");
    expect(screen.getByText("euclid_psf")).toBeTruthy();
  });

  it("confirms, then posts the typed params + resources, and reports a queued submission", async () => {
    routes["POST /api/fasrc/steps/euclid_query/submit"] = () => ({ body: { ok: true, queued: true, label: "Query",
      queue: { count: 2, names: ["x", "Query"], items: [{ id: "a", label: "x", position: 1 }, { id: "b", label: "Query", position: 2 }], halted: false } } });
    show(<StepCard step={QUERY} sshConnected />);
    fireEvent.change(await screen.findByLabelText("num stars"), { target: { value: "42" } });
    fireEvent.click(screen.getByRole("button", { name: "Submit" }));
    const dlg = await answer(/Submit “Query Euclid catalog”/, "Submit");
    expect(dlg.textContent).toContain("num_stars = 42");
    await waitFor(() => expect(posts("/api/fasrc/steps/euclid_query/submit")).toHaveLength(1));
    expect(posts("/api/fasrc/steps/euclid_query/submit")[0].form).toEqual({
      n_cpus: "1", n_gpus: "0", memory: "4G", time_limit: "30:00",
      num_stars: "42", magnitude_min: "18", mode: "a", confirm: "yes",
    });
    expect(await screen.findByText(/Queued — position 2 of 2/)).toBeTruthy();
  });

  it("blocks an invalid value and shows the server's refusal", async () => {
    routes["POST /api/fasrc/steps/euclid_query/submit"] = () => ({ status: 400, body: { ok: false, error: "num_stars: must be ≥ 1 (got '0')" } });
    show(<StepCard step={QUERY} sshConnected />);
    fireEvent.change(await screen.findByLabelText("num stars"), { target: { value: "0" } });
    expect((screen.getByRole("button", { name: "Submit" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(screen.getByLabelText("num stars"), { target: { value: "5" } });
    fireEvent.click(screen.getByRole("button", { name: "Submit" }));
    await answer(/Submit/, "Submit");
    expect(await screen.findByText(/must be ≥ 1/)).toBeTruthy();
  });

  it("hides the params the host page controls and posts them as given", async () => {
    routes["POST /api/fasrc/steps/euclid_query/submit"] = () => ({ body: { ok: true, jobid: "99" } });
    show(<StepCard step={QUERY} sshConnected extraParams={{ num_stars: 7 }} />);
    await screen.findByLabelText("magnitude min");
    expect(screen.queryByLabelText("num stars")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Submit" }));
    await answer(/Submit/, "Submit");
    await waitFor(() => expect(posts("/api/fasrc/steps/euclid_query/submit")[0]?.form.num_stars).toBe("7"));
  });

  it("clones a past run into the form", async () => {
    show(<StepCard step={QUERY} sshConnected />);
    fireEvent.click(await screen.findByRole("button", { name: "Clone run 41 into the form" }));
    await waitFor(() => expect((screen.getByLabelText("num stars") as HTMLInputElement).value).toBe("200"));
    expect(screen.getByText("Cloned from run 41")).toBeTruthy();
  });

  it("keeps the archive re-download behind its own typed confirmation (re-homed from test/archiveFields)", async () => {
    const ARCHIVE: Step = {
      step_id: "archive_field_sample", label: "Download matched multipoint Euclid archive fields", needs_gpu: false,
      defaults: { partition: "shared", n_cpus: 1, n_gpus: 0, memory: "8G", time_limit: "4:00:00" },
      task_params: [{ name: "force_redownload", type: "bool", default: false, help: "Re-download every bundle (needs its own confirmation)." }],
      last_params: null,
    };
    routes["GET /api/fasrc/steps/archive_field_sample/history"] = () => ({ body: { ok: true, history: [], match: null } });
    routes["POST /api/fasrc/steps/archive_field_sample/submit"] = () => ({ body: { ok: true, jobid: "5" } });
    show(<StepCard step={ARCHIVE} sshConnected showHistory={false} />);
    // A resume keeps the cache: no destructive token is posted.
    fireEvent.click(await screen.findByRole("button", { name: "Submit" }));
    await answer(/Submit/, "Submit");
    await waitFor(() => expect(posts("/api/fasrc/steps/archive_field_sample/submit")).toHaveLength(1));
    expect(posts("/api/fasrc/steps/archive_field_sample/submit")[0].form.force_redownload).toBe("0");
    expect(posts("/api/fasrc/steps/archive_field_sample/submit")[0].form).not.toHaveProperty("confirm_force_redownload");
    // A forced re-download is a danger confirm that must be typed.
    fireEvent.click(screen.getByRole("switch", { name: "force redownload" }));
    fireEvent.click(screen.getByRole("button", { name: "Submit" }));
    const dlg = await screen.findByRole("alertdialog", { name: /Submit/ });
    expect(dlg.textContent).toContain("Destructive: Re-download every bundle");
    const go = within(dlg).getByRole("button", { name: "Submit anyway" }) as HTMLButtonElement;
    expect(go.disabled).toBe(true);
    fireEvent.change(within(dlg).getByRole("textbox"), { target: { value: "redownload" } });
    fireEvent.click(go);
    await waitFor(() => expect(posts("/api/fasrc/steps/archive_field_sample/submit")).toHaveLength(2));
    expect(posts("/api/fasrc/steps/archive_field_sample/submit")[1].form).toMatchObject({ force_redownload: "1", confirm_force_redownload: "yes" });
  });

  it("stays disabled offline", async () => {
    show(<StepCard step={QUERY} sshConnected={false} />);
    expect(((await screen.findByRole("button", { name: "Submit" })) as HTMLButtonElement).disabled).toBe(true);
  });
});

describe("Runs › Live queue section", () => {
  it("removes one item and resumes a halted queue only after confirming", async () => {
    let halted = true;
    routes["GET /api/fasrc/queue/state"] = () => ({ body: { ok: true, queue: { count: 1, names: ["grid"], halted,
      halted_reason: "job 1 ended FAILED", active_jobid: "1", items: [{ id: "q1", label: "grid", step: "tng_grid", position: 1, queued_at: 1790000000 }] } } });
    routes["POST /api/fasrc/queue/remove"] = () => ({ body: { ok: true } });
    routes["POST /api/fasrc/queue/resume"] = () => { halted = false; return { body: { ok: true } }; };
    show(<QueuePanel />);
    expect(await screen.findByText("job 1 ended FAILED")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Remove grid" }));
    await answer(/Remove “grid”/, "Cancel");
    expect(posts("/api/fasrc/queue/remove")).toHaveLength(0);
    fireEvent.click(screen.getByRole("button", { name: "Remove grid" }));
    await answer(/Remove “grid”/, "Remove");
    await waitFor(() => expect(posts("/api/fasrc/queue/remove")[0]?.form).toEqual({ id: "q1" }));
    fireEvent.click(screen.getByRole("button", { name: "Resume" }));
    await answer("Resume the queue?", "Resume");
    await waitFor(() => expect(posts("/api/fasrc/queue/resume")).toHaveLength(1));
  });
});

/* ── SLURM monitor ───────────────────────────────────────────────────────── */

describe("SlurmMonitor", () => {
  it("says the progress once with its share, and shows the ledger facts only after completion", async () => {
    const state: string = "RUNNING";
    routes["GET /api/fasrc/jobs/77/status"] = () => ({ body: { ok: true, jobid: "77", state,
      status: { stage: "training", step: { current: 10650, total: 70000, label: "step 10650" }, has_events: true,
        resources: { gpu_percent: 78, cpu_percent: 41 } } } });
    routes["GET /api/fasrc/history"] = () => ({ body: { ok: true, total: 1, offset: 0, limit: 20, unresolved: 0,
      facets: { steps: {}, states: {} }, rows: [{ jobid: "77", step_id: "ensemble_train", state: state === "RUNNING" ? "" : "COMPLETED",
        db_state: state, req_cpus: "16", elapsed_seconds: "600" }] } });
    show(<SlurmMonitor jobid="77" />);
    expect(await screen.findByText("step 10,650 / 70,000 (15%)")).toBeTruthy();
    expect(screen.queryByText(/10650 10,650/)).toBeNull();
    expect(screen.getByText("GPU 78%")).toBeTruthy();
    expect(screen.getByText("CPU 41%")).toBeTruthy();
    await waitFor(() => expect(gets("/api/fasrc/history").length).toBeGreaterThan(0));
    expect(screen.queryByText("elapsed")).toBeNull();          // ledger facts wait for the end
    expect(screen.getByRole("link", { name: "Logs" }).getAttribute("href")).toBe("/runs/history?run=77&logs=1");
  });

  it("shows the ledger's resource use once the job has ended", async () => {
    routes["GET /api/fasrc/jobs/78/status"] = () => ({ body: { ok: true, jobid: "78", state: "COMPLETED", status: { has_events: true } } });
    routes["GET /api/fasrc/history"] = () => ({ body: { ok: true, total: 1, offset: 0, limit: 20, unresolved: 0,
      facets: { steps: {}, states: {} }, rows: [{ jobid: "78", step_id: "euclid_query", state: "COMPLETED", req_cpus: "4",
        elapsed_seconds: "600" }] } });
    show(<SlurmMonitor jobid="78" />);
    expect(await screen.findByText("elapsed")).toBeTruthy();
    expect(screen.getByText("10m 00s")).toBeTruthy();
  });
});

/* ── Live ────────────────────────────────────────────────────────────────── */

const localJob = (id: string, patch: Record<string, unknown> = {}) => ({
  job_id: id, label: `job ${id}`, kind: "evaluate", status: "running", started: 1_790_000_000, finished: null,
  duration: 5, error: null, log: null, log_truncated: false, cancellable: true, cancel_requested: false,
  result: null, progress: { current: 1, total: 4, pct: 25, label: "field" }, ...patch,
});

describe("Runs › Live", () => {
  beforeEach(() => {
    routes["GET /api/jobs"] = () => ({ body: [localJob("a"), localJob("b", { status: "done" })] });
    routes["GET /api/fasrc/current-submission"] = () => ({ body: { ok: true, current: null, queue: { count: 0, items: [], halted: false },
      live: [{ jobid: "100", state: "RUNNING", label: "Train ensemble", step_id: "ensemble_train", submitted_at: 1_790_000_100,
        params_json: JSON.stringify({ member_names: "member_199,member_200,member_201,member_202" }), nodes: "holygpu1",
        time: "2:00:00", time_limit: "3:00:00" }] } });
    routes["GET /api/fasrc/queue"] = () => ({ body: { ok: true, rows: [
      { jobid: "100", name: "train", state: "RUNNING", reason: "holygpu1" },
      { jobid: "48096695", name: "smoke-single", state: "RUNNING", time: "3:00", time_limit: "1:00:00", reason: "holy7c" },
    ] } });
    routes["GET /api/fasrc/jobs/100/status"] = () => ({ body: { ok: true, jobid: "100", state: "RUNNING", status: { has_events: false } } });
    routes["GET /api/jobs/a"] = () => ({ body: { ...localJob("a"), log: "tick" } });
  });

  it("lists the local and SLURM jobs in one table, the labels naming the members, with scope counts", async () => {
    show(<Live />);
    const grid = await screen.findByRole("grid", { name: "Running and queued jobs" });
    expect(await within(grid).findByText("Train ensemble · members 199–202")).toBeTruthy();
    expect(within(grid).getByText("job a")).toBeTruthy();
    expect(within(grid).queryByText("job b")).toBeNull();                         // finished: in History
    expect((await within(grid).findAllByText("smoke-single")).length).toBeGreaterThan(0); // CLI, read-only
    expect(screen.queryByRole("button", { name: "Cancel smoke-single" })).toBeNull();
    expect(screen.getByRole("button", { name: "Cancel Train ensemble · members 199–202" })).toBeTruthy();
    const scope = screen.getByRole("radiogroup", { name: "Scope" });
    await waitFor(() => expect(within(scope).getByRole("radio", { name: "SLURM · 2" })).toBeTruthy());
    expect(within(scope).getByRole("radio", { name: "This laptop · 1" })).toBeTruthy();
    fireEvent.click(within(scope).getByRole("radio", { name: "This laptop · 1" }));
    await waitFor(() => expect(params().get("scope")).toBe("local"));
    await waitFor(() => expect(within(grid).queryByText("Train ensemble · members 199–202")).toBeNull());
  });

  it("opens a local job's log beside the list, and a SLURM job's monitor", async () => {
    show(<Live />, "/runs/live?job=local:a");
    expect(await screen.findByText("tick")).toBeTruthy();
    const grid = await screen.findByRole("grid", { name: "Running and queued jobs" });
    fireEvent.click(await within(grid).findByText("Train ensemble · members 199–202"));
    await waitFor(() => expect(params().get("job")).toBe("100"));
    await waitFor(() => expect(gets("/api/fasrc/jobs/100/status").length).toBeGreaterThan(0));
  });

  it("never says offline or empty before the FASRC status has answered", async () => {
    routes["GET /api/jobs"] = () => ({ body: [] });
    let release: (() => void) | null = null;
    routes["GET /api/fasrc/status"] = () => ({ body: { ssh_connected: false, last_error: "timed out" } });
    const base = globalThis.fetch;
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL, init?: RequestInit) => (String(input).includes("/api/fasrc/status")
      ? new Promise<Response>((resolve) => { release = () => resolve(base(input, init) as unknown as Response); })
      : base(input, init))));
    show(<Live />);
    await screen.findByRole("grid", { name: "Running and queued jobs" });
    expect(screen.queryByText("FASRC offline")).toBeNull();                   // neither the callout nor the chip
    expect(screen.queryByText("Nothing is running")).toBeNull();
    await act(async () => { release?.(); });
    expect(await screen.findByText(/SLURM jobs appear here once the SSH session is up/)).toBeTruthy();
  });
});

/* ── History ─────────────────────────────────────────────────────────────── */

const LEDGER = {
  ok: true, total: 2, offset: 0, limit: 2000, unresolved: 1,
  facets: { steps: { euclid_query: 1, ensemble_train: 1 }, states: { UNKNOWN: 1, TIMEOUT: 1 } },
  rows: [
    { jobid: "7", step_id: "euclid_query", state: "", db_state: "UNKNOWN", state_display: "UNKNOWN",
      submitted_at: "2026-09-01T00:00:00Z", params: { num_stars: "10" }, req_cpus: "2", req_gpus: "1" },
    { jobid: "9", step_id: "ensemble_train", label: "Train ensemble", state: "TIMEOUT", submitted_at: "2026-09-02T00:00:00Z",
      params: { member_names: "member_7,member_8" }, req_gpus: "1", jobstats_gpu_util: "78", elapsed_seconds: "10800",
      log_path: "logs/pipeline/train-9.out", err_path: "logs/pipeline/train-9.err" },
  ],
};

describe("Runs › History", () => {
  beforeEach(() => {
    routes["GET /api/fasrc/history"] = () => ({ body: LEDGER });
    routes["GET /api/fasrc/steps/status"] = () => ({ body: { ssh_connected: true, steps: [
      { ...QUERY, needs_gpu: false }, { ...QUERY, step_id: "ensemble_train", label: "Train", needs_gpu: true },
    ] } });
    routes["GET /api/jobs"] = () => ({ body: [localJob("d", { status: "failed", started: 1_790_000_000 })] });
  });

  it("lists the SLURM ledger and the finished local jobs, reconciles the unresolved ones as a job", async () => {
    routes["POST /api/fasrc/refresh-accounting"] = () => ({ body: { ok: true, job_id: "j1" } });
    routes["GET /api/jobs/j1"] = () => ({ body: { job_id: "j1", label: "reconcile", status: "done", duration: 1, error: null,
      log: "", log_truncated: false, result: { ok: true, updated: 1, total: 1, resolved: { "7": "COMPLETED" } },
      progress: { current: 1, total: 1, pct: 100, label: "" } } });
    show(<History />, "/runs/history");
    const grid = await screen.findByRole("grid", { name: "Run history" });
    expect(await within(grid).findByText("UNKNOWN")).toBeTruthy();
    expect(within(grid).getByText("Train ensemble · members 7–8")).toBeTruthy();
    expect(await within(grid).findByText("job d")).toBeTruthy();                  // a finished local job
    fireEvent.click(screen.getByRole("button", { name: /Reconcile 1/ }));
    await waitFor(() => expect(posts("/api/fasrc/refresh-accounting")[0]?.form).toEqual({ scope: "unresolved" }));
  });

  it("has labelled Clone and Logs buttons, and no GPU number on a CPU step", async () => {
    show(<History />, "/runs/history");
    const grid = await screen.findByRole("grid", { name: "Run history" });
    const clone = await within(grid).findByRole("link", { name: "Clone run 9" });
    expect(clone.textContent).toBe("Clone");
    expect(clone.getAttribute("href")).toBe("/runs/steps?step=ensemble_train&clone=9");
    const query = within(grid).getByText("UNKNOWN").closest("tr") as HTMLElement;
    const train = within(grid).getByText("Train ensemble · members 7–8").closest("tr") as HTMLElement;
    expect(train.textContent).toContain("78%");
    expect(query.textContent).not.toContain("mem");                       // euclid_query is a CPU step
    fireEvent.click(within(train).getByRole("button", { name: "Logs of Train ensemble · members 7–8" }));
    await waitFor(() => expect(params().get("run")).toBe("9"));
    expect(params().get("logs")).toBe("1");
  });

  it("filters by step (the interim hstep key is read once) and by campaign", async () => {
    routes["GET /api/tracking/jobs"] = () => ({ body: { ok: true, campaign: "current", total: 1, jobids: ["9"] } });
    show(<History />, "/runs/history?hstep=ensemble_train");
    await waitFor(() => expect(params().get("step")).toBe("ensemble_train"));
    expect(params().get("hstep")).toBeNull();
    await waitFor(() => expect(gets("/api/fasrc/history?limit=2000&step=ensemble_train").length).toBeGreaterThan(0));
    fireEvent.change(screen.getByRole("combobox", { name: "Campaign" }), { target: { value: "current" } });
    await waitFor(() => expect(gets("/api/tracking/jobs?campaign=current&ids=1").length).toBe(1));
    const grid = screen.getByRole("grid", { name: "Run history" });
    await waitFor(() => expect(within(grid).queryByText("UNKNOWN")).toBeNull());
    expect(within(grid).queryByText("job d")).toBeNull();                          // campaigns log FASRC jobs only
    expect(within(grid).getByText("Train ensemble · members 7–8")).toBeTruthy();
  });

  it("opens the selected run's log and, for a training run, its wall time per 1000 steps", async () => {
    routes["GET /api/fasrc/runs/log"] = (_f, u) => ({ body: { ok: true, path: u.searchParams.get("path"), page: 0, page_size: 1000,
      total_lines: 2, start_line: 1, end_line: 2, has_older: false, has_newer: false, content: "step 1\nstep 2" } });
    routes["GET /ensemble/training-curves.json"] = () => ({ body: { members: [
      { name: "member_7", step_time: [[1000, 40], [2000, 44]] }, { name: "member_8", step_time: [[1000, 50]] },
      { name: "member_9", step_time: [[1000, 99]] },
    ] } });
    show(<History />, "/runs/history?run=9&logs=1");
    expect(await screen.findByText(/step 1/)).toBeTruthy();
    expect(gets("/api/fasrc/runs/log?path=logs%2Fpipeline%2Ftrain-9.out").length).toBeGreaterThan(0);
    expect(await screen.findByText(/Median 44 s per 1000 steps over members 7–8/)).toBeTruthy();
  });
});

/* ── Steps ───────────────────────────────────────────────────────────────── */

describe("Runs › Steps", () => {
  const OTHER: Step = { ...QUERY, step_id: "vis_noise_sample", label: "Sample VIS noise", task_params: [], last_params: null, outputs: [] };
  const TRAIN: Step = { ...QUERY, step_id: "ensemble_train", label: "Train ensemble", needs_gpu: true, task_params: [], last_params: null, outputs: [] };
  beforeEach(() => {
    routes["GET /api/fasrc/steps/status"] = () => ({ body: { ok: true, ssh_connected: true, steps: [QUERY, OTHER, TRAIN] } });
    for (const s of ["vis_noise_sample", "euclid_query", "ensemble_train"]) {
      routes[`GET /api/fasrc/steps/${s}/history`] = () => ({ body: { ok: true, step_id: s, match: null, history: [] } });
    }
  });

  it("lists every step by stage, always visible, and names the step's home tab", async () => {
    show(<Steps />, "/runs/steps?step=ensemble_train");
    const nav = await screen.findByRole("navigation", { name: "FASRC steps by stage" });
    expect(within(nav).getAllByRole("heading").map((h) => h.textContent)).toEqual(["Reference data", "Noise and fields", "Training"]);
    const item = within(nav).getByText("Train ensemble").closest("button");
    expect(item?.getAttribute("aria-current")).toBe("true");
    expect(screen.getByRole("link", { name: /Models › Train/ }).getAttribute("href")).toBe("/models/starfull/train");
  });

  it("opens on the first step of ?stage= (the old Pixels › Inputs link)", async () => {
    show(<Steps />, "/runs/steps?stage=noise-fields");
    const nav = await screen.findByRole("navigation", { name: "FASRC steps by stage" });
    await waitFor(() => expect(within(nav).getByText("Sample VIS noise").closest("button")?.getAttribute("aria-current")).toBe("true"));
    expect(screen.getByRole("link", { name: /Synthetic › Noise/ }).getAttribute("href")).toBe("/synthetic/noise?how=1");
  });

  it("with no ?step= selects the first step of the list in stage order, not the registry's first", async () => {
    routes["GET /api/fasrc/steps/status"] = () => ({ body: { ok: true, ssh_connected: true, steps: [OTHER, TRAIN, QUERY] } });
    show(<Steps />, "/runs/steps");
    const nav = await screen.findByRole("navigation", { name: "FASRC steps by stage" });
    const first = within(nav).getAllByRole("button")[0];
    expect(first.textContent).toContain(QUERY.label);
    await waitFor(() => expect(first.getAttribute("aria-current")).toBe("true"));
  });

  it("warns when the linked step is not registered", async () => {
    show(<Steps />, "/runs/steps?step=gone_step");
    expect(await screen.findByText("No step “gone_step”")).toBeTruthy();
  });
});
