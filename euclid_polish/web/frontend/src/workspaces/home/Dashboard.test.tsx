/* Home against a mocked Flask: the production numbers with their exact
 * sources, the STARFULL member count, the health checks and their actions,
 * the sky links. */
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import type { ReactElement } from "react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useJobsStore, type Job } from "../../api/jobs";
import { queryClient } from "../../api/query";
import { useInspector } from "../../state/inspector";
import { resetConfirm } from "../../ui";
import Dashboard from "./Dashboard";

type Reply = { status?: number; body: unknown };
let routes: Record<string, (form: Record<string, string>) => Reply>;
let calls: { url: string; method: string; form: Record<string, string> }[];

const GIB = 1024 ** 3;
const job = (id: string, patch: Partial<Job> = {}): Job => ({
  job_id: id, label: `job ${id}`, kind: "k", status: "done", started: 1_790_000_000, finished: 1_790_000_060,
  duration: 60, error: null, log: null, log_truncated: false, cancellable: false, cancel_requested: false,
  result: null, progress: { current: 0, total: 0, pct: 0, label: "" }, ...patch,
});

// 42 active members: 30 STARFULL + 12 starless (the carry-over bug showed 42).
const STATUS_MEMBERS = [
  ...Array.from({ length: 30 }, (_, i) => ({ name: `member_${169 + i}`, starless: false })),
  ...Array.from({ length: 12 }, (_, i) => ({ name: `member_${105 + i}`, starless: true })),
];

const KNEE = {
  available: true, stale: false, n_fields: 100, bands: ["VIS", "Y_E", "J_E", "H_E"],
  models: [
    { id: "member_0", kind: "member", label: "169·psnr", integrated: [54, 63, 60, 59] },
    { id: "member_27", kind: "member", label: "196·psnr", integrated: [54.7435, 64.1229, 60.6616, 60.3038] },
    { id: "ensemble_mean", kind: "mean", label: "ensemble mean", integrated: [53.9358, 62.9001, 60.143, 59.4462] },
    { id: "spatial_gate", kind: "combiner", label: "spatial gate", integrated: [55.7978, 64.9395, 62.0264, 61.1288] },
  ],
};

const ALERTS = {
  computed_at: new Date().toISOString(), ttl_s: 30, counts: { bad: 0, warn: 2, ok: 4, unknown: 1 },
  checks: [
    { id: "disk", label: "Disk space", state: "warn", title: "20.0 GiB free on the data disk (96 % used)", detail: "Experiments refuse…", to: "/settings/about", facts: { free_bytes: 20 * GIB } },
    { id: "real-sr", label: "Real SR products", state: "warn", title: "449 real SR products are stale", detail: "NEXUS tiles 445, Poster runs 4", to: "/sky/results" },
    { id: "evaluation", label: "Evaluation", state: "ok", title: "Evaluation current (30 members, test records)", to: "/ensemble/starfull/overview",
      action: { label: "Evaluate", method: "POST", url: "/ensemble/evaluate", params: { mode: "starfull" }, confirm: "Run the STARFULL evaluation?" } },
    { id: "records-noise", label: "Records noise model", state: "unknown", title: "Noise model of the local records unverified" },
  ],
  alerts: [] as unknown[],
};
ALERTS.alerts = ALERTS.checks.filter((c) => c.state === "warn");

const formOf = (body: BodyInit | null | undefined): Record<string, string> => {
  const out: Record<string, string> = {};
  if (body instanceof FormData) body.forEach((v, k) => { out[k] = String(v); });
  return out;
};

beforeEach(() => {
  calls = [];
  routes = {
    "GET /api/version": () => ({ body: {
      boot_commit: "1111111aaaa", boot_short: "1111111", head_commit: "2222222bbbb", head_short: "2222222",
      behind: true, dirty: true, started_at: "2026-09-26T01:00:00Z", pid: 4242, dist: null,
    } }),
    "GET /api/fasrc/status": () => ({ body: { ssh_connected: false, last_error: "ssh: connect to host login.rc: timed out" } }),
    "GET /api/jobs?summary=1": () => ({ body: [job("a", { status: "running", cancellable: true }), job("b")] }),
    "GET /api/fasrc/current-submission": () => ({ status: 503, body: { ok: false, error: "FASRC not connected", code: "fasrc_offline" } }),
    "GET /api/system/production": () => ({ body: {
      members: 30, starless_members: 12, stale: false, stale_reason: null, evaluated_at: "2026-09-25T23:32:26+00:00",
      eval_summary: {
        ensemble_psnr: 58.3753, mean_member_psnr: 57.22, ensemble_gain_db: 1.155,
        combiner_psnr: 99, // the RBF block: never the headline
        spatial_gate_combiner_psnr: 59.2354, spatial_gate_combiner_vs_mean_db: 0.8601,
        spatial_gate_combiner_vs_best_member_db: 0.2948,
      } } }),
    "GET /api/models": () => ({ body: {
      members: STATUS_MEMBERS.filter((m) => !m.starless).map((m) => `${m.name.slice(7)}·psnr`),
      production_kind: "spatial_gate",
      models: [{ spec: "production", available: true, reason: null, combiner_kind: "spatial_gate",
        label: "Production · spatial gate (convolutional, convex)",
        details: { mix_space: "linear", fitted_at: new Date(Date.now() - 3 * 3600_000).toISOString() } }],
    } }),
    "GET /ensemble/knee-psnr.json?mode=starfull": () => ({ body: KNEE }),
    "GET /api/system/alerts": () => ({ body: ALERTS }),
    "GET /api/system": () => ({ body: { disk: { free_bytes: 20 * GIB, total_bytes: 460 * GIB, used_fraction: 0.957, level: "warn" } } }),
    "GET /api/sky/layer/q1-fields": () => ({ body: { features: [
      { id: "EDF-N", ra: 269.73, dec: 66.02, radius_deg: 6, props: { name: "EDF-N" } },
      { id: "EDF-S", ra: 61.24, dec: -48.42, radius_deg: 6, props: { name: "EDF-S" } },
    ] } }),
    "GET /api/sky/layer/nexus-footprint": () => ({ body: { features: [
      { id: "nexus", polygon: [[268.2, 65.1], [268.7, 65.1], [268.7, 65.3], [268.2, 65.3]], props: {} },
    ] } }),
    "POST /ensemble/evaluate": () => ({ body: { job_id: "ev9" } }),
  };
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = String(input);
    const method = init.method ?? "GET";
    const form = formOf(init.body);
    calls.push({ url, method, form });
    const r = routes[`${method} ${url}`]?.(form) ?? { status: 404, body: { ok: false, error: `no route ${url}` } };
    return new Response(JSON.stringify(r.body), { status: r.status ?? 200 });
  }));
  queryClient.clear();
  useJobsStore.getState().reset();
  useInspector.getState().reset();
});
afterEach(() => { act(() => resetConfirm()); queryClient.clear(); });

const show = (el: ReactElement) => render(
  <QueryClientProvider client={queryClient}>
    <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>{el}</MemoryRouter>
  </QueryClientProvider>,
);

const tile = (label: string | RegExp) => screen.getByText(label).closest(".ui-kpi") as HTMLElement;

describe("Home · production numbers", () => {
  it("headlines the gate's knee-integrated PSNR vs the best member and the plain mean", async () => {
    show(<Dashboard />);
    expect(await screen.findByText("60.97")).toBeTruthy();
    const knee = tile("∫PSNR · production gate");
    expect(within(knee).getByText("+1.02 dB vs member 196")).toBeTruthy();
    expect(within(knee).getByText("+1.87 dB vs plain mean · 100 fields")).toBeTruthy();
    expect(knee.getAttribute("href")).toBe("/ensemble/starfull/knee");
  });

  it("shows the production gate's test PSNR from the spatial_gate_* keys, never the RBF", async () => {
    show(<Dashboard />);
    expect(await screen.findByText("59.24")).toBeTruthy();
    const test = tile("Test PSNR · production gate");
    expect(within(test).getByText("+0.29 dB vs best member")).toBeTruthy();
    expect(within(test).getByText("+0.86 dB vs plain mean 58.38")).toBeTruthy();
    expect(screen.queryByText("99.00")).toBeNull();
  });

  it("falls back to the ensemble status when the server predates the production endpoint", async () => {
    routes["GET /api/system/production"] = () => ({ status: 404, body: { error: "The requested URL was not found on the server." } });
    routes["GET /ensemble/status.json?mode=starfull"] = () => ({ body: {
      eval_summary: { ensemble_psnr: 58.3753, spatial_gate_combiner_psnr: 59.2354, spatial_gate_combiner_vs_best_member_db: 0.2948 },
      eval_summary_stale: false, members: STATUS_MEMBERS,
    } });
    show(<Dashboard />);
    expect(await screen.findByText("59.24", undefined, { timeout: 4000 })).toBeTruthy();
    const test = tile("Test PSNR · production gate");
    expect(within(test).getByText(/via ensemble status/)).toBeTruthy();
  });

  it("says the server needs a restart when neither endpoint answers", async () => {
    routes["GET /api/system/production"] = () => ({ status: 404, body: { error: "The requested URL was not found on the server." } });
    show(<Dashboard />);
    const test = await waitFor(() => {
      const t = tile("Test PSNR · production gate");
      if (!within(t).queryByText("not served by this server — restart it")) throw new Error("not yet");
      return t;
    }, { timeout: 4000 });
    expect(test).toBeTruthy();
  });

  it("never waits on the heavy ensemble status", async () => {
    show(<Dashboard />);
    expect(await screen.findByText("59.24")).toBeTruthy();
    expect(calls.some((c) => c.url.startsWith("/ensemble/status.json"))).toBe(false);
  });

  it("counts the active STARFULL members by regime label, not all 42", async () => {
    show(<Dashboard />);
    const members = await waitFor(() => {
      const t = tile("STARFULL members");
      if (!within(t).queryByText("30")) throw new Error("not loaded");
      return t;
    });
    expect(within(members).queryByText("42")).toBeNull();
    expect(within(members).getByText(/^linear mix · fitted .+ · 12 starless \(opt-in\)$/)).toBeTruthy();
    expect(within(members).getByText("spatial gate fits them")).toBeTruthy();
  });

  it("falls back to the labelled plain mean with its gain over the mean member", async () => {
    routes["GET /api/system/production"] = () => ({ body: { members: 30, starless_members: 12, stale: true,
      stale_reason: "The evaluation predates the current members", eval_summary: {
        ensemble_psnr: 44.123, mean_member_psnr: 43.913, ensemble_gain_db: -0.4,
      } } });
    routes["GET /ensemble/knee-psnr.json?mode=starfull"] = () => ({ body: { available: false, stale: false } });
    show(<Dashboard />);
    expect(await screen.findByText("44.12")).toBeTruthy();
    const test = tile("Test PSNR · plain mean");
    expect(within(test).getByText("+0.21 dB vs mean member")).toBeTruthy();
    expect(within(test).getByText("no production-gate score yet · summary stale")).toBeTruthy();
    expect(within(tile("∫PSNR · production gate")).getByText("not computed yet")).toBeTruthy();
  });

  it("shows connection, version, jobs and free disk", async () => {
    show(<Dashboard />);
    expect(await screen.findByText("1111111")).toBeTruthy();
    expect(screen.getByText("code changed")).toBeTruthy();
    expect(screen.getByText("Backend code changed — restart the server to load it")).toBeTruthy();
    expect(screen.queryByText(/behind|HEAD/)).toBeNull();
    expect(await screen.findByText("ssh: connect to host login.rc: timed out")).toBeTruthy();
    expect(await screen.findByText("job a")).toBeTruthy();
    const disk = await waitFor(() => {
      const t = tile("Free disk");
      if (!within(t).queryByText("20 GiB")) throw new Error("not loaded");
      return t;
    });
    expect(within(disk).getByText("96 % of 460 GiB used")).toBeTruthy();
    expect(within(disk).getByText("low")).toBeTruthy();
  });
});

describe("Home · recent jobs", () => {
  it("lists the live SLURM jobs under the local ones", async () => {
    routes["GET /api/fasrc/current-submission"] = () => ({ body: { ok: true, live: [
      { jobid: "4242", state: "RUNNING", label: "ensemble_train ×4", step_id: "ensemble_train" },
    ] } });
    show(<Dashboard />);
    const card = (await screen.findByText("Recent jobs")).closest(".ui-card") as HTMLElement;
    expect(await within(card).findByText("ensemble_train ×4")).toBeTruthy();
    expect(within(card).getByText("job a")).toBeTruthy();
    expect(within(card).getByText("SLURM")).toBeTruthy();
  });
});

describe("Home · health checks", () => {
  it("lists the alerts first, folds the passing checks and opens a check in the inspector", async () => {
    show(<Dashboard />);
    expect(await screen.findByText("449 real SR products are stale")).toBeTruthy();
    expect(screen.getByText("2 alerts")).toBeTruthy();
    expect(screen.queryByText("Evaluation current (30 members, test records)")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /1 OK · 1 unverified/ }));
    expect(screen.getByText("Evaluation current (30 members, test records)")).toBeTruthy();
    fireEvent.click(screen.getByText("20.0 GiB free on the data disk (96 % used)"));
    expect(useInspector.getState().current).toEqual({ kind: "check", id: "disk" });
  });

  it("runs a check's action only after confirm, as a tracked local job", async () => {
    show(<Dashboard />);
    await screen.findByText("449 real SR products are stale");
    fireEvent.click(screen.getByRole("button", { name: /1 OK/ }));
    const row = screen.getByText("Evaluation current (30 members, test records)").closest("li")!;
    fireEvent.click(within(row).getByRole("button", { name: "Evaluate" }));
    const dlg = await screen.findByRole("alertdialog", { name: "Evaluate?" });
    expect(calls.some((c) => c.method === "POST")).toBe(false);
    fireEvent.click(within(dlg).getByRole("button", { name: "Evaluate" }));
    await waitFor(() => expect(calls.some((c) => c.method === "POST" && c.url === "/ensemble/evaluate")).toBe(true));
    expect(calls.find((c) => c.url === "/ensemble/evaluate")!.form).toEqual({ mode: "starfull" });
    // shared with the palette / quick action: one evaluate at a time
    await waitFor(() => expect(useJobsStore.getState().keyed["run:evaluate"]).toBe("ev9"));
  });

  it("shows the server's error when the checks cannot be read", async () => {
    routes["GET /api/system/alerts"] = () => ({ status: 500, body: { error: "records dir unreadable" } });
    show(<Dashboard />);
    expect(await screen.findByText("records dir unreadable", {}, { timeout: 4000 })).toBeTruthy();
  });
});

describe("Home · quick actions and sky", () => {
  it("evaluates from the quick actions after confirm", async () => {
    show(<Dashboard />);
    fireEvent.click(await screen.findByRole("button", { name: "Evaluate" }));
    const dlg = await screen.findByRole("alertdialog", { name: "Evaluate the STARFULL ensemble?" });
    fireEvent.click(within(dlg).getByRole("button", { name: "Evaluate" }));
    await waitFor(() => expect(useJobsStore.getState().keyed["run:evaluate"]).toBe("ev9"));
  });

  it("links every Q1 field and NEXUS into the sky atlas", async () => {
    show(<Dashboard />);
    const edfn = await screen.findByRole("link", { name: /EDF-N on the sky atlas/ });
    expect(edfn.getAttribute("href")).toBe("/sky/atlas?ra=269.73&dec=66.02&fov=15");
    const nexus = await screen.findByRole("link", { name: "NEXUS F200W mosaic on the sky atlas" });
    expect(nexus.getAttribute("href")).toMatch(/^\/sky\/atlas\?ra=268\.45&dec=65\.2&fov=0\.6$/);
    // the legend repeats every target as a plain link with its coordinates
    const legend = screen.getByRole("list", { name: "Sky targets" });
    const rows = within(legend).getAllByRole("link");
    expect(rows.map((a) => a.textContent)).toEqual([
      "EDF-N269.7°, +66.0°", "EDF-S61.2°, -48.4°", "NEXUS268.4°, +65.2°",
    ]);
    expect(rows[2].getAttribute("href")).toBe(nexus.getAttribute("href"));
  });
});

describe("Home · log to tracking", () => {
  it("opens the notebook dialog pre-filled from the unlogged results and appends only on Append", async () => {
    const tracking = { id: "tracking", label: "Tracking log", state: "warn", title: "Results since the last tracking entry (2026-09-21)",
      to: "/ops/tracking", facts: { last_entry: "2026-09-21T14:42:29+00:00", unlogged: [
        { at: "2026-09-25T23:32:26+00:00", label: "evaluation" }, { at: "2026-09-25T22:30:13+00:00", label: "production gate fit" }] } };
    routes["GET /api/system/alerts"] = () => ({ body: { ...ALERTS, checks: [...ALERTS.checks, tracking], alerts: [...ALERTS.alerts, tracking] } });
    routes["POST /api/tracking/log"] = () => ({ body: { ok: true } });
    show(<Dashboard />);
    fireEvent.click(await screen.findByRole("button", { name: "Log to tracking (2)" }));
    const note = (await screen.findByRole("textbox", { name: "Markdown note" })) as HTMLTextAreaElement;
    expect(note.value).toContain("results since the last tracking entry (2026-09-21 14:42 UTC)");
    expect(note.value).toContain("- evaluation — 2026-09-25 23:32 UTC: test PSNR production gate 59.24 dB");
    expect(calls.some((c) => c.method === "POST")).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Append" }));
    await waitFor(() => expect(calls.find((c) => c.method === "POST" && c.url === "/api/tracking/log")?.form.mode).toBe("append"));
  });
});
