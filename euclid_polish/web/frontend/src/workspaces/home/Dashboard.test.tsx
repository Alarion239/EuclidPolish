/* Home against a mocked Flask: the production verdict with its exact
 * sources and real holes, its caption, the Loop strip and its problem-only
 * warnings, the running line, the cached thumbnails, and that opening the
 * page only reads (no job launchers, no tiles). */
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import type { ReactElement } from "react";
import { MemoryRouter, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useJobsStore, type Job } from "../../api/jobs";
import { queryClient } from "../../api/query";
import { useInspector } from "../../state/inspector";
import { resetConfirm } from "../../ui";
import Dashboard from "./Dashboard";

type Reply = { status?: number; body: unknown };
let routes: Record<string, (form: Record<string, string>) => Reply>;
let calls: { url: string; method: string; form: Record<string, string> }[];

const DAY = 86_400_000;
const ago = (days: number) => new Date(Date.now() - days * DAY).toISOString();
const job = (id: string, patch: Partial<Job> = {}): Job => ({
  job_id: id, label: `job ${id}`, kind: "k", status: "done", started: 1_790_000_000, finished: 1_790_000_060,
  duration: 60, error: null, log: null, log_truncated: false, cancellable: false, cancel_requested: false,
  result: null, progress: { current: 0, total: 0, pct: 0, label: "" }, ...patch,
});

type CheckRow = { id: string; label: string; state: string; title: string; detail?: string | null; to?: string | null; facts?: Record<string, unknown> };
const CHECKS: CheckRow[] = [
  { id: "disk", label: "Disk space", state: "ok", title: "200.0 GiB free on the data disk (56 % used)", to: "/system/storage" },
  { id: "real-sr", label: "Real SR products", state: "warn", title: "449 real SR products are stale", detail: "NEXUS tiles 445, Poster runs 4",
    to: "/sky/targets", facts: { current: 0, stale: 449, missing: 101 } },
  { id: "production", label: "Production model", state: "ok", title: "Production is Universal 257",
    facts: { model: "universal:member_257", label: "Universal 257" } },
  { id: "records-noise", label: "Records noise model", state: "unknown", title: "Noise model of the local records unverified" },
  { id: "tracking", label: "Tracking log", state: "ok", title: "Tracking log up to date", facts: { last_entry: ago(0.1), unlogged: [] } },
];
const alertsOf = (checks: CheckRow[]) => ({
  computed_at: new Date().toISOString(), ttl_s: 30, counts: { bad: 0, warn: 0, ok: 0, unknown: 0 },
  checks, alerts: checks.filter((c) => c.state === "warn" || c.state === "bad"),
});
const patchCheck = (id: string, patch: Partial<CheckRow>) => CHECKS.map((c) => (c.id === id ? { ...c, ...patch } : c));

const OVERVIEW = {
  gate: { ready: true, blockers: [] }, records: { generated_at: ago(2) },
  items: [
    { id: "galaxy-model", label: "Galaxies", state: "ok", title: "Galaxy model active", group: "generation", to: "/synthetic/galaxies",
      facts: { version: 15 }, records: { state: "current" } },
    { id: "star-prior", label: "Stars", state: "ok", title: "Stellar prior active", group: "generation", to: "/synthetic/stars", records: { state: "current" } },
    { id: "noise-model", label: "Noise", state: "ok", title: "Noise model v5", group: "generation", to: "/synthetic/noise",
      facts: { noise_model: "euclid-q1-mer-noise-levels-dithered-bilinear-v5" }, records: { state: "unknown" } },
    { id: "galaxy-plots", label: "Galaxy plots", state: "ok", title: "Galaxy plots current", group: "diagnostic", facts: { present: true, stale: false } },
  ],
};

const EXPERIMENTS = { experiments: [
  { id: "e-new", created: ago(1), label: "seed vs pruning on real tiles", status: "done",
    summary: { production: { n_tiles: 9, per_band: { VIS: { hole_pct: 12.5 }, Y_E: { hole_pct: 15.2 }, J_E: { hole_pct: 20.19 }, H_E: { hole_pct: 17.5 } } } } },
] };

const PLATES = { root: "output/nexus_comparisons", runs: [{ tag: "prod-0926", updated: ago(1), files: [], renders: [
  { band: "VIS", model: "production", model_label: "Production · Universal 257", model_fingerprint: "u257", created: ago(1), sheet: "nexus_tiles_VIS.png", tiles: [] },
] }] };

/** GET /ensemble/universal.json: production (257) with its own score. */
const UNIVERSAL = {
  production_model: "member_257", production: { integrated: [57.01, 67.01, 64.04, 63.22], mean: 62.82, stale: false, model: "member_257" },
  models: [{ name: "member_257", score: { scored_at: ago(0.125) } }, { name: "member_258", score: { scored_at: ago(2) } }],
};

const formOf = (body: BodyInit | null | undefined): Record<string, string> => {
  const out: Record<string, string> = {};
  if (body instanceof FormData) body.forEach((v, k) => { out[k] = String(v); });
  return out;
};

const stage = (id: string, label: string, state: string, reason: string, to: string, detail: string | null = null) =>
  ({ id, label, state, reason, detail, to });
/** The healthy verdicts of GET /api/system/loop (only Real SR stale). */
const LOOP_OK = () => ({ computed_at: new Date().toISOString(), ttl_s: 60, counts: {}, errors: {}, stages: [
  stage("priors", "Priors", "current", "galaxies v15 · stars · noise v5", "/synthetic/status"),
  stage("records", "Records", "current", "built 2 d ago, after the priors", "/synthetic/records"),
  stage("models", "Models", "current", "2 universal models", "/models/universal"),
  stage("production", "Production", "current", "Universal 257", "/models/universal"),
  stage("real-sr", "Real SR", "stale", "449 stale", "/sky/targets?state=stale"),
  stage("figures", "Figures", "current", "made with this fit", "/figures/plates"),
] });

beforeEach(() => {
  calls = [];
  routes = {
    "GET /api/system/loop": () => ({ body: LOOP_OK() }),
    "GET /api/fasrc/status": () => ({ body: { ssh_connected: true, last_error: null } }),
    "GET /api/jobs?summary=1": () => ({ body: [job("a", { status: "running", cancellable: true }), job("b")] }),
    "GET /api/fasrc/current-submission": () => ({ status: 503, body: { ok: false, error: "FASRC not connected", code: "fasrc_offline" } }),
    "GET /api/models": () => ({ body: {
      production_kind: "universal",
      production: { kind: "universal", spec: "universal:member_257", name: "member_257", label: "Universal 257", available: true, reason: null },
      models: [{ spec: "production", kind: "production", available: true, reason: null, fingerprint: "u257",
        label: "Production · Universal 257", details: { universal: true, model_spec: "universal:member_257", member: "member_257" } }],
    } }),
    "GET /ensemble/universal.json?mode=starfull": () => ({ body: UNIVERSAL }),
    "GET /api/system/alerts": () => ({ body: alertsOf(CHECKS) }),
    "GET /api/experiments": () => ({ body: EXPERIMENTS }),
    "GET /api/experiments/e-new": () => ({ body: { id: "e-new", fingerprints: { production: "u257" } } }),
    "GET /api/realism/overview": () => ({ body: OVERVIEW }),
    "GET /ensemble/training-jobs.json": () => ({ body: { jobs: [] } }),
    "GET /api/figures/nexus-plates": () => ({ body: PLATES }),
    "GET /api/figures/real-sr": () => ({ body: { total: 1, items: [{ ref: "tile/ra1_dec2", source: "tile", id: "ra1_dec2",
      source_label: "Cached 25.6″ tiles", state: "current", created: null, thumb: "/api/figures/real-sr/tile/ra1_dec2.jpg" }] } }),
    "GET /viewer/results": () => ({ body: { results: [
      { id: "vr-aaaaaaaaaaaaaaaaaaaaaaaa", label: "Poster galaxy core", regime: "real", created_utc: ago(0.2),
        source: { collection: "real", params: { source: "poster" }, object: { id: "g1", ref: "poster/g1" } } },
      { id: "vr-cccccccccccccccccccccccc", label: "synthetic lens 7", regime: "synthetic", created_utc: ago(0.1), source: { collection: "sky" } },
    ] } }),
    "GET /poster/result/status": () => ({ body: { ok: true, available: false, png: null, fits: null } }),
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

function LocationProbe() {
  const loc = useLocation();
  return <output data-testid="loc">{loc.pathname + loc.search}</output>;
}

const show = (el: ReactElement) => render(
  <QueryClientProvider client={queryClient}>
    <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>{el}<LocationProbe /></MemoryRouter>
  </QueryClientProvider>,
);
const location = () => new URL(screen.getByTestId("loc").textContent ?? "", "http://x");

const summary = () => document.querySelector(".ui-summary") as HTMLElement | null;
const summaryText = async (match: RegExp = /dB/) => waitFor(() => {
  const t = summary()?.textContent ?? "";
  if (!match.test(t)) throw new Error(`summary not yet: ${t}`);
  return t;
}, { timeout: 4000 });
const chip = (label: string) => screen.getByRole("link", { name: new RegExp(`^${label}: `) });
const chipAt = async (label: string, state: string) => waitFor(() => {
  const el = chip(label);
  expect(el.getAttribute("data-state")).toBe(state);
  return el;
}, { timeout: 4000 });

describe("Home · the production verdict", () => {
  it("states the production model and its own integrated PSNR, then its real holes, in one sentence", async () => {
    show(<Dashboard />);
    const text = await summaryText(/real holes/);
    expect(text).toMatch(/^Production is Universal 257: 62\.82\u00a0dB integrated PSNR; real holes 20% in J, its worst band \(Sky › Compare, .+\)\.$/);
    expect(text).not.toMatch(/best member|plain mean|gate/);
    expect(within(summary()!).getByRole("link", { name: "integrated PSNR" }).getAttribute("href")).toBe("/models/universal");
    const cap = await waitFor(() => {
      const el = document.querySelector(".home__caption") as HTMLElement | null;
      if (!el?.textContent?.includes("scored")) throw new Error("no caption yet");
      return el;
    });
    expect(cap.textContent).toBe("one universal model · scored 3 h ago");
    expect(within(cap).getByRole("link", { name: "one universal model" }).getAttribute("href")).toBe("/models/universal");
    // only the catalogue and the universal payload: no ensemble reads
    expect(calls.some((c) => /\/ensemble\/(overview|knee-psnr|members|status)|\/api\/system\/production/.test(c.url))).toBe(false);
  });

  it("names the production model when the newest real run scored an earlier one, and says when it is not scored yet", async () => {
    routes["GET /api/experiments/e-new"] = () => ({ body: { id: "e-new", fingerprints: { production: "u238" } } });
    const view = show(<Dashboard />);
    expect(await summaryText(/benchmark/)).toMatch(
      /^Production is Universal 257: 62\.82\u00a0dB integrated PSNR; no real benchmark for Universal 257 \(the last run, .+, scored an earlier one\)\.$/);
    view.unmount();
    queryClient.clear();
    routes["GET /ensemble/universal.json?mode=starfull"] = () => ({ body: { ...UNIVERSAL, production: null } });
    show(<Dashboard />);
    expect(await summaryText(/not scored.*benchmark/)).toMatch(/^Production is Universal 257, not scored on the test fields yet; no real benchmark for Universal 257/);
  });
});

describe("Home · the Loop strip", () => {
  it("shows the six stages in loop order, each with a state dot and one reason, linking to its tab", async () => {
    show(<Dashboard />);
    const loop = await screen.findByRole("navigation", { name: "The loop" });
    await chipAt("Figures", "current");
    await chipAt("Priors", "current");
    const chips = within(loop).getAllByRole("link");
    expect(chips.map((c) => c.getAttribute("aria-label")?.split(":")[0])).toEqual(["Priors", "Records", "Models", "Production", "Real SR", "Figures"]);
    expect(chips.map((c) => c.querySelector(".home-loop__dot")?.getAttribute("data-state")))
      .toEqual(["current", "current", "current", "current", "stale", "current"]);
    const real = chip("Real SR");
    expect(real.textContent).toBe("Real SR449 stale");
    expect(real.getAttribute("href")).toBe("/sky/targets?state=stale");
    expect(chip("Production").getAttribute("href")).toBe("/models/universal");
    expect(chip("Production").textContent).toBe("ProductionUniversal 257");
    expect(chip("Models").textContent).toBe("Models2 universal models");
    // no tiles, no badge, no job launchers
    expect(document.querySelector(".ui-kpi, .ui-stat")).toBeNull();
    expect(screen.queryByText(/\d alerts?$/)).toBeNull();
    expect(screen.queryByRole("button", { name: "Evaluate" })).toBeNull();
    expect(screen.queryByText("Quick actions")).toBeNull();
  });

  it("reads 'checking' while the staleness service answers, never 'current'", async () => {
    let release: () => void = () => {};
    const gate = new Promise<void>((r) => { release = r; });
    vi.mocked(fetch).mockImplementation(async (input: RequestInfo | URL, init: RequestInit = {}) => {
      const url = String(input);
      if (url === "/api/system/loop") await gate;
      const r = routes[`${init.method ?? "GET"} ${url}`]?.({}) ?? { status: 404, body: {} };
      return new Response(JSON.stringify(r.body), { status: r.status ?? 200 });
    });
    show(<Dashboard />);
    const priors = await chipAt("Priors", "loading");
    expect(priors.textContent).toBe("Priorschecking");
    act(() => release());
    await chipAt("Priors", "current");
  });

  it("reads 'not checked' with a note when the service fails, and never rebuilds the rules in the browser", async () => {
    routes["GET /api/system/loop"] = () => ({ status: 404, body: { ok: false, error: "not found" } });
    show(<Dashboard />);
    const priors = await chipAt("Priors", "unknown");
    expect(priors.textContent).toBe("Priorsnot checked");
    expect(await screen.findByText(/Could not read the Loop \(GET \/api\/system\/loop\)/)).toBeTruthy();
    expect(calls.some((c) => c.url === "/api/realism/overview" || c.url === "/ensemble/training-jobs.json")).toBe(false);
  });

  it("adds no warning while the disk, FASRC and the notebook are fine", async () => {
    show(<Dashboard />);
    await chipAt("Figures", "current");
    expect(screen.queryByRole("list", { name: "Needs attention" })).toBeNull();
  });

  it("warns about a low disk and a lost FASRC session, each linking to its System tab", async () => {
    routes["GET /api/system/alerts"] = () => ({ body: alertsOf(patchCheck("disk", { state: "warn", title: "20.0 GiB free on the data disk (96 % used)" })) });
    routes["GET /api/fasrc/status"] = () => ({ body: { ssh_connected: false, last_error: "ssh: connect to host login.rc: timed out" } });
    show(<Dashboard />);
    const list = await screen.findByRole("list", { name: "Needs attention" });
    await waitFor(() => expect(within(list).getAllByRole("link").length).toBe(2));
    const [disk, fasrc] = within(list).getAllByRole("link");
    expect([disk.textContent, disk.getAttribute("href")]).toEqual(["Disk: 20.0 GiB free", "/system/storage"]);
    expect([fasrc.textContent, fasrc.getAttribute("href")]).toEqual(["FASRC not connected", "/system/connections"]);
  });

  it("opens Notebook › Log prefilled from the unlogged results, appending nothing", async () => {
    routes["GET /api/system/alerts"] = () => ({ body: alertsOf(patchCheck("tracking", { state: "warn", title: "Results since the last tracking entry (2026-09-21)",
      facts: { last_entry: "2026-09-21T14:42:29+00:00", unlogged: [
        { at: "2026-09-25T23:32:26+00:00", label: "universal scores" }, { at: "2026-09-25T22:30:13+00:00", label: "experiment" }] } })) });
    show(<Dashboard />);
    const chip = await screen.findByRole("button", { name: "No notebook entry since 09-21 · 2 results" });
    expect(within(chip.closest("li")!).getByRole("link", { name: "Notebook › Log" }).getAttribute("href")).toBe("/notebook/log");
    // Wait for the headline numbers the entry quotes.
    await summaryText();
    fireEvent.click(chip);
    await waitFor(() => expect(location().pathname).toBe("/notebook/log"));
    const entry = location().searchParams.get("entry") ?? "";
    expect(location().searchParams.get("from")).toBe("Home");
    expect(entry).toContain("results since the last tracking entry (2026-09-21 14:42 UTC)");
    expect(entry).toContain("- universal scores — 2026-09-25 23:32 UTC: Universal 257 is production, ∫PSNR 62.82 dB");
    expect(entry).toContain("- experiment — 2026-09-25 22:30 UTC: see Sky › Compare");
    expect(calls.some((c) => c.method === "POST")).toBe(false);
  });
});

describe("Home · the staleness service", () => {
  const LOOP = { computed_at: new Date().toISOString(), ttl_s: 60, counts: {}, errors: {}, stages: [
    stage("priors", "Priors", "current", "galaxies v15 · stars · noise v5", "/synthetic/status"),
    stage("records", "Records", "current", "built 2 d ago, after the priors", "/synthetic/records"),
    stage("models", "Models", "stale", "7 new on FASRC", "/models/universal", "members 259–265 finished on FASRC and are not pulled"),
    stage("production", "Production", "current", "Universal 257", "/models/universal"),
    stage("real-sr", "Real SR", "stale", "450 stale", "/sky/targets?state=stale"),
    stage("figures", "Figures", "stale", "NEXUS plates use a legacy SR", "/figures/plates?plate=nexus", "Made with minibatched convex all-asinh RBF."),
  ] };

  it("shows the server's verdicts and does not rebuild them in the browser", async () => {
    routes["GET /api/system/loop"] = () => ({ body: LOOP });
    show(<Dashboard />);
    const models = await chipAt("Models", "stale");
    expect(models.textContent).toBe("Models7 new on FASRC");
    expect(chip("Figures").textContent).toBe("FiguresNEXUS plates use a legacy SR");
    expect(chip("Figures").getAttribute("href")).toBe("/figures/plates?plate=nexus");
    expect(calls.some((c) => c.url === "/api/realism/overview" || c.url === "/ensemble/training-jobs.json")).toBe(false);
  });

  it("recomputes it on Refresh Home (a read, never a job)", async () => {
    routes["GET /api/system/loop"] = () => ({ body: LOOP });
    routes["GET /api/system/loop?fresh=1"] = () => ({ body: LOOP });
    routes["GET /api/system/alerts?fresh=1"] = () => ({ body: alertsOf(CHECKS) });
    show(<Dashboard />);
    await chipAt("Models", "stale");
    fireEvent.click(screen.getByRole("button", { name: "Refresh Home" }));
    await waitFor(() => expect(calls.some((c) => c.url === "/api/system/loop?fresh=1")).toBe(true));
    expect(calls.filter((c) => c.method !== "GET")).toEqual([]);
  });
});

describe("Home · running now", () => {
  it("says what runs, on FASRC and locally, and links to Runs › Live", async () => {
    routes["GET /api/fasrc/current-submission"] = () => ({ body: { ok: true, live: [
      { jobid: "4242", state: "RUNNING", label: "Train ensemble members", step_id: "ensemble_train", progress_step: 10500, progress_total: 70000,
        gpu_util_mean: 78, params_json: JSON.stringify({ member_names: "member_199,member_200,member_201,member_202" }) },
    ] } });
    show(<Dashboard />);
    const line = await waitFor(() => {
      const el = document.querySelector(".home__running") as HTMLElement | null;
      if (!el || !/FASRC/.test(el.textContent ?? "")) throw new Error("not yet");
      return el;
    });
    expect(line.textContent).toBe("Running now: members 199–202 on FASRC · 15% · GPU 78%; job a on this laptop. Runs › Live");
    expect(within(line).getByRole("link", { name: "Runs › Live" }).getAttribute("href")).toBe("/runs/live");
  });

  it("says nothing when idle", async () => {
    routes["GET /api/jobs?summary=1"] = () => ({ body: [job("b")] });
    show(<Dashboard />);
    await chipAt("Models", "current");
    expect(document.querySelector(".home__running")).toBeNull();
  });
});

describe("Home · the latest cached thumbnails", () => {
  it("shows cached real SR tiles and crops (to their tile card) and the newest plates (to the plate), from cached files only", async () => {
    show(<Dashboard />);
    const strip = await screen.findByRole("region", { name: "Latest real SR and plates" });
    await waitFor(() => expect(within(strip).getAllByRole("link")).toHaveLength(3));
    const cards = within(strip).getAllByRole("link");
    expect(cards.map((a) => [a.querySelector(".home-thumbs__label")?.textContent, a.getAttribute("href")])).toEqual([
      ["ra1_dec2", "/sky/targets?inspect=realtile%3Atile%2Fra1_dec2"],
      ["Poster galaxy core", "/sky/targets?inspect=realtile%3Aposter%2Fg1"],
      ["NEXUS comparison", "/figures/plates?plate=nexus&run=prod-0926"],
    ]);
    expect(cards[0].querySelector("img")?.getAttribute("src")).toBe("/api/figures/real-sr/tile/ra1_dec2.jpg");
    expect(cards[1].querySelector("img")?.getAttribute("src")).toBe("/viewer/results/vr-aaaaaaaaaaaaaaaaaaaaaaaa/panel.png?size=240");
    expect(cards[2].querySelector("img")?.getAttribute("src")).toBe("/api/figures/nexus-plates/prod-0926/nexus_tiles_VIS.png?thumb=320");
    // opening Home starts nothing
    await waitFor(() => expect(calls.some((c) => c.url === "/poster/result/status")).toBe(true));
    expect(calls.filter((c) => c.method !== "GET")).toEqual([]);
  });
});
