import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import { RouterProvider, createMemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useJobsStore, type Job } from "../api/jobs";
import { queryClient } from "../api/query";
import { useShortcutRegistry } from "../hooks/useShortcut";
import { useDisplay } from "../state/display";
import { useInspector } from "../state/inspector";
import { bindPrefsToDocument, usePrefs } from "../state/prefs";
import { resetConfirm } from "../ui";
import { openInspector, registerInspector } from "./inspector";
import { MANIFEST } from "./manifest";
import { usePageActions, usePaletteRegistry } from "./palette";
import { ROUTER_FUTURE, buildRoutes, type WorkspaceLoader } from "./routes";
import { Shell } from "./Shell";
import { useShellUi } from "./shellStore";
import { Workspace, defineTabs } from "./workspace";

/* ── a fake Flask behind fetch ─────────────────────────────────────────── */
type Reply = { status?: number; body: unknown };
let routes: Record<string, (init: RequestInit) => Reply>;
let calls: { url: string; method: string; body?: BodyInit | null }[];

function job(id: string, patch: Partial<Job> = {}): Job {
  return {
    job_id: id, label: `job ${id}`, kind: null, status: "running", started: Date.now() / 1000 - 30,
    finished: null, duration: 30, error: null, log: null, log_truncated: false, cancellable: true,
    cancel_requested: false, result: null, progress: { current: 2, total: 4, pct: 50, label: "stage" },
    ...patch,
  };
}

beforeEach(() => {
  calls = [];
  routes = {
    "GET /api/version": () => ({ body: {
      boot_commit: "aaaaaaa1", boot_short: "aaaaaaa", head_commit: "bbbbbbb2", head_short: "bbbbbbb",
      behind: false, dirty: false, started_at: new Date().toISOString(), pid: 1, dist: { built_at: null, index_hash: null },
    } }),
    "GET /api/fasrc/status": () => ({ body: { ssh_connected: false, last_error: "Permission denied (publickey)" } }),
    "GET /api/jobs?summary=1": () => ({ body: [job("j1"), job("j0", { status: "done", finished: Date.now() / 1000 })] }),
    "GET /api/fasrc/current-submission": () => ({ status: 503, body: { ok: false, error: "FASRC not connected", code: "fasrc_offline" } }),
    "POST /api/jobs/j1/cancel": () => ({ body: { ok: true } }),
  };
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = String(input);
    const method = init.method ?? "GET";
    calls.push({ url, method, body: init.body });
    const route = routes[`${method} ${url}`];
    const r = route ? route(init) : { status: 404, body: { ok: false, error: `no route ${url}` } };
    return new Response(JSON.stringify(r.body), { status: r.status ?? 200 });
  }));
  queryClient.clear();
  useJobsStore.getState().reset();
  useInspector.getState().reset();
  useShellUi.getState().closeAll();
  usePrefs.getState().reset();
  useDisplay.getState().reset();
});

afterEach(() => {
  queryClient.clear();
  resetConfirm();
  usePaletteRegistry.getState().reset();
  useShortcutRegistry.getState().reset();
});

/* fake workspaces (no legacy pages) — the sky atlas registers a page action;
   realism/noise, the home page and the "probe" inspector read the theme during
   render, like the legacy pages that read colour tokens (categorical(), C.muted). */
function SkyAtlas() {
  usePageActions([{ id: "fly", label: "Fly to NEXUS", group: "Sky", run: () => { (window as unknown as { flew: number }).flew = 1; } }]);
  return <p>sky:atlas</p>;
}
function ThemeProbe({ name }: { name: string }) {
  return <p data-testid={`probe-${name}`}>{`${name}:${document.documentElement.getAttribute("data-theme")}`}</p>;
}
const NoiseTab = () => <ThemeProbe name="noise" />;
const fakes: Record<string, WorkspaceLoader> = Object.fromEntries(MANIFEST.workspaces.map((ws) => {
  const tabs = defineTabs(ws.id, Object.fromEntries(ws.tabs.map((t) => [t, {
    load: async () => ({
      default: ws.id === "sky" && t === "atlas" ? SkyAtlas
        : ws.id === "realism" && t === "noise" ? NoiseTab
          : () => <p>{`${ws.id}:${t}`}</p>,
    }),
  }])));
  const root = ws.id === "home" ? <ThemeProbe name="home" /> : <p>{`${ws.id}:root`}</p>;
  return [ws.id, async () => ({ default: () => <Workspace id={ws.id} tabs={tabs}>{root}</Workspace> })];
}));

function mount(url = "/sky/atlas") {
  const router = createMemoryRouter(buildRoutes({ components: fakes, layout: Shell }), { initialEntries: [url], future: ROUTER_FUTURE });
  render(
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} future={{ v7_startTransition: true }} />
    </QueryClientProvider>,
  );
  return router;
}

const press = (key: string, init: KeyboardEventInit = {}) => act(() => {
  window.dispatchEvent(new KeyboardEvent("keydown", { key, code: init.code ?? `Key${key.toUpperCase()}`, bubbles: true, cancelable: true, ...init }));
});
const MOD: KeyboardEventInit = /Mac|iPod|iPhone|iPad/.test(navigator.platform) ? { metaKey: true } : { ctrlKey: true };

describe("Shell", () => {
  it("renders the rail with every workspace, the active one marked, and breadcrumbs", async () => {
    mount("/ensemble/starless/knee");
    await screen.findByText("ensemble:knee");
    const rail = screen.getByRole("navigation", { name: "Workspaces" });
    for (const ws of MANIFEST.workspaces) expect(within(rail).getByText(ws.label)).toBeTruthy();
    expect(within(rail).getByText("Ensemble").closest("a")?.getAttribute("aria-current")).toBe("page");
    const crumbs = screen.getByRole("navigation", { name: "Breadcrumbs" });
    expect(crumbs.textContent).toContain("Ensemble · starless");
    expect(crumbs.textContent).toContain("Knee PSNR");
    expect(document.title).toBe("Knee PSNR · Ensemble (starless) · EuclidPolish");
  });

  it("renders the router-linked workspace tabs", async () => {
    const router = mount("/data/records");
    await screen.findByText("data:records");
    const tabs = screen.getByRole("navigation", { name: "Data tabs" });
    fireEvent.click(within(tabs).getByText("PSFs"));
    await screen.findByText("data:psfs");
    expect(router.state.location.pathname).toBe("/data/psfs");
  });

  it("collapses the rail (persisted pref)", async () => {
    mount();
    await screen.findByText("sky:atlas");
    fireEvent.click(screen.getByRole("button", { name: "Collapse navigation" }));
    expect(usePrefs.getState().railCollapsed).toBe(true);
    expect(screen.getByRole("navigation", { name: "Workspaces" }).hasAttribute("data-collapsed")).toBe(true);
  });

  it("shows FASRC offline with the real error and never as a failure", async () => {
    mount();
    const badge = await screen.findByRole("link", { name: "FASRC offline" });
    expect(badge.getAttribute("href")).toBe("/settings/connections");
  });

  it("opens the palette with ⌘/Ctrl-K, lists page actions and navigates", async () => {
    const router = mount();
    await screen.findByText("sky:atlas");
    press("k", MOD);
    const input = await screen.findByPlaceholderText(/Search pages and actions/);
    expect(await screen.findByText("Fly to NEXUS")).toBeTruthy();
    fireEvent.change(input, { target: { value: "Ops › Git" } });
    fireEvent.click(await screen.findByText("Ops › Git"));
    await screen.findByText("ops:git");
    expect(router.state.location.pathname).toBe("/ops/git");
    expect(useShellUi.getState().palette).toBe(false);
  });

  it("runs a page action from the palette", async () => {
    mount();
    await screen.findByText("sky:atlas");
    act(() => useShellUi.getState().openOnly("palette"));
    fireEvent.click(await screen.findByText("Fly to NEXUS"));
    expect((window as unknown as { flew: number }).flew).toBe(1);
  });

  it("navigates with g-sequences and lists them in the ? sheet", async () => {
    const router = mount();
    await screen.findByText("sky:atlas");
    press("g"); press("o");
    await screen.findByText("ops:jobs");
    expect(router.state.location.pathname).toBe("/ops/jobs");
    press("?", { shiftKey: true, code: "Slash" });
    const sheet = await screen.findByRole("dialog", { name: "Keyboard shortcuts" });
    expect(within(sheet).getByText("Go to Ops")).toBeTruthy();
    expect(within(sheet).getByText("Command palette")).toBeTruthy();
  });

  it("opens the Display panel and edits the C7 store", async () => {
    mount();
    await screen.findByText("sky:atlas");
    fireEvent.click(screen.getByRole("button", { name: "Display settings" }));
    const dlg = await screen.findByRole("dialog", { name: "Display" });
    fireEvent.change(within(dlg).getByLabelText("Colour"), { target: { value: "lupton" } });
    expect(useDisplay.getState().color).toBe("lupton");
    fireEvent.click(within(dlg).getByRole("switch", { name: "Invert" }));
    expect(useDisplay.getState().invert).toBe(true);
    fireEvent.click(within(dlg).getByText("Reset to defaults"));
    expect(useDisplay.getState().color).toBe("VIS");
  });

  it("lists jobs in the tray, cancels one, and shows SLURM offline (not an error)", async () => {
    mount();
    await screen.findByText("sky:atlas");
    const trayBtn = await screen.findByRole("button", { name: "Jobs: 1 running" });
    fireEvent.click(trayBtn);
    const tray = await screen.findByLabelText("Jobs", { selector: ".jobtray" });
    expect(within(tray).getByText("job j1")).toBeTruthy();
    expect(within(tray).getByText("job j0")).toBeTruthy();
    expect(within(tray).getByText("FASRC offline")).toBeTruthy();
    expect(within(tray).queryByText(/Could not read SLURM/)).toBeNull();
    fireEvent.click(within(tray).getByRole("button", { name: "Cancel job j1" }));
    await waitFor(() => expect(calls.some((c) => c.method === "POST" && c.url === "/api/jobs/j1/cancel")).toBe(true));
  });

  it("opens a job in the inspector and mirrors it to ?inspect=", async () => {
    routes["GET /api/jobs/j1"] = () => ({ body: job("j1", { log: "line one\nline two" }) });
    const router = mount();
    await screen.findByText("sky:atlas");
    fireEvent.click(await screen.findByRole("button", { name: "Jobs: 1 running" }));
    fireEvent.click(await screen.findByText("job j1"));
    const panel = await screen.findByRole("complementary", { name: "Inspector" });
    expect(within(panel).getByRole("heading", { name: "job j1" })).toBeTruthy();
    expect(await within(panel).findByText(/line one/)).toBeTruthy();
    await waitFor(() => expect(new URLSearchParams(router.state.location.search).get("inspect")).toBe("job:local/j1"));
    fireEvent.click(within(panel).getByRole("button", { name: "Close inspector" }));
    await waitFor(() => expect(router.state.location.search).toBe(""));
  });

  it("shows the version banner when the server is behind HEAD", async () => {
    routes["GET /api/version"] = () => ({ body: {
      boot_commit: "a1", boot_short: "a1", head_commit: "b2", head_short: "b2", behind: true, dirty: false,
      started_at: null, pid: 1, dist: null,
    } });
    mount();
    expect(await screen.findByText("The server is running older code")).toBeTruthy();
    const rail = screen.getByRole("navigation", { name: "Workspaces" });
    expect(within(rail).getByText("server behind HEAD")).toBeTruthy();
  });

  it("re-renders the active tab, a tabless page and the inspector on a theme flip", async () => {
    const unbind = bindPrefsToDocument();
    const unregister = registerInspector("probe", () => <ThemeProbe name="insp" />);
    try {
      usePrefs.getState().setTheme("light");
      const router = mount("/realism/noise");
      expect((await screen.findByTestId("probe-noise")).textContent).toBe("noise:light");
      act(() => openInspector({ kind: "probe", id: "x" }));
      expect((await screen.findByTestId("probe-insp")).textContent).toBe("insp:light");
      act(() => usePrefs.getState().toggleTheme());
      await waitFor(() => expect(screen.getByTestId("probe-noise").textContent).toBe("noise:dark"));
      expect(screen.getByTestId("probe-insp").textContent).toBe("insp:dark");
      await act(() => router.navigate("/"));
      expect((await screen.findByTestId("probe-home")).textContent).toBe("home:dark");
      act(() => usePrefs.getState().toggleTheme());
      await waitFor(() => expect(screen.getByTestId("probe-home").textContent).toBe("home:light"));
    } finally {
      unregister();
      unbind();
      document.documentElement.removeAttribute("data-theme");
    }
  });

  it("offers typed-text suggestions without claiming nothing matches", async () => {
    mount();
    await screen.findByText("sky:atlas");
    act(() => useShellUi.getState().openOnly("palette"));
    const input = await screen.findByPlaceholderText(/Search pages and actions/);
    fireEvent.change(input, { target: { value: "150.1 2.2" } });
    expect(await screen.findByText(/Go to RA 150\.1°, Dec 2\.2°/)).toBeTruthy();
    expect(screen.queryByText(/Nothing matches/)).toBeNull();
    fireEvent.change(input, { target: { value: "7" } });
    expect(await screen.findByText("Nothing matches “7”.")).toBeTruthy();
  });

  it("has a global command that goes to the FASRC steps console", async () => {
    const router = mount();
    await screen.findByText("sky:atlas");
    act(() => useShellUi.getState().openOnly("palette"));
    const input = await screen.findByPlaceholderText(/Search pages and actions/);
    fireEvent.change(input, { target: { value: "run job" } });
    fireEvent.click(await screen.findByText("Run a FASRC step…"));
    await screen.findByText("ops:fasrc");
    expect(router.state.location.pathname).toBe("/ops/fasrc");
  });

  it("toasts a job that finishes while watched", async () => {
    mount();
    await screen.findByText("sky:atlas");
    await screen.findByRole("button", { name: "Jobs: 1 running" });
    act(() => useJobsStore.getState().upsert(job("j1", { status: "failed", error: "boom\ntrace" })));
    expect(await screen.findByText("boom")).toBeTruthy();
  });
});
