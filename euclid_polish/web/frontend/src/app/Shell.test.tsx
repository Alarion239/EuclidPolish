import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { RouterProvider, createMemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useJobsStore, type Job } from "../api/jobs";
import { invalidate, queryClient, serverHealth } from "../api/query";
import { useShortcutRegistry } from "../hooks/useShortcut";
import { useDisplay } from "../state/display";
import { useInspector } from "../state/inspector";
import { bindPrefsToDocument, usePrefs } from "../state/prefs";
import { resetConfirm } from "../ui";
import { registerDisplaySection } from "./displaySections";
import { openInspector, registerInspector } from "./inspector";
import { MANIFEST } from "./manifest";
import { usePageActions, usePaletteRegistry } from "./palette";
import { ROUTER_FUTURE, buildRoutes, type WorkspaceLoader } from "./routes";
import { NARROW_QUERY, Shell } from "./Shell";
import shellCss from "./shell.css?raw";
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
  serverHealth.reset();
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

function mount(url = "/sky/atlas", { strict = false }: { strict?: boolean } = {}) {
  const router = createMemoryRouter(buildRoutes({ components: fakes, layout: Shell }), { initialEntries: [url], future: ROUTER_FUTURE });
  const app = (
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} future={{ v7_startTransition: true }} />
    </QueryClientProvider>
  );
  render(strict ? <StrictMode>{app}</StrictMode> : app);
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
    expect(crumbs.textContent).toContain("Ensemble (starless)");
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

  it("closing the Display panel gives focus back to the button that opened it, and the stage makes room while open", async () => {
    mount();
    await screen.findByText("sky:atlas");
    const button = screen.getByRole("button", { name: "Display settings" });
    const shell = document.querySelector(".shell")!;
    expect(shell.hasAttribute("data-display")).toBe(false);
    for (const close of ["escape", "x", "done"] as const) {
      button.focus();
      fireEvent.click(button);
      const sheet = await screen.findByRole("dialog", { name: "Display" });
      expect(shell.hasAttribute("data-display")).toBe(true);        // the body reserves the sheet's width
      await waitFor(() => expect(sheet.contains(document.activeElement)).toBe(true));
      if (close === "escape") fireEvent.keyDown(document.activeElement!, { key: "Escape" });
      else if (close === "x") fireEvent.click(within(sheet).getByRole("button", { name: "Close the Display panel" }));
      else fireEvent.click(within(sheet).getByRole("button", { name: "Done" }));
      await waitFor(() => expect(screen.queryByRole("dialog", { name: "Display" })).toBeNull());
      await waitFor(() => expect(document.activeElement).toBe(button));
      expect(shell.hasAttribute("data-display")).toBe(false);
    }
  });

  it("Shift+D from the page: closing returns focus to the Display settings button when nothing else had it", async () => {
    mount();
    await screen.findByText("sky:atlas");
    (document.activeElement as HTMLElement | null)?.blur();
    press("D", { shiftKey: true, code: "KeyD" });
    const sheet = await screen.findByRole("dialog", { name: "Display" });
    fireEvent.click(within(sheet).getByRole("button", { name: "Done" }));
    await waitFor(() => expect(document.activeElement).toBe(screen.getByRole("button", { name: "Display settings" })));
  });

  it("keeps the images live under the Display panel: no overlay, outside clicks do not close it", async () => {
    const off = registerDisplaySection({ id: "sky", title: "Sky", order: 10, Component: () => <p>sky controls</p> });
    try {
      mount();
      await screen.findByText("sky:atlas");
      fireEvent.click(screen.getByRole("button", { name: "Display settings" }));
      const sheet = await screen.findByRole("dialog", { name: "Display" });
      expect(document.querySelector(".ui-dialog__overlay")).toBeNull();
      expect(document.getElementById("main")?.closest("[aria-hidden='true']")).toBeNull();
      // the page's own section comes first
      const titles = [...sheet.querySelectorAll(".ui-section__title")].map((t) => t.textContent);
      expect(titles).toEqual(["Sky", "Image", "Viewers"]);
      // working on the page (a click, a drag on an image) leaves it open
      fireEvent.pointerDown(screen.getByText("sky:atlas"));
      fireEvent.mouseDown(screen.getByText("sky:atlas"));
      fireEvent.click(screen.getByText("sky:atlas"));
      expect(screen.getByRole("dialog", { name: "Display" })).toBe(sheet);
      // one transfer group at a time, in its own unit
      expect(within(sheet).getByRole("slider", { name: "Default knee" }).getAttribute("aria-valuetext")).toBe("100 e⁻");
      fireEvent.click(within(sheet).getByRole("radio", { name: "JWST" }));
      expect(within(sheet).getByRole("slider", { name: "JWST knee" }).getAttribute("aria-valuetext")).toBe("100 MJy/sr");
      // a training knee is typed exactly (the log slider cannot land on 3)
      fireEvent.click(within(sheet).getByRole("radio", { name: "Euclid" }));
      const knee = within(sheet).getByRole("textbox", { name: "Euclid knee (e⁻)" });
      fireEvent.change(knee, { target: { value: "3" } });
      fireEvent.keyDown(knee, { key: "Enter" });
      expect(useDisplay.getState().groups.euclid.knee).toBe(3);
      fireEvent.change(knee, { target: { value: "0,1" } });
      fireEvent.blur(knee);
      expect(useDisplay.getState().groups.euclid.knee).toBe(0.1);
      fireEvent.change(knee, { target: { value: "-4" } });
      fireEvent.blur(knee);
      expect(useDisplay.getState().groups.euclid.knee).toBe(0.1);      // not a knee: ignored
      fireEvent.click(within(sheet).getByRole("button", { name: "Done" }));
      await waitFor(() => expect(screen.queryByRole("dialog", { name: "Display" })).toBeNull());
    } finally {
      off();
    }
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

  it("asks for a restart only when a loaded backend file changed, and lists the files", async () => {
    routes["GET /api/version"] = () => ({ body: {
      boot_commit: "a1", boot_short: "a1", head_commit: "b2", head_short: "b2", behind: true, dirty: false,
      changed_files: ["euclid_polish/web/routes/real.py", "euclid_polish/web/helpers/viewer_data.py"],
      changed_count: 3, started_at: "2026-09-26T20:58:25Z", pid: 185, dist: null,
    } });
    mount();
    const title = await screen.findByText("Backend code changed — restart the server to use it.");
    const banner = title.closest(".ui-callout") as HTMLElement;
    expect(within(banner).queryByText("euclid_polish/web/routes/real.py")).toBeNull();   // collapsed
    const details = within(banner).getByRole("button", { name: "Details" });
    expect(details.getAttribute("aria-expanded")).toBe("false");
    fireEvent.click(details);
    expect(details.getAttribute("aria-expanded")).toBe("true");
    expect(within(banner).getByText("euclid_polish/web/routes/real.py")).toBeTruthy();
    expect(within(banner).getByText("euclid_polish/web/helpers/viewer_data.py")).toBeTruthy();
    expect(within(banner).getByText("and 1 more.")).toBeTruthy();
    expect(within(banner).getByRole("link", { name: "Server details" }).getAttribute("href")).toBe("/settings/about");
    const rail = screen.getByRole("navigation", { name: "Workspaces" });
    expect(within(rail).getByText("Backend code changed — restart the server")).toBeTruthy();
    fireEvent.click(within(banner).getByRole("button", { name: "Dismiss" }));
    expect(screen.queryByText(/restart the server to use it/)).toBeNull();
  });

  it("puts the restart notice in the stage's one notice strip (dense, not pinned); no strip while nothing applies", async () => {
    routes["GET /api/version"] = () => ({ body: {
      boot_commit: "a1", boot_short: "a1", head_commit: "b2", head_short: "b2", behind: true, dirty: false,
      changed_files: [], changed_count: 1, started_at: null, pid: 185, dist: null,
    } });
    mount();
    const title = await screen.findByText("Backend code changed — restart the server to use it.");
    const notice = title.closest(".ui-callout") as HTMLElement;
    expect(notice.classList.contains("ui-callout--dense")).toBe(true);
    const strips = document.querySelectorAll(".shell__banner");
    expect(strips.length).toBe(1);
    expect(strips[0].contains(notice)).toBe(true);
    expect(strips[0].closest("main.stage")).toBeTruthy();   // in the scrolling stage, under the top bar
    expect(/\.shell__banner:empty\s*\{\s*display:\s*none/.test(shellCss)).toBe(true);
    expect(/\.shell__banner\s*\{[^}]*position:\s*(sticky|fixed)/.test(shellCss)).toBe(false);
  });

  it("shows no restart banner when HEAD moved but the loaded code did not change", async () => {
    // The old false positive: the commit of the code the server already runs.
    mount();
    await screen.findByText("sky:atlas");
    await waitFor(() => expect(calls.some((c) => c.url === "/api/version")).toBe(true));
    await act(async () => { await new Promise((r) => setTimeout(r, 20)); });
    expect(screen.queryByText(/Backend code changed/)).toBeNull();
    expect(screen.queryByText(/older code/)).toBeNull();
  });

  it("marks a dead server in the top bar, keeps the last data, and clears it on recovery", async () => {
    mount();
    await screen.findByText("sky:atlas");
    await screen.findByRole("link", { name: "FASRC offline" });
    await screen.findByRole("button", { name: "Jobs: 1 running" });
    expect(screen.queryByText("Server not responding")).toBeNull();

    // Flask dies: every request fails like a refused connection.
    const ok = globalThis.fetch;
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
    await act(async () => { await invalidate(); });
    const chip = await screen.findByRole("button", { name: /Server not responding — retrying/ }, { timeout: 5000 });
    expect(chip.textContent).toContain("Server not responding");
    expect(document.querySelector(".topbar")?.getAttribute("data-server")).toBe("down");
    // FASRC's state is unknown while the local server is gone: its badge makes room.
    expect(screen.queryByRole("link", { name: "FASRC offline" })).toBeNull();
    // The job tray keeps its last snapshot, marked as such.
    const tray = screen.getByRole("button", { name: "Jobs: 1 running when the server last answered" });
    fireEvent.click(tray);
    expect(await screen.findByText(/these are the job states it last sent/)).toBeTruthy();
    act(() => useShellUi.getState().closeAll());

    // It comes back: a retry (the chip) reaches it and everything unmarks.
    vi.stubGlobal("fetch", ok);
    fireEvent.click(chip);
    await waitFor(() => expect(screen.queryByText("Server not responding")).toBeNull(), { timeout: 5000 });
    expect(document.querySelector(".topbar")?.hasAttribute("data-server")).toBe(false);
    expect(await screen.findByRole("link", { name: "FASRC offline" })).toBeTruthy();
    expect(await screen.findByRole("button", { name: "Jobs: 1 running" })).toBeTruthy();
    expect(screen.getByText("The server is answering again")).toBeTruthy();   // live region
  }, 15_000);

  it("keeps the tab strip on one line: tabs that do not fit go to More, the active one stays", async () => {
    // jsdom has no layout: give every tab its text width and the strip 360 px.
    const rect = vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (this: HTMLElement) {
      const w = this.classList.contains("ui-tab") ? 8 * (this.textContent ?? "").length + 28 : 0;
      return { x: 0, y: 0, top: 0, left: 0, bottom: 0, right: w, width: w, height: 30, toJSON: () => ({}) } as DOMRect;
    });
    const client = vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockImplementation(function (this: HTMLElement) {
      return this.classList.contains("ws__tabs-wrap") ? 360 : 0;
    });
    try {
      const router = mount("/ensemble/starfull/disagreement");
      await screen.findByText("ensemble:disagreement");
      const strip = screen.getByRole("navigation", { name: "Ensemble tabs" });
      const shown = () => within(strip).getAllByRole("link").map((a) => a.textContent);
      // Overview 92 + Members 84 + Curves 76 = 252 of 360 − More; Disagreement (124) is active
      await waitFor(() => expect(shown()).toEqual(["Overview", "Members", "Disagreement"]));
      expect(within(strip).getByRole("link", { name: "Disagreement" }).getAttribute("aria-current")).toBe("page");
      const more = screen.getByRole("button", { name: /^More tabs: / });
      expect(strip.contains(more)).toBe(true);                   // inside the tabs landmark
      expect(more.getAttribute("aria-label")).toBe("More tabs: Curves, Knee PSNR, Diagnostics, Combiners, Train");
      fireEvent.pointerDown(more, { button: 0, ctrlKey: false, pointerType: "mouse" });
      const menu = await screen.findByRole("menu");
      expect(within(menu).getAllByRole("menuitem").map((m) => m.textContent))
        .toEqual(["Curves", "Knee PSNR", "Diagnostics", "Combiners", "Train"]);
      // items are router links: a middle- or ⌘-click opens a new browser tab
      const train = within(menu).getByRole("menuitem", { name: "Train" });
      expect(train.tagName).toBe("A");
      expect(train.getAttribute("href")).toBe("/ensemble/starfull/train");
      fireEvent.click(train);
      await screen.findByText("ensemble:train");
      expect(router.state.location.pathname).toBe("/ensemble/starfull/train");
      // the new active tab takes the reserved slot; the leading tabs never move
      await waitFor(() => expect(shown()).toEqual(["Overview", "Members", "Train"]));
      // no label is ever cut: every visible tab is a whole label
      for (const a of within(strip).getAllByRole("link")) expect(a.textContent?.length).toBeGreaterThan(2);
    } finally {
      rect.mockRestore();
      client.mockRestore();
    }
  });

  it("shows every tab and no More button when the strip is wide enough", async () => {
    mount("/data/records");
    await screen.findByText("data:records");
    const strip = screen.getByRole("navigation", { name: "Data tabs" });
    expect(within(strip).getAllByRole("link")).toHaveLength(MANIFEST.workspaces.find((w) => w.id === "data")!.tabs.length);
    expect(screen.queryByRole("button", { name: /^More tabs/ })).toBeNull();
  });

  it("moves focus into the inspector on open, Esc closes it from the page, and focus returns to the opener", async () => {
    routes["GET /api/jobs/j1"] = () => ({ body: job("j1") });
    mount();
    await screen.findByText("sky:atlas");
    const opener = await screen.findByRole("button", { name: "Jobs: 1 running" });
    opener.focus();
    act(() => openInspector({ kind: "job", id: "local/j1" }));
    const panel = await screen.findByRole("complementary", { name: "Inspector" });
    await waitFor(() => expect(document.activeElement).toBe(panel));   // opening moves focus in
    // switching targets while open leaves focus where it is
    opener.focus();
    act(() => openInspector({ kind: "job", id: "local/j0" }));
    await act(async () => { await new Promise((r) => setTimeout(r, 0)); });
    expect(document.activeElement).toBe(opener);
    // Esc with focus on the page (not in the inspector) closes it
    press("Escape", { code: "Escape" });
    await waitFor(() => expect(screen.queryByRole("complementary", { name: "Inspector" })).toBeNull());

    // closing from inside the inspector hands focus back to the opener
    act(() => openInspector({ kind: "job", id: "local/j1" }));
    const again = await screen.findByRole("complementary", { name: "Inspector" });
    await waitFor(() => expect(document.activeElement).toBe(again));
    fireEvent.click(within(again).getByRole("button", { name: "Close inspector" }));
    await waitFor(() => expect(screen.queryByRole("complementary", { name: "Inspector" })).toBeNull());
    await waitFor(() => expect(document.activeElement).toBe(opener));
    expect(panel.isConnected).toBe(false);
  });

  it("keeps focus in the opened inspector under StrictMode's double effects (the dev build)", async () => {
    routes["GET /api/jobs/j1"] = () => ({ body: job("j1") });
    mount("/sky/atlas", { strict: true });
    await screen.findByText("sky:atlas");
    const opener = await screen.findByRole("button", { name: "Jobs: 1 running" });
    opener.focus();
    act(() => openInspector({ kind: "job", id: "local/j1" }));
    const panel = await screen.findByRole("complementary", { name: "Inspector" });
    await act(async () => { await new Promise((r) => setTimeout(r, 10)); });
    expect(document.activeElement).toBe(panel);
    fireEvent.click(within(panel).getByRole("button", { name: "Close inspector" }));
    await waitFor(() => expect(document.activeElement).toBe(opener));
  });

  it("Esc closes the inspector from a field too, but an open dialog or a key the page used keeps it", async () => {
    routes["GET /api/jobs/j1"] = () => ({ body: job("j1") });
    mount();
    await screen.findByText("sky:atlas");
    const field = document.createElement("input");
    document.getElementById("main")!.appendChild(field);
    const escOn = (el: Element) => act(() => {
      el.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", code: "Escape", bubbles: true, cancelable: true }));
    });
    try {
      act(() => openInspector({ kind: "job", id: "local/j1" }));
      await screen.findByRole("complementary", { name: "Inspector" });
      // a key something nearer consumed (the viewer's focus mode, a zoomed chart): kept
      const eat = (e: KeyboardEvent) => e.preventDefault();
      field.addEventListener("keydown", eat);
      field.focus();
      escOn(field);
      expect(screen.getByRole("complementary", { name: "Inspector" })).toBeTruthy();
      field.removeEventListener("keydown", eat);
      // an open dialog (the palette) takes Esc first; the inspector stays
      act(() => useShellUi.getState().openOnly("palette"));
      const palette = await screen.findByRole("dialog");
      escOn(palette);
      expect(screen.getByRole("complementary", { name: "Inspector" })).toBeTruthy();
      act(() => useShellUi.getState().closeAll());
      await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
      // typing in a field: Esc closes it
      field.focus();
      escOn(field);
      await waitFor(() => expect(screen.queryByRole("complementary", { name: "Inspector" })).toBeNull());
    } finally { field.remove(); }
  });

  it("renders one visually hidden h1 per tab, in plain words", async () => {
    mount("/data/records");
    await screen.findByText("data:records");
    const h1s = screen.getAllByRole("heading", { level: 1 });
    expect(h1s).toHaveLength(1);
    expect(h1s[0].textContent).toBe("Records, Data");
    expect(h1s[0].classList.contains("sr-only")).toBe(true);
  });

  describe("below 900 px (the drawer and the inspector sheet)", () => {
    beforeEach(() => {
      const real = window.matchMedia.bind(window);
      vi.stubGlobal("matchMedia", (q: string) => (q === NARROW_QUERY
        ? { matches: true, media: q, onchange: null, addEventListener: () => {}, removeEventListener: () => {}, addListener: () => {}, removeListener: () => {}, dispatchEvent: () => false }
        : real(q)));
    });

    it("draws the scrims UNDER the drawer and the sheet (their content takes the pointer)", () => {
      // The kit's .ui-dialog__overlay is at z-dialog (60), above the drawer
      // (45): it covered the sheet's images and ate every click.
      const rule = /\.shell__scrim\s*\{[^}]*z-index:\s*([^;]+);/.exec(shellCss);
      expect(rule?.[1].trim()).toBe("calc(var(--z-drawer) - 1)");
      for (const cls of ["rail-drawer", "inspector-sheet"]) {
        expect(new RegExp(`\\.${cls}\\s*\\{[^}]*z-index:\\s*var\\(--z-drawer\\)`).test(shellCss)).toBe(true);
      }
    });

    it("a drawer link navigates (and closes the drawer)", async () => {
      const router = mount();
      await screen.findByText("sky:atlas");
      act(() => useShellUi.getState().setOpen("drawer", true));
      const drawer = await screen.findByRole("dialog", { name: "Navigation" });
      expect(document.querySelector(".shell__scrim")).toBeTruthy();
      expect(document.querySelector(".ui-dialog__overlay")).toBeNull();
      fireEvent.click(within(drawer).getByText("Data"));
      await waitFor(() => expect(router.state.location.pathname.startsWith("/data")).toBe(true));
      await waitFor(() => expect(useShellUi.getState().drawer).toBe(false));
    });

    it("a click inside the inspector sheet keeps it open", async () => {
      routes["GET /api/jobs/j1"] = () => ({ body: job("j1") });
      const router = mount();
      await screen.findByText("sky:atlas");
      act(() => openInspector({ kind: "job", id: "local/j1" }));
      const sheet = await screen.findByRole("dialog", { name: "Inspector" });
      expect(sheet.classList.contains("inspector-sheet")).toBe(true);
      const heading = await within(sheet).findByRole("heading", { name: "job j1" });
      fireEvent.pointerDown(heading);
      fireEvent.click(heading);
      await waitFor(() => expect(new URLSearchParams(router.state.location.search).get("inspect")).toBe("job:local/j1"));
      expect(screen.getByRole("dialog", { name: "Inspector" })).toBe(sheet);
    });

    it("opening the sheet focuses the panel itself, not its first button (no tooltip, one Esc closes)", async () => {
      routes["GET /api/jobs/j1"] = () => ({ body: job("j1") });
      mount();
      await screen.findByText("sky:atlas");
      act(() => openInspector({ kind: "job", id: "local/j1" }));
      const sheet = await screen.findByRole("dialog", { name: "Inspector" });
      await within(sheet).findByRole("heading", { name: "job j1" });
      await waitFor(() => expect(document.activeElement).toBe(sheet.querySelector("aside.inspector")));
      expect(document.querySelector("[role=tooltip]")).toBeNull();
      fireEvent.keyDown(document.activeElement!, { key: "Escape" });
      await waitFor(() => expect(screen.queryByRole("dialog", { name: "Inspector" })).toBeNull());
    });
  });

  it("names the inspector's resize handle", async () => {
    routes["GET /api/jobs/j1"] = () => ({ body: job("j1") });
    mount();
    await screen.findByText("sky:atlas");
    act(() => openInspector({ kind: "job", id: "local/j1" }));
    await screen.findByRole("complementary", { name: "Inspector" });
    expect(screen.getByRole("separator", { name: "Resize inspector" })).toBeTruthy();
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

  it("ranks what was typed: Enter opens the named page; the sky lookup is last", async () => {
    mount();
    await screen.findByText("sky:atlas");
    act(() => useShellUi.getState().openOnly("palette"));
    const input = await screen.findByPlaceholderText(/Search pages and actions/);
    const selected = () => document.querySelector("[cmdk-item][aria-selected='true']")?.textContent ?? "";
    const items = () => [...document.querySelectorAll("[cmdk-item]")].map((e) => e.textContent ?? "");
    for (const [q, page] of [["git", "Ops › Git"], ["noise", "Realism › Noise"], ["members", "Ensemble (starfull) › Members"]]) {
      fireEvent.change(input, { target: { value: q } });
      await waitFor(() => expect(selected()).toContain(page));
      expect(items().at(-1)).toContain(`Find “${q}” on the sky`);
      expect(items().some((t) => t.startsWith("Open member_"))).toBe(false);
    }
    // Only when nothing else matches is the sky lookup the Enter target.
    fireEvent.change(input, { target: { value: "Vega" } });
    await waitFor(() => expect(selected()).toContain("Find “Vega” on the sky"));
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

/* ── shell polish (W-Settings+Home) ──────────────────────────────────────── */

const ALERTS = {
  computed_at: "2026-09-26T10:00:00+00:00", ttl_s: 30, counts: { bad: 1, warn: 1, ok: 3, unknown: 1 },
  checks: [],
  alerts: [
    { id: "records-noise", label: "Records noise model", state: "bad", title: "Training records use an older noise model" },
    { id: "disk", label: "Disk space", state: "warn", title: "19.0 GiB free on the data disk" },
  ],
};

describe("Shell polish", () => {
  it("puts the full breadcrumb path in a tooltip so truncated crumbs stay readable", async () => {
    mount("/ensemble/starless/knee");
    await screen.findByText("ensemble:knee");
    const crumbs = screen.getByRole("navigation", { name: "Breadcrumbs" });
    expect(crumbs.getAttribute("title")).toBe("Ensemble (starless) › Knee PSNR");
    // every crumb is a truncating text element (CSS ellipsis)
    expect(crumbs.querySelectorAll(".crumbs__text")).toHaveLength(2);
    expect(within(crumbs).getByText("Knee PSNR").getAttribute("aria-current")).toBe("page");
  });

  it("badges Home in the rail with the number of health alerts, worst tone", async () => {
    routes["GET /api/system/alerts"] = () => ({ body: ALERTS });
    mount();
    await screen.findByText("sky:atlas");
    const rail = screen.getByRole("navigation", { name: "Workspaces" });
    const home = within(rail).getByText("Home").closest("a")!;
    const badge = await waitFor(() => {
      const b = home.querySelector(".rail__badge");
      if (!b) throw new Error("no badge yet");
      return b;
    });
    expect(badge.getAttribute("data-tone")).toBe("bad");
    expect(badge.textContent).toContain("2");
    expect(badge.getAttribute("title")).toContain("Training records use an older noise model");
  });

  it("lists the global 'Run a job' actions after the page's own actions", async () => {
    mount();
    await screen.findByText("sky:atlas");
    act(() => useShellUi.getState().openOnly("palette"));
    await screen.findByText("Fly to NEXUS");
    const headings = [...document.querySelectorAll("[cmdk-group-heading]")].map((h) => h.textContent);
    expect(headings.indexOf("Sky")).toBeGreaterThanOrEqual(0);
    expect(headings.indexOf("Run a job")).toBeGreaterThan(headings.indexOf("Sky"));
    expect(screen.getByText("Evaluate the STARFULL ensemble")).toBeTruthy();
  });

  it("runs a job from the palette only after confirm, then tracks it in the tray", async () => {
    routes["POST /ensemble/evaluate"] = () => ({ body: { job_id: "ev1" } });
    mount();
    await screen.findByText("sky:atlas");
    act(() => useShellUi.getState().openOnly("palette"));
    fireEvent.click(await screen.findByText("Evaluate the STARFULL ensemble"));
    const dlg = await screen.findByRole("alertdialog", { name: "Evaluate the STARFULL ensemble?" });
    expect(calls.some((c) => c.url === "/ensemble/evaluate")).toBe(false);
    fireEvent.click(within(dlg).getByRole("button", { name: "Evaluate" }));
    await waitFor(() => expect(calls.some((c) => c.method === "POST" && c.url === "/ensemble/evaluate")).toBe(true));
    const post = calls.find((c) => c.url === "/ensemble/evaluate")!;
    expect((post.body as FormData).get("mode")).toBe("starfull");
    await waitFor(() => expect(useJobsStore.getState().keyed["run:evaluate"]).toBe("ev1"));
  });

  it("toasts a SLURM job that leaves the live list, with its final state", async () => {
    routes["GET /api/fasrc/jobs/4242/status"] = () => ({ body: { ok: true, jobid: "4242", state: "COMPLETED" } });
    mount();
    await screen.findByText("sky:atlas");
    const key = ["jobs-feed", "slurm"];
    const live = { offline: false, stale: false, queue: null, current: null,
      jobs: [{ jobid: "4242", state: "RUNNING", label: "ensemble_train ×4" }] };
    act(() => { queryClient.setQueryData(key, live); });
    act(() => { queryClient.setQueryData(key, { ...live, stale: true, jobs: [] }); });   // a slow tick: ignored
    expect(calls.some((c) => c.url === "/api/fasrc/jobs/4242/status")).toBe(false);
    act(() => { queryClient.setQueryData(key, { ...live, jobs: [] }); });
    expect(await screen.findByText("SLURM 4242 · ensemble_train ×4")).toBeTruthy();
    expect(await screen.findByText("completed")).toBeTruthy();
  });
});
