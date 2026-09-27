import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { invalidate, queryClient } from "../api/query";
import { UiProvider } from "../ui";
import { __resetLoadedBuild } from "./status";
import { BUILD_BANNER_DISMISSED_KEY, BuildBanner } from "./TopBar";

let hash: string;
let entry: string | null;
const version = () => ({
  boot_commit: "a", boot_short: "a", head_commit: "a", head_short: "a", behind: false, dirty: false,
  changed_files: [], changed_count: 0, started_at: null, pid: 1, dist: { built_at: null, index_hash: hash, entry },
});

beforeEach(() => {
  hash = "build-1";
  entry = null;
  queryClient.clear();
  __resetLoadedBuild();
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(version()), { status: 200 })));
});
afterEach(() => {
  queryClient.clear();
  vi.unstubAllGlobals();
});

function mount(fromBuild: boolean, pageEntry: string | null = null) {
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter><UiProvider><BuildBanner fromBuild={fromBuild} pageEntry={pageEntry} /></UiProvider></MemoryRouter>
    </QueryClientProvider>,
  );
}

/** The page has rendered the version answer carrying `build`. */
async function loaded(build: string) {
  await waitFor(() => expect(queryClient.getQueryData<{ dist: { index_hash: string } }>(["GET", "/api/version"])?.dist.index_hash).toBe(build));
  await act(async () => { await new Promise((r) => setTimeout(r, 0)); });
}

const reloadButton = () => screen.queryByRole("button", { name: "Reload" });

describe("BuildBanner (a newer console build is served than this page runs)", () => {
  it("appears once the served build differs from the one this page loaded", async () => {
    mount(true);
    await loaded("build-1");
    expect(screen.queryByRole("button", { name: /^Reload$/ })).toBeNull();
    hash = "build-2";
    await act(async () => { await invalidate("/api/version"); });
    expect(await screen.findByRole("button", { name: /^Reload$/ })).toBeTruthy();
  });

  it("appears when the build was replaced before the page's first version answer (entry script differs)", async () => {
    hash = "build-2";
    entry = "/static/dist/assets/index-B.js";
    mount(true, "/static/dist/assets/index-A.js");
    await loaded("build-2");
    expect(await screen.findByRole("button", { name: /^Reload$/ })).toBeTruthy();
  });

  it("stays hidden while the served entry is this page's own", async () => {
    entry = "/static/dist/assets/index-A.js";
    mount(true, "/static/dist/assets/index-A.js");
    await loaded("build-1");
    hash = "build-1b";     // index.html bytes changed, same build (e.g. whitespace)
    await act(async () => { await invalidate("/api/version"); });
    await act(async () => { await new Promise((r) => setTimeout(r, 20)); });
    expect(screen.queryByRole("button", { name: /^Reload$/ })).toBeNull();
  });

  it("never shows on the dev server (the page is not from the build)", async () => {
    mount(false);
    await loaded("build-1");
    hash = "build-2";
    await act(async () => { await invalidate("/api/version"); });
    await act(async () => { await new Promise((r) => setTimeout(r, 20)); });
    expect(screen.queryByRole("button", { name: /^Reload$/ })).toBeNull();
  });
});

describe("BuildBanner: calm, never automatic, dismissible per build", () => {
  it("says it plainly with a Reload action, and reloads only when asked", async () => {
    const reload = vi.fn();
    const real = window.location;
    Object.defineProperty(window, "location", { configurable: true, value: { ...real, reload } });
    try {
      entry = "/static/dist/assets/index-B.js";
      mount(true, "/static/dist/assets/index-A.js");
      await loaded("build-1");
      const title = await screen.findByText("A newer console build is available.");
      const banner = title.closest(".ui-callout") as HTMLElement;
      expect(banner.getAttribute("role")).toBe("status");        // polite, not an alert
      expect(banner.classList.contains("shell__notice")).toBe(true);        // a notice of the stage strip (scrolls away)
      expect(banner.classList.contains("ui-callout--dense")).toBe(true);    // one compact line, not a 52 px box
      await act(async () => { await new Promise((r) => setTimeout(r, 30)); });
      expect(reload).not.toHaveBeenCalled();                       // never by itself
      fireEvent.click(within(banner).getByRole("button", { name: "Reload" }));
      expect(reload).toHaveBeenCalledTimes(1);
    } finally {
      Object.defineProperty(window, "location", { configurable: true, value: real });
    }
  });

  it("stays dismissed for that build and comes back for the next one", async () => {
    entry = "/static/dist/assets/index-B.js";
    mount(true, "/static/dist/assets/index-A.js");
    await loaded("build-1");
    fireEvent.click(await screen.findByRole("button", { name: "Dismiss" }));
    expect(screen.queryByText("A newer console build is available.")).toBeNull();
    expect(localStorage.getItem(BUILD_BANNER_DISMISSED_KEY)).toBe("/static/dist/assets/index-B.js");
    // another rebuild: a new entry script → shown again
    entry = "/static/dist/assets/index-C.js";
    hash = "build-3";
    await act(async () => { await invalidate("/api/version"); });
    expect(await screen.findByText("A newer console build is available.")).toBeTruthy();
  });

  it("remembers the dismissal across a remount (this browser)", async () => {
    localStorage.setItem(BUILD_BANNER_DISMISSED_KEY, "/static/dist/assets/index-B.js");
    entry = "/static/dist/assets/index-B.js";
    mount(true, "/static/dist/assets/index-A.js");
    await loaded("build-1");
    await act(async () => { await new Promise((r) => setTimeout(r, 20)); });
    expect(reloadButton()).toBeNull();
  });
});
