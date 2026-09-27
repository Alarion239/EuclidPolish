import { act, render, screen, waitFor } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { invalidate, queryClient } from "../api/query";
import { UiProvider } from "../ui";
import { __resetLoadedBuild } from "./status";
import { UpdateNotice } from "./TopBar";

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
      <MemoryRouter><UiProvider><UpdateNotice fromBuild={fromBuild} pageEntry={pageEntry} /></UiProvider></MemoryRouter>
    </QueryClientProvider>,
  );
}

/** The page has rendered the version answer carrying `build`. */
async function loaded(build: string) {
  await waitFor(() => expect(queryClient.getQueryData<{ dist: { index_hash: string } }>(["GET", "/api/version"])?.dist.index_hash).toBe(build));
  await act(async () => { await new Promise((r) => setTimeout(r, 0)); });
}

describe("UpdateNotice (the console was rebuilt under this page)", () => {
  it("appears once the served build differs from the one this page loaded", async () => {
    mount(true);
    await loaded("build-1");
    expect(screen.queryByRole("button", { name: /Reload to update/ })).toBeNull();
    hash = "build-2";
    await act(async () => { await invalidate("/api/version"); });
    expect(await screen.findByRole("button", { name: /Reload to update/ })).toBeTruthy();
  });

  it("appears when the build was replaced before the page's first version answer (entry script differs)", async () => {
    hash = "build-2";
    entry = "/static/dist/assets/index-B.js";
    mount(true, "/static/dist/assets/index-A.js");
    await loaded("build-2");
    expect(await screen.findByRole("button", { name: /Reload to update/ })).toBeTruthy();
  });

  it("stays hidden while the served entry is this page's own", async () => {
    entry = "/static/dist/assets/index-A.js";
    mount(true, "/static/dist/assets/index-A.js");
    await loaded("build-1");
    hash = "build-1b";     // index.html bytes changed, same build (e.g. whitespace)
    await act(async () => { await invalidate("/api/version"); });
    await act(async () => { await new Promise((r) => setTimeout(r, 20)); });
    expect(screen.queryByRole("button", { name: /Reload to update/ })).toBeNull();
  });

  it("never shows on the dev server (the page is not from the build)", async () => {
    mount(false);
    await loaded("build-1");
    hash = "build-2";
    await act(async () => { await invalidate("/api/version"); });
    await act(async () => { await new Promise((r) => setTimeout(r, 20)); });
    expect(screen.queryByRole("button", { name: /Reload to update/ })).toBeNull();
  });
});
