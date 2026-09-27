/* Settings › Connections / Appearance / About against a mocked Flask
 * (re-homed from workspaces/newPages.test.tsx and extended). */
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import type { ReactElement } from "react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useJobsStore } from "../../api/jobs";
import { queryClient } from "../../api/query";
import { useShellUi } from "../../app/shellStore";
import { useDisplay } from "../../state/display";
import { useInspector } from "../../state/inspector";
import { usePrefs } from "../../state/prefs";
import { UiProvider, resetConfirm } from "../../ui";
import About from "./tabs/About";
import Appearance from "./tabs/Appearance";
import Connections from "./tabs/Connections";

type Reply = { status?: number; body: unknown };
let routes: Record<string, (form: Record<string, string>) => Reply>;
let calls: { url: string; method: string; form: Record<string, string> }[];
const GIB = 1024 ** 3;

const VERSION = {
  boot_commit: "1111111aaaa", boot_short: "1111111", head_commit: "2222222bbbb", head_short: "2222222",
  behind: true, dirty: true, started_at: "2026-09-26T01:00:00Z", pid: 4242,
  dist: { built_at: "2026-09-26T00:00:00Z", index_hash: "deadbeef" },
};
const ROOTS = Array.from({ length: 18 }, (_, i) => ({
  id: `data/r${i}`, label: `r${i}`, path: `/repo/data/r${i}`, group: "data", bytes: (18 - i) * GIB, files: 10 * i, exists: true,
}));
const SYSTEM = {
  python: { version: "3.12.14", implementation: "CPython", executable: "/env/bin/python" },
  platform: { system: "Darwin", release: "25.5.0", machine: "arm64", platform: "macOS-26.5.1" },
  packages: { flask: "3.1.3", tensorflow: "2.19.1", photutils: null },
  node: "v26.0.0", pid: 1, data_dir: "/repo/data", noise_model: "euclid-q1-mer-noise-levels-dithered-bilinear-v5",
  disk: { path: "/repo/data", total_bytes: 460 * GIB, free_bytes: 19 * GIB, used_bytes: 441 * GIB, used_fraction: 0.9587,
    level: "warn", warn_below_bytes: 25 * GIB, bad_below_bytes: 10 * GIB, warn_used_fraction: 0.95 },
  roots: { items: ROOTS, computed_at: new Date().toISOString(), total_bytes: ROOTS.reduce((a, r) => a + r.bytes, 0),
    stale: false, refresh_job: null },
  experiments: { cache_budget_bytes: 4 * GIB, min_free_bytes: 5 * GIB, cache_bytes: GIB, outputs_bytes: 2 * GIB },
};

const formOf = (body: BodyInit | null | undefined): Record<string, string> => {
  const out: Record<string, string> = {};
  if (body instanceof FormData) body.forEach((v, k) => { out[k] = String(v); });
  return out;
};

beforeEach(() => {
  calls = [];
  routes = {
    "GET /api/version": () => ({ body: VERSION }),
    "GET /api/fasrc/status": () => ({ body: { ssh_connected: false, last_error: "ssh: connect to host login.rc: timed out" } }),
    "POST /api/fasrc/connect": () => ({ status: 400, body: { ok: false, error: "Permission denied (publickey)" } }),
    "GET /api/fasrc/config": () => ({ body: { ssh_user: "abelo", ssh_host: "login.rc.fas.harvard.edu", control_socket: "/tmp/s.sock",
      control_persist: "8h", repo_path: "/n/repo", conda_env_path: "/n/env", data_dir: "/n/data", ckpt_dir: "/n/ckpt",
      logs_subdir: "logs", tracking_remote_dir: "", local_ckpt_mirror: "", partition: "gpu", n_gpus: 1, n_cpus: 8,
      memory: "32G", time_limit: "12:00:00" } }),
    "POST /api/fasrc/config": (form) => ({ body: { ssh_user: form.ssh_user ?? "abelo", ssh_host: "login.rc.fas.harvard.edu",
      control_socket: "/tmp/s.sock", control_persist: "8h", repo_path: "/n/repo", conda_env_path: "/n/env", data_dir: "/n/data",
      ckpt_dir: "/n/ckpt", logs_subdir: "logs", tracking_remote_dir: "", local_ckpt_mirror: "", partition: "gpu", n_gpus: 1,
      n_cpus: 8, memory: "32G", time_limit: "12:00:00" } }),
    "GET /auth/status": () => ({ body: { authenticated: false, user: null, logged_in_at: null,
      used_by: [{ id: "galaxies", label: "Realism › Galaxies (Euclid galaxy query)", to: "/realism/galaxies" }] } }),
    "POST /auth/login": (form) => ({ body: { ok: true, authenticated: true, user: form.username, logged_in_at: "2026-09-26T10:00:00+00:00" } }),
    "GET /euclid-auth/status": () => ({ body: { present: false, connected: false } }),
    "GET /tng-auth/status": () => ({ body: { present: false, connected: false } }),
    "GET /api/system": () => ({ body: SYSTEM }),
    "POST /api/system/disk-usage/refresh": () => ({ body: { ok: true, job_id: "du1" } }),
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
  usePrefs.getState().reset();
  useDisplay.getState().reset();
  useShellUi.getState().closeAll();
  useInspector.getState().reset();
});
afterEach(() => { act(() => resetConfirm()); queryClient.clear(); });

const show = (el: ReactElement, url = "/settings") => render(
  <QueryClientProvider client={queryClient}>
    <MemoryRouter initialEntries={[url]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <UiProvider>{el}</UiProvider>
    </MemoryRouter>
  </QueryClientProvider>,
);
const posts = (url: string) => calls.filter((c) => c.method === "POST" && c.url === url);

describe("Settings › Connections", () => {
  it("shows the real last error and surfaces a failed connect", async () => {
    show(<Connections />);
    expect(await screen.findByText("ssh: connect to host login.rc: timed out")).toBeTruthy();
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Connect" })); });
    await waitFor(() => expect(posts("/api/fasrc/connect")).toHaveLength(1));
    expect(await screen.findByText("Permission denied (publickey)")).toBeTruthy();
    await waitFor(() => expect(calls.filter((c) => c.url === "/api/fasrc/status").length).toBeGreaterThan(1));
  });

  it("edits the FASRC SSH settings and saves only the changed fields", async () => {
    show(<Connections />, "/settings/connections?ssh=1");
    const user = await screen.findByLabelText("SSH user");
    fireEvent.change(user, { target: { value: "someone" } });
    expect(screen.getByText("1 unsaved")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Save settings" }));
    await waitFor(() => expect(posts("/api/fasrc/config")).toHaveLength(1));
    expect(posts("/api/fasrc/config")[0].form).toEqual({ ssh_user: "someone" });
  });

  it("logs in to the ONE Euclid archive session and lists where it is used", async () => {
    show(<Connections />);
    expect(await screen.findByRole("link", { name: "Realism › Galaxies (Euclid galaxy query)" })).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Username"), { target: { value: "abelo" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "s3cret" } });
    fireEvent.click(screen.getByRole("button", { name: "Log in" }));
    await waitFor(() => expect(posts("/auth/login")).toHaveLength(1));
    expect(posts("/auth/login")[0].form).toEqual({ username: "abelo", password: "s3cret" });
    expect((screen.getByLabelText("Password") as HTMLInputElement).value).toBe("");   // never kept
  });

  it("keeps the FASRC-side credentials read-only while FASRC is offline", async () => {
    show(<Connections />);
    const card = (await screen.findByRole("form", { name: "TNG API token · FASRC" })) as HTMLElement;
    expect((within(card).getByLabelText("Token") as HTMLInputElement).disabled).toBe(true);
    expect(screen.getAllByText("Connect to FASRC to read or write this file.")).toHaveLength(2);
  });
});

describe("Settings › Appearance", () => {
  it("edits theme, accent, density, rail and the image defaults", async () => {
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
    fireEvent.change(screen.getByLabelText("NaN colour"), { target: { value: "#112233" } });
    expect(useDisplay.getState().nanColor).toBe("#112233");
    fireEvent.change(screen.getByLabelText("Stretch"), { target: { value: "sqrt" } });
    expect(useDisplay.getState().stretch).toBe("sqrt");
    fireEvent.click(screen.getByText("Open the Display panel"));
    expect(useShellUi.getState().display).toBe(true);
    fireEvent.click(screen.getByText("Reset display settings"));
    expect(useDisplay.getState().stretch).toBe("asinh-abs");
    fireEvent.click(screen.getByText("Reset appearance"));
    expect(usePrefs.getState().theme).toBe("light");
  });
});

describe("Settings › About", () => {
  it("shows boot vs HEAD, the dirty tree and the dist build", async () => {
    show(<About />);
    expect(await screen.findByText("Restart the server")).toBeTruthy();
    // short hashes, the full one in the tooltip and on the copy button
    expect(screen.getByTitle("1111111aaaa").textContent).toBe("1111111");
    expect(screen.getByTitle("2222222bbbb").textContent).toBe("2222222");
    expect(screen.getByRole("button", { name: "Copy the boot commit hash" })).toBeTruthy();
    expect(screen.getByText("uncommitted changes")).toBeTruthy();
    expect(screen.getByText("deadbeef")).toBeTruthy();
  });

  it("shows the runtime, free space with its level and the cache budget", async () => {
    show(<About />);
    expect(await screen.findByText("CPython 3.12.14")).toBeTruthy();
    expect(screen.getByText("v26.0.0")).toBeTruthy();
    expect(screen.getByText("19 GiB free")).toBeTruthy();
    expect(screen.getByText("low")).toBeTruthy();
    expect(screen.getByText("Free space is low")).toBeTruthy();
    expect(screen.getByText("1 GiB of 4 GiB budget")).toBeTruthy();
    // the data roots' share of the used disk (171 GiB of 441 GiB used)
    expect(screen.getByText("171 GiB in the data roots · 270 GiB used elsewhere on this disk")).toBeTruthy();
  });

  it("lists every data root in a table and opens one in the inspector", async () => {
    show(<About />);
    const table = await screen.findByRole("grid", { name: "Data roots" });
    expect(within(table).getByText("r0")).toBeTruthy();
    fireEvent.click(within(table).getByText("r3"));
    expect(useInspector.getState().current).toEqual({ kind: "root", id: "data/r3" });
  });

  it("measures the disk usage when the last measurement is stale (once), and on demand", async () => {
    routes["GET /api/system"] = () => ({ body: { ...SYSTEM, roots: { ...SYSTEM.roots, stale: true } } });
    show(<About />);
    await waitFor(() => expect(posts("/api/system/disk-usage/refresh")).toHaveLength(1));
    await waitFor(() => expect(useJobsStore.getState().keyed["run:disk-usage"]).toBe("du1"));
    // started by the page itself: no "started" toast
    expect(screen.queryByText(/Measure disk usage: started/)).toBeNull();
  });

  it("toasts a measurement started on demand", async () => {
    show(<About />);
    fireEvent.click(await screen.findByRole("button", { name: "Measure now" }));
    expect(await screen.findByText("Measure disk usage: started")).toBeTruthy();
    expect(posts("/api/system/disk-usage/refresh")).toHaveLength(1);
  });
});
