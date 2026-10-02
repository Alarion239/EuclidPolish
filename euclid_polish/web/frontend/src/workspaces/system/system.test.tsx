/* System › Connections / Appearance / Storage / Code / Lineage against a
 * mocked Flask (re-homed from Settings and Ops, and extended for the
 * console regrouping). */
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
import Appearance from "./tabs/Appearance";
import Code from "./tabs/Code";
import Connections from "./tabs/Connections";
import Lineage from "./tabs/Lineage";
import Storage from "./tabs/Storage";

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
      used_by: [{ id: "galaxies", label: "Synthetic › Galaxies (Euclid galaxy query)", to: "/synthetic/galaxies" }] } }),
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

const show = (el: ReactElement, url = "/system") => render(
  <QueryClientProvider client={queryClient}>
    <MemoryRouter initialEntries={[url]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <UiProvider>{el}</UiProvider>
    </MemoryRouter>
  </QueryClientProvider>,
);
const posts = (url: string) => calls.filter((c) => c.method === "POST" && c.url === url);
const answer = async (title: RegExp | string, button: string) => {
  const dlg = await screen.findByRole("alertdialog", { name: title });
  fireEvent.click(within(dlg).getByRole("button", { name: button }));
  await waitFor(() => expect(screen.queryByRole("alertdialog", { name: title })).toBeNull());
  return dlg;
};

describe("System › Connections", () => {
  it("shows the real last error and surfaces a failed connect", async () => {
    show(<Connections />);
    expect(await screen.findByText("ssh: connect to host login.rc: timed out")).toBeTruthy();
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Connect" })); });
    await waitFor(() => expect(posts("/api/fasrc/connect")).toHaveLength(1));
    expect(await screen.findByText("Permission denied (publickey)")).toBeTruthy();
    await waitFor(() => expect(calls.filter((c) => c.url === "/api/fasrc/status").length).toBeGreaterThan(1));
  });

  it("edits the FASRC SSH settings and saves only the changed fields", async () => {
    show(<Connections />, "/system/connections?ssh=1");
    const user = await screen.findByLabelText("SSH user");
    fireEvent.change(user, { target: { value: "someone" } });
    expect(screen.getByText("1 unsaved")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Save settings" }));
    await waitFor(() => expect(posts("/api/fasrc/config")).toHaveLength(1));
    expect(posts("/api/fasrc/config")[0].form).toEqual({ ssh_user: "someone" });
  });

  it("logs in to the ONE Euclid archive session and lists where it is used", async () => {
    show(<Connections />);
    expect(await screen.findByRole("link", { name: "Synthetic › Galaxies (Euclid galaxy query)" })).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Username"), { target: { value: "abelo" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "s3cret" } });
    fireEvent.click(screen.getByRole("button", { name: "Log in" }));
    await waitFor(() => expect(posts("/auth/login")).toHaveLength(1));
    expect(posts("/auth/login")[0].form).toEqual({ username: "abelo", password: "s3cret" });
    expect((screen.getByLabelText("Password") as HTMLInputElement).value).toBe("");   // never kept
  });

  it("says since when FASRC is connected, keeps the socket collapsed and asks before disconnecting", async () => {
    routes["GET /api/fasrc/status"] = () => ({ body: { ssh_connected: true, connected_at: new Date(Date.now() - 16 * 60_000).toISOString(),
      socket: "/tmp/cm.sock" } });
    routes["POST /api/fasrc/disconnect"] = () => ({ body: { ok: true } });
    show(<Connections />);
    expect(await screen.findByText(/Connected 16 min ago/)).toBeTruthy();
    expect(screen.getByText("Control socket").closest("details")?.hasAttribute("open")).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Disconnect" }));
    await answer("Disconnect from FASRC?", "Cancel");
    expect(posts("/api/fasrc/disconnect")).toHaveLength(0);
  });

  it("keeps the FASRC-side credentials read-only while FASRC is offline", async () => {
    show(<Connections />);
    const card = (await screen.findByRole("form", { name: "TNG API token · FASRC" })) as HTMLElement;
    expect((within(card).getByLabelText("Token") as HTMLInputElement).disabled).toBe(true);
    expect(screen.getAllByText("Connect to FASRC to read or write this file.")).toHaveLength(2);
  });
});

describe("System › Appearance", () => {
  it("edits theme, accent, density and rail, and says in one line how images are shown", async () => {
    show(<Appearance />);
    fireEvent.click(screen.getByRole("radio", { name: "Dark" }));
    expect(usePrefs.getState().theme).toBe("dark");
    fireEvent.click(screen.getByRole("button", { name: "Accent teal" }));
    expect(usePrefs.getState().accent).toBe("teal");
    fireEvent.click(screen.getByRole("radio", { name: "Compact" }));
    expect(usePrefs.getState().density).toBe("compact");
    fireEvent.click(screen.getByRole("switch", { name: /Collapse the navigation rail/ }));
    expect(usePrefs.getState().railCollapsed).toBe(true);
    expect(screen.queryByLabelText("Stretch")).toBeNull();
    expect(screen.getByText("VIS, absolute asinh, knee 100 e⁻")).toBeTruthy();
    act(() => useDisplay.getState().set({ stretch: "sqrt" }));
    expect(await screen.findByText("VIS, square root, knee 100 e⁻")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Open Display panel" }));
    expect(useShellUi.getState().display).toBe(true);
    fireEvent.click(screen.getByText("Reset appearance"));
    expect(usePrefs.getState().theme).toBe("light");
  });
});

describe("System › Storage", () => {
  it("tones the disk by the one threshold, and says how much of the used space is ours", async () => {
    show(<Storage />);
    expect(await screen.findByText("19 GiB free")).toBeTruthy();
    expect(screen.getByText("low")).toBeTruthy();
    expect(screen.getByText(/171 GiB of 441 GiB used is ours/)).toBeTruthy();
    expect(screen.getByText(/Warns below 25 GiB free or at 95% used \(here and on Home\)/)).toBeTruthy();
    expect(screen.getByText("Experiments: member-SR cache 1 GiB of its 4 GiB budget, outputs 2 GiB; 5 GiB kept free.")).toBeTruthy();
  });

  it("shows no badge when the disk is fine", async () => {
    routes["GET /api/system"] = () => ({ body: { ...SYSTEM, disk: { ...SYSTEM.disk, free_bytes: 200 * GIB, used_fraction: 0.5, level: "ok" } } });
    show(<Storage />);
    expect(await screen.findByText("200 GiB free")).toBeTruthy();
    expect(screen.queryByText("healthy")).toBeNull();
    expect(screen.queryByText("low")).toBeNull();
  });

  it("lists every data root in a table and opens one in the inspector", async () => {
    show(<Storage />);
    const table = await screen.findByRole("grid", { name: "Data roots" });
    expect(within(table).getByText("r0")).toBeTruthy();
    fireEvent.click(within(table).getByText("r3"));
    expect(useInspector.getState().current).toEqual({ kind: "root", id: "data/r3" });
  });

  it("never measures on its own: a stale measurement is only flagged; Measure now starts the job", async () => {
    routes["GET /api/system"] = () => ({ body: { ...SYSTEM, roots: { ...SYSTEM.roots, stale: true } } });
    show(<Storage />);
    expect(await screen.findByText("stale")).toBeTruthy();
    await new Promise((r) => setTimeout(r, 50));
    expect(posts("/api/system/disk-usage/refresh")).toHaveLength(0);
    fireEvent.click(screen.getByRole("button", { name: "Measure now" }));
    expect(await screen.findByText("Measure disk usage: started")).toBeTruthy();
    expect(posts("/api/system/disk-usage/refresh")).toHaveLength(1);
  });

  it("lists the FASRC sizes as a sortable table, and asks before the maintenance actions", async () => {
    routes["GET /api/fasrc/status"] = () => ({ body: { ssh_connected: true } });
    routes["GET /api/fasrc/data-listing"] = () => ({ body: { ok: true, data_dir: "/n/netscratch/data",
      du: [["12G", "/n/netscratch/data/images"], ["1.5T", "/n/netscratch/data/euclid_stars"], ["512K", "/n/netscratch/data/misc"]] } });
    routes["GET /api/fasrc/files"] = () => ({ body: { ok: true, dir: null, crumbs: [], entries: [], truncated: false } });
    routes["POST /api/evaluation/sync"] = () => ({ body: { ok: true, n_ok: 3, n: 4 } });
    routes["POST /api/evaluation/rerender"] = () => ({ body: { ok: true, removed: 7 } });
    show(<Storage />, "/system/storage?side=fasrc");
    const du = await screen.findByRole("grid", { name: "Remote folder sizes" });
    await waitFor(() => expect(within(du).getAllByRole("row")[1].textContent).toContain("euclid_stars"));   // largest first
    fireEvent.click(screen.getByRole("button", { name: "Sync from FASRC…" }));
    await answer("Sync the evaluation results from FASRC?", "Cancel");
    expect(posts("/api/evaluation/sync")).toHaveLength(0);
    fireEvent.click(screen.getByRole("button", { name: "Sync from FASRC…" }));
    await answer("Sync the evaluation results from FASRC?", "Sync and delete local-only");
    await waitFor(() => expect(posts("/api/evaluation/sync")[0]?.form).toEqual({ confirm: "1" }));
    fireEvent.click(screen.getByRole("button", { name: "Drop cached PNGs…" }));
    await answer("Drop the cached eye/solar PNGs?", "Drop cached PNGs");
    await waitFor(() => expect(posts("/api/evaluation/rerender")).toHaveLength(1));
  });
});

describe("System › Code", () => {
  const FILES = [
    { xy: " M", path: "a.py", orig: null, staged: false, unstaged: true, untracked: false, size: 10, guard: null },
    { xy: "??", path: "big.fits", orig: null, staged: false, unstaged: true, untracked: true, size: 12e6, guard: "file > 10 MB" },
  ];
  const status = (ahead: number) => () => ({ body: {
    status: { in_repo: true, root: "/repo", branch: "main", upstream: "origin/main", ahead, behind: 0, files: FILES,
      last: { hash: "2222222", subject: "x", relative: "now" } }, log: [] } });
  beforeEach(() => {
    routes["GET /api/git/status"] = status(2);
    routes["GET /api/git/diff"] = () => ({ body: { diff: "", staged: false, path: null } });
    routes["GET /api/git/diff?path=a.py"] = () => ({ body: { diff: "", staged: false, path: "a.py" } });
    routes["GET /api/git/log?skip=0&limit=100"] = () => ({ body: { commits: [], total: 0, skip: 0, limit: 100, has_more: false } });
  });
  async function typeMessage() {
    show(<Code />);
    fireEvent.change(await screen.findByLabelText("Commit message"), { target: { value: "msg" } });
  }

  it("answers in one sentence whether laptop, server and FASRC are on the same commit", async () => {
    routes["GET /api/fasrc/status"] = () => ({ body: { ssh_connected: true } });
    routes["GET /api/fasrc/git-status"] = () => ({ body: { ok: true, repo: "/n/repo", branch: "main", ahead: 0, behind: 0,
      head: "2222222ffff", local_head: "2222222bbbb", relation: { relation: "same", ahead: 0, behind: 0 }, dirty: false, last: {} } });
    show(<Code />);
    expect(await screen.findByText(/Laptop and FASRC are on 2222222; the server started at 1111111\./)).toBeTruthy();
    expect(screen.getByText(/This laptop has uncommitted changes\./)).toBeTruthy();
    expect(screen.queryByText(/This laptop has 2 uncommitted/)).toBeNull();
    // the server facts sit in Details, with the restart notice
    const details = screen.getByText("Server and runtime").closest("details") as HTMLElement;
    expect(details.hasAttribute("open")).toBe(false);
    expect(within(details).getByText("Backend code changed — restart the server to load it")).toBeTruthy();
  });

  it("shows the changes in words, and makes Push primary only when ahead", async () => {
    show(<Code />);
    const grid = await screen.findByRole("grid", { name: "Changed files" });
    expect(within(grid).getByText("modified")).toBeTruthy();
    expect(within(grid).getByText("untracked")).toBeTruthy();
    expect(screen.getByText("1 modified · 1 untracked")).toBeTruthy();
    const push = screen.getByRole("button", { name: "Push 2" });
    expect(push.className).toContain("primary");
    routes["GET /api/git/status"] = status(0);
    queryClient.clear();
  });

  it("keeps Push quiet when nothing is ahead", async () => {
    routes["GET /api/git/status"] = status(0);
    show(<Code />);
    const push = await screen.findByRole("button", { name: "Push" });
    expect(push.className).not.toContain("primary");
  });

  it("confirms the changed files, then commits all=1", async () => {
    routes["POST /git/commit"] = () => ({ body: { ok: true, stdout: "[main abc] msg", committed: ["a.py", "big.fits"] } });
    await typeMessage();
    fireEvent.click(screen.getByRole("button", { name: "Commit all" }));
    const dlg = await answer("Commit all 2 changed files?", "Commit all");
    expect(dlg.textContent).toContain("big.fits  (file > 10 MB)");
    await waitFor(() => expect(posts("/git/commit")).toHaveLength(1));
    expect(posts("/git/commit")[0].form).toEqual({ message: "msg", all: "1" });
  });

  it("shows refused files on 409 and retries with force=1 when confirmed", async () => {
    routes["POST /git/commit"] = (form) => (form.force === "1"
      ? { body: { ok: true, stdout: "forced", committed: ["a.py", "big.fits"] } }
      : { status: 409, body: { ok: false, code: "refused_files", error: "refused", refused: [{ path: "big.fits", size: 12e6, reason: "file > 10 MB" }] } });
    await typeMessage();
    fireEvent.click(screen.getByRole("button", { name: "Commit all" }));
    await answer("Commit all 2 changed files?", "Commit all");
    const dlg = await answer("Commit 1 large or binary file anyway?", "Force commit");
    expect(dlg.textContent).toContain("big.fits · 12.0 MB — file > 10 MB");
    await waitFor(() => expect(posts("/git/commit")).toHaveLength(2));
    expect(posts("/git/commit")[1].form).toEqual({ message: "msg", all: "1", force: "1" });
  });

  it("commits exactly the selected paths and stages per file", async () => {
    routes["POST /git/commit"] = () => ({ body: { ok: true, committed: ["a.py"] } });
    routes["POST /git/stage"] = () => ({ body: { ok: true, staged: ["a.py"] } });
    await typeMessage();
    fireEvent.click(screen.getByRole("button", { name: "Stage a.py" }));
    await waitFor(() => expect(posts("/git/stage")[0]?.form).toEqual({ paths: "a.py" }));
    const table = screen.getByRole("grid", { name: "Changed files" });
    fireEvent.click(within(table).getAllByRole("checkbox")[1]);
    fireEvent.click(screen.getByRole("button", { name: "Commit 1 selected" }));
    await answer("Commit 1 selected file?", "Commit");
    await waitFor(() => expect(posts("/git/commit")[0]?.form).toEqual({ message: "msg", paths: "a.py" }));
  });

  it("confirms before pushing", async () => {
    routes["POST /git/push"] = () => ({ body: { ok: true, stdout: "pushed" } });
    show(<Code />);
    fireEvent.click(await screen.findByRole("button", { name: "Push 2" }));
    const dlg = await answer("Push main to origin/main?", "Push");
    expect(dlg.textContent).toContain("2 local commits will be published.");
    await waitFor(() => expect(posts("/git/push")).toHaveLength(1));
  });
});

describe("System › Lineage", () => {
  const row = (id: string, patch: Record<string, unknown> = {}) => ({ id, kind: "srcutoutartifact", category: "artifact",
    source: "sidecar", file: `data/x/${id}.json`, created_at: "2026-02-01T00:00:00+00:00", status: null, path: null, format: "fits",
    label: `x/${id}.fits`, git: "abc", dirty: false, config_type: null, seed: null, produced_by: null, parents: [], inputs: [],
    outputs: [], ra: null, dec: null, member: null, verdict: "unknown", models: [], n_upstream: 0, n_downstream: 0, ...patch });
  /* The staleness service Home's Loop reads: the verdicts come from here. */
  const LOOP = { computed_at: "2026-09-27T23:00:00+00:00", ttl_s: 60, errors: {}, counts: { current: 1, stale: 2, blocked: 0, unknown: 0 },
    stages: [
      { id: "records", label: "Records", state: "stale", reason: "predate the stellar prior", to: "/synthetic/records" },
      { id: "members", label: "Members", state: "current", reason: "30 active", to: "/models/members" },
      { id: "real-sr", label: "Real SR", state: "stale", reason: "449 stale", to: "/sky/targets" },
    ] };
  const summary = (kinds: Record<string, number>, total: number, verdicts = { current: 0, stale: 0, unknown: total }) => ({
    body: { ok: true, total, counts: { kinds, verdicts }, roots: [], current_models: [], truncated: false, duplicates: 0,
      built_at: 1790000000, build_seconds: 0.1 } });

  it("searches the records and opens one's lineage in the side card, with its Loop stage's verdict", async () => {
    routes["GET /api/system/loop"] = () => ({ body: LOOP });
    routes["GET /api/provenance/summary"] = () => summary({ srcutoutartifact: 2 }, 2, { current: 0, stale: 1, unknown: 1 });
    const r = row("77777777", { path: "./data/x/SR.fits", verdict: "current", parents: ["33333333"], models: [{ id: "33333333", member: "member_01" }], n_upstream: 1 });
    routes["GET /api/provenance/records?limit=1000"] = () => ({ body: { ok: true, total: 1, offset: 0, limit: 1000, records: [r] } });
    routes["GET /api/provenance/record/77777777"] = () => ({ body: { ok: true, entry: r, record: { id: "77777777" },
      upstream: [{ id: "33333333", role: "parent", exists: true, kind: "checkpointartifact", label: "member_01", member: "member_01" }],
      downstream: [], ancestors: { total: 1, items: [] }, descendants: { total: 0, items: [] }, models: r.models,
      current_models: [], inspect_path: "data/x/SR.fits" } });
    show(<Lineage />);
    const grid = await screen.findByRole("grid", { name: "Provenance records" });
    // The index's own model check says "current"; the Loop (as on Home) says Real SR is stale — the Loop wins.
    expect(await within(grid).findByText("Real SR · stale")).toBeTruthy();
    fireEvent.click(await within(grid).findByText("77777777"));
    expect(await screen.findByText("Real SR: 449 stale")).toBeTruthy();
    expect(screen.getByText("Open the tab that fixes it").getAttribute("href")).toBe("/sky/targets");
    expect(screen.getByText("33333333 (member_01)")).toBeTruthy();
    expect(screen.getByText("Open the FITS in Files").getAttribute("href")).toBe("/files?fits=data%2Fx%2FSR.fits");
    expect(screen.queryByText(/Model check/)).toBeNull();
    expect(screen.queryByText(/carry no model id/)).toBeNull();
  });

  it("counts records per Loop verdict on the verdict control, and says once that model ids are missing", async () => {
    routes["GET /api/system/loop"] = () => ({ body: LOOP });
    routes["GET /api/provenance/summary"] = () => summary({ srcutoutartifact: 11094, checkpointartifact: 42, generationrun: 209 }, 11345, { current: 0, stale: 0, unknown: 11094 });
    routes["GET /api/provenance/records?limit=1000"] = () => ({ body: { ok: true, total: 11345, offset: 0, limit: 1000,
      records: [row("11111111"), row("22222222"), row("33333333")] } });
    routes["GET /api/provenance/records?limit=1000&kind=generationrun%2Csrcutoutartifact"] = () => ({ body: { ok: true, total: 11303,
      offset: 0, limit: 1000, records: [row("44444444")] } });
    show(<Lineage />);
    const note = await screen.findByText(/carry no model id/);
    expect(note.closest(".ui-callout")?.textContent).toBe("98% of records carry no model id, so each record takes its Loop stage's verdict.");
    // The bottom "no active member carries a provenance id" note would repeat it.
    expect(screen.queryByText(/No active member carries a provenance id/)).toBeNull();
    await waitFor(() => expect(screen.getAllByRole("radio").map((x) => x.textContent)).toEqual(["All", "Current · 42", "Stale · 11,303"]));
    // One count: the rows that loaded out of the server's total (no second "N rows").
    expect(await screen.findByText("showing 3 of 11,345")).toBeTruthy();
    expect(screen.queryByText("3 rows")).toBeNull();
    expect(screen.getByText(/Verdicts are the Loop's, as on Home/).textContent).toContain("Records stale (predate the stellar prior)");
    // Stale narrows the server's kind filter to the stale stages' kinds.
    fireEvent.click(screen.getByRole("radio", { name: "Stale · 11,303" }));
    expect(await screen.findByText("44444444")).toBeTruthy();
  });

  it("says so when the staleness service does not answer, and gives no verdict", async () => {
    routes["GET /api/system/loop"] = () => ({ status: 404, body: { ok: false, error: "not found" } });
    routes["GET /api/provenance/summary"] = () => summary({ srcutoutartifact: 1 }, 1);
    routes["GET /api/provenance/records?limit=1000"] = () => ({ body: { ok: true, total: 1, offset: 0, limit: 1000, records: [row("11111111")] } });
    show(<Lineage />);
    expect(await screen.findByText("The staleness service did not answer")).toBeTruthy();
    const grid = await screen.findByRole("grid", { name: "Provenance records" });
    expect(await within(grid).findByText("11111111")).toBeTruthy();
    expect(within(grid).queryByText(/· stale|· current/)).toBeNull();
  });
});

