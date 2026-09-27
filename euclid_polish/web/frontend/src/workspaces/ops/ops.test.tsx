/* Ops workspace against a mocked Flask: the schema-driven step card, the
 * FASRC queue and history, the Git tab's commit guard (re-homed from the
 * legacy pages/contracts.test.tsx), tracking confirmations and time travel,
 * the provenance browser and the local job centre. */
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import type { ReactElement } from "react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useJobsStore } from "../../api/jobs";
import { queryClient } from "../../api/query";
import { resetConfirm } from "../../ui";
import { HistoryPanel } from "./fasrc/History";
import { LiveJobs } from "./fasrc/Live";
import { QueuePanel } from "./fasrc/Queue";
import { StepCard } from "./steps/StepCard";
import type { Step } from "./steps/stepForm";
import GitTab from "./tabs/Git";
import JobsTab from "./tabs/Jobs";
import ProvenanceTab from "./tabs/Provenance";
import FasrcTab from "./tabs/Fasrc";
import TrackingTab from "./tabs/Tracking";

type Reply = { status?: number; body: unknown };
type Call = { url: string; method: string; form: Record<string, string | string[]> };
let routes: Record<string, (form: Call["form"], url: URL) => Reply>;
let calls: Call[];

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
    "GET /api/fasrc/current-submission": () => ({ body: { ok: true, current: null, live: [], queue: { count: 0, names: [], items: [], halted: false } } }),
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

const show = (el: ReactElement, url = "/ops") => render(
  <QueryClientProvider client={queryClient}>
    <MemoryRouter initialEntries={[url]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>{el}</MemoryRouter>
  </QueryClientProvider>,
);
const posts = (path: string) => calls.filter((c) => c.method === "POST" && c.url === path);
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

/* ── queue + history ─────────────────────────────────────────────────────── */

describe("FASRC queue", () => {
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

describe("FASRC history", () => {
  it("lists runs and reconciles the unresolved ones as a job", async () => {
    routes["GET /api/fasrc/history"] = () => ({ body: { ok: true, total: 1, offset: 0, limit: 2000, unresolved: 1,
      facets: { steps: { euclid_query: 1 }, states: { UNKNOWN: 1 } },
      rows: [{ jobid: "7", step_id: "euclid_query", state: "", db_state: "UNKNOWN", state_display: "UNKNOWN",
        submitted_at: "2026-09-01T00:00:00Z", params: { num_stars: "10" } }] } });
    routes["POST /api/fasrc/refresh-accounting"] = () => ({ body: { ok: true, job_id: "j1" } });
    routes["GET /api/jobs/j1"] = () => ({ body: { job_id: "j1", label: "reconcile", status: "done", duration: 1, error: null,
      log: "", log_truncated: false, result: { ok: true, updated: 1, total: 1, resolved: { "7": "COMPLETED" } },
      progress: { current: 1, total: 1, pct: 100, label: "" } } });
    show(<HistoryPanel fasrcConnected />);
    expect(await screen.findByText("UNKNOWN")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Reconcile 1/ }));
    await waitFor(() => expect(posts("/api/fasrc/refresh-accounting")[0]?.form).toEqual({ scope: "unresolved" }));
  });
});

/* ── git (re-homed from pages/contracts.test.tsx) ────────────────────────── */

describe("FASRC live jobs", () => {
  it("lists CLI-submitted squeue jobs next to the console's, read-only", async () => {
    routes["GET /api/fasrc/current-submission"] = () => ({ body: { ok: true, current: null,
      live: [{ jobid: "100", state: "RUNNING", label: "train run", step_id: "ensemble_train" }],
      queue: { count: 0, names: [], items: [], halted: false } } });
    routes["GET /api/fasrc/queue"] = () => ({ body: { ok: true, rows: [
      { jobid: "100", name: "train", state: "RUNNING", reason: "holygpu1" },
      { jobid: "48096695", name: "smoke-single", state: "RUNNING", time: "3:00", time_limit: "1:00:00", reason: "holy7c" },
    ] } });
    routes["GET /api/fasrc/jobs/100/status"] = () => ({ body: { ok: true, has_events: false } });
    show(<LiveJobs />, "/ops?job=48096695");
    expect((await screen.findAllByText("smoke-single")).length).toBeGreaterThan(0);
    expect(screen.getAllByText("CLI").length).toBeGreaterThan(0);
    expect(await screen.findByText("Submitted outside the console")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Cancel 48096695" })).toBeNull();
    expect(screen.getByRole("button", { name: "Cancel 100" })).toBeTruthy();
  });
});

describe("FASRC steps deep link", () => {
  const OTHER: Step = { ...QUERY, step_id: "vis_noise_sample", label: "Sample VIS noise", task_params: [], last_params: null, outputs: [] };
  beforeEach(() => {
    routes["GET /api/fasrc/steps/status"] = () => ({ body: { ok: true, ssh_connected: true, steps: [QUERY, OTHER] } });
    routes["GET /api/fasrc/steps/vis_noise_sample/history"] = () => ({ body: { ok: true, step_id: "vis_noise_sample", match: null, history: [] } });
    routes["GET /api/fasrc/steps/euclid_query/history"] = () => ({ body: { ok: true, step_id: "euclid_query", match: null, history: [] } });
  });

  it("opens the step named by ?view=steps&step= (Settings chips + the palette link here)", async () => {
    show(<FasrcTab />, "/ops/fasrc?view=steps&step=vis_noise_sample");
    const nav = await screen.findByRole("navigation", { name: "FASRC steps" });
    const item = within(nav).getByText("Sample VIS noise").closest("button");
    expect(item?.getAttribute("aria-current")).toBe("true");
    expect(screen.queryByText(/No step/)).toBeNull();
  });

  it("warns when the linked step is not registered", async () => {
    show(<FasrcTab />, "/ops/fasrc?view=steps&step=gone_step");
    expect(await screen.findByText("No step “gone_step”")).toBeTruthy();
  });
});

describe("Git tab", () => {
  const FILES = [
    { xy: " M", path: "a.py", orig: null, staged: false, unstaged: true, untracked: false, size: 10, guard: null },
    { xy: "??", path: "big.fits", orig: null, staged: false, unstaged: true, untracked: true, size: 12e6, guard: "file > 10 MB" },
  ];
  beforeEach(() => {
    routes["GET /api/git/status"] = () => ({ body: {
      status: { in_repo: true, root: "/repo", branch: "main", upstream: "origin/main", ahead: 2, behind: 0, files: FILES, last: null },
      log: [] } });
    routes["GET /api/git/diff"] = () => ({ body: { diff: "", staged: false, path: null } });
    routes["GET /api/git/log"] = () => ({ body: { commits: [], total: 0, skip: 0, limit: 100, has_more: false } });
  });
  async function typeMessage() {
    show(<GitTab />);
    fireEvent.change(await screen.findByLabelText("Commit message"), { target: { value: "msg" } });
  }

  it("confirms the changed files, then commits all=1", async () => {
    routes["POST /git/commit"] = () => ({ body: { ok: true, stdout: "[main abc] msg", committed: ["a.py", "big.fits"] } });
    await typeMessage();
    fireEvent.click(screen.getByRole("button", { name: "Commit all" }));
    const dlg = await answer("Commit all 2 changed files?", "Commit all");
    expect(dlg.textContent).toContain("a.py");
    expect(dlg.textContent).toContain("big.fits  (file > 10 MB)");
    await waitFor(() => expect(posts("/git/commit")).toHaveLength(1));
    expect(posts("/git/commit")[0].form).toEqual({ message: "msg", all: "1" });
  });

  it("does nothing when the file list is not confirmed", async () => {
    await typeMessage();
    fireEvent.click(screen.getByRole("button", { name: "Commit all" }));
    await answer("Commit all 2 changed files?", "Cancel");
    expect(posts("/git/commit")).toHaveLength(0);
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
    expect(await screen.findByText(/forced/)).toBeTruthy();
  });

  it("explains a 400 nothing_selected", async () => {
    routes["POST /git/commit"] = () => ({ status: 400, body: { ok: false, code: "nothing_selected", error: "no changed file matches" } });
    await typeMessage();
    fireEvent.click(screen.getByRole("button", { name: "Commit all" }));
    await answer("Commit all 2 changed files?", "Commit all");
    expect(await screen.findByText(/Nothing to commit/)).toBeTruthy();
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

  it("pages the history with skip (the server caps one page at 500)", async () => {
    const commit = (i: number) => ({ hash: `h${i}`, full: `full${i}`, author: "a", relative: "now", subject: `c${i}` });
    routes["GET /api/git/log"] = (_f, url) => {
      const skip = Number(url.searchParams.get("skip")), limit = Number(url.searchParams.get("limit"));
      const commits = Array.from({ length: limit }, (_, k) => commit(skip + k)).filter((c) => Number(c.hash.slice(1)) < 250);
      return { body: { commits, total: 250, skip, limit, has_more: skip + commits.length < 250 } };
    };
    show(<GitTab />);
    expect(await screen.findByText("100 of 250 commits")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Load 100 more" }));
    expect(await screen.findByText("200 of 250 commits")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Load 100 more" }));
    expect(await screen.findByText("250 of 250 commits")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Load 100 more" })).toBeNull();
    const logCalls = calls.filter((c) => c.url.startsWith("/api/git/log")).map((c) => c.url);
    expect(logCalls).toEqual(expect.arrayContaining(["/api/git/log?skip=100&limit=100", "/api/git/log?skip=200&limit=100"]));
  });

  it("confirms before pushing", async () => {
    routes["POST /git/push"] = () => ({ body: { ok: true, stdout: "pushed" } });
    show(<GitTab />);
    fireEvent.click(await screen.findByRole("button", { name: "Push" }));
    const dlg = await answer("Push main to origin/main?", "Push");
    expect(dlg.textContent).toContain("2 local commits will be published.");
    await waitFor(() => expect(posts("/git/push")).toHaveLength(1));
  });
});

/* ── tracking (re-homed from pages/contracts.test.tsx) ───────────────────── */

describe("Tracking tab", () => {
  beforeEach(() => {
    routes["GET /api/tracking/state"] = () => ({ body: {
      active: { title: "gate-sweep", slug: "gate-sweep", created_commit: { short: "fff0000" } },
      archived: [{ title: "old run", slug: "old-run", _dir: "old-run-2", saved_commit: { short: "abc1234", hash: "abc1234ffff" }, models: [] }],
      backups: { models: [], fits: [], images: [] }, jobs_count: 0, unassigned_count: 0, log_md: "# gate-sweep\n\n## 2026-09-01T00:00:00Z\n\nA **result**.",
      ssh_connected: false,
      sandboxes: [{ short: "tt01", source: { kind: "campaign", slug: "old-run" }, source_label: "campaign old-run", running: true }],
    } });
    routes["POST /api/tracking/timetravel/remove"] = () => ({ body: { ok: true } });
    routes["POST /api/tracking/save"] = () => ({ body: { ok: true } });
    routes["POST /api/tracking/timetravel/restore"] = () => ({ body: { ok: true, short: "abc1234", url: "http://127.0.0.1:8766/", warning: null } });
  });

  it("renders the notebook as markdown", async () => {
    show(<TrackingTab />);
    expect((await screen.findByText("result")).tagName).toBe("STRONG");
  });

  it("opens the notebook newest first, with a jump-to-day menu, and has no repeated page title", async () => {
    routes["GET /api/tracking/state"] = () => ({ body: {
      active: { title: "gate-sweep", slug: "gate-sweep", created_commit: { short: "fff0000" } }, archived: [],
      backups: { models: [], fits: [], images: [] }, jobs_count: 0, unassigned_count: 0, ssh_connected: false, sandboxes: [],
      log_md: "# gate-sweep\n\n## 2026-07-02T02:37:41Z\n\nold one\n\n## 2026-09-21T01:36:49Z\n\nmiddle\n\n## 2026-09-21T14:42:29Z\n\nnewest",
    } });
    show(<TrackingTab />);
    await screen.findByText("newest");
    const doc = document.querySelector(".ops-notebook__doc") as HTMLElement;
    const order = [...doc.querySelectorAll("h3, h2")].map((h) => h.textContent).filter((t) => t?.startsWith("2026"));
    expect(order).toEqual(["2026-09-21T14:42:29Z", "2026-09-21T01:36:49Z", "2026-07-02T02:37:41Z"]);
    expect(screen.getByRole("radio", { name: "Newest first" }).getAttribute("aria-checked")).toBe("true");
    expect(screen.getByText("3 entries, 2026-07-02 to 2026-09-21")).toBeTruthy();
    const jump = screen.getByRole("combobox", { name: "Jump to a day" });
    expect([...jump.querySelectorAll("option")].map((o) => o.textContent)).toEqual(["Jump to a day…", "Sep 21, 2026", "Jul 2, 2026"]);
    expect(screen.queryByRole("heading", { level: 1, name: "Tracking" })).toBeNull();
    fireEvent.click(screen.getByRole("radio", { name: "Oldest first" }));
    await waitFor(() => expect([...doc.querySelectorAll("h3, h2")].map((h) => h.textContent).filter((t) => t?.startsWith("2026"))[0])
      .toBe("2026-07-02T02:37:41Z"));
  });

  it("renders the sandbox source_label (source is an object) and confirms removal", async () => {
    show(<TrackingTab />, "/ops/tracking?view=sandboxes");
    expect(await screen.findByText("campaign old-run")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Remove" }));
    await answer("Remove sandbox tt01?", "Cancel");
    expect(posts("/api/tracking/timetravel/remove")).toHaveLength(0);
    fireEvent.click(screen.getByRole("button", { name: "Remove" }));
    await answer("Remove sandbox tt01?", "Remove");
    await waitFor(() => expect(posts("/api/tracking/timetravel/remove")[0]?.form).toEqual({ short: "tt01" }));
  });

  it("confirms saving a snapshot", async () => {
    show(<TrackingTab />);
    fireEvent.click(await screen.findByRole("button", { name: "Save snapshot" }));
    await answer("Save “gate-sweep”?", "Save snapshot");
    await waitFor(() => expect(posts("/api/tracking/save")).toHaveLength(1));
  });

  it("time-travels an archived campaign by its archive dir", async () => {
    show(<TrackingTab />, "/ops/tracking?view=archive");
    fireEvent.click(await screen.findByRole("button", { name: "Time-travel to old run" }));
    const dlg = await screen.findByRole("dialog");
    expect(dlg.textContent).toContain("abc1234");
    fireEvent.click(within(dlg).getByRole("button", { name: "Start sandbox" }));
    await waitFor(() => expect(posts("/api/tracking/timetravel/restore")[0]?.form).toEqual({ campaign: "old-run-2", remote: "0" }));
    expect(await screen.findByText("Sandbox abc1234 is running")).toBeTruthy();
  });
});

/* ── provenance ──────────────────────────────────────────────────────────── */

describe("Provenance tab", () => {
  it("searches the records and opens one's lineage", async () => {
    routes["GET /api/provenance/summary"] = () => ({ body: { ok: true, total: 2, counts: { kinds: { srcutoutartifact: 1, checkpointartifact: 1 },
      verdicts: { current: 0, stale: 1, unknown: 0 } }, roots: [], current_models: [], truncated: false, duplicates: 0, built_at: 1790000000, build_seconds: 0.1 } });
    const row = { id: "77777777", kind: "srcutoutartifact", category: "artifact", source: "sidecar", file: "data/x/77777777.srcutoutartifact.json",
      created_at: "2026-02-01T00:00:00+00:00", status: null, path: "./data/x/SR.fits", format: "fits", label: "x/SR.fits", git: "abc", dirty: false,
      config_type: null, seed: null, produced_by: null, parents: ["33333333"], inputs: [], outputs: [], ra: null, dec: null, member: null,
      verdict: "stale", models: [{ id: "33333333", member: "member_01" }], n_upstream: 1, n_downstream: 0 };
    routes["GET /api/provenance/records"] = (_f, url) => ({ body: { ok: true, total: 1, offset: 0, limit: 1000,
      records: url.searchParams.get("verdict") === "current" ? [] : [row] } });
    routes["GET /api/provenance/record/77777777"] = () => ({ body: { ok: true, entry: row, record: { id: "77777777" },
      upstream: [{ id: "33333333", role: "parent", exists: true, kind: "checkpointartifact", label: "member_01", member: "member_01" }],
      downstream: [], ancestors: { total: 1, items: [] }, descendants: { total: 0, items: [] }, models: row.models,
      current_models: [], inspect_path: "data/x/SR.fits" } });
    show(<ProvenanceTab />);
    const grid = await screen.findByRole("grid", { name: "Provenance records" });
    fireEvent.click(await within(grid).findByText("77777777"));
    expect(await screen.findByText("Model check: stale")).toBeTruthy();
    expect(screen.getByText("Open the FITS in Inspect").getAttribute("href")).toBe("/inspect?fits=data%2Fx%2FSR.fits");
    expect(screen.queryByText(/carry no model id/)).toBeNull();          // verdicts mean something here
    fireEvent.click(screen.getByRole("radio", { name: "Current (0)" }));
    await waitFor(() => expect(calls.some((c) => c.url.includes("verdict=current"))).toBe(true));
  });

  it("says once that the verdicts mean nothing yet, with counts only on the verdict control", async () => {
    routes["GET /api/provenance/summary"] = () => ({ body: { ok: true, total: 11345, counts: { kinds: { srcutoutartifact: 11094 },
      verdicts: { current: 0, stale: 0, unknown: 11094 } }, roots: [{ path: "data/_prov", records: 11345 }],
      current_models: Array.from({ length: 42 }, (_, i) => ({ id: `m${i}`, member: `member_${i}` })),
      truncated: false, duplicates: 0, built_at: 1790000000, build_seconds: 0.1 } });
    const row = (id: string) => ({ id, kind: "srcutoutartifact", category: "artifact", source: "sidecar", file: `data/x/${id}.json`,
      created_at: "2026-02-01T00:00:00+00:00", status: null, path: null, format: "fits", label: `x/${id}.fits`, git: "abc", dirty: false,
      config_type: null, seed: null, produced_by: null, parents: [], inputs: [], outputs: [], ra: null, dec: null, member: null,
      verdict: "unknown", models: [], n_upstream: 0, n_downstream: 0 });
    routes["GET /api/provenance/records"] = () => ({ body: { ok: true, total: 11345, offset: 0, limit: 1000,
      records: [row("11111111"), row("22222222"), row("33333333")] } });
    show(<ProvenanceTab />);
    const note = await screen.findByText(/carry no model id/);
    expect(note.closest(".ui-callout")?.textContent).toBe("98% of records carry no model id, so current/stale verdicts are not meaningful yet.");
    expect(document.querySelector(".ui-kpi, .ops-kpis")).toBeNull();
    expect(screen.getAllByRole("radio").map((r) => r.textContent)).toEqual(["All", "Current (0)", "Stale (0)", "No model (11,094)"]);
    expect(await screen.findByText("11,345 match in total")).toBeTruthy();  // the total, beside the table's own "3 rows"
    expect(screen.getAllByText(/^3 rows$/)).toHaveLength(1);
    expect(screen.queryByText(/showing/)).toBeNull();                   // the loaded count is not repeated
    expect(screen.queryByText("42")).toBeNull();                         // the active-models tile is gone
  });
});

/* ── local jobs ──────────────────────────────────────────────────────────── */

describe("Jobs tab", () => {
  it("filters by status and shows the selected job's log", async () => {
    const job = { job_id: "a1", label: "Evaluate", kind: "evaluate", status: "failed", started: 1790000000, finished: 1790000060,
      duration: 60, error: "boom", log: null, log_truncated: false, progress: { current: 0, total: 0, pct: 0, label: "" } };
    routes["GET /api/jobs"] = () => ({ body: [job, { ...job, job_id: "b2", label: "Knee", status: "done", error: null }] });
    routes["GET /api/jobs/a1"] = () => ({ body: { ...job, log: "line one\nTraceback: boom" } });
    show(<JobsTab />, "/ops/jobs?status=failed");
    const grid = await screen.findByRole("grid", { name: "Local jobs" });
    await waitFor(() => expect(within(grid).queryByText("Knee")).toBeNull());
    fireEvent.click(within(grid).getByText("Evaluate"));
    expect(await screen.findByText(/Traceback: boom/)).toBeTruthy();
  });
});
