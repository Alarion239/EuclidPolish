/* Backend-contract compatibility of legacy pages (plan WP-F3 grant, from
 * WP-B1b): Git commit/push confirmations and the refused-files guard,
 * Evaluation's confirmed --delete-after sync, Tracking's sandbox
 * `source_label` and confirmations, TNG's radii refresh job, and Config's
 * dirty-only save with base_version and 409 conflicts. */
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import type { ReactElement } from "react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useJobsStore } from "../api/jobs";
import { queryClient } from "../api/query";
import { resetConfirm } from "../ui";

vi.mock("../legacy", () => ({ CutoutViewer: () => null, loadColorEngine: () => new Promise(() => {}) }));

type Reply = { status?: number; body: unknown };
type Call = { url: string; method: string; form: Record<string, string> };
let routes: Record<string, (form: Record<string, string>) => Reply>;
let calls: Call[];

const formOf = (body: BodyInit | null | undefined): Record<string, string> => {
  if (!(body instanceof FormData)) return {};
  const out: Record<string, string> = {};
  body.forEach((v, k) => { out[k] = String(v); });
  return out;
};

beforeEach(() => {
  calls = [];
  routes = {};
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
});
afterEach(() => {
  act(() => resetConfirm());
  queryClient.clear();
});

const show = (el: ReactElement) => render(
  <QueryClientProvider client={queryClient}>
    <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>{el}</MemoryRouter>
  </QueryClientProvider>,
);
const posts = (url: string) => calls.filter((c) => c.method === "POST" && c.url === url);
const answer = async (title: RegExp | string, button: string) => {
  const dlg = await screen.findByRole("alertdialog", { name: title });
  fireEvent.click(within(dlg).getByRole("button", { name: button }));
  await waitFor(() => expect(screen.queryByRole("alertdialog", { name: title })).toBeNull());
  return dlg;
};

/* ── Git ─────────────────────────────────────────────────────────────── */

describe("Git page", () => {
  beforeEach(() => {
    routes["GET /api/git/status"] = () => ({ body: {
      status: { in_repo: true, root: "/repo", branch: "main", upstream: "origin/main", ahead: 2, behind: 0,
        files: [{ xy: " M", path: "a.py" }, { xy: "??", path: "big.fits" }], last: null },
      log: [],
    } });
  });

  async function typeMessage() {
    const { default: GitPage } = await import("./Git");
    show(<GitPage />);
    fireEvent.change(await screen.findByPlaceholderText("commit message…"), { target: { value: "msg" } });
  }

  it("confirms the changed files, then commits all=1", async () => {
    routes["POST /git/commit"] = () => ({ body: { ok: true, stdout: "[main abc] msg", committed: ["a.py", "big.fits"] } });
    await typeMessage();
    fireEvent.click(screen.getByRole("button", { name: "Stage all + commit" }));
    const dlg = await answer("Commit all 2 changed files?", "Commit all");
    expect(dlg.textContent).toContain("a.py");
    expect(dlg.textContent).toContain("big.fits");
    await waitFor(() => expect(posts("/git/commit")).toHaveLength(1));
    expect(posts("/git/commit")[0].form).toEqual({ message: "msg", all: "1" });
  });

  it("does nothing when the file list is not confirmed", async () => {
    await typeMessage();
    fireEvent.click(screen.getByRole("button", { name: "Stage all + commit" }));
    await answer("Commit all 2 changed files?", "Cancel");
    expect(posts("/git/commit")).toHaveLength(0);
  });

  it("shows refused files on 409 and retries with force=1 when confirmed", async () => {
    routes["POST /git/commit"] = (form) => (form.force === "1"
      ? { body: { ok: true, stdout: "forced", committed: ["a.py", "big.fits"] } }
      : { status: 409, body: { ok: false, code: "refused_files", error: "refused", refused: [{ path: "big.fits", size: 12e6, reason: "larger than 10 MB" }] } });
    await typeMessage();
    fireEvent.click(screen.getByRole("button", { name: "Stage all + commit" }));
    await answer("Commit all 2 changed files?", "Commit all");
    const dlg = await answer("Commit 1 large or binary file anyway?", "Force commit");
    expect(dlg.textContent).toContain("big.fits · 12.0 MB — larger than 10 MB");
    await waitFor(() => expect(posts("/git/commit")).toHaveLength(2));
    expect(posts("/git/commit")[1].form).toEqual({ message: "msg", all: "1", force: "1" });
    expect(await screen.findByText(/forced/)).toBeTruthy();
  });

  it("explains a 400 nothing_selected", async () => {
    routes["POST /git/commit"] = () => ({ status: 400, body: { ok: false, code: "nothing_selected", error: "no changed file matches" } });
    await typeMessage();
    fireEvent.click(screen.getByRole("button", { name: "Stage all + commit" }));
    await answer("Commit all 2 changed files?", "Commit all");
    expect(await screen.findByText(/nothing to commit/)).toBeTruthy();
  });

  it("confirms before pushing", async () => {
    routes["POST /git/push"] = () => ({ body: { ok: true, stdout: "pushed" } });
    const { default: GitPage } = await import("./Git");
    show(<GitPage />);
    fireEvent.click(await screen.findByRole("button", { name: "Push" }));
    const dlg = await answer("Push main to origin/main?", "Push");
    expect(dlg.textContent).toContain("2 local commits will be published.");
    await waitFor(() => expect(posts("/git/push")).toHaveLength(1));
  });
});

/* ── Evaluation ──────────────────────────────────────────────────────── */

describe("Evaluation page", () => {
  it("syncs only after confirming the --delete-after, with confirm=1", async () => {
    routes["GET /api/evaluation/runs"] = () => ({ body: { n: 0, n_ok: 0, rows: [] } });
    routes["POST /api/evaluation/sync"] = (form) => (form.confirm === "1"
      ? { body: { ok: true, stdout: "synced" } }
      : { status: 400, body: { ok: false, code: "confirm_required", error: "confirm" } });
    const { default: EvaluationPage } = await import("./Evaluation");
    show(<EvaluationPage />);
    const btn = await screen.findByRole("button", { name: "⟳ Sync results (FASRC)" });
    fireEvent.click(btn);
    const dlg = await answer("Sync evaluation results from FASRC?", "Cancel");
    expect(dlg.textContent).toContain("delete local-only results");
    expect(posts("/api/evaluation/sync")).toHaveLength(0);
    fireEvent.click(btn);
    await answer("Sync evaluation results from FASRC?", "Sync and delete local-only");
    await waitFor(() => expect(posts("/api/evaluation/sync")).toHaveLength(1));
    expect(posts("/api/evaluation/sync")[0].form).toEqual({ confirm: "1" });
  });
});

/* ── Tracking ────────────────────────────────────────────────────────── */

describe("Tracking page", () => {
  beforeEach(() => {
    routes["GET /api/tracking/state"] = () => ({ body: {
      active: { title: "gate-sweep", slug: "gate-sweep" }, archived: [
        { title: "old run", slug: "old-run", _dir: "d", saved_commit: { short: "abc1234" }, models: [] },
      ],
      backups: { models: [], fits: [], images: [] }, jobs: [], log_md: "", ssh_connected: false,
      sandboxes: [{ short: "tt01", source: { kind: "campaign", slug: "old-run" }, source_label: "campaign old-run", running: true }],
    } });
    routes["POST /api/tracking/timetravel/remove"] = () => ({ body: { ok: true } });
    routes["POST /api/tracking/save"] = () => ({ body: { ok: true } });
    routes["POST /api/tracking/timetravel/restore"] = () => ({ body: { ok: true } });
  });

  it("renders the sandbox source_label (source is an object) and confirms removal", async () => {
    const { default: TrackingPage } = await import("./Tracking");
    show(<TrackingPage />);
    expect(await screen.findByText("campaign old-run")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "remove" }));
    await answer("Remove sandbox tt01?", "Cancel");
    expect(posts("/api/tracking/timetravel/remove")).toHaveLength(0);
    fireEvent.click(screen.getByRole("button", { name: "remove" }));
    await answer("Remove sandbox tt01?", "Remove");
    await waitFor(() => expect(posts("/api/tracking/timetravel/remove")[0]?.form).toEqual({ short: "tt01" }));
  });

  it("confirms saving a snapshot and time travel", async () => {
    const { default: TrackingPage } = await import("./Tracking");
    show(<TrackingPage />);
    fireEvent.click(await screen.findByRole("button", { name: "Save snapshot" }));
    await answer("Save a snapshot of “gate-sweep”?", "Save snapshot");
    await waitFor(() => expect(posts("/api/tracking/save")).toHaveLength(1));
    fireEvent.click(screen.getByRole("button", { name: "⏱ time-travel" }));
    const dlg = await answer("Time-travel to “old run”?", "Start sandbox");
    expect(dlg.textContent).toContain("abc1234");
    await waitFor(() => expect(posts("/api/tracking/timetravel/restore")[0]?.form).toEqual({ campaign: "old-run", remote: "0" }));
  });
});

/* ── TNG ─────────────────────────────────────────────────────────────── */

describe("TNG page radii", () => {
  beforeEach(() => {
    routes["GET /tng-auth/status"] = () => ({ body: { present: true, connected: true, chars: 32 } });
    routes["GET /api/fasrc/steps/status"] = () => ({ body: { ssh_connected: true, steps: [], artifacts: {}, remote_paths: {} } });
    routes["GET /api/fasrc/status"] = () => ({ body: { ssh_connected: true } });
  });

  it("starts the refresh job when the cache is stale and FASRC is connected, and shows it", async () => {
    routes["GET /api/tng/radii/status"] = () => ({ body: { valid: false, stale: true, connected: true, refresh_job: null, expected_count: 10, valid_count: 0 } });
    routes["POST /api/tng/radii/refresh"] = () => ({ body: { ok: true, job_id: "r1" } });
    routes["GET /api/jobs/r1"] = () => ({ body: {
      job_id: "r1", label: "TNG radii validation", kind: "tng-radii", status: "running", started: 1, finished: null,
      duration: 1, error: null, log: "", log_truncated: false, cancellable: true, cancel_requested: false, result: null,
      progress: { current: 0, total: 0, pct: 0, label: "" },
    } });
    const { default: TngPage } = await import("./Tng");
    show(<TngPage />);
    await waitFor(() => expect(posts("/api/tng/radii/refresh")).toHaveLength(1));
    expect(await screen.findByText("TNG radii validation")).toBeTruthy();
  });

  it("does not start it while offline or when a refresh already runs", async () => {
    routes["GET /api/tng/radii/status"] = () => ({ body: { valid: false, stale: true, connected: false, refresh_job: null } });
    const { default: TngPage } = await import("./Tng");
    const { unmount } = show(<TngPage />);
    expect(await screen.findByText(/connect to FASRC to re-check it/)).toBeTruthy();
    unmount();
    queryClient.clear();
    routes["GET /api/tng/radii/status"] = () => ({ body: { valid: false, stale: true, connected: true, refresh_job: "other" } });
    show(<TngPage />);
    await screen.findByText(/frames valid/);
    expect(posts("/api/tng/radii/refresh")).toHaveLength(0);
  });
});

/* ── Config ──────────────────────────────────────────────────────────── */

describe("Config page", () => {
  const CONFIG = { vis_pixels: 301, n_train: 1000, n_valid: 100, n_test: 100, hr_image_size: 600, plateau_lr_enabled: 0, plateau_lr_metric: "combined_loss" };

  it("posts only the edited fields with base_version", async () => {
    routes["GET /api/config"] = () => ({ body: { ok: true, config: CONFIG, version: "v1" } });
    routes["POST /api/config/save"] = (form) => ({ body: { ok: true, config: { ...CONFIG, n_train: Number(form.n_train) }, version: "v2", note: null } });
    const { default: ConfigPage } = await import("./Config");
    show(<ConfigPage />);
    const input = await screen.findByLabelText("Train scenes");
    fireEvent.change(input, { target: { value: "2000" } });
    fireEvent.click(screen.getByRole("button", { name: "Save config" }));
    await waitFor(() => expect(posts("/api/config/save")).toHaveLength(1));
    expect(posts("/api/config/save")[0].form).toEqual({ n_train: "2000", base_version: "v1" });
  });

  it("shows a 409 conflict and rebases on the server values, keeping other edits", async () => {
    routes["GET /api/config"] = () => ({ body: { ok: true, config: CONFIG, version: "v1" } });
    let n = 0;
    routes["POST /api/config/save"] = (form) => {
      n += 1;
      if (n === 1) {
        return { status: 409, body: {
          ok: false, code: "config_conflict", error: "the config changed since you loaded it: n_train",
          conflicts: { n_train: { base: 1000, current: 5000 } },
          config: { ...CONFIG, n_train: 5000 }, version: "v9",
        } };
      }
      return { body: { ok: true, config: { ...CONFIG, n_train: 5000, n_valid: Number(form.n_valid) }, version: "v10" } };
    };
    const { default: ConfigPage } = await import("./Config");
    show(<ConfigPage />);
    fireEvent.change(await screen.findByLabelText("Train scenes"), { target: { value: "2000" } });
    fireEvent.change(screen.getByLabelText("Validate scenes"), { target: { value: "150" } });
    fireEvent.click(screen.getByRole("button", { name: "Save config" }));
    expect(await screen.findByText("The config changed since you loaded it")).toBeTruthy();
    expect(screen.getByText("n_train: yours 2000 · now 5000")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Reload server values" }));
    expect((screen.getByLabelText("Train scenes") as HTMLInputElement).value).toBe("5000");
    expect((screen.getByLabelText("Validate scenes") as HTMLInputElement).value).toBe("150");
    fireEvent.click(screen.getByRole("button", { name: "Save config" }));
    await waitFor(() => expect(posts("/api/config/save")).toHaveLength(2));
    expect(posts("/api/config/save")[1].form).toEqual({ n_valid: "150", base_version: "v9" });
  });
});

/* ── Ensemble ────────────────────────────────────────────────────────── */

describe("Ensemble page", () => {
  it("archives a member only after the kit confirm()", async () => {
    routes["GET /ensemble/status.json?mode=starfull"] = () => ({ body: {
      base_dir: "ckpt", members: [{ name: "member_7", psnr: 44.1, loss: "l1", blocks: 16, asinh_knee: 100, step: 1000 }],
      archived: [], n_members: 1, n_models: 1, test_present: true, psnr_fields: 10,
      evaluations_available: false, eval_summary: null,
    } });
    routes["POST /ensemble/archive-member"] = () => ({ body: { ok: true } });
    const confirmSpy = vi.spyOn(window, "confirm");
    const { default: EnsemblePage } = await import("./Ensemble");
    render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={["/ensemble/starfull/members"]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <Routes><Route path="/ensemble/:mode/*" element={<EnsemblePage />} /></Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    );
    const archive = await screen.findByTitle(/zip → tracking/);
    fireEvent.click(archive);
    await answer("Archive member_7?", "Cancel");
    expect(posts("/ensemble/archive-member")).toHaveLength(0);
    fireEvent.click(archive);
    await answer("Archive member_7?", "Archive");
    await waitFor(() => expect(posts("/ensemble/archive-member")[0]?.form).toEqual({ member: "member_7" }));
    expect(confirmSpy).not.toHaveBeenCalled();
  });
});
