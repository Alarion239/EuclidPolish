/* The Models tabs against a mocked Flask (console regrouping, Team M):
 * Leaderboard (status line, verdict, comparison table with the real
 * benchmark, TIMEOUT alert line, knee curves and the leaderboard), Members
 * (roster, selection toolbar, archive only after confirm(), Images link, the
 * Pull banner), Train (preview, regime in the button, resources), Combiner
 * (current membership + history, real holes from ONE Sky › Compare run, gate
 * share → Members, held-out loss scale, promote guard, never the RBF),
 * Diagnostics (facet legend, the spread answer + coverage, old d= values,
 * SR-scale focus, real field, recovery), Images (viewer first, ONE member
 * picker side panel, stamps, Generate SR asks first), the member inspector
 * and the pixel back-trace. */
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import type { ReactElement } from "react";
import { Link, MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, onTestFinished, vi } from "vitest";
import { useJobsStore } from "../../api/jobs";
import { queryClient } from "../../api/query";
import { useInspector } from "../../state/inspector";
import { useSelection } from "../../state/selection";
import { resetConfirm } from "../../ui";
import type { CombinersPayload, KneeModel, MemberDetail, MemberRow, Overview as OverviewData } from "./api";

const viewerProps: Record<string, unknown>[] = [];
vi.mock("../../viewer", () => ({
  ImageViewer: (props: { onReady?: (api: unknown) => void; collection: string }) => {
    viewerProps.push(props);
    props.onReady?.({ getState: () => ({ tiers: ["lr", "sr"] }), setTiers: vi.fn(), setMorphMembers: vi.fn(), reload: vi.fn(), goToId: vi.fn() });
    return <div data-testid="viewer" data-collection={props.collection} />;
  },
  renderCubeImageData: vi.fn(),
}));

type Reply = { status?: number; body: unknown };
let routes: Record<string, (form: Record<string, string>) => Reply>;
let calls: { url: string; method: string; form: Record<string, string> }[];
const posts = (u: string) => calls.filter((c) => c.method === "POST" && c.url === u);
const gets = (u: string) => calls.filter((c) => c.method === "GET" && c.url === u);

const formOf = (body: BodyInit | null | undefined): Record<string, string> => {
  const out: Record<string, string> = {};
  if (body instanceof FormData) body.forEach((v, k) => { out[k] = String(v); });
  else if (body instanceof URLSearchParams) body.forEach((v, k) => { out[k] = v; });
  else if (typeof body === "string") new URLSearchParams(body).forEach((v, k) => { out[k] = v; });
  return out;
};

const member = (n: number, patch: Partial<MemberRow> = {}): MemberRow => ({
  name: `member_${n}`, label: `${n}·psnr`, starless: false, regime: "starfull", origin: { seed: n },
  loss: "l2", blocks: 32, asinh_knee: 10, step: 70000, target_steps: 70000, fraction: 1,
  status: "complete", timeout: false, job: { jobid: "48107719", state: "COMPLETED" },
  psnr: 61.8, psnr_rank: 1, knee_integrated: { VIS: 53.5, Y_E: 62.4, J_E: 59.7, H_E: 58.8, mean: 58.6 }, knee_rank: 1,
  gate_usage: { VIS: 0.05, Y_E: 0.05, J_E: 0.05, H_E: 0.05 }, gate_usage_source: null, coherence: { overall: 0.92, sr: 0.58 },
  ...patch,
});

const MEMBERS = {
  regime: "starfull", other_regime_members: 12, psnr_fields: 100, eval_subset: "test",
  knee: { available: true, stale: false, n_fields: 100 }, gate: { available: true, stale: false, n_members: 30 },
  members: [
    member(196, { asinh_knee: null, asinh_knees: [0.1, 1, 10, 100, 1000, 10000], output_knee: 10, knee_loss: "balanced",
      knee_integrated: { VIS: 54.7, Y_E: 64.1, J_E: 60.7, H_E: 60.3, mean: 59.96 } }),
    member(197, { asinh_knee: null, asinh_knees: [0.1, 1, 10, 100, 1000, 10000], output_knee: null }),
    member(178, { step: 52000, fraction: 52 / 70, status: "timeout", timeout: true }),
  ],
  archived: [{ name: "member_09", archived_at: "2026-07-02T19:58:50+00:00", zip: "models/ensemble-member-09.zip",
    commit: "ea19f95", zip_found: true, zip_path: "/t/current/models/ensemble-member-09.zip", campaign: "current", size_bytes: 45e6 }],
};

const OVERVIEW: OverviewData = {
  regime: "starfull", active_members: ["178·psnr", "196·psnr", "197·psnr"], n_members: 3, test_present: true,
  eval_subset: "test", evaluated_at: "2026-09-25T19:32:00Z", summary: { ensemble_psnr: 58.3753 },
  headline: {
    metric: "vis_asinh", knee_e: 100, n_scored: 100,
    production: { psnr: 59.2355, vs_mean_db: 0.8601, vs_best_member_db: 0.2948 },
    mean: { psnr: 58.3753, vs_mean_member_db: 1.1553 },
    best_member: { psnr: 58.9407, label: "171·psnr", mean_member_psnr: 57.22 },
    knee: { available: true, stale: false, n_fields: 100, integration: { from_e: 0.1, to_e: 10000 },
      production: 60.9731, production_bands: [55.8, 64.9, 62.0, 61.1], mean: 59.1063, best_member: 59.958, best_member_label: "196·psnr" },
  },
  checks: [
    { id: "eval-members", ok: true, tone: "good", title: "Evaluation vs members", detail: "matches", action: "evaluate" },
    { id: "gate-members", ok: false, tone: "warn", title: "Production gate vs members", detail: "Fitted for 30 members; 31 are active", action: "combiners" },
  ],
  production_gate: { available: true, n_members: 30, mix_space: "linear", fitted_at: "2026-09-25T19:00:00Z" },
};

const KNEE = {
  available: true, stale: false, n_fields: 100, knees: [0.1, 1, 10, 100, 1000, 10000], bands: ["VIS", "Y_E", "J_E", "H_E"],
  integration: { from_e: 0.1, to_e: 10000 },
  models: [
    { id: "member_0", kind: "member", label: "196·psnr", asinh_knees: [0.1, 10], output_knee: 10, psnr: Array.from({ length: 6 }, () => [55, 64, 61, 60]), integrated: [55, 64, 61, 60] },
    { id: "member_1", kind: "member", label: "178·psnr", asinh_knee: 10, psnr: Array.from({ length: 6 }, (_, k) => (k < 3 ? [58, 66, 63, 62] : [50, 58, 55, 54])), integrated: [54, 62, 59, 58] },
    { id: "ensemble_mean", kind: "mean", label: "ensemble mean", psnr: Array.from({ length: 6 }, () => [54, 63, 60, 59]), integrated: [54, 63, 60, 59] },
    { id: "spatial_gate", kind: "combiner", label: "spatial gate", psnr: Array.from({ length: 6 }, () => [56, 65, 62, 61]), integrated: [56, 65, 62, 61] },
  ],
};

const variant = (name: string, patch: Partial<CombinersPayload["variants"][number]> = {}) => ({
  name, kind: "gate" as const, spec: name === "spatial_gate_combiner" ? "production" : `gate:${name.slice(13)}`,
  production: name === "spatial_gate_combiner", backup: name.startsWith("spatial_gate_backup_"),
  member_labels: ["178·psnr", "196·psnr", "197·psnr"], reads: ["178·psnr", "196·psnr", "197·psnr"], n_members: 3, n_reads: 3,
  pruned: false, mix_space: "linear", use_lr: false, width: 32, fitted_at: "2026-09-25T19:00:00Z",
  membership: { current: true, missing: [], extra: [] }, applies_to_test_cubes: true, fit: { steps: 2000 },
  selected: { step: 2000, loss: 0.65 }, baseline: { step: 0, loss: 0.96 }, history: [{ step: 0, loss: 0.96 }, { step: 2000, loss: 0.65 }],
  test: null, knee: null, ...patch,
});
const COMBINERS: CombinersPayload = {
  regime: "starfull", production: "spatial_gate_combiner", active_members: ["178·psnr", "196·psnr", "197·psnr"],
  cube_members: ["178·psnr", "196·psnr", "197·psnr"], compare: null, reports: [],
  variants: [
    variant("spatial_gate_combiner", { eval: { psnr: 59.2355 } }),
    variant("spatial_gate_p2", { member_labels: ["178·psnr", "196·psnr", "197·psnr"], reads: ["196·psnr", "197·psnr"], n_reads: 2, pruned: true }),
    variant("spatial_gate_26m", { member_labels: ["196·psnr"], reads: ["196·psnr"], n_members: 1, n_reads: 1,
      membership: { current: false, missing: [], extra: ["178·psnr", "197·psnr"] } }),
  ],
};

const DETAIL: MemberDetail = {
  name: "member_196", label: "196·psnr", active: true, archived: null, regime: "starfull",
  row: MEMBERS.members[0] as MemberRow,
  curves: { psnr: [[1000, 40], [2000, 42]], band_psnr: { VIS: [[1000, 39]] }, loss_series: [[1000, 0.2]], train_loss: [], gnorm: [], gnorm_max: [], step_time: [] },
  knee: { knees: KNEE.knees, bands: KNEE.bands, stale: false, models: KNEE.models as unknown as KneeModel[] },
  gate: null,
};

const TRAIN_JOB = {
  // ended yesterday: a recent batch (the Pull banner looks back 14 days)
  jobid: "48107719", state: "COMPLETED", submitted_at: "2026-09-24T04:54:24Z", ended_at: new Date(Date.now() - 86_400_000).toISOString(), mode: "add",
  member_names: ["member_195", "member_196"], req_time_limit: "3:00:00", req_memory: "32G", req_cpus: 4,
  params: { mode: "add", count: 2, steps: "70000", member_spec: JSON.stringify([
    { loss: "l2", bootstrap: 0.7, asinh_knees: [0.1, 1, 10, 100, 1000, 10000], knee_loss: "balanced", output_knee: 10, num_res_blocks: 32, icnr: true },
    { loss: "l2", bootstrap: 0.7, asinh_knee: 3000, num_res_blocks: 32, icnr: true }]) },
};

beforeEach(() => {
  calls = [];
  viewerProps.length = 0;
  routes = {
    "GET /ensemble/overview.json?mode=starfull": () => ({ body: OVERVIEW }),
    "GET /ensemble/members.json?mode=starfull": () => ({ body: MEMBERS }),
    "GET /ensemble/knee-psnr.json?mode=starfull": () => ({ body: KNEE }),
    "GET /ensemble/combiners.json?mode=starfull": () => ({ body: COMBINERS }),
    "GET /api/experiments": () => ({ body: { experiments: [] } }),
    "GET /api/models": () => ({ body: { regime: "starfull", models: [] } }),
    "GET /ensemble/training-jobs.json": () => ({ body: { jobs: [TRAIN_JOB] } }),
    "GET /api/fasrc/steps/status": () => ({ body: { ssh_connected: false, steps: [{ step_id: "ensemble_train",
      defaults: { partition: "gpu", n_cpus: 4, n_gpus: 1, memory: "32G", time_limit: "48:00:00" }, fixed_gpus: 1 }] } }),
    "GET /api/fasrc/status": () => ({ body: { ssh_connected: false, last_error: "timed out" } }),
    "GET /api/config": () => ({ body: { config: { psf_warp_prob: 1, psf_warp_alpha_max: 5, psf_warp_sigma: 3, saturation_mask_prob: 0.5,
      lr_peak: 0.0002, lr_final: 1e-6, lr_warmup_steps: 2000, plateau_lr_enabled: false } } }),
    "GET /ensemble/member/member_196.json": () => ({ body: DETAIL }),
    "POST /ensemble/evaluate": () => ({ body: { job_id: "ev1" } }),
    "POST /ensemble/archive-member": () => ({ body: { job_id: "ar1" } }),
    "GET /api/jobs/ar1": () => ({ body: { job_id: "ar1", status: "done", label: "archive", progress: null } }),
    "POST /ensemble/combiners/promote": () => ({ body: { ok: true, job_id: "pr1" } }),
    "POST /ensemble/combiners/fit": () => ({ body: { ok: true, job_id: "fit1", variant: "spatial_gate_trial" } }),
    "POST /ensemble/train/preview": (form) => ({ body: {
      ok: true, mode: form.mode, member_names: ["member_199", "member_200"], count: 2, array: { tasks: 2, max_parallel: 2 },
      command: ["python", "scripts/train_ensemble.py"], command_text: `python scripts/train_ensemble.py --member-spec '${form.member_spec}'`,
      base_seed: null, star_prior: true,
    } }),
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
  useSelection.getState().clear();
});
afterEach(() => { act(() => resetConfirm()); queryClient.clear(); });

let lastLocation = "";
function Spy() { const l = useLocation(); lastLocation = l.pathname + l.search; return null; }

const show = (el: ReactElement, url = "/models/starfull/leaderboard") => render(
  <QueryClientProvider client={queryClient}>
    <MemoryRouter initialEntries={[url]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Routes><Route path="/models/:mode/*" element={<>{el}<Spy /></>} /><Route path="*" element={<Spy />} /></Routes>
    </MemoryRouter>
  </QueryClientProvider>,
);

async function answer(title: RegExp | string, button: string) {
  const dlg = await screen.findByRole("alertdialog", { name: title });
  fireEvent.click(within(dlg).getByRole("button", { name: button }));
}

const legendLabels = (root: ParentNode = document) =>
  [...root.querySelectorAll(".plot-legend__label")].map((n) => n.textContent);

/* ── Leaderboard ───────────────────────────────────────────────────────── */
describe("leaderboard", () => {
  const tableRows = () => [...document.querySelectorAll("table[aria-label='Production vs references'] tbody tr")]
    .map((tr) => [...tr.querySelectorAll("td")].map((td) => td.textContent));
  const load = async () => (await import("./tabs/Leaderboard")).default;

  it("states the production gate against the best member and the plain mean, with no tiles", async () => {
    const Leaderboard = await load();
    show(<Leaderboard />);
    const line = await waitFor(() => {
      const el = document.querySelector(".ui-summary");
      if (!el || !/61\.00/.test(el.textContent ?? "")) throw new Error("not yet");
      return el as HTMLElement;
    });
    expect(line.textContent).toBe("Production gate 61.00 dB integrated PSNR, +1.00 dB over the best member (#196) and +2.00 dB over the plain mean");
    expect(document.querySelector(".ui-kpi, .ens-kpis")).toBeNull();
  });

  it("compares gate, plain mean and the best member on one metric per band, and says there is no real benchmark yet", async () => {
    const Leaderboard = await load();
    show(<Leaderboard />);
    await waitFor(() => expect(tableRows()).toHaveLength(3));
    const head = [...document.querySelectorAll("table[aria-label='Production vs references'] thead th")].map((th) => th.textContent);
    expect(head).toEqual(["", "∫PSNR [dB]", "∫VIS [dB]", "∫Y [dB]", "∫J [dB]", "∫H [dB]"]);
    expect(tableRows()[0]).toEqual(["Production gate", "61.00", "56.00", "65.00", "62.00", "61.00"]);
    expect(await screen.findByText(/no real benchmark yet: no Sky › Compare run has scored production/)).toBeTruthy();
  });

  it("adds the real holes and R̃ from the newest Sky › Compare run of THIS production, with its date and a link", async () => {
    const agg = (vis: number, y: number, j: number, h: number, r: number) => ({
      n_tiles: 3, per_band: { VIS: { hole_pct: vis }, Y_E: { hole_pct: y }, J_E: { hole_pct: j }, H_E: { hole_pct: h } },
      summary: { hole_pct_mean: (vis + y + j + h) / 4, hole_pct_max: Math.max(vis, y, j, h), median_R: r } });
    routes["GET /api/experiments"] = () => ({ body: { experiments: [{ id: "e1", created: "2026-09-27T20:38:56Z", label: "poster galaxy", tiles: ["t/a"],
      summary: { production: agg(17, 18, 22, 20, 1.18), mean: agg(20, 25, 26, 30, 1.2) } }] } });
    routes["GET /api/experiments/e1"] = () => ({ body: { id: "e1", fingerprints: { production: "p1", mean: "m1" } } });
    routes["GET /api/models"] = () => ({ body: { regime: "starfull", models: [{ spec: "production", fingerprint: "p1" }, { spec: "mean", fingerprint: "m1" }] } });
    const Leaderboard = await load();
    show(<Leaderboard />);
    await waitFor(() => expect(tableRows()[0]).toEqual(["Production gate", "61.00", "56.00", "65.00", "62.00", "61.00", "J 22", "1.18"]));
    expect(tableRows()[1].slice(-2)).toEqual(["H 30", "1.20"]);
    expect(tableRows()[2].slice(-2)).toEqual(["—", "—"]);
    const link = screen.getByRole("link", { name: "“poster galaxy”" });
    expect(link.getAttribute("href")).toBe("/sky/compare?exp=e1");
  });

  it("shows each failing check once in the status line, with its fix", async () => {
    const Leaderboard = await load();
    show(<Leaderboard />);
    const status = await screen.findByRole("list", { name: "Staleness" });
    const item = within(status).getByText("Production gate vs members").closest("li") as HTMLElement;
    expect(item.textContent).toContain("Fitted for 30 members; 31 are active");
    expect(within(item).getByRole("link", { name: "Combiner" }).getAttribute("href")).toBe("/models/starfull/combiner");
    expect(within(status).queryByText("Evaluation vs members")).toBeNull();
  });

  it("offers each fix once and runs it only after confirm", async () => {
    routes["GET /ensemble/overview.json?mode=starfull"] = () => ({ body: { ...OVERVIEW, checks: [
      { id: "eval-members", ok: false, tone: "warn", title: "Evaluation vs members", detail: "Evaluated 2 members; 3 are active now.", action: "evaluate" },
      { id: "eval-gate", ok: false, tone: "warn", title: "Evaluation vs production gate", detail: "The production gate changed.", action: "evaluate" },
      { id: "knee", ok: false, tone: "warn", title: "Knee curves", detail: "The cubes changed.", action: "knee" },
    ] } });
    routes["POST /ensemble/knee-psnr"] = () => ({ body: { job_id: "kn1" } });
    const Leaderboard = await load();
    show(<Leaderboard />);
    const status = await screen.findByRole("list", { name: "Staleness" });
    expect(within(status).getAllByRole("button", { name: "Evaluate 100 fields…" })).toHaveLength(1);
    fireEvent.click(within(status).getByRole("button", { name: "Knee PSNR" }));
    await answer("Recompute PSNR vs knee?", "Compute");
    await waitFor(() => expect(posts("/ensemble/knee-psnr")[0]?.form).toMatchObject({ mode: "starfull" }));
    // The Evaluate dialog is the confirm: Cancel starts nothing.
    fireEvent.click(within(status).getByRole("button", { name: "Evaluate 100 fields…" }));
    let dlg = await screen.findByRole("dialog", { name: /Evaluate the starfull ensemble/ });
    fireEvent.click(within(dlg).getByRole("button", { name: "Cancel" }));
    expect(posts("/ensemble/evaluate")).toHaveLength(0);
    // It chooses the field count (the old Overview's ?n=) and says when scores stop being comparable.
    routes["POST /ensemble/evaluate"] = () => ({ body: { job_id: "ev1" } });
    fireEvent.click(within(status).getByRole("button", { name: "Evaluate 100 fields…" }));
    dlg = await screen.findByRole("dialog", { name: /Evaluate the starfull ensemble/ });
    fireEvent.change(within(dlg).getByRole("spinbutton", { name: /Test fields/ }), { target: { value: "250" } });
    expect(within(dlg).getByText(/not comparable/)).toBeTruthy();
    fireEvent.click(within(dlg).getByRole("button", { name: "Evaluate 250 fields" }));
    await waitFor(() => expect(posts("/ensemble/evaluate")[0]?.form).toMatchObject({ mode: "starfull", num_images: "250", force: "0" }));
  });

  it("says all current in one quiet line when every check passes", async () => {
    routes["GET /ensemble/overview.json?mode=starfull"] = () => ({ body: { ...OVERVIEW, checks: OVERVIEW.checks.map((c) => ({ ...c, ok: true, tone: "good" })) } });
    routes["GET /ensemble/members.json?mode=starfull"] = () => ({ body: { ...MEMBERS, members: MEMBERS.members.filter((m) => !m.timeout) } });
    const Leaderboard = await load();
    show(<Leaderboard />);
    expect(await screen.findByText("All current")).toBeTruthy();
    expect(screen.queryByRole("list", { name: "Staleness" })).toBeNull();
  });

  it("puts the TIMEOUT members in ONE alert line with Continue", async () => {
    const Leaderboard = await load();
    show(<Leaderboard />);
    const line = await screen.findByText("1 member stopped short of the target steps: 178");
    expect(line.closest(".ui-callout")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Continue" }).getAttribute("href"))
      .toBe("/models/starfull/train?mode=continue&members=member_178");
  });

  it("logs the evaluation and the leaderboard on Notebook › Log, appending nothing itself", async () => {
    const Leaderboard = await load();
    show(<Leaderboard />, "/models/starfull/leaderboard?range=1,100");
    await screen.findByRole("grid", { name: "Knee-integrated PSNR leaderboard" });
    fireEvent.click(screen.getByRole("button", { name: "Log to notebook" }));
    await waitFor(() => expect(lastLocation.startsWith("/notebook/log?")).toBe(true));
    const q = new URLSearchParams(lastLocation.split("?")[1]);
    expect(q.get("from")).toBe("Models › Leaderboard");
    expect(q.get("entry")).toContain("**Ensemble evaluation · starfull**");
    expect(q.get("entry")).toContain("∫ over 1–100 e⁻ (a sub-range of 0.1–10k e⁻, uniform in log knee)");
    expect(posts("/api/tracking/log")).toHaveLength(0);
  });

  it("ranks the leaderboard over the integration range, hides Test VIS and Test 4b, and colours the legend", async () => {
    const Leaderboard = await load();
    const view = show(<Leaderboard />);
    const table = await screen.findByRole("grid", { name: "Knee-integrated PSNR leaderboard" });
    expect(within(within(table).getAllByRole("row")[1]).getByText("production gate")).toBeTruthy();
    const heads = within(table).getAllByRole("columnheader").map((h) => h.textContent);
    expect(heads.some((h) => /Test VIS|Test 4b/.test(h ?? ""))).toBe(false);
    expect(legendLabels()).toEqual(["production gate", "plain mean", "10 e⁻", "multi-knee → 1 image"]);
    view.unmount();
    show(<Leaderboard />, "/models/starfull/leaderboard?range=0.1,10");
    const narrow = await screen.findByRole("grid", { name: "Knee-integrated PSNR leaderboard" });
    const first = within(narrow).getAllByRole("row")[1];
    expect(within(first).getByText("#178")).toBeTruthy();
    expect(within(first).getByText("▲3")).toBeTruthy();
  });
});

/* ── Members ───────────────────────────────────────────────────────────── */
describe("members", () => {
  const load = async () => (await import("./tabs/Members")).default;

  it("shows the roster: Member, Status, Steps, Recipe, ∫PSNR and the gate share bar; the rest in Columns", async () => {
    const Members = await load();
    show(<Members />, "/models/starfull/members");
    expect(await screen.findByText("multi ×6 → 10")).toBeTruthy();
    const grid = screen.getByRole("grid", { name: "starfull members" });
    const heads = within(grid).getAllByRole("columnheader").map((h) => h.textContent?.trim()).filter(Boolean);
    expect(heads).toEqual(["Member", "Status", "Steps", "Recipe", "∫PSNR", "Gate share"]);
    expect(screen.getAllByText("TIMEOUT").length).toBeGreaterThan(0);
    fireEvent.click(screen.getByText("#196"));
    expect(useInspector.getState().current).toEqual({ kind: "member", id: "member_196" });
  });

  it("keeps the selection toolbar visible, archives only after confirm(), typed beyond 3", async () => {
    const confirmSpy = vi.spyOn(window, "confirm");
    const Members = await load();
    show(<Members />, "/models/starfull/members");
    await screen.findByText("multi ×6 → 10");
    const tools = screen.getByRole("group", { name: "Selection" });
    expect((within(tools).getByRole("button", { name: "Continue" }) as HTMLButtonElement).disabled).toBe(true);
    act(() => useSelection.getState().select("member", ["member_178"]));
    const archive = within(tools).getByRole("button", { name: "Archive" });
    await waitFor(() => expect((archive as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(archive);
    await answer("Archive member_178?", "Cancel");
    expect(posts("/ensemble/archive-member")).toHaveLength(0);
    fireEvent.click(archive);
    await answer("Archive member_178?", "Archive");
    await waitFor(() => expect(posts("/ensemble/archive-member")[0]?.form).toEqual({ member: "member_178" }));
    expect(confirmSpy).not.toHaveBeenCalled();
  });

  it("sends the selection to Images and to Train", async () => {
    const Members = await load();
    show(<Members />, "/models/starfull/members");
    await screen.findByText("multi ×6 → 10");
    act(() => useSelection.getState().select("member", ["member_196", "member_178"]));
    const tools = screen.getByRole("group", { name: "Selection" });
    fireEvent.click(within(tools).getByRole("button", { name: "Images" }));
    await waitFor(() => expect(lastLocation).toBe("/models/starfull/images?sel=196,178"));
  });

  it("shows a core specialist's peak share and marks what the gate does not read", async () => {
    routes["GET /ensemble/members.json?mode=starfull"] = () => ({ body: { ...MEMBERS, members: [
      member(195, { gate_usage: { VIS: 0.0002, Y_E: 0.0001, J_E: 0.0001, H_E: 0.0001 },
        gate_usage_peak: { value: 0.48, band: "VIS", bin: "core" }, used_by_gate: true }),
      member(169, { gate_usage: { VIS: 0.00001, Y_E: 0.00001, J_E: 0.00001, H_E: 0.00001 }, gate_usage_peak: 0.0031, used_by_gate: false }),
    ] } });
    const Members = await load();
    show(<Members />, "/models/starfull/members");
    // The cell carries one value (the peak and where), so it never clips; the full reading is its tooltip.
    const cell = await screen.findByText("48% VIS cores");
    expect(cell.closest("[aria-label]")?.getAttribute("aria-label")).toBe("Gate share 48% VIS cores");
    expect(screen.queryByText("<0.1% mean · 48% peak (VIS cores)")).toBeNull();
    const row169 = screen.getByText("#169").closest("tr") as HTMLElement;
    expect(within(row169).getByText("not read")).toBeTruthy();
  });

  it("offers the Pull banner only when finished members are waiting on FASRC", async () => {
    const Members = await load();
    const view = show(<Members />, "/models/starfull/members");
    await screen.findByText("multi ×6 → 10");
    // member_195 finished on FASRC and is not local
    expect(await screen.findByText(/1 new member finished on FASRC: 195/)).toBeTruthy();
    view.unmount();
    routes["GET /ensemble/training-jobs.json"] = () => ({ body: { jobs: [{ ...TRAIN_JOB, member_names: ["member_196"] }] } });
    queryClient.clear();
    show(<Members />, "/models/starfull/members");
    await screen.findByText("multi ×6 → 10");
    expect(screen.queryByText(/finished on FASRC/)).toBeNull();
    expect(screen.getByRole("button", { name: "Pull from FASRC…" })).toBeTruthy();
  });

  it("lists the archived members with Restore behind a confirm", async () => {
    routes["POST /ensemble/restore-member"] = () => ({ body: { job_id: "rs1" } });
    const Members = await load();
    show(<Members />, "/models/starfull/members?view=archived");
    const grid = await screen.findByRole("grid", { name: "Archived members" });
    expect(within(grid).getByText("#09")).toBeTruthy();
    fireEvent.click(within(grid).getByRole("button", { name: "Restore" }));
    await answer("Restore member_09?", "Restore");
    await waitFor(() => expect(posts("/ensemble/restore-member")[0]?.form).toEqual({ member: "member_09" }));
  });
});

/* ── Train ─────────────────────────────────────────────────────────────── */
describe("train", () => {
  const load = async () => (await import("./tabs/Train")).default;

  it("previews names + command, repeats the last batch, names the regime on the submit and needs FASRC", async () => {
    const Train = await load();
    show(<Train />, "/models/starfull/train");
    expect(await screen.findByText("member_199")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Repeat last batch" }));
    await waitFor(() => {
      const spec = JSON.parse(posts("/ensemble/train/preview").at(-1)!.form.member_spec);
      expect(spec[0]).toMatchObject({ asinh_knees: [0.1, 1, 10, 100, 1000, 10000], output_knee: 10, knee_loss: "balanced" });
      expect(spec[1]).toMatchObject({ asinh_knee: 3000 });
    });
    const submit = await screen.findByRole("button", { name: "Submit 2 STARFULL members to SLURM" });
    expect((submit as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText(/FASRC offline/)).toBeTruthy();
  });

  it("shows the forward-model values System › Config owns, read-only, and never sends them", async () => {
    const Train = await load();
    show(<Train />, "/models/starfull/train");
    expect(await screen.findByText("PSF warp α max")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Edit in System › Config" }).getAttribute("href")).toBe("/system/config");
    await waitFor(() => expect(posts("/ensemble/train/preview").length).toBeGreaterThan(0));
    expect(posts("/ensemble/train/preview").at(-1)!.form).not.toHaveProperty("psf_warp_prob");
  });

  it("links back to System › Config with the count of knobs changed from their defaults", async () => {
    const used = (keys: string[]) => Object.fromEntries(keys.map((k) => [k, ["ensemble_train"]]));
    routes["GET /api/config"] = () => ({ body: {
      config: { psf_warp_prob: 0.5, psf_warp_alpha_max: 5, psf_warp_sigma: 3, saturation_mask_prob: 0.25, lr_peak: 0.0002, lr_final: 1e-6, lr_warmup_steps: 2000, plateau_lr_enabled: 0 },
      defaults: { psf_warp_prob: 1, psf_warp_alpha_max: 5, psf_warp_sigma: 3, saturation_mask_prob: 0.5, lr_peak: 0.0002, lr_final: 1e-6, lr_warmup_steps: 1000, plateau_lr_enabled: 0 },
      used_by: used(["psf_warp_prob", "psf_warp_alpha_max", "psf_warp_sigma", "saturation_mask_prob", "lr_peak", "lr_final", "lr_warmup_steps", "plateau_lr_enabled"]),
    } });
    const Train = await load();
    show(<Train />, "/models/starfull/train");
    const forward = await screen.findByRole("link", { name: "2 knobs changed · Edit" });
    expect(forward.getAttribute("href")).toBe("/system/config");
    expect(screen.getByRole("link", { name: "1 knob changed · Edit" }).getAttribute("href")).toBe("/system/config");
    expect(screen.queryByRole("link", { name: "Edit in System › Config" })).toBeNull();
  });

  it("re-reads the members when an in-app link opens Train for other members", async () => {
    const Train = await load();
    show(<><Train /><Link to="/models/starfull/train?mode=continue&members=member_179">next</Link></>,
      "/models/starfull/train?mode=continue&members=member_178");
    await waitFor(() => expect(posts("/ensemble/train/preview").at(-1)?.form.members).toBe("member_178"));
    fireEvent.click(screen.getByRole("link", { name: "next" }));
    await waitFor(() => expect(posts("/ensemble/train/preview").at(-1)?.form.members).toBe("member_179"));
  });

  it("starts from the last finished batch's resources and continues TIMEOUT members up to their target", async () => {
    const Train = await load();
    const view = show(<Train />, "/models/starfull/train");
    const cpus = await screen.findByRole("spinbutton", { name: "CPUs / model" });
    await waitFor(() => expect((cpus as HTMLInputElement).value).toBe("4"));
    view.unmount();
    show(<Train />, "/models/starfull/train?mode=continue&members=member_178");
    const upTo = await screen.findByRole("spinbutton", { name: "Up to step" });
    expect((upTo as HTMLInputElement).value).toBe("70000");
    await waitFor(() => expect(posts("/ensemble/train/preview").at(-1)?.form).toMatchObject({ continue_basis: "target", target_steps: "70000" }));
  });

  it("trains starless members from the starless workspace, and the regime is an explicit field", async () => {
    routes["GET /ensemble/members.json?mode=starless"] = () => ({ body: { ...MEMBERS, regime: "starless", members: [] } });
    const Train = await load();
    show(<Train />, "/models/starless/train");
    expect(await screen.findByText("member_199")).toBeTruthy();
    expect(posts("/ensemble/train/preview").at(-1)!.form.starless).toBe("1");
    expect(await screen.findByRole("button", { name: "Submit 1 STARLESS member to SLURM" })).toBeTruthy();
    fireEvent.click(screen.getByRole("radio", { name: "starfull" }));
    await waitFor(() => expect(posts("/ensemble/train/preview").at(-1)!.form).not.toHaveProperty("starless"));
    expect(screen.getByRole("button", { name: "Submit 1 STARFULL member to SLURM" })).toBeTruthy();
  });
});

/* ── Combiner ──────────────────────────────────────────────────────────── */
describe("combiner", () => {
  const load = async () => (await import("./tabs/Combiner")).default;
  const EXP = (id: string, created: string, label: string, specs: Record<string, number[]>) => ({
    id, created, label, tiles: ["tile/a"],
    summary: Object.fromEntries(Object.entries(specs).map(([spec, [vis, y, j, h]]) => [spec, {
      n_tiles: 1, per_band: { VIS: { hole_pct: vis }, Y_E: { hole_pct: y }, J_E: { hole_pct: j }, H_E: { hole_pct: h } },
      summary: { hole_pct_mean: (vis + y + j + h) / 4, hole_pct_max: Math.max(vis, y, j, h), median_R: 1, pct_R_lt_0p8: 0 } }])),
  });

  it("never tells production's gate share to 'compare it with production' (loading or unreadable members.json)", async () => {
    routes["GET /ensemble/members.json?mode=starfull"] = () => ({ status: 500, body: { ok: false, error: "boom" } });
    const Combiner = await load();
    show(<Combiner />, "/models/starfull/combiner");
    await screen.findByRole("grid", { name: "Combiner variants" });
    expect(screen.queryByText(/compare it with production to measure one/)).toBeNull();          // while members.json loads
    // (the query layer retries a failed GET before it reports the error)
    expect(await screen.findByText(/The production gate's share per member is not readable/, undefined, { timeout: 4000 })).toBeTruthy();
    expect(screen.queryByText(/compare it with production to measure one/)).toBeNull();
  });

  it("shows production and the variants of the current membership; the rest behind History", async () => {
    const Combiner = await load();
    show(<Combiner />, "/models/starfull/combiner");
    const grid = await screen.findByRole("grid", { name: "Combiner variants" });
    expect(within(grid).getByText("combiner")).toBeTruthy();
    expect(within(grid).getByText("p2")).toBeTruthy();
    expect(within(grid).queryByText("26m")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "History 1" }));
    await waitFor(() => expect(within(grid).getByText("26m")).toBeTruthy());
    expect(lastLocation).toContain("history=1");
  });

  it("scores the real holes per band from ONE Sky › Compare run, the header linking to it", async () => {
    routes["GET /api/experiments"] = () => ({ body: { experiments: [
      EXP("e-new", "2026-09-27T02:00:00Z", "cached tile ra0273", { mean: [14, 26, 26, 33], "gate:p2": [9, 9, 9, 9] }),
      EXP("e-prod", "2026-09-26T02:00:00Z", "NEXUS core", { production: [19, 13, 19, 30], "gate:p2": [12, 11, 12, 20] }),
    ] } });
    routes["GET /api/experiments/e-prod"] = () => ({ body: { id: "e-prod", fingerprints: { production: "p1", "gate:p2": "g2" } } });
    routes["GET /api/experiments/e-new"] = () => ({ body: { id: "e-new", fingerprints: { mean: "m1", "gate:p2": "g2" } } });
    routes["GET /api/models"] = () => ({ body: { regime: "starfull", models: [
      { spec: "production", fingerprint: "p1" }, { spec: "mean", fingerprint: "m1" }, { spec: "gate:p2", fingerprint: "g2" }] } });
    const Combiner = await load();
    show(<Combiner />, "/models/starfull/combiner");
    const grid = await screen.findByRole("grid", { name: "Combiner variants" });
    const picker = screen.getByRole("combobox", { name: "Real holes from (Sky › Compare run)" }) as HTMLSelectElement;
    await waitFor(() => expect(picker.value).toBe("e-prod"));
    expect(within(grid).getByRole("link", { name: "Real holes [%]" }).getAttribute("href")).toBe("/sky/compare?exp=e-prod");
    const prodRow = within(grid).getByText("combiner").closest("tr") as HTMLElement;
    await waitFor(() => expect(prodRow.querySelector(".mdl-holes")).toBeTruthy());
    const holes = prodRow.querySelector(".mdl-holes") as HTMLElement;
    expect(holes.textContent).toBe("19 · 13 · 19 · 30");
    expect(holes.querySelector("[data-worst]")?.textContent).toBe("30");
    fireEvent.change(picker, { target: { value: "e-new" } });
    await waitFor(() => expect(lastLocation).toContain("bench=e-new"));
    await waitFor(() => expect((within(grid).getByText("combiner").closest("tr") as HTMLElement).querySelector(".mdl-holes")).toBeNull());
  });

  it("never shows real holes the run scored for an earlier fit as current", async () => {
    routes["GET /api/experiments"] = () => ({ body: { experiments: [
      EXP("e-prod", "2026-09-26T02:00:00Z", "NEXUS core", { production: [15, 19, 20, 18] }),
    ] } });
    // The run scored production d9…; production is now 11…
    routes["GET /api/experiments/e-prod"] = () => ({ body: { id: "e-prod", fingerprints: { production: "d9efa5b5" } } });
    routes["GET /api/models"] = () => ({ body: { regime: "starfull", models: [{ spec: "production", fingerprint: "1188a841" }] } });
    const Combiner = await load();
    show(<Combiner />, "/models/starfull/combiner");
    const grid = await screen.findByRole("grid", { name: "Combiner variants" });
    const prodRow = within(grid).getByText("combiner").closest("tr") as HTMLElement;
    await waitFor(() => expect(within(prodRow).getByText("earlier fit")).toBeTruthy());
    expect(prodRow.querySelector(".mdl-holes")).toBeNull();
    expect(prodRow.textContent).not.toContain("15");
    expect(screen.getByText(/That run scored an earlier production gate/)).toBeTruthy();
  });

  it("never lists the legacy RBF, never offers it in a compare, and shows no report card without a report", async () => {
    routes["GET /ensemble/combiners.json?mode=starfull"] = () => ({ body: { ...COMBINERS, variants: [
      ...COMBINERS.variants, variant("raw_incremental_minmeanmax_rbf", { kind: "rbf", spec: "rbf", production: false, backup: false }),
    ] } });
    routes["POST /ensemble/combiners/compare"] = () => ({ body: { ok: true, job_id: "cmp1" } });
    const Combiner = await load();
    show(<Combiner />, "/models/starfull/combiner?history=1");
    const grid = await screen.findByRole("grid", { name: "Combiner variants" });
    expect(within(grid).queryByText(/rbf/i)).toBeNull();
    expect(screen.queryByText("Compare report")).toBeNull();
    expect(screen.queryByRole("combobox", { name: "Compare report" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Compare…" }));
    const dlg = await screen.findByRole("dialog", { name: "Compare gate variants" });
    expect(within(dlg).queryByText(/RBF/)).toBeNull();
    fireEvent.click(within(dlg).getByRole("button", { name: /^Compare \d/ }));
    await answer("Run the compare?", "Compare");
    await waitFor(() => expect(posts("/ensemble/combiners/compare")).toHaveLength(1));
    expect(posts("/ensemble/combiners/compare")[0].form).not.toHaveProperty("include_rbf");
  });

  it("says how many members production runs", async () => {
    routes["GET /ensemble/combiners.json?mode=starfull"] = () => ({ body: { ...COMBINERS, variants: [
      variant("spatial_gate_combiner", { n_members: 30, n_reads: 20, pruned: true, fit: { steps: 2000, prune_threshold: 0.005 } }),
    ] } });
    const Combiner = await load();
    show(<Combiner />, "/models/starfull/combiner");
    expect(await screen.findByText(/^Runs 20 of 30 members: those with ≥ 0\.5% of the gate's weight somewhere/)).toBeTruthy();
  });

  it("lists the production gate's share per member and opens Members with the picked ones", async () => {
    routes["GET /ensemble/members.json?mode=starfull"] = () => ({ body: { ...MEMBERS, members: [
      member(196, { gate_usage: { VIS: 0.6, Y_E: 0.6, J_E: 0.6, H_E: 0.6 }, used_by_gate: true }),
      member(178, { gate_usage: { VIS: 0, Y_E: 0, J_E: 0, H_E: 0 }, gate_usage_peak: { value: 0, band: "VIS" }, used_by_gate: false }),
    ] } });
    const Combiner = await load();
    show(<Combiner />, "/models/starfull/combiner");
    const list = await screen.findByRole("grid", { name: "Gate share per member" });
    await waitFor(() => expect(within(list).getAllByRole("row")).toHaveLength(3));
    expect(within(within(list).getAllByRole("row")[1]).getByText("#196")).toBeTruthy();   // largest share first
    // An unread member with no weight says "0%" and "not read" once, with no band tag and no Read column.
    const row178 = within(list).getByText("#178").closest("tr") as HTMLElement;
    expect(within(row178).getByText("0%")).toBeTruthy();
    expect(within(row178).getAllByText("not read")).toHaveLength(1);
    expect(within(row178).queryByText(/VIS/)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Select the 1 not read" }));
    fireEvent.click(screen.getByRole("button", { name: "Open in Members with this selection (1)" }));
    await waitFor(() => expect(lastLocation).toBe("/models/starfull/members"));
    expect(useSelection.getState().get("member")).toEqual(["member_178"]);
  });

  it("fits a named variant and refuses the production name and reserved names", async () => {
    const Combiner = await load();
    show(<Combiner />, "/models/starfull/combiner");
    await screen.findByRole("grid", { name: "Combiner variants" });
    fireEvent.click(screen.getByRole("button", { name: "Fit variant…" }));
    const name = await screen.findByRole("textbox", { name: /Variant name/ });
    fireEvent.change(name, { target: { value: "combiner" } });
    expect(await screen.findByText("that is the production gate")).toBeTruthy();
    fireEvent.change(name, { target: { value: "comparison.json" } });
    expect(await screen.findByText("reserved for compare reports / eval sidecars")).toBeTruthy();
    fireEvent.change(name, { target: { value: "trial" } });
    fireEvent.click(screen.getByRole("button", { name: "Fit variant" }));
    await answer(/spatial_gate_trial/, "Fit");
    await waitFor(() => expect(posts("/ensemble/combiners/fit")[0]?.form).toMatchObject({
      out_name: "trial", mode: "starfull", mix_space: "linear", loss_knees: "all", compare_after: "1" }));
  });

  it("does not rank a held-out loss on another scale, and leaves it out of the loss curves", async () => {
    routes["GET /ensemble/combiners.json?mode=starfull"] = () => ({ body: { ...COMBINERS, variants: [
      variant("spatial_gate_combiner", { fit: { loss: "per-field relative asinh MSE" } }),
      variant("spatial_gate_v1", { fit: { loss: "band-weighted asinh squared error" }, selected: { step: 3000, loss: 0.1775 } }),
    ] } });
    const Combiner = await load();
    show(<Combiner />, "/models/starfull/combiner");
    expect(await screen.findByText("other scale")).toBeTruthy();
    expect(screen.getByText("0.6500")).toBeTruthy();
    expect(screen.getByText(/Not drawn: v1, whose loss is on another scale than production's/)).toBeTruthy();
  });

  it("promotes a variant fitted for other members only with typed confirmation + force", async () => {
    const Combiner = await load();
    show(<Combiner />, "/models/starfull/combiner?history=1");
    const grid = await screen.findByRole("grid", { name: "Combiner variants" });
    const row = within(grid).getByText("26m").closest("tr") as HTMLElement;
    fireEvent.pointerDown(within(row).getByRole("button", { name: /actions/ }), { button: 0 });
    fireEvent.click(await screen.findByRole("menuitem", { name: /Promote/ }));
    const dlg = await screen.findByRole("alertdialog", { name: /Promote spatial_gate_26m/ });
    expect(within(dlg).getByText(/WARNING/)).toBeTruthy();
    fireEvent.change(within(dlg).getByRole("textbox"), { target: { value: "promote" } });
    fireEvent.click(within(dlg).getByRole("button", { name: "Promote" }));
    await waitFor(() => expect(posts("/ensemble/combiners/promote")[0]?.form).toEqual({
      mode: "starfull", variant: "spatial_gate_26m", force: "1" }));
  });
});

/* ── Diagnostics ───────────────────────────────────────────────────────── */
describe("diagnostics", () => {
  const load = async () => (await import("./tabs/Diagnostics")).default;
  const EVALS = {
    n_fields: 5, n_members: 3, subset: "test", guides: { lr_scale: 0.1, vis_fwhm: 0.16, theta_min: 0.05 },
    members: [{ label: "196·psnr", loss: "l2" }, { label: "178·psnr", loss: "l1" }, { label: "170·psnr", loss: "l2" }],
    ps: { theta: [0.1, 0.2, 0.4, 2], r: [0.95, 0.9, 0.8, 0.99], r_lr: [0.5, 0.4, 0.3, 0.9],
      r_members: [[0.9, 0.8, 0.7, 0.99], [0.8, 0.7, 0.6, 0.99], [0.85, 0.75, 0.65, 0.99]] },
  };
  const block = { edges: [-1, 0, 1], hist: [[1, 2], [3, 4]], med_std: [-0.5, 0.5], med_err: [-0.4, 0.6], n_fields: 3 };
  const SPREAD = {
    ...EVALS,
    std_err: { ...block, models: { ensemble_mean: block, spatial_gate: block, raw_incremental_minmeanmax_rbf: block } },
    calibration: {
      z_edges: [-2, 0, 2], pdf: [0.2, 0.89], field_std: [1, 2, 4], field_rmse: [10, 30, 40],
      stats: { cover1: 0.8893, cover2: 0.9531, cover3: 0.977, sigma_z: 0.45 },
    },
  };

  it("gives the members ONE legend entry per colour facet, and a band switch that really switches the data", async () => {
    routes["GET /ensemble/evals.json?mode=starfull"] = () => ({ body: EVALS });
    const Diagnostics = await load();
    const view = show(<Diagnostics />, "/models/starfull/diagnostics");
    await screen.findByLabelText("Cross-correlation r(k)");
    expect(legendLabels()).toEqual(["members 3", "LR (bicubic)", "plain mean"]);
    const bands = screen.getByRole("radiogroup", { name: "Band" });
    expect(within(bands).getAllByRole("radio").map((r) => r.textContent)).toEqual(["VIS", "Y", "J", "H"]);
    expect(screen.getByText(/· VIS band$/)).toBeTruthy();
    // Y: its own payload (?band=Y_E) with its own guides and caption
    routes["GET /ensemble/evals.json?mode=starfull&band=Y_E"] = () => ({ body: {
      ...EVALS, band: "Y_E", stale: false, guides: { ...EVALS.guides, band: "Y_E", psf_fwhm: 0.4 } } });
    fireEvent.click(within(bands).getByRole("radio", { name: "Y" }));
    await waitFor(() => expect(lastLocation).toContain("band=Y_E"));
    expect(await screen.findByText(/· Y band$/)).toBeTruthy();
    expect(gets("/ensemble/evals.json?mode=starfull&band=Y_E")).toHaveLength(1);
    expect(posts("/ensemble/evals/bands")).toHaveLength(0);                 // opening computes nothing
    view.unmount();
    show(<Diagnostics />, "/models/starfull/diagnostics?color=loss");
    await screen.findByLabelText("Cross-correlation r(k)");
    expect(legendLabels()).toEqual(["l1", "l2", "LR (bicubic)", "plain mean"]);
  });

  it("offers to compute a band that is not measured yet, and flags one measured for an earlier evaluation", async () => {
    routes["POST /ensemble/evals/bands"] = () => ({ body: { ok: true, job_id: "bands1" } });
    const Diagnostics = await load();
    const view = show(<Diagnostics />, "/models/starfull/diagnostics?band=J_E");
    const empty = await screen.findByText("The J diagnostics are not computed yet");
    expect(empty).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Compute Y, J and H…" }));
    await answer("Compute the Y, J and H diagnostics?", "Compute");
    await waitFor(() => expect(posts("/ensemble/evals/bands")[0]?.form).toEqual({ mode: "starfull" }));
    view.unmount();
    routes["GET /ensemble/evals.json?mode=starfull&band=H_E"] = () => ({ body: { ...EVALS, band: "H_E", stale: true } });
    show(<Diagnostics />, "/models/starfull/diagnostics?band=H_E");
    expect(await screen.findByText("These H diagnostics belong to an earlier evaluation")).toBeTruthy();
    expect(screen.getByText(/· H band$/)).toBeTruthy();
  });

  it("answers the spread in one sentence and a coverage table (the old calibration URL lands here)", async () => {
    routes["GET /ensemble/evals.json?mode=starfull"] = () => ({ body: SPREAD });
    const Diagnostics = await load();
    show(<Diagnostics />, "/models/starfull/diagnostics?d=calibration");
    const line = await waitFor(() => {
      const el = document.querySelector(".ui-summary");
      if (!el) throw new Error("not yet");
      return el as HTMLElement;
    });
    expect(line.textContent).toBe("Cross-member σ is not an error bar: per test field, the RMSE is ≈10× the mean σ (median over 3 fields).");
    const table = screen.getByRole("table", { name: "Coverage of |z|" });
    expect([...table.querySelectorAll("tbody tr")].map((tr) => [...tr.querySelectorAll("td")].map((td) => td.textContent))).toEqual([
      ["|z| < 1", "88.9%", "68.3%"], ["|z| < 2", "95.3%", "95.4%"], ["|z| < 3", "97.7%", "99.7%"],
    ]);
    expect(screen.getByLabelText("Disagreement vs error")).toBeTruthy();
    expect(screen.getByLabelText("z-score distribution")).toBeTruthy();
    expect(document.querySelector(".ui-kpi, .ens-kpis")).toBeNull();
  });

  it("lands the deleted axes view on spread and never offers the RBF", async () => {
    routes["GET /ensemble/evals.json?mode=starfull"] = () => ({ body: SPREAD });
    const Diagnostics = await load();
    show(<Diagnostics />, "/models/starfull/diagnostics?d=axes");
    await screen.findByLabelText("Disagreement vs error");
    expect(screen.queryByRole("radio", { name: /Combiner axes/ })).toBeNull();
    const chips = within(screen.getByRole("group", { name: "Error of" })).getAllByRole("button").map((c) => c.textContent);
    expect(chips).toEqual(["plain mean", "production gate"]);
  });

  it("shows the legacy real field beside its synthetic twin, naming both ensembles", async () => {
    routes["GET /ensemble/evals.json?mode=starfull"] = () => ({ body: { ...EVALS, n_members: 30, ps: { ...EVALS.ps, r_pairs: [[0.9, 0.8, 0.7, 0.6]], r_cross: [0.9, 0.8, 0.7, 0.6] } } });
    routes["GET /api/inference/field.json"] = () => ({ body: { field: { field_id: "f1", ra: 269.3, dec: 65.1, count: 100,
      member_labels: Array.from({ length: 30 }, (_, i) => `${i}·psnr`), run_member_labels: Array.from({ length: 14 }, (_, i) => `${i}·psnr`), member_scope: "gate" } } });
    routes["GET /api/inference/diagnostics.json"] = () => ({ body: { diagnostics: { version: 2, member_labels: ["a", "b", "c"],
      model_power: { k: [0.5, 1, 2], r_pairs: [[0.9, 0.8, 0.7]], r_cross: [0.9, 0.8, 0.7], pixel_scale_arcsec: 0.05 },
      std_brightness: { x_edges: [0, 1, 2], y_edges: [0, 1, 2], counts: [[1, 2], [3, 4]], x_label: "brightness", y_label: "σ" } } } });
    const Diagnostics = await load();
    show(<Diagnostics />, "/models/starfull/diagnostics?d=real-field&fd=cross");
    expect(await screen.findByLabelText("Cross-correlation r(d), field-f1-rd")).toBeTruthy();
    expect(screen.getByLabelText("Cross-correlation r(d), synthetic-rd")).toBeTruthy();
    expect(await screen.findByText(/^14 real vs 30 synthetic members/)).toBeTruthy();
    const bands = screen.getByRole("radiogroup", { name: "Band" });                // same place as every section
    expect(within(bands).getByRole("radio", { name: "J" }).hasAttribute("disabled")).toBe(true);
    expect(within(bands).getByRole("radio", { name: "J" }).getAttribute("title")).toMatch(/legacy real field has no Y, J or H/);
    expect(screen.queryByText(/occupancy/i)).toBeNull();
    expect(posts("/inference/refresh-combiners")).toHaveLength(0);          // opening never recomputes
  });

  it("draws the SR → HR recovery of the synthetic stamps from the evaluation rows", async () => {
    routes["GET /api/evaluation/runs"] = () => ({ body: { rows: [
      { id: "a", grade: "syn-lens", ok: "True", psnr_lr_hr: "30", psnr_sr_hr: "33", flux_ratio_sr_over_lr: "0.95", viewer_id: "a" },
      { id: "b", grade: "syn-gal", ok: "True", psnr_lr_hr: "31", psnr_sr_hr: "30", flux_ratio_sr_over_lr: "1.02", viewer_id: "b" },
      { id: "c", grade: "A", ok: "True" },
    ] } });
    const Diagnostics = await load();
    show(<Diagnostics />, "/models/starfull/diagnostics?d=recovery");
    expect(await screen.findByLabelText("SR vs HR recovery of the synthetic stamps")).toBeTruthy();
    expect(document.querySelector(".ui-summary")?.textContent)
      .toBe("SR is closer to the HR truth than LR on 1 of 2 synthetic stamps; Syn lens median 30.00 → 33.00 dB; Syn gal median 31.00 → 30.00 dB.");
    expect(gets("/api/evaluation/angular-power-spectrum")).toHaveLength(0);   // the figure renders only on request
    expect(await screen.findByText("Not measured yet")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Measure…" })).toBeTruthy();
    expect(posts("/api/evaluation/angular-power-spectrum")).toHaveLength(0);
  });

  it("draws the angular power spectrum interactively, per band, from the cached curves", async () => {
    const curves = (t: number) => ({ theta: [0.05, 0.1, 0.5, 2], T: [t, 0.9, 1, 1], T_lo: [t - 0.1, 0.8, 0.9, 0.9], T_hi: [t + 0.1, 1, 1.1, 1.1],
      r: [0.3, 0.7, 0.95, 0.99], r_lo: [0.2, 0.6, 0.9, 0.98], r_hi: [0.4, 0.8, 0.99, 1], count: [10, 10, 10, 10] });
    routes["GET /api/evaluation/angular-power-spectrum.json"] = () => ({ body: {
      subset: "validate", n_fields: 10, field_n: 510, pixel_scale: 0.05, lr_scale: 0.1, theta_max: 4.2,
      band_names: ["VIS", "Y_E", "J_E", "H_E"],
      bands: Object.fromEntries(["VIS", "Y_E", "J_E", "H_E"].map((b, i) => [b, { psf_fwhm: [0.16, 0.4, 0.45, 0.48][i], linear: curves(0.5), asinh: curves(0.6) }])),
    } });
    routes["POST /api/evaluation/angular-power-spectrum"] = () => ({ body: { ok: true, rendered: true } });
    const Diagnostics = await load();
    show(<Diagnostics />, "/models/starfull/diagnostics?d=recovery&band=Y_E");
    expect(await screen.findByLabelText("Transfer function, Y")).toBeTruthy();
    expect(screen.getByLabelText("Cross-correlation, Y")).toBeTruthy();
    expect(screen.getByText(/^10 validate fields of 510×510 px at 0\.05″ · per-field median, shaded 16–84% of fields/)).toBeTruthy();
    expect(within(screen.getByRole("radiogroup", { name: "Measured on" })).getAllByRole("radio").map((r) => r.textContent)).toEqual(["asinh", "linear"]);
    fireEvent.click(screen.getByRole("button", { name: "Re-measure…" }));
    await answer("Measure the angular power spectrum?", "Measure");
    await waitFor(() => expect(posts("/api/evaluation/angular-power-spectrum")).toHaveLength(1));
  });
});

/* ── Images ────────────────────────────────────────────────────────────── */
describe("images", () => {
  const load = async () => (await import("./tabs/Images")).default;
  const META = { count: 100, member_labels: ["178·psnr", "196·psnr", "197·psnr"] };

  it("puts the viewer first with ONE member picker beside it (no popover in the tab strip)", async () => {
    routes["GET /viewer/meta/ensemble?mode=starfull"] = () => ({ body: META });
    const Images = await load();
    show(<Images />, "/models/starfull/images?sel=196,178");
    const panel = await screen.findByRole("complementary", { name: "Members" });
    const viewer = screen.getByTestId("viewer");
    expect(viewer.dataset.collection).toBe("ensemble");
    expect(viewer.compareDocumentPosition(panel) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(within(panel).getByRole("status").textContent).toBe("Movie over 2 members: #196, #178");
    const picker = within(panel).getByRole("group", { name: "Members in the movie" });
    await waitFor(() => expect(within(picker).getAllByRole("button")).toHaveLength(3));
    fireEvent.change(within(panel).getByRole("searchbox", { name: "Find members" }), { target: { value: "#178" } });
    await waitFor(() => expect(within(picker).getAllByRole("button")).toHaveLength(1));
    fireEvent.click(within(picker).getByRole("button"));
    await waitFor(() => expect(lastLocation).toContain("sel=196"));
    expect(lastLocation).not.toContain("178");
    expect(screen.getAllByRole("complementary")).toHaveLength(1);
    expect(within(panel).getByRole("link", { name: "Leaderboard" }).getAttribute("href")).toBe("/models/starfull/leaderboard");
  });

  it("gives no member-table verdict while members.json loads", async () => {
    routes["GET /viewer/meta/ensemble?mode=starfull"] = () => ({ body: META });
    routes["GET /ensemble/members.json?mode=starfull"] = () => ({ status: 500, body: { ok: false, error: "boom" } });
    const Images = await load();
    show(<Images />, "/models/starfull/images");
    const panel = await screen.findByRole("complementary", { name: "Members" });
    const picker = within(panel).getByRole("group", { name: "Members in the movie" });
    await waitFor(() => expect(within(picker).getAllByRole("button")).toHaveLength(3));
    expect(within(picker).queryByText("not in the members table")).toBeNull();          // while it loads
    expect((within(panel).getByRole("button", { name: /Top 5 by ∫PSNR/ }) as HTMLButtonElement).disabled).toBe(true);
    // (the query layer retries a failed GET before it reports the error)
    await waitFor(() => expect(within(picker).getAllByText("members table not readable")).toHaveLength(3), { timeout: 4000 });
    expect(within(picker).queryByText("not in the members table")).toBeNull();
  });

  it("gives the test fields' PSNR vs HR in the caption", async () => {
    routes["GET /viewer/meta/ensemble?mode=starfull"] = () => ({ body: META });
    const Images = await load();
    show(<Images />);
    expect(await screen.findByText(/^100 starfull test fields · PSNR vs HR \(VIS, knee 100 e⁻\): production 59\.24 dB, plain mean 58\.38 dB, best member 58\.94 dB · production ∫PSNR 60\.97 dB/)).toBeTruthy();
  });

  it("shows the synthetic stamps of one group with its PSNR vs HR caption, a row opening the stamp", async () => {
    routes["GET /api/evaluation/runs"] = () => ({ body: { rows: [
      { id: "l1", grade: "syn-lens", ok: "True", psnr_lr_hr: "30", psnr_sr_hr: "33", viewer_id: "syn-lens_0001_0", state: "stale", n_members: 9 },
      { id: "l2", grade: "syn-lens", ok: "True", psnr_lr_hr: "31", psnr_sr_hr: "32", viewer_id: "syn-lens_0002_0", state: "stale", n_members: 9 },
      { id: "g1", grade: "syn-gal", ok: "True", psnr_lr_hr: "40", psnr_sr_hr: "41", viewer_id: "syn-gal_0001_0", state: "current", n_members: 20 },
    ] } });
    const Images = await load();
    show(<Images />, "/models/starfull/images?set=stamps&g=syn-lens");
    expect(await screen.findByRole("button", { name: "Syn lens 2" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Syn gal 1" })).toBeTruthy();
    const viewer = await screen.findByTestId("viewer");
    expect(viewer.dataset.collection).toBe("evaluation");
    expect(viewerProps.at(-1)).toMatchObject({ initialId: "syn-lens_0001_0" });   // the largest SR gain first
    expect(screen.getByText(/^Median PSNR vs HR: LR 30\.50 → SR 32\.50 dB, a gain of \+2\.00 dB · SR is closer to the truth on every stamp/)).toBeTruthy();
    expect(screen.getByText("All predate the current model.")).toBeTruthy();
    fireEvent.click(screen.getByText("0002_0"));
    await waitFor(() => expect(lastLocation).toContain("id=syn-lens_0002_0"));
  });

  it("shows the production SR over the local records (the Records SR tier), per split, with a link back to Records", async () => {
    routes["GET /api/sky/sr-status"] = () => ({ body: { records: true, checkpoint: true, can_generate: true, subsets: ["test", "validate"],
      sr: { test: 12, validate: 0 }, splits: { test: { count: 20, present: true, sr: { state: "stale", reasons: ["model changed"], count: 12 } },
        validate: { count: 10, present: true, sr: { state: "missing", reasons: [], count: 0 } } } } });
    const Images = await load();
    show(<Images />, "/models/starfull/images?set=records&id=test:3");
    expect(await screen.findByRole("radio", { name: "Records 12" })).toBeTruthy();
    const viewer = await screen.findByTestId("viewer");
    expect(viewer.dataset.collection).toBe("sky");
    expect(viewerProps.at(-1)).toMatchObject({ params: { subset: "test" }, initialId: "test:3", tiers: ["dirty", "sr", "hr"] });
    // one split with SR: no split switch
    expect(screen.queryByRole("radiogroup", { name: "Records split" })).toBeNull();
    expect(screen.getByText(/12 of 20 test records carry the production SR/)).toBeTruthy();
    expect(screen.getByText(/predates the current production model \(model changed\)/)).toBeTruthy();
    expect(screen.getByRole("link", { name: "Synthetic › Records" }).getAttribute("href")).toBe("/synthetic/records");
  });

  it("generates SR over the local records only after the confirm", async () => {
    routes["GET /viewer/meta/ensemble?mode=starfull"] = () => ({ body: META });
    routes["GET /api/sky/sr-status"] = () => ({ body: { records: true, checkpoint: true, can_generate: true, subsets: ["test", "validate"],
      sr: { test: 0 }, splits: { test: { count: 10, present: true, sr: { state: "missing", reasons: [], count: 0 } },
        validate: { count: 10, present: true, sr: { state: "missing", reasons: [], count: 0 } } } } });
    routes["POST /api/sky/generate-sr"] = () => ({ body: { job_id: "gs1" } });
    const Images = await load();
    show(<Images />);
    const trigger = await screen.findByRole("button", { name: "Generate SR over local records…" });
    await waitFor(() => expect((screen.getByRole("button", { name: "Generate SR over local records…" }) as HTMLButtonElement).disabled).toBe(false));
    expect(trigger).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Generate SR over local records…" }));
    fireEvent.click(await screen.findByRole("button", { name: "Generate…" }));
    await answer(/Generate the production SR for test \+ validate/, "Cancel");
    expect(posts("/api/sky/generate-sr")).toHaveLength(0);
    fireEvent.click(screen.getByRole("button", { name: "Generate SR over local records…" }));
    fireEvent.click(await screen.findByRole("button", { name: "Generate…" }));
    await answer(/Generate the production SR/, "Generate");
    await waitFor(() => expect(posts("/api/sky/generate-sr")[0]?.form).toEqual({ subsets: "test,validate", overwrite: "0" }));
  });
});

/* ── inspector and back-trace ──────────────────────────────────────────── */
describe("member inspector", () => {
  it("shows the member's recipe, curves and actions", async () => {
    const { default: MemberInspector } = await import("./MemberInspector");
    show(<MemberInspector id="196" />);
    await screen.findByText("Training curves");
    expect(document.querySelector(".mdl-insp__title")?.textContent).toBe("#196");
    expect(screen.getAllByText("multi ×6 → 10").length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "Continue" })).toBeTruthy();
  });
});

describe("pixel back-trace", () => {
  const stamp = (hr: number, std: number, err: number) => ({
    field: 7, y: 20, x: 20, center: 20, sr_is_combiner: true, model_kind: "spatial_gate",
    hr: btoa(String.fromCharCode(...new Uint8Array(new Float32Array(4).buffer))),
    sr: btoa(String.fromCharCode(...new Uint8Array(new Float32Array(4).buffer))),
    std: btoa(String.fromCharCode(...new Uint8Array(new Float32Array(4).buffer))),
    hr_val: hr, sr_val: hr, std_val: std, err_val: err, bright_asinh: 0,
  });

  it("gives each row a knee at its traced pixel's level and scrolls the trace into view once it is in", async () => {
    routes["GET /ensemble/pixel-trace.json?mode=starfull&diag=std_err&i=32&j=32"] = () => ({ body: {
      diag: "std_err", i: 32, j: 32, half: 1, size: 2, bands: ["VIS"], stretch: 1,
      stamps: [stamp(3.1, 1.2, 1.6), stamp(11, 1.61, 1.5)],
    } });
    routes["GET /viewer/meta/ensemble?mode=starfull"] = () => ({ body: { color: { default_asinh: 100 } } });
    const scrolled = vi.fn();
    const had = Element.prototype.scrollIntoView;
    Element.prototype.scrollIntoView = scrolled;
    onTestFinished(() => { Element.prototype.scrollIntoView = had; });
    const { PixelTrace } = await import("./PixelTrace");
    show(<PixelTrace mode="starfull" pick={{ diag: "std_err", i: 32, j: 32 }} cellLabel="σ 1.2–1.61 e⁻" targetLabel="HR" onClose={() => {}} />);
    expect(await screen.findByText("knee 3.1 e⁻")).toBeTruthy();
    expect(screen.getByText("knee 11 e⁻")).toBeTruthy();
    await waitFor(() => expect(scrolled).toHaveBeenCalledWith(expect.objectContaining({ block: "nearest" })));
  });
});
