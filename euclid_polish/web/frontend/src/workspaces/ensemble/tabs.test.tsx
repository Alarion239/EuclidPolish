/* The Ensemble tabs against a mocked Flask: headline numbers and their
 * sources, the joined members table (multi-knee text, TIMEOUT, archive only
 * after confirm()), the knee leaderboard, the combiner registry's fit /
 * promote guards, the train preview and the member inspector. */
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import { useState, type ReactElement } from "react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, onTestFinished, vi } from "vitest";
import { useJobsStore } from "../../api/jobs";
import { queryClient } from "../../api/query";
import { useInspector } from "../../state/inspector";
import { useSelection } from "../../state/selection";
import { resetConfirm } from "../../ui";
import { TabAsideSlot } from "./aside";
import type { CombinersPayload, KneeModel, MemberDetail, MemberRow, Overview as OverviewData } from "./api";

vi.mock("../../viewer", () => ({
  ImageViewer: ({ onReady }: { onReady?: (api: unknown) => void }) => {
    onReady?.({ getState: () => ({ tiers: ["lr", "sr"] }), setTiers: vi.fn(), setMorphMembers: vi.fn(), reload: vi.fn() });
    return <div data-testid="viewer" />;
  },
  renderCubeImageData: vi.fn(),
}));

type Reply = { status?: number; body: unknown };
let routes: Record<string, (form: Record<string, string>) => Reply>;
let calls: { url: string; method: string; form: Record<string, string> }[];
const posts = (u: string) => calls.filter((c) => c.method === "POST" && c.url === u);

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

beforeEach(() => {
  calls = [];
  routes = {
    "GET /ensemble/overview.json?mode=starfull": () => ({ body: OVERVIEW }),
    "GET /ensemble/members.json?mode=starfull": () => ({ body: MEMBERS }),
    "GET /ensemble/knee-psnr.json?mode=starfull": () => ({ body: KNEE }),
    "GET /ensemble/combiners.json?mode=starfull": () => ({ body: COMBINERS }),
    "GET /api/experiments": () => ({ body: { experiments: [] } }),
    "GET /ensemble/training-jobs.json": () => ({ body: { jobs: [{
      jobid: "48107719", state: "COMPLETED", submitted_at: "2026-09-24T04:54:24Z", mode: "add",
      member_names: ["member_195", "member_196"], req_time_limit: "3:00:00", req_memory: "32G", req_cpus: 4,
      params: { mode: "add", count: 2, steps: "70000", member_spec: JSON.stringify([
        { loss: "l2", bootstrap: 0.7, asinh_knees: [0.1, 1, 10, 100, 1000, 10000], knee_loss: "balanced", output_knee: 10, num_res_blocks: 32, icnr: true },
        { loss: "l2", bootstrap: 0.7, asinh_knee: 3000, num_res_blocks: 32, icnr: true }]) },
    }] } }),
    "GET /api/fasrc/steps/status": () => ({ body: { ssh_connected: false, steps: [{ step_id: "ensemble_train",
      defaults: { partition: "gpu", n_cpus: 4, n_gpus: 1, memory: "32G", time_limit: "48:00:00" }, fixed_gpus: 1 }] } }),
    "GET /api/fasrc/status": () => ({ body: { ssh_connected: false, last_error: "timed out" } }),
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

const show = (el: ReactElement, url = "/ensemble/starfull/overview") => render(
  <QueryClientProvider client={queryClient}>
    <MemoryRouter initialEntries={[url]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Routes><Route path="/ensemble/:mode/*" element={<>{el}<Spy /></>} /></Routes>
    </MemoryRouter>
  </QueryClientProvider>,
);

async function answer(title: RegExp | string, button: string) {
  const dlg = await screen.findByRole("alertdialog", { name: title });
  fireEvent.click(within(dlg).getByRole("button", { name: button }));
}


const legendLabels = (root: ParentNode = document) =>
  [...root.querySelectorAll(".plot-legend__label")].map((n) => n.textContent);

describe("overview", () => {
  it("headlines the production gate's knee-integrated and test PSNR with their deltas", async () => {
    const { default: Overview } = await import("./tabs/Overview");
    show(<Overview />);
    expect(await screen.findByText("60.97")).toBeTruthy();
    const knee = screen.getByText("∫PSNR · production gate").closest(".ui-kpi") as HTMLElement;
    expect(within(knee).getByText("+1.02 dB vs member 196")).toBeTruthy();
    expect(within(knee).getByText("+1.87 dB vs plain mean")).toBeTruthy();
    const test = screen.getByText("Test PSNR · production gate").closest(".ui-kpi") as HTMLElement;
    expect(within(test).getByText("59.24")).toBeTruthy();
    expect(within(test).getByText("+0.29 dB vs best member")).toBeTruthy();
    expect(screen.getAllByText("Production gate vs members")).toHaveLength(2);   // banner + checks list
    expect(screen.getByRole("alert").textContent).toMatch(/Fitted for 30 members; 31 are active/);
  });

  it("never reports members complete while members.json is loading or failed", async () => {
    routes["GET /ensemble/members.json?mode=starfull"] = () => ({ status: 404, body: { ok: false, error: "registry unreadable" } });
    const { default: Overview } = await import("./tabs/Overview");
    show(<Overview />);
    expect(await screen.findByText("60.97")).toBeTruthy();
    expect(await screen.findByText("registry unreadable")).toBeTruthy();
    expect(screen.queryByText("Every member reached its target steps.")).toBeNull();
  });

  it("lists TIMEOUT members with a continue link and evaluates only after confirm", async () => {
    const { default: Overview } = await import("./tabs/Overview");
    show(<Overview />);
    await screen.findByText("60.97");
    expect(await screen.findByText("#178 52k/70k")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Continue them…" }).getAttribute("href"))
      .toBe("/ensemble/starfull/train?mode=continue&members=member_178");
    fireEvent.click(screen.getByRole("checkbox", { name: "force" }));
    fireEvent.click(screen.getByRole("button", { name: "Evaluate" }));
    await answer(/Evaluate the starfull ensemble/, "Cancel");
    expect(posts("/ensemble/evaluate")).toHaveLength(0);
    fireEvent.click(screen.getByRole("button", { name: "Evaluate" }));
    await answer(/Evaluate the starfull ensemble/, "Evaluate");
    await waitFor(() => expect(posts("/ensemble/evaluate")[0]?.form).toMatchObject({ mode: "starfull", num_images: "100", force: "1" }));
  });
});

describe("members", () => {
  it("shows multi-knee members as multi ×N, flags TIMEOUT, opens the inspector", async () => {
    const { default: Members } = await import("./tabs/Members");
    show(<Members />, "/ensemble/starfull/members");
    expect(await screen.findByText("multi ×6 → 10")).toBeTruthy();
    expect(screen.getByText("multi ×6 heads")).toBeTruthy();
    expect(screen.queryByText("100e")).toBeNull();
    expect(screen.getAllByText("TIMEOUT").length).toBeGreaterThan(0);
    fireEvent.click(screen.getByText("#196"));
    expect(useInspector.getState().current).toEqual({ kind: "member", id: "member_196" });
    expect(screen.getByText("Archived members")).toBeTruthy();
  });

  it("names both per-member test PSNRs and advertises a filter that works", async () => {
    routes["GET /ensemble/members.json?mode=starfull"] = () => ({ body: {
      ...MEMBERS, vis_psnr: { metric: "vis_asinh", knee_e: 100, n_scored: 100 },
      members: MEMBERS.members.map((m, i) => ({ ...m, vis_psnr: 58.1 + i })) } });
    const { default: Members } = await import("./tabs/Members");
    show(<Members />, "/ensemble/starfull/members");
    await screen.findByText("multi ×6 → 10");
    expect(screen.getByText("Test VIS")).toBeTruthy();
    expect(screen.getByText("Test 4b")).toBeTruthy();
    expect(screen.queryByText("Test")).toBeNull();
    expect(screen.getByText("58.100")).toBeTruthy();
    const filter = screen.getByPlaceholderText(/knee_mean>59/);
    expect((filter as HTMLInputElement).placeholder).not.toMatch(/∫/);
    fireEvent.change(filter, { target: { value: "knee_mean>59" } });
    await waitFor(() => expect(screen.queryByText("#178")).toBeNull());
    expect(screen.getByText("#196")).toBeTruthy();
  });

  it("archives the selection only after the kit confirm()", async () => {
    const confirmSpy = vi.spyOn(window, "confirm");
    const { default: Members } = await import("./tabs/Members");
    show(<Members />, "/ensemble/starfull/members");
    await screen.findByText("multi ×6 → 10");
    act(() => useSelection.getState().select("member", ["member_178"]));
    const archive = await screen.findByRole("button", { name: "Archive" });
    fireEvent.click(archive);
    await answer("Archive member_178?", "Cancel");
    expect(posts("/ensemble/archive-member")).toHaveLength(0);
    fireEvent.click(archive);
    await answer("Archive member_178?", "Archive");
    await waitFor(() => expect(posts("/ensemble/archive-member")[0]?.form).toEqual({ member: "member_178" }));
    expect(confirmSpy).not.toHaveBeenCalled();
  });

  it("sends the selection to the disagreement movie", async () => {
    const { default: Members } = await import("./tabs/Members");
    show(<Members />, "/ensemble/starfull/members");
    await screen.findByText("multi ×6 → 10");
    act(() => useSelection.getState().select("member", ["member_196", "member_178"]));
    fireEvent.click(await screen.findByRole("button", { name: "Disagreement" }));
    await waitFor(() => expect(lastLocation).toBe("/ensemble/starfull/disagreement?sel=196,178"));
  });
});

describe("members gate use", () => {
  it("shows the gate's share as the mean over bands (the VIS weight alone hid NISP use), per band on hover", async () => {
    routes["GET /ensemble/members.json?mode=starfull"] = () => ({ body: { ...MEMBERS, members: [
      member(190, { gate_usage: { VIS: 0.00018, Y_E: 0.374, J_E: 0.366, H_E: 0.329 } }),
    ] } });
    const { default: Members } = await import("./tabs/Members");
    show(<Members />, "/ensemble/starfull/members");
    const cell = await screen.findByLabelText(/^Gate use 26\.7% \(mean over bands\)/);
    expect(cell.getAttribute("aria-label")).toBe("Gate use 26.7% (mean over bands): VIS 0.0% · Y 37.4% · J 36.6% · H 32.9%");
    expect(cell.querySelectorAll(".ens-gate__bars i")).toHaveLength(4);
  });
});

describe("disagreement", () => {
  const META = { count: 100, member_labels: ["178·psnr", "196·psnr", "197·psnr"] };
  it("puts the viewer first and ONE compact, searchable member panel under it", async () => {
    routes["GET /viewer/meta/ensemble?mode=starfull"] = () => ({ body: META });
    const { default: Disagreement } = await import("./tabs/Disagreement");
    show(<Disagreement />, "/ensemble/starfull/disagreement?sel=196,178");
    const panel = await screen.findByRole("region", { name: "Members" });
    const viewer = screen.getByTestId("viewer");
    // The viewer comes before the member panel in the page (nothing above its bar).
    expect(viewer.compareDocumentPosition(panel) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(within(panel).getByRole("status").textContent).toBe("Movie over 2 members: #196, #178");
    const picker = within(panel).getByRole("group", { name: "Members in the movie" });
    await waitFor(() => expect(within(picker).getAllByRole("button")).toHaveLength(3));
    expect(within(picker).getAllByRole("button").filter((b) => b.getAttribute("aria-pressed") === "true")).toHaveLength(2);
    // Search by number or knee description.
    fireEvent.change(within(panel).getByRole("searchbox", { name: "Find members" }), { target: { value: "multi" } });
    await waitFor(() => expect(within(picker).getAllByRole("button").map((b) => b.textContent?.slice(0, 4))).toEqual(["#196", "#197"]));
    fireEvent.change(within(panel).getByRole("searchbox", { name: "Find members" }), { target: { value: "#178" } });
    await waitFor(() => expect(within(picker).getAllByRole("button")).toHaveLength(1));
    fireEvent.click(within(picker).getByRole("button"));
    await waitFor(() => expect(lastLocation).toContain("sel=196"));
    expect(lastLocation).not.toContain("178");
    fireEvent.click(within(panel).getByRole("button", { name: "Clear selection" }));
    await waitFor(() => expect(within(panel).getByRole("status").textContent).toMatch(/^Pick members/));
  });

  it("offers the members from a menu in the tab strip, reachable without scrolling", async () => {
    routes["GET /viewer/meta/ensemble?mode=starfull"] = () => ({ body: META });
    const { default: Disagreement } = await import("./tabs/Disagreement");
    function WithStrip() {
      const [slot, setSlot] = useState<HTMLElement | null>(null);
      return (
        <TabAsideSlot.Provider value={slot}>
          <nav aria-label="Tab strip"><span ref={setSlot} /></nav>
          <Disagreement />
        </TabAsideSlot.Provider>
      );
    }
    show(<WithStrip />, "/ensemble/starfull/disagreement?sel=196");
    const strip = screen.getByRole("navigation", { name: "Tab strip" });
    const trigger = await within(strip).findByRole("button", { name: "1 member" });
    fireEvent.click(trigger);
    const menu = await screen.findByRole("dialog", { name: "Pick members" });
    expect(menu.textContent).toContain("Showing member #196");
    const picker = within(menu).getByRole("group", { name: "Members in the movie" });
    await waitFor(() => expect(within(picker).getAllByRole("button")).toHaveLength(3));
    fireEvent.click(within(picker).getAllByRole("button").find((b) => b.textContent?.startsWith("#178"))!);
    await waitFor(() => expect(lastLocation).toMatch(/sel=196(,|%2C)178/));
    expect(await within(strip).findByRole("button", { name: "2 members" })).toBeTruthy();
    fireEvent.click(within(menu).getByRole("button", { name: "Clear selection" }));
    await waitFor(() => expect(within(strip).getByRole("button", { name: "Pick members" })).toBeTruthy());
  });
});

describe("knee", () => {
  it("ranks the leaderboard over the selected range and draws no 100 e⁻ line", async () => {
    const { default: Knee } = await import("./tabs/Knee");
    show(<Knee />, "/ensemble/starfull/knee");
    const table = await screen.findByRole("grid", { name: "Knee-integrated PSNR leaderboard" });
    const first = within(table).getAllByRole("row")[1];
    expect(within(first).getByText("production gate")).toBeTruthy();
    expect(screen.queryByText(/100 e⁻ reference/)).toBeNull();
  });

  it("builds the legend from the active colouring", async () => {
    const { default: Knee } = await import("./tabs/Knee");
    const view = show(<Knee />, "/ensemble/starfull/knee");
    await screen.findByRole("grid", { name: "Knee-integrated PSNR leaderboard" });
    expect(legendLabels()).toEqual(["production gate", "plain mean", "10 e⁻", "multi-knee → 1 image"]);
    view.unmount();
    show(<Knee />, "/ensemble/starfull/knee?color=loss");
    await screen.findByRole("grid", { name: "Knee-integrated PSNR leaderboard" });
    expect(legendLabels()).toEqual(["production gate", "plain mean", "L1"]);
  });

  it("re-ranks when the integration range narrows (URL ?range=)", async () => {
    const { default: Knee } = await import("./tabs/Knee");
    show(<Knee />, "/ensemble/starfull/knee?range=0.1,10");
    const table = await screen.findByRole("grid", { name: "Knee-integrated PSNR leaderboard" });
    const first = within(table).getAllByRole("row")[1];
    expect(within(first).getByText("#178")).toBeTruthy();       // strongest at low knees
    expect(within(first).getByText("▲3")).toBeTruthy();
  });
});

describe("combiners", () => {
  it("fits a named variant and refuses the production name", async () => {
    const { default: Combiners } = await import("./tabs/Combiners");
    show(<Combiners />, "/ensemble/starfull/combiners");
    fireEvent.click(await screen.findByRole("button", { name: "Fit variant…" }));
    const name = await screen.findByRole("textbox", { name: /Variant name/ });
    fireEvent.change(name, { target: { value: "combiner" } });
    expect(await screen.findByText("that is the production gate")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Fit variant" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(name, { target: { value: "trial" } });
    fireEvent.click(screen.getByRole("button", { name: "Fit variant" }));
    await answer(/spatial_gate_trial/, "Fit");
    await waitFor(() => expect(posts("/ensemble/combiners/fit")[0]?.form).toMatchObject({
      out_name: "trial", mode: "starfull", mix_space: "linear", loss_knees: "all", compare_after: "1" }));
  });

  it("refuses names reserved for compare reports and eval sidecars, with no bogus time estimate", async () => {
    const { default: Combiners } = await import("./tabs/Combiners");
    show(<Combiners />, "/ensemble/starfull/combiners");
    fireEvent.click(await screen.findByRole("button", { name: "Fit variant…" }));
    const name = await screen.findByRole("textbox", { name: /Variant name/ });
    for (const bad of ["comparisons", "comparison.json", "combiner_evals"]) {
      fireEvent.change(name, { target: { value: bad } });
      expect(await screen.findByText("reserved for compare reports / eval sidecars")).toBeTruthy();
    }
    fireEvent.change(name, { target: { value: "trial" } });
    fireEvent.click(screen.getByRole("button", { name: "Fit variant" }));
    const dlg = await screen.findByRole("alertdialog", { name: /spatial_gate_trial/ });
    expect(dlg.textContent).not.toMatch(/per 100 steps/);
    expect(dlg.textContent).toMatch(/2,000 steps/);
    fireEvent.click(within(dlg).getByRole("button", { name: "Cancel" }));
  });

  it("does not rank a held-out loss measured on another scale against production's", async () => {
    routes["GET /ensemble/combiners.json?mode=starfull"] = () => ({ body: { ...COMBINERS, variants: [
      variant("spatial_gate_combiner", { fit: { loss: "per-field relative asinh MSE" } }),
      variant("spatial_gate_v1", { fit: { loss: "band-weighted asinh squared error" }, selected: { step: 3000, loss: 0.1775 } }),
    ] } });
    const { default: Combiners } = await import("./tabs/Combiners");
    show(<Combiners />, "/ensemble/starfull/combiners");
    const odd = await screen.findByText("0.1775 ≠");
    expect(odd.className).toContain("ens-faint");
    expect(screen.getByText("0.6500")).toBeTruthy();
  });

  it("promotes a variant fitted for other members only with typed confirmation + force", async () => {
    const { default: Combiners } = await import("./tabs/Combiners");
    show(<Combiners />, "/ensemble/starfull/combiners");
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

describe("diagnostics", () => {
  const EVALS = {
    n_fields: 5, n_members: 3, subset: "test",
    members: [{ label: "196·psnr", loss: "l2" }, { label: "178·psnr", loss: "l1" }, { label: "170·psnr", loss: "l2" }],
    ps: { theta: [0.1, 0.2, 0.4], r: [0.95, 0.9, 0.8], r_lr: [0.5, 0.4, 0.3],
      r_members: [[0.9, 0.8, 0.7], [0.8, 0.7, 0.6], [0.85, 0.75, 0.65]] },
  };
  it("gives the members ONE legend entry per colour facet", async () => {
    routes["GET /ensemble/evals.json?mode=starfull"] = () => ({ body: EVALS });
    const { default: Diagnostics } = await import("./tabs/Diagnostics");
    const view = show(<Diagnostics />, "/ensemble/starfull/diagnostics");
    await screen.findByLabelText("Cross-correlation r(k)");
    expect(legendLabels()).toEqual(["members", "LR (bicubic)", "plain mean"]);
    view.unmount();
    show(<Diagnostics />, "/ensemble/starfull/diagnostics?color=loss");
    await screen.findByLabelText("Cross-correlation r(k)");
    expect(legendLabels()).toEqual(["l1", "l2", "LR (bicubic)", "plain mean"]);
  });
});

describe("train", () => {
  it("previews names + command, clones the last batch with its multi-knee rows, and needs FASRC to submit", async () => {
    const { default: Train } = await import("./tabs/Train");
    show(<Train />, "/ensemble/starfull/train");
    expect(await screen.findByText("member_199")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Repeat last batch" }));
    await waitFor(() => {
      const spec = JSON.parse(posts("/ensemble/train/preview").at(-1)!.form.member_spec);
      expect(spec[0]).toMatchObject({ asinh_knees: [0.1, 1, 10, 100, 1000, 10000], output_knee: 10, knee_loss: "balanced" });
      expect(spec[1]).toMatchObject({ asinh_knee: 3000 });
    });
    expect((screen.getByRole("button", { name: "Submit to SLURM" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText(/FASRC offline/)).toBeTruthy();
  });
});

describe("train resources", () => {
  it("starts from the last finished batch's resources (not the step's 4 CPUs / 48 h) and names them in the confirm", async () => {
    routes["GET /ensemble/training-jobs.json"] = () => ({ body: { jobs: [{
      jobid: "48107719", state: "COMPLETED", submitted_at: "2026-09-24T04:54:24Z", mode: "add", member_names: ["member_195"],
      req_time_limit: "3:00:00", req_memory: "32G", req_cpus: 16, params: { mode: "add", count: 1, steps: "70000" },
    }] } });
    const { default: Train } = await import("./tabs/Train");
    show(<Train />, "/ensemble/starfull/train");
    const cpus = await screen.findByRole("spinbutton", { name: "CPUs / model" });
    expect((cpus as HTMLInputElement).value).toBe("16");
    expect(screen.getByDisplayValue("3:00:00")).toBeTruthy();
    expect(screen.getByText("As job 48107719 (the last finished new batch)")).toBeTruthy();
  });

  it("continues TIMEOUT members up to their target, not a fixed +20k", async () => {
    const { default: Train } = await import("./tabs/Train");
    show(<Train />, "/ensemble/starfull/train?mode=continue&members=member_178");
    const upTo = await screen.findByRole("spinbutton", { name: "Up to step" });
    expect((upTo as HTMLInputElement).value).toBe("70000");
    await waitFor(() => expect(posts("/ensemble/train/preview").at(-1)?.form).toMatchObject({ continue_basis: "target", target_steps: "70000" }));
  });
});

describe("train regime", () => {
  it("submits starless members from the starless workspace, with no per-row regime knob", async () => {
    routes["GET /ensemble/members.json?mode=starless"] = () => ({ body: { ...MEMBERS, regime: "starless", members: [] } });
    const { default: Train } = await import("./tabs/Train");
    show(<Train />, "/ensemble/starless/train");
    expect(await screen.findByText("member_199")).toBeTruthy();
    const last = posts("/ensemble/train/preview").at(-1)!.form;
    expect(last.starless).toBe("1");
    expect(JSON.parse(last.member_spec)[0]).not.toHaveProperty("starless");
    expect(screen.queryByText("Regime")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Repeat last batch" }));
    await waitFor(() => expect(JSON.parse(posts("/ensemble/train/preview").at(-1)!.form.member_spec)[0])
      .toMatchObject({ output_knee: 10 }));
    expect(posts("/ensemble/train/preview").at(-1)!.form.starless).toBe("1");
  });

  it("sends no regime flag from the starfull workspace", async () => {
    const { default: Train } = await import("./tabs/Train");
    show(<Train />, "/ensemble/starfull/train");
    expect(await screen.findByText("member_199")).toBeTruthy();
    expect(posts("/ensemble/train/preview").at(-1)!.form).not.toHaveProperty("starless");
  });
});

describe("member inspector", () => {
  it("shows the member's recipe, curves and actions", async () => {
    const { default: MemberInspector } = await import("./MemberInspector");
    show(<MemberInspector id="196" />);
    await screen.findByText("Training curves");
    expect(document.querySelector(".ens-insp__title")?.textContent).toBe("#196");
    expect(screen.getAllByText("multi ×6 → 10").length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "Continue" })).toBeTruthy();
    expect(screen.getByText("Training curves")).toBeTruthy();
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
    // the Display panel's knee is one click away
    fireEvent.click(screen.getByRole("radio", { name: "Display panel knee" }));
    expect(screen.getAllByText("knee 100 e⁻")).toHaveLength(2);
  });
});
