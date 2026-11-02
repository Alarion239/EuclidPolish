/* The Models tabs against a mocked Flask: Universal (summary, confirmed
 * score / pull / pull again / try pull, the plots' legends, the models table
 * with stale marks, real fields from the newest run of THIS model, the Sky ›
 * Compare links, make production), Train (preview, the count in the button,
 * no star-regime flag, resources and their past-run advice) and the model
 * inspector. */
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import type { ReactElement } from "react";
import { Link, MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useJobsStore } from "../../api/jobs";
import { queryClient } from "../../api/query";
import { useInspector } from "../../state/inspector";
import { useSelection } from "../../state/selection";
import { resetConfirm } from "../../ui";
import type { MemberDetail, MemberRow } from "./api";
import { payload as universalPayload, universalProduction } from "./testing/universalFixture";
import MemberInspector from "./MemberInspector";
import Train from "./tabs/Train";
import Universal from "./tabs/Universal";

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
let calls: { url: string; method: string; form: Record<string, string>; json?: unknown }[];
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
  name: `member_${n}`, label: `${n}·psnr`, origin: { seed: n },
  loss: "l2", blocks: 32, asinh_knee: 10, step: 70000, target_steps: 70000, fraction: 1,
  status: "complete", timeout: false, job: { jobid: "48107719", state: "COMPLETED" },
  psnr: 61.8, psnr_rank: 1, knee_integrated: { VIS: 53.5, Y_E: 62.4, J_E: 59.7, H_E: 58.8, mean: 58.6 }, knee_rank: 1,
  gate_usage: { VIS: 0.05, Y_E: 0.05, J_E: 0.05, H_E: 0.05 }, gate_usage_source: null, coherence: { overall: 0.92, sr: 0.58 },
  ...patch,
});

const MEMBERS = {
  psnr_fields: 100, eval_subset: "test",
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

const DETAIL: MemberDetail = {
  name: "member_196", label: "196·psnr", active: true, archived: null,
  row: MEMBERS.members[0] as MemberRow,
  curves: { psnr: [[1000, 40], [2000, 42]], band_psnr: { VIS: [[1000, 39]] }, loss_series: [[1000, 0.2]], train_loss: [], gnorm: [], gnorm_max: [], step_time: [] },
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
    "GET /ensemble/members.json?mode=starfull": () => ({ body: MEMBERS }),
    "GET /api/experiments": () => ({ body: { experiments: [] } }),
    "GET /api/models": () => ({ body: { models: [] } }),
    "GET /ensemble/training-jobs.json": () => ({ body: { jobs: [TRAIN_JOB] } }),
    "GET /api/fasrc/steps/status": () => ({ body: { ssh_connected: false, steps: [{ step_id: "ensemble_train",
      defaults: { partition: "gpu", n_cpus: 4, n_gpus: 1, memory: "32G", time_limit: "48:00:00" }, fixed_gpus: 1 }] } }),
    "GET /api/fasrc/status": () => ({ body: { ssh_connected: false, last_error: "timed out" } }),
    "GET /api/config": () => ({ body: { config: { psf_warp_prob: 1, psf_warp_alpha_max: 5, psf_warp_sigma: 3, saturation_mask_prob: 0.5,
      lr_peak: 0.0002, lr_final: 1e-6, lr_warmup_steps: 2000, plateau_lr_enabled: false } } }),
    "GET /ensemble/member/member_196.json": () => ({ body: DETAIL }),
    "POST /ensemble/archive-member": () => ({ body: { job_id: "ar1" } }),
    "GET /api/jobs/ar1": () => ({ body: { job_id: "ar1", status: "done", label: "archive", progress: null } }),
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
    calls.push({ url, method, form, json: typeof init.body === "string" ? JSON.parse(init.body) : undefined });
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

const show = (el: ReactElement, url = "/models/universal") => render(
  <QueryClientProvider client={queryClient}>
    <MemoryRouter initialEntries={[url]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Routes><Route path="/models/*" element={<>{el}<Spy /></>} /><Route path="*" element={<Spy />} /></Routes>
    </MemoryRouter>
  </QueryClientProvider>,
);

async function answer(title: RegExp | string, button: string) {
  const dlg = await screen.findByRole("alertdialog", { name: title });
  fireEvent.click(within(dlg).getByRole("button", { name: button }));
}

/* ── Universal ─────────────────────────────────────────────────────────── */
describe("universal", () => {
  const URL_ = "GET /ensemble/universal.json?mode=starfull";
  const grid = (name: string) => screen.findByRole("grid", { name });
  beforeEach(() => {
    routes[URL_] = () => ({ body: universalPayload() });
    routes["POST /ensemble/universal/score"] = () => ({ body: { job_id: "us1" } });
    routes["POST /ensemble/universal/pull"] = () => ({ body: { job_id: "up1" } });
  });

  it("states the best model against production, and starts nothing on open", async () => {
    show(<Universal />, "/models/universal");
    const line = await waitFor(() => {
      const el = document.querySelector(".ui-summary");
      if (!el) throw new Error("not yet");
      return el as HTMLElement;
    });
    expect(line.textContent?.replace(/\u00a0/g, " ")).toBe(
      "5 universal models; the best, #211, scores 61.60 dB integrated PSNR after 12.7 h of training, +1.10 dB over production, Universal 257 (60.50 dB)");
    // no combiners: no gate rows, no weighting switch
    expect(screen.queryByRole("grid", { name: "Combiners by training time" })).toBeNull();
    expect(screen.queryByRole("radiogroup", { name: "Combiner weighting" })).toBeNull();
    await grid("Universal models");
    expect(calls.filter((c) => c.method === "POST")).toHaveLength(0);
    expect(document.querySelector(".ui-kpi")).toBeNull();
  });

  it("scores the models without a current score only after the confirm", async () => {
    show(<Universal />, "/models/universal");
    const btn = await screen.findByRole("button", { name: "Score 1 model on the 100 test fields" });
    expect(screen.getByText("1 universal model has no current score: 215.")).toBeTruthy();
    fireEvent.click(btn);
    await answer("Score 1 universal model on the 100 test fields?", "Cancel");
    expect(posts("/ensemble/universal/score")).toHaveLength(0);
    fireEvent.click(btn);
    await answer("Score 1 universal model on the 100 test fields?", "Score");
    await waitFor(() => expect(posts("/ensemble/universal/score")[0]?.form).toEqual({ mode: "starfull", members: "member_215", force: "0" }));
  });

  it("pulls what finished on FASRC only when connected, and only after the confirm", async () => {
    const view = show(<Universal />, "/models/universal");
    expect(await screen.findByText("1 universal model finished on FASRC and is not here: 213; 1 model still training: 218.")).toBeTruthy();
    const off = await screen.findByRole("button", { name: "Pull 1 model from FASRC" });
    await waitFor(() => expect((off as HTMLButtonElement).disabled).toBe(true));
    view.unmount();
    routes["GET /api/fasrc/status"] = () => ({ body: { ssh_connected: true } });
    queryClient.clear();
    show(<Universal />, "/models/universal");
    const on = await screen.findByRole("button", { name: "Pull 1 model from FASRC" });
    await waitFor(() => expect((on as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(on);
    await answer("Pull 1 universal model from FASRC?", "Pull");
    await waitFor(() => expect(posts("/ensemble/universal/pull")[0]?.form).toEqual({ members: "member_213" }));
    expect(screen.getByRole("table", { name: "Universal models on FASRC" })).toBeTruthy();
  });

  it("links Sky › Compare's new comparison for all local models and for one", async () => {
    show(<Universal />, "/models/universal");
    const all = await screen.findByRole("link", { name: "Compare all on real fields" });
    expect(all.getAttribute("href")).toBe(
      "/sky/compare?new=1&models=production,universal:member_211,universal:member_210,universal:member_208,universal:member_215");
    const table = await grid("Universal models");
    const r211 = within(table).getByText("#211").closest("tr") as HTMLElement;
    fireEvent.pointerDown(within(r211).getByRole("button", { name: "#211 actions" }), { button: 0 });
    fireEvent.click(await screen.findByRole("menuitem", { name: "Compare on real fields" }));
    await waitFor(() => expect(lastLocation).toBe("/sky/compare?new=1&models=production,universal:member_211"));
  });

  it("takes the real holes from the newest Sky › Compare run that scored THIS checkpoint", async () => {
    const agg = (h: number, r: number) => ({ n_tiles: 2, per_band: { VIS: { hole_pct: h }, Y_E: { hole_pct: h - 1 }, J_E: { hole_pct: h }, H_E: { hole_pct: h } },
      summary: { hole_pct_mean: h, hole_pct_max: h, median_R: r } });
    routes["GET /api/experiments"] = () => ({ body: { experiments: [
      { id: "e2", created: "2026-10-13T09:00:00Z", label: "poster", summary: { "universal:member_211": agg(30, 0.8), "universal:member_210": agg(12, 0.97) } },
      { id: "e1", created: "2026-10-12T09:00:00Z", label: "nexus", summary: { "universal:member_211": agg(6, 1.03) } },
    ] } });
    routes["GET /api/experiments/e2"] = () => ({ body: { id: "e2", fingerprints: { "universal:member_211": "old", "universal:member_210": "u210" } } });
    routes["GET /api/experiments/e1"] = () => ({ body: { id: "e1", fingerprints: { "universal:member_211": "u211" } } });
    routes["GET /api/models"] = () => ({ body: { models: [{ spec: "universal:member_211", fingerprint: "u211" }, { spec: "universal:member_210", fingerprint: "u210" }] } });
    show(<Universal />, "/models/universal");
    const table = await grid("Universal models");
    const r211 = within(table).getByText("#211").closest("tr") as HTMLElement;
    const r210 = within(table).getByText("#210").closest("tr") as HTMLElement;
    await waitFor(() => expect(within(r211).getByRole("link", { name: "VIS 6" }).getAttribute("href")).toBe("/sky/compare?exp=e1"));
    expect(within(r211).getByText("1.03")).toBeTruthy();
    expect(within(r210).getByRole("link", { name: "VIS 12" }).getAttribute("href")).toBe("/sky/compare?exp=e2");
    expect(within(table).getByText("#208").closest("tr")?.textContent).not.toMatch(/checking/);
  });

  it("pulls only what left a checkpoint; a failed job is tried from its row, a registered model with none here pulled again", async () => {
    routes["GET /api/fasrc/status"] = () => ({ body: { ssh_connected: true } });
    const fasrc = [
      { name: "member_213", jobids: ["1"], state: "COMPLETED", num_res_blocks: 16, steps: 250_000, knee: "learned" as const, partition: "gpu",
        finished: true, pullable: true, registered: false },
      { name: "member_220", jobids: ["2"], state: "OUT_OF_MEMORY", num_res_blocks: 64, steps: 200_000, knee: "learned" as const, partition: "gpu_test",
        finished: true, pullable: false, registered: false },
      { name: "member_216", jobids: ["3"], state: "COMPLETED", num_res_blocks: 16, steps: 500_000, knee: "learned" as const, partition: "gpu",
        finished: true, pullable: true, registered: true },
    ];
    routes[URL_] = () => ({ body: universalPayload({ fasrc }) });
    show(<Universal />, "/models/universal");
    expect(await screen.findByText("1 universal model finished on FASRC and is not here: 213; 1 model registered here with no checkpoint "
      + "(the last pull failed): 216; 1 model ended without a clean finish (220 out_of_memory): try it from the table below.")).toBeTruthy();
    const pull = await screen.findByRole("button", { name: "Pull 1 model from FASRC" });
    await waitFor(() => expect((pull as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(pull);
    await answer("Pull 1 universal model from FASRC?", "Pull");
    await waitFor(() => expect(posts("/ensemble/universal/pull")[0]?.form).toEqual({ members: "member_213" }));
    const table = screen.getByRole("table", { name: "Universal models on FASRC" });
    expect(within(table).getByText("out_of_memory")).toBeTruthy();
    expect(within(table).getByText("completed · not pulled")).toBeTruthy();
    // the running row has no action; the failed one is a try, its state in the confirm
    act(() => useJobsStore.getState().reset());
    fireEvent.click(within(table).getByRole("button", { name: "Try pull #220" }));
    const tryDlg = await screen.findByRole("alertdialog", { name: "Pull 1 universal model from FASRC?" });
    expect(tryDlg.textContent).toMatch(/Its job ended OUT_OF_MEMORY: when FASRC holds no checkpoint the pull refuses and registers nothing/);
    fireEvent.click(within(tryDlg).getByRole("button", { name: "Pull" }));
    await waitFor(() => expect(posts("/ensemble/universal/pull")[1]?.form).toEqual({ members: "member_220" }));
    act(() => useJobsStore.getState().reset());
    fireEvent.click(within(table).getByRole("button", { name: "Pull again #216" }));
    await answer("Pull again 1 universal model from FASRC?", "Pull again");
    await waitFor(() => expect(posts("/ensemble/universal/pull")[2]?.form).toEqual({ members: "member_216" }));
  });

  it("gives the production model's row the real holes of a run that scored production, which runs that model", async () => {
    routes[URL_] = () => ({ body: universalProduction() });
    const agg = (h: number, r: number) => ({ n_tiles: 2, per_band: { VIS: { hole_pct: h }, Y_E: { hole_pct: h - 1 }, J_E: { hole_pct: h }, H_E: { hole_pct: h } },
      summary: { hole_pct_mean: h, hole_pct_max: h, median_R: r } });
    routes["GET /api/experiments"] = () => ({ body: { experiments: [
      { id: "e5", created: "2026-10-21T09:00:00Z", label: "poster", summary: { production: agg(5, 1.01), "universal:member_211": agg(8, 0.99) } },
    ] } });
    routes["GET /api/experiments/e5"] = () => ({ body: { id: "e5", fingerprints: { production: "u210", "universal:member_211": "u211" } } });
    routes["GET /api/models"] = () => ({ body: { models: [{ spec: "production", fingerprint: "u210" },
      { spec: "universal:member_211", fingerprint: "u211" }, { spec: "universal:member_210", fingerprint: "u210" }] } });
    show(<Universal />, "/models/universal");
    const table = await grid("Universal models");
    const r210 = within(table).getByText("#210").closest("tr") as HTMLElement;
    await waitFor(() => expect(within(r210).getByRole("link", { name: "VIS 5" }).getAttribute("href")).toBe("/sky/compare?exp=e5"));
    expect(within(r210).getByText("1.01")).toBeTruthy();
    const r211 = within(table).getByText("#211").closest("tr") as HTMLElement;
    expect(within(r211).getByRole("link", { name: "VIS 8" })).toBeTruthy();
  });

  it("makes a registered model with a checkpoint production only after a confirm, then re-reads what names production", async () => {
    routes[URL_] = () => ({ body: universalProduction() });
    routes["POST /ensemble/production"] = () => ({ body: { ok: true, production: "member_211", spec: "universal:member_211" } });
    show(<Universal />, "/models/universal");
    const table = await grid("Universal models");
    const menuOf = async (num: string) => {
      const row = within(table).getByText(`#${num}`).closest("tr") as HTMLElement;
      fireEvent.pointerDown(within(row).getByRole("button", { name: `#${num} actions` }), { button: 0 });
      await screen.findByRole("menuitem", { name: "Inspect" });
    };
    // Escape hands focus back to the trigger a tick later: a menu opened before
    // that loses focus to it and closes (the kit's menus are not modal).
    const closeMenuOf = async (num: string) => {
      fireEvent.keyDown(document.activeElement ?? document.body, { key: "Escape" });
      await waitFor(() => expect(document.activeElement).toBe(within(table).getByRole("button", { name: `#${num} actions` })));
    };
    await menuOf("211");
    fireEvent.click(screen.getByRole("menuitem", { name: "Make production…" }));
    const dlg = await screen.findByRole("alertdialog", { name: "Make Universal 211 production?" });
    expect(dlg.textContent).toMatch(/Every page that defaults to “production” — Sky, the NEXUS plates, the figures and inference — will run Universal 211 instead of Universal 210\./);
    expect(dlg.textContent).toMatch(/The production outputs made so far \(real SR tiles, plates, figures\) turn stale until they are run again\./);
    fireEvent.click(within(dlg).getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(screen.queryByRole("alertdialog")).toBeNull());
    expect(posts("/ensemble/production")).toHaveLength(0);
    const before = { universal: gets("/ensemble/universal.json?mode=starfull").length, models: gets("/api/models").length };
    await menuOf("211");
    fireEvent.click(screen.getByRole("menuitem", { name: "Make production…" }));
    await answer("Make Universal 211 production?", "Make production");
    await waitFor(() => expect(posts("/ensemble/production")[0]?.form).toEqual({ model: "universal:member_211" }));
    await waitFor(() => expect(gets("/ensemble/universal.json?mode=starfull").length).toBeGreaterThan(before.universal));
    await waitFor(() => expect(gets("/api/models").length).toBeGreaterThan(before.models));
    // production itself offers no switch; a model with no checkpoint cannot be production
    await menuOf("210");
    expect(screen.queryByRole("menuitem", { name: "Make production…" })).toBeNull();
    await closeMenuOf("210");
    await menuOf("212");
    expect(screen.getByRole("menuitem", { name: "Make production…" }).getAttribute("aria-disabled")).toBe("true");
  });

  it("says when there is no universal model yet, and when production's score is stale", async () => {
    routes[URL_] = () => ({ body: universalPayload({ available: false, models: [], fasrc: [] }) });
    const view = show(<Universal />, "/models/universal");
    expect(await screen.findByText("No universal models yet")).toBeTruthy();
    view.unmount();
    routes[URL_] = () => ({ body: universalPayload({ production: { integrated: [55, 64, 62, 61], mean: 60.5, stale: true, model: "member_257" },
      notes: ["no universal model is production — make one production from its row", "Done."] }) });
    queryClient.clear();
    show(<Universal />, "/models/universal");
    expect(await screen.findByText(/Production's ∫PSNR \(Universal 257\) is stale/)).toBeTruthy();
    // the server's notes, as sentences
    expect(screen.getByText("No universal model is production — make one production from its row.")).toBeTruthy();
    expect(screen.getByText("Done.")).toBeTruthy();
  });
});

/* ── Train ─────────────────────────────────────────────────────────────── */
describe("train", () => {

  it("previews names + command, repeats the last batch, names the count on the submit and needs FASRC", async () => {
    show(<Train />, "/models/train");
    expect(await screen.findByText("member_199")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Repeat last batch" }));
    await waitFor(() => {
      const spec = JSON.parse(posts("/ensemble/train/preview").at(-1)!.form.member_spec);
      expect(spec[0]).toMatchObject({ asinh_knees: [0.1, 1, 10, 100, 1000, 10000], output_knee: 10, knee_loss: "balanced" });
      expect(spec[1]).toMatchObject({ asinh_knee: 3000 });
    });
    const submit = await screen.findByRole("button", { name: "Submit 2 members to SLURM" });
    expect((submit as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText(/FASRC offline/)).toBeTruthy();
  });

  it("shows the forward-model values System › Config owns, read-only, and never sends them", async () => {
    show(<Train />, "/models/train");
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
    show(<Train />, "/models/train");
    const forward = await screen.findByRole("link", { name: "2 knobs changed · Edit" });
    expect(forward.getAttribute("href")).toBe("/system/config");
    expect(screen.getByRole("link", { name: "1 knob changed · Edit" }).getAttribute("href")).toBe("/system/config");
    expect(screen.queryByRole("link", { name: "Edit in System › Config" })).toBeNull();
  });

  it("re-reads the members when an in-app link opens Train for other members", async () => {
    show(<><Train /><Link to="/models/train?mode=continue&members=member_179">next</Link></>,
      "/models/train?mode=continue&members=member_178");
    await waitFor(() => expect(posts("/ensemble/train/preview").at(-1)?.form.members).toBe("member_178"));
    fireEvent.click(screen.getByRole("link", { name: "next" }));
    await waitFor(() => expect(posts("/ensemble/train/preview").at(-1)?.form.members).toBe("member_179"));
  });

  it("sends a test run to the step's test partition", async () => {
    routes["GET /api/fasrc/steps/status"] = () => ({ body: { ssh_connected: true, steps: [{ step_id: "ensemble_train",
      defaults: { partition: "gpu", n_cpus: 24, n_gpus: 1, memory: "32G", time_limit: "48:00:00" }, fixed_gpus: 1, test_partition: "gpu_test" }] } });
    routes["GET /api/fasrc/status"] = () => ({ body: { ssh_connected: true } });
    routes["POST /api/fasrc/steps/ensemble_train/submit"] = () => ({ body: { ok: true, jobid: "52300000" } });
    show(<Train />, "/models/train");
    fireEvent.click(await screen.findByRole("switch", { name: /Test run on gpu_test/ }));
    expect(await screen.findByText(/gpu_test partition/)).toBeTruthy();
    const submit = await screen.findByRole("button", { name: /^Submit 1 member/ });
    await waitFor(() => expect((submit as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(submit);
    await answer(/^Submit 1 member/, "Submit");
    await waitFor(() => expect(posts("/api/fasrc/steps/ensemble_train/submit").at(-1)?.form)
      .toMatchObject({ partition: "gpu_test", n_gpus: "1", xla: "0", bf16: "0" }));
  });

  it("starts from the last finished batch's resources and continues TIMEOUT members up to their target", async () => {
    const view = show(<Train />, "/models/train");
    const cpus = await screen.findByRole("spinbutton", { name: "CPUs / model" });
    await waitFor(() => expect((cpus as HTMLInputElement).value).toBe("4"));
    view.unmount();
    show(<Train />, "/models/train?mode=continue&members=member_178");
    const upTo = await screen.findByRole("spinbutton", { name: "Up to step" });
    expect((upTo as HTMLInputElement).value).toBe("70000");
    await waitFor(() => expect(posts("/ensemble/train/preview").at(-1)?.form).toMatchObject({ continue_basis: "target", target_steps: "70000" }));
  });

  it("advises the resources for the built params; Apply sets them per model and keeps the 'As job' hint", async () => {
    routes["POST /api/fasrc/resources/ensemble_train/recommend"] = () => ({ body: {
      ok: true, step_id: "ensemble_train", available: true, confidence: "high",
      resources: { n_cpus: "6", n_gpus: "1", memory: "36G", time_limit: "3:30:00" },
      current: { n_cpus: "4", n_gpus: "1", memory: "32G", time_limit: "3:00:00" },
      changes: [
        { field: "n_cpus", current: "4", recommended: "6", reason: "GPU waits on the input pipeline (starved): add CPUs" },
        { field: "memory", current: "32G", recommended: "36G", reason: "p90 peak 29 GB of 32 GB over 6 runs" },
        { field: "time_limit", current: "3:00:00", recommended: "3:30:00", reason: "p90 0.15 s per step × 70,000 steps × 1.2" },
      ],
      basis: { level: "similar", level_label: "same kind, batch and depth", n_runs: 6, jobids: [] }, notes: [], warnings: [],
    } });
    show(<Train />, "/models/train");
    const cpus = await screen.findByRole("spinbutton", { name: "CPUs / model" });
    await waitFor(() => expect((cpus as HTMLInputElement).value).toBe("4"));
    expect(await screen.findByText(/Recommended from 6 past runs/)).toBeTruthy();
    const ask = posts("/api/fasrc/resources/ensemble_train/recommend").at(-1)!;
    const preview = posts("/ensemble/train/preview").at(-1)!;
    expect((ask.json as { params: unknown }).params).toEqual(preview.form);          // the same built params
    expect((ask.json as { resources: unknown }).resources).toEqual({ n_cpus: "4", n_gpus: "1", memory: "32G", time_limit: "3:00:00" });
    expect(document.querySelector(".radv__field")?.textContent).toBe("CPUs / model");  // per array task
    fireEvent.click(screen.getByRole("button", { name: "Apply" }));
    await waitFor(() => expect((cpus as HTMLInputElement).value).toBe("6"));
    expect((screen.getByRole("textbox", { name: "Memory / model" }) as HTMLInputElement).value).toBe("36G");
    expect((screen.getByRole("textbox", { name: "Time limit / model" }) as HTMLInputElement).value).toBe("3:30:00");
    expect(screen.getByText(/As job 48107719/)).toBeTruthy();
  });

  it("offers no star regime and never sends one, even for a clone of a legacy starless batch", async () => {
    routes["GET /ensemble/training-jobs.json"] = () => ({ body: { jobs: [
      { ...TRAIN_JOB, jobid: "47000001", params: { ...TRAIN_JOB.params, starless: "1" } }] } });
    show(<Train />, "/models/train?from=47000001");
    await waitFor(() => expect(JSON.parse(posts("/ensemble/train/preview").at(-1)!.form.member_spec)).toHaveLength(2));
    expect(posts("/ensemble/train/preview").every((c) => !("starless" in c.form))).toBe(true);
    expect(screen.queryByRole("radiogroup", { name: "Regime" })).toBeNull();
    expect(screen.queryByText(/starless|starfull|regime/i)).toBeNull();
    expect(await screen.findByRole("button", { name: "Submit 2 members to SLURM" })).toBeTruthy();
  });
});

/* ── inspector and back-trace ──────────────────────────────────────────── */
describe("member inspector", () => {

  it("opens a registered universal model as one (not archived): Universal, Sky › Compare, Archive after confirm", async () => {
    routes["GET /ensemble/member/member_211.json"] = () => ({ body: {
      ...DETAIL, name: "member_211", label: "211·psnr", active: false, archived: null, knee: null, gate: null,
      row: member(211, { class: "universal", asinh_knee: null, asinh_knees: [0.1, 1, 10, 100, 1000, 10000], output_knee: null,
        learned_output_knee: true, blocks: 16, knee_integrated: null, gate_usage: null }),
    } });
    show(<MemberInspector id="member_211" />);
    await screen.findByText("Training curves");
    expect(screen.getByText("universal")).toBeTruthy();
    expect(screen.queryByText("archived")).toBeNull();
    expect(screen.queryByRole("button", { name: "Continue" })).toBeNull();
    expect(screen.queryByText("Production-gate usage")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Archive" }));
    await answer("Archive member_211?", "Archive");
    await waitFor(() => expect(posts("/ensemble/archive-member")[0]?.form).toEqual({ member: "member_211" }));
    fireEvent.click(screen.getByRole("button", { name: "Compare on real fields" }));
    await waitFor(() => expect(lastLocation).toBe("/sky/compare?new=1&models=production,universal:member_211"));
  });

  it("restores an archived universal model in its own words", async () => {
    routes["POST /ensemble/restore-member"] = () => ({ body: { job_id: "rs3" } });
    routes["GET /ensemble/member/member_211.json"] = () => ({ body: {
      ...DETAIL, name: "member_211", label: "211·psnr", active: false, universal: false, class: "universal", row: null, curves: null,
      archived: { name: "member_211", archived_at: "2026-10-12T10:00:00Z", zip: "models/ensemble-member-211.zip", commit: "abc1234",
        zip_found: true, campaign: "current", class: "universal" },
    } });
    show(<MemberInspector id="member_211" />);
    expect(await screen.findByText("archived", { selector: ".ui-badge" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Restore from zip" }));
    const dlg = await screen.findByRole("alertdialog", { name: "Restore member_211?" });
    expect(dlg.textContent).toMatch(/registers it as a universal model again/);
    fireEvent.click(within(dlg).getByRole("button", { name: "Restore" }));
    await waitFor(() => expect(posts("/ensemble/restore-member")[0]?.form).toEqual({ member: "member_211" }));
  });
});
