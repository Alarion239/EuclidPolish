/* Sky › Real results / Experiments / Catalog eval and the realtile /
 * experiment inspector cards against a mocked backend (C9 + /api/evaluation). */
import { QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactElement } from "react";
import { Suspense } from "react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useJobsStore } from "../../../api/jobs";
import { queryClient } from "../../../api/query";
import { registerInspector, useInspectorRegistry } from "../../../app/inspector";
import { useInspector } from "../../../state/inspector";
import { useSelection } from "../../../state/selection";
import { resetConfirm } from "../../../ui";
import CatalogEval from "../tabs/CatalogEval";
import Experiments from "../tabs/Experiments";
import Results from "../tabs/Results";
import ExperimentInspector from "./ExperimentInspector";
import { ModelPicker } from "./ModelPicker";
import RealTileInspector from "./RealTileInspector";
import "./register";

type MockViewerProps = {
  collection: string; initialId?: string; tiers?: string[]; params?: Record<string, string>; nav?: boolean;
  onReady?: (api: unknown) => void; onState?: (s: unknown) => void;
};
/* The viewer engine is mocked: it mounts on initialId, reports its object id
 * through getState/onState and records every goToId. */
const viewerMock = vi.hoisted(() => ({ id: null as string | null, goTo: [] as string[], mounts: 0 }));
vi.mock("../../../viewer", async () => {
  const { useEffect } = await import("react");
  return {
    ImageViewer: (p: MockViewerProps) => {
      useEffect(() => {
        viewerMock.mounts += 1;
        viewerMock.id = p.initialId ?? null;
        const api = {
          getState: () => ({ id: viewerMock.id }),
          goToId: async (id: string) => { viewerMock.goTo.push(id); viewerMock.id = id; p.onState?.({ id }); return true; },
          setTiers: () => undefined,
        };
        p.onReady?.(api);
        return () => p.onReady?.(null);
        // eslint-disable-next-line react-hooks/exhaustive-deps
      }, []);
      return (
        <div data-testid="viewer" data-nav={String(p.nav ?? true)}>
          {p.collection}|{p.initialId ?? ""}|{(p.tiers ?? []).join(",")}|{p.params?.models ?? ""}
        </div>
      );
    },
  };
});

type Reply = { status?: number; body: unknown };
let routes: Record<string, (form: Record<string, string>) => Reply>;
let posts: { url: string; form: Record<string, string> }[];

const formOf = (body: BodyInit | null | undefined): Record<string, string> => {
  const out: Record<string, string> = {};
  if (body instanceof FormData) body.forEach((v, k) => { out[k] = String(v); });
  return out;
};

function Probe() {
  const loc = useLocation();
  return <output data-testid="loc">{loc.pathname}{loc.search}</output>;
}

const show = (el: ReactElement, url = "/sky/results") => render(
  <QueryClientProvider client={queryClient}>
    <MemoryRouter initialEntries={[url]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Routes><Route path="*" element={<>{el}<Probe /></>} /></Routes>
    </MemoryRouter>
  </QueryClientProvider>,
);

const answer = async (title: RegExp | string, button: string) => {
  const dlg = await screen.findByRole("alertdialog", { name: title });
  fireEvent.click(within(dlg).getByRole("button", { name: button }));
  await waitFor(() => expect(screen.queryByRole("alertdialog", { name: title })).toBeNull());
  return dlg;
};

const MODELS = {
  regime: "starfull", production_kind: "spatial_gate", members: ["1·psnr", "2·psnr"],
  models: [
    { spec: "production", kind: "production", label: "Production · spatial gate", available: true, n_members: 2, reads: ["1·psnr", "2·psnr"] },
    { spec: "mean", kind: "mean", label: "Mean of 2", available: true, n_members: 2, members: ["1·psnr", "2·psnr"] },
    { spec: "gate:pruned", kind: "gate", label: "Pruned gate", available: true, n_members: 20, reads: ["2·psnr", "7·psnr", "8·psnr", "9·psnr", "10·psnr", "11·psnr"] },
    { spec: "rbf", kind: "rbf", label: "RBF", available: false, reason: "fitted for 20 archived members" },
    { spec: "member:member_1", kind: "member", label: "Member 1·psnr", available: true, n_members: 1 },
    { spec: "member:member_2", kind: "member", label: "Member 2·psnr", available: true, n_members: 1 },
  ],
};

const NEXUS = {
  source: "nexus", label: "NEXUS", count: 2, tiles: [
    { source: "nexus", id: "f200w-0001", ref: "nexus/f200w-0001", label: "NEXUS tile 1", ra: 268.4, dec: 65.1, field: "EDF-N",
      shape: [255, 255], has_jwst: true, production_state: "stale", extras: { field_id: "nf" },
      models: { rbf: { state: "current", legacy: true, summary: { hole_pct_max: 4.25, median_R: 0.97 } } } },
    { source: "nexus", id: "f200w-0002", ref: "nexus/f200w-0002", label: "NEXUS tile 2", ra: 268.5, dec: 65.2, field: "EDF-N",
      shape: [255, 255], has_jwst: true, production_state: "current", extras: { field_id: "nf" },
      models: { production: { state: "current", summary: { hole_pct_max: 1.5 } } } },
  ],
};
const POSTER = {
  source: "poster", label: "Poster", count: 1, tiles: [
    { source: "poster", id: "p1", ref: "poster/p1", label: "Poster p1", ra: 273.23, dec: 68.36, field: "EDF-N",
      shape: [1024, 1024], has_jwst: false, production_state: "missing", models: {} },
  ],
};
const SOURCES_PAYLOAD = { sources: [
  { id: "nexus", label: "NEXUS", count: 2 }, { id: "tile", label: "Cached tiles", count: 0 },
  { id: "field", label: "Fields", count: 0 }, { id: "archive", label: "Archive", count: 0 },
  { id: "eval", label: "Eval", count: 0 }, { id: "poster", label: "Poster", count: 1 }, { id: "pair", label: "Pairs", count: 0 },
] };
const EMPTY = (source: string) => ({ source, label: source, count: 0, tiles: [] });

const CARD = {
  ...NEXUS.tiles[0], model_ready: true, runnable_models: ["production", "mean"],
  models: {
    rbf: { state: "current", legacy: true, origin: "nexus-field", label: "RBF",
      metrics: { per_band: { VIS: { hole_pct: 4.25, median_R: 0.97, flux_ratio: 0.99, n_peaks: 3 } }, summary: { hole_pct_max: 4.25 } } },
    "member:member_1": { state: "current", label: "Member 1", experiment_id: "20260926-101010-abcdef",
      metrics: { per_band: { VIS: { hole_pct: 1.25 } }, summary: { hole_pct_max: 1.25 } } },
  },
  image_urls: { lr: "/a", jwst: "/b", "m:rbf": "/c", "m:member:member_1": "/d" },
  experiments: ["20260926-101010-abcdef"], disk: { total_bytes: 4_000_000, output_bytes: 1_000_000 },
  q1_tile: { tile: "102158584", levels_e: [27.7, 13, 14, 13], rejected: null },
};

const RECORD = {
  id: "20260926-101010-abcdef", label: "core check", status: "done", created: "2026-09-26T10:10:10Z",
  duration_s: 42, tiles: ["nexus/f200w-0001", "poster/p1"], models: ["production", "member:member_1"], skipped: {},
  model_labels: { production: "Production", "member:member_1": "Member 1" },
  summary: {
    production: { bands: ["VIS"], per_band: { VIS: { hole_pct: 5.5, pct_R_lt_0p8: 10, median_R: 0.9, flux_ratio: 0.99 } }, summary: { pct_R_lt_0p8: 10, median_R: 0.9 } },
    "member:member_1": { bands: ["VIS"], per_band: { VIS: { hole_pct: 2.5 } }, summary: {} },
  },
  results: { "poster/p1": { production: { state: "computed", metrics: { bands: ["VIS"], per_band: { VIS: { hole_pct: 7 } },
    gate_core_weights: { VIS: [["2·psnr", 0.6], ["1·psnr", 0.4]] } } } } },
  errors: {}, counts: { members_computed: 2, members_reused: 0, outputs_computed: 4, outputs_reused: 0 },
};

beforeEach(() => {
  routes = {
    "GET /api/models": () => ({ body: MODELS }),
    "GET /api/real/sources": () => ({ body: SOURCES_PAYLOAD }),
    "GET /api/real/nexus": () => ({ body: NEXUS }),
    "GET /api/real/poster": () => ({ body: POSTER }),
    "GET /api/real/nexus/f200w-0001": () => ({ body: CARD }),
    "GET /api/experiments": () => ({ body: { experiments: [RECORD] } }),
    "GET /api/experiments/20260926-101010-abcdef": () => ({ body: RECORD }),
  };
  for (const s of ["tile", "field", "archive", "eval", "pair"]) routes[`GET /api/real/${s}`] = () => ({ body: EMPTY(s) });
  posts = [];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = String(input);
    const method = init.method ?? "GET";
    const form = formOf(init.body);
    if (method === "POST") posts.push({ url, form });
    const r = routes[`${method} ${url}`]?.(form) ?? { status: 404, body: { ok: false, error: `no route ${method} ${url}` } };
    return new Response(JSON.stringify(r.body), { status: r.status ?? 200 });
  }));
  queryClient.clear();
  useJobsStore.getState().reset();
  useSelection.getState().clear();
  useInspector.getState().clear();
  Object.assign(viewerMock, { id: null, goTo: [], mounts: 0 });
});
afterEach(() => {
  act(() => resetConfirm());
  queryClient.clear();
  vi.unstubAllGlobals();
});

describe("inspector registration", () => {
  it("registers realtile and experiment on import", () => {
    const kinds = useInspectorRegistry.getState().kinds;
    expect(kinds.realtile?.title).toBeTypeOf("function");
    expect(kinds.experiment).toBeTruthy();
  });

  it("turns a realtile: target into tile: in place (one card, one kind label)", async () => {
    const off = registerInspector("tile", () => <p>tile card</p>, { title: (id) => `Tile ${id}` });
    try {
      act(() => useInspector.getState().show({ kind: "realtile", id: "nexus/f200w-0040" }));
      const Alias = useInspectorRegistry.getState().kinds.realtile.Component;
      render(<Suspense fallback={null}><Alias id="nexus/f200w-0040" /></Suspense>);   // the inspector panel provides this boundary
      await waitFor(() => expect(useInspector.getState().current).toEqual({ kind: "tile", id: "nexus/f200w-0040" }));
      expect(useInspector.getState().back).toEqual([]);                  // replaced, not a history step
    } finally { off(); }
  });
});

describe("Real results tab", () => {
  it("lists every source's tiles with state, models and metrics; a row opens the tile card", async () => {
    show(<Results />);
    expect(await screen.findByText("f200w-0001")).toBeTruthy();
    expect(screen.getByText("p1")).toBeTruthy();
    expect(screen.getByText("4.3")).toBeTruthy();                      // headline holes (rbf, legacy)
    fireEvent.click(screen.getByText("f200w-0001"));
    expect(useInspector.getState().current).toEqual({ kind: "tile", id: "nexus/f200w-0001" });
  });

  it("filters by production state through the URL", async () => {
    show(<Results />, "/sky/results?state=stale");
    expect(await screen.findByText("f200w-0001")).toBeTruthy();
    expect(screen.queryByText("f200w-0002")).toBeNull();
    expect(screen.queryByText("p1")).toBeNull();
  });

  it("loads only the chosen sources", async () => {
    show(<Results />, "/sky/results?src=poster");
    expect(await screen.findByText("p1")).toBeTruthy();
    expect(screen.queryByText("f200w-0001")).toBeNull();
    const gets = (vi.mocked(fetch).mock.calls).map(([u]) => String(u));
    expect(gets).not.toContain("/api/real/nexus");
  });

  it("deletes the selected tiles' outputs only after a danger confirm", async () => {
    routes["POST /api/real/nexus/f200w-0001/delete-outputs"] = () => ({ body: { ok: true, removed_count: 3, cache_bytes_freed: 2048 } });
    useSelection.getState().select("tile", ["nexus/f200w-0001"]);
    show(<Results />);
    await screen.findByText("f200w-0001");
    const more = screen.getByRole("button", { name: "More actions" });
    fireEvent.pointerDown(more, { button: 0 });
    fireEvent.click(await screen.findByRole("menuitem", { name: /Delete model outputs/ }));
    await answer(/Delete the model outputs of 1 tile/, "Cancel");
    expect(posts).toHaveLength(0);
    fireEvent.pointerDown(screen.getByRole("button", { name: "More actions" }), { button: 0 });
    fireEvent.click(await screen.findByRole("menuitem", { name: /Delete model outputs/ }));
    await answer(/Delete the model outputs of 1 tile/, "Delete outputs");
    await waitFor(() => expect(posts.map((p) => p.url)).toEqual(["/api/real/nexus/f200w-0001/delete-outputs"]));
  });

  it("computes the missing metrics as an experiment over only the unscored outputs", async () => {
    const unscored = { ...NEXUS.tiles[1], models: { production: { state: "current" }, mean: { state: "current", summary: { hole_pct_max: 1 } } } };
    routes["GET /api/real/nexus"] = () => ({ body: { ...NEXUS, tiles: [NEXUS.tiles[0], unscored] } });
    routes["POST /api/experiments"] = () => ({ body: { ok: true, job_id: "m1", experiment_id: "20260926-131313-222222", skipped: {} } });
    useSelection.getState().select("tile", ["nexus/f200w-0001", "nexus/f200w-0002"]);
    show(<Results />);
    await screen.findByText("f200w-0002");
    fireEvent.pointerDown(screen.getByRole("button", { name: "More actions" }), { button: 0 });
    fireEvent.click(await screen.findByRole("menuitem", { name: /Compute metrics of 1 unscored output/ }));
    await answer(/Compute the metrics of 1 output on 1 tile/, "Compute metrics");
    await waitFor(() => expect(posts).toEqual([{
      url: "/api/experiments", form: { tiles: "nexus/f200w-0002", models: "production", label: "metrics" },
    }]));
  });

  it("caches a 25.6″ tile after the Q1 check and a confirm", async () => {
    routes["GET /api/sky/at?ra=273.230900&dec=68.363700"] = () => ({ body: { q1_verdict: "observed", best_tile: "102160000" } });
    routes["POST /api/real/tiles"] = () => ({ body: { ok: true, job_id: "j1", id: "t", ref: "tile/t" } });
    show(<Results />);
    fireEvent.click(await screen.findByRole("button", { name: "Cache tile…" }));
    const input = await screen.findByPlaceholderText("273.2309 68.3637");
    fireEvent.change(input, { target: { value: "273.2309 68.3637" } });
    fireEvent.click(screen.getByRole("button", { name: "Cache tile" }));
    await answer("Cache a 25.6″ tile?", "Cache tile");
    await waitFor(() => expect(posts[0]).toEqual({ url: "/api/real/tiles", form: { ra: "273.230900", dec: "68.363700", run: "production,mean" } }));
  });
});

describe("real-field diagnostics", () => {
  const FIELD = { field: { field_id: "ra0267_decp064", ra: 267.4229, dec: 64.8873, count: 100, member_labels: ["1·psnr", "2·psnr", "3·psnr"] }, field_size: 2560 };
  const DIAGNOSTICS = { diagnostics: {
    version: 2, member_labels: ["1·psnr", "2·psnr", "3·psnr"],
    model_power: { k: [0.5, 0.05], r_pairs: [[0.9, 0.3], [0.8, 0.2], [0.85, 0.25]], r_cross: [0.85, 0.25], pixel_scale_arcsec: 0.05 },
    std_brightness: { x_edges: [0, 1, 2], y_edges: [-3, -2], counts: [[4], [2]], x_label: "mean brightness (asinh)", y_label: "log10(member std)" },
    combiners: { raw_incremental_minmeanmax_rbf: { mode: "histogram", x_edges: [0, 1, 2], counts: [9, 99], x_label: "RBF weight", pixel_count: 108 } },
  } };

  it("shows the field's r(d) beside the synthetic one, switches views in the URL, recomputes after a confirm", async () => {
    routes["GET /api/inference/field.json"] = () => ({ body: FIELD });
    routes["GET /api/inference/diagnostics.json"] = () => ({ body: DIAGNOSTICS });
    routes["GET /ensemble/evals.json?mode=starfull"] = () => ({ body: { ps: { theta: [1, 2], r_cross: [0.7, 0.6], r_pairs: [[0.7, 0.6]] } } });
    routes["POST /inference/refresh-combiners"] = () => ({ body: { job_id: "fd1" } });
    show(<Results />, "/sky/results?diag=1");
    const section = (await screen.findByText("Field diagnostics")).closest("section") as HTMLElement;
    expect(await within(section).findByText("Real field")).toBeTruthy();
    expect(within(section).getByText("Synthetic STARFULL")).toBeTruthy();
    expect(within(section).getByText(/3 member pairs/)).toBeTruthy();
    expect(within(section).getByText(/ra0267_decp064/)).toBeTruthy();
    fireEvent.click(within(section).getByRole("radio", { name: "RBF occupancy" }));
    await waitFor(() => expect(screen.getByTestId("loc").textContent).toContain("fd=occupancy"));
    expect(within(section).getByText(/RBF · 108 real pixels/)).toBeTruthy();
    fireEvent.click(within(section).getByRole("button", { name: "Recompute" }));
    await answer(/Recompute the real-field diagnostics/, "Recompute");
    await waitFor(() => expect(posts.map((p) => p.url)).toEqual(["/inference/refresh-combiners"]));
    fireEvent.click(within(section).getByRole("button", { name: "Hide field diagnostics" }));
    await waitFor(() => expect(screen.queryByText("Field diagnostics")).toBeNull());
    expect(screen.getByTestId("loc").textContent).not.toContain("diag=1");
  });

  it("explains a field without diagnostics", async () => {
    routes["GET /api/inference/field.json"] = () => ({ body: FIELD });
    routes["GET /api/inference/diagnostics.json"] = () => ({ body: { diagnostics: null } });
    routes["GET /ensemble/evals.json?mode=starfull"] = () => ({ status: 500, body: { ok: false, error: "no evals" } });
    show(<Results />, "/sky/results?diag=1");
    expect(await screen.findByText("No diagnostics for this field yet")).toBeTruthy();
  });
});

describe("model picker", () => {
  it("disables unavailable specs with the reason and offers quick picks", async () => {
    const seen: string[][] = [];
    show(<ModelPicker value={[]} onChange={(v) => seen.push(v)} />);
    const rbf = await screen.findByRole("checkbox", { name: /rbf/ });
    expect((rbf as HTMLInputElement).disabled).toBe(true);
    expect(screen.getByText("fitted for 20 archived members")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Production + mean" }));
    expect(seen.at(-1)).toEqual(["production", "mean"]);
    fireEvent.click(screen.getByRole("button", { name: "+ all members" }));
    expect(seen.at(-1)).toEqual(["member:member_1", "member:member_2"]);
    // a pruned gate says how many members it reads, not how many it was fitted on
    expect(screen.getByText(/6 of 20 members/)).toBeTruthy();
  });
});

describe("realtile inspector", () => {
  it("shows the one real-tile card: the viewer first (only the tile's own tiers), models table, per-band metrics", async () => {
    const { container } = show(<RealTileInspector id="nexus/f200w-0001" />);
    const viewer = await screen.findByTestId("viewer");
    // two large frames: LR and the first model output (JWST one chip away); the picker lists exactly this tile's outputs
    expect(viewer.textContent).toBe("real|f200w-0001|lr,m:rbf|rbf,member:member_1");
    // "Open large" puts the viewer in focus mode (the frames fill the stage)
    expect(screen.getByRole("button", { name: "Open large" })).toBeTruthy();
    expect(container.querySelector(".res-card")?.firstElementChild?.contains(viewer)).toBe(true);
    expect(screen.getByText("m1")).toBeTruthy();
    expect(screen.getByText("Metrics of rbf")).toBeTruthy();
    expect(screen.getByText("0.970")).toBeTruthy();                     // median R, VIS
    fireEvent.click(screen.getByText("m1"));
    expect(screen.getByText("Metrics of m1")).toBeTruthy();
    // the atlas card's actions are on this card too
    expect(screen.getByRole("button", { name: "Compare models…" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Overlay on the sky" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Show on sky" }));
    expect(screen.getByTestId("loc").textContent).toMatch(/^\/sky\/atlas\?ra=268\.4\d*&dec=65\.1\d*&fov=/);
  });

  it("a tile without model outputs asks the viewer for no model tier at all", async () => {
    routes["GET /api/real/poster/p1"] = () => ({ body: { ...POSTER.tiles[0], model_ready: true, image_urls: { lr: "/a" } } });
    show(<RealTileInspector id="poster/p1" />);
    expect((await screen.findByTestId("viewer")).textContent).toBe("real|p1|lr|,");
    expect(screen.getByText(/No SR for this tile yet/)).toBeTruthy();
  });

  it("adds the catalogue-eval provenance for an eval object", async () => {
    routes["GET /api/real/eval/lensA"] = () => ({ body: { ...CARD, source: "eval", id: "lensA", ref: "eval/lensA", has_jwst: false, models: {}, image_urls: { lr: "/a" } } });
    routes["GET /api/evaluation/objects/lensA"] = () => ({ body: {
      id: "lensA", state: "stale", state_reason: "membership changed: made by 22 member(s), 30 active STARFULL now",
      members: { member_labels: Array(22).fill("x") }, current: { n_members: 30, combiner_kind: "spatial_gate" },
      provenance: [{ id: "2941ea92", git: "e969277", dirty: true, created_at: "2026-07-26T16:04:34Z" }],
      downloads: { LR: "/eval-files/lensA/original_stack.fits", SR: "/eval-files/lensA/SR.fits" },
    } });
    show(<RealTileInspector id="eval/lensA" />);
    fireEvent.click(await screen.findByRole("button", { name: "Catalogue evaluation" }));
    expect(await screen.findByText(/membership changed/)).toBeTruthy();
    expect(screen.getByText("22 members · combiner not recorded")).toBeTruthy();
    expect(screen.getByText("30 STARFULL · spatial_gate")).toBeTruthy();
    expect(screen.getByRole("link", { name: "SR" }).getAttribute("href")).toBe("/eval-files/lensA/SR.fits");
  });

  it("explains an unknown tile with the server's error", async () => {
    routes["GET /api/real/nexus/nope"] = () => ({ status: 404, body: { ok: false, error: "unknown nexus tile 'nope'" } });
    show(<RealTileInspector id="nexus/nope" />);
    expect(await screen.findByText("unknown nexus tile 'nope'")).toBeTruthy();
  });

  it("runs models on the tile as an experiment after a confirm", async () => {
    routes["POST /api/experiments"] = () => ({ body: { ok: true, job_id: "j9", experiment_id: "20260926-111111-000000", tiles: ["nexus/f200w-0001"], models: ["production", "mean"], skipped: {} } });
    show(<RealTileInspector id="nexus/f200w-0001" />);
    fireEvent.click(await screen.findByRole("button", { name: "Run models…" }));
    fireEvent.click(await screen.findByRole("button", { name: "Run 2" }));
    await answer(/Run 2 models on 1 tile/, "Run");
    await waitFor(() => expect(posts[0]).toEqual({ url: "/api/experiments", form: { tiles: "nexus/f200w-0001", models: "production,mean" } }));
    expect(useJobsStore.getState().keyed["sky:experiment"]).toBe("j9");
  });
});

describe("Experiments tab", () => {
  it("preselects ?tiles= and starts the experiment with the picked models", async () => {
    routes["POST /api/experiments"] = () => ({ body: { ok: true, job_id: "j2", experiment_id: "20260926-121212-111111", skipped: {} } });
    show(<Experiments />, "/sky/experiments?tiles=nexus%2Ff200w-0001%2Cposter%2Fp1");
    expect(await screen.findByText("nexus/f200w-0001")).toBeTruthy();
    expect(screen.getByText("poster/p1")).toBeTruthy();
    await screen.findByRole("checkbox", { name: /production/ });
    // the cost is stated before anything runs
    expect(screen.getByText(/4 outputs \(2 models on 2 tiles\)\. Needs 2 member SRs per tile: at most 4 member inferences/)).toBeTruthy();
    // tiles handed over: the form is the point, no experiment opens by itself
    expect(screen.queryByTestId("viewer")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Run 2 models on 2 tiles" }));
    const dlg = await answer(/Run 2 models on 2 tiles/, "Run");
    expect(dlg.textContent).toContain("4 outputs (2 models on 2 tiles).");
    await waitFor(() => expect(posts[0]).toEqual({
      url: "/api/experiments", form: { tiles: "nexus/f200w-0001,poster/p1", models: "production,mean" },
    }));
    await waitFor(() => expect(screen.getByTestId("loc").textContent).toContain("exp=20260926-121212-111111"));
  });

  it("shows the detail first: comparison viewer, then metrics per model × band and core weights; history below", async () => {
    show(<Experiments />, "/sky/experiments?exp=20260926-101010-abcdef&scope=poster%2Fp1");
    expect(await screen.findByText("core check", { selector: "strong" })).toBeTruthy();
    const viewer = await screen.findByTestId("viewer");
    expect(viewer.textContent).toBe("real|p1|lr,m:production,m:member:member_1|production,member:member_1");
    const history = screen.getByRole("grid", { name: "Experiments" });
    expect(viewer.compareDocumentPosition(history) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(screen.getByText("7.0")).toBeTruthy();                       // production hole %, poster tile
    expect(screen.getByLabelText("Gate core weights")).toBeTruthy();
  });

  it("a plain visit (the tab link) opens the newest experiment, form folded, URL untouched", async () => {
    const older = { ...RECORD, id: "20260901-000000-000000", label: "older", created: "2026-09-01T00:00:00" };
    routes["GET /api/experiments"] = () => ({ body: { experiments: [older, { ...RECORD, created: "2026-09-26T10:10:10" }] } });
    show(<Experiments />, "/sky/experiments");
    expect(await screen.findByText("core check", { selector: "strong" })).toBeTruthy();
    expect(await screen.findByTestId("viewer")).toBeTruthy();
    const toggle = screen.getAllByRole("button", { name: /New experiment/ }).find((el) => el.hasAttribute("aria-expanded"));
    expect(toggle?.getAttribute("aria-expanded")).toBe("false");
    expect(screen.getByTestId("loc").textContent).toBe("/sky/experiments");
  });

  it("keeps the comparison viewer on the Scope tile (no own navigation)", async () => {
    const two = { ...RECORD, id: "x1", tiles: ["nexus/f200w-0001", "nexus/f200w-0002"], results: {} };
    routes["GET /api/experiments"] = () => ({ body: { experiments: [two] } });
    routes["GET /api/experiments/x1"] = () => ({ body: two });
    show(<Experiments />, "/sky/experiments?exp=x1");
    const viewer = await screen.findByTestId("viewer");
    expect(viewer.textContent).toMatch(/^real\|f200w-0001\|/);
    expect(viewer.dataset.nav).toBe("false");
    fireEvent.change(screen.getByRole("combobox", { name: "Metric scope" }), { target: { value: "nexus/f200w-0002" } });
    await waitFor(() => expect(viewerMock.goTo).toEqual(["f200w-0002"]));
    expect(viewerMock.id).toBe("f200w-0002");
    expect(viewerMock.mounts).toBe(1);                                   // same source: moved, not remounted
    expect(screen.getByTestId("loc").textContent).toContain("scope=nexus%2Ff200w-0002");
    fireEvent.change(screen.getByRole("combobox", { name: "Metric scope" }), { target: { value: "pooled" } });
    await waitFor(() => expect(viewerMock.goTo).toEqual(["f200w-0002", "f200w-0001"]));  // pooled shows the first tile
  });

  it("logs a markdown summary to the tracking notebook", async () => {
    routes["POST /api/tracking/log"] = () => ({ body: { ok: true, log_md: "" } });
    show(<Experiments />, "/sky/experiments?exp=20260926-101010-abcdef");
    fireEvent.click(await screen.findByRole("button", { name: "Log to tracking" }));
    const note = await screen.findByRole("textbox", { name: "Markdown note" });
    expect((note as HTMLTextAreaElement).value).toContain("**Real-data experiment `20260926-101010-abcdef`** — core check");
    fireEvent.click(screen.getByRole("button", { name: "Append" }));
    await waitFor(() => expect(posts[0]?.url).toBe("/api/tracking/log"));
    expect(posts[0].form.mode).toBe("append");
    expect(posts[0].form.text).toContain("| `production` | 5.5");
  });
});

describe("Experiments tab before any experiment", () => {
  it("reads the metric definitions without an experiment", async () => {
    routes["GET /api/experiments"] = () => ({ body: { experiments: [] } });
    show(<Experiments />, "/sky/experiments");
    expect(await screen.findByText("No experiments yet")).toBeTruthy();
    expect(await screen.findByText("What the metrics measure")).toBeTruthy();
    expect(screen.getByText(/SR pixels under the brightest 1 % of LR pixels/)).toBeTruthy();
    expect(screen.getByText(/Median enclosed-flux ratio over the peaks/)).toBeTruthy();
  });
});

describe("experiment inspector", () => {
  it("summarises the record and links to the full comparison", async () => {
    show(<ExperimentInspector id="20260926-101010-abcdef" />, "/sky/atlas");
    expect(await screen.findByText("core check")).toBeTruthy();
    expect(screen.queryByTestId("viewer")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Open in Experiments" }));
    expect(screen.getByTestId("loc").textContent).toBe("/sky/experiments?exp=20260926-101010-abcdef");
  });
});

const RUNS = {
  name: "eval_results", run: "eval_results", n: 3, n_ok: 2, columns: ["id"],
  current: { n_members: 30, combiner_kind: "spatial_gate" }, counts: { current: 1, stale: 1, unknown: 0 },
  groups: { A: 1, "syn-gal": 1 },
  rows: [
    { id: "lensA", ra: "57.98", dec: "-50.85", grade: "A", ok: "True", out_subdir: "lensA", flux_ratio_sr_over_lr: "0.67",
      kind: "lens", field: "EDF-S", viewer_id: "lensA", realtile: "eval/lensA", state: "stale", state_reason: "membership changed" },
    { id: "syn1", ra: "", dec: "", grade: "syn-gal", ok: "True", out_subdir: "syn1", kind: "synthetic", viewer_id: "syn1", state: "current", psnr_sr_hr: "44.1" },
    { id: "bad", ra: "1", dec: "2", grade: "A", ok: "False", error: "RuntimeError: VIS: empty", out_subdir: "bad", kind: "lens" },
  ],
};

describe("Catalog eval tab", () => {
  beforeEach(() => {
    routes["GET /api/evaluation/runs"] = () => ({ body: RUNS });
    routes["GET /auth/status"] = () => ({ body: { authenticated: false } });
  });

  it("puts the reconstruction viewer (LR beside SR) before the object list", async () => {
    show(<CatalogEval />, "/sky/catalog-eval");
    const viewer = await screen.findByTestId("viewer");
    expect(viewer.textContent).toBe("evaluation||LR,SR|");
    const table = screen.getByRole("grid", { name: "Evaluation objects" });
    expect(viewer.compareDocumentPosition(table) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("lists objects with their SR model state and hides failures unless asked", async () => {
    show(<CatalogEval />, "/sky/catalog-eval");
    expect(await screen.findByText("lensA")).toBeTruthy();
    expect(screen.getByText("syn1")).toBeTruthy();
    expect(screen.queryByText("bad")).toBeNull();
    expect(screen.getByText(/reconstructions predate the current model/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Failed too" }));
    expect(await screen.findByText("bad")).toBeTruthy();
  });

  it("syncs only after confirming the --delete-after, with confirm=1", async () => {
    routes["POST /api/evaluation/sync"] = (form) => (form.confirm === "1"
      ? { body: { ok: true, stdout: "synced", n: 3, n_ok: 2 } }
      : { status: 400, body: { ok: false, code: "confirm_required", error: "confirm" } });
    show(<CatalogEval />, "/sky/catalog-eval");
    await screen.findByText("lensA");
    const open = async () => {
      fireEvent.pointerDown(screen.getByRole("button", { name: "More catalogue actions" }), { button: 0 });
      fireEvent.click(await screen.findByRole("menuitem", { name: /Sync results from FASRC/ }));
    };
    await open();
    const dlg = await answer("Sync evaluation results from FASRC?", "Cancel");
    expect(dlg.textContent).toContain("delete local-only results");
    expect(posts.filter((p) => p.url === "/api/evaluation/sync")).toHaveLength(0);
    await open();
    await answer("Sync evaluation results from FASRC?", "Sync and delete local-only");
    await waitFor(() => expect(posts.filter((p) => p.url === "/api/evaluation/sync")).toHaveLength(1));
    expect(posts.find((p) => p.url === "/api/evaluation/sync")!.form).toEqual({ confirm: "1" });
  });

  it("runs the grouped analysis with the chosen size", async () => {
    routes["POST /api/evaluation/run-grouped"] = () => ({ body: { ok: true, job_id: "g1" } });
    routes["GET /api/jobs/g1"] = () => ({ body: { job_id: "g1", status: "running", label: "grouped", log: "" } });
    show(<CatalogEval />, "/sky/catalog-eval");
    await screen.findByText("lensA");
    fireEvent.click(screen.getAllByRole("button", { name: "Grouped analysis…" })[0]);
    fireEvent.change(await screen.findByRole("spinbutton"), { target: { value: "4" } });
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(posts[0]).toEqual({ url: "/api/evaluation/run-grouped", form: { n: "4", synthetic: "1" } }));
  });

  it("needs the Euclid session to query galaxies (links to Settings)", async () => {
    show(<CatalogEval />, "/sky/catalog-eval");
    await screen.findByText("lensA");
    fireEvent.click(screen.getByRole("button", { name: "Query galaxies…" }));
    expect(await screen.findByRole("link", { name: "Settings › Connections" })).toBeTruthy();
    expect((screen.getByRole("button", { name: "Query" }) as HTMLButtonElement).disabled).toBe(true);
  });
});
