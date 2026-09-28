/* Test kit of the Sky tabs and cards (imported by tests only): a mocked
 * backend (C9 + /api/evaluation) behind a stubbed `fetch`, its fixtures, and
 * the render / confirm helpers. Each test file mocks the image viewer itself
 * (vi.mock must be hoisted in the test file). */
import { QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactElement } from "react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { expect, vi } from "vitest";
import { useJobsStore } from "../../../api/jobs";
import { queryClient } from "../../../api/query";
import { useInspector } from "../../../state/inspector";
import { useSelection } from "../../../state/selection";
import { resetConfirm } from "../../../ui";

export type Reply = { status?: number; body: unknown };
export type Routes = Record<string, (form: Record<string, string>) => Reply>;
export type Post = { url: string; form: Record<string, string> };

export const MODELS = {
  regime: "starfull", production_kind: "spatial_gate", members: ["1·psnr", "2·psnr"],
  models: [
    { spec: "production", kind: "production", label: "Production · spatial gate", available: true, n_members: 2, reads: ["1·psnr", "2·psnr"] },
    { spec: "mean", kind: "mean", label: "Mean of 2", available: true, n_members: 2, members: ["1·psnr", "2·psnr"] },
    { spec: "gate:pruned", kind: "gate", label: "Pruned gate", available: true, n_members: 20, reads: ["2·psnr", "7·psnr", "8·psnr", "9·psnr", "10·psnr", "11·psnr"] },
    { spec: "gate:old", kind: "gate", label: "Old gate", available: false, reason: "fitted for 20 archived members" },
    { spec: "rbf", kind: "rbf", label: "RBF", available: false, reason: "fitted for 20 archived members" },
    { spec: "member:member_1", kind: "member", label: "Member 1·psnr", available: true, n_members: 1 },
    { spec: "member:member_2", kind: "member", label: "Member 2·psnr", available: true, n_members: 1 },
  ],
};

export const NEXUS = {
  source: "nexus", label: "NEXUS", count: 2, tiles: [
    { source: "nexus", id: "f200w-0001", ref: "nexus/f200w-0001", label: "NEXUS tile 1", ra: 268.4, dec: 65.1, field: "EDF-N",
      shape: [255, 255], has_jwst: true, production_state: "stale", extras: { field_id: "nf" },
      models: { rbf: { state: "current", legacy: true, label: "minibatched convex all-asinh RBF", summary: { hole_pct_max: 4.25, median_R: 0.97 }, flux_ratio: { VIS: 0.5 } } } },
    { source: "nexus", id: "f200w-0002", ref: "nexus/f200w-0002", label: "NEXUS tile 2", ra: 268.5, dec: 65.2, field: "EDF-N",
      shape: [255, 255], has_jwst: true, production_state: "current", extras: { field_id: "nf" },
      models: { production: { state: "current", label: "Production · spatial gate", summary: { hole_pct_max: 1.5, median_R: 1.01 }, flux_ratio: { VIS: 0.98 } } } },
  ],
};
export const POSTER = {
  source: "poster", label: "Poster", count: 1, tiles: [
    { source: "poster", id: "p1", ref: "poster/p1", label: "Poster p1", ra: 273.23, dec: 68.36, field: "EDF-N",
      shape: [1024, 1024], has_jwst: false, production_state: "missing", models: {} },
  ],
};
export const SOURCES_PAYLOAD = { sources: [
  { id: "nexus", label: "NEXUS", count: 2 }, { id: "tile", label: "Cached tiles", count: 0 },
  { id: "field", label: "Fields", count: 0 }, { id: "archive", label: "Archive", count: 0 },
  { id: "eval", label: "Eval", count: 3 }, { id: "poster", label: "Poster", count: 1 }, { id: "pair", label: "Pairs", count: 0 },
] };
export const EMPTY = (source: string) => ({ source, label: source, count: 0, tiles: [] });

const evalRow = (id: string, grade: string, over: Record<string, unknown> = {}) => ({
  id, grade, ok: "True", out_subdir: id, viewer_id: id, ra: "57.1", dec: "-49.5", field: "EDF-S",
  kind: grade === "gal" ? "galaxy" : grade.startsWith("syn") ? "synthetic" : "lens", realtile: `eval/${id}`,
  state: "stale", state_reason: "membership changed: made by 22 member(s), the production gate is fitted for 30 now",
  n_members: 22, combiner_kind: null, flux_ratio_sr_over_lr: "0.67", ...over,
});
export const EVAL_RUNS = {
  name: "eval_results", run: "eval_results", n: 6, n_ok: 5,
  current: { n_members: 30, combiner_kind: "spatial_gate" },
  counts: { current: 0, stale: 5, unknown: 0 },
  groups: { A: 1, B: 1, gal: 1, "syn-gal": 1 },
  rows: [
    evalRow("lensA", "A", { flux_ratio_sr_over_lr: "0.61" }), evalRow("lensB", "B", { flux_ratio_sr_over_lr: "0.73" }),
    evalRow("bad", "A", { ok: "False", state: null, error: "RuntimeError: VIS: downloaded file is empty", realtile: null, flux_ratio_sr_over_lr: "" }),
    evalRow("gal1", "gal", { flux_ratio_sr_over_lr: "0.82" }), evalRow("syn1", "syn-gal", { realtile: null }),
  ],
};
export const EVAL_TILES = {
  source: "eval", label: "Evaluation objects", count: 3, tiles: [
    { source: "eval", id: "lensA", ref: "eval/lensA", label: "lensA · A", ra: 57.1, dec: -49.5, extras: { kind: "lens", grade: "A", flux_ratio_sr_over_lr: 0.61 }, production_state: "stale", models: {} },
    { source: "eval", id: "lensB", ref: "eval/lensB", label: "lensB · B", ra: 57.2, dec: -49.6, extras: { kind: "lens", grade: "B" }, production_state: "stale", models: {} },
    { source: "eval", id: "gal1", ref: "eval/gal1", label: "gal1 · gal", ra: 57.3, dec: -49.7, extras: { kind: "galaxy", grade: "gal" }, production_state: "stale", models: {} },
  ],
};

export const CARD = {
  ...NEXUS.tiles[0], model_ready: true, runnable_models: ["production", "mean"],
  models: {
    rbf: { state: "current", legacy: true, origin: "nexus-field", label: "RBF",
      metrics: { per_band: { VIS: { hole_pct: 4.25, median_R: 0.97, flux_ratio: 0.31, n_peaks: 3 }, J_E: { hole_pct: 2 } }, summary: { hole_pct_max: 4.25, median_R: 0.97 } } },
    "member:member_1": { state: "current", label: "Member 1", experiment_id: "20260926-101010-abcdef",
      metrics: { per_band: { VIS: { hole_pct: 1.25, flux_ratio: 0.98 } }, summary: { hole_pct_max: 1.25 } } },
  },
  image_urls: { lr: "/a", jwst: "/b", "m:rbf": "/c", "m:member:member_1": "/d" },
  files: { lr: "real_tiles/nexus/f200w-0001/lr.fits", "m:member:member_1": "real_outputs/nexus/f200w-0001/member_1.fits" },
  experiments: ["20260926-101010-abcdef"], disk: { total_bytes: 4_000_000, output_bytes: 1_000_000 },
  q1_tile: { tile: "102158584", levels_e: [27.7, 13, 14, 13], rejected: null },
};

export const RECORD = {
  id: "20260926-101010-abcdef", label: "core check", status: "done", created: "2026-09-26T10:10:10Z",
  duration_s: 42, tiles: ["nexus/f200w-0001", "poster/p1"], models: ["production", "mean", "member:member_1"], skipped: {},
  model_labels: { production: "Production", mean: "Mean of 2", "member:member_1": "Member 1" },
  summary: {
    production: { bands: ["VIS", "J_E"], per_band: { VIS: { hole_pct: 5.5, pct_R_lt_0p8: 10, median_R: 0.9, flux_ratio: 0.99 }, J_E: { hole_pct: 8.25, flux_ratio: 1.3 } },
      summary: { hole_pct_max: 8.25, pct_R_lt_0p8: 10, median_R: 0.9 } },
    mean: { bands: ["VIS"], per_band: { VIS: { hole_pct: 30.5, flux_ratio: 0.31 } }, summary: { hole_pct_max: 30.5 } },
    "member:member_1": { bands: ["VIS"], per_band: { VIS: { hole_pct: 2.5 } }, summary: {} },
  },
  results: {
    "nexus/f200w-0001": {
      production: { state: "computed", metrics: { per_band: { VIS: { hole_pct: 4, flux_ratio: 0.96 } } } },
      mean: { state: "computed", metrics: { per_band: { VIS: { hole_pct: 31, flux_ratio: 0.3 } } } },
    },
    "poster/p1": { production: { state: "computed", metrics: { bands: ["VIS"], per_band: { VIS: { hole_pct: 7 } },
      gate_core_weights: { VIS: [["2·psnr", 0.6], ["1·psnr", 0.4]] } } } },
  },
  errors: {}, counts: { members_computed: 2, members_reused: 0, outputs_computed: 4, outputs_reused: 0 },
};

const formOf = (body: BodyInit | null | undefined): Record<string, string> => {
  const out: Record<string, string> = {};
  if (body instanceof FormData) body.forEach((v, k) => { out[k] = String(v); });
  return out;
};

/** Stub `fetch` with the default routes (a test adds or replaces entries in
 *  the returned `routes`); every POST is recorded in `posts`. Also resets the
 *  query cache and the stores the pages share. */
export function installBackend(): { routes: Routes; posts: Post[] } {
  const routes: Routes = {
    "GET /api/models": () => ({ body: MODELS }),
    "GET /api/real/sources": () => ({ body: SOURCES_PAYLOAD }),
    "GET /api/real/nexus": () => ({ body: NEXUS }),
    "GET /api/real/poster": () => ({ body: POSTER }),
    "GET /api/real/eval": () => ({ body: EVAL_TILES }),
    "GET /api/real/nexus/f200w-0001": () => ({ body: CARD }),
    "GET /api/experiments": () => ({ body: { experiments: [RECORD] } }),
    "GET /api/experiments/20260926-101010-abcdef": () => ({ body: RECORD }),
    "GET /api/evaluation/runs": () => ({ body: EVAL_RUNS }),
  };
  for (const s of ["tile", "field", "archive", "pair"]) routes[`GET /api/real/${s}`] = () => ({ body: EMPTY(s) });
  const posts: Post[] = [];
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
  return { routes, posts };
}

export function teardownBackend(): void {
  act(() => resetConfirm());
  queryClient.clear();
  vi.unstubAllGlobals();
}

function Probe() {
  const loc = useLocation();
  return <output data-testid="loc">{loc.pathname}{loc.search}</output>;
}

/** Render inside the query client and a router at `url`; `loc` shows the location. */
export const show = (el: ReactElement, url = "/sky/targets") => render(
  <QueryClientProvider client={queryClient}>
    <MemoryRouter initialEntries={[url]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Routes><Route path="*" element={<>{el}<Probe /></>} /></Routes>
    </MemoryRouter>
  </QueryClientProvider>,
);

/** Answer the confirm dialog titled `title` with `button` (typing `typed` first, for a typed confirmation). */
export const answer = async (title: RegExp | string, button: string, typed?: string) => {
  const dlg = await screen.findByRole("alertdialog", { name: title });
  if (typed != null) fireEvent.change(within(dlg).getByRole("textbox"), { target: { value: typed } });
  fireEvent.click(within(dlg).getByRole("button", { name: button }));
  await waitFor(() => expect(screen.queryByRole("alertdialog", { name: title })).toBeNull());
  return dlg;
};
