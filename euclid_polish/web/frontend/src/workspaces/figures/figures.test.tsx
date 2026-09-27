/* Figures › Grid / Plates / Results and the `figure` inspector card against a
 * mocked backend (/viewer/results, /viewer/grid-layouts, /api/figures/*,
 * /poster/result/*). */
import { QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactElement } from "react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useJobsStore } from "../../api/jobs";
import { queryClient } from "../../api/query";
import { useInspectorRegistry } from "../../app/inspector";
import { useInspector } from "../../state/inspector";
import { useSelection } from "../../state/selection";
import { resetConfirm, toast } from "../../ui";
import FigureInspector from "./FigureInspector";
import Grid from "./tabs/Grid";
import Plates from "./tabs/Plates";
import Results from "./tabs/Results";
import "./register";

vi.mock("../../fasrc", () => ({ StepById: ({ stepId }: { stepId: string }) => <div data-testid="step">{stepId}</div> }));

type Reply = { status?: number; body: unknown };
let routes: Record<string, (body: Record<string, unknown>) => Reply>;
let posts: { url: string; body: Record<string, unknown> }[];

const bodyOf = (body: BodyInit | null | undefined): Record<string, unknown> => {
  if (body instanceof FormData) {
    const out: Record<string, string> = {};
    body.forEach((v, k) => { out[k] = String(v); });
    return out;
  }
  if (typeof body === "string") { try { return JSON.parse(body); } catch { return {}; } }
  return {};
};

function Probe() {
  const loc = useLocation();
  return <output data-testid="loc">{loc.pathname}{loc.search}</output>;
}

const show = (el: ReactElement, url: string) => render(
  <QueryClientProvider client={queryClient}>
    <MemoryRouter initialEntries={[url]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Routes><Route path="*" element={<>{el}<Probe /></>} /></Routes>
    </MemoryRouter>
  </QueryClientProvider>,
);

const loc = () => decodeURIComponent(screen.getByTestId("loc").textContent ?? "");

const answer = async (title: RegExp | string, button: string) => {
  const dlg = await screen.findByRole("alertdialog", { name: title });
  fireEvent.click(within(dlg).getByRole("button", { name: button }));
  await waitFor(() => expect(screen.queryByRole("alertdialog", { name: title })).toBeNull());
};

const REAL = {
  id: "vr-aaaaaaaaaaaaaaaaaaaaaaaa", label: "NEXUS F200W tile 0040", default_label: "NEXUS F200W tile 0040",
  regime: "real", created_utc: "2026-09-26T10:00:00+00:00",
  source: { collection: "real", index: 40, params: { source: "nexus", models: "rbf" },
    object: { id: "f200w-0040", ref: "nexus/f200w-0040", ra: 268.47, dec: 65.14, label: "NEXUS F200W tile 0040" } },
  selection: { u: 0.5, v: 0.5, mode: "angular", angular_side_arcsec: 1.2 },
  logical_tiers: ["dirty", "sr", "jwst"], bands: { dirty: ["VIS", "Y_E", "J_E", "H_E"] },
  pixscale_arcsec: { dirty: 0.1, sr: 0.05, jwst: 0.03 },
  recipes: ["dirty:VIS", "dirty:H_E", "dirty:VIS_H", "sr:VIS", "sr:H_E", "sr:VIS_H", "jwst:native"],
  thumbnail: "sr:VIS_H", files: { dirty: { shape_hwc: [12, 12, 4], wcs: true, source_label: "LR" }, sr: { wcs: true }, jwst: { wcs: true } },
  bytes: 12345, inspect_paths: { sr: "data/viewer_results/vr-a/sr.fits" },
  wcs_preserved: true, wcs_tiers: ["dirty", "sr", "jwst"], center: { ra: 268.47, dec: 65.14 },
};
const REAL2 = { ...REAL, id: "vr-bbbbbbbbbbbbbbbbbbbbbbbb", label: "tile 42", created_utc: "2026-09-25T10:00:00+00:00",
  recipes: ["dirty:VIS", "sr:VIS"], wcs_preserved: false, wcs_tiers: [], center: null };
const SYN = {
  id: "vr-cccccccccccccccccccccccc", label: "synthetic lens 7", regime: "synthetic", created_utc: "2026-09-24T10:00:00+00:00",
  source: { collection: "evaluation", index: 0, object: { id: "syn-lens-7", label: "synthetic lens 7" } },
  logical_tiers: ["dirty", "sr", "hr"], bands: {}, pixscale_arcsec: {},
  recipes: ["dirty:VIS", "sr:VIS", "hr:VIS", "dirty:H_E", "sr:H_E", "hr:H_E", "sr:VIS_H"],
  wcs_preserved: false, wcs_tiers: [], center: null,
};
const INDEX = { schema_version: 1, limits: { max_results: 12, max_rows: 16 },
  supported: { logical_tiers: ["dirty", "sr", "hr", "jwst"], modes: ["VIS", "H_E", "VIS_H", "native"] },
  results: [REAL, REAL2, SYN] };

const PLATES = {
  root: "output/nexus_comparisons", bands: ["VIS", "Y_E", "J_E", "H_E", "temp"],
  defaults: { tiles: [40, 42], band: "VIS", max_tiles: 24 },
  runs: [{
    tag: "m169-188", updated: "2026-09-21T10:00:00+00:00",
    files: [{ name: "nexus_tiles_temp.png", size: 1000, kind: "sheet", band: "temp", tile_index: null, model_slug: null },
      { name: "nexus_tile040_temp.png", size: 500, kind: "tile", band: "temp", tile_index: 40, model_slug: null }],
    renders: [{ band: "temp", model: null, legacy: true, model_label: "minibatched RBF", sheet: "nexus_tiles_temp.png",
      tiles: [{ index: 40, id: "f200w-0040", ref: "nexus/f200w-0040", file: "nexus_tile040_temp.png", ra_deg: 268.47, dec_deg: 65.14, legacy: true }] }],
  }],
};
const MODELS = { models: [
  { spec: "production", label: "Production · gate", available: true },
  { spec: "rbf", label: "RBF", available: true },
] };
const NEXUS = { tiles: [
  { id: "f200w-0040", ref: "nexus/f200w-0040", models: { rbf: { state: "current" } } },
  { id: "f200w-0042", ref: "nexus/f200w-0042", models: { rbf: { state: "current" }, production: { state: "current" } } },
] };

class FakeImage {}

beforeEach(() => {
  routes = {
    "GET /viewer/results": () => ({ body: INDEX }),
    [`GET /viewer/results/${REAL.id}`]: () => ({ body: { result: REAL } }),
    "GET /viewer/grid-layouts": () => ({ body: { layouts: [{ id: "gl-000000000001", name: "Poster sheet", results: [REAL.id], rows: ["sr:VIS", "jwst:native"], regime: "real" }] } }),
    "POST /viewer/grid-layouts": (b) => ({ status: 201, body: { ok: true, created: true, layout: { id: "gl-000000000002", ...b } } }),
    [`POST /viewer/results/${REAL.id}/delete`]: () => ({ body: { ok: true, id: REAL.id } }),
    [`POST /viewer/results/${REAL.id}/rename`]: (b) => ({ body: { ok: true, result: { ...REAL, label: b.label } } }),
    "GET /api/figures/nexus-plates": () => ({ body: PLATES }),
    "POST /api/figures/nexus-plates": () => ({ body: { ok: true, job_id: "job00001", tag: "rbf-20260926" } }),
    "GET /api/jobs/job00001": () => ({ body: { job_id: "job00001", label: "NEXUS plates", status: "done", result: { tag: "rbf-20260926", band: "VIS", model: "rbf" } } }),
    "GET /api/models": () => ({ body: MODELS }),
    "GET /api/real/nexus": () => ({ body: NEXUS }),
    "GET /poster/result/status": () => ({ body: { ok: true, available: false, png: null, fits: null } }),
    "POST /poster/result/pull": () => ({ status: 503, body: { ok: false, error: "FASRC not connected", code: "fasrc_offline" } }),
    "GET /api/vis/list.json": () => ({ body: { pngs: [] } }),
    "GET /api/fasrc/status": () => ({ body: { ssh_connected: true, connected_at: null, socket: null, last_error: null } }),
    "GET /view/population-atlas?format=png&dpi=150&inline=1": () => ({ status: 404, body: { error: "Euclid joint fit has no publication diagnostics" } }),
  };
  posts = [];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = String(input);
    const method = init.method ?? "GET";
    const body = bodyOf(init.body);
    if (method === "POST") posts.push({ url, body });
    const r = routes[`${method} ${url}`]?.(body) ?? { status: 404, body: { ok: false, error: `no route ${method} ${url}` } };
    return new Response(JSON.stringify(r.body), { status: r.status ?? 200 });
  }));
  vi.stubGlobal("Image", FakeImage);
  queryClient.clear();
  useJobsStore.getState().reset();
  useSelection.getState().clear();
  useInspector.getState().clear();
});
afterEach(() => {
  act(() => resetConfirm());
  queryClient.clear();
  vi.unstubAllGlobals();
});

describe("registration", () => {
  it("registers the figure inspector kind on import", () => {
    const title = useInspectorRegistry.getState().kinds.figure?.title;
    expect(typeof title === "function" ? title("vr-aaaaaaaaaaaaaaaaaaaaaaaa") : title).toMatch(/^Saved result vr-/);
  });
});

describe("Results tab", () => {
  it("lists saved results with source, WCS state and thumbnails; a row opens the figure inspector", async () => {
    show(<Results />, "/figures/results");
    expect(await screen.findByText("tile 42")).toBeTruthy();
    expect(screen.getAllByText("NEXUS F200W tile 0040").length).toBeGreaterThan(0);
    expect(screen.getAllByText("WCS").length).toBe(2);                  // the header + REAL's badge
    expect(screen.getAllByText("no WCS").length).toBe(2);
    const thumb = document.querySelector<HTMLImageElement>(`img[src^="/viewer/results/${REAL.id}/panel.png"]`);
    expect(thumb?.getAttribute("src")).toBe(`/viewer/results/${REAL.id}/panel.png?size=88`);
    fireEvent.click(screen.getByText("tile 42"));
    expect(useInspector.getState().current).toEqual({ kind: "figure", id: REAL2.id });
  });

  it("filters by regime through the URL", async () => {
    show(<Results />, "/figures/results?regime=synthetic");
    expect(await screen.findByText("synthetic lens 7")).toBeTruthy();
    expect(screen.queryByText("tile 42")).toBeNull();
  });

  it("builds a grid from a one-regime selection", async () => {
    show(<Results />, "/figures/results?regime=real");
    await screen.findByText("tile 42");
    const grid = screen.getByRole("grid");
    const boxes = within(grid).getAllByRole("checkbox");
    fireEvent.click(boxes[boxes.length - 1]);
    fireEvent.click(boxes[boxes.length - 2]);
    fireEvent.click(screen.getByRole("button", { name: /^Grid$/ }));
    await waitFor(() => expect(loc()).toMatch(/^\/figures\/grid\?regime=real&cols=vr-/));
    expect(loc().split("cols=")[1].split(",").sort()).toEqual([REAL.id, REAL2.id].sort());
  });

  it("deletes after a danger confirm (singular wording for one result)", async () => {
    show(<Results />, "/figures/results?regime=real");
    await screen.findByText("tile 42");
    const grid = screen.getByRole("grid");
    const row = within(grid).getAllByRole("row").find((r) => r.textContent?.includes("f200w-0040") && r.textContent?.includes("NEXUS F200W tile 0040"))!;
    fireEvent.click(within(row).getByRole("checkbox"));
    fireEvent.click(screen.getByRole("button", { name: /^Delete$/ }));
    const dlg = await screen.findByRole("alertdialog", { name: /Delete “NEXUS F200W tile 0040”\?/ });
    expect(dlg.textContent).toMatch(/Its FITS crops are removed/);
    await answer(/Delete “NEXUS F200W tile 0040”\?/, "Delete");
    await waitFor(() => expect(posts.some((p) => p.url === `/viewer/results/${REAL.id}/delete`)).toBe(true));
  });
});

describe("Grid tab", () => {
  it("previews the grid of the URL's columns and rows", async () => {
    show(<Grid />, `/figures/grid?regime=real&cols=${REAL.id}&rows=dirty:VIS,sr:VIS_H`);
    await waitFor(() => {
      const img = document.querySelector<HTMLImageElement>('img[src^="/viewer/results/grid.png"]');
      expect(img?.getAttribute("src")).toBe(`/viewer/results/grid.png?result=${REAL.id}&row=dirty%3AVIS&row=sr%3AVIS_H&dpi=120&inline=1`);
    }, { timeout: 2000 });
    expect(screen.getByText("2 × 1")).toBeTruthy();
    const png = screen.getByRole("link", { name: /PNG/ });
    expect(png.getAttribute("href")).toContain("dpi=300");
  });

  it("keeps the last preview up (dimmed) while an edit settles and renders", async () => {
    show(<Grid />, `/figures/grid?regime=real&cols=${REAL.id}&rows=dirty:VIS,sr:VIS_H`);
    const first = `/viewer/results/grid.png?result=${REAL.id}&row=dirty%3AVIS&row=sr%3AVIS_H&dpi=120&inline=1`;
    const next = `/viewer/results/grid.png?result=${REAL.id}&row=sr%3AVIS_H&row=dirty%3AVIS&dpi=120&inline=1`;
    const img = await waitFor(() => {
      const el = document.querySelector<HTMLImageElement>(`img[src="${first}"]`);
      expect(el).toBeTruthy();
      return el!;
    });
    fireEvent.load(img);
    expect(screen.queryByText("Updating")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Move row 1 down" }));
    // during the settle delay: the old render stays, marked as updating
    await waitFor(() => expect(screen.getByText("Updating")).toBeTruthy());
    expect(document.querySelector(`img[src="${first}"]`)).toBeTruthy();
    // the settled render loads behind the old one, which stays until it lands
    const fresh = await waitFor(() => {
      const el = document.querySelector<HTMLImageElement>(`img[src="${next}"]`);
      expect(el).toBeTruthy();
      return el!;
    }, { timeout: 2000 });
    expect(document.querySelector(`img[src="${first}"]`)?.className).toBe("is-stale");
    fireEvent.load(fresh);
    await waitFor(() => expect(document.querySelector(`img[src="${first}"]`)).toBeNull());
    expect(screen.queryByText("Updating")).toBeNull();
  });

  it("collapses the preview through the URL", async () => {
    show(<Grid />, `/figures/grid?regime=real&cols=${REAL.id}&rows=dirty:VIS`);
    await waitFor(() => expect(document.querySelector('img[src^="/viewer/results/grid.png"]')).toBeTruthy());
    fireEvent.click(screen.getByRole("button", { name: "Collapse the preview" }));
    await waitFor(() => expect(loc()).toContain("preview=0"));
    expect(document.querySelector('img[src^="/viewer/results/grid.png"]')).toBeNull();
  });

  it("reports cells a column cannot render", async () => {
    show(<Grid />, `/figures/grid?regime=real&cols=${REAL.id},${REAL2.id}&rows=dirty:VIS,jwst:native`);
    expect((await screen.findAllByText("1 cell unavailable")).length).toBeGreaterThan(0);
    expect(document.querySelector('img[src^="/viewer/results/grid.png"]')).toBeNull();
    expect(screen.getByText(/1 missing/)).toBeTruthy();
  });

  it("applies a preset and a saved layout from the template picker", async () => {
    show(<Grid />, "/figures/grid");
    await screen.findByText("tile 42");
    const picker = screen.getByRole("combobox", { name: "Template" });
    fireEvent.change(picker, { target: { value: "synthetic-composite" } });
    await waitFor(() => expect(loc()).toContain("regime=synthetic"));
    expect(loc()).toContain("rows=dirty:VIS_H,sr:VIS_H,hr:VIS_H");
    await screen.findByText("★ Poster sheet");
    fireEvent.change(screen.getByRole("combobox", { name: "Template" }), { target: { value: "layout:gl-000000000001" } });
    await waitFor(() => expect(loc()).toContain(`cols=${REAL.id}`));
    expect(loc()).toContain("rows=sr:VIS,jwst:native");
    expect(loc()).not.toContain("regime=synthetic");                    // real = the default regime
  });

  it("adds the newest compatible crops and saves the layout", async () => {
    show(<Grid />, "/figures/grid");
    await screen.findByText("tile 42");
    fireEvent.click(screen.getByRole("button", { name: "Add newest" }));
    await waitFor(() => expect(loc()).toContain(`cols=${REAL.id}`));
    fireEvent.click(screen.getByRole("button", { name: /^Save$/ }));
    const name = await screen.findByRole("textbox", { name: "Layout name" });
    fireEvent.change(name, { target: { value: "Sheet A" } });
    fireEvent.click(screen.getByRole("button", { name: "Save layout" }));
    await waitFor(() => expect(posts.find((p) => p.url === "/viewer/grid-layouts")?.body).toEqual({
      name: "Sheet A", results: [REAL.id], rows: ["dirty:VIS", "dirty:H_E", "sr:VIS_H", "jwst:native"], regime: "real",
    }));
    await waitFor(() => expect(loc()).toContain("tpl=layout:gl-000000000002"));
  });

  it("edits rows and marks the template custom", async () => {
    show(<Grid />, `/figures/grid?cols=${REAL.id}`);
    await screen.findByText("tile 42");
    fireEvent.click(screen.getByRole("button", { name: "Remove row 1" }));
    await waitFor(() => expect(loc()).toContain("tpl=custom"));
    expect(loc()).toContain("rows=dirty:H_E,sr:VIS_H,jwst:native");
    fireEvent.change(screen.getByRole("combobox", { name: "Row 1 band" }), { target: { value: "VIS" } });
    await waitFor(() => expect(loc()).toContain("rows=dirty:VIS,sr:VIS_H,jwst:native"));
  });
});

describe("Plates tab", () => {
  it("shows the server's error text when a plate cannot render", async () => {
    show(<Plates />, "/figures/plates");
    const img = document.querySelector<HTMLImageElement>('img[src="/view/population-atlas?format=png&dpi=150&inline=1"]');
    expect(img).toBeTruthy();
    expect(screen.getByRole("link", { name: /PDF/ }).getAttribute("href")).toBe("/view/population-atlas?format=pdf&dpi=300");
    fireEvent.error(img!);
    expect(await screen.findByText("Euclid joint fit has no publication diagnostics")).toBeTruthy();
    // nothing to download from a plate the server cannot render
    await waitFor(() => expect(screen.getByRole("button", { name: /PDF/ }).hasAttribute("disabled")).toBe(true));
  });

  it("switches plates through the URL; the galaxy plate carries the training toggle", async () => {
    show(<Plates />, "/figures/plates?plate=galaxies&training=1");
    expect(document.querySelector('img[src="/view/galaxy-distribution-plate?format=png&dpi=150&inline=1&include_training=1"]')).toBeTruthy();
    fireEvent.click(screen.getByRole("radio", { name: "Stars" }));
    await waitFor(() => expect(loc()).toContain("plate=stars"));
  });

  it("renders NEXUS plates as a job, validates model coverage and lists runs with provenance", async () => {
    show(<Plates />, "/figures/plates?plate=nexus&model=production");
    expect(await screen.findByText(/production has not run on 1 tile/)).toBeTruthy();
    expect(screen.getByRole("link", { name: "Run in Experiments" }).getAttribute("href"))
      .toBe("/sky/experiments?tiles=nexus%2Ff200w-0040&models=production");
    expect(screen.getByRole("button", { name: "Render" }).hasAttribute("disabled")).toBe(true);
    // the legacy run with its sheet and tile plates
    expect(await screen.findByText("minibatched RBF")).toBeTruthy();
    expect(document.querySelector('img[src="/api/figures/nexus-plates/m169-188/nexus_tiles_temp.png?thumb=1400"]')).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Enlarge tile 40" }));
    await waitFor(() => expect(loc()).toContain("tile=40"));
    const dialog = await screen.findByRole("dialog", { name: "NEXUS tile 40" });
    // a legacy tile resolved to its real tile links to it
    expect(within(dialog).getByRole("link", { name: "Open the real tile" }).getAttribute("href"))
      .toBe("/sky/results?inspect=realtile%3Anexus%2Ff200w-0040");
  });

  it("posts a render for a covered model", async () => {
    show(<Plates />, "/figures/plates?plate=nexus&model=rbf&band=temp");
    await screen.findByText("minibatched RBF");
    await waitFor(() => expect(screen.getByRole("button", { name: "Render" }).hasAttribute("disabled")).toBe(false));
    fireEvent.click(screen.getByRole("button", { name: "Render" }));
    await waitFor(() => expect(posts.find((p) => p.url === "/api/figures/nexus-plates")?.body).toEqual({
      tiles: "f200w-0040,f200w-0042", band: "temp", model: "rbf", tag: "",
    }));
    await waitFor(() => expect(loc()).toContain("run=rbf-20260926"), { timeout: 4000 });
  });

  it("shows the poster state; a pull asks first and reports the server's failure", async () => {
    const error = vi.spyOn(toast, "error");
    show(<Plates />, "/figures/plates?plate=poster");
    expect(await screen.findByText(/No cutout pulled yet/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Pull latest" }));
    await answer(/Pull the latest poster cutout/, "Cancel");
    expect(posts.some((p) => p.url === "/poster/result/pull")).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Pull latest" }));
    await answer(/Pull the latest poster cutout/, "Pull");
    await waitFor(() => expect(error).toHaveBeenCalledWith(expect.stringMatching(/FASRC is not connected/)));
    fireEvent.click(screen.getByRole("button", { name: "Generate on FASRC" }));
    expect(screen.getByTestId("step").textContent).toBe("poster_cutout");
  });

  it("disables the poster pull while FASRC is offline", async () => {
    routes["GET /api/fasrc/status"] = () => ({ body: { ssh_connected: false, last_error: "socket missing" } });
    show(<Plates />, "/figures/plates?plate=poster");
    await screen.findByText(/No cutout pulled yet/);
    await waitFor(() => expect(screen.getByRole("button", { name: "Pull latest" }).hasAttribute("disabled")).toBe(true));
    fireEvent.click(screen.getByRole("button", { name: "Pull latest" }));
    expect(screen.queryByRole("alertdialog")).toBeNull();
  });
});

describe("figure inspector", () => {
  it("shows the saved result's panels, crop and actions", async () => {
    show(<FigureInspector id={REAL.id} />, "/figures/results");
    expect(await screen.findByText("NEXUS F200W tile 0040")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Open the real tile" }).getAttribute("href"))
      .toBe("/sky/results?inspect=realtile%3Anexus%2Ff200w-0040");
    expect(screen.getByRole("link", { name: /Sky/ }).getAttribute("href")).toBe("/sky/atlas?ra=268.470000&dec=65.140000&fov=0.01");
    expect(screen.getByText(/^1\.20*″$/)).toBeTruthy();
    expect(screen.getAllByRole("button", { pressed: false }).length).toBeGreaterThan(3);
    fireEvent.click(screen.getByRole("button", { name: /VIS Dirty/ }));
    await waitFor(() => expect(document.querySelector(`img[src="/viewer/results/${REAL.id}/panel.png?tier=dirty&mode=VIS&size=512"]`)).toBeTruthy());
    fireEvent.click(screen.getByRole("button", { name: "Rename" }));
    const input = await screen.findByRole("textbox", { name: "Label" });
    fireEvent.change(input, { target: { value: "Lens A" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(posts.find((p) => p.url.endsWith("/rename"))?.body).toEqual({ label: "Lens A" }));
  });

  it("shows the server error for an unknown result", async () => {
    routes[`GET /viewer/results/vr-x`] = () => ({ status: 404, body: { error: "saved viewer result not found" } });
    show(<FigureInspector id="vr-x" />, "/figures/results");
    expect(await screen.findByText("saved viewer result not found")).toBeTruthy();
  });
});
