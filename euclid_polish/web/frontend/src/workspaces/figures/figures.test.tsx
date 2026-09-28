/* Figures › Sheet / Plates and the `figure` inspector card against a mocked
 * backend (/viewer/results, /viewer/grid-layouts, /api/figures/*,
 * /api/realism/overview, /poster/result/*). */
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
import Plates from "./tabs/Plates";
import Sheet from "./tabs/Sheet";
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

const DAY = 86_400_000;
const ago = (days: number) => new Date(Date.now() - days * DAY).toISOString();
const OVERVIEW = { items: [
  { id: "galaxy-model", state: "ok", facts: { version: 15, is_active: true, active_fingerprint: "g1", candidate_fingerprint: "g1", candidate_valid: true },
    records: { prior_at: ago(13) } },
  { id: "star-prior", state: "ok", facts: { is_active: true, active_fingerprint: "s1", candidate_fingerprint: "s1", candidate_valid: true },
    records: { prior_at: ago(46) } },
  { id: "galaxy-plots", state: "warn", facts: { present: true, stale: true, reason: "the plot schema changed since the last build", built_at: ago(0.4) } },
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
    "GET /api/realism/overview": () => ({ body: OVERVIEW }),
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


describe("Sheet tab · the saved-crop pool", () => {
  it("lists the regime's saved crops with source, WCS state and thumbnails; a row opens the figure inspector", async () => {
    show(<Sheet />, "/figures/sheet");
    expect(await screen.findByText("tile 42")).toBeTruthy();
    expect(screen.getAllByText("NEXUS F200W tile 0040").length).toBeGreaterThan(0);
    expect(screen.queryByText("synthetic lens 7")).toBeNull();                 // real is the sheet's regime
    // the counts sit on the regime control, once
    expect(screen.getByRole("radio", { name: "Real 2" })).toBeTruthy();
    expect(screen.getByRole("radio", { name: "Synthetic 1" })).toBeTruthy();
    expect(screen.getAllByText("no WCS").length).toBe(1);
    const thumb = document.querySelector<HTMLImageElement>(`img[src^="/viewer/results/${REAL.id}/panel.png"]`);
    expect(thumb?.getAttribute("src")).toBe(`/viewer/results/${REAL.id}/panel.png?size=112`);
    expect(screen.getByText(/freeze a region with the lens in any viewer and press S/)).toBeTruthy();
    fireEvent.click(screen.getByText("tile 42"));
    expect(useInspector.getState().current).toEqual({ kind: "figure", id: REAL2.id });
  });

  it("puts a ticked crop in the sheet as a column and previews it", async () => {
    show(<Sheet />, "/figures/sheet?rows=dirty:VIS");
    await screen.findByText("tile 42");
    const grid = screen.getByRole("grid", { name: "real saved crops" });
    const row = within(grid).getAllByRole("row").find((r) => r.textContent?.includes("tile 42"))!;
    fireEvent.click(within(row).getByRole("checkbox"));
    await waitFor(() => expect(loc()).toContain(`cols=${REAL2.id}`));
    expect(within(screen.getByRole("list", { name: "Column order" })).getByText("tile 42")).toBeTruthy();
    await waitFor(() => expect(document.querySelector('img[src^="/viewer/results/grid.png"]')?.getAttribute("src"))
      .toBe(`/viewer/results/grid.png?result=${REAL2.id}&row=dirty%3AVIS&dpi=120&inline=1`));
  });

  it("opens a thumbnail full size (every panel, pixels kept) without opening the inspector", async () => {
    show(<Sheet />, "/figures/sheet");
    await screen.findByText("tile 42");
    useInspector.getState().clear();
    fireEvent.click(screen.getByRole("button", { name: "View NEXUS F200W tile 0040 full size" }));
    const dlg = await screen.findByRole("dialog", { name: "NEXUS F200W tile 0040" });
    expect(useInspector.getState().current).toBeNull();
    const shown = () => within(dlg).getByRole("img", { name: /NEXUS F200W tile 0040, / }).getAttribute("src");
    expect(shown()).toBe(`/viewer/results/${REAL.id}/panel.png?tier=sr&mode=VIS_H`);
    fireEvent.click(within(dlg).getByRole("button", { name: "VIS Dirty" }));
    expect(shown()).toBe(`/viewer/results/${REAL.id}/panel.png?tier=dirty&mode=VIS`);
    fireEvent.click(within(dlg).getByRole("button", { name: "Open its card" }));
    expect(useInspector.getState().current).toEqual({ kind: "figure", id: REAL.id });
  });

  it("shows the crops as a gallery (?pool=gallery, where /figures/results lands), newest first, with one find box", async () => {
    show(<Sheet />, "/figures/sheet?pool=gallery");
    await screen.findByText("tile 42");
    const list = screen.getByRole("list", { name: "real saved crops" });
    expect(within(list).getAllByRole("button", { name: /full size$/ }).map((b) => b.getAttribute("aria-label")))
      .toEqual(["View NEXUS F200W tile 0040 full size", "View tile 42 full size"]);    // REAL is the newer crop
    expect(list.querySelectorAll("img")[1]?.getAttribute("src")).toBe(`/viewer/results/${REAL2.id}/panel.png?size=300`);
    fireEvent.click(within(list).getByRole("checkbox", { name: "Put tile 42 in the sheet" }));
    await waitFor(() => expect(loc()).toContain(`cols=${REAL2.id}`));
    fireEvent.change(screen.getByRole("searchbox", { name: "Find saved crops" }), { target: { value: "tile vr-aaaa" } });
    await waitFor(() => expect(loc()).toMatch(/q=tile(\+| )vr-aaaa/));
    expect(within(list).getAllByRole("button", { name: /full size$/ }).map((b) => b.getAttribute("aria-label")))
      .toEqual(["View NEXUS F200W tile 0040 full size"]);
  });

  it("honours the old /figures/results ?view=table over the redirect's ?pool=gallery, once", async () => {
    show(<Sheet />, "/figures/sheet?view=table&pool=gallery");
    await screen.findByText("tile 42");
    expect(screen.queryByRole("list", { name: "real saved crops" })).toBeNull();
    fireEvent.click(within(screen.getByRole("radiogroup", { name: "Show the crops as" })).getByRole("radio", { name: /Gallery/ }));
    await waitFor(() => expect(screen.getByRole("list", { name: "real saved crops" })).toBeTruthy());
    expect(loc()).not.toContain("view=");
    expect(loc()).toContain("pool=gallery");
  });

  it("returns focus to the thumbnail that opened the full-size view", async () => {
    show(<Sheet />, "/figures/sheet?pool=gallery");
    await screen.findByText("tile 42");
    const open = screen.getByRole("button", { name: "View tile 42 full size" });
    open.focus();
    fireEvent.click(open);
    const dlg = await screen.findByRole("dialog", { name: "tile 42" });
    fireEvent.keyDown(dlg, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "tile 42" })).toBeNull());
    await waitFor(() => expect(document.activeElement).toBe(open));
  });

  it("switches the regime through the URL", async () => {
    show(<Sheet />, "/figures/sheet?regime=synthetic");
    expect(await screen.findByText("synthetic lens 7")).toBeTruthy();
    expect(screen.queryByText("tile 42")).toBeNull();
  });

  it("opens a crop's source, the sky and Files from its menu, and deletes after a danger confirm", async () => {
    show(<Sheet />, "/figures/sheet");
    await screen.findByText("tile 42");
    fireEvent.pointerDown(screen.getAllByRole("button", { name: "Actions for NEXUS F200W tile 0040" })[0], { button: 0, pointerType: "mouse" });
    const menu = await screen.findByRole("menu");
    expect(within(menu).getByRole("menuitem", { name: "Open the real tile" })).toBeTruthy();
    expect(within(menu).getByRole("menuitem", { name: "Show on sky" })).toBeTruthy();
    expect(within(menu).getByRole("menuitem", { name: /Open in Files/ })).toBeTruthy();
    fireEvent.click(within(menu).getByRole("menuitem", { name: "Delete…" }));
    const dlg = await screen.findByRole("alertdialog", { name: /Delete “NEXUS F200W tile 0040”\?/ });
    expect(dlg.textContent).toMatch(/Its FITS crops are removed/);
    await answer(/Delete “NEXUS F200W tile 0040”\?/, "Delete");
    await waitFor(() => expect(posts.some((p) => p.url === `/viewer/results/${REAL.id}/delete`)).toBe(true));
  });

  it("deletes the ticked crops together after one danger confirm, and drops them from the sheet", async () => {
    routes[`POST /viewer/results/${REAL2.id}/delete`] = () => ({ body: { ok: true, id: REAL2.id } });
    show(<Sheet />, `/figures/sheet?cols=${REAL.id},${REAL2.id}`);
    fireEvent.click(await screen.findByRole("button", { name: "Delete the 2 ticked…" }));
    const dlg = await screen.findByRole("alertdialog", { name: "Delete 2 saved results?" });
    expect(dlg.textContent).toMatch(/Their FITS crops are removed/);
    expect(posts).toEqual([]);
    await answer("Delete 2 saved results?", "Delete");
    await waitFor(() => expect(posts.map((p) => p.url)).toEqual([`/viewer/results/${REAL.id}/delete`, `/viewer/results/${REAL2.id}/delete`]));
    await waitFor(() => expect(loc()).not.toContain("cols="));
  });

  it("reads the table's Find box as the filter language, with hidden columns and a CSV export", async () => {
    show(<Sheet />, "/figures/sheet?q=wcs=none");
    await screen.findByText("tile 42");
    const grid = screen.getByRole("grid", { name: "real saved crops" });
    await waitFor(() => expect(within(grid).queryByText("NEXUS F200W tile 0040")).toBeNull());
    expect(within(grid).getByText("tile 42")).toBeTruthy();
    expect(screen.getByRole("button", { name: /^CSV/ })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Columns" })).toBeTruthy();
    // one Find box: the table has no second filter
    expect(screen.queryByRole("searchbox", { name: "Filter rows" })).toBeNull();
  });

  it("names the column limit only once it is reached", async () => {
    routes["GET /viewer/results"] = () => ({ body: { ...INDEX, limits: { max_results: 2, max_rows: 16 } } });
    show(<Sheet />, `/figures/sheet?cols=${REAL.id}`);
    await screen.findByText("tile 42");
    expect(screen.queryByText(/the most a sheet holds/)).toBeNull();
    const grid = screen.getByRole("grid", { name: "real saved crops" });
    fireEvent.click(within(within(grid).getAllByRole("row").find((r) => r.textContent?.includes("tile 42"))!).getByRole("checkbox"));
    expect(await screen.findByText("2 columns: the most a sheet holds")).toBeTruthy();
    expect(screen.getByText(/The sheet is full/)).toBeTruthy();
  });
});

describe("Sheet tab · rows and the live preview", () => {
  it("previews the sheet of the URL's columns and rows", async () => {
    show(<Sheet />, `/figures/sheet?regime=real&cols=${REAL.id}&rows=dirty:VIS,sr:VIS_H`);
    await waitFor(() => {
      const img = document.querySelector<HTMLImageElement>('img[src^="/viewer/results/grid.png"]');
      expect(img?.getAttribute("src")).toBe(`/viewer/results/grid.png?result=${REAL.id}&row=dirty%3AVIS&row=sr%3AVIS_H&dpi=120&inline=1`);
    }, { timeout: 2000 });
    expect(screen.getByText("2 × 1 ready")).toBeTruthy();
    const png = screen.getByRole("link", { name: /PNG/ });
    expect(png.getAttribute("href")).toContain("dpi=300");
  });

  it("explains the colours the preview uses: one band grey, the composite VIS azure + H_E amber", async () => {
    show(<Sheet />, `/figures/sheet?cols=${REAL.id}&rows=dirty:VIS,sr:VIS_H`);
    const legend = await screen.findByRole("list", { name: "Colours in the sheet" });
    expect(within(legend).getAllByRole("listitem").map((li) => li.textContent)).toEqual(["one band: grey", "VIS + H_E: VIS azure, H_E amber"]);
    expect([...legend.querySelectorAll("i")].map((i) => i.getAttribute("data-tone"))).toEqual(["grey", "vis", "h"]);
    // the rows carry the same colour as their marker
    expect(document.querySelectorAll('.fig-row[data-tone="grey"]').length).toBe(1);
    expect(document.querySelectorAll('.fig-row[data-tone="vis-h"]').length).toBe(1);
  });

  it("offers the Y and J bands once the backend lists them", async () => {
    routes["GET /viewer/results"] = () => ({ body: { ...INDEX, supported: { ...INDEX.supported, modes: ["VIS", "Y_E", "J_E", "H_E", "VIS_H", "native"] } } });
    show(<Sheet />, `/figures/sheet?cols=${REAL.id}&rows=sr:VIS`);
    await screen.findByText("tile 42");
    const band = await screen.findByRole("combobox", { name: "Row 1 band" });
    expect([...band.querySelectorAll("option")].map((o) => o.textContent)).toEqual(["VIS", "Y_E", "J_E", "H_E", "VIS + H_E", "native band"]);
    fireEvent.change(band, { target: { value: "J_E" } });
    await waitFor(() => expect(loc()).toContain("rows=sr:J_E"));
  });

  it("opens the sheet full size (a sharper render) from the preview, with Fit / Fit width / Actual size", async () => {
    show(<Sheet />, `/figures/sheet?regime=real&cols=${REAL.id}&rows=dirty:VIS,sr:VIS_H`);
    fireEvent.click(await screen.findByRole("button", { name: "View the sheet full size" }));
    const dlg = await screen.findByRole("dialog", { name: "Figure sheet" });
    const img = within(dlg).getByRole("img", { name: /Figure sheet, 2 rows × 1 column$/ });
    expect(dlg.textContent).toContain("2 rows × 1 column · rendered at 200 dpi");
    expect(img.getAttribute("src")).toBe(`/viewer/results/grid.png?result=${REAL.id}&row=dirty%3AVIS&row=sr%3AVIS_H&dpi=200&inline=1`);
    fireEvent.click(within(dlg).getByRole("radio", { name: "Fit width" }));
    expect(dlg.querySelector(".fig-lightbox__stage")?.getAttribute("data-scale")).toBe("width");
    fireEvent.click(within(dlg).getByRole("radio", { name: "Actual size" }));
    expect(dlg.querySelector(".fig-lightbox__stage")?.getAttribute("data-scale")).toBe("actual");
    expect(within(dlg).getByRole("link", { name: /PNG · 300 dpi/ }).getAttribute("href")).toContain("dpi=300");
    fireEvent.click(within(dlg).getByRole("button", { name: "Close" }));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Figure sheet" })).toBeNull());
  });

  it("keeps the last preview up (dimmed) while an edit settles and renders", async () => {
    show(<Sheet />, `/figures/sheet?regime=real&cols=${REAL.id}&rows=dirty:VIS,sr:VIS_H`);
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
    await waitFor(() => expect(screen.getByText("Updating")).toBeTruthy());
    expect(document.querySelector(`img[src="${first}"]`)).toBeTruthy();
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
    show(<Sheet />, `/figures/sheet?regime=real&cols=${REAL.id}&rows=dirty:VIS`);
    await waitFor(() => expect(document.querySelector('img[src^="/viewer/results/grid.png"]')).toBeTruthy());
    fireEvent.click(screen.getByRole("button", { name: "Collapse the preview" }));
    await waitFor(() => expect(loc()).toContain("preview=0"));
    expect(document.querySelector('img[src^="/viewer/results/grid.png"]')).toBeNull();
  });

  it("draws every available cell and marks the one a column cannot render in place", async () => {
    show(<Sheet />, `/figures/sheet?regime=real&cols=${REAL.id},${REAL2.id}&rows=dirty:VIS,jwst:native`);
    expect(await screen.findByText("1 cell not available (grey in the sheet)")).toBeTruthy();
    await waitFor(() => expect(document.querySelector('img[src^="/viewer/results/grid.png"]')).not.toBeNull());
    const src = document.querySelector('img[src^="/viewer/results/grid.png"]')!.getAttribute("src")!;
    expect(src).toContain("row=dirty%3AVIS");
    expect(src).toContain("row=jwst%3Anative");
    expect(src).toContain("missing=blank");
    expect(screen.getByRole("link", { name: "PNG" }).getAttribute("href")).toContain("missing=blank");
    expect(screen.getByText(/1 missing/)).toBeTruthy();
    expect(screen.getByText(/lacks 1 row/)).toBeTruthy();                        // the pool marks the crop
    expect(screen.getByRole("button", { name: "Only the rows every column has" })).toBeTruthy();
  });

  it("applies a preset and a saved layout from the template picker", async () => {
    show(<Sheet />, "/figures/sheet");
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
    show(<Sheet />, "/figures/sheet");
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
    show(<Sheet />, `/figures/sheet?cols=${REAL.id}`);
    await screen.findByText("tile 42");
    fireEvent.click(screen.getByRole("button", { name: "Remove row 1" }));
    await waitFor(() => expect(loc()).toContain("tpl=custom"));
    expect(loc()).toContain("rows=dirty:H_E,sr:VIS_H,jwst:native");
    fireEvent.change(screen.getByRole("combobox", { name: "Row 1 band" }), { target: { value: "VIS" } });
    await waitFor(() => expect(loc()).toContain("rows=dirty:VIS,sr:VIS_H,jwst:native"));
  });
});

describe("Plates tab", () => {
  it("names the plates after their titles, in the spec's order", async () => {
    show(<Plates />, "/figures/plates");
    const picker = screen.getByRole("radiogroup", { name: "Plate" });
    expect(within(picker).getAllByRole("radio").map((r) => r.textContent)).toEqual([
      "Galaxy population calibration", "Galaxy distributions", "Stellar population calibration", "NEXUS comparison", "Synthetic poster scene",
    ]);
  });

  it("captions a calibration plate with what it is made with, whether that is current and when", async () => {
    show(<Plates />, "/figures/plates");
    expect(await screen.findByText(/made with galaxy model v15/)).toBeTruthy();
    expect(document.querySelector(".fig-plate__caption")?.textContent).toMatch(/^made with galaxy model v15 · current · activated \d+ d ago$/);
    fireEvent.click(screen.getByRole("radio", { name: "Galaxy distributions" }));
    await waitFor(() => expect(loc()).toContain("plate=galaxies"));
    const cap = await screen.findByText("stale");
    expect(cap.closest(".fig-plate__caption")?.textContent).toMatch(/^made with galaxy model v15 · stale · built \d+ (h|d) ago$/);
  });

  it("shows the server's error text when a plate cannot render", async () => {
    show(<Plates />, "/figures/plates");
    const img = document.querySelector<HTMLImageElement>('img[src="/view/population-atlas?format=png&dpi=150&inline=1"]');
    expect(img).toBeTruthy();
    expect(screen.getByRole("link", { name: /PDF/ }).getAttribute("href")).toBe("/view/population-atlas?format=pdf&dpi=300");
    fireEvent.error(img!);
    expect(await screen.findByText("Euclid joint fit has no publication diagnostics")).toBeTruthy();
    await waitFor(() => expect(screen.getByRole("button", { name: /PDF/ }).hasAttribute("disabled")).toBe(true));
  });

  it("switches plates through the URL; the galaxy plate carries the training toggle and the resolution", async () => {
    show(<Plates />, "/figures/plates?plate=galaxies&training=1");
    expect(document.querySelector('img[src="/view/galaxy-distribution-plate?format=png&dpi=150&inline=1&include_training=1"]')).toBeTruthy();
    fireEvent.change(screen.getByRole("combobox", { name: "Download resolution" }), { target: { value: "600" } });
    await waitFor(() => expect(screen.getByRole("link", { name: /PNG/ }).getAttribute("href")).toContain("dpi=600"));
    fireEvent.click(screen.getByRole("radio", { name: "Stellar population calibration" }));
    await waitFor(() => expect(loc()).toContain("plate=stars"));
  });

  it("defaults the NEXUS model to the production gate and waits for the tile list before calling a tile unknown", async () => {
    let release: () => void = () => {};
    const gate = new Promise<void>((r) => { release = r; });
    const reply = routes["GET /api/real/nexus"];
    routes["GET /api/real/nexus"] = () => ({ body: NEXUS });
    vi.mocked(fetch).mockImplementation(async (input: RequestInfo | URL, init: RequestInit = {}) => {
      const url = String(input);
      const method = init.method ?? "GET";
      if (url === "/api/real/nexus") await gate;
      const r = routes[`${method} ${url}`]?.(bodyOf(init.body)) ?? { status: 404, body: { ok: false, error: `no route ${method} ${url}` } };
      return new Response(JSON.stringify(r.body), { status: r.status ?? 200 });
    });
    show(<Plates />, "/figures/plates?plate=nexus");
    await screen.findByText(/made with minibatched RBF/);
    expect(screen.queryByText(/Unknown:/)).toBeNull();                          // the tile list is still loading
    expect(screen.getByRole("button", { name: "Render" }).hasAttribute("disabled")).toBe(true);
    act(() => release());
    // then production, the default, is missing on tile 40
    expect(await screen.findByText(/production has not run on 1 tile/)).toBeTruthy();
    routes["GET /api/real/nexus"] = reply;
  });

  it("renders NEXUS plates as a job, validates model coverage and lists runs with a one-line caption", async () => {
    show(<Plates />, "/figures/plates?plate=nexus&model=production");
    expect(await screen.findByText(/production has not run on 1 tile/)).toBeTruthy();
    expect(screen.getByRole("link", { name: "Run in Sky › Compare" }).getAttribute("href"))
      .toBe("/sky/compare?tiles=nexus%2Ff200w-0040&models=production");
    expect(screen.getByRole("button", { name: "Render" }).hasAttribute("disabled")).toBe(true);
    const caption = await screen.findByText(/made with minibatched RBF \(legacy SR\)/);
    expect(caption.closest(".fig-plate__caption")?.textContent).toMatch(/^temperature colour · 1 tile · made with minibatched RBF \(legacy SR\) · stale · rendered \d+ d ago$/);
    expect(document.querySelector('img[src="/api/figures/nexus-plates/m169-188/nexus_tiles_temp.png?thumb=1400"]')).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Enlarge tile 40" }));
    await waitFor(() => expect(loc()).toContain("tile=40"));
    const dialog = await screen.findByRole("dialog", { name: "NEXUS tile 40" });
    expect(within(dialog).getByRole("link", { name: "Open the real tile" }).getAttribute("href"))
      .toBe("/sky/targets?inspect=realtile%3Anexus%2Ff200w-0040");
  });

  it("offers a rendered run at 150 / 300 / 600 dpi as PNG, PDF or SVG (a legacy run keeps its PNG)", async () => {
    routes["GET /api/figures/nexus-plates"] = () => ({ body: { ...PLATES, runs: [{
      tag: "prod-run", updated: "2026-09-27T10:00:00+00:00", files: [],
      renders: [{ band: "VIS", model: "production", model_label: "Production · gate", model_fingerprint: "f1", created: "2026-09-27T10:00:00+00:00",
        sheet: "nexus_tiles_VIS__production.png",
        tiles: [{ index: 42, id: "f200w-0042", ref: "nexus/f200w-0042", file: "nexus_tile042_VIS__production.png" }] }],
    }, ...PLATES.runs] } });
    show(<Plates />, "/figures/plates?plate=nexus&dpi=600");
    const pdf = await screen.findByRole("link", { name: "PDF" });
    expect(pdf.getAttribute("href")).toBe("/api/figures/nexus-plates/prod-run/export?band=VIS&model=production&format=pdf&dpi=600");
    expect(screen.getByRole("link", { name: "SVG" }).getAttribute("href")).toContain("format=svg&dpi=600");
    fireEvent.click(screen.getByRole("button", { name: "Enlarge tile 42" }));
    const dialog = await screen.findByRole("dialog", { name: "NEXUS tile 42" });
    expect(within(dialog).getByRole("link", { name: "PNG" }).getAttribute("href"))
      .toBe("/api/figures/nexus-plates/prod-run/export?band=VIS&model=production&format=png&dpi=600&tile=42");
  });

  it("asks before it renders plates for a covered model", async () => {
    show(<Plates />, "/figures/plates?plate=nexus&model=rbf&band=temp");
    await screen.findByText(/minibatched RBF/);
    await waitFor(() => expect(screen.getByRole("button", { name: "Render" }).hasAttribute("disabled")).toBe(false));
    fireEvent.click(screen.getByRole("button", { name: "Render" }));
    await answer(/Render 2 NEXUS plates with RBF\?/, "Cancel");
    expect(posts.some((p) => p.url === "/api/figures/nexus-plates")).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Render" }));
    await answer(/Render 2 NEXUS plates with RBF\?/, "Render");
    await waitFor(() => expect(posts.find((p) => p.url === "/api/figures/nexus-plates")?.body).toEqual({
      tiles: "f200w-0040,f200w-0042", band: "temp", model: "rbf", tag: "",
    }));
    await waitFor(() => expect(loc()).toContain("run=rbf-20260926"), { timeout: 4000 });
  });

  it("shows the poster scene; a pull asks first and reports the server's failure; the step is in its drawer", async () => {
    const error = vi.spyOn(toast, "error");
    show(<Plates />, "/figures/plates?plate=poster");
    expect(await screen.findByText(/No scene pulled yet/)).toBeTruthy();
    expect(screen.getByRole("link", { name: "Steps" }).getAttribute("href")).toBe("/runs/steps?step=poster_cutout");
    fireEvent.click(screen.getByRole("button", { name: "Pull latest" }));
    await answer(/Pull the latest poster cutout/, "Cancel");
    expect(posts.some((p) => p.url === "/poster/result/pull")).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Pull latest" }));
    await answer(/Pull the latest poster cutout/, "Pull");
    await waitFor(() => expect(error).toHaveBeenCalledWith(expect.stringMatching(/FASRC is not connected — connect in System › Connections/)));
    expect(screen.queryByTestId("step")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "How this is produced" }));
    await waitFor(() => expect(loc()).toContain("how=1"));
    expect(screen.getByTestId("step").textContent).toBe("poster_cutout");
  });

  it("offers the pulled poster scene for print at a chosen dpi", async () => {
    routes["GET /poster/result/status"] = () => ({ body: { ok: true, available: true, png: { size: 10, mtime: 1, pulled_at: "x" }, fits: { size: 20, mtime: 1, pulled_at: "x" } } });
    show(<Plates />, "/figures/plates?plate=poster&dpi=150");
    const pdf = await screen.findByRole("link", { name: "PDF" });
    expect(pdf.getAttribute("href")).toBe("/poster/result/export?format=pdf&dpi=150");
    expect(screen.getByRole("link", { name: "SVG" }).getAttribute("href")).toBe("/poster/result/export?format=svg&dpi=150");
    expect(screen.getByRole("link", { name: "FITS" }).getAttribute("href")).toBe("/poster/result/cutout.fits");
  });

  it("disables the poster pull while FASRC is offline", async () => {
    routes["GET /api/fasrc/status"] = () => ({ body: { ssh_connected: false, last_error: "socket missing" } });
    show(<Plates />, "/figures/plates?plate=poster");
    await screen.findByText(/No scene pulled yet/);
    await waitFor(() => expect(screen.getByRole("button", { name: "Pull latest" }).hasAttribute("disabled")).toBe(true));
    fireEvent.click(screen.getByRole("button", { name: "Pull latest" }));
    expect(screen.queryByRole("alertdialog")).toBeNull();
  });
});

describe("figure inspector", () => {
  it("shows the saved result's panels, crop and actions", async () => {
    show(<FigureInspector id={REAL.id} />, "/figures/sheet");
    expect(await screen.findByText("NEXUS F200W tile 0040")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Open the real tile" }).getAttribute("href"))
      .toBe("/sky/targets?inspect=realtile%3Anexus%2Ff200w-0040");
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
    show(<FigureInspector id="vr-x" />, "/figures/sheet");
    expect(await screen.findByText("saved viewer result not found")).toBeTruthy();
  });
});
