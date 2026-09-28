/* Files (was Inspect) against a mocked Flask: browser → file → HDUs → views,
 * the per-frame statistics, table paging, header filter, track, errors, the
 * `?path=` alias and the `fits` inspector kind. */
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { queryClient } from "../../api/query";
import { useInspectorRegistry } from "../../app/inspector";
import { resetConfirm } from "../../ui";
import type { BrowseResponse, InspectResponse, TablePage } from "./api";
import FitsInspector from "./FitsInspector";
import FilesPage from "./FilesPage";
import { unregisterFitsInspector } from "./register";

type Reply = { status?: number; body: unknown };
let routes: Record<string, (url: URL, init: RequestInit) => Reply>;
let calls: { url: string; method: string; body?: string }[];
let location = "";

const ROOT = { id: "eval", label: "Evaluation results", path: "/r/data/eval_results", rel: "data/eval_results", exists: true };

const ROOTS: BrowseResponse = {
  dir: null, root: null, crumbs: [], query: "", roots: [ROOT], truncated: false, other: 0,
  entries: [{ name: "Evaluation results", rel: "data/eval_results", kind: "root", size: null, mtime: null, root_id: "eval", exists: true }],
};
const LISTING: BrowseResponse = {
  dir: "data/eval_results", root: ROOT, crumbs: [{ name: "Evaluation results", rel: "data/eval_results" }],
  query: "", roots: [ROOT], truncated: false, other: 2,
  entries: [
    { name: "gal_1", rel: "data/eval_results/gal_1", kind: "dir", size: null, mtime: 1_790_000_000 },
    { name: "cat.fits", rel: "data/eval_results/cat.fits", kind: "fits", size: 20480, mtime: 1_790_000_000 },
  ],
};

const FILE = "data/eval_results/cat.fits";
const SUMMARY: InspectResponse = {
  file: { abspath: "/r/data/eval_results/cat.fits", basename: "cat.fits", size: 20480, size_kb: 20, mtime: 1_790_000_000, compressed: false },
  hdus: [
    { index: 0, hdu_index: 0, name: "PRIMARY", kind: "PrimaryHDU", type: "image", shape: [4, 16, 20], dtype: ">f4", ndim: 3,
      planes: 4, plane_axes: [4], bunit: "electron", bands: ["VIS", "Y_E", "J_E", "H_E"], bands_assumed: false, viewable: true,
      wcs: { ctype: ["RA---TAN", "DEC--TAN"], ra: 150, dec: 2, pixscale_arcsec: 0.1, width_arcsec: 2, height_arcsec: 1.6,
        fov_deg: 2 / 3600, corners: [], constructed: false },
      cards: [["SIMPLE", "True", "conforms"], ["CRVAL1", "150.0", "ref RA"], ["BUNIT", "electron", ""]] },
    { index: 1, hdu_index: 1, name: "CAT", kind: "BinTableHDU", type: "table", shape: null, dtype: null, viewable: true,
      nrows: 450, ncols: 2, columns: [{ name: "id", format: "J", unit: null, dim: null, null: null }, { name: "flux", format: "E", unit: "e-", dim: null, null: null }],
      cards: [["XTENSION", "BINTABLE", ""], ["TTYPE1", "id", ""]] },
  ],
  band_groups: [], scan_truncated: false, stamp: { id: "aaaaaaaa", produced_by: "bbbbbbbb", parents: [], schema_version: 3 },
  rel: FILE, root: ROOT, allowed_roots: [], roots: [ROOT],
};

function tablePage(url: URL): TablePage {
  const offset = Number(url.searchParams.get("offset") ?? 0);
  const limit = Number(url.searchParams.get("limit") ?? 200);
  const rows: unknown[][] = [];
  for (let i = offset; i < Math.min(450, offset + 3); i++) rows.push([i, i * 1.5]);
  return {
    hdu: 1, total: 450, offset, limit, sort: url.searchParams.get("sort"), desc: url.searchParams.get("desc") === "1",
    columns: [{ name: "id", format: "J", unit: null, dim: null, null: null, kind: "numeric" },
      { name: "flux", format: "E", unit: "e-", dim: null, null: null, kind: "numeric" }],
    rows, row_index: rows.map((r) => r[0] as number),
  };
}

function LocationProbe() {
  const loc = useLocation();
  location = `${loc.pathname}${loc.search}`;
  return null;
}

const show = (url: string) => render(
  <QueryClientProvider client={queryClient}>
    <MemoryRouter initialEntries={[url]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <FilesPage />
      <LocationProbe />
    </MemoryRouter>
  </QueryClientProvider>,
);

const params = () => new URLSearchParams(location.split("?")[1] ?? "");

beforeEach(() => {
  calls = [];
  localStorage.clear();
  routes = {
    "GET /api/inspect/browse": (u) => ({ body: u.searchParams.get("dir") ? LISTING : ROOTS }),
    "GET /api/inspect": (u) => (u.searchParams.get("fits") === FILE ? { body: SUMMARY }
      : { status: 404, body: { error: `no such FITS file: ${u.searchParams.get("fits")}` } }),
    "GET /api/inspect/image/stats": (u) => ({ body: {
      n: 320, n_finite: 320, n_nan: 0, n_posinf: 0, n_neginf: 0, n_zero: 1, n_negative: 0,
      min: 0, max: 319, mean: 159.5, std: 92.4, median: 159.5, mad_std: 118.6, sum: 51040,
      percentiles: { "1": 3.19, "99": 315.8 }, histogram: { edges: [0, 160, 320], counts: [160, 160], below: 0, above: 0 },
      hdu: Number(u.searchParams.get("hdu")), plane: Number(u.searchParams.get("plane") ?? 0), shape: [16, 20], sampled: null,
    } }),
    "GET /api/inspect/table": (u) => ({ body: tablePage(u) }),
    "GET /api/inspect/table/stats": () => ({ body: { hdu: 1, total: 450, sampled: null, columns: [
      { name: "id", kind: "numeric", n: 450, n_null: 0, min: 0, max: 449, median: 224.5, std: 130, n_unique: 450,
        histogram: { edges: [0, 225, 450], counts: [225, 225], below: 0, above: 0 } },
      { name: "flux", unit: "e-", kind: "numeric", n: 450, n_null: 3, min: 0, max: 673.5, median: 336, std: 195 },
    ] } }),
    "GET /api/inspect/provenance": () => ({ body: {
      stamp: SUMMARY.stamp, stale_sidecars: 0, related: [],
      sidecars: [{ file: "data/eval_results/aaaaaaaa.srcutoutartifact.json", id: "aaaaaaaa", kind: "srcutoutartifact", current: true,
        record: { id: "aaaaaaaa", created_at: "2026-07-26T16:25:02Z", git: { short: "e969277", dirty: true } } }],
    } }),
    "GET /viewer/meta/fits": () => ({ status: 404, body: { error: "viewer offline in tests" } }),
    "POST /api/tracking/backup": () => ({ body: { ok: true, record: { name: "cat" }, warning: null, sync: { ok: true } } }),
  };
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = new URL(String(input), "http://localhost");
    const method = init.method ?? "GET";
    const body = init.body instanceof FormData ? new URLSearchParams(init.body as unknown as Record<string, string>).toString() : undefined;
    calls.push({ url: `${url.pathname}${url.search}`, method, body });
    const r = routes[`${method} ${url.pathname}`]?.(url, init) ?? { status: 404, body: { error: `no route ${url.pathname}` } };
    return new Response(JSON.stringify(r.body), { status: r.status ?? 200, headers: { "Content-Type": "application/json" } });
  }));
  queryClient.clear();
});
afterEach(() => { queryClient.clear(); resetConfirm(); vi.unstubAllGlobals(); });

describe("file browser", () => {
  it("starts at the roots, opens a folder and then a file", async () => {
    show("/inspect");
    expect(await screen.findByText("Open a FITS file")).toBeTruthy();
    // each root says which pipeline stage it belongs to
    expect((await screen.findByTitle("data/eval_results")).textContent).toBe("Evaluation · data/eval_results");
    fireEvent.click(await screen.findByText("Evaluation results"));
    await waitFor(() => expect(params().get("dir")).toBe("data/eval_results"));
    expect(await screen.findByText("gal_1")).toBeTruthy();
    expect(screen.getByText("+2 other files")).toBeTruthy();
    fireEvent.click(screen.getByText("cat.fits"));
    await waitFor(() => expect(params().get("fits")).toBe(FILE));
    expect(await screen.findByRole("heading", { name: "cat.fits" })).toBeTruthy();
  });

  it("filters as you type and deep-searches on Enter", async () => {
    show("/inspect?dir=data/eval_results");
    await screen.findByText("gal_1");
    const box = screen.getByRole("textbox", { name: /Filter this folder/ });
    fireEvent.change(box, { target: { value: "cat" } });
    await waitFor(() => expect(screen.queryByText("gal_1")).toBeNull());
    fireEvent.keyDown(box, { key: "Enter" });
    await waitFor(() => expect(params().get("q")).toBe("cat"));
    expect(calls.some((c) => c.url === "/api/inspect/browse?dir=data%2Feval_results&q=cat")).toBe(true);
  });

  it("names a loading search's folder and hides the row count until it lands", async () => {
    const base = globalThis.fetch;
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL, init?: RequestInit) =>
      (String(input).includes("q=") ? new Promise<Response>(() => {}) : base(input, init))));
    show("/inspect?dir=data/eval_results&q=cat");
    const status = await screen.findByRole("status");
    await waitFor(() => expect(status.textContent).toContain("under Evaluation results"));
    expect(status.textContent).not.toContain("all roots");
    expect(screen.queryByText(/^0 rows$/)).toBeNull();
  });
});

describe("layout", () => {
  it("folds the browser by the host's measured width, not the viewport", async () => {
    let width = 1440;
    const observers: (() => void)[] = [];
    vi.stubGlobal("ResizeObserver", class {
      constructor(private cb: () => void) {}
      observe() { observers.push(this.cb); }
      disconnect() {}
      unobserve() {}
    });
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (this: HTMLElement) {
      return { width: this.classList.contains("insp-host") ? width : 0, height: 0, x: 0, y: 0, top: 0, left: 0, right: 0, bottom: 0, toJSON: () => ({}) } as DOMRect;
    });
    const { container } = show(`/inspect?fits=${FILE}`);
    await screen.findByRole("heading", { name: "cat.fits" });
    // wide host: the browser sits beside the file (two-column class on)
    expect(container.querySelector(".insp")?.classList.contains("insp--browser")).toBe(true);
    expect(container.querySelector(".insp-host .insp__browser")).toBeTruthy();
    // the host narrows below 1000 px: the browser folds away above the file
    width = 957;
    act(() => observers.forEach((cb) => cb()));
    await waitFor(() => expect(container.querySelector(".insp__browser")).toBeNull());
    fireEvent.click(screen.getByRole("button", { name: "Show files" }));
    expect(container.querySelector(".insp__browser")).toBeTruthy();
  });
});

describe("a file", () => {
  it("lists HDUs, opens the image view and requests the viewer + stats", async () => {
    show(`/inspect?fits=${encodeURIComponent(FILE)}`);
    const hdus = await screen.findByRole("grid", { name: "HDUs" });
    expect(within(hdus).getByText("PRIMARY")).toBeTruthy();
    expect(within(hdus).getByText("20 × 16 × 4")).toBeTruthy();
    // the file's facts are one muted line in the file bar
    expect(screen.getByText(/^20 KiB · 2 HDUs · modified .* · PROVID aaaaaaaa$/)).toBeTruthy();
    // the image view: viewer meta with the file + HDU, statistics of the VIS band
    expect(await screen.findByText("viewer offline in tests")).toBeTruthy();
    expect(calls.some((c) => c.url.startsWith("/viewer/meta/fits?path=data%2Feval_results%2Fcat.fits&hdu=0"))).toBe(true);
    // one statistics row per shown frame: median, σ (MAD), p99, Σ flux (no
    // non-finite column: every pixel is finite); no min/max, no preview card
    const table = await screen.findByRole("table", { name: "Statistics of the frames shown" });
    const row = await within(table).findByRole("row", { name: /PRIMARY · VIS/ });
    // the colour cube has one row per band: VIS, Y, J, H
    expect(within(table).getAllByRole("rowheader").map((h) => h.textContent)).toEqual(
      ["PRIMARY · VIS", "PRIMARY · Y", "PRIMARY · J", "PRIMARY · H"]);
    await waitFor(() => expect(row.textContent).toContain("159.5"));
    expect(row.textContent).toContain("118.6");
    expect(row.textContent).toContain("315.8");
    expect(row.textContent).toContain("51,000");                     // Σ flux to 3 significant figures
    expect(within(table).getByRole("columnheader", { name: "Median [electron]" })).toBeTruthy();
    expect(within(table).queryByRole("columnheader", { name: /Non-finite/ })).toBeNull();
    expect(screen.queryByText(/Preview/)).toBeNull();
    expect(screen.queryByText("0 / 319 electron")).toBeNull();
    expect(calls.some((c) => c.url === "/api/inspect/image/stats?fits=data%2Feval_results%2Fcat.fits&hdu=0&plane=0")).toBe(true);
    // the sky is one caption line; the file bar has the one sky link
    expect(screen.getByText(/Centre 10h00m00\.00s \+02°00′00\.0″ · 0\.1″ pixels/)).toBeTruthy();
    const sky = screen.getAllByRole("link", { name: /Show on sky/ });
    expect(sky.length).toBe(1);
    expect(sky[0].getAttribute("href")).toMatch(/^\/sky\/atlas\?ra=150\.000000&dec=2\.000000&fov=/);
    // the page's HDU picker is labelled as what it is (not a copy of the viewer's chips): the name first
    const pick = screen.getByRole("combobox", { name: "HDU for the header, table and statistics" });
    expect(pick.textContent).toContain("PRIMARY (HDU 0)");
    // a crumb keeps its whole name for assistive tech while its middle may give way
    const crumbs = screen.getByRole("navigation", { name: "File location" });
    expect(within(crumbs).getAllByRole("button").map((b) => b.getAttribute("aria-label"))).toContain("Evaluation results");
  });

  it("adds a non-finite column only when a frame has non-finite pixels, and puts the clipping in a tooltip", async () => {
    const base = routes["GET /api/inspect/image/stats"];
    routes["GET /api/inspect/image/stats"] = (u, init) => {
      const r = base(u, init);
      const body = r.body as Record<string, unknown>;
      return { body: { ...body, n_nan: 7, histogram: { edges: [0, 160, 320], counts: [160, 160], below: 1049, above: 3 } } };
    };
    show(`/inspect?fits=${encodeURIComponent(FILE)}`);
    const table = await screen.findByRole("table", { name: "Statistics of the frames shown" });
    expect(await within(table).findByRole("columnheader", { name: "Non-finite px" })).toBeTruthy();
    expect(await screen.findByLabelText("1,049 pixels below and 3 above the plotted range")).toBeTruthy();
    expect(screen.queryByText(/1,049 below/)).toBeNull();
  });

  it("switches to the table HDU and pages on the server", async () => {
    show(`/inspect?fits=${encodeURIComponent(FILE)}`);
    const hdus = await screen.findByRole("grid", { name: "HDUs" });
    fireEvent.click(within(hdus).getByText("CAT"));
    await waitFor(() => expect(params().get("hdu")).toBe("1"));
    expect(await screen.findByText("1–200 of 450")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    expect(await screen.findByText("201–400 of 450")).toBeTruthy();
    expect(params().get("toff")).toBe("200");
    expect(calls.some((c) => c.url.includes("/api/inspect/table?") && c.url.includes("offset=200"))).toBe(true);
    // column statistics, then one column's histogram
    fireEvent.click(await screen.findByText("flux", { selector: "code" }));
    await waitFor(() => expect(params().get("tcol")).toBe("flux"));
  });

  it("opens a table row as a card, with a sky link for RA/Dec columns", async () => {
    routes["GET /api/inspect/table"] = (u) => {
      const page = tablePage(u);
      page.columns = [...page.columns, { name: "RA", format: "D", unit: "deg", dim: null, null: null, kind: "numeric" },
        { name: "DEC", format: "D", unit: "deg", dim: null, null: null, kind: "numeric" }];
      page.rows = page.rows.map((r) => [...r, 266.8, 67.45]);
      return { body: page };
    };
    show(`/inspect?fits=${encodeURIComponent(FILE)}&hdu=1`);
    const grid = await screen.findByRole("grid", { name: "Rows of HDU 1" });
    fireEvent.click(await within(grid).findByText("1.5"));
    await waitFor(() => expect(params().get("trow")).toBe("1"));
    expect(await screen.findByText("Row 1")).toBeTruthy();
    const sky = screen.getByRole("link", { name: /Show on sky/ });
    expect(sky.getAttribute("href")).toBe("/sky/atlas?ra=266.800000&dec=67.450000&fov=0.0200");
    fireEvent.click(screen.getByRole("button", { name: "Close the row" }));
    await waitFor(() => expect(params().get("trow")).toBeNull());
  });

  it("filters the header cards and shows provenance", async () => {
    show(`/inspect?fits=${encodeURIComponent(FILE)}&view=header`);
    const grid = await screen.findByRole("grid", { name: "Header of HDU 0" });
    expect(within(grid).getByText("CRVAL1")).toBeTruthy();
    fireEvent.change(screen.getByRole("searchbox", { name: /Filter/ }), { target: { value: "key:BUNIT" } });
    await waitFor(() => expect(within(grid).queryByText("CRVAL1")).toBeNull());
    expect(within(grid).getByText("BUNIT")).toBeTruthy();
    fireEvent.mouseDown(screen.getByRole("tab", { name: "Provenance" }));
    await waitFor(() => expect(params().get("view")).toBe("provenance"));
    expect(await screen.findByText("This file's record")).toBeTruthy();
  });

  it("tracks the file after the dialog", async () => {
    show(`/inspect?fits=${encodeURIComponent(FILE)}&view=header`);
    await screen.findByRole("grid", { name: "Header of HDU 0" });
    fireEvent.click(screen.getByRole("button", { name: "Track" }));
    const dialog = await screen.findByRole("dialog");
    fireEvent.change(within(dialog).getByRole("textbox", { name: /Comment/ }), { target: { value: "for the poster" } });
    await act(async () => { fireEvent.click(within(dialog).getByRole("button", { name: "Track" })); });
    await waitFor(() => expect(calls.some((c) => c.method === "POST" && c.url === "/api/tracking/backup")).toBe(true));
    const post = calls.find((c) => c.method === "POST")!;
    const form = new URLSearchParams(post.body ?? "");
    expect(form.get("kind")).toBe("fits");
    expect(form.get("path")).toBe(FILE);
    expect(form.get("comment")).toBe("for the poster");
  });

  it("shows the server's error for a missing file", async () => {
    show("/inspect?fits=data/nope.fits");
    expect(await screen.findByText("File not found")).toBeTruthy();
    expect(screen.getByText("no such FITS file: data/nope.fits")).toBeTruthy();
  });

  it("turns ?slice= into the viewer's plane (a band cube switches to planes)", async () => {
    show(`/inspect?path=${encodeURIComponent(FILE)}&hdu=0&slice=J_E`);
    await waitFor(() => expect(params().get("v.fits.id")).toBe("p2"));
    expect(params().get("slice")).toBeNull();
    expect(params().get("stack")).toBe("planes");
    await waitFor(() => expect(calls.some((c) => c.url.startsWith("/viewer/meta/fits?") && c.url.includes("stack=planes"))).toBe(true));
    // the viewer never mounted on the band cube first
    expect(calls.some((c) => c.url.startsWith("/viewer/meta/fits?") && !c.url.includes("stack=planes"))).toBe(false);
  });

  it("accepts ?path= as an alias of ?fits=", async () => {
    show(`/inspect?path=${encodeURIComponent(FILE)}`);
    await waitFor(() => expect(params().get("fits")).toBe(FILE));
    expect(params().get("path")).toBeNull();
  });
});

describe("the fits inspector kind", () => {
  it("is registered with a file-name title and renders a compact view", async () => {
    const reg = useInspectorRegistry.getState().kinds.fits;
    expect(reg).toBeTruthy();
    expect(typeof reg.title === "function" && reg.title("data/x/SR.fits")).toBe("SR.fits");
    render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <FitsInspector id={FILE} />
        </MemoryRouter>
      </QueryClientProvider>,
    );
    const open = await screen.findByRole("link", { name: /Open in Files/ });
    expect(open.getAttribute("href")).toBe(`/files?fits=${encodeURIComponent(FILE)}&hdu=0`);
    expect(screen.getByText("PRIMARY", { exact: false })).toBeTruthy();
    expect(typeof unregisterFitsInspector).toBe("function");
  });
});
