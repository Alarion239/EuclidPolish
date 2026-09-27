import { beforeEach, describe, expect, it } from "vitest";
import type { BandGroup, HduSummary, InspectResponse } from "./api";
import { browseUrl, inspectPageHref, previewUrl, tableUrl } from "./api";
import {
  basename, cardRows, defaultHduKey, defaultView, dirname, fileCrumbs, flatIndex, fovArcsec, hduByKey, hduFacts,
  histogramSeries, isGroupKey, pageLabel, planeAxesLabel, planeIndex, pushRecent, readRecent, shapeText,
  normalizeSummary, skyAt, skyColumns, skyHref, sliceTarget, sortToServer, viewerParams, viewsFor,
  axisLabel, INSPECT_WIDE_PX, isNarrowWidth, searchScope,
} from "./model";

const hdu = (over: Partial<HduSummary>): HduSummary => ({
  index: 0, hdu_index: 0, name: "PRIMARY", kind: "PrimaryHDU", type: "image", shape: [8, 8],
  dtype: ">f4", viewable: true, planes: 1, plane_axes: [], ...over,
});

const group: BandGroup = {
  id: "b:LR_", prefix: "LR_", label: "LR · 4-band colour", hdus: [1, 2, 3, 4],
  bands: ["VIS", "Y_E", "J_E", "H_E"], shape: [8, 8], wcs: null, bunit: "electron",
};

const summary = (hdus: HduSummary[], groups: BandGroup[] = []): InspectResponse => ({
  file: { abspath: "/x/a.fits", basename: "a.fits", size: 1, size_kb: 0, mtime: 0, compressed: false },
  hdus, band_groups: groups, scan_truncated: false, stamp: null, rel: "data/a.fits", root: null,
  allowed_roots: [], roots: [],
});

describe("paths", () => {
  it("basename / dirname", () => {
    expect(basename("data/eval/x/SR.fits")).toBe("SR.fits");
    expect(basename("SR.fits")).toBe("SR.fits");
    expect(dirname("data/eval/x/SR.fits")).toBe("data/eval/x");
    expect(dirname("SR.fits")).toBe("");
  });

  it("file crumbs start at the root label", () => {
    const root = { id: "eval", label: "Evaluation results", path: "/r/data/eval_results", rel: "data/eval_results", exists: true };
    expect(fileCrumbs("data/eval_results/gal_1/SR.fits", root)).toEqual([
      { name: "Evaluation results", rel: "data/eval_results" },
      { name: "gal_1", rel: "data/eval_results/gal_1" },
    ]);
    expect(fileCrumbs("poster/x.fits", null)).toEqual([{ name: "poster", rel: "poster" }]);
  });
});

describe("planes", () => {
  it("flat ↔ multi index (row-major, numpy order)", () => {
    expect(planeIndex([2, 3], 4)).toEqual([1, 1]);
    expect(planeIndex([4], 3)).toEqual([3]);
    expect(planeIndex([], 0)).toEqual([]);
    expect(flatIndex([2, 3], [1, 1])).toBe(4);
    expect(flatIndex([2, 3], [5, -1])).toBe(3);   // clamped to [1, 0]
    for (let k = 0; k < 24; k++) expect(flatIndex([2, 3, 4], planeIndex([2, 3, 4], k))).toBe(k);
  });

  it("shape text is FITS order (NAXIS1 first)", () => {
    expect(shapeText([4, 106, 106])).toBe("106 × 106 × 4");
    expect(shapeText([16, 20])).toBe("20 × 16");
    expect(shapeText(null)).toBe("—");
    expect(planeAxesLabel([2, 3])).toBe("NAXIS4 × NAXIS3");
  });
});

describe("HDU selection", () => {
  it("defaults to a colour group, else the first viewable image, a table, then 0", () => {
    expect(defaultHduKey(summary([hdu({ type: "empty", viewable: false }), hdu({ index: 1 })]))).toBe("1");
    expect(defaultHduKey(summary([hdu({ type: "empty", viewable: false })], [group]))).toBe("b:LR_");
    expect(defaultHduKey(summary([hdu({ type: "empty", viewable: false }), hdu({ index: 1, name: "LR_VIS" })], [group]))).toBe("b:LR_");
    expect(defaultHduKey(summary([hdu({ type: "empty", viewable: false }), hdu({ index: 1, type: "table" })]))).toBe("1");
    expect(defaultHduKey(summary([hdu({ type: "empty", viewable: false })]))).toBe("0");
  });

  it("resolves keys to an HDU or a band group", () => {
    const s = summary([hdu({}), hdu({ index: 1, name: "LR_VIS" })], [group]);
    expect(hduByKey(s, "1")?.hdu?.name).toBe("LR_VIS");
    expect(hduByKey(s, "b:LR_")?.group?.id).toBe("b:LR_");
    expect(hduByKey(s, "9")).toBeNull();
    expect(hduByKey(s, "x")).toBeNull();
    expect(isGroupKey("b:")).toBe(true);
    expect(isGroupKey("3")).toBe(false);
  });

  it("summarises a selection in one line", () => {
    const cube = hdu({ ndim: 3, shape: [4, 106, 106], planes: 4, plane_axes: [4], bunit: "electron",
      bands: ["VIS", "Y_E", "J_E", "H_E"] });
    expect(hduFacts({ key: "0", hdu: cube, group: null })).toBe("106 × 106 × 4 · f4 · electron · VIS Y J H");
    expect(hduFacts({ key: "1", hdu: hdu({ index: 1, type: "table", shape: null, dtype: null, nrows: 1234, ncols: 7 }), group: null }))
      .toBe("1,234 rows × 7 columns");
    expect(hduFacts({ key: "b:LR_", hdu: null, group })).toBe("8 × 8 · 4 bands · electron");
    expect(hduFacts({ key: "0", hdu: hdu({ type: "empty", shape: null, dtype: null, viewable: false }), group: null })).toBe("no data");
  });

  it("offers the views an HDU supports", () => {
    expect(viewsFor(hdu({}))).toEqual(["image", "header", "provenance"]);
    expect(viewsFor(hdu({ type: "table" }))).toEqual(["table", "header", "provenance"]);
    expect(viewsFor(hdu({ type: "vector" }))).toEqual(["plot", "header", "provenance"]);
    expect(viewsFor(hdu({ type: "empty", viewable: false }))).toEqual(["header", "provenance"]);
    expect(viewsFor(null, group)).toEqual(["image", "provenance"]);
    expect(defaultView(hdu({ type: "table" }))).toBe("table");
    expect(defaultView(hdu({ type: "empty", viewable: false }))).toBe("header");
  });
});

describe("normalizeSummary", () => {
  it("upgrades the pre-rework /api/inspect payload", () => {
    const old = {
      file: { basename: "a.fits", size_kb: 1, mtime: 0 }, rel: "a.fits", allowed_roots: [],
      hdus: [
        { hdu_index: 0, name: "PRIMARY", kind: "PrimaryHDU", shape: [4, 8, 8], dtype: ">f4", cards: [] },
        { hdu_index: 1, name: "CAT", kind: "BinTableHDU", shape: null, dtype: null, cards: [] },
        { hdu_index: 2, name: "EMPTY", kind: "ImageHDU", shape: null, dtype: null, cards: [] },
      ],
    } as unknown as InspectResponse;
    const s = normalizeSummary(old);
    expect(s.band_groups).toEqual([]);
    expect(s.hdus.map((h) => [h.index, h.type, h.planes, h.viewable])).toEqual([
      [0, "image", 4, true], [1, "table", 0, false], [2, "empty", 0, false],
    ]);
    expect(s.hdus[0].plane_axes).toEqual([4]);
    expect(defaultHduKey(s)).toBe("0");
  });
});

describe("viewer params", () => {
  it("leave the defaults out", () => {
    const cube = hdu({ ndim: 3, shape: [4, 8, 8], planes: 4, plane_axes: [4], bands: ["VIS", "Y_E", "J_E", "H_E"] });
    const sel = { key: "0", hdu: cube, group: null };
    expect(viewerParams("a.fits", sel, { stack: "bands", bin: "auto", render: "asinh" })).toEqual({ path: "a.fits", hdu: "0" });
    expect(viewerParams("a.fits", sel, { stack: "planes", bin: "2", render: "log" }))
      .toEqual({ path: "a.fits", hdu: "0", stack: "planes", bin: "2", render: "log" });
    // only a band cube has a stacking choice
    expect(viewerParams("a.fits", { key: "1", hdu: hdu({ index: 1 }), group: null }, { stack: "planes", bin: "auto", render: "" }))
      .toEqual({ path: "a.fits", hdu: "1" });
  });
});

describe("?slice= (the spec's plane / band link)", () => {
  const cube = hdu({ ndim: 3, shape: [4, 8, 8], planes: 4, plane_axes: [4], bands: ["VIS", "Y_E", "J_E", "H_E"] });
  const stack = hdu({ ndim: 4, shape: [2, 3, 8, 8], planes: 6, plane_axes: [2, 3] });
  const sel = (h: HduSummary) => ({ key: String(h.index), hdu: h, group: null });

  it("picks a plane by flat index, band name or multi-index (clamped)", () => {
    expect(sliceTarget(sel(stack), "4")).toEqual({ id: "p4", planes: false });
    expect(sliceTarget(sel(stack), "99")).toEqual({ id: "p5", planes: false });
    expect(sliceTarget(sel(stack), "1,1")).toEqual({ id: "p4", planes: false });
    expect(sliceTarget(sel(stack), "[0, 2]")).toEqual({ id: "p2", planes: false });
    // a band cube switches to one-plane-at-a-time
    expect(sliceTarget(sel(cube), "2")).toEqual({ id: "p2", planes: true });
    expect(sliceTarget(sel(cube), "j_e")).toEqual({ id: "p2", planes: true });
    expect(sliceTarget(sel(cube), "H")).toEqual({ id: "p3", planes: true });
  });

  it("ignores what has no planes to pick", () => {
    expect(sliceTarget(sel(hdu({})), "3")).toBeNull();                 // a 2-D image
    expect(sliceTarget({ key: "b:LR_", hdu: null, group }, "1")).toBeNull();
    expect(sliceTarget(sel(stack), "nope")).toBeNull();
    expect(sliceTarget(sel(stack), "")).toBeNull();
    expect(sliceTarget(sel(cube), "K")).toBeNull();
  });
});

describe("header cards", () => {
  it("numbers the cards and keeps duplicates distinct", () => {
    const rows = cardRows([["SIMPLE", "True", "std"], ["COMMENT", "a", ""], ["COMMENT", "b", ""]]);
    expect(rows.map((r) => r.i)).toEqual([0, 1, 2]);
    expect(rows[2]).toEqual({ i: 2, key: "COMMENT", value: "b", comment: "" });
  });
});

describe("sky + table helpers", () => {
  it("links to the atlas with a context field of view", () => {
    const wcs = { ctype: ["RA---TAN", "DEC--TAN"], ra: 266.8, dec: 67.45, pixscale_arcsec: 0.1, width_arcsec: 10.6,
      height_arcsec: 10.6, fov_deg: 10.6 / 3600, corners: [], constructed: false };
    const href = skyHref(wcs);
    const q = new URLSearchParams(href.split("?")[1]);
    expect(href.startsWith("/sky/atlas?")).toBe(true);
    expect(Number(q.get("ra"))).toBeCloseTo(266.8, 6);
    expect(Number(q.get("dec"))).toBeCloseTo(67.45, 6);
    expect(Number(q.get("fov"))).toBeGreaterThanOrEqual(0.01);
    expect(fovArcsec(wcs)).toBeCloseTo(10.6, 6);
  });

  it("finds a table's RA/Dec columns (numeric, common spellings)", () => {
    const col = (name: string, kind: "numeric" | "text" = "numeric") => ({ name, format: "D", unit: null, dim: null, null: null, kind });
    expect(skyColumns([col("id"), col("RA"), col("DEC")])).toEqual({ ra: "RA", dec: "DEC" });
    expect(skyColumns([col("ALPHA_J2000"), col("DELTA_J2000")])).toEqual({ ra: "ALPHA_J2000", dec: "DELTA_J2000" });
    expect(skyColumns([col("right_ascension"), col("declination")])).toEqual({ ra: "right_ascension", dec: "declination" });
    expect(skyColumns([col("ra", "text"), col("dec")])).toBeNull();
    expect(skyColumns([col("ra")])).toBeNull();
    expect(skyColumns([col("rating"), col("decay")])).toBeNull();
    const href = skyAt(266.8, 67.45, 0.02);
    expect(href).toBe("/sky/atlas?ra=266.800000&dec=67.450000&fov=0.0200");
    expect(skyAt(Number.NaN, 1)).toBeNull();
    expect(skyAt(10, 95)).toBeNull();
  });

  it("maps the first sort key to the server", () => {
    expect(sortToServer([])).toEqual({ sort: null, desc: false });
    expect(sortToServer([{ id: "flux", desc: true }, { id: "id", desc: false }])).toEqual({ sort: "flux", desc: true });
    expect(sortToServer([{ id: "#", desc: false }])).toEqual({ sort: null, desc: false });
  });

  it("labels a page window", () => {
    expect(pageLabel(0, 200, 5000)).toBe("1–200 of 5,000");
    expect(pageLabel(4900, 200, 5000)).toBe("4,901–5,000 of 5,000");
    expect(pageLabel(0, 200, 0)).toBe("0 rows");
  });

  it("turns histogram edges into bar centres", () => {
    expect(histogramSeries({ edges: [0, 1, 2], counts: [3, 4], below: 0, above: 0 })).toEqual({ x: [0.5, 1.5], y: [3, 4] });
  });
});

describe("recent files", () => {
  beforeEach(() => localStorage.clear());

  it("keeps the newest first, de-duplicated, capped", () => {
    for (let k = 0; k < 12; k++) pushRecent(`f${k}.fits`);
    pushRecent("f5.fits");
    const recent = readRecent();
    expect(recent[0]).toBe("f5.fits");
    expect(recent.length).toBe(8);
    expect(new Set(recent).size).toBe(8);
  });

  it("survives corrupt storage", () => {
    localStorage.setItem("ep-inspect-recent", "{nope");
    expect(readRecent()).toEqual([]);
  });
});

describe("urls", () => {
  it("builds the endpoint URLs", () => {
    expect(browseUrl("", "")).toBe("/api/inspect/browse");
    expect(browseUrl("data/eval results", " sr ")).toBe("/api/inspect/browse?dir=data%2Feval%20results&q=sr");
    expect(tableUrl("a.fits", 1, { offset: 0, limit: 200, sort: "flux", desc: true }))
      .toBe("/api/inspect/table?fits=a.fits&hdu=1&limit=200&sort=flux&desc=1");
    expect(previewUrl("a.fits", { hdu: 2, plane: 0, size: 128 })).toBe("/inspect/preview.png?fits=a.fits&hdu=2&size=128");
    expect(inspectPageHref("data/a b.fits", "b:LR_")).toBe("/inspect?fits=data%2Fa%20b.fits&hdu=b%3ALR_");
  });
});

describe("layout + labels", () => {
  it("folds the browser below the layout breakpoint, measured on the host", () => {
    expect(INSPECT_WIDE_PX).toBe(1000);
    expect(isNarrowWidth(720, false)).toBe(true);
    expect(isNarrowWidth(999, false)).toBe(true);
    expect(isNarrowWidth(1000, true)).toBe(false);
    expect(isNarrowWidth(1440, true)).toBe(false);
    // unmeasured (0: tests, before layout) → the viewport fallback
    expect(isNarrowWidth(0, true)).toBe(true);
    expect(isNarrowWidth(0, false)).toBe(false);
  });

  it("writes compact axis labels for large values", () => {
    expect(axisLabel(0)).toBe("0");
    expect(axisLabel(999)).toBe("999");
    expect(axisLabel(5000)).toBe("5k");
    expect(axisLabel(1234)).toBe("1.23k");
    expect(axisLabel(10000)).toBe("10k");
    expect(axisLabel(25000)).toBe("25k");
    expect(axisLabel(-12500)).toBe("-12.5k");
    expect(axisLabel(2.5e6)).toBe("2.5M");
    expect(axisLabel(0.25)).toBe("0.25");
    expect(axisLabel(3e-5)).toBe("3e-5");
  });

  it("names the search scope from the folder even before the listing loads", () => {
    const root = { id: "eval", label: "Evaluation results", path: "/r/data/eval_results", rel: "data/eval_results", exists: true };
    expect(searchScope("", null)).toBe("all roots");
    // not loaded yet: the folder's own name (or its root's label)
    expect(searchScope("data/eval_results/gal_1", null)).toBe("gal_1");
    expect(searchScope("data/eval_results", null, [root])).toBe("Evaluation results");
    // loaded for this folder: its last crumb
    const loaded = { dir: "data/eval_results", crumbs: [{ name: "Evaluation results", rel: "data/eval_results" }] };
    expect(searchScope("data/eval_results", loaded)).toBe("Evaluation results");
    // a stale listing of another folder is ignored
    expect(searchScope("data/eval_results/gal_1", loaded)).toBe("gal_1");
  });
});
