import { describe, expect, it } from "vitest";
import {
  axis, clampView, drawable, nearestPoint, panDomain, readoutAt, sameInputs, seriesKey, seriesToCSV,
  stepDrawable, tooltipReadout, viewTicks, zoomDomain, type PlotGeometry,
} from "./plotModel";
import type { Series } from "./types";

const geo = (over: Partial<PlotGeometry> = {}): PlotGeometry => ({
  m: { l: 40, r: 20, t: 10, b: 30 }, iw: 400, ih: 200,
  x: axis([0, 10], "linear", 40, 440), y: axis([0, 100], "linear", 210, 10),
  ...over,
});

describe("axis", () => {
  it("maps linear domains to pixels and back", () => {
    const a = axis([0, 10], "linear", 40, 440);
    expect(a.toPx(5)).toBe(240);
    expect(a.fromPx(240)).toBe(5);
    expect(a.ok(-3)).toBe(true);
  });
  it("maps log domains through log10 and rejects non-positive values", () => {
    const a = axis([1, 1000], "log", 0, 300);
    expect(a.toPx(10)).toBeCloseTo(100);
    expect(a.fromPx(200)).toBeCloseTo(100);
    expect(a.ok(0)).toBe(false);
    expect(a.ok(-1)).toBe(false);
    expect(Number.isFinite(a.toPx(0))).toBe(false);
  });
  it("inverts y (top pixel = domain max)", () => {
    const g = geo();
    expect(g.y.toPx(100)).toBe(10);
    expect(g.y.toPx(0)).toBe(210);
  });
  it("survives a degenerate or non-positive log domain", () => {
    const a = axis([0, 100], "log", 0, 100);
    expect(Number.isFinite(a.toPx(50))).toBe(true);
    const z = axis([5, 5], "linear", 0, 100);
    expect(Number.isFinite(z.toPx(5))).toBe(true);
  });
});

describe("zoom and pan", () => {
  it("zooms about an anchor fraction", () => {
    expect(zoomDomain([0, 10], "linear", 0.5, 0.5)).toEqual([2.5, 7.5]);
    expect(zoomDomain([0, 10], "linear", 0, 0.5)).toEqual([0, 5]);
    const [a, b] = zoomDomain([1, 1e4], "log", 0.5, 0.5);
    expect(a).toBeCloseTo(10);
    expect(b).toBeCloseTo(1000);
  });
  it("pans by a fraction of the span (in log space for log axes)", () => {
    expect(panDomain([0, 10], "linear", 0.1)).toEqual([1, 11]);
    const [a, b] = panDomain([1, 100], "log", 0.5);
    expect(a).toBeCloseTo(10);
    expect(b).toBeCloseTo(1000);
  });
  it("orders and bounds a view", () => {
    expect(clampView([5, 1], "linear")).toEqual([1, 5]);
    expect(clampView([3, 3], "linear")).toBeNull();
    expect(clampView([-1, 10], "log")).toBeNull();
  });
});

describe("hit testing", () => {
  const lines: Series[] = [
    { x: [0, 5, 10], y: [10, 50, 90], color: "red", name: "mean" },
    { x: [0, 5, 10], y: [20, 30, null], color: "blue", name: "gate" },
    { x: [2, 8], y: [80, 20], color: "green", mode: "scatter", name: "pts" },
  ];

  it("finds the nearest visible point in pixels", () => {
    const g = geo();
    const near = nearestPoint(lines, new Set(), { x: g.x.toPx(5) + 3, y: g.y.toPx(32) }, g)!;
    expect(near.series).toBe(1);
    expect(near.index).toBe(1);
    expect(near.y).toBe(30);
    const scat = nearestPoint(lines, new Set(), { x: g.x.toPx(8), y: g.y.toPx(22) }, g)!;
    expect(scat.series).toBe(2);
  });

  it("skips hidden series, gaps and far-away points", () => {
    const g = geo();
    const near = nearestPoint(lines, new Set([seriesKey(lines[1], 1)]), { x: g.x.toPx(5), y: g.y.toPx(32) }, g)!;
    expect(near.series).toBe(0);
    expect(nearestPoint(lines, new Set(), { x: g.x.toPx(10), y: g.y.toPx(0) }, g, 20)).toBeNull();
  });

  it("reads every visible line at the nearest x", () => {
    const rows = readoutAt(lines, new Set(), 4.6);
    expect(rows.map((r) => [r.series, r.y])).toEqual([[0, 50], [1, 30]]);
  });

  it("tooltipReadout keeps the nearest series and picks the other rows by pixel distance", () => {
    const flat = Array.from({ length: 20 }, (_, i): Series => ({ x: [0, 5, 10], y: [5 * i, 5 * i, 5 * i], color: "c" }));
    const g = geo();
    // cursor at y = 52 (px): nearest lines y = 50, 55, 45, 60 …; series 0 (y = 0) is forced in
    const out = tooltipReadout(flat, new Set(), 5, { limit: 4, cursorPy: g.y.toPx(52), y: g.y, first: 0 });
    expect(out.rows.map((r) => r.series)).toEqual([0, 9, 10, 11]);   // series order
    expect(out.more).toBe(16);
    // everything fits: array order, nothing hidden
    const few = tooltipReadout(flat.slice(0, 3), new Set(), 5, { limit: 8, cursorPy: 0, y: g.y });
    expect(few.rows.map((r) => r.series)).toEqual([0, 1, 2]);
    expect(few.more).toBe(0);
    // hidden series never count
    expect(tooltipReadout(flat.slice(0, 3), new Set(["#1"]), 5, { limit: 8, cursorPy: 0, y: g.y }).rows
      .map((r) => r.series)).toEqual([0, 2]);
  });

  it("keys series by key, then name, then label, then index", () => {
    expect(seriesKey({ x: [], y: [], color: "", key: "k", name: "n" }, 3)).toBe("k");
    expect(seriesKey({ x: [], y: [], color: "", name: "n" }, 3)).toBe("n");
    expect(seriesKey({ x: [], y: [], color: "", label: "l" }, 3)).toBe("l");
    expect(seriesKey({ x: [], y: [], color: "" }, 3)).toBe("#3");
  });
});

describe("keyboard readout steps", () => {
  const logY = geo({ y: axis([1, 100], "log", 210, 10) });
  const s: Series = { x: [0, 1, 2, 3, 4], y: [0, 10, -5, null, 50], color: "red" };

  it("knows which points the axes can draw", () => {
    expect(drawable(s, 0, logY)).toBe(false);          // 0 on a log axis is a gap
    expect(drawable(s, 1, logY)).toBe(true);
    expect(drawable(s, 3, logY)).toBe(false);          // null
    expect(drawable(s, 0, geo())).toBe(true);           // 0 is fine on a linear axis
    expect(drawable(s, 9, geo())).toBe(false);          // out of range
  });

  it("steps over the gaps, stays at the ends and finds the nearest drawable point", () => {
    expect(stepDrawable(s, 0, 0, logY)).toBe(1);
    expect(stepDrawable(s, 1, 1, logY)).toBe(4);
    expect(stepDrawable(s, 4, 1, logY)).toBe(4);
    expect(stepDrawable(s, 4, -1, logY)).toBe(1);
    expect(stepDrawable(s, 1, -1, logY)).toBe(1);
    expect(stepDrawable(s, 2, 0, logY)).toBe(1);
    expect(stepDrawable(s, 99, 0, logY)).toBe(4);
    expect(stepDrawable({ x: [0, 1], y: [0, -1], color: "red" }, 0, 0, logY)).toBe(-1);
    expect(stepDrawable({ x: [], y: [], color: "red" }, 0, 1, logY)).toBe(-1);
  });
});

describe("viewTicks", () => {
  const ticks = [0, 2, 4, 6, 8, 10].map((v) => ({ v, label: `t${v}` }));
  it("keeps the caller's ticks when enough fall inside the view", () => {
    expect(viewTicks(ticks, [1, 9], "linear").map((t) => t.label)).toEqual(["t2", "t4", "t6", "t8"]);
  });
  it("generates ticks when too few caller ticks remain", () => {
    const out = viewTicks(ticks, [2.1, 3.9], "linear");
    expect(out.length).toBeGreaterThanOrEqual(3);
    for (const t of out) expect(t.v >= 2.1 && t.v <= 3.9).toBe(true);
  });
  it("uses a formatter for generated ticks", () => {
    const out = viewTicks(undefined, [2, 3], "log", (v) => `~${v.toFixed(1)}`);
    expect(out.every((t) => t.label.startsWith("~"))).toBe(true);
  });
});

describe("seriesToCSV", () => {
  it("writes long-format rows with the optional band/error columns", () => {
    const csv = seriesToCSV([
      { x: [1, 2], y: [3, null], low: [2, 1], high: [4, 5], color: "r", name: "a" },
      { x: [1], y: [7], color: "b", label: "b,c" },
    ]);
    expect(csv).toBe('series,x,y,low,high\r\na,1,3,2,4\r\na,2,,1,5\r\n"b,c",1,7,,\r\n');
  });

  it("defuses spreadsheet formulas in series names but keeps negative numbers", () => {
    const csv = seriesToCSV([{ x: [-1], y: [-2.5], color: "r", name: "=HYPERLINK(\"x\")" }]);
    expect(csv).toBe('series,x,y\r\n"\'=HYPERLINK(""x"")",-1,-2.5\r\n');
  });
});

describe("sameInputs", () => {
  const x = [1, 2], y = [3, 4];
  const base = { xDomain: [0, 1] as [number, number], yDomain: [0, 1] as [number, number], series: [{ x, y, color: "r" }] };
  it("treats rebuilt wrappers around the same data as equal", () => {
    expect(sameInputs(base, { ...base, xDomain: [0, 1], series: [{ x, y, color: "r" }] })).toBe(true);
  });
  it("detects changed data, colours, domains and handlers are ignored", () => {
    expect(sameInputs(base, { ...base, series: [{ x, y: [3, 5], color: "r" }] })).toBe(false);
    expect(sameInputs(base, { ...base, series: [{ x, y, color: "b" }] })).toBe(false);
    expect(sameInputs(base, { ...base, yDomain: [0, 2] })).toBe(false);
    expect(sameInputs({ ...base, onPlotClick: () => {} }, { ...base, onPlotClick: () => {} })).toBe(true);
  });
  it("compares drawn functions nested in the inputs (heat.color) by identity", () => {
    const z = [[1]], e = [0, 1];
    const red = () => "red";
    expect(sameInputs({ ...base, heat: { z, xEdges: e, yEdges: e, color: red } },
      { ...base, heat: { z, xEdges: e, yEdges: e, color: red } })).toBe(true);
    expect(sameInputs({ ...base, heat: { z, xEdges: e, yEdges: e, color: red } },
      { ...base, heat: { z, xEdges: e, yEdges: e, color: () => "red" } })).toBe(false);
    // top-level functions (handlers, formatters) are not compared here
    expect(sameInputs({ ...base, xFormat: (v: number) => `${v}` }, { ...base, xFormat: (v: number) => `${v} e` })).toBe(true);
  });
});
