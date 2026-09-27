import { describe, expect, it } from "vitest";
import type { ExperimentRecord, ModelSpecRow, TileList, TileRow } from "./api";
import { atlasHref, experimentsHref, splitRef, URLS } from "./api";
import {
  bandSeries, experimentMarkdown, filterByState, filterEvalRows, flattenTiles, formatMetric, groupModels,
  headlineSpec, metricRows, metricsPlan, num, parseRefs, productionCounts, recordSpecs, runnableSelection, seriesDomain,
  sortSpecs, specShort, tileModels,
} from "./model";

const tile = (ref: string, state?: string, models: TileRow["models"] = {}): TileRow => {
  const [source, id] = splitRef(ref);
  return { source, id, ref, label: ref, ra: 1, dec: 2, production_state: state, models };
};

const RECORD: ExperimentRecord = {
  id: "20260926-101010-abcdef", label: "poster core", status: "done",
  tiles: ["poster/p1", "nexus/f200w-0040"], models: ["member:member_192", "production", "gate:26m"],
  skipped: { "gate:26m": "member 171 archived" },
  model_labels: { production: "Production · spatial gate", "member:member_192": "Member 192·psnr" },
  summary: {
    production: {
      bands: ["VIS", "Y_E"], per_band: {
        VIS: { hole_pct: 5.2, pct_R_lt_0p8: 10, median_R: 0.91, flux_ratio: 0.998 },
        Y_E: { hole_pct: 6.1, median_R: 0.95 },
      }, summary: { pct_R_lt_0p8: 12.5, median_R: 0.93 },
    },
    "member:member_192": { bands: ["VIS"], per_band: { VIS: { hole_pct: 2.0, flux_ratio: 1.0 } }, summary: {} },
  },
  results: {
    "poster/p1": { production: { state: "computed", metrics: { bands: ["VIS"], per_band: { VIS: { hole_pct: 7 } } } } },
  },
  errors: { "nexus/f200w-0040|production": "RuntimeError: boom" },
};

describe("value helpers", () => {
  it("reads numbers from JSON and CSV strings", () => {
    expect(num(3)).toBe(3);
    expect(num("0.67")).toBe(0.67);
    expect(num("")).toBeNull();
    expect(num("nan")).toBeNull();
    expect(num(Infinity)).toBeNull();
    expect(num(null)).toBeNull();
  });
  it("formats metrics with their precision", () => {
    expect(formatMetric("hole_pct", 5.234)).toBe("5.2");
    expect(formatMetric("median_R", "0.91234")).toBe("0.912");
    expect(formatMetric("n_peaks", 12)).toBe("12");
    expect(formatMetric("hole_pct", null)).toBe("—");
  });
});

describe("refs and URLs", () => {
  it("splits refs at the first slash", () => {
    expect(splitRef("nexus/f200w-0001")).toEqual(["nexus", "f200w-0001"]);
    expect(splitRef("bare")).toEqual(["bare", ""]);
  });
  it("builds C9 URLs with encoded ids", () => {
    expect(URLS.card("eval/102_NEG1")).toBe("/api/real/eval/102_NEG1");
    expect(URLS.image("tile/ra1_dec2", "m:production", "VIS")).toBe(
      "/api/real/tile/ra1_dec2/image.fits?tier=m%3Aproduction&band=VIS");
    expect(URLS.deleteOutputs("nexus/f200w-0001")).toBe("/api/real/nexus/f200w-0001/delete-outputs");
  });
  it("links to the atlas and to experiments", () => {
    expect(atlasHref(268.4, 65.2, "nexus/f200w-0001")).toBe(
      "/sky/atlas?ra=268.400000&dec=65.200000&inspect=realtile%3Anexus%2Ff200w-0001");
    expect(experimentsHref(["a/1", "b/2"])).toBe("/sky/experiments?tiles=a%2F1%2Cb%2F2");
    expect(experimentsHref([])).toBe("/sky/experiments");
  });
  it("parses pasted refs, keeping known sources once", () => {
    expect(parseRefs("nexus/f200w-0001, tile/x\npair/p1 nexus/f200w-0001 bogus/1 nope eval/a/b"))
      .toEqual(["nexus/f200w-0001", "tile/x", "pair/p1"]);
  });
});

describe("model specs", () => {
  it("orders production, mean, rbf, gates, members (natural)", () => {
    expect(sortSpecs(["member:member_10", "gate:b", "rbf", "member:member_2", "production", "mean", "gate:a"]))
      .toEqual(["production", "mean", "rbf", "gate:a", "gate:b", "member:member_2", "member:member_10"]);
  });
  it("shortens specs", () => {
    expect(specShort("member:member_196")).toBe("m196");
    expect(specShort("gate:26m")).toBe("gate 26m");
    expect(specShort("production")).toBe("production");
  });
  it("groups the catalogue and drops unavailable selections", () => {
    const models: ModelSpecRow[] = [
      { spec: "production", kind: "production", label: "P", available: true },
      { spec: "mean", kind: "mean", label: "M", available: true },
      { spec: "rbf", kind: "rbf", label: "R", available: false, reason: "stale" },
      { spec: "member:member_1", kind: "member", label: "m1", available: true },
      { spec: "gate:x", kind: "gate", label: "gx", available: false, reason: "member gone" },
    ];
    expect(groupModels(models).map((g) => [g.id, g.items.map((i) => i.spec)])).toEqual([
      ["production", ["production"]], ["mean", ["mean"]], ["rbf", ["rbf"]],
      ["gate", ["gate:x"]], ["member", ["member:member_1"]],
    ]);
    expect(runnableSelection(["rbf", "mean", "gate:x", "unknown"], models)).toEqual(["mean"]);
  });
});

describe("real tiles", () => {
  const lists: TileList[] = [
    { source: "nexus", label: "N", count: 2, tiles: [tile("nexus/a", "stale"), tile("nexus/b", "current")] },
    { source: "tile", label: "T", count: 1, tiles: [tile("tile/c"), tile("nexus/a", "stale")] },
  ];
  it("flattens every source once and counts production states", () => {
    const rows = flattenTiles([...lists, null]);
    expect(rows.map((r) => r.ref)).toEqual(["nexus/a", "nexus/b", "tile/c"]);
    expect(productionCounts(rows)).toEqual({ total: 3, current: 1, stale: 1, missing: 1 });
    expect(filterByState(rows, "missing").map((r) => r.ref)).toEqual(["tile/c"]);
    expect(filterByState(rows, "all")).toHaveLength(3);
  });
  it("lists a tile's models in order and picks the headline spec", () => {
    const row = tile("nexus/a", "stale", {
      "member:member_3": { state: "current" }, rbf: { state: "current", legacy: true, summary: { median_R: 1 } },
      production: { state: "stale" },
    });
    expect(tileModels(row)).toEqual([
      { spec: "production", state: "stale", legacy: false },
      { spec: "rbf", state: "current", legacy: true },
      { spec: "member:member_3", state: "current", legacy: false },
    ]);
    expect(headlineSpec(row)).toBe("rbf");
    expect(headlineSpec(tile("x/y"))).toBeNull();
  });
});

describe("experiment metrics", () => {
  it("collects every computed spec (not the skipped ones)", () => {
    expect(recordSpecs(RECORD)).toEqual(["production", "member:member_192"]);
  });
  it("tabulates model × band, pooled or per tile", () => {
    const pooled = metricRows(RECORD, "pooled");
    expect(pooled.map((r) => r.key)).toEqual(["production|VIS", "production|Y_E", "member:member_192|VIS"]);
    expect(pooled[0]).toMatchObject({ spec: "production", band: "VIS", hole_pct: 5.2, label: "Production · spatial gate" });
    const one = metricRows(RECORD, "poster/p1");
    expect(one.map((r) => [r.key, r.hole_pct])).toEqual([["production|VIS", 7]]);
    expect(metricRows(RECORD, "nope/none")).toEqual([]);
  });
  it("builds per-band series with per-model offsets", () => {
    const series = bandSeries(RECORD, "pooled", "hole_pct");
    expect(series.map((s) => s.spec)).toEqual(["production", "member:member_192"]);
    expect(series[0].y).toEqual([5.2, 6.1, null, null]);
    expect(series[1].y).toEqual([2, null, null, null]);
    expect(series[0].x[0]).toBeLessThan(0);
    expect(series[1].x[0]).toBeGreaterThan(0);
    const one = bandSeries({ ...RECORD, models: ["production"], summary: { production: RECORD.summary!.production }, results: {} }, "pooled", "hole_pct");
    expect(one[0].x).toEqual([0, 1, 2, 3]);
  });
  it("pads chart domains: % from zero, ratios around one", () => {
    expect(seriesDomain(bandSeries(RECORD, "pooled", "hole_pct"), "hole_pct")[0]).toBe(0);
    const [lo, hi] = seriesDomain(bandSeries(RECORD, "pooled", "median_R"), "median_R");
    expect(lo).toBeLessThan(0.91);
    expect(hi).toBeGreaterThan(1);
    expect(seriesDomain([], "hole_pct")).toEqual([0, 1]);
  });
  it("summarises the experiment for the tracking log", () => {
    const md = experimentMarkdown(RECORD);
    expect(md).toContain("**Real-data experiment `20260926-101010-abcdef`** — poster core (done)");
    expect(md).toContain("- tiles (2): `poster/p1`, `nexus/f200w-0040`");
    expect(md).toContain("`gate:26m` (member 171 archived)");
    expect(md).toContain("| `production` | 5.2 / 6.1 / — / — | 12.5 | 0.930 | 0.998 |");
    expect(md).toContain("Errors (1)");
    expect(md.startsWith("#")).toBe(false);          // the store adds the heading
  });
});

describe("catalogue evaluation rows", () => {
  const rows = [
    { id: "a", grade: "A", ok: "True", state: "stale" },
    { id: "b", grade: "gal", ok: "True", state: "current" },
    { id: "c", grade: "A", ok: "False", state: null },
    { id: "d", grade: "syn-gal", ok: "True", state: "unknown" },
  ];
  it("filters by group, state and success", () => {
    expect(filterEvalRows(rows, [], "all", false).map((r) => r.id)).toEqual(["a", "b", "c", "d"]);
    expect(filterEvalRows(rows, ["A"], "all", true).map((r) => r.id)).toEqual(["a"]);
    expect(filterEvalRows(rows, [], "unknown", false).map((r) => r.id)).toEqual(["c", "d"]);
    expect(filterEvalRows(rows, ["gal", "syn-gal"], "current", true).map((r) => r.id)).toEqual(["b"]);
  });
});

describe("metricsPlan", () => {
  it("groups the tiles by the current outputs that still lack metrics (never an SR a tile does not have)", () => {
    const rows = [
      tile("nexus/a", "current", { production: { state: "current" }, mean: { state: "current", summary: { hole_pct_max: 1 } } }),
      tile("nexus/b", "current", { production: { state: "current" } }),
      tile("poster/p", "stale", { production: { state: "stale" }, rbf: { state: "current", legacy: true } }),
      tile("nexus/c", "current", { production: { state: "current", summary: { hole_pct_max: 2 } } }),
    ];
    expect(metricsPlan(rows)).toEqual([
      { specs: ["production"], refs: ["nexus/a", "nexus/b"] },
      { specs: ["rbf"], refs: ["poster/p"] },
    ]);
    expect(metricsPlan([rows[3]])).toEqual([]);
  });
});
