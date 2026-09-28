import { describe, expect, it } from "vitest";
import {
  LAYERS_CODEC, OVERLAYS_CODEC, REGION_CODEC, experimentsHref, featureView, fmtCoord, fmtFovParam, parseGoto,
  parsePointId, parseProjection, patchSearch, pointTargetId, toggleLayer, updateLayer, type LayerSetting,
} from "./urlState";

describe("atlas URL codecs", () => {
  it("layers: ids with optional opacity and colour", () => {
    expect(LAYERS_CODEC.parse("moc-q1,nexus-tiles:0.5,stars:0.8:cat-3,q1-tiles::by.field")).toEqual([
      { id: "moc-q1" }, { id: "nexus-tiles", opacity: 0.5 }, { id: "stars", opacity: 0.8, color: "cat-3" },
      { id: "q1-tiles", color: "by.field" },
    ]);
    expect(LAYERS_CODEC.serialize([{ id: "a" }, { id: "b", opacity: 0.25 }, { id: "c", color: "bad" }])).toBe("a,b:0.25,c::bad");
    // Everything off is "-", distinct from the default (absent).
    expect(LAYERS_CODEC.parse("-")).toEqual([]);
    expect(LAYERS_CODEC.serialize([])).toBe("-");
    // Junk is dropped; opacity is clamped; duplicates collapse (last wins).
    expect(LAYERS_CODEC.parse("a:9,,b:x,a:0.3")).toEqual([{ id: "a", opacity: 0.3 }, { id: "b" }]);
    expect(LAYERS_CODEC.parse("a:-1")).toEqual([{ id: "a", opacity: 0 }]);
    expect(LAYERS_CODEC.parse("bad id!")).toEqual([]);
  });

  it("round-trips layers", () => {
    const raw = "moc-q1,nexus-tiles:0.5,stars:0.8:cat-3";
    expect(LAYERS_CODEC.serialize(LAYERS_CODEC.parse(raw)!)).toBe(raw);
  });

  it("toggles and updates layer settings immutably", () => {
    const base: LayerSetting[] = [{ id: "a" }, { id: "b", opacity: 0.4 }];
    expect(toggleLayer(base, "a")).toEqual([{ id: "b", opacity: 0.4 }]);
    expect(toggleLayer(base, "c")).toEqual([...base, { id: "c" }]);
    expect(updateLayer(base, "b", { opacity: 0.9 })).toEqual([{ id: "a" }, { id: "b", opacity: 0.9 }]);
    expect(updateLayer(base, "b", { color: undefined, opacity: undefined })).toEqual([{ id: "a" }, { id: "b" }]);
    expect(updateLayer(base, "z", { opacity: 0.1 })).toBe(base);
  });

  it("overlay HiPS: ids with opacity", () => {
    expect(OVERLAYS_CODEC.parse("jwst-nircam:0.6,cds-f200w")).toEqual([{ id: "jwst-nircam", opacity: 0.6 }, { id: "cds-f200w" }]);
    expect(OVERLAYS_CODEC.serialize([])).toBe("");
  });

  it("regions: circles and polygons in degrees", () => {
    expect(REGION_CODEC.parse("c:268.46,65.2,0.1")).toEqual({ type: "circle", ra: 268.46, dec: 65.2, r: 0.1 });
    expect(REGION_CODEC.parse("p:1,2;3,4;5,-6")).toEqual({ type: "polygon", points: [[1, 2], [3, 4], [5, -6]] });
    expect(REGION_CODEC.parse("p:1,2;3,4")).toBeUndefined();
    expect(REGION_CODEC.parse("c:1,2")).toBeUndefined();
    expect(REGION_CODEC.parse("c:1,2,-3")).toBeUndefined();
    expect(REGION_CODEC.parse("x")).toBeUndefined();
    expect(REGION_CODEC.serialize({ type: "circle", ra: 268.4612345, dec: 65.2, r: 0.1 })).toBe("c:268.46123,65.2,0.1");
    expect(REGION_CODEC.serialize({ type: "polygon", points: [[1, 2], [3, 4], [5, 6]] })).toBe("p:1,2;3,4;5,6");
    expect(REGION_CODEC.serialize(null)).toBeNull();
  });

  it("formats view numbers compactly", () => {
    expect(fmtCoord(268.461234567)).toBe("268.46123");
    expect(fmtCoord(-0.000001)).toBe("0");
    expect(fmtFovParam(360)).toBe("360");
    expect(fmtFovParam(0.123456)).toBe("0.1235");
    expect(fmtFovParam(12.3456)).toBe("12.35");
  });

  it("parses projections and goto targets", () => {
    expect(parseProjection("ait")).toBe("AIT");
    expect(parseProjection("XYZ")).toBeUndefined();
    expect(parseGoto("268.46 65.2")).toEqual({ kind: "coord", ra: 268.46, dec: 65.2 });
    expect(parseGoto("17:53:50.8 +65:11:47")).toMatchObject({ kind: "coord" });
    expect(parseGoto("  M31 ")).toEqual({ kind: "name", name: "M31" });
    expect(parseGoto("   ")).toBeNull();
  });

  it("point card ids round-trip", () => {
    expect(pointTargetId(268.461234567, -8.4)).toBe("at/268.46123,-8.4");
    expect(parsePointId("at/268.46123,-8.4")).toEqual({ ra: 268.46123, dec: -8.4 });
    expect(parsePointId("268.4,65.2")).toEqual({ ra: 268.4, dec: 65.2 });
    expect(parsePointId("at/400,0")).toBeNull();
    expect(parsePointId("at/x,y")).toBeNull();
  });

  it("links to Sky › Compare with tiles preselected", () => {
    expect(experimentsHref(["nexus/f200w-0001", "archive/007"])).toBe("/sky/compare?tiles=nexus%2Ff200w-0001%2Carchive%2F007");
    expect(experimentsHref([])).toBe("/sky/compare");
  });

  it("patches a query string keeping the other params' spelling", () => {
    const q = "?layers=moc-q1,nexus-tiles&ra=1&inspect=tile:nexus/12&goto=M31";
    expect(patchSearch(q, { ra: "268.4", dec: "65.2", goto: null }))
      .toBe("?layers=moc-q1,nexus-tiles&ra=268.4&inspect=tile:nexus/12&dec=65.2");
    expect(patchSearch("", { ra: "1", fov: null })).toBe("?ra=1");
    expect(patchSearch("?goto=x", { goto: null })).toBe("");
    // Duplicates of a patched key collapse; values are encoded (not , : /).
    expect(patchSearch("?a=1&a=2&b=x%20y", { a: "p q,r" })).toBe("?a=p%20q,r&b=x%20y");
  });
});

describe("featureView", () => {
  it("frames an inspected feature a few times its size", () => {
    const tile = featureView({ ra: 268.39, dec: 65.1, sizeDeg: 25.6 / 3600 });
    expect(tile.ra).toBe(268.39);
    expect(tile.fov).toBeCloseTo((25.6 * 6) / 3600, 6);          // ~2.6′
    expect(featureView({ ra: 1, dec: 2, sizeDeg: 0 }).fov).toBe(0.02);   // a point: 1.2′
    expect(featureView({ ra: 1, dec: 2, sizeDeg: 20 }).fov).toBe(30);    // a deep field: capped
  });
});
