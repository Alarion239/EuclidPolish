import { describe, expect, it } from "vitest";
import { featureAt, featuresAt, preferHit } from "./hitTest";
import { normalisePayload, type LayerInfo, type LayerPayload, type SkyFeature } from "./layerModel";
import type { RenderSpec } from "./render";

const info = (id: string, kind: LayerInfo["kind"]): LayerInfo => ({
  id, label: id, group: "results", kind, count: 0, bbox: null, style: {}, ready: true, reason: null,
  fill_action: null, description: "", url: null,
});
const sq = (ra: number, dec: number, side: number): [number, number][] =>
  [[ra, dec], [ra + side, dec], [ra + side, dec + side], [ra, dec + side]];

const TILES = normalisePayload({
  id: "nexus-tiles", label: "NEXUS", kind: "polygons", count: 2,
  features: [
    { id: "t0", polygon: sq(268.40, 65.10, 0.007), props: {}, inspect: { kind: "realtile", id: "nexus/f200w-0000" } },
    { id: "t1", polygon: sq(268.42, 65.10, 0.007), props: {}, inspect: { kind: "realtile", id: "nexus/f200w-0001" } },
  ],
} as LayerPayload);
const Q1 = normalisePayload({
  id: "q1-tiles", label: "Q1", kind: "polygons", count: 1,
  features: [{ id: "102159776", polygon: sq(268.0, 64.9, 0.6), props: {}, inspect: { kind: "source", id: "q1-tiles/102159776" } }],
} as LayerPayload);
const FIELDS = normalisePayload({
  id: "q1-fields", label: "fields", kind: "circles", count: 1,
  features: [{ id: "EDF-N", ra: 269.733, dec: 66.018, radius_deg: 6, props: {}, inspect: { kind: "source", id: "q1-fields/EDF-N" } }],
} as LayerPayload);
const MAST = normalisePayload({
  id: "jwst-mast", label: "MAST", kind: "points", count: 1,
  columns: ["ra", "dec", "obs_id", "polygons"],
  rows: [[268.5, 65.3, "jw01", [sq(268.49, 65.29, 0.02)]]],
  inspect: { kind: "source", prefix: "jwst-mast/", id_column: "obs_id" },
} as LayerPayload);

const spec = (id: string, kind: LayerInfo["kind"], features: SkyFeature[], over: Partial<RenderSpec> = {}): RenderSpec => ({
  info: info(id, kind), features, dataVersion: 1, visible: true, opacity: 1,
  scale: { type: "fixed", color: "#fff", label: id }, lod: "shapes", markerScale: 1, ...over,
});

const ALL = [spec("q1-fields", "circles", FIELDS), spec("q1-tiles", "polygons", Q1), spec("nexus-tiles", "polygons", TILES)];

describe("sky hit testing", () => {
  it("a click inside a tile picks the tile, not the Q1 tile or the field around it", () => {
    const f = featureAt(268.4035, 65.1035, ALL);
    expect(f?.inspect).toEqual({ kind: "tile", id: "nexus/f200w-0000" });
    expect(featuresAt(268.4035, 65.1035, ALL).map((x) => x.layer)).toEqual(["nexus-tiles", "q1-tiles", "q1-fields"]);
  });

  it("between tiles the containing Q1 tile wins; outside it the field circle", () => {
    expect(featureAt(268.415, 65.1035, ALL)?.inspect?.id).toBe("q1-tiles/102159776");
    expect(featureAt(271.0, 66.0, ALL)?.inspect?.id).toBe("q1-fields/EDF-N");
    expect(featureAt(10, -40, ALL)).toBeNull();
  });

  it("ignores hidden layers, marker-LOD layers and MOCs", () => {
    const hidden = [spec("nexus-tiles", "polygons", TILES, { visible: false })];
    expect(featureAt(268.4035, 65.1035, hidden)).toBeNull();
    const markers = [spec("nexus-tiles", "polygons", TILES, { lod: "markers" })];
    expect(featureAt(268.4035, 65.1035, markers)).toBeNull();
    expect(featureAt(268.4035, 65.1035, [spec("moc-q1", "moc", [])])).toBeNull();
  });

  it("points with observation footprints hit inside their footprints when zoomed in", () => {
    const s = [spec("jwst-mast", "points", MAST)];
    expect(featureAt(268.5, 65.3, s)?.inspect).toEqual({ kind: "source", id: "jwst-mast/jw01" });
    expect(featureAt(268.6, 65.3, s)).toBeNull();
    expect(featureAt(268.5, 65.3, [spec("jwst-mast", "points", MAST, { lod: "markers" })])).toBeNull();
  });

  it("prefers the more specific of Aladin's outline hit and ours", () => {
    const [tile] = TILES, [q1] = Q1;
    expect(preferHit(q1, tile)).toBe(tile);
    expect(preferHit(tile, q1)).toBe(tile);
    expect(preferHit(null, tile)).toBe(tile);
    expect(preferHit(q1, null)).toBe(q1);
    const marker = { ...tile, sizeDeg: 0 };
    expect(preferHit(marker, tile)).toBe(marker);
  });
});
