import { describe, expect, it } from "vitest";
import {
  DEFAULT_LAYERS, atlasTarget, featureFacts, findFeatureByTarget, footprintFeatures, groupLayers, normalisePayload,
  sourceTargetParts, stubLayerInfo, tileRefOf, tileTargetId, withClientLayers, withStubLayers, type LayerInfo,
  type LayerPayload,
} from "./layerModel";

const info = (over: Partial<LayerInfo>): LayerInfo => ({
  id: "x", label: "X", group: "real", kind: "polygons", count: 0, bbox: null, style: {},
  ready: true, reason: null, fill_action: null, description: "", url: "/api/sky/layer/x", ...over,
});

const NEXUS: LayerPayload = {
  id: "nexus-tiles", label: "NEXUS × Euclid tiles", group: "results", kind: "polygons", count: 1,
  features: [{
    id: "f200w-0000", inspect: { id: "nexus/f200w-0000", kind: "realtile" },
    polygon: [[268.3855965, 65.09367503], [268.36877695, 65.09368528], [268.36879904, 65.10076859], [268.38562306, 65.10075833]],
    props: { dec: 65.0972175, ra: 268.3771838, label: "NEXUS F200W tile 0000", state: "stale", field: "EDF-N", ref: "nexus/f200w-0000" },
  }],
};

const STARS: LayerPayload = {
  id: "stars", label: "Stars (FASRC catalogue)", group: "catalogues", kind: "points", count: 2,
  columns: ["ra", "dec", "mag", "flags"],
  rows: [[272.873461, 68.387353, 17.8, 7], [64.928031, -48.34014, 18.2, 15]],
  inspect: { id_column: null, kind: "source", prefix: "stars/" },
};

const LENSES: LayerPayload = {
  id: "lens-candidates", label: "Q1 lens candidates", group: "catalogues", kind: "points", count: 1,
  columns: ["ra", "dec", "grade", "subset", "id", "field"],
  rows: [[58.6875229, -51.2840675, "C", "discovery_engine", "102018212_NEG586875228512840674", "EDF-S"]],
  inspect: { id_column: "id", kind: "source", prefix: "lens-candidates/" },
};

const FIELDS: LayerPayload = {
  id: "q1-fields", label: "Q1 deep fields", group: "coverage", kind: "circles", count: 1,
  features: [{ id: "EDF-N", ra: 269.733, dec: 66.018, radius_deg: 6, props: { name: "EDF-N" }, inspect: { id: "q1-fields/EDF-N", kind: "source" } }],
};

describe("layer model", () => {
  it("normalises polygons with their recorded centre, size and atlas inspector target", () => {
    const [f] = normalisePayload(NEXUS);
    expect(f.layer).toBe("nexus-tiles");
    expect(f.key).toBe("f200w-0000");
    expect(f.label).toBe("NEXUS F200W tile 0000");
    expect(f.ra).toBeCloseTo(268.3771838, 6);
    expect(f.polygon).toHaveLength(4);
    expect(f.sizeDeg * 3600).toBeGreaterThan(30);
    // Real tiles open the atlas tile card (kind `tile`), not the results card.
    expect(f.inspect).toEqual({ kind: "tile", id: "nexus/f200w-0000" });
    expect(tileRefOf(f)).toBe("nexus/f200w-0000");
  });

  it("normalises point rows: keys from the id column or the row index", () => {
    const stars = normalisePayload(STARS);
    expect(stars.map((s) => s.key)).toEqual(["0", "1"]);
    expect(stars[1].inspect).toEqual({ kind: "source", id: "stars/1" });
    expect(stars[0].props.mag).toBe(17.8);
    expect(stars[0].label).toMatch(/17\.8/);
    const [lens] = normalisePayload(LENSES);
    expect(lens.key).toBe("102018212_NEG586875228512840674");
    expect(lens.inspect).toEqual({ kind: "source", id: "lens-candidates/102018212_NEG586875228512840674" });
    expect(lens.props.grade).toBe("C");
    expect(tileRefOf(lens)).toBeNull();
  });

  it("normalises circles with a diameter", () => {
    const [c] = normalisePayload(FIELDS);
    expect(c.radius).toBe(6);
    expect(c.sizeDeg).toBe(12);
    expect(c.label).toBe("EDF-N");
  });

  it("skips rows and features without finite coordinates", () => {
    const bad: LayerPayload = { ...STARS, rows: [[null, 1, 2, 3], ["x", 2, 3, 4], [1, 2, 3, 4]] };
    expect(normalisePayload(bad)).toHaveLength(1);
  });

  it("maps inspector targets for the atlas", () => {
    expect(atlasTarget({ kind: "realtile", id: "archive/000" })).toEqual({ kind: "tile", id: "archive/000" });
    expect(atlasTarget({ kind: "source", id: "stars/3" })).toEqual({ kind: "source", id: "stars/3" });
    expect(atlasTarget(null)).toBeNull();
  });

  it("maps palette tile ids to real-tile refs", () => {
    expect(tileTargetId("nexus/12")).toBe("nexus/f200w-0012");
    expect(tileTargetId("nexus/0012")).toBe("nexus/f200w-0012");
    expect(tileTargetId("12")).toBe("nexus/f200w-0012");
    expect(tileTargetId("nexus/f200w-0012")).toBe("nexus/f200w-0012");
    expect(tileTargetId("archive/007")).toBe("archive/007");
    expect(tileTargetId("eval/gal_1/x")).toBe("eval/gal_1/x");
  });

  it("splits source targets into layer and id", () => {
    expect(sourceTargetParts("lens-candidates/1020_NEG")).toEqual({ layer: "lens-candidates", id: "1020_NEG" });
    expect(sourceTargetParts("at/268.4,65.2")).toEqual({ layer: "at", id: "268.4,65.2" });
    expect(sourceTargetParts("nothing")).toBeNull();
  });

  it("adds the client-side coverage MOCs and groups layers in a stable order", () => {
    const server = [info({ id: "stars", group: "inputs" }), info({ id: "q1-tiles", group: "coverage" }), info({ id: "nexus-tiles", group: "real" }),
      info({ id: "lens-candidates", group: "targets" })];
    const all = withClientLayers(server);
    expect(all.map((l) => l.id).slice(0, 2)).toEqual(["moc-q1", "moc-jwst"]);
    expect(all.find((l) => l.id === "moc-q1")!.kind).toBe("moc");
    const groups = groupLayers(all);
    expect(groups.map((g) => [g.group, g.label])).toEqual([
      ["real", "Real tiles"], ["targets", "Targets"], ["inputs", "Scene inputs"], ["coverage", "Coverage"],
    ]);
    expect(groups[3].layers.map((l) => l.id)).toEqual(["moc-q1", "moc-jwst", "q1-tiles"]);
  });

  it("files an older server's groups under the new ones", () => {
    const all = withClientLayers([info({ id: "stars", group: "catalogues" as never }), info({ id: "nexus-tiles", group: "results" as never })]);
    expect(Object.fromEntries(all.map((l) => [l.id, l.group]))).toMatchObject({ stars: "inputs", "nexus-tiles": "real", "moc-q1": "coverage" });
    expect(DEFAULT_LAYERS).toContain("nexus-tiles");
    expect(DEFAULT_LAYERS).toContain("moc-q1");
  });

  it("turns per-view JWST footprints into polygon features that inspect as the MAST row", () => {
    const feats = footprintFeatures({
      ra: 268.4, dec: 65.2, r: 0.5, footprints: [{
        obs_id: "jw01234-o001", instrument: "NIRCAM", filters: "F200W", target: "NEXUS",
        polygons: [[[268.4, 65.2], [268.5, 65.2], [268.5, 65.3]], [[1, 2]]],
      }],
    });
    expect(feats).toHaveLength(1);
    expect(feats[0].inspect).toEqual({ kind: "source", id: "jwst-mast/jw01234-o001" });
    expect(feats[0].label).toBe("NEXUS");
    expect(footprintFeatures(null)).toEqual([]);
  });

  it("finds the loaded feature an inspector target points at", () => {
    const byLayer = { "nexus-tiles": normalisePayload(NEXUS), stars: normalisePayload(STARS) };
    expect(findFeatureByTarget({ kind: "tile", id: "nexus/0" }, byLayer)?.key).toBe("f200w-0000");
    // `realtile:` opens the same real-tile card, so the atlas highlights it too
    expect(findFeatureByTarget({ kind: "realtile", id: "nexus/f200w-0000" }, byLayer)?.key).toBe("f200w-0000");
    expect(findFeatureByTarget({ kind: "source", id: "stars/1" }, byLayer)?.key).toBe("1");
    expect(findFeatureByTarget({ kind: "source", id: "at/1,2" }, byLayer)).toBeNull();
    expect(findFeatureByTarget({ kind: "member", id: "member_1" }, byLayer)).toBeNull();
    expect(findFeatureByTarget(null, byLayer)).toBeNull();
  });

  it("lists a few human facts for hover tooltips", () => {
    const [f] = normalisePayload(NEXUS);
    const facts = featureFacts(f);
    expect(facts).toContainEqual(["state", "stale"]);
    expect(facts).toContainEqual(["field", "EDF-N"]);
    expect(facts.length).toBeLessThanOrEqual(4);
  });

  it("stands in for layers whose payload arrived before the catalogue", () => {
    const payload = { id: "nexus-tiles", label: "NEXUS × Euclid tiles", group: "results", kind: "polygons", count: 445, features: [] } as LayerPayload;
    const stub = stubLayerInfo("nexus-tiles", payload);
    expect(stub).toMatchObject({ id: "nexus-tiles", label: "NEXUS × Euclid tiles", group: "real", kind: "polygons", count: 445, url: null, style: {} });
    expect(stubLayerInfo("x").group).toBe("real");
    const known = withClientLayers([]);
    const merged = withStubLayers(known, ["moc-q1", "nexus-tiles", "stars"], { "nexus-tiles": { payload }, stars: { payload: null } });
    expect(merged.map((l) => l.id)).toEqual([...known.map((l) => l.id), "nexus-tiles"]);
    // Nothing to add: a copy of the catalogue.
    expect(withStubLayers(known, ["moc-q1"], {})).toEqual(known);
  });
});
