import { describe, expect, it } from "vitest";
import { makeFakeAladin } from "../../../sky/testing/fakeAladin";
import type { SkyPalette } from "./colorScale";
import { scaleFor } from "./colorScale";
import { CLIENT_LAYERS, normalisePayload, type LayerInfo, type LayerPayload } from "./layerModel";
import { LayerRenderer, type RenderSpec } from "./render";

const P: SkyPalette = {
  accent: "#4c9ffe", good: "#56d68a", warn: "#ffcc66", bad: "#ff7a7a", info: "#6fb2ff",
  muted: "#8a97ab", ink: "#e5edf7", select: "#00e5ff", hover: "#ffffff",
  cat: ["#111111", "#222222", "#333333", "#444444", "#555555", "#666666", "#777777", "#888888"],
};

const TILES_INFO: LayerInfo = {
  id: "nexus-tiles", label: "NEXUS × Euclid tiles", group: "real", kind: "polygons", count: 2, bbox: null,
  style: { color_by: "state" }, ready: true, reason: null, fill_action: null, description: "", url: "/api/sky/layer/nexus-tiles",
};
const poly = (ra: number): [number, number][] => [[ra, 65], [ra + 0.01, 65], [ra + 0.01, 65.007], [ra, 65.007]];
const TILES = normalisePayload({
  id: "nexus-tiles", label: "NEXUS", kind: "polygons", count: 2,
  features: [
    { id: "f200w-0000", polygon: poly(268.3), props: { state: "stale" }, inspect: { kind: "realtile", id: "nexus/f200w-0000" } },
    { id: "f200w-0001", polygon: poly(268.4), props: { state: "current" }, inspect: { kind: "realtile", id: "nexus/f200w-0001" } },
  ],
} as LayerPayload);

const spec = (over: Partial<RenderSpec> = {}): RenderSpec => ({
  info: TILES_INFO, features: TILES, dataVersion: 1, visible: true, opacity: 0.6,
  scale: scaleFor(TILES_INFO, TILES, P), lod: "shapes", markerScale: 1, ...over,
});

describe("layer renderer", () => {
  it("draws polygons zoomed in and centroid markers zoomed out", () => {
    const fake = makeFakeAladin();
    const r = new LayerRenderer({ A: fake.A, al: fake.al }, () => P);
    r.sync([spec()]);
    expect(fake.overlays).toHaveLength(1);
    expect(fake.overlays[0].items).toHaveLength(2);
    const [first] = fake.overlays[0].items as { opts: Record<string, unknown> }[];
    expect(first.opts.color).toBe(P.warn);        // stale
    expect(first.opts.opacity).toBe(0.6);
    // Resolves Aladin objects back to features (click → inspector).
    expect(r.featureOf(fake.overlays[0].items[1])?.inspect).toEqual({ kind: "tile", id: "nexus/f200w-0001" });

    r.sync([spec({ lod: "markers" })]);
    expect(fake.overlays).toHaveLength(0);
    expect(fake.catalogs).toHaveLength(1);
    const src = fake.catalogs[0].sources[0];
    expect(src.ra).toBeCloseTo(TILES[0].ra);
    expect(r.featureOf(src)?.key).toBe("f200w-0000");
  });

  it("hides without rebuilding and rebuilds only when the signature changes", () => {
    const fake = makeFakeAladin();
    const r = new LayerRenderer({ A: fake.A, al: fake.al }, () => P);
    r.sync([spec()]);
    const overlay = fake.overlays[0];
    r.sync([spec({ visible: false })]);
    expect(overlay.visible).toBe(false);
    expect(fake.overlays[0]).toBe(overlay);
    r.sync([spec()]);
    expect(overlay.visible).toBe(true);
    expect(fake.overlays[0]).toBe(overlay);
    r.sync([spec({ opacity: 0.2 })]);
    expect(fake.overlays).toHaveLength(1);
    expect(fake.overlays[0]).not.toBe(overlay);
    // A layer that leaves the spec list is removed.
    r.sync([]);
    expect(fake.overlays).toHaveLength(0);
  });

  it("coverage MOCs are created once; opacity is a live setter", () => {
    const fake = makeFakeAladin();
    const r = new LayerRenderer({ A: fake.A, al: fake.al }, () => P);
    const moc = CLIENT_LAYERS[0];
    const s = (over: Partial<RenderSpec>) => ({ info: moc, features: [], dataVersion: 0, visible: true, opacity: 0.3, scale: { type: "fixed" as const, color: P.accent, label: "" }, lod: "shapes" as const, markerScale: 1, ...over });
    r.sync([s({})]);
    expect(fake.mocs).toHaveLength(1);
    expect(fake.mocs[0].url).toMatch(/Moc\.fits$/);
    r.sync([s({ opacity: 0.7 })]);
    expect(fake.mocs).toHaveLength(1);
    expect(fake.mocs[0].opacity).toBe(0.7);
    r.sync([s({ visible: false })]);
    expect(fake.mocs[0].visible).toBe(false);
  });

  it("the MOC fill is a live switch (outline only when zoomed in)", () => {
    const fake = makeFakeAladin();
    const r = new LayerRenderer({ A: fake.A, al: fake.al }, () => P);
    const moc = CLIENT_LAYERS[0];
    const s = (over: Partial<RenderSpec>) => ({ info: moc, features: [], dataVersion: 0, visible: true, opacity: 0.3, scale: { type: "fixed" as const, color: P.accent, label: "" }, lod: "shapes" as const, markerScale: 1, ...over });
    r.sync([s({ fill: false })]);
    expect(fake.mocs[0].opts.fill).toBe(false);
    expect(fake.mocs[0].opts.perimeter).toBe(true);
    r.sync([s({ fill: true })]);
    expect(fake.mocs).toHaveLength(1);
    expect((fake.mocs[0] as unknown as { fill: boolean }).fill).toBe(true);
    r.sync([s({ fill: false })]);
    expect((fake.mocs[0] as unknown as { fill: boolean }).fill).toBe(false);
  });

  it("a coverage shape is drawn without its fill when zoomed in, and rebuilt when the fill returns", () => {
    const fake = makeFakeAladin();
    const r = new LayerRenderer({ A: fake.A, al: fake.al }, () => P);
    const fields = { ...TILES_INFO, id: "q1-fields", group: "coverage" as const };
    r.sync([spec({ info: fields, fill: false })]);
    const items = () => fake.overlays[fake.overlays.length - 1].items as { opts: Record<string, unknown> }[];
    expect(items()[0].opts.fill).toBe(false);
    r.sync([spec({ info: fields, fill: true })]);
    expect(fake.overlays).toHaveLength(1);
    expect(items()[0].opts.fill).toBe(true);
  });

  it("never asks Aladin for markers below its 5 px minimum (its sprite radius is size/2 − 2)", () => {
    const fake = makeFakeAladin();
    const r = new LayerRenderer({ A: fake.A, al: fake.al }, () => P);
    const stars = { ...TILES_INFO, id: "stars", kind: "points" as const, style: { shape: "square", size: 2 } };
    r.sync([spec({ info: stars, features: TILES, scale: scaleFor(stars, TILES, P) })]);
    expect(fake.catalogs[0].opts.sourceSize).toBe(5);
  });

  it("reports a layer Aladin cannot draw instead of throwing", () => {
    const fake = makeFakeAladin();
    fake.A.graphicOverlay = () => { throw new Error("boom"); };
    const errors: string[] = [];
    const r = new LayerRenderer({ A: fake.A, al: fake.al }, () => P, (l, e) => errors.push(`${l.id}:${(e as Error).message}`));
    expect(() => r.sync([spec()])).not.toThrow();
    expect(errors).toEqual(["nexus-tiles:boom"]);
  });

  it("outlines the selection region and the inspected feature", () => {
    const fake = makeFakeAladin();
    const r = new LayerRenderer({ A: fake.A, al: fake.al }, () => P);
    r.setRegion({ type: "circle", ra: 268.4, dec: 65.2, r: 0.1 });
    expect(fake.overlays).toHaveLength(1);
    r.setRegion(null);
    expect(fake.overlays).toHaveLength(0);
    r.setFocus(TILES[0]);
    expect(fake.overlays).toHaveLength(1);
    r.setFocus(null);
    expect(fake.overlays).toHaveLength(0);
  });
});
