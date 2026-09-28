import { describe, expect, it } from "vitest";
import {
  colorOf, colorOptions, legendFor, lerpColor, scaleFor, withAlpha, type SkyPalette,
} from "./colorScale";
import type { LayerInfo, SkyFeature } from "./layerModel";

const P: SkyPalette = {
  accent: "#4c9ffe", good: "#56d68a", warn: "#ffcc66", bad: "#ff7a7a", info: "#6fb2ff",
  muted: "#8a97ab", ink: "#e5edf7", select: "#00e5ff", hover: "#ffffff",
  cat: ["#000001", "#000002", "#000003", "#000004", "#000005", "#000006", "#000007", "#000008"],
};

const layer = (over: Partial<LayerInfo>): LayerInfo => ({
  id: "x", label: "X", group: "real", kind: "polygons", count: 0, bbox: null, style: {},
  ready: true, reason: null, fill_action: null, description: "", url: null, ...over,
});
const feat = (props: Record<string, unknown>, key = "k"): SkyFeature => ({
  layer: "x", key, label: key, ra: 0, dec: 0, sizeDeg: 0, props, inspect: null,
});

describe("colour scales", () => {
  it("interpolates and adds alpha to token colours", () => {
    expect(lerpColor("#000000", "#ffffff", 0.5)).toBe("#808080");
    expect(lerpColor("#102030", "#102030", 0.3)).toBe("#102030");
    expect(withAlpha("#ff0000", 0.5)).toBe("rgba(255, 0, 0, 0.5)");
    expect(withAlpha("#f00", 1)).toBe("rgba(255, 0, 0, 1)");
    expect(withAlpha("gray", 0.5)).toBe("gray");
  });

  it("colours real tiles by SR state (current / stale / missing)", () => {
    const s = scaleFor(layer({ id: "nexus-tiles", style: { color_by: "state" } }), [], P);
    expect(colorOf(s, { state: "current" })).toBe(P.good);
    expect(colorOf(s, { state: "stale" })).toBe(P.warn);
    expect(colorOf(s, { state: "missing" })).toBe(P.muted);
    expect(legendFor(s).items!.map((i) => i.label)).toEqual(["current", "stale", "missing"]);
  });

  it("colours lens candidates by grade and archive fields by field", () => {
    const g = scaleFor(layer({ id: "lens-candidates", kind: "points", style: { color_by: "grade" } }), [], P);
    expect(colorOf(g, { grade: "A" })).toBe(P.bad);
    expect(colorOf(g, { grade: "B" })).toBe(P.warn);
    expect(colorOf(g, { grade: "Z" })).toBe(P.muted);
    const f = scaleFor(layer({ id: "archive-fields", style: { color_by: "field" } }), [], P);
    expect(colorOf(f, { field: "EDF-N" })).toBe(P.cat[0]);
    expect(colorOf(f, { field: "EDF-S" })).toBe(P.cat[1]);
  });

  it("colours flux ratios on a diverging scale around 1 (holes are red)", () => {
    const s = scaleFor(layer({ id: "eval-objects", kind: "points", style: { color_by: "flux_ratio_sr_over_lr" } }), [], P);
    expect(s.type).toBe("diverging");
    expect(colorOf(s, { flux_ratio_sr_over_lr: 1 })).toBe(P.good);
    expect(colorOf(s, { flux_ratio_sr_over_lr: 0.2 })).toBe(P.bad);
    expect(colorOf(s, { flux_ratio_sr_over_lr: 3 })).toBe(P.info);
    expect(colorOf(s, { flux_ratio_sr_over_lr: null })).toBe(P.muted);
  });

  it("colours Q1 tiles by sky level from the data range; rejected tiles are marked", () => {
    const feats = [feat({ vis_level_e: 20 }), feat({ vis_level_e: 40 }), feat({ vis_level_e: null, state: "rejected" })];
    const s = scaleFor(layer({ id: "q1-tiles", style: { color_by: "vis_level_e" } }), feats, P);
    expect(s.type).toBe("sequential");
    if (s.type !== "sequential") return;
    expect(s.domain).toEqual([20, 40]);
    expect(colorOf(s, { vis_level_e: 20 })).not.toBe(colorOf(s, { vis_level_e: 40 }));
    expect(colorOf(s, { state: "rejected" })).toBe(P.bad);
    const lg = legendFor(s);
    expect(lg.gradient!.min).toBe("20");
    expect(lg.items!.map((i) => i.label)).toContain("rejected");
  });

  it("stars by magnitude: brighter is the high end of the ramp", () => {
    const feats = [15, 16, 17, 18, 19].map((m) => feat({ mag: m }));
    const s = scaleFor(layer({ id: "stars", kind: "points", style: { color_by: "mag" } }), feats, P);
    expect(s.type === "sequential" && s.reverse).toBe(true);
    expect(colorOf(s, { mag: 15 })).not.toBe(colorOf(s, { mag: 19 }));
  });

  it("fixed colours come from tokens, never from the backend hex", () => {
    const s = scaleFor(layer({ id: "poster", style: { color: "#e84393" } }), [], P);
    expect(s.type).toBe("fixed");
    expect(colorOf(s, {})).not.toBe("#e84393");
    expect(P.cat).toContain(colorOf(s, {}));
  });

  it("honours a colour override: a token or colour-by a property", () => {
    const l = layer({ id: "nexus-tiles", style: { color_by: "state" } });
    expect(colorOf(scaleFor(l, [], P, "bad"), { state: "current" })).toBe(P.bad);
    expect(colorOf(scaleFor(l, [], P, "cat-3"), {})).toBe(P.cat[3]);
    const byField = scaleFor(l, [feat({ field: "EDF-N" })], P, "by.field");
    expect(byField.type).toBe("categorical");
    expect(colorOf(byField, { field: "EDF-N" })).toBe(P.cat[0]);
  });

  it("offers colour-by choices from the data", () => {
    const feats = [feat({ state: "stale", field: "EDF-N", ra: 1, label: "a" }), feat({ state: "current", field: "EDF-S", ra: 2, label: "b" })];
    const opts = colorOptions(layer({ id: "nexus-tiles", style: { color_by: "state" } }), feats);
    const values = opts.map((o) => o.value);
    expect(values[0]).toBe("");
    expect(values).toContain("by.field");
    expect(values).not.toContain("by.ra");
    expect(values).not.toContain("by.label");
    expect(values).toContain("cat-0");
  });
});
