/* The pure shaping behind the Combiner, Diagnostics and Images tabs (plan
 * "Console phases 2–6", Team M): held-out curves on one loss scale with
 * unique styles, the gate share per member (production from members.json, a
 * variant from its compare report), the per-set stamp caption, the spectrum's
 * SR-scale focus and non-colliding guides, the spread section's coverage
 * table, answer and z-pdf domain, the coherence labels, and SR's Δm vs LR
 * from the viewer's cubes. */
import { describe, expect, it } from "vitest";
import { bandIndex, changedConfigKnobs, share } from "./model";

describe("Δm vs LR from the viewer's cubes", () => {

  it("follows the viewer's band; a colour composite reads VIS", () => {
    expect(bandIndex(["VIS", "Y_E", "J_E", "H_E"], "J_E")).toBe(2);
    expect(bandIndex(["VIS", "Y_E"], "lupton")).toBe(0);
    expect(bandIndex(undefined, "Y_E")).toBe(0);
  });
});

describe("one rule for the numbers the Models tabs repeat", () => {
  it("formats every gate share to two significant figures", () => {
    expect([0.48, 0.046, 0.4, 0.47, 0.0048, 0.0002, 0, null].map(share)).toEqual(["48%", "4.6%", "40%", "47%", "0.48%", "<0.1%", "0%", "—"]);
  });

  it("counts the Config knobs a step reads that differ from their defaults", () => {
    const payload = {
      config: { a: 1, b: "x", c: 0.30000000000000004, d: 5 }, defaults: { a: 2, b: "x", c: 0.3, d: 5 },
      used_by: { a: ["ensemble_train"], b: ["ensemble_train"], c: ["ensemble_train"], d: ["generate"] },
    };
    expect(changedConfigKnobs(payload, "ensemble_train")).toEqual(["a"]);
    expect(changedConfigKnobs({ ...payload, config: { ...payload.config, d: 6 } }, "ensemble_train")).toEqual(["a"]);
    expect(changedConfigKnobs(payload, "ensemble_train", { except: ["a"] })).toEqual([]);
    expect(changedConfigKnobs(null, "ensemble_train")).toEqual([]);
  });
});
