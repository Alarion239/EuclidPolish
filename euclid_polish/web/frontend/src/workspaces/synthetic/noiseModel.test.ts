import { describe, expect, it } from "vitest";
import { fieldsLabel, noiseProvenance, noiseSteps, stackedTopsByField } from "./noiseModel";
import { NOISE } from "./testFixtures";

describe("Synthetic › Noise words", () => {
  it("names the fields once (EDF-N/S/F) and the provenance in one footer line", () => {
    expect(fieldsLabel(["EDF-N", "EDF-S", "EDF-F"])).toBe("EDF-N/S/F");
    expect(fieldsLabel(["EDF-N", "COSMOS"])).toBe("EDF-N, COSMOS");
    expect(noiseProvenance(NOISE)).toBe("NOISE_MODEL v5 · Q1_R1 · retrieved 2026-09-19 · mer_noise_levels.json");
  });

  it("says how a scene gets its noise in three sentences from the generator's settings", () => {
    const [pick, depth, draw] = noiseSteps(NOISE);
    expect(pick).toBe("Each scene takes the four band levels of one of the 3 measured Q1 positions, picked uniformly.");
    expect(depth).toBe("Its depth is scaled by ×0.99–1.01, and in 10% of scenes a 20–50% strip steps ×1.1–1.45, like a pointing seam.");
    expect(draw).toMatch(/^The noise is σ = √\(level² \+ signal\) × scale/);
    const plain = noiseSteps({ ...NOISE, generator: { ...NOISE.generator, draws_measured_levels: false, scene_scale: null, region: null } });
    expect(plain[0]).toMatch(/band median levels/);
    expect(plain[1]).toBe("Its depth is not jittered.");
  });

  it("stacks only the shown fields, the top bar first", () => {
    const { tops, order } = stackedTopsByField({ a: [1, 2], b: [3, 0], c: [1, 1] }, ["a", "b", "c"], ["b"], 2);
    expect(order).toEqual(["c", "a"]);
    expect(tops).toEqual([[2, 3], [1, 2]]);
  });
});
