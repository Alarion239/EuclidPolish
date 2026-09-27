import { describe, expect, it } from "vitest";
import { areaFactor, areaReference, parsePerArea } from "./area";

const LR = { pixscale: 0.1, unit: "e-" };
const HR = { pixscale: 0.05, unit: "e-" };
const JWST = { pixscale: 0.031, unit: "MJy/sr" };

describe("per-unit-area display normalisation", () => {
  it("references the coarsest shown e⁻ pixel, only when the scales differ", () => {
    expect(areaReference([LR, HR])).toBe(0.1);
    expect(areaReference([HR, HR])).toBe(0);                 // one scale: nothing to normalise
    expect(areaReference([HR, JWST])).toBe(0);               // MJy/sr is already per area
    expect(areaReference([LR, HR, JWST])).toBe(0.1);
    expect(areaReference([])).toBe(0);
  });

  it("shows an HR / SR pixel ×4 next to LR (the same surface brightness looks the same)", () => {
    // Records record 0: LR 1323 e⁻, HR 433.7 e⁻ at the galaxy centre (VIS 21.48 vs 21.51 AB)
    expect(areaFactor(HR, 0.1, true)).toBeCloseTo(4);
    expect(433.7 * areaFactor(HR, 0.1, true)).toBeGreaterThan(1323 * 0.9);
    expect(areaFactor(LR, 0.1, true)).toBe(1);
  });

  it("is 1 when off, without a reference, or for a tier in another unit", () => {
    expect(areaFactor(HR, 0.1, false)).toBe(1);
    expect(areaFactor(HR, 0, true)).toBe(1);
    expect(areaFactor(JWST, 0.1, true)).toBe(1);
    expect(areaFactor({ pixscale: 0, unit: "e-" }, 0.1, true)).toBe(1);
  });

  it("is on unless turned off", () => {
    expect(parsePerArea(null)).toBe(true);
    expect(parsePerArea("1")).toBe(true);
    expect(parsePerArea("0")).toBe(false);
  });
});
