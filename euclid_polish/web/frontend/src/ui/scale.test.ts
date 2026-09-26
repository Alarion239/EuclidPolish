import { describe, expect, it, vi } from "vitest";
import { LOG_STEPS, effectiveScale, fromSliderPos, roundSig, toSliderPos } from "./scale";

describe("slider scale", () => {
  it("passes linear values through, snapping to the step and clamping", () => {
    const o = { min: 0, max: 10, step: 0.5 };
    expect(toSliderPos(3.2, o)).toBe(3.2);
    expect(fromSliderPos(3.2, o)).toBe(3);
    expect(fromSliderPos(3.3, o)).toBe(3.5);
    expect(fromSliderPos(12, o)).toBe(10);
    expect(fromSliderPos(0.1 + 0.2, { min: 0, max: 1, step: 0.1 })).toBe(0.3);   // no float fuzz
  });
  it("maps a log scale over decades with readable snapping", () => {
    const o = { min: 0.1, max: 1e4, scale: "log" as const };
    expect(toSliderPos(0.1, o)).toBe(0);
    expect(toSliderPos(1e4, o)).toBe(LOG_STEPS);
    expect(toSliderPos(100, o)).toBe(600);             // 3 of 5 decades
    expect(fromSliderPos(600, o)).toBe(100);
    expect(fromSliderPos(0, o)).toBe(0.1);
    expect(fromSliderPos(LOG_STEPS, o)).toBe(1e4);
    // round trip within 1% everywhere
    for (const v of [0.13, 2.7, 55, 999, 4321]) {
      expect(Math.abs(fromSliderPos(toSliderPos(v, o), o) / v - 1)).toBeLessThan(0.012);
    }
  });
  it("clamps non-positive log inputs to min", () => {
    expect(toSliderPos(-5, { min: 1, max: 100, scale: "log" })).toBe(0);
  });
  it("falls back to linear (with a warning) for a log range it cannot map, never throwing or NaN", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const bad = { min: 0, max: 100, scale: "log" as const };
    expect(effectiveScale(bad)).toBe("linear");
    expect(toSliderPos(40, bad)).toBe(40);
    expect(fromSliderPos(40, bad)).toBe(40);
    const flat = { min: 5, max: 5, scale: "log" as const };
    expect(effectiveScale(flat)).toBe("linear");
    expect(toSliderPos(5, flat)).toBe(5);
    expect(fromSliderPos(5, flat)).toBe(5);
    expect(effectiveScale({ min: 0.1, max: 10, scale: "log" })).toBe("log");
    expect(effectiveScale({ min: 0.1, max: 10 })).toBe("linear");
    expect(warn).toHaveBeenCalledTimes(2);                       // once per bad range
    toSliderPos(1, bad);
    expect(warn).toHaveBeenCalledTimes(2);
    warn.mockRestore();
  });
  it("rounds to significant digits", () => {
    expect(roundSig(123456, 3)).toBe(123000);
    expect(roundSig(0.0012345, 2)).toBe(0.0012);
    expect(roundSig(0)).toBe(0);
  });
});
