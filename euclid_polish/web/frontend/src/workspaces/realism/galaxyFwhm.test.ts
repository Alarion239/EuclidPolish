/* The FWHM | VIS 2FWHM interval (was test/galaxyFwhm.test.ts). */
import { describe, expect, it } from "vitest";
import { conditionalFwhmInterval } from "./galaxyFwhm";

describe("conditionalFwhmInterval", () => {
  it("interpolates the 16th and 84th percentiles inside FWHM histogram bins", () => {
    const interval = conditionalFwhmInterval([1, 1], [[0.5, 0.5], [0, 1]], [0, 1, 2]);
    expect(interval.low).toEqual([0.32, 1.16]);
    expect(interval.high).toEqual([1.68, 1.8399999999999999]);
  });

  it("shows intervals only for populated observed magnitude bins", () => {
    const interval = conditionalFwhmInterval([null, 1, 1], [[0.5, 0.5], [0, 0], [1]], [0, 1, 2]);
    expect(interval.low).toEqual([null, null, null]);
    expect(interval.high).toEqual([null, null, null]);
  });
});
