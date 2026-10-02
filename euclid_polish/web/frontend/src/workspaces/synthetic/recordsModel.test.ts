import { describe, expect, it } from "vitest";
import type { FieldCensus, SourcesCensus } from "./dataApi";
import { censusHistogram, lensDensity, nearestRecord, recordArea, recordSrLink } from "./recordsModel";

const f = (field_index: number, total: number, star: number | null, lens = 0): FieldCensus => ({
  field_index, total_vis_e: total, brightest_star_mag: star, brightest_galaxy_mag: 22, n: 5, galaxy: 5, star: star == null ? 0 : 1,
  lens, other: 0, off_field: 0,
});
const FIELDS = [f(0, 1e5, null), f(1, 2e5, 18.5, 1), f(2, 1.8e7, 15.2), f(3, 3e5, 21.9)];
const CENSUS: SourcesCensus = {
  subset: "test", present: true, fields: FIELDS,
  geometry: { hr: { width: 510, height: 510, pixscale: 0.05 }, lr: { width: 255, height: 255, pixscale: 0.1 } },
};

describe("records census", () => {
  it("bins Σ VIS in log decades and the brightest star in half magnitudes", () => {
    const total = censusHistogram(FIELDS, "total")!;
    expect(total.n).toBe(4);
    expect(total.domain[0]).toBeCloseTo(1e5, 3);
    expect(total.counts.reduce((a, b) => a + b, 0)).toBe(4);
    const star = censusHistogram(FIELDS, "star")!;
    expect(star.n).toBe(3);
    expect(star.domain).toEqual([15, 22]);
    expect(censusHistogram([f(0, 1, null)], "star")).toBeNull();
  });

  it("a click opens the nearest record (log distance for Σ VIS)", () => {
    expect(nearestRecord(FIELDS, "total", 1.5e7)).toBe(2);
    expect(nearestRecord(FIELDS, "total", 1.4e5)).toBe(0);
    expect(nearestRecord(FIELDS, "star", 15.4)).toBe(2);
    expect(nearestRecord(FIELDS, "star", 30)).toBe(3);
    expect(nearestRecord(FIELDS, "total", -1)).toBeNull();
  });

  it("measures the record area and the lens density", () => {
    expect(recordArea(CENSUS)).toBeCloseTo(0.180625, 6);
    const lens = lensDensity(CENSUS)!;
    expect(lens.count).toBe(1);
    expect(lens.area).toBeCloseTo(4 * 0.180625, 6);
    expect(lens.density).toBeCloseTo(1 / (4 * 0.180625), 6);
  });
});

describe("recordSrLink (the record's SR is one click away, in Models › Images)", () => {
  it("opens this record on the records set when its split has an SR", () => {
    expect(recordSrLink("test", 12, 100)).toEqual({
      to: "/models/images?set=records&split=test&id=test%3A12", label: "Open its SR in Models › Images",
    });
  });
  it("points at Generate SR when the split has none, and at the set without a record", () => {
    expect(recordSrLink("validate", 3, 0)).toEqual({ to: "/models/images?set=records", label: "Generate its SR in Models › Images" });
    expect(recordSrLink("test", null, 5).to).toBe("/models/images?set=records&split=test");
  });
});
