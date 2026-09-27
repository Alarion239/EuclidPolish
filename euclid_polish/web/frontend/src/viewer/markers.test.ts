import { describe, expect, it } from "vitest";
import { markerShapes, markersOnTier, type ViewerMarkers } from "./markers";
import type { FrameLayout } from "./selection";

// A 510² HR tier drawn 1:1 into a 510-px frame.
const fit = (w: number): FrameLayout => ({ sx: 0, sy: 0, sw: w, sh: w, dx: 0, dy: 0, dw: 510, dh: 510 });

const base: ViewerMarkers = {
  grid: { width: 510, height: 510 },
  items: [
    { key: "0", x: 100, y: 200, r: 4, kind: "galaxy" },
    { key: "1", x: 509, y: 0, r: 3, kind: "star" },
  ],
};

describe("viewer markers", () => {
  it("draw on every tier by default, or only on the listed tiers", () => {
    expect(markersOnTier("hr", base)).toBe(true);
    expect(markersOnTier("dirty", base)).toBe(true);
    expect(markersOnTier("dirty", { ...base, tiers: ["hr"] })).toBe(false);
    expect(markersOnTier("hr", { ...base, tiers: ["hr"] })).toBe(true);
    expect(markersOnTier("hr", null)).toBe(false);
  });

  it("put a marker on its pixel centre (grid pixel centres are integers)", () => {
    const [m] = markerShapes(fit(510), 510, base, { width: 510, height: 510 });
    expect(m.key).toBe("0");
    expect(m.cx).toBeCloseTo(100.5);
    expect(m.cy).toBeCloseTo(200.5);
    expect(m.r).toBeCloseTo(4);
  });

  it("scale positions and radii onto a coarser tier (LR at half the HR grid)", () => {
    const L: FrameLayout = { sx: 0, sy: 0, sw: 255, sh: 255, dx: 0, dy: 0, dw: 510, dh: 510 };
    const [m] = markerShapes(L, 510, base, { width: 255, height: 255 });
    // HR (100, 200) → LR image (50.25, 100.25) → frame ×2
    expect(m.cx).toBeCloseTo(100.5);
    expect(m.cy).toBeCloseTo(200.5);
    expect(m.r).toBeCloseTo(4);
  });

  it("follow the pan/zoom view and drop markers outside the frame", () => {
    // zoomed 2× onto the top-left quarter
    const L: FrameLayout = { sx: 0, sy: 0, sw: 255, sh: 255, dx: 0, dy: 0, dw: 510, dh: 510 };
    const shapes = markerShapes(L, 510, base, { width: 510, height: 510 });
    expect(shapes.map((s) => s.key)).toEqual(["0"]);            // the corner star is out of view
    expect(shapes[0].cx).toBeCloseTo(201);
    expect(shapes[0].r).toBeCloseTo(8);
  });

  it("flag the active marker and keep the dim flag", () => {
    const shapes = markerShapes(fit(510), 510, {
      ...base, activeKey: "1", items: [...base.items, { key: "2", x: 10, y: 10, r: 2, dim: true }],
    }, { width: 510, height: 510 });
    expect(shapes.find((s) => s.key === "1")?.active).toBe(true);
    expect(shapes.find((s) => s.key === "0")?.active).toBe(false);
    expect(shapes.find((s) => s.key === "2")?.dim).toBe(true);
  });
});
