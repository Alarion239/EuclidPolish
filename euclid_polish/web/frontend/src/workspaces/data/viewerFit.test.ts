import { describe, expect, it } from "vitest";
import { fitFrames } from "../../viewer/fit";
import { besideWidth, frameCount, frameFit, gridWidth, maxFirstRowSide } from "./viewerFit";

/* The viewer fits its frames to the WHOLE stage height (minus its own bar and
   readout); a page that puts rows above it would push a height-limited frame
   row below the fold. frameFit is the grid the viewer would lay out over the
   height left under its top — the viewer stays full width (its bar keeps its
   rows), only the frames shrink. */

describe("maxFirstRowSide", () => {
  it("is the stage height below the table's top, less the chrome and the margin", () => {
    // 672 px stage, table at 97 px, bar 37 + readout 28, 8 px margin
    expect(maxFirstRowSide({ stageHeight: 672, tableTop: 97, chrome: 65 })).toBe(502);
    expect(maxFirstRowSide({ stageHeight: 672, tableTop: 97, chrome: 65, margin: 0 })).toBe(510);
    expect(maxFirstRowSide({ stageHeight: 100, tableTop: 97, chrome: 65 })).toBe(0);
  });
});

describe("frameFit", () => {
  // Records at 792 × 720: a 728 px frame grid (two 363 px frames), the table
  // 120 px under the stage top, a two-row bar + readout of 95 px.
  const records792 = { mode: "auto" as const, width: 728, stageHeight: 672, tableTop: 120, chrome: 95 };

  it("leaves a width-limited viewer alone", () => {
    expect(fitFrames({ n: 2, width: 728, height: 672 - 95 - 8, mode: "auto" }).side).toBe(363);
    expect(frameFit({ ...records792, n: 2 })).toBeNull();
  });

  it("fits a single frame (blink / swipe) to the height under the viewer, at the same chrome", () => {
    // the viewer alone would make it 569 px: 120 px of it below the fold
    expect(frameFit({ ...records792, n: 1 })).toEqual({ columns: 1, side: 449 });
  });

  it("never makes a blink frame smaller than the two-up frames it replaces", () => {
    for (const tableTop of [60, 120, 170, 220]) {
      const o = { ...records792, tableTop };
      const two = frameFit({ ...o, n: 2 })?.side ?? fitFrames({ n: 2, width: 728, height: 672 - 95 - 8, mode: "auto" }).side;
      const one = frameFit({ ...o, n: 1 })?.side ?? 728;
      expect(one).toBeGreaterThanOrEqual(two);
    }
  });

  it("does not fit when too little height is left (a viewer far down a page scrolls into view)", () => {
    expect(frameFit({ ...records792, n: 1, tableTop: 420 })).toBeNull();
    expect(frameFit({ ...records792, n: 1, tableTop: 380, minSide: 100 })).toEqual({ columns: 1, side: 189 });
  });

  it("re-chooses the auto columns for the smaller height", () => {
    // four frames, 1600 px wide, a tall stage but 380 px left under the viewer:
    // the viewer's own fit would pick two rows of 499 px frames; under the
    // top, one row of four 380 px frames is the largest arrangement
    const o = { n: 4, mode: "auto" as const, width: 1600, stageHeight: 1073, tableTop: 620, chrome: 65 };
    // (2 × 2 and 3 + 1 give the same 499 px: 2 × 2 has no empty cell)
    expect(fitFrames({ n: 4, width: 1600, height: 1073 - 65 - 8, mode: "auto" }).columns).toBe(2);
    expect(frameFit({ ...o, minSide: 200 })).toEqual({ columns: 4, side: 380 });
  });

  it("keeps the fixed layouts' column count", () => {
    const o = { n: 3, width: 1500, stageHeight: 800, tableTop: 327, chrome: 65 };   // 400 px left
    expect(frameFit({ ...o, mode: "one-row" })).toEqual({ columns: 3, side: 400 });
    // a stack of three: 132 px each fits, relaxed to the viewer's 160 px floor
    expect(frameFit({ ...o, mode: "stack" })).toEqual({ columns: 1, side: 160 });
    expect(frameFit({ ...o, n: 1, mode: "stack" })).toEqual({ columns: 1, side: 400 });
  });

  it("returns nothing for no width", () => {
    expect(frameFit({ ...records792, n: 1, width: 0 })).toBeNull();
  });
});

describe("besideWidth", () => {
  const fit = { columns: 1, side: 449 };

  it("is the frames' width (+ the dock beside them) when the panel fits in the rest", () => {
    expect(gridWidth({ columns: 2, side: 363 })).toBe(728);
    expect(besideWidth({ rowWidth: 717, fit, asideMin: 150, gap: 12 })).toBe(449);
    expect(besideWidth({ rowWidth: 1000, fit, extraWidth: 272, asideMin: 150, gap: 12 })).toBe(721);
  });

  it("never narrows the viewer below its bar's two-row width (no extra bar row)", () => {
    expect(besideWidth({ rowWidth: 717, fit: { columns: 1, side: 300 }, barMin: 452, asideMin: 150, gap: 12 })).toBe(452);
    // the bar's width leaves the panel too little: the panel goes below
    expect(besideWidth({ rowWidth: 717, fit: { columns: 1, side: 300 }, barMin: 560, asideMin: 150, gap: 12 })).toBeNull();
  });

  it("keeps the panel below without a fit (a full-width viewer) or room", () => {
    expect(besideWidth({ rowWidth: 717, fit: null, asideMin: 150, gap: 12 })).toBeNull();
    expect(besideWidth({ rowWidth: 717, fit, asideMin: 300, gap: 12 })).toBeNull();
  });
});

describe("frameCount", () => {
  it("counts the frames the viewer shows, one while comparing, the hint while loading", () => {
    expect(frameCount({ frames: 3, stacked: false, loading: false }, 2)).toBe(3);
    expect(frameCount({ frames: 2, stacked: true, loading: false }, 2)).toBe(1);
    expect(frameCount({ frames: 1, stacked: false, loading: true }, 2)).toBe(2);
    expect(frameCount({ frames: 0, stacked: false, loading: false }, 0)).toBe(1);
  });
});
