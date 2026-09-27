import { describe, expect, it } from "vitest";
import { availableHeight, columnsFor, figureLayout, fitFrames, heightUnderTop, parseLayout, sideFor } from "./fit";

describe("sideFor", () => {
  it("is the smaller of the width per column and the height per row (2 px gaps)", () => {
    expect(sideFor(2, 2, 780, 640)).toBe(389);            // (780 − 2) / 2
    expect(sideFor(2, 1, 780, 640)).toBe(319);            // (640 − 2) / 2
    expect(sideFor(1, 1, 780, 500)).toBe(500);
  });
  it("never goes negative and treats n < 1 as one frame", () => {
    expect(sideFor(0, 1, 100, 100)).toBe(100);
    expect(sideFor(3, 3, 2, 2)).toBe(0);
  });
});

describe("columnsFor", () => {
  it("auto: side by side when the viewport is wide", () => {
    expect(columnsFor("auto", 2, 780, 640)).toBe(2);
  });
  it("auto: a 2 × 2 grid for four frames when one row would be short", () => {
    // one row: 248 px; 2 × 2: min(499, 349) = 349; 3 columns: 332
    expect(columnsFor("auto", 4, 1000, 700)).toBe(2);
  });
  it("auto: a stack in a tall narrow box (the inspector)", () => {
    expect(columnsFor("auto", 2, 380, 900)).toBe(1);
  });
  it("auto: three frames on a 792 × 644 stage go 2 + 1 (321 px) rather than one row (262 px)", () => {
    expect(columnsFor("auto", 3, 792, 644)).toBe(2);
  });
  it("auto: one row wins when its side is within 2 % of the largest", () => {
    // two frames in 600 × 610: one row 299 px, a stack 304 px (299 ≥ 0.98 · 304)
    expect(columnsFor("auto", 2, 600, 610)).toBe(2);
    // …and a grid clearly larger with no empty cell wins: 4 frames in 1000 × 700 (see above)
  });
  it("auto: an empty cell costs 6 % — a 2 + 1 grid must beat one row by more than that", () => {
    // Disagreement at 720 × 720: 654 × 458, one row 216 px, 2 + 1 228 px (+5.6 %) → one row
    expect(columnsFor("auto", 3, 654, 458)).toBe(3);
    expect(fitFrames({ n: 3, width: 654, height: 458, mode: "auto" }).side).toBe(216);
    // Experiments at 1024 × 768 (review): 728 × 500, one row 241 px, 2 + 1 249 px (+3 %) → one row
    expect(sideFor(3, 3, 728, 500)).toBe(241);
    expect(sideFor(3, 2, 728, 500)).toBe(249);
    expect(columnsFor("auto", 3, 728, 500)).toBe(3);
    // a much larger 2 + 1 still wins: 728 × 560, one row 241, 2 + 1 279 (+16 %)
    expect(columnsFor("auto", 3, 728, 560)).toBe(2);
    // a grid with no empty cell pays nothing (4 frames, 2 × 2)
    expect(columnsFor("auto", 4, 1000, 700)).toBe(2);
  });
  it("auto: a layout whose rows do not fit the height (the minimum side) loses to one that fits", () => {
    // 3 frames in 700 × 300: one row 232 px fits; 2 + 1 would be raised to the
    // 160 px minimum and its second row would end at 322 > 300 (below the fold)
    expect(columnsFor("auto", 3, 700, 300)).toBe(3);
    // 2 frames in 400 × 300: a stack (149 → 160 px, 322 px tall) does not fit, one row (199 px) does
    expect(columnsFor("auto", 2, 400, 300)).toBe(2);
  });
  it("auto: near-ties (within 2 %) go to the fewest empty cells, then to more columns", () => {
    // 3 frames in 600 × 900: 2 + 1 is 299 px (one empty cell), a stack 298 px (none)
    expect(columnsFor("auto", 3, 600, 900)).toBe(1);
  });
  it("auto: fewer columns only when the frames get clearly larger", () => {
    // 3 frames in 900 × 450: one row 298 px; 2 + 1 min(449, 224) = 224 → one row
    expect(columnsFor("auto", 3, 900, 450)).toBe(3);
  });
  it("one-row, grid and stack are fixed counts", () => {
    expect(columnsFor("one-row", 5, 100, 100)).toBe(5);
    expect(columnsFor("stack", 5, 5000, 100)).toBe(1);
    expect([1, 2, 3, 4, 5, 9, 10].map((n) => columnsFor("grid", n, 1000, 1000))).toEqual([1, 2, 2, 2, 3, 3, 4]);
  });
});

describe("fitFrames", () => {
  it("fits the frames to the height left in the viewport", () => {
    expect(fitFrames({ n: 2, width: 744, height: 644, mode: "auto" })).toEqual({ columns: 2, rows: 1, side: 371 });
    expect(fitFrames({ n: 1, width: 744, height: 600, mode: "auto" })).toEqual({ columns: 1, rows: 1, side: 600 });
  });
  it("relaxes the height down to the minimum side, but never the width", () => {
    expect(fitFrames({ n: 1, width: 800, height: 100, mode: "auto" }).side).toBe(160);
    expect(fitFrames({ n: 2, width: 300, height: 1000, mode: "one-row" }).side).toBe(149);
  });
  it("stacked compare modes are one frame", () => {
    expect(fitFrames({ n: 1, width: 1200, height: 700, mode: "one-row" })).toEqual({ columns: 1, rows: 1, side: 700 });
  });
});

describe("availableHeight", () => {
  it("is the viewport minus the bar, the readout and the margin", () => {
    expect(availableHeight(720, 40 + 28)).toBe(644);
    expect(availableHeight(50, 100)).toBe(0);
  });
});

describe("heightUnderTop", () => {
  it("subtracts how far below its scroll box's top the viewer starts", () => {
    // Disagreement at 1024 × 768: stage 720, the table 110 px down (banner,
    // tabs), a two-row bar 67 + readout 28: 720 − 110 − 95 − 8
    expect(heightUnderTop({ viewport: 720, chrome: 95, lead: 110 })).toBe(507);
    // one frame: side 507, so the frame AND the readout end inside the stage
    const side = fitFrames({ n: 1, width: 790, height: heightUnderTop({ viewport: 720, chrome: 95, lead: 110 }), mode: "auto" }).side;
    expect(110 + 67 + side + 28).toBeLessThanOrEqual(720);
  });
  it("is the whole-viewport height at the top of the box (no lead)", () => {
    expect(heightUnderTop({ viewport: 720, chrome: 65 })).toBe(availableHeight(720, 65));
    expect(heightUnderTop({ viewport: 720, chrome: 65, lead: 0 })).toBe(647);
  });
  it("keeps fitting under the top however far down the viewer starts, down to the minimum side", () => {
    // 647 px full; a 330 px lead leaves 317 px (no fallback to the whole viewport)
    expect(heightUnderTop({ viewport: 720, chrome: 65, lead: 330 })).toBe(317);
    expect(heightUnderTop({ viewport: 720, chrome: 65, lead: 480 })).toBe(167);
    // clamped by the minimum side
    expect(heightUnderTop({ viewport: 720, chrome: 65, lead: 490 })).toBe(160);
  });
  it("a viewer that starts below the first screen fits the whole viewport (it is scrolled to)", () => {
    // not even a minimum frame is in sight under its top
    expect(heightUnderTop({ viewport: 720, chrome: 65, lead: 720 })).toBe(647);
    expect(heightUnderTop({ viewport: 720, chrome: 65, lead: 1500 })).toBe(647);
    expect(heightUnderTop({ viewport: 300, chrome: 65, lead: 100 })).toBe(227);
  });
  it("a fractional lead rounds up (the readout never ends a pixel under the fold)", () => {
    expect(heightUnderTop({ viewport: 720, chrome: 95, lead: 109.2 })).toBe(507);
  });
  it("the Disagreement case of the review: 3 frames under a 110 px lead end inside the 720 px stage", () => {
    const height = heightUnderTop({ viewport: 720, chrome: 67 + 28, lead: 110 });
    const f = fitFrames({ n: 3, width: 728, height, mode: "auto" });
    const bottom = 48 + 110 + 67 + f.rows * f.side + (f.rows - 1) * 2 + 28;
    expect(bottom).toBeLessThanOrEqual(768);
    // Cutouts at 1280 × 800: one frame under a 131 px lead, bar 37 + readout 28 in a 752 px stage
    const one = fitFrames({ n: 1, width: 1000, height: heightUnderTop({ viewport: 752, chrome: 65, lead: 131 }), mode: "auto" });
    expect(48 + 131 + 37 + one.side + 28).toBeLessThanOrEqual(800);
  });
});

describe("parseLayout", () => {
  it("reads the saved mode and migrates the old key", () => {
    expect(parseLayout("grid")).toBe("grid");
    expect(parseLayout("stack", "two-rows")).toBe("stack");
    expect(parseLayout(null, "two-rows")).toBe("grid");
    expect(parseLayout(null, "one-row")).toBe("auto");
    expect(parseLayout(null, null)).toBe("auto");
    expect(parseLayout("two-rows")).toBe("auto");
  });
  it("maps the screen fit onto the figure's two layouts", () => {
    expect(figureLayout({ rows: 2 }, 4)).toBe("two-rows");
    expect(figureLayout({ rows: 1 }, 4)).toBe("one-row");
    expect(figureLayout({ rows: 2 }, 2)).toBe("one-row");
    expect(figureLayout(null, 3)).toBe("one-row");
  });
});
