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
  it("auto: three frames stay in one row unless a grid makes them ≥ 8 % larger (no empty cell for a few px)", () => {
    // Disagreement at 720 × 720: one row 216 px, 2 + 1 would be 228 px (+5.6 %)
    expect(columnsFor("auto", 3, 654, 458)).toBe(3);
    expect(fitFrames({ n: 3, width: 654, height: 458, mode: "auto" }).side).toBe(216);
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
  it("a viewer far down a page fits the whole viewport (you scroll to it)", () => {
    // 647 px full; 330 px lead leaves 317 < half of it → the full height
    expect(heightUnderTop({ viewport: 720, chrome: 65, lead: 330 })).toBe(647);
    expect(heightUnderTop({ viewport: 720, chrome: 65, lead: 320 })).toBe(327);
    // never below the minimum side
    expect(heightUnderTop({ viewport: 300, chrome: 65, lead: 100 })).toBe(227);
  });
  it("a fractional lead rounds up (the readout never ends a pixel under the fold)", () => {
    expect(heightUnderTop({ viewport: 720, chrome: 95, lead: 109.2 })).toBe(507);
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
