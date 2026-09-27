import { describe, expect, it } from "vitest";
import { fitBoxHeight, fitBoxSlack } from "./fitMath";

describe("fitBoxHeight: the viewer's box ends at the bottom of the stage", () => {
  it("is the stage height below the box's top (content coordinates)", () => {
    // Inspect at 1024×768: 720 px stage, the viewer 149 px down the page.
    expect(fitBoxHeight({ stage: 720, offset: 149 })).toBe(571);
    // the atlas card: the inspector body, 12 px padding above the viewer
    expect(fitBoxHeight({ stage: 672, offset: 12 })).toBe(660);
  });
  it("keeps a floor: a viewer far down the page gets a screenful (one scroll shows all of it)", () => {
    expect(fitBoxHeight({ stage: 720, offset: 600, min: 360 })).toBe(360);
    // but never more than the stage itself
    expect(fitBoxHeight({ stage: 300, offset: 250, min: 360 })).toBe(300);
  });
  it("subtracts a bottom gap and never goes negative", () => {
    expect(fitBoxHeight({ stage: 720, offset: 100, gap: 20 })).toBe(600);
    expect(fitBoxHeight({ stage: 0, offset: 10 })).toBe(0);
  });
  it("rounds down (a fractional box would raise a scrollbar)", () => {
    expect(fitBoxHeight({ stage: 720.6, offset: 100.2 })).toBe(620);
  });
});

describe("fitBoxSlack: space the viewer does not use is given back to the page", () => {
  it("is the negative margin that pulls the next content up", () => {
    expect(fitBoxSlack(600, 420)).toBe(180);   // width-limited frames: 180 px unused
    expect(fitBoxSlack(600, 600)).toBe(0);
    expect(fitBoxSlack(600, 640)).toBe(0);     // never negative
    expect(fitBoxSlack(600, 0)).toBe(0);       // nothing measured yet
  });
});
