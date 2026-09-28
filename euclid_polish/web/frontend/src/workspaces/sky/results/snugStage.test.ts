/* The tile card's stage: shrink to width-limited frames, never below what
 * the fit used; back to the cap when the viewer outgrows it. */
import { describe, expect, it } from "vitest";
import { FIT_MARGIN } from "../../../viewer/fit";
import { SNUG_SLACK, snugStep } from "./snugStage";

describe("snugStep", () => {
  it("shrinks the stage to a viewer shorter than the cap, keeping the fit's margin and some slack", () => {
    expect(snugStep(342, 448)).toBe(342 + FIT_MARGIN + SNUG_SLACK);
  });
  it("is stable once snug (the next check keeps it)", () => {
    const snug = snugStep(342, 448) as number;
    expect(snugStep(342, snug)).toBe("keep");
  });
  it("keeps a viewer that fills the stage, and resets one that outgrows it", () => {
    expect(snugStep(440, 448)).toBe("keep");
    expect(snugStep(500, 448)).toBe("reset");
  });
  it("does nothing before the viewer is measured", () => {
    expect(snugStep(0, 448)).toBe("keep");
  });
});
