import { describe, expect, it } from "vitest";
import { fitTabs, sameIndices } from "./tabFit";

// The old (pre-regroup) ensemble strip at ~720 px: Overview … Train, More ≈ 70 px.
const W = [90, 82, 75, 96, 113, 98, 121, 67];
const ALL = W.map((_, i) => i);
const sum = (idx: number[]) => idx.reduce((s, i) => s + W[i], 0);

describe("fitTabs", () => {
  it("shows every tab when they all fit (no More button)", () => {
    expect(fitTabs(W, 742, 70, 0)).toEqual(ALL);
    expect(fitTabs(W, 1000, 70, 6)).toEqual(ALL);
  });

  it("an exact fit (sub-pixel rounding) still shows every tab", () => {
    expect(fitTabs(W, sum(ALL) - 0.4, 70, 0)).toEqual(ALL);
  });

  it("keeps a fixed run of leading tabs and reserves one slot (the widest hidden tab) before More", () => {
    // 90+82+75 = 247, + widest of the rest (Disagreement 121) = 368 ≤ 460;
    // one more (Knee 96) would need 343 + 121 = 464 > 460.
    const shown = fitTabs(W, 530, 70, 0);
    expect(shown).toEqual([0, 1, 2]);
    expect(sum(shown) + 121 + 70).toBeLessThanOrEqual(530);
  });

  it("the leading run is the same whichever tab is active (tabs never trade places)", () => {
    const lead = fitTabs(W, 530, 70, 0);
    for (let active = 0; active < W.length; active++) {
      const shown = fitTabs(W, 530, 70, active);
      expect(shown.slice(0, lead.length)).toEqual(lead);
      expect(shown).toContain(active);
      expect(sum(shown) + 70).toBeLessThanOrEqual(530);
    }
  });

  it("an active tab past the leading run takes the reserved slot, in strip order", () => {
    expect(fitTabs(W, 530, 70, 6)).toEqual([0, 1, 2, 6]);   // Disagreement is active
    expect(fitTabs(W, 530, 70, 3)).toEqual([0, 1, 2, 3]);
    expect(fitTabs(W, 530, 70, 7)).toEqual([0, 1, 2, 7]);
  });

  it("an active tab inside the leading run leaves the slot empty (nothing else moves in and out)", () => {
    expect(fitTabs(W, 530, 70, 1)).toEqual([0, 1, 2]);
  });

  it("never skips a tab to squeeze a later, narrower one in", () => {
    // 90+82 = 172, + widest rest 121 = 293 > 250; 90 + 121 = 211 fits: one tab leads.
    expect(fitTabs(W, 280, 30, 0)).toEqual([0]);
  });

  it("shows the active tab alone when even it barely fits", () => {
    expect(fitTabs(W, 150, 70, 6)).toEqual([6]);
    expect(fitTabs(W, 10, 70, 6)).toEqual([6]);
  });

  it("with no active tab, a strip too narrow for any tab is all More", () => {
    expect(fitTabs(W, 60, 70, -1)).toEqual([]);
  });

  it("unmeasured strips (jsdom: every width 0) show every tab", () => {
    expect(fitTabs([0, 0, 0], 0, 0, 1)).toEqual([0, 1, 2]);
  });

  it("an empty strip is empty", () => {
    expect(fitTabs([], 500, 70, -1)).toEqual([]);
  });
});

describe("sameIndices", () => {
  it("compares index lists by value", () => {
    expect(sameIndices([0, 1], [0, 1])).toBe(true);
    expect(sameIndices([0, 1], [0, 2])).toBe(false);
    expect(sameIndices([0], [0, 1])).toBe(false);
    expect(sameIndices(null, null)).toBe(true);
    expect(sameIndices(null, [0])).toBe(false);
  });
});
