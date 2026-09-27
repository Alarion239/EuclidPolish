import { describe, expect, it } from "vitest";
import { SHARP_MAX_AREA, drawPlan } from "./draw";

const whole = (w: number, S: number) => ({ sx: 0, sy: 0, sw: w, sh: w, dx: 0, dy: 0, dw: S, dh: S });

describe("drawPlan", () => {
  it("draws an integer magnification nearest-neighbour (equal k × k pixels)", () => {
    expect(drawPlan(whole(255, 510), 1, 255, 255)).toEqual({ kind: "nearest" });
    expect(drawPlan(whole(255, 255), 2, 255, 255)).toEqual({ kind: "nearest" });
    // less than half a device pixel off over the whole frame still counts
    expect(drawPlan(whole(100, 200.2), 1, 100, 100).kind).toBe("nearest");
  });

  it("goes sharp-bilinear at a non-integer magnification: ⌈scale⌉ first, then down", () => {
    // Records at 1024: LR 255 px into a 726 device-px canvas (2.85×)
    const p = drawPlan(whole(255, 363), 2, 255, 255);
    expect(p.kind).toBe("sharp");
    if (p.kind !== "sharp") return;
    expect(p.k).toBe(3);
    expect([p.ix, p.iy, p.iw, p.ih]).toEqual([0, 0, 255, 255]);
    expect([p.tx, p.ty, p.tw, p.th]).toEqual([0, 0, 765, 765]);
    expect([p.dx, p.dy, p.dw, p.dh]).toEqual([0, 0, 726, 726]);
    // HR 510 px at 1.42×
    const h = drawPlan(whole(510, 363), 2, 510, 510);
    expect(h.kind === "sharp" && h.k).toBe(2);
  });

  it("smooths a downsampled frame (nearest would drop rows)", () => {
    expect(drawPlan(whole(328, 142), 2, 328, 328)).toEqual({ kind: "smooth" });
  });

  it("maps a zoomed crop with fractional edges onto whole scratch pixels", () => {
    const L = { sx: 10.5, sy: 20.25, sw: 30, sh: 30, dx: 0, dy: 0, dw: 400, dh: 400 };
    const p = drawPlan(L, 1, 255, 255);
    expect(p.kind).toBe("sharp");
    if (p.kind !== "sharp") return;
    expect(p.k).toBe(14);                                  // 13.33× → 14
    expect([p.ix, p.iy, p.iw, p.ih]).toEqual([10, 20, 31, 31]);
    expect(p.tx).toBeCloseTo(7);                           // 0.5 px into the block × 14
    expect(p.ty).toBeCloseTo(3.5);
    expect(p.tw).toBeCloseTo(420);
    expect([p.dx, p.dy, p.dw, p.dh].map((v) => Math.round(v))).toEqual([0, 0, 400, 400]);
  });

  it("clips a view that overhangs the image edge", () => {
    const L = { sx: -5, sy: 0, sw: 20, sh: 20, dx: 0, dy: 0, dw: 250, dh: 250 };   // 12.5×
    const p = drawPlan(L, 1, 100, 100);
    expect(p.kind).toBe("sharp");
    if (p.kind !== "sharp") return;
    expect(p.ix).toBe(0);
    expect(p.iw).toBe(15);
    expect(p.dx).toBeCloseTo(62.5);                        // the overhang stays empty
    expect(p.dw).toBeCloseTo(187.5);
  });

  it("falls back to plain bilinear when the scratch canvas would be huge", () => {
    const big = 2100;
    const p = drawPlan(whole(big, big * 1.01), 1, big, big);
    expect(big * 2 * big * 2).toBeGreaterThan(SHARP_MAX_AREA);
    expect(p).toEqual({ kind: "smooth" });
  });

  it("degenerate layouts draw nothing special", () => {
    expect(drawPlan({ sx: 0, sy: 0, sw: 0, sh: 0, dx: 0, dy: 0, dw: 10, dh: 10 }, 1, 10, 10)).toEqual({ kind: "nearest" });
  });
});
