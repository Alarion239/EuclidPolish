import { describe, expect, it } from "vitest";
import { SHARP_MAX_AREA, SNAP_MIN_FILL, ZOOM_PRESETS, drawPlan, snappedDrawSide, zoomPreset } from "./draw";
import { frameLayout } from "./selection";

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

describe("snappedDrawSide (pixel-exact scale in fit mode)", () => {
  it("snaps the whole image to the largest integer multiple of native pixels in device px", () => {
    // Records at 1024 × 768: a 363 css px cell at dpr 2 = 726 device px; LR 255 px, HR 510 px.
    // HR (the finest tier that is magnified) at 1× = 510 device px → LR at exactly 2×.
    expect(snappedDrawSide(363, 2, [255, 510])).toBe(255);
    // the same cell at dpr 1: HR is downsampled, LR snaps to 1× (255 css px)
    expect(snappedDrawSide(363, 1, [255, 510])).toBe(255);
    // one LR frame in a 600 css px cell at dpr 2: 1200 device px → 4 × 255 = 1020 → 510 css px
    expect(snappedDrawSide(600, 2, [255])).toBe(510);
    // an exact fit keeps the whole cell
    expect(snappedDrawSide(255, 2, [255])).toBe(255);
    // every magnification is an integer, in device pixels
    for (const [S, dpr, ext] of [[363, 2, [255, 510]], [617, 2, [255, 510]], [241, 1.5, [128, 256]]] as const) {
      const D = snappedDrawSide(S, dpr, ext) * dpr;
      for (const e of ext) if (e <= D) expect(Number.isInteger(Math.round(D * 1e6) / 1e6 / e)).toBe(true);
    }
  });
  it("fills the cell when every tier is downsampled (smoothed), or without extents", () => {
    expect(snappedDrawSide(300, 1, [853])).toBe(300);
    expect(snappedDrawSide(300, 2, [])).toBe(300);
    expect(snappedDrawSide(0, 2, [100])).toBe(0);
  });
  it("frames of different pixel scales keep ONE drawn side (blink, swipe and side-by-side line up)", () => {
    // NEXUS: LR 256 px and JWST 853 px in a 363 css px cell at dpr 2 (726 device px):
    // LR 2× = 512 device px; JWST is drawn into the same 512 (downsampled, smoothed)
    expect(snappedDrawSide(363, 2, [256, 853])).toBe(256);
  });
});

describe("snappedDrawSide: a snap that shrinks the image too much keeps the full cell", () => {
  it("snaps only when it keeps at least 80 % of the cell (unless pixel-exact is asked for)", () => {
    expect(SNAP_MIN_FILL).toBe(0.8);
    // Ensemble at 1024 × 768: a 252 css px cell at dpr 2 (504 px), HR 256 px → 1× = 256 px = 51 %: full cell
    expect(snappedDrawSide(252, 2, [128, 256], SNAP_MIN_FILL)).toBe(252);
    expect(snappedDrawSide(252, 2, [128, 256])).toBe(128);             // pixel-exact: always
    // Records at 1024: 510 of 726 device px = 70 %: full cell (sharp-bilinear)
    expect(snappedDrawSide(363, 2, [255, 510], SNAP_MIN_FILL)).toBe(363);
    // a small cutout magnified 18×: 1152 of 1200 device px = 96 % → snapped
    expect(snappedDrawSide(600, 2, [64], SNAP_MIN_FILL)).toBe(576);
    // exactly the threshold snaps
    expect(snappedDrawSide(30, 1, [4, 8], SNAP_MIN_FILL)).toBe(24);
  });
});

describe("frameLayout with a snapped draw side", () => {
  it("centres the snapped image in its cell on whole device pixels", () => {
    const LR = { tier: "lr", width: 255, height: 255, pixscale: 0.1, ready: true };
    const L = frameLayout(LR, null, 363, 255, 2);
    expect(L.dw).toBe(255);
    expect(L.dh).toBe(255);
    expect(L.dx * 2).toBe(Math.round(L.dx * 2));          // a whole device pixel
    expect(L.dx).toBe(54);
    expect(drawPlan(L, 2, 255, 255)).toEqual({ kind: "nearest" });   // exactly 2×
    // a zoomed view still fills the whole cell (user zoom may be arbitrary)
    const Z = frameLayout(LR, { u: 0.5, v: 0.5, angularSideArcsec: null, relativeSide: 0.5 }, 363, 255, 2);
    expect([Z.dx, Z.dw]).toEqual([0, 363]);
  });
});

describe("zoomPreset (integer device-pixel zoom steps)", () => {
  it("steps to the preset nearest the target, at least one step in that direction", () => {
    expect(ZOOM_PRESETS.slice(0, 6)).toEqual([1, 2, 3, 4, 6, 8]);
    const o = { fit: 1, full: 1.42, max: 64 };
    expect(zoomPreset(1, 1.5, o)).toBe(2);
    expect(zoomPreset(2, 1.5, o)).toBe(3);
    expect(zoomPreset(3, 1.5, o)).toBe(4);
    expect(zoomPreset(4, 1.5, o)).toBe(6);
    expect(zoomPreset(6, 1 / 1.5, o)).toBe(4);
    expect(zoomPreset(4, 1 / 1.5, o)).toBe(3);
    expect(zoomPreset(3, 1 / 1.5, o)).toBe(2);
    // below the smallest zoomed magnification: back to the fit
    expect(zoomPreset(2, 1 / 1.5, o)).toBe("fit");
    // an arbitrary (wheel) zoom snaps back onto the presets
    expect(zoomPreset(2.7, 1.5, o)).toBe(4);
    expect(zoomPreset(2.7, 1 / 1.5, o)).toBe(2);
  });
  it("a downsampled fit zooms in to 1× first; the maximum clamps", () => {
    expect(zoomPreset(0.85, 1.5, { fit: 0.85, full: 0.85, max: 64 })).toBe(1);
    expect(zoomPreset(64, 1.5, { fit: 1, full: 1, max: 64 })).toBe(64);
    expect(zoomPreset(48, 1.5, { fit: 1, full: 1, max: 64 })).toBe(64);
    expect(zoomPreset(1, 1 / 1.5, { fit: 1, full: 1, max: 64 })).toBe("fit");
  });
  it("above the largest preset (a small image in a big frame) the steps go on continuously", () => {
    // a ~20 px ePSF in a 700 css px frame at dpr 2: the fit is already 70×
    const o = { fit: 70, full: 70, max: 70 * 64 };
    expect(zoomPreset(70, 1.5, o)).toBeCloseTo(105);
    expect(zoomPreset(105, 1.5, o)).toBeCloseTo(157.5);
    expect(zoomPreset(157.5, 1 / 1.5, o)).toBeCloseTo(105);
    expect(zoomPreset(105, 1 / 1.5, o)).toBe("fit");
    // the maximum still clamps, and there it stops
    expect(zoomPreset(4000, 1.5, o)).toBe(4480);
    expect(zoomPreset(4480, 1.5, o)).toBe(4480);
    // a continuous zoom above 64 steps back down onto the presets
    expect(zoomPreset(100, 1 / 1.5, { fit: 1, full: 1, max: 6400 })).toBeCloseTo(66.667, 2);
    expect(zoomPreset(66.667, 1 / 1.5, { fit: 1, full: 1, max: 6400 })).toBe(48);
  });
});
