import { describe, expect, it } from "vitest";
import {
  LENS_DEFAULT_ZOOM, LENS_MAX_ZOOM, LENS_ZOOM_STEP, chooseLensPosition, clampSelectionToFrames, cornerCandidates,
  frameLayout, frameToImage, imageToFrame, lensSide, normalizeSelectionScale, panSelection, placeLensPopup,
  receptiveFieldLabels, resolveCrop, resolveLensOverlaps, selectionAt, serializeSelection, zoomSelection,
  type FrameGeom, type Selection, type Viewport,
} from "./selection";

const LR: FrameGeom = { tier: "lr", width: 256, height: 256, pixscale: 0.1, ready: true };
const SR: FrameGeom = { tier: "sr", width: 512, height: 512, pixscale: 0.05, ready: true };
const JW: FrameGeom = { tier: "jwst", width: 850, height: 850, pixscale: 0.03, ready: true };
const NOPIX: FrameGeom = { tier: "psf", width: 64, height: 64, pixscale: null, ready: true };
const VP: Viewport = { width: 1400, height: 900, scrollX: 0, scrollY: 0 };

describe("crops are matched in angular size across pixel scales", () => {
  const sel: Selection = { u: 0.5, v: 0.5, angularSideArcsec: 3.2, relativeSide: null };
  it("the same arcsec side on every tier", () => {
    const a = resolveCrop(LR, sel)!, b = resolveCrop(SR, sel)!, c = resolveCrop(JW, sel)!;
    expect(a.side).toBeCloseTo(32);
    expect(b.side).toBeCloseTo(64);
    expect(c.side).toBeCloseTo(3.2 / 0.03);
    for (const crop of [a, b, c]) expect(crop.angularSideArcsec).toBeCloseTo(3.2);
    expect(a.cx).toBe(128);
    expect(b.cx).toBe(256);
  });
  it("a frame without a pixel scale uses the relative side", () => {
    const crop = resolveCrop(NOPIX, { u: 0.5, v: 0.5, angularSideArcsec: null, relativeSide: 0.25 })!;
    expect(crop.side).toBe(16);
    expect(crop.angularSideArcsec).toBeNull();
    const fallback = resolveCrop(NOPIX, { u: 0.5, v: 0.5, angularSideArcsec: null, relativeSide: null })!;
    expect(fallback.side).toBeCloseTo(64 / LENS_DEFAULT_ZOOM);
  });
  it("keeps the crop inside the frame near an edge", () => {
    const crop = resolveCrop(LR, { u: 0, v: 1, angularSideArcsec: 3.2, relativeSide: null })!;
    expect(crop.x).toBe(0);
    expect(crop.y + crop.side).toBe(256);
  });
});

describe("normalise and clamp against every ready frame", () => {
  it("bounds the angular side by the smallest frame and the coarsest pixel", () => {
    const tiny: FrameGeom = { tier: "t", width: 20, height: 20, pixscale: 0.1, ready: true };
    const n = normalizeSelectionScale({ u: 0.5, v: 0.5, angularSideArcsec: 99, relativeSide: 0.9 }, [LR, tiny]);
    expect(n.angularSideArcsec).toBeCloseTo(2);            // 20 px × 0.1″
    // scaled with the angular side (0.9·2/99 ≈ 0.018), floored at one pixel of the 20-px frame
    expect(n.relativeSide).toBeCloseTo(1 / 20);
    const small = normalizeSelectionScale({ u: 0.5, v: 0.5, angularSideArcsec: 0.001, relativeSide: null }, [LR, SR]);
    expect(small.angularSideArcsec).toBeCloseTo(0.1);      // never below one coarse pixel
  });
  it("clamps the shared centre to the intersection valid in every frame", () => {
    const c = clampSelectionToFrames({ u: 0, v: 0.999, angularSideArcsec: 5.1, relativeSide: null }, [LR, SR]);
    // LR: ceil(51) = 51 px → half-width 25.5/256.
    expect(c.u).toBeCloseTo(25.5 / 256);
    expect(c.v).toBeCloseTo(1 - 25.5 / 256);
  });
  it("ignores frames that are not ready", () => {
    const loading = { ...JW, ready: false, width: 10, height: 10 };
    const n = normalizeSelectionScale({ u: 0.5, v: 0.5, angularSideArcsec: 12, relativeSide: null }, [LR, loading]);
    expect(n.angularSideArcsec).toBe(12);
  });
});

describe("the hover selection and the lens zoom", () => {
  it("a first hover starts at the default zoom", () => {
    const s = selectionAt(LR, 0.25, 0.75, null, [LR, SR], lensSide(VP))!;
    const crop = resolveCrop(LR, s)!;
    expect(lensSide(VP) / crop.side).toBeCloseTo(LENS_DEFAULT_ZOOM, 5);
    expect(s.sourceTier).toBe("lr");
    expect(s.u).toBeCloseTo(0.25);
  });
  it("wheel steps zoom by 1.18× and stop at 16×", () => {
    let s = selectionAt(LR, 0.5, 0.5, null, [LR], lensSide(VP))!;
    const z0 = lensSide(VP) / resolveCrop(LR, s)!.side;
    s = zoomSelection(LR, s, LENS_ZOOM_STEP, [LR], lensSide(VP));
    expect(lensSide(VP) / resolveCrop(LR, s)!.side).toBeCloseTo(z0 * LENS_ZOOM_STEP, 5);
    for (let i = 0; i < 60; i++) s = zoomSelection(LR, s, LENS_ZOOM_STEP, [LR], lensSide(VP));
    expect(lensSide(VP) / resolveCrop(LR, s)!.side).toBeLessThanOrEqual(LENS_MAX_ZOOM + 1e-9);
    for (let i = 0; i < 60; i++) s = zoomSelection(LR, s, 1 / LENS_ZOOM_STEP, [LR], lensSide(VP));
    expect(resolveCrop(LR, s)!.side).toBeLessThanOrEqual(256);
  });
  it("lens side follows the window (160–280 px)", () => {
    expect(lensSide(VP)).toBe(280);
    expect(lensSide({ ...VP, width: 150 })).toBe(160);
  });
});

describe("receptive-field tags", () => {
  const fields = [
    { angular_side_arcsec: 2.1, blocks: 8, label: "8b" },
    { angular_side_arcsec: 3.7, blocks: 16, label: "16b" },
    { angular_side_arcsec: 6.9, blocks: 32 },
  ];
  it("match within half a wheel step (log-symmetric)", () => {
    expect(receptiveFieldLabels(2.1, fields)).toEqual(["8b"]);
    expect(receptiveFieldLabels(2.1 * Math.sqrt(LENS_ZOOM_STEP) * 0.999, fields)).toEqual(["8b"]);
    expect(receptiveFieldLabels(2.1 * LENS_ZOOM_STEP, fields)).toEqual([]);
    expect(receptiveFieldLabels(6.9, fields)).toEqual(["32b"]);   // label from blocks
    expect(receptiveFieldLabels(null, fields)).toEqual([]);
  });
});

describe("the wire shape of a frozen crop", () => {
  it("serialises angular crops and marks relative-only ones", () => {
    expect(serializeSelection({ u: 0.4, v: 0.6, angularSideArcsec: 3, relativeSide: 0.1, revision: 7 })).toEqual({
      u: 0.4, v: 0.6, angular_side_arcsec: 3, relative_side: 0.1, revision: 7,
    });
    expect(serializeSelection({ u: 0.4, v: 0.6, angularSideArcsec: null, relativeSide: 0.1, revision: 2 })).toEqual({
      u: 0.4, v: 0.6, angular_side_arcsec: null, relative_side: 0.1, revision: 2, relative_fallback_safe: true,
    });
    expect(serializeSelection(null)).toBeNull();
  });
});

describe("frame layout (pan/zoom view)", () => {
  it("contains a non-square image without distortion at the full view", () => {
    const wide: FrameGeom = { tier: "w", width: 200, height: 100, pixscale: 0.1, ready: true };
    const L = frameLayout(wide, null, 400);
    expect(L).toMatchObject({ sx: 0, sy: 0, sw: 200, sh: 100, dx: 0, dy: 100, dw: 400, dh: 200 });
    expect(frameToImage(L, 200, 200)).toEqual({ x: 100, y: 50 });
    expect(imageToFrame(L, 100, 50)).toEqual({ x: 200, y: 200 });
    expect(frameToImage(L, 200, 20)).toBeNull();            // letterbox
  });
  it("a zoomed view fills the frame with the matched square crop", () => {
    const view: Selection = { u: 0.5, v: 0.5, angularSideArcsec: 6.4, relativeSide: null };
    const L = frameLayout(LR, view, 320);
    expect(L).toMatchObject({ sx: 96, sy: 96, sw: 64, sh: 64, dx: 0, dy: 0, dw: 320, dh: 320 });
    const S = frameLayout(SR, view, 320);
    expect(S.sw).toBeCloseTo(128);
  });
  it("panning moves the centre and stays inside every frame", () => {
    const view: Selection = { u: 0.5, v: 0.5, angularSideArcsec: 6.4, relativeSide: null };
    const p = panSelection(view, 0.1, -0.2, [LR, SR]);
    expect(p.u).toBeCloseTo(0.6);
    expect(p.v).toBeCloseTo(0.3);
    const far = panSelection(view, 5, 5, [LR, SR]);
    expect(far.u).toBeCloseTo(1 - 32 / 256);
  });
});

describe("lens popup layout", () => {
  it("places a hover popup beside the cursor and flips at the window edge", () => {
    expect(placeLensPopup(280, 100, 100, VP, false)).toEqual({ left: 118, top: 118 });
    expect(placeLensPopup(280, 1300, 800, VP, false)).toEqual({ left: 1300 - 280 - 18, top: 800 - 280 - 18 });
    // A frozen popup is positioned in document coordinates (follows the page).
    expect(placeLensPopup(280, 100, 100, { ...VP, scrollY: 500 }, true)).toEqual({ left: 118, top: 618 });
  });
  it("offers four corners around the source crop", () => {
    const src = { left: 400, top: 300, right: 500, bottom: 400 };
    const c = cornerCandidates(src, 280, 280);
    expect(c.map((x) => x.corner)).toEqual(["top-left", "top-right", "bottom-left", "bottom-right"]);
    expect(c[3]).toMatchObject({ left: 512, top: 412, right: 792, bottom: 692 });
  });
  it("never overlaps two popups and keeps an existing corner", () => {
    const pos = (left: number) => ({ current: { left: 520, top: 420, right: 800, bottom: 700, width: 280, height: 280 },
      sourceRect: { left, top: 300, width: 100, height: 100 }, corner: null as string | null });
    const placed = resolveLensOverlaps([pos(400), pos(420)], VP);
    const [a, b] = placed;
    const overlap = a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;
    expect(overlap).toBe(false);
    const kept = chooseLensPosition(pos(400).current, [], { ...pos(400), corner: "top-left" }, VP);
    expect(kept?.corner).toBe("top-left");
  });
});
