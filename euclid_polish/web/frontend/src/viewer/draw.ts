/* How a frame's rendered cube is drawn into its visible canvas (pure plan +
 * the canvas calls; Frame.tsx), and the pixel-exact scale of the fit.
 *
 * Nearest-neighbour at a NON-integer magnification gives native pixels
 * uneven widths (2 and 3 device px side by side at 2.85×), which jitters the
 * noise texture and star cores. So:
 *
 *   fit (whole image)     the drawn image is SNAPPED to the largest integer
 *                         multiple of native pixels, in device pixels, that
 *                         fits its cell (`snappedDrawSide`, one side for
 *                         every frame of the grid so blink / swipe / side by
 *                         side line up) and centred on whole device pixels —
 *                         when that keeps at least SNAP_MIN_FILL (80 %) of
 *                         the cell; a snap that would shrink the image more
 *                         (a 1.97× fit snaps to 1×: half the side) keeps the
 *                         full cell, drawn sharp-bilinear below. "Pixel-exact
 *                         fit" (the layout menu) always snaps.
 *   scale ≈ integer ≥ 1   nearest neighbour (every pixel k × k device px)
 *   scale > 1 otherwise   (a user zoom, a tier whose grid is not an integer
 *                         multiple of the finest) "sharp bilinear": nearest
 *                         neighbour to the next integer multiple k = ⌈scale⌉
 *                         in a scratch canvas, then bilinear down to the
 *                         frame (square pixels of equal size, ≤ 1 device px
 *                         of blend at their edges)
 *   scale < 1             bilinear, high quality (downsampling: nearest
 *                         would drop rows)
 *
 * The zoom steps (+ / −, `zoomBy`) prefer integer device-pixel
 * magnifications of the finest tier (`zoomPreset`); the wheel and a pinch
 * stay continuous. */
import type { FrameLayout } from "./selection";

/** Scratch canvases larger than this (device px²) fall back to plain bilinear. */
export const SHARP_MAX_AREA = 16_000_000;

export type DrawPlan =
  | { kind: "nearest" }
  | { kind: "smooth" }
  | {
      kind: "sharp"; k: number;
      /** The integer block of source pixels copied (k×) into the scratch canvas. */
      ix: number; iy: number; iw: number; ih: number;
      /** The visible part of the source, in scratch-canvas pixels. */
      tx: number; ty: number; tw: number; th: number;
      /** Where it lands in the visible canvas (device px). */
      dx: number; dy: number; dw: number; dh: number;
    };

/** The drawing plan for layout `L` (frame CSS px) at `dpr` over a source of
 *  `srcW` × `srcH` pixels. */
export function drawPlan(L: FrameLayout, dpr: number, srcW: number, srcH: number): DrawPlan {
  if (!(L.sw > 0 && L.sh > 0 && L.dw > 0 && L.dh > 0 && srcW > 0 && srcH > 0)) return { kind: "nearest" };
  const scale = (L.dw * dpr) / L.sw;
  if (scale < 1 - 1e-6) return { kind: "smooth" };
  const nearest = Math.round(scale);
  if (Math.abs(scale - nearest) * Math.max(L.sw, L.sh) < 0.5) return { kind: "nearest" };   // < ½ device px off over the frame
  const k = Math.ceil(scale);
  // The source rectangle clipped to the image (a view may overhang an edge).
  const cx0 = Math.max(0, L.sx), cy0 = Math.max(0, L.sy);
  const cx1 = Math.min(srcW, L.sx + L.sw), cy1 = Math.min(srcH, L.sy + L.sh);
  if (!(cx1 > cx0 && cy1 > cy0)) return { kind: "nearest" };
  const ix = Math.floor(cx0), iy = Math.floor(cy0);
  const iw = Math.ceil(cx1) - ix, ih = Math.ceil(cy1) - iy;
  if (iw * k * ih * k > SHARP_MAX_AREA) return { kind: "smooth" };
  const fx = (L.dw * dpr) / L.sw, fy = (L.dh * dpr) / L.sh;
  return {
    kind: "sharp", k, ix, iy, iw, ih,
    tx: (cx0 - ix) * k, ty: (cy0 - iy) * k, tw: (cx1 - cx0) * k, th: (cy1 - cy0) * k,
    dx: L.dx * dpr + (cx0 - L.sx) * fx, dy: L.dy * dpr + (cy0 - L.sy) * fy,
    dw: (cx1 - cx0) * fx, dh: (cy1 - cy0) * fy,
  };
}

type Ctx = CanvasRenderingContext2D;

/** Draw `source` through `L` into `ctx` (device px = CSS × dpr), using
 *  `scratch` for the sharp-bilinear step. */
export function drawFrame(ctx: Ctx, source: HTMLCanvasElement, L: FrameLayout, dpr: number, scratch: HTMLCanvasElement) {
  const plan = drawPlan(L, dpr, source.width, source.height);
  if (plan.kind === "sharp") {
    const w = plan.iw * plan.k, h = plan.ih * plan.k;
    if (scratch.width !== w || scratch.height !== h) { scratch.width = w; scratch.height = h; }
    const sctx = scratch.getContext("2d");
    if (sctx) {
      sctx.imageSmoothingEnabled = false;
      sctx.clearRect(0, 0, w, h);
      sctx.drawImage(source, plan.ix, plan.iy, plan.iw, plan.ih, 0, 0, w, h);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "low";   // bilinear: no ringing at the pixel edges
      ctx.drawImage(scratch, plan.tx, plan.ty, plan.tw, plan.th, plan.dx, plan.dy, plan.dw, plan.dh);
      return;
    }
  }
  ctx.imageSmoothingEnabled = plan.kind === "smooth";
  if (plan.kind === "smooth") ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, L.sx, L.sy, L.sw, L.sh, L.dx * dpr, L.dy * dpr, L.dw * dpr, L.dh * dpr);
}

/** The device-pixel ratio the frames draw at (1–2, as Frame.tsx sizes its canvas). */
export function frameDpr(): number {
  const d = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
  return Math.max(1, Math.min(d, 2));
}

/** A snap that keeps less of the cell than this draws the full cell instead
 *  (sharp-bilinear: equal-size pixels, ≤ 1 device px of blend at their edges). */
export const SNAP_MIN_FILL = 0.8;

/** The side (css px) the whole image is drawn at inside a square cell of
 *  side `S` css px, at `dpr`, for frames whose images are `extents` native
 *  pixels across (max(w, h) of each shown tier): the largest integer
 *  multiple of the finest tier that is magnified (extent ≤ the cell in
 *  device px), floor(S·dpr / e)·e / dpr. Every coarser tier whose grid is
 *  an integer fraction of it is then an integer magnification too (LR
 *  0.1″ next to HR 0.05″); a finer, downsampled tier is drawn into the same
 *  side (smoothed). When every tier is downsampled — or the snap would keep
 *  less than `minFill` of the cell — the image fills the cell. */
export function snappedDrawSide(S: number, dpr: number, extents: readonly number[], minFill = 0): number {
  if (!(S > 0) || !(dpr > 0)) return Math.max(0, S || 0);
  const D = Math.floor(S * dpr + 1e-6);
  const up = extents.filter((e) => e > 0 && Number.isFinite(e) && e <= D);
  if (!up.length) return S;
  const e = Math.max(...up);
  const side = (Math.floor(D / e) * e) / dpr;
  return side >= S * minFill - 1e-9 ? side : S;
}

/** Magnifications (device px per native pixel of the finest tier) that the
 *  zoom steps land on. */
export const ZOOM_PRESETS: readonly number[] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64];

/** The magnification one zoom step by `factor` goes to from `m`: the preset
 *  nearest (in log) to m·factor, at least one step in that direction.
 *  `fit` is the whole image's (snapped) magnification, `full` the one at
 *  which a crop covers the whole image (smaller zoomed views do not exist),
 *  `max` the largest allowed. Zooming out past the smallest zoomed preset
 *  returns "fit". Above the largest preset (a small image in a big frame,
 *  whose fit is already past 64×) the steps go on continuously, by
 *  `factor`, up to `max`, so + never stalls short of the maximum. */
export function zoomPreset(m: number, factor: number, o: { fit: number; full: number; max: number }): number | "fit" {
  const eps = 1e-6;
  if (!(m > 0) || !(factor > 0) || factor === 1) return m;
  const target = m * factor;
  const top = ZOOM_PRESETS[ZOOM_PRESETS.length - 1];
  const dist = (p: number) => Math.abs(Math.log(p / target));
  if (factor > 1) {
    const cands = ZOOM_PRESETS.filter((p) => p > m * (1 + eps) && p > o.full * (1 + eps) && p <= o.max * (1 + eps));
    if (!cands.length) {
      const next = Math.min(target, o.max);
      return next > m * (1 + eps) ? next : m;
    }
    return cands.reduce((a, b) => (dist(b) < dist(a) ? b : a));
  }
  // out from above the presets: continuous while the target stays above them
  if (m > top * (1 + eps) && target > top * (1 + eps) && target > o.full * (1 + eps)) return target;
  const cands: (number | "fit")[] = ZOOM_PRESETS.filter((p) => p < m * (1 - eps) && p > o.full * (1 + eps));
  cands.push("fit");
  const value = (p: number | "fit") => (p === "fit" ? o.fit : p);
  return cands.reduce((a, b) => (dist(value(b)) < dist(value(a)) ? b : a));
}
