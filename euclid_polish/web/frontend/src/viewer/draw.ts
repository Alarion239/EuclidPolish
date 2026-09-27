/* How a frame's rendered cube is drawn into its visible canvas (pure plan +
 * the canvas calls; Frame.tsx).
 *
 * Nearest-neighbour at a NON-integer magnification gives native pixels
 * uneven widths (2 and 3 device px side by side at 2.85×), which jitters the
 * noise texture and star cores. So:
 *
 *   scale ≈ integer ≥ 1   nearest neighbour (every pixel k × k device px)
 *   scale > 1 otherwise   "sharp bilinear": nearest neighbour to the next
 *                         integer multiple k = ⌈scale⌉ in a scratch canvas,
 *                         then bilinear down to the frame (square pixels of
 *                         equal size, ≤ 1 device px of blend at their edges)
 *   scale < 1             bilinear (downsampling: nearest would drop rows)
 *
 * The frame keeps its fitted size — snapping the side to an integer multiple
 * instead would shrink the images by up to half. */
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
