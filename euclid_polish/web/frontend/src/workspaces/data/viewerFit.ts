/* The Data pages' image-first fit (pure; ViewerStage.tsx measures, this decides).
 *
 * The viewer sizes its square frames to the WHOLE stage height minus its own
 * bar and readout (viewer/fit.ts). A page puts the tab strip, its toolbar and
 * a caption above the viewer, so a height-limited frame row (one frame —
 * Cutouts, PSFs, blink / swipe — or a big pane) would run that far below the
 * fold. `frameFit` is the grid the viewer would lay out if it fitted the
 * height under its own top instead: the same `fitFrames` choice (auto / one
 * row / grid / stack) over the same width, with the smaller height. The
 * viewer stays full width — its bar keeps the rows it has, the frames centre
 * on the light table — and ViewerStage only narrows the frames (CSS on
 * `.cv-frames`). A width-limited viewer is left alone; so is one that sits
 * too far down the page (it scrolls into view).
 *
 * `besideWidth`: with a side panel (the Cutouts gallery, the PSF clusters),
 * the viewer may give the width it does not need to the panel — never below
 * the width its bar needs for two unwrapped rows, so narrowing it never adds a
 * bar row (which would shrink the frames again). */
import { FIT_MARGIN, FRAME_GAP, availableHeight, fitFrames, type LayoutMode } from "../../viewer/fit";

/** Below this height left under the viewer's top, the frames keep the viewer's own fit and the page scrolls. */
export const MIN_CAP_SIDE = 240;

/** The height left for the frames under the viewer's top: the stage height
 *  below the table's top, less the table's chrome (bar + readout, and a
 *  Display dock stacked under the frames) and the viewer's fit margin. */
export function maxFirstRowSide(o: { stageHeight: number; tableTop: number; chrome: number; margin?: number }): number {
  return Math.max(0, Math.floor(o.stageHeight - o.tableTop - o.chrome - (o.margin ?? FIT_MARGIN)));
}

export type FrameFitInput = {
  /** Frames the viewer lays out (1 while blinking or swiping). */
  n: number;
  mode: LayoutMode;
  /** The frame grid's width with the viewer at full width (css px: the row
   *  less the width the table spends beside the frames, the Display dock). */
  width: number;
  /** The stage's viewport height (the viewer's own fit height source). */
  stageHeight: number;
  /** The table's top inside the stage content (at scroll 0). */
  tableTop: number;
  /** Bar + readout (+ a dock under the frames): the table less its frames. */
  chrome: number;
  minSide?: number;
};

export type FrameFit = { columns: number; side: number };

/** The frame grid that fits under the viewer's top (columns and square
 *  side), or null when the viewer's own fit already does (width-limited) or
 *  too little height is left for a sensible fit. */
export function frameFit(o: FrameFitInput): FrameFit | null {
  const width = Math.floor(o.width);
  if (!(width > 0)) return null;
  const n = Math.max(1, Math.floor(o.n || 1));
  const target = maxFirstRowSide(o);
  if (target < (o.minSide ?? MIN_CAP_SIDE)) return null;
  const own = fitFrames({ n, width, height: availableHeight(o.stageHeight, o.chrome), mode: o.mode });
  const fit = fitFrames({ n, width, height: target, mode: o.mode });
  if (fit.side >= own.side && fit.columns === own.columns) return null;
  return { columns: fit.columns, side: fit.side };
}

/** The grid's width for a fit (the frames and the gaps between them). */
export function gridWidth(fit: FrameFit, gap = FRAME_GAP): number {
  return fit.columns * fit.side + gap * Math.max(0, fit.columns - 1);
}

/** The viewer width that leaves a side panel room beside it (css px), or
 *  null to keep the panel below: the frames (+ the dock beside them), but
 *  never narrower than the bar's own two-row width. */
export function besideWidth(o: {
  rowWidth: number; fit: FrameFit | null; extraWidth?: number; barMin?: number; asideMin: number; gap: number;
}): number | null {
  if (!o.fit || !(o.rowWidth > 0)) return null;
  const viewer = Math.ceil(Math.max(gridWidth(o.fit) + Math.max(0, o.extraWidth ?? 0), o.barMin ?? 0));
  return o.rowWidth - viewer - o.gap >= o.asideMin ? viewer : null;
}

/** The frames the viewer lays out: one while blink / swipe stacks them, the
 *  page's expected count while it is still loading (so it does not jump). */
export function frameCount(dom: { frames: number; stacked: boolean; loading: boolean }, hint: number): number {
  if (dom.stacked) return 1;
  if (dom.loading || dom.frames < 1) return Math.max(1, Math.floor(hint || 1));
  return dom.frames;
}
