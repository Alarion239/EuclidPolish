/* Fit sizing of the image viewer's frame grid (pure; src/viewer/README.md).
 *
 * Every frame is a square. Its side is the largest that fits BOTH the width
 * per column and the height per row of the space the viewer has: the grid's
 * own width, and the height of the stage viewport (the page's scroll box, or
 * the focus-mode surround) below the viewer's own top (`heightUnderTop`:
 * the rows above it on the page are subtracted too), minus the viewer's bar
 * and readout and a small margin. The layout picks the column count:
 *
 *   auto     the count that gives the largest side for the visible frames
 *   one-row  every frame side by side
 *   grid     ⌈√n⌉ columns (a near-square grid)
 *   stack    one column
 *
 * A side never exceeds the width per column. The height may be relaxed down
 * to MIN_FRAME_SIDE (a short window still gets usable frames; the page
 * scrolls). */

export type LayoutMode = "auto" | "one-row" | "grid" | "stack";
export const LAYOUT_MODES: readonly LayoutMode[] = ["auto", "one-row", "grid", "stack"];
export const LAYOUT_LABEL: Record<LayoutMode, string> = {
  auto: "Auto", "one-row": "One row", grid: "Grid", stack: "Stack",
};
export const LAYOUT_HINT: Record<LayoutMode, string> = {
  auto: "The arrangement that makes the frames largest",
  "one-row": "Every frame side by side",
  grid: "A near-square grid",
  stack: "One frame per row",
};

/** Gap between frames (css px): frames sit edge to edge on the surround. */
export const FRAME_GAP = 2;
/** The height may shrink a frame down to this side, never the width. */
export const MIN_FRAME_SIDE = 160;
/** Breathing room kept below the readout inside the viewport. */
export const FIT_MARGIN = 8;

export type Fit = { columns: number; rows: number; side: number };

/** The largest square side (floored css px, unclamped) for `n` frames in
 *  `columns` columns inside width × height. */
export function sideFor(n: number, columns: number, width: number, height: number, gap = FRAME_GAP): number {
  const count = Math.max(1, Math.floor(n));
  const c = Math.max(1, Math.min(count, Math.floor(columns)));
  const rows = Math.ceil(count / c);
  const byWidth = (width - gap * (c - 1)) / c;
  const byHeight = (height - gap * (rows - 1)) / rows;
  return Math.max(0, Math.floor(Math.min(byWidth, byHeight)));
}

/** "Auto" treats arrangements whose frames are within this factor of the
 *  largest as equally good, and among those takes the one with the fewest
 *  empty cells, then the most columns: frames side by side read as a
 *  comparison, and a 2 + 1 grid should not leave an empty cell for a few
 *  pixels (Disagreement at 720 px: 216 px in one row vs 228 px in 2 + 1). */
export const FEWER_COLUMNS_GAIN = 1.08;

/** The column count of a layout mode for `n` frames. "auto": the largest
 *  side, near-ties (FEWER_COLUMNS_GAIN) going to the fewest empty cells,
 *  then to more columns. */
export function columnsFor(mode: LayoutMode, n: number, width: number, height: number, gap = FRAME_GAP): number {
  const count = Math.max(1, Math.floor(n));
  if (mode === "one-row") return count;
  if (mode === "stack") return 1;
  if (mode === "grid") return Math.ceil(Math.sqrt(count));
  const cands: { c: number; side: number; empty: number }[] = [];
  for (let c = count; c >= 1; c--) {
    cands.push({ c, side: sideFor(count, c, width, height, gap), empty: Math.ceil(count / c) * c - count });
  }
  const max = Math.max(...cands.map((k) => k.side));
  const near = cands.filter((k) => k.side * FEWER_COLUMNS_GAIN >= max || k.side >= max - 0.5);
  near.sort((a, b) => a.empty - b.empty || b.c - a.c);
  return near[0]?.c ?? count;
}

/** Columns, rows and the frame side for `n` frames. */
export function fitFrames(o: { n: number; width: number; height: number; mode: LayoutMode; gap?: number; minSide?: number }): Fit {
  const gap = o.gap ?? FRAME_GAP;
  const minSide = o.minSide ?? MIN_FRAME_SIDE;
  const count = Math.max(1, Math.floor(o.n || 0));
  const width = Math.max(0, o.width);
  const height = Math.max(0, o.height);
  const columns = columnsFor(o.mode, count, width, height, gap);
  const rows = Math.ceil(count / columns);
  const byWidth = Math.max(0, Math.floor((width - gap * (columns - 1)) / columns));
  const fitted = sideFor(count, columns, width, height, gap);
  return { columns, rows, side: Math.min(byWidth, Math.max(minSide, fitted)) };
}

/** Height left for the frames: the viewport minus the viewer's own chrome
 *  (bar + readout) and the margin. */
export function availableHeight(viewport: number, chrome: number, margin = FIT_MARGIN): number {
  return Math.max(0, Math.floor(viewport - chrome - margin));
}

/** A viewer that starts this far down its scroll box is fitted under its own
 *  top only while that leaves at least this share of the whole-viewport
 *  height (and MIN_FRAME_SIDE); further down, it fits the whole viewport
 *  and you scroll to it. */
export const LEAD_KEEP = 0.5;

/** Height for the frames of a viewer whose table starts `lead` css px below
 *  the top of its scroll box's content (the tab strip, a toolbar, a caption
 *  and the version banner above it, measured at scroll 0): the viewport
 *  below the viewer's top, minus its chrome and the margin — so the first
 *  frame row AND the readout are in sight without scrolling. A viewer far
 *  down a page (less than LEAD_KEEP of the full height left) gets the whole
 *  viewport's height instead. */
export function heightUnderTop(o: { viewport: number; chrome: number; lead?: number; margin?: number }): number {
  const full = availableHeight(o.viewport, o.chrome, o.margin ?? FIT_MARGIN);
  const lead = Math.max(0, Math.ceil(o.lead ?? 0));
  if (!(lead > 0)) return full;
  const below = Math.max(0, full - lead);
  return below >= Math.max(MIN_FRAME_SIDE, full * LEAD_KEEP) ? below : full;
}

export const LAYOUT_STORAGE_KEY = "euclid-polish.viewer.layout";
/** The pre-2026-09-27 key ("one-row" | "two-rows"). */
export const LEGACY_LAYOUT_STORAGE_KEY = "euclid-polish.cutout-viewer.layout";

/** The saved layout. Without one, the old key's explicit "two rows" becomes
 *  "grid"; its "one row" was the old default, so it becomes "auto". */
export function parseLayout(raw: string | null | undefined, legacy?: string | null): LayoutMode {
  if (raw && (LAYOUT_MODES as readonly string[]).includes(raw)) return raw as LayoutMode;
  if (raw == null && legacy === "two-rows") return "grid";
  return "auto";
}

/** The old two-value layout the publication figure understands. */
export function figureLayout(fit: Pick<Fit, "rows"> | null, n: number): "one-row" | "two-rows" {
  return fit && fit.rows > 1 && n > 2 ? "two-rows" : "one-row";
}
