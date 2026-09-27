/* Page-side fit for an image viewer that is not at the top of its scroll box.
 *
 * The viewer (src/viewer/TierGrid.tsx) sizes its frames to the height of its
 * nearest scrolling ancestor minus its own bar and readout — it does not
 * subtract how far down that box it sits. On a page with a file bar, filters
 * or a card head above the images the frames therefore ran past the fold.
 * `FitBox` (FitBox.tsx) wraps the viewer in its own scroll box whose height is
 * exactly what is left of the stage below the box's top, so the viewer's fit
 * is right; the pure maths is here. */

export type FitBoxInput = {
  /** The scroll box's (the stage's / inspector body's) client height. */
  stage: number;
  /** The FitBox's top in the scroll box's content coordinates. */
  offset: number;
  /** A floor: a viewer far down a page still gets this much (then one scroll
   *  shows all of it). Capped by the stage height. */
  min?: number;
  /** Space kept free under the box. */
  gap?: number;
};

/** The FitBox height: the stage below its top, at least `min` (≤ the stage). */
export function fitBoxHeight({ stage, offset, min = 0, gap = 0 }: FitBoxInput): number {
  const room = Math.max(0, stage - gap);
  const left = Math.max(0, room - Math.max(0, offset));
  return Math.floor(Math.max(left, Math.min(min, room)));
}

/** Height the viewer leaves unused in a box of `box` px (width-limited
 *  frames): given back to the page as a negative bottom margin. */
export function fitBoxSlack(box: number, content: number): number {
  if (!(content > 0)) return 0;
  return Math.max(0, Math.floor(box - content));
}
