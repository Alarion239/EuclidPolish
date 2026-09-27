/* The docked inspector's width (FOUNDATION §10.4).
 *
 * The user's width is remembered per browser (prefs.inspectorWidth, saved
 * after a drag; the store's localStorage access is try/catch-guarded). Until
 * they drag it, the panel opens at a default fitted to the window: wide
 * enough for a ~480 px viewer inside (512 px, the body padding is 16 px a
 * side) when the window allows, narrower so the main content keeps about
 * 560 px — but never below 380 px (a ~348 px viewer, the old default) when
 * the window cannot give both: the image wins over the main content, which
 * then keeps only its hard 320 px minimum. The default width preference
 * (`DEFAULT_PREFS.inspectorWidth`) stands for "never resized". */
import { useSyncExternalStore } from "react";
import { DEFAULT_PREFS, INSPECTOR_WIDTH_RANGE } from "../state/prefs";

/** A ~480 px viewer inside the panel (16 px body padding each side). */
export const INSPECTOR_TARGET_PX = 512;
/** What the main content keeps when the inspector opens at its default. */
export const MAIN_MIN_PX = 560;
/** The default never opens narrower than this (the old fixed default) when
 *  the window cannot give both the target viewer and MAIN_MIN_PX. */
export const INSPECTOR_FLOOR_PX = 380;
/** The resize separator (shell.css .shell__sep). */
export const SEPARATOR_PX = 6;
/** The stage's hard minimum (the stage Panel's minSize). */
export const STAGE_MIN_PX = 320;

/** The rail's width (tokens.css --rail-w / --rail-w-collapsed). */
export function railWidth(collapsed: boolean): number {
  return collapsed ? 56 : 232;
}

/** The default width for a shell body (window − rail) of `bodyWidth` px. */
export function defaultInspectorWidth(bodyWidth: number): number {
  const [wMin, wMax] = INSPECTOR_WIDTH_RANGE;
  const room = Math.floor(bodyWidth - SEPARATOR_PX - MAIN_MIN_PX);
  const hardRoom = Math.floor(bodyWidth - SEPARATOR_PX - STAGE_MIN_PX);
  const fitted = Math.max(INSPECTOR_FLOOR_PX, Math.min(INSPECTOR_TARGET_PX, room));
  return Math.max(wMin, Math.min(wMax, fitted, hardRoom));
}

/** The width the panel opens at: the saved one (trimmed so the stage keeps
 *  its 320 px minimum), or the fitted default while none was chosen. */
export function dockedInspectorWidth(saved: number, bodyWidth: number): number {
  const [wMin, wMax] = INSPECTOR_WIDTH_RANGE;
  if (saved === DEFAULT_PREFS.inspectorWidth) return defaultInspectorWidth(bodyWidth);
  const room = Math.floor(bodyWidth - SEPARATOR_PX - STAGE_MIN_PX);
  return Math.max(wMin, Math.min(wMax, saved, room));
}

function subscribeResize(fn: () => void): () => void {
  window.addEventListener("resize", fn);
  return () => window.removeEventListener("resize", fn);
}

/** The window's inner width (re-renders on resize). */
export function useWindowWidth(): number {
  return useSyncExternalStore(subscribeResize, () => window.innerWidth, () => 1280);
}
