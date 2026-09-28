/* Land on a section of a page and stay there while the page settles.
 *
 * A link can open a page on one of its sections (a synthetic drawer, the
 * records census, System › Code's FASRC side). The content above the section
 * — figures, viewers, tables — arrives after the first paint and pushes it
 * down, so a single scroll leaves the reader at the top. holdInView scrolls
 * the section to the top of the page and re-aligns it whenever it moves, for
 * a few seconds or until the reader scrolls, clicks or types — never after.
 * It polls instead of using requestAnimationFrame or ResizeObserver, so it
 * also settles in a hidden tab. */
import { useEffect, useRef, type RefObject } from "react";

const HOLD_MS = 5000;
const HOLD_EVERY_MS = 250;
const HOLD_RELEASE = ["wheel", "touchstart", "keydown", "pointerdown"] as const;

/** Keep `target` (an element or its id) aligned to the top; returns the
 *  release function. `onRelease` runs once, however it is released. */
export function holdInView(
  target: string | HTMLElement,
  { settleMs = HOLD_MS, onRelease }: { settleMs?: number; onRelease?: () => void } = {},
): () => void {
  let last: number | null = null;
  let released = false;
  const node = () => (typeof target === "string" ? document.getElementById(target) : target);
  const align = () => {
    const el = node();
    if (!el || released) return;
    const top = el.getBoundingClientRect().top;
    if (last !== null && Math.abs(top - last) <= 1) return;
    el.scrollIntoView?.({ block: "start" });
    last = el.getBoundingClientRect().top;
  };
  const release = () => {
    if (released) return;
    released = true;
    window.clearInterval(timer);
    window.clearTimeout(deadline);
    for (const type of HOLD_RELEASE) window.removeEventListener(type, release, true);
    onRelease?.();
  };
  const timer = window.setInterval(align, HOLD_EVERY_MS);
  const deadline = window.setTimeout(release, settleMs);
  for (const type of HOLD_RELEASE) window.addEventListener(type, release, { capture: true, passive: true });
  align();
  return release;
}

/** holdInView for `ref` once `enabled` (the URL asks for this section) and
 *  `ready` (the section has rendered). Torn down before it settled
 *  (StrictMode's double mount, an unmount), the next mount may try again. */
export function useArrivalScroll(ref: RefObject<HTMLElement | null>, enabled: boolean, ready: boolean,
  settleMs = HOLD_MS): void {
  const done = useRef(false);
  useEffect(() => {
    if (!enabled || !ready || done.current) return undefined;
    const el = ref.current;
    if (!el) return undefined;
    done.current = true;
    let settled = false;
    const release = holdInView(el, { settleMs, onRelease: () => { settled = true; } });
    return () => {
      const wasSettled = settled;
      release();
      if (!wasSettled) done.current = false;
    };
  }, [ref, enabled, ready, settleMs]);
}
