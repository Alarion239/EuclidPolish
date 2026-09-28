/* The tile card's viewer stage: a capped scroll box (results.css
 * `.res-card__stage`), so the viewer fits its frames under the cap and the
 * Δm footer, the status sentence and the headline numbers stay in the first
 * screen. A narrow card makes the frames width-limited, shorter than the
 * cap: the stage then shrinks to the viewer (no blank band under the
 * images). It goes back to the cap when the card's width or the frame count
 * changes, or when the viewer outgrows it, and the viewer refits. */
import { useEffect, type RefObject } from "react";
import { FIT_MARGIN } from "../../../viewer/fit";

/** Slack kept under a snug viewer (px): the fit never sees less room than it
 *  used, so shrinking the stage cannot shrink the frames. */
export const SNUG_SLACK = 4;
const VAR = "--res-stage-h";
/** How long the viewer gets to refit into a reset stage before the next look. */
const REFIT_MS = 250;
const MAX_WAITS = 60;

/** What the stage should do for a viewer `host` px tall in a stage `stage`
 *  px tall: "keep", "reset" (back to the cap; the viewer overflows it) or a
 *  snug height in px. */
export function snugStep(host: number, stage: number): "keep" | "reset" | number {
  if (!(host > 0) || !(stage > 0)) return "keep";
  if (host > stage - FIT_MARGIN + 1) return host > stage ? "reset" : "keep";
  const snug = Math.ceil(host + FIT_MARGIN + SNUG_SLACK);
  return snug + SNUG_SLACK < stage ? snug : "keep";
}

export function useSnugStage(ref: RefObject<HTMLElement>, key: string): void {
  useEffect(() => {
    const stage = ref.current;
    if (!stage || typeof ResizeObserver === "undefined") return;
    stage.style.removeProperty(VAR);
    let width = stage.clientWidth;
    let frames = -1;
    let waits = 0;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const check = () => {
      timer = null;
      const host = stage.firstElementChild as HTMLElement | null;
      if (!host) return;
      const n = stage.querySelectorAll(".cv-canvas").length;
      // Back to the cap, then look again once the viewer has refitted.
      const reset = () => { stage.style.removeProperty(VAR); again(REFIT_MS); };
      if (stage.clientWidth !== width || (frames >= 0 && n !== frames)) {
        width = stage.clientWidth;
        frames = n;
        reset();
        return;
      }
      frames = n;
      // Still loading: nothing to fit to yet; look again for a while (a
      // hidden tab never delivers ResizeObserver notes, timers still run).
      if (!n) { if (waits++ < MAX_WAITS) again(REFIT_MS); return; }
      const step = snugStep(host.offsetHeight, stage.clientHeight);
      if (step === "reset") reset();
      else if (step !== "keep") stage.style.setProperty(VAR, `${step}px`);
    };
    // A task later, like the viewer's own fit (never inside the observer callback).
    function again(ms = 50) { if (timer == null) timer = setTimeout(check, ms); }
    const ro = new ResizeObserver(() => again());
    ro.observe(stage);
    const watchHost = () => { const h = stage.firstElementChild; if (h) ro.observe(h); };
    watchHost();
    const mo = typeof MutationObserver !== "undefined" ? new MutationObserver(() => { watchHost(); again(); }) : null;
    mo?.observe(stage, { childList: true });
    again();
    return () => {
      ro.disconnect();
      mo?.disconnect();
      if (timer != null) clearTimeout(timer);
      stage.style.removeProperty(VAR);
    };
  }, [ref, key]);
}
