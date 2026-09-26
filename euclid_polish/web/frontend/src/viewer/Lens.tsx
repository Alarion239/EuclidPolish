/* The magnifier lens: one popup per ready frame showing the SAME sky region
 * (a shared selection matched in arcsec across pixel scales), laid out at the
 * source crop's corners without collisions. Hover popups are fixed to the
 * viewport; a frozen crop's popups keep their document position. Drawn with
 * nearest-neighbour sampling from each frame's natural-resolution canvas. */
import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { useController, useViewer } from "./hooks";
import { LENS_CANVAS, currentViewport, lensSide, receptiveFieldLabels, type Selection } from "./selection";
import type { LensPlacement } from "./types";

function LensPopup({ tier, placement, selection, frozen, order }: {
  tier: string; placement: LensPlacement; selection: Selection; frozen: boolean; order: number;
}) {
  const ctrl = useController();
  const drawn = useViewer((s) => s.drawn);
  const shown = useViewer((s) => s.shown[tier]);
  const meta = useViewer((s) => s.meta);
  const ref = useRef<HTMLCanvasElement>(null);
  const vp = currentViewport();
  const side = lensSide(vp);
  // The shared selection resolved on this tier (centre matched through the WCS).
  const crop = ctrl.cropOf(tier, selection);

  useEffect(() => {
    const cv = ref.current, h = ctrl.frames.get(tier);
    const ctx = cv?.getContext("2d");
    if (!cv || !ctx || !h || !crop) return;
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(h.source, crop.x, crop.y, crop.side, crop.side, 0, 0, cv.width, cv.height);
  });
  void drawn; void shown;

  if (!crop) return null;
  const labels = [`${(side / crop.side).toFixed(1)}×`, ...receptiveFieldLabels(crop.angularSideArcsec, meta?.receptive_fields).map((l) => `${l} RF`)];
  if (frozen) labels.push("click to unfreeze");
  const style = frozen
    ? { position: "absolute" as const, left: placement.left, top: placement.top, width: side, height: side, zIndex: 1000 + order }
    : { position: "fixed" as const, left: placement.left - vp.scrollX, top: placement.top - vp.scrollY, width: side, height: side, zIndex: 1000 + order };
  return (
    <div className={`cv-lens${frozen ? " cv-lens--frozen" : ""}`} style={{ ...style, display: "block" }} aria-hidden="true" data-tier={tier}>
      <canvas ref={ref} className="cv-lens__canvas" width={LENS_CANVAS} height={LENS_CANVAS} />
      <span className="cv-lens__label">{labels.join(" · ")}</span>
    </div>
  );
}

export function LensLayer() {
  const ctrl = useController();
  const hover = useViewer((s) => s.hover);
  const frozen = useViewer((s) => s.frozen);
  const lens = useViewer((s) => s.lens);
  const selection = frozen || hover;

  // Hover popups follow page scroll and resize (fixed → recompute placements).
  useEffect(() => {
    if (!selection) return;
    const onChange = () => requestAnimationFrame(() => ctrl.refreshLenses());
    window.addEventListener("resize", onChange);
    window.addEventListener("scroll", onChange, { passive: true, capture: true });
    return () => {
      window.removeEventListener("resize", onChange);
      window.removeEventListener("scroll", onChange, { capture: true });
    };
  }, [ctrl, selection]);

  if (!selection || typeof document === "undefined") return null;
  const keys = ctrl.frameKeys().filter((k) => lens[k]);
  return createPortal(
    <>{keys.map((k, i) => <LensPopup key={k} tier={k} placement={lens[k]} selection={selection} frozen={!!frozen} order={i} />)}</>,
    document.body,
  );
}
