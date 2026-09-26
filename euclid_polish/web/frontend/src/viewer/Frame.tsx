/* One image frame: the tier's rendered cube (a natural-resolution source
 * canvas) drawn into the visible canvas through the shared pan/zoom view,
 * with its overlays (label + magnitude, lens box, synchronized crosshair,
 * profile geometry, temperature legend, movie progress, messages) and the
 * pointer interactions (pan drag, wheel/pinch zoom, lens hover/freeze,
 * shift-drag line profile, click radial profile, double-click reset). */
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { useDisplay } from "../state/display";
import { planckianXY, srgbGamma, xyToLinearSrgb } from "./color";
import { useController, useSettings, useViewer } from "./hooks";
import { contentBoxOrigin, frameToImage, frameToImageClamped, imageToFrame, type FrameLayout, type Selection } from "./selection";

type Drag =
  | { kind: "pan"; x: number; y: number; start: Selection; moved: boolean }
  | { kind: "profile"; x: number; y: number; p0: { x: number; y: number }; moved: boolean }
  | { kind: "click"; x: number; y: number; moved: boolean };

function drawLegend(canvas: HTMLCanvasElement | null) {
  const ctx = canvas?.getContext("2d");
  if (!canvas || !ctx) return;
  const n = 160;
  const img = ctx.createImageData(14, n);
  const lo = Math.log(3000), hi = Math.log(20000);
  for (let row = 0; row < n; row++) {
    const T = Math.exp(hi - (hi - lo) * (row / (n - 1)));   // top hot, bottom cool
    const [x, y] = planckianXY(T);
    const [r, g, b] = xyToLinearSrgb(x, y);
    for (let col = 0; col < 14; col++) {
      const o = (row * 14 + col) * 4;
      img.data[o] = srgbGamma(r) * 255; img.data[o + 1] = srgbGamma(g) * 255; img.data[o + 2] = srgbGamma(b) * 255; img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

export function Frame({ tier, hidden = false, clip, label, labelRight = false }: { tier: string; hidden?: boolean; clip?: string; label?: string; labelRight?: boolean }) {
  const ctrl = useController();
  const shown = useViewer((s) => s.shown[tier]);
  const status = useViewer((s) => s.status[tier]);
  const overlay = useViewer((s) => s.overlay[tier]);
  const view = useViewer((s) => s.view);
  const hover = useViewer((s) => s.hover);
  const frozen = useViewer((s) => s.frozen);
  const readout = useViewer((s) => s.readout);
  const profile = useViewer((s) => s.profile);
  const progress = useViewer((s) => s.movieProgress[tier] ?? null);
  const settings = useSettings();
  const elRef = useRef<HTMLDivElement>(null);
  const visRef = useRef<HTMLCanvasElement>(null);
  const legendRef = useRef<HTMLCanvasElement>(null);
  const [source] = useState(() => document.createElement("canvas"));
  const sizeRef = useRef(0);
  const [size, setSize] = useState(0);
  const drag = useRef<Drag | null>(null);
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const pinch = useRef<{ d: number } | null>(null);
  const [profileDraft, setProfileDraft] = useState<{ p0: { x: number; y: number }; p1: { x: number; y: number } } | null>(null);

  // The view resolved on this tier (its centre matched through the WCS).
  const layout = useCallback((): FrameLayout | null => {
    if (!sizeRef.current) return null;
    return ctrl.layoutOf(tier, sizeRef.current);
  }, [ctrl, tier]);

  const redraw = useCallback(() => {
    const cv = visRef.current;
    const S = sizeRef.current;
    if (!cv || !S) return;
    const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
    const px = Math.max(1, Math.round(S * dpr));
    if (cv.width !== px || cv.height !== px) { cv.width = px; cv.height = px; }
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, px, px);
    const L = layout();
    if (!L || source.width < 1 || source.height < 1) return;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(source, L.sx, L.sy, L.sw, L.sh, L.dx * dpr, L.dy * dpr, L.dw * dpr, L.dh * dpr);
  }, [layout, source]);

  // Register with the engine (movie frames, lens popups, exports).
  useLayoutEffect(() => {
    const el = elRef.current, vis = visRef.current;
    if (!el || !vis) return;
    return ctrl.registerFrame({ tier, source, visible: vis, element: el, size: () => sizeRef.current, redraw });
  }, [ctrl, tier, source, redraw]);

  // Track the frame's CSS side.
  useLayoutEffect(() => {
    const el = elRef.current;
    if (!el) return;
    const measure = () => {
      const w = el.clientWidth;
      if (w !== sizeRef.current) { sizeRef.current = w; setSize(w); }
    };
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Render the cube with the display settings (the movie draws itself).
  useEffect(() => {
    if (!shown || tier === "morph") return;
    const img = ctrl.renderShown(shown, settings);
    if (!img) return;
    if (source.width !== img.width || source.height !== img.height) { source.width = img.width; source.height = img.height; }
    source.getContext("2d")?.putImageData(img, 0, 0);
    redraw();
    ctrl.bumpDrawn();
  }, [ctrl, tier, shown, settings, source, redraw]);

  useEffect(() => { redraw(); }, [redraw, view, size, shown]);

  const tempMode = !!shown && ctrl.preparedMode(shown, settings) === "temp";
  useEffect(() => { if (tempMode) drawLegend(legendRef.current); }, [tempMode]);

  // ---- pointer → image ----
  // Frame CSS coordinates are measured from the padding box (the canvas and
  // the SVG sit there; S = clientWidth), not the 1 px border's outer edge.
  // `clamp` pins a point outside the image onto its edge (profile drags).
  const toImage = (clientX: number, clientY: number, clamp = false) => {
    const el = elRef.current, L = layout();
    if (!el || !L) return null;
    const o = contentBoxOrigin(el);
    return (clamp ? frameToImageClamped : frameToImage)(L, clientX - o.left, clientY - o.top);
  };
  const uv = (p: { x: number; y: number }) => {
    const g = ctrl.geomOf(tier);
    return g ? { u: p.x / g.width, v: p.y / g.height } : null;
  };

  const raf = useRef<number | null>(null);
  const pending = useRef<{ x: number; y: number; alt: boolean } | null>(null);
  const onHoverMove = (clientX: number, clientY: number, alt: boolean) => {
    pending.current = { x: clientX, y: clientY, alt };
    if (raf.current != null) return;
    raf.current = requestAnimationFrame(() => {
      raf.current = null;
      const p = pending.current;
      if (!p) return;
      // A pointer that was already over the viewer (no mouseenter) still owns the keys.
      if (!ctrl.s.hot || !ctrl.isActive()) ctrl.activate();
      ctrl.setAltLens(p.alt);
      const q = toImage(p.x, p.y);
      if (!q) { ctrl.clearReadout(); return; }
      ctrl.hoverAt(tier, q.x, q.y);
      if (ctrl.lensActive() && !ctrl.s.frozen) {
        const n = uv(q);
        if (n) ctrl.lensHover(tier, n.u, n.v);
      }
    });
  };
  useEffect(() => () => { if (raf.current != null) cancelAnimationFrame(raf.current); }, []);

  const focusViewer = () => {
    const root = elRef.current?.closest<HTMLElement>(".cv-root");
    if (root && !root.contains(document.activeElement)) root.focus({ preventScroll: true });
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    focusViewer();
    ctrl.activate();
    if (pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()];
      pinch.current = { d: Math.hypot(a.x - b.x, a.y - b.y) };
      drag.current = null;
      return;
    }
    const q = toImage(e.clientX, e.clientY);
    if (e.shiftKey && q) {
      drag.current = { kind: "profile", x: e.clientX, y: e.clientY, p0: q, moved: false };
    } else if (ctrl.s.view && !ctrl.lensActive()) {
      drag.current = { kind: "pan", x: e.clientX, y: e.clientY, start: ctrl.s.view, moved: false };
    } else {
      drag.current = { kind: "click", x: e.clientX, y: e.clientY, moved: false };
    }
    try { (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId); } catch { /* synthetic pointer */ }
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (pointers.current.has(e.pointerId)) pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pinch.current && pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinch.current.d > 0 && d > 0) {
        const mid = toImage((a.x + b.x) / 2, (a.y + b.y) / 2);
        const n = mid ? uv(mid) : null;
        ctrl.zoomView(tier, d / pinch.current.d, n ?? undefined);
      }
      pinch.current.d = d;
      return;
    }
    const d = drag.current;
    if (d) {
      const dx = e.clientX - d.x, dy = e.clientY - d.y;
      if (!d.moved && Math.hypot(dx, dy) > 3) d.moved = true;
      if (d.kind === "pan" && d.moved) { ctrl.panBy(tier, dx, dy, d.start); return; }
      if (d.kind === "profile" && d.moved) {
        const q = toImage(e.clientX, e.clientY, true);
        if (q) setProfileDraft({ p0: d.p0, p1: q });
        return;
      }
    }
    onHoverMove(e.clientX, e.clientY, e.altKey);
  };

  const onPointerUp = (e: React.PointerEvent) => {
    pointers.current.delete(e.pointerId);
    if (pointers.current.size < 2) pinch.current = null;
    const d = drag.current;
    drag.current = null;
    if (!d) return;
    if (d.kind === "profile") {
      // A line released outside the frame ends on the image edge (clamped).
      const q = toImage(e.clientX, e.clientY, d.moved);
      if (d.moved && q) ctrl.setProfile({ kind: "line", tier, p0: d.p0, p1: q });
      else if (q) ctrl.setProfile({ kind: "radial", tier, c: q });
      setProfileDraft(null);
      return;
    }
    if (d.moved) return;
    const q = toImage(e.clientX, e.clientY);
    const n = q ? uv(q) : null;
    if (!q || !n) return;
    if (ctrl.lensActive() || e.altKey) ctrl.toggleFrozen(tier, n.u, n.v);
    else if (ctrl.s.profileOpen) ctrl.setProfile({ kind: "radial", tier, c: q });
  };

  const onPointerLeave = () => {
    pending.current = null;
    ctrl.clearReadout();
    const s = ctrl.s;
    if (s.hover && s.hover.sourceTier === tier && !s.frozen) ctrl.hideHover();
  };

  // Wheel: zoom (focused / ⌘-Ctrl / always), the lens zoom while it is active,
  // horizontal wheel = brightness. A plain wheel otherwise scrolls the page.
  useEffect(() => {
    const el = elRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      const q = toImage(e.clientX, e.clientY);
      const n = q ? uv(q) : null;
      const horizontal = Math.abs(e.deltaX) > Math.abs(e.deltaY);
      if (ctrl.lensActive() && !ctrl.s.frozen && n) {
        e.preventDefault(); e.stopPropagation();
        if (horizontal) ctrl.wheelGain(tier, e.deltaX);
        else ctrl.lensZoom(tier, n.u, n.v, e.deltaY);
        return;
      }
      const mode = useDisplay.getState().wheel;
      const root = el.closest(".cv-root");
      const focused = !!root && root.contains(document.activeElement);
      const zoom = e.ctrlKey || e.metaKey || mode === "always-zoom" || (mode === "zoom-when-focused" && focused);
      if (!zoom) return;
      e.preventDefault(); e.stopPropagation();
      if (horizontal && !e.ctrlKey && !e.metaKey) { ctrl.wheelGain(tier, e.deltaX); return; }
      const unit = e.deltaMode === 1 ? 33 : e.deltaMode === 2 ? 400 : 1;
      // One mouse notch ≈ 100 px → ×1.28; clamped so a large delta never jumps far.
      const delta = Math.max(-150, Math.min(150, e.deltaY * unit * (e.ctrlKey ? 4 : 1)));
      const factor = Math.exp(-delta * 0.0025);
      ctrl.zoomView(tier, factor, n ?? undefined);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
    // toImage/uv read the latest layout through refs and the controller
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ctrl, tier]);

  // ---- overlays ----
  const S = size;
  const L = S ? ctrl.layoutOf(tier, S, view) : null;
  const g = ctrl.geomOf(tier);
  const sel = frozen || hover;
  const crop = sel && g && g.ready ? ctrl.cropOf(tier, sel) : null;
  const box = crop && L ? { a: imageToFrame(L, crop.x, crop.y), b: imageToFrame(L, crop.x + crop.side, crop.y + crop.side) } : null;
  const here = readout?.tiers.find((t) => t.tier === tier);
  const cross = here && L ? imageToFrame(L, here.fx, here.fy) : null;
  const inFrame = (p: { x: number; y: number } | null) => !!p && p.x >= 0 && p.y >= 0 && p.x <= S && p.y <= S;
  const prof = profileDraft ? { kind: "line" as const, tier, p0: profileDraft.p0, p1: profileDraft.p1 } : profile;
  let profShape: { line?: [{ x: number; y: number }, { x: number; y: number }]; dot?: { x: number; y: number } } | null = null;
  if (prof && L) {
    if (prof.kind === "line") {
      const a = ctrl.mapPoint(prof.tier, tier, prof.p0.x, prof.p0.y), b = ctrl.mapPoint(prof.tier, tier, prof.p1.x, prof.p1.y);
      if (a && b) profShape = { line: [imageToFrame(L, a.x, a.y), imageToFrame(L, b.x, b.y)] };
    } else {
      const c = ctrl.mapPoint(prof.tier, tier, prof.c.x, prof.c.y);
      if (c) profShape = { dot: imageToFrame(L, c.x, c.y) };
    }
  }

  const message = status?.kind === "error" || status?.kind === "missing" ? status.message : "";
  const hint = status?.kind === "error" ? status.hint : undefined;
  const loading = status?.kind === "loading" || (tier === "morph" && progress != null);
  const text = message ? "" : (label ?? overlay ?? "");

  return (
    <div ref={elRef} className={`cv-frame${loading ? " cv-loading" : ""}${hidden ? " cv-frame--hidden" : ""}`}
      data-tier={tier} role="img" aria-label={text || ctrl.tierLabel(tier)}
      style={clip ? { clipPath: clip } : undefined}
      onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp} onPointerLeave={onPointerLeave}
      onDoubleClick={() => ctrl.resetView()}>
      <canvas ref={visRef} className="cv-canvas" />
      <svg className="cv-svg" width={S || 1} height={S || 1} aria-hidden="true">
        {box && <rect className="cv-lens-rect" x={box.a.x} y={box.a.y} width={Math.max(1, box.b.x - box.a.x)} height={Math.max(1, box.b.y - box.a.y)} />}
        {cross && inFrame(cross) && <g className="cv-cross">
          <line x1={cross.x} y1={0} x2={cross.x} y2={S} /><line x1={0} y1={cross.y} x2={S} y2={cross.y} />
        </g>}
        {profShape?.line && <g className="cv-prof">
          <line x1={profShape.line[0].x} y1={profShape.line[0].y} x2={profShape.line[1].x} y2={profShape.line[1].y} />
          <circle cx={profShape.line[0].x} cy={profShape.line[0].y} r={3} /><circle cx={profShape.line[1].x} cy={profShape.line[1].y} r={3} />
        </g>}
        {profShape?.dot && <g className="cv-prof"><circle cx={profShape.dot.x} cy={profShape.dot.y} r={5} />
          <line x1={profShape.dot.x - 8} y1={profShape.dot.y} x2={profShape.dot.x + 8} y2={profShape.dot.y} />
          <line x1={profShape.dot.x} y1={profShape.dot.y - 8} x2={profShape.dot.x} y2={profShape.dot.y + 8} /></g>}
      </svg>
      {tempMode && !message && (
        <div className="cv-legend-wrap">
          <span className="cv-legend-tick">20k</span>
          <canvas ref={legendRef} className="cv-legend" width={14} height={160} />
          <span className="cv-legend-tick">3k</span>
        </div>
      )}
      {text && <div className={`cv-overlay${labelRight ? " cv-overlay--right" : ""}`}>{text}</div>}
      {message && <div className="cv-msg"><span>{message}{hint && hint !== message && <><br /><em>{hint}</em></>}</span></div>}
      {tier === "morph" && progress != null && (
        <div className="cv-movie-prog" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress * 100)}>
          <div className="cv-movie-prog__fill" style={{ width: `${Math.round(progress * 100)}%` }} />
          <div className="cv-movie-prog__lbl">caching movie… {Math.round(progress * 100)}%</div>
        </div>
      )}
    </div>
  );
}
