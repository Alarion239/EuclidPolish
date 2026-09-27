/* The frame grid: one square frame per selected tier (canonical order, then
 * residual tiers), edge to edge with 2 px gaps on the light table. The side
 * comes from fit.ts: the largest square that fits the grid's width per column
 * AND the height the viewer has — the stage viewport (the page's scroll box,
 * else the window; the focus-mode surround in focus mode) below the viewer's
 * own top (the tab strip, toolbars and banner above it: `heightUnderTop`),
 * minus the bar, the readout and a small margin (and, in a narrow viewer, the
 * Display dock stacked under the frames; in focus mode the profile panel).
 * Content above the viewer that grows or goes (a banner dismissed) refits
 * it. The layout (auto / one row / grid / stack)
 * picks the column count. Blink stacks every frame in one cell and cycles
 * them; swipe overlays the first two with a draggable divider. */
import { useLayoutEffect, useRef, type CSSProperties } from "react";
import { fitFrames, heightUnderTop } from "./fit";
import { Frame } from "./Frame";
import { useController, useViewer } from "./hooks";

function scrollParent(el: HTMLElement | null): HTMLElement | null {
  for (let p = el?.parentElement ?? null; p; p = p.parentElement) {
    const oy = getComputedStyle(p).overflowY;
    if ((oy === "auto" || oy === "scroll") && p.clientHeight > 0) return p;
  }
  return null;
}

const outerHeight = (el: HTMLElement | null | undefined) => {
  if (!el) return 0;
  const cs = getComputedStyle(el);
  return el.offsetHeight + (parseFloat(cs.marginTop) || 0) + (parseFloat(cs.marginBottom) || 0);
};

/** Lay the grid out for `n` frames (also while loading: one frame). */
function useFit(ref: React.RefObject<HTMLDivElement>, n: number) {
  const ctrl = useController();
  const layout = useViewer((s) => s.layout);
  const focus = useViewer((s) => s.focus);
  // The bar's height changes with its row count: refit in the same layout
  // pass (before paint), not a frame later through the ResizeObserver.
  const barRows = useViewer((s) => s.bar.rows);
  const dock = useViewer((s) => s.dock);
  const profileOpen = useViewer((s) => s.profileOpen);
  useLayoutEffect(() => {
    const grid = ref.current;
    if (!grid) return;
    const root = grid.closest<HTMLElement>(".cv-root");
    const stage = focus ? null : scrollParent(root);
    const fit = () => {
      const bar = root?.querySelector<HTMLElement>(".cv-bar");
      const readout = root?.querySelector<HTMLElement>(".cv-readout");
      // The Display dock takes height only when it sits below the frames (a
      // narrow viewer); beside them it takes width, which the grid measures.
      const body = grid.parentElement;
      const dockEl = body?.querySelector<HTMLElement>(":scope > .cv-dock");
      const dockBelow = dockEl && body && getComputedStyle(body).flexDirection === "column" ? outerHeight(dockEl) : 0;
      // In focus mode the profile panel sits under the table, inside the surround.
      const panels = focus ? outerHeight(root?.querySelector<HTMLElement>(":scope > .cv-panels")) : 0;
      const chrome = outerHeight(bar) + outerHeight(readout) + dockBelow + panels;
      let viewport: number;
      let lead = 0;
      if (focus && root) {
        const cs = getComputedStyle(root);
        viewport = root.clientHeight - (parseFloat(cs.paddingTop) || 0) - (parseFloat(cs.paddingBottom) || 0);
      } else {
        viewport = stage ? stage.clientHeight : (window.visualViewport?.height ?? window.innerHeight);
        // How far below the top of its scroll box's content the viewer starts
        // (at scroll 0): the rows above it on the page.
        const top = root?.getBoundingClientRect().top ?? 0;
        lead = stage
          ? top - (stage.getBoundingClientRect().top + stage.clientTop) + stage.scrollTop
          : top + window.scrollY;
        watchAbove();
      }
      const width = grid.clientWidth;
      if (!(width > 0)) return;
      const f = fitFrames({ n, width, height: heightUnderTop({ viewport, chrome, lead }), mode: layout });
      grid.style.setProperty("--cv-frame-size", `${f.side}px`);
      grid.style.setProperty("--cv-columns", `${f.columns}`);
      ctrl.setFit(f);
    };
    // A ResizeObserver callback refits a task later (resizing the frames in
    // the callback would re-trigger it: the "ResizeObserver loop" error). A
    // timer, not requestAnimationFrame: rAF never runs in a hidden tab, and
    // the frames must be right when the tab comes back.
    let timer: ReturnType<typeof setTimeout> | null = null;
    const again = () => { if (timer == null) timer = setTimeout(() => { timer = null; fit(); }, 0); };
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(again) : null;
    // What sits above the viewer in its scroll box moves it without resizing
    // it: watch what precedes the viewer or one of its ancestors (up to the
    // scroll box) — a banner dismissed, a caption or toolbar that wraps — and
    // the ancestors themselves (a row inserted above grows them). An observed
    // element that leaves the DOM reports once more (size 0). The fit reads
    // only positions above the frames, so refitting settles in one pass.
    // A row INSERTED above (the version banner arriving, a caption once the
    // data loads) is caught by a childList observer on each ancestor (not the
    // subtree: the readout's text under the pointer never reaches it).
    const watched = new Set<Element>();
    const mo = typeof MutationObserver !== "undefined" ? new MutationObserver(again) : null;
    const watch = (el: Element) => {
      if (watched.has(el)) return;
      watched.add(el);
      ro?.observe(el);
      if (el !== root && el.contains(root)) mo?.observe(el, { childList: true });
    };
    function watchAbove() {
      for (let node: Element | null = root ?? null; node && node !== stage; node = node.parentElement) {
        if (node !== root) watch(node);
        for (let sib = node.previousElementSibling; sib; sib = sib.previousElementSibling) watch(sib);
      }
      if (stage && !watched.has(stage)) { watched.add(stage); mo?.observe(stage, { childList: true }); }
    }
    fit();
    ro?.observe(grid);
    const bar = root?.querySelector<HTMLElement>(".cv-bar");
    if (bar) ro?.observe(bar);
    // the readout takes a second line when one would cut values off (ReadoutBar)
    const readoutEl = root?.querySelector<HTMLElement>(".cv-readout");
    if (readoutEl) ro?.observe(readoutEl);
    const dockEl = grid.parentElement?.querySelector<HTMLElement>(":scope > .cv-dock");
    if (dockEl) ro?.observe(dockEl);
    const panelsEl = focus ? root?.querySelector<HTMLElement>(":scope > .cv-panels") : null;
    if (panelsEl) ro?.observe(panelsEl);
    if (stage) ro?.observe(stage);
    if (focus && root) ro?.observe(root);
    window.addEventListener("resize", fit);
    window.visualViewport?.addEventListener("resize", fit);
    return () => {
      if (timer != null) clearTimeout(timer);
      ro?.disconnect();
      mo?.disconnect();
      window.removeEventListener("resize", fit);
      window.visualViewport?.removeEventListener("resize", fit);
    };
  }, [ctrl, ref, n, layout, focus, barRows, dock, profileOpen]);
}

export function TierGrid() {
  const ctrl = useController();
  const tiers = useViewer((s) => s.tiers);
  const residuals = useViewer((s) => s.residuals);
  const meta = useViewer((s) => s.meta);
  const compare = useViewer((s) => s.compare);
  const blinkAt = useViewer((s) => s.blinkAt);
  const swipe = useViewer((s) => s.swipe);
  const metaError = useViewer((s) => s.metaError);
  const zoomed = useViewer((s) => !!s.view);
  const ref = useRef<HTMLDivElement>(null);
  const keys = meta ? ctrl.frameKeys() : [];
  const stacked = compare !== "off" && keys.length > 1;
  useFit(ref, stacked ? 1 : Math.max(1, keys.length));
  void tiers; void residuals;   // (frameKeys reads them; subscribed for re-render)

  if (metaError || !meta) {
    return (
      <div ref={ref} className="cv-frames">
        <div className={`cv-frame cv-frame--message${metaError ? "" : " cv-loading"}`}>
          <div className="cv-msg"><span>{metaError || "Loading…"}</span></div>
        </div>
      </div>
    );
  }

  if (stacked && compare === "blink") {
    const active = keys[blinkAt % keys.length];
    return (
      <div ref={ref} className={`cv-frames${zoomed ? " cv-frames--zoomed" : ""}`}>
        <div className="cv-stack">
          {keys.map((k) => <Frame key={k} tier={k} hidden={k !== active} />)}
        </div>
      </div>
    );
  }
  if (stacked && compare === "swipe") {
    const [a, b] = keys;
    const pct = Math.round(swipe * 1000) / 10;
    const drag = (e: React.PointerEvent<HTMLDivElement>) => {
      const box = (e.currentTarget.parentElement as HTMLElement).getBoundingClientRect();
      if (!(box.width > 0)) return;
      ctrl.setPanels({ swipe: Math.max(0, Math.min(1, (e.clientX - box.left) / box.width)) });
    };
    return (
      <div ref={ref} className={`cv-frames${zoomed ? " cv-frames--zoomed" : ""}`}>
        <div className="cv-stack">
          <Frame tier={a} />
          <Frame tier={b} clip={`inset(0 0 0 ${pct}%)`} labelRight />
          <div className="cv-swipe" style={{ left: `${pct}%` } as CSSProperties} role="slider" tabIndex={0}
            aria-label="Swipe position" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(pct)}
            onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); e.stopPropagation(); }}
            onPointerMove={(e) => { if (e.currentTarget.hasPointerCapture(e.pointerId)) { e.stopPropagation(); drag(e); } }}
            onKeyDown={(e) => {
              if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
                e.preventDefault(); e.stopPropagation();
                ctrl.setPanels({ swipe: Math.max(0, Math.min(1, swipe + (e.key === "ArrowLeft" ? -0.05 : 0.05))) });
              }
            }} />
        </div>
      </div>
    );
  }
  return (
    <div ref={ref} className={`cv-frames${keys.length > 1 ? " cv-multi" : ""}${zoomed ? " cv-frames--zoomed" : ""}`}>
      {keys.map((k) => <Frame key={k} tier={k} />)}
    </div>
  );
}
