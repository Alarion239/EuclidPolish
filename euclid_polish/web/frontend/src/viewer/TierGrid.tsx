/* The frame grid: one square frame per selected tier (canonical order, then
 * residual tiers), edge to edge with 2 px gaps on the light table. The side
 * comes from fit.ts: the largest square that fits the grid's width per column
 * AND the height the viewer has — the stage viewport (the page's scroll box,
 * else the window; the focus-mode surround in focus mode) below the viewer's
 * own top (the tab strip, toolbars and banner above it: `heightUnderTop`),
 * minus the viewer's own chrome (every row of the light table but the frames:
 * the bar, the Display row, the readout; in focus mode also the profile
 * panel) and a small margin. Content above the viewer that grows, wraps or
 * goes (a banner dismissed) refits it. The layout (auto / one row / grid /
 * stack) picks the column count. Blink stacks every frame in one cell and
 * cycles them; swipe overlays the first two with a draggable divider over
 * the drawn image. */
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

const outerHeight = (el: Element | null | undefined) => {
  if (!(el instanceof HTMLElement)) return 0;
  const cs = getComputedStyle(el);
  return el.offsetHeight + (parseFloat(cs.marginTop) || 0) + (parseFloat(cs.marginBottom) || 0);
};

/** The light table's height less its frames: every row but `.cv-body`. */
function tableChrome(table: HTMLElement | null | undefined): number {
  if (!table) return 0;
  let h = 0;
  for (const el of Array.from(table.children)) if (!el.classList.contains("cv-body")) h += outerHeight(el);
  return h;
}

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
    const table = grid.closest<HTMLElement>(".cv-table");
    const stage = focus ? null : scrollParent(root);
    const fit = () => {
      // In focus mode the profile panel sits under the table, inside the surround.
      const panels = focus ? outerHeight(root?.querySelector<HTMLElement>(":scope > .cv-panels")) : 0;
      const chrome = tableChrome(table) + panels;
      let viewport: number;
      let lead = 0;
      if (focus && root) {
        const cs = getComputedStyle(root);
        viewport = root.clientHeight - (parseFloat(cs.paddingTop) || 0) - (parseFloat(cs.paddingBottom) || 0);
      } else {
        viewport = stage ? stage.clientHeight : (window.visualViewport?.height ?? window.innerHeight);
        // How far below the top of its scroll box's content the viewer starts
        // (at scroll 0): viewerRoot.top − stage.top + stage.scrollTop.
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
    // it. The cheap watch: a ResizeObserver on each element that precedes the
    // viewer or one of its ancestors (up to the scroll box) — a banner
    // dismissed, a caption or toolbar that wraps — and a childList observer
    // on each ancestor, for a row INSERTED above (the version banner
    // arriving, a caption once the data loads). Not the subtree: the
    // readout's text under the pointer never reaches it. An observed element
    // that leaves the DOM reports once more (size 0). The fit reads only
    // positions above the frames, so refitting settles in one pass.
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
    // the table's own rows: the bar (its rows), the Display row, the readout (its lines)
    if (table) for (const el of Array.from(table.children)) if (!el.classList.contains("cv-body")) ro?.observe(el);
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
  const fitSide = useViewer((s) => s.fit?.side ?? 0);
  useViewer((s) => s.shown);   // the drawn rectangle follows the shown cubes
  useViewer((s) => s.pixelExact);
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
    // The divider moves over the drawn image (the snapped whole image, or
    // the zoomed view filling the frame), not the surround around it.
    const L = fitSide > 0 ? ctrl.layoutOf(a, fitSide) : null;
    const x0 = L ? Math.max(0, L.dx) : 0;
    const w = L ? Math.min(fitSide, L.dx + L.dw) - x0 : 0;
    const at = (f: number) => (w > 0 ? `${x0 + f * w}px` : `${f * 100}%`);
    const pct = Math.round(swipe * 1000) / 10;
    const drag = (e: React.PointerEvent<HTMLDivElement>) => {
      const box = (e.currentTarget.parentElement as HTMLElement).getBoundingClientRect();
      if (!(box.width > 0)) return;
      const f = w > 0 ? (e.clientX - box.left - x0) / w : (e.clientX - box.left) / box.width;
      ctrl.setPanels({ swipe: Math.max(0, Math.min(1, f)) });
    };
    return (
      <div ref={ref} className={`cv-frames${zoomed ? " cv-frames--zoomed" : ""}`}>
        <div className="cv-stack">
          <Frame tier={a} />
          <Frame tier={b} clip={`inset(0 0 0 ${at(swipe)})`} labelRight />
          <div className="cv-swipe" style={{ left: at(swipe) } as CSSProperties} role="slider" tabIndex={0}
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
