/* The frame grid: one square frame per selected tier (canonical order, then
 * residual tiers), sized so every tile is as large as the visible stage
 * allows (ported from the old fitFramesToViewport: the layout control is an
 * exact column count — one row, or ceil(n/2) columns on two rows — with a
 * single column below 760 px). Blink stacks every frame in one cell and
 * cycles them; swipe overlays the first two with a draggable divider. */
import { useLayoutEffect, useRef, type CSSProperties } from "react";
import { Frame } from "./Frame";
import { useController, useViewer } from "./hooks";

function scrollParent(el: HTMLElement | null): HTMLElement | null {
  for (let p = el?.parentElement ?? null; p; p = p.parentElement) {
    const oy = getComputedStyle(p).overflowY;
    if ((oy === "auto" || oy === "scroll") && p.scrollHeight >= p.clientHeight) return p;
  }
  return null;
}

export function TierGrid() {
  const ctrl = useController();
  const tiers = useViewer((s) => s.tiers);
  const residuals = useViewer((s) => s.residuals);
  const meta = useViewer((s) => s.meta);
  const layout = useViewer((s) => s.layout);
  const compare = useViewer((s) => s.compare);
  const blinkAt = useViewer((s) => s.blinkAt);
  const swipe = useViewer((s) => s.swipe);
  const metaError = useViewer((s) => s.metaError);
  const zoomed = useViewer((s) => !!s.view);
  const ref = useRef<HTMLDivElement>(null);
  const keys = ctrl.frameKeys();
  const stacked = compare !== "off" && keys.length > 1;
  const count = stacked ? 1 : keys.length;

  useLayoutEffect(() => {
    const grid = ref.current;
    if (!grid) return;
    const root = grid.closest<HTMLElement>(".cv-root");
    const fit = () => {
      if (!count) { grid.style.removeProperty("--cv-frame-size"); grid.style.removeProperty("--cv-columns"); return; }
      const style = getComputedStyle(grid);
      const columnGap = parseFloat(style.columnGap) || 14;
      const rowGap = parseFloat(style.rowGap) || columnGap;
      const width = grid.clientWidth;
      const stage = scrollParent(root);
      const visible = stage ? stage.clientHeight : (window.visualViewport?.height ?? window.innerHeight);
      let chrome = 16;
      for (const sel of [".cv-toolbar", ".cv-nav", ".cv-readout"]) {
        const el = root?.querySelector<HTMLElement>(sel);
        if (el) {
          const cs = getComputedStyle(el);
          chrome += el.offsetHeight + (parseFloat(cs.marginTop) || 0) + (parseFloat(cs.marginBottom) || 0);
        }
      }
      const columns = window.innerWidth <= 760 ? 1 : layout === "two-rows" ? Math.ceil(count / 2) : count;
      const rows = Math.ceil(count / columns);
      const widthLimit = (width - columnGap * (columns - 1)) / columns;
      const heightLimit = (visible - chrome - rowGap * (rows - 1)) / rows;
      const side = Math.floor(Math.max(160, Math.min(widthLimit, heightLimit)));
      if (side > 0) {
        grid.style.setProperty("--cv-frame-size", `${Math.min(side, Math.floor(widthLimit))}px`);
        grid.style.setProperty("--cv-columns", `${columns}`);
      }
    };
    fit();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(() => requestAnimationFrame(fit)) : null;
    ro?.observe(grid);
    window.addEventListener("resize", fit);
    window.visualViewport?.addEventListener("resize", fit);
    return () => {
      ro?.disconnect();
      window.removeEventListener("resize", fit);
      window.visualViewport?.removeEventListener("resize", fit);
    };
  }, [count, layout]);

  if (metaError) {
    return (
      <div ref={ref} className="cv-frames">
        <div className="cv-frame cv-frame--message"><div className="cv-msg"><span>{metaError}</span></div></div>
      </div>
    );
  }
  if (!meta) {
    return (
      <div ref={ref} className="cv-frames">
        <div className="cv-frame cv-frame--message cv-loading"><div className="cv-msg"><span>Loading…</span></div></div>
      </div>
    );
  }
  void tiers; void residuals;   // (frameKeys reads them; subscribed for re-render)

  if (stacked && compare === "blink") {
    const active = keys[blinkAt % keys.length];
    return (
      <div ref={ref} className={`cv-frames${zoomed ? " cv-frames--zoomed" : ""}`}>
        <div className="cv-stack" aria-live="polite">
          {keys.map((k) => <Frame key={k} tier={k} hidden={k !== active} />)}
          <div className="cv-stack__badge">blink · {ctrl.tierLabel(active)}</div>
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
          <Frame tier={a} label={`${ctrl.tierLabel(a)}  ◀`} />
          <Frame tier={b} clip={`inset(0 0 0 ${pct}%)`} label={`▶  ${ctrl.tierLabel(b)}`} labelRight />
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
