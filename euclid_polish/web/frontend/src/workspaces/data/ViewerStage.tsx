/* <ViewerStage> — the image-first slot of a Data page: the viewer first,
 * with its frames fitted to the height under its own top, and an optional
 * panel (a gallery, a cluster list) beside it when there is room, else below.
 *
 *   <ViewerStage layout={state?.layout} frames={2} aside={<Gallery/>} asideLabel="Cached cutouts">
 *     <ImageViewer … />
 *   </ViewerStage>
 *
 * The viewer fits its frames to the whole stage height (viewer/fit.ts); the
 * rows above it (tab strip, toolbar, caption) would push a height-limited
 * frame row below the fold. This measures the viewer's own DOM (`.cv-table`,
 * `.cv-bar`, `.cv-frames`, `.cv-readout`) in its stage (the nearest scrolling
 * ancestor, as the viewer does) and applies `frameFit` (viewerFit.ts) to the
 * frame grid only — the viewer stays full width, so its bar keeps the rows it
 * has and the smaller frames centre on the light table. With a side panel,
 * `besideWidth` narrows the viewer to its frames (never below its bar's
 * two-row width) when that leaves the panel `asideMin`. Nothing is measured
 * or applied in focus mode (the viewer is lifted out of the page). */
import { useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import type { LayoutMode } from "../../viewer/fit";
import { besideWidth, frameCount, frameFit, type FrameFit } from "./viewerFit";

/** Gap between the viewer and the side panel (css px; = var(--s3)). */
const ASIDE_GAP = 12;

type Fit = { frames: FrameFit | null; beside: number | null; height: number };
const NO_FIT: Fit = { frames: null, beside: null, height: 0 };

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

/** The bar's width for two unwrapped rows: its widest row's natural width
 *  (the groups side by side, not the spacer) plus its padding. */
function barMinWidth(bar: HTMLElement | null | undefined): number {
  if (!bar) return 0;
  // Measured with the button labels shown: a bar that went icon-only (compact)
  // still asks for its labelled width. The attribute is lifted and restored in
  // the same task, so nothing paints or reaches a ResizeObserver in between.
  const compact = bar.hasAttribute("data-compact");
  if (compact) bar.removeAttribute("data-compact");
  const cs = getComputedStyle(bar);
  let widest = 0;
  for (const row of bar.querySelectorAll<HTMLElement>(":scope > .cv-bar__row")) {
    const kids = [...row.children].filter((k) => !k.classList.contains("cv-bar__spacer"));
    const gap = parseFloat(getComputedStyle(row).columnGap) || 0;
    const w = kids.reduce((s, k) => s + k.getBoundingClientRect().width, 0) + gap * Math.max(0, kids.length - 1);
    widest = Math.max(widest, w);
  }
  const pad = (parseFloat(cs.paddingLeft) || 0) + (parseFloat(cs.paddingRight) || 0);
  if (compact) bar.setAttribute("data-compact", "");
  return widest > 0 ? Math.ceil(widest + pad + 2) : 0;
}

function measure(row: HTMLElement, box: HTMLElement | null, o: { layout: LayoutMode; hint: number; aside: boolean; asideMin: number }): Fit | null {
  const root = box?.querySelector<HTMLElement>(".cv-root");
  if (root?.hasAttribute("data-focus")) return null;               // lifted out: keep the last fit
  const table = box?.querySelector<HTMLElement>(".cv-table");
  const frames = table?.querySelector<HTMLElement>(".cv-frames");
  const stage = scrollParent(row);
  const width = row.clientWidth;
  if (!table || !frames || !stage || !(width > 0)) return NO_FIT;
  const stageRect = stage.getBoundingClientRect();
  const tableTop = table.getBoundingClientRect().top - stageRect.top + stage.scrollTop;
  const n = frameCount({
    frames: frames.querySelectorAll(":scope > .cv-frame").length,
    stacked: !!frames.querySelector(":scope > .cv-stack"),
    loading: !!frames.querySelector(":scope > .cv-frame--message"),
  }, o.hint);
  const bar = table.querySelector<HTMLElement>(":scope > .cv-bar");
  // The Display dock takes height only when it sits under the frames (a narrow
  // viewer); beside them it takes width.
  const body = frames.parentElement;
  const dock = body?.querySelector<HTMLElement>(":scope > .cv-dock");
  const dockBelow = dock && body && getComputedStyle(body).flexDirection === "column" ? outerHeight(dock) : 0;
  const chrome = outerHeight(bar) + outerHeight(table.querySelector<HTMLElement>(":scope > .cv-readout")) + dockBelow;
  const extraWidth = Math.max(0, table.clientWidth - frames.clientWidth);
  const fit = frameFit({ n, mode: o.layout, width: width - extraWidth, stageHeight: stage.clientHeight, tableTop, chrome });
  const beside = o.aside
    ? besideWidth({ rowWidth: width, fit, extraWidth, barMin: barMinWidth(bar), asideMin: o.asideMin, gap: ASIDE_GAP })
    : null;
  return { frames: fit, beside, height: box?.offsetHeight ?? 0 };
}

const sameFit = (a: Fit, b: Fit) => a.beside === b.beside && Math.abs(a.height - b.height) < 1
  && a.frames?.side === b.frames?.side && a.frames?.columns === b.frames?.columns;

/** Where the side panel landed: beside the viewer (as tall as it) or below it. */
export type AsidePlace = { beside: boolean; height: number };

export function ViewerStage({ children, aside, asideLabel, asideMin = 176, asideFill = false, layout = "auto", frames = 1, className }: {
  children: ReactNode;
  /** A panel beside the viewer when the fitted viewer leaves `asideMin` px, else
   *  below it; a function gets where it landed (e.g. to size a table to the viewer). */
  aside?: ReactNode | ((place: AsidePlace) => ReactNode);
  asideLabel?: string;
  asideMin?: number;
  /** Beside the viewer, the panel is exactly as tall as it and its own child
   *  scrolls inside (a table / thumbnail list that fills the column), rather
   *  than the whole panel scrolling. */
  asideFill?: boolean;
  /** The viewer's layout mode (`ViewerState.layout`). */
  layout?: LayoutMode;
  /** The frames the page expects before the viewer has loaded (its `tiers` count). */
  frames?: number;
  className?: string;
}) {
  const rowRef = useRef<HTMLDivElement>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const [fit, setFit] = useState<Fit>(NO_FIT);
  const hasAside = aside != null && aside !== false;

  useLayoutEffect(() => {
    const row = rowRef.current;
    if (!row) return;
    const opts = { layout, hint: frames, aside: hasAside, asideMin };
    let raf = 0;
    const observed = new Set<Element>();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(() => schedule()) : null;
    const watch = (el: Element | null | undefined) => {
      if (el && ro && !observed.has(el)) { ro.observe(el); observed.add(el); }
    };
    const run = () => {
      raf = 0;
      const next = measure(row, boxRef.current, opts);
      // The viewer's table and grid appear once it has mounted / loaded its meta.
      const table = boxRef.current?.querySelector(".cv-table");
      watch(table); watch(table?.querySelector(".cv-frames")); watch(boxRef.current);
      // Rows appearing above the viewer (a toolbar that wraps, the shell's
      // version banner) move it down without resizing it: they resize one of
      // its ancestors up to the stage, so watch those too.
      const stage = scrollParent(row);
      for (let p = row.parentElement; p && p !== stage; p = p.parentElement) watch(p);
      watch(stage);
      if (next) setFit((prev) => (sameFit(prev, next) ? prev : next));
    };
    function schedule() { if (!raf) raf = requestAnimationFrame(run); }
    watch(row);
    run();
    window.addEventListener("resize", schedule);
    // The viewer swaps its placeholder for the table, its frames for a message
    // or a blink stack, and enters / leaves focus mode: re-measure (and
    // re-observe) then — not on the readout / bar text changes under the pointer.
    const structural = (m: MutationRecord) => m.type === "attributes"
      || (m.target instanceof Element && m.target.matches(".dt-vstage__viewer, .cv-host, .cv-root, .cv-table, .cv-body, .cv-frames"));
    const mo = typeof MutationObserver !== "undefined" && boxRef.current
      ? new MutationObserver((records) => { if (records.some(structural)) schedule(); }) : null;
    if (mo && boxRef.current) mo.observe(boxRef.current, { childList: true, subtree: true, attributes: true, attributeFilter: ["data-focus"] });
    return () => {
      if (raf) cancelAnimationFrame(raf);
      ro?.disconnect();
      mo?.disconnect();
      window.removeEventListener("resize", schedule);
    };
  }, [layout, frames, hasAside, asideMin]);

  const beside = fit.beside != null;
  const style = {
    ...(fit.frames ? { "--dt-cols": `${fit.frames.columns}`, "--dt-side": `${fit.frames.side}px` } : {}),
    ...(beside ? { "--dt-vw": `${fit.beside}px` } : {}),
    ...(beside && fit.height > 0 ? { "--dt-vh": `${fit.height}px` } : {}),
  } as CSSProperties;
  return (
    <div ref={rowRef} className={`dt-vstage${className ? ` ${className}` : ""}`} style={style}
      data-fit={fit.frames ? "" : undefined} data-beside={beside || undefined}>
      <div ref={boxRef} className="dt-vstage__viewer">{children}</div>
      {hasAside && (
        <div className="dt-vstage__aside" role="region" aria-label={asideLabel} data-fill={asideFill || undefined}>
          {typeof aside === "function" ? aside({ beside, height: fit.height }) : aside}
        </div>
      )}
    </div>
  );
}
