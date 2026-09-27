/* <FitBox>: the page-side fit for an image viewer below other content (a file
 * bar, a card head, filters). See fitMath.ts for why and for the maths.
 *
 * The box is its own scroll box (`overflow-y: auto`), so the viewer's fit
 * measures IT — and its height is what is left of the page's scroll box (the
 * stage, or the inspector body) below the box's top. The frames then end at
 * the bottom of the stage and the readout stays in sight. Height the viewer
 * does not use (width-limited frames) is given back to the page as a
 * negative bottom margin, so nothing below moves down for it.
 *
 * Exceptions, where the box steps aside (plain flow, the viewer's own fit):
 * the viewer's profile panel open under the table (it is meant to be
 * scrolled to), and before the first measurement. Focus mode is unaffected:
 * the viewer lifts itself out of the page. If the viewer ever subtracts its
 * own offset, the box still works (the viewer's offset inside it is 0). */
import { useLayoutEffect, useRef, type ReactNode } from "react";
import { fitBoxHeight, fitBoxSlack } from "./fitMath";
import "./fitbox-layout.css";

function scrollParent(el: HTMLElement | null): HTMLElement | null {
  for (let p = el?.parentElement ?? null; p; p = p.parentElement) {
    const oy = getComputedStyle(p).overflowY;
    if ((oy === "auto" || oy === "scroll") && p.clientHeight > 0) return p;
  }
  return null;
}

export function FitBox({ children, className, min = 360, gap = 0, label }: {
  children: ReactNode; className?: string;
  /** The floor for a viewer far down the page (see fitBoxHeight). */
  min?: number;
  /** Space kept free under the box. */
  gap?: number;
  /** An accessible name: the box is then a labelled region. */
  label?: string;
}) {
  const box = useRef<HTMLDivElement>(null);
  const inner = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const el = box.current;
    const content = inner.current;
    if (!el || !content) return undefined;
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(() => later()) : null;
    let stage: HTMLElement | null = null;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const watched = new Set<Element>();
    const watch = (node: Element) => { if (!watched.has(node)) { watched.add(node); ro?.observe(node); } };
    const apply = () => {
      timer = null;
      // The scroll box is found on every pass: a card that mounted while its
      // panel was still opening (0 px tall) would otherwise bind to the page.
      const sp = scrollParent(el);
      if (sp !== stage) { stage = sp; if (sp) watch(sp); }
      // Content above the box growing or shrinking moves it: watch what
      // precedes the box or an ancestor, up to the stage (the ancestors
      // themselves often keep a min-height, so they do not resize).
      for (let node: HTMLElement | null = el; node && node !== stage; node = node.parentElement) {
        for (let sib = node.previousElementSibling; sib; sib = sib.previousElementSibling) watch(sib);
      }
      // The profile panel sits under the table outside focus mode: let the page scroll to it.
      const off = !!content.querySelector(".cv-root:not([data-focus]) > .cv-panels");
      el.classList.toggle("fitbox--off", off);
      if (off) { el.style.height = ""; el.style.marginBottom = ""; return; }
      const top = stage ? stage.getBoundingClientRect().top + stage.clientTop : 0;
      const offset = el.getBoundingClientRect().top - top + (stage ? stage.scrollTop : window.scrollY);
      const h = fitBoxHeight({ stage: stage ? stage.clientHeight : window.innerHeight, offset, min, gap });
      if (!(h > 0)) return;
      if (el.style.height !== `${h}px`) el.style.height = `${h}px`;
      const slack = fitBoxSlack(h, content.offsetHeight);
      el.style.marginBottom = slack ? `-${slack}px` : "";
    };
    // A timer, not requestAnimationFrame: rAF never runs in a hidden tab, and
    // the box must be right when the tab comes back. Coalesced per task.
    function later() { if (timer == null) timer = setTimeout(apply, 0); }
    apply();
    watch(content);                               // the viewer's own height (slack)
    // Content inserted or removed around the box (a callout, a loaded list,
    // the profile panel) moves or changes it; the viewer's own updates inside
    // the box (readout text) are ignored unless they add or drop the panel.
    const isPanels = (n: Node) => n instanceof HTMLElement && n.classList.contains("cv-panels");
    const mo = typeof MutationObserver !== "undefined"
      ? new MutationObserver((records) => {
        if (records.some((r) => !el.contains(r.target)
          || Array.from(r.addedNodes).some(isPanels) || Array.from(r.removedNodes).some(isPanels))) later();
      })
      : null;
    mo?.observe(document.body, { childList: true, subtree: true });
    window.addEventListener("resize", later);
    return () => {
      if (timer != null) clearTimeout(timer);
      ro?.disconnect();
      mo?.disconnect();
      window.removeEventListener("resize", later);
    };
  }, [min, gap]);
  return (
    <div ref={box} className={`fitbox${className ? ` ${className}` : ""}`}
      {...(label ? { role: "region", "aria-label": label } : {})}>
      <div ref={inner} className="fitbox__inner">{children}</div>
    </div>
  );
}
