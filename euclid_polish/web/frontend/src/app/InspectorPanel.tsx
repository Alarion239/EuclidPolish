/* The right-hand Inspector panel (spec §4): renders the current
 * `{kind, id}` target through the registry (app/inspector.ts), with
 * back/forward, pin, copy-link and close. The shell docks it in a resizable
 * panel (width persisted in prefs) on wide screens and shows it as a sheet
 * below 900 px. Escape closes it from anywhere — the page, a field, the
 * inspector — unless a dialog, popover or menu is open or something nearer
 * takes the key first (the viewer leaving focus mode or unfreezing its lens,
 * a zoomed chart, a table clearing its selection: they preventDefault).
 * Opening it moves focus into it (the panel); closing it hands focus back to
 * what opened it (or to the stage). Switching targets while it is open leaves
 * focus where it is (a table's arrow keys keep working). */
import { Suspense, useEffect, useLayoutEffect, useRef, type RefObject } from "react";
import { useLocation } from "react-router-dom";
import { useShortcut } from "../hooks/useShortcut";
import { formatInspectParam, sameTarget, useInspector, type InspectTarget } from "../state/inspector";
import { Callout, Chip, CopyButton, IconButton, JsonTree, Skeleton } from "../ui";
import { ErrorBoundary } from "./ErrorBoundary";
import { inspectHref, inspectorTitle, openInspector, useInspectorKind, useInspectorRegistry } from "./inspector";
import { useTokenRerender } from "./workspace";

function UnknownKind({ target }: { target: InspectTarget }) {
  return (
    <div className="insp-unknown">
      <Callout tone="info" title={`No inspector for “${target.kind}”`}>
        Every workspace registers its inspectors at start and none handles this kind, so the
        link is probably mistyped or from another version of the console. Its target is below.
      </Callout>
      <JsonTree data={target} expandDepth={1} />
    </div>
  );
}

function Pins({ current }: { current: InspectTarget }) {
  const pinned = useInspector((s) => s.pinned);
  useInspectorRegistry((s) => s.kinds); // re-title when kinds register
  if (!pinned.length) return null;
  return (
    <nav className="inspector__pins" aria-label="Pinned">
      {pinned.map((p) => (
        <Chip key={formatInspectParam(p)} on={sameTarget(p, current)} onClick={() => openInspector(p)}
          title={formatInspectParam(p)}>
          {inspectorTitle(p)}
        </Chip>
      ))}
    </nav>
  );
}

/** "tile" → "Tile", "realtile" → "Realtile": the kind above the title, in
 *  sentence case (it is a label, not data). */
export function kindLabel(kind: string): string {
  const k = kind.replace(/[_-]+/g, " ").trim();
  return k ? k[0].toUpperCase() + k.slice(1) : kind;
}

const EDITABLE = "input, textarea, select, [contenteditable]:not([contenteditable='false'])";
/** An open layer that owns Escape: a dialog (the palette, confirm, the
 *  Display sheet, a popover), a menu or a listbox. */
const LAYER = "[role='dialog'], [role='alertdialog'], [role='menu'], [role='listbox']";

/** Whether Escape belongs to a layer other than the docked inspector. */
export function escapeTakenByLayer(event: KeyboardEvent, panel: HTMLElement | null): boolean {
  const t = event.target;
  const inLayer = t instanceof Element ? t.closest(LAYER) : null;
  if (inLayer && !(panel && inLayer.contains(panel))) return true;
  return [...document.querySelectorAll(LAYER)].some((el) => !(panel && (el.contains(panel) || panel.contains(el))));
}

/** Remember what had focus (outside the panel) when a target opened; move
 *  focus into the panel when it opens (not when the target changes while it
 *  is open, and not away from a field being typed in or from a dialog); give
 *  focus back to the opener when the panel closes while focus is inside it —
 *  otherwise focus would drop to <body>. Falls back to the stage. Inside the
 *  narrow-screen sheet (a Radix dialog) the dialog moves and restores focus. */
function useReturnFocus(panel: RefObject<HTMLElement>, target: string | null) {
  const opener = useRef<HTMLElement | null>(null);
  useEffect(() => {
    if (!target) return;
    const active = document.activeElement;
    if (active instanceof HTMLElement && active !== document.body && !panel.current?.contains(active)) {
      opener.current = active;
    }
  }, [target, panel]);
  const open = target != null;
  useEffect(() => {
    const el = panel.current;
    if (!open || !el || el.closest("[role='dialog']")) return;
    const active = document.activeElement;
    if (active instanceof Element && (active.matches(EDITABLE) || active.closest(LAYER))) return;
    if (el.contains(active)) return;
    el.focus({ preventScroll: true });
  }, [open, panel]);
  // A layout effect: its cleanup runs while the panel is still in the DOM.
  // Focus goes back only once the panel has really left the page (not on
  // StrictMode's simulated unmount, which keeps the element connected).
  useLayoutEffect(() => {
    if (!open) return undefined;
    const el = panel.current;
    return () => {
      if (!el || !el.contains(document.activeElement)) return;
      const back = opener.current;
      queueMicrotask(() => {
        if (el.isConnected) return;
        const to = back?.isConnected ? back : document.getElementById("main");
        to?.focus({ preventScroll: true });
      });
    };
  }, [open, panel]);
}

export function InspectorPanel() {
  const current = useInspector((s) => s.current);
  const canBack = useInspector((s) => s.back.length > 0);
  const canForward = useInspector((s) => s.forward.length > 0);
  const pinned = useInspector((s) => (s.current ? s.isPinned(s.current) : false));
  const reg = useInspectorKind(current?.kind);
  useTokenRerender(); // a theme flip repaints the inspected entity (token reads in render)
  const location = useLocation();
  const ref = useRef<HTMLElement>(null);
  const store = useInspector.getState;
  // Window-level: runs after the viewer (window capture) and anything else
  // that uses Esc and preventDefaults it; works from fields too; an open
  // dialog, popover or menu keeps the key (it closes first).
  useShortcut("Escape", (e) => {
    if (escapeTakenByLayer(e, ref.current)) return false;
    store().hide();
  }, {
    description: "Close the inspector", scope: "Inspector", hidden: true, allowInInputs: true,
    enabled: current != null,
  });
  useReturnFocus(ref, current ? formatInspectParam(current) : null);
  if (!current) return null;
  const title = inspectorTitle(current);
  const link = () => `${window.location.origin}${inspectHref(current, location)}`;
  return (
    <aside className="inspector" aria-label="Inspector" ref={ref} tabIndex={-1}>
      <header className="inspector__head">
        <IconButton icon="chevronLeft" size="sm" label="Back" disabled={!canBack} onClick={() => store().goBack()} />
        <IconButton icon="chevronRight" size="sm" label="Forward" disabled={!canForward} onClick={() => store().goForward()} />
        <div className="inspector__title">
          <span className="inspector__kind">{kindLabel(current.kind)}</span>
          <h2 title={title}>{title}</h2>
        </div>
        <IconButton icon="pin" size="sm" label={pinned ? "Unpin" : "Pin"} pressed={pinned}
          onClick={() => store().togglePin()} />
        <CopyButton value={link} label="Copy link" />
        <IconButton icon="close" size="sm" label="Close inspector" onClick={() => store().hide()} />
      </header>
      <Pins current={current} />
      <div className="inspector__body">
        <ErrorBoundary resetKey={formatInspectParam(current)} label="The inspector">
          <Suspense fallback={<Skeleton lines={5} />}>
            {reg ? <reg.Component key={formatInspectParam(current)} id={current.id} /> : <UnknownKind target={current} />}
          </Suspense>
        </ErrorBoundary>
      </div>
    </aside>
  );
}
