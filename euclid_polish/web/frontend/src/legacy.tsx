/* React wrappers around the reused imperative browser modules that already ship
   with the classic UI (served at /static/*.js). We do NOT reimplement the
   canvas cutout engine — we mount it and manage its lifecycle from React. */
import { useEffect, useRef } from "react";
import "./viewer.css";

type ViewerState = {
  index: number; tier: string; tiers: string[]; color: string;
  params: Record<string, string>;
};
export type ViewerApi = {
  goTo(i: number): void;
  setTiers(keys: string | string[]): void;
  setMorphMembers(csv: string | null): void;
  getIndex(): number;
  isReady(): boolean;
  getState(): ViewerState;
  reload(): Promise<void>;
  destroy(): void;
};
type ViewerModule = {
  mountCutoutViewer(root: HTMLElement, opts: Record<string, unknown>): ViewerApi;
};

let viewerMod: Promise<ViewerModule> | null = null;
function loadViewer(): Promise<ViewerModule> {
  // Non-literal specifier: loaded at runtime from Flask's /static, NOT bundled
  // by Vite and not statically resolved by TS (the engine ships with classic UI).
  const url = "/static/cutout_viewer.js";
  if (!viewerMod) viewerMod = import(/* @vite-ignore */ url) as Promise<ViewerModule>;
  return viewerMod;
}

export type CutoutViewerProps = {
  collection: string;
  params?: Record<string, string>;
  compact?: boolean;
  initialIndex?: number;
  onChange?: (s: ViewerState) => void;
  /** Called with the imperative engine handle once mounted (and null on
   *  teardown) — lets a sibling panel drive tiers / the movie subset without
   *  re-mounting the viewer. */
  onReady?: (api: ViewerApi | null) => void;
  className?: string;
};

/** Mount the shared canvas cutout viewer for a `collection`, re-mounting when
 *  the collection or its params change. Cleans up on unmount. */
export function CutoutViewer(
  { collection, params, compact, initialIndex, onChange, onReady, className }: CutoutViewerProps,
) {
  const host = useRef<HTMLDivElement>(null);
  const paramsKey = JSON.stringify(params ?? {});
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;
  const onReadyRef = useRef(onReady);
  onReadyRef.current = onReady;

  useEffect(() => {
    let api: ViewerApi | null = null;
    let dead = false;
    const el = host.current;
    if (!el) return;
    loadViewer().then((mod) => {
      if (dead || !el) return;
      el.innerHTML = "";
      api = mod.mountCutoutViewer(el, {
        collection,
        params: params ?? {},
        compact,
        initialIndex,
        onChange: (s: ViewerState) => onChangeRef.current?.(s),
      });
      onReadyRef.current?.(api);
    });
    return () => {
      dead = true;
      onReadyRef.current?.(null);
      try { api?.destroy(); } catch { /* noop */ }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [collection, paramsKey, compact, initialIndex]);

  return <div ref={host} className={`cv-root ${className ?? ""}`} />;
}
