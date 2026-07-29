/* React wrappers around the reused imperative browser modules that already ship
   with the classic UI (served at /static/*.js). We do NOT reimplement the
   canvas cutout engine — we mount it and manage its lifecycle from React. */
import { useEffect, useRef } from "react";
import "./viewer.css";

type ViewerState = {
  index: number; tier: string; tiers: string[]; color: string;
  layout: "one-row" | "two-rows";
  knee: number; gain: number;
  transfers: Record<string, { knee: number; gain: number }>;
  params: Record<string, string>;
};
export type ViewerApi = {
  goTo(i: number): void;
  setTiers(keys: string | string[]): void;
  /** Apply the shared browser-side colour/asinh/brightness transfer. */
  setView(patch: { color?: string; knee?: number; gain?: number }): void;
  /** Patch cube-query parameters and refresh the visible pixels in place. */
  setParams(patch: Record<string, string>): Promise<void>;
  setMorphMembers(csv: string | null): void;
  getIndex(): number;
  isReady(): boolean;
  getState(): ViewerState;
  reload(): Promise<void>;
  destroy(): void;
};
/** A raw N-band stamp for the shared colour renderer. */
export type CubeRec = { data: Float32Array; h: number; w: number; c: number };
/** The field viewer's `color` meta block (band constants + rgb scheme). */
export type ColorMeta = {
  band_names: string[];
  bands: Record<string, unknown>;
  rgb_scheme: string[];
  default_asinh?: number;
};
export type RenderOpts = { color: string; knee: number; gain: number; K0: number };
type ViewerModule = {
  mountCutoutViewer(root: HTMLElement, opts: Record<string, unknown>): ViewerApi;
  /** Render one cube to ImageData with the viewer's exact colour pipeline —
   *  reused by the ensemble back-trace stamps for viewer-parity colour. */
  renderCubeImageData(rec: CubeRec, colorMeta: ColorMeta, opts: RenderOpts): ImageData;
};

let viewerMod: Promise<ViewerModule> | null = null;
function loadViewer(): Promise<ViewerModule> {
  // Non-literal specifier: loaded at runtime from Flask's /static, NOT bundled
  // by Vite and not statically resolved by TS (the engine ships with classic UI).
  // Bump this token whenever the standalone engine changes: the browser caches
  // dynamic module imports by URL, independently of the rebuilt Vite bundle.
  const url = "/static/cutout_viewer.js?v=20260729-nexus-layout";
  if (!viewerMod) viewerMod = import(/* @vite-ignore */ url) as Promise<ViewerModule>;
  return viewerMod;
}

/** Load the shared colour engine (same module as the viewer). Lets a non-viewer
 *  surface — the back-trace stamps — render byte-identically to the field
 *  viewer's colour / knee / brightness. */
export function loadColorEngine(): Promise<ViewerModule["renderCubeImageData"]> {
  return loadViewer().then((m) => m.renderCubeImageData);
}

export type CutoutViewerProps = {
  collection: string;
  params?: Record<string, string>;
  compact?: boolean;
  hideToolbar?: boolean;
  initialTier?: string;
  initialTiers?: string[];
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
  { collection, params, compact, hideToolbar, initialTier, initialTiers, initialIndex,
    onChange, onReady, className }: CutoutViewerProps,
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
        hideToolbar,
        initialTier,
        initialTiers,
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
  }, [collection, paramsKey, compact, hideToolbar, initialTier, initialIndex]);

  return <div ref={host} className={`cv-root ${className ?? ""}`} />;
}
