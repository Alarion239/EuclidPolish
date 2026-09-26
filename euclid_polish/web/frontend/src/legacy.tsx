/* Compatibility wrappers for the pre-rework pages. The static
 * /static/cutout_viewer.js engine is gone; `CutoutViewer` is now the bundled
 * viewer engine v2 (<ImageViewer>, src/viewer/) with the old props, and
 * `loadColorEngine` resolves to the same TS colour pipeline. New code imports
 * from "./viewer" directly. */
import { useEffect, useRef } from "react";
import { ImageViewer, renderCubeImageData } from "./viewer";
import type { ViewerApi, ViewerState } from "./viewer";

export type { ViewerApi, ViewerState } from "./viewer";

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
type RenderFn = (rec: CubeRec, colorMeta: ColorMeta, opts: RenderOpts) => ImageData;

const render: RenderFn = (rec, colorMeta, opts) =>
  renderCubeImageData(rec, colorMeta as unknown as Parameters<typeof renderCubeImageData>[1], opts);

/** The viewer's exact colour pipeline for non-viewer surfaces (the ensemble
 *  back-trace stamps). Kept a Promise for the old call sites. */
export function loadColorEngine(): Promise<RenderFn> {
  return Promise.resolve(render);
}

export type CutoutViewerProps = {
  collection: string;
  params?: Record<string, string>;
  /** Image frames only (no toolbar, no navigation). */
  compact?: boolean;
  /** No toolbar (navigation stays). */
  hideToolbar?: boolean;
  initialTier?: string;
  initialTiers?: string[];
  /** Applied at mount; later changes move the viewer (no remount). */
  initialIndex?: number;
  onChange?: (s: ViewerState) => void;
  /** The engine handle once mounted (null on teardown). */
  onReady?: (api: ViewerApi | null) => void;
  className?: string;
  /** URL-state prefix (default: the collection); "" turns URL state off. */
  urlKey?: string;
};

/** The pre-rework viewer props on top of <ImageViewer>. */
export function CutoutViewer(
  { collection, params, compact, hideToolbar, initialTier, initialTiers, initialIndex, onChange, onReady, className, urlKey }: CutoutViewerProps,
) {
  const api = useRef<ViewerApi | null>(null);
  const onReadyRef = useRef(onReady);
  onReadyRef.current = onReady;
  const tiers = initialTiers ?? (initialTier ? [initialTier] : undefined);

  // The JWST carousel feeds the viewer's index back as initialIndex: follow it
  // without remounting (the old wrapper remounted on every step).
  useEffect(() => {
    const a = api.current;
    if (a && a.isReady() && initialIndex != null && a.getIndex() !== initialIndex) a.goTo(initialIndex);
  }, [initialIndex]);

  return (
    <ImageViewer key={(tiers ?? []).join(",")}
      collection={collection} params={params} tiers={tiers} initialIndex={initialIndex}
      urlKey={urlKey === "" ? undefined : (urlKey ?? collection)}
      toolbar={compact || hideToolbar ? "none" : "full"} nav={!compact}
      onState={onChange}
      onReady={(a) => { api.current = a; onReadyRef.current?.(a); }}
      className={className} />
  );
}
