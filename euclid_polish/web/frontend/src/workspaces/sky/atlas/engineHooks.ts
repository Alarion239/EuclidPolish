/* Hooks around the one engine: its layer renderer, "fly to" navigation and
 * "overlay these pixels on the sky" (from any page). */
import { useCallback } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import type { SkyEngine } from "../../../sky/engine";
import { useSelection } from "../../../state/selection";
import { toast } from "../../../ui";
import { readSkyPalette } from "./colorScale";
import { IMG_CODEC, withOverlays, type PixelOverlaySetting } from "./pixelOverlays";
import { LayerRenderer } from "./render";
import { experimentsHref, fmtCoord, fmtFovParam, patchSearch } from "./urlState";

let renderer: { engine: SkyEngine; r: LayerRenderer } | null = null;

/** The renderer bound to the engine (it lives as long as the engine does). */
export function rendererFor(engine: SkyEngine): LayerRenderer {
  if (!renderer || renderer.engine !== engine) {
    renderer = {
      engine,
      r: new LayerRenderer(engine, () => readSkyPalette(), (layer, err) => {
        console.error(`sky: could not draw layer "${layer.id}"`, err);
        toast.error(`Could not draw ${layer.label}: ${err instanceof Error ? err.message : String(err)}`);
      }),
    };
  }
  return renderer.r;
}

export const ATLAS_PATH = "/sky/atlas";

/** Centre the atlas on a position (navigates there from any page; one
 *  history entry, so Back returns to the previous view). */
export function useFlyTo(): (v: { ra: number; dec: number; fov?: number }) => void {
  const show = useShowOnSky();
  return useCallback((v) => show({ view: v }), [show]);
}

type View = { ra: number; dec: number; fov?: number };

const viewPatch = (v: View) => ({
  ra: fmtCoord(v.ra), dec: fmtCoord(v.dec), ...(v.fov != null ? { fov: fmtFovParam(v.fov) } : {}), goto: null,
});

/** Add pixel overlays to the atlas's `img` param and/or centre it, from any
 *  page (on the atlas the other params are kept). Moving the view pushes a
 *  history entry; adding overlays alone edits the current one. */
export function useShowOnSky(): (o: { view?: View; overlays?: readonly PixelOverlaySetting[] }) => void {
  const navigate = useNavigate();
  const location = useLocation();
  return useCallback(({ view, overlays }) => {
    const onAtlas = location.pathname === ATLAS_PATH;
    const base = onAtlas ? location.search : "";
    const patch: Record<string, string | null> = view ? viewPatch(view) : {};
    if (overlays?.length) {
      const cur = IMG_CODEC.parse(new URLSearchParams(base).get("img") ?? "") ?? [];
      patch.img = IMG_CODEC.serialize(withOverlays(cur, overlays));
    }
    navigate(`${ATLAS_PATH}${patchSearch(base, patch)}`, { replace: onAtlas && !view });
  }, [navigate, location.pathname, location.search]);
}

/** "Compare models…" handoff to Sky › Compare: exactly these tiles. The
 *  `tile` selection scope is REPLACED (an earlier pick never leaks into a
 *  one-tile comparison) and `?tiles=<ref,…>` carries the same refs, so the
 *  link also works on a fresh load. */
export function useCompareModels(): (refs: readonly string[]) => void {
  const navigate = useNavigate();
  return useCallback((refs) => {
    useSelection.getState().select("tile", refs);
    navigate(experimentsHref(refs));
  }, [navigate]);
}
