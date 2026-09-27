/* Point markers drawn over the frames (e.g. the truth sources of a training
 * record on its HR image). Coordinates live on one reference pixel grid with
 * pixel centres at integers (the numpy convention of the source catalogues);
 * every tier maps them onto its own grid by the width/height ratio, and the
 * shared pan/zoom view maps them onto the frame. Pure geometry here; the
 * drawing and pointer handling are in Frame.tsx. */
import { createContext, useContext } from "react";
import { imageToFrame, type FrameLayout } from "./selection";

export type ViewerMarker = {
  key: string;
  /** Position and radius on `ViewerMarkers.grid` (pixel centres at integers). */
  x: number;
  y: number;
  r: number;
  /** Styling hook (`galaxy`, `star`, `lens`, …); a star draws as a cross. */
  kind?: string;
  title?: string;
  /** Drawn faded and dashed (e.g. a source centred outside the frame). */
  dim?: boolean;
};

export type ViewerMarkers = {
  grid: { width: number; height: number };
  items: readonly ViewerMarker[];
  /** Tiers to draw on (default: every tier, scaled to its grid). */
  tiers?: readonly string[];
  activeKey?: string | null;
  onHover?: (key: string | null) => void;
  onPick?: (key: string) => void;
};

export type MarkerShape = {
  key: string; cx: number; cy: number; r: number;
  kind: string; title: string; dim: boolean; active: boolean;
};

export const MarkersContext = createContext<ViewerMarkers | null>(null);
export const useMarkers = (): ViewerMarkers | null => useContext(MarkersContext);

export function markersOnTier(tier: string, markers: ViewerMarkers | null | undefined): boolean {
  if (!markers || !markers.items.length) return false;
  return !markers.tiers || markers.tiers.includes(tier);
}

/** Frame-space shapes (CSS px of a frame of side S) of the markers visible
 *  through layout L on a tier whose image is `geom` pixels. */
export function markerShapes(
  L: FrameLayout, S: number, markers: ViewerMarkers, geom: { width: number; height: number },
): MarkerShape[] {
  const { grid } = markers;
  if (!(grid.width > 0 && grid.height > 0 && geom.width > 0 && geom.height > 0 && L.sw > 0 && L.dw > 0)) return [];
  const sx = geom.width / grid.width, sy = geom.height / grid.height;
  const zoom = L.dw / L.sw;
  const out: MarkerShape[] = [];
  for (const m of markers.items) {
    const p = imageToFrame(L, (m.x + 0.5) * sx, (m.y + 0.5) * sy);
    const r = Math.max(0, m.r * sx * zoom);
    const pad = Math.max(r, 6);
    if (p.x < -pad || p.y < -pad || p.x > S + pad || p.y > S + pad) continue;
    out.push({
      key: m.key, cx: p.x, cy: p.y, r, kind: m.kind ?? "other", title: m.title ?? "",
      dim: !!m.dim, active: markers.activeKey != null && markers.activeKey === m.key,
    });
  }
  return out;
}
