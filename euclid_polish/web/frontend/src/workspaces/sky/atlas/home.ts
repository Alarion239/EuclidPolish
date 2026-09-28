/* Where the atlas opens when the URL names no position (console
 * regrouping): framed on the last real tile you inspected (remembered in
 * this browser), else on EDF-N, the Q1 deep field that holds the NEXUS
 * mosaic, the poster galaxy and the legacy real field. The other deep fields
 * are one quick jump away. Not the all-sky Mollweide: it showed no data. */
import { readStorage, writeStorage } from "../../../state/storage";
import { featureView } from "./urlState";

export const LAST_TILE_KEY = "ep.sky.lastTile";

export type HomeView = { ra: number; dec: number; fov: number; ref?: string };

/** EDF-N (the deep field with nearly every real tile), as its quick jump frames it. */
export const Q1_HOME: HomeView = { ra: 269.733, dec: 66.018, fov: 14 };

const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

/** The remembered tile view, or null when missing or malformed. */
export function parseLastTile(raw: string | null): HomeView | null {
  if (!raw) return null;
  let v: unknown;
  try { v = JSON.parse(raw); } catch { return null; }
  if (!v || typeof v !== "object") return null;
  const { ra, dec, fov, ref } = v as Record<string, unknown>;
  if (!finite(ra) || !finite(dec) || !finite(fov)) return null;
  if (ra < 0 || ra >= 360 || Math.abs(dec) > 90 || fov <= 0 || fov > 360) return null;
  return { ra, dec, fov, ...(typeof ref === "string" && ref ? { ref } : {}) };
}

/** Remember a real tile the inspector showed (its centre, framed a few times its size). */
export function rememberTile(ref: string, ra: number | null | undefined, dec: number | null | undefined, sizeDeg: number): void {
  if (!finite(ra) || !finite(dec)) return;
  const v = featureView({ ra, dec, sizeDeg });
  writeStorage(LAST_TILE_KEY, JSON.stringify({ ...v, ref }));
}

/** The atlas's opening view without URL coordinates. */
export function atlasHome(): HomeView {
  return parseLastTile(readStorage(LAST_TILE_KEY)) ?? Q1_HOME;
}
