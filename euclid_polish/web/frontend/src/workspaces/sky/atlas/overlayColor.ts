/* Colour of the FITS pixel overlays on the sky (pure).
 *
 * By default an overlay follows the Display panel (C7), like every image
 * viewer: its colormap, stretch and invert, and — for the electron tiers (LR,
 * SR) under an absolute stretch — the Euclid transfer group's black point and
 * white = 30·knee/gain e⁻ (the locked absolute-asinh anchors). Aladin's asinh
 * is not the viewer's exact curve, so the overlay matches in feel and anchors,
 * not bit for bit. JWST (MJy/sr) and the auto stretches use Aladin's own cuts
 * from the data. The Sky section can switch "follow" off and set its own. */
import type { DisplaySettings } from "../../../state/display";
import { transferFor } from "../../../state/display";
import type { BaseColor } from "../../../sky/engine";
import type { OverlayStretch } from "./store";

const CMAP: Record<DisplaySettings["colormap"], string> = {
  gray: "grayscale", viridis: "viridis", magma: "magma", inferno: "inferno", cividis: "cividis", rdbu: "rdbu",
};
const STRETCH: Record<DisplaySettings["stretch"], string> = {
  "asinh-abs": "asinh", linear: "linear", log: "log", sqrt: "sqrt", "asinh-auto": "asinh", zscale: "linear",
};
const ABSOLUTE = new Set<DisplaySettings["stretch"]>(["asinh-abs", "linear", "log", "sqrt"]);

/** White point of the locked absolute transfer, in knees. */
export const WHITE_KNEES = 30;

export const isElectronTier = (tier: string) => tier === "lr" || tier.startsWith("m:");

export function overlayColor(
  tier: string,
  display: Pick<DisplaySettings, "colormap" | "stretch" | "invert" | "groups">,
  own: OverlayStretch & { follow?: boolean },
): BaseColor {
  if (own.follow === false) {
    return {
      colormap: own.colormap, stretch: own.stretch, reversed: false,
      ...(own.minCut != null && own.maxCut != null ? { minCut: own.minCut, maxCut: own.maxCut } : {}),
    };
  }
  const out: BaseColor = {
    colormap: CMAP[display.colormap] ?? "grayscale",
    stretch: STRETCH[display.stretch] ?? "asinh",
    reversed: !!display.invert,
  };
  if (isElectronTier(tier) && ABSOLUTE.has(display.stretch)) {
    const t = transferFor(display, "euclid");
    const white = (WHITE_KNEES * t.knee) / (t.gain > 0 ? t.gain : 1);
    const black = Number.isFinite(t.black) ? t.black : 0;
    if (white > black) { out.minCut = black; out.maxCut = white; }
  }
  return out;
}

/** Stable key of a colour (a change re-applies it to the layer). */
export const colorKey = (c: BaseColor) => JSON.stringify([c.colormap, c.stretch, c.reversed, c.minCut, c.maxCut]);
