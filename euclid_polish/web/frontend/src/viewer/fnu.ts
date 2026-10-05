/* JWST on the Euclid f_ν scale (pure; spec 2026-10-03-jwst-fnu-scale-design).
 *
 * A JWST frame in MJy/sr is normally scaled by its own brightest pixels (the
 * served robust display scale) and has its own transfer group, so Euclid and
 * JWST are shown on unrelated brightness scales. With the Display option
 * "JWST on the Euclid scale (f_ν)" (`DisplaySettings.jwstFollowsEuclid`, ON
 * by default) a JWST frame FOLLOWS Euclid instead: its MJy/sr values are
 * converted through the AB system into the electrons the shown Euclid band b
 * would collect in the reference pixel,
 *
 *   φ = E_b(1″) · p_ref² · D_ref,   E_b(1″) = meta.color.bands[b].e_per_mjy_sr_arcsec2
 *
 * (`photometry.mjy_per_sr_to_electrons_factor`, served per band), and it uses
 * the Euclid knee, brightness and black point. φ replaces the served display
 * scale: the controller applies `φ / displayScale` beside the per-area factor
 * (knee, black point and white reference ÷ it), so the prepared frame is not
 * rebuilt. The reference pixel is the coarsest shown Euclid e⁻ frame (with
 * its own display factor D_ref), else `meta.color.lr_pixscale` with D_ref = 1.
 * Readout, histogram values and exports stay native MJy/sr. */
import type { ColorMeta } from "./color";

export type FnuRec = {
  transferGroup?: string;
  unit?: string | null;
  pixscale?: number | null;
  displayScale?: number;
  directRgb?: boolean;
};

/** The f_ν constants the server adds to `meta.color`. */
export type FnuColorMeta = Pick<ColorMeta, "bands" | "lr_pixscale">;

const norm = (unit: string | null | undefined) => String(unit ?? "").trim().toLowerCase().replace(/\s+/g, "");

export const isMjySr = (unit: string | null | undefined): boolean => {
  const u = norm(unit);
  return u === "mjy/sr" || u === "mjysr-1" || u === "mjy/steradian";
};

const isElectrons = (unit: string | null | undefined) => {
  const u = norm(unit);
  return u === "" || u === "e-" || u === "e⁻" || u === "electron" || u === "electrons";
};

/** Whether a frame follows Euclid: the option is on, the frame is a one-image
 *  JWST frame in MJy/sr (not a colour composite) and the collection has a
 *  Euclid transfer group to follow. */
export function fnuFollows(rec: FnuRec | null | undefined, on: boolean, hasEuclid: boolean): boolean {
  return !!rec && on && hasEuclid && rec.transferGroup === "jwst" && !rec.directRgb && isMjySr(rec.unit);
}

/** The Euclid band the JWST frame is translated through: the shown colour
 *  when it is a calibrated Euclid band with an f_ν constant, else VIS (the
 *  colour modes' knee is already VIS-equivalent). */
export function fnuBand(color: string, colorMeta: FnuColorMeta | null | undefined): string {
  const b = colorMeta?.bands?.[color];
  return b && !b.display_only && Number(b.e_per_mjy_sr_arcsec2) > 0 ? color : "VIS";
}

/** The reference pixel: the coarsest shown Euclid e⁻ frame (its pixel scale
 *  and display factor `displayOf(rec)`), else the served LR pixel, factor 1. */
export function fnuReference<R extends FnuRec>(recs: readonly R[], displayOf: (rec: R) => number,
  colorMeta: FnuColorMeta | null | undefined): { pixscale: number; factor: number } {
  let best: R | null = null;
  for (const r of recs) {
    const p = Number(r.pixscale);
    if (r.transferGroup !== "euclid" || !isElectrons(r.unit) || !(p > 0) || !Number.isFinite(p)) continue;
    if (!best || p > Number(best.pixscale)) best = r;
  }
  if (best) {
    const d = displayOf(best);
    return { pixscale: Number(best.pixscale), factor: Number.isFinite(d) && d > 0 ? d : 1 };
  }
  const p = Number(colorMeta?.lr_pixscale);
  return { pixscale: p > 0 && Number.isFinite(p) ? p : 0.1, factor: 1 };
}

/** φ: display units (Euclid e⁻ in the reference pixel) per MJy/sr in band b;
 *  0 when the band has no served constant. */
export function fnuScale(colorMeta: FnuColorMeta | null | undefined, band: string,
  ref: { pixscale: number; factor: number }): number {
  const e = Number(colorMeta?.bands?.[band]?.e_per_mjy_sr_arcsec2);
  const phi = e * ref.pixscale * ref.pixscale * ref.factor;
  return Number.isFinite(phi) && phi > 0 ? phi : 0;
}
