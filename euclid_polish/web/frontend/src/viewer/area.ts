/* Per-unit-area display normalisation (pure).
 *
 * The linked stretch is in e⁻ per pixel. An HR / SR pixel (0.05″) collects
 * about a quarter of the flux of an LR pixel (0.1″) of the same surface
 * brightness, so under one knee the HR and SR frames looked nearly empty
 * next to LR even when their integrated magnitudes matched. With
 * "Match surface brightness" on (the default), every e⁻ frame is DISPLAYED
 * as if it had the coarsest shown pixel: its values × (ref / pixscale)²
 * before the stretch (knee, black point and white reference ÷ that factor —
 * the same image). The readout, magnitudes, histogram values and exports
 * stay in native e⁻ per pixel; the knee reads "e⁻ per <ref>″ pixel". Tiers
 * already in a surface-brightness unit (JWST MJy/sr) and collections with a
 * single pixel scale are untouched. */

export const PER_AREA_STORAGE_KEY = "euclid-polish.viewer.per-area";

type AreaRec = { pixscale?: number | null; unit?: string | null };

const isElectrons = (unit: string | null | undefined) => {
  const u = String(unit ?? "").trim().toLowerCase();
  return u === "" || u === "e-" || u === "e⁻" || u === "electron" || u === "electrons";
};

/** The reference pixel scale (″): the coarsest of the shown e⁻ frames, or 0
 *  when they share one scale (nothing to normalise). */
export function areaReference(recs: readonly AreaRec[]): number {
  const scales = new Set<number>();
  for (const r of recs) {
    const p = Number(r.pixscale);
    if (isElectrons(r.unit) && p > 0 && Number.isFinite(p)) scales.add(Math.round(p * 1e6) / 1e6);
  }
  return scales.size > 1 ? Math.max(...scales) : 0;
}

/** The display factor of one frame: (ref / pixscale)², 1 when off, without a
 *  reference, or for a non-e⁻ tier. */
export function areaFactor(rec: AreaRec, ref: number, on: boolean): number {
  const p = Number(rec.pixscale);
  if (!on || !(ref > 0) || !(p > 0) || !isElectrons(rec.unit)) return 1;
  const f = (ref / p) ** 2;
  return Number.isFinite(f) && f > 0 ? f : 1;
}

/** The saved preference ("0" = off; anything else, or nothing, = on). */
export function parsePerArea(raw: string | null | undefined): boolean {
  return raw !== "0";
}
