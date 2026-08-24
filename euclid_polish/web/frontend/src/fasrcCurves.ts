export type CurveRec = {
  step: number; psnr_stretched?: number | null; psnr_raw?: number | null;
  loss?: number | null; psnr_vis?: number | null; psnr_y_e?: number | null;
  psnr_j_e?: number | null; psnr_h_e?: number | null;
  member?: number | string | null;
  total?: number | null;
};

/** Keep only chartable Reporter metrics with finite cumulative steps. */
export function curveRecords(value: unknown): CurveRec[] {
  if (!Array.isArray(value)) return [];
  const records = value.filter((item): item is CurveRec => {
    if (item == null || typeof item !== "object") return false;
    const step = (item as { step?: unknown }).step;
    return typeof step === "number" && Number.isFinite(step);
  });
  return records;
}
