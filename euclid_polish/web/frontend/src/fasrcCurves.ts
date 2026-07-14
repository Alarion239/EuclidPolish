export type CurveRec = {
  step: number; psnr_stretched?: number | null; psnr_raw?: number | null;
  loss?: number | null; psnr_vis?: number | null; psnr_y_e?: number | null;
  psnr_j_e?: number | null; psnr_h_e?: number | null;
  member?: number | string | null;
  total?: number | null;
};

/** Keep only chartable Reporter metrics and repair old member-local x axes.
 *
 * Older lens-isolation jobs tagged each sample with `member`, but restarted
 * `step` at zero for every member.  When a member transition rewinds the raw
 * step, offset the next segment by the previous member's declared total.  New
 * jobs and production ensemble jobs already emit monotonically cumulative
 * steps, so they pass through unchanged.
 */
export function curveRecords(value: unknown): CurveRec[] {
  if (!Array.isArray(value)) return [];
  const records = value.filter((item): item is CurveRec => {
    if (item == null || typeof item !== "object") return false;
    const step = (item as { step?: unknown }).step;
    return typeof step === "number" && Number.isFinite(step);
  });
  let offset = 0;
  let previousRaw = -Infinity;
  let previousMember: string | null = null;
  let previousTotal = 0;
  return records.map((record) => {
    const member = record.member == null ? null : String(record.member);
    const raw = record.step;
    if (member != null && previousMember != null
        && member !== previousMember && raw <= previousRaw) {
      offset += previousTotal > 0 ? previousTotal : Math.max(0, previousRaw);
    }
    const normalized = offset ? { ...record, step: offset + raw } : record;
    previousRaw = raw;
    previousMember = member;
    previousTotal = typeof record.total === "number" && Number.isFinite(record.total)
      ? Math.max(0, record.total) : 0;
    return normalized;
  });
}
