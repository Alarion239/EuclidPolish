/* Small pure pieces of the tracking-notebook notes every workspace builds
 * (ensemble/notes.ts, home/homeModel.ts). */

/** An ISO time as "2026-09-25 23:32 UTC" (the raw text when unparseable). */
export function utcText(iso: string | null | undefined): string {
  if (!iso) return "—";
  const t = new Date(iso);
  return Number.isNaN(t.getTime()) ? iso : `${t.toISOString().slice(0, 16).replace("T", " ")} UTC`;
}
