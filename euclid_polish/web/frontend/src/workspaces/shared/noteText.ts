/* Small pure pieces of the tracking-notebook notes every workspace builds
 * (models/notes.ts, home/homeModel.ts) and where they are sent. */
import { pagePath } from "../../app/nav";

/** An ISO time as "2026-09-25 23:32 UTC" (the raw text when unparseable). */
export function utcText(iso: string | null | undefined): string {
  if (!iso) return "—";
  const t = new Date(iso);
  return Number.isNaN(t.getTime()) ? iso : `${t.toISOString().slice(0, 16).replace("T", " ")} UTC`;
}

/** Where a "Log to notebook" button lands: Notebook › Log with the entry
 *  prefilled (`?entry=<markdown>&from=<page label>`); nothing is appended
 *  until the entry card's Append. */
export function notebookEntryUrl(entry: string, from?: string): string {
  const q = new URLSearchParams({ entry, ...(from ? { from } : {}) });
  return `${pagePath("notebook", { tab: "log" })}?${q.toString()}`;
}
