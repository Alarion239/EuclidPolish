/* Pure helpers of the Notebook workspace (no React): the notebook's entry
 * order and day outline, the backups filter and the stored-copy paths.
 * Unit-tested in model.test.ts. */
import type { Backups, Commit } from "./api";

/** Where a "Log to notebook" button lands (shared with every workspace). */
export { notebookEntryUrl } from "../shared/noteText";

/* ── tracking notebook ────────────────────────────────────────────────────── */

/** log.md with its entries (`## ` headings) newest first — the preamble
 *  (before the first entry) stays on top — or as written. */
export function notebookOrder(text: string, newestFirst: boolean): string {
  if (!newestFirst) return text;
  const parts = text.split(/\n(?=## )/);
  return parts.length > 1 ? [parts[0], ...parts.slice(1).reverse()].join("\n") : text;
}

export type NotebookDay = { day: string; label: string; month: string; count: number; id: string };

const ENTRY_TIME = /^(\d{4})-(\d{2})-(\d{2})T\d{2}:\d{2}/;
const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

/** The notebook's dated entries (`## <ISO time>` headings) by UTC day, in the
 *  order they are shown; `id` is the day's first shown entry (the jump
 *  target). Headings that are not a timestamp are left out. */
export function notebookDays(heads: readonly { level: number; text: string; id: string }[]): NotebookDay[] {
  const out: NotebookDay[] = [];
  const byDay = new Map<string, NotebookDay>();
  for (const h of heads) {
    const m = h.level === 2 ? ENTRY_TIME.exec(h.text.trim()) : null;
    if (!m) continue;
    const day = `${m[1]}-${m[2]}-${m[3]}`;
    const seen = byDay.get(day);
    if (seen) { seen.count += 1; continue; }
    const month = MONTHS[Number(m[2]) - 1] ?? m[2];
    const entry = { day, label: `${month.slice(0, 3)} ${Number(m[3])}`, month: `${month} ${m[1]}`, count: 1, id: h.id };
    byDay.set(day, entry);
    out.push(entry);
  }
  return out;
}

/* ── backups ──────────────────────────────────────────────────────────────── */

export type Show = "models" | "fits" | "images" | "campaigns";
export const SHOWS: Show[] = ["models", "fits", "images", "campaigns"];
export const SHOW_LABEL: Record<Show, string> = { models: "Models", fits: "FITS", images: "Images", campaigns: "Archived campaigns" };

/** The Backups filter from `?show=`, else the old tracking page's `?bk=`
 *  (models | fits | images), else models. */
export function parseShow(show: string, bk: string): Show {
  if ((SHOWS as string[]).includes(show)) return show as Show;
  if ((SHOWS as string[]).includes(bk) && bk !== "campaigns") return bk as Show;
  return "models";
}

export function backupCounts(b: Pick<Backups, "models" | "fits" | "images"> | { models: unknown[]; fits: unknown[]; images: unknown[] } | null | undefined,
  archived: number): Record<Show, number> {
  return { models: b?.models.length ?? 0, fits: b?.fits.length ?? 0, images: b?.images.length ?? 0, campaigns: archived };
}

/** `tracking/<current | archive/<dir>>/<kind>/<name>`: the stored copy Files opens. */
export function storedPath(trackingDir: string | undefined, campaignDir: string, kind: Exclude<Show, "campaigns">, name: string): string {
  const root = trackingDir && !/\/tracking\/?$/.test(trackingDir) ? trackingDir : "tracking";
  const sub = campaignDir === "current" ? "current" : `archive/${campaignDir}`;
  return `${root}/${sub}/${kind}/${name}`;
}

export const commitText = (c?: Commit | string | null): string =>
  !c ? "—" : typeof c === "string" ? c : c.short ?? c.hash?.slice(0, 7) ?? "—";
