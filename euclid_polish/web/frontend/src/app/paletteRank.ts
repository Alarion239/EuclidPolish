/* Ranking of the ⌘K palette (UX: the page or command the user named comes
 * first, not a job that happens to contain the letters).
 *
 *   rankPalette("git", [{ heading: "Pages", items: pages, bias: 2 }, …])
 *
 * Each entry scores against its label, keywords and hint (0 = no match):
 *   exact label 100 · label prefix 90 · a label word starts with it 80 ·
 *   keyword exact 75 · keyword prefix 65 · label substring 60 · every word of
 *   a multi-word query matches somewhere 50 · keyword substring 45 · hint
 *   word 40 · hint substring 30 · fuzzy (letters in order, from a word
 *   start, 3+ letters) 10.
 * A group's `bias` (a few points) breaks near-ties toward it (pages and
 * commands over the global job launchers). Groups are ordered by their best
 * item, items by score within a group; ties keep the given order. An empty
 * query returns the groups unchanged. Pure, so it is unit-tested. */

export type Rankable = { label: string; hint?: string; keywords?: readonly string[] };
export type RankGroup<T extends Rankable> = { heading: string; items: T[]; bias?: number };

const norm = (text: string) => text.toLowerCase().replace(/\s+/g, " ").trim();
const isWordChar = (c: string | undefined) => !!c && /[\p{L}\p{N}]/u.test(c);

/** Does `q` occur in `text` at the start of a word? */
function atWordStart(text: string, q: string): boolean {
  for (let i = text.indexOf(q); i >= 0; i = text.indexOf(q, i + 1)) {
    if (i === 0 || !isWordChar(text[i - 1])) return true;
  }
  return false;
}

/** The letters of `q` in order in `text`, the first at a word start. */
function fuzzy(text: string, q: string): boolean {
  if (q.length < 3) return false;
  for (let start = 0; start < text.length; start++) {
    if (text[start] !== q[0] || (start > 0 && isWordChar(text[start - 1]))) continue;
    let j = 1;
    for (let i = start + 1; i < text.length && j < q.length; i++) if (text[i] === q[j]) j++;
    if (j === q.length) return true;
  }
  return false;
}

/** How well one query word matches (label/keywords/hint), for multi-word queries. */
function wordMatches(entry: { label: string; hint: string; keywords: string[] }, w: string): boolean {
  return atWordStart(entry.label, w) || entry.keywords.some((k) => k.includes(w)) || atWordStart(entry.hint, w);
}

export function scoreEntry(query: string, entry: Rankable): number {
  const q = norm(query);
  if (!q) return 0;
  const label = norm(entry.label);
  const hint = norm(entry.hint ?? "");
  const keywords = (entry.keywords ?? []).map(norm).filter(Boolean);
  if (label === q) return 100;
  if (label.startsWith(q)) return 90;
  if (atWordStart(label, q)) return 80;
  if (keywords.includes(q)) return 75;
  if (keywords.some((k) => k.startsWith(q))) return 65;
  if (label.includes(q)) return 60;
  const words = q.split(" ");
  if (words.length > 1 && words.every((w) => wordMatches({ label, hint, keywords }, w))) return 50;
  if (keywords.some((k) => k.includes(q))) return 45;
  if (hint && atWordStart(hint, q)) return 40;
  if (hint.includes(q)) return 30;
  if (fuzzy(label, q)) return 10;
  return 0;
}

export function rankPalette<T extends Rankable>(query: string, groups: RankGroup<T>[]): RankGroup<T>[] {
  if (!norm(query)) return groups;
  const ranked = groups.map((g, gi) => {
    const bias = g.bias ?? 0;
    const scored = g.items
      .map((item, i) => ({ item, i, score: scoreEntry(query, item) }))
      .filter((s) => s.score > 0)
      .sort((a, b) => b.score - a.score || a.i - b.i);
    return { g, gi, best: scored.length ? scored[0].score + bias : 0, items: scored.map((s) => s.item) };
  }).filter((r) => r.items.length > 0);
  ranked.sort((a, b) => b.best - a.best || a.gi - b.gi);
  return ranked.map((r) => ({ ...r.g, items: r.items }));
}
