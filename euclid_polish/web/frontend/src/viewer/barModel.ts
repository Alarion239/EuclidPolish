/* Pure model of the viewer's control bar (src/viewer/README.md "Control
 * bar"): which tiers get a chip, the colour choices, short labels, the
 * display unit of a transfer group, number formatting for the knee field,
 * and the keyboard-sequence guard. */
import type { TierMeta } from "./types";

/** Colour keys of the old engine: Q W E R T Y → the bands, then Lupton, Temp. */
export const COLOR_KEYS = ["q", "w", "e", "r", "t", "y"];
export const COLOR_MODES_EXTRA = [
  { key: "lupton", label: "Lupton", title: "4-band solar-balanced Lupton RGB" },
  { key: "temp", label: "Temp", title: "Per-pixel blackbody-T colour (Planckian locus)" },
];

/** The knee slider spans the research knee grid (training and scoring knees). */
export const KNEE_SLIDER_RANGE: [number, number] = [0.1, 1e4];
/** Brightness (gain) slider range. */
export const GAIN_SLIDER_RANGE: [number, number] = [0.1, 10];
/** At most this many tier chips sit in the bar; the rest are in the tier menu. */
export const MAX_TIER_CHIPS = 7;

/** A short chip / frame name from a meta tier label: the part before the
 *  first " · " or " (" ("SR · production gate" → "SR", "BHR (blurred HR)" →
 *  "BHR", "SR 169·psnr" → "SR 169"). An all-lowercase label gets a capital
 *  ("disagreement movie" → "Disagreement movie"). */
export function shortTierLabel(label: string): string {
  const full = String(label ?? "").trim();
  const parts = full.split(/\s*·\s*|\s+\(/);
  const head = parts[0].trim() || full;
  // A FITS HDU tier ("1 · LR_VIS"): the HDU name says what it is, its index does not.
  if (/^\d+$/.test(head) && parts[1]?.trim()) return hduName(parts[1].trim());
  return sentenceLabel(head);
}

/** A tier's name in the one-line readout: the short label, cut to its first
 *  word when longer than 12 characters ("Mean of 30 starfull members" →
 *  "Mean"), so every tier's value fits; the full label is in the tooltip. */
export function readoutTierName(label: string): string {
  const short = shortTierLabel(label);
  return short.length > 12 ? short.split(/\s+/)[0] : short;
}

/** The readout names of the shown tiers, each unique: a name shared by two
 *  tiers ("Gate full30s1" and "Gate full30s2" both cut to "Gate") keeps its
 *  short label, or the full label if that is shared too. */
export function readoutTierNames(labels: readonly string[]): string[] {
  const names = labels.map(readoutTierName);
  const count = (xs: string[], x: string) => xs.filter((y) => y === x).length;
  const longer = names.map((n, i) => (count(names, n) > 1 ? shortTierLabel(labels[i]) : n));
  return longer.map((n, i) => (count(longer, n) > 1 ? String(labels[i] ?? "").trim() || n : n));
}

/** "LR_VIS" → "LR VIS", "SR_Y_E" → "SR Y" (the NISP suffix dropped). */
function hduName(name: string): string {
  return name.replace(/_([YJH])_E$/, "_$1").replace(/_/g, " ").trim();
}

/** A served tier label in the console's plain words: the regime names in
 *  lower case, as everywhere else ("Mean of 30 STARFULL members" → "Mean of
 *  30 starfull members"), and a trailing "· native" as "(native)". */
export function plainLabel(label: string): string {
  return String(label ?? "").trim()
    .replace(/\bSTAR(FULL|LESS)\b/g, (m) => m.toLowerCase())
    .replace(/\s*·\s*native$/i, " (native)");
}

/** A label in sentence case: an all-lowercase phrase gets a capital
 *  ("disagreement movie" → "Disagreement movie"); names with capitals
 *  ("stdSR", "SR 169·psnr") are kept as they are (in plain words, plainLabel). */
export function sentenceLabel(label: string): string {
  const s = plainLabel(label);
  return /^[a-z][a-z0-9 -]*$/.test(s) ? s[0].toUpperCase() + s.slice(1) : s;
}

/** "Y_E" → "Y" (Euclid NISP bands); other names as they are. */
export function bandLabel(name: string): string {
  return /^[YJH]_E$/.test(name) ? name[0] : name;
}

/** The tiers with a chip in the bar (canonical order): every tier that is
 *  not `hidden`; past MAX_TIER_CHIPS, the first ones plus any selected one. */
export function chipTiers(tiers: TierMeta[], selected: readonly string[], max = MAX_TIER_CHIPS): TierMeta[] {
  const shown = tiers.filter((t) => !t.hidden);
  if (shown.length <= max) return shown;
  const keep = new Set(shown.slice(0, max - 1).map((t) => t.key));
  for (const t of shown) if (selected.includes(t.key)) keep.add(t.key);
  return shown.filter((t) => keep.has(t.key));
}

export type ColourOption = { key: string; label: string; title: string; shortcut?: string };

/** The band / colour choices: the collection's bands, then Lupton and Temp,
 *  each with its old key (Q–Y). None for a log-rendered collection (PSFs)
 *  or when every shown tier is a single plane (a native one-filter image:
 *  there is no band to choose). */
export function colourOptions(bandNames: readonly string[], o: { logMode: boolean; singlePlane: boolean }): ColourOption[] {
  if (o.logMode || o.singlePlane) return [];
  return [
    ...bandNames.map((n) => ({ key: n, label: bandLabel(n), title: n })),
    ...COLOR_MODES_EXTRA.map((m) => ({ key: m.key, label: m.label, title: m.title })),
  ].map((c, i) => ({ ...c, shortcut: COLOR_KEYS[i]?.toUpperCase() }));
}

/** True when there is at least one shown cube and every one has one plane. */
export function isSinglePlane(recs: readonly { c: number }[]): boolean {
  return recs.length > 0 && recs.every((r) => r.c === 1);
}

type UnitRec = { transferGroup?: string; unit?: string; displayScale?: number };

/** The unit a transfer group's knee and black point are shown in, and the
 *  factor from native to transfer units: a JWST group reads MJy/sr (the
 *  knee is kept in display units = native × display scale); Euclid groups
 *  read e⁻. From the shown cubes of the group, else the meta tier units. */
export function groupUnit(recs: readonly UnitRec[], inGroup: (r: UnitRec) => boolean, fallbackUnit = ""): { unit: string; scale: number } {
  const rec = recs.find(inGroup);
  const raw = (rec?.unit || fallbackUnit || "").trim();
  const unit = raw === "e-" || raw.toLowerCase() === "electron" ? "e⁻" : raw === "arb" ? "arb." : raw;
  const scale = rec && rec.displayScale && rec.displayScale > 0 ? rec.displayScale : 1;
  return { unit, scale };
}

/** A compact number for a control: 3 significant digits, no exponent
 *  between 0.001 and 1e6, a real minus sign ("0.1", "3.16", "100", "1e4" →
 *  "10000"). */
export function formatSig(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  if (v === 0) return "0";
  const a = Math.abs(v);
  const s = a >= 1e6 || a < 1e-3 ? v.toExponential(2).replace("e+", "e") : String(Number(v.toPrecision(3)));
  return s.replace(/^-/, "−");
}

/** A typed number ("1e4", "0,5", "−3", " 100 ") or null. */
export function parseNumber(text: string): number | null {
  const t = String(text ?? "").trim().replace(/−/g, "-").replace(",", ".");
  if (!t) return null;
  const v = Number(t);
  return Number.isFinite(v) ? v : null;
}

/** Whether a key press may be the second key of a shell sequence ("g s",
 *  "g e" …): the previous plain key was "g" less than `windowMs` ago. The
 *  viewer then leaves the key to the shell. */
export function sequencePending(prev: { key: string; t: number } | null, now: number, windowMs = 1000): boolean {
  return !!prev && prev.key === "g" && now - prev.t >= 0 && now - prev.t < windowMs;
}

export type BarItem = {
  width: number;
  row: 1 | 2;
  /** The group's id (Bar.tsx `data-g`): what `overflow` / `collapsed` name. */
  id?: string;
  /** May move into the bar's More menu in a narrow viewer; lower goes first. */
  overflow?: number;
  /** Its width when collapsed (the band chips become one select). */
  shrink?: number;
};
/** rows: one or two; compact: the button texts hidden; overflow: the groups
 *  moved into the More menu (a narrow viewer, 300–480 px); collapsed: the
 *  groups drawn collapsed (the band chips as one select); wrap: the rows
 *  still too wide (below ~300 px) — they wrap onto another line. No control
 *  is ever cut off or scrolled out of sight. */
export type BarLayout = { rows: 1 | 2; compact: boolean; wrap: (1 | 2)[]; overflow: string[]; collapsed: string[] };

/** How the bar lays out: ONE row when every item fits (with the Display
 *  text, else icon-only); otherwise two rows — what is shown (tiers, bands)
 *  above how it is shown — with the text unless the second row only fits
 *  icon-only. When a row does not fit even then (a narrow viewer: the
 *  inspector panel, the bottom sheet), the first row collapses its band
 *  chips into a select, then its tier chips to the selected ones (the tier
 *  menu has the rest), and the second moves its rarely used groups into a
 *  More menu (`moreWidth`), lowest `overflow` rank first, until it fits;
 *  only what still does not fit wraps. `textWidth` is the width the second
 *  row's button texts take when shown. */
export function barLayout(o: { items: readonly BarItem[]; textWidth: number; gap: number; available: number; moreWidth?: number }): BarLayout {
  const { items, gap, available } = o;
  const done = (rows: 1 | 2, compact: boolean, wrap: (1 | 2)[] = [], overflow: string[] = [], collapsed: string[] = []): BarLayout =>
    ({ rows, compact, wrap, overflow, collapsed });
  if (!(available > 0) || !items.length) return done(1, false);
  const span = (list: readonly BarItem[]) => list.reduce((s, it) => s + it.width, 0) + gap * Math.max(0, list.length - 1);
  const all = span(items);
  if (all <= available) return done(1, false);
  if (all - o.textWidth <= available) return done(1, true);
  const row1 = items.filter((it) => it.row === 1);
  const row2 = items.filter((it) => it.row === 2);
  const compact = span(row2) > available;
  let first = span(row1);
  let second = compact ? span(row2) - o.textWidth : span(row2);
  const collapsed: string[] = [];
  const overflow: string[] = [];
  // the first row collapses from its end (the band chips, then the tier chips)
  for (const it of [...row1].reverse()) {
    if (first <= available) break;
    if (it.id && it.shrink != null && it.shrink < it.width) { collapsed.push(it.id); first -= it.width - it.shrink; }
  }
  if (second > available) {
    const movable = row2.filter((it) => it.id && it.overflow != null).sort((a, b) => (a.overflow as number) - (b.overflow as number));
    const more = Math.max(0, o.moreWidth ?? 28);
    for (const it of movable) {
      if (second <= available) break;
      second -= it.width + gap;
      if (!overflow.length) second += more + gap;
      overflow.push(it.id as string);
    }
  }
  const wrap: (1 | 2)[] = [];
  if (first > available) wrap.push(1);
  if (second > available) wrap.push(2);
  return done(2, compact, wrap, overflow, collapsed);
}

export function sameBarLayout(a: BarLayout, b: BarLayout): boolean {
  return a.rows === b.rows && a.compact === b.compact && a.wrap.join() === b.wrap.join()
    && a.overflow.join() === b.overflow.join() && a.collapsed.join() === b.collapsed.join();
}

/** Height reserved for the bar before the meta arrives, so nothing below it
 *  jumps: the rows it had last time for this collection, else two rows when
 *  the viewer is narrower than a one-row bar usually needs. */
export const ONE_ROW_MIN_WIDTH = 760;
export const BAR_ROWS_STORAGE_KEY = "euclid-polish.viewer.bar-rows";
type Store = { getItem(k: string): string | null; setItem(k: string, v: string): void };

function readRowsMap(store: Store | null): Record<string, 1 | 2> {
  try {
    const raw = store?.getItem(BAR_ROWS_STORAGE_KEY);
    const m = raw ? JSON.parse(raw) : {};
    return m && typeof m === "object" ? m : {};
  } catch { return {}; }
}

export function reservedBarRows(key: string, width: number, store: Store | null): 1 | 2 {
  const v = readRowsMap(store)[key];
  if (v === 1 || v === 2) return v;
  return width > 0 && width < ONE_ROW_MIN_WIDTH ? 2 : 1;
}

/** Remember the bar's rows for `key` (a collection); storage errors are ignored. */
export function rememberBarRows(key: string, rows: 1 | 2, store: Store | null) {
  const m = readRowsMap(store);
  if (m[key] === rows) return;
  m[key] = rows;
  const keys = Object.keys(m);
  if (keys.length > 64) delete m[keys[0]];
  try { store?.setItem(BAR_ROWS_STORAGE_KEY, JSON.stringify(m)); } catch { /* private window */ }
}

/** The page's storage, or null where it throws (private window, sandbox). */
export function safeStorage(): Store | null {
  try { return typeof window !== "undefined" ? window.localStorage : null; } catch { return null; }
}

/** The navigation counter: a 1-based position and the object count
 *  ("1 / 100" for index 0 of 100), as people count. */
export function navPosition(index: number, count: number): { position: string; total: string } {
  return { position: String(Math.max(0, Math.floor(index)) + 1), total: String(Math.max(0, Math.floor(count))) };
}

/** A typed 1-based position → the 0-based index (null when not a number;
 *  the caller clamps). */
export function parsePosition(text: string): number | null {
  const v = parseNumber(text);
  return v == null ? null : Math.round(v) - 1;
}

/** The two shapes of "More display settings" (css px, measured at 1024 × 768):
 *  one column, or two columns (wide and short). */
export const MORE_SHAPES = { narrow: { w: 340, h: 340 }, wide: { w: 600, h: 240 } } as const;
export type Box = { left: number; top: number; right: number; bottom: number };

/** Which shape of "More display settings" hides less of the frames: the
 *  popover opens under its trigger (`anchor`), aligned to the trigger's
 *  right edge, inside the viewport (8 px of padding). The frames are the
 *  on-screen frame rectangles. Ties go to the narrow one. */
export function moreShape(frames: readonly Box[], anchor: Box, viewportWidth: number): "narrow" | "wide" {
  const covered = (w: number, h: number) => {
    const right = Math.min(anchor.right, viewportWidth - 8);
    const box = { left: Math.max(8, right - w), top: anchor.bottom + 6, right, bottom: anchor.bottom + 6 + h };
    return frames.reduce((sum, f) => {
      const x = Math.max(0, Math.min(f.right, box.right) - Math.max(f.left, box.left));
      const y = Math.max(0, Math.min(f.bottom, box.bottom) - Math.max(f.top, box.top));
      return sum + x * y;
    }, 0);
  };
  const n = covered(MORE_SHAPES.narrow.w, MORE_SHAPES.narrow.h);
  const w = covered(MORE_SHAPES.wide.w, MORE_SHAPES.wide.h);
  return w < n ? "wide" : "narrow";
}
