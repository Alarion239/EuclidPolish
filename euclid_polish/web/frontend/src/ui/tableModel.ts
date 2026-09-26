/* DataTable model (tableModel.ts): the pure (React-free) half of `ui/DataTable.tsx`.
 *
 * Column definitions, value access, natural multi-key sorting, the filter
 * mini-language, CSV export, range selection and column-width estimation.
 * Everything here is deterministic and unit-tested (`tableModel.test.ts`).
 */
import type { ReactNode } from "react";

export type SortSpec = { id: string; desc: boolean };
/** Ordered sort keys: the first entry is the primary sort. */
export type SortState = SortSpec[];

export type DataColumn<T> = {
  /** Stable id: sort/filter/visibility key, default accessor (`row[id]`), filter prefix. */
  id: string;
  header: ReactNode;
  /** Plain-text header for CSV, filter prefixes and ARIA (default: `header` if it is a string, else `id`). */
  headerText?: string;
  /** Value used for sorting, filtering, CSV and the default cell (default `row[id]`). */
  accessor?: (row: T) => unknown;
  /** Cell renderer (default: the formatted accessor value). */
  cell?: (row: T, index: number) => ReactNode;
  /** Default true. */
  sortable?: boolean;
  /** Custom comparator for ascending order (empty values are NOT special-cased). */
  sortFn?: (a: T, b: T) => number;
  /** Default true: the column's text takes part in the free-text filter. */
  filterable?: boolean;
  /** Text the filter matches (default: `cellText`). */
  filterText?: (row: T) => string;
  /** CSV cell (default: `cellText`); `false` leaves the column out of the export. */
  csv?: ((row: T) => string | number | null | undefined) | false;
  align?: "left" | "right" | "center";
  /** Right-aligned tabular figures. */
  numeric?: boolean;
  /** px (number) or any CSS width. Default: estimated from the content. */
  width?: number | string;
  /** Initially hidden (the column menu can show it). */
  hidden?: boolean;
  /** Default true: listed in the column-visibility menu. */
  hideable?: boolean;
  /** Extra class on this column's cells. */
  className?: string;
};

/* ─── values ──────────────────────────────────────────────────────────────── */

export function columnValue<T>(col: DataColumn<T>, row: T): unknown {
  if (col.accessor) return col.accessor(row);
  if (row != null && typeof row === "object") return (row as Record<string, unknown>)[col.id];
  return undefined;
}

export function headerText<T>(col: DataColumn<T>): string {
  if (col.headerText != null) return col.headerText;
  if (typeof col.header === "string" || typeof col.header === "number") return String(col.header);
  return col.id;
}

const isEmpty = (v: unknown): boolean =>
  v == null || v === "" || (typeof v === "number" && Number.isNaN(v));

/** Plain text of a cell: "" for empty, raw numbers, "yes"/"no" for booleans. */
export function cellText<T>(col: DataColumn<T>, row: T): string {
  return valueText(columnValue(col, row));
}

function valueText(v: unknown): string {
  if (isEmpty(v)) return "";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (v instanceof Date) return Number.isNaN(v.getTime()) ? "" : v.toISOString();
  if (Array.isArray(v)) return v.map(valueText).join(", ");
  if (typeof v === "object") {
    try { return JSON.stringify(v); } catch { return String(v); }
  }
  return String(v);
}

/* ─── sorting ─────────────────────────────────────────────────────────────── */

const COLLATOR = new Intl.Collator("en", { numeric: true, sensitivity: "base" });

function rank(v: unknown): number {
  if (typeof v === "number" || typeof v === "bigint") return 0;
  if (v instanceof Date) return 0;
  if (typeof v === "boolean") return 1;
  return 2;
}

/** Ascending comparison of two non-empty values: numbers (and dates)
 *  numerically, booleans false < true, text naturally ("m_2" < "m_10") and
 *  case-insensitively; numbers sort before text in mixed columns. */
export function compareValues(a: unknown, b: unknown): number {
  const ra = rank(a), rb = rank(b);
  if (ra !== rb) return ra - rb;
  if (ra === 0) {
    const x = a instanceof Date ? a.getTime() : Number(a);
    const y = b instanceof Date ? b.getTime() : Number(b);
    return x < y ? -1 : x > y ? 1 : 0;
  }
  if (ra === 1) return Number(a) - Number(b);
  return COLLATOR.compare(valueText(a), valueText(b));
}

/** Stable multi-key sort. Empty values (null, undefined, "", NaN) sort last in
 *  both directions. Unknown column ids are ignored. Always returns a new array. */
export function sortRows<T>(rows: readonly T[], columns: DataColumn<T>[], sort: SortState): T[] {
  const byId = new Map(columns.map((c) => [c.id, c]));
  const keys = sort.map((s) => ({ s, col: byId.get(s.id) })).filter((k) => k.col) as
    { s: SortSpec; col: DataColumn<T> }[];
  const indexed = rows.map((row, i) => ({ row, i }));
  if (!keys.length) return indexed.map((x) => x.row);
  // Pre-compute accessor values once per (row, key).
  const vals = keys.map(({ col }) => (col.sortFn ? null : indexed.map(({ row }) => columnValue(col, row))));
  indexed.sort((A, B) => {
    for (let k = 0; k < keys.length; k++) {
      const { s, col } = keys[k];
      let c: number;
      if (col.sortFn) {
        c = col.sortFn(A.row, B.row);
      } else {
        const va = vals[k]![A.i], vb = vals[k]![B.i];
        const ea = isEmpty(va), eb = isEmpty(vb);
        if (ea || eb) {
          if (ea && eb) continue;
          return ea ? 1 : -1;            // empty last, whatever the direction
        }
        c = compareValues(va, vb);
      }
      if (c !== 0) return s.desc ? -c : c;
    }
    return A.i - B.i;
  });
  return indexed.map((x) => x.row);
}

/** Header-click transition. Plain click: asc → desc → none on that column
 *  (replacing any other sort). Multi (shift) click: same cycle on that key
 *  while keeping the others; a new key is appended. */
export function nextSort(sort: SortState, id: string, multi: boolean): SortState {
  const at = sort.findIndex((s) => s.id === id);
  if (!multi) {
    const cur = at >= 0 ? sort[at] : null;
    if (!cur) return [{ id, desc: false }];
    if (!cur.desc) return [{ id, desc: true }];
    return [];
  }
  if (at < 0) return [...sort, { id, desc: false }];
  if (!sort[at].desc) return sort.map((s, i) => (i === at ? { id, desc: true } : s));
  return sort.filter((_, i) => i !== at);
}

/** URL form: "-psnr,name" (leading "-" = descending). */
export function serializeSort(sort: SortState): string {
  return sort.map((s) => `${s.desc ? "-" : ""}${s.id}`).join(",");
}

export function parseSort(text: string | null | undefined): SortState {
  if (!text) return [];
  return text.split(",").map((p) => p.trim()).filter((p) => p && p !== "-").map((p) =>
    p.startsWith("-") ? { id: p.slice(1), desc: true } : { id: p, desc: false });
}

/* ─── filtering ───────────────────────────────────────────────────────────── */

export type FilterOp = ":" | "=" | ">" | "<" | ">=" | "<=";
export type FilterToken =
  | { value: string; negate: boolean; column?: undefined; op?: undefined }
  | { column: string; op: FilterOp; value: string; negate: boolean };

const SCOPED = /^([A-Za-z_][\w.-]*)(>=|<=|:|=|>|<)(.*)$/;

/** The filter mini-language (documented in FOUNDATION.md):
 *  `text` (substring, any column) · `"two words"` · `-text` / `!text` (not) ·
 *  `col:text` (substring in that column) · `col=v` (equal; numeric when both
 *  parse) · `col>n` `col>=n` `col<n` `col<=n` (numeric). Tokens are ANDed.
 *  A `-` before a bare number (`-0.3`, `-1e3`) is the negative value, not a
 *  negation; negate a number with `!` (`!0.3`). */
export function parseFilter(query: string): FilterToken[] {
  const out: FilterToken[] = [];
  // 1: negation · 2+3: col<op>"quoted" · 4: "quoted" · 5: bare token
  const re = /(-|!)?(?:([A-Za-z_][\w.-]*(?:>=|<=|:|=|>|<))"([^"]*)"|"([^"]*)"|(\S+))/g;
  for (const m of query.matchAll(re)) {
    const negate = !!m[1];
    if (m[2] != null) {
      const head = SCOPED.exec(`${m[2]}x`)!;
      out.push({ column: head[1], op: head[2] as FilterOp, value: m[3] ?? "", negate });
      continue;
    }
    if (m[4] != null) {
      if (m[4]) out.push({ value: m[4], negate });
      continue;
    }
    const raw = m[5] ?? "";
    if (!raw) continue;
    if (m[1] === "-" && Number.isFinite(Number(`-${raw}`))) {
      out.push({ value: `-${raw}`, negate: false });           // a negative number
      continue;
    }
    const scoped = SCOPED.exec(raw);
    if (scoped && scoped[3] !== "") {
      out.push({ column: scoped[1], op: scoped[2] as FilterOp, value: scoped[3], negate });
    } else {
      out.push({ value: raw, negate });
    }
  }
  return out;
}

const norm = (s: string) => s.toLowerCase();
const squash = (s: string) => s.toLowerCase().replace(/[\s_-]+/g, "");

function findColumn<T>(columns: DataColumn<T>[], name: string): DataColumn<T> | undefined {
  const n = norm(name), q = squash(name);
  return columns.find((c) => norm(c.id) === n)
    ?? columns.find((c) => squash(headerText(c)) === q);
}

function toNumber(v: unknown): number | null {
  if (typeof v === "number") return Number.isFinite(v) ? v : null;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (v instanceof Date) return v.getTime();
  if (typeof v === "string" && v.trim() !== "") {
    const x = Number(v);
    return Number.isFinite(x) ? x : null;
  }
  return null;
}

function filterTextOf<T>(col: DataColumn<T>, row: T): string {
  return col.filterText ? col.filterText(row) : cellText(col, row);
}

type Compiled<T> = (row: T) => boolean;

function compileToken<T>(tok: FilterToken, searchable: DataColumn<T>[], all: DataColumn<T>[]): Compiled<T> {
  let test: Compiled<T>;
  const col = tok.column != null ? findColumn(all, tok.column) : undefined;
  if (tok.column != null && col) {
    const needle = norm(tok.value);
    const op = tok.op;
    if (op === ":") {
      test = (row) => norm(filterTextOf(col, row)).includes(needle);
    } else {
      const target = toNumber(tok.value);
      test = (row) => {
        const v = columnValue(col, row);
        const x = toNumber(v);
        if (op === "=") {
          if (target != null && x != null) return x === target;
          return norm(filterTextOf(col, row)) === needle;
        }
        if (target == null || x == null) return false;
        return op === ">" ? x > target : op === ">=" ? x >= target : op === "<" ? x < target : x <= target;
      };
    }
  } else {
    const needle = norm(tok.column != null ? `${tok.column}${tok.op}${tok.value}` : tok.value);
    test = (row) => searchable.some((c) => norm(filterTextOf(c, row)).includes(needle));
  }
  return tok.negate ? (row) => !test(row) : test;
}

/** Rows matching every token of `query` (see `parseFilter`). A blank query
 *  returns a copy of all rows. */
export function filterRows<T>(rows: readonly T[], columns: DataColumn<T>[], query: string): T[] {
  const tokens = parseFilter(query ?? "");
  if (!tokens.length) return rows.slice();
  const searchable = columns.filter((c) => c.filterable !== false);
  const tests = tokens.map((t) => compileToken(t, searchable, columns));
  return rows.filter((row) => tests.every((t) => t(row)));
}

/* ─── CSV ─────────────────────────────────────────────────────────────────── */

/** One RFC-4180 CSV cell. Numbers are written raw (non-finite → empty); a
 *  text cell that a spreadsheet would run as a formula (leading = + - @ tab
 *  CR, and not itself a number) is defused with a leading `'`. Shared by
 *  `toCSV` and the Plot CSV export. */
export function csvCell(v: string | number | null | undefined): string {
  if (v == null) return "";
  if (typeof v === "number") return Number.isFinite(v) ? String(v) : "";
  let s = String(v);
  // Spreadsheet formula injection: a text cell starting with = + - @ (or a
  // tab/CR) is executed by Excel/Sheets. Prefix a quote unless it is a number.
  if (/^[=+\-@\t\r]/.test(s) && !Number.isFinite(Number(s))) s = `'${s}`;
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, "\"\"")}"` : s;
}

/** RFC-4180 CSV (CRLF line ends, trailing CRLF) of `rows` over `columns`
 *  (columns with `csv: false` are left out). */
export function toCSV<T>(rows: readonly T[], columns: DataColumn<T>[]): string {
  const cols = columns.filter((c) => c.csv !== false);
  const lines = [cols.map((c) => csvCell(headerText(c))).join(",")];
  for (const row of rows) {
    lines.push(cols.map((c) => {
      if (typeof c.csv === "function") return csvCell(c.csv(row));
      const v = columnValue(c, row);
      return csvCell(typeof v === "number" ? v : cellText(c, row));
    }).join(","));
  }
  return lines.join("\r\n") + "\r\n";
}

/* ─── selection / layout helpers ──────────────────────────────────────────── */

/** Keys from `anchor` to `target` inclusive, in view order; just `[target]`
 *  when the anchor is unknown or not in view. */
export function rangeKeys(order: readonly string[], anchor: string | null | undefined, target: string): string[] {
  const b = order.indexOf(target);
  const a = anchor == null ? -1 : order.indexOf(anchor);
  if (a < 0 || b < 0) return [target];
  const [lo, hi] = a <= b ? [a, b] : [b, a];
  return order.slice(lo, hi + 1);
}

export const MIN_COL_WIDTH = 56;
export const MAX_COL_WIDTH = 420;

/** Stable px widths per column id, estimated from the header and the first
 *  `sample` rows (so virtualised rows scrolling in never reflow the table).
 *  A numeric `width` is used as-is; string widths are left to CSS (absent). */
export function estimateWidths<T>(
  rows: readonly T[], columns: DataColumn<T>[], opts: { sample?: number; charPx?: number; pad?: number } = {},
): Record<string, number> {
  const { sample = 200, charPx = 7.4, pad = 28 } = opts;
  const out: Record<string, number> = {};
  const n = Math.min(rows.length, sample);
  for (const c of columns) {
    if (typeof c.width === "number") { out[c.id] = c.width; continue; }
    if (typeof c.width === "string") continue;
    let chars = headerText(c).length + (c.sortable === false ? 0 : 2);
    for (let i = 0; i < n; i++) chars = Math.max(chars, Math.min(64, cellText(c, rows[i]).length));
    out[c.id] = Math.round(Math.max(MIN_COL_WIDTH, Math.min(MAX_COL_WIDTH, chars * charPx + pad)));
  }
  return out;
}
