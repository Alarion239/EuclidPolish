/* Shared display formatting — numbers, SI prefixes, bytes, durations,
 * magnitudes, sky coordinates and dates. Replaces the per-page `num`/`fmt`/
 * `pct`/`formatSpan` helpers and ad-hoc `toFixed` calls.
 *
 * Conventions: every formatter takes `unknown`-ish input (null, undefined,
 * NaN, ±Infinity) and returns `DASH` ("—") for it rather than "NaN". Output
 * is locale-independent (en-US digits and grouping, ASCII minus) so values are
 * copyable and tests are deterministic.
 */

export const DASH = "—";

type Maybe = number | null | undefined;

/** True for a finite number (narrows `unknown`). */
export function isFiniteNumber(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

const NF_CACHE = new Map<string, Intl.NumberFormat>();
function nf(opts: Intl.NumberFormatOptions): Intl.NumberFormat {
  const key = JSON.stringify(opts);
  let f = NF_CACHE.get(key);
  if (!f) { f = new Intl.NumberFormat("en-US", opts); NF_CACHE.set(key, f); }
  return f;
}

/** `2.50e-5` → `2.5e-5`, `3.2e+9` → `3.2e9`. */
function compactExp(s: string): string {
  return s.replace(/\.?0+e/, "e").replace("e+", "e");
}

export type NumberOpts = {
  /** Fixed number of decimals (overrides the significant-digit default). */
  digits?: number;
  /** Significant digits (default 3 below 1000; integers from 1000 to 1e6). */
  sig?: number;
  /** Thousands separators (default true). */
  grouping?: boolean;
  /** Always show a sign ("+0.25"). */
  signed?: boolean;
  /** Appended after a space ("43.2 dB"). */
  unit?: string;
  fallback?: string;
};

/**
 * General-purpose number formatting.
 * - `digits`: fixed decimals → `formatNumber(3.14159, {digits: 2})` = "3.14"
 * - default: 3 significant digits below 1000, whole numbers up to 1e6,
 *   compact exponent notation beyond 1e6 or below 1e-3 ("2.5e-5").
 */
export function formatNumber(v: Maybe, opts: NumberOpts = {}): string {
  const { digits, sig, grouping = true, signed = false, unit, fallback = DASH } = opts;
  if (!isFiniteNumber(v)) return fallback;
  let s: string;
  const a = Math.abs(v);
  if (digits != null) {
    s = nf({ minimumFractionDigits: digits, maximumFractionDigits: digits, useGrouping: grouping }).format(v);
  } else if (v === 0) {
    s = "0";
  } else if (a >= 1e6 || a < 1e-3) {
    s = compactExp(v.toExponential(Math.max(0, (sig ?? 3) - 1)));
  } else if (sig != null) {
    s = nf({ maximumSignificantDigits: sig, useGrouping: grouping }).format(v);
  } else if (a >= 1000) {
    s = nf({ maximumFractionDigits: 0, useGrouping: grouping }).format(v);
  } else {
    s = nf({ maximumSignificantDigits: 3, useGrouping: grouping }).format(v);
  }
  if (s === "-0" || /^-0\.?0*$/.test(s)) s = s.slice(1);
  if (signed && v > 0 && !s.startsWith("+")) s = `+${s}`;
  return unit ? `${s} ${unit}` : s;
}

/** Whole-number count with grouping: 43401 → "43,401". */
export function formatCount(v: Maybe, fallback = DASH): string {
  return isFiniteNumber(v) ? nf({ maximumFractionDigits: 0 }).format(Math.round(v)) : fallback;
}

/** Fraction → percent: 0.1234 → "12.3%". */
export function formatPercent(fraction: Maybe, digits = 1, fallback = DASH): string {
  return isFiniteNumber(fraction) ? `${(fraction * 100).toFixed(digits)}%` : fallback;
}

const SI = [
  { exp: 12, p: "T" }, { exp: 9, p: "G" }, { exp: 6, p: "M" }, { exp: 3, p: "k" }, { exp: 0, p: "" },
  { exp: -3, p: "m" }, { exp: -6, p: "µ" }, { exp: -9, p: "n" }, { exp: -12, p: "p" },
];

/** SI-prefixed value: 1234 → "1.23 k", (2.5e6, {unit: "e⁻"}) → "2.5 Me⁻". */
export function formatSI(v: Maybe, opts: { unit?: string; sig?: number; fallback?: string } = {}): string {
  const { unit = "", sig = 3, fallback = DASH } = opts;
  if (!isFiniteNumber(v)) return fallback;
  if (v === 0) return unit ? `0 ${unit}` : "0";
  const a = Math.abs(v);
  let i = SI.findIndex((s) => a >= 10 ** s.exp);
  if (i < 0) i = SI.length - 1;
  let mant = Number((v / 10 ** SI[i].exp).toPrecision(sig));
  if (Math.abs(mant) >= 1000 && i > 0) { i -= 1; mant = Number((v / 10 ** SI[i].exp).toPrecision(sig)); }
  const tail = `${SI[i].p}${unit}`;
  return tail ? `${mant} ${tail}` : String(mant);
}

const BYTE_UNITS = ["B", "KB", "MB", "GB", "TB", "PB"];

/** Byte count in binary units: 1536 → "1.5 KB", 10 MiB → "10 MB". */
export function formatBytes(n: Maybe, fallback = DASH): string {
  if (!isFiniteNumber(n) || n < 0) return fallback;
  let i = 0;
  let v = n;
  while (v >= 1024 && i < BYTE_UNITS.length - 1) { v /= 1024; i += 1; }
  if (i === 0) return `${Math.round(v)} B`;
  let s = v < 10 ? v.toFixed(1) : String(Math.round(v));
  if (s === "1024" && i < BYTE_UNITS.length - 1) { s = "1"; i += 1; }
  return `${s.replace(/\.0$/, "")} ${BYTE_UNITS[i]}`;
}

/** Elapsed time: 4.2s · 42s · 3m 05s · 2h 04m · 3d 4h. */
export function formatDuration(seconds: Maybe, fallback = DASH): string {
  if (!isFiniteNumber(seconds) || seconds < 0) return fallback;
  if (seconds < 9.95) return `${seconds.toFixed(1)}s`;
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 1440) return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, "0")}m`;
  const hours = Math.round(seconds / 3600);
  return `${Math.floor(hours / 24)}d ${hours % 24}h`;
}

/** AB magnitude: 19.2345 → "19.23"; with sigma → "19.23 ± 0.05". */
export function formatMagnitude(
  mag: Maybe,
  opts: { digits?: number; sigma?: number | null; unit?: boolean; fallback?: string } = {},
): string {
  const { digits = 2, sigma, unit = false, fallback = DASH } = opts;
  if (!isFiniteNumber(mag)) return fallback;
  let s = mag.toFixed(digits);
  if (isFiniteNumber(sigma)) s += ` ± ${sigma.toFixed(digits)}`;
  return unit ? `${s} mag` : s;
}

const SUP: Record<string, string> = {
  "-": "⁻", "+": "⁺", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹",
};

/** Superscript digits/sign: -12 → "⁻¹²". */
export function superscript(n: number | string): string {
  return String(n).split("").map((c) => SUP[c] ?? c).join("");
}

/** Power-of-ten label: 0 → "1", 1 → "10", -3 → "10⁻³". */
export function formatPow10(exp: number): string {
  if (exp === 0) return "1";
  if (exp === 1) return "10";
  return `10${superscript(exp)}`;
}

/* ── sky coordinates ─────────────────────────────────────────────────────── */

const pad2 = (n: number) => String(n).padStart(2, "0");

/** Split a non-negative total (in the smallest unit, already rounded) into
 *  [big, mid, small] parts with base-60 carries. */
function sexagesimal(total: number, decimals: number): [number, number, string] {
  const scale = 10 ** decimals;
  const units = Math.round(total * 3600 * scale);   // integer count of 1/scale small units
  const small = units % (60 * scale);
  const rest = (units - small) / (60 * scale);       // whole minutes
  const mid = rest % 60;
  const big = (rest - mid) / 60;
  const smallStr = (small / scale).toFixed(decimals).padStart(decimals ? decimals + 3 : 2, "0");
  return [big, mid, smallStr];
}

/** Right ascension (deg) as hours: 267.4229 → "17h49m41.50s" (or "17:49:41.50"). */
export function formatRA(
  deg: Maybe,
  opts: { digits?: number; style?: "hms" | "colon"; fallback?: string } = {},
): string {
  const { digits = 2, style = "hms", fallback = DASH } = opts;
  if (!isFiniteNumber(deg)) return fallback;
  const hours = (((deg % 360) + 360) % 360) / 15;
  const [hRaw, m, s] = sexagesimal(hours, digits);
  const h = hRaw % 24;   // 23h59m59.999s rounds up to 24h → 00h
  return style === "colon" ? `${pad2(h)}:${pad2(m)}:${s}` : `${pad2(h)}h${pad2(m)}m${s}s`;
}

/** Declination (deg), always signed: 64.8873 → "+64°53′14.3″" (or "+64:53:14.3"). */
export function formatDec(
  deg: Maybe,
  opts: { digits?: number; style?: "dms" | "colon"; fallback?: string } = {},
): string {
  const { digits = 1, style = "dms", fallback = DASH } = opts;
  if (!isFiniteNumber(deg)) return fallback;
  const sign = deg < 0 ? "-" : "+";
  const [d, m, s] = sexagesimal(Math.abs(deg), digits);
  return style === "colon" ? `${sign}${pad2(d)}:${pad2(m)}:${s}` : `${sign}${pad2(d)}°${pad2(m)}′${s}″`;
}

/** Decimal degrees: 267.4229 → "267.42290°". */
export function formatDeg(deg: Maybe, digits = 5, opts: { signed?: boolean; fallback?: string } = {}): string {
  if (!isFiniteNumber(deg)) return opts.fallback ?? DASH;
  const s = deg.toFixed(digits);
  return `${opts.signed && deg >= 0 ? "+" : ""}${s}°`;
}

/** An (RA, Dec) pair: sexagesimal (default), decimal degrees, or both. */
export function formatRaDec(
  ra: Maybe,
  dec: Maybe,
  opts: { mode?: "sexagesimal" | "degrees" | "both"; digits?: number } = {},
): string {
  const { mode = "sexagesimal", digits = 5 } = opts;
  if (!isFiniteNumber(ra) || !isFiniteNumber(dec)) return DASH;
  const sex = `${formatRA(ra)} ${formatDec(dec)}`;
  const degs = `${formatDeg(ra, digits)} ${formatDeg(dec, digits, { signed: true })}`;
  if (mode === "degrees") return degs;
  if (mode === "both") return `${sex} (${formatDeg(ra, digits)}, ${formatDeg(dec, digits, { signed: true })})`;
  return sex;
}

const NUM = String.raw`\d+(?:\.\d*)?|\.\d+`;
const SEX_RE = new RegExp(
  String.raw`^([+-]?)(${NUM})\s*(?:[:hd°\s])\s*(${NUM})\s*(?:[:m′'\s])\s*(${NUM})\s*[s″"]?$`, "i",
);

/** One coordinate token → value in its natural unit (hours for RA sexagesimal,
 *  degrees otherwise); `sexa` says whether it was sexagesimal. */
function parseComponent(text: string): { value: number; sexa: boolean } | null {
  const t = text.trim().replace(/[−–]/g, "-");
  if (new RegExp(`^[+-]?(?:${NUM})$`).test(t)) return { value: Number(t), sexa: false };
  const m = SEX_RE.exec(t);
  if (!m) return null;
  const [a, b, c] = [Number(m[2]), Number(m[3]), Number(m[4])];
  if (b >= 60 || c >= 60) return null;
  const v = a + b / 60 + c / 3600;
  return { value: m[1] === "-" ? -v : v, sexa: true };
}

/**
 * Parse "RA Dec" typed by a person: decimal degrees ("267.4229 64.8873",
 * "53.16, -27.78") or sexagesimal with RA in hours ("17:49:41.5 +64:53:14",
 * "17h49m41.5s -27d46m48s", "03h32m38.4s −27°46′48″", or six numbers).
 * Returns degrees, or null when the text is not a coordinate pair.
 */
export function parseSkyCoord(text: string): { ra: number; dec: number } | null {
  const t = text.trim().replace(/[−–]/g, "-").replace(/,/g, " ");
  if (!t) return null;
  let raTok: string, decTok: string;
  const words = t.split(/\s+/);
  if (words.length === 2) {
    [raTok, decTok] = words;
  } else if (words.length === 6 && words.every((w) => new RegExp(`^[+-]?(?:${NUM})$`).test(w))) {
    raTok = words.slice(0, 3).join(":");
    decTok = words.slice(3).join(":");
  } else {
    // "17h49m41.5s -27d46m48s" with inner spaces: split before the Dec sign.
    const m = /^(.+?)\s+([+-].+)$/.exec(t);
    if (!m) return null;
    [raTok, decTok] = [m[1].replace(/\s+/g, ""), m[2].replace(/\s+/g, "")];
  }
  const ra = parseComponent(raTok);
  const dec = parseComponent(decTok);
  if (!ra || !dec) return null;
  const raDeg = ra.sexa ? ra.value * 15 : ra.value;
  if (ra.value < 0 || raDeg >= 360 || Math.abs(dec.value) > 90) return null;
  return { ra: raDeg, dec: dec.value };
}

/* ── dates ───────────────────────────────────────────────────────────────── */

/** Epoch seconds (< 1e11), epoch ms, ISO string or Date → Date (null if invalid). */
export function parseTimestamp(t: unknown): Date | null {
  let d: Date;
  if (t instanceof Date) d = new Date(t.getTime());
  else if (isFiniteNumber(t)) d = new Date(Math.abs(t) < 1e11 ? t * 1000 : t);
  else if (typeof t === "string" && t.trim()) d = new Date(t);
  else return null;
  return Number.isNaN(d.getTime()) ? null : d;
}

type DateOpts = { utc?: boolean; fallback?: string };

function parts(d: Date, utc: boolean) {
  return utc
    ? [d.getUTCFullYear(), d.getUTCMonth() + 1, d.getUTCDate(), d.getUTCHours(), d.getUTCMinutes(), d.getUTCSeconds()]
    : [d.getFullYear(), d.getMonth() + 1, d.getDate(), d.getHours(), d.getMinutes(), d.getSeconds()];
}

/** "2026-09-25" (local time unless utc). */
export function formatDate(t: unknown, opts: DateOpts = {}): string {
  const d = parseTimestamp(t);
  if (!d) return opts.fallback ?? DASH;
  const [y, mo, da] = parts(d, !!opts.utc);
  return `${y}-${pad2(mo)}-${pad2(da)}`;
}

/** "2026-09-25 14:03" (":07" with seconds). */
export function formatDateTime(t: unknown, opts: DateOpts & { seconds?: boolean } = {}): string {
  const d = parseTimestamp(t);
  if (!d) return opts.fallback ?? DASH;
  const [y, mo, da, h, mi, s] = parts(d, !!opts.utc);
  return `${y}-${pad2(mo)}-${pad2(da)} ${pad2(h)}:${pad2(mi)}${opts.seconds ? `:${pad2(s)}` : ""}`;
}

/** "just now" · "2 min ago" · "3 h ago" · "2 d ago" · "in 1 h". */
export function formatRelative(t: unknown, now: number = Date.now(), fallback = DASH): string {
  const d = parseTimestamp(t);
  if (!d) return fallback;
  const dt = (d.getTime() - now) / 1000;
  const a = Math.abs(dt);
  if (a < 45) return "just now";
  let n: number, u: string;
  if (a < 45 * 60) { n = Math.round(a / 60); u = "min"; }
  else if (a < 22 * 3600) { n = Math.round(a / 3600); u = "h"; }
  else { n = Math.round(a / 86400); u = "d"; }
  return dt < 0 ? `${n} ${u} ago` : `in ${n} ${u}`;
}
