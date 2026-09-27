/* Pixel readout and the magnitude overlay of the image viewer (pure).
 *
 * The overlay magnitude is the whole cube's integrated flux in the shown band
 * (composite colours fall back to band 0):
 *   mag = zeropoint_ab_e_total − 2.5·log₁₀(Σe⁻)
 * with the stack zeropoint served precomputed (BandConfig.sim_zeropoint_e —
 * the same anchor as euclid_polish/photometry.py; never re-derived here). On
 * the SR frame of a collection with a `std` tier the one-sigma ensemble
 * disagreement adds ± (2.5/ln 10)·Σσ/ΣSR. Held to photometry.py by
 * readout.test.ts (golden fixture). */
import { bandLabel } from "./barModel";
import type { ColorMeta } from "./color";

export type MagInfo = { name: string; tot: number; mag: number | null };

type SummableCube = { data: Float32Array; c: number; bands?: string[]; sums?: Record<number, number>; noCache?: boolean; unit?: string };

const isElectrons = (unit: string | undefined) => !unit || unit === "e-" || unit === "e⁻" || unit.toLowerCase() === "electron";

export function magInfo(rec: SummableCube, colorMeta: ColorMeta, color: string): MagInfo | null {
  if (colorMeta.render_mode === "log") return null;
  // The zeropoint is for electrons over the stack (ADU/s cutouts, MJy/sr JWST
  // and arbitrary-unit kernels have no AB magnitude here).
  if (!isElectrons(rec.unit)) return null;
  const bands = colorMeta.bands;
  if (!bands || rec.noCache || !rec.data) return null;
  const names = rec.bands && rec.bands.length === rec.c ? rec.bands : (colorMeta.band_names || []);
  const bi = Math.max(0, names.indexOf(color));   // composite → band 0
  const c = rec.c || 1;
  const idx = Math.min(bi, c - 1);
  if (!rec.sums) rec.sums = {};
  if (!(idx in rec.sums)) {
    let s = 0;
    const d = rec.data;
    for (let i = idx; i < d.length; i += c) s += d[i];
    rec.sums[idx] = s;
  }
  const tot = rec.sums[idx];
  const name = names[Math.min(idx, names.length - 1)] || "VIS";
  const b = bands[name];
  if (!b || b.display_only) return null;   // native JWST approximation: no fake AB magnitude
  const mag = tot > 0 && b.zeropoint_ab_e_total ? b.zeropoint_ab_e_total - 2.5 * Math.log10(tot) : null;
  return { name, tot, mag };
}

/** δm of the SR magnitude from the std cube's sum. */
export function sigmaMagnitude(sr: MagInfo, std: MagInfo): number {
  return (2.5 / Math.LN10) * (std.tot / sr.tot);
}

/** " · VIS 21.43 AB" (or "" without a magnitude). */
export function magLabel(mi: MagInfo | null, dm?: number | null): string {
  if (!mi || mi.mag == null) return "";
  return dm != null && Number.isFinite(dm)
    ? ` · ${bandLabel(mi.name)} ${mi.mag.toFixed(2)} ± ${dm.toFixed(2)} AB`
    : ` · ${bandLabel(mi.name)} ${mi.mag.toFixed(2)} AB`;
}

type PixelCube = { data: Float32Array; w: number; h: number; c: number };

/** Every band's value at integer pixel (x, y); null outside the image. */
export function pixelValues(rec: PixelCube, x: number, y: number): number[] | null {
  if (!Number.isInteger(x) || !Number.isInteger(y) || x < 0 || y < 0 || x >= rec.w || y >= rec.h) return null;
  const o = (y * rec.w + x) * rec.c;
  return Array.from(rec.data.subarray(o, o + rec.c));
}

const emptyCache = new WeakMap<Float32Array, boolean>();

/** True when a cube has no finite pixel at all (a JWST cutout outside the
 *  mosaic): the frame shows a caption instead of a flat NaN-coloured square.
 *  Cached per data array. */
export function cubeIsEmpty(rec: { data: Float32Array } | null | undefined): boolean {
  const d = rec?.data;
  if (!d || d.length === 0) return false;
  const hit = emptyCache.get(d);
  if (hit != null) return hit;
  let empty = true;
  for (let i = 0; i < d.length; i++) if (Number.isFinite(d[i])) { empty = false; break; }
  emptyCache.set(d, empty);
  return empty;
}

/** Continuous image coordinates → the integer pixel under them. */
export function pixelAt(rec: { w: number; h: number }, fx: number, fy: number): { x: number; y: number } | null {
  const x = Math.floor(fx), y = Math.floor(fy);
  if (!(x >= 0 && y >= 0 && x < rec.w && y < rec.h)) return null;
  return { x, y };
}

/** Display form of a served unit ("e-" → "e⁻"). */
export function unitLabel(unit: string | null | undefined): string {
  const u = String(unit ?? "").trim();
  if (!u) return "";
  if (u === "e-" || u === "e⁻" || u.toLowerCase() === "electron") return "e⁻";
  if (u === "arb") return "arb.";
  return u;
}

/** Compact pixel value: 1 decimal from 10, 3 significant digits from 0.01,
 *  exponent notation below (and from 1e5); a real minus sign. */
export function formatValue(v: number): string {
  if (Number.isNaN(v)) return "NaN";
  if (v === Infinity) return "+∞";
  if (v === -Infinity) return "−∞";
  const a = Math.abs(v);
  let s: string;
  if (a === 0) s = "0";
  else if (a >= 1e5 || a < 1e-2) s = v.toExponential(2).replace("e+", "e");
  else if (a >= 10) s = v.toFixed(1);
  else s = v.toPrecision(3);
  return s.replace(/^-/, "−");
}

/** The widest the one-line readout gets for these tiers (css px), from its
 *  parts: hovering — "x y", RA Dec (+ the copy button), the band, then each
 *  tier's name, a worst-case value and its unit; idle — RA Dec, each tier's
 *  name + band + magnitude, the field size. `measure` gives a string's width
 *  (the readout's font). The readout reserves two lines when this does not
 *  fit (ReadoutBar), so no value is ever cut off and hovering never changes
 *  its height. */
export function readoutLineWidth(tiers: readonly { name: string; unit: string }[], measure: (text: string) => number, o: { gap?: number; copy?: number; hasSky?: boolean } = {}): number {
  const gap = o.gap ?? 14;
  const copy = o.copy ?? 25;
  const sky = o.hasSky === false ? [] : [measure("00h00m00.00s +00°00′00.0″") + copy];
  const sum = (parts: number[]) => parts.reduce((a, b) => a + b, 0) + gap * Math.max(0, parts.length - 1);
  const hover = sum([
    measure("x 9999  y 9999"), ...sky, measure("VIS"),
    ...tiers.map((t) => measure(`${t.name} −0.000 ${unitLabel(t.unit)}`.trim())),
  ]);
  const idle = sum([...sky, ...tiers.map((t) => measure(`${t.name} VIS 00.00 AB`)), measure("0.0× 00.0″")]);
  return Math.ceil(Math.max(hover, idle));
}
