/* Colour scales of the atlas layers (pure, given a palette).
 *
 * Every colour comes from the sky tokens (`--sky-*` in atlas.css: bright
 * values that read on dark sky imagery in BOTH app themes) or the viridis
 * ramp; the backend's style hex values are ignored (only `color_by` is used).
 * A layer's colour is one of: a fixed token, a categorical map (SR state,
 * lens grade, field…), a sequential viridis ramp over the data range (sky
 * level, magnitude, FWHM…) or a diverging ramp (SR/LR flux ratio around 1).
 * Anything with `state: "rejected"` (unobserved Q1 tiles) is drawn in `bad`. */
import { viridis } from "../../../colors";
import type { LayerInfo, SkyFeature } from "./layerModel";

export type SkyPalette = {
  accent: string; good: string; warn: string; bad: string; info: string;
  muted: string; ink: string; select: string; hover: string;
  cat: string[];
};

export const SKY_TOKENS = {
  accent: "--sky-accent", good: "--sky-good", warn: "--sky-warn", bad: "--sky-bad", info: "--sky-info",
  muted: "--sky-muted", ink: "--sky-ink", select: "--sky-select", hover: "--sky-hover",
} as const;
export const SKY_CAT_TOKENS = [
  "--sky-cat-0", "--sky-cat-1", "--sky-cat-2", "--sky-cat-3",
  "--sky-cat-4", "--sky-cat-5", "--sky-cat-6", "--sky-cat-7",
] as const;

/** Read the sky palette from the tokens (runtime). */
export function readSkyPalette(el: Element = document.documentElement): SkyPalette {
  const style = getComputedStyle(el);
  const read = (name: string) => style.getPropertyValue(name).trim().toLowerCase() || "gray";
  const out = Object.fromEntries(Object.entries(SKY_TOKENS).map(([k, v]) => [k, read(v)])) as Omit<SkyPalette, "cat">;
  return { ...out, cat: SKY_CAT_TOKENS.map(read) };
}

/* ── colour maths ────────────────────────────────────────────────────── */

function parseHex(c: string): [number, number, number] | null {
  const m = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(c.trim());
  if (!m) return null;
  const h = m[1].length === 3 ? m[1].split("").map((x) => x + x).join("") : m[1];
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)) as [number, number, number];
}

const hex2 = (v: number) => Math.round(Math.max(0, Math.min(255, v))).toString(16).padStart(2, "0");

export function lerpColor(a: string, b: string, t: number): string {
  const pa = parseHex(a), pb = parseHex(b);
  if (!pa || !pb) return t < 0.5 ? a : b;
  const k = Math.max(0, Math.min(1, t));
  return `#${pa.map((x, i) => hex2(x + (pb[i] - x) * k)).join("")}`;
}

/** `#rrggbb` → `rgba(r, g, b, a)`; other colour strings pass through. */
export function withAlpha(color: string, alpha: number): string {
  const p = parseHex(color);
  if (!p) return color;
  return `rgba(${p[0]}, ${p[1]}, ${p[2]}, ${Math.max(0, Math.min(1, alpha))})`;
}

/* ── scales ──────────────────────────────────────────────────────────── */

type Extra = { label: string; color: string };

export type ColorScale =
  | { type: "fixed"; color: string; label: string }
  | { type: "categorical"; prop: string; label: string; entries: { value: string; label: string; color: string }[]; other: string; extras: Extra[]; rejected: string }
  | { type: "sequential"; prop: string; label: string; domain: [number, number]; reverse: boolean; missing: string; extras: Extra[]; rejected: string }
  | { type: "diverging"; prop: string; label: string; domain: [number, number, number]; stops: [string, string, string]; missing: string; extras: Extra[]; rejected: string };

export type Legend = {
  title: string;
  items?: Extra[];
  gradient?: { stops: string[]; min: string; max: string; mid?: string };
};

export const FIXED_TOKENS = ["accent", "good", "warn", "bad", "info", ...SKY_CAT_TOKENS.map((_, i) => `cat-${i}`)] as const;

/** The token a layer uses when it has no colour-by (stable per layer). */
const LAYER_TOKEN: Record<string, string> = {
  "moc-q1": "accent", "moc-jwst": "cat-4", "q1-fields": "cat-3", "q1-tiles": "cat-0",
  "nexus-footprint": "cat-2", "nexus-tiles": "cat-2", "real-tiles": "cat-5", "real-fields": "cat-6",
  poster: "cat-4", pairs: "cat-5", "archive-fields": "cat-3", "eval-objects": "cat-1",
  experiments: "cat-6", "lens-candidates": "bad", galaxies: "cat-1", stars: "cat-7",
  "psf-clusters": "cat-2", "noise-positions": "cat-0", "population-cones": "cat-0",
  "gaia-fields": "cat-7", "jwst-mast": "cat-6", "jwst-footprints": "cat-6", selection: "accent",
};

function tokenColor(token: string, p: SkyPalette): string | null {
  const cat = /^cat-(\d)$/.exec(token);
  if (cat) return p.cat[Number(cat[1]) % p.cat.length] ?? null;
  if (token in SKY_TOKENS) return p[token as keyof typeof SKY_TOKENS];
  return null;
}

function hashIndex(s: string, n: number): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  return Math.abs(h) % n;
}

export function layerToken(id: string): string {
  return LAYER_TOKEN[id] ?? `cat-${hashIndex(id, 8)}`;
}

const PROP_LABEL: Record<string, string> = {
  state: "SR state", grade: "grade", field: "field", kind: "kind",
  flux_ratio_sr_over_lr: "SR/LR flux ratio", vis_level_e: "VIS sky level (e⁻)", mag: "magnitude",
  fwhm_arcsec: "PSF FWHM (″)", VIS: "VIS noise (e⁻)", Y_E: "Y noise (e⁻)", J_E: "J noise (e⁻)",
  H_E: "H noise (e⁻)", rows: "rows", exptime_s: "exposure (s)",
};

const numeric = (v: unknown): number | null => {
  const n = typeof v === "number" ? v : typeof v === "string" && v.trim() !== "" ? Number(v) : NaN;
  return Number.isFinite(n) ? n : null;
};

function categoricalFor(prop: string, p: SkyPalette, features: readonly SkyFeature[]) {
  const fixed: Record<string, [string, string][]> = {
    state: [["current", p.good], ["stale", p.warn], ["missing", p.muted]],
    grade: [["A", p.bad], ["B", p.warn], ["C", p.cat[3]]],
    field: [["EDF-N", p.cat[0]], ["EDF-S", p.cat[1]], ["EDF-F", p.cat[2]], ["LDN1641", p.cat[3]]],
    kind: [["lens", p.cat[4]], ["galaxy", p.cat[1]]],
  };
  if (fixed[prop]) return fixed[prop].map(([value, color]) => ({ value, label: value, color }));
  const counts = new Map<string, number>();
  for (const f of features) {
    const v = f.props[prop];
    if (v == null || v === "") continue;
    counts.set(String(v), (counts.get(String(v)) ?? 0) + 1);
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, 8)
    .map(([value], i) => ({ value, label: value, color: p.cat[i % p.cat.length] }));
}

function percentile(sorted: number[], q: number): number {
  const i = Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))));
  return sorted[i];
}

function domainOf(prop: string, features: readonly SkyFeature[]): [number, number] {
  const vals = features.map((f) => numeric(f.props[prop])).filter((v): v is number => v != null).sort((a, b) => a - b);
  if (!vals.length) return [0, 1];
  const [lo, hi] = vals.length >= 100 ? [percentile(vals, 0.01), percentile(vals, 0.99)] : [vals[0], vals[vals.length - 1]];
  return lo === hi ? [lo - 0.5, hi + 0.5] : [lo, hi];
}

function extrasFor(prop: string, features: readonly SkyFeature[], p: SkyPalette, layerId: string): Extra[] {
  const out: Extra[] = [];
  if (layerId === "q1-tiles" || features.some((f) => f.props.state === "rejected")) out.push({ label: "rejected", color: p.bad });
  if (features.some((f) => f.props.state !== "rejected" && numeric(f.props[prop]) == null && typeof f.props[prop] !== "string")) {
    out.push({ label: "no data", color: p.muted });
  }
  return out;
}

const SEQUENTIAL = new Set(["vis_level_e", "mag", "fwhm_arcsec", "VIS", "Y_E", "J_E", "H_E", "rows", "exptime_s"]);
const REVERSED = new Set(["mag"]);

function autoScale(prop: string, info: LayerInfo, features: readonly SkyFeature[], p: SkyPalette): ColorScale {
  const label = PROP_LABEL[prop] ?? prop;
  if (prop === "flux_ratio_sr_over_lr") {
    return {
      type: "diverging", prop, label, domain: [0.5, 1, 1.5], stops: [p.bad, p.good, p.info], missing: p.muted,
      extras: extrasFor(prop, features, p, info.id), rejected: p.bad,
    };
  }
  const values = features.map((f) => f.props[prop]).filter((v) => v != null && v !== "");
  const isNumeric = SEQUENTIAL.has(prop) || (values.length > 0 && values.every((v) => numeric(v) != null) && !["state", "grade", "field", "kind"].includes(prop));
  if (isNumeric) {
    return {
      type: "sequential", prop, label, domain: domainOf(prop, features), reverse: REVERSED.has(prop), missing: p.muted,
      extras: extrasFor(prop, features, p, info.id), rejected: p.bad,
    };
  }
  return {
    type: "categorical", prop, label, entries: categoricalFor(prop, p, features), other: p.muted,
    extras: info.id === "q1-tiles" || features.some((f) => f.props.state === "rejected") ? [{ label: "rejected", color: p.bad }] : [],
    rejected: p.bad,
  };
}

/** The scale of a layer. `override`: a fixed token (`cat-3`, `good`, …) or `by.<prop>`. */
export function scaleFor(info: LayerInfo, features: readonly SkyFeature[], p: SkyPalette, override?: string | null): ColorScale {
  if (override) {
    const fixed = tokenColor(override, p);
    if (fixed) return { type: "fixed", color: fixed, label: info.label };
    if (override.startsWith("by.") && override.length > 3) return autoScale(override.slice(3), info, features, p);
  }
  const by = info.style.color_by;
  if (by) return autoScale(by, info, features, p);
  return { type: "fixed", color: tokenColor(layerToken(info.id), p) ?? p.accent, label: info.label };
}

export function colorOf(s: ColorScale, props: Record<string, unknown>): string {
  if (s.type === "fixed") return s.color;
  if (props.state === "rejected") return s.rejected;
  const v = props[s.prop];
  if (s.type === "categorical") return s.entries.find((e) => e.value === String(v))?.color ?? s.other;
  const n = numeric(v);
  if (n == null) return s.missing;
  if (s.type === "sequential") {
    const [lo, hi] = s.domain;
    const t = Math.max(0, Math.min(1, (n - lo) / (hi - lo)));
    return viridis(s.reverse ? 1 - t : t);
  }
  const [lo, mid, hi] = s.domain;
  if (n <= mid) return lerpColor(s.stops[0], s.stops[1], (n - lo) / (mid - lo));
  return lerpColor(s.stops[1], s.stops[2], (n - mid) / (hi - mid));
}

const fmt = (v: number) => String(Number(v.toPrecision(3)));

export function legendFor(s: ColorScale): Legend {
  if (s.type === "fixed") return { title: s.label, items: [{ label: s.label, color: s.color }] };
  if (s.type === "categorical") return { title: s.label, items: [...s.entries.map(({ label, color }) => ({ label, color })), ...s.extras] };
  if (s.type === "sequential") {
    const ts = [0, 0.25, 0.5, 0.75, 1];
    return {
      title: s.label, items: s.extras,
      gradient: { stops: ts.map((t) => viridis(s.reverse ? 1 - t : t)), min: fmt(s.domain[0]), max: fmt(s.domain[1]) },
    };
  }
  return {
    title: s.label, items: s.extras,
    gradient: { stops: [...s.stops], min: `≤ ${fmt(s.domain[0])}`, mid: fmt(s.domain[1]), max: `≥ ${fmt(s.domain[2])}` },
  };
}

/* ── colour choices for a layer row ──────────────────────────────────── */

const NOT_COLOURABLE = new Set([
  "ra", "dec", "label", "id", "ref", "name", "obs_id", "tile", "field_id", "star_id", "polygon",
  "path", "file", "target", "out_subdir", "cluster", "flags", "level_position", "stored_field",
  "position_name", "outline", "rejected", "models", "levels_e", "polygons",
]);

export type ColorChoice = { value: string; label: string };

/** "" (the layer default), `by.<prop>` for colourable props, then fixed tokens. */
export function colorOptions(info: LayerInfo, features: readonly SkyFeature[]): ColorChoice[] {
  const by = info.style.color_by;
  const out: ColorChoice[] = [{ value: "", label: by ? `Default · ${PROP_LABEL[by] ?? by}` : "Default" }];
  const sample = features.slice(0, 500);
  const keys = new Set<string>();
  for (const f of sample) for (const k of Object.keys(f.props)) keys.add(k);
  for (const k of [...keys].sort()) {
    if (NOT_COLOURABLE.has(k) || k === by) continue;
    const vals = sample.map((f) => f.props[k]).filter((v) => v != null && v !== "");
    if (vals.length < Math.max(1, sample.length * 0.3)) continue;
    const allNum = vals.every((v) => numeric(v) != null && typeof v !== "boolean");
    const distinct = new Set(vals.map(String)).size;
    if (allNum ? distinct > 1 : distinct >= 2 && distinct <= 12) out.push({ value: `by.${k}`, label: `By ${PROP_LABEL[k] ?? k}` });
  }
  for (const t of FIXED_TOKENS) out.push({ value: t, label: t.startsWith("cat-") ? `Colour ${Number(t.slice(4)) + 1}` : t });
  return out;
}
