/* Synthetic › Galaxies › templates (the TNG50-1 SKIRT atlas the generator
   draws morphologies from): pure helpers, templates.test.ts.
   - zeroStrip / floorDomain: galaxies whose plotted property is exactly zero
     (SFR = 0: the quenched ones) cannot sit on a log axis, so they are drawn
     on a labelled strip just below the axis's positive range instead of
     silently dropped;
   - marginalGuides: the histogram under the explorer (its marginal) marks
     the median and the 16–84% range, labelled at 2 significant figures in
     the property's unit;
   - templatesCaption / radiusManifestLine: the atlas caption and the radius
     manifest's ONE state (its fix lives on Status). */
import type { Guide } from "../../../charts/Plot";
import { C } from "../../../colors";
import { formatCount, formatNumber, formatRelative } from "../../../format";
import type { TngPayload, TngRadius } from "../dataApi";
import { axisOk, tngPropMeta, tngValue, type Stats, type TngRow } from "../dataModel";

export type ZeroStrip = {
  /** The axis whose zeros are drawn on the strip. */
  axis: "x" | "y";
  count: number;
  label: string;
  ids: (number | string)[];
  /** The galaxies' values on the OTHER axis (their position along the strip). */
  values: number[];
};

/** The galaxies at exactly zero on a log axis (the other axis showable), or
 *  null when neither log axis has any. The y axis is checked first. */
export function zeroStrip(
  rows: readonly TngRow[], x: string, y: string, { xlog, ylog }: { xlog: boolean; ylog: boolean },
): ZeroStrip | null {
  const pick = (axis: "x" | "y"): ZeroStrip | null => {
    const [zeroKey, otherKey, otherLog] = axis === "y" ? [y, x, xlog] : [x, y, ylog];
    const ids: (number | string)[] = [];
    const values: number[] = [];
    for (const r of rows) {
      const z = tngValue(r, zeroKey), o = tngValue(r, otherKey);
      if (z === 0 && axisOk(o, otherLog)) { ids.push(r.id); values.push(o); }
    }
    if (!ids.length) return null;
    return { axis, count: ids.length, label: `${tngPropMeta(zeroKey).label} = 0: ${formatCount(ids.length)}`, ids, values };
  };
  return (ylog ? pick("y") : null) ?? (xlog ? pick("x") : null);
}

/** A log domain extended downwards by a strip (12% of its decades) where the
 *  zero-valued galaxies are drawn (`at`, the strip's geometric middle). */
export function floorDomain(domain: [number, number], log: boolean): { domain: [number, number]; strip: [number, number]; at: number } {
  if (!log || !(domain[0] > 0)) return { domain, strip: [domain[0], domain[0]], at: domain[0] };
  const a = Math.log10(domain[0]), b = Math.log10(domain[1]);
  const lo = 10 ** (a - Math.max(0.12 * (b - a), 0.15));
  return { domain: [lo, domain[1]], strip: [lo, domain[0]], at: Math.sqrt(lo * domain[0]) };
}

/** A value at 2 significant figures, trailing zeros kept ("5.0", "0.050",
 *  "150"); masses and rates beyond 10⁴ or below 10⁻² in exponent form ("3.9e11"). */
export function sig2(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e4 || a < 1e-2) return v.toExponential(1).replace("e+", "e");
  if (a >= 100) return formatNumber(Number(v.toPrecision(2)));
  return v.toPrecision(2);
}

/** The median and 16–84% guides of the explorer's marginal histogram. */
export function marginalGuides(stats: Stats, unit: string): Guide[] {
  if (!stats) return [];
  const v = (value: number) => `${sig2(value)} ${unit}`;
  return [
    { axis: "x", v: stats.p16, color: C.muted, dash: [3, 4], width: 1, label: `p16 ${v(stats.p16)}` },
    { axis: "x", v: stats.median, color: C.mean, width: 1.6, label: `median ${v(stats.median)}` },
    { axis: "x", v: stats.p84, color: C.muted, dash: [3, 4], width: 1, label: `p84 ${v(stats.p84)}` },
  ];
}

/** "TNG50-1 atlas: 1,154 galaxies (1,140 with measured Rₑ)". */
export function templatesCaption(summary: TngPayload["summary"] | undefined): string | null {
  if (!summary) return null;
  return `TNG50-1 atlas: ${formatCount(summary.n)} galaxies (${formatCount(summary.n_in_atlas)} with measured Rₑ)`;
}

/** The radius manifest in one line: its state, the words, and whether the
 *  fix (Validate TNG radii on FASRC, on Status) is needed. */
export function radiusManifestLine(r: TngRadius | null | undefined, now = Date.now()): {
  state: "ok" | "bad" | "unknown"; text: string; fix: boolean;
} {
  if (!r || (r.cached === false && r.valid_count == null)) return { state: "unknown", text: "Radius manifest not validated yet", fix: true };
  if (!r.valid) {
    const reason = r.reasons?.[0];
    return { state: "bad", text: `Radius manifest invalid${reason ? `: ${reason}` : ""}`, fix: true };
  }
  const counts = r.expected_count != null ? `${formatCount(r.valid_count ?? 0)} of ${formatCount(r.expected_count)} measured radii valid` : "Measured radii valid";
  const checked = r.checked_at ? ` · checked ${formatRelative(r.checked_at, now)}` : "";
  return { state: "ok", text: `${counts}${checked}`, fix: false };
}
