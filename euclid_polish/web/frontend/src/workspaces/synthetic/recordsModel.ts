/* Synthetic › Records census (pure; recordsModel.test.ts): the Σ VIS and
   brightest-star histograms, the record a histogram click opens, the record
   area and the generated lens density; and where the record's SR is. */
import { pagePath } from "../../app/nav";
import { extent } from "../../ticks";
import type { FieldCensus, SourcesCensus } from "./dataApi";
import { histogram } from "./dataModel";

export type CensusHistogram = { centers: number[]; counts: number[]; domain: [number, number]; n: number };

const value = (f: FieldCensus, which: "total" | "star"): number | null => {
  const v = which === "total" ? f.total_vis_e : f.brightest_star_mag;
  return v != null && Number.isFinite(v) && (which === "star" || v > 0) ? v : null;
};

/** Σ VIS per record in log10 bins (drawn on a log axis), or the brightest
 *  star's VIS magnitude in 0.5-mag bins; null without values. */
export function censusHistogram(fields: readonly FieldCensus[], which: "total" | "star"): CensusHistogram | null {
  const values = fields.map((f) => value(f, which)).filter((v): v is number => v != null);
  const span = extent(values);
  if (!span) return null;
  if (which === "total") {
    let [lo, hi] = [Math.log10(span[0]), Math.log10(span[1])];
    lo = Math.floor(lo * 5) / 5; hi = Math.max(lo + 0.2, Math.ceil(hi * 5) / 5);
    const bins = Math.max(1, Math.round((hi - lo) / 0.2));
    const h = histogram(values.map(Math.log10), lo, hi, bins);
    return { centers: h.centers.map((c) => 10 ** c), counts: h.counts, domain: [10 ** lo, 10 ** hi], n: values.length };
  }
  const lo = Math.floor(span[0] * 2) / 2, hi = Math.max(lo + 0.5, Math.ceil(span[1] * 2) / 2);
  const h = histogram(values, lo, hi, Math.max(1, Math.round((hi - lo) / 0.5)));
  return { centers: h.centers, counts: h.counts, domain: [lo, hi], n: values.length };
}

/** The record whose value is nearest a click at `x` (log distance for Σ VIS). */
export function nearestRecord(fields: readonly FieldCensus[], which: "total" | "star", x: number): number | null {
  if (!Number.isFinite(x) || (which === "total" && x <= 0)) return null;
  const t = (v: number) => (which === "total" ? Math.log10(v) : v);
  let best: number | null = null;
  let bestD = Infinity;
  for (const f of fields) {
    const v = value(f, which);
    if (v == null) continue;
    const d = Math.abs(t(v) - t(x));
    if (d < bestD) { bestD = d; best = f.field_index; }
  }
  return best;
}

/** One record's LR field area in arcmin² (its LR grid × pixel scale), or null. */
export function recordArea(census: Pick<SourcesCensus, "geometry">): number | null {
  const lr = census.geometry.lr;
  if (!lr || !(lr.pixscale > 0)) return null;
  return (lr.width * lr.pixscale / 60) * (lr.height * lr.pixscale / 60);
}

/** Lenses per arcmin² over the split's records, with the count and area. */
export function lensDensity(census: SourcesCensus): { count: number; area: number; density: number } | null {
  const area1 = recordArea(census);
  if (area1 == null || !census.fields.length) return null;
  const count = census.fields.reduce((n, f) => n + (f.lens ?? 0), 0);
  const area = area1 * census.fields.length;
  return { count, area, density: count / area };
}

/** Where a record's SR is: Models › Images, records set, on this record when
 *  its split has SR cubes (`nSr`), else the set itself (its Generate SR). */
export function recordSrLink(split: string, index: number | null, nSr: number): { to: string; label: string } {
  const base = pagePath("models", { tab: "images" });
  if (!nSr) return { to: `${base}?set=records`, label: "Generate its SR in Models › Images" };
  const q = new URLSearchParams({ set: "records", split, ...(index != null ? { id: `${split}:${index}` } : {}) });
  return { to: `${base}?${q.toString()}`, label: "Open its SR in Models › Images" };
}
