/* Synthetic › Status and the realism verdicts shared by the ingredient tabs
   (pure; statusModel.test.ts): each row's one verdict number with its unit,
   the generation gate as words, the records tick, the last field statistics
   (a stale cache keeps its result, flagged) and the realised background
   noise per band. No React, no fetch. */
import type { Tone } from "../../ui";
import { formatCount, formatNumber, formatRelative } from "../../format";
import type {
  Band, Comparison, FieldComparison, GalaxyPayload, OverviewItem, OverviewPayload, PixelsPayload, RecordsTick, StarPayload,
} from "./api";
import { generatedDensity } from "./stars/model";

export type Verdict = { text: string; tone?: Tone };

/** The payloads a verdict may read (each optional: a row shows what it has). */
export type VerdictContext = {
  galaxies?: GalaxyPayload | null;
  stars?: StarPayload | null;
  pixels?: PixelsPayload | null;
};

/** Relative difference that turns a synthetic-vs-reference number warn-toned. */
export const TOLERANCE = { density: 0.05, noise: 0.1, power: 0.25 } as const;

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const sig3 = (v: number) => formatNumber(v, { sig: 3 });
const pct = (fraction: number) => `${Math.round(100 * fraction)}%`;
const bandShort = (band: string) => band.replace(/_E$/, "");
/** A signed display number with the typographic minus ("−0.15"), as the console writes negatives. */
export const minus = (text: string) => text.replace(/^-/, "\u2212");
/** "a–b" for a range of two-decimal values, one number when they agree. */
const range2 = (values: readonly number[]) => {
  const lo = Math.min(...values).toFixed(2), hi = Math.max(...values).toFixed(2);
  return lo === hi ? lo : `${lo}–${hi}`;
};
/** "Y", "Y and H", "Y, J and H". */
const bandList = (bands: readonly string[]) => listText(bands.map(bandShort));

/** "generated 5.03 vs prior 5.08 stars arcmin⁻²", warn-toned outside the tolerance. */
function versus(generated: number | null, prior: number | null, unit: string): Verdict | null {
  if (prior == null) return null;
  if (generated == null) return { text: `prior ${sig3(prior)} ${unit}` };
  const off = prior > 0 ? Math.abs(generated / prior - 1) > TOLERANCE.density : false;
  return { text: `generated ${sig3(generated)} vs prior ${sig3(prior)} ${unit}`, tone: off ? "warn" : undefined };
}

/** Generated vs prior galaxy surface density over the prior's whole range. */
export function galaxyDensities(galaxies: GalaxyPayload | null | undefined): { generated: number | null; prior: number | null } {
  const synthetic = galaxies?.sources.synthetic;
  const generated = synthetic?.available && synthetic.rows != null && synthetic.area_arcmin2
    ? synthetic.rows / synthetic.area_arcmin2 : null;
  return { generated, prior: num(galaxies?.calibration.candidate?.generation.surface_density_arcmin2) };
}

/** The last field statistics: the current cache, else the one built at an
 *  older schema (`stale`), else none. A current-schema cache is stale too
 *  when its inputs changed (the availability says so). */
export function lastComparison(p: PixelsPayload | null | undefined): { comparison: Comparison | null; stale: boolean } {
  if (!p) return { comparison: null, stale: false };
  if (p.comparison) return { comparison: p.comparison, stale: !p.availability.comparison_cache?.fresh };
  return { comparison: p.previous ?? null, stale: !!p.previous };
}

export type BandNoise = { band: Band; synthetic: number; real: number; ratio: number };

/** Realised background noise per band: the median over fields of the robust
 *  σ (1.4826 × MAD) of the synthetic and real LR, and their ratio. */
export function backgroundNoise(fields: FieldComparison): BandNoise[] {
  return fields.bands.flatMap((band) => {
    const synthetic = num(fields.summary.synthetic[band]?.robust_std?.median);
    const real = num(fields.summary.real[band]?.robust_std?.median);
    return synthetic != null && real != null && real > 0 ? [{ band, synthetic, real, ratio: synthetic / real }] : [];
  });
}

/** "VIS 1.00 · Y 0.88 · J 0.95 · H 0.86": the ratio per band (synthetic ÷ real). */
export const noiseRatiosText = (rows: readonly BandNoise[]) =>
  rows.map((r) => `${bandShort(r.band)} ${r.ratio.toFixed(2)}`).join(" · ");

/** Worst |ratio − 1| of the realised noise is outside the tolerance. */
export const noiseOff = (rows: readonly BandNoise[]) => rows.some((r) => Math.abs(r.ratio - 1) > TOLERANCE.noise);

/** The realised-noise verdict over every band: "1.00" when one band, else the
 *  range with the worst band named ("0.86–1.00, H worst"), warn-toned when any
 *  band is outside the tolerance; null without rows. */
export function noiseVerdict(rows: readonly BandNoise[]): Verdict | null {
  if (!rows.length) return null;
  const worst = rows.reduce((a, b) => (Math.abs(b.ratio - 1) > Math.abs(a.ratio - 1) ? b : a));
  const span = range2(rows.map((r) => r.ratio));
  const off = noiseOff(rows);
  const text = rows.length === 1 ? `${span} (${bandShort(worst.band)})` : off ? `${span}, ${bandShort(worst.band)} worst` : `${span} in every band`;
  return { text, tone: off ? "warn" : undefined };
}

export type ScaleScore = { band: Band; overlap: number; power: number; powerOff: boolean };

/** The power ratio (synthetic ÷ real) is outside the tolerance. */
export const powerOff = (ratio: number) => Math.abs(ratio - 1) > TOLERANCE.power;

/** The per-band scale-spectrum scores (medians), VIS first, and the bands
 *  whose power ratio is off (in band order); null without any score. */
export function scaleVerdict(fields: FieldComparison): { vis: ScaleScore | null; scores: ScaleScore[]; off: ScaleScore[] } | null {
  const scores = fields.bands.flatMap((band): ScaleScore[] => {
    const s = fields.scale_similarity[band];
    const overlap = num(s?.overlap?.median), power = num(s?.variance_ratio?.median);
    return overlap != null && power != null ? [{ band, overlap, power, powerOff: powerOff(power) }] : [];
  });
  if (!scores.length) return null;
  const off = scores.filter((s) => s.powerOff);
  return { vis: scores.find((s) => s.band === "VIS") ?? null, scores, off };
}

/** One Status row's verdict: a number with its unit, or null. */
export function rowVerdict(item: OverviewItem, ctx: VerdictContext = {}, now = Date.now()): Verdict | null {
  const f = item.facts;
  switch (item.id) {
    case "galaxy-model": {
      const d = galaxyDensities(ctx.galaxies);
      return versus(d.generated, d.prior ?? num(f.surface_density_arcmin2), "galaxies arcmin⁻²");
    }
    case "star-prior": {
      const c = ctx.stars?.distribution?.density_comparison;
      return versus(c ? generatedDensity(c) : null, num(c?.model_density_arcmin2) ?? num(f.density_arcmin2), "stars arcmin⁻²");
    }
    case "noise-model": {
      const last = lastComparison(ctx.pixels);
      const rows = last.comparison ? backgroundNoise(last.comparison.fields) : [];
      const v = noiseVerdict(rows);
      if (v) return { text: `background σ syn/real ${v.text}${last.stale ? " (last result)" : ""}`, tone: v.tone };
      const positions = num(f.positions);
      return positions ? { text: `${formatCount(positions)} measured Q1 positions` } : null;
    }
    case "psf": {
      // What the last generation run recorded (its records' provenance) beats the synced ePSFs.
      const recorded = f.records_psf_kinds as Record<string, string> | null | undefined;
      if (recorded && Object.keys(recorded).length) {
        const gauss = Object.entries(recorded).filter(([, k]) => k !== "empirical").map(([b]) => b);
        const n = Object.keys(recorded).length;
        return gauss.length
          ? { text: `records used the Gaussian fallback in ${gauss.join(", ")}`, tone: "warn" }
          : { text: `records used empirical ePSFs in ${n} of ${n} bands` };
      }
      const empirical = (f.empirical as string[] | undefined) ?? [];
      const fallback = (f.gaussian_fallback as string[] | undefined) ?? [];
      const uncached = (f.not_cached as string[] | undefined) ?? [];
      const total = empirical.length + fallback.length + uncached.length;
      if (!total) return null;
      if (fallback.length) return { text: `Gaussian fallback in ${fallback.join(", ")}`, tone: "warn" };
      if (!uncached.length) return { text: `empirical in ${total} of ${total} bands` };
      return { text: empirical.length ? `empirical in ${empirical.join(", ")}; ${uncached.join(", ")} not synced here` : "not synced here" };
    }
    case "tng-radii": {
      const valid = num(f.valid_count), expected = num(f.expected_count);
      if (valid == null || expected == null) return null;
      return { text: `${formatCount(valid)} of ${formatCount(expected)} radii valid`, tone: valid < expected ? "warn" : undefined };
    }
    case "saturation": {
      const base = num(f.base_probability), bright = num(f.bright_probability);
      const ramp = f.ramp_well_ratios as number[] | undefined;
      if (base == null || bright == null || !ramp || ramp.length !== 2) return null;
      return { text: `blackout ${pct(base)} → ${pct(bright)} of cores from ${formatNumber(ramp[0])}× to ${formatNumber(ramp[1])}× the well` };
    }
    case "training-catalog": {
      const all = num(f.population_fields_with_training), shown = num(f.population_fields);
      return f.cached && all != null && shown != null && all > shown ? { text: `${formatCount(all - shown)} training fields` } : null;
    }
    case "galaxy-plots": {
      const at = f.built_at;
      return typeof at === "string" && at ? { text: `built ${formatRelative(at, now)}` } : null;
    }
    case "comparison-cache": {
      const last = lastComparison(ctx.pixels);
      const v = last.comparison ? scaleVerdict(last.comparison.fields) : null;
      if (!v) return null;
      const text = v.off.length
        ? `power syn/real ${range2(v.off.map((s) => s.power))} in ${bandList(v.off.map((s) => s.band))}`
        : v.vis ? `VIS overlap ${v.vis.overlap.toFixed(2)}, power within ${pct(TOLERANCE.power)} in every band` : null;
      return text ? { text: `${text}${last.stale ? " (last result)" : ""}`, tone: v.off.length ? "warn" : undefined } : null;
    }
    default:
      return null;
  }
}

/** The Status groups: what generation reads, then the diagnostic caches
 *  (a row the server did not group reads as a generation input). */
export function statusGroups(items: readonly OverviewItem[]): { generation: OverviewItem[]; diagnostic: OverviewItem[] } {
  return {
    generation: items.filter((i) => i.group !== "diagnostic"),
    diagnostic: items.filter((i) => i.group === "diagnostic"),
  };
}

/** The records tick of a row in words: its short label, tone and tooltip. */
export function recordsTickText(tick: RecordsTick | null | undefined): { label: string; tone: Tone; tip: string } | null {
  if (!tick) return null;
  if (tick.state === "current") return { label: "records built with it", tone: "good", tip: tick.detail };
  if (tick.state === "predates") return { label: "records predate it", tone: "warn", tip: tick.detail };
  return { label: "records unverified", tone: "neutral", tip: tick.detail };
}

/** The gate as words: "Ready to generate" or "Blocked by N", and the rows
 *  whose ingredient the local records predate (regenerate to use them). */
export function gateSummary(data: OverviewPayload): { ready: boolean; headline: string; blockers: string[]; predates: string[] } {
  const g = data.gate;
  const blockers = g.blockers.map((b) => b.message);
  const predates = data.items.filter((i) => i.records?.state === "predates").map((i) => i.label);
  const headline = g.ready ? "Ready to generate" : `Blocked by ${g.blockers.length}`;
  return { ready: g.ready, headline, blockers, predates };
}

/** "the stellar prior and the noise model" style list of row labels. */
export function listText(labels: readonly string[]): string {
  if (labels.length <= 1) return labels.join("");
  return `${labels.slice(0, -1).join(", ")} and ${labels[labels.length - 1]}`;
}
