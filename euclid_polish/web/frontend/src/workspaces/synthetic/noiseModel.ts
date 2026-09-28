/* Synthetic › Noise: the tab's words and the stacked histogram tops (pure;
   noiseModel.test.ts). No React, no fetch. */
import { formatCount, formatDate } from "../../format";
import type { NoisePayload } from "./api";
import { stackedTops } from "./chartKit";

/** "EDF-N/S/F" for fields sharing a prefix, else the names joined. */
export function fieldsLabel(names: readonly string[]): string {
  const parts = names.map((n) => { const i = n.lastIndexOf("-"); return i > 0 ? [n.slice(0, i), n.slice(i + 1)] : [n, ""]; });
  const prefix = parts[0]?.[0];
  if (names.length > 1 && parts.every(([p, rest]) => p === prefix && rest)) return `${prefix}-${parts.map(([, rest]) => rest).join("/")}`;
  return names.join(", ");
}

/** The footer: "NOISE_MODEL v5 · Q1_R1 · retrieved 2026-09-19 · mer_noise_levels.json". */
export function noiseProvenance(payload: Pick<NoisePayload, "generator" | "source">): string {
  const version = /-v(\d+)$/.exec(payload.generator.noise_model)?.[1];
  const file = payload.source.table_path.split("/").pop() ?? payload.source.table_path;
  return [
    version ? `NOISE_MODEL v${version}` : `NOISE_MODEL ${payload.generator.noise_model}`,
    payload.source.release,
    `retrieved ${formatDate(payload.source.retrieved_last, { utc: true })}`,
    file,
  ].join(" · ");
}

const pct = (v: number) => `${Math.round(100 * v)}%`;

/** How a scene gets its noise, as three sentences (pick a position, vary the
 *  depth, draw the noise), from the generator's own settings. */
export function noiseSteps(payload: Pick<NoisePayload, "generator" | "source">): [string, string, string] {
  const { generator, source } = payload;
  const pick = generator.draws_measured_levels
    ? `Each scene takes the four band levels of one of the ${formatCount(source.position_count)} measured Q1 positions, picked uniformly.`
    : "Each scene takes the band median levels: the generator does not draw the measured positions.";
  const scale = generator.scene_scale;
  const region = generator.region;
  const depth = (scale ? `Its depth is scaled by ×${scale[0]}–${scale[1]}` : "Its depth is not jittered")
    + (region
      ? `, and in ${pct(region.probability)} of scenes a ${Math.round(100 * region.fraction[0])}–${pct(region.fraction[1])} strip steps ×${region.step[0]}–${region.step[1]}, like a pointing seam.`
      : ".");
  const draw = "The noise is σ = √(level² + signal) × scale, drawn on a dithered, resampled unit field.";
  return [pick, depth, draw];
}

/** The stacked bar tops of the fields still shown (a hidden field drops out
 *  of the stack), with the fields in stacking order (the top bar first). */
export function stackedTopsByField(
  countsByField: Record<string, number[]>, fields: readonly string[], hidden: readonly string[], bins: number,
): { tops: number[][]; order: string[] } {
  const visible = fields.filter((f) => !hidden.includes(f));
  const tops = stackedTops(visible.map((f) => countsByField[f] ?? []), bins);
  return { tops, order: [...visible].reverse() };
}
