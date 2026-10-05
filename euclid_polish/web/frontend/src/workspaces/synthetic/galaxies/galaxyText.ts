/* Synthetic › Galaxies: the tab's words and gates (pure; synthetic.test.tsx) —
   magnitude ranges at one shared precision, whether the cached Q1 brackets
   are complete enough to refit without an archive query, and the one caption
   under each marginal panel (what it is normalised to, which samples it
   draws), built from the payload so no number is made up. */
import { formatCount, formatNumber } from "../../../format";
import type { GalaxyPayload } from "../api";

/** Magnitudes at one shared precision: integers stay "14–28", otherwise every
 *  value gets the fewest decimals (≤ 2) the least round one needs
 *  ("14.0–28.5", never "14–28.5"). */
export function magRange(...values: number[]): string {
  const need = (v: number) => (Math.abs(v - Math.round(v)) < 1e-6 ? 0 : Math.abs(v * 10 - Math.round(v * 10)) < 1e-6 ? 1 : 2);
  const digits = Math.max(...values.map(need));
  return values.map((v) => v.toFixed(digits)).join("–");
}

/** The Q1 aperture checkpoints and the Rₑ brackets are all cached, so the
 *  prior can be refitted without querying the archive. */
export function galaxyFitReady(api: Pick<GalaxyPayload, "q1_counts" | "q1_radius"> | null | undefined): boolean {
  const q1 = api?.q1_counts;
  const r = api?.q1_radius;
  if (!q1 || !r) return false;
  const done = q1.completed_queries ?? q1.query_count;
  const countsComplete = q1.complete !== false && (q1.total_queries == null || (done != null && done >= q1.total_queries));
  return countsComplete && r.complete && r.completed_queries >= r.total_queries;
}

export type MarginalCaptions = { brightness: string; radius: string; colours: string; shape: string };

/** "5,489 galaxies in 200 test + validate scenes (36.4 arcmin²)", or null. */
function generatedText(api: GalaxyPayload): string | null {
  const s = api.sources.synthetic;
  if (!s?.available || s.rows == null) return null;
  const splits = api.training_included ? "train + test + validate" : "test + validate";
  const scenes = s.fields ? ` in ${formatCount(s.fields)} ${splits} scenes` : ` in the ${splits} scenes`;
  const area = s.area_arcmin2 ? ` (${formatNumber(s.area_arcmin2, { sig: 3 })} arcmin²)` : "";
  return `${formatCount(s.rows)} galaxies${scenes}${area}`;
}

/** The caption under each distributions panel. */
export function marginalCaptions(api: GalaxyPayload): MarginalCaptions {
  const footprint = api.q1_counts?.footprint_area_deg2;
  const q1Area = footprint != null ? ` over the ${formatNumber(footprint, { digits: 1 })} deg² deep fields` : "";
  const generated = generatedText(api);
  const measured = api.sources.synthetic?.measured_radius_rows;
  return {
    brightness: [`Q1: PHZ-weighted MER brackets${q1Area}`, generated ? `generated: ${generated}` : null,
      "per arcmin² per magnitude"].filter(Boolean).join(" · "),
    radius: [
      "Circularized Sérsic Rₑ: the Q1 brackets, the generated galaxies' requested Rₑ"
        + (measured ? ` and ${formatCount(measured)} half-light radii measured on their clean images` : ""),
      "per arcmin² per dex",
    ].join(" · "),
    colours: "Q1: raw forced-photometry colours, measurement noise included; generated: the deconvolved draws, "
      + "so their scatter sits below Q1 at fixed depth · per arcmin² per magnitude",
    shape: "Each curve integrates to one over log radius; the dashed curve is the full faint extension the generator draws",
  };
}

/** How the joint views name the model's colours: noised with the Q1
 *  flux-ratio errors (forward noised, so contour widths compare with the raw
 *  Q1 colours) or, on a cache built before that, deconvolved. */
export function modelColourWording(noise?: { applied: boolean } | null): string {
  return noise?.applied ? "colours with Q1 measurement noise" : "deconvolved colours (an older build: rebuild the plots to add Q1 noise)";
}
