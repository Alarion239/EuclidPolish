/* Job helpers of the Models workspace: the shared job keys (so the palette
   and these tabs never start the same job twice), the confirmed starters of
   its local jobs (the universal models' score, spectra and pull, a
   knee-attribution run, a pixel-types analysis or comparison), the
   confirmed production switch (an instant POST, no job) and an end-of-job
   effect. Jobs start through `startJob` (app/RunActions), which confirms,
   registers the job under its key and toasts the start; the shell toasts
   the end. */
import { useEffect, useRef } from "react";
import { apiPost } from "../../api/client";
import { isTerminal, type Job } from "../../api/jobs";
import { invalidate } from "../../api/query";
import { startJob } from "../../app/RunActions";
import { confirm, toast } from "../../ui";
import { REGIME, url } from "./api";
import { kneeUrl } from "./kneesApi";
import { numbersText } from "./universal";

export const JOB = {
  restore: "ensemble:restore",
  archive: "ensemble:archive",
  universalScore: "ensemble:universal-score",
  universalPull: "ensemble:universal-pull",
  kneeAttribution: "models:knee-attribution",
  /** A pixel-types analysis or comparison (one at a time: both load the run's fields). */
  kneePixelTypes: "models:knee-pixel-types",
} as const;

/** The start of the universal score job's label (routes/ensemble.py), so a
 *  page follows that job whoever started it (api/jobs useTrackedJob). */
export const UNIVERSAL_SCORE_LABEL = "universal: score";

const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;
const numbers = (names: readonly string[]) => names.map((n) => n.replace(/^member_/, "")).join(", ");

/** Score universal models on the test fields, the Leaderboard's ∫PSNR
 *  (confirmed). `names`: those models only; none = every universal model
 *  whose score is missing or stale. `force` re-scores them even when their
 *  score is current. */
export function scoreUniversal(names: readonly string[], nFields: number, force = false) {
  const what = names.length ? plural(names.length, "universal model") : force ? "every universal model" : "the unscored universal models";
  const verb = force ? "Re-score" : "Score";
  return startJob({
    key: JOB.universalScore, url: url.universalScore(),
    label: `${verb} ${what}`,
    data: { mode: REGIME, ...(names.length ? { members: names.join(",") } : {}), force: force ? "1" : "0" },
    question: {
      title: `${verb} ${what} on the ${nFields} test fields?`,
      message: `Loads ${names.length ? numbers(names) : "each model"} (TensorFlow) and scores its PSNR at every knee against the blurred HR truth: the Leaderboard's metric. It writes only the universal scores, never the ensemble's cubes. Minutes per model.`,
      confirmLabel: verb,
    },
  });
}

/** Compute the universal models' spectra (Models › Spectra; confirmed): the
 *  universal score job with `spectra=1`, which runs every universal model
 *  whose score or spectra are missing or stale. `missing`: the models the
 *  confirm names; `estimate`: its cost line (spectra.ts spectraEstimate). */
export function computeSpectra(missing: readonly string[], nFields: number, estimate: string) {
  return startJob({
    key: JOB.universalScore, url: url.universalScore(), label: "Compute spectra",
    data: { spectra: "1" },
    question: {
      title: `Compute the spectra of ${plural(missing.length, "universal model")}?`,
      message: `${estimate.charAt(0).toUpperCase()}${estimate.slice(1)}. The universal score job loads ${numbersText(missing)} on this machine (TensorFlow, never FASRC) and measures their power spectra and direction diagnostics on the ${nFields} test fields; a model without a current score is scored too (minutes per model). It writes only the universal scores and spectra.`,
      confirmLabel: "Compute",
    },
  });
}

/** Pull universal models from FASRC as standalone models (confirmed).
 *  `again`: registered ones whose last pull left no checkpoint here;
 *  `state`: a job that ended without COMPLETED / TIMEOUT (a try: FASRC may
 *  hold no checkpoint, and then nothing is registered). */
export function pullUniversal(names: readonly string[], opts: { again?: boolean; state?: string | null } = {}) {
  const what = plural(names.length, "universal model");
  const message = opts.again
    ? `Copies ${numbers(names)} from FASRC again: ${names.length === 1 ? "it is" : "they are"} registered as universal ${names.length === 1 ? "model" : "models"}, but the last pull left no checkpoint here.`
    : `Registers ${numbers(names)} as universal models and copies ${names.length === 1 ? "its checkpoint" : "their checkpoints"} here. They stay out of the ensemble: the gates, the plain mean and Evaluate never read them.`;
  const tried = opts.state ? ` ${names.length === 1 ? "Its job" : "Their jobs"} ended ${opts.state}: when FASRC holds no checkpoint the pull refuses and registers nothing.` : "";
  return startJob({
    key: JOB.universalPull, url: url.universalPull(),
    label: `${opts.again ? "Pull again" : "Pull"} ${what}`,
    data: { members: names.join(",") },
    question: {
      title: `${opts.again ? "Pull again" : "Pull"} ${what} from FASRC?`,
      message: message + tried,
      confirmLabel: opts.again ? "Pull again" : "Pull",
    },
  });
}

/** Start a knee-attribution run (Models › Knees; confirmed): `fields` are the
 *  POSTed form (knees.ts runFormData), `estimate` the confirm's cost line
 *  (knees.ts estimateText). Resolves the job id, or null when cancelled or
 *  refused (the refusal is toasted). */
export function startKneeAttribution(fields: Record<string, string>, estimate: string) {
  return startJob({
    key: JOB.kneeAttribution, url: kneeUrl.start(), label: "Knee attribution",
    data: fields,
    question: {
      title: "Run knee attribution?",
      message: `${estimate.charAt(0).toUpperCase()}${estimate.slice(1)}. Loads each model (TensorFlow) and back-propagates every sampled pixel through it on this machine (no FASRC); it writes a new run and changes nothing else.`,
      confirmLabel: "Run",
    },
  });
}

/** Analyse a run's pixel types (confirmed): `fields` = {run_id, models?}. */
export function startPixelTypes(fields: Record<string, string>, estimate: string) {
  return startJob({
    key: JOB.kneePixelTypes, url: kneeUrl.pixelAnalyse(), label: "Pixel types",
    data: fields,
    question: {
      title: "Analyse the pixel types?",
      message: `${estimate.charAt(0).toUpperCase()}${estimate.slice(1)}. Reads the run's samples and stored fields and draws the figures on this machine (no model is loaded); it rewrites this run's pixel-types folder and changes nothing else.`,
      confirmLabel: "Analyse",
    },
  });
}

/** Compare two models on the same pixels (confirmed): `fields` = {a, b, tag?}. */
export function startPixelCompare(fields: Record<string, string>, what: string) {
  return startJob({
    key: JOB.kneePixelTypes, url: kneeUrl.pixelCompare(), label: "Pixel types: compare",
    data: fields,
    question: {
      title: `Compare ${what}?`,
      message: "Local job, ~2 min. Pairs the pixels both runs sampled and draws the comparison on this machine (no model is loaded); it writes one comparison folder and changes nothing else.",
      confirmLabel: "Compare",
    },
  });
}

/** The reads that name production: the catalogue and the Universal payload
 *  (Home reads both). */
const PRODUCTION_READS = [url.models(), url.universal()];

/** Make a registered universal model production — `target` its spec and
 *  label (universal:member_257, "Universal 257") — after a confirm. An
 *  instant POST (no job, nothing deleted); then every read that names
 *  production is refreshed. `current`: what production is now, in words
 *  ("Universal 257"). Resolves true when production changed; a refusal (the
 *  server's 400) is toasted. */
export async function setProduction(target: { spec: string; label: string }, current: string): Promise<boolean> {
  const ok = await confirm({
    title: `Make ${target.label} production?`,
    message: `Every page that defaults to “production” — Sky, the NEXUS plates, the figures and inference — will run ${target.label} instead of ${current}. The production outputs made so far (real SR tiles, plates, figures) turn stale until they are run again. Instant: no job runs and nothing is deleted.`,
    confirmLabel: "Make production",
  });
  if (!ok) return false;
  try {
    const r = await apiPost<{ ok?: boolean; error?: string }>(url.production(), { model: target.spec });
    if (r.ok === false || r.error) throw new Error(r.error ?? "refused");
  } catch (e) {
    toast.error("Production: not changed", { description: e instanceof Error ? e.message : String(e) });
    return false;
  }
  toast.success(`${target.label} is production`);
  await Promise.all(PRODUCTION_READS.map((u) => invalidate(u)));
  return true;
}

/** Run `onEnd` once when `job` goes from running to a terminal state (and
 *  refresh every /ensemble/ resource). */
export function useOnJobEnd(job: Job | null, onEnd?: (job: Job) => void) {
  const seen = useRef<string | null>(null);
  const cb = useRef(onEnd);
  cb.current = onEnd;
  useEffect(() => {
    if (!job) return;
    if (job.status === "running") { seen.current = job.job_id; return; }
    if (isTerminal(job.status) && seen.current === job.job_id) {
      seen.current = null;
      void invalidate("/ensemble/");
      cb.current?.(job);
    }
  }, [job]);
}
