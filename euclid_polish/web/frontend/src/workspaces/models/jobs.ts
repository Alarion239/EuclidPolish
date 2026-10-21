/* Job helpers of the Models workspace: the shared job keys (so the palette
   and these tabs never start the same job twice), the
   confirmed starters of the loop's local jobs (evaluate, PSNR vs knee,
   member PSNR, the universal models' score, spectra and pull, a
   knee-attribution run) and an end-of-job effect. Jobs start through
   `startJob` (app/RunActions), which confirms, registers the job under its
   key and toasts the start; the shell toasts the end. */
import { useEffect, useRef } from "react";
import { isTerminal, type Job } from "../../api/jobs";
import { invalidate } from "../../api/query";
import { startJob } from "../../app/RunActions";
import { confirm } from "../../ui";
import { REGIME, url } from "./api";
import { kneeUrl } from "./kneesApi";
import { numbersText } from "./universal";

export const JOB = {
  evaluate: "run:evaluate",        // == app/RunActions (same url → same key)
  knee: "run:knee",
  memberPsnr: "run:member-psnr",
  pull: "ensemble:pull",
  pullCheck: "ensemble:pull-check",
  mirror: "fasrc:mirror",          // == System › Storage's checkpoint pull
  compare: "ensemble:compare",
  fit: "ensemble:gate-fit",
  promote: "ensemble:gate-promote",
  restore: "ensemble:restore",
  archive: "ensemble:archive",
  generateSr: "data:generate-sr",  // == Synthetic › Records (same job)
  fieldDiagnostics: "sky:field-diagnostics",
  bandEvals: "ensemble:band-evals",
  universalScore: "ensemble:universal-score",
  universalPull: "ensemble:universal-pull",
  kneeAttribution: "models:knee-attribution",
} as const;

/** The start of the universal score job's label (routes/ensemble.py), so a
 *  page follows that job whoever started it (api/jobs useTrackedJob). */
export const UNIVERSAL_SCORE_LABEL = "universal: score";

/** Evaluate the ensemble on `n` test fields (confirmed; `asked`: the
 *  caller's own dialog already asked, so no second confirm). */
export function evaluate(n: number, force = false, opts: { asked?: boolean } = {}) {
  return startJob({
    key: JOB.evaluate, url: "/ensemble/evaluate",
    label: `Evaluate the ensemble${force ? " (forced)" : ""}`,
    data: { mode: REGIME, num_images: String(n), force: force ? "1" : "0" },
    question: opts.asked ? undefined : {
      title: `Evaluate the ensemble on ${n} test fields?`,
      message: force
        ? "Forced: every member is re-run (TensorFlow) even if an identical evaluation is cached. Several minutes."
        : "Loads every active member (TensorFlow) unless an identical evaluation is cached. Several minutes.",
      confirmLabel: "Evaluate",
    },
  });
}

/** Recompute the PSNR-vs-knee curves from the cached test cubes (confirmed). */
export const computeKnee = () => startJob({
  key: JOB.knee, url: "/ensemble/knee-psnr", label: "PSNR vs knee", data: { mode: REGIME },
  question: { title: "Recompute PSNR vs knee?", message: "Scores every member, the mean and each combiner at every knee from the cached test cubes.", confirmLabel: "Compute" },
});

/** Generate the production SR over the local synthetic records (the SR tier
 *  of Synthetic › Records and the angular power spectrum). Asks first; a
 *  danger confirm when the existing SR is deleted. */
export async function generateSr(subsets: readonly string[], overwrite: boolean) {
  const ok = await confirm({
    title: `Generate the production SR for ${subsets.join(" + ")}?`,
    message: `Loads every active member (TensorFlow) and the production gate${overwrite ? "; the existing SR of these splits is deleted first" : ""}.`,
    confirmLabel: "Generate", tone: overwrite ? "danger" : "default",
  });
  if (!ok) return null;
  return startJob({
    key: JOB.generateSr, url: "/api/sky/generate-sr", label: "Generate SR",
    data: { subsets: subsets.join(","), overwrite: overwrite ? "1" : "0" },
  });
}

/** Re-apply the newest combiners to the cached legacy real field
 *  and rewrite its diagnostics (Diagnostics › Real field; confirmed). */
export const refreshFieldDiagnostics = (fieldId: string) => startJob({
  key: JOB.fieldDiagnostics, url: "/inference/refresh-combiners", label: "Real-field diagnostics",
  question: {
    title: "Recompute the real-field diagnostics?",
    message: `Applies the newest combiners to field ${fieldId} (TensorFlow, local job; stale member SRs are rebuilt, the 100 sub-tiles re-downloaded only if the member cache is stale).`,
    confirmLabel: "Recompute",
  },
});

/** Compute the Y, J and H evaluation diagnostics from the cached test cubes
 *  (Diagnostics' band switch; confirmed). */
export const computeBandEvals = () => startJob({
  key: JOB.bandEvals, url: "/ensemble/evals/bands", label: "Y, J, H diagnostics", data: { mode: REGIME },
  question: {
    title: "Compute the Y, J and H diagnostics?",
    message: "Re-measures the spectra, the coherence and the spread in each NISP band from the cached test cubes (no model runs; a local job of several minutes). Evaluate keeps them current afterwards.",
    confirmLabel: "Compute",
  },
});

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

/** Measure the gates' member shares (Models › Universal's combiner hours)
 *  from the validate cubes: the score job with no models named, which
 *  measures the gates first, then scores whatever has no current score
 *  (`unscored`, named in the confirm). */
export function measureGateWeights(gates: readonly string[], unscored: readonly string[], nFields: number) {
  return startJob({
    key: JOB.universalScore, url: url.universalScore(),
    label: "Measure gate weights",
    data: { mode: REGIME, force: "0" },
    question: {
      title: `Measure the weights of ${plural(gates.length, "gate")}?`,
      message: `Reads ${gates.join(", ")} over the validate cubes (no model runs) for the members' gate shares, which set the combiners' training hours.${unscored.length
        ? ` The same job then scores ${plural(unscored.length, "universal model")} without a current score (${numbers(unscored)}) on the ${nFields} test fields (TensorFlow, minutes per model).`
        : ""}`,
      confirmLabel: "Measure",
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
    : `Registers ${numbers(names)} as universal models and copies ${names.length === 1 ? "its checkpoint" : "their checkpoints"} here. They stay out of the ensemble: the production gate, the plain mean and Evaluate never read them.`;
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

/** Score the members' test PSNR (changed or unscored ones only; confirmed). */
export const refreshMemberPsnr = () => startJob({
  key: JOB.memberPsnr, url: "/ensemble/member-psnr", label: "Re-score the members' test PSNR",
  question: { title: "Re-score the members' test PSNR?", message: "Only changed or unscored members are evaluated (TensorFlow).", confirmLabel: "Re-score" },
});

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
