/* Job helpers of the Models workspace: the shared job keys (so the palette
   and these tabs never start the same job twice), the
   confirmed starters of the loop's local jobs (evaluate, PSNR vs knee,
   member PSNR) and an end-of-job effect. Jobs start through `startJob`
   (app/RunActions), which confirms, registers the job under its key and
   toasts the start; the shell toasts the end. */
import { useEffect, useRef } from "react";
import { isTerminal, type Job } from "../../api/jobs";
import { invalidate } from "../../api/query";
import { startJob } from "../../app/RunActions";
import { confirm } from "../../ui";
import { REGIME } from "./api";

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
} as const;

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
