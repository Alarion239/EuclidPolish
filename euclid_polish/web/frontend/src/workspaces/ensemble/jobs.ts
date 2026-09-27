/* Job helpers of the Ensemble workspace: the shared job keys (so the palette,
   Home's quick actions and these tabs never start the same job twice) and an
   end-of-job effect. Jobs start through `startJob` (app/RunActions), which
   confirms, registers the job under its key and toasts the start; the shell
   toasts the end. */
import { useEffect, useRef } from "react";
import { isTerminal, type Job } from "../../api/jobs";
import { invalidate } from "../../api/query";

export const JOB = {
  evaluate: "run:evaluate",        // == app/RunActions (same url → same key)
  knee: "run:knee",
  memberPsnr: "run:member-psnr",
  pull: "ensemble:pull",
  pullCheck: "ensemble:pull-check",
  compare: "ensemble:compare",
  fit: "ensemble:gate-fit",
  promote: "ensemble:gate-promote",
  restore: "ensemble:restore",
  archive: "ensemble:archive",
} as const;

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
