/* Realism jobs: one key per job kind (the palette, the overview checklist and
   the tabs never start the same job twice; a remounted tab re-attaches to it),
   started through app/RunActions' `startJob` (confirm → POST → register →
   toast), followed with `useJob(key)`. The shell toasts the end. */
import { useEffect, useRef } from "react";
import { isTerminal, useJob, type Job } from "../../api/jobs";
import { invalidate } from "../../api/query";
import { startJob } from "../../app/RunActions";
import type { ItemAction } from "./api";

export const JOB = {
  galaxyQuery: "realism:galaxy-query",
  galaxyCones: "realism:galaxy-cones",
  galaxyBuild: "realism:galaxy-build",
  galaxyActivate: "realism:galaxy-activate",
  starQuery: "realism:star-query",
  starFit: "realism:star-fit",
  starActivate: "realism:star-activate",
  pixelsBuild: "realism:pixels-build",
  archiveSync: "realism:archive-sync",
  trainingSync: "realism:training-sync",
  tngRadii: "realism:tng-radii",
} as const;
export type JobKey = (typeof JOB)[keyof typeof JOB];

/** The resources every Realism job can change. */
export const REALISM_PREFIXES = [
  "/api/realism/", "/api/galaxy-distributions", "/api/star-distribution", "/api/population-comparison",
  "/viewer/meta/archive-fields", "/api/archive-fields", "/api/tng/radii",
];

export function invalidateRealism() {
  for (const prefix of REALISM_PREFIXES) void invalidate(prefix);
}

/** The job key an endpoint's action runs under (so an overview action and
 *  the tab's own button share one slot). */
const KEY_BY_URL: Record<string, JobKey> = {
  "/api/galaxy-distributions/query-q1-counts": JOB.galaxyQuery,
  "/api/galaxy-distributions/refresh-population-cones": JOB.galaxyCones,
  "/api/galaxy-distributions/build": JOB.galaxyBuild,
  "/api/galaxy-distributions/activate": JOB.galaxyActivate,
  "/api/star-distribution/query": JOB.starQuery,
  "/api/star-distribution/fit": JOB.starFit,
  "/api/star-distribution/activate": JOB.starActivate,
  "/api/population-comparison/build": JOB.pixelsBuild,
  "/api/archive-fields/sync": JOB.archiveSync,
  "/api/population-comparison/sync-training-catalog": JOB.trainingSync,
  "/api/tng/radii/refresh": JOB.tngRadii,
};

export const keyForUrl = (url: string): string => KEY_BY_URL[url] ?? `realism:${url}`;

/** Start one Realism job; `question` confirms first (remote or destructive). */
export function runJob(spec: {
  url: string; label: string; data?: Record<string, string>;
  question?: { title: string; message: string; confirmLabel: string };
}): Promise<string | null> {
  return startJob({ key: keyForUrl(spec.url), url: spec.url, label: spec.label, data: spec.data, question: spec.question });
}

/** Run a server-described action (overview items, the training sync). */
export function runAction(action: ItemAction, label = action.label): Promise<string | null> {
  return runJob({
    url: action.url, label, data: action.params,
    question: action.confirm ? { title: `${label}?`, message: action.confirm, confirmLabel: action.label } : undefined,
  });
}

/** An action while FASRC is offline: a gated endpoint (`requires_fasrc`)
 *  is disabled; a self-connecting job stays enabled and says it connects
 *  first (it reports a failed connection in the job). */
export function offlinePolicy(
  action: { requires_fasrc?: boolean; self_connects?: boolean }, offline: boolean,
): { disabled: boolean; hint: string | null } {
  if (!offline) return { disabled: false, hint: null };
  if (action.requires_fasrc) return { disabled: true, hint: "FASRC is offline" };
  if (action.self_connects) return { disabled: false, hint: "FASRC is offline: the job connects first" };
  return { disabled: false, hint: null };
}

/** Follow a keyed job; `onEnd` runs once when it leaves "running", after
 *  every Realism resource is refetched. */
export function useRealismJob(key: JobKey, onEnd?: (job: Job) => void) {
  const job = useJob(key);
  const seen = useRef<string | null>(null);
  const cb = useRef(onEnd);
  cb.current = onEnd;
  const current = job.job;
  useEffect(() => {
    if (!current) return;
    if (current.status === "running") { seen.current = current.job_id; return; }
    if (isTerminal(current.status) && seen.current === current.job_id) {
      seen.current = null;
      invalidateRealism();
      cb.current?.(current);
    }
  }, [current]);
  return job;
}
