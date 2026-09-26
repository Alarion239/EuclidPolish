/* Local background jobs — compatibility module for the pages written before
   the foundation rework. The hooks live in the data layer (`api/jobs.ts`:
   useJob, useTrackedJob, useJobsFeed, cancelJob) and the progress panel in
   the UI kit (`ui/JobProgress.tsx`: JobProgress, compat JobProgressView).
   New code imports from those modules directly. */
export { cancelJob, isTerminal, useJob, useJobsFeed, useTrackedJob } from "./api/jobs";
export type { Job, JobProgress, JobStatus, RunOpts, UseJob } from "./api/jobs";
export { JobProgressView } from "./ui";
