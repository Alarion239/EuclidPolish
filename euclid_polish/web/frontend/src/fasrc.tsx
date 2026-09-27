/* FASRC pipeline components shared across workspaces — the public facade of
 * the Ops workspace's step and monitor modules (workspaces/ops/…):
 *
 *   <StepById stepId="euclid_query" />   the schema-driven step card (C5)
 *   <StepCard step={…} sshConnected />   the same for a step already loaded
 *   useStepsStatus()                     the step registry (/api/fasrc/steps/status)
 *   <SlurmMonitor jobid />               live SLURM job monitor (job:slurm/<id> inspector)
 *   <JobStatusBody status />, <TrainingCurve … />, jobStateTone(state)
 *   <ConnectionBar />, <CurrentSubmission />   connection toggle / live jobs panel
 *
 * Props stay backward compatible with the pre-rework module; import from
 * here (not from workspaces/ops) in other workspaces. */
export { StepById, StepCard, StepHistory, useStepsStatus } from "./workspaces/ops/steps/StepCard";
export type { Step, StepCardProps, StepDefaults, StepsStatus, TaskParam } from "./workspaces/ops/steps/StepCard";
export { JobStatusBody, SlurmMonitor, TrainingCurve } from "./workspaces/ops/steps/SlurmMonitor";
export type { SlurmStatus } from "./workspaces/ops/steps/SlurmMonitor";
export { jobStateTone } from "./workspaces/ops/model";
export { ConnectionBar } from "./workspaces/ops/fasrc/Connection";
export { LiveJobs as CurrentSubmission } from "./workspaces/ops/fasrc/Live";
