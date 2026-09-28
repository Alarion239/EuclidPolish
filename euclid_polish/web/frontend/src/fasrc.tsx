/* FASRC pipeline components shared across workspaces — the public facade of
 * the Runs workspace's step and monitor modules (workspaces/runs/…):
 *
 *   <StepById stepId="euclid_query" />   the schema-driven step card (C5)
 *   <StepCard step={…} sshConnected />   the same for a step already loaded
 *   useStepsStatus()                     the step registry (/api/fasrc/steps/status)
 *   <SlurmMonitor jobid />               live SLURM job monitor (job:slurm/<id> inspector)
 *   <JobStatusBody status />, <TrainingCurve … />, jobStateTone(state)
 *   <ConnectionBar />, <CurrentSubmission />   connection toggle / live jobs panel
 *
 * Props stay backward compatible with the pre-rework module; import from
 * here (not from workspaces/runs) in other workspaces. */
export { StepById, StepCard, StepHistory, useStepsStatus } from "./workspaces/runs/steps/StepCard";
export type { Step, StepCardProps, StepDefaults, StepsStatus, TaskParam } from "./workspaces/runs/steps/StepCard";
export { JobStatusBody, SlurmMonitor, TrainingCurve } from "./workspaces/runs/steps/SlurmMonitor";
export type { SlurmStatus } from "./workspaces/runs/steps/SlurmMonitor";
export { jobStateTone } from "./workspaces/runs/model";
export { ConnectionBar } from "./workspaces/runs/Connection";
export { LiveJobs as CurrentSubmission } from "./workspaces/runs/tabs/Live";
