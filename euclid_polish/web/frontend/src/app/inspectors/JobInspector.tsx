/* Inspector kind `job` (registered by the shell):
 *   job:local/<job_id>  — a local background job: status, progress, cancel,
 *                         the full log (search / follow / copy) and its result;
 *   job:slurm/<jobid>   — a SLURM job: live stage, progress, resources,
 *                         warnings and errors (the FASRC monitor).
 */
import { useState } from "react";
import { useJobsStore, type Job } from "../../api/jobs";
import { useResource, type Resource } from "../../api/query";
import { SlurmMonitor } from "../../fasrc";
import { formatDateTime, formatDuration } from "../../format";
import { Callout, DefList, JobProgress, JsonTree, LogView, Section, Skeleton } from "../../ui";

/** Poll while the job runs: the freshest status (the detail, else the jobs
 *  feed) decides; unknown yet → poll until the first answer; a 404 (the server
 *  no longer knows the job, e.g. a stale shared link after a restart) or any
 *  other error with nothing to show stops it. */
function shouldPoll(detail: Resource<Job>, stored: Job | undefined): boolean {
  if ((detail.error ?? detail.staleError)?.status === 404) return false;
  const status = detail.data?.status ?? stored?.status;
  return status ? status === "running" : !detail.error;
}

function LocalJob({ id }: { id: string }) {
  const stored = useJobsStore((s) => s.jobs[id]) as Job | undefined;
  // Decided from the previous answer (state derived during render: a change
  // re-renders at once with the new polling options).
  const [poll, setPoll] = useState(() => !stored || stored.status === "running");
  const detail = useResource<Job>(`/api/jobs/${encodeURIComponent(id)}`, [], {
    ttl: poll ? 1_000 : 60_000, poll: poll ? 2_000 : undefined,
  });
  const next = shouldPoll(detail, stored);
  if (next !== poll) setPoll(next);
  const job = detail.data ?? stored ?? null;
  if (!job) {
    if (detail.loading) return <Skeleton lines={4} />;
    return (
      <Callout tone="warn" title="Job not found">
        {detail.error?.status === 404
          ? "The local server no longer knows this job (it may have restarted)."
          : detail.error?.message ?? "No data."}
      </Callout>
    );
  }
  return (
    <div className="insp-job">
      <DefList dense items={[
        ["job", <code className="mono">{job.job_id}</code>],
        job.kind ? ["kind", <code className="mono">{job.kind}</code>] : null,
        ["started", job.started ? formatDateTime(job.started) : "—"],
        job.finished ? ["finished", formatDateTime(job.finished)] : null,
        ["duration", formatDuration(job.duration)],
      ]} />
      <JobProgress job={{ ...job, log: null }} />
      <Section title="Log" sub={job.log_truncated ? "tail" : undefined}>
        <LogView text={job.log ?? ""} title={job.label} exportName={`job-${job.job_id}`} />
      </Section>
      {job.result != null && (
        <Section title="Result" collapsible defaultOpen>
          <JsonTree data={job.result} expandDepth={1} />
        </Section>
      )}
    </div>
  );
}

export function JobInspector({ id }: { id: string }) {
  const slash = id.indexOf("/");
  const source = slash > 0 ? id.slice(0, slash) : "local";
  const rest = slash > 0 ? id.slice(slash + 1) : id;
  if (source === "slurm") return <SlurmMonitor jobid={rest} />;
  if (source !== "local") return <Callout tone="warn" title="Unknown job source">{source}</Callout>;
  return <LocalJob id={rest} />;
}

export function jobTitle(id: string): string {
  const slash = id.indexOf("/");
  const source = slash > 0 ? id.slice(0, slash) : "local";
  const rest = slash > 0 ? id.slice(slash + 1) : id;
  if (source === "slurm") return `SLURM job ${rest}`;
  return useJobsStore.getState().jobs[rest]?.label ?? `Job ${rest}`;
}
