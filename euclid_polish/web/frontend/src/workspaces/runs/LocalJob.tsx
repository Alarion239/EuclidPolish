/* One local job of this server (C2): its state, progress while it runs,
 * the full searchable log, cancel (confirmed) and the JSON result. The side
 * panel of a local job in Runs › Live and Runs › History. */
import { useState } from "react";
import { cancelJob, type Job } from "../../api/jobs";
import { useResource } from "../../api/query";
import { openInspector } from "../../app/inspector";
import { formatDateTime, formatDuration } from "../../format";
import {
  Badge, Button, Callout, Card, CardBody, CardHead, DefList, IconButton, JobProgress, JsonTree, LogView, Section,
  Skeleton, confirm, toast,
} from "../../ui";
import { localJobTone } from "./model";
import "./runs.css";

export async function askCancel(job: Job): Promise<void> {
  if (!(await confirm({ title: `Cancel “${job.label}”?`, message: "The job stops at its next progress tick; partial output stays.",
    tone: "danger", confirmLabel: "Cancel job" }))) return;
  const r = await cancelJob(job.job_id);
  if (r.ok) toast.info("Cancel requested"); else toast.error(r.error ?? "cancel refused");
}

export function LocalJobCard({ id, stored }: { id: string; stored: Job | undefined }) {
  const [poll, setPoll] = useState(() => !stored || stored.status === "running");
  const detail = useResource<Job>(`/api/jobs/${encodeURIComponent(id)}`, [id], { ttl: poll ? 1_000 : 60_000, poll: poll ? 2_000 : undefined });
  const job = detail.data ?? stored ?? null;
  const running = (detail.data?.status ?? stored?.status) === "running";
  if (running !== poll && !(detail.error?.status === 404)) setPoll(running);
  if (!job) {
    return detail.loading ? <Skeleton lines={6} /> : (
      <Callout tone="warn" title="Job not found">
        {detail.error?.status === 404 ? "The server no longer knows this job (restarted or evicted)." : detail.error?.message}
      </Callout>
    );
  }
  return (
    <Card className="runs-jobdetail">
      <CardHead title={job.label} sub={<code className="mono">{job.job_id}{job.kind ? ` · ${job.kind}` : ""}</code>}
        right={<div className="runs-row">
          <Badge tone={localJobTone(job.status)} dot={job.status === "running"}>{job.status}</Badge>
          <IconButton size="sm" icon="panelRight" label="Open in the inspector" onClick={() => openInspector({ kind: "job", id: `local/${job.job_id}` })} />
          {job.status === "running" && job.cancellable !== false && !job.cancel_requested && (
            <Button size="sm" variant="ghost" icon="stop" onClick={() => void askCancel(job)}>Cancel</Button>
          )}
        </div>} />
      <CardBody>
        <DefList dense items={[
          ["started", job.started ? formatDateTime(job.started) : "—"],
          job.finished ? ["finished", formatDateTime(job.finished)] : null,
          ["duration", formatDuration(job.duration)],
        ]} />
        <JobProgress job={{ ...job, log: null }} cancel={false} />
        <Section title="Log" sub={job.log_truncated ? "tail" : undefined}>
          <LogView text={job.log ?? ""} title={job.label} exportName={`job-${job.job_id}`} maxHeight="min(56vh, 620px)" />
        </Section>
        {job.result != null && (
          <Section title="Result" collapsible defaultOpen>
            <JsonTree data={job.result} expandDepth={1} />
          </Section>
        )}
      </CardBody>
    </Card>
  );
}

