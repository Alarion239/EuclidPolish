/* Data workspace — shared pieces: the sticky tab toolbar, freshness badges,
 * load/error states that show the server's text, sky links, and the job
 * starter every sync / generate button uses (confirm → POST → job tray →
 * invalidate the data resources → toast). */
import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { invalidate } from "../../api/query";
import type { Job, UseJob } from "../../api/jobs";
import type { FormRecord } from "../../api/client";
import { useFasrcStatus } from "../../app/status";
import { formatDateTime, formatRelative } from "../../format";
import { Badge, Button, Callout, JobProgress, Skeleton, Tooltip, confirm, toast, type Tone } from "../../ui";
import { DATA_PREFIXES } from "./api";
import { atlasHref } from "./model";

/** The tab's sticky toolbar (under the workspace tab strip). */
export function DataBar({ label, children }: { label: string; children: ReactNode }) {
  return <div className="dt-bar" role="toolbar" aria-label={label}>{children}</div>;
}

export const Spacer = () => <span className="dt-bar__spacer" />;

/** A labelled cluster inside the toolbar. */
export function BarGroup({ label, children }: { label: string; children: ReactNode }) {
  return <div className="dt-bar__group" role="group" aria-label={label}>{children}</div>;
}

/** "synced 3 h ago" with the absolute time in a tooltip. */
export function Freshness({ at, label = "synced", tone, stale = 7 * 24 * 3600 }: {
  at: number | null | undefined; label?: string; tone?: Tone; stale?: number;
}) {
  if (at == null) return <Badge size="sm">never {label}</Badge>;
  const age = Date.now() / 1000 - at;
  return (
    <Tooltip content={`${label} ${formatDateTime(at * 1000)}`}>
      <span tabIndex={0} className="dt-fresh">
        <Badge size="sm" tone={tone ?? (age > stale ? "warn" : "neutral")}>{label} {formatRelative(at * 1000)}</Badge>
      </span>
    </Tooltip>
  );
}

/** Loading skeleton, or the server's error with a retry, else the children. */
export function LoadState({ loading, error, onRetry, lines = 4, children }: {
  loading: boolean; error: { message: string } | null | undefined; onRetry?: () => void; lines?: number; children: ReactNode;
}) {
  if (loading) return <Skeleton lines={lines} />;
  if (error) {
    return (
      <Callout tone="bad" title="Could not load" action={onRetry ? <Button size="sm" onClick={onRetry}>Retry</Button> : undefined}>
        <span className="dt-pre">{error.message}</span>
      </Callout>
    );
  }
  return <>{children}</>;
}

/** Router link to the Sky atlas (a small ghost button). */
export function SkyButton({ ra, dec, fov = 0.05, layers, inspect, label = "On sky", hint }: {
  ra?: number | null; dec?: number | null; fov?: number; layers?: string[]; inspect?: string; label?: string; hint?: string;
}) {
  const link = (
    <Button asChild size="sm" variant="ghost" icon="globe">
      <Link to={atlasHref({ ra, dec, fov, layers, inspect })}>{label}</Link>
    </Button>
  );
  return hint ? <Tooltip content={hint}>{link}</Tooltip> : link;
}

/** FASRC connection state for gating the remote actions. */
export function useFasrcOnline(): { online: boolean; known: boolean } {
  const s = useFasrcStatus().data;
  return { online: !!s?.ssh_connected, known: s != null };
}

export const OFFLINE_HINT = "Needs the FASRC connection (Settings › Connections).";

export type StartOpts = {
  /** The confirm dialog; omit for no confirmation. */
  question?: { title: string; message: string; confirmLabel: string; tone?: "danger" | "default" };
  label: string;
  onDone?: (job: Job) => void;
};

/** Confirm, POST a job endpoint through `job` (tray + re-attach by key), then
 *  refresh every Data resource and toast the outcome. */
export async function startDataJob(job: UseJob, url: string, data: FormRecord, opts: StartOpts): Promise<void> {
  if (opts.question && !(await confirm(opts.question))) return;
  await job.run(url, data, {
    onDone: (j) => {
      for (const prefix of DATA_PREFIXES) void invalidate(prefix);
      if (j.status === "done") toast.success(`${opts.label}: done`);
      else if (j.status === "failed") toast.error(`${opts.label}: failed`, { description: (j.error ?? "").split("\n")[0] });
      else if (j.status === "cancelled") toast.warning(`${opts.label}: cancelled`);
      opts.onDone?.(j);
    },
  });
}

/** A running / just-finished job under the toolbar (hidden when idle). */
export function JobStrip({ job }: { job: UseJob }) {
  if (!job.job && !job.error) return null;
  return (
    <div className="dt-jobstrip">
      <JobProgress job={job.job} error={job.error} />
      {job.job && job.job.status !== "running" && (
        <Button size="sm" variant="ghost" icon="close" onClick={job.reset} aria-label="Dismiss the job">Dismiss</Button>
      )}
    </div>
  );
}
