/* The FASRC job records a campaign logged (GET /api/tracking/jobs: paged
 * server-side, without the embedded calibration blobs). `campaign` =
 * current | <archived dir> | unassigned. */
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useResource } from "../../api/query";
import { formatDateTime, formatRelative } from "../../format";
import {
  Badge, Callout, DataTable, IconButton, Input, Tooltip, type DataColumn,
} from "../../ui";
import { trackingJobsUrl, type TrackedJob, type TrackingJobsResp } from "./api";
import { paramsSummary } from "../runs/model";
import { commitText } from "./model";

const LIMIT = 100;

/** "k=v · k=v" of a job's first user-facing params: the Runs › History rule
 *  (defining knobs first; resources, array bookkeeping, seeds and
 *  `_`-prefixed internals left out). */
const paramsLine = (j: TrackedJob): string => paramsSummary(j.params, 5);

export function TrackedJobs({ campaign, compact = false }: { campaign: string; compact?: boolean }) {
  const [offset, setOffset] = useState(0);
  const [q, setQ] = useState("");
  const [needle, setNeedle] = useState("");
  const res = useResource<TrackingJobsResp>(trackingJobsUrl(campaign, offset, LIMIT, needle), [campaign, offset, needle], { ttl: 30_000 });
  const d = res.data;
  const columns = useMemo<DataColumn<TrackedJob>[]>(() => [
    { id: "logged_at", header: "Logged", width: 110,
      cell: (j) => <span className="nb-dim nb-small" title={formatDateTime(j.logged_at)}>{formatRelative(j.logged_at)}</span> },
    { id: "jobid", header: "Job", width: 92, cell: (j) => <code className="mono">{j.jobid}</code> },
    { id: "step_id", header: "Step", width: 150, cell: (j) => <code className="mono nb-small">{j.step_id ?? "—"}</code> },
    { id: "label", header: "Label", hidden: compact },
    { id: "params", header: "Params", accessor: paramsLine,
      cell: (j) => {
        const omitted = Object.keys(j.params_omitted ?? {});
        return <span className="nb-row">
          <span className="mono nb-small nb-ellipsis" title={paramsLine(j)}>{paramsLine(j) || "—"}</span>
          {omitted.length > 0 && <Tooltip content={`Embedded payloads not shown: ${omitted.join(", ")}`}>
            <span tabIndex={0}><Badge size="sm">+{omitted.length} blob{omitted.length === 1 ? "" : "s"}</Badge></span></Tooltip>}
        </span>;
      } },
    { id: "commit", header: "Commit", width: 90, accessor: (j) => commitText(j.commit),
      cell: (j) => <code className="mono">{commitText(j.commit)}</code> },
    { id: "actions", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 48,
      cell: (j) => <Tooltip content="Its run in Runs › History: resources and logs"><Link className="nb-iconlink" aria-label={`History of ${j.jobid}`}
        to={`/runs/history?run=${encodeURIComponent(j.jobid)}`}>≡</Link></Tooltip> },
  ], [compact]);
  if (res.error && !d) return <Callout tone="bad" title="Could not read the job records">{res.error.message}</Callout>;
  const last = d ? Math.min(d.total, offset + d.jobs.length) : 0;
  return (
    <DataTable rows={d?.jobs ?? []} columns={columns} rowKey={(j) => `${j.jobid}:${j.logged_at}`} dense
      aria-label="Campaign FASRC jobs" loading={res.loading && !d} height={compact ? 320 : 520}
      inspect={(j) => ({ kind: "job", id: `slurm/${j.jobid}` })} searchable={false} exportName={`tracking-jobs-${campaign}`}
      empty={needle ? "No job matches." : "No FASRC job was logged in this campaign."}
      toolbar={<>
        <form onSubmit={(e) => { e.preventDefault(); setOffset(0); setNeedle(q.trim()); }}>
          <Input size="sm" value={q} onChange={setQ} icon="search" clearable placeholder="Search jobs…" aria-label="Search jobs" />
        </form>
        <IconButton size="sm" icon="chevronLeft" label="Newer" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - LIMIT))} />
        <span className="mono nb-small nb-dim">{d ? `${d.total ? offset + 1 : 0}–${last} of ${d.total}` : "…"}</span>
        <IconButton size="sm" icon="chevronRight" label="Older" disabled={!d || last >= d.total} onClick={() => setOffset(offset + LIMIT)} />
      </>} />
  );
}
