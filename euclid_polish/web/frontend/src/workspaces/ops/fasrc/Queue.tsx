/* Ops › FASRC › Queue: the local fail-stop submission queue (one cluster job
 * at a time; a failure halts it). Remove one item, clear all, resume after
 * a halt. Reads the local `/api/fasrc/queue/state` (works offline). */
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { apiPost } from "../../../api/client";
import { refreshJobsFeed } from "../../../api/jobs";
import { invalidate, useResource } from "../../../api/query";
import { formatDateTime, formatRelative } from "../../../format";
import {
  Badge, Button, Callout, Card, CardBody, CardHead, DataTable, EmptyState, IconButton, Skeleton, confirm, toast,
  type DataColumn,
} from "../../../ui";
import { QUEUE_STATE_URL, type QueueItem, type QueueState } from "../api";

type Resp = { ok: boolean; queue: QueueState };

export function QueuePanel() {
  const res = useResource<Resp>(QUEUE_STATE_URL, [], { ttl: 5_000, poll: 10_000 });
  const [busy, setBusy] = useState<string | null>(null);
  const q = res.data?.queue;

  async function act(url: string, body: Record<string, string>, done: string) {
    setBusy(url);
    try {
      await apiPost(url, body);
      toast.success(done);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
      void invalidate(QUEUE_STATE_URL);
      void refreshJobsFeed();
    }
  }
  const remove = async (it: QueueItem) => {
    if (await confirm({ title: `Remove “${it.label}” from the queue?`, message: "It will not be submitted.",
      tone: "danger", confirmLabel: "Remove" })) await act("/api/fasrc/queue/remove", { id: it.id }, `Removed “${it.label}”`);
  };
  const clear = async () => {
    if (await confirm({ title: `Clear all ${q?.count ?? 0} queued submissions?`, message: "Nothing queued will be submitted; the running job is not touched.",
      tone: "danger", confirmLabel: "Clear queue" })) await act("/api/fasrc/queue/clear", {}, "Queue cleared");
  };
  const resume = async () => {
    if (await confirm({ title: "Resume the queue?", message: "The next queued submission starts at the next poll, even though the previous job failed.",
      confirmLabel: "Resume" })) await act("/api/fasrc/queue/resume", {}, "Queue resumed");
  };

  const columns = useMemo<DataColumn<QueueItem>[]>(() => [
    { id: "position", header: "#", numeric: true, width: 48 },
    { id: "label", header: "Submission" },
    { id: "step", header: "Step", cell: (it) => <code className="mono">{it.step ?? "—"}</code> },
    { id: "queued_at", header: "Queued", width: 140,
      cell: (it) => <span className="ops-dim ops-small" title={formatDateTime(it.queued_at)}>{formatRelative(it.queued_at)}</span> },
    { id: "actions", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 48,
      cell: (it) => <IconButton size="sm" icon="close" label={`Remove ${it.label}`} disabled={busy != null}
        onClick={() => void remove(it)} /> },
  // eslint-disable-next-line react-hooks/exhaustive-deps
  ], [busy]);

  if (res.loading && !q) return <Skeleton lines={4} />;
  if (res.error && !q) return <Callout tone="bad" title="Could not read the queue">{res.error.message}</Callout>;
  return (
    <div className="ops-stack">
      {q?.halted && (
        <Callout tone="bad" title="The queue is halted" action={<Button size="sm" loading={busy === "/api/fasrc/queue/resume"} onClick={resume}>Resume</Button>}>
          {q.halted_reason || "A job failed; nothing further is submitted."}
        </Callout>
      )}
      <Card>
        <CardHead title="Submission queue" sub={q ? `${q.count} queued${q.active_jobid ? ` · lane: job ${q.active_jobid}` : ""}` : undefined}
          right={<div className="ops-row">
            {q?.active_jobid && <Button asChild size="sm" variant="ghost"><Link to={`/ops/fasrc?view=live&job=${q.active_jobid}`}>Active job</Link></Button>}
            <Button size="sm" variant="ghost" icon="close" disabled={!q?.count || busy != null} onClick={clear}>Clear</Button>
          </div>} />
        <CardBody>
          <DataTable rows={q?.items ?? []} columns={columns} rowKey={(it) => it.id} dense height="auto"
            aria-label="Queued submissions" hideToolbar={(q?.items.length ?? 0) < 8}
            empty={<EmptyState compact icon="layers" title="Nothing queued">
              A submit while a job runs waits here and starts when that job succeeds.
            </EmptyState>} />
          {q && !q.halted && q.count > 0 && <p className="ops-note"><Badge size="sm" tone="info">fail-stop</Badge> a failed or OOM job halts the queue.</p>}
        </CardBody>
      </Card>
    </div>
  );
}
