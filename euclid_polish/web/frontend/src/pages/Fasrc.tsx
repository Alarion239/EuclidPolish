/* FASRC — the SLURM ops console: connection, remote config + git, the live
   squeue, recent jobs with a live monitor + cancel, remote storage, and the
   granular pipeline steps (reused StepById cards). */
import { useState } from "react";
import { postForm } from "../api";
import { useResource, usePolling } from "../hooks";
import { ConnectionBar, SlurmMonitor, useHstStatus, StepCard } from "../fasrc";
import {
  Badge, Button, Card, CardBody, CardHead, DefList, Empty, LogTail, Page,
  PageHead, Spinner, Table, type Column,
} from "../ui";

type QueueRow = { jobid: string; name: string; state: string; time: string; nodes: string; reason?: string };
type QueueResp = { ok: boolean; rows: QueueRow[] };
type JobRow = { jobid: string; label: string; state: string; submitted_at?: number; started_at?: number; ended_at?: number; eta_seconds?: number };
type JobsResp = { jobs: JobRow[] };
type GitStatus = { ok: boolean; repo?: string; branch?: string; ahead?: number; behind?: number; last?: { hash: string; subject: string; relative: string } };
type DataListing = {
  ok: boolean; data_dir?: string; ckpt_dir?: string;
  du?: [string, string][]; tfrecords?: { path: string; size: string }[];
  checkpoints?: { path: string; size: string; mtime?: string }[];
};

const stateTone = (s: string): "good" | "warn" | "bad" | undefined =>
  s === "RUNNING" ? undefined : s === "PENDING" ? "warn"
    : s === "COMPLETED" || s === "DONE" ? "good"
    : s === "FAILED" || s === "CANCELLED" || s === "TIMEOUT" ? "bad" : undefined;

const QUEUE_COLS: Column<QueueRow>[] = [
  { header: "job", cell: (r) => <code className="mono">{r.jobid}</code> },
  { header: "name", cell: (r) => r.name },
  { header: "state", cell: (r) => <Badge tone={stateTone(r.state)}>{r.state}</Badge> },
  { header: "time", cell: (r) => <span className="mono">{r.time}</span>, align: "right" },
  { header: "nodes", cell: (r) => <span className="muted">{r.nodes || r.reason || "—"}</span> },
];

export default function FasrcPage() {
  const cfg = useResource<Record<string, string>>("/api/fasrc/config");
  const git = useResource<GitStatus>("/api/fasrc/git-status");
  const queue = useResource<QueueResp>("/api/fasrc/queue");
  const jobs = useResource<JobsResp>("/api/fasrc/jobs");
  const data = useResource<DataListing>("/api/fasrc/data-listing");
  const hst = useHstStatus();
  const [sel, setSel] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);

  usePolling(() => { queue.reload(); jobs.reload(); }, 10000);

  async function control(url: string, body: Record<string, string>, msg: string) {
    setBusy(url); setNote(null);
    try {
      const r = await postForm<{ ok?: boolean; error?: string }>(url, body);
      setNote(r.error ? `✗ ${r.error}` : `✓ ${msg}`);
      queue.reload(); jobs.reload();
    } catch (e) { setNote(`✗ ${e instanceof Error ? e.message : String(e)}`); }
    finally { setBusy(null); }
  }

  const jobCols: Column<JobRow>[] = [
    { header: "job", cell: (r) => <code className="mono">{r.jobid}</code> },
    { header: "label", cell: (r) => r.label },
    { header: "state", cell: (r) => <Badge tone={stateTone(r.state)}>{r.state}</Badge> },
    { header: "", align: "right", cell: (r) => (
      <div className="row" style={{ gap: 6, justifyContent: "flex-end" }}>
        <Button size="sm" onClick={() => setSel(sel === r.jobid ? null : r.jobid)}>{sel === r.jobid ? "hide" : "monitor"}</Button>
        <Button size="sm" variant="ghost" disabled={busy != null}
          onClick={() => control("/api/fasrc/cancel", { jobid: r.jobid }, `cancelled ${r.jobid}`)}>cancel</Button>
      </div>
    ) },
  ];

  return (
    <Page>
      <PageHead eyebrow="ops · fasrc" title="FASRC"
        sub="The SLURM cluster console: connection, remote code + storage, the live queue, and the training pipeline."
        right={<ConnectionBar />} />

      {note && <div className="job-panel" style={{ marginBottom: "var(--s4)" }}><LogTail text={note} /></div>}

      <div className="grid" style={{ gridTemplateColumns: "1fr", gap: "var(--s4)" }}>
        <div className="row" style={{ gap: "var(--s4)", alignItems: "stretch" }}>
          <Card style={{ flex: "1 1 360px" }}>
            <CardHead title="Remote config" />
            <CardBody>
              {cfg.loading ? <Empty><Spinner /> loading…</Empty>
                : <DefList items={Object.entries(cfg.data ?? {}).slice(0, 10).map(([k, v]) => [k, <code className="mono">{String(v)}</code>])} />}
            </CardBody>
          </Card>
          <Card style={{ flex: "1 1 360px" }}>
            <CardHead title="Remote git"
              right={<Button size="sm" disabled={busy != null}
                onClick={() => control("/api/fasrc/git-pull", {}, "pulled")}>git pull</Button>} />
            <CardBody>
              {git.data?.ok ? (
                <DefList items={[
                  ["branch", <code className="mono">{git.data.branch}</code>],
                  ["ahead / behind", `${git.data.ahead ?? 0} / ${git.data.behind ?? 0}`],
                  ["last", git.data.last ? <span><code className="mono">{git.data.last.hash}</code> {git.data.last.subject}</span> : "—"],
                ]} />
              ) : <Empty>connect to FASRC to read the remote repo</Empty>}
            </CardBody>
          </Card>
        </div>

        <Card>
          <CardHead title="Live queue" sub="squeue · refreshes every 10s"
            right={<Button size="sm" variant="ghost" onClick={() => queue.reload()}>↻</Button>} />
          <CardBody>
            <Table columns={QUEUE_COLS} rows={queue.data?.rows ?? []} rowKey={(r) => r.jobid}
              empty={queue.loading ? "loading…" : "queue empty"} />
          </CardBody>
        </Card>

        <Card>
          <CardHead title="Recent jobs" sub="last 30 submissions" />
          <CardBody>
            <Table columns={jobCols} rows={jobs.data?.jobs ?? []} rowKey={(r) => r.jobid}
              empty={jobs.loading ? "loading…" : "no jobs yet"} />
            {sel && <SlurmMonitor jobid={sel} />}
          </CardBody>
        </Card>

        <Card>
          <CardHead title="Remote storage" sub={data.data?.data_dir} />
          <CardBody>
            {data.loading ? <Empty><Spinner /> loading…</Empty> : (
              <>
                {!!data.data?.du?.length && (
                  <Table
                    columns={[
                      { header: "size", cell: (r: [string, string]) => <span className="mono">{r[0]}</span>, width: 90 },
                      { header: "path", cell: (r: [string, string]) => <code className="mono">{r[1]}</code> },
                    ]}
                    rows={data.data.du} rowKey={(r, i) => i} />
                )}
                <div className="muted" style={{ marginTop: "var(--s3)", fontSize: 12 }}>
                  {data.data?.tfrecords?.length ?? 0} tfrecords · {data.data?.checkpoints?.length ?? 0} checkpoints
                </div>
              </>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHead title="Pipeline steps" sub="submit a SLURM job for any stage" />
          <CardBody>
            {hst.loading ? <Empty><Spinner /> loading step registry…</Empty> : (
              <div className="grid" style={{ gap: "var(--s4)" }}>
                {(hst.data?.steps ?? []).map((s) => (
                  <StepCard key={s.step_id} step={s} sshConnected={!!hst.data?.ssh_connected} />
                ))}
              </div>
            )}
          </CardBody>
        </Card>
      </div>
    </Page>
  );
}
