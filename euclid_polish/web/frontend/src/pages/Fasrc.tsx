/* FASRC — the SLURM ops console, as a classic-style tabbed dashboard:
   Current Submission · Git · Storage · Logs. Step-submit forms live on their
   own domain pages now (Train members, Sky, Cutouts, …); this page is purely
   monitoring + cluster ops. */
import { useEffect, useRef, useState } from "react";
import { postForm } from "../api";
import { asArray } from "../data";
import { useResource, usePolling } from "../hooks";
import { ConnectionBar, CurrentSubmission } from "../fasrc";
import {
  Badge, Button, Card, CardBody, CardHead, DefList, Empty, LogTail, Page,
  PageHead, Spinner, Table, Tabs, type Column,
} from "../ui";
import {
  buildLogPageUrl, hasRunLogs, logPath, preferredLogKind, type LogKind,
} from "./fasrcLogs";

type GitStatus = { ok: boolean; repo?: string; branch?: string; ahead?: number; behind?: number; last?: { hash: string; subject: string; relative: string } };
type DataListing = {
  ok: boolean; error?: string; data_dir?: string; ckpt_dir?: string;
  du?: [string, string][]; tfrecords?: { path: string; size: string }[];
  checkpoints?: { path: string; size: string; mtime?: string }[];
};
type RunRow = {
  name: string; jobid?: string | null; label?: string | null; state?: string | null;
  submitted_at?: number; out_size?: number; err_size?: number; missing?: boolean;
  out_path?: string | null; err_path?: string | null;
  array_count?: number;
  tasks?: RunTask[];
};
type RunTask = {
  index: number; member: string; jobid: string; name: string; state?: string | null;
  out_path?: string | null; err_path?: string | null;
  out_size?: number; err_size?: number; missing?: boolean;
};
type RunsResp = { ok: boolean; log_dir?: string; runs: RunRow[]; total_runs: number; page: number; page_size: number };
type LogResp = {
  ok: boolean; path: string; page: number; page_size: number;
  total_lines: number; start_line: number; end_line: number;
  has_older: boolean; has_newer: boolean; content: string;
};

const TABS = [
  { id: "current", label: "Current submission" },
  { id: "git", label: "Git" },
  { id: "storage", label: "Storage" },
  { id: "logs", label: "Logs" },
] as const;
type TabId = typeof TABS[number]["id"];

const stateTone = (s?: string | null): "good" | "warn" | "bad" | undefined =>
  s === "RUNNING" ? undefined : s === "PENDING" ? "warn"
    : s === "COMPLETED" || s === "DONE" ? "good"
    : s === "FAILED" || s === "CANCELLED" || s === "TIMEOUT" ? "bad" : undefined;

export default function FasrcPage() {
  const status = useResource<{ ssh_connected: boolean }>("/api/fasrc/status");
  const [tab, setTab] = useState<TabId>("current");
  const connected = !!status.data?.ssh_connected;

  return (
    <Page>
      <PageHead eyebrow="ops · fasrc" title="FASRC"
        sub="The SLURM cluster console — what's running, the remote repo, storage, and past-run logs. Submit jobs from their own pages."
        right={<ConnectionBar />} />

      <div style={{ marginBottom: "var(--s4)" }}>
        <Tabs<TabId> value={tab} tabs={TABS.map((t) => ({ id: t.id, label: t.label }))} onChange={setTab} />
      </div>

      {tab === "current" && <CurrentSubmission />}
      {tab === "git" && <GitTab connected={connected} />}
      {tab === "storage" && <StorageTab connected={connected} />}
      {tab === "logs" && <LogsTab connected={connected} />}
    </Page>
  );
}

/* ── Git ─────────────────────────────────────────────────────────────────── */
function GitTab({ connected }: { connected: boolean }) {
  const git = useResource<GitStatus>(connected ? "/api/fasrc/git-status" : null, [connected]);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  async function pull() {
    setBusy(true); setNote(null);
    try {
      const r = await postForm<{ ok?: boolean; error?: string }>("/api/fasrc/git-pull", {});
      setNote(r.error ? `✗ ${r.error}` : "✓ pulled");
    } catch (e) { setNote(`✗ ${e instanceof Error ? e.message : String(e)}`); }
    finally { setBusy(false); git.reload(); }
  }
  return (
    <Card>
      <CardHead title="Repo on FASRC"
        right={<div className="row" style={{ gap: 8 }}>
          <Button size="sm" variant="ghost" onClick={() => git.reload()} disabled={!connected}>↻</Button>
          <Button size="sm" variant="primary" onClick={pull} disabled={busy || !connected}>git pull</Button>
        </div>} />
      <CardBody>
        {!connected ? <Empty>connect to FASRC to read the remote repo</Empty>
          : git.loading ? <Empty><Spinner /> loading…</Empty>
          : git.data?.ok ? (
            <DefList items={[
              ["repo", <code className="mono">{git.data.repo}</code>],
              ["branch", <code className="mono">{git.data.branch}</code>],
              ["ahead / behind", `${git.data.ahead ?? 0} / ${git.data.behind ?? 0}`],
              ["last commit", git.data.last
                ? <span><code className="mono">{git.data.last.hash}</code> {git.data.last.subject} <span className="muted">· {git.data.last.relative}</span></span>
                : "—"],
            ]} />
          ) : <Empty>couldn't read the remote repo</Empty>}
        {note && <div className="job-panel" style={{ marginTop: "var(--s3)" }}><LogTail text={note} /></div>}
      </CardBody>
    </Card>
  );
}

/* ── Storage ─────────────────────────────────────────────────────────────── */
function StorageTab({ connected }: { connected: boolean }) {
  // Only fetch once connected — the old page fired on mount before the SSH
  // session existed and got stuck on an empty/loading listing forever.
  const data = useResource<DataListing>(connected ? "/api/fasrc/data-listing" : null, [connected]);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  async function control(url: string, msg: string) {
    setBusy(true); setNote(null);
    try {
      const r = await postForm<{ ok?: boolean; error?: string }>(url, {});
      setNote(r.error ? `✗ ${r.error}` : `✓ ${msg}`);
    } catch (e) { setNote(`✗ ${e instanceof Error ? e.message : String(e)}`); }
    finally { setBusy(false); data.reload(); }
  }
  const duRows = asArray<[string, string]>(data.data?.du);
  const tfrecords = asArray(data.data?.tfrecords);
  const checkpoints = asArray(data.data?.checkpoints);
  if (!connected) return <Card><CardBody><Empty>connect to FASRC to browse remote storage</Empty></CardBody></Card>;
  return (
    <div className="grid" style={{ gap: "var(--s4)" }}>
      <Card>
        <CardHead title="Remote storage" sub={data.data?.data_dir}
          right={<div className="row" style={{ gap: 8 }}>
            <Button size="sm" variant="ghost" onClick={() => data.reload()}>↻</Button>
            <Button size="sm" disabled={busy} onClick={() => control("/api/fasrc/bootstrap-data", "synced from holylabs")}>sync from holylabs</Button>
            <Button size="sm" disabled={busy} onClick={() => control("/api/fasrc/mirror/trigger", "pulling checkpoints to local")}>pull ckpts → local</Button>
          </div>} />
        <CardBody>
          {data.loading ? <Empty><Spinner /> scanning netscratch… (remote du can take ~20 s)</Empty>
            : (data.data && data.data.ok === false) ? <Empty>{data.data.error || "listing unavailable"}</Empty>
            : !duRows.length ? <Empty>no listing — is the data dir set?</Empty>
            : (<>
              <Table
                columns={[
                  { header: "size", cell: (r: [string, string]) => <span className="mono">{r[0]}</span>, width: 90 },
                  { header: "path", cell: (r: [string, string]) => <code className="mono">{r[1]}</code> },
                ]}
                rows={duRows} rowKey={(_r, i) => i} />
              <div className="muted" style={{ marginTop: "var(--s3)", fontSize: 12 }}>
                {tfrecords.length} tfrecords · {checkpoints.length} checkpoints
                {data.data?.ckpt_dir ? ` · ${data.data.ckpt_dir}` : ""}
              </div>
            </>)}
          {note && <div className="job-panel" style={{ marginTop: "var(--s3)" }}><LogTail text={note} /></div>}
        </CardBody>
      </Card>
    </div>
  );
}

/* ── Logs (past runs) ────────────────────────────────────────────────────── */
const LOG_PAGE_SIZE = 1000;

function RunLogDetail({ run, onBack }: { run: RunRow; onBack: () => void }) {
  const targets: (RunTask | RunRow)[] = run.tasks?.length ? run.tasks : [run];
  const [targetIndex, setTargetIndex] = useState(0);
  const target = targets[targetIndex] ?? targets[0];
  const [which, setWhich] = useState<LogKind>(() => preferredLogKind(targets[0]) ?? "out");
  const [page, setPage] = useState(0);
  const contentRef = useRef<HTMLPreElement>(null);
  const path = logPath(target, which);
  const url = path ? buildLogPageUrl(path, page, LOG_PAGE_SIZE) : null;
  const log = useResource<LogResp>(url, [path, page], { ttl: 0 });
  const data = log.data?.path === path ? log.data : null;

  useEffect(() => {
    const pre = contentRef.current;
    if (!pre || !data) return;
    pre.scrollTop = page === 0 ? pre.scrollHeight : 0;
  }, [data, page]);

  function show(kind: LogKind) {
    if (!logPath(target, kind)) return;
    setWhich(kind);
    setPage(0);
  }

  function selectTarget(index: number) {
    const next = targets[index];
    if (!next) return;
    setTargetIndex(index);
    setWhich(preferredLogKind(next) ?? "out");
    setPage(0);
  }

  const taskCols: Column<RunTask>[] = [
    { header: "task", width: 72, cell: (task) => <code className="mono">#{task.index}</code> },
    { header: "member", cell: (task) => <code className="mono">{task.member}</code> },
    { header: "SLURM job", cell: (task) => <code className="mono">{task.jobid}</code> },
    { header: "state", cell: (task) => task.state
      ? <Badge tone={stateTone(task.state)}>{task.state}</Badge>
      : <span className="muted">—</span> },
    { header: "", align: "right", cell: (task) => (
      <Button size="sm" variant={task.index === targetIndex ? "primary" : "ghost"}>
        {task.index === targetIndex ? "viewing" : "view log"}
      </Button>
    ) },
  ];

  return (
    <Card>
      <CardHead title="Run logs"
        sub={<>
          <code className="mono">{run.name}</code>{run.jobid ? ` · job ${run.jobid}` : ""}
          {run.tasks?.length ? ` · ${run.tasks.length} array tasks` : ""}
        </>}
        right={<div className="fasrc-log__toolbar">
          <Button size="sm" variant="ghost" onClick={onBack}>← Back to past runs</Button>
          <Button size="sm" variant={which === "out" ? "primary" : "default"}
            disabled={!target.out_path} onClick={() => show("out")}>.out</Button>
          <Button size="sm" variant={which === "err" ? "primary" : "default"}
            disabled={!target.err_path} onClick={() => show("err")}>.err</Button>
        </div>} />
      <CardBody>
        {run.tasks?.length ? <div className="fasrc-log__tasks">
          <div className="muted fasrc-log__task-caption">
            Select an array task to read that member's own stdout or stderr.
          </div>
          <Table columns={taskCols} rows={run.tasks}
            rowKey={(task) => task.jobid}
            onRowClick={(task) => selectTarget(task.index)} />
        </div> : null}
        {run.tasks?.length ? <div className="fasrc-log__selected muted">
          task #{(target as RunTask).index} · <code className="mono">{(target as RunTask).member}</code>
          {target.missing ? " · not seen by the history scan yet" : ""}
        </div> : null}
        {!data
          ? log.error
            ? <Empty>couldn't load this log</Empty>
            : <Empty><Spinner /> loading {which}…</Empty>
          : <pre ref={contentRef} className="ui-logtail fasrc-log__content">{data.content || "(empty)"}</pre>}
        {data && <div className="fasrc-log__pager">
          <Button size="sm" variant="ghost" disabled={!data.has_older}
            onClick={() => setPage((p) => p + 1)}>← older</Button>
          <Button size="sm" variant="ghost" disabled={!data.has_newer}
            onClick={() => setPage((p) => Math.max(0, p - 1))}>newer →</Button>
          <Button size="sm" variant="ghost" disabled={!data.has_newer}
            onClick={() => setPage(0)}>newest ↓</Button>
          <span className="muted mono">
            {data.total_lines
              ? `lines ${data.start_line}–${data.end_line} of ${data.total_lines}`
              : "empty file"}
          </span>
        </div>}
      </CardBody>
    </Card>
  );
}

function LogsTab({ connected }: { connected: boolean }) {
  const [page, setPage] = useState(0);
  const runs = useResource<RunsResp>(connected ? `/api/fasrc/runs?page=${page}` : null, [connected, page]);
  const [selected, setSelected] = useState<RunRow | null>(null);

  function openRun(run: RunRow) {
    if (hasRunLogs(run)) setSelected(run);
  }

  const cols: Column<RunRow>[] = [
    { header: "job", cell: (r) => <code className="mono">{r.jobid ?? "—"}</code> },
    { header: "label", cell: (r) => <div>
      <div>{r.label ?? r.name}</div>
      {r.tasks?.length ? <div className="muted" style={{ fontSize: 11 }}>
        array · {r.tasks.length} members
      </div> : null}
    </div> },
    { header: "state", cell: (r) => r.state ? <Badge tone={stateTone(r.state)}>{r.state}</Badge> : <span className="muted">—</span> },
    { header: "", align: "right", cell: (r) => hasRunLogs(r)
      ? <Button size="sm" title="Open raw logs">
        {r.tasks?.length ? `open ${r.tasks.length} task logs →` : "open logs →"}
      </Button>
      : <span className="muted">no files</span> },
  ];
  if (!connected) return <Card><CardBody><Empty>connect to FASRC to browse past runs</Empty></CardBody></Card>;
  if (selected) return <RunLogDetail run={selected} onBack={() => setSelected(null)} />;
  const d = runs.data;
  const hasOlder = d ? (d.page + 1) * d.page_size < d.total_runs : false;
  return (
    <Card>
      <CardHead title="Past runs" sub={d ? `${d.total_runs} total` : undefined}
        right={<div className="row" style={{ gap: 8 }}>
          <Button size="sm" variant="ghost" disabled={page === 0} onClick={() => setPage((p) => Math.max(0, p - 1))}>← newer</Button>
          <span className="muted mono" style={{ fontSize: 12 }}>pg {page + 1}</span>
          <Button size="sm" variant="ghost" disabled={!hasOlder} onClick={() => setPage((p) => p + 1)}>older →</Button>
          <Button size="sm" variant="ghost" onClick={() => runs.reload()}>↻</Button>
        </div>} />
      <CardBody>
        <Table columns={cols} rows={asArray<RunRow>(d?.runs)} rowKey={(r) => r.name}
          onRowClick={openRun} isRowClickable={(r) => hasRunLogs(r)}
          empty={runs.loading ? "loading…" : "no past runs found"} />
      </CardBody>
    </Card>
  );
}
