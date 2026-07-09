/* Git — repo status, stage-all+commit, push/pull/fetch, diff + recent log.
   Second exemplar page: useResource for status, postForm for the sync actions,
   Table/DefList from the kit. */
import { useState } from "react";
import { getJSON, postForm } from "../api";
import { useResource } from "../hooks";
import {
  Badge, Button, Card, CardBody, CardHead, DefList, Empty, LogTail, Page,
  PageHead, Spinner, Table, Textarea, type Column,
} from "../ui";

type Last = { hash: string; subject: string; relative: string };
type GitStatus = {
  in_repo: boolean; root?: string; branch?: string; upstream?: string | null;
  ahead?: number; behind?: number; files?: { xy: string; path: string }[]; last?: Last | null;
};
type LogEntry = { hash: string; author: string; subject: string; relative: string };
type StatusResp = { status: GitStatus; log: LogEntry[] };

const FILE_COLS: Column<{ xy: string; path: string }>[] = [
  { header: "status", cell: (f) => <code className="mono">{f.xy}</code>, width: 80 },
  { header: "path", cell: (f) => <code className="mono">{f.path}</code> },
];
const LOG_COLS: Column<LogEntry>[] = [
  { header: "commit", cell: (c) => <code className="mono">{c.hash}</code>, width: 90 },
  { header: "subject", cell: (c) => c.subject },
  { header: "author", cell: (c) => <span className="muted">{c.author}</span> },
  { header: "when", cell: (c) => <span className="muted">{c.relative}</span>, align: "right" },
];

export default function GitPage() {
  const { data, loading, reload } = useResource<StatusResp>("/api/git/status");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [note, setNote] = useState<{ ok: boolean; text: string } | null>(null);
  const [diff, setDiff] = useState<string | null>(null);

  const s = data?.status;

  async function act(url: string, body?: Record<string, string>) {
    setBusy(url); setNote(null);
    try {
      const r = await postForm<{ ok: boolean; error?: string; stdout?: string }>(url, body ?? {});
      setNote({ ok: !!r.ok, text: r.ok ? (r.stdout || "done") : (r.error || "failed") });
      if (r.ok) reload();
    } catch (e) {
      setNote({ ok: false, text: e instanceof Error ? e.message : String(e) });
    } finally { setBusy(null); }
  }

  async function loadDiff() {
    if (diff != null) { setDiff(null); return; }
    const r = await getJSON<{ diff: string }>("/api/git/diff?staged=0");
    setDiff(r?.diff ?? "(no unstaged changes)");
  }

  return (
    <Page>
      <PageHead eyebrow="ops · git" title="Git"
        sub="Repository status, commit, and sync with the remote — from the console." />

      {loading && <Card><CardBody><Empty><Spinner /> loading…</Empty></CardBody></Card>}

      {s && !s.in_repo && (
        <Card><CardBody><Empty>not inside a git repository</Empty></CardBody></Card>
      )}

      {s?.in_repo && (
        <div className="grid" style={{ gridTemplateColumns: "1fr", gap: "var(--s4)" }}>
          <Card>
            <CardHead title="Status" sub={<code className="mono">{s.root}</code>}
              right={
                <div className="row" style={{ gap: 8 }}>
                  {(s.ahead ?? 0) > 0 && <Badge tone="warn">↑ {s.ahead}</Badge>}
                  {(s.behind ?? 0) > 0 && <Badge tone="warn">↓ {s.behind}</Badge>}
                  {!s.ahead && !s.behind && <Badge tone="good">in sync</Badge>}
                </div>
              } />
            <CardBody>
              <DefList items={[
                ["branch", <code className="mono">{s.branch}</code>],
                ["upstream", <code className="mono">{s.upstream ?? "—"}</code>],
                ["last commit", s.last
                  ? <span><code className="mono">{s.last.hash}</code> {s.last.subject} <span className="muted">· {s.last.relative}</span></span>
                  : "—"],
              ]} />
              <div className="row" style={{ marginTop: "var(--s4)", gap: "var(--s2)" }}>
                <Button onClick={() => act("/git/fetch")} disabled={busy != null}>Fetch</Button>
                <Button onClick={() => act("/git/pull")} disabled={busy != null}>Pull (ff-only)</Button>
                <Button variant="primary" onClick={() => act("/git/push")} disabled={busy != null}>Push</Button>
                <Button variant="ghost" size="sm" onClick={loadDiff}>{diff != null ? "hide diff" : "show unstaged diff"}</Button>
              </div>
              {note && (
                <div className={`job-panel job-panel--${note.ok ? "done" : "err"}`} style={{ marginTop: "var(--s3)" }}>
                  <LogTail text={note.text} />
                </div>
              )}
              {diff != null && <div style={{ marginTop: "var(--s3)" }}><LogTail text={diff} style={{ maxHeight: 360 }} /></div>}
            </CardBody>
          </Card>

          <Card>
            <CardHead title="Commit" sub="stages all changes, then commits" />
            <CardBody>
              <Textarea value={message} onChange={setMessage} rows={3} placeholder="commit message…" />
              <div className="row" style={{ marginTop: "var(--s3)" }}>
                <Button variant="primary" disabled={!message.trim() || busy != null}
                  onClick={() => act("/git/commit", { message })}>
                  Stage all + commit
                </Button>
              </div>
            </CardBody>
          </Card>

          <Card>
            <CardHead title="Changed files"
              sub={`${s.files?.length ?? 0} modified`} />
            <CardBody>
              <Table columns={FILE_COLS} rows={s.files ?? []} empty="working tree clean"
                rowKey={(f) => f.path} />
            </CardBody>
          </Card>

          <Card>
            <CardHead title="Recent commits" />
            <CardBody>
              <Table columns={LOG_COLS} rows={data?.log ?? []} rowKey={(c) => c.hash} />
            </CardBody>
          </Card>
        </div>
      )}
    </Page>
  );
}
