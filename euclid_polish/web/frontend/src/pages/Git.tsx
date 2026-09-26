/* Git — repo status, commit, push/pull/fetch, diff + recent log.
   Second exemplar page: useResource for status, postForm for the sync actions,
   Table/DefList from the kit. "Stage all + commit" confirms the file list and
   posts `all=1` (never an implicit `git add -A`); the server's large/binary
   guard (409 refused_files) offers a `force=1` retry; push confirms first. */
import { useState } from "react";
import { asArray } from "../data";
import { ApiError, getJSON, postForm } from "../api";
import { useResource } from "../hooks";
import {
  Badge, Button, Card, CardBody, CardHead, DefList, Empty, LogTail, Page,
  PageHead, Spinner, Table, Textarea, confirm, type Column,
} from "../ui";

type Last = { hash: string; subject: string; relative: string };
type GitStatus = {
  in_repo: boolean; root?: string; branch?: string; upstream?: string | null;
  ahead?: number; behind?: number; files?: { xy: string; path: string }[]; last?: Last | null;
};
type LogEntry = { hash: string; author: string; subject: string; relative: string };
type StatusResp = { status: GitStatus; log: LogEntry[] };
type Refused = { path: string; size?: number; reason?: string };
type CommitResp = { ok: boolean; error?: string; stdout?: string; committed?: string[] };

const LIST_MAX = 25;

/** A context-free file list for confirm() (plain elements only). */
function fileList(rows: { key: string; text: string }[]) {
  const shown = rows.slice(0, LIST_MAX);
  return (
    <div>
      <ul className="mono" style={{ margin: "6px 0 0", paddingLeft: 18, maxHeight: 260, overflow: "auto", fontSize: 12 }}>
        {shown.map((r) => <li key={r.key}>{r.text}</li>)}
      </ul>
      {rows.length > LIST_MAX && <p style={{ margin: "6px 0 0" }}>… and {rows.length - LIST_MAX} more</p>}
    </div>
  );
}

const mbText = (bytes?: number) => (bytes == null ? "" : ` · ${(bytes / 1e6).toFixed(1)} MB`);

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
  const files = asArray<NonNullable<GitStatus["files"]>[number]>(s?.files);
  const logEntries = asArray<LogEntry>(data?.log);

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

  async function commitAll() {
    const changed = files.map((f) => ({ key: f.path, text: `${f.xy.trim() || "?"}  ${f.path}` }));
    const ok = await confirm({
      title: `Commit all ${files.length} changed file${files.length === 1 ? "" : "s"}?`,
      message: fileList(changed),
      confirmLabel: "Commit all",
    });
    if (!ok) return;
    await postCommit({ message, all: "1" });
  }

  async function postCommit(body: Record<string, string>) {
    setBusy("/git/commit"); setNote(null);
    try {
      const r = await postForm<CommitResp>("/git/commit", body);
      setNote({ ok: !!r.ok, text: r.ok ? (r.stdout || `committed ${r.committed?.length ?? 0} file(s)`) : (r.error || "failed") });
      if (r.ok) { setMessage(""); reload(); }
    } catch (e) {
      if (e instanceof ApiError && e.status === 409 && e.code === "refused_files") {
        const refused = asArray<Refused>((e.body as { refused?: unknown })?.refused);
        setNote({ ok: false, text: `refused (large or untracked binary files):\n${refused.map((f) => `${f.path}${mbText(f.size)} — ${f.reason ?? "refused"}`).join("\n")}` });
        setBusy(null);
        const force = await confirm({
          title: `Commit ${refused.length} large or binary file${refused.length === 1 ? "" : "s"} anyway?`,
          message: fileList(refused.map((f) => ({ key: f.path, text: `${f.path}${mbText(f.size)} — ${f.reason ?? "refused"}` }))),
          tone: "danger", confirmLabel: "Force commit",
        });
        if (force) await postCommit({ ...body, force: "1" });
        return;
      }
      if (e instanceof ApiError && e.status === 400 && (e.code === "no_selection" || e.code === "nothing_selected")) {
        setNote({ ok: false, text: e.code === "nothing_selected" ? "nothing to commit: no changed file matches the selection" : "nothing selected to commit" });
        return;
      }
      setNote({ ok: false, text: e instanceof Error ? e.message : String(e) });
    } finally { setBusy(null); }
  }

  async function push() {
    const ahead = s?.ahead ?? 0;
    const ok = await confirm({
      title: `Push ${s?.branch ?? "this branch"} to ${s?.upstream ?? "its upstream"}?`,
      message: ahead ? `${ahead} local commit${ahead === 1 ? "" : "s"} will be published.` : "No local commits ahead of the upstream.",
      confirmLabel: "Push",
    });
    if (ok) await act("/git/push");
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
                <Button variant="primary" onClick={push} disabled={busy != null}>Push</Button>
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
            <CardHead title="Commit" sub="commits every changed file (you confirm the list first)" />
            <CardBody>
              <Textarea value={message} onChange={setMessage} rows={3} placeholder="commit message…" />
              <div className="row" style={{ marginTop: "var(--s3)" }}>
                <Button variant="primary" disabled={!message.trim() || busy != null || files.length === 0}
                  onClick={commitAll}>
                  Stage all + commit
                </Button>
              </div>
            </CardBody>
          </Card>

          <Card>
            <CardHead title="Changed files"
              sub={`${files.length} modified`} />
            <CardBody>
              <Table columns={FILE_COLS} rows={files} empty="working tree clean"
                rowKey={(f) => f.path} />
            </CardBody>
          </Card>

          <Card>
            <CardHead title="Recent commits" />
            <CardBody>
              <Table columns={LOG_COLS} rows={logEntries} rowKey={(c) => c.hash} />
            </CardBody>
          </Card>
        </div>
      )}
    </Page>
  );
}
