/* ops/git (spec §8.7): the local repository. Per-file staging, a diff viewer
 * (unstaged / staged, one file or all), commit exactly the selected files or
 * all of them (never an implicit `git add -A`; the size guard's refusals are
 * shown before committing and a forced retry is confirmed), fetch / pull /
 * push (confirmed) and the paged history (a commit opens in the inspector).
 * URL: `file`, `side` (unstaged|staged), the tables' `g.*` / `gl.*`. */
import { useMemo, useState } from "react";
import { ApiError, apiPost } from "../../../api/client";
import { invalidate, useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import { formatBytes, formatDateTime } from "../../../format";
import { useShortcut } from "../../../hooks/useShortcut";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Badge, Button, Callout, Card, CardBody, CardHead, DataTable, EmptyState, IconButton, Page, Segmented,
  Skeleton, Textarea, Tooltip, confirm, toast, type DataColumn,
} from "../../../ui";
import {
  gitDiffUrl, gitLogUrl, type GitActionResp, type GitCommit, type GitDiffResp, type GitFile, type GitLogResp,
  type GitStatusResp,
} from "../api";
import { gitStatusText, mergeCommitPages } from "../model";
import { DiffView } from "../git/DiffView";
import "../ops.css";

const STATUS_URL = "/api/git/status";
const LIST_MAX = 25;
const mb = (bytes?: number | null) => (bytes == null ? "" : ` · ${(bytes / 1e6).toFixed(1)} MB`);

/** A context-free file list for confirm() (plain elements only). */
function fileList(lines: string[]) {
  return (
    <span>
      <ul className="mono" style={{ margin: "6px 0 0", paddingLeft: 18, maxHeight: 260, overflow: "auto", fontSize: 12 }}>
        {lines.slice(0, LIST_MAX).map((l) => <li key={l}>{l}</li>)}
      </ul>
      {lines.length > LIST_MAX && <span>… and {lines.length - LIST_MAX} more</span>}
    </span>
  );
}

type Side = "unstaged" | "staged";

const LOG_PAGE = 100;

function History() {
  // Skip-based paging: the server caps one page at 500 commits, so "load
  // more" fetches the next page and appends it instead of re-asking for more.
  const [skip, setSkip] = useState(0);
  const [older, setOlder] = useState<GitCommit[]>([]);
  const PAGE = LOG_PAGE;
  const res = useResource<GitLogResp>(gitLogUrl(skip, PAGE), [skip], { ttl: 30_000 });
  const columns = useMemo<DataColumn<GitCommit>[]>(() => [
    { id: "hash", header: "Commit", width: 86, cell: (c) => <code className="mono">{c.hash}</code> },
    { id: "subject", header: "Subject", cell: (c) => <span className="ops-ellipsis" title={c.subject}>{c.subject}</span> },
    { id: "author", header: "Author", width: 150, cell: (c) => <span className="ops-dim">{c.author}</span> },
    { id: "date", header: "When", width: 120, accessor: (c) => c.date ?? "",
      cell: (c) => <span className="ops-dim ops-small" title={formatDateTime(c.date)}>{c.relative}</span> },
  ], []);
  const d = res.data;
  const commits = useMemo(() => (skip === 0 ? d?.commits ?? [] : mergeCommitPages(older, d?.commits ?? [])), [skip, older, d]);
  const loadMore = () => { setOlder(commits); setSkip(skip + PAGE); };
  return (
    <Card>
      <CardHead title="History" sub={d ? `${commits.length} of ${d.total} commits` : undefined} />
      <CardBody>
        <DataTable rows={commits} columns={columns} rowKey={(c) => c.full ?? c.hash} aria-label="Commits"
          loading={res.loading && !commits.length} height={420} urlKey="gl" exportName="git-log"
          inspect={(c) => ({ kind: "commit", id: c.full ?? c.hash })}
          empty="No commits yet."
          toolbar={d?.has_more ? <Button size="sm" variant="ghost" disabled={res.loading} onClick={loadMore}>Load {PAGE} more</Button> : undefined} />
      </CardBody>
    </Card>
  );
}

export default function Git() {
  const res = useResource<GitStatusResp>(STATUS_URL, [], { ttl: 5_000, poll: 30_000 });
  const [file, setFile] = useUrlState("file", "");
  const [side, setSide] = useUrlState<Side>("side", "unstaged", { parse: (r) => (r === "staged" ? "staged" : r === "unstaged" ? "unstaged" : undefined) });
  const [selected, setSelected] = useState<string[]>([]);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [note, setNote] = useState<{ ok: boolean; text: string } | null>(null);
  const diff = useResource<GitDiffResp>(gitDiffUrl(file || null, side === "staged"), [file, side, res.data], { ttl: 5_000 });
  const s = res.data?.status;
  const files = useMemo(() => s?.files ?? [], [s]);
  const byPath = useMemo(() => new Map(files.map((f) => [f.path, f])), [files]);
  const chosen = selected.filter((p) => byPath.has(p));
  const guarded = (list: GitFile[]) => list.filter((f) => f.guard);

  const refresh = () => { void invalidate(STATUS_URL); void invalidate("/api/git/"); };

  async function act(url: string, body: Record<string, string | string[]> = {}, done?: string): Promise<GitActionResp | null> {
    setBusy(url); setNote(null);
    try {
      const form = new FormData();
      for (const [k, v] of Object.entries(body)) for (const x of Array.isArray(v) ? v : [v]) form.append(k, x);
      const r = await apiPost<GitActionResp>(url, form);
      if (r.ok === false) { setNote({ ok: false, text: r.error || "failed" }); return r; }
      setNote({ ok: true, text: r.stdout || done || "done" });
      if (done) toast.success(done);
      return r;
    } catch (e) {
      if (e instanceof ApiError) throw e;
      setNote({ ok: false, text: e instanceof Error ? e.message : String(e) });
      return null;
    } finally { setBusy(null); refresh(); }
  }
  async function simple(url: string, done: string) {
    try { await act(url, {}, done); } catch (e) { setNote({ ok: false, text: e instanceof Error ? e.message : String(e) }); }
  }

  async function stage(paths: string[], un = false) {
    if (!paths.length) return;
    try {
      await act(un ? "/git/unstage" : "/git/stage", { paths }, `${un ? "Unstaged" : "Staged"} ${paths.length} file${paths.length === 1 ? "" : "s"}`);
    } catch (e) { setNote({ ok: false, text: e instanceof Error ? e.message : String(e) }); }
  }

  async function commit(mode: "selected" | "all") {
    const targets = mode === "all" ? files : chosen.map((p) => byPath.get(p)!);
    if (!message.trim() || !targets.length) return;
    const ok = await confirm({
      title: mode === "all" ? `Commit all ${targets.length} changed file${targets.length === 1 ? "" : "s"}?`
        : `Commit ${targets.length} selected file${targets.length === 1 ? "" : "s"}?`,
      message: fileList(targets.map((f) => `${f.xy.trim() || "?"}  ${f.path}${f.guard ? `  (${f.guard})` : ""}`)),
      confirmLabel: mode === "all" ? "Commit all" : "Commit",
    });
    if (!ok) return;
    const body: Record<string, string | string[]> = mode === "all" ? { message, all: "1" } : { message, paths: targets.map((f) => f.path) };
    await postCommit(body);
  }

  async function postCommit(body: Record<string, string | string[]>) {
    try {
      const r = await act("/git/commit", body);
      if (r?.ok) {
        toast.success(`Committed ${r.committed?.length ?? 0} file${r.committed?.length === 1 ? "" : "s"}`);
        setMessage(""); setSelected([]);
      }
    } catch (e) {
      if (e instanceof ApiError && e.status === 409 && e.code === "refused_files") {
        const refused = ((e.body as { refused?: GitActionResp["refused"] })?.refused) ?? [];
        const lines = refused.map((f) => `${f.path}${mb(f.size)} — ${f.reason ?? "refused"}`);
        setNote({ ok: false, text: `Refused (large or untracked binary files):\n${lines.join("\n")}` });
        const force = await confirm({
          title: `Commit ${refused.length} large or binary file${refused.length === 1 ? "" : "s"} anyway?`,
          message: fileList(lines), tone: "danger", confirmLabel: "Force commit",
        });
        if (force) await postCommit({ ...body, force: "1" });
        return;
      }
      if (e instanceof ApiError && e.status === 400 && (e.code === "no_selection" || e.code === "nothing_selected")) {
        setNote({ ok: false, text: e.code === "nothing_selected" ? "Nothing to commit: no changed file matches the selection." : "Nothing selected to commit." });
        return;
      }
      setNote({ ok: false, text: e instanceof Error ? e.message : String(e) });
    }
  }

  async function push() {
    const ahead = s?.ahead ?? 0;
    if (!(await confirm({ title: `Push ${s?.branch ?? "this branch"} to ${s?.upstream || "its upstream"}?`,
      message: ahead ? `${ahead} local commit${ahead === 1 ? "" : "s"} will be published.` : "No local commits ahead of the upstream.",
      confirmLabel: "Push" }))) return;
    await simple("/git/push", "Pushed");
  }
  async function pull() {
    if (!(await confirm({ title: `Pull ${s?.upstream || "the upstream"}?`, message: "Fast-forward only (git pull --ff-only).", confirmLabel: "Pull" }))) return;
    await simple("/git/pull", "Pulled");
  }

  useShortcut("$mod+Enter", () => { if (message.trim() && chosen.length) { void commit("selected"); return true; } return false; },
    { description: "Commit the selected files", scope: "Git", allowInInputs: true });
  usePageActions([
    { id: "git-fetch", label: "git fetch", group: "Git", run: () => void simple("/git/fetch", "Fetched") },
    { id: "git-pull", label: "git pull (ff-only)", group: "Git", run: () => void pull() },
    { id: "git-push", label: "git push", group: "Git", run: () => void push() },
    { id: "git-stage-selected", label: "Stage the selected files", group: "Git", disabled: !chosen.length, run: () => void stage(chosen) },
    { id: "git-refresh", label: "Refresh git status", group: "Git", run: refresh },
  ]);

  const columns = useMemo<DataColumn<GitFile>[]>(() => [
    { id: "xy", header: "Status", width: 118, accessor: (f) => gitStatusText(f.xy),
      cell: (f) => (
        <span className="ops-row">
          <code className="mono" title={`porcelain ${JSON.stringify(f.xy)}`}>{f.xy.replace(/ /g, "·")}</code>
          {f.staged && <Badge size="sm" tone="good">staged</Badge>}
        </span>
      ) },
    { id: "path", header: "Path", cell: (f) => (
      <span className="ops-cell2">
        <span className="mono">{f.path}</span>
        {f.orig && <span className="ops-dim ops-small mono">from {f.orig}</span>}
      </span>
    ) },
    { id: "size", header: "Size", numeric: true, width: 84, accessor: (f) => f.size ?? -1,
      cell: (f) => <span className="mono ops-small">{f.size == null ? "—" : formatBytes(f.size)}</span> },
    { id: "guard", header: "Guard", width: 150, accessor: (f) => f.guard ?? "",
      cell: (f) => (f.guard ? <Tooltip content="Commits refuse it unless forced"><span tabIndex={0}><Badge size="sm" tone="warn">{f.guard}</Badge></span></Tooltip> : null) },
    { id: "actions", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 72,
      cell: (f) => (
        <span className="ops-row-actions">
          {f.unstaged && <IconButton size="sm" icon="plus" label={`Stage ${f.path}`} onClick={() => void stage([f.path])} />}
          {f.staged && <IconButton size="sm" icon="minus" label={`Unstage ${f.path}`} onClick={() => void stage([f.path], true)} />}
        </span>
      ) },
  // eslint-disable-next-line react-hooks/exhaustive-deps
  ], []);

  if (res.loading && !s) return <Page className="ops-page"><Skeleton lines={10} /></Page>;
  if (res.error && !s) return <Page className="ops-page"><Callout tone="bad" title="Could not read git status">{res.error.message}</Callout></Page>;
  if (s && !s.in_repo) return <Page className="ops-page"><EmptyState icon="info" title="Not inside a git repository" /></Page>;
  const flaggedChosen = guarded(chosen.map((p) => byPath.get(p)!));
  const stagedChosen = chosen.filter((p) => byPath.get(p)?.staged);
  const unstagedChosen = chosen.filter((p) => byPath.get(p)?.unstaged);
  return (
    <Page className="ops-page">
      <div className="ops-bar" role="toolbar" aria-label="Repository">
        <code className="mono">{s?.branch}</code>
        {s?.upstream ? <span className="ops-dim mono ops-small">→ {s.upstream}</span> : <Badge size="sm" tone="warn">no upstream</Badge>}
        {(s?.ahead ?? 0) > 0 && <Badge size="sm" tone="warn">↑ {s?.ahead}</Badge>}
        {(s?.behind ?? 0) > 0 && <Badge size="sm" tone="warn">↓ {s?.behind}</Badge>}
        {!s?.ahead && !s?.behind && s?.upstream && <Badge size="sm" tone="good">in sync</Badge>}
        {s?.last?.hash && <span className="ops-dim ops-small ops-ellipsis" title={s.last.subject}>{s.last.hash} · {s.last.subject}</span>}
        <span className="ops-spacer" />
        <Button size="sm" variant="ghost" loading={busy === "/git/fetch"} onClick={() => void simple("/git/fetch", "Fetched")}>Fetch</Button>
        <Button size="sm" variant="ghost" loading={busy === "/git/pull"} onClick={() => void pull()}>Pull</Button>
        <Button size="sm" variant="primary" loading={busy === "/git/push"} onClick={() => void push()}>Push</Button>
        <IconButton size="sm" icon="reset" label="Refresh" onClick={refresh} />
      </div>
      {note && (
        <Callout tone={note.ok ? "good" : "bad"} onDismiss={() => setNote(null)}><span className="ops-pre">{note.text}</span></Callout>
      )}
      <div className="ops-split">
        <div className="ops-split__main">
          <Card>
            <CardHead title="Changes" sub={files.length ? `${files.length} changed` : "working tree clean"} />
            <CardBody>
              <DataTable rows={files} columns={columns} rowKey={(f) => f.path} aria-label="Changed files" urlKey="g"
                selectable selected={chosen} onSelectedChange={(keys) => setSelected(keys)}
                activeKey={file || null} onRowClick={(f) => { setFile(f.path); setSide(f.unstaged ? "unstaged" : "staged"); }}
                height={380} dense empty="Working tree clean."
                toolbar={<>
                  <Button size="sm" variant="ghost" disabled={!unstagedChosen.length || busy != null} onClick={() => void stage(unstagedChosen)}>Stage{unstagedChosen.length ? ` ${unstagedChosen.length}` : ""}</Button>
                  <Button size="sm" variant="ghost" disabled={!stagedChosen.length || busy != null} onClick={() => void stage(stagedChosen, true)}>Unstage{stagedChosen.length ? ` ${stagedChosen.length}` : ""}</Button>
                </>} />
            </CardBody>
          </Card>
          <Card>
            <CardHead title="Commit" sub="exactly the selected files, as they are on disk" />
            <CardBody className="ops-editor">
              <Textarea value={message} onChange={setMessage} rows={3} placeholder="Commit message… (⌘/Ctrl-Enter commits the selection)" aria-label="Commit message" />
              {flaggedChosen.length > 0 && (
                <Callout tone="warn" title={`${flaggedChosen.length} selected file${flaggedChosen.length === 1 ? " is" : "s are"} guarded`}>
                  {flaggedChosen.map((f) => `${f.path} (${f.guard})`).join(", ")} — the commit asks before forcing.
                </Callout>
              )}
              <div className="ops-row">
                <Button variant="primary" size="sm" loading={busy === "/git/commit"} disabled={!message.trim() || !chosen.length}
                  onClick={() => void commit("selected")}>Commit {chosen.length || ""} selected</Button>
                <Button size="sm" disabled={!message.trim() || !files.length || busy != null} onClick={() => void commit("all")}>Commit all</Button>
              </div>
            </CardBody>
          </Card>
          <History />
        </div>
        <aside className="ops-split__side" aria-label="Diff">
          <Card>
            <CardHead title={file ? <code className="mono">{file}</code> : "Diff"} sub={file ? undefined : "all files"}
              right={<div className="ops-row">
                <Segmented<Side> size="sm" value={side} onChange={setSide} aria-label="Diff side"
                  options={[{ value: "unstaged", label: "Unstaged" }, { value: "staged", label: "Staged" }]} />
                {file && <IconButton size="sm" icon="close" label="Show every file" onClick={() => setFile("")} />}
                {file && <IconButton size="sm" icon="panelRight" label="Inspect the file"
                  onClick={() => /\.fits?(\.gz)?$/i.test(file) ? openInspector({ kind: "fits", id: file }) : undefined}
                  disabled={!/\.fits?(\.gz)?$/i.test(file)} />}
              </div>} />
            <CardBody>
              {diff.loading && !diff.data ? <Skeleton lines={8} />
                : diff.error && !diff.data ? <Callout tone="bad">{diff.error.message}</Callout>
                : <DiffView text={diff.data?.diff} empty={side === "staged" ? "Nothing staged." : "No unstaged changes."} />}
            </CardBody>
          </Card>
        </aside>
      </div>
    </Page>
  );
}
