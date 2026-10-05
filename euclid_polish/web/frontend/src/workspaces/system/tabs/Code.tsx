/* System › Code (`/system/code`): one question — are this laptop, the
 * running server and the FASRC checkout on the same commit? One sentence
 * answers it; then this laptop's checkout (the changes in words, stage /
 * unstage, commit exactly the selected files or all of them — never an
 * implicit `git add -A`; the size guard's refusals are shown and a forced
 * retry is confirmed —, fetch / pull, push, primary only when ahead, and a
 * diff side card), the FASRC checkout (its HEAD against this laptop's, git
 * pull, Update env) and, collapsed, the server's boot commit, the
 * restart-needed notice and the runtime versions.
 * URL: `file`, `dside` (unstaged|staged), `side=fasrc` (open on the FASRC
 * checkout), the tables' `g.*` / `gl.*`. */
import { useMemo, useRef, useState, version as reactVersion, type ReactNode } from "react";
import { ApiError, apiPost } from "../../../api/client";
import { useJob } from "../../../api/jobs";
import { invalidate, useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import { useConsoleUpdate, useFasrcStatus, useVersion } from "../../../app/status";
import { formatBytes, formatDateTime, formatRelative } from "../../../format";
import { useShortcut } from "../../../hooks/useShortcut";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Badge, Button, Callout, Caption, Card, CardBody, CardHead, CopyButton, DataTable, DefList, Details, EmptyState, FactsList,
  IconButton, JobProgress, LogTail, LogView, Page, Section, Segmented, Skeleton, SummaryLine, Textarea, Toolbar,
  ToolbarSpacer, ToolbarText, Tooltip, confirm, toast, type DataColumn,
} from "../../../ui";
import { ConnectionBar } from "../../../fasrc";
import {
  FASRC_GIT_URL, GIT_STATUS_URL, SYSTEM_URL, gitDiffUrl, gitLogUrl, type GitActionResp, type GitCommit, type GitDiffResp,
  type GitFile, type GitLogResp, type GitPullResp, type GitStatusResp, type RemoteGit, type SystemInfo,
} from "../api";
import { codeSentence, gitStatusText, mergeCommitPages, relationText } from "../model";
import { DiffView } from "../git/DiffView";
import { useArrivalScroll } from "../../../hooks/arrival";
import "../system.css";

const STATUS_URL = GIT_STATUS_URL;
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
    { id: "subject", header: "Subject", cell: (c) => <span className="sys-ellipsis" title={c.subject}>{c.subject}</span> },
    { id: "author", header: "Author", width: 150, cell: (c) => <span className="sys-dim">{c.author}</span> },
    { id: "date", header: "When", width: 120, accessor: (c) => c.date ?? "",
      cell: (c) => <span className="sys-dim sys-small" title={formatDateTime(c.date)}>{c.relative}</span> },
  ], []);
  const d = res.data;
  const commits = useMemo(() => (skip === 0 ? d?.commits ?? [] : mergeCommitPages(older, d?.commits ?? [])), [skip, older, d]);
  const loadMore = () => { setOlder(commits); setSkip(skip + PAGE); };
  return (
    <Section title="Commit history" sub={d ? `${commits.length} of ${d.total} commits` : undefined} collapsible defaultOpen={false}>
        <DataTable rows={commits} columns={columns} rowKey={(c) => c.full ?? c.hash} aria-label="Commits"
          loading={res.loading && !commits.length} height={420} urlKey="gl" exportName="git-log"
          inspect={(c) => ({ kind: "commit", id: c.full ?? c.hash })}
          empty="No commits yet."
          toolbar={d?.has_more ? <Button size="sm" variant="ghost" disabled={res.loading} onClick={loadMore}>Load {PAGE} more</Button> : undefined} />
    </Section>
  );
}

/** This laptop's checkout: changes, commit, fetch / pull / push, diff. */
function LocalCheckout({ res }: { res: ReturnType<typeof useResource<GitStatusResp>> }) {
  const [file, setFile] = useUrlState("file", "");
  const [side, setSide] = useUrlState<Side>("dside", "unstaged", { parse: (r) => (r === "staged" ? "staged" : r === "unstaged" ? "unstaged" : undefined) });
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
    { id: "xy", header: "Change", width: 132, accessor: (f) => gitStatusText(f.xy),
      cell: (f) => (
        <span className="sys-row">
          <Tooltip content={`git status --porcelain: ${JSON.stringify(f.xy)}`}><span tabIndex={0}>{gitStatusText(f.xy)}</span></Tooltip>
          {f.staged && <Badge size="sm" tone="good">staged</Badge>}
        </span>
      ) },
    { id: "path", header: "Path", width: 220, cell: (f) => (
      <span className="sys-cell2" title={f.path}>
        <span className="mono">{f.path}</span>
        {f.orig && <span className="sys-dim sys-small mono">from {f.orig}</span>}
      </span>
    ) },
    { id: "size", header: "Size", numeric: true, width: 84, priority: 2, accessor: (f) => f.size ?? -1,
      cell: (f) => <span className="mono sys-small">{f.size == null ? "—" : formatBytes(f.size)}</span> },
    { id: "guard", header: "Guard", width: 150, priority: 1, accessor: (f) => f.guard ?? "",
      cell: (f) => (f.guard ? <Tooltip content="Commits refuse it unless forced"><span tabIndex={0}><Badge size="sm" tone="warn">{f.guard}</Badge></span></Tooltip> : null) },
    { id: "actions", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 72,
      cell: (f) => (
        <span className="sys-row-actions">
          {f.unstaged && <IconButton size="sm" icon="plus" label={`Stage ${f.path}`} onClick={() => void stage([f.path])} />}
          {f.staged && <IconButton size="sm" icon="minus" label={`Unstage ${f.path}`} onClick={() => void stage([f.path], true)} />}
        </span>
      ) },
  // eslint-disable-next-line react-hooks/exhaustive-deps
  ], []);

  if (res.loading && !s) return <Skeleton lines={10} />;
  if (res.error && !s) return <Callout tone="bad" title="Could not read git status">{res.error.message}</Callout>;
  if (s && !s.in_repo) return <EmptyState icon="info" title="Not inside a git repository" />;
  const flaggedChosen = guarded(chosen.map((p) => byPath.get(p)!));
  const stagedChosen = chosen.filter((p) => byPath.get(p)?.staged);
  const unstagedChosen = chosen.filter((p) => byPath.get(p)?.unstaged);
  const ahead = s?.ahead ?? 0;
  const behind = s?.behind ?? 0;
  const words = [
    ahead ? `${ahead} commit${ahead === 1 ? "" : "s"} to push` : "",
    behind ? `${behind} to pull` : "",
    !ahead && !behind && s?.upstream ? "in sync with its upstream" : "",
  ].filter(Boolean).join(" · ");
  return (
    <>
      <Toolbar label="This laptop's checkout">
        <code className="mono">{s?.branch}</code>
        {s?.upstream ? <ToolbarText><span className="mono">→ {s.upstream}</span></ToolbarText> : <Badge size="sm" tone="warn">no upstream</Badge>}
        {words && <ToolbarText>{words}</ToolbarText>}
        <ToolbarSpacer />
        <Button size="sm" variant="ghost" loading={busy === "/git/fetch"} onClick={() => void simple("/git/fetch", "Fetched")}>Fetch</Button>
        <Button size="sm" variant="ghost" loading={busy === "/git/pull"} onClick={() => void pull()}>Pull</Button>
        <Button size="sm" variant={ahead > 0 ? "primary" : "ghost"} loading={busy === "/git/push"} onClick={() => void push()}>
          Push{ahead > 0 ? ` ${ahead}` : ""}
        </Button>
        <IconButton size="sm" icon="reset" label="Refresh" onClick={refresh} />
      </Toolbar>
      {note && (
        <Callout tone={note.ok ? "good" : "bad"} onDismiss={() => setNote(null)}><span className="sys-pre">{note.text}</span></Callout>
      )}
      <div className="sys-split">
        <div className="sys-split__main">
          <Card>
            <CardHead title="Changes" sub={files.length ? changesText(files) : "working tree clean"} />
            <CardBody>
              <DataTable rows={files} columns={columns} rowKey={(f) => f.path} aria-label="Changed files" urlKey="g"
                selectable selected={chosen} onSelectedChange={(keys) => setSelected(keys)}
                activeKey={file || null} onRowClick={(f) => { setFile(f.path); setSide(f.unstaged ? "unstaged" : "staged"); }}
                height={380} dense empty="Working tree clean." countText={null}
                toolbar={<>
                  <Button size="sm" variant="ghost" disabled={!unstagedChosen.length || busy != null} onClick={() => void stage(unstagedChosen)}>Stage{unstagedChosen.length ? ` ${unstagedChosen.length}` : ""}</Button>
                  <Button size="sm" variant="ghost" disabled={!stagedChosen.length || busy != null} onClick={() => void stage(stagedChosen, true)}>Unstage{stagedChosen.length ? ` ${stagedChosen.length}` : ""}</Button>
                </>} />
            </CardBody>
          </Card>
          <Card>
            <CardHead title="Commit" sub="exactly the selected files, as they are on disk" />
            <CardBody className="sys-editor">
              <Textarea value={message} onChange={setMessage} rows={3} placeholder="Commit message… (⌘/Ctrl-Enter commits the selection)" aria-label="Commit message" />
              {flaggedChosen.length > 0 && (
                <Callout tone="warn" title={`${flaggedChosen.length} selected file${flaggedChosen.length === 1 ? " is" : "s are"} guarded`}>
                  {flaggedChosen.map((f) => `${f.path} (${f.guard})`).join(", ")} — the commit asks before forcing.
                </Callout>
              )}
              <div className="sys-row">
                <Button variant="primary" size="sm" loading={busy === "/git/commit"} disabled={!message.trim() || !chosen.length}
                  onClick={() => void commit("selected")}>Commit {chosen.length || ""} selected</Button>
                <Button size="sm" disabled={!message.trim() || !files.length || busy != null} onClick={() => void commit("all")}>Commit all</Button>
              </div>
            </CardBody>
          </Card>
          <History />
        </div>
        <aside className="sys-split__side" aria-label="Diff">
          <Card>
            <CardHead title={file ? <code className="mono">{file}</code> : "Diff"} sub={file ? undefined : "all files"}
              right={<div className="sys-row">
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
    </>
  );
}

/** "3 untracked · 5 modified": the changes in words, counted by kind. */
function changesText(files: readonly GitFile[]): string {
  const counts = new Map<string, number>();
  for (const f of files) counts.set(gitStatusText(f.xy), (counts.get(gitStatusText(f.xy)) ?? 0) + 1);
  return [...counts.entries()].map(([k, n]) => `${n} ${k}`).join(" · ");
}

/** The FASRC checkout against this laptop's HEAD: git pull, Update env. */
function FasrcCheckout({ git, connected }: { git: ReturnType<typeof useResource<RemoteGit>>; connected: boolean }) {
  const env = useJob("fasrc:env-update");
  const [pulling, setPulling] = useState(false);
  const [pullOut, setPullOut] = useState<GitPullResp | null>(null);
  const d = git.data;
  const rel = relationText(d?.relation as never);

  async function pull() {
    if (!(await confirm({ title: `git pull on FASRC (${d?.branch ?? "branch"})?`,
      message: "Fast-forward only. Running jobs keep their already-loaded code.", confirmLabel: "Pull" }))) return;
    setPulling(true); setPullOut(null);
    try {
      const r = await apiPost<GitPullResp>("/api/fasrc/git-pull");
      setPullOut(r);
      if (r.ok) toast.success(r.changed_files?.length ? `Pulled ${r.changed_files.length} changed files` : "Already up to date");
      else toast.error(r.error || "git pull failed");
    } catch (e) {
      setPullOut({ ok: false, error: e instanceof Error ? e.message : String(e) });
    } finally { setPulling(false); void invalidate(FASRC_GIT_URL); }
  }
  async function updateEnv() {
    if (!(await confirm({ title: "Update the conda environment on FASRC?",
      message: "In the FASRC checkout: `module load python`, then `yes | mamba env update -p <Conda env> -f environment.yml` "
        + "(minutes; Conda env is the prefix in System › Connections › SSH settings). A cancel can leave the env half-updated; "
        + "re-run to finish.",
      confirmLabel: "Update env" }))) return;
    await env.run("/api/fasrc/env-update", {}, {
      onDone: (j) => { if (j.status === "done") toast.success("Conda environment updated"); },
    });
  }
  usePageActions([
    { id: "fasrc-git-pull", label: "git pull on FASRC", group: "Code", disabled: !connected, run: () => void pull() },
    { id: "fasrc-env-update", label: "Update the FASRC conda environment", group: "Code", keywords: ["mamba", "environment.yml"],
      disabled: !connected || env.busy, run: () => void updateEnv() },
  ]);

  return (
    <Card>
      <CardHead title="FASRC checkout" sub={d?.repo ? <code className="mono">{d.repo}</code> : undefined} right={
        connected ? (
          <div className="sys-row">
            <Button size="sm" variant={d?.relation?.relation === "remote_behind" ? "primary" : "default"} loading={pulling} onClick={() => void pull()}>git pull</Button>
            <Button size="sm" loading={env.busy} onClick={() => void updateEnv()}>Update env</Button>
            <IconButton size="sm" icon="reset" label="Refresh (fetches on FASRC)" onClick={() => git.reload()} />
          </div>
        ) : <ConnectionBar />} />
      <CardBody>
        {!connected ? <p className="sys-note">Connect to FASRC to read its checkout.</p>
          : git.loading && !d ? <Skeleton lines={3} />
          : git.error && !d ? <Callout tone="bad">{git.error.message}</Callout>
          : d && (
            <FactsList facts={[
              { label: "Against this laptop", value: <Tooltip content={rel.hint}><span tabIndex={0}>{rel.label}</span></Tooltip>, tone: rel.tone === "good" ? undefined : rel.tone },
              { label: "FASRC HEAD", value: <code className="mono">{d.head?.slice(0, 7) || "—"}</code>, unit: d.branch },
              ((d.ahead ?? 0) > 0 || (d.behind ?? 0) > 0) && { label: "Against its upstream", value: `${d.ahead ?? 0} ahead · ${d.behind ?? 0} behind`, tone: "warn" },
              d.dirty && { label: "Uncommitted on FASRC", value: `${d.dirty_files?.length ?? 0} files`, tone: "warn" },
            ]} />
          )}
        {connected && d?.last?.subject && <Caption>Last commit on FASRC: {d.last.subject}{d.last.relative ? ` · ${d.last.relative}` : ""}</Caption>}
        {d?.dirty && d.dirty_files && <Details summary="Uncommitted files on FASRC"><LogTail text={d.dirty_files.join("\n")} style={{ maxHeight: 160 }} /></Details>}
        {pullOut && (
          <Callout tone={pullOut.ok ? (pullOut.env_update_needed ? "warn" : "good") : "bad"}
            title={pullOut.ok ? (pullOut.env_update_needed ? "environment.yml changed" : "Pulled") : "git pull failed"}
            action={pullOut.env_update_needed ? <Button size="sm" onClick={() => void updateEnv()}>Update env</Button> : undefined}
            onDismiss={() => setPullOut(null)}>
            <span className="sys-pre">{pullOut.ok ? (pullOut.stdout || "Already up to date.") : pullOut.error}</span>
          </Callout>
        )}
        {(env.job || env.error) && (
          <div className="sys-stack">
            <JobProgress job={env.job ? { ...env.job, log: null } : null} error={env.error} />
            {env.job && <LogView text={env.job.log} title="mamba env update" exportName="fasrc-env-update" maxHeight={420} />}
          </div>
        )}
      </CardBody>
    </Card>
  );
}

/** A short commit hash with the full one in its tooltip and a copy button. */
function Commit({ full, short, label }: { full: string | null | undefined; short: string | null | undefined; label: string }) {
  if (!full) return <span className="sys-dim">—</span>;
  return (
    <span className="sys-row">
      <code className="mono" title={full}>{short || full.slice(0, 7)}</code>
      <CopyButton value={full} label={`Copy the ${label} hash`} />
    </span>
  );
}

/** The server process and the runtime (provenance-level facts, collapsed). */
function ServerDetails() {
  const version = useVersion();
  const v = version.data;
  const consoleUpdated = useConsoleUpdate();
  const sys = useResource<SystemInfo>(SYSTEM_URL, [], { ttl: 30_000 });
  const s = sys.data;
  return (
    <Details summary="Server and runtime">
      <div className="sys-stack">
        {v?.behind && (
          <Callout tone="warn" title="Backend code changed — restart the server to load it">
            {v.changed_files?.length
              ? <>Changed since it started{v.changed_count && v.changed_count > v.changed_files.length ? ` (${v.changed_count} files, newest first)` : ""}:{" "}
                {v.changed_files.map((f, i) => <span key={f}>{i > 0 && ", "}<code className="mono">{f}</code></span>)}.</>
              : <>A backend file it loaded changed on disk after it started.</>}
          </Callout>
        )}
        {!v?.behind && consoleUpdated && (
          <Callout tone="warn" title="The console build changed — reload"
            action={<Button size="sm" onClick={() => window.location.reload()}>Reload</Button>}>
            This page was loaded from an older build; reload to get the new one.
          </Callout>
        )}
        {v && (
          <DefList dense items={[
            ["Server boot commit", <Commit full={v.boot_commit} short={v.boot_short} label="boot commit" />],
            ["Checkout HEAD", <Commit full={v.head_commit} short={v.head_short} label="HEAD" />],
            ["Started", v.started_at ? `${formatDateTime(v.started_at)} (${formatRelative(v.started_at)})` : "—"],
            ["Process", v.pid != null ? <code className="mono">{v.pid}</code> : "—"],
            ["Console build", v.dist?.built_at ? `${formatDateTime(v.dist.built_at)}${v.dist.index_hash ? ` · ${v.dist.index_hash}` : ""}` : "—"],
          ]} />
        )}
        {s && (
          <DefList dense items={[
            ["Python", <span title={s.python.executable}>{s.python.implementation} {s.python.version}</span>],
            ["Platform", `${s.platform.system} ${s.platform.release} · ${s.platform.machine}`],
            ["Node", s.node ?? <span className="sys-dim">not on the server PATH</span>],
            ["Bundle", `React ${reactVersion} · ${import.meta.env.MODE}`],
            ...Object.entries(s.packages).map(([name, ver]) => [name, ver ?? <span className="sys-dim">not installed</span>] as [string, ReactNode]),
            ["Noise model", <code className="mono">{s.noise_model}</code>],
          ]} />
        )}
        {sys.error && !s && <Callout tone="bad" title="Could not read /api/system">{sys.error.message}</Callout>}
      </div>
    </Details>
  );
}

export default function Code() {
  const [side] = useUrlState("side", "");
  const res = useResource<GitStatusResp>(STATUS_URL, [], { ttl: 5_000, poll: 30_000 });
  const version = useVersion();
  const fasrc = useFasrcStatus();
  const connected = !!fasrc.data?.ssh_connected;
  const git = useResource<RemoteGit>(connected ? FASRC_GIT_URL : null, [connected], { ttl: 60_000 });
  const fasrcRef = useRef<HTMLElement>(null);
  // ?side=fasrc (the old FASRC › Git view) opens on the FASRC checkout once
  // the local checkout above it has loaded, and keeps it in view while the
  // diff card above settles.
  const localReady = !!res.data || !!res.error;
  useArrivalScroll(fasrcRef, side === "fasrc", localReady);
  const laptop = res.data?.status.last?.hash?.slice(0, 7) ?? version.data?.head_short ?? null;
  const server = version.data?.boot_short ?? null;
  const sentence = codeSentence({
    laptop, server,
    fasrc: !fasrc.data || (connected && !git.data && !git.error) ? "loading"
      : connected && git.data?.ok !== false && git.data ? { head: git.data.head, relation: git.data.relation?.relation, ahead: git.data.relation?.ahead, behind: git.data.relation?.behind }
      : null,
  });
  // The count lives in the Changes card (in words and on its table); the
  // sentence only says there are some.
  const dirty = (res.data?.status.files ?? []).length;
  return (
    <Page className="sys-page">
      <SummaryLine className="sys-summary" >
        {sentence.tone === "warn" && <Badge tone="warn" dot>differs</Badge>}{" "}
        {sentence.text}{dirty ? " This laptop has uncommitted changes." : ""}
      </SummaryLine>
      <section className="sys-stack" aria-labelledby="code-local">
        <h2 id="code-local" className="sys-section">This laptop</h2>
        <LocalCheckout res={res} />
      </section>
      <section className="sys-stack" aria-labelledby="code-fasrc" ref={fasrcRef}>
        <h2 id="code-fasrc" className="sys-section">FASRC</h2>
        <FasrcCheckout git={git} connected={connected} />
      </section>
      <ServerDetails />
    </Page>
  );
}
