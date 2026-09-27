/* Ops › FASRC › Logs: the runs that left a log on FASRC (UI and CLI
 * submissions), and one run's .out / .err — paged from the end, a live
 * "follow" tail, and a server-side search over the whole file.
 * URL: `job` (a ledger job id — the robust deep link) or `run` (a runs-page
 * row), `task`, `lkind` (out|err), `lpage`, `follow`, `lq` (search), `rp`
 * (runs page). */
import { useMemo, useState } from "react";
import { useResource } from "../../../api/query";
import { formatBytes, formatRelative } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Badge, Button, Callout, DataTable, EmptyState, IconButton, Input, LogView, Segmented, Select, Skeleton, Switch,
  type DataColumn,
} from "../../../ui";
import {
  historyUrl, logGrepUrl, runsUrl, type HistoryResp, type LogGrep, type LogPage, type RunRow, type RunsResp,
} from "../api";
import {
  buildLogPageUrl, hasRunLogs, logPath, logTargetFromRow, preferredLogKind, type LogKind, type LogTarget,
} from "../fasrcLogs";
import { isLiveState, jobStateTone, pageForLine } from "../model";

const PAGE_SIZE = 1000;

function LogViewer({ target, onClose }: { target: LogTarget; onClose: () => void }) {
  const tasks = target.tasks ?? [];
  const [taskIdx, setTaskIdx] = useUrlState("task", 0);
  const task = tasks.length ? tasks[Math.min(Math.max(0, taskIdx), tasks.length - 1)] : null;
  const files = task ?? target;
  const [kindParam, setKind] = useUrlState<string>("lkind", "");
  const kind: LogKind = (kindParam === "err" || kindParam === "out") && logPath(files, kindParam)
    ? kindParam : preferredLogKind(files) ?? "out";
  const [page, setPage] = useUrlState("lpage", 0);
  const [follow, setFollow] = useUrlState("follow", false);
  const [query, setQuery] = useUrlState("lq", "");
  const [draft, setDraft] = useState(query);
  const path = logPath(files, kind);
  const live = isLiveState(target.state);
  const log = useResource<LogPage>(path ? buildLogPageUrl(path, follow ? 0 : page, PAGE_SIZE) : null,
    [path, follow ? 0 : page], { ttl: follow ? 0 : 30_000, poll: follow ? 5_000 : undefined });
  const grep = useResource<LogGrep>(path && query ? logGrepUrl(path, query) : null, [path, query], { ttl: 30_000 });
  const data = log.data?.path === path ? log.data : null;

  function jumpTo(line: number) {
    if (!data) return;
    setFollow(false);
    setPage(pageForLine(line, data.total_lines, PAGE_SIZE));
  }

  return (
    <div className="ops-stack">
      <div className="ops-toolbar" role="toolbar" aria-label="Log controls">
        <Button size="sm" variant="ghost" icon="chevronLeft" onClick={onClose}>Runs</Button>
        <strong className="ops-ellipsis">{target.label || target.name}</strong>
        {target.jobid && <code className="mono ops-dim">#{target.jobid}</code>}
        {target.state && <Badge size="sm" tone={jobStateTone(target.state)}>{target.state}</Badge>}
        <span className="ops-spacer" />
        {tasks.length > 0 && (
          <Select size="sm" aria-label="Array task" value={String(task?.index ?? 0)}
            onChange={(v) => { setTaskIdx(Number(v)); setPage(0); }}
            options={tasks.map((t) => ({ value: String(t.index), label: `#${t.index} · ${t.member}` }))} />
        )}
        <Segmented<LogKind> size="sm" aria-label="Stream" value={kind} onChange={(k) => { setKind(k); setPage(0); }}
          options={[{ value: "out", label: ".out", disabled: !files.out_path }, { value: "err", label: ".err", disabled: !files.err_path }]} />
        <Switch size="sm" checked={follow} onChange={(on) => { setFollow(on); if (on) setPage(0); }}>
          Follow{live ? "" : " (finished)"}
        </Switch>
      </div>
      <form className="ops-row" onSubmit={(e) => { e.preventDefault(); setQuery(draft.trim()); }}>
        <Input size="sm" value={draft} onChange={setDraft} icon="search" clearable placeholder="Search the whole file on FASRC…"
          aria-label="Search the log file" style={{ flex: "1 1 240px" }} />
        <Button size="sm" type="submit" disabled={!draft.trim() || !path}>Search</Button>
        {query && <Button size="sm" variant="ghost" onClick={() => { setQuery(""); setDraft(""); }}>Clear</Button>}
      </form>
      {query && (
        <div className="ops-grep" aria-live="polite">
          {grep.loading ? <Skeleton lines={2} />
            : grep.error ? <Callout tone="bad">{grep.error.message}</Callout>
            : grep.data && !grep.data.matches.length ? <span className="ops-dim ops-small">No line contains “{query}”.</span>
            : grep.data && (
              <>
                <span className="ops-dim ops-small">{grep.data.matches.length}{grep.data.truncated ? "+" : ""} matching lines</span>
                <ol className="ops-grep__list">
                  {grep.data.matches.map((m) => (
                    <li key={m.line}>
                      <button type="button" className="ops-grep__hit" onClick={() => jumpTo(m.line)}>
                        <span className="mono ops-dim">{m.line}</span> <span className="mono">{m.text}</span>
                      </button>
                    </li>
                  ))}
                </ol>
              </>
            )}
        </div>
      )}
      {!path ? <EmptyState compact icon="fileSearch" title="No log file recorded for this run" />
        : !data ? (log.error ? <Callout tone="bad" title="Could not read the log">{log.error.message}</Callout> : <Skeleton lines={8} />)
        : (
          <>
            <LogView text={data.content} title={<span className="mono ops-small">{path}</span>} maxHeight="min(66vh, 760px)"
              follow={follow} exportName={target.name + (kind === "err" ? ".err" : ".out")} empty="(empty file)" />
            <div className="ops-row">
              <Button size="sm" variant="ghost" icon="chevronLeft" disabled={!data.has_older} onClick={() => { setFollow(false); setPage(page + 1); }}>Older</Button>
              <Button size="sm" variant="ghost" iconRight="chevronRight" disabled={!data.has_newer || follow} onClick={() => setPage(Math.max(0, page - 1))}>Newer</Button>
              <Button size="sm" variant="ghost" disabled={page === 0 && !follow} onClick={() => setPage(0)}>Newest</Button>
              <span className="ops-dim mono ops-small">
                {data.total_lines ? `lines ${data.start_line.toLocaleString()}–${data.end_line.toLocaleString()} of ${data.total_lines.toLocaleString()}` : "empty file"}
              </span>
            </div>
          </>
        )}
    </div>
  );
}

function runTarget(run: RunRow): LogTarget {
  return {
    name: run.name, jobid: String(run.jobid ?? ""), label: run.label ?? null, state: run.state ?? null,
    out_path: run.out_path ?? null, err_path: run.err_path ?? null,
    tasks: run.tasks?.map((t) => ({ index: t.index, member: t.member, jobid: t.jobid, name: t.name,
      out_path: t.out_path ?? null, err_path: t.err_path ?? null })),
  };
}

export function LogsPanel({ fasrcConnected }: { fasrcConnected: boolean }) {
  const [job, setJob] = useUrlState("job", "");
  const [run, setRun] = useUrlState("run", "");
  const [runsPage, setRunsPage] = useUrlState("rp", 0);
  const clearTarget = useUrlState("task", 0)[1];
  const ledger = useResource<HistoryResp>(job ? historyUrl({ q: job, limit: 20 }) : null, [job], { ttl: 60_000 });
  const runs = useResource<RunsResp>(fasrcConnected && !job ? runsUrl(runsPage) : null, [runsPage, fasrcConnected], { ttl: 30_000 });
  const row = ledger.data?.rows.find((r) => String(r.jobid) === job) ?? null;
  const pageRun = run ? runs.data?.runs.find((r) => r.name === run) ?? null : null;
  const close = () => { setJob(""); setRun(""); clearTarget(0); };

  const columns = useMemo<DataColumn<RunRow>[]>(() => [
    { id: "state", header: "State", width: 104, accessor: (r) => r.state ?? "",
      cell: (r) => (r.state ? <Badge size="sm" tone={jobStateTone(r.state)}>{r.state}</Badge> : <span className="ops-dim">CLI</span>) },
    { id: "jobid", header: "Job", width: 92, cell: (r) => <code className="mono">{r.jobid ?? "—"}</code> },
    { id: "label", header: "Run", accessor: (r) => r.label ?? r.name,
      cell: (r) => <span className="ops-cell2"><span>{r.label ?? r.name}</span>
        <span className="ops-dim ops-small">{r.tasks?.length ? `array · ${r.tasks.length} tasks` : r.name}</span></span> },
    { id: "size", header: "Size", numeric: true, width: 96, accessor: (r) => (r.out_size ?? 0) + (r.err_size ?? 0),
      cell: (r) => <span className="mono ops-small">{formatBytes((r.out_size ?? 0) + (r.err_size ?? 0))}</span> },
    { id: "mtime", header: "Updated", width: 104, accessor: (r) => r.mtime ?? 0,
      cell: (r) => <span className="ops-dim ops-small">{formatRelative(r.mtime)}</span> },
  ], []);

  if (job) {
    if (ledger.loading && !ledger.data) return <Skeleton lines={6} />;
    if (!row) return <Callout tone="warn" title={`Job ${job} is not in the local job ledger`} onDismiss={close}>Pick its run from the list instead.</Callout>;
    if (!fasrcConnected) return <Callout tone="warn" title="FASRC offline">Logs are read from FASRC over SSH.</Callout>;
    return <LogViewer key={job} target={logTargetFromRow(row)} onClose={close} />;
  }
  if (!fasrcConnected) return <Callout tone="warn" title="FASRC offline">The run logs live on FASRC; connect to browse them.</Callout>;
  if (run && pageRun) return <LogViewer key={run} target={runTarget(pageRun)} onClose={close} />;
  const d = runs.data;
  return (
    <div className="ops-stack">
      {run && d && !pageRun && <Callout tone="warn" onDismiss={() => setRun("")}>Run {run} is not on this page.</Callout>}
      {runs.error && !d && <Callout tone="bad" title="Could not list the runs">{runs.error.message}</Callout>}
      <DataTable rows={d?.runs ?? []} columns={columns} rowKey={(r) => r.name} aria-label="Runs with logs"
        loading={runs.loading && !d} height={620} exportName="fasrc-runs"
        onRowClick={(r) => { if (!hasRunLogs(r)) return; if (r.jobid) { setRun(""); setJob(String(r.jobid)); } else setRun(r.name); }}
        empty="No run logs found." filterPlaceholder="Filter runs on this page…"
        toolbar={<>
          <IconButton size="sm" icon="chevronLeft" label="Newer runs" disabled={runsPage === 0} onClick={() => setRunsPage(Math.max(0, runsPage - 1))} />
          <span className="mono ops-small ops-dim">page {runsPage + 1}{d ? ` · ${d.total_runs} runs` : ""}</span>
          <IconButton size="sm" icon="chevronRight" label="Older runs" disabled={!d?.has_older} onClick={() => setRunsPage(runsPage + 1)} />
          <IconButton size="sm" icon="reset" label="Reload" onClick={() => runs.reload()} />
        </>} />
    </div>
  );
}
