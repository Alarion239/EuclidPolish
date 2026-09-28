/* One run's .out / .err on FASRC: paged from the end, a live "follow"
 * tail, and a server-side search over the whole file; one select per array
 * task (named by member). The log side panel of Runs › History.
 * URL: `task`, `lkind` (out|err), `lpage`, `follow`, `lq` (search). */
import { useState } from "react";
import { useResource } from "../../api/query";
import { useUrlState } from "../../hooks/useUrlState";
import {
  Button, Callout, EmptyState, Input, LogView, Segmented, Select, Skeleton, Switch,
} from "../../ui";
import { logGrepUrl, type LogGrep, type LogPage, type RunRow } from "./api";
import { buildLogPageUrl, logPath, preferredLogKind, type LogKind, type LogTarget } from "./fasrcLogs";
import { isLiveState, pageForLine } from "./model";
import "./runs.css";

const PAGE_SIZE = 1000;

export function LogViewer({ target }: { target: LogTarget }) {
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
    <div className="runs-stack">
      <div className="runs-row" role="toolbar" aria-label="Log controls">
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
      <form className="runs-row" onSubmit={(e) => { e.preventDefault(); setQuery(draft.trim()); }}>
        <Input size="sm" value={draft} onChange={setDraft} icon="search" clearable placeholder="Search the whole file on FASRC…"
          aria-label="Search the log file" style={{ flex: "1 1 240px" }} />
        <Button size="sm" type="submit" disabled={!draft.trim() || !path}>Search</Button>
        {query && <Button size="sm" variant="ghost" onClick={() => { setQuery(""); setDraft(""); }}>Clear</Button>}
      </form>
      {query && (
        <div className="runs-grep" aria-live="polite">
          {grep.loading ? <Skeleton lines={2} />
            : grep.error ? <Callout tone="bad">{grep.error.message}</Callout>
            : grep.data && !grep.data.matches.length ? <span className="runs-dim runs-small">No line contains “{query}”.</span>
            : grep.data && (
              <>
                <span className="runs-dim runs-small">{grep.data.matches.length}{grep.data.truncated ? "+" : ""} matching lines</span>
                <ol className="runs-grep__list">
                  {grep.data.matches.map((m) => (
                    <li key={m.line}>
                      <button type="button" className="runs-grep__hit" onClick={() => jumpTo(m.line)}>
                        <span className="mono runs-dim">{m.line}</span> <span className="mono">{m.text}</span>
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
            <LogView text={data.content} title={<span className="mono runs-small">{path}</span>} maxHeight="min(66vh, 760px)"
              follow={follow} exportName={target.name + (kind === "err" ? ".err" : ".out")} empty="(empty file)" />
            <div className="runs-row">
              <Button size="sm" variant="ghost" icon="chevronLeft" disabled={!data.has_older} onClick={() => { setFollow(false); setPage(page + 1); }}>Older</Button>
              <Button size="sm" variant="ghost" iconRight="chevronRight" disabled={!data.has_newer || follow} onClick={() => setPage(Math.max(0, page - 1))}>Newer</Button>
              <Button size="sm" variant="ghost" disabled={page === 0 && !follow} onClick={() => setPage(0)}>Newest</Button>
              <span className="runs-dim mono runs-small">
                {data.total_lines ? `lines ${data.start_line.toLocaleString()}–${data.end_line.toLocaleString()} of ${data.total_lines.toLocaleString()}` : "empty file"}
              </span>
            </div>
          </>
        )}
    </div>
  );
}

export function runTarget(run: RunRow): LogTarget {
  return {
    name: run.name, jobid: String(run.jobid ?? ""), label: run.label ?? null, state: run.state ?? null,
    out_path: run.out_path ?? null, err_path: run.err_path ?? null,
    tasks: run.tasks?.map((t) => ({ index: t.index, member: t.member, jobid: t.jobid, name: t.name,
      out_path: t.out_path ?? null, err_path: t.err_path ?? null })),
  };
}

