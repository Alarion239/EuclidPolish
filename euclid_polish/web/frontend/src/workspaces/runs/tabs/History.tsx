/* Runs › History (`/runs/history`): every run that has ended or is on
 * record — the FASRC job ledger (works offline) and this server's finished
 * local jobs — in ONE table, filtered by source, step, state and campaign
 * (the jobs a tracking campaign logged). Each row has labelled Clone and Logs
 * buttons; the selected run opens beside the table: its resource use, its
 * .out / .err log (paged, follow, search the whole file on FASRC) and, for
 * ensemble_train, the wall time per 1000 steps of the members it trained.
 * The "Log files" source lists the run logs on FASRC, CLI submissions too.
 * URL: `src` (all | slurm | local | files), `step`, `state`, `campaign`
 * (+ `logged=1`: the jobs that campaign logged, with their commit),
 * `run` (a SLURM job id, `local:<id>` or `log:<name>`), `logs=1`, the
 * table's `h.q` / `h.sort`, the log viewer's `task/lkind/lpage/follow/lq`
 * and the log-file pages' `rp`. The interim `hstep` / `hstate` keys are
 * read once as `step` / `state`. */
import { useEffect, useMemo, useRef } from "react";
import { Link } from "react-router-dom";
import { useJob, useJobsFeed } from "../../../api/jobs";
import { invalidate, useResource } from "../../../api/query";
import { pagePath } from "../../../app/nav";
import { usePageActions } from "../../../app/palette";
import { useFasrcStatus } from "../../../app/status";
import Plot from "../../../charts/Plot";
import { categorical } from "../../../colors";
import { formatBytes, formatCount, formatDateTime, formatDuration, formatNumber, formatRelative } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Badge, Button, Callout, Caption, Card, CardBody, CardHead, DataTable, EmptyState, FactsList, IconButton, JobProgress,
  Menu, Page, Segmented, Select, Skeleton, Toolbar, ToolbarGroup, ToolbarSpacer, Tooltip, confirm, toast,
  type DataColumn,
} from "../../../ui";
import {
  TRACKING_STATE_URL, TRAINING_CURVES_URL, campaignJobIdsUrl, historyUrl, runsUrl, type CampaignChoices,
  type CampaignJobIds, type HistoryResp, type RunRow, type RunsResp,
} from "../api";
import { TrackedJobs } from "../../notebook/TrackedJobs";
import { ConnectionBar } from "../Connection";
import { hasRunLogs, logTargetFromRow } from "../fasrcLogs";
import { LocalJobCard } from "../LocalJob";
import { LogViewer, runTarget } from "../LogViewer";
import {
  accountingNotes, cpuUsage, exitCodeText, formatMemory, gpuUsage, hasGpu, historyItems, jobLabel, jobStateTone, memberNames, membersText,
  memoryUsage, paramsSummary, revealBelowFold, rowParams, wallPer1k, type HistoryItem, type HistoryRow, type StepTimeCurve,
} from "../model";
import { useStepsStatus } from "../steps/StepCard";
import "../runs.css";

type Source = "all" | "slurm" | "local" | "files";
const SOURCES: Source[] = ["all", "slurm", "local", "files"];
const SOURCE_LABEL: Record<Source, string> = { all: "All", slurm: "SLURM", local: "This laptop", files: "Log files on FASRC" };
const parseSource = (raw: string): Source | undefined => ((SOURCES as string[]).includes(raw) ? raw as Source : undefined);

const pct = (v: number | null | undefined) => (v == null ? "—" : `${v.toFixed(0)}%`);
const LOCAL_DONE: Record<string, string> = { DONE: "COMPLETED" };

/** The seconds per 1000 steps of the members an ensemble_train run trained. */
function WallTime({ row }: { row: HistoryRow }) {
  const members = useMemo(() => memberNames(rowParams(row)), [row]);
  const curves = useResource<{ members: StepTimeCurve[] }>(members.length ? TRAINING_CURVES_URL : null, [], { ttl: 120_000 });
  const w = useMemo(() => wallPer1k(curves.data?.members ?? [], members), [curves.data, members]);
  if (!members.length) return null;
  const series = w.members.map((m, i) => ({
    x: m.points.map((p) => p[0]), y: m.points.map((p) => p[1]), color: categorical(i), name: m.name.replace("member_", "#"),
  }));
  return (
    <section className="runs-stack" aria-label="Wall time per 1000 steps">
      <h3 className="runs-section-title">Wall time per 1000 steps</h3>
      {curves.loading && !curves.data ? <Skeleton height={160} />
        : curves.error && !curves.data ? <Callout tone="bad">{curves.error.message}</Callout>
        : !series.some((s) => s.x.length) ? <p className="runs-note">No training log of {membersText(members)} on this laptop yet (pull the members first).</p>
        : (
          <>
            <Plot xDomain={[0, Math.max(1, ...series.flatMap((s) => s.x))]} yDomain={[0, Math.max(1, ...series.flatMap((s) => s.y)) * 1.1]}
              xLabel="step" yLabel="s / 1k steps" series={series} height={180} legend="auto" exportName={`wall-time-${row.jobid}`}
              aria-label={`Wall time per 1000 steps, ${membersText(members)}`} />
            <Caption>
              Median {formatNumber(w.median, { sig: 3 })} s per 1000 steps over {membersText(w.members.map((m) => m.name))};
              each member's whole training log on this laptop.
            </Caption>
          </>
        )}
    </section>
  );
}

/** The selected SLURM run: resource use once it ended, its log, its wall time. */
function SlurmRunPanel({ row, logs, onLogs, gpuStep, fasrcConnected }: {
  row: HistoryRow; logs: boolean; onLogs: (on: boolean) => void; gpuStep: boolean; fasrcConnected: boolean;
}) {
  const cpu = cpuUsage(row);
  const mem = memoryUsage(row);
  const gpu = gpuUsage(row);
  const notes = accountingNotes(row);
  const exitCode = exitCodeText(row);
  const params = paramsSummary(rowParams(row), 8);
  const state = String(row.state_display || row.state || row.db_state || "PENDING").toUpperCase();
  return (
    <Card>
      <CardHead title={<span className="runs-side__title">{jobLabel(row) || row.jobid}</span>}
        sub={<span className="mono">#{row.jobid}{row.step_id ? ` · ${row.step_id}` : ""}</span>}
        right={<Badge size="sm" tone={jobStateTone(state)}>{state}</Badge>} />
      <CardBody>
        <div className="runs-stack">
          <FactsList facts={[
            { label: "Submitted", value: formatDateTime(row.submitted_at) },
            row.elapsed_seconds ? { label: "Elapsed", value: formatDuration(Number(row.elapsed_seconds)), unit: row.req_time_limit ? `of ${row.req_time_limit}` : undefined } : null,
            cpu.requested != null && { label: "CPU", value: `${cpu.used == null ? "—" : cpu.used.toFixed(1)} / ${cpu.requested}`, unit: `cores · peak ${pct(cpu.pct)}` },
            mem.requested != null && { label: "Memory", value: `${formatMemory(mem.used)} / ${formatMemory(mem.requested)}` },
            gpuStep && hasGpu(row) && { label: "GPU", value: pct(gpu.mean), unit: `mean · memory ${pct(gpu.memPct)}` },
            exitCode ? { label: "Exit code", value: <code className="mono">{exitCode}</code> } : null,
          ]} />
          {notes && <Caption>Jobstats: {notes}</Caption>}
          {params && <Caption>{params}</Caption>}
          <div className="runs-row">
            {row.step_id && (
              <Button asChild size="sm" icon="copy">
                <Link to={`${pagePath("runs", { tab: "steps" })}?${new URLSearchParams({ step: String(row.step_id), clone: row.jobid }).toString()}`}>Clone</Link>
              </Button>
            )}
            <Button size="sm" variant={logs ? "primary" : "default"} onClick={() => onLogs(!logs)}>{logs ? "Hide logs" : "Logs"}</Button>
          </div>
          {logs && (!fasrcConnected
            ? <Callout tone="warn" title="FASRC offline" action={<ConnectionBar />}>The logs are read from FASRC over SSH.</Callout>
            : <LogViewer key={row.jobid} target={logTargetFromRow(row)} />)}
          {row.step_id === "ensemble_train" && <WallTime row={row} />}
        </div>
      </CardBody>
    </Card>
  );
}

/** The run logs on FASRC (console and CLI submissions), page by page. */
function LogFiles({ fasrcConnected, onOpen, active }: {
  fasrcConnected: boolean; onOpen: (run: RunRow) => void; active: string | null;
}) {
  const [page, setPage] = useUrlState("rp", 0);
  const runs = useResource<RunsResp>(fasrcConnected ? runsUrl(page) : null, [page, fasrcConnected], { ttl: 30_000 });
  const columns = useMemo<DataColumn<RunRow>[]>(() => [
    { id: "state", header: "State", width: 104, accessor: (r) => r.state ?? "",
      cell: (r) => (r.state ? <Badge size="sm" tone={jobStateTone(r.state)}>{r.state}</Badge> : <span className="runs-dim">CLI</span>) },
    { id: "label", header: "Run", accessor: (r) => r.label ?? r.name,
      cell: (r) => <span className="runs-cell2"><span>{r.label ?? r.name}</span>
        <span className="runs-dim runs-small mono">{r.jobid ? `#${r.jobid}` : r.name}{r.tasks?.length ? ` · ${r.tasks.length} array tasks` : ""}</span></span> },
    { id: "size", header: "Log size", numeric: true, width: 96, accessor: (r) => (r.out_size ?? 0) + (r.err_size ?? 0),
      cell: (r) => formatBytes((r.out_size ?? 0) + (r.err_size ?? 0)) },
    { id: "mtime", header: "Updated", width: 104, accessor: (r) => r.mtime ?? 0,
      cell: (r) => <span className="runs-dim runs-small" title={formatDateTime(r.mtime)}>{formatRelative(r.mtime)}</span> },
  ], []);
  if (!fasrcConnected) {
    return <Callout tone="warn" title="FASRC offline" action={<ConnectionBar />}>The run logs live on FASRC; connect to browse them.</Callout>;
  }
  const d = runs.data;
  return (
    <>
      {runs.error && !d && <Callout tone="bad" title="Could not list the run logs">{runs.error.message}</Callout>}
      <DataTable rows={d?.runs ?? []} columns={columns} rowKey={(r) => r.name} aria-label="Run logs on FASRC"
        loading={runs.loading && !d} height={620} exportName="fasrc-run-logs" activeKey={active}
        onRowClick={(r) => { if (hasRunLogs(r)) onOpen(r); }}
        empty="No run logs found." filterPlaceholder="Filter the runs on this page…"
        toolbar={<>
          <IconButton size="sm" icon="chevronLeft" label="Newer runs" disabled={page === 0} onClick={() => setPage(Math.max(0, page - 1))} />
          <span className="mono runs-small runs-dim">page {page + 1}{d ? ` of ${Math.max(1, Math.ceil(d.total_runs / d.page_size))}` : ""}</span>
          <IconButton size="sm" icon="chevronRight" label="Older runs" disabled={!d?.has_older} onClick={() => setPage(page + 1)} />
        </>} />
    </>
  );
}

export default function History() {
  const [source, setSource] = useUrlState<Source>("src", "all", { parse: parseSource });
  const [step, setStep] = useUrlState("step", "");
  const [state, setState] = useUrlState("state", "");
  const [campaign, setCampaign] = useUrlState("campaign", "");
  const [run, setRun] = useUrlState("run", "", { replace: false });
  const [logs, setLogs] = useUrlState("logs", false);
  const [logged, setLogged] = useUrlState("logged", false);
  const [oldStep, setOldStep] = useUrlState("hstep", "");
  const [oldState, setOldState] = useUrlState("hstate", "");
  // Links written during the regrouping carry the interim keys: read them once.
  useEffect(() => {
    if (oldStep) { if (!step) setStep(oldStep); setOldStep(""); }
    if (oldState) { if (!state) setState(oldState); setOldState(""); }
  }, [oldStep, oldState, step, state, setStep, setState, setOldStep, setOldState]);

  const fasrc = useFasrcStatus();
  const fasrcConnected = !!fasrc.data?.ssh_connected;
  const feed = useJobsFeed({ slurm: false });
  const steps = useStepsStatus();
  const res = useResource<HistoryResp>(historyUrl({ step, state }), [step, state], { ttl: 20_000 });
  const camps = useResource<CampaignChoices>(TRACKING_STATE_URL, [], { ttl: 60_000 });
  const campIds = useResource<CampaignJobIds>(campaign ? campaignJobIdsUrl(campaign) : null, [campaign], { ttl: 60_000 });
  const reconcile = useJob("fasrc:accounting");
  const d = res.data;

  const needsGpu = useMemo(() => new Map((steps.data?.steps ?? []).map((s) => [s.step_id, s.needs_gpu])), [steps.data]);
  const gpuStep = (r: HistoryRow | undefined) => !!r && (needsGpu.get(String(r.step_id ?? "")) ?? hasGpu(r));

  const all = useMemo(() => {
    let local = feed.jobs;
    if (step) local = local.filter((j) => j.kind === step);
    if (state) local = state === "unresolved" ? [] : local.filter((j) => (LOCAL_DONE[j.status.toUpperCase()] ?? j.status.toUpperCase()) === state);
    let items = historyItems(d?.rows ?? [], local);
    if (campaign) {
      const ids = new Set(campIds.data?.jobids ?? []);
      items = items.filter((i) => i.source === "slurm" && ids.has(i.id));
    }
    return items;
  }, [feed.jobs, d, step, state, campaign, campIds.data]);
  // What the campaign logged but the ledger lacks (a job the ledger never
  // recorded, or one beyond its newest 2000 rows): counted, and listed with its
  // logged commit and params on demand, so no logged job goes missing.
  const campaignGap = useMemo(() => {
    if (!campaign || !campIds.data || !d) return null;
    const ledger = new Set(d.rows.map((r) => String(r.jobid)));
    const ids = campIds.data.jobids;
    return { logged: ids.length, missing: step || state ? null : ids.filter((id) => !ledger.has(String(id))).length };
  }, [campaign, campIds.data, d, step, state]);
  const campaignName = campaign === "current" ? camps.data?.active?.title ?? "The active campaign"
    : campaign === "unassigned" ? "No campaign" : camps.data?.archived.find((a) => a._dir === campaign)?.title ?? campaign;
  const counts = useMemo(() => ({
    all: all.length, slurm: all.filter((i) => i.source === "slurm").length, local: all.filter((i) => i.source === "local").length,
  }), [all]);
  const rows = useMemo(() => (source === "all" || source === "files" ? all : all.filter((i) => i.source === source)), [all, source]);
  const showGpu = rows.some((i) => gpuStep(i.row));

  const selected = run && !run.startsWith("log:") ? all.find((i) => i.key === run) ?? null : null;
  const localId = run.startsWith("local:") ? run.slice("local:".length) : "";
  // A SLURM run named in the URL but filtered out (or beyond the page) is
  // still looked up in the ledger by its id.
  const lookup = useResource<HistoryResp>(run && !selected && !localId && !run.startsWith("log:")
    ? historyUrl({ q: run, limit: 20 }) : null, [run], { ttl: 60_000 });
  const slurmRow = selected?.row ?? lookup.data?.rows.find((r) => String(r.jobid) === run) ?? null;
  const filesRuns = useResource<RunsResp>(run.startsWith("log:") && fasrcConnected ? runsUrl(0) : null, [run], { ttl: 30_000 });
  const fileRun = run.startsWith("log:") ? filesRuns.data?.runs.find((r) => r.name === run.slice(4)) ?? null : null;

  async function runReconcile(scope: "unresolved" | "all") {
    if (scope === "all" && !(await confirm({ title: "Re-pull accounting for every finished job?",
      message: "One sacct + Jobstats query per job over SSH; this can take several minutes. Runs as a cancellable job.",
      confirmLabel: "Re-pull all" }))) return;
    await reconcile.run("/api/fasrc/refresh-accounting", { scope }, {
      onDone: (j) => {
        void invalidate("/api/fasrc/history");
        void invalidate("/api/fasrc/steps/");
        const r = j.result as { updated?: number; total?: number; resolved?: Record<string, string> } | null;
        if (j.status === "done") {
          const n = Object.keys(r?.resolved ?? {}).length;
          toast.success(scope === "unresolved" ? `Resolved ${n} of ${r?.total ?? 0} job state${(r?.total ?? 0) === 1 ? "" : "s"}` : `Re-recorded ${r?.updated ?? 0} jobs`);
        }
      },
    });
  }
  usePageActions([
    { id: "history-reconcile", label: "Reconcile unresolved SLURM job states", group: "Runs", keywords: ["sacct", "unknown"],
      disabled: !fasrcConnected || reconcile.busy, run: () => void runReconcile("unresolved") },
    { id: "history-failed", label: "Show failed runs", group: "Runs", run: () => setState("FAILED") },
    { id: "history-train", label: "Show the training runs", group: "Runs", keywords: ["ensemble_train"], run: () => setStep("ensemble_train") },
  ]);

  const stepOptions = useMemo(() => [{ value: "", label: "All steps" },
    ...Object.entries(d?.facets.steps ?? {}).sort((a, b) => b[1] - a[1]).map(([s, n]) => ({ value: s, label: `${s} (${n})` }))],
  [d?.facets.steps]);
  const stateOptions = useMemo(() => [{ value: "", label: "All states" },
    { value: "unresolved", label: `Unresolved (${d?.unresolved ?? 0})` },
    ...Object.entries(d?.facets.states ?? {}).sort((a, b) => b[1] - a[1]).map(([s, n]) => ({ value: s, label: `${s} (${n})` }))],
  [d?.facets.states, d?.unresolved]);
  const campaignOptions = useMemo(() => {
    const c = camps.data;
    return [{ value: "", label: "All campaigns" },
      ...(c?.active ? [{ value: "current", label: `${c.active.title} (active, ${c.jobs_count})` }] : []),
      { value: "unassigned", label: `Unassigned (${c?.unassigned_count ?? 0})` },
      ...(c?.archived ?? []).map((a) => ({ value: a._dir, label: a.title })),
      ...(campaign && c && campaign !== "current" && campaign !== "unassigned" && !c.archived.some((a) => a._dir === campaign)
        ? [{ value: campaign, label: campaign }] : []),
    ];
  }, [camps.data, campaign]);

  const openRun = (item: HistoryItem, withLogs: boolean) => { setRun(item.key); setLogs(withLogs); };
  // A narrow page stacks the run's card under the table: bring it into view.
  const sideRef = useRef<HTMLElement>(null);
  useEffect(() => { if (run) requestAnimationFrame(() => revealBelowFold(sideRef.current)); }, [run, logs]);
  const columns = useMemo<DataColumn<HistoryItem>[]>(() => [
    /* The Run cell's sub-line carries the id, the step and the date, so
       Submitted and Step are hidden columns (the column menu shows them) and
       the params keep their place at desktop width. */
    { id: "submitted", header: "Submitted", width: 132, hidden: true, accessor: (i) => i.submitted ?? 0,
      cell: (i) => <span className="mono runs-small" title={formatDateTime(i.submitted)}>{formatDateTime(i.submitted)}</span> },
    { id: "label", header: "Run", width: 240, accessor: (i) => `${i.label} ${i.id} ${i.step}`,
      cell: (i) => (
        <span className="runs-cell2">
          <span title={i.label}>{i.label}</span>
          <span className="runs-dim runs-small mono">
            {[i.source === "local" ? "this laptop" : `#${i.id}`, i.step, formatDateTime(i.submitted)].filter(Boolean).join(" · ")}
          </span>
        </span>
      ) },
    { id: "step", header: "Step", width: 150, hidden: true, cell: (i) => <code className="mono runs-small">{i.step || "—"}</code> },
    { id: "state", header: "State", width: 104,
      cell: (i) => {
        const r = i.row;
        const tip = !r ? `local job: ${i.state.toLowerCase()}` : r.state ? `sacct: ${r.state}${r.db_state ? ` · DB: ${r.db_state}` : ""}` : `no sacct verdict yet · DB: ${r.db_state ?? "—"}`;
        return <Tooltip content={tip}><span tabIndex={0}><Badge size="sm" tone={jobStateTone(LOCAL_DONE[i.state] ?? i.state)}>{i.state}</Badge></span></Tooltip>;
      } },
    { id: "elapsed", header: "Elapsed", numeric: true, width: 84, priority: 2, accessor: (i) => i.elapsed, cell: (i) => formatDuration(i.elapsed) },
    { id: "cpu", header: "CPU used", numeric: true, width: 92, priority: 3, accessor: (i) => (i.row ? cpuUsage(i.row).pct : null),
      cell: (i) => { if (!i.row) return ""; const u = cpuUsage(i.row); return <span className="mono runs-small" title={`peak ${pct(u.pct)} · mean ${pct(u.mean)}`}>{u.used == null ? "—" : u.used.toFixed(1)} / {u.requested ?? "—"}</span>; } },
    { id: "mem", header: "Memory", numeric: true, width: 110, priority: 3, accessor: (i) => (i.row ? memoryUsage(i.row).pct : null),
      cell: (i) => { if (!i.row) return ""; const u = memoryUsage(i.row); return <span className="mono runs-small">{formatMemory(u.used)} / {formatMemory(u.requested)}</span>; } },
    ...(showGpu ? [{ id: "gpu", header: "GPU", numeric: true, width: 110, priority: 2,
      accessor: (i: HistoryItem) => (i.row && gpuStep(i.row) ? gpuUsage(i.row).mean : null),
      cell: (i: HistoryItem) => { if (!i.row || !gpuStep(i.row)) return ""; const g = gpuUsage(i.row); return <span className="mono runs-small">{pct(g.mean)} · mem {pct(g.memPct)}</span>; } }] : []),
    { id: "params", header: "Params", width: 200, priority: 1, accessor: (i) => (i.row ? paramsSummary(rowParams(i.row)) : ""),
      cell: (i) => { const t = i.row ? paramsSummary(rowParams(i.row)) : ""; return <span className="mono runs-small runs-ellipsis" title={t}>{t || "—"}</span>; } },
    { id: "actions", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 136,
      cell: (i) => (
        <span className="runs-row-actions">
          {i.source === "slurm" && i.step && (
            <Button asChild size="sm" variant="ghost">
              <Link aria-label={`Clone run ${i.id}`}
                to={`${pagePath("runs", { tab: "steps" })}?${new URLSearchParams({ step: i.step, clone: i.id }).toString()}`}>Clone</Link>
            </Button>
          )}
          <Button size="sm" variant="ghost" aria-label={`Logs of ${i.label}`} onClick={() => openRun(i, true)}>Logs</Button>
        </span>
      ) },
  // eslint-disable-next-line react-hooks/exhaustive-deps
  ], [showGpu, needsGpu]);

  const side = source === "files" ? (fileRun ? (
    <Card>
      <CardHead title={<span className="runs-side__title">{fileRun.label ?? fileRun.name}</span>}
        sub={<span className="mono">{fileRun.jobid ? `#${fileRun.jobid}` : fileRun.name}</span>}
        right={<IconButton size="sm" icon="close" label="Close" onClick={() => setRun("")} />} />
      <CardBody><LogViewer key={fileRun.name} target={runTarget(fileRun)} /></CardBody>
    </Card>
  ) : null) : localId ? (
    <LocalJobCard key={localId} id={localId} stored={feed.jobs.find((j) => j.job_id === localId)} />
  ) : slurmRow ? (
    <SlurmRunPanel key={slurmRow.jobid} row={slurmRow} logs={logs} onLogs={setLogs} gpuStep={gpuStep(slurmRow)}
      fasrcConnected={fasrcConnected} />
  ) : run && !run.startsWith("log:") && (lookup.loading || res.loading) ? <Skeleton lines={5} />
    : run && !run.startsWith("log:") ? <Callout tone="warn" title={`Run ${run} is not in the job ledger`} onDismiss={() => setRun("")}>
      Look for it under the log files on FASRC.
    </Callout> : null;

  return (
    <Page className="runs-page">
      <Toolbar label="Run history filters">
        <ToolbarGroup label="Source" hideLabel>
          <Segmented<Source> size="sm" value={source} onChange={(v) => { setSource(v); if (v === "files" || run.startsWith("log:")) setRun(""); }}
            aria-label="Source" options={SOURCES.map((s) => ({
              value: s, label: s !== "files" && d ? `${SOURCE_LABEL[s]} · ${counts[s as "all" | "slurm" | "local"]}` : SOURCE_LABEL[s],
            }))} />
        </ToolbarGroup>
        {source !== "files" && <>
          <Select size="sm" value={step} onChange={setStep} options={stepOptions} aria-label="Step" />
          <Select size="sm" value={state} onChange={setState} options={stateOptions} aria-label="State" />
          <Select size="sm" value={campaign} onChange={setCampaign} options={campaignOptions} aria-label="Campaign" />
        </>}
        <ToolbarSpacer />
        {source !== "files" && (
          <Tooltip content={fasrcConnected ? "Re-pull sacct for the jobs without a final verdict (blank, UNKNOWN, DONE, or a stale RUNNING / PENDING)" : "Needs FASRC"}>
            <span>
              <Button size="sm" icon="reset" loading={reconcile.busy} disabled={!fasrcConnected || !d?.unresolved}
                onClick={() => void runReconcile("unresolved")}>
                Reconcile{d?.unresolved ? ` ${d.unresolved}` : ""}
              </Button>
            </span>
          </Tooltip>
        )}
        <Menu label="More history actions" trigger={<IconButton size="sm" icon="more" label="More history actions" />} items={[
          { label: "Re-pull accounting for every job", disabled: !fasrcConnected || reconcile.busy, onSelect: () => void runReconcile("all") },
          { label: "Reload", onSelect: () => { res.reload(); feed.refresh(); } },
        ]} />
      </Toolbar>
      {(reconcile.job || reconcile.error) && <JobProgress job={reconcile.job} error={reconcile.error} />}
      {res.error && !d && source !== "files" && <Callout tone="bad" title="Could not read the run history">{res.error.message}</Callout>}
      {campaign && campIds.error && <Callout tone="warn" title="Could not read the campaign's jobs">{campIds.error.message}</Callout>}
      {campaignGap && source !== "files" && (
        <div className="runs-row">
          <Caption>
            {campaign === "unassigned" ? "Unassigned jobs" : `“${campaignName}”`}: {formatCount(campaignGap.logged)} logged job{campaignGap.logged === 1 ? "" : "s"}
            {campaignGap.missing ? `, ${formatCount(campaignGap.missing)} not in the ledger` : ""}.
          </Caption>
          {campaignGap.logged > 0 && (
            <Button size="sm" variant="ghost" onClick={() => setLogged(!logged)}>{logged ? "Hide the logged jobs" : "Show the logged jobs"}</Button>
          )}
        </div>
      )}
      <div className="runs-split">
        <div className="runs-split__main">
          {source === "files" ? (
            <LogFiles fasrcConnected={fasrcConnected} active={run.startsWith("log:") ? run.slice(4) : null}
              onOpen={(r) => setRun(`log:${r.name}`)} />
          ) : (
            <DataTable rows={rows} columns={columns} rowKey={(i) => i.key} aria-label="Run history"
              loading={res.loading && !d} urlKey="h" exportName="run-history" height={640} activeKey={run || null}
              countText={null /* the Source chips carry the counts */}
              onRowClick={(i) => openRun(i, logs)}
              empty={<EmptyState compact icon="table" title={step || state || campaign ? "No run matches the filters" : "No runs recorded yet"} />}
              filterPlaceholder="Filter: text, state:FAILED, step:euclid_query…" />
          )}
          {campaign && logged && source !== "files" && (
            <Card>
              <CardHead title="Jobs the campaign logged" sub="Logged time, the commit at submission and the params, as the campaign recorded them" />
              <CardBody><TrackedJobs campaign={campaign} compact /></CardBody>
            </Card>
          )}
        </div>
        {side && <aside className="runs-split__side" aria-label="Selected run" ref={sideRef}>{side}</aside>}
      </div>
    </Page>
  );
}
