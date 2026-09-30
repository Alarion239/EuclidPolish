/* Runs › Resources (`/runs/resources`): what past runs of each FASRC step
 * asked for against what they used — the local job ledger's sacct + Jobstats
 * accounting, read offline (`GET /api/fasrc/resources[/<step>]`) — and what to
 * ask for next (the resource advisor, spec 2026-09-30). The step list (runs,
 * completed share, OOM / timeouts, median CPU efficiency and GPU utilisation)
 * sits beside the picked step: one summary sentence, the medians and the
 * allocated-but-unused hours, the recommendation "for the next run like the
 * last one" with a link to where the step is submitted (none for a retired
 * step the console no longer registers), the per-run charts
 * (peak memory vs requested, elapsed vs time limit; OOM and TIMEOUT runs
 * marked) and the recent runs with requested-vs-used bars (a row opens the job
 * inspector). Every allocation is per array task. Opening the page only reads.
 * URL: `step` (default ensemble_train), the run table's `r.q` / `r.sort`. */
import { useEffect, useMemo, useRef } from "react";
import { Link } from "react-router-dom";
import { useResource } from "../../../api/query";
import { pagePath } from "../../../app/nav";
import { usePageActions } from "../../../app/palette";
import Plot from "../../../charts/Plot";
import { C, statusColor } from "../../../colors";
import { formatCount, formatDateTime, formatDuration, formatNumber } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Badge, Button, Callout, Caption, Card, CardBody, CardHead, DataTable, Details, EmptyState, FactsList, IconButton,
  Num, Page, Skeleton, SummaryLine, type DataColumn,
} from "../../../ui";
import { AdviceChanges, AdviceNotes } from "../../shared/ResourceAdvice";
import { PageLead } from "../../shared/PageLead";
import {
  ADVICE_FIELDS, adviceHeadline, adviceRows, fieldLabel, levelText, rateText, type Recommendation,
} from "../../shared/resourceAdviceModel";
import {
  RESOURCES_URL, stepResourcesUrl, type ResourcesResp, type RunUsage, type StepResourcesResp, type StepSummary,
} from "../api";
import { formatMemory, jobStateTone, revealBelowFold } from "../model";
import {
  barTone, chronological, finishedRuns, hoursText, memorySeries, outcomeClauses, pctText, runAt, runTicks, stepMeta,
  submitHome, timeSeries, usedFraction, utilText, waste, workText, yAxis, type RunSeries,
} from "../resourcesModel";
import "../runs.css";

const DEFAULT_STEP = "ensemble_train";
/** ensemble_train is an array job: one member per task. */
const perTaskWord = (stepId: string) => (stepId === "ensemble_train" ? "member" : undefined);

/* ── the step list ────────────────────────────────────────────────────────── */

function StepList({ steps, current, onPick }: { steps: StepSummary[]; current: string; onPick: (id: string) => void }) {
  return (
    <nav className="runs-steplist" aria-label="Steps with past runs">
      <ul>
        {steps.map((s) => (
          <li key={s.step_id}>
            <button type="button" className="runs-steplist__item" onClick={() => onPick(s.step_id)}
              aria-current={s.step_id === current ? "true" : undefined}>
              <span className="runs-steplist__label">{s.label || s.step_id}</span>
              <span className="runs-steplist__meta">
                <code className="mono">{s.step_id}</code>
                <Badge size="sm">{s.needs_gpu ? "GPU" : "CPU"}</Badge>
                {s.states.oom > 0 && <Badge size="sm" tone="bad">{s.states.oom} OOM</Badge>}
                {s.states.timeout > 0 && <Badge size="sm" tone="warn">{s.states.timeout} timeout{s.states.timeout === 1 ? "" : "s"}</Badge>}
              </span>
              <span className="runs-steplist__meta">{stepMeta(s)}</span>
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
}

/* ── the summary: one sentence, the medians, the unused hours ─────────────── */

function Overview({ s }: { s: StepSummary }) {
  const finished = finishedRuns(s.states);
  const clauses = outcomeClauses(s.states);
  const cpu = waste(s.cpu_hours_alloc, s.cpu_hours_used);
  const gpu = s.needs_gpu ? waste(s.gpu_hours_alloc, s.gpu_hours_used) : null;
  const mem = waste(s.mem_gb_hours_alloc, s.mem_gb_hours_used);
  const idleFact = (label: string, w: typeof cpu, unit: string) => w && {
    label, value: hoursText(w.idle), unit: `of ${hoursText(w.alloc)} ${unit} (${pctText(w.share)})`,
    tone: w.share > 0.5 ? "warn" as const : undefined,
  };
  return (
    <section className="runs-stack" aria-label="Usage summary">
      <div>
        <SummaryLine>
          <Num>{s.states.completed}</Num> of <Num>{finished}</Num> finished run{finished === 1 ? "" : "s"} completed
          {s.success_rate != null && <> (<Num>{pctText(s.success_rate)}</Num>)</>}
          {clauses.map((c, i) => (
            <span key={c.text}>{i === 0 ? "; " : i === clauses.length - 1 ? " and " : ", "}<Num tone={c.tone}>{c.n}</Num> {c.text}</span>
          ))}.
        </SummaryLine>
        {s.last_submitted_at && <Caption>Last submitted {formatDateTime(s.last_submitted_at)}.</Caption>}
      </div>
      <div className="runs-res__facts">
        <FactsList title="Median use" facts={[
          { label: "CPU efficiency", value: pctText(s.cpu_efficiency), hint: "Cores busy ÷ cores allocated over the run (sacct)" },
          s.needs_gpu && { label: "GPU utilisation", value: utilText(s.gpu_util), hint: "Mean GPU utilisation (Jobstats)" },
          { label: "Memory used", value: pctText(s.mem_ratio), unit: "of requested",
            hint: "Peak memory (sacct MaxRSS or Jobstats' node memory, the larger) ÷ requested" },
          s.peak_mem_p90_mb != null && { label: "Peak memory, p90", value: formatMemory(s.peak_mem_p90_mb) },
          { label: "Time used", value: pctText(s.time_ratio), unit: "of the limit" },
        ]} />
        <FactsList title="Allocated but unused" facts={[
          idleFact("CPU-hours", cpu, "CPU-h"),
          idleFact("GPU-hours", gpu, "GPU-h"),
          idleFact("Memory GB-hours", mem, "GB-h"),
        ]} />
      </div>
      <Caption>Per array task: an array job counts one task's allocation and its slowest task's elapsed.</Caption>
    </section>
  );
}

/* ── the recommendation for the next run like the last one ────────────────── */

function NextRun({ rec, stepId, needsGpu, registered }: {
  rec: Recommendation | null; stepId: string; needsGpu: boolean; registered: boolean;
}) {
  const home = submitHome(stepId);
  const perTask = perTaskWord(stepId);
  const fields = needsGpu ? ADVICE_FIELDS : ADVICE_FIELDS.filter((f) => f !== "n_gpus");
  const rows = adviceRows(rec, fields, perTask);
  const changed = new Set(rows.map((r) => r.field));
  // A retired step (only its ledger rows left) has nowhere to be submitted.
  const submit = registered ? (
    <Button asChild size="sm" iconRight="chevronRight"><Link to={home.to}>Submit in {home.label}</Link></Button>
  ) : (
    <Badge size="sm" title="This console no longer registers the step: these are its past runs only">Retired step</Badge>
  );
  if (!rec?.available) {
    return (
      <Card>
        <CardHead title="For the next run like the last one" right={submit} />
        <CardBody>
          <EmptyState compact icon="table" title="Nothing to recommend from yet">
            No run of this step completed, ran out of memory or timed out with its accounting recorded.
          </EmptyState>
        </CardBody>
      </Card>
    );
  }
  const b = rec.basis;
  const rate = rateText(b?.rate_s_per_unit, b?.units_label);
  return (
    <Card>
      <CardHead title="For the next run like the last one" sub={adviceHeadline(rec)} right={submit} />
      <CardBody>
        <div className="runs-stack">
          <div className="runs-res__facts">
            <FactsList title="Ask for" facts={fields.map((f) => ({
              label: fieldLabel(f, perTask), value: String(rec.resources[f] ?? "—"),
              unit: changed.has(f) ? `instead of ${String(rec.current[f] ?? "—")}` : "no change",
            }))} />
            <FactsList title="Based on" facts={[
              b && { label: "Runs", value: String(b.n_runs), unit: levelText(b) },
              b?.units != null && { label: "Planned work", value: formatCount(b.units), unit: b.units_label ?? undefined },
              !!rate && { label: "Rate", value: rate },
            ]} />
          </div>
          {rows.length > 0 && <AdviceChanges rows={rows} />}
          <AdviceNotes rec={rec} />
          {!!b?.jobids?.length && (
            <Details summary={`The ${b.jobids.length} runs it is based on`}>
              <p className="mono runs-small runs-dim">{b.jobids.map((j) => `#${j}`).join(" · ")}</p>
            </Details>
          )}
        </div>
      </CardBody>
    </Card>
  );
}

/* ── per-run charts ───────────────────────────────────────────────────────── */

function RunChart({ runs, s, title, unit, requestedName, usedName, markedName, markedColor, yFormat, exportName }: {
  runs: RunUsage[]; s: RunSeries; title: string; unit: string; requestedName: string; usedName: string;
  markedName: string; markedColor: string; yFormat: (v: number) => string; exportName: string;
}) {
  const ticks = useMemo(() => runTicks(runs), [runs]);
  const y = useMemo(() => yAxis(s), [s]);
  const series = [
    { x: s.x, y: s.requested, color: C.cross, dash: [5, 4], width: 1.5, name: requestedName },
    { x: s.x, y: s.used, color: C.mean, width: 2, dots: true, name: usedName },
    ...(s.marked.x.length ? [{ x: s.marked.x, y: s.marked.y, color: markedColor, mode: "scatter" as const,
      marker: "diamond" as const, width: 3, name: `${markedName} (${s.marked.x.length})` }] : []),
  ];
  return (
    <figure className="runs-res__chart">
      <figcaption className="runs-section-title">{title}</figcaption>
      <Plot xDomain={[0.5, Math.max(1.5, runs.length + 0.5)]} yDomain={y.domain} yScale={y.scale} xTicks={ticks}
        xLabel="Run" yLabel={unit} series={series} height={210} legend="auto" zoomAxes="x"
        xFormat={(v) => runAt(runs, v)} yFormat={yFormat} exportName={exportName} aria-label={title} />
    </figure>
  );
}

function UsageCharts({ runs, stepId }: { runs: RunUsage[]; stepId: string }) {
  const ordered = useMemo(() => chronological(runs), [runs]);
  const mem = useMemo(() => memorySeries(ordered), [ordered]);
  const time = useMemo(() => timeSeries(ordered), [ordered]);
  if (!ordered.length) return null;
  return (
    <Card>
      <CardHead title="Per run" sub="Oldest to newest; the dashed line is what each run asked for" />
      <CardBody>
        <div className="runs-res__charts">
          <RunChart runs={ordered} s={mem} title="Peak memory vs requested" unit="GB" requestedName="Requested"
            usedName="Peak used" markedName="Out of memory" markedColor={statusColor("bad")}
            yFormat={(v) => `${formatNumber(v, { sig: 3 })} GB`} exportName={`${stepId}-memory`} />
          <RunChart runs={ordered} s={time} title="Elapsed vs time limit" unit="hours" requestedName="Time limit"
            usedName="Elapsed" markedName="Timed out" markedColor={statusColor("warn")}
            yFormat={(v) => formatDuration(v * 3600)} exportName={`${stepId}-time`} />
        </div>
        <Caption>
          One point per run (per array task). A diamond marks a run that ran out of memory or hit its time limit: it
          needed more than it got, so its point is a lower bound.
        </Caption>
      </CardBody>
    </Card>
  );
}

/* ── the run table ────────────────────────────────────────────────────────── */

function UseBar({ used, requested, text, tone }: {
  used: number | null; requested: number | null; text: string; tone?: "bad" | "warn";
}) {
  const f = usedFraction(used, requested);
  return (
    <span className="runs-use" data-tone={tone}>
      <span className="runs-use__bar" aria-hidden="true">
        {f != null && <span className="runs-use__fill" style={{ width: `${Math.min(100, f * 100)}%` }} />}
      </span>
      <span className="mono runs-small">{text}</span>
    </span>
  );
}

function runColumns(needsGpu: boolean): DataColumn<RunUsage>[] {
  return [
    { id: "run", header: "Run", width: 220, accessor: (r) => `${r.label ?? ""} ${r.jobid} ${r.submitted_at ?? ""}`,
      sortFn: (a, b) => String(a.submitted_at ?? "").localeCompare(String(b.submitted_at ?? "")),
      cell: (r) => (
        <span className="runs-cell2">
          <span title={r.label ?? r.jobid}>{r.label || `#${r.jobid}`}</span>
          <span className="runs-dim runs-small mono">#{r.jobid} · {formatDateTime(r.submitted_at)}</span>
        </span>
      ) },
    { id: "state", header: "State", width: 116,
      cell: (r) => <Badge size="sm" tone={jobStateTone(r.state)}>{r.state}</Badge> },
    { id: "cpu", header: "CPU cores used", numeric: true, width: 150, priority: 3,
      accessor: (r) => usedFraction(r.cores_used, r.cpus),
      cell: (r) => <UseBar used={r.cores_used} requested={r.cpus}
        text={`${r.cores_used == null ? "—" : formatNumber(r.cores_used, { sig: 2 })} / ${r.cpus ?? "—"}`} /> },
    { id: "mem", header: "Peak memory", numeric: true, width: 170,
      accessor: (r) => usedFraction(r.peak_mem_mb, r.req_memory_mb),
      cell: (r) => { const f = usedFraction(r.peak_mem_mb, r.req_memory_mb);
        return <UseBar used={r.peak_mem_mb} requested={r.req_memory_mb} tone={barTone("memory", f, r.state)}
          text={`${formatMemory(r.peak_mem_mb)} / ${formatMemory(r.req_memory_mb)}`} />; } },
    { id: "time", header: "Elapsed", numeric: true, width: 170,
      accessor: (r) => usedFraction(r.elapsed_s, r.req_time_s),
      cell: (r) => { const f = usedFraction(r.elapsed_s, r.req_time_s);
        return <UseBar used={r.elapsed_s} requested={r.req_time_s} tone={barTone("time", f, r.state)}
          text={`${formatDuration(r.elapsed_s)} / ${formatDuration(r.req_time_s)}`} />; } },
    ...(needsGpu ? [{ id: "gpu", header: "GPU util", numeric: true, width: 88, priority: 2,
      accessor: (r: RunUsage) => r.gpu_util, cell: (r: RunUsage) => utilText(r.gpu_util) }] : []),
    { id: "work", header: "Work", numeric: true, width: 120, priority: 2, accessor: (r) => r.units,
      cell: (r) => <span className="mono runs-small">{workText(r) || "—"}</span> },
    { id: "plan", header: "Plan", width: 200, priority: 1, accessor: (r) => r.key_label ?? "",
      cell: (r) => <span className="runs-small runs-ellipsis" title={r.key_label ?? ""}>{r.key_label || "—"}</span> },
    ...(needsGpu ? [{ id: "gpumem", header: "GPU memory", numeric: true, width: 110, hidden: true,
      accessor: (r: RunUsage) => r.gpu_mem_used_mb, cell: (r: RunUsage) => formatMemory(r.gpu_mem_used_mb) }] : []),
    { id: "partition", header: "Partition", width: 100, hidden: true, accessor: (r) => r.partition ?? "" },
  ];
}

/* ── the tab ──────────────────────────────────────────────────────────────── */

export default function Resources() {
  const [stepId, setStepId] = useUrlState("step", DEFAULT_STEP);
  const list = useResource<ResourcesResp>(RESOURCES_URL, [], { ttl: 60_000 });
  const detail = useResource<StepResourcesResp>(stepId ? stepResourcesUrl(stepId) : null, [stepId], { ttl: 60_000 });
  const steps = useMemo(() => list.data?.steps ?? [], [list.data]);
  const d = detail.data?.step_id === stepId ? detail.data : null;
  const summary = d?.summary ?? steps.find((s) => s.step_id === stepId) ?? null;
  const needsGpu = !!summary?.needs_gpu;
  const columns = useMemo(() => runColumns(needsGpu), [needsGpu]);
  const missing = detail.error?.status === 404;

  const reload = () => { list.reload(); detail.reload(); };
  usePageActions([{ id: "resources-reload", label: "Reload the resource usage", group: "Runs",
    keywords: ["sacct", "memory", "cpu"], run: reload }]);

  // A narrow page stacks the step under the list: show the picked step.
  const bodyRef = useRef<HTMLDivElement>(null);
  const picked = useRef(false);
  const pick = (id: string) => { picked.current = true; setStepId(id); };
  useEffect(() => {
    if (picked.current) requestAnimationFrame(() => revealBelowFold(bodyRef.current));
    picked.current = false;
  }, [stepId]);

  const lead = (
    <PageLead right={<IconButton size="sm" icon="reset" label="Reload" onClick={reload} />}>
      What past runs of each FASRC step asked for against what they used (the local job ledger, sacct and Jobstats),
      and what to ask for next.
    </PageLead>
  );
  if (list.loading && !list.data) return <Page className="runs-page">{lead}<Skeleton lines={6} /></Page>;
  if (list.error && !list.data) {
    return <Page className="runs-page">{lead}<Callout tone="bad" title="Could not read the job ledger">{list.error.message}</Callout></Page>;
  }
  if (!steps.length && !d) {
    return (
      <Page className="runs-page">
        {lead}
        <EmptyState icon="table" title="No FASRC runs recorded yet">
          Runs appear here once a step was submitted from the console and its accounting came back.
        </EmptyState>
      </Page>
    );
  }
  return (
    <Page className="runs-page">
      {lead}
      <div className="runs-steps">
        <StepList steps={steps} current={stepId} onPick={pick} />
        <div className="runs-stack" ref={bodyRef}>
          {missing ? (
            <Callout tone="warn" title={`No runs of “${stepId}” in the job ledger`}
              onDismiss={steps[0] ? () => setStepId(steps[0].step_id) : undefined}>
              Pick a step from the list.
            </Callout>
          ) : detail.error && !d ? (
            <Callout tone="bad" title="Could not read this step's runs">{detail.error.message}</Callout>
          ) : !d ? <Skeleton lines={8} /> : (
            <>
              <div className="runs-res__head">
                <h2 className="runs-res__title">{d.summary.label || d.step_id}</h2>
                <code className="mono runs-dim runs-small">{d.step_id}</code>
                <span className="runs-spacer" />
                <Button asChild size="sm" variant="ghost" iconRight="chevronRight">
                  <Link to={`${pagePath("runs", { tab: "history" })}?${new URLSearchParams({ step: d.step_id }).toString()}`}>Logs in History</Link>
                </Button>
              </div>
              <Overview s={d.summary} />
              <NextRun rec={d.recommendation} stepId={d.step_id} needsGpu={needsGpu}
                registered={d.summary.registered !== false} />
              <UsageCharts runs={d.runs} stepId={d.step_id} />
              <Card>
                <CardHead title="Recent runs" sub="Requested vs used, per array task; a row opens the job" />
                <CardBody>
                  <DataTable rows={d.runs} columns={columns} rowKey={(r) => String(r.jobid)} aria-label={`${d.summary.label || d.step_id} runs`}
                    urlKey="r" exportName={`${d.step_id}-resources`} height={520} dense
                    inspect={(r) => ({ kind: "job", id: `slurm/${r.jobid}` })}
                    empty="No runs of this step recorded." filterPlaceholder="Filter: text, state:TIMEOUT…" />
                  {needsGpu && (
                    <Caption>
                      GPU memory (in Columns) is not a usage number: TensorFlow preallocates the whole card, so it
                      reads ~98 % whatever a run needs.
                    </Caption>
                  )}
                </CardBody>
              </Card>
            </>
          )}
        </div>
      </div>
    </Page>
  );
}
