/* Live SLURM job views, shared by every step card, Runs › Live and the
 * `job:slurm/<jobid>` inspector (re-exported from src/fasrc.tsx):
 *   <SlurmMonitor jobid compact?> — polls /api/fasrc/jobs/<jobid>/status and
 *     folds the Reporter event stream into stage, progress ("step 10,650 /
 *     70,000 (15%)", once), GPU / CPU %, warnings/errors and one card per
 *     array task (in the full form, each task's live training curves under
 *     the cards); the full form adds cancel (confirmed), the ledger's
 *     resource use once the job has finished, and links to its logs and its
 *     row in Runs › History;
 *   <JobStatusBody status> — that status body alone;
 *   <TrainingCurve …> — per-step validation curves of a training run. */
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { cancelSlurmJob } from "../../../api/jobs";
import { useResource } from "../../../api/query";
import Plot from "../../../charts/Plot";
import { bandColor, C } from "../../../colors";
import { curveRecords, type CurveRec } from "../../../fasrcCurves";
import { formatDuration } from "../../../format";
import { useResolvedTheme } from "../../../state/prefs";
import { linearTicks, paddedDomain } from "../../../ticks";
import {
  Badge, Button, Callout, DefList, EmptyState, ProgressBar, Segmented, Skeleton, confirm, toast,
} from "../../../ui";
import { historyUrl, type HistoryResp } from "../api";
import {
  cpuUsage, finiteNumber, formatMemory, gpuUsage, hasGpu, isLiveState, jobStateTone, memoryUsage, progressText,
  rowState, type HistoryRow,
} from "../model";
import "./steps.css";

type SlurmStep = { current: number; total: number; label?: string };
type SlurmEvent = { ts: number; msg: string };
type SlurmResources = {
  cpu_percent?: number; gpu_percent?: number; gpu_mem_percent?: number; cpu_peak?: number; gpu_peak?: number;
};
export type SlurmStatus = {
  stage?: string; step?: SlurmStep | null;
  warnings?: SlurmEvent[]; errors?: SlurmEvent[];
  resources?: SlurmResources | null;
  metrics?: CurveRec[];
  has_events?: boolean;
};
type ArrayTask = { index: number; member: string; jobid: string; state?: string; reason?: string; status?: SlurmStatus };
type JobStatusResp = {
  ok: boolean; jobid: string; state: string; status?: SlurmStatus | null;
  array?: { count: number; max_parallel?: number; tasks?: ArrayTask[] } | null; error?: string;
};

const TERMINAL = new Set(["COMPLETED", "DONE", "TIMEOUT", "FAILED", "CANCELLED", "UNKNOWN", "OUT_OF_MEMORY", "NODE_FAIL"]);
const list = <T,>(v: T[] | null | undefined): T[] => (Array.isArray(v) ? v : []);

/** The stage / progress / gauges / warnings / errors of a job's live status. */
export function JobStatusBody({ status }: { status?: SlurmStatus | null }) {
  const st = status ?? {};
  const step = st.step;
  const res = st.resources;
  const warnings = list(st.warnings);
  const errors = list(st.errors);
  return (
    <div className="ops-mon__body">
      {st.stage && <div className="ops-mon__stage">{st.stage}</div>}
      {step && step.total > 0 && (
        <ProgressBar value={step.current} max={step.total} label={progressText(step)} />
      )}
      {res && (res.gpu_percent != null || res.cpu_percent != null) && (
        <div className="ops-mon__res">
          {res.gpu_percent != null && <span>GPU {res.gpu_percent.toFixed(0)}%</span>}
          {res.gpu_mem_percent != null && <span>GPU mem {res.gpu_mem_percent.toFixed(0)}%</span>}
          {res.cpu_percent != null && <span>CPU {res.cpu_percent.toFixed(0)}%</span>}
        </div>
      )}
      {!!warnings.length && (
        <ul className="ops-mon__events ops-mon__events--warn" aria-label="Warnings">
          {warnings.slice(-4).map((e, i) => <li key={i}>{e.msg}</li>)}
        </ul>
      )}
      {!!errors.length && (
        <ul className="ops-mon__events ops-mon__events--err" aria-label="Errors">
          {errors.slice(-4).map((e, i) => <li key={i}>{e.msg}</li>)}
        </ul>
      )}
    </div>
  );
}

function ArrayTasks({ tasks, count, maxParallel, curves = false }:
  { tasks: ArrayTask[]; count: number; maxParallel?: number; curves?: boolean }) {
  return (
    <div className="ops-mon__tasks">
      <div className="ops-mon__caption">{count} array tasks{maxParallel ? ` · ≤ ${maxParallel} at once` : ""}</div>
      {tasks.map((t) => (
        <div className="ops-mon__task" key={t.index}>
          <div className="ops-mon__head">
            <strong className="mono">{t.member}</strong>
            <span className="mono ops-dim">#{t.jobid}</span>
            {t.state && <Badge size="sm" tone={jobStateTone(t.state)}>{t.state}</Badge>}
          </div>
          {t.reason && t.state === "PENDING" && <div className="ops-dim">{t.reason}</div>}
          <JobStatusBody status={t.status} />
        </div>
      ))}
      {curves && tasks.filter((t) => curveRecords(t.status?.metrics).length > 0).map((t) => (
        <TrainingCurve key={t.index} title={`Training curves · ${t.member}`} eventRecords={t.status?.metrics} startedAt={1} />
      ))}
    </div>
  );
}

/** What the job ledger recorded for one SLURM job (resources, elapsed). */
function LedgerFacts({ row }: { row: HistoryRow }) {
  const cpu = cpuUsage(row);
  const mem = memoryUsage(row);
  const gpu = gpuUsage(row);
  return (
    <DefList dense items={[
      ["step", <code className="mono">{row.step_id || "—"}</code>],
      ["ledger", <Badge size="sm" tone={jobStateTone(rowState(row))}>{rowState(row)}</Badge>],
      ["elapsed", formatDuration(finiteNumber(row.elapsed_seconds))],
      ["CPU", `${cpu.used == null ? "—" : cpu.used.toFixed(1)} / ${cpu.requested ?? "—"} cores`],
      ["memory", `${formatMemory(mem.used)} / ${formatMemory(mem.requested)}`],
      hasGpu(row) ? ["GPU", `${gpu.mean == null ? "—" : `${gpu.mean.toFixed(0)}%`} mean · memory ${gpu.memPct == null ? "—" : `${gpu.memPct.toFixed(0)}%`}`] : null,
      row.exit_code ? ["exit", <code className="mono">{row.exit_code}</code>] : null,
    ]} />
  );
}

/** Live monitor of one SLURM job. Stops polling at a terminal state. */
export function SlurmMonitor({ jobid, compact = false }: { jobid: string; compact?: boolean }) {
  const [terminal, setTerminal] = useState(false);
  const status = useResource<JobStatusResp>(`/api/fasrc/jobs/${encodeURIComponent(jobid)}/status`, [jobid],
    { ttl: 1_000, poll: terminal ? undefined : 3_000 });
  const state = status.data?.state ?? "";
  const isTerminal = TERMINAL.has(state);
  if (isTerminal !== terminal) setTerminal(isTerminal);
  const ledger = useResource<HistoryResp>(compact ? null : historyUrl({ q: jobid, limit: 20 }), [jobid, terminal],
    { ttl: 30_000 });
  const row = ledger.data?.rows.find((r) => String(r.jobid) === String(jobid)) ?? null;

  async function cancel() {
    if (!(await confirm({ title: `Cancel SLURM job ${jobid}?`, message: "scancel on FASRC; the job stops at once.",
      tone: "danger", confirmLabel: "Cancel job" }))) return;
    const r = await cancelSlurmJob(jobid);
    if (r.ok) { toast.success(`Cancelled ${jobid}`); status.reload(); } else toast.error(r.error ?? "cancel refused");
  }

  if (status.loading && !status.data) return <div className="ops-mon"><Skeleton lines={3} /></div>;
  if (status.error && !status.data) {
    return (
      <Callout tone={status.error.status === 404 ? "warn" : "bad"} title={`Job ${jobid}`}>
        {status.error.status === 404 ? "This job is not in the local job DB (submitted elsewhere?)." : status.error.message}
      </Callout>
    );
  }
  const d = status.data!;
  const tasks = list(d.array?.tasks);
  return (
    <div className="ops-mon" data-compact={compact || undefined}>
      <div className="ops-mon__head">
        <Badge tone={jobStateTone(state)} dot={isLiveState(state)}>{state || "…"}</Badge>
        <span className="mono ops-dim">#{jobid}</span>
        <span className="ops-spacer" />
        {!compact && isLiveState(state) && <Button size="sm" variant="ghost" icon="stop" onClick={cancel}>Cancel</Button>}
      </div>
      {d.status && !d.status.has_events && isLiveState(state) && !d.status.stage && (
        <div className="ops-dim ops-small">No Reporter events yet.</div>
      )}
      <JobStatusBody status={d.status} />
      {d.array && <ArrayTasks tasks={tasks} count={d.array.count} maxParallel={d.array.max_parallel} curves={!compact} />}
      {!compact && (
        <>
          {/* the ledger's numbers are final only once the job has ended */}
          {row && isTerminal && <LedgerFacts row={row} />}
          <div className="ops-mon__links">
            <Button asChild size="sm" variant="ghost"><Link to={`/runs/history?run=${encodeURIComponent(jobid)}&logs=1`}>Logs</Link></Button>
            <Button asChild size="sm" variant="ghost"><Link to={`/runs/history?run=${encodeURIComponent(jobid)}`}>History</Link></Button>
          </div>
          {d.status && curveRecords(d.status.metrics).length > 0 && (
            <TrainingCurve eventRecords={d.status.metrics} startedAt={1} />
          )}
        </>
      )}
    </div>
  );
}

/* ── training curves ─────────────────────────────────────────────────────── */

type CurveResp = { ok: boolean; member?: string; records: CurveRec[] };
type CurveMetric = "psnr" | "loss";
const BANDS: [keyof CurveRec, string][] = [["psnr_vis", "VIS"], ["psnr_y_e", "Y_E"], ["psnr_j_e", "J_E"], ["psnr_h_e", "H_E"]];

/** Per-step validation PSNR / loss of a run: the Reporter metric events when
 *  present (live), else the run's wall-time window of the training log.
 *  `title` names it (an array task's curve names its member). */
export function TrainingCurve(
  { startedAt, endedAt, stepId, eventRecords, title = "Training curves" }:
  { startedAt?: number; endedAt?: number; stepId?: string | null; eventRecords?: CurveRec[]; title?: string },
) {
  const [metric, setMetric] = useState<CurveMetric>("psnr");
  const theme = useResolvedTheme();
  const live = curveRecords(eventRecords);
  const url = startedAt && !live.length && startedAt > 1
    ? `/api/fasrc/runs/training-curve.json?started_at=${startedAt}${endedAt ? `&ended_at=${endedAt}` : ""}${stepId ? `&step_id=${encodeURIComponent(stepId)}` : ""}`
    : null;
  const file = useResource<CurveResp>(url, [startedAt, endedAt, stepId], { ttl: 15_000, poll: endedAt ? undefined : 20_000 });
  const recs = live.length ? live : curveRecords(file.data?.records);
  const chart = useMemo(() => {
    if (!recs.length) return null;
    const xs = recs.map((r) => r.step);
    const series: { x: number[]; y: (number | null)[]; color: string; width?: number; name: string }[] = [];
    const ys: number[] = [];
    const push = (key: keyof CurveRec, color: string, name: string, width = 1.4) => {
      const y = recs.map((r) => { const v = r[key]; return typeof v === "number" && Number.isFinite(v) ? v : null; });
      if (!y.some((v) => v != null)) return;
      series.push({ x: xs, y, color, width, name });
      for (const v of y) if (v != null) ys.push(v);
    };
    if (metric === "psnr") {
      for (const [k, band] of BANDS) push(k, bandColor(band), band);
      push("psnr_stretched", C.mean, "joint", 2.4);
    } else push("loss", C.baseline, "loss", 2.2);
    if (!series.length) return null;
    const xDomain: [number, number] = [Math.min(...xs), Math.max(Math.max(...xs), Math.min(...xs) + 1)];
    const yDomain = paddedDomain(ys, { pad: 0.06, minSpan: 0.5 }) as [number, number];
    return { series, xDomain, yDomain, xTicks: linearTicks(xDomain, { count: 5 }), yTicks: linearTicks(yDomain, { count: 5 }) };
    // theme: token colours are read at build time
  }, [recs, metric, theme]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!startedAt) return null;
  return (
    <section className="ops-curve" aria-label={title}>
      <div className="ops-curve__head">
        <strong>{title}</strong>
        <span className="ops-dim ops-small">{live.length ? "live events" : file.data?.member || ""}</span>
        <span className="ops-spacer" />
        <Segmented<CurveMetric> size="sm" value={metric} onChange={setMetric} aria-label="Curve metric"
          options={[{ value: "psnr", label: "PSNR" }, { value: "loss", label: "loss" }]} />
      </div>
      {!recs.length ? (
        file.loading ? <Skeleton lines={3} />
          : <EmptyState compact icon="activity" title={file.error ? "Curves unavailable" : "No evaluation logged yet"}>
            {file.error ? file.error.message : "The curve appears after the first validation."}
          </EmptyState>
      ) : chart ? (
        <Plot xDomain={chart.xDomain} yDomain={chart.yDomain} xTicks={chart.xTicks} yTicks={chart.yTicks}
          xLabel="step" yLabel={metric === "psnr" ? "PSNR [dB]" : "loss"} series={chart.series} aspect={0.42}
          legend="auto" syncKey="ops-train" exportName={`training-${metric}`}
          aria-label={metric === "psnr" ? "Validation PSNR by step" : "Training loss by step"} />
      ) : <EmptyState compact title={`No ${metric} values`} />}
    </section>
  );
}
