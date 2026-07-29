/* FASRC pipeline: typed React components for the SLURM step registry + live
   job monitoring. `<StepCard>` submits one pipeline step with editable
   resources; `<SlurmMonitor>` folds the event stream into stage/progress/
   warnings/errors. Shared by Catalog, PSFs, TNG, Lens-finder and the FASRC
   page — the DRY replacement for the classic `fasrc_step_card.js`. */
import { useMemo, useState, type ReactNode } from "react";
import { getJSON, postForm } from "./api";
import { curveRecords, type CurveRec } from "./fasrcCurves";
import { asArray } from "./data";
import { useResource, usePolling } from "./hooks";
import Plot, { Legend } from "./charts/Plot";
import {
  Badge, Button, Card, CardBody, CardHead, ConnBadge, DefList, Empty, LogTail,
  NumberField, Field, Input, ProgressBar, Segmented, Spinner, Table, type Column,
} from "./ui";

export type StepDefaults = {
  partition: string; n_cpus: number; n_gpus: number; memory: string; time_limit: string;
};
export type Step = {
  step_id: string; label: string; needs_gpu: boolean;
  fixed_cpus?: number | null; fixed_gpus?: number | null; defaults: StepDefaults;
};
export type HstStatus = {
  ssh_connected: boolean;
  steps: Step[];
  artifacts: Record<string, unknown>;
  remote_paths: Record<string, string>;
};

export function useHstStatus() {
  return useResource<HstStatus>("/api/fasrc/hst/status");
}

type SlurmStep = { current: number; total: number; label?: string };
type SlurmEvent = { ts: number; msg: string };
type SlurmResources = {
  cpu_percent?: number; gpu_percent?: number; gpu_mem_percent?: number;
  cpu_peak?: number; gpu_peak?: number;
};
export type SlurmStatus = {
  stage?: string; step?: SlurmStep | null;
  warnings?: SlurmEvent[]; errors?: SlurmEvent[];
  resources?: SlurmResources | null;
  metrics?: CurveRec[];
};
type JobStatusResp = {
  ok: boolean; jobid: string; state: string; status?: SlurmStatus;
  array?: CSArray | null; error?: string;
};

const TERMINAL_SLURM = new Set(["COMPLETED", "DONE", "TIMEOUT", "FAILED", "CANCELLED"]);
export const jobStateTone = (s: string): "good" | "warn" | "bad" | undefined =>
  s === "COMPLETED" || s === "DONE" ? "good"
    : s === "FAILED" || s === "CANCELLED" || s === "TIMEOUT" ? "bad"
    : s === "RUNNING" ? undefined : "warn";

/** The stage/progress/gauges/warnings/errors body of a job's live status —
 *  shared by the inline `<SlurmMonitor>` and the Current Submission panel. */
export function JobStatusBody({ status }: { status?: SlurmStatus | null }) {
  const st = status ?? {};
  const step = st.step;
  const res = st.resources;
  const warnings = asArray<SlurmEvent>(st.warnings);
  const errors = asArray<SlurmEvent>(st.errors);
  return (
    <>
      {st.stage && <div className="fasrc-mon__stage">{st.stage}</div>}
      {step && step.total > 0 && (
        <ProgressBar value={step.current} max={step.total}
          label={`${step.label ? step.label + " " : ""}${step.current}/${step.total}`} />
      )}
      {res && (res.gpu_percent != null || res.cpu_percent != null) && (
        <div className="fasrc-mon__res mono">
          {res.gpu_percent != null && <span>GPU {res.gpu_percent.toFixed(0)}%</span>}
          {res.gpu_mem_percent != null && <span>mem {res.gpu_mem_percent.toFixed(0)}%</span>}
          {res.cpu_percent != null && <span>CPU {res.cpu_percent.toFixed(0)}%</span>}
        </div>
      )}
      {!!warnings.length && (
        <ul className="fasrc-mon__events fasrc-mon__events--warn">
          {warnings.slice(-4).map((e, i) => <li key={i}>{e.msg}</li>)}
        </ul>
      )}
      {!!errors.length && (
        <ul className="fasrc-mon__events fasrc-mon__events--err">
          {errors.slice(-4).map((e, i) => <li key={i}>{e.msg}</li>)}
        </ul>
      )}
    </>
  );
}

/** Live SLURM job monitor — polls `/api/fasrc/jobs/<jobid>/status` and folds the
 *  event stream into a stage chip, progress bar, resource gauges + warn/err
 *  lists. Stops polling once the job reaches a terminal state. */
export function SlurmMonitor({ jobid }: { jobid: string }) {
  const [resp, setResp] = useState<JobStatusResp | null>(null);
  const terminal = resp ? TERMINAL_SLURM.has(resp.state) : false;

  usePolling(() => {
    getJSON<JobStatusResp>(`/api/fasrc/jobs/${jobid}/status`).then((r) => r && setResp(r));
  }, 1500, !terminal);

  if (!resp) return <div className="fasrc-mon"><Spinner /> <span className="mono">job {jobid}</span></div>;
  return (
    <div className="fasrc-mon">
      <div className="fasrc-mon__head">
        <Badge tone={jobStateTone(resp.state)}>{resp.state || "…"}</Badge>
        <span className="mono fasrc-mon__jobid">#{jobid}</span>
      </div>
      <JobStatusBody status={resp.status} />
      {resp.array && <ArrayTaskStatuses array={resp.array} />}
    </div>
  );
}

/* ── Current Submission ─────────────────────────────────────────────────── */
type CSJob = {
  jobid?: string; label?: string; state?: string; nodes?: string;
  time?: string; time_limit?: string; start_time?: string; reason?: string;
  started_at?: number; ended_at?: number;
  step_id?: string | null;
};
type JobstatsSnapshot = {
  accounting_source?: string;
  jobstats_cpu_util?: number | string;
  jobstats_cpu_memory_util?: number | string;
  jobstats_cpu_memory_used_mb?: number | string;
  jobstats_cpu_memory_alloc_mb?: number | string;
  jobstats_gpu_util?: number | string;
  jobstats_gpu_memory_util?: number | string;
  jobstats_gpu_memory_used_mb?: number | string;
  jobstats_gpu_memory_total_mb?: number | string;
};
type CSQueue = {
  count: number; halted: boolean; halted_reason?: string | null;
  names: string[]; active_jobid?: string | null;
};
type CSArrayTask = {
  index: number; member: string; jobid: string; state?: string;
  reason?: string; nodes?: string; time?: string; status?: SlurmStatus;
};
type CSArray = { count: number; max_parallel?: number; tasks?: CSArrayTask[] };
type CurrentSubmissionResp = {
  ok: boolean; stale?: boolean;
  current: { job: CSJob; status?: SlurmStatus | null; array?: CSArray | null;
    accounting?: JobstatsSnapshot | null } | null;
  queue: CSQueue;
};

function ArrayTaskStatuses({ array }: { array: CSArray }) {
  const tasks = asArray<CSArrayTask>(array.tasks);
  return <div className="grid" style={{ gap: "var(--s3)", marginTop: "var(--s3)" }}>
    <div className="muted">{array.count} independent model tasks
      {array.max_parallel ? ` · up to ${array.max_parallel} running at once` : ""}</div>
    {tasks.map((task) => <div className="fasrc-mon" key={task.index}>
      <div className="fasrc-mon__head">
        <strong className="mono">{task.member}</strong>
        <span className="mono fasrc-mon__jobid">#{task.jobid}</span>
        {task.state && <Badge tone={jobStateTone(task.state)}>{task.state}</Badge>}
      </div>
      {task.reason && task.state === "PENDING" && <div className="muted">{task.reason}</div>}
      <JobStatusBody status={task.status} />
    </div>)}
  </div>;
}

function JobstatsSnapshotBody({ snapshot }: { snapshot?: JobstatsSnapshot | null }) {
  if (!snapshot) return null;
  const cpu = finiteNumber(snapshot.jobstats_cpu_util);
  const cpuMemory = finiteNumber(snapshot.jobstats_cpu_memory_util);
  const gpu = finiteNumber(snapshot.jobstats_gpu_util);
  const gpuMemory = finiteNumber(snapshot.jobstats_gpu_memory_util);
  const gpuUsed = finiteNumber(snapshot.jobstats_gpu_memory_used_mb);
  const gpuTotal = finiteNumber(snapshot.jobstats_gpu_memory_total_mb);
  if (cpu == null && cpuMemory == null && gpu == null && gpuMemory == null) return null;
  return (
    <div className="fasrc-mon__res mono" title="Throttled Jobstats snapshot">
      <span>Jobstats</span>
      {cpu != null && <span>CPU {cpu.toFixed(0)}%</span>}
      {cpuMemory != null && <span>CPU mem {cpuMemory.toFixed(0)}%</span>}
      {gpu != null && <span>GPU {gpu.toFixed(0)}%</span>}
      {gpuMemory != null && (
        <span>GPU mem {gpuUsed != null && gpuTotal != null
          ? `${formatMemory(gpuUsed)} / ${formatMemory(gpuTotal)} (${gpuMemory.toFixed(0)}%)`
          : `${gpuMemory.toFixed(0)}%`}</span>
      )}
    </div>
  );
}

/** The live "what's running right now" panel — the local submission queue plus
 *  the current PENDING/RUNNING job (state, elapsed, limit + extend, cancel,
 *  stage/progress/warnings/errors). Job-agnostic; polls current-submission. */
export function CurrentSubmission() {
  const [resp, setResp] = useState<CurrentSubmissionResp | null>(null);
  const [busy, setBusy] = useState(false);
  const load = () =>
    getJSON<CurrentSubmissionResp>("/api/fasrc/current-submission").then((r) => r && setResp(r));
  usePolling(load, 5000);

  async function control(url: string, body: Record<string, string>) {
    setBusy(true);
    try { await postForm(url, body); } catch { /* re-polled below */ }
    finally { setBusy(false); load(); }
  }

  const cur = resp?.current;
  const q = resp?.queue;
  const queueNames = asArray<string>(q?.names);
  return (
    <div className="grid" style={{ gap: "var(--s4)" }}>
      {q && (q.count > 0 || q.halted) && (
        <Card>
          <CardHead title="Submission queue" sub={`${q.count} queued`}
            right={q.count > 0 && <Button size="sm" variant="ghost" disabled={busy}
              onClick={() => control("/api/fasrc/queue/clear", {})}>clear</Button>} />
          <CardBody>
            {q.halted && <div className="job-panel job-panel--err"><LogTail text={q.halted_reason || "queue halted"} /></div>}
            {queueNames.length
              ? <ul className="fasrc-queue">{queueNames.map((n, i) => <li key={i}>{n}</li>)}</ul>
              : <span className="muted">nothing queued</span>}
          </CardBody>
        </Card>
      )}
      {!cur ? (
        <Card><CardBody><Empty>{resp
          ? "No active job — submit one from its page (Train members, Sky, Cutouts, …)."
          : <><Spinner /> loading…</>}</Empty></CardBody></Card>
      ) : (
        <Card>
          <CardHead title={cur.job.label || "job"} sub={<code className="mono">#{cur.job.jobid}</code>}
            right={<Badge tone={jobStateTone(cur.job.state || "")}>{cur.job.state || "…"}{resp?.stale ? " · stale" : ""}</Badge>} />
          <CardBody>
            <DefList items={[
              ["node", cur.job.nodes || "—"],
              ["elapsed", <span className="mono">{cur.job.time || "—"}</span>],
              ["limit", <span className="mono">{cur.job.time_limit || "—"}</span>],
              ...(cur.job.state === "PENDING" ? [
                ["est. start", cur.job.start_time || "—"] as [string, ReactNode],
                ["reason", cur.job.reason || "—"] as [string, ReactNode],
              ] : []),
            ]} />
            <div className="row" style={{ gap: 8, marginTop: "var(--s3)" }}>
              <Button size="sm" variant="ghost" disabled={busy}
                onClick={() => control("/api/fasrc/cancel", { jobid: cur.job.jobid || "" })}>cancel job</Button>
            </div>
            <div style={{ marginTop: "var(--s3)" }}>
              <JobStatusBody status={cur.status} />
              {cur.array && <ArrayTaskStatuses array={cur.array} />}
              <JobstatsSnapshotBody snapshot={resp?.current?.accounting} />
            </div>
          </CardBody>
        </Card>
      )}
      {cur?.job.started_at && !cur.array && <TrainingCurve startedAt={cur.job.started_at}
        endedAt={cur.job.ended_at} stepId={cur.job.step_id}
        eventRecords={cur.status?.metrics} />}
    </div>
  );
}

/* ── Training curves (browser-rendered) ─────────────────────────────────────
   The per-step validation records for a run's wall-time window, drawn live in
   the browser with the shared <Plot> — no server matplotlib. Polls while the
   card is up so an in-flight run's curve grows. */
type CurveResp = { ok: boolean; member?: string; records: CurveRec[] };
type CurveMetric = "psnr" | "loss";

const BANDS: [keyof CurveRec, string, string][] = [
  ["psnr_vis", "VIS", "--series-mean"],
  ["psnr_y_e", "Y_E", "--cat-6"],
  ["psnr_j_e", "J_E", "--series-comb"],
  ["psnr_h_e", "H_E", "--series-baseline"],
];

function niceTicks(lo: number, hi: number, kilo = false): { v: number; label: string }[] {
  if (!isFinite(lo) || !isFinite(hi) || hi <= lo) return [];
  const out: { v: number; label: string }[] = [];
  for (let i = 0; i <= 4; i++) {
    const v = lo + ((hi - lo) * i) / 4;
    out.push({ v, label: kilo ? `${Math.round(v / 1000)}k` : (v >= 100 ? v.toFixed(0) : v.toFixed(2)) });
  }
  return out;
}

export function TrainingCurve(
  { startedAt, endedAt, stepId, eventRecords }:
  { startedAt?: number; endedAt?: number; stepId?: string | null; eventRecords?: CurveRec[] },
) {
  const [resp, setResp] = useState<CurveResp | null>(null);
  // `loading` only until the FIRST fetch resolves; after that we show the curve,
  // a "no eval yet" note, or the error — never an endless spinner. `err` holds a
  // transient failure (e.g. FASRC SSH dropped); we keep polling so the curve
  // self-heals when the connection returns, and keep any prior curve on screen.
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string>("");
  const [metric, setMetric] = useState<CurveMetric>("psnr");
  usePolling(() => {
    if (!startedAt) return;
    const url = `/api/fasrc/runs/training-curve.json?started_at=${startedAt}${endedAt ? `&ended_at=${endedAt}` : ""}${stepId ? `&step_id=${encodeURIComponent(stepId)}` : ""}`;
    fetch(url, { headers: { Accept: "application/json" } })
      .then(async (r) => {
        const j = (await r.json().catch(() => null)) as (CurveResp & { error?: string }) | null;
        setLoading(false);
        if (!r.ok || !j || j.ok === false) { setErr(j?.error || `HTTP ${r.status}`); return; }
        setErr(""); setResp(j);
      })
      .catch(() => { setLoading(false); setErr("request failed"); });
  }, 20000, !!startedAt && !curveRecords(eventRecords).length);

  // Reporter events are the canonical live source and already arrive with the
  // current-submission poll.  The file endpoint remains a compatibility
  // fallback for older jobs that did not emit metric events.
  const liveRecs = curveRecords(eventRecords);
  const recs = liveRecs.length ? liveRecs : curveRecords(resp?.records);
  const liveMember = liveRecs.length ? liveRecs[liveRecs.length - 1].member : null;
  const sourceReady = liveRecs.length > 0 || resp != null;
  const chart = useMemo(() => {
    if (!recs.length) return null;
    const css = (n: string, f: string) => {
      if (typeof document === "undefined") return f;
      return getComputedStyle(document.documentElement).getPropertyValue(n).trim() || f;
    };
    const xs = recs.map((r) => r.step);
    const xDomain: [number, number] = [Math.min(...xs), Math.max(...xs) || 1];
    const series: { x: number[]; y: (number | null)[]; color: string; width?: number; alpha?: number }[] = [];
    const legend: { label: string; color: string }[] = [];
    const ys: number[] = [];
    const push = (key: keyof CurveRec, color: string, label: string, width = 2) => {
      const y = recs.map((r) => { const v = r[key]; return typeof v === "number" && isFinite(v) ? v : null; });
      if (!y.some((v) => v != null)) return;
      series.push({ x: xs, y, color, width });
      legend.push({ label, color });
      for (const v of y) if (v != null) ys.push(v);
    };
    if (metric === "psnr") {
      BANDS.forEach(([k, lbl, tok]) => push(k, css(tok, "#888"), lbl, 1));
      push("psnr_stretched", css("--text", "#111"), "PSNR (joint)", 2.4);
    } else {
      push("loss", css("--series-baseline", "#e11d48"), "loss", 2.4);
    }
    if (!series.length || !ys.length) return null;
    let lo = Math.min(...ys), hi = Math.max(...ys);
    const pad = (hi - lo) * 0.06 || 0.5;
    lo -= pad; hi += pad;
    return {
      series, legend, xDomain, yDomain: [lo, hi] as [number, number],
      xTicks: niceTicks(xDomain[0], xDomain[1], true),
      yTicks: niceTicks(lo, hi),
    };
  }, [recs, metric]);

  if (!startedAt) return null;
  return (
    <Card>
      <CardHead title="Training curves"
        sub={liveRecs.length ? `live events${liveMember != null ? ` · member ${liveMember}` : ""}` : resp?.member || (resp ? "this run" : undefined)}
        right={<Segmented<CurveMetric> value={metric} onChange={setMetric}
          options={[{ value: "psnr", label: "PSNR" }, { value: "loss", label: "loss" }]} />} />
      <CardBody>
        {loading && !liveRecs.length ? <Empty><Spinner /> loading curves…</Empty>
          : !sourceReady ? <Empty>curves unavailable{err ? ` — ${err}` : ""} · retrying every 20 s</Empty>
          : !recs.length ? <Empty>no eval logged yet — the curve appears after the first validation</Empty>
          : !chart ? <Empty>no {metric} data for this run</Empty>
          : (<>
            {err && <div className="muted" style={{ fontSize: 12, marginBottom: 6 }}>last refresh failed ({err}) — showing the last good curve</div>}
            <Plot title={metric === "psnr" ? "validation PSNR vs step [dB]" : "training loss vs step"}
              xDomain={chart.xDomain} yDomain={chart.yDomain} xTicks={chart.xTicks} yTicks={chart.yTicks}
              xLabel="step" yLabel={metric === "psnr" ? "PSNR [dB]" : "loss"}
              series={chart.series} aspect={0.42} />
            <Legend items={chart.legend} />
          </>)}
      </CardBody>
    </Card>
  );
}

type HistoryRow = {
  jobid: string; submitted_at?: string; state?: string; params_json?: string;
  partition?: string; req_cpus?: string | number; req_gpus?: string | number;
  req_memory?: string; req_time_limit?: string; elapsed_seconds?: string;
  alloc_cpus?: string | number; alloc_gpus?: string | number;
  cpu_efficiency?: string; max_rss_mb?: string; alloc_memory_mb?: string;
  cpu_util_mean?: string; cpu_util_peak?: string; gpu_util_mean?: string;
  gpu_util_peak?: string; gpu_mem_peak?: string; gpu_mem_peak_mb?: string;
  gpu_mem_util_peak?: string; accounting_source?: string;
  jobstats_cpu_util?: string; jobstats_cpu_memory_util?: string;
  jobstats_cpu_memory_used_mb?: string; jobstats_cpu_memory_alloc_mb?: string;
  jobstats_gpu_util?: string; jobstats_gpu_memory_util?: string;
  jobstats_gpu_memory_used_mb?: string; jobstats_gpu_memory_total_mb?: string;
  jobstats_notes_json?: string;
};
type HistoryResp = { ok: boolean; history?: HistoryRow[] };
type HistoryParamsMap = Record<string, unknown>;

const finiteNumber = (value: unknown): number | null => {
  const number = Number(value);
  return value !== "" && value != null && isFinite(number) ? number : null;
};

const formatElapsed = (value: unknown): string => {
  const seconds = finiteNumber(value);
  if (seconds == null) return "—";
  const minutes = Math.max(0, Math.round(seconds / 60));
  return `${Math.floor(minutes / 60)}:${String(minutes % 60).padStart(2, "0")}`;
};

const memoryMegabytes = (value: unknown): number | null => {
  const number = finiteNumber(value);
  if (number != null) return number;
  const match = String(value ?? "").trim().match(/^([\d.]+)\s*([KMGT])?i?B?$/i);
  if (!match) return null;
  const scale = { k: 1 / 1024, m: 1, g: 1024, t: 1024 * 1024 }[String(match[2] || "m").toLowerCase() as "k" | "m" | "g" | "t"];
  return Number(match[1]) * scale;
};

const formatMemory = (megabytes: number | null): string => megabytes == null ? "—"
  : megabytes >= 1024 ? `${(megabytes / 1024).toFixed(megabytes >= 10240 ? 0 : 1)} GB`
    : `${Math.round(megabytes)} MB`;

const formatPercent = (value: unknown, scale = 1): string => {
  const number = finiteNumber(value);
  return number == null ? "—" : `${(number * scale).toFixed(0)}%`;
};

const accountingNote = (row: HistoryRow): string => {
  try {
    const notes = JSON.parse(row.jobstats_notes_json || "[]");
    return Array.isArray(notes) ? notes.filter((note) => typeof note === "string").join(" ") : "";
  } catch { return ""; }
};

const parseHistoryParams = (raw?: string): HistoryParamsMap => {
  try {
    const parsed = JSON.parse(raw || "{}");
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed;
  } catch { /* malformed legacy rows render as unavailable */ }
  return {};
};

// The pure-TNG generator fixes this at Config.TNG_GAL_DENSITY_ARCMIN2; it is
// shown here because it affects field work but is not a per-run form knob.
const DEFAULT_TNG_GALAXY_DENSITY = 60;
const param = (params: HistoryParamsMap, key: string): unknown => {
  const value = params[key];
  return value === "" || value == null ? null : value;
};
const paramText = (params: HistoryParamsMap, key: string, fallback = "—"): string => {
  const value = param(params, key);
  return value == null ? fallback : typeof value === "object" ? JSON.stringify(value) : String(value);
};
const paramCount = (params: HistoryParamsMap, key: string, fallback = "—"): string => {
  const value = param(params, key);
  const number = finiteNumber(value);
  return number == null ? paramText(params, key, fallback) : number.toLocaleString();
};
const paramSum = (params: HistoryParamsMap, keys: string[]): string => {
  let total = 0; let found = false;
  for (const key of keys) {
    const number = finiteNumber(param(params, key));
    if (number != null) { total += number; found = true; }
  }
  return found ? total.toLocaleString() : "—";
};
const regenerationSplits = (params: HistoryParamsMap): string[] =>
  paramText(params, "regenerate_splits", "").split(",")
    .map((value) => value.trim()).filter(Boolean);
const syntheticFieldCount = (params: HistoryParamsMap): string => {
  const selected = regenerationSplits(params);
  const keyBySplit: Record<string, string> = {
    train: "n_train", validate: "n_valid", test: "n_test",
  };
  return paramSum(params, selected.length
    ? selected.map((split) => keyBySplit[split]).filter(Boolean)
    : ["n_train", "n_valid", "n_test"]);
};
const syntheticSplitMode = (params: HistoryParamsMap): string => {
  const selected = regenerationSplits(params);
  return selected.length ? `rebuild ${selected.join("+")}` : "resume all";
};
const ensembleMemberCount = (params: HistoryParamsMap): number | null => {
  const explicit = finiteNumber(param(params, "count")) ?? finiteNumber(param(params, "n_members"));
  if (explicit != null) return explicit;
  const members = paramText(params, "members", "").split(",").map((value) => value.trim()).filter(Boolean);
  return members.length || null;
};
const ensembleTotalSteps = (params: HistoryParamsMap): string => {
  const members = ensembleMemberCount(params);
  const steps = finiteNumber(param(params, "steps")) ?? finiteNumber(param(params, "extra_steps"));
  return members != null && steps != null ? (members * steps).toLocaleString() : "—";
};
const taskColumn = (header: string, render: (params: HistoryParamsMap) => string, width?: number): Column<HistoryRow> => ({
  header, width,
  cell: (row) => <span className="mono fasrc-history__value">{render(parseHistoryParams(row.params_json))}</span>,
});

function taskColumnsFor(stepId: string): Column<HistoryRow>[] {
  switch (stepId) {
    case "synthetic_generate":
      return [
        taskColumn("splits", syntheticSplitMode, 132),
        taskColumn("nfields", syntheticFieldCount, 92),
        taskColumn("galaxies / arcmin²", (p) => paramText(p, "tng_density_arcmin2", String(DEFAULT_TNG_GALAXY_DENSITY)), 128),
        taskColumn("lenses / arcmin²", (p) => paramText(p, "lens_density_arcmin2"), 122),
      ];
    case "lensfinder_generate":
      return [
        taskColumn("nfields", (p) => paramSum(p, ["n_train", "n_valid", "n_test"]), 92),
        taskColumn("galaxies / arcmin²", (p) => paramText(p, "tng_density_arcmin2", String(DEFAULT_TNG_GALAXY_DENSITY)), 128),
        taskColumn("lenses / arcmin²", (p) => paramText(p, "lens_density_arcmin2"), 122),
      ];
    case "lens_isolation_generate":
      return [taskColumn("nfields", (p) => paramSum(p, ["ntrain", "nvalid", "ntest"]), 92)];
    case "lens_isolation_train":
      return [taskColumn("source members", (p) => paramText(p, "sources"), 150), taskColumn("steps", (p) => paramCount(p, "steps"), 76), taskColumn("batch", (p) => paramCount(p, "batch_size"), 70)];
    case "lens_isolation_evaluate":
      return [taskColumn("fields", (p) => paramCount(p, "limit"), 76), taskColumn("crop px", (p) => paramCount(p, "crop_size"), 82)];
    case "download_euclid_cutouts":
      return [taskColumn("VIS px", (p) => paramCount(p, "vis_pixels"), 76), taskColumn("workers", (p) => paramCount(p, "workers"), 82)];
    case "extract_euclid_psf":
      return [taskColumn("stars / PSF", (p) => paramCount(p, "stars_per_psf"), 92), taskColumn("max stars", (p) => paramCount(p, "num_stars"), 88), taskColumn("kernel px", (p) => paramCount(p, "output_size"), 88)];
    case "psf_rotation_pool":
      return [taskColumn("rotations", (p) => paramCount(p, "rotations"), 82), taskColumn("crop px", (p) => paramCount(p, "crop"), 78)];
    case "download_tng_skirt":
      return [taskColumn("galaxies", (p) => paramCount(p, "limit", "all"), 86), taskColumn("workers", (p) => paramCount(p, "workers"), 82)];
    case "tng_grid":
      return [taskColumn("band", (p) => paramText(p, "band"), 72), taskColumn("downsample", (p) => paramText(p, "downsample"), 96)];
    case "tng_stack":
      return [taskColumn("band", (p) => paramText(p, "band"), 72), taskColumn("galaxy", (p) => paramText(p, "galaxy_id", "selected"), 100)];
    case "poster_cutout":
      return [taskColumn("object", (p) => paramText(p, "mode"), 100), taskColumn("HR px", (p) => paramCount(p, "image_size"), 76)];
    case "euclid_query":
      return [taskColumn("stars", (p) => paramCount(p, "num_stars"), 82), taskColumn("mag range", (p) => `${paramText(p, "magnitude_min")}–${paramText(p, "magnitude_limit")}`, 110), taskColumn("min SNR", (p) => paramCount(p, "snr_min"), 82)];
    case "euclid_verify_photometry":
      return [taskColumn("stars", (p) => paramCount(p, "n"), 72), taskColumn("cutout px", (p) => paramCount(p, "size"), 92)];
    case "lensfinder_build_stamps":
      return [taskColumn("fields", (p) => paramCount(p, "max_fields", "all"), 82), taskColumn("neg / lens", (p) => paramCount(p, "neg_per_lens"), 92), taskColumn("stamp px", (p) => paramCount(p, "stamp_m"), 88)];
    case "lensfinder_sr_infer":
      return [taskColumn("subset", (p) => paramText(p, "subset", "all"), 82)];
    case "lensfinder_train":
      return [taskColumn("epochs", (p) => paramCount(p, "epochs"), 76), taskColumn("batch", (p) => paramCount(p, "batch_size"), 70), taskColumn("mode", (p) => paramText(p, "training_mode"), 104)];
    case "ensemble_train":
      return [
        taskColumn("total steps", ensembleTotalSteps, 104),
        taskColumn("batch", (p) => paramCount(p, "batch_size", "4"), 68),
        taskColumn("HR side", (p) => paramCount(p, "hr_crop_size", "256"), 76),
        taskColumn("examples / field", (p) => paramCount(p, "crops_per_field", "8"), 108),
      ];
    case "download":
      return [taskColumn("tiles", (p) => paramCount(p, "n_tiles"), 72)];
    case "extract_psf":
      return [taskColumn("stars", (p) => paramCount(p, "n_stars"), 72), taskColumn("PSF px", (p) => { const half = finiteNumber(param(p, "half_side")); return half == null ? "—" : (2 * half + 1).toLocaleString(); }, 82)];
    default:
      return [];
  }
}

function PreviousRuns({ stepId, refreshKey }: { stepId: string; refreshKey?: string | null }) {
  const history = useResource<HistoryResp>(`/api/fasrc/hst/${stepId}/history`, [refreshKey]);
  const [showAll, setShowAll] = useState(false);
  const rows = asArray<HistoryRow>(history.data?.history);
  const shown = showAll ? rows : rows.slice(0, 8);
  const taskColumns = taskColumnsFor(stepId);
  const gpuMemoryPercent = (row: HistoryRow): number | null =>
    finiteNumber(row.jobstats_gpu_memory_util)
    ?? finiteNumber(row.gpu_mem_util_peak)
    // ``gpu_mem_peak`` is a legacy live-sampler percentage.  New sacct
    // rows also carry the explicit ``gpu_mem_peak_mb`` field, so do not
    // accidentally render absolute MB as a percentage.
    ?? (finiteNumber(row.gpu_mem_peak_mb) == null ? finiteNumber(row.gpu_mem_peak) : null);
  const gpuUtilMean = (row: HistoryRow): number | null =>
    finiteNumber(row.jobstats_gpu_util) ?? finiteNumber(row.gpu_util_mean);
  const hasGpu = rows.some((row) =>
    (finiteNumber(row.req_gpus) ?? 0) > 0 || (finiteNumber(row.alloc_gpus) ?? 0) > 0 ||
    gpuUtilMean(row) != null || finiteNumber(row.gpu_util_peak) != null ||
    gpuMemoryPercent(row) != null || finiteNumber(row.jobstats_gpu_memory_used_mb) != null,
  );
  const columns: Column<HistoryRow>[] = [
    {
      header: "run", width: 122,
      cell: (row) => {
        const date = row.submitted_at ? new Date(row.submitted_at) : null;
        const note = accountingNote(row);
        return <div className="fasrc-history__stack mono" title={note || undefined}>
          <span className="fasrc-history__date">{date && !isNaN(date.getTime())
            ? date.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })
            : row.submitted_at || "—"}</span>
          {row.accounting_source && <span className="muted" style={{ fontSize: 10 }}>{row.accounting_source}{note ? " · note" : ""}</span>}
        </div>;
      },
    },
    {
      header: "state", width: 96,
      cell: (row) => <Badge tone={jobStateTone(row.state || "")}>{row.state || "—"}</Badge>,
    },
    { header: "elapsed (H:MM)", width: 96, cell: (row) => <span className="mono fasrc-history__value">{formatElapsed(row.elapsed_seconds)}</span> },
    ...taskColumns,
    {
      header: "CPU max used / requested (peak %) · mean %", width: 220,
      cell: (row) => {
        const requested = finiteNumber(row.req_cpus) ?? finiteNumber(row.alloc_cpus);
        const peak = finiteNumber(row.cpu_util_peak)
          ?? finiteNumber(row.jobstats_cpu_util)
          ?? (finiteNumber(row.cpu_efficiency) == null ? null : finiteNumber(row.cpu_efficiency)! * 100);
        const mean = finiteNumber(row.jobstats_cpu_util) ?? finiteNumber(row.cpu_util_mean);
        const used = requested != null && peak != null ? requested * peak / 100 : null;
        return <span className="mono fasrc-history__value">
          {used == null ? "—" : used.toFixed(1)} / {requested == null ? "—" : requested} ({peak == null ? "—" : `${peak.toFixed(0)}%`}) · {mean == null ? "—" : `${mean.toFixed(0)}%`}
        </span>;
      },
    },
    {
      header: "CPU memory max used / requested (%)", width: 210,
      cell: (row) => {
        const used = finiteNumber(row.jobstats_cpu_memory_used_mb)
          ?? finiteNumber(row.max_rss_mb);
        const requested = finiteNumber(row.jobstats_cpu_memory_alloc_mb)
          ?? memoryMegabytes(row.alloc_memory_mb)
          ?? memoryMegabytes(row.req_memory);
        const percent = used != null && requested != null && requested > 0 ? `${(100 * used / requested).toFixed(0)}%` : "—";
        return <span className="mono fasrc-history__value">{formatMemory(used)} / {formatMemory(requested)} ({percent})</span>;
      },
    },
    ...(hasGpu ? [{
      header: "GPU max used / requested (peak %) · mean % · memory %", width: 260,
      cell: (row) => {
        const requested = finiteNumber(row.req_gpus) ?? finiteNumber(row.alloc_gpus);
        const peak = finiteNumber(row.gpu_util_peak);
        const mean = gpuUtilMean(row);
        const memoryUsed = finiteNumber(row.jobstats_gpu_memory_used_mb);
        const memoryTotal = finiteNumber(row.jobstats_gpu_memory_total_mb);
        const memoryText = memoryUsed != null && memoryTotal != null
          ? `${formatMemory(memoryUsed)} / ${formatMemory(memoryTotal)} (${formatPercent(gpuMemoryPercent(row))})`
          : `memory ${formatPercent(gpuMemoryPercent(row))}`;
        const used = requested != null && peak != null ? requested * peak / 100 : null;
        return <span className="mono fasrc-history__value">
          {used == null ? "—" : used.toFixed(1)} / {requested == null ? "—" : requested} ({peak == null ? "—" : `${peak.toFixed(0)}%`}) · {mean == null ? "—" : `${mean.toFixed(0)}%`} · {memoryText}
        </span>;
      },
    } as Column<HistoryRow>] : []),
  ];
  return (
    <section className="fasrc-history">
      <div className="fasrc-history__head">
        <div>
          <div className="eyebrow">Previous runs</div>
          <div className="muted fasrc-history__note">Work-driving inputs, elapsed time, and observed resource use.</div>
        </div>
        <div className="row" style={{ gap: 8 }}>
          <Badge>{rows.length}</Badge>
          <Button size="sm" variant="ghost" onClick={() => history.reload()}>↻</Button>
        </div>
      </div>
      {history.loading && !rows.length
        ? <Empty><Spinner /> loading history…</Empty>
        : <Table columns={columns} rows={shown} rowKey={(row) => row.jobid}
            empty="no previous runs for this step" />}
      {rows.length > 8 && (
        <Button size="sm" variant="ghost" onClick={() => setShowAll((value) => !value)}>
          {showAll ? "show newest 8" : `show all ${rows.length}`}
        </Button>
      )}
    </section>
  );
}

/** Submit one FASRC pipeline step. Renders the step's resources (prefilled from
 *  server defaults), a confirm-guarded submit, then a live `<SlurmMonitor>`.
 *  `extraParams` carries any step-specific task params the page wants to pass.
 *  Embedded mode removes the surrounding card for pages that already provide
 *  the workflow card and only need a highlighted submission section. */
export function StepCard(
  { step, extraParams, sshConnected, embedded = false, showHistory = true,
    submitDisabled = false, submitDisabledHint }: {
    step: Step; extraParams?: Record<string, string | number>; sshConnected: boolean;
    embedded?: boolean; showHistory?: boolean; submitDisabled?: boolean;
    submitDisabledHint?: string;
  },
) {
  const d = step.defaults;
  const lockedCpus = step.fixed_cpus != null;
  const lockedGpus = step.fixed_gpus != null;
  const hasWorkerControl = step.step_id === "download_euclid_cutouts";
  const perModel = step.step_id === "ensemble_train" ? " / model" : "";
  const [partition, setPartition] = useState(d.partition);
  const [nCpus, setNCpus] = useState(String(step.fixed_cpus ?? d.n_cpus));
  const [nGpus, setNGpus] = useState(String(step.fixed_gpus ?? d.n_gpus));
  const [memory, setMemory] = useState(d.memory);
  const [timeLimit, setTimeLimit] = useState(d.time_limit);
  const [workers, setWorkers] = useState("8");
  const [jobid, setJobid] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    if (!window.confirm(`Submit “${step.label}” to SLURM?`)) return;
    setBusy(true); setError(null);
    try {
      const res = await postForm<{ ok?: boolean; jobid?: string; slurm_id?: string; error?: string }>(
        `/api/fasrc/hst/${step.step_id}/submit`,
        {
          partition, n_cpus: nCpus, n_gpus: nGpus, memory, time_limit: timeLimit,
          confirm: "yes", ...extraParams,
          ...(hasWorkerControl ? { workers } : {}),
        },
      );
      if (res.error) setError(res.error);
      else setJobid(res.jobid ?? res.slurm_id ?? null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally { setBusy(false); }
  }

  const content = (
    <>
      {embedded && (
        <div className="fasrc-step-inline__head">
          <div>
            <div className="eyebrow">SLURM submission</div>
            <strong>{step.label}</strong> <code className="mono">{step.step_id}</code>
          </div>
          {step.needs_gpu ? <Badge>GPU</Badge> : <Badge>CPU</Badge>}
        </div>
      )}
        {hasWorkerControl && (
          <div className="fasrc-step__res" style={{ marginBottom: "var(--s3)" }}>
            <NumberField
              label="parallel workers"
              value={workers}
              onChange={setWorkers}
              min={1}
              max={32}
              step={1}
              hint="per band; all four bands can run concurrently"
            />
          </div>
        )}
        <div className="fasrc-step__res">
          <Field label="partition"><Input value={partition} onChange={setPartition} /></Field>
          <NumberField label={`cpus${perModel}`} value={nCpus} onChange={setNCpus} min={1} disabled={lockedCpus} />
          {step.needs_gpu && <NumberField label={`gpus${perModel}`} value={nGpus} onChange={setNGpus} min={0} disabled={lockedGpus} />}
          <Field label={`memory${perModel}`}><Input value={memory} onChange={setMemory} /></Field>
          <Field label={`time limit${perModel}`}><Input value={timeLimit} onChange={setTimeLimit} /></Field>
        </div>
        <div className="row" style={{ marginTop: "var(--s3)" }}>
          <Button variant="primary" onClick={submit}
            disabled={busy || !sshConnected || submitDisabled}>
            {busy ? "submitting…" : "Submit to SLURM"}
          </Button>
          {!sshConnected && <span className="ui-field__hint">connect to FASRC first</span>}
          {sshConnected && submitDisabled && submitDisabledHint &&
            <span className="ui-field__hint">{submitDisabledHint}</span>}
        </div>
        {error && <div className="job-panel job-panel--err"><pre>{error}</pre></div>}
        {jobid && <SlurmMonitor jobid={jobid} />}
        {showHistory && <PreviousRuns stepId={step.step_id} refreshKey={jobid} />}
    </>
  );
  if (embedded) return <div className="fasrc-step-inline">{content}</div>;
  return (
    <Card className="fasrc-step">
      <CardHead
        title={step.label}
        sub={<code className="mono">{step.step_id}</code>}
        right={step.needs_gpu ? <Badge>GPU</Badge> : <Badge>CPU</Badge>}
      />
      <CardBody>{content}</CardBody>
    </Card>
  );
}

/** Look up a step by id in the registry and render its `<StepCard>` (or a
 *  loading/empty state). The one-liner pages call to embed a pipeline step. */
export function StepById(
  { stepId, extraParams, embedded = false, showHistory = true,
    submitDisabled = false, submitDisabledHint }: {
    stepId: string; extraParams?: Record<string, string | number>;
    embedded?: boolean; showHistory?: boolean; submitDisabled?: boolean;
    submitDisabledHint?: string;
  },
) {
  const { data, loading } = useHstStatus();
  if (loading) return embedded
    ? <div className="fasrc-step-inline"><Empty><Spinner /> loading step…</Empty></div>
    : <Card><CardBody><Empty><Spinner /> loading step…</Empty></CardBody></Card>;
  const step = asArray<Step>(data?.steps).find((s) => s.step_id === stepId);
  if (!step) {
    const empty = <Empty>step <code>{stepId}</code> is not registered on the server</Empty>;
    return embedded ? <div className="fasrc-step-inline">{empty}</div> : <Card><CardBody>{empty}</CardBody></Card>;
  }
  return <StepCard step={step} extraParams={extraParams} sshConnected={!!data?.ssh_connected}
    embedded={embedded} showHistory={showHistory} submitDisabled={submitDisabled}
    submitDisabledHint={submitDisabledHint} />;
}

/** FASRC SSH connection status + connect/disconnect. */
export function ConnectionBar() {
  const { data, reload } = useResource<{ ssh_connected: boolean; error?: string }>("/api/fasrc/status");
  const [busy, setBusy] = useState(false);
  const connected = !!data?.ssh_connected;
  async function toggle() {
    setBusy(true);
    try { await postForm(connected ? "/api/fasrc/disconnect" : "/api/fasrc/connect"); }
    catch { /* surfaced via reload */ }
    finally { setBusy(false); reload(); }
  }
  return (
    <div className="row" style={{ gap: "var(--s3)" }}>
      <ConnBadge ok={connected} labels={["FASRC connected", "FASRC offline"]} />
      <Button size="sm" onClick={toggle} disabled={busy}>
        {busy ? "…" : connected ? "disconnect" : "connect"}
      </Button>
      {data?.error && !connected && <span className="ui-field__hint">{data.error}</span>}
    </div>
  );
}
