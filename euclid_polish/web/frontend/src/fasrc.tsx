/* FASRC pipeline: typed React components for the SLURM step registry + live
   job monitoring. `<StepCard>` submits one pipeline step with editable
   resources; `<SlurmMonitor>` folds the event stream into stage/progress/
   warnings/errors. Shared by Catalog, PSFs, TNG, Lens-finder and the FASRC
   page — the DRY replacement for the classic `fasrc_step_card.js`. */
import { useMemo, useState, type ReactNode } from "react";
import { getJSON, postForm } from "./api";
import { useResource, usePolling } from "./hooks";
import Plot, { Legend } from "./charts/Plot";
import {
  Badge, Button, Card, CardBody, CardHead, ConnBadge, DefList, Empty, LogTail,
  NumberField, Field, Input, ProgressBar, Segmented, Spinner,
} from "./ui";

export type StepDefaults = {
  partition: string; n_cpus: number; n_gpus: number; memory: string; time_limit: string;
};
export type Step = {
  step_id: string; label: string; needs_gpu: boolean;
  fixed_cpus?: number | null; defaults: StepDefaults;
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
};
type JobStatusResp = {
  ok: boolean; jobid: string; state: string; status?: SlurmStatus; error?: string;
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
      {!!st.warnings?.length && (
        <ul className="fasrc-mon__events fasrc-mon__events--warn">
          {st.warnings.slice(-4).map((e, i) => <li key={i}>{e.msg}</li>)}
        </ul>
      )}
      {!!st.errors?.length && (
        <ul className="fasrc-mon__events fasrc-mon__events--err">
          {st.errors.slice(-4).map((e, i) => <li key={i}>{e.msg}</li>)}
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
    </div>
  );
}

/* ── Current Submission ─────────────────────────────────────────────────── */
type CSJob = {
  jobid?: string; label?: string; state?: string; nodes?: string;
  time?: string; time_limit?: string; start_time?: string; reason?: string;
  started_at?: number; ended_at?: number;
};
type CSQueue = {
  count: number; halted: boolean; halted_reason?: string | null;
  names: string[]; active_jobid?: string | null;
};
type CurrentSubmissionResp = {
  ok: boolean; stale?: boolean;
  current: { job: CSJob; status?: SlurmStatus | null } | null;
  queue: CSQueue;
};

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
  return (
    <div className="grid" style={{ gap: "var(--s4)" }}>
      {q && (q.count > 0 || q.halted) && (
        <Card>
          <CardHead title="Submission queue" sub={`${q.count} queued`}
            right={q.count > 0 && <Button size="sm" variant="ghost" disabled={busy}
              onClick={() => control("/api/fasrc/queue/clear", {})}>clear</Button>} />
          <CardBody>
            {q.halted && <div className="job-panel job-panel--err"><LogTail text={q.halted_reason || "queue halted"} /></div>}
            {q.names?.length
              ? <ul className="fasrc-queue">{q.names.map((n, i) => <li key={i}>{n}</li>)}</ul>
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
            <div style={{ marginTop: "var(--s3)" }}><JobStatusBody status={cur.status} /></div>
          </CardBody>
        </Card>
      )}
      {cur?.job.started_at && <TrainingCurve startedAt={cur.job.started_at} endedAt={cur.job.ended_at} />}
    </div>
  );
}

/* ── Training curves (browser-rendered) ─────────────────────────────────────
   The per-step validation records for a run's wall-time window, drawn live in
   the browser with the shared <Plot> — no server matplotlib. Polls while the
   card is up so an in-flight run's curve grows. */
type CurveRec = {
  step: number; psnr_stretched?: number | null; psnr_raw?: number | null;
  loss?: number | null; psnr_vis?: number | null; psnr_y_e?: number | null;
  psnr_j_e?: number | null; psnr_h_e?: number | null;
};
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

export function TrainingCurve({ startedAt, endedAt }: { startedAt?: number; endedAt?: number }) {
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
    const url = `/api/fasrc/runs/training-curve.json?started_at=${startedAt}${endedAt ? `&ended_at=${endedAt}` : ""}`;
    fetch(url, { headers: { Accept: "application/json" } })
      .then(async (r) => {
        const j = (await r.json().catch(() => null)) as (CurveResp & { error?: string }) | null;
        setLoading(false);
        if (!r.ok || !j || j.ok === false) { setErr(j?.error || `HTTP ${r.status}`); return; }
        setErr(""); setResp(j);
      })
      .catch(() => { setLoading(false); setErr("request failed"); });
  }, 20000, !!startedAt);

  const recs = resp?.records ?? [];
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
      <CardHead title="Training curves" sub={resp?.member || (resp ? "this run" : undefined)}
        right={<Segmented<CurveMetric> value={metric} onChange={setMetric}
          options={[{ value: "psnr", label: "PSNR" }, { value: "loss", label: "loss" }]} />} />
      <CardBody>
        {loading ? <Empty><Spinner /> loading curves…</Empty>
          : !resp ? <Empty>curves unavailable{err ? ` — ${err}` : ""} · retrying every 20 s</Empty>
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

/** Submit one FASRC pipeline step. Renders the step's resources (prefilled from
 *  server defaults), a confirm-guarded submit, then a live `<SlurmMonitor>`.
 *  `extraParams` carries any step-specific task params the page wants to pass. */
export function StepCard(
  { step, extraParams, sshConnected }: {
    step: Step; extraParams?: Record<string, string | number>; sshConnected: boolean;
  },
) {
  const d = step.defaults;
  const lockedCpus = step.fixed_cpus != null;
  const [partition, setPartition] = useState(d.partition);
  const [nCpus, setNCpus] = useState(String(step.fixed_cpus ?? d.n_cpus));
  const [nGpus, setNGpus] = useState(String(d.n_gpus));
  const [memory, setMemory] = useState(d.memory);
  const [timeLimit, setTimeLimit] = useState(d.time_limit);
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
        },
      );
      if (res.error) setError(res.error);
      else setJobid(res.jobid ?? res.slurm_id ?? null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally { setBusy(false); }
  }

  return (
    <Card className="fasrc-step">
      <CardHead
        title={step.label}
        sub={<code className="mono">{step.step_id}</code>}
        right={step.needs_gpu ? <Badge>GPU</Badge> : <Badge>CPU</Badge>}
      />
      <CardBody>
        <div className="fasrc-step__res">
          <Field label="partition"><Input value={partition} onChange={setPartition} /></Field>
          <NumberField label="cpus" value={nCpus} onChange={setNCpus} min={1} disabled={lockedCpus} />
          {step.needs_gpu && <NumberField label="gpus" value={nGpus} onChange={setNGpus} min={0} />}
          <Field label="memory"><Input value={memory} onChange={setMemory} /></Field>
          <Field label="time limit"><Input value={timeLimit} onChange={setTimeLimit} /></Field>
        </div>
        <div className="row" style={{ marginTop: "var(--s3)" }}>
          <Button variant="primary" onClick={submit} disabled={busy || !sshConnected}>
            {busy ? "submitting…" : "Submit to SLURM"}
          </Button>
          {!sshConnected && <span className="ui-field__hint">connect to FASRC first</span>}
        </div>
        {error && <div className="job-panel job-panel--err"><pre>{error}</pre></div>}
        {jobid && <SlurmMonitor jobid={jobid} />}
      </CardBody>
    </Card>
  );
}

/** Look up a step by id in the registry and render its `<StepCard>` (or a
 *  loading/empty state). The one-liner pages call to embed a pipeline step. */
export function StepById(
  { stepId, extraParams }: { stepId: string; extraParams?: Record<string, string | number> },
) {
  const { data, loading } = useHstStatus();
  if (loading) return <Card><CardBody><Empty><Spinner /> loading step…</Empty></CardBody></Card>;
  const step = data?.steps.find((s) => s.step_id === stepId);
  if (!step) {
    return <Card><CardBody><Empty>step <code>{stepId}</code> is not registered on the server</Empty></CardBody></Card>;
  }
  return <StepCard step={step} extraParams={extraParams} sshConnected={!!data?.ssh_connected} />;
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
