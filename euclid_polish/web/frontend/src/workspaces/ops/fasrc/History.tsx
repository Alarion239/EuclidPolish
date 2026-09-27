/* Ops › FASRC › History: every run of every step (the local job ledger,
 * works offline), filterable by step / state / text, with "reconcile
 * unresolved states" (a local job re-pulling sacct for UNKNOWN / DONE /
 * blank rows and stale RUNNING / PENDING snapshots the DB has finalised)
 * and clone / logs / inspect per run. */
import { useMemo } from "react";
import { Link } from "react-router-dom";
import { useJob } from "../../../api/jobs";
import { invalidate, useResource } from "../../../api/query";
import { usePageActions } from "../../../app/palette";
import { formatDateTime, formatDuration } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Badge, Button, Callout, DataTable, IconButton, JobProgress, Menu, Select, Tooltip, confirm, toast, type DataColumn,
} from "../../../ui";
import { historyUrl, type HistoryResp } from "../api";
import {
  cpuUsage, finiteNumber, formatMemory, gpuUsage, hasGpu, jobStateTone, memoryUsage, paramsSummary, rowParams, rowState,
  type HistoryRow,
} from "../model";

const pct = (v: number | null | undefined) => (v == null ? "—" : `${v.toFixed(0)}%`);

const rowSummary = (row: HistoryRow): string => paramsSummary(rowParams(row));

export function HistoryPanel({ fasrcConnected }: { fasrcConnected: boolean }) {
  const [step, setStep] = useUrlState("hstep", "");
  const [state, setState] = useUrlState("hstate", "");
  const res = useResource<HistoryResp>(historyUrl({ step, state }), [step, state], { ttl: 20_000 });
  const reconcile = useJob("fasrc:accounting");
  const d = res.data;

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
    { id: "fasrc-reconcile", label: "Reconcile unresolved SLURM job states", group: "FASRC", keywords: ["sacct", "unknown"],
      disabled: !fasrcConnected || reconcile.busy, run: () => void runReconcile("unresolved") },
  ]);

  const stepOptions = useMemo(() => [{ value: "", label: "All steps" },
    ...Object.entries(d?.facets.steps ?? {}).sort((a, b) => b[1] - a[1]).map(([s, n]) => ({ value: s, label: `${s} (${n})` }))],
  [d?.facets.steps]);
  const stateOptions = useMemo(() => [{ value: "", label: "All states" },
    { value: "unresolved", label: `Unresolved (${d?.unresolved ?? 0})` },
    ...Object.entries(d?.facets.states ?? {}).sort((a, b) => b[1] - a[1]).map(([s, n]) => ({ value: s, label: `${s} (${n})` }))],
  [d?.facets.states, d?.unresolved]);
  const rows = useMemo(() => d?.rows ?? [], [d]);
  const gpu = useMemo(() => rows.some(hasGpu), [rows]);

  const columns = useMemo<DataColumn<HistoryRow>[]>(() => [
    { id: "submitted_at", header: "Submitted", width: 132,
      cell: (r) => <span className="mono ops-small">{formatDateTime(r.submitted_at)}</span> },
    { id: "jobid", header: "Job", width: 92, cell: (r) => <code className="mono">{r.jobid}</code> },
    { id: "step_id", header: "Step", width: 150, cell: (r) => <code className="mono ops-small">{r.step_id || "—"}</code> },
    { id: "state", header: "State", width: 116, accessor: (r) => rowState(r),
      cell: (r) => {
        const s = rowState(r);
        const tip = r.state ? `sacct: ${r.state}${r.db_state ? ` · DB: ${r.db_state}` : ""}` : `no sacct verdict yet · DB: ${r.db_state ?? "—"}`;
        return <Tooltip content={tip}><span tabIndex={0}><Badge size="sm" tone={jobStateTone(s)}>{s}</Badge></span></Tooltip>;
      } },
    { id: "elapsed", header: "Elapsed", numeric: true, width: 84, accessor: (r) => finiteNumber(r.elapsed_seconds),
      cell: (r) => formatDuration(finiteNumber(r.elapsed_seconds)) },
    { id: "cpu", header: "CPU used", numeric: true, width: 92, accessor: (r) => cpuUsage(r).pct,
      cell: (r) => { const u = cpuUsage(r); return <span className="mono ops-small" title={`peak ${pct(u.pct)} · mean ${pct(u.mean)}`}>{u.used == null ? "—" : u.used.toFixed(1)} / {u.requested ?? "—"}</span>; } },
    { id: "mem", header: "Memory", numeric: true, width: 120, accessor: (r) => memoryUsage(r).pct,
      cell: (r) => { const u = memoryUsage(r); return <span className="mono ops-small">{formatMemory(u.used)} / {formatMemory(u.requested)}</span>; } },
    ...(gpu ? [{ id: "gpu", header: "GPU", numeric: true, width: 110, accessor: (r: HistoryRow) => gpuUsage(r).mean,
      cell: (r: HistoryRow) => { const g = gpuUsage(r); return <span className="mono ops-small">{pct(g.mean)} · mem {pct(g.memPct)}</span>; } }] : []),
    { id: "params", header: "Params", accessor: rowSummary, hidden: false,
      cell: (r) => <span className="mono ops-small ops-ellipsis" title={rowSummary(r)}>{rowSummary(r) || "—"}</span> },
    { id: "label", header: "Label", hidden: true },
    { id: "actions", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 76,
      cell: (r) => (
        <span className="ops-row-actions">
          {r.step_id && (
            <Tooltip content="Clone this run into the step form">
              <Link className="ops-iconlink" aria-label={`Clone ${r.jobid}`}
                to={`/ops/fasrc?view=steps&step=${encodeURIComponent(String(r.step_id))}&clone=${encodeURIComponent(r.jobid)}`}>⧉</Link>
            </Tooltip>
          )}
          <Tooltip content="Logs"><Link className="ops-iconlink" aria-label={`Logs of ${r.jobid}`}
            to={`/ops/fasrc?view=logs&job=${encodeURIComponent(r.jobid)}`}>≡</Link></Tooltip>
        </span>
      ) },
  ], [gpu]);

  return (
    <div className="ops-stack">
      {res.error && !d && <Callout tone="bad" title="Could not read the run history">{res.error.message}</Callout>}
      {(reconcile.job || reconcile.error) && <JobProgress job={reconcile.job} error={reconcile.error} />}
      <DataTable rows={rows} columns={columns} rowKey={(r) => String(r.jobid)} aria-label="FASRC run history"
        loading={res.loading && !d} urlKey="h" exportName="fasrc-history" height={620}
        inspect={(r) => ({ kind: "job", id: `slurm/${r.jobid}` })}
        empty="No runs recorded yet." filterPlaceholder="Filter: text, state:FAILED, jobid=…"
        toolbar={<>
          <Select size="sm" value={step} onChange={setStep} options={stepOptions} aria-label="Step" />
          <Select size="sm" value={state} onChange={setState} options={stateOptions} aria-label="State" />
          <Tooltip content={fasrcConnected ? "Re-pull sacct for the jobs without a final verdict (blank, UNKNOWN, DONE, or a stale RUNNING / PENDING)" : "Needs FASRC"}>
            <span>
              <Button size="sm" icon="reset" loading={reconcile.busy} disabled={!fasrcConnected || !d?.unresolved}
                onClick={() => void runReconcile("unresolved")}>
                Reconcile{d?.unresolved ? ` ${d.unresolved}` : ""}
              </Button>
            </span>
          </Tooltip>
          <Menu label="More history actions" trigger={<IconButton size="sm" icon="more" label="More" />} items={[
            { label: "Re-pull accounting for every job", disabled: !fasrcConnected || reconcile.busy, onSelect: () => void runReconcile("all") },
            { label: "Reload", onSelect: () => res.reload() },
          ]} />
          {d && <Badge size="sm">{d.total} runs</Badge>}
        </>} />
    </div>
  );
}
