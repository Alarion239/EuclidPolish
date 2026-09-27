/* Ops › FASRC › Steps: every registered pipeline step, one of them open in
 * the schema-driven card. `?step=` picks it, `?clone=<jobid>` fills the form
 * from that run (the History and step-history "clone" actions link here). */
import { useMemo } from "react";
import { useResource } from "../../../api/query";
import { usePageActions } from "../../../app/palette";
import { useUrlState } from "../../../hooks/useUrlState";
import { Badge, Callout, Input, Select, Skeleton } from "../../../ui";
import { historyUrl, type HistoryResp } from "../api";
import { rowParams } from "../model";
import { StepCard, useStepsStatus } from "../steps/StepCard";

export function StepsPanel() {
  const steps = useStepsStatus();
  const [stepId, setStepId] = useUrlState("step", "");
  const [cloneId, setCloneId] = useUrlState("clone", "");
  const [q, setQ] = useUrlState("sq", "");
  const all = useMemo(() => steps.data?.steps ?? [], [steps.data]);
  const current = all.find((s) => s.step_id === stepId) ?? all[0];
  // A deep link (`/ops/fasrc?view=steps&step=<id>` from Settings or the
  // palette) naming a step this backend does not register.
  const unknownStep = !!stepId && all.length > 0 && !all.some((s) => s.step_id === stepId);
  const clone = useResource<HistoryResp>(cloneId ? historyUrl({ q: cloneId, limit: 20 }) : null, [cloneId], { ttl: 60_000 });
  const cloneRow = clone.data?.rows.find((r) => String(r.jobid) === cloneId) ?? null;

  usePageActions(all.map((s) => ({
    id: `submit-${s.step_id}`, label: `Submit ${s.label}`, group: "FASRC steps", keywords: [s.step_id, "slurm", "run"],
    run: () => { setCloneId(""); setStepId(s.step_id); },
  })));

  const needle = q.trim().toLowerCase();
  const shown = needle ? all.filter((s) => `${s.label} ${s.step_id}`.toLowerCase().includes(needle)) : all;
  if (steps.loading && !steps.data) return <Skeleton lines={6} />;
  if (steps.error && !steps.data) return <Callout tone="bad" title="Could not load the FASRC steps">{steps.error.message}</Callout>;
  const initial = cloneRow && current && String(cloneRow.step_id) === current.step_id
    ? { params: rowParams(cloneRow), resources: cloneRow, jobid: cloneId } : null;
  return (
    <div className="ops-split ops-split--steps">
      <div className="ops-steppick">
        <Select searchable size="sm" aria-label="Step" value={current?.step_id ?? ""}
          onChange={(v) => { setCloneId(""); setStepId(v); }}
          options={all.map((s) => ({ value: s.step_id, label: s.label, hint: `${s.step_id} · ${s.needs_gpu ? "GPU" : "CPU"}` }))} />
      </div>
      <nav className="ops-steplist" aria-label="FASRC steps">
        <Input size="sm" value={q} onChange={setQ} icon="search" clearable placeholder="Filter steps…" aria-label="Filter steps" />
        <ul>
          {shown.map((s) => (
            <li key={s.step_id}>
              <button type="button" className="ops-steplist__item" aria-current={s.step_id === current?.step_id ? "true" : undefined}
                onClick={() => { setCloneId(""); setStepId(s.step_id); }}>
                <span className="ops-steplist__label">{s.label}</span>
                <span className="ops-steplist__meta">
                  <code className="mono">{s.step_id}</code>
                  <Badge size="sm">{s.needs_gpu ? "GPU" : "CPU"}</Badge>
                  {(s.outputs ?? []).some((o) => o.exists === false) && <Badge size="sm" tone="warn">output missing</Badge>}
                </span>
              </button>
            </li>
          ))}
          {!shown.length && <li className="ops-dim ops-small">No step matches.</li>}
        </ul>
      </nav>
      <div className="ops-split__main">
        {unknownStep && (
          <Callout tone="warn" title={`No step “${stepId}”`} onDismiss={() => setStepId("")}>
            This backend does not register it; showing {current?.label ?? "the first step"} instead.
          </Callout>
        )}
        {cloneId && clone.data && !cloneRow && (
          <Callout tone="warn" title={`Run ${cloneId} not found`} onDismiss={() => setCloneId("")}>It is not in the local job ledger.</Callout>
        )}
        {cloneRow && current && String(cloneRow.step_id) !== current.step_id && (
          <Callout tone="warn" title="Clone of another step" onDismiss={() => setCloneId("")}>
            Run {cloneId} belongs to <code>{String(cloneRow.step_id)}</code>.
          </Callout>
        )}
        {current && (cloneId && clone.loading ? <Skeleton lines={5} /> : (
          <StepCard key={`${current.step_id}:${initial ? cloneId : ""}`} step={current}
            sshConnected={!!steps.data?.ssh_connected} initial={initial} />
        ))}
      </div>
    </div>
  );
}
