/* Runs › Steps (`/runs/steps`): the FASRC step catalogue as a list that is
 * always visible, grouped by pipeline stage (Reference data, Noise and
 * fields, Generation, Training, Figures), and the selected step's card
 * (params, resources, a confirmed Queue / Submit, its previous runs). Each
 * step names its home: the tab whose drawer embeds it (ensemble_train →
 * Models › Train). URL: `step`, `clone=<jobid>` (fill the form from that
 * run; History's Clone links here), `stage` (the group to open on), `sq`
 * (the list filter). */
import { useEffect, useMemo, useRef } from "react";
import { Link } from "react-router-dom";
import { useResource } from "../../../api/query";
import { usePageActions } from "../../../app/palette";
import { useUrlState } from "../../../hooks/useUrlState";
import { Badge, Callout, Icon, Input, Page, Skeleton } from "../../../ui";
import { ConnectionBar } from "../Connection";
import { historyUrl, type HistoryResp } from "../api";
import { revealBelowFold, rowParams, stageOfStep, stepHome, stepsByStage } from "../model";
import { StepCard, useStepsStatus } from "../steps/StepCard";
import "../runs.css";

export default function Steps() {
  const steps = useStepsStatus();
  const [stepId, setStepId] = useUrlState("step", "");
  const [cloneId, setCloneId] = useUrlState("clone", "");
  const [stage, setStage] = useUrlState("stage", "");
  const [q, setQ] = useUrlState("sq", "");
  const all = useMemo(() => steps.data?.steps ?? [], [steps.data]);
  const groups = useMemo(() => stepsByStage(all), [all]);
  // `?stage=` opens on the first step of that stage (the Noise and fields
  // inputs link lands here); an explicit `?step=` wins.
  const stageFirst = stage ? groups.find((g) => g.id === stage)?.steps[0] : undefined;
  const current = all.find((s) => s.step_id === stepId) ?? stageFirst ?? groups[0]?.steps[0];
  const unknownStep = !!stepId && all.length > 0 && !all.some((s) => s.step_id === stepId);
  const clone = useResource<HistoryResp>(cloneId ? historyUrl({ q: cloneId, limit: 20 }) : null, [cloneId], { ttl: 60_000 });
  const cloneRow = clone.data?.rows.find((r) => String(r.jobid) === cloneId) ?? null;
  const groupRefs = useRef(new Map<string, HTMLElement>());
  const focusStage = stage || (current ? stageOfStep(current.step_id) : "");

  useEffect(() => {
    if (!stage || !groups.length) return;
    const el = groupRefs.current.get(stage);
    el?.scrollIntoView?.({ block: "nearest" });
  }, [stage, groups.length]);

  const cardRef = useRef<HTMLDivElement>(null);
  const picked = useRef(false);
  const pick = (id: string) => { picked.current = true; setCloneId(""); setStage(""); setStepId(id); };
  // A narrow page stacks the card under the stage list: show the picked step.
  useEffect(() => {
    if (picked.current) requestAnimationFrame(() => revealBelowFold(cardRef.current));
    picked.current = false;
  }, [stepId]);
  usePageActions(all.map((s) => ({
    id: `submit-${s.step_id}`, label: `Submit ${s.label}`, group: "FASRC steps", keywords: [s.step_id, "slurm", "run"],
    run: () => pick(s.step_id),
  })));

  const needle = q.trim().toLowerCase();
  const shown = needle
    ? groups.map((g) => ({ ...g, steps: g.steps.filter((s) => `${s.label} ${s.step_id}`.toLowerCase().includes(needle)) }))
      .filter((g) => g.steps.length)
    : groups;
  const home = current ? stepHome(current.step_id) : null;
  const initial = cloneRow && current && String(cloneRow.step_id) === current.step_id
    ? { params: rowParams(cloneRow), resources: cloneRow, jobid: cloneId } : null;
  return (
    <Page className="runs-page">
      {steps.loading && !steps.data ? <Skeleton lines={6} />
        : steps.error && !steps.data ? <Callout tone="bad" title="Could not load the FASRC steps">{steps.error.message}</Callout>
        : (
          <div className="runs-steps">
            <nav className="runs-steplist" aria-label="FASRC steps by stage">
              <Input size="sm" value={q} onChange={setQ} icon="search" clearable placeholder="Filter steps…" aria-label="Filter steps" />
              <div className="runs-steplist__groups">
                {shown.map((g) => (
                  <section key={g.id} className="runs-steplist__group" data-focus={g.id === focusStage || undefined}
                    aria-label={g.label} ref={(el) => { if (el) groupRefs.current.set(g.id, el); else groupRefs.current.delete(g.id); }}>
                    <h3 className="runs-steplist__stage">{g.label}</h3>
                    <ul>
                      {g.steps.map((s) => (
                        <li key={s.step_id}>
                          <button type="button" className="runs-steplist__item" onClick={() => pick(s.step_id)}
                            aria-current={s.step_id === current?.step_id ? "true" : undefined}>
                            <span className="runs-steplist__label">{s.label}</span>
                            <span className="runs-steplist__meta">
                              <code className="mono">{s.step_id}</code>
                              <Badge size="sm">{s.needs_gpu ? "GPU" : "CPU"}</Badge>
                              {(s.outputs ?? []).some((o) => o.exists === false) && <Badge size="sm" tone="warn">output missing</Badge>}
                            </span>
                          </button>
                        </li>
                      ))}
                    </ul>
                  </section>
                ))}
                {!shown.length && <p className="runs-note">No step matches.</p>}
              </div>
            </nav>
            <div className="runs-stack">
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
              {current && (
                <div ref={cardRef}>
                  <div className="runs-stephome">
                    {home ? <>
                      <span>Home tab</span>
                      <Link to={home.to}>{home.label} <Icon name="chevronRight" size={12} /></Link>
                    </> : <span>No console tab embeds this step yet.</span>}
                    <span className="runs-spacer" />
                    <ConnectionBar />
                  </div>
                  {cloneId && clone.loading ? <Skeleton lines={5} /> : (
                    <StepCard key={`${current.step_id}:${initial ? cloneId : ""}`} step={current}
                      sshConnected={!!steps.data?.ssh_connected} initial={initial} />
                  )}
                </div>
              )}
            </div>
          </div>
        )}
    </Page>
  );
}
