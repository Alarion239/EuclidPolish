/* The schema-driven FASRC step card (spec §8.7, contract C5), shared by every
 * page that embeds a pipeline step (re-exported from src/fasrc.tsx):
 *
 *   <StepById stepId="euclid_query" />                      // looks the step up
 *   <StepCard step={step} sshConnected extraParams={…} />   // host-controlled params hidden
 *
 * It renders the step's `task_params` generically (type, range, choices, help
 * popover), prefilled from the last successful run (`last_params`) or from a
 * cloned past run, the SLURM resources, a confirmed submit that reports a
 * queued submission, and re-attaches to the step's live job (from the shared
 * jobs feed) after navigating away. Under the resources, the resource
 * advisor's "Recommended from N past runs" callout (shared/ResourceAdvice:
 * asked with exactly the task params the submit posts; Apply sets the
 * editable resource fields). Output artifacts and the run history (clone a
 * run) sit under the form. Props are backward compatible with the pre-rework
 * StepCard. */
import { useMemo, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { ApiError, apiPost, isFasrcOffline } from "../../../api/client";
import { refreshJobsFeed, useJobsFeed } from "../../../api/jobs";
import { invalidate, useResource } from "../../../api/query";
import { formatDateTime, formatDuration } from "../../../format";
import {
  Badge, Button, Callout, Card, CardBody, CardHead, DataTable, Field, IconButton, Input, NumberField, Section,
  Segmented, Select, Skeleton, Switch, Textarea, Tooltip, confirm, toast, type DataColumn,
} from "../../../ui";
import {
  STEPS_STATUS_URL, stepHistoryUrl, type StepHistoryResp, type StepsStatus, type SubmitResp,
} from "../api";
import {
  cpuUsage, finiteNumber, formatMemory, gpuUsage, hasGpu, isLiveState, jobStateTone, memoryUsage, paramText,
  parentDir, rowParams, rowState, varyingParams, type HistoryRow,
} from "../model";
import {
  applyAdvisedResources, changedParams, dangerReasons, defaultResources, defaultValues, editableResources, humanName,
  initialValues, isChanged, paramFacts, resourcesFromRow, submitBody, submitParams, validateResources, validateValues,
  visibleParams, type FormValues, type Resources, type Step, type TaskParam,
} from "./stepForm";
import { ResourceAdvice } from "../../shared/ResourceAdvice";
import { SlurmMonitor } from "./SlurmMonitor";
import "./steps.css";

export type { Step, StepDefaults, TaskParam } from "./stepForm";
export type { StepsStatus } from "../api";

/** The shared step registry (`/api/fasrc/steps/status`, one cache entry). */
export function useStepsStatus() {
  return useResource<StepsStatus>(STEPS_STATUS_URL, [], { ttl: 60_000 });
}

export type StepCardProps = {
  step: Step;
  /** Task params the host page controls: posted as given and hidden from the form. */
  extraParams?: Record<string, string | number>;
  sshConnected: boolean;
  /** No surrounding card (the host page already provides one). */
  embedded?: boolean;
  showHistory?: boolean;
  submitDisabled?: boolean;
  submitDisabledHint?: string;
  /** More params to hide (besides the `extraParams` keys). */
  hideParams?: string[];
  /** Start from these values (e.g. a clone from the URL); `jobid` labels it. */
  initial?: { params: Record<string, unknown>; resources?: Record<string, unknown>; jobid?: string } | null;
  /** Called after a successful submit (queued or not). */
  onSubmitted?: (result: SubmitResp) => void;
};

type Source = { kind: "defaults" | "last" | "clone"; jobid?: string };
type Outcome = { kind: "submitted"; jobid: string } | { kind: "queued"; position: number; count: number; label?: string };

const RESOURCE_LABEL: Record<keyof Resources, string> = {
  n_cpus: "CPUs", n_gpus: "GPUs", memory: "Memory", time_limit: "Time limit",
};

/* ── one task-param control ───────────────────────────────────────────────── */

function ParamField({ param, value, error, onChange, disabled }: {
  param: TaskParam; value: string; error?: string; onChange: (v: string) => void; disabled?: boolean;
}) {
  const changed = isChanged(param, value);
  const hint = (
    <span className="ops-param__hint">
      {param.help && <span>{param.help}</span>}
      <span className="ops-dim mono">{param.name} · {param.type} · {paramFacts(param)}</span>
    </span>
  );
  let control: ReactNode;
  switch (param.type) {
    case "bool":
      control = <Switch checked={value === "1" || value.toLowerCase() === "true"} disabled={disabled}
        onChange={(on) => onChange(on ? "1" : "0")} aria-label={humanName(param.name)} />;
      break;
    case "choice": {
      const choices = param.choices ?? [];
      control = choices.length <= 3 && choices.every((c) => c.length <= 10)
        ? <Segmented size="sm" value={value} onChange={onChange} disabled={disabled} aria-label={humanName(param.name)}
            options={choices.map((c) => ({ value: c, label: c }))} />
        : <Select size="sm" value={value} onChange={onChange} disabled={disabled}
            options={choices.map((c) => ({ value: c, label: c }))} />;
      break;
    }
    case "int":
    case "float":
      control = <NumberField size="sm" value={value} onChange={onChange} disabled={disabled}
        min={param.min} max={param.max} step={param.type === "int" ? 1 : "any"}
        placeholder={param.default == null ? "unset" : undefined} aria-label={humanName(param.name)} />;
      break;
    case "json":
      control = <Textarea value={value} onChange={onChange} rows={2} disabled={disabled} spellCheck={false}
        className="mono" placeholder={param.default == null ? "unset" : undefined} />;
      break;
    default:
      control = <Input size="sm" value={value} onChange={onChange} disabled={disabled} spellCheck={false}
        autoComplete="off" placeholder={param.default == null ? "unset (default)" : undefined} />;
  }
  return (
    <div className="ops-param" data-changed={changed || undefined} data-type={param.type}>
      <Field label={humanName(param.name)} hint={hint} error={error}>{control}</Field>
    </div>
  );
}

/* ── run history of one step ──────────────────────────────────────────────── */

const pct = (v: number | null | undefined) => (v == null ? "—" : `${v.toFixed(0)}%`);

export function StepHistory({ step, refreshKey, onClone }: {
  step: Step; refreshKey?: string | number | null; onClone?: (row: HistoryRow) => void;
}) {
  const history = useResource<StepHistoryResp>(stepHistoryUrl(step.step_id), [refreshKey], { ttl: 30_000 });
  const rows = useMemo(() => history.data?.history ?? [], [history.data]);
  const names = useMemo(() => (step.task_params ?? []).map((p) => p.name), [step.task_params]);
  const varying = useMemo(() => varyingParams(rows, names, 4), [rows, names]);
  // A CPU step has no GPU column (a GPU someone requested by hand stays in the ledger).
  const gpu = useMemo(() => step.needs_gpu && rows.some(hasGpu), [rows, step.needs_gpu]);
  const columns = useMemo<DataColumn<HistoryRow>[]>(() => [
    { id: "submitted_at", header: "Submitted", width: 128,
      cell: (r) => <span className="mono ops-small">{formatDateTime(r.submitted_at)}</span> },
    { id: "jobid", header: "Job", width: 92, cell: (r) => <code className="mono">{r.jobid}</code> },
    { id: "state", header: "State", width: 104, accessor: (r) => rowState(r),
      cell: (r) => <Badge size="sm" tone={jobStateTone(rowState(r))}>{rowState(r)}</Badge> },
    { id: "elapsed", header: "Elapsed", numeric: true, width: 84, accessor: (r) => finiteNumber(r.elapsed_seconds),
      cell: (r) => formatDuration(finiteNumber(r.elapsed_seconds)) },
    ...varying.map<DataColumn<HistoryRow>>((name) => ({
      id: `p.${name}`, header: humanName(name), headerText: name, accessor: (r) => paramText(rowParams(r)[name]),
      cell: (r) => <span className="mono ops-small">{paramText(rowParams(r)[name])}</span>,
    })),
    { id: "cpu", header: "CPU used", numeric: true, width: 96, accessor: (r) => cpuUsage(r).pct,
      cell: (r) => { const u = cpuUsage(r); return <Tooltip content={`peak ${pct(u.pct)} · mean ${pct(u.mean)}`}>
        <span className="mono ops-small" tabIndex={0}>{u.used == null ? "—" : u.used.toFixed(1)} / {u.requested ?? "—"}</span></Tooltip>; } },
    { id: "mem", header: "Memory", numeric: true, width: 118, accessor: (r) => memoryUsage(r).pct,
      cell: (r) => { const u = memoryUsage(r); return <span className="mono ops-small">{formatMemory(u.used)} / {formatMemory(u.requested)}</span>; } },
    ...(gpu ? [{ id: "gpu", header: "GPU", numeric: true, width: 110, accessor: (r: HistoryRow) => gpuUsage(r).mean,
      cell: (r: HistoryRow) => { const g = gpuUsage(r); return <span className="mono ops-small">{pct(g.mean)} · mem {pct(g.memPct)}</span>; } }] : []),
    { id: "actions", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 132,
      cell: (r) => (
        <span className="ops-row-actions">
          {onClone && <Button size="sm" variant="ghost" icon="copy" aria-label={`Clone run ${r.jobid} into the form`}
            onClick={() => onClone(r)}>Clone</Button>}
          <Button asChild size="sm" variant="ghost">
            <Link to={`/runs/history?run=${encodeURIComponent(r.jobid)}&logs=1`} aria-label={`Logs of ${r.jobid}`}>Logs</Link>
          </Button>
        </span>
      ) },
  ], [varying, gpu, onClone]);
  if (history.error && !rows.length) return <Callout tone="bad" title="Could not load the run history">{history.error.message}</Callout>;
  return (
    <DataTable rows={rows} columns={columns} rowKey={(r) => String(r.jobid)} dense height={300}
      loading={history.loading} aria-label={`${step.label} runs`} exportName={`${step.step_id}-runs`}
      inspect={(r) => ({ kind: "job", id: `slurm/${r.jobid}` })}
      empty="No runs of this step yet." filterPlaceholder="Filter runs…"
      toolbar={<IconButton size="sm" icon="reset" label="Reload the history" onClick={() => history.reload()} />} />
  );
}

/* ── the card ─────────────────────────────────────────────────────────────── */

function OutputChips({ step }: { step: Step }) {
  const outs = step.outputs ?? [];
  if (!outs.length) return null;
  return (
    <span className="ops-step__outputs" aria-label="Outputs">
      {outs.map((o) => (
        <Tooltip key={o.key} content={`${o.path}${o.exists == null ? " (not probed: FASRC offline)" : o.exists ? " exists" : " not found"}`}>
          <Link className="ops-out" data-state={o.exists == null ? "unknown" : o.exists ? "yes" : "no"}
            to={`/system/storage?side=fasrc&dir=${encodeURIComponent(parentDir(o.path))}`}>
            <span className="ops-out__dot" aria-hidden="true" />{o.key}
          </Link>
        </Tooltip>
      ))}
    </span>
  );
}

export function StepCard({
  step, extraParams, sshConnected, embedded = false, showHistory = true, submitDisabled = false,
  submitDisabledHint, hideParams, initial, onSubmitted,
}: StepCardProps) {
  const params = useMemo(() => step.task_params ?? [], [step.task_params]);
  const hidden = useMemo(() => new Set([...Object.keys(extraParams ?? {}), ...(hideParams ?? [])]),
    [extraParams, hideParams]);
  const [values, setValues] = useState<FormValues>(() => (initial
    ? initialValues(step, { clone: initial.params }).values : initialValues(step).values));
  const [resources, setResources] = useState<Resources>(() => (initial?.resources
    ? resourcesFromRow(step, initial.resources) : defaultResources(step)));
  const [source, setSource] = useState<Source>(() => (initial
    ? { kind: "clone", jobid: initial.jobid } : { kind: initialValues(step).source }));
  const [expanded, setExpanded] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<{ text: string; offline?: boolean } | null>(null);
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const [historyKey, setHistoryKey] = useState(0);
  const feed = useJobsFeed();

  const errors = useMemo(() => validateValues(params, values, hidden), [params, values, hidden]);
  const resErrors = useMemo(() => validateResources(step, resources), [step, resources]);
  const invalid = Object.keys(errors).length > 0 || Object.keys(resErrors).length > 0;
  const { shown, more } = visibleParams(params, values, { hidden, expanded, errors });
  const changed = changedParams(params, values, hidden);
  // What the advisor is asked about: the task params this card submits.
  const advisedParams = useMemo(() => submitParams(step, values, extraParams, hidden), [step, values, extraParams, hidden]);
  const advisedFields = useMemo(() => editableResources(step), [step]);
  const live = feed.slurm.filter((j) => j.step_id === step.step_id && isLiveState(j.state));
  const queued = (feed.slurmQueue?.items as { id: string; step?: string; position: number }[] | undefined ?? [])
    .filter((it) => it.step === step.step_id);
  const laneBusy = feed.slurm.some((j) => isLiveState(j.state));
  const ownJob = outcome?.kind === "submitted" ? outcome.jobid : null;
  const shownJob = live[0]?.jobid ?? ownJob;

  const setValue = (name: string, v: string) => setValues((cur) => ({ ...cur, [name]: v }));
  function resetDefaults() {
    setValues(defaultValues(params));
    setResources(defaultResources(step));
    setSource({ kind: "defaults" });
  }
  function useLast() {
    const init = initialValues(step);
    setValues(init.values);
    setSource({ kind: init.source });
  }
  function clone(row: HistoryRow) {
    setValues(initialValues(step, { clone: rowParams(row) }).values);
    setResources(resourcesFromRow(step, row));
    setSource({ kind: "clone", jobid: String(row.jobid) });
    toast.info(`Form filled from run ${row.jobid}`);
  }

  async function submit() {
    const danger = dangerReasons(step, values, hidden);
    const forceRedownload = step.step_id === "archive_field_sample" && danger.length > 0
      && values.force_redownload === "1";
    const changes = changed.map((p) => `${p.name} = ${values[p.name] === "" ? "(unset)" : values[p.name]}`);
    const ok = await confirm({
      title: laneBusy ? `Queue “${step.label}”?` : `Submit “${step.label}”?`,
      message: [
        laneBusy ? "Another FASRC job is running: this one waits in the local queue and starts when it succeeds."
          : "Submits to SLURM on FASRC.",
        changes.length ? ` Changed: ${changes.join(", ")}.` : " All task params at their defaults.",
        ` ${resources.n_cpus} CPU · ${step.needs_gpu ? `${resources.n_gpus} GPU · ` : ""}${resources.memory} · ${resources.time_limit}.`,
        danger.length ? ` Destructive: ${danger.join("; ")}.` : "",
      ].join(""),
      tone: danger.length ? "danger" : "default",
      confirmLabel: laneBusy ? "Queue" : danger.length ? "Submit anyway" : "Submit",
      requireText: forceRedownload ? "redownload" : undefined,
    });
    if (!ok) return;
    setBusy(true); setError(null); setOutcome(null);
    const body = submitBody(step, values, resources, extraParams, hidden);
    if (forceRedownload) body.confirm_force_redownload = "yes";
    try {
      const r = await apiPost<SubmitResp>(`/api/fasrc/steps/${encodeURIComponent(step.step_id)}/submit`, body);
      if (r.error) { setError({ text: r.error }); return; }
      if (r.queued) {
        const items = r.queue?.items ?? [];
        const position = items.length || r.queue?.count || 1;
        setOutcome({ kind: "queued", position, count: r.queue?.count ?? position, label: r.label });
        toast.info(`Queued “${step.label}” (position ${position})`);
      } else {
        const jobid = String(r.jobid ?? r.slurm_id ?? "");
        setOutcome({ kind: "submitted", jobid });
        toast.success(`Submitted “${step.label}”${jobid ? ` as job ${jobid}` : ""}`);
      }
      onSubmitted?.(r);
      setHistoryKey((k) => k + 1);
      void refreshJobsFeed();
      void invalidate("/api/fasrc/history");
    } catch (e) {
      setError({ text: e instanceof Error ? e.message : String(e), offline: isFasrcOffline(e) || (e instanceof ApiError && e.status === 503) });
    } finally { setBusy(false); }
  }

  const sourceLine = source.kind === "last" ? "Prefilled from the last successful run"
    : source.kind === "clone" ? `Cloned from run ${source.jobid ?? ""}`.trim() : null;
  const hasLast = !!step.last_params && initialValues(step).source === "last";

  const content = (
    <div className="ops-step">
      {(live.length > 0 || queued.length > 0) && (
        <div className="ops-step__live" role="status">
          {live.map((j) => (
            <Link key={j.jobid} className="ops-step__livejob" to={`/runs/live?job=${encodeURIComponent(j.jobid)}`}>
              <Badge size="sm" tone={jobStateTone(j.state)} dot>{j.state}</Badge>
              <span className="mono">#{j.jobid}</span>
              {j.time && <span className="ops-dim mono">{j.time}{j.time_limit ? ` / ${j.time_limit}` : ""}</span>}
            </Link>
          ))}
          {queued.map((q) => <Badge key={q.id} size="sm" tone="warn">queued · #{q.position}</Badge>)}
        </div>
      )}
      {(sourceLine || changed.length > 0 || hasLast) && (
        <div className="ops-step__prefill">
          {sourceLine && <Badge size="sm" tone={source.kind === "clone" ? "accent" : "info"}>{sourceLine}</Badge>}
          {changed.length > 0 && <span className="ops-dim ops-small">{changed.length} changed from the defaults</span>}
          <span className="ops-spacer" />
          {hasLast && source.kind !== "last" && <Button size="sm" variant="ghost" onClick={useLast}>Use last run</Button>}
          {(changed.length > 0 || source.kind !== "defaults") && <Button size="sm" variant="ghost" icon="reset" onClick={resetDefaults}>Defaults</Button>}
        </div>
      )}
      {shown.length > 0 && (
        <div className="ops-step__grid">
          {shown.map((p) => (
            <ParamField key={p.name} param={p} value={values[p.name] ?? ""} error={errors[p.name]}
              onChange={(v) => setValue(p.name, v)} disabled={busy} />
          ))}
        </div>
      )}
      {(more > 0 || expanded) && (
        <Button size="sm" variant="ghost" icon={expanded ? "chevronUp" : "chevronDown"} onClick={() => setExpanded((x) => !x)}>
          {expanded ? "Fewer params" : `${more} more param${more === 1 ? "" : "s"}`}
        </Button>
      )}
      <div className="ops-step__res" aria-label="Resources">
        {(Object.keys(RESOURCE_LABEL) as (keyof Resources)[])
          .filter((k) => k !== "n_gpus" || step.needs_gpu)
          .map((k) => {
            const locked = (k === "n_cpus" && step.fixed_cpus != null) || (k === "n_gpus" && step.fixed_gpus != null);
            const perModel = step.step_id === "ensemble_train" ? " / member" : "";
            return (
              <Field key={k} label={`${RESOURCE_LABEL[k]}${perModel}`} error={resErrors[k]}
                hint={locked ? "Fixed for this step." : k === "time_limit" ? "SLURM time limit, e.g. 2:00:00 or 1-00:00:00." : undefined}>
                <Input size="sm" value={resources[k]} disabled={locked || busy} spellCheck={false}
                  onChange={(v) => setResources((r) => ({ ...r, [k]: v }))} />
              </Field>
            );
          })}
        <Tooltip content="The partition is fixed per step (the server forces it).">
          <span className="ops-step__partition" tabIndex={0}><Badge size="sm">{step.defaults.partition}</Badge></span>
        </Tooltip>
      </div>
      <ResourceAdvice stepId={step.step_id} params={advisedParams} resources={resources} fields={advisedFields}
        perTask={step.step_id === "ensemble_train" ? "member" : undefined}
        onApply={(advised) => setResources((cur) => applyAdvisedResources(step, cur, advised))} />
      <div className="ops-step__submit">
        <Button variant="primary" icon="server" loading={busy} onClick={submit}
          disabled={!sshConnected || submitDisabled || invalid}>
          {laneBusy ? "Queue" : "Submit"}
        </Button>
        {!sshConnected && <span className="ops-dim ops-small">FASRC offline — <Link to="/system/connections">connect</Link></span>}
        {sshConnected && submitDisabled && submitDisabledHint && <span className="ops-dim ops-small">{submitDisabledHint}</span>}
        {sshConnected && invalid && <span className="ops-bad ops-small">Fix the highlighted fields</span>}
        <span className="ops-spacer" />
        <OutputChips step={step} />
      </div>
      {error && (
        <Callout tone="bad" title={error.offline ? "FASRC not connected" : "Submission refused"} onDismiss={() => setError(null)}
          action={error.offline ? <Button asChild size="sm"><Link to="/system/connections">Connect</Link></Button> : undefined}>
          <span className="ops-pre">{error.text}</span>
        </Callout>
      )}
      {outcome?.kind === "queued" && (
        <Callout tone="info" title={`Queued — position ${outcome.position} of ${outcome.count}`} onDismiss={() => setOutcome(null)}
          action={<Button asChild size="sm"><Link to="/runs/live">Queue</Link></Button>}>
          It is submitted when the running job succeeds; a failure halts the queue.
        </Callout>
      )}
      {shownJob && <SlurmMonitor jobid={shownJob} compact />}
      {showHistory && (
        <Section title="Previous runs" collapsible defaultOpen={!embedded} className="ops-step__history">
          <StepHistory step={step} refreshKey={historyKey} onClone={clone} />
        </Section>
      )}
    </div>
  );
  if (embedded) {
    return (
      <div className="ops-step-inline fasrc-step-inline">
        <div className="ops-step-inline__head fasrc-step-inline__head">
          <div><div className="eyebrow">SLURM submission</div><strong>{step.label}</strong> <code className="mono ops-dim">{step.step_id}</code></div>
          <Badge size="sm">{step.needs_gpu ? "GPU" : "CPU"}</Badge>
        </div>
        {content}
      </div>
    );
  }
  return (
    <Card className="ops-stepcard">
      <CardHead title={step.label} sub={<code className="mono">{step.step_id}</code>}
        right={<Badge size="sm">{step.needs_gpu ? "GPU" : "CPU"}</Badge>} />
      <CardBody>{content}</CardBody>
    </Card>
  );
}

/** Look a step up in the registry and render its card (loading / missing states). */
export function StepById({
  stepId, extraParams, embedded = false, showHistory = true, submitDisabled = false, submitDisabledHint,
  hideParams, initial, onSubmitted,
}: Omit<StepCardProps, "step" | "sshConnected"> & { stepId: string }) {
  const { data, loading, error } = useStepsStatus();
  const wrap = (node: ReactNode) => (embedded ? <div className="ops-step-inline">{node}</div> : <Card><CardBody>{node}</CardBody></Card>);
  if (loading && !data) return wrap(<Skeleton lines={4} />);
  if (error && !data) return wrap(<Callout tone="bad" title="Could not load the FASRC steps">{error.message}</Callout>);
  const step = (data?.steps ?? []).find((s) => s.step_id === stepId);
  if (!step) return wrap(<Callout tone="warn" title="Unknown step">Step <code>{stepId}</code> is not registered on the server.</Callout>);
  return <StepCard key={`${step.step_id}:${initial?.jobid ?? ""}`} step={step} extraParams={extraParams}
    sshConnected={!!data?.ssh_connected} embedded={embedded} showHistory={showHistory}
    submitDisabled={submitDisabled} submitDisabledHint={submitDisabledHint} hideParams={hideParams}
    initial={initial} onSubmitted={onSubmitted} />;
}
