/* Models › Train (`/models/train`): submit an ensemble_train SLURM job.
   From the top: the "Running batch" strip (the live SLURM ensemble_train
   jobs, the feed Runs › Live reads; nothing when none runs); the bar — Add /
   Continue / Fork, Repeat last batch, Clone a past job, Recipe reset; the
   member rows (loss, depth, single or MULTI-knee, noise aug, bootstrap, ICNR,
   seed), or the Continue picker (TIMEOUT members one click away), or the Fork
   source; Scheduling (models at once, base seed, evaluate every, batch, and
   the LR schedule System › Config sets); Forward model (the PSF-warp and
   saturation values System › Config owns, read-only with an edit link — the
   form never sends them, so the step fills them from Config — and the
   trainer's own knobs, shown as the recipe and editable for one batch);
   Resources (CPUs, memory, time; under them the resource advisor's
   "Recommended from N past runs" for the built params, applied only by its
   Apply, per model = per array task); the live command preview (POST
   /ensemble/train/preview: the member names the submit allocates, the exact
   argv; a read-only exemption, nothing reaches FASRC); and the confirmed
   submit, whose label repeats the count ("Submit 4 members to SLURM").
   URL: ?mode=, ?members= (continue), ?member= (fork), ?from=<jobid>
   (clone). */
import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { ApiError, apiPost } from "../../../api/client";
import { useJobsFeed } from "../../../api/jobs";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { pagePath } from "../../../app/nav";
import { usePageActions } from "../../../app/palette";
import { useFasrcStatus } from "../../../app/status";
import { useUrlState } from "../../../hooks/useUrlState";
import { LOSS_COLOR } from "../../../colors";
import {
  Badge, Button, Callout, Caption, Card, CardBody, CardHead, Checkbox, CopyButton, FactsList, Field, IconButton, Input,
  NumberField, Page, Segmented, Select, Switch, Toolbar, ToolbarGroup, ToolbarSeparator, ToolbarSpacer, Tooltip, confirm, toast,
} from "../../../ui";
import { url, useMembers, useTrainingJobs, type MemberRow, type TrainingJob } from "../api";
import { CONFIG_FORWARD_KEYS, changedConfigKnobs, forwardModelFacts, kneeText, knobsChangedText, lrScheduleText, memberName, memberNumber, runningBatches, stepsText, submitLabel } from "../model";
import {
  DEFAULT_MULTI_KNEES, KNEE_LOSSES, LOSSES, RECIPE_RESOURCES, buildParams, continueTarget, defaultForm, defaultResources,
  formFromJob, lastBatch, newRow, recipeSummary, validate, type Resources, type SpecRow, type TrainForm, type TrainMode,
} from "../trainModel";
import { ResourceAdvice } from "../../shared/ResourceAdvice";
import type { AdviceField } from "../../shared/resourceAdviceModel";
import "../models.css";

type Preview = { ok: boolean; mode: string; member_names: string[]; count: number; array: { tasks: number; max_parallel: number } | null;
  command: string[]; command_text: string; base_seed: string | number | null; star_prior: boolean };
type StepInfo = { step_id: string; defaults: { partition: string; n_cpus: number; n_gpus: number; memory: string; time_limit: string }; fixed_gpus?: number | null };
type ConfigPayload = { config?: Record<string, unknown>; defaults?: Record<string, unknown>; used_by?: Record<string, string[]> };

const CONFIG_PATH = "/system/config";
/** The resources this form edits (the GPU count is fixed per model). */
const ADVICE_FIELDS: readonly AdviceField[] = ["n_cpus", "memory", "time_limit"];

function SpecRowEditor({ row, i, mode, onChange, onRemove, onDuplicate, canRemove }: {
  row: SpecRow; i: number; mode: TrainMode; onChange: (p: Partial<SpecRow>) => void; onRemove: () => void;
  onDuplicate: () => void; canRemove: boolean;
}) {
  const multi = row.kneeMode === "multi";
  return (
    <div className="mdl-spec__row" style={{ borderLeft: `3px solid ${LOSS_COLOR[row.loss]}` }}>
      <span className="mdl-spec__idx">#{i + 1}</span>
      <Field label="Loss"><Select value={row.loss} onChange={(v) => onChange({ loss: v })} options={LOSSES.map((v) => ({ value: v, label: v.toUpperCase() }))} /></Field>
      {mode === "add" && <NumberField label="Depth" value={row.blocks} onChange={(v) => onChange({ blocks: v })} min={4} max={64} />}
      <Field label="Knee" hint="single: one asinh knee (100 = the per-band default). multi: stretch at every knee at once.">
        <Select value={row.kneeMode} onChange={(v) => onChange({ kneeMode: v as SpecRow["kneeMode"] })}
          options={[{ value: "single", label: "single" }, { value: "multi", label: "multi-knee" }]} />
      </Field>
      {multi ? <>
        <Field label="Knees [e⁻]" className="mdl-spec__wide"><Input value={row.knees} onChange={(v) => onChange({ knees: v })} placeholder={DEFAULT_MULTI_KNEES} /></Field>
        <Field label="Output knee" hint="One output image stretched at this knee and scored at every knee (option 2, best so far). Blank: one image per knee (option 1).">
          <Input value={row.outputKnee} onChange={(v) => onChange({ outputKnee: v })} placeholder="per-knee heads" />
        </Field>
        <Field label="Knee loss" hint="balanced: every knee's channel weighs the same; plain: raw sum.">
          <Select value={row.kneeLoss} onChange={(v) => onChange({ kneeLoss: v as SpecRow["kneeLoss"] })} options={KNEE_LOSSES.map((v) => ({ value: v, label: v }))} />
        </Field>
      </> : <NumberField label="Knee [e⁻]" value={row.knee} onChange={(v) => onChange({ knee: v })} min={0.01} step="any" />}
      <NumberField label="Noise aug" value={row.noise} onChange={(v) => onChange({ noise: v })} min={0} max={5} step={0.25} />
      <NumberField label="Bootstrap" value={row.boot} onChange={(v) => onChange({ boot: v })} min={0} max={0.99} step={0.05} placeholder="off" />
      <NumberField label="Seed" value={row.seed} onChange={(v) => onChange({ seed: v })} placeholder="auto" />
      <div className="mdl-row mdl-spec__tools">
        {mode === "add" && <Checkbox checked={row.icnr} onChange={(v) => onChange({ icnr: v })}>ICNR</Checkbox>}
        <IconButton icon="copy" size="sm" label={`Duplicate member ${i + 1}`} onClick={onDuplicate} />
        <IconButton icon="close" size="sm" label={`Remove member ${i + 1}`} onClick={onRemove} disabled={!canRemove} />
      </div>
    </div>
  );
}

function ContinuePicker({ rows, picked, onChange }: { rows: MemberRow[]; picked: string[]; onChange: (v: string[]) => void }) {
  const set = new Set(picked);
  const timeouts = rows.filter((m) => m.timeout).map((m) => m.name);
  return (
    <div className="mdl-stack mdl-stack--tight">
      <div className="mdl-row">
        <span className="mdl-muted">{picked.length} selected</span>
        <Button size="sm" variant="ghost" disabled={!timeouts.length} onClick={() => onChange(timeouts)}>TIMEOUT members ({timeouts.length})</Button>
        <Button size="sm" variant="ghost" onClick={() => onChange(rows.map((m) => m.name))}>all</Button>
        <Button size="sm" variant="ghost" disabled={!picked.length} onClick={() => onChange([])}>none</Button>
      </div>
      <div className="mdl-picker" role="group" aria-label="Members to continue">
        {rows.map((m) => (
          <label key={m.name} className="mdl-pick" data-on={set.has(m.name)} style={{ ["--sw" as string]: LOSS_COLOR[m.loss] }}>
            <span className="mdl-pick__top">
              <span>#{memberNumber(m.name)}{m.timeout ? " ⚠" : ""}</span>
              <Checkbox checked={set.has(m.name)} aria-label={`Continue ${m.name}`}
                onChange={(on) => onChange(on ? [...picked, m.name] : picked.filter((x) => x !== m.name))} />
            </span>
            <span className="mdl-pick__meta">{stepsText(m.step, m.target_steps)} · {m.loss} · {kneeText(m).text}</span>
          </label>
        ))}
      </div>
    </div>
  );
}

/** The live SLURM ensemble_train jobs (Runs › Live's feed), one line each. */
function RunningBatch() {
  const feed = useJobsFeed();
  const jobs = useTrainingJobs();
  const batches = useMemo(() => runningBatches(feed.slurm, jobs.data?.jobs ?? []), [feed.slurm, jobs.data]);
  if (!batches.length) return null;
  return (
    <section className="mdl-batch" aria-label="Running batch">
      {batches.map((b) => (
        <p key={b.jobid} className="mdl-batch__line">
          <span className="mdl-dot" data-tone={b.state === "PENDING" ? "info" : "good"} aria-hidden />
          <strong>Running batch</strong>
          <span>{b.text}</span>
          <Button size="sm" variant="ghost" onClick={() => openInspector({ kind: "job", id: `slurm/${b.jobid}` })}>Monitor</Button>
          <Link to={pagePath("runs", { tab: "live" })}>Runs › Live</Link>
        </p>
      ))}
    </section>
  );
}

/** Which jobs seed the resources: continue jobs, or new batches (add, fork). */
const resKind = (m: TrainMode): "new" | "continue" => (m === "continue" ? "continue" : "new");

/** The Train tab. Other pages hand members over in the URL (?members= to
 *  continue, ?member= to fork); the form reads them when it mounts, so a new
 *  handover remounts it. A mode switch made inside the form keeps its edits. */
export default function Train() {
  const [urlMembers] = useUrlState("members", "");
  const [urlMember] = useUrlState("member", "");
  return <TrainPage key={`${urlMembers}|${urlMember}`} />;
}

function TrainPage() {
  const members = useMembers();
  const jobs = useTrainingJobs();
  const steps = useResource<{ ssh_connected: boolean; steps: StepInfo[] }>("/api/fasrc/steps/status", [], { ttl: 60_000 });
  const config = useResource<ConfigPayload>(url.config(), [], { ttl: 60_000 });
  const fasrc = useFasrcStatus().data;
  const [urlMode, setUrlMode] = useUrlState<TrainMode>("mode", "add");
  const [urlMembers] = useUrlState("members", "");
  const [urlMember] = useUrlState("member", "");
  const [from, setFrom] = useUrlState("from", "");
  const [form, setForm] = useState<TrainForm>(() => {
    const f = defaultForm();
    f.mode = urlMode;
    if (urlMembers) f.members = urlMembers.split(",").map((n) => memberName(n)).filter((n): n is string => !!n);
    if (urlMember) f.forkFrom = memberName(urlMember) ?? "";
    return f;
  });
  const [override, setOverride] = useState(false);
  const [res, setRes] = useState<Resources | null>(null);
  /** Where the resources came from: a job id, null = the recipe. */
  const [resFrom, setResFrom] = useState<string | null>(null);
  /** The kind of job the resources were seeded from; null once the user owns them. */
  const [resSeed, setResSeed] = useState<"new" | "continue" | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState<{ jobid?: string; queued?: boolean } | null>(null);
  const cloned = useRef<string | null>(null);

  const step = steps.data?.steps.find((s) => s.step_id === "ensemble_train");
  // Resources start from the last finished batch of the same kind (else the
  // recipe: 16 CPUs, 3 h), never the step's generic 4 CPUs / 48 h; switching
  // the mode re-seeds until the resources are edited.
  useEffect(() => {
    if (!jobs.data && !jobs.error) return;
    if (res && (resSeed == null || resSeed === resKind(form.mode))) return;
    const d = defaultResources(jobs.data?.jobs ?? [], form.mode);
    setRes({ n_cpus: d.n_cpus, memory: d.memory, time_limit: d.time_limit });
    setResFrom(d.from);
    setResSeed(resKind(form.mode));
  }, [jobs.data, jobs.error, res, resSeed, form.mode]);
  const ownRes = (r: Resources) => { setRes(r); setResSeed(null); };
  // "Continue" from Members or the Leaderboard (?members=): up to each
  // member's recorded target, not a fixed +20k (once, when the rows arrive).
  const targetSet = useRef(false);
  useEffect(() => {
    if (targetSet.current || form.mode !== "continue" || !urlMembers || !members.data) return;
    targetSet.current = true;
    const t = continueTarget(members.data.members, form.members);
    if (t) setForm((f) => ({ ...f, continueBasis: "target", targetSteps: String(t) }));
  }, [members.data]); // eslint-disable-line react-hooks/exhaustive-deps

  const patch = (p: Partial<TrainForm>) => setForm((f) => ({ ...f, ...p }));
  const setMode = (m: TrainMode) => { patch({ mode: m }); setUrlMode(m); };
  const setGeometry = (p: Partial<TrainForm["geometry"]>) => setForm((f) => ({ ...f, geometry: { ...f.geometry, ...p } }));
  const setRow = (i: number, p: Partial<SpecRow>) => setForm((f) => ({ ...f, rows: f.rows.map((r, j) => (j === i ? { ...r, ...p } : r)) }));
  const reset = () => { setForm({ ...defaultForm(), mode: form.mode }); setFrom(""); ownRes({ ...RECIPE_RESOURCES }); setResFrom(null); setOverride(false); };

  const clone = (job: TrainingJob) => {
    const f = formFromJob(job);
    setForm(f);
    setUrlMode(f.mode);
    setFrom(job.jobid);
    if (job.req_time_limit || job.req_memory || job.req_cpus) {
      setResSeed(null);
      setRes((r) => ({ n_cpus: job.req_cpus ? String(job.req_cpus) : r?.n_cpus ?? RECIPE_RESOURCES.n_cpus,
        memory: job.req_memory || r?.memory || RECIPE_RESOURCES.memory, time_limit: job.req_time_limit || r?.time_limit || RECIPE_RESOURCES.time_limit }));
      setResFrom(job.jobid);
    }
    toast.info(`Cloned job ${job.jobid}`, { description: recipeSummary(job) });
  };
  // ?from=<jobid> deep link → clone once the job list arrives
  useEffect(() => {
    if (!from || cloned.current === from) return;
    const job = jobs.data?.jobs.find((j) => j.jobid === from);
    if (job) { cloned.current = from; clone(job); }
  }, [from, jobs.data]); // eslint-disable-line react-hooks/exhaustive-deps

  const params = useMemo(() => buildParams(form), [form]);
  const errors = useMemo(() => validate(form), [form]);
  const body = JSON.stringify(params);
  useEffect(() => {
    if (errors.length) { setPreview(null); setPreviewError(null); return; }
    const ctl = new AbortController();
    const t = window.setTimeout(() => {
      apiPost<Preview>("/ensemble/train/preview", params)
        .then((p) => { if (!ctl.signal.aborted) { setPreview(p); setPreviewError(null); } })
        .catch((e) => { if (!ctl.signal.aborted) { setPreview(null); setPreviewError(e instanceof ApiError ? e.message : String(e)); } });
    }, 350);
    return () => { ctl.abort(); window.clearTimeout(t); };
  }, [body, errors.length]); // eslint-disable-line react-hooks/exhaustive-deps

  const offline = fasrc ? !fasrc.ssh_connected : true;
  const count = form.mode === "continue" ? form.members.length : form.rows.length;
  const label = submitLabel(form.mode, count);
  async function submit() {
    if (errors.length || !res) return;
    const what = form.mode === "continue"
      ? `Continue ${form.members.length} member(s)`
      : `${form.rows.length} ${form.mode === "fork" ? "fork" : "new"} member(s): ${preview?.member_names.join(", ") ?? "…"}`;
    const ok = await confirm({ title: `${label}?`, message: `${what} · per model ${res.n_cpus} CPUs, ${res.memory}, ${res.time_limit} on ${step?.defaults.partition ?? "gpu"}. It queues locally when another job is active.`, confirmLabel: "Submit" });
    if (!ok) return;
    setSubmitting(true);
    try {
      const r = await apiPost<{ ok?: boolean; jobid?: string; slurm_id?: string; queued?: boolean; error?: string }>(
        "/api/fasrc/steps/ensemble_train/submit",
        { ...params, partition: step?.defaults.partition ?? "gpu", n_cpus: res.n_cpus, n_gpus: String(step?.fixed_gpus ?? 1),
          memory: res.memory, time_limit: res.time_limit, confirm: "yes" });
      if (r.error || r.ok === false) throw new Error(r.error ?? "refused");
      const jobid = r.jobid ?? r.slurm_id;
      setSubmitted({ jobid, queued: r.queued });
      if (jobid) toast.success(`Submitted SLURM job ${jobid}`, { action: { label: "Monitor", onClick: () => openInspector({ kind: "job", id: `slurm/${jobid}` }) } });
      else toast.success(r.queued ? "Queued behind the active FASRC job" : "Submitted");
      void jobs.reload();
    } catch (e) {
      toast.error("Submit failed", { description: e instanceof Error ? e.message : String(e) });
    } finally { setSubmitting(false); }
  }

  const last = lastBatch(jobs.data?.jobs ?? []);
  usePageActions([
    { id: "train-last", label: "Train: repeat the last batch recipe", group: "Train", disabled: !last, run: () => last && clone(last) },
    { id: "train-recipe", label: "Train: reset to the standard recipe", group: "Train", run: reset },
    { id: "train-add-multi", label: "Train: add a multi-knee member row", group: "Train", run: () => patch({ rows: [...form.rows, newRow({ kneeMode: "multi" })] }) },
    { id: "train-submit", label: `Train: ${label}`, group: "Train", disabled: offline || !!errors.length, run: () => void submit() },
  ]);

  const activeRows = members.data?.members ?? [];
  const jobOptions = (jobs.data?.jobs ?? []).slice(0, 40).map((j) => ({
    value: j.jobid, label: `${j.jobid} · ${j.submitted_at?.slice(0, 10) ?? ""} · ${recipeSummary(j)}`, hint: j.state ?? "",
  }));
  const g = form.geometry;
  const cfg = config.data?.config;
  const forwardConfig = forwardModelFacts(cfg);
  const schedule = lrScheduleText(cfg);
  const forwardKeys = CONFIG_FORWARD_KEYS.map((k) => k.key);
  const forwardChanged = knobsChangedText(changedConfigKnobs(config.data, "ensemble_train", { only: forwardKeys }));
  const scheduleChanged = knobsChangedText(changedConfigKnobs(config.data, "ensemble_train", { except: forwardKeys }));
  const editLink = (changed: string | null, what: string) => (
    <Button size="sm" variant="ghost" asChild>
      <Link to={CONFIG_PATH} title={`The ${what} values live in System › Config`}>{changed ? `${changed} · Edit` : "Edit in System › Config"}</Link>
    </Button>
  );
  const recipeFacts = [
    { label: "Live forward model", value: g.forward_onthefly ? "on" : "off" },
    { label: "HR example side", value: g.hr_crop_size, unit: "px" },
    { label: "Examples per field", value: g.crops_per_field },
    { label: "PSF clusters per member", value: g.psf_subset },
    { label: "Target PSF FWHM", value: g.target_psf_fwhm_arcsec, unit: "″" },
  ];
  return (
    <Page className="mdl-page">
      <RunningBatch />
      <Toolbar label="Train controls">
        <Segmented<TrainMode> size="sm" aria-label="Train mode" value={form.mode} onChange={setMode}
          options={[{ value: "add", label: "Add" }, { value: "continue", label: "Continue" }, { value: "fork", label: "Fork" }]} />
        <ToolbarSeparator />
        <Tooltip content={last ? recipeSummary(last) : "No past batch in the job log"}>
          <span><Button size="sm" disabled={!last} onClick={() => last && clone(last)}>Repeat last batch</Button></span>
        </Tooltip>
        <ToolbarGroup label="Clone">
          <Select searchable size="sm" aria-label="Clone a past job" value={from} placeholder="a past job…"
            onChange={(id) => { const j = jobs.data?.jobs.find((x) => x.jobid === id); if (j) clone(j); }} options={jobOptions} />
        </ToolbarGroup>
        <ToolbarSpacer />
        <Button size="sm" variant="ghost" icon="reset" onClick={reset}>Recipe</Button>
      </Toolbar>
      {form.mode === "continue" ? (
        <Card>
          <CardHead title="Continue members" sub="each member becomes one task of a capped job array" />
          <CardBody>
            <div className="mdl-form mdl-form--gap">
              <Field label="Continue by"><Select value={form.continueBasis} onChange={(v) => patch({ continueBasis: v as TrainForm["continueBasis"] })}
                options={[{ value: "extra", label: "extra steps" }, { value: "target", label: "up to step N" }]} /></Field>
              {form.continueBasis === "target"
                ? <NumberField label="Up to step" value={form.targetSteps} onChange={(v) => patch({ targetSteps: v })} min={1000} step={1000}
                    hint="members already at N are skipped" />
                : <NumberField label="Extra steps" value={form.extraSteps} onChange={(v) => patch({ extraSteps: v })} min={1000} step={1000} />}
            </div>
            <ContinuePicker rows={activeRows} picked={form.members} onChange={(v) => patch({ members: v })} />
          </CardBody>
        </Card>
      ) : (
        <Card>
          <CardHead title={form.mode === "fork" ? "Fork into new members" : "New members"}
            sub={form.mode === "fork" ? "weights from an existing member; depth and init inherited" : "one row per model"}
            right={<div className="mdl-row">
              <Button size="sm" icon="plus" onClick={() => patch({ rows: [...form.rows, newRow(form.rows[form.rows.length - 1])] })}>member</Button>
              <Button size="sm" icon="plus" onClick={() => patch({ rows: [...form.rows, newRow({ kneeMode: "multi" })] })}>multi-knee</Button>
            </div>} />
          <CardBody>
            <div className="mdl-form mdl-form--gap">
              <NumberField label="Steps" value={form.steps} onChange={(v) => patch({ steps: v })} min={1000} step={1000} />
              {form.mode === "fork" && <>
                <Field label="Fork from">
                  <Select searchable value={form.forkFrom} onChange={(v) => patch({ forkFrom: v })} placeholder="pick a member"
                    options={activeRows.map((m) => ({ value: m.name, label: `#${memberNumber(m.name)}`, hint: `${m.loss} · ${kneeText(m).text}` }))} />
                </Field>
                <Field label="Track"><Select value={form.forkTrack} onChange={(v) => patch({ forkTrack: v as TrainForm["forkTrack"] })}
                  options={[{ value: "psnr", label: "PSNR-best" }, { value: "loss", label: "loss-best" }]} /></Field>
              </>}
            </div>
            <div className="mdl-spec">
              {form.rows.map((r, i) => (
                <SpecRowEditor key={i} row={r} i={i} mode={form.mode} onChange={(p) => setRow(i, p)} canRemove={form.rows.length > 1}
                  onRemove={() => patch({ rows: form.rows.filter((_, j) => j !== i) })}
                  onDuplicate={() => patch({ rows: [...form.rows.slice(0, i + 1), newRow(r), ...form.rows.slice(i + 1)] })} />
              ))}
            </div>
          </CardBody>
        </Card>
      )}
      <div className="mdl-grid">
        <Card>
          <CardHead title="Scheduling" right={scheduleChanged ? editLink(scheduleChanged, "LR schedule") : undefined} />
          <CardBody>
            <div className="mdl-form">
              <NumberField label="Models at once" value={form.arrayMaxParallel} onChange={(v) => patch({ arrayMaxParallel: v })} min={1} />
              <NumberField label="Base seed" value={form.baseSeed} onChange={(v) => patch({ baseSeed: v })} placeholder="drawn at submit" />
              <NumberField label="Evaluate every" value={form.evaluateEvery} onChange={(v) => patch({ evaluateEvery: v })} placeholder="default" unit="steps" />
              <NumberField label="Batch" value={g.batch_size} onChange={(v) => setGeometry({ batch_size: v })} min={1} max={64} />
            </div>
            {schedule && <Caption>{schedule}, from <Link to={CONFIG_PATH}>System › Config</Link>.</Caption>}
          </CardBody>
        </Card>
        <Card>
          <CardHead title="Forward model" right={editLink(forwardChanged, "forward-model")} />
          <CardBody>
            <div className="mdl-stack mdl-stack--tight">
              {forwardConfig.length > 0
                ? <FactsList title="From System › Config" facts={forwardConfig} />
                : <p className="mdl-note">{config.loading ? "Reading System › Config…" : "System › Config is not readable; the step uses its defaults."}</p>}
              {override ? (
                <div className="mdl-form">
                  <Switch checked={g.forward_onthefly} onChange={(v) => setGeometry({ forward_onthefly: v })}>live forward model</Switch>
                  <NumberField label="HR example side" value={g.hr_crop_size} onChange={(v) => setGeometry({ hr_crop_size: v })} min={2} max={508} step={2} unit="px" />
                  <NumberField label="Examples per field" value={g.crops_per_field} onChange={(v) => setGeometry({ crops_per_field: v })} min={1} max={25} />
                  <NumberField label="PSF clusters per member" value={g.psf_subset} onChange={(v) => setGeometry({ psf_subset: v })} min={1} />
                  <NumberField label="Target PSF FWHM" value={g.target_psf_fwhm_arcsec} onChange={(v) => setGeometry({ target_psf_fwhm_arcsec: v })} min={0} max={2} step={0.001} unit="″" />
                </div>
              ) : <FactsList title="Trainer recipe" facts={recipeFacts} />}
              <Switch size="sm" checked={override} onChange={setOverride}>Override the trainer recipe for this batch</Switch>
            </div>
          </CardBody>
        </Card>
        <Card>
          <CardHead title="Resources" sub={step ? `${step.defaults.partition} partition · ${step.fixed_gpus ?? 1} GPU per model` : undefined} />
          <CardBody>
            {res && (
              <div className="mdl-stack mdl-stack--tight">
                <div className="mdl-form">
                  <NumberField label="CPUs / model" value={res.n_cpus} onChange={(v) => ownRes({ ...res, n_cpus: v })} min={1}
                    hint={resFrom ? `As job ${resFrom} (the last finished ${form.mode === "continue" ? "continue job" : "new batch"})` : "The recipe: 16 keep the GPU fed"} />
                  <Field label="Memory / model"><Input value={res.memory} onChange={(v) => ownRes({ ...res, memory: v })} /></Field>
                  <Field label="Time limit / model" hint="The 70k-step recipe takes 2.5–3 h on the gpu partition."><Input value={res.time_limit} onChange={(v) => ownRes({ ...res, time_limit: v })} /></Field>
                </div>
                <ResourceAdvice stepId="ensemble_train" params={params} fields={ADVICE_FIELDS} perTask="model"
                  resources={{ n_cpus: res.n_cpus, n_gpus: String(step?.fixed_gpus ?? 1), memory: res.memory, time_limit: res.time_limit }}
                  onApply={(r) => ownRes({ n_cpus: r.n_cpus, memory: r.memory, time_limit: r.time_limit })} />
              </div>
            )}
          </CardBody>
        </Card>
      </div>
      <Card>
        <CardHead title="Command" sub="what the submit runs, built locally; nothing reaches FASRC until Submit" />
        <CardBody>
          <div className="mdl-stack mdl-stack--tight">
            {errors.length > 0 && <Callout tone="warn" title="Fix before submitting">{errors.join(" · ")}</Callout>}
            {previewError && <Callout tone="bad" title="The submit would be refused"><span className="mdl-mono">{previewError}</span></Callout>}
            {preview && <>
              <div className="mdl-row">
                <span className="mdl-muted">{preview.mode === "continue" ? "Continues" : "New members"}</span>
                <div className="mdl-names">{preview.member_names.map((n) => <Badge key={n} tone={preview.mode === "continue" ? undefined : "accent"}>{n}</Badge>)}</div>
                {preview.array && <span className="mdl-faint">array of {preview.array.tasks}, {preview.array.max_parallel} at once</span>}
                <span className="mdl-faint">seed {preview.base_seed ?? "drawn at submit"}</span>
              </div>
              <div className="mdl-row mdl-row--top">
                <pre className="mdl-cmd" aria-label="Final command">{preview.command_text}</pre>
                <CopyButton value={preview.command_text} label="Copy the command" />
              </div>
            </>}
            <div className="mdl-row">
              <Button variant="primary" icon="server" loading={submitting} disabled={offline || !!errors.length || !!previewError || !res}
                onClick={() => void submit()}>{label}</Button>
              {offline && <span className="mdl-warn">FASRC offline{fasrc?.last_error ? `: ${fasrc.last_error}` : ""}</span>}
              {submitted?.jobid && <Button size="sm" variant="ghost" onClick={() => openInspector({ kind: "job", id: `slurm/${submitted.jobid}` })}>Monitor job {submitted.jobid}</Button>}
              {submitted?.queued && <Badge tone="info">queued locally</Badge>}
            </div>
          </div>
        </CardBody>
      </Card>
    </Page>
  );
}
