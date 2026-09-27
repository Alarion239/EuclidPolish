/* ensemble/train (spec §8.2): submit an ensemble_train SLURM job — add new
   members with per-member knobs (loss, depth, single or MULTI-knee:
   asinh_knees / output_knee / knee_loss, noise aug, bootstrap, ICNR,
   seed; the star regime is the workspace's, never a per-row knob), continue members (by extra steps or up to N; TIMEOUT members one
   click away) or fork one (psnr / loss track). Presets: the 09-20+ recipe and
   "repeat last batch"; clone any past job. The run geometry lives here. A
   live preview shows the member names the submit will allocate (tombstones
   never reused) and the exact train_ensemble.py command. URL: ?mode=,
   ?members= (continue), ?member= (fork), ?from=<jobid> (clone). */
import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, apiPost } from "../../../api/client";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import { useFasrcStatus } from "../../../app/status";
import { useUrlState } from "../../../hooks/useUrlState";
import { LOSS_COLOR } from "../../../colors";
import {
  Badge, Button, Callout, Card, CardBody, CardHead, Checkbox, CopyButton, Field, IconButton, Input,
  NumberField, Page, Segmented, Select, Switch, Tooltip, confirm, toast,
} from "../../../ui";
import { useMembers, useMode, useTrainingJobs, type MemberRow, type TrainingJob } from "../api";
import { BarGroup, EnsBar } from "../common";
import { kneeText, memberName, memberNumber, stepsText } from "../model";
import {
  DEFAULT_MULTI_KNEES, KNEE_LOSSES, LOSSES, buildParams, defaultForm, formFromJob, jobRegime, lastBatch, newRow,
  recipeSummary, validate, type SpecRow, type TrainForm, type TrainMode,
} from "../trainModel";
import "../ensemble.css";

type Preview = { ok: boolean; mode: string; member_names: string[]; count: number; array: { tasks: number; max_parallel: number } | null;
  command: string[]; command_text: string; base_seed: string | number | null; star_prior: boolean };
type StepInfo = { step_id: string; defaults: { partition: string; n_cpus: number; n_gpus: number; memory: string; time_limit: string }; fixed_gpus?: number | null };
type Resources = { n_cpus: string; memory: string; time_limit: string };

function SpecRowEditor({ row, i, mode, onChange, onRemove, onDuplicate, canRemove }: {
  row: SpecRow; i: number; mode: TrainMode; onChange: (p: Partial<SpecRow>) => void; onRemove: () => void;
  onDuplicate: () => void; canRemove: boolean;
}) {
  const multi = row.kneeMode === "multi";
  return (
    <div className="ens-spec__row" style={{ borderLeft: `3px solid ${LOSS_COLOR[row.loss]}` }}>
      <span className="ens-spec__idx">#{i + 1}</span>
      <Field label="Loss"><Select value={row.loss} onChange={(v) => onChange({ loss: v })} options={LOSSES.map((v) => ({ value: v, label: v.toUpperCase() }))} /></Field>
      {mode === "add" && <NumberField label="Depth" value={row.blocks} onChange={(v) => onChange({ blocks: v })} min={4} max={64} />}
      <Field label="Knee" hint="single: one asinh knee (100 = the per-band default). multi: stretch at every knee at once.">
        <Select value={row.kneeMode} onChange={(v) => onChange({ kneeMode: v as SpecRow["kneeMode"] })}
          options={[{ value: "single", label: "single" }, { value: "multi", label: "multi-knee" }]} />
      </Field>
      {multi ? <>
        <Field label="Knees [e⁻]" className="ens-spec__wide"><Input value={row.knees} onChange={(v) => onChange({ knees: v })} placeholder={DEFAULT_MULTI_KNEES} /></Field>
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
      <div className="ens-row" style={{ alignSelf: "center" }}>
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
    <div className="ens-stack" style={{ gap: "var(--s2)" }}>
      <div className="ens-row">
        <Badge tone="accent">{picked.length} selected</Badge>
        <Button size="sm" variant="ghost" disabled={!timeouts.length} onClick={() => onChange(timeouts)}>TIMEOUT members ({timeouts.length})</Button>
        <Button size="sm" variant="ghost" onClick={() => onChange(rows.map((m) => m.name))}>all</Button>
        <Button size="sm" variant="ghost" disabled={!picked.length} onClick={() => onChange([])}>none</Button>
      </div>
      <div className="ens-picker" role="group" aria-label="Members to continue">
        {rows.map((m) => (
          <label key={m.name} className="ens-pick" data-on={set.has(m.name)} style={{ ["--sw" as string]: LOSS_COLOR[m.loss] }}>
            <span className="ens-pick__top">
              <span>#{memberNumber(m.name)}{m.timeout ? " ⚠" : ""}</span>
              <Checkbox checked={set.has(m.name)} aria-label={`Continue ${m.name}`}
                onChange={(on) => onChange(on ? [...picked, m.name] : picked.filter((x) => x !== m.name))} />
            </span>
            <span className="ens-pick__meta">{stepsText(m.step, m.target_steps)} · {m.loss} · {kneeText(m).text}</span>
          </label>
        ))}
      </div>
    </div>
  );
}

export default function Train() {
  const mode = useMode();
  const members = useMembers(mode);
  const jobs = useTrainingJobs();
  const steps = useResource<{ ssh_connected: boolean; steps: StepInfo[] }>("/api/fasrc/steps/status", [], { ttl: 60_000 });
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
  const [res, setRes] = useState<Resources | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState<{ jobid?: string; queued?: boolean } | null>(null);
  const cloned = useRef<string | null>(null);

  const step = steps.data?.steps.find((s) => s.step_id === "ensemble_train");
  useEffect(() => {
    if (step && !res) setRes({ n_cpus: String(step.defaults.n_cpus), memory: step.defaults.memory, time_limit: step.defaults.time_limit });
  }, [step, res]);

  const patch = (p: Partial<TrainForm>) => setForm((f) => ({ ...f, ...p }));
  const setMode = (m: TrainMode) => { patch({ mode: m }); setUrlMode(m); };
  const setGeometry = (p: Partial<TrainForm["geometry"]>) => setForm((f) => ({ ...f, geometry: { ...f.geometry, ...p } }));
  const setRow = (i: number, p: Partial<SpecRow>) => setForm((f) => ({ ...f, rows: f.rows.map((r, j) => (j === i ? { ...r, ...p } : r)) }));

  const clone = (job: TrainingJob) => {
    const f = formFromJob(job);
    setForm(f);
    setUrlMode(f.mode);
    setFrom(job.jobid);
    if (job.req_time_limit || job.req_memory) {
      setRes((r) => ({ n_cpus: job.req_cpus ? String(job.req_cpus) : r?.n_cpus ?? "4",
        memory: job.req_memory || r?.memory || "32G", time_limit: job.req_time_limit || r?.time_limit || "3:00:00" }));
    }
    const jr = jobRegime(job);
    if (jr !== mode && f.mode !== "continue") {
      toast.warning(`Cloned job ${job.jobid} trained ${jr} members`, { description: `This workspace submits ${mode} ones. Switch the workspace regime to repeat it as it was.` });
    } else toast.info(`Cloned job ${job.jobid}`, { description: recipeSummary(job) });
  };
  // ?from=<jobid> deep link → clone once the job list arrives
  useEffect(() => {
    if (!from || cloned.current === from) return;
    const job = jobs.data?.jobs.find((j) => j.jobid === from);
    if (job) { cloned.current = from; clone(job); }
  }, [from, jobs.data]); // eslint-disable-line react-hooks/exhaustive-deps

  const params = useMemo(() => buildParams(form, mode), [form, mode]);
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
  async function submit() {
    if (errors.length || !res) return;
    const what = form.mode === "continue" ? `continue ${form.members.length} member(s)` : `${form.rows.length} ${form.mode === "fork" ? "fork" : "new"} member(s): ${preview?.member_names.join(", ") ?? "…"}`;
    const ok = await confirm({ title: "Submit ensemble_train to SLURM?", message: `${what} · ${res.time_limit} per model on ${step?.defaults.partition ?? "gpu"}. It queues locally when another job is active.`, confirmLabel: "Submit" });
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
    { id: "train-recipe", label: "Train: reset to the standard recipe", group: "Train", run: () => { setForm({ ...defaultForm(), mode: form.mode }); setFrom(""); } },
    { id: "train-add-multi", label: "Train: add a multi-knee member row", group: "Train", run: () => patch({ rows: [...form.rows, newRow({ kneeMode: "multi" })] }) },
    { id: "train-submit", label: "Train: submit to SLURM", group: "Train", disabled: offline || !!errors.length, run: () => void submit() },
  ]);

  const activeRows = members.data?.members ?? [];
  const jobOptions = (jobs.data?.jobs ?? []).slice(0, 40).map((j) => ({
    value: j.jobid, label: `${j.jobid} · ${j.submitted_at?.slice(0, 10) ?? ""} · ${recipeSummary(j)}`, hint: j.state ?? "",
  }));
  const g = form.geometry;
  return (
    <Page>
      <EnsBar label="Train controls">
        <Segmented<TrainMode> size="sm" aria-label="Train mode" value={form.mode} onChange={setMode}
          options={[{ value: "add", label: "Add" }, { value: "continue", label: "Continue" }, { value: "fork", label: "Fork" }]} />
        <span className="ens-bar__sep" aria-hidden />
        <Tooltip content={last ? recipeSummary(last) : "No past batch in the job log"}>
          <span><Button size="sm" disabled={!last} onClick={() => last && clone(last)}>Repeat last batch</Button></span>
        </Tooltip>
        <BarGroup label="Clone">
          <Select searchable size="sm" aria-label="Clone a past job" value={from} placeholder="a past job…"
            onChange={(id) => { const j = jobs.data?.jobs.find((x) => x.jobid === id); if (j) clone(j); }} options={jobOptions} />
        </BarGroup>
        <span className="ens-bar__spacer" />
        <Tooltip content={form.mode === "continue" ? "Continued members keep their recorded regime." : form.mode === "fork"
          ? "A fork keeps its source member's regime." : `New members train ${mode === "starless" ? "starless (erase stars, clean target)" : "starfull (reconstruct stars)"}. Set by the workspace regime switch.`}>
          <span><Badge tone={mode === "starless" ? "warn" : undefined} dot>{mode}</Badge></span>
        </Tooltip>
        <Button size="sm" variant="ghost" icon="reset" onClick={() => { setForm({ ...defaultForm(), mode: form.mode }); setFrom(""); }}>Recipe</Button>
      </EnsBar>
      <div className="ens-stack">
        {form.mode === "continue" ? (
          <Card>
            <CardHead title="Continue members" sub="each member becomes one task of a capped job array" />
            <CardBody>
              <div className="ens-form" style={{ marginBottom: "var(--s3)" }}>
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
              right={<div className="ens-row">
                <Button size="sm" icon="plus" onClick={() => patch({ rows: [...form.rows, newRow(form.rows[form.rows.length - 1])] })}>member</Button>
                <Button size="sm" icon="plus" onClick={() => patch({ rows: [...form.rows, newRow({ kneeMode: "multi" })] })}>multi-knee</Button>
              </div>} />
            <CardBody>
              <div className="ens-form" style={{ marginBottom: "var(--s3)" }}>
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
              <div className="ens-spec">
                {form.rows.map((r, i) => (
                  <SpecRowEditor key={i} row={r} i={i} mode={form.mode} onChange={(p) => setRow(i, p)} canRemove={form.rows.length > 1}
                    onRemove={() => patch({ rows: form.rows.filter((_, j) => j !== i) })}
                    onDuplicate={() => patch({ rows: [...form.rows.slice(0, i + 1), newRow(r), ...form.rows.slice(i + 1)] })} />
                ))}
              </div>
            </CardBody>
          </Card>
        )}
        <Card>
          <CardHead title="Run" sub="run-wide knobs and the live forward model's geometry" />
          <CardBody>
            <div className="ens-form">
              <NumberField label="Models at once" value={form.arrayMaxParallel} onChange={(v) => patch({ arrayMaxParallel: v })} min={1} />
              <NumberField label="Base seed" value={form.baseSeed} onChange={(v) => patch({ baseSeed: v })} placeholder="drawn at submit" />
              <NumberField label="Evaluate every" value={form.evaluateEvery} onChange={(v) => patch({ evaluateEvery: v })} placeholder="default" unit="steps" />
              <NumberField label="Batch" value={g.batch_size} onChange={(v) => setGeometry({ batch_size: v })} min={1} max={64} />
              <NumberField label="HR example side" value={g.hr_crop_size} onChange={(v) => setGeometry({ hr_crop_size: v })} min={2} max={508} step={2} unit="px" />
              <NumberField label="Examples / field" value={g.crops_per_field} onChange={(v) => setGeometry({ crops_per_field: v })} min={1} max={25} />
              <NumberField label="PSF clusters / member" value={g.psf_subset} onChange={(v) => setGeometry({ psf_subset: v })} min={1} />
              <NumberField label="PSF warp prob" value={g.psf_warp_prob} onChange={(v) => setGeometry({ psf_warp_prob: v })} min={0} max={1} step={0.05} />
              <NumberField label="PSF warp α max" value={g.psf_warp_alpha_max} onChange={(v) => setGeometry({ psf_warp_alpha_max: v })} min={0} max={100} />
              <NumberField label="PSF warp σ" value={g.psf_warp_sigma} onChange={(v) => setGeometry({ psf_warp_sigma: v })} min={0.1} max={100} step={0.5} unit="HR px" />
              <NumberField label="Saturation mask prob" value={g.saturation_mask_prob} onChange={(v) => setGeometry({ saturation_mask_prob: v })} min={0} max={0.5} step={0.05} />
              <NumberField label="Target PSF FWHM" value={g.target_psf_fwhm_arcsec} onChange={(v) => setGeometry({ target_psf_fwhm_arcsec: v })} min={0} max={2} step={0.001} unit="″" />
              <Switch checked={g.forward_onthefly} onChange={(v) => setGeometry({ forward_onthefly: v })}>live forward model</Switch>
            </div>
          </CardBody>
        </Card>
        <Card>
          <CardHead title="Submit" sub={step ? `${step.defaults.partition} partition · ${step.fixed_gpus ?? 1} GPU per model` : "loading the step…"} />
          <CardBody>
            <div className="ens-stack">
              {res && (
                <div className="ens-form">
                  <NumberField label="CPUs / model" value={res.n_cpus} onChange={(v) => setRes({ ...res, n_cpus: v })} min={1} />
                  <Field label="Memory / model"><Input value={res.memory} onChange={(v) => setRes({ ...res, memory: v })} /></Field>
                  <Field label="Time limit / model" hint="The 70k-step recipe takes 2.5–3 h on the gpu partition."><Input value={res.time_limit} onChange={(v) => setRes({ ...res, time_limit: v })} /></Field>
                </div>
              )}
              {errors.length > 0 && <Callout tone="warn" title="Fix before submitting">{errors.join(" · ")}</Callout>}
              {previewError && <Callout tone="bad" title="The submit would be refused"><span className="ens-mono">{previewError}</span></Callout>}
              {preview && (
                <div className="ens-stack" style={{ gap: "var(--s2)" }}>
                  <div className="ens-row">
                    <span className="ens-bar__label">{preview.mode === "continue" ? "Continues" : "New members"}</span>
                    <div className="ens-names">{preview.member_names.map((n) => <Badge key={n} tone={preview.mode === "continue" ? undefined : "accent"}>{n}</Badge>)}</div>
                    {preview.array && <span className="ens-faint">array of {preview.array.tasks}, {preview.array.max_parallel} at once</span>}
                    <span className="ens-faint">seed {preview.base_seed ?? "drawn at submit"}</span>
                  </div>
                  <div className="ens-row" style={{ alignItems: "flex-start" }}>
                    <pre className="ens-cmd" style={{ flex: 1 }} aria-label="Final command">{preview.command_text}</pre>
                    <CopyButton value={preview.command_text} label="Copy the command" />
                  </div>
                </div>
              )}
              <div className="ens-row">
                <Button variant="primary" icon="server" loading={submitting} disabled={offline || !!errors.length || !!previewError || !res}
                  onClick={() => void submit()}>Submit to SLURM</Button>
                {offline && <span className="ens-warn">FASRC offline{fasrc?.last_error ? `: ${fasrc.last_error}` : ""}</span>}
                {submitted?.jobid && <Button size="sm" variant="ghost" onClick={() => openInspector({ kind: "job", id: `slurm/${submitted.jobid}` })}>Monitor job {submitted.jobid}</Button>}
                {submitted?.queued && <Badge tone="info">queued locally</Badge>}
              </div>
            </div>
          </CardBody>
        </Card>
      </div>
    </Page>
  );
}
