/* "Freeze study…" — the dialog that freezes the whole (starfull) ensemble
 * into an immutable model study (spec 2026-09-28 "The freeze dialog"; opened
 * from Models › Leaderboard, Figures › Studies and the palette). Three steps:
 *   1. what will be frozen: the ensemble and each numbers block's state (a
 *      stale block names its reason and where it is fixed);
 *   2. the field gallery (0–10), grouped by kind, sizes as upper bounds;
 *   3. name, note and a summary, then "Freeze".
 * Only GETs run until "Freeze" (the candidates and their thumbnails): the
 * POST /api/studies of the last step is the one write. Every refusal (400,
 * 409 stale / busy, 503 FASRC offline, 507 disk) is said in plain words.
 * Mount it only while open: `{open && <FreezeStudyDialog … onClose />}`. */
import { Fragment, useEffect, useId, useMemo, useRef, useState, type ReactNode } from "react";
import { Link, useNavigate } from "react-router-dom";
import { apiPost } from "../../api/client";
import { refreshJobsFeed, useJob, useJobsStore } from "../../api/jobs";
import { invalidate, useResource } from "../../api/query";
import { formatBytes, formatCount, formatRelative } from "../../format";
import {
  Button, Callout, Caption, Checkbox, Dialog, Field, FactsList, Input, JobProgress, Num, Skeleton, SummaryLine, Textarea,
} from "../../ui";
import {
  FREEZE_JOB_KEY, KIND_HINT, KIND_TITLE, MAX_FIELDS, STUDY_MODE, blockFix, candidatesUrl, counterText, groupFields, refusalMessage, studyPath, togglePick,
  uploadBytes, type CandidateBlock, type CandidateField, type Candidates, type FieldGroup, type FreezeReply,
} from "./studyFreeze";
import "./freezeStudy.css";

type Step = "what" | "fields" | "confirm" | "running";
type Refusal = ReturnType<typeof refusalMessage>;

function BlockRow({ block, onFix }: { block: CandidateBlock; onFix: (to: string) => void }) {
  const fix = blockFix(block);
  const tone = block.state === "current" ? "good" : block.state === "stale" ? "warn" : "neutral";
  return (
    <li className="sfz-block" data-tone={tone}>
      <span className="sfz-dot" aria-hidden />
      <span className="sfz-block__text">
        <span className="sfz-block__title">{block.title}</span>
        {block.state !== "current" && <span className="sfz-block__state"> · {block.state}</span>}
        <span className="sfz-block__detail">{block.detail}</span>
      </span>
      {fix && <Button size="sm" variant="ghost" onClick={() => onFix(fix.to)}>{fix.label}</Button>}
    </li>
  );
}

/** The ids of the visible text that says why a field cannot be picked. */
type ReasonIds = { group: string; offline: string; limit: string };

function FieldItem({ field, picked, locked, offline, sharedReason, ids, onToggle }: {
  field: CandidateField; picked: boolean; locked: boolean; offline: boolean; sharedReason: string | null; ids: ReasonIds; onToggle: () => void;
}) {
  const [thumbFailed, setThumbFailed] = useState(false);
  const own = useId();
  const disabled = !field.available || offline || (locked && !picked);
  const why = !field.available ? field.reason : null;
  const ownReason = why && why !== sharedReason ? why : null;
  const describedBy = [
    // Only ids of text that is on the page: its own reason, else the group's shared one.
    !field.available && (ownReason ? own : sharedReason ? ids.group : null),
    offline && ids.offline,
    field.available && !offline && locked && !picked && ids.limit,
  ].filter(Boolean).join(" ");
  return (
    <li className="sfz-field" data-disabled={disabled || undefined} data-picked={picked || undefined}>
      {field.thumb_url && !thumbFailed
        ? <img className="sfz-field__thumb" src={field.thumb_url} alt="" loading="lazy" decoding="async" width={96} height={96} onError={() => setThumbFailed(true)} />
        : <span className="sfz-field__thumb sfz-field__thumb--empty" aria-hidden />}
      <Checkbox checked={picked} disabled={disabled} onChange={onToggle} className="sfz-field__check" aria-describedby={describedBy}>
        <span className="sfz-field__label">{field.label}</span>
      </Checkbox>
      <span className="sfz-field__size">≤ {formatBytes(field.bytes)}</span>
      {ownReason && <span id={own} className="sfz-field__reason">{ownReason}</span>}
    </li>
  );
}

function FieldGallery({ groups, picked, max, offline, offlineId, limitId, onToggle }: {
  groups: FieldGroup[]; picked: string[]; max: number; offline: boolean; offlineId: string; limitId: string; onToggle: (fid: string) => void;
}) {
  const locked = picked.length >= max;
  const base = useId();
  if (!groups.length) return <p className="sfz-note">No field has SR for every member under this setup.</p>;
  return (
    <div className="sfz-gallery">
      {groups.map((g) => (
        <section key={g.kind} className="sfz-group" aria-labelledby={`sfz-g-${g.kind}`}>
          <h3 id={`sfz-g-${g.kind}`} className="sfz-group__title">
            {KIND_TITLE[g.kind]} <span className="sfz-group__count">{formatCount(g.available)} of {formatCount(g.fields.length)} available</span>
          </h3>
          <p className="sfz-group__hint">{KIND_HINT[g.kind]}</p>
          {g.sharedReason && <p id={`${base}-${g.kind}`} className="sfz-group__reason">Unavailable: {g.sharedReason}</p>}
          {g.available === 0 && !g.sharedReason && (g.fields.length <= 6 ? (
            <ul className="sfz-group__reasons" aria-label={`Why no ${KIND_TITLE[g.kind].toLowerCase()} can be attached`}>
              {g.fields.map((f) => <li key={f.fid}><span className="sfz-group__who">{f.label}</span>: {f.reason ?? "no reason given"}</li>)}
            </ul>
          ) : (
            <p className="sfz-group__reason">Unavailable: {[...new Set(g.fields.map((f) => f.reason ?? "no reason given"))].join("; ")}</p>
          ))}
          {/* A group with nothing to pick is its header and reason only. */}
          {g.available > 0 && (
            <ul className="sfz-fields">
              {g.fields.map((f) => (
                <FieldItem key={f.fid} field={f} picked={picked.includes(f.fid)} locked={locked} offline={offline}
                  sharedReason={g.sharedReason} ids={{ group: `${base}-${g.kind}`, offline: offlineId, limit: limitId }}
                  onToggle={() => onToggle(f.fid)} />
              ))}
            </ul>
          )}
        </section>
      ))}
    </div>
  );
}

export function FreezeStudyDialog({ onClose }: { onClose: () => void }) {
  const navigate = useNavigate();
  const job = useJob(FREEZE_JOB_KEY);
  // A freeze started earlier (this session) and still running re-attaches.
  const [step, setStep] = useState<Step>(() => (job.job?.status === "running" ? "running" : "what"));
  const [picked, setPicked] = useState<string[]>([]);
  const [name, setName] = useState("");
  const [nameTouched, setNameTouched] = useState(false);
  const [note, setNote] = useState("");
  const [posting, setPosting] = useState(false);
  const [refusal, setRefusal] = useState<Refusal | null>(null);
  const [studyId, setStudyId] = useState<string | null>(null);

  useEffect(() => {
    // A finished freeze of an earlier opening is not this dialog's job.
    if (step !== "running" && job.job && job.job.status !== "running") useJobsStore.getState().forgetKey(FREEZE_JOB_KEY);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const cand = useResource<Candidates>(step === "running" ? null : candidatesUrl(), [], { ttl: 15_000 });
  const c = cand.data;
  const max = c?.max_fields ?? MAX_FIELDS;
  const offline = c ? !c.fasrc_connected : false;
  const groups = useMemo(() => groupFields(c?.fields ?? []), [c]);
  const bytes = uploadBytes(c?.fields ?? [], picked);
  // Re-attached to a running freeze: its study id comes from the listing once,
  // and is kept, so "Open the study" survives the job finishing.
  const listing = useResource<{ freezing: { job_id: string; study_id: string } | null }>(
    step === "running" && !studyId ? "/api/studies" : null, [step, studyId], { ttl: 2_000 });
  const listedStudy = listing.data?.freezing?.job_id === job.job?.job_id ? listing.data?.freezing?.study_id ?? null : null;
  useEffect(() => { if (!studyId && listedStudy) setStudyId(listedStudy); }, [studyId, listedStudy]);
  const resultStudy = (job.job?.result as { study_id?: string } | null | undefined)?.study_id ?? null;
  const runningStudy = studyId ?? resultStudy;
  const finished = job.job != null && job.job.status !== "running";
  const ids = { offline: useId(), limit: useId() };

  // Each step lands focus on its first control: the gallery's first free
  // field, the Name input; the running step its body (never the job's
  // Cancel, which a second Enter would press); else the step's body.
  const bodyRef = useRef<HTMLDivElement>(null);
  const firstStep = useRef(true);
  useEffect(() => {
    if (firstStep.current) { firstStep.current = false; return; }
    const root = bodyRef.current;
    if (!root) return;
    const order = step === "fields" ? [".sfz-fields input:not(:disabled)", ".sfz-counter"]
      : step === "confirm" ? ["input"] : step === "running" ? [] : ["input:not(:disabled), button:not(:disabled)"];
    const target = order.map((sel) => root.querySelector<HTMLElement>(sel)).find((el) => el != null);
    (target ?? root).focus();
  }, [step]);

  useEffect(() => {
    if (finished) invalidate("/api/studies");
  }, [finished]);

  const leave = (to: string) => { onClose(); navigate(to); };

  async function freeze() {
    setNameTouched(true);
    if (!name.trim() || posting) return;
    setPosting(true);
    setRefusal(null);
    try {
      const r = await apiPost<FreezeReply>("/api/studies", { name: name.trim(), note, mode: STUDY_MODE, fields: picked.join(",") });
      if (!r.ok || !r.job_id) {
        setRefusal({ title: "The freeze did not start", text: r.error ?? "The server gave no job.", retry: null });
        return;
      }
      useJobsStore.getState().register(r.job_id, FREEZE_JOB_KEY);
      void refreshJobsFeed();
      setStudyId(r.study_id ?? null);
      setStep("running");
      invalidate("/api/studies");
    } catch (e) {
      const m = refusalMessage(e);
      setRefusal(m);
      if (m.retry === "candidates") cand.reload();
    } finally {
      setPosting(false);
    }
  }

  const gate = c?.ensemble.gate;
  const stale = c?.ensemble.blocks.filter((b) => b.state !== "current") ?? [];

  let body: ReactNode;
  let footer: ReactNode;
  const cancel = <Button variant="ghost" onClick={onClose}>Cancel</Button>;

  if (step === "running") {
    body = (
      <div className="sfz-stack">
        <p className="sfz-note">
          Freezing the ensemble. The numbers step takes about 2–3 minutes; attached fields then upload to holylabs one
          product at a time. You can close this dialog: the job keeps running in the job tray.
        </p>
        <JobProgress job={job.job} error={job.error} />
        {job.job?.status === "done" && <Callout tone="good" title="Study frozen">Its numbers and fields are sealed; nothing in it changes when members do.</Callout>}
        {job.job && job.job.status !== "running" && job.job.status !== "done" && (
          <Callout tone="warn" title="The study is incomplete">Open it in Figures › Studies to resume or delete it.</Callout>
        )}
      </div>
    );
    footer = (
      <>
        <Button variant="ghost" onClick={onClose}>Close</Button>
        {runningStudy && (
          <Button variant={finished ? "primary" : "default"} asChild>
            <Link to={studyPath(runningStudy)} onClick={onClose}>Open the study</Link>
          </Button>
        )}
      </>
    );
  } else if (!c) {
    body = cand.error
      ? <Callout tone="bad" title="Could not read what would be frozen" action={<Button size="sm" onClick={cand.reload}>Retry</Button>}>{cand.error.message}</Callout>
      : <Skeleton lines={6} />;
    footer = cancel;
  } else if (step === "what") {
    body = (
      <div className="sfz-stack">
        <SummaryLine>
          <Num>{formatCount(c.ensemble.n_members)}</Num> member{c.ensemble.n_members === 1 ? "" : "s"}
          {gate?.available ? <> · production gate <Num>{gate.name ?? "—"}</Num></> : " · no production gate"}
          {" · "}{c.ensemble.evaluated_at ? <>evaluated <Num>{formatRelative(c.ensemble.evaluated_at)}</Num></> : "not evaluated"}
        </SummaryLine>
        <ul className="sfz-blocks" aria-label="Numbers blocks">
          {c.ensemble.blocks.map((b) => <BlockRow key={b.id} block={b} onFix={leave} />)}
        </ul>
        {!c.can_freeze && (
          <Callout tone="bad" title="Cannot freeze now">{c.blocking ?? "Nothing to freeze."}</Callout>
        )}
        {c.can_freeze && stale.length > 0 && (
          <Callout tone="warn" title={`${stale.length} of ${c.ensemble.blocks.length} blocks ${stale.length === 1 ? "is" : "are"} not current`}>
            They are frozen as they are now and stay that way in the study. To freeze them current, fix them first (the button on each row), then open this dialog again.
          </Callout>
        )}
        <Caption>
          The numbers (≈ {formatBytes(c.ensemble.numbers_bytes)}) are written locally and mirrored to holylabs; attached fields go to holylabs only.
          Nothing is written until you press Freeze.
        </Caption>
      </div>
    );
    footer = (
      <>
        {cancel}
        <Button disabled={!c.can_freeze} onClick={() => { setPicked([]); setStep("confirm"); }}>Freeze without fields</Button>
        <Button variant="primary" disabled={!c.can_freeze} onClick={() => setStep("fields")}>Choose fields…</Button>
      </>
    );
  } else if (step === "fields") {
    body = (
      <div className="sfz-stack">
        <p className="sfz-counter" aria-live="polite" tabIndex={-1}>{counterText(picked.length, max, bytes)}</p>
        <Caption>Sizes are upper bounds (uncompressed); the study records the compressed sizes.</Caption>
        {offline && (
          <Callout tone="info" title="FASRC is not connected">
            <span id={ids.offline}>{c.fields_note ?? "Fields are stored on holylabs, so attaching them needs the connection."} You can still freeze without fields.</span>
          </Callout>
        )}
        {picked.length >= max && <p id={ids.limit} className="sfz-note" role="status">At most {max} fields: unpick one to choose another.</p>}
        <FieldGallery groups={groups} picked={picked} max={max} offline={offline} offlineId={ids.offline} limitId={ids.limit}
          onToggle={(fid) => setPicked((p) => togglePick(p, fid, max))} />
      </div>
    );
    footer = (
      <>
        <Button variant="ghost" onClick={() => setStep("what")}>Back</Button>
        {cancel}
        <Button variant="primary" onClick={() => setStep("confirm")}>
          {picked.length ? `Continue with ${picked.length} field${picked.length === 1 ? "" : "s"}` : "Continue without fields"}
        </Button>
      </>
    );
  } else {
    const nameError = nameTouched && !name.trim() ? "A study needs a name" : undefined;
    body = (
      <div className="sfz-stack">
        <Field label="Name" description="Required. Shown in Figures › Studies and on every exported figure." error={nameError}>
          <Input value={name} onChange={setName} onEnter={() => void freeze()} placeholder="Loss and knee choice, September" />
        </Field>
        <Field label="Note" description="Optional: why this study, what it should show. Editable later.">
          <Textarea value={note} onChange={setNote} rows={3} />
        </Field>
        <FactsList title="Will be written" facts={[
          { label: "Members", value: formatCount(c.ensemble.n_members) },
          { label: "Production gate", value: gate?.available ? gate.name ?? "—" : "none" },
          { label: "Numbers", value: `≈ ${formatBytes(c.ensemble.numbers_bytes)}`, hint: "Written locally, mirrored to holylabs" },
          { label: "Fields", value: picked.length ? `${picked.length}, ≤ ${formatBytes(bytes)}` : "none", hint: picked.length ? "Uploaded to holylabs (upper bound)" : undefined },
          stale.length > 0 && { label: "Not current", value: stale.map((b) => b.title).join(", "), tone: "warn" },
        ]} />
        {refusal && (
          <Callout tone="bad" title={refusal.title} action={
            refusal.retry === "fields" && picked.length ? <Button size="sm" onClick={() => { setPicked([]); setRefusal(null); }}>Drop the fields</Button>
              : refusal.retry === "candidates" ? <Button size="sm" onClick={() => { setRefusal(null); setStep("what"); }}>Check again</Button> : undefined
          }>{refusal.text}</Callout>
        )}
        <Caption>The numbers step takes about 2–3 minutes; you can close the dialog while it runs.</Caption>
      </div>
    );
    footer = (
      <>
        <Button variant="ghost" onClick={() => setStep(picked.length ? "fields" : "what")}>Back</Button>
        {cancel}
        <Button variant="primary" loading={posting} disabled={!name.trim()} onClick={() => void freeze()}>Freeze</Button>
      </>
    );
  }

  return (
    <Dialog open onOpenChange={(v) => { if (!v) onClose(); }} size="lg" className="sfz"
      title={step === "fields" ? "Choose fields" : step === "confirm" ? "Name the study" : step === "running" ? "Freezing the study" : "Freeze a study"}
      description={step === "what" ? "Freeze the whole ensemble — every active member, the production gate and the combiner comparison — into a study that outlives the members." : undefined}
      footer={<Fragment key={step}>{footer}</Fragment>}>
      <div key={step} ref={bodyRef} tabIndex={-1} className="sfz-step">{body}</div>
    </Dialog>
  );
}
