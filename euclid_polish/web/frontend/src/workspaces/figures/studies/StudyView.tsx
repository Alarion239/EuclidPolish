/* One study (`/figures/studies?study=<id>`): what was frozen, the selection
 * (members, grouping, colour, named selections saved in the study's
 * sidecar), the six charts from the frozen numbers with their exports, and
 * the attached fields. Reads only on open; every write is a button. */
import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { apiPost } from "../../../api/client";
import { useJob } from "../../../api/jobs";
import { invalidate } from "../../../api/query";
import { usePageActions } from "../../../app/palette";
import { formatBytes, formatCount, formatDate, formatRelative } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { useResolvedTheme } from "../../../state/prefs";
import {
  Button, Callout, Details, FactsList, Field, IconButton, Input, JobProgress, MultiSelect, Num, Popover, Section, Select, Skeleton,
  SummaryLine, Tabs, Textarea, Toolbar, ToolbarGroup, toast,
} from "../../../ui";
import { FREEZE_JOB_KEY } from "../../shared/studyFreeze";
import { deleteStudy, resumeStudy } from "./actions";
import { studyUrl, useStudies, useStudy, type SavedSelection, type StudyDetail, type StudyMember } from "./api";
import { FieldsSection } from "./FieldsSection";
import { CHART_COMPONENT, makePalette, type ChartProps } from "./StudyCharts";
import { CHARTS, CHART_TAB, CHART_TITLE, GROUP_LABEL, groupNames, memberKey, memberName, type Chart, type ChartSelection } from "./model";

const LIST_PATH = "/figures/studies";

const commitText = (c: StudyDetail["manifest"]["commit"]) => {
  if (!c) return "—";
  if (typeof c === "string") return c;
  return `${c.short ?? c.hash?.slice(0, 8) ?? "—"}${c.dirty ? " (uncommitted changes)" : ""}`;
};

function NoteEditor({ detail, onSaved }: { detail: StudyDetail; onSaved: () => void }) {
  const [text, setText] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const value = text ?? detail.note ?? "";
  const changed = text != null && text !== (detail.note ?? "");
  async function save() {
    setSaving(true);
    try {
      await apiPost(`${studyUrl(detail.study.id)}/note`, { note: value });
      toast.success("Note saved");
      setText(null);
      onSaved();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }
  return (
    <div className="stu-note">
      <Field label="Note" description="The study's only editable text besides its named selections; the frozen numbers never change.">
        <Textarea value={value} onChange={setText} rows={2} />
      </Field>
      {changed && (
        <div className="stu-note__actions">
          <Button size="sm" variant="ghost" onClick={() => setText(null)}>Discard</Button>
          <Button size="sm" variant="primary" loading={saving} onClick={() => void save()}>Save note</Button>
        </div>
      )}
    </div>
  );
}

function SaveSelection({ detail, current, onSaved }: {
  detail: StudyDetail; current: { members: readonly string[] | null; group: string | null }; onSaved: () => void;
}) {
  const [name, setName] = useState("");
  const [saving, setSaving] = useState(false);
  const saved = detail.selections ?? [];
  async function write(next: SavedSelection[], done: string) {
    setSaving(true);
    try {
      await apiPost(`${studyUrl(detail.study.id)}/selections`, { selections: next }, { json: true });
      toast.success(done);
      setName("");
      onSaved();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }
  const trimmed = name.trim();
  const save = () => {
    if (!trimmed) return;
    const entry: SavedSelection = { name: trimmed, ...(current.members?.length ? { members: [...current.members] } : {}), ...(current.group ? { group: current.group } : {}) };
    void write([...saved.filter((s) => s.name !== trimmed), entry], `Saved the selection “${trimmed}”`);
  };
  return (
    <Popover label="Save selection" trigger={<Button size="sm">Save selection…</Button>}>
      <div className="stu-save">
        <Field label="Name" description={saved.some((s) => s.name === trimmed) ? "Replaces the saved selection of that name." : "Saved in the study (its selections sidecar)."}>
          <Input value={name} onChange={setName} onEnter={save} placeholder="L1 vs L2 at knee 100" />
        </Field>
        <Button size="sm" variant="primary" disabled={!trimmed} loading={saving} onClick={save}>Save</Button>
        {saved.length > 0 && (
          <ul className="stu-save__list" aria-label="Saved selections">
            {saved.map((s) => (
              <li key={s.name}>
                <span>{s.name}</span>
                <IconButton size="sm" icon="close" label={`Delete the saved selection ${s.name}`}
                  onClick={() => void write(saved.filter((x) => x.name !== s.name), `Deleted the selection “${s.name}”`)} />
              </li>
            ))}
          </ul>
        )}
      </div>
    </Popover>
  );
}

function Loaded({ detail, reload }: { detail: StudyDetail; reload: () => void }) {
  const m = detail.manifest;
  const all = useMemo<StudyMember[]>(() => m.ensemble?.members ?? [], [m]);
  const labels = useMemo(() => all.map((x) => x.label), [all]);
  const [chartRaw, setChart] = useUrlState("chart", "knee");
  const chart: Chart = (CHARTS as readonly string[]).includes(chartRaw) ? (chartRaw as Chart) : "knee";
  const [groupRaw, setGroup] = useUrlState("group", "");
  const [colourRaw, setColour] = useUrlState("colour", "loss");
  const [membersRaw, setMembers] = useUrlState<string[]>("members", []);
  const [reference, setReference] = useUrlState("ref", "");
  const [source, setSource] = useUrlState("src", "");
  const [metric, setMetric] = useUrlState("metric", "");
  const [experiment, setExperiment] = useUrlState("exp", "");
  const [view, setView] = useUrlState<"absolute" | "relative">("view", "absolute");
  const [dpi, setDpi] = useUrlState("dpi", 300);
  const freezeJob = useJob(FREEZE_JOB_KEY);
  const navigate = useNavigate();
  const listing = useStudies();
  const freezing = listing.data?.freezing ?? null;
  const [resumedHere, setResumedHere] = useState(false);
  // The freeze job's progress belongs here only when it freezes THIS study.
  const freezingHere = resumedHere || freezing?.study_id === detail.study.id;
  const resumeBlocked = freezing && freezing.study_id !== detail.study.id ? "Another study is being frozen; resume this one when it finishes."
    : freezeJob.busy && freezingHere ? "Its freeze is running (below)." : null;
  const deleteBlocked = freezing?.study_id === detail.study.id || (freezingHere && freezeJob.busy) ? "It is being frozen: cancel the freeze to delete it." : null;

  const groupFields = detail.group_fields ?? [];
  const group = groupFields.includes(groupRaw) ? groupRaw : "";
  const colourBy = groupFields.includes(colourRaw) ? colourRaw : "loss";
  const picked = membersRaw.filter((l) => labels.includes(l));
  const members = useMemo(() => (picked.length ? all.filter((x) => picked.includes(x.label)) : all), [picked.join(","), all]); // eslint-disable-line react-hooks/exhaustive-deps
  // One group → colour map for every chart (StudyCharts makePalette).
  const theme = useResolvedTheme();
  const palette = useMemo(() => makePalette(members), [members, theme]); // eslint-disable-line react-hooks/exhaustive-deps
  const pickMembers = (next: string[]) => {
    setMembers(next);
    // A member reference outside the new subset is dropped (the backend would refuse it).
    if (reference && labels.includes(reference) && next.length && !next.includes(reference)) setReference("");
    // So is a group reference whose group has no member left in the subset.
    if (reference.startsWith("group:") && group) {
      const kept = next.length ? all.filter((x) => next.includes(x.label)) : all;
      if (!groupNames(kept, group).includes(reference.slice("group:".length))) setReference("");
    }
  };
  const sel = useMemo<ChartSelection>(() => ({
    members: picked.length && picked.length < labels.length ? picked : null, group: group || null,
    reference: reference || null, source: source || null, metric: metric || null, experiment: experiment || null,
  }), [picked.join(","), labels.length, group, reference, source, metric, experiment]); // eslint-disable-line react-hooks/exhaustive-deps

  const set = (patch: Partial<ChartSelection>) => {
    if ("reference" in patch) setReference(patch.reference ?? "");
    if ("source" in patch) setSource(patch.source ?? "");
    if ("metric" in patch) setMetric(patch.metric ?? "");
    if ("experiment" in patch) setExperiment(patch.experiment ?? "");
  };
  const applySaved = (name: string) => {
    const s = detail.selections.find((x) => x.name === name);
    if (!s) return;
    setMembers((s.members ?? []).filter((l) => labels.includes(l)));
    setGroup(s.group ?? "");
    setReference("");
  };

  usePageActions(CHARTS.map((c) => ({ id: `study-chart-${c}`, label: `Study: show ${CHART_TITLE[c]}`, group: "Studies", run: () => setChart(c) })));

  const gate = m.gate ?? {};
  const nKneeFields = detail.numbers?.knee_psnr?.fields?.length ?? m.knee_fields?.length ?? null;
  const fieldsBytes = detail.study.fields_bytes;
  const groupOptions = [{ value: "", label: "each member" }, ...groupFields.map((f) => ({ value: f, label: `${GROUP_LABEL[f] ?? f} (${groupNames(members, f).length})` }))];
  const chartProps: ChartProps = {
    detail, chart, sel, members, colourBy, palette, dpi, setDpi, set, view, setView,
  };
  const ChartBody = CHART_COMPONENT[chart];
  const complete = detail.study.state === "complete";

  return (
    <div className="stu-stack">
      <div className="stu-head">
        <Button size="sm" variant="ghost" icon="chevronLeft" asChild><Link to={LIST_PATH}>All studies</Link></Button>
        <h2 className="stu-title">{detail.study.name}</h2>
        <span className="stu-head__right">
          {deleteBlocked && <span id="stu-delete-why" className="stu-reason">{deleteBlocked}</span>}
          <Button size="sm" variant="ghost" disabled={!!deleteBlocked} aria-describedby={deleteBlocked ? "stu-delete-why" : undefined}
            onClick={() => void deleteStudy(detail.study).then((gone) => { if (gone) navigate(LIST_PATH); })}>Delete study…</Button>
        </span>
      </div>
      <SummaryLine>
        <Num>{formatCount(all.length)}</Num> {m.regime ?? ""} members{gate.name ? <> and the production gate <Num>{gate.name}</Num></> : ""}, frozen{" "}
        <Num>{formatRelative(m.completed ?? m.created)}</Num>{nKneeFields != null ? <> on <Num>{formatCount(nKneeFields)}</Num> test fields</> : null}
        {detail.fields.length ? <>, with <Num>{formatCount(detail.fields.length)}</Num> attached field{detail.fields.length === 1 ? "" : "s"}</> : null}.
      </SummaryLine>
      {(m.warnings?.length ?? 0) > 0 && (
        <Callout tone="warn" title={`Frozen with ${m.warnings!.length} block${m.warnings!.length === 1 ? "" : "s"} not current`}>
          <ul className="stu-warnings">{m.warnings!.map((w) => <li key={w}>{w}</li>)}</ul>
        </Callout>
      )}
      {!complete && (
        <Callout tone="warn" title="This study is incomplete" action={(
          <span className="stu-row-actions">
            <Button size="sm" disabled={!!resumeBlocked} aria-describedby={resumeBlocked ? "stu-resume-why" : undefined}
              onClick={() => void resumeStudy(detail.study.id).then((id) => { if (id) setResumedHere(true); })}>Resume</Button>
          </span>
        )}>
          {detail.study.reason ?? "It was never sealed."} Its charts appear once it is complete.
          {resumeBlocked && <span id="stu-resume-why" className="stu-reason"> {resumeBlocked}</span>}
        </Callout>
      )}
      {freezingHere && (freezeJob.job || freezeJob.error) && <JobProgress job={freezeJob.job} error={freezeJob.error} />}
      <FactsList title="Frozen" facts={[
        { label: "Created", value: formatDate(m.created) },
        { label: "Code", value: commitText(m.commit) },
        gate.name ? { label: "Production gate", value: gate.name, hint: [gate.reads ? `reads ${gate.reads.length} of ${all.length} members` : null, gate.mix_space ? `${gate.mix_space} mix` : null].filter(Boolean).join(" · ") || undefined } : { label: "Production gate", value: "none" },
        { label: "Evaluated", value: formatDate(m.evaluation?.evaluated_at) },
        { label: "Fields", value: detail.fields.length ? `${formatCount(detail.fields.length)} · ${formatBytes(fieldsBytes)}` : "none", hint: detail.fields.length ? "Compressed, on holylabs" : undefined },
      ]} />
      <NoteEditor detail={detail} onSaved={reload} />
      <Details summary="Provenance">
        <dl className="stu-prov">
          <dt>Study id</dt><dd className="mono">{detail.study.id}</dd>
          <dt>Manifest sha256</dt><dd className="mono">{detail.manifest_sha256}</dd>
          <dt>Citation</dt><dd>{detail.citation}</dd>
          {m.records?.records_fp && <><dt>Test records</dt><dd className="mono">{m.records.records_fp}</dd></>}
          {gate.fingerprint && <><dt>Gate fingerprint</dt><dd className="mono">{gate.fingerprint}</dd></>}
          <dt>Numbers</dt><dd>{formatBytes(detail.study.numbers_bytes)}</dd>
        </dl>
      </Details>

      {complete && (
        <>
          <Toolbar label="Selection">
            <ToolbarGroup label="Members">
              <MultiSelect size="sm" aria-label="Members" value={picked} onChange={pickMembers} placeholder={`all ${formatCount(all.length)}`}
                options={all.map((x) => ({ value: x.label, label: memberName(x.label), hint: [memberKey(x, "loss"), `knee ${memberKey(x, "training_knee")}`].join(" · ") }))} />
              {picked.length > 0 && <IconButton size="sm" icon="reset" label="Every member" onClick={() => pickMembers([])} />}
            </ToolbarGroup>
            <ToolbarGroup label="Group by">
              <Select size="sm" aria-label="Group by" value={group} onChange={(v) => { setGroup(v); setReference(""); }} options={groupOptions} />
            </ToolbarGroup>
            {!group && (
              <ToolbarGroup label="Colour by">
                <Select size="sm" aria-label="Colour by" value={colourBy} onChange={setColour}
                  options={groupFields.map((f) => ({ value: f, label: GROUP_LABEL[f] ?? f }))} />
              </ToolbarGroup>
            )}
            {detail.selections.length > 0 && (
              <ToolbarGroup label="Saved">
                <Select size="sm" aria-label="Apply a saved selection" value="" onChange={applySaved} placeholder={`${detail.selections.length} saved`}
                  options={detail.selections.map((s) => ({ value: s.name, label: s.name }))} />
              </ToolbarGroup>
            )}
            <SaveSelection detail={detail} current={{ members: sel.members ?? null, group: sel.group ?? null }} onSaved={reload} />
          </Toolbar>
          <Tabs<Chart> aria-label="Charts" value={chart} onChange={setChart} variant="line"
            tabs={CHARTS.map((c) => ({ id: c, label: CHART_TAB[c] }))}>
            <ChartBody {...chartProps} />
          </Tabs>
        </>
      )}

      <Section title="Fields" sub={detail.fields.length ? `${detail.fields.length} attached` : undefined}>
        <FieldsSection detail={detail} />
      </Section>
    </div>
  );
}

export function StudyView({ id }: { id: string }) {
  const res = useStudy(id);
  if (res.error && !res.data) {
    return (
      <div className="stu-stack">
        <Button size="sm" variant="ghost" icon="chevronLeft" asChild><Link to={LIST_PATH}>All studies</Link></Button>
        <Callout tone={res.error.status === 404 ? "neutral" : "bad"} title={res.error.status === 404 ? "No such study" : "Could not read the study"}
          action={res.error.status === 404 ? undefined : <Button size="sm" onClick={res.reload}>Retry</Button>}>{res.error.message}</Callout>
      </div>
    );
  }
  if (!res.data) return <Skeleton lines={8} />;
  return <Loaded detail={res.data} reload={() => { invalidate(studyUrl(id)); res.reload(); }} />;
}

