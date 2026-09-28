/* Figures › Studies, the list: every study, newest first — name, created,
 * members, gate, fields, note and (only on a problem) its state. An
 * incomplete study offers Resume; Delete is confirmed. A freeze running now
 * shows its progress here too. Reads `/api/studies` only. */
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useJob, useJobsStore } from "../../../api/jobs";
import { usePageActions } from "../../../app/palette";
import { formatBytes, formatCount, formatDate } from "../../../format";
import { Badge, Button, Callout, DataTable, EmptyState, JobProgress, Tooltip, type DataColumn } from "../../../ui";
import { FreezeStudyDialog } from "../../shared/FreezeStudyDialog";
import { PageLead } from "../../shared/PageLead";
import { FREEZE_JOB_KEY, studyPath } from "../../shared/studyFreeze";
import { deleteStudy, resumeStudy } from "./actions";
import { useStudies, type StudySummary } from "./api";

export function StudyList() {
  const list = useStudies();
  const navigate = useNavigate();
  const [freezeOpen, setFreezeOpen] = useState(false);
  const freezeJob = useJob(FREEZE_JOB_KEY);
  const freezing = list.data?.freezing ?? null;

  useEffect(() => {
    // A freeze started elsewhere (another tab, before a reload): follow it here.
    if (freezing && useJobsStore.getState().keyed[FREEZE_JOB_KEY] !== freezing.job_id) {
      useJobsStore.getState().register(freezing.job_id, FREEZE_JOB_KEY);
    }
  }, [freezing]);
  const done = freezeJob.job != null && freezeJob.job.status !== "running";
  useEffect(() => {
    // The freeze ended: re-read the list, and let go of the job (the shell's
    // toast reports how it ended) — unless the dialog is open and still shows
    // it (its outcome, its link); closing the dialog lets go then.
    if (!done) return;
    list.reload();
    if (!freezeOpen) useJobsStore.getState().forgetKey(FREEZE_JOB_KEY);
  }, [done]); // eslint-disable-line react-hooks/exhaustive-deps
  const closeFreeze = () => {
    setFreezeOpen(false);
    const j = useJobsStore.getState();
    const id = j.keyed[FREEZE_JOB_KEY];
    if (id && j.jobs[id] && j.jobs[id].status !== "running") j.forgetKey(FREEZE_JOB_KEY);
  };
  const busyWhy = freezing || freezeJob.busy
    ? `A study is being frozen${freezing ? ` (“${list.data?.studies.find((x) => x.id === freezing.study_id)?.name ?? freezing.study_id}”)` : ""}: Resume waits for it to finish, and the study being frozen cannot be deleted.`
    : null;

  usePageActions([
    { id: "study-freeze", label: "Freeze a study of the starfull ensemble…", group: "Studies", keywords: ["study", "paper", "snapshot"], run: () => setFreezeOpen(true) },
  ]);

  const columns = useMemo<DataColumn<StudySummary>[]>(() => [
    { id: "name", header: "Study", cell: (s) => <Link to={studyPath(s.id)} className="stu-name">{s.name}</Link> },
    { id: "created", header: "Created", width: 118, cell: (s) => formatDate(s.created), priority: 3 },
    { id: "regime", header: "Regime", width: 92, priority: 4 },
    { id: "members", header: "Members", numeric: true, width: 90, cell: (s) => formatCount(s.members) },
    { id: "gate", header: "Gate", width: 120, priority: 2, cell: (s) => s.gate ?? "—" },
    { id: "fields", header: "Fields", numeric: true, width: 72, cell: (s) => formatCount(s.fields) },
    { id: "note", header: "Note", priority: 5, cell: (s) => <span className="stu-note-cell" title={s.note}>{s.note || "—"}</span> },
    { id: "state", header: "State", width: 150, accessor: (s) => s.state, cell: (s) => (s.state === "complete"
      ? <span className="muted">complete</span>
      : <Tooltip content={s.reason ?? "incomplete"}><span tabIndex={0}><Badge tone="warn" dot>incomplete</Badge></span></Tooltip>) },
    { id: "actions", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 190, cell: (s) => (
      <span className="stu-row-actions">
        {s.state === "incomplete" && (
          <Button size="sm" disabled={!!busyWhy} aria-describedby={busyWhy ? "stu-list-busy" : undefined} onClick={() => void resumeStudy(s.id)}>Resume</Button>
        )}
        <Button size="sm" variant="ghost" disabled={freezing?.study_id === s.id} aria-describedby={freezing?.study_id === s.id ? "stu-list-busy" : undefined}
          onClick={() => void deleteStudy(s)}>Delete</Button>
      </span>
    ) },
  ], [freezing, busyWhy]);

  const studies = list.data?.studies ?? [];
  const bytes = studies.reduce((t, s) => t + (s.numbers_bytes || 0), 0);

  return (
    <div className="stu-stack">
      <PageLead right={<Button variant="primary" onClick={() => setFreezeOpen(true)}>Freeze study…</Button>}>
        Frozen comparisons of the whole ensemble. A study keeps every member's numbers after the member is archived, and exports each chart as a
        publication figure with its CSV.
      </PageLead>
      {busyWhy && <p id="stu-list-busy" className="stu-reason" role="status">{busyWhy}</p>}
      {(freezeJob.job || freezeJob.error) && <JobProgress job={freezeJob.job} error={freezeJob.error} />}
      {list.error && !list.data && (
        <Callout tone="bad" title="Could not read the studies" action={<Button size="sm" onClick={list.reload}>Retry</Button>}>{list.error.message}</Callout>
      )}
      {list.data && !studies.length ? (
        <EmptyState icon="layers" title="No studies yet" action={<Button variant="primary" onClick={() => setFreezeOpen(true)}>Freeze study…</Button>}>
          Freeze the ensemble from Models › Leaderboard or here: the dialog shows what will be frozen and which fields can be attached before anything is written.
        </EmptyState>
      ) : (
        <DataTable rows={studies} columns={columns} rowKey={(s) => s.id} aria-label="Studies" loading={list.loading}
          onRowClick={(s) => navigate(studyPath(s.id))} height="auto" countText={null}
          caption={studies.length ? `${formatCount(studies.length)} stud${studies.length === 1 ? "y" : "ies"} · numbers ${formatBytes(bytes)} locally` : undefined} />
      )}
      {freezeOpen && <FreezeStudyDialog mode="starfull" onClose={closeFreeze} />}
    </div>
  );
}
