/* ops/tracking (spec §8.7): the experiment lab notebook. The active campaign
 * (save a snapshot, push to holylabs, back up a model / FITS / image), its
 * notebook, backups, logged FASRC jobs, the archived campaigns and the
 * time-travel sandboxes. `?view=` selects: notebook | backups | jobs |
 * archive | sandboxes. */
import { useCallback, useState } from "react";
import { apiPost } from "../../../api/client";
import { invalidate, useResource } from "../../../api/query";
import { usePageActions } from "../../../app/palette";
import { formatDateTime } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Badge, Button, Callout, Card, CardBody, Dialog, EmptyState, Field, Input, Page, PageHead, Segmented, Select,
  Skeleton, Textarea, Tooltip, confirm, toast,
} from "../../../ui";
import { TRACKING_STATE_URL, type TrackingState } from "../api";
import { ArchiveTable } from "../tracking/Archive";
import { BackupDialog, BackupTables } from "../tracking/Backups";
import { NotebookView } from "../tracking/Notebook";
import { SandboxTable } from "../tracking/Sandboxes";
import { TimeTravelDialog, commitText, type TimeTravelTarget } from "../tracking/TimeTravel";
import { TrackedJobs } from "../tracking/TrackedJobs";
import "../ops.css";

const VIEWS = ["notebook", "backups", "jobs", "archive", "sandboxes"] as const;
type View = typeof VIEWS[number];
const parseView = (raw: string): View | undefined => (VIEWS as readonly string[]).includes(raw) ? raw as View : undefined;

function NewCampaignDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const [title, setTitle] = useState("");
  const [desc, setDesc] = useState("");
  const [busy, setBusy] = useState(false);
  async function create() {
    setBusy(true);
    try {
      await apiPost("/api/tracking/new", { title: title.trim(), description: desc.trim() });
      toast.success(`Campaign “${title.trim()}” started`);
      setTitle(""); setDesc(""); onOpenChange(false);
      void invalidate(TRACKING_STATE_URL);
    } catch (e) { toast.error(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }
  return (
    <Dialog open={open} onOpenChange={onOpenChange} title="New campaign"
      description="Stamps the current commit; backups, notes and every FASRC job you submit are collected in it until you save it."
      footer={<><Button variant="ghost" onClick={() => onOpenChange(false)}>Cancel</Button>
        <Button variant="primary" loading={busy} disabled={!title.trim()} onClick={create}>Create</Button></>}>
      <div className="ops-stack">
        <Field label="Title"><Input value={title} onChange={setTitle} placeholder="e.g. multi-knee members" onEnter={() => title.trim() && void create()} /></Field>
        <Field label="Description"><Textarea value={desc} onChange={setDesc} rows={3} placeholder="What are we testing?" /></Field>
      </div>
    </Dialog>
  );
}

export default function Tracking() {
  const res = useResource<TrackingState>(TRACKING_STATE_URL, [], { ttl: 15_000, poll: 30_000 });
  const [view, setView] = useUrlState<View>("view", "notebook", { parse: parseView, replace: false });
  const [newOpen, setNewOpen] = useState(false);
  const [backupOpen, setBackupOpen] = useState(false);
  const [tt, setTt] = useState<TimeTravelTarget | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const s = res.data;
  const active = s?.active ?? null;
  const onTimeTravel = useCallback((t: TimeTravelTarget) => setTt(t), []);

  async function saveSnapshot() {
    if (!active) return;
    if (!(await confirm({ title: `Save “${active.title}”?`,
      message: "Archives the campaign with its backups, notebook and job log at the current commit; a new campaign can then start.",
      confirmLabel: "Save snapshot" }))) return;
    setBusy("save");
    try {
      const r = await apiPost<{ ok?: boolean; error?: string; warning?: string | null; sync?: { ok?: boolean; error?: string } }>("/api/tracking/save");
      if (r.error) toast.error(r.error);
      else {
        toast.success(`Saved “${active.title}”${r.sync?.ok ? " and pushed to holylabs" : ""}`);
        if (r.warning) toast.warning(r.warning, { duration: 12_000 });
        if (r.sync && !r.sync.ok) toast.info(`holylabs push skipped: ${r.sync.error ?? "offline"}`);
      }
    } catch (e) { toast.error(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(null); void invalidate(TRACKING_STATE_URL); }
  }
  async function push() {
    if (!(await confirm({ title: "Push the tracking store to holylabs?", message: `rsync of ${s?.tracking_dir ?? "./tracking"} → ${s?.remote_dir ?? "holylabs"}.`,
      confirmLabel: "Push" }))) return;
    setBusy("push");
    try {
      await apiPost("/api/tracking/sync");
      toast.success("Pushed to holylabs");
    } catch (e) { toast.error(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(null); }
  }

  usePageActions([
    { id: "track-new", label: "New tracking campaign", group: "Tracking", disabled: !!active, run: () => setNewOpen(true) },
    { id: "track-backup", label: "Back up a model / FITS / image", group: "Tracking", keywords: ["track", "backup"],
      disabled: !active, run: () => setBackupOpen(true) },
    { id: "track-save", label: "Save the active campaign", group: "Tracking", disabled: !active, run: () => void saveSnapshot() },
    { id: "track-push", label: "Push tracking to holylabs", group: "Tracking", disabled: !s?.ssh_connected, run: () => void push() },
    { id: "track-notebook", label: "Tracking notebook", group: "Tracking", run: () => setView("notebook") },
  ]);

  const counts: Record<View, number | null> = {
    notebook: null, backups: s ? s.backups.models.length + s.backups.fits.length + s.backups.images.length : null,
    jobs: s ? s.jobs_count : null, archive: s ? s.archived.length : null, sandboxes: s ? s.sandboxes.length : null,
  };
  return (
    <Page className="ops-page">
      <PageHead eyebrow="ops · tracking" title="Tracking" />
      <div className="ops-bar" role="toolbar" aria-label="Tracking">
        {active ? (
          <span className="ops-campaign">
            <Badge tone="info" dot>active</Badge>
            <Tooltip content={active.description || "No description"}><strong tabIndex={0}>{active.title}</strong></Tooltip>
            <code className="mono ops-dim">{commitText(active.created_commit)}</code>
          </span>
        ) : <span className="ops-dim">No active campaign</span>}
        <span className="ops-spacer" />
        {active ? <>
          <Button size="sm" icon="plus" onClick={() => setBackupOpen(true)}>Back up…</Button>
          <Button size="sm" variant="primary" loading={busy === "save"} onClick={saveSnapshot}>Save snapshot</Button>
        </> : <Button size="sm" variant="primary" icon="plus" onClick={() => setNewOpen(true)}>New campaign</Button>}
        <Tooltip content={s?.ssh_connected ? `→ ${s?.remote_dir ?? "holylabs"}` : "FASRC offline"}>
          <span><Button size="sm" variant="ghost" loading={busy === "push"} disabled={!s?.ssh_connected} onClick={push}>Push</Button></span>
        </Tooltip>
      </div>
      <div className="ops-row" style={{ marginBottom: "var(--s3)" }}>
        <Segmented<View> value={view} onChange={setView} aria-label="Tracking view"
          options={VIEWS.map((v) => ({ value: v, label: `${v[0].toUpperCase()}${v.slice(1)}${counts[v] ? ` · ${counts[v]}` : ""}` }))} />
      </div>
      {res.loading && !s && <Skeleton lines={8} />}
      {res.error && !s && <Callout tone="bad" title="Could not read the tracking store">{res.error.message}</Callout>}
      {s && view === "notebook" && (active
        ? <NotebookView text={s.log_md} editable title={`Notebook · ${active.title}`} />
        : (
          <Card><CardBody>
            <EmptyState icon="info" title="No active campaign"
              action={<Button variant="primary" onClick={() => setNewOpen(true)}>New campaign</Button>}>
              {s.archived.length ? `${s.archived.length} saved campaigns are in the archive.` : "Start one to collect backups, notes and FASRC jobs."}
            </EmptyState>
          </CardBody></Card>
        ))}
      {s && view === "backups" && (active
        ? <BackupTables backups={s.backups} campaign="current" trackingDir={s.tracking_dir} title={active.title} onTimeTravel={onTimeTravel} />
        : <Callout tone="info">Backups belong to a campaign — open an archived one from the archive.</Callout>)}
      {s && view === "jobs" && <JobsView state={s} />}
      {s && view === "archive" && <ArchiveTable archived={s.archived} onTimeTravel={onTimeTravel} />}
      {s && view === "sandboxes" && <SandboxTable sandboxes={s.sandboxes} />}
      {s && active && view === "notebook" && (
        <p className="ops-note">Started {formatDateTime(active.created_at)} · store {s.tracking_dir}</p>
      )}
      <NewCampaignDialog open={newOpen} onOpenChange={setNewOpen} />
      <BackupDialog open={backupOpen} onOpenChange={setBackupOpen} />
      <TimeTravelDialog target={tt} onClose={() => setTt(null)} fasrcConnected={!!s?.ssh_connected} />
    </Page>
  );
}

function JobsView({ state }: { state: TrackingState }) {
  const [which, setWhich] = useUrlState("jcamp", "current");
  const options = [
    { value: "current", label: `Active campaign · ${state.jobs_count}` },
    { value: "unassigned", label: `Unassigned · ${state.unassigned_count}` },
    ...state.archived.map((a) => ({ value: a._dir, label: a.title })),
  ];
  return (
    <div className="ops-stack">
      <div className="ops-row">
        <Segmented size="sm" value={which === "unassigned" ? "unassigned" : which === "current" ? "current" : "archived"}
          aria-label="Job log" onChange={(v) => setWhich(v === "archived" ? (state.archived[0]?._dir ?? "current") : v)}
          options={[{ value: "current", label: `Active · ${state.jobs_count}` }, { value: "unassigned", label: `Unassigned · ${state.unassigned_count}` },
            { value: "archived", label: "Archived…", disabled: !state.archived.length }]} />
        {which !== "current" && which !== "unassigned" && (
          <Select searchable size="sm" value={which} onChange={setWhich} aria-label="Archived campaign" options={options.slice(2)} />
        )}
      </div>
      {which === "current" && !state.active
        ? <Callout tone="info">No active campaign: jobs submitted now are logged as unassigned.</Callout>
        : <TrackedJobs key={which} campaign={which} />}
    </div>
  );
}
