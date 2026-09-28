/* The campaign bar of Notebook › Log (and Backups): the active campaign with
 * the commit it started at, Back up…, Push (the store to holylabs) and New
 * campaign…; Save snapshot is secondary (the menu). A new campaign while one
 * is active saves that one first — the dialog says so. Every write is
 * confirmed or goes through a dialog. */
import { useState } from "react";
import { apiPost } from "../../api/client";
import { invalidate } from "../../api/query";
import { usePageActions } from "../../app/palette";
import { formatDateTime, formatRelative } from "../../format";
import {
  Badge, Button, Dialog, Field, IconButton, Input, Menu, Textarea, Toolbar, ToolbarSpacer, ToolbarText, Tooltip,
  confirm, toast,
} from "../../ui";
import { TRACKING_STATE_URL, type TrackingState } from "./api";
import { BackupDialog } from "./Backups";
import { commitText } from "./model";

type SaveResp = { ok?: boolean; error?: string; warning?: string | null; sync?: { ok?: boolean; error?: string } };

async function saveActive(title: string): Promise<boolean> {
  try {
    const r = await apiPost<SaveResp>("/api/tracking/save");
    if (r.error) { toast.error(r.error); return false; }
    toast.success(`Saved “${title}”${r.sync?.ok ? " and pushed to holylabs" : ""}`);
    if (r.warning) toast.warning(r.warning, { duration: 12_000 });
    if (r.sync && !r.sync.ok) toast.info(`holylabs push skipped: ${r.sync.error ?? "offline"}`);
    return true;
  } catch (e) {
    toast.error(e instanceof Error ? e.message : String(e));
    return false;
  } finally {
    void invalidate(TRACKING_STATE_URL);
  }
}

function NewCampaignDialog({ open, onOpenChange, active }: {
  open: boolean; onOpenChange: (o: boolean) => void; active: string | null;
}) {
  const [title, setTitle] = useState("");
  const [desc, setDesc] = useState("");
  const [busy, setBusy] = useState(false);
  async function create() {
    setBusy(true);
    try {
      if (active && !(await saveActive(active))) return;
      await apiPost("/api/tracking/new", { title: title.trim(), description: desc.trim() });
      toast.success(`Campaign “${title.trim()}” started`);
      setTitle(""); setDesc(""); onOpenChange(false);
    } catch (e) { toast.error(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); void invalidate(TRACKING_STATE_URL); }
  }
  return (
    <Dialog open={open} onOpenChange={onOpenChange} title="New campaign"
      description={active
        ? `Saves “${active}” first (archived with its backups, notebook and job log at the current commit), then starts the new one at this commit.`
        : "Stamps the current commit; backups, notes and every FASRC job you submit are collected in it until you save it."}
      footer={<><Button variant="ghost" onClick={() => onOpenChange(false)}>Cancel</Button>
        <Button variant="primary" loading={busy} disabled={!title.trim()} onClick={() => void create()}>
          {active ? `Save “${active}” and start` : "Create"}
        </Button></>}>
      <div className="nb-stack">
        <Field label="Title"><Input value={title} onChange={setTitle} placeholder="e.g. multi-knee members" onEnter={() => title.trim() && void create()} /></Field>
        <Field label="Description"><Textarea value={desc} onChange={setDesc} rows={3} placeholder="What are we testing?" /></Field>
      </div>
    </Dialog>
  );
}

export function CampaignBar({ state }: { state: TrackingState | null }) {
  const [newOpen, setNewOpen] = useState(false);
  const [backupOpen, setBackupOpen] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const active = state?.active ?? null;

  async function saveSnapshot() {
    if (!active) return;
    if (!(await confirm({ title: `Save “${active.title}”?`,
      message: "Archives the campaign with its backups, notebook and job log at the current commit; a new campaign can then start.",
      confirmLabel: "Save snapshot" }))) return;
    setBusy("save");
    await saveActive(active.title);
    setBusy(null);
  }
  async function push() {
    if (!(await confirm({ title: "Push the tracking store to holylabs?", message: `rsync of ${state?.tracking_dir ?? "./tracking"} → ${state?.remote_dir ?? "holylabs"}.`,
      confirmLabel: "Push" }))) return;
    setBusy("push");
    try {
      await apiPost("/api/tracking/sync");
      toast.success("Pushed to holylabs");
    } catch (e) { toast.error(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(null); }
  }

  usePageActions([
    { id: "nb-new", label: "New notebook campaign", group: "Notebook", run: () => setNewOpen(true) },
    { id: "nb-backup", label: "Back up a model / FITS / image", group: "Notebook", keywords: ["track", "backup"],
      disabled: !active, run: () => setBackupOpen(true) },
    { id: "nb-save", label: "Save the active campaign", group: "Notebook", disabled: !active, run: () => void saveSnapshot() },
    { id: "nb-push", label: "Push the notebook store to holylabs", group: "Notebook", disabled: !state?.ssh_connected, run: () => void push() },
  ]);

  return (
    <>
      <Toolbar label="Campaign">
        {active ? (
          <span className="nb-campaign">
            <Tooltip content={active.description || "No description"}><strong tabIndex={0}>{active.title}</strong></Tooltip>
            <ToolbarText>
              started <span title={formatDateTime(active.created_at)}>{formatRelative(active.created_at)}</span> at{" "}
              <code className="mono">{commitText(active.created_commit)}</code>
            </ToolbarText>
          </span>
        ) : state ? <Badge tone="warn">No active campaign</Badge> : null}
        <ToolbarSpacer />
        <Button size="sm" icon="plus" disabled={!active} onClick={() => setBackupOpen(true)}>Back up…</Button>
        <Tooltip content={state?.ssh_connected ? `→ ${state?.remote_dir ?? "holylabs"}` : "Needs FASRC"}>
          <span><Button size="sm" loading={busy === "push"} disabled={!state?.ssh_connected} onClick={() => void push()}>Push</Button></span>
        </Tooltip>
        <Button size="sm" variant={active ? "default" : "primary"} icon="plus" onClick={() => setNewOpen(true)}>New campaign…</Button>
        <Menu label="More campaign actions" align="end" items={[
          { label: busy === "save" ? "Saving…" : "Save snapshot", disabled: !active || busy != null, onSelect: () => void saveSnapshot() },
        ]} trigger={<IconButton size="sm" icon="more" label="More campaign actions" />} />
      </Toolbar>
      <NewCampaignDialog open={newOpen} onOpenChange={setNewOpen} active={active?.title ?? null} />
      <BackupDialog open={backupOpen} onOpenChange={setBackupOpen} />
    </>
  );
}
