/* A campaign's model / FITS / image backups (each with its comment and
 * commit stamp), per-model time travel, and the "Back up…" dialog (a model
 * checkpoint dir — e.g. an active ensemble member —, a FITS file or an image). */
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { apiPost } from "../../../api/client";
import { invalidate, useResource } from "../../../api/query";
import { formatBytes, formatDateTime, formatRelative } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Badge, Button, Callout, DataTable, Dialog, Field, IconButton, Input, Segmented, Select, Textarea, Tooltip, toast,
  type DataColumn,
} from "../../../ui";
import { TRACKING_STATE_URL, type BackupRec, type Backups } from "../api";
import { commitText, type TimeTravelTarget } from "./TimeTravel";

type Kind = "models" | "fits" | "images";
const KIND_LABEL: Record<Kind, string> = { models: "Models", fits: "FITS", images: "Images" };

/** `tracking/<campaign dir>/<sub>/<name>` for the Inspect workspace. */
function storedPath(trackingDir: string | undefined, campaignDir: string, kind: Kind, name: string): string {
  const root = trackingDir && /\/tracking\/?$/.test(trackingDir) ? "tracking" : (trackingDir ?? "tracking");
  const sub = campaignDir === "current" ? "current" : `archive/${campaignDir}`;
  return `${root}/${sub}/${kind}/${name}`;
}

export function BackupTables({ backups, campaign, trackingDir, title, onTimeTravel }: {
  backups: Backups; campaign: string; trackingDir?: string; title: string;
  onTimeTravel: (t: TimeTravelTarget) => void;
}) {
  const [kind, setKind] = useUrlState<Kind>("bk", "models", { parse: (r) => (r in KIND_LABEL ? r as Kind : undefined) });
  const rows = backups[kind] ?? [];
  const columns = useMemo<DataColumn<BackupRec>[]>(() => [
    { id: "name", header: "Name", cell: (r) => <code className="mono">{r.name}</code> },
    { id: "comment", header: "Comment", cell: (r) => <span className="ops-ellipsis" title={r.comment}>{r.comment || <span className="ops-dim">—</span>}</span> },
    { id: "size_bytes", header: "Size", numeric: true, width: 90, cell: (r) => formatBytes(r.size_bytes) },
    { id: "commit", header: "Commit", width: 104, accessor: (r) => commitText(r.commit),
      cell: (r) => <span className="ops-row"><code className="mono">{commitText(r.commit)}</code>{typeof r.commit === "object" && r.commit?.dirty && <Badge size="sm" tone="warn">dirty</Badge>}</span> },
    { id: "created_at", header: "Saved", width: 104,
      cell: (r) => <span className="ops-dim ops-small" title={formatDateTime(r.created_at)}>{formatRelative(r.created_at)}</span> },
    { id: "actions", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 80,
      cell: (r) => (
        <span className="ops-row-actions">
          {kind === "models" && (
            <IconButton size="sm" icon="reset" label={`Time-travel to ${r.name}`}
              onClick={() => onTimeTravel({ campaign, model: r.name, title: r.name, commit: r.commit })} />
          )}
          {kind === "fits" && (
            <Tooltip content="Open in Inspect">
              <Link className="ops-iconlink" aria-label={`Inspect ${r.name}`}
                to={`/inspect?fits=${encodeURIComponent(storedPath(trackingDir, campaign, kind, r.name))}`}>⌕</Link>
            </Tooltip>
          )}
        </span>
      ) },
  ], [kind, campaign, trackingDir, onTimeTravel]);
  return (
    <DataTable rows={rows} columns={columns} rowKey={(r) => r.name} aria-label={`${title} ${KIND_LABEL[kind]}`}
      height={420} dense exportName={`backups-${kind}`}
      empty={`No ${KIND_LABEL[kind].toLowerCase()} backed up in this campaign.`}
      toolbar={<Segmented<Kind> size="sm" value={kind} onChange={setKind} aria-label="Backup kind"
        options={(Object.keys(KIND_LABEL) as Kind[]).map((k) => ({ value: k, label: `${KIND_LABEL[k]} · ${backups[k]?.length ?? 0}` }))} />} />
  );
}

type ModelsResp = { members?: string[] };
type BackupKind = "model" | "fits" | "image";

/** "Back up…" into the active campaign (POST /api/tracking/backup). */
export function BackupDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const models = useResource<ModelsResp>(open ? "/api/models" : null, [open], { ttl: 5 * 60_000 });
  const [kind, setKind] = useState<BackupKind>("model");
  const [member, setMember] = useState("");
  const [path, setPath] = useState("");
  const [comment, setComment] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [warning, setWarning] = useState<string | null>(null);
  const members = (models.data?.members ?? []).map((label) => {
    const n = label.replace(/·.*$/, "");
    return `member_${n.padStart(2, "0")}`;
  });
  const ckpt = kind === "model" ? (member ? `ckpt/ensemble/${member}` : path) : "";

  async function save() {
    setBusy(true); setWarning(null);
    try {
      const r = await apiPost<{ ok?: boolean; error?: string; warning?: string | null; record?: { name?: string }; sync?: { ok?: boolean; error?: string } }>(
        "/api/tracking/backup", {
          kind, comment, name: name.trim() || undefined,
          ckpt_dir: kind === "model" ? ckpt : undefined, path: kind === "model" ? undefined : path,
        });
      if (r.error) { toast.error(r.error); return; }
      toast.success(`Backed up ${r.record?.name ?? kind}${r.sync && !r.sync.ok ? " (holylabs push skipped)" : ""}`);
      if (r.warning) setWarning(r.warning); else onOpenChange(false);
      void invalidate(TRACKING_STATE_URL);
    } catch (e) { toast.error(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }
  return (
    <Dialog open={open} onOpenChange={onOpenChange} title="Back up into the campaign"
      description="Copies it into the active campaign with a comment and the current commit stamp, then pushes to holylabs (best effort)."
      footer={<>
        <Button variant="ghost" onClick={() => onOpenChange(false)}>{warning ? "Close" : "Cancel"}</Button>
        <Button variant="primary" loading={busy} disabled={kind === "model" ? !ckpt : !path.trim()} onClick={save}>Back up</Button>
      </>}>
      <div className="ops-stack">
        <Segmented<BackupKind> value={kind} onChange={setKind} aria-label="What to back up"
          options={[{ value: "model", label: "Model" }, { value: "fits", label: "FITS" }, { value: "image", label: "Image" }]} />
        {kind === "model" ? (
          <>
            <Field label="Ensemble member" hint="Its checkpoint dir (ckpt/ensemble/member_NN) is copied restorably.">
              <Select searchable value={member} onChange={setMember} placeholder={models.loading ? "Loading…" : "Pick a member"}
                options={[{ value: "", label: "Custom path…" }, ...members.map((m) => ({ value: m, label: m }))]} />
            </Field>
            {!member && (
              <Field label="Checkpoint dir" hint="Under the checkpoint root, e.g. ckpt/ensemble/member_196.">
                <Input value={path} onChange={setPath} spellCheck={false} placeholder="ckpt/ensemble/member_…" />
              </Field>
            )}
          </>
        ) : (
          <Field label={kind === "fits" ? "FITS file" : "Image file"} hint="A project path, e.g. data/eval_results/…/SR.fits.">
            <Input value={path} onChange={setPath} spellCheck={false} placeholder={kind === "fits" ? "data/…/file.fits" : "figures/…/plot.png"} />
          </Field>
        )}
        <Field label="Name" hint="Optional stored name (defaults to the source's).">
          <Input value={name} onChange={setName} spellCheck={false} />
        </Field>
        <Field label="Comment"><Textarea value={comment} onChange={setComment} rows={2} placeholder="Why this one?" /></Field>
        {warning && <Callout tone="warn" title="Saved, but not exactly reproducible">{warning}</Callout>}
      </div>
    </Dialog>
  );
}
