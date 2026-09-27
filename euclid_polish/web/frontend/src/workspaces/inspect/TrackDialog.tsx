/* "📌 Track": back the FITS file up into the active tracking campaign
   (POST /api/tracking/backup kind=fits; best-effort push to holylabs). */
import { useState } from "react";
import { apiPost, ApiError } from "../../api/client";
import { invalidate } from "../../api/query";
import { Button, Dialog, Field, Input, Textarea, toast } from "../../ui";
import { basename } from "./model";

type BackupReply = {
  ok: boolean;
  error?: string;
  record?: { name?: string };
  warning?: string | null;
  sync?: { ok?: boolean; error?: string } | null;
};

export function TrackDialog({ fits, open, onOpenChange }: {
  fits: string; open: boolean; onOpenChange: (open: boolean) => void;
}) {
  const [comment, setComment] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit() {
    setBusy(true);
    try {
      const reply = await apiPost<BackupReply>("/api/tracking/backup", { kind: "fits", path: fits, comment, name: name.trim() || undefined });
      if (!reply.ok) throw new Error(reply.error ?? "backup failed");
      toast.success(`Tracked ${reply.record?.name ?? basename(fits)}`);
      if (reply.warning) toast.warning(reply.warning);
      if (reply.sync && reply.sync.ok === false && reply.sync.error) toast.info(`Not pushed to holylabs: ${reply.sync.error}`);
      void invalidate("/api/tracking");
      setComment(""); setName("");
      onOpenChange(false);
    } catch (e) {
      toast.error(e instanceof ApiError || e instanceof Error ? e.message : "backup failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange} title="Track this FITS file"
      description={<>Copies <code className="mono">{basename(fits)}</code> into the active tracking campaign (with the commit), then tries to push it to holylabs.</>}
      footer={(
        <>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>Cancel</Button>
          <Button variant="primary" icon="pin" loading={busy} onClick={() => void submit()}>Track</Button>
        </>
      )}>
      <div className="insp-track">
        <Field label="Comment" description="What this file shows, for the lab notebook.">
          <Textarea value={comment} onChange={setComment} rows={3} />
        </Field>
        <Field label="Name" description="Optional; defaults to the file name.">
          <Input value={name} onChange={setName} placeholder={basename(fits)} onEnter={() => void submit()} />
        </Field>
      </div>
    </Dialog>
  );
}
