/* "📌 Log to tracking": review / edit the markdown note, then append it to the
 * active tracking campaign (POST /api/tracking/log mode=append). */
import { useState, type ReactElement } from "react";
import { Button, Dialog, Textarea } from "../../../ui";
import { logToTracking } from "./actions";

export function LogToTrackingButton({ note, trigger, disabled }: {
  /** Builds the note when the dialog opens (the record may still be updating). */
  note: () => string; trigger?: ReactElement; disabled?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const onOpenChange = (next: boolean) => {
    if (next) setText(note());
    setOpen(next);
  };
  const submit = async () => {
    setBusy(true);
    const ok = await logToTracking(text);
    setBusy(false);
    if (ok) setOpen(false);
  };
  return (
    <Dialog open={open} onOpenChange={onOpenChange} size="lg" title="Log to the tracking notebook"
      description="Appended to the active campaign's log.md under a timestamped heading."
      trigger={trigger ?? <Button size="sm" icon="pin" disabled={disabled}>Log to tracking</Button>}
      footer={<>
        <Button variant="ghost" onClick={() => setOpen(false)}>Cancel</Button>
        <Button variant="primary" icon="pin" loading={busy} disabled={!text.trim()} onClick={submit}>Append</Button>
      </>}>
      <Textarea value={text} onChange={setText} rows={14} className="res-mdnote mono" aria-label="Markdown note" />
    </Dialog>
  );
}
