/* "Log to tracking", shared by every workspace (Sky experiments, the
 * Ensemble loop: Evaluate summary, knee leaderboard, gate fit / compare /
 * promote; Home's quick action): review / edit a markdown note pre-filled
 * from the facts on the page, then append it to the active tracking
 * campaign's log.md (POST /api/tracking/log mode=append). Nothing is sent
 * until "Append" is pressed.
 *
 *   <LogToTrackingButton note={() => markdown} />               a trigger button
 *   <LogToTrackingDialog open onOpenChange note={() => md} />   controlled (a menu item, a Home action)
 */
import { useState, type ReactElement } from "react";
import { apiGet, apiPost, ApiError } from "../../api/client";
import { invalidate } from "../../api/query";
import { Button, Dialog, Textarea, toast, type ButtonSize } from "../../ui";
import "./shared.css";

export const TRACKING_LOG_URL = "/api/tracking/log";
const ALERTS_URL = "/api/system/alerts";

/** Append a markdown note to the active campaign's log; true when it landed. */
export async function appendTrackingNote(text: string): Promise<boolean> {
  try {
    const r = await apiPost<{ ok?: boolean; error?: string }>(TRACKING_LOG_URL, { text, mode: "append" });
    if (r?.ok === false || r?.error) throw new Error(r.error ?? "refused");
    toast.success("Logged to the tracking notebook");
    void invalidate("/api/tracking/");
    // The Home "Results since the last tracking entry" check reads log.md:
    // recompute it now (quietly) instead of after its 30 s memo.
    void apiGet(`${ALERTS_URL}?fresh=1`).then(() => invalidate(ALERTS_URL)).catch(() => invalidate(ALERTS_URL));
    return true;
  } catch (e) {
    toast.error("Could not log to tracking", { description: e instanceof ApiError || e instanceof Error ? e.message : String(e) });
    return false;
  }
}

type DialogProps = {
  open: boolean; onOpenChange: (open: boolean) => void;
  /** Builds the note when the dialog opens (the page may still be updating). */
  note: () => string;
  /** The dialog trigger (uncontrolled use goes through LogToTrackingButton). */
  trigger?: ReactElement;
};

export function LogToTrackingDialog({ open, onOpenChange, note, trigger }: DialogProps) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  // Fill the editor as the dialog opens (a trigger click or a controlled open).
  const [filledFor, setFilledFor] = useState(false);
  if (open && !filledFor) { setFilledFor(true); setText(note()); }
  if (!open && filledFor) setFilledFor(false);
  const submit = async () => {
    setBusy(true);
    const ok = await appendTrackingNote(text);
    setBusy(false);
    if (ok) onOpenChange(false);
  };
  return (
    <Dialog open={open} onOpenChange={onOpenChange} size="lg" title="Log to the tracking notebook"
      description="Appended to the active campaign's log.md under a timestamped heading. Edit it first if you like."
      trigger={trigger}
      footer={<>
        <Button variant="ghost" onClick={() => onOpenChange(false)}>Cancel</Button>
        <Button variant="primary" icon="pin" loading={busy} disabled={!text.trim()} onClick={() => void submit()}>Append</Button>
      </>}>
      <Textarea value={text} onChange={setText} rows={14} className="log-note mono" aria-label="Markdown note" spellCheck />
    </Dialog>
  );
}

export function LogToTrackingButton({ note, trigger, disabled, size = "sm", label = "Log to tracking", title }: {
  /** Builds the note when the dialog opens (the record may still be updating). */
  note: () => string; trigger?: ReactElement; disabled?: boolean;
  size?: ButtonSize; label?: string; title?: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <LogToTrackingDialog open={open} onOpenChange={setOpen} note={note}
      trigger={trigger ?? <Button size={size} icon="pin" disabled={disabled} title={title}>{label}</Button>} />
  );
}
