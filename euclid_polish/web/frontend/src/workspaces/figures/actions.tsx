/* Saved-crop actions shared by the Sheet tab (its crop pool) and the
 * `figure` inspector: confirmed delete and the rename dialog. */
import { useEffect, useState } from "react";
import { Button, Dialog, Field, Input, confirm, toast } from "../../ui";
import { deleteResults, renameResult, type SavedResult } from "./api";
import { closeInspector } from "../../app/inspector";
import { useInspector } from "../../state/inspector";

/** Delete saved results after a danger confirm; toasts the outcome and
 *  closes an inspector showing a deleted one. Resolves with the deleted ids. */
export async function confirmDeleteResults(results: readonly Pick<SavedResult, "id" | "label">[]): Promise<string[]> {
  if (!results.length) return [];
  const one = results.length === 1;
  const ok = await confirm({
    title: one ? `Delete “${results[0].label}”?` : `Delete ${results.length} saved results?`,
    message: `${one ? "Its" : "Their"} FITS crops are removed from the results store and dropped from every saved grid layout. This cannot be undone.`,
    tone: "danger", confirmLabel: "Delete",
  });
  if (!ok) return [];
  const { deleted, failed } = await deleteResults(results.map((r) => r.id));
  if (deleted.length) toast.success(one ? "Saved result deleted" : `${deleted.length} saved result${deleted.length === 1 ? "" : "s"} deleted`);
  if (failed.length) toast.error(`${failed.length} not deleted: ${failed[0][1]}`);
  const cur = useInspector.getState().current;
  if (cur?.kind === "figure" && deleted.includes(cur.id)) closeInspector();
  return deleted;
}

/** Rename a saved result (an empty label restores its default label). */
export function RenameDialog({ result, open, onOpenChange }: {
  result: SavedResult | null; open: boolean; onOpenChange: (open: boolean) => void;
}) {
  const [label, setLabel] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (open && result) { setLabel(result.label); setError(null); }
  }, [open, result]);
  if (!result) return null;
  const save = async (value: string) => {
    setBusy(true);
    setError(null);
    try {
      await renameResult(result.id, value);
      toast.success(value.trim() ? "Renamed" : "Label reset");
      onOpenChange(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const tooLong = label.length > 120;
  return (
    <Dialog open={open} onOpenChange={onOpenChange} title="Rename saved result" size="sm"
      footer={<>
        {result.default_label && result.label !== result.default_label && (
          <Button variant="ghost" disabled={busy} onClick={() => void save("")}>Reset to “{result.default_label}”</Button>
        )}
        <Button variant="ghost" onClick={() => onOpenChange(false)}>Cancel</Button>
        <Button variant="primary" loading={busy} disabled={!label.trim() || tooLong} onClick={() => void save(label)}>Save</Button>
      </>}>
      <Field label="Label" error={tooLong ? "At most 120 characters" : error ?? undefined}>
        <Input value={label} onChange={setLabel} onEnter={() => { if (label.trim() && !tooLong) void save(label); }} autoFocus />
      </Field>
    </Dialog>
  );
}
