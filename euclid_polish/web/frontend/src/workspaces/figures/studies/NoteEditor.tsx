/* The study's note: its only editable text besides its named selections
 * (the frozen numbers never change). Saved only on "Save note". */
import { useState } from "react";
import { apiPost } from "../../../api/client";
import { Button, Field, Textarea, toast } from "../../../ui";
import { studyUrl, type StudyDetail } from "./api";

export function NoteEditor({ detail, onSaved }: { detail: StudyDetail; onSaved: () => void }) {
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
