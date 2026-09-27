/* The campaign notebook (log.md): rendered with the safe Markdown renderer,
 * an outline of its entries, and an editor that appends a timestamped entry
 * or (confirmed) replaces the whole file. ⌘/Ctrl-Enter saves. */
import { useMemo, useState } from "react";
import { apiPost } from "../../../api/client";
import { invalidate } from "../../../api/query";
import { useShortcut } from "../../../hooks/useShortcut";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Button, Card, CardBody, CardHead, CopyButton, EmptyState, Segmented, Textarea, confirm, toast,
} from "../../../ui";
import { TRACKING_STATE_URL } from "../api";
import { Markdown, outline } from "../markdown";

type Mode = "append" | "replace";

export function NotebookView({ text, editable, title }: { text: string; editable: boolean; title?: string }) {
  const [mode, setMode] = useUrlState<Mode>("nbmode", "append", { parse: (r) => (r === "replace" ? "replace" : r === "append" ? "append" : undefined) });
  const [draft, setDraft] = useState("");
  const [replaceDraft, setReplaceDraft] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [newestFirst, setNewestFirst] = useUrlState("nbnew", false);
  const heads = useMemo(() => outline(text).filter((h) => h.level <= 2), [text]);
  const body = mode === "replace" ? (replaceDraft ?? text) : draft;

  async function save() {
    if (!body.trim() && mode === "append") return;
    if (mode === "replace" && !(await confirm({ title: "Replace the whole notebook?",
      message: "log.md is overwritten with the editor's text; earlier entries not in it are lost.", tone: "danger",
      confirmLabel: "Replace log.md" }))) return;
    setBusy(true);
    try {
      await apiPost("/api/tracking/log", { text: body, mode });
      toast.success(mode === "append" ? "Entry added" : "Notebook replaced");
      if (mode === "append") setDraft(""); else setReplaceDraft(null);
      void invalidate(TRACKING_STATE_URL);
    } catch (e) { toast.error(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }
  useShortcut("$mod+Enter", () => { if (editable && body.trim()) { void save(); return true; } return false; },
    { description: "Save the notebook entry", scope: "Tracking", allowInInputs: true, enabled: editable });

  // Newest first: the entries (## headings) in reverse, the preamble kept on top.
  const shown = useMemo(() => {
    if (!newestFirst) return text;
    const parts = text.split(/\n(?=## )/);
    return parts.length > 1 ? [parts[0], ...parts.slice(1).reverse()].join("\n") : text;
  }, [text, newestFirst]);

  return (
    <div className="ops-stack">
      {editable && (
        <Card>
          <CardHead title="Write" right={<Segmented<Mode> size="sm" value={mode} onChange={setMode} aria-label="Edit mode"
            options={[{ value: "append", label: "Append entry" }, { value: "replace", label: "Edit whole file" }]} />} />
          <CardBody className="ops-editor">
            <Textarea value={body} onChange={(v) => (mode === "append" ? setDraft(v) : setReplaceDraft(v))}
              rows={mode === "append" ? 4 : 14} spellCheck
              placeholder={mode === "append" ? "A result, a decision, a job to check… (Markdown; ⌘/Ctrl-Enter saves)" : undefined}
              aria-label={mode === "append" ? "New notebook entry" : "Whole notebook"} />
            <div className="ops-row">
              <Button variant="primary" size="sm" loading={busy} disabled={mode === "append" ? !draft.trim() : replaceDraft == null}
                onClick={save}>{mode === "append" ? "Add entry" : "Replace log.md"}</Button>
              {mode === "replace" && replaceDraft != null && <Button size="sm" variant="ghost" onClick={() => setReplaceDraft(null)}>Discard edits</Button>}
              {mode === "append" && draft.trim() && <span className="ops-dim ops-small">Saved under a new timestamped heading.</span>}
            </div>
            {mode === "append" && draft.trim() && <div className="ops-preview"><Markdown text={draft} /></div>}
          </CardBody>
        </Card>
      )}
      <Card>
        <CardHead title={title ?? "Notebook"} sub={heads.length ? `${Math.max(0, heads.length - 1)} entries` : undefined}
          right={<div className="ops-row">
            <Segmented size="sm" value={newestFirst ? "new" : "old"} onChange={(v) => setNewestFirst(v === "new")} aria-label="Order"
              options={[{ value: "old", label: "Oldest first" }, { value: "new", label: "Newest first" }]} />
            <CopyButton value={() => text} label="Copy log.md" />
          </div>} />
        <CardBody>
          {!text.trim() ? <EmptyState compact icon="info" title="The notebook is empty" /> : (
            <div className="ops-notebook">
              <Markdown text={shown} className="ops-notebook__doc" />
              {heads.length > 2 && (
                <nav className="ops-outline" aria-label="Notebook entries">
                  {(newestFirst ? [...heads].reverse() : heads).map((h) => (
                    <a key={h.id} href={`#${h.id}`} onClick={(e) => {
                      e.preventDefault();
                      document.getElementById(h.id)?.scrollIntoView({ block: "start", behavior: "smooth" });
                    }}>{h.text}</a>
                  ))}
                </nav>
              )}
            </div>
          )}
        </CardBody>
      </Card>
    </div>
  );
}
