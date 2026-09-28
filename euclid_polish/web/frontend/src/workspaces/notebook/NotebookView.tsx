/* The campaign notebook (log.md): an editor that appends a timestamped
 * entry (prefilled when a page's "Log to notebook" sent one) or, confirmed,
 * replaces the whole file (⌘/Ctrl-Enter saves); then the notebook rendered
 * with the safe Markdown renderer, NEWEST entry first by default (?nbnew=0
 * for oldest first), its size and date span in the card head, and a compact
 * date outline (a "Jump to a day" menu, the day list beside the text on a
 * wide page). */
import { useMemo, useState } from "react";
import { apiPost } from "../../api/client";
import { invalidate } from "../../api/query";
import { useShortcut } from "../../hooks/useShortcut";
import { useUrlState } from "../../hooks/useUrlState";
import {
  Button, Card, CardBody, CardHead, CopyButton, EmptyState, Segmented, Select, Textarea, confirm, toast,
} from "../../ui";
import { TRACKING_STATE_URL } from "./api";
import { Markdown, outline } from "./markdown";
import { notebookDays, notebookOrder, type NotebookDay } from "./model";

/** Scroll an entry to the top of the stage (instantly under reduced motion). */
function jumpTo(id: string) {
  const reduce = typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  document.getElementById(id)?.scrollIntoView({ block: "start", behavior: reduce ? "auto" : "smooth" });
}

/** The day list beside the notebook, grouped by month. */
function DayOutline({ days }: { days: NotebookDay[] }) {
  const months: { month: string; days: NotebookDay[] }[] = [];
  for (const d of days) {
    const last = months[months.length - 1];
    if (last?.month === d.month) last.days.push(d); else months.push({ month: d.month, days: [d] });
  }
  return (
    <nav className="nb-outline" aria-label="Notebook entries by day">
      {months.map((m) => (
        <div key={m.month} className="nb-outline__month">
          <span className="nb-outline__label">{m.month}</span>
          <div className="nb-outline__days">
            {m.days.map((d) => (
              <a key={d.day} href={`#${d.id}`} title={`${d.day}: ${d.count} entr${d.count === 1 ? "y" : "ies"}`}
                onClick={(e) => { e.preventDefault(); jumpTo(d.id); }}>
                {d.label}{d.count > 1 && <span className="nb-outline__n">{d.count}</span>}
              </a>
            ))}
          </div>
        </div>
      ))}
    </nav>
  );
}

type Mode = "append" | "replace";

export function NotebookView({ text, editable, title, initialDraft = "", prefillFrom, onAppended }: {
  text: string; editable: boolean; title?: string;
  /** A prefilled entry (a page's "Log to notebook"), edited before it is added. */
  initialDraft?: string;
  /** Where the prefill came from ("Models › Leaderboard"), said above the editor. */
  prefillFrom?: string;
  onAppended?: () => void;
}) {
  const [mode, setMode] = useUrlState<Mode>("nbmode", "append", { parse: (r) => (r === "replace" ? "replace" : r === "append" ? "append" : undefined) });
  const [draft, setDraft] = useState(initialDraft);
  const [replaceDraft, setReplaceDraft] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [newestFirst, setNewestFirst] = useUrlState("nbnew", true);
  // Newest first: the entries (## headings) in reverse, the preamble kept on top.
  const shown = useMemo(() => notebookOrder(text, newestFirst), [text, newestFirst]);
  const heads = useMemo(() => outline(shown).filter((h) => h.level <= 2), [shown]);
  const days = useMemo(() => notebookDays(heads), [heads]);
  const entries = heads.filter((h) => h.level === 2).length;
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
      if (mode === "append") { setDraft(""); onAppended?.(); } else setReplaceDraft(null);
      void invalidate(TRACKING_STATE_URL);
    } catch (e) { toast.error(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }
  useShortcut("$mod+Enter", () => { if (editable && body.trim()) { void save(); return true; } return false; },
    { description: "Save the notebook entry", scope: "Notebook", allowInInputs: true, enabled: editable });

  return (
    <div className="nb-stack">
      {editable && (
        <Card>
          <CardHead title={mode === "append" ? "New entry" : "Edit log.md"}
            sub={mode === "append" && initialDraft && draft === initialDraft ? `Prefilled${prefillFrom ? ` from ${prefillFrom}` : ""}: edit it, then add it` : undefined}
            right={<Segmented<Mode> size="sm" value={mode} onChange={setMode} aria-label="Edit mode"
            options={[{ value: "append", label: "Append entry" }, { value: "replace", label: "Edit whole file" }]} />} />
          <CardBody className="nb-editor">
            <Textarea value={body} onChange={(v) => (mode === "append" ? setDraft(v) : setReplaceDraft(v))}
              rows={mode === "append" ? 4 : 14} spellCheck
              placeholder={mode === "append" ? "A result, a decision, a job to check… (Markdown; ⌘/Ctrl-Enter saves)" : undefined}
              aria-label={mode === "append" ? "New notebook entry" : "Whole notebook"} />
            <div className="nb-row">
              <Button variant="primary" size="sm" loading={busy} disabled={mode === "append" ? !draft.trim() : replaceDraft == null}
                onClick={save}>{mode === "append" ? "Add entry" : "Replace log.md"}</Button>
              {mode === "replace" && replaceDraft != null && <Button size="sm" variant="ghost" onClick={() => setReplaceDraft(null)}>Discard edits</Button>}
              {mode === "append" && draft.trim() && <span className="nb-dim nb-small">Saved under a new timestamped heading.</span>}
            </div>
            {mode === "append" && draft.trim() && <div className="nb-preview"><Markdown text={draft} /></div>}
          </CardBody>
        </Card>
      )}
      <Card>
        <CardHead title={title ?? "Notebook"} sub={entries ? `${entries} entries${days.length ? `, ${days[newestFirst ? days.length - 1 : 0].day} to ${days[newestFirst ? 0 : days.length - 1].day}` : ""}` : undefined}
          right={<div className="nb-row">
            <Segmented size="sm" value={newestFirst ? "new" : "old"} onChange={(v) => setNewestFirst(v === "new")} aria-label="Order"
              options={[{ value: "new", label: "Newest first" }, { value: "old", label: "Oldest first" }]} />
            {days.length > 1 && (
              <Select size="sm" aria-label="Jump to a day" placeholder="Jump to a day…" value=""
                onChange={(id) => { if (id) jumpTo(id); }}
                options={days.map((d) => ({ value: d.id, label: `${d.label}, ${d.day.slice(0, 4)}`, hint: `${d.count} entr${d.count === 1 ? "y" : "ies"}` }))} />
            )}
            <CopyButton value={() => text} label="Copy log.md" />
          </div>} />
        <CardBody>
          {!text.trim() ? <EmptyState compact icon="info" title="The notebook is empty" /> : (
            <div className="nb-notebook">
              <Markdown text={shown} className="nb-notebook__doc" />
              {days.length > 1 && <DayOutline days={days} />}
            </div>
          )}
        </CardBody>
      </Card>
    </div>
  );
}
