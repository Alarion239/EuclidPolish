/* Unified-diff viewer: per-file cards with +/− counts, numbered old/new
 * columns, add/delete/hunk rows in the status tints, optional wrapping. */
import { useMemo, useState } from "react";
import { Badge, EmptyState, Switch } from "../../../ui";
import { diffStat, parseDiff, type DiffFile } from "../diff";

function FileDiff({ file, wrap }: { file: DiffFile; wrap: boolean }) {
  const [open, setOpen] = useState(true);
  const body = file.lines.filter((l) => l.kind !== "meta");
  return (
    <section className={`ops-diff__file${wrap ? " ops-diff--wrap" : ""}`}>
      <header className="ops-diff__head">
        <button type="button" className="ops-linkbtn" aria-expanded={open} onClick={() => setOpen(!open)}>
          {open ? "▾" : "▸"} {file.oldPath ? `${file.oldPath} → ${file.path}` : file.path || "(diff)"}
        </button>
        <span className="ops-spacer" />
        {file.binary && <Badge size="sm">binary</Badge>}
        <span className="ops-diff__add">+{file.added}</span>
        <span className="ops-diff__del">−{file.removed}</span>
      </header>
      {open && body.length > 0 && (
        <div className="ops-diff__body">
          <table className="ops-diff__table">
            <tbody>
              {body.map((l, i) => (
                <tr key={i} className={`ops-diff__row--${l.kind}`}>
                  <td className="ops-diff__num">{l.old ?? ""}</td>
                  <td className="ops-diff__num">{l.new ?? ""}</td>
                  <td>{l.kind === "add" ? "+" : l.kind === "del" ? "−" : l.kind === "ctx" ? " " : ""}{l.text}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

export function DiffView({ text, empty = "No changes." }: { text: string | null | undefined; empty?: string }) {
  const files = useMemo(() => parseDiff(text ?? ""), [text]);
  const [wrap, setWrap] = useState(false);
  if (!files.length) return <EmptyState compact icon="check" title={empty} />;
  const stat = diffStat(files);
  return (
    <div className="ops-diff">
      <div className="ops-row">
        <span className="ops-dim ops-small">{files.length} file{files.length === 1 ? "" : "s"} ·{" "}
          <span className="ops-diff__add">+{stat.added}</span> <span className="ops-diff__del">−{stat.removed}</span></span>
        <span className="ops-spacer" />
        <Switch size="sm" checked={wrap} onChange={setWrap}>Wrap</Switch>
      </div>
      {files.map((f, i) => <FileDiff key={`${f.path}:${i}`} file={f} wrap={wrap} />)}
    </div>
  );
}
