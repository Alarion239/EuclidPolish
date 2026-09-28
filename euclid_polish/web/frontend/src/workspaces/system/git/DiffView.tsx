/* Unified-diff viewer: per-file cards with +/− counts, numbered old/new
 * columns, add/delete/hunk rows in the status tints, optional wrapping. */
import { useMemo, useState } from "react";
import { Badge, EmptyState, Switch } from "../../../ui";
import { diffStat, parseDiff, type DiffFile } from "../diff";

function FileDiff({ file, wrap }: { file: DiffFile; wrap: boolean }) {
  const [open, setOpen] = useState(true);
  const body = file.lines.filter((l) => l.kind !== "meta");
  return (
    <section className={`sys-diff__file${wrap ? " sys-diff--wrap" : ""}`}>
      <header className="sys-diff__head">
        <button type="button" className="sys-linkbtn" aria-expanded={open} onClick={() => setOpen(!open)}>
          {open ? "▾" : "▸"} {file.oldPath ? `${file.oldPath} → ${file.path}` : file.path || "(diff)"}
        </button>
        <span className="sys-spacer" />
        {file.binary && <Badge size="sm">binary</Badge>}
        <span className="sys-diff__add">+{file.added}</span>
        <span className="sys-diff__del">−{file.removed}</span>
      </header>
      {open && body.length > 0 && (
        <div className="sys-diff__body">
          <table className="sys-diff__table">
            <tbody>
              {body.map((l, i) => (
                <tr key={i} className={`sys-diff__row--${l.kind}`}>
                  <td className="sys-diff__num">{l.old ?? ""}</td>
                  <td className="sys-diff__num">{l.new ?? ""}</td>
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
    <div className="sys-diff">
      <div className="sys-row">
        <span className="sys-dim sys-small">{files.length} file{files.length === 1 ? "" : "s"} ·{" "}
          <span className="sys-diff__add">+{stat.added}</span> <span className="sys-diff__del">−{stat.removed}</span></span>
        <span className="sys-spacer" />
        <Switch size="sm" checked={wrap} onChange={setWrap}>Wrap</Switch>
      </div>
      {files.map((f, i) => <FileDiff key={`${f.path}:${i}`} file={f} wrap={wrap} />)}
    </div>
  );
}
