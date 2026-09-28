/* The head every plate shares: its title, the one caption line ("made with
 * <model> · current/stale · rendered N d ago", plateStatus.ts) and the
 * plate's own tools (resolution, downloads, the tab its data comes from). */
import type { ReactNode } from "react";
import { Caption, Tooltip } from "../../../ui";
import { captionParts, type PlateCaption } from "./plateStatus";

export function PlateCaptionLine({ caption, prefix }: { caption: PlateCaption; prefix?: string[] }) {
  const parts = [...(prefix ?? []), ...captionParts(caption)];
  if (!parts.length) return null;
  const stateWord = caption.state === "stale" ? "stale" : caption.state === "not-active" ? "not active" : null;
  return (
    <Caption className="fig-plate__caption">
      {parts.map((p, i) => {
        const sep = i ? " · " : "";
        if (p === stateWord && caption.tone === "warn") {
          const word = <span className="fig-plate__state" data-tone="warn">{p}</span>;
          return (
            <span key={p}>{sep}
              {caption.reason
                ? <Tooltip content={caption.reason}><span tabIndex={0} className="fig-plate__reason">{word}</span></Tooltip>
                : word}
            </span>
          );
        }
        return <span key={`${i}-${p}`}>{sep}{p}</span>;
      })}
    </Caption>
  );
}

export function PlateHead({ id, title, sub, caption, prefix, tools }: {
  id: string; title: string; sub?: ReactNode; caption: PlateCaption; prefix?: string[]; tools?: ReactNode;
}) {
  return (
    <header className="fig-plate__head">
      <div className="fig-plate__heading">
        <h2 id={`fig-plate-${id}`}>{title}</h2>
        {sub && <p className="muted fig-plate__sub">{sub}</p>}
        <PlateCaptionLine caption={caption} prefix={prefix} />
      </div>
      {tools && <div className="fig-plate__tools">{tools}</div>}
    </header>
  );
}
