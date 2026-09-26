/* LogView: a log/terminal output panel with search (highlight, match count,
   next/previous, only-matching filter), follow (sticks to the end while you
   are at the end; scrolling up pauses it), wrap toggle, copy and download.
   LogTail is the compat, toolbar-less variant with the same follow logic. */
import {
  useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode,
  type RefObject,
} from "react";
import { CopyButton } from "./display";
import { Input } from "./controls";
import { downloadText, safeFileName } from "./download";
import { IconButton } from "./Button";
import { cx } from "./slot";

export type LogMatch = { line: number; start: number; end: number };

/** Case-insensitive plain-text matches of `query` in `lines` (pure). */
export function findMatches(lines: readonly string[], query: string, limit = 5000): LogMatch[] {
  const q = query.toLowerCase();
  if (!q) return [];
  const out: LogMatch[] = [];
  for (let i = 0; i < lines.length && out.length < limit; i++) {
    const l = lines[i].toLowerCase();
    let at = l.indexOf(q);
    while (at >= 0 && out.length < limit) {
      out.push({ line: i, start: at, end: at + q.length });
      at = l.indexOf(q, at + Math.max(1, q.length));
    }
  }
  return out;
}

const NEAR_END_PX = 24;

function useFollow(el: RefObject<HTMLElement | null>, text: string, follow: boolean) {
  const atEnd = useRef(true);
  const [paused, setPaused] = useState(false);
  useLayoutEffect(() => {
    const node = el.current;
    if (node && follow && atEnd.current) node.scrollTop = node.scrollHeight;
  }, [el, text, follow]);
  const onScroll = () => {
    const node = el.current;
    if (!node) return;
    const end = node.scrollHeight - node.scrollTop - node.clientHeight <= NEAR_END_PX;
    atEnd.current = end;
    setPaused(!end);
  };
  const jumpToEnd = () => {
    const node = el.current;
    if (!node) return;
    node.scrollTop = node.scrollHeight;
    atEnd.current = true;
    setPaused(false);
  };
  return { onScroll, paused, jumpToEnd };
}

/** Compat: plain scrolling log tail (no toolbar), following the end. */
export function LogTail({ text, style, className }: { text: string; style?: CSSProperties; className?: string }) {
  const ref = useRef<HTMLPreElement>(null);
  const { onScroll } = useFollow(ref, text, true);
  return (
    <pre ref={ref} className={cx("ui-logtail", className)} style={style} onScroll={onScroll} tabIndex={0}>
      {text || "(no output)"}
    </pre>
  );
}

export function LogView(
  { text, title, follow: followProp = true, maxHeight = 360, wrap: wrapDefault = true, exportName,
    toolbar, empty = "(no output)", className, style }: {
    text: string | null | undefined;
    title?: ReactNode;
    /** Auto-scroll to the end on new output while at the end (default true). */
    follow?: boolean;
    maxHeight?: number | string;
    wrap?: boolean;
    /** Enables "download .log" with this file name. */
    exportName?: string;
    /** Extra controls on the right of the toolbar. */
    toolbar?: ReactNode;
    empty?: ReactNode;
    className?: string;
    style?: CSSProperties;
  },
) {
  const body = text ?? "";
  const [query, setQuery] = useState("");
  const [onlyMatching, setOnlyMatching] = useState(false);
  const [wrap, setWrap] = useState(wrapDefault);
  const [follow, setFollow] = useState(followProp);
  const [cursor, setCursor] = useState(0);
  useEffect(() => setFollow(followProp), [followProp]);
  const ref = useRef<HTMLPreElement>(null);
  const { onScroll, paused, jumpToEnd } = useFollow(ref, body, follow && !query);
  const lines = useMemo(() => body.split("\n"), [body]);
  const matches = useMemo(() => findMatches(lines, query), [lines, query]);
  useEffect(() => setCursor(0), [query]);
  const active = matches.length ? matches[Math.min(cursor, matches.length - 1)] : null;

  useEffect(() => {
    if (!active) return;
    const node = ref.current?.querySelector<HTMLElement>(`[data-match="${active.line}:${active.start}"]`);
    node?.scrollIntoView?.({ block: "nearest" });
  }, [active]);

  const step = (d: number) => {
    if (!matches.length) return;
    setCursor((c) => (c + d + matches.length) % matches.length);
  };

  const content = useMemo(() => {
    if (!body) return null;
    if (!query) return body;
    const byLine = new Map<number, LogMatch[]>();
    for (const m of matches) {
      const arr = byLine.get(m.line);
      if (arr) arr.push(m); else byLine.set(m.line, [m]);
    }
    const out: ReactNode[] = [];
    lines.forEach((line, i) => {
      const ms = byLine.get(i);
      if (onlyMatching && !ms) return;
      if (!ms) { out.push(line, "\n"); return; }
      let at = 0;
      const parts: ReactNode[] = [];
      for (const m of ms) {
        if (m.start > at) parts.push(line.slice(at, m.start));
        const isActive = active != null && active.line === m.line && active.start === m.start;
        parts.push(
          <mark key={m.start} data-match={`${m.line}:${m.start}`} className={cx("ui-log__hit", isActive && "is-active")}>
            {line.slice(m.start, m.end)}
          </mark>,
        );
        at = m.end;
      }
      if (at < line.length) parts.push(line.slice(at));
      out.push(<span key={`l${i}`} className="ui-log__line">{parts}</span>, "\n");
    });
    return out;
  }, [body, query, lines, matches, onlyMatching, active]);

  return (
    <div className={cx("ui-log", className)} style={style}>
      <div className="ui-log__bar">
        {title != null && <span className="ui-log__title">{title}</span>}
        <Input type="search" value={query} onChange={setQuery} size="sm" icon="search" clearable placeholder="Search log"
          aria-label="Search log" className="ui-log__search"
          onKeyDown={(e) => {
            if (e.key === "Enter") { e.preventDefault(); step(e.shiftKey ? -1 : 1); }
            if (e.key === "Escape" && query) { e.preventDefault(); setQuery(""); }
          }} />
        {query && (
          <span className="ui-log__count mono" aria-live="polite">
            {matches.length ? `${Math.min(cursor, matches.length - 1) + 1}/${matches.length}` : "0 matches"}
          </span>
        )}
        {query && <>
          <IconButton size="sm" icon="chevronUp" label="Previous match" onClick={() => step(-1)} disabled={!matches.length} />
          <IconButton size="sm" icon="chevronDown" label="Next match" onClick={() => step(1)} disabled={!matches.length} />
          <IconButton size="sm" icon="filter" label="Only matching lines" pressed={onlyMatching}
            onClick={() => setOnlyMatching((v) => !v)} />
        </>}
        <span className="ui-log__spacer" />
        {toolbar}
        <IconButton size="sm" icon="arrowDown" label={follow ? "Following the end (click to stop)" : "Follow the end"}
          pressed={follow} onClick={() => { setFollow((f) => !f); if (!follow) jumpToEnd(); }} />
        <button type="button" className="ui-chip ui-log__wrap" data-on={wrap} aria-pressed={wrap}
          onClick={() => setWrap((w) => !w)}>wrap</button>
        <CopyButton value={() => body} label="Copy log" />
        {exportName && (
          <IconButton size="sm" icon="download" label="Download log"
            onClick={() => downloadText(`${safeFileName(exportName, "log")}.log`, body)} />
        )}
      </div>
      <pre ref={ref} className={cx("ui-log__text", !wrap && "is-nowrap")} style={{ maxHeight }}
        onScroll={onScroll} tabIndex={0} aria-label={typeof title === "string" ? title : "Log output"}>
        {content ?? <span className="ui-log__empty">{empty}</span>}
      </pre>
      {follow && paused && !query && body && (
        <button type="button" className="ui-log__jump" onClick={jumpToEnd}>Jump to end ↓</button>
      )}
    </div>
  );
}
