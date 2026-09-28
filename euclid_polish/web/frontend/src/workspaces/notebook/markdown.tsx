/* A tiny, safe Markdown renderer for the tracking notebook (no dependency,
 * no innerHTML: the text becomes React elements, so nothing in it can run).
 *
 * Blocks: ATX headings, paragraphs, `---` rules, `>` quotes, `-`/`*`/`+` and
 * `1.` lists (one level), fenced ``` code, GFM pipe tables. Inline: `code`,
 * **bold**, *italic* / _italic_, [text](url) links (http(s), mailto,
 * site-relative and #anchors only; anything else renders as text).
 *
 * `parseMarkdown` is the pure block parser (unit-tested); `<Markdown>` renders
 * it. Headings get ids (`slugify`) so an outline can link to them. */
import type { ReactNode } from "react";

export type Inline = string;
export type Block =
  | { type: "heading"; level: number; text: Inline; id: string }
  | { type: "paragraph"; text: Inline }
  | { type: "rule" }
  | { type: "quote"; blocks: Block[] }
  | { type: "list"; ordered: boolean; items: Inline[] }
  | { type: "code"; lang: string; text: string }
  | { type: "table"; head: Inline[]; rows: Inline[][] };

export const slugify = (text: string): string =>
  text.toLowerCase().replace(/[`*_[\]()]/g, "").replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "section";

const HEADING = /^(#{1,6})\s+(.*?)\s*#*\s*$/;
const RULE = /^\s{0,3}([-*_])(\s*\1){2,}\s*$/;
const FENCE = /^\s{0,3}(```+|~~~+)\s*([\w+-]*)\s*$/;
const BULLET = /^\s{0,3}[-*+]\s+(.*)$/;
const ORDERED = /^\s{0,3}\d{1,9}[.)]\s+(.*)$/;
const QUOTE = /^\s{0,3}>\s?(.*)$/;
const TABLE_SEP = /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/;

const splitRow = (line: string): string[] =>
  line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());

export function parseMarkdown(src: string): Block[] {
  const lines = src.replace(/\r\n?/g, "\n").split("\n");
  const blocks: Block[] = [];
  const ids = new Map<string, number>();
  let i = 0;
  const uniqueId = (text: string) => {
    const base = slugify(text);
    const n = ids.get(base) ?? 0;
    ids.set(base, n + 1);
    return n ? `${base}-${n + 1}` : base;
  };
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i += 1; continue; }
    const fence = line.match(FENCE);
    if (fence) {
      const close = fence[1];
      const body: string[] = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith(close)) { body.push(lines[i]); i += 1; }
      i += 1;
      blocks.push({ type: "code", lang: fence[2], text: body.join("\n") });
      continue;
    }
    const h = line.match(HEADING);
    if (h) { blocks.push({ type: "heading", level: h[1].length, text: h[2], id: uniqueId(h[2]) }); i += 1; continue; }
    if (RULE.test(line)) { blocks.push({ type: "rule" }); i += 1; continue; }
    if (QUOTE.test(line)) {
      const body: string[] = [];
      while (i < lines.length && QUOTE.test(lines[i])) { body.push(lines[i].match(QUOTE)![1]); i += 1; }
      blocks.push({ type: "quote", blocks: parseMarkdown(body.join("\n")) });
      continue;
    }
    if (line.includes("|") && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1])) {
      const head = splitRow(line);
      const rows: string[][] = [];
      i += 2;
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) { rows.push(splitRow(lines[i])); i += 1; }
      blocks.push({ type: "table", head, rows });
      continue;
    }
    const listRe = BULLET.test(line) ? BULLET : ORDERED.test(line) ? ORDERED : null;
    if (listRe) {
      const items: string[] = [];
      while (i < lines.length) {
        const m = lines[i].match(listRe);
        if (m) { items.push(m[1]); i += 1; continue; }
        // A continuation line (indented, not blank) joins the previous item.
        if (lines[i].trim() && /^\s{2,}\S/.test(lines[i]) && items.length) {
          items[items.length - 1] += ` ${lines[i].trim()}`;
          i += 1;
          continue;
        }
        break;
      }
      blocks.push({ type: "list", ordered: listRe === ORDERED, items });
      continue;
    }
    const para: string[] = [];
    while (i < lines.length && lines[i].trim() && !HEADING.test(lines[i]) && !RULE.test(lines[i])
      && !FENCE.test(lines[i]) && !QUOTE.test(lines[i]) && !BULLET.test(lines[i]) && !ORDERED.test(lines[i])) {
      para.push(lines[i].trim());
      i += 1;
    }
    if (para.length) blocks.push({ type: "paragraph", text: para.join(" ") });
    else i += 1;
  }
  return blocks;
}

/** Only these link targets become anchors; anything else stays text. */
export function safeHref(url: string): string | null {
  const u = url.trim();
  if (u.startsWith("//") || u.includes("\\")) return null;   // protocol-relative / backslash tricks
  if (/^(https?:|mailto:)/i.test(u)) return u;
  if (/^(\/(?!\/)|#|\.\/|\.\.\/)/.test(u)) return u;
  if (/^[\w./-]+$/.test(u) && !u.includes(":")) return u;
  return null;
}

const INLINE = /(`+)([\s\S]*?)\1|\*\*([^*]+?)\*\*|__([^_]+?)__|\*([^*\s][^*]*?)\*|(?<![\w])_([^_\s][^_]*?)_(?![\w])|\[([^\]]+)\]\(([^)\s]+)\)/g;

export function renderInline(text: string, keyBase = "i"): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  let k = 0;
  for (const m of text.matchAll(INLINE)) {
    const at = m.index ?? 0;
    if (at > last) out.push(text.slice(last, at));
    const key = `${keyBase}-${k++}`;
    if (m[1]) out.push(<code key={key}>{m[2]}</code>);
    else if (m[3] || m[4]) out.push(<strong key={key}>{renderInline(m[3] ?? m[4], key)}</strong>);
    else if (m[5] || m[6]) out.push(<em key={key}>{renderInline(m[5] ?? m[6], key)}</em>);
    else if (m[7]) {
      const href = safeHref(m[8]);
      out.push(href
        ? <a key={key} href={href} target={/^https?:/i.test(href) ? "_blank" : undefined}
            rel={/^https?:/i.test(href) ? "noreferrer noopener" : undefined}>{renderInline(m[7], key)}</a>
        : `[${m[7]}](${m[8]})`);
    }
    last = at + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

function renderBlocks(blocks: Block[], keyBase: string): ReactNode[] {
  return blocks.map((b, n) => {
    const key = `${keyBase}-${n}`;
    switch (b.type) {
      case "heading": {
        const Tag = `h${Math.min(6, b.level + 1)}` as "h2";
        return <Tag key={key} id={b.id} className={`md-h md-h${b.level}`}>{renderInline(b.text, key)}</Tag>;
      }
      case "paragraph": return <p key={key}>{renderInline(b.text, key)}</p>;
      case "rule": return <hr key={key} />;
      case "quote": return <blockquote key={key}>{renderBlocks(b.blocks, key)}</blockquote>;
      case "list": {
        const items = b.items.map((it, j) => <li key={j}>{renderInline(it, `${key}-${j}`)}</li>);
        return b.ordered ? <ol key={key}>{items}</ol> : <ul key={key}>{items}</ul>;
      }
      case "code": return <pre key={key} className="md-code"><code>{b.text}</code></pre>;
      case "table":
        return (
          <div key={key} className="md-table">
            <table>
              <thead><tr>{b.head.map((c, j) => <th key={j}>{renderInline(c, `${key}-h${j}`)}</th>)}</tr></thead>
              <tbody>{b.rows.map((r, j) => (
                <tr key={j}>{r.map((c, x) => <td key={x}>{renderInline(c, `${key}-${j}-${x}`)}</td>)}</tr>
              ))}</tbody>
            </table>
          </div>
        );
      default: return null;
    }
  });
}

export function Markdown({ text, className }: { text: string; className?: string }) {
  return <div className={`md${className ? ` ${className}` : ""}`}>{renderBlocks(parseMarkdown(text), "b")}</div>;
}

/** The headings of a document (for an outline / jump list). */
export function outline(text: string): { level: number; text: string; id: string }[] {
  return parseMarkdown(text)
    .filter((b): b is Extract<Block, { type: "heading" }> => b.type === "heading")
    .map((b) => ({ level: b.level, text: b.text, id: b.id }));
}
