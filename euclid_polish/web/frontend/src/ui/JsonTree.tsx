/* JsonTree: collapsible view of any JSON-like value (API payloads, job
   results, origin.json, FITS headers). Nested plain lists: objects/arrays
   are disclosure buttons (aria-expanded + aria-controls while open), so Tab
   and Enter/Space are the whole keyboard model. Deliberately NOT ARIA
   tree/treeitem: those promise arrow-key navigation this view does not have.
   Long collections show the first `pageSize` children with "show more";
   each node can copy its value or its path. */
import { useId, useState, type ReactNode } from "react";
import { CopyButton } from "./display";
import { Icon } from "./icons";
import { cx } from "./slot";

type Json = unknown;

const isObj = (v: Json): v is Record<string, Json> => v !== null && typeof v === "object" && !Array.isArray(v);
const isBranch = (v: Json) => Array.isArray(v) || isObj(v);

/** JS-style access path: `a.b[3]["odd key"]`. */
export function jsonPath(parts: readonly (string | number)[]): string {
  return parts.map((p, i) => {
    if (typeof p === "number") return `[${p}]`;
    if (/^[A-Za-z_$][\w$]*$/.test(p)) return i === 0 ? p : `.${p}`;
    return `[${JSON.stringify(p)}]`;
  }).join("");
}

function Scalar({ v }: { v: Json }) {
  if (v === null) return <span className="ui-json__null">null</span>;
  if (v === undefined) return <span className="ui-json__null">undefined</span>;
  if (typeof v === "string") return <span className="ui-json__str">{JSON.stringify(v)}</span>;
  if (typeof v === "number") return <span className="ui-json__num">{String(v)}</span>;
  if (typeof v === "boolean") return <span className="ui-json__bool">{String(v)}</span>;
  return <span className="ui-json__str">{String(v)}</span>;
}

function summary(v: Json): string {
  if (Array.isArray(v)) return `[${v.length}]`;
  if (isObj(v)) { const n = Object.keys(v).length; return `{${n} key${n === 1 ? "" : "s"}}`; }
  return "";
}

function Node(
  { name, value, path, depth, expandDepth, pageSize }: {
    name: ReactNode; value: Json; path: (string | number)[]; depth: number; expandDepth: number;
    pageSize: number;
  },
) {
  const [open, setOpen] = useState(depth < expandDepth);
  const [shown, setShown] = useState(pageSize);
  const childrenId = useId();
  const copyValue = () => {
    try { return typeof value === "string" ? value : JSON.stringify(value, null, 2) ?? String(value); }
    catch { return String(value); }
  };
  const pathText = jsonPath(path);
  if (!isBranch(value)) {
    return (
      <li className="ui-json__row">
        <span className="ui-json__leaf">
          {name != null && <span className="ui-json__key">{name}</span>}
          {name != null && <span className="ui-json__colon">: </span>}
          <Scalar v={value} />
          <span className="ui-json__tools"><CopyButton value={copyValue} label={`Copy ${pathText || "value"}`} /></span>
        </span>
      </li>
    );
  }
  const entries: [string | number, Json][] = Array.isArray(value)
    ? value.map((v, i) => [i, v])
    : Object.entries(value as Record<string, Json>);
  return (
    <li className="ui-json__row">
      <span className="ui-json__branch">
        <button type="button" className="ui-json__toggle" onClick={() => setOpen((o) => !o)}
          aria-expanded={open} aria-controls={open ? childrenId : undefined}>
          <Icon name={open ? "chevronDown" : "chevronRight"} size={12} />
          {name != null ? <span className="ui-json__key">{name}</span> : <span className="sr-only">root</span>}
          <span className="ui-json__sum"> {summary(value)}</span>
        </button>
        <span className="ui-json__tools">
          {pathText && <CopyButton value={pathText} label={`Copy path ${pathText}`} />}
          <CopyButton value={copyValue} label={`Copy ${pathText || "value"} as JSON`} />
        </span>
      </span>
      {open && (
        <ul className="ui-json__children" id={childrenId}>
          {entries.slice(0, shown).map(([k, v]) => (
            <Node key={String(k)} name={Array.isArray(value) ? <span className="ui-json__idx">{k}</span> : String(k)}
              value={v} path={[...path, k]} depth={depth + 1} expandDepth={expandDepth} pageSize={pageSize} />
          ))}
          {entries.length > shown && (
            <li className="ui-json__row">
              <button type="button" className="ui-json__more" onClick={() => setShown((s) => s + pageSize)}>
                show {Math.min(pageSize, entries.length - shown)} more of {entries.length - shown}
              </button>
            </li>
          )}
        </ul>
      )}
    </li>
  );
}

export function JsonTree(
  { data, expandDepth = 1, pageSize = 100, rootLabel, className, "aria-label": ariaLabel }: {
    data: Json;
    /** Levels open initially (0 = collapsed root). */
    expandDepth?: number;
    /** Children rendered per "show more" page. */
    pageSize?: number;
    rootLabel?: ReactNode;
    className?: string;
    "aria-label"?: string;
  },
) {
  return (
    <ul className={cx("ui-json", className)} aria-label={ariaLabel ?? "JSON"}>
      <Node name={rootLabel ?? null} value={data} path={[]} depth={0} expandDepth={expandDepth} pageSize={pageSize} />
    </ul>
  );
}
