/* DataTable: the console's table for anything list-shaped (members, tiles,
   runs, stars, jobs). Virtualised (TanStack Virtual) so thousands of rows
   stay fast; multi-key sort, a filter mini-language, column visibility,
   checkbox selection with shift-ranges, keyboard row navigation, row →
   inspector, CSV export, sticky header, optional URL-bound sort/filter.

   The pure model (sorting, filtering, CSV, ranges, widths) is `tableModel.ts`.
   Rows must be distinct objects (or unique primitives) with unique keys. */
import { useVirtualizer } from "@tanstack/react-virtual";
import {
  useCallback, useId, useLayoutEffect, useMemo, useRef, useState,
  type KeyboardEvent, type MouseEvent, type ReactNode,
} from "react";
import { formatNumber } from "../format";
import { useUrlState } from "../hooks/useUrlState";
import { sameTarget, useInspector, type InspectTarget } from "../state/inspector";
import { Button, IconButton } from "./Button";
import { Checkbox, Input } from "./controls";
import {
  columnValue, estimateWidths, filterRows, headerText, nextSort, parseSort, rangeKeys, serializeSort,
  sortRows, toCSV, type DataColumn, type SortState,
} from "./tableModel";
import { EmptyState, Skeleton } from "./display";
import { downloadText, safeFileName } from "./download";
import { Icon } from "./icons";
import { Menu, type MenuItem } from "./overlays";
import { cx } from "./slot";

export type { DataColumn, SortSpec, SortState } from "./tableModel";

export type DataTableProps<T> = {
  rows: readonly T[];
  columns: DataColumn<T>[];
  /** Unique, stable key per row (selection, cursor, React keys). */
  rowKey: (row: T, index: number) => string;
  "aria-label"?: string;
  /** Visually hidden table caption (screen readers). */
  caption?: string;

  /* sorting (controlled with `sort`, or uncontrolled from `defaultSort`) */
  sort?: SortState;
  defaultSort?: SortState;
  onSortChange?: (sort: SortState) => void;

  /* filtering (see `parseFilter` for the syntax) */
  filter?: string;
  defaultFilter?: string;
  onFilterChange?: (query: string) => void;
  /** Show the filter box (default true). */
  searchable?: boolean;
  filterPlaceholder?: string;

  /* selection */
  selectable?: boolean;
  selected?: readonly string[];
  defaultSelected?: readonly string[];
  /** Keys in view order (then any selected keys filtered out of view), and
   *  their rows (same length: keys whose row left `rows` are dropped). */
  onSelectedChange?: (keys: string[], rows: T[]) => void;

  /* columns */
  columnVisibility?: Record<string, boolean>;
  onColumnVisibilityChange?: (visibility: Record<string, boolean>) => void;

  /* rows */
  /** Row click / Enter. `index` is the row's index in `rows`. */
  onRowClick?: (row: T, index: number) => void;
  /** Row click / Enter opens this entity in the inspector (and highlights the inspected row). */
  inspect?: (row: T) => InspectTarget | null;
  /** Explicitly highlighted row (overrides the inspector highlight). */
  activeKey?: string | null;
  rowClassName?: (row: T) => string | undefined;

  /* export + layout */
  /** Adds a CSV button; the file is `<exportName>.csv`. */
  exportName?: string;
  /** Scroll viewport max-height (px or CSS). "auto" = no inner scroll (disables virtualisation). */
  height?: number | string;
  /** Row height estimate in px (default 34; rows are measured). */
  rowHeight?: number;
  /** true / false, or the row count above which rows are virtualised (default 150). */
  virtualize?: boolean | number;
  /** Extra controls at the right of the toolbar. */
  toolbar?: ReactNode;
  /** Hide the whole toolbar (filter, count, columns, CSV). */
  hideToolbar?: boolean;
  empty?: ReactNode;
  loading?: boolean;
  dense?: boolean;
  className?: string;
  /** Keep sort and filter in the URL as `<urlKey>.sort` / `<urlKey>.q` (needs a router). */
  urlKey?: string;
};

export function DataTable<T>(props: DataTableProps<T>) {
  if (props.urlKey) return <UrlDataTable {...props} urlKey={props.urlKey} />;
  return <DataTableView {...props} />;
}

function UrlDataTable<T>(props: DataTableProps<T> & { urlKey: string }) {
  const { urlKey, onFilterChange, onSortChange } = props;
  const [q, setQ] = useUrlState(`${urlKey}.q`, props.defaultFilter ?? "");
  const [sortRaw, setSortRaw] = useUrlState(`${urlKey}.sort`, serializeSort(props.defaultSort ?? []));
  const sort = useMemo(() => parseSort(sortRaw), [sortRaw]);
  return (
    <DataTableView {...props}
      filter={props.filter ?? q}
      onFilterChange={(v) => { setQ(v); onFilterChange?.(v); }}
      sort={props.sort ?? sort}
      onSortChange={(s) => { setSortRaw(serializeSort(s)); onSortChange?.(s); }} />
  );
}

const INTERACTIVE = "button, a, input, select, textarea, label, [role='button'], [role='checkbox'], [data-row-ignore]";

function defaultCell(v: unknown): ReactNode {
  if (v == null || (typeof v === "number" && Number.isNaN(v))) return <span className="ui-dt__nil">—</span>;
  if (typeof v === "number") return formatNumber(v);
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (v instanceof Date) return v.toISOString().replace("T", " ").slice(0, 19);
  if (Array.isArray(v)) return v.join(", ");
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function useControlled<V>(prop: V | undefined, initial: V | (() => V)): [V, (v: V) => void, boolean] {
  const [inner, setInner] = useState<V>(initial);
  const controlled = prop !== undefined;
  return [controlled ? (prop as V) : inner, (v: V) => { if (!controlled) setInner(v); }, controlled];
}

function DataTableView<T>(p: DataTableProps<T>) {
  const {
    rows, columns, rowKey, selectable = false, searchable = true, exportName, hideToolbar = false,
    dense = false, rowHeight, virtualize, height = 560, loading = false,
  } = p;
  const uid = useId().replace(/:/g, "");

  /* ── state (controlled or not) ── */
  const [sort, setSortInner] = useControlled<SortState>(p.sort, () => p.defaultSort ?? []);
  const setSort = (s: SortState) => { setSortInner(s); p.onSortChange?.(s); };
  const [filter, setFilterInner] = useControlled<string>(p.filter, () => p.defaultFilter ?? "");
  const setFilter = (q: string) => { setFilterInner(q); p.onFilterChange?.(q); };
  const [selected, setSelectedInner] = useControlled<readonly string[]>(p.selected, () => p.defaultSelected ?? []);
  const [visInner, setVisInner] = useState<Record<string, boolean>>({});
  const visibility = p.columnVisibility ?? visInner;
  const isVisible = (c: DataColumn<T>) => visibility[c.id] ?? !c.hidden;
  const visibleCols = columns.filter(isVisible);
  const [cursor, setCursor] = useState<string | null>(null);
  /* The anchor is the row last clicked, activated (Enter) or toggled; a Shift
     range runs from it. Moving the cursor alone does not move it. */
  const anchor = useRef<string | null>(null);
  /* The selection a shift-range is added to: captured when a range starts
     from the anchor, cleared by every other selection change and whenever the
     anchor moves, so moving the range end back (Shift+↑ after Shift+↓)
     shrinks it and a new range keeps the earlier ones. */
  const rangeBase = useRef<Set<string> | null>(null);
  const setAnchor = (key: string | null) => { anchor.current = key; rangeBase.current = null; };

  /* ── derived rows ── */
  const entries = useMemo(() => {
    const byRow = new Map<T, { key: string; index: number }>();
    const byKey = new Map<string, { row: T; index: number }>();
    rows.forEach((row, index) => {
      const key = rowKey(row, index);
      byRow.set(row, { key, index });
      byKey.set(key, { row, index });
    });
    return { byRow, byKey };
    // rowKey is usually an inline arrow; keys are a function of the rows.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows]);
  const filtered = useMemo(() => filterRows(rows, columns, filter), [rows, columns, filter]);
  const view = useMemo(() => sortRows(filtered, columns, sort), [filtered, columns, sort]);
  const viewKeys = useMemo(() => view.map((r) => entries.byRow.get(r)!.key), [view, entries]);
  const viewIndex = useMemo(() => new Map(viewKeys.map((k, i) => [k, i])), [viewKeys]);
  // Selected keys whose rows are gone (a data refresh) are ignored for display,
  // counts and callbacks; they are dropped from state on the next change.
  const liveSelected = useMemo(() => selected.filter((k) => entries.byKey.has(k)), [selected, entries]);
  const selSet = useMemo(() => new Set(liveSelected), [liveSelected]);
  const widths = useMemo(() => estimateWidths(rows, columns), [rows, columns]);

  /* ── selection ── */
  const commitSelection = useCallback((next: Set<string>, keepRange = false) => {
    if (!keepRange) rangeBase.current = null;
    // View order first, then selected keys that are filtered out of view;
    // keys without a row are dropped, so `keys` and `rows` always pair up.
    const keys: string[] = [];
    const seen = new Set<string>();
    for (const k of viewKeys) if (next.has(k)) { keys.push(k); seen.add(k); }
    for (const k of [...liveSelected, ...next]) {
      if (next.has(k) && !seen.has(k) && entries.byKey.has(k)) { keys.push(k); seen.add(k); }
    }
    setSelectedInner(keys);
    p.onSelectedChange?.(keys, keys.map((k) => entries.byKey.get(k)!.row));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewKeys, liveSelected, entries, p.onSelectedChange]);

  const toggleKey = (key: string, shift: boolean) => {
    const next = new Set(selSet);
    if (shift && anchor.current != null) {
      const span = rangeKeys(viewKeys, anchor.current, key);
      const on = !selSet.has(key);
      for (const k of span) { if (on) next.add(k); else next.delete(k); }
    } else {
      if (next.has(key)) next.delete(key); else next.add(key);
      setAnchor(key);
    }
    commitSelection(next);
  };
  /** Shift-click / Shift+navigation: the range anchor → `key` on top of the
   *  selection that existed when the range started (rebuilt, not grown). */
  const extendTo = (key: string) => {
    const from = anchor.current ?? cursor ?? key;
    if (anchor.current == null) anchor.current = from;
    if (rangeBase.current == null) rangeBase.current = new Set(selSet);
    commitSelection(new Set([...rangeBase.current, ...rangeKeys(viewKeys, from, key)]), true);
  };
  const allInView = viewKeys.length > 0 && viewKeys.every((k) => selSet.has(k));
  const someInView = !allInView && viewKeys.some((k) => selSet.has(k));
  const toggleAll = () => {
    const next = new Set(selSet);
    if (allInView) for (const k of viewKeys) next.delete(k);
    else for (const k of viewKeys) next.add(k);
    commitSelection(next);
  };

  /* ── activation / inspector ──
     `inspect` writes the inspector STORE (state/inspector.ts). The shell
     (WP-F3, app/inspector.ts) owns the `?inspect=kind:id` URL and must mirror
     store → URL (and URL → store) so a row opened here is shareable. */
  const inspected = useInspector((s) => s.current);
  /** A plain click or Enter: the row becomes the anchor of the next Shift range. */
  const activate = (key: string) => {
    const e = entries.byKey.get(key);
    if (!e) return;
    setAnchor(key);
    p.onRowClick?.(e.row, e.index);
    const target = p.inspect?.(e.row);
    if (target) useInspector.getState().show(target);
    if (!p.onRowClick && !p.inspect && selectable) toggleKey(key, false);
  };
  const isActive = (row: T, key: string) => {
    if (p.activeKey !== undefined) return p.activeKey === key;
    if (!p.inspect || !inspected) return false;
    return sameTarget(p.inspect(row), inspected);
  };

  /* ── virtualisation ── */
  const scrollRef = useRef<HTMLDivElement>(null);
  const headRef = useRef<HTMLTableSectionElement>(null);
  const rowH = rowHeight ?? (dense ? 28 : 34);
  const threshold = typeof virtualize === "number" ? virtualize : 150;
  const virtualOn = height !== "auto" && virtualize !== false && (virtualize === true || view.length > threshold);
  const [headH, setHeadH] = useState(rowH);
  useLayoutEffect(() => {
    // The sticky header's height offsets the virtual rows (scrollMargin).
    const el = headRef.current;
    if (!el) return;
    const read = () => {
      const h = el.getBoundingClientRect().height;
      if (h > 0) setHeadH((old) => (Math.abs(old - h) > 0.5 ? h : old));
    };
    read();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const virtualizer = useVirtualizer({
    count: view.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => rowH,
    overscan: 10,
    enabled: virtualOn,
    getItemKey: (i) => viewKeys[i] ?? i,
    scrollMargin: headH,
    scrollPaddingStart: headH,
    initialRect: { width: 800, height: typeof height === "number" ? height : 480 },
    measureElement: (el, entry) => {
      const h = entry?.borderBoxSize?.[0]?.blockSize ?? el.getBoundingClientRect().height;
      return h > 0 ? h : rowH;
    },
  });
  const items = virtualOn ? virtualizer.getVirtualItems() : null;
  const indices = items ? items.map((it) => it.index) : view.map((_, i) => i);
  const padTop = items && items.length ? items[0].start - headH : 0;
  const padBottom = items && items.length
    ? Math.max(0, virtualizer.getTotalSize() - (items[items.length - 1].end - headH)) : 0;

  const scrollToIndex = (i: number) => {
    if (virtualOn) { virtualizer.scrollToIndex(i, { align: "auto" }); return; }
    const el = scrollRef.current?.querySelector<HTMLElement>(`[data-vindex="${i}"]`);
    el?.scrollIntoView?.({ block: "nearest" });
  };
  const moveCursor = (i: number, shift: boolean) => {
    if (!viewKeys.length) return;
    const j = Math.max(0, Math.min(viewKeys.length - 1, i));
    const key = viewKeys[j];
    if (shift && selectable) extendTo(key);
    setCursor(key);
    scrollToIndex(j);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTableElement>) => {
    if (e.target !== e.currentTarget) return;
    const at = cursor != null ? (viewIndex.get(cursor) ?? -1) : -1;
    const page = Math.max(5, Math.floor((scrollRef.current?.clientHeight || rowH * 10) / rowH) - 1);
    const mod = e.metaKey || e.ctrlKey;
    switch (e.key) {
      case "ArrowDown": e.preventDefault(); moveCursor(at < 0 ? 0 : at + 1, e.shiftKey); break;
      case "ArrowUp": e.preventDefault(); moveCursor(at < 0 ? viewKeys.length - 1 : at - 1, e.shiftKey); break;
      case "PageDown": e.preventDefault(); moveCursor(at + page, e.shiftKey); break;
      case "PageUp": e.preventDefault(); moveCursor(Math.max(0, at - page), e.shiftKey); break;
      case "Home": e.preventDefault(); moveCursor(0, e.shiftKey); break;
      case "End": e.preventDefault(); moveCursor(viewKeys.length - 1, e.shiftKey); break;
      case " ":
        if (cursor != null && selectable) { e.preventDefault(); toggleKey(cursor, e.shiftKey); }
        break;
      case "Enter":
        if (cursor != null) { e.preventDefault(); activate(cursor); }
        break;
      case "Escape":
        if (selectable && selSet.size) { e.preventDefault(); commitSelection(new Set()); setAnchor(null); }
        break;
      default:
        if (mod && (e.key === "a" || e.key === "A") && selectable) {
          e.preventDefault();
          commitSelection(new Set([...selSet, ...viewKeys]));
        }
    }
  };

  const onRowClick = (e: MouseEvent<HTMLTableRowElement>, key: string) => {
    const t = e.target as HTMLElement;
    const hit = t.closest?.(INTERACTIVE);
    if (hit && hit !== e.currentTarget && e.currentTarget.contains(hit)) return;
    setCursor(key);
    if (selectable && (e.metaKey || e.ctrlKey)) { toggleKey(key, false); return; }
    if (selectable && e.shiftKey) { extendTo(key); return; }
    activate(key);
  };

  /* ── export / columns menu ──
     With a selection, the CSV holds EVERY selected row (the count on the
     button), also those the filter hides, in the current sort order. */
  const exportCsv = () => {
    const out = selSet.size
      ? sortRows(rows.filter((r) => selSet.has(entries.byRow.get(r)!.key)), columns, sort)
      : view;
    downloadText(`${safeFileName(exportName ?? "table")}.csv`, toCSV(out, visibleCols), "text/csv;charset=utf-8");
  };
  const selHidden = selSet.size ? selSet.size - viewKeys.reduce((n, k) => n + (selSet.has(k) ? 1 : 0), 0) : 0;
  const csvTitle = !selSet.size ? "Export the rows shown"
    : `Export the ${selSet.size.toLocaleString("en-US")} selected row${selSet.size === 1 ? "" : "s"}`
      + (selHidden ? ` (${selHidden.toLocaleString("en-US")} hidden by the filter)` : "");
  const hideable = columns.filter((c) => c.hideable !== false);
  const columnItems: MenuItem[] = [
    { type: "label", label: "Columns" },
    ...hideable.map((c): MenuItem => ({
      type: "checkbox", id: c.id, label: headerText(c), checked: isVisible(c),
      disabled: isVisible(c) && visibleCols.length <= 1,
      onCheckedChange: (on) => {
        const next = { ...visibility, [c.id]: on };
        setVisInner(next);
        p.onColumnVisibilityChange?.(next);
      },
    })),
    { type: "separator" },
    {
      label: "Show all", onSelect: () => {
        const next = Object.fromEntries(columns.map((c) => [c.id, true]));
        setVisInner(next);
        p.onColumnVisibilityChange?.(next);
      },
    },
  ];

  /* ── render ── */
  const nCols = visibleCols.length + (selectable ? 1 : 0);
  const sumWidth = visibleCols.reduce((s, c) => s + (widths[c.id] ?? 120), selectable ? 36 : 0);
  const cursorIdx = cursor != null ? viewIndex.get(cursor) : undefined;
  const rowId = (i: number) => `${uid}-r${i}`;
  const countText = filter.trim() && view.length !== rows.length
    ? `${view.length.toLocaleString("en-US")} of ${rows.length.toLocaleString("en-US")} rows`
    : `${rows.length.toLocaleString("en-US")} row${rows.length === 1 ? "" : "s"}`;

  const bodyRows = indices.map((i) => {
    const row = view[i];
    const { key, index } = entries.byRow.get(row)!;
    const on = selSet.has(key);
    return (
      <tr key={key} id={rowId(i)} data-key={key} data-vindex={i} data-index={i}
        ref={virtualOn ? virtualizer.measureElement : undefined}
        aria-rowindex={i + 2} aria-selected={selectable ? on : undefined}
        data-cursor={cursorIdx === i || undefined} data-active={isActive(row, key) || undefined}
        className={cx("ui-dt__row", on && "is-selected", p.rowClassName?.(row))}
        onMouseDown={(e) => { if (e.shiftKey && selectable) e.preventDefault(); }}
        onClick={(e) => onRowClick(e, key)}>
        {selectable && (
          <td className="ui-dt__selcell" role="gridcell">
            <Checkbox checked={on} onChange={() => { /* handled on click (keeps shift) */ }}
              aria-label={`Select ${key}`}
              onClick={(e) => { e.stopPropagation(); setCursor(key); toggleKey(key, e.shiftKey); }} />
          </td>
        )}
        {visibleCols.map((c) => (
          <td key={c.id} role="gridcell"
            className={cx(c.numeric && "is-num", c.className)}
            style={{ textAlign: c.align ?? (c.numeric ? "right" : undefined) }}>
            {c.cell ? c.cell(row, index) : defaultCell(columnValue(c, row))}
          </td>
        ))}
      </tr>
    );
  });

  let bodyExtra: ReactNode = null;
  if (!view.length) {
    const msg = loading && !rows.length
      ? <Skeleton lines={3} />
      : rows.length
        ? (
          <EmptyState compact icon="filter" title={`No rows match “${filter}”`}
            action={searchable ? <Button size="sm" variant="ghost" onClick={() => setFilter("")}>Clear filter</Button> : undefined} />
        )
        : <EmptyState compact title={p.empty ?? "Nothing here yet"} />;
    bodyExtra = <tr className="ui-dt__emptyrow"><td colSpan={Math.max(1, nCols)}>{msg}</td></tr>;
  }

  return (
    <div className={cx("ui-dt", dense && "ui-dt--dense", p.className)}>
      {!hideToolbar && (
        <div className="ui-dt__bar">
          {searchable && (
            <Input type="search" value={filter} onChange={setFilter} size="sm" icon="search" clearable
              placeholder={p.filterPlaceholder ?? "Filter… (col:x, col>n, -not)"} aria-label="Filter rows"
              className="ui-dt__search" />
          )}
          <span className="ui-dt__count mono" aria-live="polite">{countText}</span>
          {selectable && selSet.size > 0 && (
            <span className="ui-dt__selcount">
              <span className="mono">{selSet.size.toLocaleString("en-US")} selected</span>
              <button type="button" className="ui-dt__clear" onClick={() => { commitSelection(new Set()); setAnchor(null); }}>
                clear
              </button>
            </span>
          )}
          <span className="ui-dt__spacer" />
          {p.toolbar}
          {hideable.length > 1 && (
            <Menu align="end" label="Columns" items={columnItems}
              trigger={<IconButton size="sm" icon="columns" label="Columns" />} />
          )}
          {exportName && (
            <Button size="sm" variant="ghost" icon="download" onClick={exportCsv} title={csvTitle}>
              {selSet.size ? `CSV (${selSet.size})` : "CSV"}
            </Button>
          )}
        </div>
      )}
      <div ref={scrollRef} className="ui-dt__scroll" style={{ maxHeight: height === "auto" ? undefined : height }}>
        <table className="ui-dt__table" role="grid" tabIndex={0}
          aria-label={p["aria-label"]} aria-rowcount={view.length + 1} aria-colcount={nCols}
          aria-multiselectable={selectable || undefined}
          aria-activedescendant={cursorIdx != null && indices.includes(cursorIdx) ? rowId(cursorIdx) : undefined}
          style={{ width: `max(100%, ${Math.round(sumWidth)}px)` }}
          onKeyDown={onKeyDown}>
          {p.caption && <caption className="sr-only">{p.caption}</caption>}
          <colgroup>
            {selectable && <col style={{ width: 36 }} />}
            {visibleCols.map((c) => (
              <col key={c.id} style={{ width: typeof c.width === "string" ? c.width : widths[c.id] }} />
            ))}
          </colgroup>
          <thead ref={headRef}>
            <tr aria-rowindex={1}>
              {selectable && (
                <th className="ui-dt__selcell" scope="col">
                  <Checkbox checked={allInView} indeterminate={someInView} onChange={toggleAll}
                    aria-label="Select all rows" disabled={!viewKeys.length} />
                </th>
              )}
              {visibleCols.map((c) => {
                const at = sort.findIndex((s) => s.id === c.id);
                const dir = at >= 0 ? (sort[at].desc ? "desc" : "asc") : null;
                const ariaSort = c.sortable === false ? undefined
                  : at === 0 ? (dir === "asc" ? "ascending" : "descending") : at > 0 ? "other" : "none";
                const align = c.align ?? (c.numeric ? "right" : undefined);
                return (
                  <th key={c.id} scope="col" aria-sort={ariaSort} style={{ textAlign: align }}
                    className={cx(c.numeric && "is-num")}>
                    {c.sortable === false ? c.header : (
                      <button type="button" className={cx("ui-dt__sort", align === "right" && "ui-dt__sort--right")}
                        onClick={(e) => setSort(nextSort(sort, c.id, e.shiftKey))}
                        title={`Sort by ${headerText(c)} (shift-click to add a key)`}>
                        <span className="ui-dt__hlabel">{c.header}</span>
                        <span className="ui-dt__sorticon" aria-hidden="true">
                          {dir && <Icon name={dir === "asc" ? "arrowUp" : "arrowDown"} size={12} />}
                          {dir && sort.length > 1 && <span className="ui-dt__sortn">{at + 1}</span>}
                        </span>
                      </button>
                    )}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {padTop > 0 && <tr aria-hidden="true" className="ui-dt__pad" style={{ height: padTop }}><td colSpan={nCols} /></tr>}
            {bodyRows}
            {padBottom > 0 && <tr aria-hidden="true" className="ui-dt__pad" style={{ height: padBottom }}><td colSpan={nCols} /></tr>}
            {bodyExtra}
          </tbody>
        </table>
      </div>
    </div>
  );
}
