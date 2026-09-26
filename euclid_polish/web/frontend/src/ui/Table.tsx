/* Compat: the simple non-virtualised Table of the pre-rework pages. New code
   uses DataTable (sorting, filtering, selection, virtualisation, CSV).
   Clickable rows are keyboard-reachable (Tab, Enter/Space). */
import type { ReactNode } from "react";
import { Empty } from "./display";
import { cx } from "./slot";

export type Column<T> = {
  header: ReactNode;
  cell: (row: T, i: number) => ReactNode;
  align?: "left" | "right" | "center";
  width?: string | number;
};

export function Table<T>(
  { columns, rows, empty, rowKey, onRowClick, isRowClickable, className, "aria-label": ariaLabel }: {
    columns: Column<T>[]; rows: T[]; empty?: ReactNode;
    rowKey?: (row: T, i: number) => string | number;
    onRowClick?: (row: T, i: number) => void;
    isRowClickable?: (row: T, i: number) => boolean;
    className?: string; "aria-label"?: string;
  },
) {
  if (!rows.length) return <Empty>{empty ?? "nothing here yet"}</Empty>;
  return (
    <div className="ui-table-wrap">
      <table className={cx("ui-table", className)} aria-label={ariaLabel}>
        <thead>
          <tr>{columns.map((c, i) => (
            <th key={i} scope="col" style={{ textAlign: c.align, width: c.width }}>{c.header}</th>
          ))}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const clickable = !!onRowClick && (isRowClickable?.(r, i) ?? true);
            return (
              <tr key={rowKey ? rowKey(r, i) : i}
                className={clickable ? "ui-table__row--action" : undefined}
                tabIndex={clickable ? 0 : undefined}
                onClick={clickable ? () => onRowClick(r, i) : undefined}
                onKeyDown={clickable ? (e) => {
                  if (e.target !== e.currentTarget) return;
                  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onRowClick(r, i); }
                } : undefined}>
                {columns.map((c, j) => (
                  <td key={j} style={{ textAlign: c.align }}>{c.cell(r, i)}</td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
