/* Small shared pieces of the Ensemble tabs: the sticky toolbar, the error /
   empty states that show the server's own message, facet colours and the
   colour-by control. */
import { useMemo, type ReactNode } from "react";
import type { ApiError } from "../../api/client";
import { openInspector } from "../../app/inspector";
import { C, LOSS_COLOR, categorical } from "../../colors";
import { Button, EmptyState, Select, Skeleton } from "../../ui";
import { COLOR_BY_OPTIONS, facetOf, facetValues, type ColorBy } from "./model";
import type { KneeInfo } from "./api";

export function EnsBar({ children, label }: { children: ReactNode; label?: string }) {
  return <div className="ens-bar" role="toolbar" aria-label={label ?? "Tab controls"}>{children}</div>;
}

export function BarGroup({ label, children }: { label?: string; children: ReactNode }) {
  return (
    <div className="ens-bar__group">
      {label && <span className="ens-bar__label">{label}</span>}
      {children}
    </div>
  );
}

/** Loading → skeleton; error → the server's message with a retry; empty → `empty`. */
export function LoadState(
  { loading, error, onRetry, empty, children, lines = 4 }: {
    loading: boolean; error: ApiError | null; onRetry?: () => void; empty?: ReactNode | false;
    children?: ReactNode; lines?: number;
  },
) {
  if (loading) return <Skeleton lines={lines} />;
  if (error) {
    return (
      <EmptyState icon="warn" title={error.status === 404 ? "Nothing here yet" : "Could not load"}
        action={onRetry && <Button size="sm" icon="reset" onClick={onRetry}>Retry</Button>}>
        <span className="ens-mono">{error.message}</span>
      </EmptyState>
    );
  }
  if (empty) return <>{empty}</>;
  return <>{children}</>;
}

type FacetRow = KneeInfo & { loss?: string | null; loss_norm?: string | null; blocks?: number | null };

/** Facet value → colour (loss colours for losses, the categorical palette
 *  otherwise, the ensemble-mean colour for "uniform"). Read on render, so a
 *  theme flip (the tab re-renders) picks up the new tokens. */
export function useFacetColors<T extends FacetRow>(rows: readonly T[], by: ColorBy) {
  const values = useMemo(() => facetValues(rows, by), [rows, by]);
  const color = (v: string) => (by === "uniform" ? C.mean
    : by === "loss" ? LOSS_COLOR[v] : categorical(Math.max(0, values.indexOf(v))));
  return {
    values,
    of: (row: T) => color(facetOf(row, by)),
    legend: values.map((v) => ({ label: v, color: color(v), key: v })),
  };
}

export function ColorBySelect({ value, onChange }: { value: ColorBy; onChange: (v: ColorBy) => void }) {
  return <Select<ColorBy> size="sm" aria-label="Colour by" value={value} onChange={onChange} options={COLOR_BY_OPTIONS} />;
}

export const openMember = (name: string) => openInspector({ kind: "member", id: name });
