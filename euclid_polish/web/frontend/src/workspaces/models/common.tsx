/* Small shared pieces of the Models tabs: the error / empty states that show
   the server's own message, facet colours, the colour-by control and the
   gate-share bar. Tab control bars are the kit's Toolbar. */
import { useMemo, type ReactNode } from "react";
import type { ApiError } from "../../api/client";
import { openInspector } from "../../app/inspector";
import { C, LOSS_COLOR, categorical, viridis } from "../../colors";
import { Button, EmptyState, Select, Skeleton } from "../../ui";
import { COLOR_BY_OPTIONS, facetOf, facetValues, kneeText, type ColorBy } from "./model";
import type { KneeInfo } from "./api";

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
        <span className="mdl-mono">{error.message}</span>
      </EmptyState>
    );
  }
  if (empty) return <>{empty}</>;
  return <>{children}</>;
}

type FacetRow = KneeInfo & { loss?: string | null; loss_norm?: string | null; blocks?: number | null };

/** The single training knees present, ascending: the viridis ramp's stops. */
export function kneeOrderOf(rows: readonly KneeInfo[]): number[] {
  return [...new Set(rows.filter((m) => kneeText(m).kind !== "multi").map((m) => kneeText(m).sort))].sort((a, b) => a - b);
}

/** THE knee / multi-knee colour of a member, shared by every Models tab (the
 *  Leaderboard curves, Members › Curves, Diagnostics): single knees on an
 *  ordered viridis ramp (low knee dark, high knee light), multi-knee members
 *  in two fixed hues (one image vs heads). Under "multi" the single-knee
 *  members are muted. */
export function kneeColor(m: KneeInfo, order: readonly number[], by: "knee" | "multi" = "knee"): string {
  const k = kneeText(m);
  if (k.kind === "multi") return categorical(m.output_knee != null ? 1 : 3);
  if (by === "multi") return C.muted;
  const i = Math.max(0, order.indexOf(k.sort));
  return viridis(order.length > 1 ? 0.1 + (0.8 * i) / (order.length - 1) : 0.5);
}

/** Facet value → colour (loss colours for losses, the shared knee colours
 *  for knee / multi-knee, the categorical palette for depth, the
 *  ensemble-mean colour for "uniform"). Read on render, so a theme flip (the
 *  tab re-renders) picks up the new tokens. */
export function useFacetColors<T extends FacetRow>(rows: readonly T[], by: ColorBy) {
  const values = useMemo(() => facetValues(rows, by), [rows, by]);
  const order = useMemo(() => kneeOrderOf(rows), [rows]);
  const sample = useMemo(() => {
    const m = new Map<string, T>();
    for (const r of rows) { const v = facetOf(r, by); if (!m.has(v)) m.set(v, r); }
    return m;
  }, [rows, by]);
  const color = (v: string, row?: T) => {
    if (by === "uniform") return C.mean;
    if (by === "loss") return LOSS_COLOR[v];
    if (by === "knee" || by === "multi") {
      const r = row ?? sample.get(v);
      return r ? kneeColor(r, order, by) : C.muted;
    }
    return categorical(Math.max(0, values.indexOf(v)));
  };
  return {
    values,
    of: (row: T) => color(facetOf(row, by), row),
    legend: values.map((v) => ({ label: v, color: color(v), key: v })),
  };
}

export function ColorBySelect({ value, onChange }: { value: ColorBy; onChange: (v: ColorBy) => void }) {
  return <Select<ColorBy> size="sm" aria-label="Colour by" value={value} onChange={onChange} options={COLOR_BY_OPTIONS} />;
}

export const openMember = (name: string) => openInspector({ kind: "member", id: name });

/** A member's share of the gate's weight as a bar (length ∝ share, scaled to
 *  the table's largest) with its text; "not read" members are dimmed. */
export function ShareBar({ value, max, text, read, label }: {
  value: number | null; max: number; text: string; read?: boolean | null; label?: string;
}) {
  if (value == null) return <span className="ui-dt__nil">—</span>;
  const w = Math.max(2, Math.min(100, (100 * value) / Math.max(max, 1e-9)));
  return (
    <span className="mdl-share" data-read={read === false ? "no" : undefined} aria-label={label ?? `Gate share ${text}${read === false ? ", not read by the gate" : ""}`}>
      <span className="mdl-share__track" aria-hidden><span className="mdl-share__fill" style={{ width: `${w}%` }} /></span>
      <span className="mdl-num">{text}</span>
      {read === false && <span className="mdl-faint">not read</span>}
    </span>
  );
}
