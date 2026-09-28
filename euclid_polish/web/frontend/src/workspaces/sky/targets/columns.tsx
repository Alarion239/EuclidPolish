/* The Sky › Targets table: one row per target, sorted by flux SR/LR (the
 * worst first). Target, State and Flux never drop and fit, with the select
 * column, the ~340 px the tile card leaves at 1024 × 768 (columns.test.tsx). The state reads current / stale / missing everywhere, with
 * a badge only on a problem; "Made by" names the model in words. Holes and
 * R̃ exist only for scored real tiles, so those columns show only when every
 * row in view is scored (Targets' "Show only them"; the per-set sentence
 * carries the median otherwise) — never a mostly blank column. Narrow tables drop
 * the columns with a priority, highest first: RA / Dec, field, grade, R̃,
 * holes, set, then "made by" (which says the most about a stale row). */
import { formatDeg, formatNumber } from "../../../format";
import { Badge, Tooltip, type DataColumn } from "../../../ui";
import { deltaMagWarn, fluxDeltaMag, formatMetric } from "../results/model";
import { SET_BY_ID, type TargetRow } from "./model";

export function StateCell({ row }: { row: Pick<TargetRow, "state" | "reason"> }) {
  const body = row.state === "current" ? <span className="tg-state tg-state--current">current</span>
    : <Badge size="sm" dot tone={row.state === "stale" ? "warn" : "neutral"}>{row.state}</Badge>;
  if (!row.reason) return body;
  return <Tooltip content={row.reason}><span tabIndex={0} className="tg-state__tip">{body}</span></Tooltip>;
}

const flux = (r: TargetRow) => {
  if (r.flux == null) return <span className="muted">—</span>;
  const warn = deltaMagWarn(fluxDeltaMag(r.flux));
  return <span className={warn ? "tg-warn" : undefined}>{formatNumber(r.flux, { digits: 2 })}</span>;
};

export function targetColumns(opts: { scored: boolean; manySets: boolean; lenses: boolean }): DataColumn<TargetRow>[] {
  const cols: DataColumn<TargetRow>[] = [
    { id: "id", header: "Target", accessor: (r) => r.id, filterText: (r) => `${r.id} ${r.label}${r.hasJwst ? " jwst" : ""}`, width: 118,
      cell: (r) => (
        <span className="res-tilecell">
          <span className="mono res-ellipsis" title={r.label}>{r.id}</span>
          {r.hasJwst && <Badge size="sm" tone="info" title="Has a JWST image">JWST</Badge>}
        </span>
      ) },
    { id: "set", header: "Set", accessor: (r) => SET_BY_ID[r.set].label, width: 120, hidden: !opts.manySets, priority: 3 },
    { id: "state", header: "State", accessor: (r) => r.state, width: 78, cell: (r) => <StateCell row={r} /> },
    { id: "madeBy", header: "Made by", accessor: (r) => r.madeBy ?? "", width: 210, priority: 2,
      cell: (r) => (r.madeBy ? <span className="res-ellipsis" title={r.madeBy}>{r.madeBy}</span> : <span className="muted">—</span>) },
    { id: "flux", header: "Flux SR/LR", headerText: "Flux SR/LR (VIS)", numeric: true, accessor: (r) => r.flux, width: 100, cell: flux,
      csv: (r) => (r.flux == null ? "" : String(r.flux)) },
  ];
  if (opts.scored) {
    cols.push(
      { id: "holes", header: "Holes %", headerText: "Holes % (worst band)", numeric: true, accessor: (r) => (r.scored ? r.holes : null),
        cell: (r) => (r.scored ? formatMetric("hole_pct", r.holes) : ""), width: 78, priority: 4 },
      { id: "medR", header: "R̃", headerText: "Median R", numeric: true, accessor: (r) => (r.scored ? r.medR : null),
        cell: (r) => (r.scored ? formatMetric("median_R", r.medR) : ""), width: 64, priority: 5 },
    );
  }
  if (opts.lenses) cols.push({ id: "grade", header: "Grade", accessor: (r) => r.grade ?? "", width: 64, priority: 6 });
  cols.push(
    { id: "field", header: "Field", accessor: (r) => r.field ?? "", width: 64, priority: 7 },
    { id: "ra", header: "RA, Dec", headerText: "RA", numeric: true, accessor: (r) => r.ra, width: 96, priority: 8,
      cell: (r) => (r.ra == null || r.dec == null ? <span className="muted">—</span> : (
        <span className="res-radec" title={`${formatDeg(r.ra, 6)} ${formatDeg(r.dec, 6, { signed: true })}`}>
          <span>{formatDeg(r.ra, 4)}</span><span>{formatDeg(r.dec, 4, { signed: true })}</span>
        </span>
      )) },
    { id: "dec", header: "Dec", numeric: true, accessor: (r) => r.dec, cell: (r) => formatDeg(r.dec, 4, { signed: true }), hidden: true },
    { id: "reason", header: "Why", accessor: (r) => r.reason ?? "", hidden: true },
  );
  return cols;
}
