/* The Real results table's columns (Sky › Real results). Narrow tables drop
 * columns by `priority`, highest first: Source (the source chips already say
 * it), Field, Models, R̃, Holes. Tile, RA / Dec and Production never drop and
 * are sized so that, with the select column, they fit the table the tile
 * inspector leaves at 1024×768 (INSPECTOR_TABLE_PX) without a sideways scroll. */
import { formatDeg } from "../../../format";
import { Badge, type DataColumn } from "../../../ui";
import type { TileRow } from "./api";
import { StateBadge } from "./common";
import { formatMetric, headlineSpec, specShort, tileModels } from "./model";

/** The table's client width with the tile inspector open at 1024×768. */
export const INSPECTOR_TABLE_PX = 340;
/** The DataTable's checkbox column. */
export const SELECT_COL_PX = 36;

function headline(row: TileRow) {
  const spec = headlineSpec(row);
  return { spec, summary: spec ? row.models?.[spec]?.summary ?? null : null };
}

export const REAL_TILE_COLUMNS: DataColumn<TileRow>[] = [
  // narrow tables (the inspector open, a ~720 px pane) drop the columns with a
  // priority, highest first: Source (the source chips already say it), Field,
  // Models, R̃, Holes; Tile, RA / Dec and Production never drop (kit DataTable
  // `priority`; the column menu shows a dropped one again)
  { id: "source", header: "Source", cell: (r) => <Badge size="sm">{r.source}</Badge>, width: 76, priority: 6 },
  // the id over its JWST badge: the column stays narrow enough for the
  // inspector-open table (see INSPECTOR_TABLE_PX)
  { id: "id", header: "Tile", filterText: (r) => `${r.id} ${r.label}${r.has_jwst ? " jwst" : ""}`, width: 110,
    cell: (r) => (
      <span className="res-tilecell">
        <span className="mono res-ellipsis" title={r.label}>{r.id}</span>
        {r.has_jwst && <Badge size="sm" tone="info" title="Has a JWST image">JWST</Badge>}
      </span>
    ) },
  { id: "label", header: "Label", hidden: true },
  { id: "field", header: "Field", accessor: (r) => r.field ?? "", width: 64, priority: 5 },
  // the whole value, never cut: RA over Dec, two tabular lines of 9 characters
  // ("268.3772°" / "+65.0985°") in a column a third as wide as one line
  { id: "ra", header: "RA, Dec", headerText: "RA", numeric: true, width: 96,
    cell: (r) => (
      <span className="res-radec" title={`${formatDeg(r.ra, 6)} ${formatDeg(r.dec, 6, { signed: true })}`}>
        <span>{formatDeg(r.ra, 4)}</span><span>{formatDeg(r.dec, 4, { signed: true })}</span>
      </span>
    ) },
  { id: "dec", header: "Dec", numeric: true, cell: (r) => formatDeg(r.dec, 4, { signed: true }), hidden: true },
  { id: "shape", header: "Grid", accessor: (r) => (r.shape ? r.shape[0] * r.shape[1] : null),
    cell: (r) => (r.shape ? `${r.shape[1]}×${r.shape[0]}` : "—"), csv: (r) => (r.shape ? `${r.shape[1]}x${r.shape[0]}` : ""), hidden: true },
  { id: "jwst", header: "JWST", accessor: (r) => (r.has_jwst ? "jwst" : ""), hidden: true },
  { id: "production", header: "Production", accessor: (r) => r.production_state ?? "missing",
    cell: (r) => <StateBadge state={r.production_state} />, width: 90 },
  { id: "models", header: "Models", accessor: (r) => Object.keys(r.models ?? {}).length,
    filterText: (r) => Object.keys(r.models ?? {}).join(" "),
    csv: (r) => Object.keys(r.models ?? {}).join(" "),
    cell: (r) => {
      const ms = tileModels(r);
      if (!ms.length) return <span className="muted">—</span>;
      return (
        <span className="res-chips res-chips--tight">
          {ms.slice(0, 3).map((m) => (
            <Badge key={m.spec} size="sm" dot tone={m.state === "current" ? "good" : m.state === "stale" ? "warn" : "neutral"}
              title={`${m.spec} · ${m.state}${m.legacy ? " · legacy" : ""}`}>{specShort(m.spec)}</Badge>
          ))}
          {ms.length > 3 && <span className="muted">+{ms.length - 3}</span>}
        </span>
      );
    }, width: 170, priority: 4 },
  { id: "holes", header: "Holes %", headerText: "Holes % (worst band)", numeric: true, accessor: (r) => headline(r).summary?.hole_pct_max ?? null,
    cell: (r) => { const h = headline(r); return <span title={h.spec ? `${h.spec}: worst band` : undefined}>{formatMetric("hole_pct", h.summary?.hole_pct_max)}</span>; },
    width: 80, priority: 2 },
  { id: "R08", header: "% R<0.8", numeric: true, accessor: (r) => headline(r).summary?.pct_R_lt_0p8 ?? null,
    cell: (r) => formatMetric("pct_R_lt_0p8", headline(r).summary?.pct_R_lt_0p8), hidden: true },
  { id: "medR", header: "R̃", headerText: "Median R", numeric: true, accessor: (r) => headline(r).summary?.median_R ?? null,
    cell: (r) => formatMetric("median_R", headline(r).summary?.median_R), width: 64, priority: 3 },
  { id: "peaks", header: "Peaks", numeric: true, accessor: (r) => headline(r).summary?.n_peaks ?? null, hidden: true },
  { id: "grade", header: "Grade", accessor: (r) => (typeof r.extras?.grade === "string" ? r.extras.grade : ""), hidden: true },
  { id: "legacy", header: "Legacy SR", accessor: (r) => {
      const l = r.extras?.legacy_sr as { origin?: string; kind?: string } | null | undefined;
      return l ? `${l.origin ?? ""} ${l.kind ?? ""}`.trim() : "";
    }, hidden: true },
];
