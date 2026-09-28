/* The HDU list of a file (+ its 4-band colour groups): one row per HDU; a
   click selects it. */
import { useMemo } from "react";
import { formatBytes, formatCount } from "../../format";
import { Badge, DataTable, Tooltip, type DataColumn, type Tone } from "../../ui";
import type { BandGroup, HduSummary, InspectResponse } from "./api";
import { bandsText, shapeText } from "./model";

export type HduRow = {
  key: string;
  label: string;
  name: string;
  type: string;
  dims: string;
  dtype: string;
  unit: string;
  size: number | null;
  wcs: "wcs" | "constructed" | "";
  bands: string;
  hdu: HduSummary | null;
  group: BandGroup | null;
};

const TYPE_TONE: Record<string, Tone> = {
  image: "accent", table: "info", vector: "good", colour: "warn", empty: "neutral", other: "neutral",
};

export function hduRows(s: InspectResponse): HduRow[] {
  const rows: HduRow[] = s.hdus.map((h) => ({
    key: String(h.index),
    label: String(h.index),
    name: h.name,
    type: h.type,
    dims: h.type === "table" ? `${formatCount(h.nrows ?? 0)} rows × ${h.ncols ?? 0}` : shapeText(h.shape),
    dtype: h.type === "table" || h.type === "empty" ? "" : (h.dtype ?? "").replace(/^[<>|]/, "") + (h.scaling ? " (scaled)" : ""),
    unit: h.bunit ?? "",
    size: h.size_bytes ?? null,
    wcs: h.wcs ? (h.wcs.constructed ? "constructed" : "wcs") : "",
    bands: bandsText(h.bands ?? (h.band ? [h.band] : null)),
    hdu: h,
    group: null,
  }));
  for (const g of s.band_groups) {
    rows.push({
      key: g.id, label: "★", name: g.prefix.replace(/[_\-\s]+$/, "") || "4 bands", type: "colour", dims: shapeText(g.shape), dtype: "",
      unit: g.bunit, size: null, wcs: g.wcs ? (g.wcs.constructed ? "constructed" : "wcs") : "",
      bands: `HDUs ${g.hdus.join(" ")}`, hdu: null, group: g,
    });
  }
  return rows;
}

const COLUMNS: DataColumn<HduRow>[] = [
  { id: "label", header: "#", width: 40, sortable: false },
  { id: "name", header: "Name", cell: (r) => <span className="mono">{r.name}</span> },
  {
    id: "type", header: "Type", width: 84,
    cell: (r) => (
      <Tooltip content={r.hdu?.reason ?? (r.group ? "the four band HDUs as one colour cube" : r.hdu?.kind ?? "")}>
        <span tabIndex={-1}><Badge size="sm" tone={TYPE_TONE[r.type] ?? "neutral"}>{r.type}</Badge></span>
      </Tooltip>
    ),
  },
  {
    id: "dims", header: "Size", accessor: (r) => (r.dtype ? `${r.dims} · ${r.dtype}` : r.dims),
    cell: (r) => (
      <span className="mono">{r.dims}{r.dtype && <span className="insp-dim"> · {r.dtype}</span>}</span>
    ),
  },
  { id: "unit", header: "Unit", cell: (r) => (r.unit ? <span className="mono">{r.unit}</span> : "") },
  {
    id: "wcs", header: "WCS", width: 60,
    cell: (r) => (r.wcs === "wcs" ? <Badge size="sm" tone="good">sky</Badge>
      : r.wcs === "constructed" ? <Tooltip content="Built from RA/DEC/PIXSCALE (north up)"><span tabIndex={-1}><Badge size="sm" tone="warn">built</Badge></span></Tooltip>
        : ""),
  },
  { id: "bands", header: "Bands", hidden: false, cell: (r) => <span className="mono insp-dim">{r.bands}</span> },
  { id: "size", header: "Bytes", numeric: true, hidden: true, cell: (r) => (r.size ? formatBytes(r.size) : "") },
];

export function HduList({ summary, selected, onSelect }: {
  summary: InspectResponse; selected: string; onSelect: (key: string) => void;
}) {
  const rows = useMemo(() => hduRows(summary), [summary]);
  const many = rows.length > 8;
  return (
    <DataTable<HduRow>
      rows={rows} columns={COLUMNS} rowKey={(r) => r.key} aria-label="HDUs"
      activeKey={selected} onRowClick={(r) => onSelect(r.key)} dense
      height={many ? 280 : "auto"} hideToolbar={!many} filterPlaceholder="Filter HDUs…" />
  );
}
