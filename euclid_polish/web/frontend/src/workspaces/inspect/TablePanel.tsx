/* A table HDU: server-paged rows (sorting the whole table on the server; the
   filter box narrows the loaded page) and per-column statistics with a
   histogram / top values of the chosen column. */
import { useMemo, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { useResource } from "../../api/query";
import { formatCount, formatNumber } from "../../format";
import { useUrlState } from "../../hooks/useUrlState";
import {
  Badge, Button, Callout, Card, CardBody, CardHead, CopyButton, DataTable, DefList, Icon, IconButton, Section,
  Select, Skeleton, type DataColumn, type SortState,
} from "../../ui";
import {
  tableStatsUrl, tableUrl, type ColumnStats, type HduSummary, type TableColumn, type TablePage,
  type TableStats,
} from "./api";
import { HistogramPlot } from "./charts";
import { basename, pageLabel, ROW_COLUMN, skyAt, skyColumns, sortToServer } from "./model";

type Row = { i: number; cells: unknown[] };

const PAGE_SIZES = ["100", "200", "500", "1000", "2000"].map((v) => ({ value: v, label: `${v} rows` }));

function cellNode(v: unknown): ReactNode {
  if (v == null) return <span className="insp-dim">—</span>;
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : formatNumber(v, { sig: 9, grouping: false });
  if (typeof v === "boolean") return v ? "T" : "F";
  if (Array.isArray(v)) return <span className="mono">[{v.map((x) => (typeof x === "number" ? formatNumber(x, { sig: 4, grouping: false }) : String(x ?? "—"))).join(", ")}]</span>;
  return String(v);
}

function parseSortParam(raw: string): SortState {
  if (!raw) return [];
  return raw.startsWith("-") ? [{ id: raw.slice(1), desc: true }] : [{ id: raw, desc: false }];
}

function ColumnDetail({ c, fileStem }: { c: ColumnStats; fileStem: string }) {
  if (c.kind === "numeric" && c.histogram) {
    return <HistogramPlot hist={c.histogram} label={c.name} unit={c.unit ?? undefined} exportName={`${fileStem}_${c.name}_hist`}
      markers={[{ v: c.median, label: "median" }]} height={170} />;
  }
  if (c.kind === "text" && c.top?.length) {
    return (
      <ol className="insp-top">
        {c.top.map(([value, n]) => (
          <li key={value}><code className="mono">{value || "(empty)"}</code><span className="insp-dim">{formatCount(n)}</span></li>
        ))}
      </ol>
    );
  }
  if (c.kind === "bool") return <p>{formatCount(c.n_true)} true · {formatCount(c.n_false)} false</p>;
  return <p className="insp-dim">{c.kind === "array" ? `Array cells${c.shape ? ` of shape ${c.shape.join("×")}` : ""}.` : "No values."}</p>;
}

/** One row of the page, every column as a line (the table is often wider
 *  than the pane); a sky link when the table has RA/Dec columns. */
function RowCard({ row, columns, onClose }: { row: Row; columns: TableColumn[]; onClose: () => void }) {
  const sky = skyColumns(columns);
  const at = (name: string) => row.cells[columns.findIndex((c) => c.name === name)];
  const href = sky ? skyAt(Number(at(sky.ra)), Number(at(sky.dec))) : null;
  const text = columns.map((c, j) => `${c.name}\t${JSON.stringify(row.cells[j] ?? null)}`).join("\n");
  return (
    <Card className="insp-card insp-rowcard">
      <CardHead title={`Row ${formatCount(row.i)}`}
        right={(
          <span className="insp-rowcard__actions">
            {href && (
              <Button asChild size="sm">
                <Link to={href}><Icon name="globe" /><span className="ui-btn__label">Show on sky</span></Link>
              </Button>
            )}
            <CopyButton value={text} label="Copy the row (name, value per line)" />
            <IconButton size="sm" icon="close" label="Close the row" onClick={onClose} />
          </span>
        )} />
      <CardBody>
        <DefList dense items={columns.map((c, j) => [
          <span title={[c.format, c.unit].filter(Boolean).join(" · ")}><code className="mono">{c.name}</code></span>,
          <span className="mono insp-break">{cellNode(row.cells[j])}{c.unit ? <span className="insp-dim"> {c.unit}</span> : null}</span>,
        ])} />
      </CardBody>
    </Card>
  );
}

const PRECISE = { sig: 7, grouping: false } as const;

const STAT_COLUMNS: DataColumn<ColumnStats>[] = [
  { id: "name", header: "Column", cell: (c) => <code className="mono">{c.name}</code> },
  { id: "kind", header: "Kind", width: 80, cell: (c) => <Badge size="sm">{c.kind}</Badge> },
  { id: "unit", header: "Unit", accessor: (c) => c.unit ?? "", cell: (c) => (c.unit ? <span className="mono">{c.unit}</span> : "") },
  { id: "n_null", header: "Null", numeric: true, accessor: (c) => c.n_null ?? c.n_empty ?? null },
  { id: "min", header: "Min", numeric: true, accessor: (c) => c.min ?? null, cell: (c) => formatNumber(c.min, PRECISE) },
  { id: "max", header: "Max", numeric: true, accessor: (c) => c.max ?? null, cell: (c) => formatNumber(c.max, PRECISE) },
  { id: "median", header: "Median", numeric: true, accessor: (c) => c.median ?? null, cell: (c) => formatNumber(c.median, PRECISE) },
  { id: "std", header: "σ", headerText: "std", numeric: true, accessor: (c) => c.std ?? null, cell: (c) => formatNumber(c.std) },
  { id: "n_unique", header: "Unique", numeric: true, accessor: (c) => c.n_unique ?? null },
];

export function TablePanel({ fits, hdu }: { fits: string; hdu: HduSummary }) {
  const [offset, setOffset] = useUrlState("toff", 0);
  const [limitRaw, setLimit] = useUrlState("tlim", "200");
  const [sortRaw, setSortRaw] = useUrlState("tsort", "");
  const [colName, setCol] = useUrlState("tcol", "");
  const [filter, setFilter] = useUrlState("tq", "");
  const [rowParam, setRowParam] = useUrlState("trow", -1);
  const limit = Number(limitRaw) || 200;
  const sort = useMemo(() => parseSortParam(sortRaw), [sortRaw]);
  const server = sortToServer(sort);
  const page = useResource<TablePage>(tableUrl(fits, hdu.index, { offset, limit, sort: server.sort, desc: server.desc }), [], { ttl: 60_000 });
  const stats = useResource<TableStats>(tableStatsUrl(fits, hdu.index), [], { ttl: 5 * 60_000 });
  const data = page.data;
  const total = data?.total ?? hdu.nrows ?? 0;
  const stem = basename(fits).replace(/\.fits?(\.gz|\.fz)?$/i, "");

  const columns = useMemo<DataColumn<Row>[]>(() => {
    const cols = data?.columns ?? hdu.columns ?? [];
    return [
      { id: ROW_COLUMN, header: "#", numeric: true, width: 64, accessor: (r) => r.i, sortable: false },
      ...cols.map((c, j): DataColumn<Row> => ({
        id: c.name,
        header: c.unit ? <span title={`${c.format} · ${c.unit}`}>{c.name} <span className="insp-dim">[{c.unit}]</span></span> : <span title={c.format}>{c.name}</span>,
        headerText: c.name,
        numeric: c.kind === "numeric",
        sortable: c.kind !== "array",
        accessor: (r) => r.cells[j],
        cell: (r) => cellNode(r.cells[j]),
      })),
    ];
  }, [data?.columns, hdu.columns]);
  const rows = useMemo<Row[]>(() => (data?.rows ?? []).map((cells, k) => ({ i: data!.row_index[k], cells })), [data]);

  const onSort = (next: SortState) => {
    const s = sortToServer(next);
    setSortRaw(s.sort ? `${s.desc ? "-" : ""}${s.sort}` : "");
    setOffset(0);
  };
  const last = Math.max(0, Math.floor((total - 1) / limit) * limit);
  const chosen = stats.data?.columns.find((c) => c.name === colName) ?? null;
  const pageColumns = data?.columns ?? hdu.columns ?? [];
  const openRow = rows.find((r) => r.i === rowParam) ?? null;

  return (
    <div className="insp-table">
      <div className="insp-toolbar insp-toolbar--sub" role="toolbar" aria-label="Pages">
        <Button size="sm" variant="ghost" disabled={offset <= 0} onClick={() => setOffset(0)}>First</Button>
        <IconButton size="sm" icon="chevronLeft" label="Previous page" disabled={offset <= 0}
          onClick={() => setOffset(Math.max(0, offset - limit))} />
        <span className="mono insp-toolbar__label" aria-live="polite">{pageLabel(offset, limit, total)}</span>
        <IconButton size="sm" icon="chevronRight" label="Next page" disabled={offset + limit >= total}
          onClick={() => setOffset(offset + limit)} />
        <Button size="sm" variant="ghost" disabled={offset >= last} onClick={() => setOffset(last)}>Last</Button>
        <span className="insp-toolbar__spacer" />
        {server.sort && <Badge size="sm" tone="info">sorted by {server.sort}{server.desc ? " ↓" : " ↑"} (whole table)</Badge>}
        <Select size="sm" aria-label="Rows per page" value={String(limit)} options={PAGE_SIZES}
          onChange={(v) => { setLimit(v); setOffset(0); }} />
      </div>
      {page.error ? (
        <Callout tone="bad" title="Could not read the table">{page.error.message}</Callout>
      ) : (
        <DataTable<Row>
          rows={rows} columns={columns} rowKey={(r) => String(r.i)} aria-label={`Rows of HDU ${hdu.index}`}
          loading={page.loading} dense height="min(64vh, 640px)" sort={sort} onSortChange={onSort}
          filter={filter} onFilterChange={setFilter} filterPlaceholder="Filter this page… (flux>10, name:gal)"
          activeKey={openRow ? String(openRow.i) : null}
          onRowClick={(r) => setRowParam(r.i === rowParam ? -1 : r.i)}
          exportName={`${stem}_hdu${hdu.index}_rows_${offset + 1}`} empty="No rows." />
      )}
      {openRow && <RowCard row={openRow} columns={pageColumns} onClose={() => setRowParam(-1)} />}
      <Section title="Columns" sub={stats.data?.sampled ? `statistics over every ${stats.data.sampled}th row` : `${hdu.ncols ?? 0} columns`}>
        {stats.loading && <Skeleton lines={4} />}
        {stats.error && <Callout tone="bad" title="No column statistics">{stats.error.message}</Callout>}
        {stats.data && (
          <div className="insp-colstats">
            <DataTable<ColumnStats>
              rows={stats.data.columns} columns={STAT_COLUMNS} rowKey={(c) => c.name} aria-label="Column statistics"
              dense height={stats.data.columns.length > 12 ? 360 : "auto"} activeKey={chosen?.name ?? null}
              onRowClick={(c) => setCol(c.name === colName ? "" : c.name)} exportName={`${stem}_hdu${hdu.index}_columns`}
              hideToolbar={stats.data.columns.length <= 12} />
            {chosen && (
              <Card className="insp-card">
                <CardHead title={<code className="mono">{chosen.name}</code>} sub={chosen.kind}
                  right={<IconButton size="sm" icon="close" label="Close column" onClick={() => setCol("")} />} />
                <CardBody><ColumnDetail c={chosen} fileStem={stem} /></CardBody>
              </Card>
            )}
          </div>
        )}
      </Section>
    </div>
  );
}
