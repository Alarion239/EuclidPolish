/* Figures › Sheet, left panel: the saved-crop pool of the sheet's regime,
 * newest first — a table (thumbnail, crop with its source and side, WCS) or
 * a gallery of large
 * thumbnails (`?pool=gallery`), one Find box for both (`?q=`). A tick puts
 * the crop in the sheet as a column; a thumbnail opens it full size; a
 * table row opens its `figure` card; each crop's menu opens its source
 * viewer, the sky, Files, a FITS download, Rename and Delete. The ticked
 * crops can be deleted together. The table keeps the console's filter
 * language in the Find box (`collection:real tiers:jwst wcs=wcs side<2`),
 * its column menu (tiers, source, crop side, position, saved, size and id
 * start hidden)
 * and a CSV export. */
import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { openInspector } from "../../../app/inspector";
import { formatBytes, formatDateTime, formatDeg, formatNumber, formatRelative } from "../../../format";
import {
  Button, Caption, Checkbox, DataTable, EmptyState, IconButton, Input, Menu, Segmented, Skeleton,
  type DataColumn, type MenuItem,
} from "../../../ui";
import { confirmDeleteResults } from "../actions";
import { fitsUrl, type SavedResult } from "../api";
import { ResultThumb, ThumbButton, WcsBadge } from "../common";
import { cropSideArcsec, galleryMatches, inspectLink, missingRecipes, skyLink, sourceLabel, viewerLink } from "../model";

export type PoolView = "table" | "gallery";

/** Thumbnail sides (css px): the table's, the gallery's. */
const TABLE_THUMB = 56;
const GALLERY_THUMB = 150;

export const ADD_CROP_HINT = "To add a crop, freeze a region with the lens in any viewer and press S.";

/** A saved crop's actions (the pool's row menu). */
export function cropMenu(r: SavedResult, rename: (r: SavedResult) => void, navigate: (to: string) => void): MenuItem[] {
  const link = viewerLink(r);
  const sky = skyLink(r);
  const tiers = r.logical_tiers.filter((t) => r.files?.[t] || r.inspect_paths?.[t]);
  const inFiles = tiers.filter((t) => inspectLink(r, t));
  return [
    { label: link?.label ?? "Source viewer unknown", disabled: !link, onSelect: () => link && navigate(link.to) },
    { label: "Show on sky", disabled: !sky, onSelect: () => sky && navigate(sky) },
    { type: "sub", label: "Open in Files", disabled: !inFiles.length,
      items: inFiles.map((t) => ({ label: `${t}.fits`, onSelect: () => { const to = inspectLink(r, t); if (to) navigate(to); } })) },
    { type: "sub", label: "Download FITS", disabled: !tiers.length,
      items: tiers.map((t) => ({ label: `${t}.fits`, onSelect: () => { window.location.assign(fitsUrl(r.id, t)); } })) },
    { type: "separator" },
    { label: "Open its card", onSelect: () => openInspector({ kind: "figure", id: r.id }) },
    { label: "Rename…", onSelect: () => rename(r) },
    { label: "Delete…", tone: "danger", onSelect: () => { void confirmDeleteResults([r]); } },
  ];
}

/** The text a crop's position, source and tiers are found by. */
const objectText = (r: SavedResult) => {
  const o = r.source?.object;
  return [o?.id, o?.label, sourceLabel(r), r.source?.collection, ...r.logical_tiers].filter(Boolean).join(" ");
};

export function CropPool({ pool, loading, regime, view, onView, find, onFind, columns, onColumns, onToggle, rows, onOpen, onRename, onDeleteTicked, capped }: {
  pool: readonly SavedResult[]; loading: boolean; regime: string;
  view: PoolView; onView: (v: PoolView) => void; find: string; onFind: (q: string) => void;
  /** The sheet's columns (ticked crops), in sheet order. */
  columns: readonly string[]; onColumns: (keys: string[]) => void; onToggle: (id: string, on: boolean) => void;
  /** The sheet's rows (a crop lacking one is marked). */
  rows: readonly string[];
  onOpen: (r: SavedResult) => void; onRename: (r: SavedResult) => void;
  /** Delete the ticked crops (one danger confirm for all of them). */
  onDeleteTicked: () => void;
  /** No more columns fit: unticked crops cannot be added. */
  capped: boolean;
}) {
  const navigate = useNavigate();
  const newest = useMemo(() => [...pool]
    .sort((a, b) => String(b.created_utc ?? "").localeCompare(String(a.created_utc ?? ""))), [pool]);
  // The gallery matches words; the table reads the same box as its filter language.
  const shown = useMemo(() => newest.filter((r) => galleryMatches(r, find)), [newest, find]);
  const tableColumns = useMemo<DataColumn<SavedResult>[]>(() => [
    { id: "thumb", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: TABLE_THUMB + 16,
      cell: (r) => <ThumbButton result={r} size={TABLE_THUMB} onOpen={() => onOpen(r)} /> },
    { id: "label", header: "Saved crop", width: 180, accessor: (r) => r.label, filterText: (r) => `${r.label} ${r.id} ${objectText(r)}`,
      cell: (r) => {
        const missing = missingRecipes(r, rows).length;
        const side = cropSideArcsec(r);
        return (
          <span className="fig-cell-stack">
            <span className="fig-ellipsis" title={r.label}>{r.label}</span>
            <span className="fig-ellipsis muted">
              {sourceLabel(r)}{side != null ? ` · ${formatNumber(side, { digits: 2 })}″` : ""}
              {missing ? <span className="fig-warn"> · lacks {missing} row{missing === 1 ? "" : "s"}</span> : null}
            </span>
          </span>
        );
      } },
    { id: "wcs", header: "WCS", accessor: (r) => (r.wcs_preserved ? "wcs" : (r.wcs_tiers ?? []).length ? "partial" : "none"), width: 92,
      cell: (r) => <WcsBadge result={r} /> },
    { id: "collection", header: "Source", accessor: (r) => sourceLabel(r), width: 110, hidden: true,
      filterText: (r) => `${sourceLabel(r)} ${r.source?.collection ?? ""}` },
    { id: "tiers", header: "Tiers", accessor: (r) => r.logical_tiers.join(" "), width: 140, hidden: true },
    { id: "side", header: "Crop", numeric: true, accessor: (r) => cropSideArcsec(r), width: 70, hidden: true,
      cell: (r) => { const v = cropSideArcsec(r); return v == null ? "—" : `${formatNumber(v, { digits: 2 })}″`; },
      csv: (r) => String(cropSideArcsec(r) ?? "") },
    { id: "radec", header: "RA, Dec", numeric: true, width: 180, hidden: true,
      accessor: (r) => r.center?.ra ?? r.source?.object?.ra ?? null,
      filterText: (r) => { const c = r.center ?? r.source?.object; return c?.ra != null ? `${c.ra} ${c.dec}` : ""; },
      cell: (r) => {
        const c = r.center ?? (r.source?.object?.ra != null ? { ra: r.source.object.ra, dec: r.source.object.dec ?? NaN } : null);
        return c ? <span className="mono">{formatDeg(c.ra, 4)} {formatDeg(c.dec, 4, { signed: true })}</span> : <span className="muted">—</span>;
      } },
    { id: "created", header: "Saved", accessor: (r) => r.created_utc ?? "", width: 96, hidden: true,
      cell: (r) => <span title={formatDateTime(r.created_utc)}>{formatRelative(r.created_utc)}</span> },
    { id: "bytes", header: "Size", numeric: true, accessor: (r) => r.bytes ?? null, width: 80, hidden: true, cell: (r) => formatBytes(r.bytes) },
    { id: "id", header: "Id", accessor: (r) => r.id, width: 200, hidden: true, cell: (r) => <span className="mono">{r.id}</span> },
    { id: "actions", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 40,
      cell: (r) => (
        <Menu label={`Actions for ${r.label}`} align="end" items={cropMenu(r, onRename, navigate)}
          trigger={<IconButton icon="more" size="sm" label={`Actions for ${r.label}`} />} />
      ) },
  ], [navigate, onOpen, onRename, rows]);

  return (
    <section className="fig-panel fig-pool" aria-labelledby="fig-pool-title">
      <header className="fig-panel__head">
        <h3 id="fig-pool-title">Saved crops</h3>
        <Segmented<PoolView> size="sm" value={view} onChange={onView} aria-label="Show the crops as"
          options={[{ value: "table", label: "Table" }, { value: "gallery", label: "Gallery", title: "Large thumbnails" }]} />
        <Input size="sm" type="search" icon="search" clearable value={find} onChange={onFind}
          placeholder={view === "table" ? "Find: label, tiers:jwst, side<2" : "Find: label, tier, source"}
          aria-label="Find saved crops" className="fig-pool__find" />
        {columns.length > 0 && (
          <Button size="sm" variant="ghost" icon="close" onClick={onDeleteTicked}
            title="Delete the ticked crops from the results store (one confirm for all of them)">
            Delete the {columns.length === 1 ? "ticked crop" : `${columns.length} ticked`}…
          </Button>
        )}
      </header>
      {pool.length > 0 && <Caption className="fig-pool__hint">{ADD_CROP_HINT} A tick adds the crop to the sheet.</Caption>}
      {capped && <p className="fig-note fig-warn" role="status">The sheet is full: untick a column to add another.</p>}
      {!loading && !pool.length ? (
        <EmptyState compact icon="image" title={`No ${regime} saved crops yet`}>{ADD_CROP_HINT}</EmptyState>
      ) : view === "gallery" ? (loading && !pool.length ? <Skeleton lines={4} /> : (
        <ul className="fig-gallery" aria-label={`${regime} saved crops`}>
          {shown.map((r) => {
            const side = cropSideArcsec(r);
            const on = columns.includes(r.id);
            return (
              <li key={r.id} className="fig-gallery__item" data-on={on || undefined}>
                <button type="button" className="fig-gallery__open" onClick={() => onOpen(r)} aria-label={`View ${r.label} full size`}>
                  <ResultThumb result={r} size={GALLERY_THUMB} className="fig-gallery__img" />
                </button>
                <div className="fig-gallery__cap">
                  <Checkbox checked={on} disabled={!on && capped} onChange={(v) => onToggle(r.id, v)} aria-label={`Put ${r.label} in the sheet`} />
                  <button type="button" className="fig-gallery__label fig-ellipsis" title={`${r.label} — open its card`}
                    onClick={() => openInspector({ kind: "figure", id: r.id })}>{r.label}</button>
                  <Menu label={`Actions for ${r.label}`} align="end" items={cropMenu(r, onRename, navigate)}
                    trigger={<IconButton icon="more" size="sm" label={`Actions for ${r.label}`} />} />
                </div>
                <div className="fig-gallery__meta">
                  <span>{sourceLabel(r)}</span>
                  {side != null && <span>{formatNumber(side, { digits: 2 })}″</span>}
                  <span title={formatDateTime(r.created_utc)}>{formatRelative(r.created_utc)}</span>
                </div>
              </li>
            );
          })}
          {!shown.length && <li className="fig-gallery__none muted">Nothing matches “{find.trim()}”.</li>}
        </ul>
      )) : (
        <DataTable rows={newest} columns={tableColumns} rowKey={(r) => r.id} aria-label={`${regime} saved crops`}
          selectable selected={columns} onSelectedChange={(keys) => onColumns(keys)}
          filter={find} onFilterChange={onFind} searchable={false} exportName={`saved-crops-${regime}`}
          inspect={(r) => ({ kind: "figure", id: r.id })} rowHeight={TABLE_THUMB + 10} dense
          height={newest.length > 6 ? 420 : "auto"} loading={loading && !pool.length}
          empty={find.trim() ? `Nothing matches “${find.trim()}”.` : "No saved crops."} />
      )}
    </section>
  );
}
