/* Figures › Results (spec §8.5; image-first pass 2026-09-27): every crop
 * saved from a viewer (lens freeze, then S / "Save crop to results") — as a
 * table (thumbnail, source, tiers, crop size, WCS badge, sky position) or a
 * gallery of large thumbnails (`?view=gallery`). A thumbnail opens the crop
 * full size (Lightbox: every panel, pixels kept); a table row opens the
 * `figure` inspector; per-row actions (open the source viewer, sky,
 * inspect/download FITS, rename, delete) and bulk "build a grid" / delete.
 * Regime filter, view, table filter and sort, and the gallery's own text
 * filter (`?q=`, newest first) live in the URL. */
import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import { formatBytes, formatDateTime, formatDeg, formatNumber, formatRelative } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Badge, Button, Callout, Checkbox, DataTable, EmptyState, IconButton, Input, Menu, Page, Segmented, Skeleton, Tooltip,
  type DataColumn, type MenuItem,
} from "../../../ui";
import { confirmDeleteResults, RenameDialog } from "../actions";
import { URLS, fitsUrl, type SavedResult } from "../api";
import { RegimeBadge, ResultThumb, ThumbButton, WcsBadge } from "../common";
import { ResultLightbox } from "../Lightbox";
import { cropSideArcsec, galleryMatches, gridHref, inspectLink, normalizeIndex, resultRegime, skyLink, sourceLabel, viewerLink } from "../model";
import { useMediaQuery } from "../../../hooks/useMediaQuery";
import "../figures.css";
import "../register";

const REGIMES = ["all", "real", "synthetic"] as const;
type RegimeFilter = (typeof REGIMES)[number];
const REGIME_LABEL: Record<RegimeFilter, string> = { all: "All", real: "Real", synthetic: "Synthetic" };
type View = "table" | "gallery";
/** Thumbnail sides (css px): the table's, the gallery's. */
const TABLE_THUMB = 64;
const GALLERY_THUMB = 200;

function objectText(r: SavedResult): string {
  const o = r.source?.object;
  return [o?.id, o?.label !== r.label ? o?.label : null].filter(Boolean).join(" · ");
}

function rowMenu(r: SavedResult, rename: (r: SavedResult) => void, navigate: (to: string) => void): MenuItem[] {
  const link = viewerLink(r);
  const sky = skyLink(r);
  const regime = resultRegime(r) ?? "real";
  const tiers = r.logical_tiers.filter((t) => r.files?.[t] || r.inspect_paths?.[t]);
  return [
    { label: link?.label ?? "Source viewer unknown", disabled: !link, onSelect: () => link && navigate(link.to) },
    { label: "Open in the grid", onSelect: () => navigate(gridHref([r.id], regime)) },
    { label: "Show on sky", disabled: !sky, onSelect: () => sky && navigate(sky) },
    { type: "separator" },
    { type: "sub", label: "Inspect FITS", disabled: !tiers.length,
      items: tiers.map((t) => ({ label: t, disabled: !inspectLink(r, t), onSelect: () => { const to = inspectLink(r, t); if (to) navigate(to); } })) },
    { type: "sub", label: "Download FITS", disabled: !tiers.length,
      items: tiers.map((t) => ({ label: `${t}.fits`, onSelect: () => { window.location.assign(fitsUrl(r.id, t)); } })) },
    { type: "separator" },
    { label: "Rename…", onSelect: () => rename(r) },
    { label: "Delete…", tone: "danger", onSelect: () => { void confirmDeleteResults([r]); } },
  ];
}

export default function Results() {
  const navigate = useNavigate();
  const [regime, setRegime] = useUrlState<string>("regime", "all");
  const [viewRaw, setView] = useUrlState<string>("view", "table");
  const view: View = viewRaw === "gallery" ? "gallery" : "table";
  const [find, setFind] = useUrlState<string>("q", "");
  const [full, setFull] = useState<SavedResult | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [renaming, setRenaming] = useState<SavedResult | null>(null);
  const index = useResource<unknown>(URLS.results, [], { ttl: 15_000 });
  const norm = useMemo(() => normalizeIndex(index.data), [index.data]);
  const all = norm.results;
  const counts = useMemo(() => ({
    all: all.length,
    real: all.filter((r) => resultRegime(r) === "real").length,
    synthetic: all.filter((r) => resultRegime(r) === "synthetic").length,
  }), [all]);
  const filter = (REGIMES as readonly string[]).includes(regime) ? (regime as RegimeFilter) : "all";
  const rows = useMemo(() => (filter === "all" ? all : all.filter((r) => resultRegime(r) === filter)), [all, filter]);
  const byId = useMemo(() => new Map(all.map((r) => [r.id, r])), [all]);
  const picked = selected.map((id) => byId.get(id)).filter((r): r is SavedResult => !!r);
  const pickedRegimes = new Set(picked.map((r) => resultRegime(r) ?? "real"));
  const gridable = picked.length > 0 && pickedRegimes.size === 1 && picked.length <= norm.maxResults;
  const gridReason = !picked.length ? "Select saved results first"
    : pickedRegimes.size > 1 ? "A grid holds one regime: select only real or only synthetic results"
      : picked.length > norm.maxResults ? `A grid holds at most ${norm.maxResults} columns` : "";
  const buildGrid = () => { if (gridable) navigate(gridHref(picked.map((r) => r.id), [...pickedRegimes][0])); };
  const removeSelected = async () => {
    const gone = await confirmDeleteResults(picked);
    if (gone.length) setSelected((cur) => cur.filter((id) => !gone.includes(id)));
  };

  usePageActions([
    { id: "fig-results-refresh", label: "Refresh saved results", group: "Figures", run: () => void index.reload() },
    { id: "fig-results-grid", label: "Build a grid from the selected results", group: "Figures", disabled: !gridable, run: buildGrid },
    { id: "fig-results-delete", label: "Delete the selected saved results…", group: "Figures", disabled: !picked.length, run: () => void removeSelected() },
    { id: "fig-results-real", label: "Show real saved results", group: "Figures", run: () => setRegime("real") },
    { id: "fig-results-synthetic", label: "Show synthetic saved results", group: "Figures", run: () => setRegime("synthetic") },
    { id: "fig-results-view", label: view === "gallery" ? "Show saved results as a table" : "Show saved results as a gallery", group: "Figures",
      run: () => setView(view === "gallery" ? "table" : "gallery") },
  ]);

  // Below 1280 px the source (the result's second line names it), position
  // and saved-at columns start hidden (the column menu brings them back), so
  // the table fits the pane without a hidden horizontal scroll.
  const narrowPane = useMediaQuery("(max-width: 1279px)");
  const columns = useMemo<DataColumn<SavedResult>[]>(() => [
    { id: "thumb", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: TABLE_THUMB + 20,
      cell: (r) => <ThumbButton result={r} size={TABLE_THUMB} onOpen={() => setFull(r)} /> },
    { id: "label", header: "Result", width: 220, filterText: (r) => `${r.label} ${r.id} ${objectText(r)}`,
      cell: (r) => (
        <span className="fig-cell-stack">
          <span className="fig-ellipsis" title={r.label}>{r.label}</span>
          <span className="fig-ellipsis mono muted" title={r.id}>{objectText(r) || r.id}</span>
        </span>
      ) },
    { id: "regime", header: "Regime", accessor: (r) => resultRegime(r) ?? "", width: 88, hidden: true, cell: (r) => <RegimeBadge regime={resultRegime(r)} /> },
    { id: "collection", header: "Source", accessor: (r) => sourceLabel(r), width: 110, hidden: narrowPane,
      filterText: (r) => `${sourceLabel(r)} ${r.source?.collection ?? ""}`,
      cell: (r) => <Badge size="sm" title={`viewer collection ${r.source?.collection ?? "—"}`}>{sourceLabel(r)}</Badge> },
    { id: "tiers", header: "Tiers", accessor: (r) => r.logical_tiers.join(" "), width: 150,
      cell: (r) => (
        <span className="fig-chips">
          {r.logical_tiers.map((t) => (
            <Badge key={t} size="sm" title={r.files?.[t]?.source_label ?? t}>{t}</Badge>
          ))}
        </span>
      ) },
    { id: "side", header: "Crop", numeric: true, accessor: (r) => cropSideArcsec(r), width: 70,
      cell: (r) => { const s = cropSideArcsec(r); return s == null ? "—" : `${formatNumber(s, { digits: 2 })}″`; },
      csv: (r) => String(cropSideArcsec(r) ?? "") },
    { id: "wcs", header: "WCS", accessor: (r) => (r.wcs_preserved ? "wcs" : (r.wcs_tiers ?? []).length ? "partial" : "none"), width: 96,
      cell: (r) => <WcsBadge result={r} /> },
    { id: "radec", header: "RA, Dec", numeric: true, width: 186, hidden: narrowPane,
      accessor: (r) => r.center?.ra ?? r.source?.object?.ra ?? null,
      filterText: (r) => { const c = r.center ?? r.source?.object; return c?.ra != null ? `${c.ra} ${c.dec}` : ""; },
      cell: (r) => {
        const c = r.center ?? (r.source?.object?.ra != null ? { ra: r.source.object.ra, dec: r.source.object.dec ?? NaN } : null);
        return c ? <span className="mono">{formatDeg(c.ra, 4)} {formatDeg(c.dec, 4, { signed: true })}</span> : <span className="muted">—</span>;
      } },
    { id: "created", header: "Saved", accessor: (r) => r.created_utc ?? "", width: 96, hidden: narrowPane,
      cell: (r) => <span title={formatDateTime(r.created_utc)}>{formatRelative(r.created_utc)}</span> },
    { id: "bytes", header: "Size", numeric: true, accessor: (r) => r.bytes ?? null, hidden: true, cell: (r) => formatBytes(r.bytes) },
    { id: "id", header: "Id", hidden: true, cell: (r) => <span className="mono">{r.id}</span> },
    { id: "actions", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 44,
      cell: (r) => (
        <Menu label={`Actions for ${r.label}`} align="end" items={rowMenu(r, setRenaming, navigate)}
          trigger={<IconButton icon="more" size="sm" label={`Actions for ${r.label}`} />} />
      ) },
  ], [navigate, narrowPane]);
  const newest = useMemo(() => rows.filter((r) => galleryMatches(r, find))
    .sort((a, b) => String(b.created_utc ?? "").localeCompare(String(a.created_utc ?? ""))), [rows, find]);
  const toggle = (id: string, on: boolean) => setSelected((cur) => (on ? [...new Set([...cur, id])] : cur.filter((x) => x !== id)));

  return (
    <Page className="fig-page">
      <div className="fig-bar" role="toolbar" aria-label="Saved results">
        <Segmented size="sm" value={filter} onChange={setRegime} aria-label="Regime"
          options={REGIMES.map((r) => ({ value: r, label: `${REGIME_LABEL[r]} ${counts[r]}` }))} />
        <Segmented<View> size="sm" value={view} onChange={setView} aria-label="View"
          options={[{ value: "table", label: "Table" }, { value: "gallery", label: "Gallery", title: "Large thumbnails" }]} />
        {view === "gallery" && (
          <Input size="sm" type="search" icon="search" clearable value={find} onChange={setFind}
            placeholder="Find: label, tier, source" aria-label="Find saved results" className="fig-bar__find" />
        )}
        <span className="fig-bar__spacer" />
        <span className="fig-bar__note" aria-live="polite">
          {picked.length ? `${picked.length} selected`
            : view === "gallery" && rows.length ? `Newest first · ${newest.length === rows.length ? `all ${rows.length}` : `${newest.length} of ${rows.length}`}` : ""}
        </span>
        <Tooltip content={gridReason || "Open the selection as grid columns"}>
          <span tabIndex={gridable ? -1 : 0}>
            <Button size="sm" icon="columns" disabled={!gridable} onClick={buildGrid}>Grid</Button>
          </span>
        </Tooltip>
        <Button size="sm" variant={picked.length ? "danger" : "default"} icon="close" disabled={!picked.length} onClick={() => void removeSelected()}>Delete</Button>
        <IconButton icon="reset" label="Refresh" size="sm" onClick={() => void index.reload()} />
      </div>
      {index.error && !index.data && (
        <Callout tone="bad" title="Saved results did not load" action={<Button size="sm" onClick={() => void index.reload()}>Retry</Button>}>
          {index.error.message}
        </Callout>
      )}
      {norm.malformed && <Callout tone="bad" title="The results index is malformed">The server sent an unexpected payload.</Callout>}
      {norm.dropped > 0 && <Callout tone="warn" title={`${norm.dropped} malformed saved result${norm.dropped === 1 ? " was" : "s were"} skipped`} />}
      {!index.loading && !index.error && !all.length ? (
        <EmptyState icon="image" title="No saved crops yet"
          action={<Button asChild size="sm"><Link to="/sky/results">Open real results</Link></Button>}>
          In any viewer, freeze a region with the lens and press S (Save crop to results).
        </EmptyState>
      ) : (
        view === "gallery" ? (index.loading && !index.data ? <Skeleton lines={4} /> : (
          <ul className="fig-gallery" aria-label="Saved viewer results">
            {newest.map((r) => {
              const side = cropSideArcsec(r);
              const on = selected.includes(r.id);
              return (
                <li key={r.id} className="fig-gallery__item" data-on={on || undefined}>
                  <button type="button" className="fig-gallery__open" onClick={() => setFull(r)} aria-label={`View ${r.label} full size`}>
                    <ResultThumb result={r} size={GALLERY_THUMB} className="fig-gallery__img" />
                  </button>
                  <div className="fig-gallery__cap">
                    <Checkbox checked={on} onChange={(v) => toggle(r.id, v)} aria-label={`Select ${r.label}`} />
                    <button type="button" className="fig-gallery__label fig-ellipsis" title={`${r.label} — open its card`}
                      onClick={() => openInspector({ kind: "figure", id: r.id })}>{r.label}</button>
                    <Menu label={`Actions for ${r.label}`} align="end" items={rowMenu(r, setRenaming, navigate)}
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
            {!newest.length && (
              <li className="fig-gallery__none muted">
                {find.trim() && rows.length ? `Nothing matches “${find.trim()}”.` : filter === "all" ? "No saved results." : `No ${filter} saved results.`}
              </li>
            )}
          </ul>
        )) : (
          <DataTable rows={rows} columns={columns} rowKey={(r) => r.id} aria-label="Saved viewer results"
            selectable selected={selected} onSelectedChange={(keys) => setSelected(keys)}
            inspect={(r) => ({ kind: "figure", id: r.id })} rowHeight={TABLE_THUMB + 12}
            exportName="saved-viewer-results" urlKey="fr" loading={index.loading && !index.data}
            height="max(420px, calc(100vh - 240px))" defaultSort={[{ id: "created", desc: true }]}
            filterPlaceholder="Filter: collection:real  tiers:jwst  wcs=wcs  side<2"
            empty={filter === "all" ? "No saved results." : `No ${filter} saved results.`} />
        )
      )}
      <RenameDialog result={renaming} open={!!renaming} onOpenChange={(o) => { if (!o) setRenaming(null); }} />
      <ResultLightbox result={full} onClose={() => setFull(null)} />
    </Page>
  );
}
