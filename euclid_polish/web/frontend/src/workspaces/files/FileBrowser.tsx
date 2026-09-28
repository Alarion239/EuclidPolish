/* The Files browser: every inspectable root (in pipeline-stage order, each
   labelled with its stage), breadcrumbs, a folder listing (filter as you type)
   and a bounded deep search below the folder (Enter). A row opens a folder or
   a FITS file; the side-panel button opens a file in the inspector without
   leaving the page. The name column takes the width: size and date drop out
   first when the browser is narrow. */
import { forwardRef, useEffect, useMemo, useState } from "react";
import { useResource } from "../../api/query";
import { openInspector } from "../../app/inspector";
import { formatBytes, formatDateTime, formatRelative } from "../../format";
import {
  Button, Callout, DataTable, EmptyState, Icon, IconButton, Input, type DataColumn,
} from "../../ui";
import { browseUrl, type BrowseEntry, type BrowseResponse } from "./api";
import { basename, dirname, readRecent, rootStage, searchScope, sortRootsByStage } from "./model";

type Props = {
  /** The folder listed ("" = the roots). */
  dir: string;
  onDir: (dir: string) => void;
  /** The deep-search query ("" = none). */
  query: string;
  onQuery: (q: string) => void;
  /** The open file (highlighted). */
  current: string;
  onOpen: (rel: string) => void;
  /** Listing max-height (shorter when the browser stacks above the file). */
  height?: number | string;
};

function entryIcon(e: BrowseEntry) {
  if (e.kind === "fits") return <Icon name="image" size={14} />;
  if (e.kind === "root") return <Icon name="database" size={14} />;
  return <Icon name="layers" size={14} />;
}

export const FileBrowser = forwardRef<HTMLInputElement, Props>(function FileBrowser(
  { dir, onDir, query, onQuery, current, onOpen, height = "min(64vh, 640px)" }, searchRef,
) {
  const res = useResource<BrowseResponse>(browseUrl(dir, query), [], { ttl: 15_000 });
  const data = res.data;
  // The roots listing (cached) names a search's root before its own listing loads.
  const rootsRes = useResource<BrowseResponse>(query && dir ? browseUrl("", "") : null, [], { ttl: 60_000 });
  const [filter, setFilter] = useState(query);
  // A new folder or search starts with the query text in the box.
  useEffect(() => { setFilter(query); }, [dir, query]);

  const searching = !!query;
  const columns = useMemo<DataColumn<BrowseEntry>[]>(() => [
    {
      id: "name", header: "Name",
      filterText: (e) => (searching ? `${e.name} ${e.rel}` : e.kind === "root" ? `${e.name} ${e.rel} ${rootStage(e.root_id)}` : e.name),
      cell: (e) => (
        <span className="insp-fb__name">
          <span className={`insp-fb__icon insp-fb__icon--${e.kind}`} aria-hidden="true">{entryIcon(e)}</span>
          <span className="insp-fb__label">
            <span className="insp-fb__text">{e.name}</span>
            {searching && <span className="insp-fb__sub mono">{dirname(e.rel)}</span>}
            {e.kind === "root" && (
              <span className="insp-fb__sub" title={e.rel}>{rootStage(e.root_id)} · <span className="mono">{e.rel}</span></span>
            )}
          </span>
        </span>
      ),
    },
    {
      id: "size", header: "Size", numeric: true, width: 76, priority: 1,
      cell: (e) => (e.kind === "fits" ? formatBytes(e.size) : ""),
    },
    {
      id: "mtime", header: "Modified", width: 88, hidden: false, priority: 2,
      cell: (e) => (e.mtime ? <span title={formatDateTime(e.mtime)}>{formatRelative(e.mtime)}</span> : ""),
    },
    {
      id: "open", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 52,
      cell: (e) => (e.kind === "fits" ? (
        <IconButton size="sm" icon="panelRight" label="Open in the side panel" data-row-ignore
          onClick={() => openInspector({ kind: "fits", id: e.rel })} />
      ) : null),
    },
  ], [searching]);

  const recent = !dir && !query ? readRecent() : [];
  const entries = useMemo(() => (!dir && !query ? sortRootsByStage(data?.entries ?? []) : data?.entries ?? []),
    [data, dir, query]);
  const up = data?.crumbs && data.crumbs.length > 1 ? data.crumbs[data.crumbs.length - 2].rel : "";
  const where = searchScope(dir, data, data?.roots ?? rootsRes.data?.roots ?? []);

  return (
    <div className="insp-fb">
      <div className="insp-fb__bar">
        <Input ref={searchRef} value={filter} onChange={setFilter} icon="search" clearable size="sm"
          placeholder={dir ? "Filter · Enter searches below" : "Enter searches every root"}
          aria-label="Filter this folder; press Enter to search below it"
          onEnter={() => onQuery(filter.trim())}
          onKeyDown={(e) => { if (e.key === "Escape" && (filter || query)) { setFilter(""); onQuery(""); } }} />
        <IconButton size="sm" icon="reset" label="Refresh the listing" onClick={res.reload} />
      </div>
      <nav className="insp-crumbs" aria-label="Folder">
        <button type="button" className="insp-crumbs__item" onClick={() => onDir("")}
          aria-current={!dir ? "location" : undefined}>
          <Icon name="home" size={12} /> Roots
        </button>
        {(data?.crumbs ?? []).map((c, i, all) => (
          <span key={c.rel} className="insp-crumbs__seg">
            <Icon name="chevronRight" size={12} />
            <button type="button" className="insp-crumbs__item" onClick={() => onDir(c.rel)}
              aria-current={i === all.length - 1 ? "location" : undefined}>{c.name}</button>
          </span>
        ))}
        {dir && <IconButton size="sm" icon="arrowUp" label="Up one folder" className="insp-crumbs__up" onClick={() => onDir(up)} />}
      </nav>
      {searching && (
        <div className="insp-fb__search" role="status">
          <span>Results for <b>“{query}”</b> under {where}</span>
          {data?.truncated && <span className="muted"> · stopped early, refine the search</span>}
          <Button size="sm" variant="ghost" onClick={() => onQuery("")}>Clear</Button>
        </div>
      )}
      {res.error ? (
        <Callout tone="bad" title="Cannot list this folder"
          action={<Button size="sm" onClick={() => onDir("")}>Back to roots</Button>}>
          {res.error.message}
        </Callout>
      ) : (
        <DataTable<BrowseEntry>
          rows={entries} columns={columns} rowKey={(e) => `${e.kind}:${e.rel}`}
          aria-label={searching ? "Search results" : "Folder contents"}
          loading={res.loading} hideToolbar={res.loading} dense searchable={false} height={height}
          filter={searching ? "" : filter} activeKey={current ? `fits:${current}` : null}
          columnVisibility={dir || searching ? undefined : { size: false, mtime: false, open: false }}
          onRowClick={(e) => (e.kind === "fits" ? onOpen(e.rel) : onDir(e.rel))}
          empty={searching ? "No FITS file matches." : "No folders or FITS files here."}
          toolbar={data && !searching && (data.other ?? 0) > 0
            ? <span className="muted insp-fb__other">+{data.other} other files</span> : undefined} />
      )}
      {recent.length > 0 && (
        <section className="insp-fb__recent" aria-label="Recent files">
          <h3 className="insp-fb__heading">Recent</h3>
          <ul>
            {recent.map((rel) => (
              <li key={rel}>
                <button type="button" className="insp-fb__recentItem" onClick={() => onOpen(rel)} title={rel}>
                  <Icon name="image" size={12} /> <span>{basename(rel)}</span>
                  <span className="insp-fb__sub mono">{dirname(rel)}</span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}
      {!res.loading && !res.error && !dir && !searching && !(data?.entries.length) && (
        <EmptyState compact icon="database" title="No data roots on this machine">
          None of the inspectable folders exist yet.
        </EmptyState>
      )}
    </div>
  );
});
