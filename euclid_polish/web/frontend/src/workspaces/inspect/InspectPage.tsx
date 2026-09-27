/* Inspect workspace (spec §8.6): a file browser over every inspectable root,
   then one FITS file, image first — a plain file bar, ONE row that picks the
   HDU and the view, and at once the HDU's content: the image in the viewer
   (planes, stats, histogram, sky below it), the table (paged rows, column
   stats), a 1-D plot, the header or the file's provenance; the full HDU table
   is under the view. Every choice is in the URL:
     fits   the open file (project-relative; `?path=` is accepted as an alias)
     dir    the browser folder ("@" = the roots; absent = the file's folder)
     q      the deep search in the browser
     hdu    the selected HDU ("3") or 4-band group ("b:LR_")
     slice  one-shot: a plane ("3", "1,2") or band ("J_E") of the HDU; it
            becomes the viewer's object (`v.fits.id`) before the viewer mounts
     view   image | plot | table | header | provenance
   plus the viewer's `v.fits.*`, the image `stack/bin/render/band`, the table
   `toff/tlim/tsort/tq/tcol/trow` and the header filter `hdr.q`. */
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type RefObject } from "react";
import { invalidate, useResource } from "../../api/query";
import { openInspector } from "../../app/inspector";
import { usePageActions } from "../../app/palette";
import { formatBytes, formatDateTime, formatRelative } from "../../format";
import { useMediaQuery } from "../../hooks/useMediaQuery";
import { useUrlState } from "../../hooks/useUrlState";
import { readStorage, writeStorage } from "../../state/storage";
import {
  Badge, Button, Callout, Card, CardBody, CardHead, DefList, Icon, IconButton, Input, Menu, Select,
  Skeleton, Tabs, copyText, toast,
} from "../../ui";
import { downloadUrl, inspectUrl, type InspectResponse } from "./api";
import { FileBrowser } from "./FileBrowser";
import { HduList } from "./HduList";
import { HeaderPanel } from "./HeaderPanel";
import { ImagePanel } from "./ImagePanel";
import {
  basename, defaultHduKey, dirname, fileCrumbs, groupOptionLabel, hduByKey, hduFacts, hduOptionLabel, INSPECT_WIDE_PX,
  isNarrowWidth, middleSplit, normalizeSummary, pushRecent, sliceTarget, VIEW_LABELS, viewsFor, type Selected, type View,
} from "./model";
import { ProvenancePanel } from "./ProvenancePanel";
import { SkyLink } from "./SkyLink";
import { TablePanel } from "./TablePanel";
import { TrackDialog } from "./TrackDialog";
import { VectorPanel } from "./VectorPanel";
import "./inspect.css";

const ROOTS = "@";

function hdusSub(hdus: number, groups: number): string {
  const g = groups ? ` + ${groups} colour group${groups > 1 ? "s" : ""}` : "";
  return `${hdus} HDU${hdus === 1 ? "" : "s"}${g}`;
}
const BROWSER_KEY = "ep-inspect-browser";

/** Whether the page host is below the two-column breakpoint — measured on
 *  the host (the same box the CSS container query sees), so the JS fold and
 *  the layout always agree; the viewport query covers an unmeasured host. */
function useNarrowHost(ref: RefObject<HTMLElement | null>): boolean {
  const viewportNarrow = useMediaQuery(`(max-width: ${INSPECT_WIDE_PX - 1}px)`);
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const measure = () => setWidth(el.getBoundingClientRect().width);
    measure();
    if (typeof ResizeObserver === "undefined") return undefined;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return isNarrowWidth(width, viewportNarrow);
}

/** Clears every per-HDU URL key (viewer object/tiers/view/colour, table
 *  page/sort/filter/column/row, stats band, header filter/sort), so a newly
 *  selected HDU or file starts clean. The setters coalesce into one write. */
function useHduScopedReset(): () => void {
  const setters = [
    useUrlState("v.fits.id", "")[1], useUrlState("v.fits.i", "")[1], useUrlState("v.fits.t", "")[1],
    useUrlState("v.fits.r", "")[1], useUrlState("v.fits.z", "")[1], useUrlState("v.fits.c", "")[1],
    useUrlState("toff", "")[1], useUrlState("tsort", "")[1], useUrlState("tq", "")[1],
    useUrlState("tcol", "")[1], useUrlState("trow", "")[1], useUrlState("band", "")[1], useUrlState("hdr.q", "")[1],
    useUrlState("hdr.sort", "")[1],
  ];
  const ref = useRef(setters);
  ref.current = setters;
  return () => ref.current.forEach((set) => set(""));
}

/** "LR VIS (HDU 1)" / "LR colour (HDUs 1–4)": the selected HDU in words. */
function hduTitle(sel: Selected): string {
  return sel.group ? groupOptionLabel(sel.group) : hduOptionLabel(sel.hdu ?? { index: 0, name: "", type: "image" });
}

/** The file's and the selected HDU's facts, under the view (the top of the
 *  page belongs to the image). */
function FileFacts({ data, sel }: { data: InspectResponse; sel: Selected }) {
  return (
    <Card className="insp-facts">
      <CardBody>
        <DefList dense items={[
          ["file", <span className="insp-id">
            <span>{formatBytes(data.file.size)}, {data.hdus.length} HDU{data.hdus.length === 1 ? "" : "s"}</span>
            {data.file.compressed && <Badge size="sm">compressed</Badge>}
            {data.stamp && <Badge size="sm" tone="good" dot>PROVID {data.stamp.id}</Badge>}
          </span>],
          ["modified", <span title={formatDateTime(data.file.mtime)}>{formatRelative(data.file.mtime)}</span>],
          ["HDU", <span className="mono">{hduTitle(sel)}: {hduFacts(sel)}</span>],
          sel.hdu?.reason && sel.hdu.type !== "empty" ? ["note", sel.hdu.reason] : null,
        ]} />
      </CardBody>
    </Card>
  );
}

/** The one row above the HDU's content: which HDU (a picker when the file
 *  has several), which view; the HDU's facts when the view has room (not the
 *  image view: its drawing controls share the row). */
function ViewHead({ data, sel, view, views, onHdu, onView }: {
  data: InspectResponse; sel: Selected; view: View | null; views: View[];
  onHdu: (key: string) => void; onView: (v: View) => void;
}) {
  const many = data.hdus.length > 1 || data.band_groups.length > 0;
  const options = [
    // the HDU's name first ("LR VIS (HDU 1)"); an image is the default kind, the others say theirs
    ...data.hdus.map((h) => ({ value: String(h.index), label: hduOptionLabel(h) })),
    ...data.band_groups.map((g) => ({ value: g.id, label: groupOptionLabel(g) })),
  ];
  return (
    <>
      {/* The file's part the page describes (header, table, statistics, the
          viewer's first images); the viewer's chips choose what is drawn. */}
      {many ? (
        <label className="insp-viewbar__pick" title="The HDU the header, table and statistics describe; the chips in the viewer choose the images shown">
          <span className="insp-viewbar__picklabel">HDU</span>
          <Select size="sm" aria-label="HDU for the header, table and statistics" value={sel.key} onChange={onHdu} options={options} className="insp-viewbar__hdu" />
        </label>
      ) : view !== "image" && <span className="insp-viewbar__name mono">{hduTitle(sel)}</span>}
      {view && (
        <Tabs value={view} onChange={(v) => onView(v)} variant="line" aria-label="HDU view"
          className="insp-view__tabs" tabs={views.map((v) => ({ id: v, label: VIEW_LABELS[v] }))} />
      )}
      {view !== "image" && <span className="insp-viewbar__facts mono" title={sel.hdu?.reason ?? undefined}>{hduFacts(sel)}</span>}
    </>
  );
}

function OpenByPath({ onOpen }: { onOpen: (rel: string) => void }) {
  const [text, setText] = useState("");
  return (
    <div className="insp-openpath">
      <Input value={text} onChange={setText} icon="search" placeholder="data/eval_results/…/SR.fits"
        aria-label="Open a FITS path" onEnter={() => text.trim() && onOpen(text.trim())} />
      <Button onClick={() => text.trim() && onOpen(text.trim())} disabled={!text.trim()}>Open</Button>
    </div>
  );
}

export default function InspectPage() {
  const [fits, setFits] = useUrlState("fits", "", { replace: false });
  const [alias, setAlias] = useUrlState("path", "");
  const [dirParam, setDirParam] = useUrlState("dir", "");
  const [query, setQuery] = useUrlState("q", "");
  const [hduParam, setHduParam] = useUrlState("hdu", "");
  const [viewParam, setViewParam] = useUrlState("view", "");
  const [slice, setSlice] = useUrlState("slice", "");
  const [, setStackParam] = useUrlState("stack", "bands");
  const [, setViewerId] = useUrlState("v.fits.id", "");
  const resetScoped = useHduScopedReset();
  const hostRef = useRef<HTMLDivElement>(null);
  const narrow = useNarrowHost(hostRef);
  const [browserOpen, setBrowserOpen] = useState(() => {
    const saved = readStorage(BROWSER_KEY);
    return saved ? saved === "1" : true;
  });
  const [tracking, setTracking] = useState(false);
  const searchRef = useRef<HTMLInputElement>(null);

  // `?path=` (the spec's spelling) is the same as `?fits=`.
  useEffect(() => {
    if (alias && !fits) { setFits(alias); setAlias(""); }
  }, [alias, fits, setFits, setAlias]);

  const res = useResource<InspectResponse>(fits ? inspectUrl(fits) : null, [], { ttl: 30_000 });
  const data = useMemo(() => (res.data ? normalizeSummary(res.data) : null), [res.data]);
  useEffect(() => { if (data?.rel) pushRecent(data.rel); }, [data?.rel]);

  const browseDir = dirParam === ROOTS ? "" : dirParam || (fits ? dirname(fits) : "");
  const onDir = (d: string) => { setDirParam(d === "" ? (fits ? ROOTS : "") : d); setQuery(""); };
  // Wide: the browser sits beside the file (remembered). Narrow: it stacks
  // above the file, so it starts folded once a file is open.
  const [narrowOpen, setNarrowOpen] = useState(false);
  const showBrowser = !fits || (narrow ? narrowOpen : browserOpen);
  const toggleBrowser = () => {
    const next = !showBrowser;
    if (narrow) { setNarrowOpen(next); return; }
    setBrowserOpen(next);
    writeStorage(BROWSER_KEY, next ? "1" : "0");
  };

  const openFile = (rel: string) => {
    if (rel !== fits) { resetScoped(); setHduParam(""); setViewParam(""); }
    setFits(rel);
    setNarrowOpen(false);
  };
  const sel = data ? hduByKey(data, hduParam) ?? hduByKey(data, defaultHduKey(data)) : null;
  const selectHdu = (key: string) => {
    if (key === sel?.key) return;
    resetScoped();
    setHduParam(key);
  };
  // `?slice=` → the viewer's object (and one plane at a time for a band
  // cube), written before the image view mounts the viewer.
  useEffect(() => {
    if (!slice || !data || !sel) return;
    const target = sliceTarget(sel, slice);
    if (target) {
      if (target.planes) setStackParam("planes");
      setViewerId(target.id);
    }
    setSlice("");
  }, [slice, data, sel, setSlice, setStackParam, setViewerId]);
  const views = sel ? viewsFor(sel.hdu, sel.group) : [];
  const view: View | null = views.length ? (views.includes(viewParam as View) ? viewParam as View : views[0]) : null;
  const wcs = sel?.group?.wcs ?? sel?.hdu?.wcs ?? null;
  const unit = sel?.hdu?.bunit ?? sel?.group?.bunit ?? "";
  const rel = data?.rel ?? fits;

  const stepHdu = (delta: number) => {
    if (!data || !sel) return false;
    const keys = [...data.hdus.map((h) => String(h.index)), ...data.band_groups.map((g) => g.id)];
    const next = keys[keys.indexOf(sel.key) + delta];
    if (next == null) return false;
    selectHdu(next);
    return true;
  };

  usePageActions([
    { id: "inspect-search", label: "Search FITS files", group: "Inspect", keywords: ["browse", "find", "open"], shortcut: "/",
      run: () => { if (!showBrowser) toggleBrowser(); setTimeout(() => searchRef.current?.focus(), 0); } },
    { id: "inspect-roots", label: "Browse the data roots", group: "Inspect", run: () => { onDir(""); if (!showBrowser) toggleBrowser(); } },
    { id: "inspect-files", label: showBrowser ? "Hide the file browser" : "Show the file browser", group: "Inspect", disabled: !fits, run: toggleBrowser },
    { id: "inspect-next-hdu", label: "Next HDU", group: "Inspect", shortcut: "j", disabled: !data, run: () => stepHdu(1) },
    { id: "inspect-prev-hdu", label: "Previous HDU", group: "Inspect", shortcut: "k", disabled: !data, run: () => stepHdu(-1) },
    ...views.map((v) => ({ id: `inspect-view-${v}`, label: `Show the ${VIEW_LABELS[v].toLowerCase()}`, group: "Inspect", run: () => setViewParam(v) })),
    { id: "inspect-download", label: "Download this FITS file", group: "Inspect", disabled: !data,
      run: () => { window.location.href = downloadUrl(rel); } },
    { id: "inspect-track", label: "Track this FITS file…", group: "Inspect", keywords: ["backup", "pin", "tracking"], disabled: !data, run: () => setTracking(true) },
    { id: "inspect-panel", label: "Open this file in the side panel", group: "Inspect", disabled: !data, run: () => openInspector({ kind: "fits", id: rel }) },
    { id: "inspect-copy", label: "Copy the file path", group: "Inspect", disabled: !data,
      run: () => { void copyText(rel).then(() => toast.success("Path copied")); } },
  ]);

  const crumbs = data ? fileCrumbs(data.rel, data.root) : [];
  return (
    <div className="insp-host" ref={hostRef}>
      <div className={`insp${showBrowser ? " insp--browser" : ""}${fits ? "" : " insp--start"}`}>
        {!fits && (
          <header className="insp-start">
            <div className="insp-start__text">
              <h2 className="insp-start__title"><Icon name="fileSearch" size={18} /> Open a FITS file</h2>
              <p className="insp-dim">Browse a root below, press Enter in the filter to search, or paste a path.</p>
            </div>
            <OpenByPath onOpen={openFile} />
          </header>
        )}
        {showBrowser && (
          <aside className="insp__browser" aria-label="Files">
            <FileBrowser ref={searchRef} dir={browseDir} onDir={onDir} query={query} onQuery={setQuery}
              current={data?.rel ?? fits} onOpen={openFile}
              height={narrow && fits ? "min(40vh, 360px)" : "min(64vh, 640px)"} />
          </aside>
        )}
        {fits && <div className="insp__main">
          {(
            <header className="insp-filebar">
              <div className="insp-filebar__title">
                <IconButton icon="sidebar" label={showBrowser ? "Hide files" : "Show files"} pressed={showBrowser}
                  onClick={toggleBrowser} />
                {/* one line: where the file is, then its name */}
                <div className="insp-filebar__names">
                  <nav className="insp-crumbs insp-crumbs--file" aria-label="File location">
                    {crumbs.map((c) => {
                      // squeezed, a crumb gives way in the middle ("Post…(repo)"), never to one letter
                      const [head, tail] = middleSplit(c.name);
                      return (
                        <span key={c.rel} className="insp-crumbs__seg">
                          <button type="button" className="insp-crumbs__item insp-crumbs__item--mid" title={`Show ${c.rel} in the browser`}
                            aria-label={c.name} onClick={() => { onDir(c.rel); if (!showBrowser) toggleBrowser(); }}>
                            <span className="insp-crumbs__head">{head}</span>{tail && <span className="insp-crumbs__tail">{tail}</span>}
                          </button>
                          <Icon name="chevronRight" size={11} />
                        </span>
                      );
                    })}
                  </nav>
                  <h2 className="insp-filebar__name mono" title={rel}>{basename(rel)}</h2>
                </div>
              </div>
              <div className="insp-filebar__actions" role="toolbar" aria-label="File actions">
                <Button size="sm" icon="download" href={downloadUrl(rel)} download disabled={!data}>Download</Button>
                <Button size="sm" icon="pin" onClick={() => setTracking(true)} disabled={!data}>Track</Button>
                {wcs && <SkyLink wcs={wcs} />}
                <Menu label="More file actions" align="end" items={[
                  { label: "Open in the side panel", disabled: !data, onSelect: () => openInspector({ kind: "fits", id: rel }) },
                  { label: "Copy the path", onSelect: () => { void copyText(rel).then((ok) => { if (ok) toast.success("Path copied"); }); } },
                  { label: "Reload the file", onSelect: () => { void invalidate("/api/inspect"); void invalidate("/viewer/meta/fits"); } },
                ]} trigger={<IconButton size="sm" icon="more" label="More file actions" />} />
              </div>
            </header>
          )}
          {fits && res.loading && <Skeleton lines={6} />}
          {fits && res.error && (
            <Callout tone="bad" title={res.error.status === 404 ? "File not found" : "Cannot open this file"}
              action={<Button size="sm" onClick={() => { setFits(""); setAlias(""); }}>Close</Button>}>
              {res.error.message}
            </Callout>
          )}
          {data && sel && (
            <>
              {data.scan_truncated && (
                <Callout tone="info" title="Only the primary HDU was read">
                  This gzip file is too large to scan for extensions; download it to see them.
                </Callout>
              )}
              {/* Image first: one row picks the HDU and the view (and, on the
                  image view, how it is drawn); the viewer follows at once. The
                  full HDU table is below the view. */}
              <section className="insp-view" aria-labelledby="insp-hdu-heading">
                <h2 id="insp-hdu-heading" className="sr-only">{hduTitle(sel)}</h2>
                {(() => {
                  const head = (
                    <ViewHead data={data} sel={sel} view={view} views={views} onHdu={selectHdu} onView={setViewParam} />
                  );
                  if (view === "image") {
                    return slice ? <><div className="insp-viewbar">{head}</div><Skeleton height={320} /></>
                      : <ImagePanel fits={data.rel} summary={data} sel={sel} unit={unit} head={head} />;
                  }
                  return (
                    <>
                      <div className="insp-viewbar">{head}</div>
                      <div className="insp-view__body">
                        {view === "plot" && sel.hdu && <VectorPanel fits={data.rel} hdu={sel.hdu} unit={unit} />}
                        {view === "table" && sel.hdu && <TablePanel fits={data.rel} hdu={sel.hdu} />}
                        {view === "header" && sel.hdu && <HeaderPanel hdu={sel.hdu} fileName={data.file.basename} />}
                        {view === "provenance" && <ProvenancePanel fits={data.rel} />}
                        {!view && <p className="insp-note">{sel.hdu?.reason ?? "Nothing to show for this HDU."}</p>}
                      </div>
                    </>
                  );
                })()}
              </section>
              <FileFacts data={data} sel={sel} />
              {(data.hdus.length > 1 || data.band_groups.length > 0) && (
                <Card className="insp-hdus">
                  <CardHead title="HDUs" sub={hdusSub(data.hdus.length, data.band_groups.length)} />
                  <CardBody><HduList summary={data} selected={sel.key} onSelect={selectHdu} /></CardBody>
                </Card>
              )}
            </>
          )}
        </div>}
        {data && <TrackDialog fits={data.rel} open={tracking} onOpenChange={setTracking} />}
      </div>
    </div>
  );
}
