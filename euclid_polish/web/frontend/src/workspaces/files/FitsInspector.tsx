/* Inspector kind `fits` (`fits:<project-relative path>`): any page opens a
   FITS file in the side panel, image first — the HDU picker and the view
   tabs, the selected image in a bar-less viewer (or a table's first rows, or
   its header), then the file's facts and a link to the full Files page.
   Registered by ./register.ts. */
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useResource } from "../../api/query";
import { formatBytes, formatCount } from "../../format";
import {
  Badge, Button, Callout, DataTable, DefList, Icon, Select, Skeleton, Tabs, type DataColumn,
} from "../../ui";
import { ImageViewer } from "../../viewer";
import {
  downloadUrl, inspectPageHref, inspectUrl, tableUrl, type InspectResponse, type TablePage,
} from "./api";
import { HeaderPanel } from "./HeaderPanel";
import {
  defaultHduKey, hduByKey, hduFacts, normalizeSummary, viewerParams, viewsFor, VIEW_LABELS, type View,
} from "./model";
import { SkyLink } from "./SkyLink";
import "./files.css";

function MiniTable({ fits, hdu }: { fits: string; hdu: number }) {
  const page = useResource<TablePage>(tableUrl(fits, hdu, { limit: 100 }), [], { ttl: 60_000 });
  const columns = useMemo<DataColumn<unknown[]>[]>(() => (page.data?.columns ?? []).map((c, j) => ({
    id: c.name, header: c.name, numeric: c.kind === "numeric", accessor: (r) => {
      const v = r[j];
      return Array.isArray(v) ? v.join(", ") : v;
    },
  })), [page.data]);
  if (page.error) return <Callout tone="bad" title="Could not read the table">{page.error.message}</Callout>;
  return (
    <>
      <DataTable<unknown[]> rows={page.data?.rows ?? []} columns={columns} rowKey={(_r, i) => String(i)}
        aria-label="First rows" loading={page.loading} dense height={320} />
      {page.data && page.data.total > 100 && (
        <p className="insp-dim">First 100 of {formatCount(page.data.total)} rows.</p>
      )}
    </>
  );
}

export default function FitsInspector({ id }: { id: string }) {
  const res = useResource<InspectResponse>(inspectUrl(id), [], { ttl: 30_000 });
  const [key, setKey] = useState<string | null>(null);
  const [view, setView] = useState<View | null>(null);
  const s = useMemo(() => (res.data ? normalizeSummary(res.data) : null), [res.data]);
  if (res.loading) return <Skeleton lines={6} />;
  if (res.error || !s) {
    return <Callout tone="bad" title="Cannot open this FITS file">{res.error?.message ?? "No data."}</Callout>;
  }
  const sel = hduByKey(s, key ?? defaultHduKey(s)) ?? hduByKey(s, defaultHduKey(s));
  const views: View[] = viewsFor(sel?.hdu ?? null, sel?.group).filter((v) => v !== "provenance" && v !== "plot");
  const current = view && views.includes(view) ? view : views[0];
  const wcs = sel?.group?.wcs ?? sel?.hdu?.wcs;
  const options = [
    ...s.hdus.map((h) => ({ value: String(h.index), label: `${h.index} ${h.name} (${h.type})` })),
    ...s.band_groups.map((g) => ({ value: g.id, label: g.label })),
  ];
  // Image first: the HDU picker and the view tabs, then the content; the
  // file's facts and actions follow.
  return (
    <div className="fits-insp">
      {options.length > 1
        ? <Select aria-label="HDU" value={sel?.key ?? "0"} onChange={(v) => { setKey(v); setView(null); }} options={options} />
        : null}
      {current && (
        <Tabs value={current} onChange={(v) => setView(v as View)} variant="line" aria-label="FITS view"
          tabs={views.map((v) => ({ id: v, label: VIEW_LABELS[v] }))}>
          {sel && current === "image" && (
            // No toolbar: the frame gets the panel's width; colour follows the
            // Display panel, the full controls are one click away (Open in Files).
            <ImageViewer collection="fits" params={viewerParams(s.rel, sel, { stack: "bands", bin: "auto", render: "asinh" })}
              toolbar="none" nav={false} />
          )}
          {sel?.hdu && current === "table" && <MiniTable fits={s.rel} hdu={sel.hdu.index} />}
          {sel?.hdu && current === "header" && (
            <HeaderPanel hdu={sel.hdu} fileName={s.file.basename} urlKey="" height={360} />
          )}
        </Tabs>
      )}
      {sel && <p className="insp-dim fits-insp__meta">{hduFacts(sel)}</p>}
      <div className="fits-insp__actions">
        <Button asChild size="sm" variant="primary">
          <Link to={inspectPageHref(s.rel, sel?.key)}><Icon name="fileSearch" /><span className="ui-btn__label">Open in Files</span></Link>
        </Button>
        <Button size="sm" icon="download" href={downloadUrl(s.rel)} download>Download</Button>
        {wcs && <SkyLink wcs={wcs} label="Sky" />}
      </div>
      <DefList dense items={[
        ["file", <code className="mono insp-break">{s.rel}</code>],
        ["size", `${formatBytes(s.file.size)}, ${s.hdus.length} HDU${s.hdus.length === 1 ? "" : "s"}`],
        s.stamp ? ["PROVID", <Badge size="sm" tone="good">{s.stamp.id}</Badge>] : null,
      ]} />
    </div>
  );
}
