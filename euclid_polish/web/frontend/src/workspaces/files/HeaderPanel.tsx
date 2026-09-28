/* One HDU's header as a searchable table (key / value / comment), with the
   raw 80-column text a click away. */
import { useMemo } from "react";
import { Button, CopyButton, DataTable, safeFileName, downloadText, type DataColumn } from "../../ui";
import type { HduSummary } from "./api";
import { cardRows, type CardRow } from "./model";

const WCS_KEY = /^(CTYPE|CRVAL|CRPIX|CDELT|CUNIT|CD\d_|PC\d_|WCSAXES|RADESYS|EQUINOX|LONPOLE|LATPOLE)/;

const COLUMNS: DataColumn<CardRow>[] = [
  { id: "i", header: "#", width: 48, numeric: true },
  {
    id: "key", header: "Key", width: 110,
    cell: (r) => <code className={`mono insp-hdr__key${WCS_KEY.test(r.key) ? " insp-hdr__key--wcs" : ""}`}>{r.key}</code>,
  },
  { id: "value", header: "Value", cell: (r) => <code className="mono insp-hdr__value">{r.value}</code> },
  { id: "comment", header: "Comment", cell: (r) => <span className="insp-dim">{r.comment}</span> },
];

/** The header as FITS-like text lines (`KEY     = value / comment`). */
export function headerText(rows: CardRow[]): string {
  return rows.map((r) => {
    if (r.key === "COMMENT" || r.key === "HISTORY" || r.key === "") return `${r.key.padEnd(8)} ${r.value}`;
    const value = r.value.length ? `= ${r.value}` : "=";
    return `${r.key.padEnd(8)}${value}${r.comment ? ` / ${r.comment}` : ""}`;
  }).join("\n");
}

export function HeaderPanel({ hdu, fileName, urlKey = "hdr", height = "min(70vh, 720px)" }: {
  hdu: HduSummary; fileName: string; urlKey?: string; height?: number | string;
}) {
  const rows = useMemo(() => cardRows(hdu.cards), [hdu.cards]);
  const stem = safeFileName(`${fileName.replace(/\.fits?(\.gz|\.fz)?$/i, "")}_hdu${hdu.index}_header`);
  return (
    <DataTable<CardRow>
      rows={rows} columns={COLUMNS} rowKey={(r) => String(r.i)} aria-label={`Header of HDU ${hdu.index}`}
      urlKey={urlKey} dense height={height} exportName={stem}
      filterPlaceholder="Filter cards… (key:CRVAL, value:VIS)"
      empty="This HDU has no header cards."
      toolbar={(
        <>
          <CopyButton value={() => headerText(rows)} label="Copy the header as text" />
          <Button size="sm" variant="ghost" icon="download"
            onClick={() => downloadText(`${stem}.txt`, headerText(rows))}>.txt</Button>
        </>
      )} />
  );
}
