import { useSearchParams } from "react-router-dom";
import { useResource } from "../hooks";
import {
  Card, CardBody, CardHead, DefList, Empty, Page, PageHead, Spinner,
  Table, type Column,
} from "../ui";

type FileInfo = { basename: string; size_kb: number; mtime: number };
type Hdu = {
  hdu_index: number;
  name: string;
  kind: string;
  shape: number[] | null;
  dtype: string | null;
  cards: [string, string, string][];
};
type InspectResponse = {
  file: FileInfo;
  hdus: Hdu[];
  rel: string;
  allowed_roots: string[];
};
type HeaderRow = { key: string; value: string; comment: string };

const headerColumns: Column<HeaderRow>[] = [
  { header: "key", cell: (r) => <code className="mono">{r.key}</code> },
  { header: "value", cell: (r) => <code className="mono">{r.value}</code> },
  { header: "comment", cell: (r) => <span className="muted">{r.comment}</span> },
];

export default function InspectPage() {
  const [params] = useSearchParams();
  const rel = params.get("fits") ?? "";
  const resource = useResource<InspectResponse>(
    rel ? `/api/inspect?fits=${encodeURIComponent(rel)}` : "",
  );
  const data = resource.data;

  return (
    <Page>
      <PageHead
        eyebrow="tools · fits"
        title="FITS inspector"
        sub="Inspect a project-local FITS file, preview its data, and download the original artifact."
      />
      {!rel ? (
        <Card><CardBody><Empty>Choose a FITS artifact from another page first.</Empty></CardBody></Card>
      ) : resource.loading ? (
        <Card><CardBody><Empty><Spinner /> loading {rel}…</Empty></CardBody></Card>
      ) : !data ? (
        <Card><CardBody><Empty>Could not inspect <code className="mono">{rel}</code>.</Empty></CardBody></Card>
      ) : (
        <>
          <div className="grid" style={{ gridTemplateColumns: "minmax(260px, 0.8fr) minmax(360px, 1.2fr)", gap: "var(--s4)" }}>
            <Card>
              <CardHead title="File" />
              <CardBody>
                <DefList items={[
                  ["name", <code className="mono">{data.file.basename}</code>],
                  ["size", `${data.file.size_kb} KB`],
                  ["path", <code className="mono">{data.rel}</code>],
                  ["HDUs", data.hdus.length],
                ]} />
                <p style={{ marginTop: "var(--s3)" }}>
                  <a className="ui-btn ui-btn--primary" href={`/inspect/download?fits=${encodeURIComponent(data.rel)}`} download>
                    ↓ Download FITS
                  </a>
                </p>
              </CardBody>
            </Card>
            <Card>
              <CardHead title="Preview" />
              <CardBody>
                <div className="ui-figure__paper">
                  <img
                    src={`/inspect/preview.png?fits=${encodeURIComponent(data.rel)}&size=480`}
                    alt={`preview of ${data.file.basename}`}
                    style={{ maxWidth: "100%", height: "auto", display: "block", margin: "0 auto" }}
                  />
                </div>
              </CardBody>
            </Card>
          </div>
          {data.hdus.map((hdu) => (
            <Card key={hdu.hdu_index}>
              <CardHead
                title={`HDU ${hdu.hdu_index} · ${hdu.name}`}
                sub={`${hdu.kind}${hdu.shape ? ` · shape=${hdu.shape.join("×")}` : ""}${hdu.dtype ? ` · dtype=${hdu.dtype}` : ""}`}
              />
              <CardBody>
                <Table
                  columns={headerColumns}
                  rows={hdu.cards.map(([key, value, comment]) => ({ key, value, comment }))}
                  empty="no header cards"
                />
              </CardBody>
            </Card>
          ))}
          <Card>
            <CardHead title="Allowed data roots" />
            <CardBody>
              <ul className="mono">
                {data.allowed_roots.map((root) => <li key={root}>{root}</li>)}
              </ul>
            </CardBody>
          </Card>
        </>
      )}
    </Page>
  );
}
