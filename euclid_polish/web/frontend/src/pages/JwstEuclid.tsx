import { useEffect, useMemo, useState } from "react";
import { useResource } from "../hooks";
import { JobProgressView, useJob } from "../jobs";
import { Badge, Button, Card, CardBody, CardHead, Empty, Field, Input, Page, PageHead, Select, Spinner, Stat } from "../ui";
import "./jwst-euclid.css";

type FieldRow = {
  field_id: string;
  jwst_archive: string;
  jwst_observation_id: string;
  jwst_target_name?: string;
  jwst_instrument?: string;
  jwst_filters?: string;
  euclid_tile_index: string;
  euclid_ra_deg?: number;
  euclid_dec_deg?: number;
  jwst_distance_deg?: number;
  footprint_status?: string;
  available: boolean;
};
type FieldIndex = {
  fields: FieldRow[];
  status: { partial?: boolean; count: number; source_files?: string[]; source_manifest?: Record<string, unknown> };
};
type Manifest = {
  field_id: string;
  jwst_archive: string;
  jwst_observation_id: string;
  jwst_product: string;
  euclid_tile_index: string;
  target_name?: string;
  ra_deg: number;
  dec_deg: number;
  size_arcsec: number;
  shape: [number, number];
  alignment: { method: string; target_grid: string; source_units: string; target_units: string };
  display: { euclid: { display_min: number; display_max: number }; jwst: { display_min: number; display_max: number } };
  files: { euclid_png: string; jwst_png: string };
};

const formatCoord = (value: number | undefined, positive: string, negative: string) => {
  if (value == null) return "—";
  return `${Math.abs(value).toFixed(5)}° ${value >= 0 ? positive : negative}`;
};

function FieldOption({ row }: { row: FieldRow }) {
  const source = row.jwst_target_name || row.jwst_observation_id;
  const instrument = row.jwst_instrument?.split("/")[0] || "JWST";
  return <>{source} · {instrument} · Euclid {row.euclid_tile_index}{row.available ? " · cached" : ""}</>;
}

export default function JwstEuclidPage() {
  const index = useResource<FieldIndex>("/api/jwst-euclid/fields", [], { ttl: 60_000 });
  const [selectedId, setSelectedId] = useState("");
  const [search, setSearch] = useState("");
  const [archive, setArchive] = useState("esa");
  const pairJob = useJob();

  const fields = index.data?.fields ?? [];
  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();
    if (!query) return fields.slice(0, 500);
    return fields.filter((row) => [
      row.jwst_target_name, row.jwst_observation_id, row.jwst_instrument,
      row.jwst_filters, row.euclid_tile_index,
    ].some((value) => value?.toLowerCase().includes(query))).slice(0, 500);
  }, [fields, search]);
  const selected = fields.find((row) => row.field_id === selectedId) ?? filtered[0] ?? null;
  const effectiveId = selected?.field_id ?? "";
  const manifest = useResource<Manifest>(
    effectiveId && selected?.available ? `/api/jwst-euclid/field.json?id=${encodeURIComponent(effectiveId)}` : null,
  );

  useEffect(() => {
    if (!selectedId && filtered[0]) setSelectedId(filtered[0].field_id);
  }, [filtered, selectedId]);

  useEffect(() => {
    if (selected?.jwst_archive) setArchive(selected.jwst_archive);
  }, [selected?.field_id, selected?.jwst_archive]);

  const runDownload = () => {
    if (!selected) return;
    void pairJob.run("/api/jwst-euclid/download", {
      jwst_archive: archive,
      euclid_tile_index: selected.euclid_tile_index,
      jwst_observation_id: selected.jwst_observation_id,
      size_arcsec: 30,
    }, {
      onDone: () => {
        window.setTimeout(() => index.reload(), 250);
      },
    });
  };

  const pair = manifest.data;
  const canDownload = !!selected && !pairJob.busy;
  const sourceStatus = index.data?.status;

  return (
    <Page>
      <PageHead
        eyebrow="reference imaging · archive bridge"
        title="JWST × Euclid"
        sub="Download a matched field, register the JWST image on the Euclid VIS grid, and inspect both views side-by-side."
        right={<Badge tone={sourceStatus?.partial ? "warn" : "good"}>
          {sourceStatus?.partial ? "overlap index partial" : "overlap index ready"}
        </Badge>}
      />

      <section className="jwst-euclid__hero" aria-label="JWST Euclid field controls">
        <div className="jwst-euclid__hero-mark" aria-hidden>
          <span className="jwst-euclid__orbit jwst-euclid__orbit--one" />
          <span className="jwst-euclid__orbit jwst-euclid__orbit--two" />
          <span className="jwst-euclid__register" />
        </div>
        <div>
          <div className="eyebrow">same sky · different instruments</div>
          <p className="jwst-euclid__hero-copy">
            The Euclid cutout is the reference grid. JWST stays in its native file for provenance,
            then appears again after a WCS-only remap onto that grid.
          </p>
        </div>
        <div className="jwst-euclid__hero-stats">
          <Stat k="candidate pairs" v={sourceStatus?.count ?? "—"} />
          <Stat k="cached pairs" v={fields.filter((row) => row.available).length} />
        </div>
      </section>

      <Card className="jwst-euclid__control-card">
        <CardHead title="Choose an overlap" sub="Discovery rows are cached locally; downloading is explicit and cache-first." right={
          <Button size="sm" variant="ghost" onClick={index.reload}>↻ refresh index</Button>
        } />
        <CardBody>
          {index.loading && !index.data ? <Empty><Spinner /> loading overlap rows…</Empty> : fields.length === 0 ? (
            <Empty>
              <span>No cached overlap table is available. Run <code>scripts/find_jwst_euclid_overlap.py --jwst-archive esa</code> first.</span>
            </Empty>
          ) : (
            <div className="jwst-euclid__controls">
              <Field label="field">
                <Select value={effectiveId} onChange={setSelectedId} options={filtered.map((row) => ({
                  value: row.field_id,
                  label: `${row.jwst_target_name || row.jwst_observation_id} · ${row.jwst_instrument?.split("/")[0] || "JWST"} · ${row.euclid_tile_index}${row.available ? " · cached" : ""}`,
                }))} />
              </Field>
              <Field label="filter list">
                <Input value={search} onChange={setSearch} placeholder="target, observation, tile…" />
              </Field>
              <div className="jwst-euclid__size-note"><span>cutout side</span><strong>30″</strong><small>Euclid VIS grid</small></div>
              <Field label="JWST archive">
                <Select value={archive} onChange={setArchive} options={[
                  { value: "esa", label: "ESA archive" },
                  { value: "mast", label: "MAST" },
                ]} />
              </Field>
              <Button variant="primary" onClick={runDownload} disabled={!canDownload}>
                {selected?.available ? "use cached pair" : "download + align"}
              </Button>
            </div>
          )}
          {selected && <div className="jwst-euclid__selection mono">
            <span><FieldOption row={selected} /></span>
            <span>{formatCoord(selected.euclid_ra_deg, "N", "S")} · {formatCoord(selected.euclid_dec_deg, "E", "W")}</span>
            <span>{selected.footprint_status || "candidate"}</span>
          </div>}
          <JobProgressView job={pairJob.job} error={pairJob.error} />
        </CardBody>
      </Card>

      {pair ? (
        <section className="jwst-euclid__viewer" aria-label="Aligned JWST and Euclid images">
          <div className="jwst-euclid__viewer-head">
            <div>
              <div className="eyebrow">registered comparison</div>
              <h2>{pair.target_name || "Unnamed field"}</h2>
              <p>{formatCoord(pair.ra_deg, "N", "S")} · {formatCoord(pair.dec_deg, "E", "W")} · {pair.size_arcsec.toFixed(1)}″ field · {pair.shape.join(" × ")} px</p>
            </div>
            <div className="jwst-euclid__viewer-actions">
              <Badge tone="good">WCS aligned</Badge>
              <a className="ui-btn ui-btn--sm ui-btn--ghost" href={`/api/jwst-euclid/field/${pair.field_id}/download/jwst_aligned`}>
                aligned FITS
              </a>
            </div>
          </div>
          <div className="jwst-euclid__pair">
            <figure className="jwst-euclid__frame jwst-euclid__frame--euclid">
              <div className="jwst-euclid__frame-label"><span>EUCLID</span><small>VIS · reference grid</small></div>
              <img src={`/api/jwst-euclid/field/${pair.field_id}/euclid_png`} alt={`Euclid VIS view of ${pair.target_name || pair.field_id}`} />
              <figcaption>native archive cutout · {pair.alignment.target_units}</figcaption>
            </figure>
            <div className="jwst-euclid__seam" aria-hidden><span>same WCS</span></div>
            <figure className="jwst-euclid__frame jwst-euclid__frame--jwst">
              <div className="jwst-euclid__frame-label"><span>JWST</span><small>{pair.jwst_product}</small></div>
              <img src={`/api/jwst-euclid/field/${pair.field_id}/jwst_png`} alt={`JWST aligned view of ${pair.target_name || pair.field_id}`} />
              <figcaption>resampled to Euclid VIS pixels · {pair.alignment.source_units}</figcaption>
            </figure>
          </div>
          <div className="jwst-euclid__meta">
            <Stat k="Euclid tile" v={pair.euclid_tile_index} />
            <Stat k="JWST observation" v={pair.jwst_observation_id} />
            <Stat k="remap" v={pair.alignment.method} />
          </div>
        </section>
      ) : (
        <Card className="jwst-euclid__empty-view"><CardBody><Empty>
          {selected?.available ? "Loading the cached paired field…" : "Choose a candidate and download it to open the registered comparison."}
        </Empty></CardBody></Card>
      )}
    </Page>
  );
}
