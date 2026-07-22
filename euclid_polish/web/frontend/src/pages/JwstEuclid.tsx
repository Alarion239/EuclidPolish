import { useEffect, useMemo, useState } from "react";
import { useResource } from "../hooks";
import { JobProgressView, useJob } from "../jobs";
import { Badge, Button, Card, CardBody, CardHead, Empty, Field, Input, Page, PageHead, Segmented, Select, Spinner, Stat } from "../ui";
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
  jwst_instrument?: string;
  jwst_filters?: string;
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

const targetName = (row: FieldRow) => row.jwst_target_name?.trim() || "Unnamed JWST field";
const instrumentName = (row: FieldRow) => row.jwst_instrument?.split("/")[0]?.trim() || "JWST imaging";
const filterName = (row: FieldRow) => row.jwst_filters?.trim() || "filter not listed";
const footprintName = (row: FieldRow) => row.footprint_status === "exact_intersection" ? "footprints intersect" : "nearby candidate";
type FieldView = "all" | "exact" | "saved";

export default function JwstEuclidPage() {
  const index = useResource<FieldIndex>("/api/jwst-euclid/fields", [], { ttl: 60_000 });
  const [selectedId, setSelectedId] = useState("");
  const [search, setSearch] = useState("");
  const [instrument, setInstrument] = useState("all");
  const [view, setView] = useState<FieldView>("all");
  const pairJob = useJob();

  const fields = index.data?.fields ?? [];
  const instruments = useMemo(() => Array.from(new Set(fields.map(instrumentName))).sort(), [fields]);
  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();
    return fields.filter((row) => {
      const searchable = [targetName(row), instrumentName(row), filterName(row),
        row.euclid_ra_deg?.toFixed(4), row.euclid_dec_deg?.toFixed(4)].join(" ").toLowerCase();
      const matchesSearch = !query || searchable.includes(query);
      const matchesInstrument = instrument === "all" || instrumentName(row) === instrument;
      const matchesView = view === "all"
        || (view === "exact" && row.footprint_status === "exact_intersection")
        || (view === "saved" && row.available);
      return matchesSearch && matchesInstrument && matchesView;
    }).slice(0, 500);
  }, [fields, instrument, search, view]);
  const selected = filtered.find((row) => row.field_id === selectedId) ?? filtered[0] ?? null;
  const effectiveId = selected?.field_id ?? "";
  const manifest = useResource<Manifest>(
    effectiveId && selected?.available ? `/api/jwst-euclid/field.json?id=${encodeURIComponent(effectiveId)}` : null,
  );

  useEffect(() => {
    if (!filtered.some((row) => row.field_id === selectedId)) setSelectedId(filtered[0]?.field_id ?? "");
  }, [filtered, selectedId]);

  const runDownload = () => {
    if (!selected) return;
    void pairJob.run("/api/jwst-euclid/download", {
      jwst_archive: selected.jwst_archive,
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
          <Stat k="fields to browse" v={sourceStatus?.count ?? "—"} />
          <Stat k="saved comparisons" v={fields.filter((row) => row.available).length} />
        </div>
      </section>

      <Card className="jwst-euclid__control-card">
        <CardHead title="Pick a field" sub="Choose by target, instrument, filters, and sky position. Archive identifiers stay behind the scenes." right={
          <Button size="sm" variant="ghost" onClick={index.reload}>↻ refresh index</Button>
        } />
        <CardBody>
          {index.loading && !index.data ? <Empty><Spinner /> loading overlap rows…</Empty> : fields.length === 0 ? (
            <Empty>
              <span>No cached overlap table is available. Run <code>scripts/find_jwst_euclid_overlap.py --jwst-archive esa</code> first.</span>
            </Empty>
          ) : (
            <>
            <div className="jwst-euclid__browse-bar">
              <Field label="search fields">
                <Input value={search} onChange={setSearch} placeholder="target name or coordinates…" />
              </Field>
              <Field label="instrument">
                <Select value={instrument} onChange={setInstrument} options={[
                  { value: "all", label: "All instruments" },
                  ...instruments.map((name) => ({ value: name, label: name })),
                ]} />
              </Field>
              <div className="jwst-euclid__view-filter">
                <span>show</span>
                <Segmented<FieldView> value={view} onChange={setView} options={[
                  { value: "all", label: "All" },
                  { value: "exact", label: "Footprint matches" },
                  { value: "saved", label: "Saved" },
                ]} />
              </div>
            </div>
            <div className="jwst-euclid__results-head">
              <span>{filtered.length.toLocaleString()} fields shown</span>
              <span className="mono">30″ Euclid VIS reference cutout</span>
            </div>
            {filtered.length > 0 ? <div className="jwst-euclid__field-grid" role="listbox" aria-label="Available overlapping fields">
              {filtered.map((row) => {
                const isSelected = selected?.field_id === row.field_id;
                return <button
                  key={row.field_id}
                  type="button"
                  role="option"
                  aria-selected={isSelected}
                  className="jwst-euclid__field-card"
                  data-selected={isSelected}
                  onClick={() => setSelectedId(row.field_id)}
                >
                  <span className="jwst-euclid__field-card-head">
                    <strong>{targetName(row)}</strong>
                    <Badge tone={row.available ? "good" : row.footprint_status === "exact_intersection" ? "good" : "warn"}>
                      {row.available ? "saved" : row.footprint_status === "exact_intersection" ? "matched" : "candidate"}
                    </Badge>
                  </span>
                  <span className="jwst-euclid__field-card-meta">
                    <span>{instrumentName(row)}</span><span>{filterName(row)}</span>
                  </span>
                  <span className="jwst-euclid__field-card-coords">
                    {formatCoord(row.euclid_ra_deg, "N", "S")} · {formatCoord(row.euclid_dec_deg, "E", "W")}
                  </span>
                  <span className="jwst-euclid__field-card-action">{isSelected ? "selected" : "choose field"} <span>→</span></span>
                </button>;
              })}
            </div> : <Empty>No fields match these choices. Clear the search or choose a broader instrument/footprint view.</Empty>}
            {selected && <div className="jwst-euclid__selection">
              <div>
                <span className="eyebrow">selected field</span>
                <strong>{targetName(selected)}</strong>
                <span>{instrumentName(selected)} · {filterName(selected)} · {footprintName(selected)}</span>
              </div>
              <Button variant="primary" onClick={runDownload} disabled={!canDownload}>
                {pairJob.busy ? "preparing field…" : selected.available ? "open saved comparison" : "download + align"}
              </Button>
            </div>}
            </>
          )}
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
              <img src={`/api/jwst-euclid/field/${pair.field_id}/euclid_png`} alt={`Euclid VIS view of ${pair.target_name || "selected field"}`} />
              <figcaption>native archive cutout · {pair.alignment.target_units}</figcaption>
            </figure>
            <div className="jwst-euclid__seam" aria-hidden><span>same WCS</span></div>
            <figure className="jwst-euclid__frame jwst-euclid__frame--jwst">
              <div className="jwst-euclid__frame-label"><span>JWST</span><small>{pair.jwst_instrument || "imaging"} · {pair.jwst_filters || "filter not listed"}</small></div>
              <img src={`/api/jwst-euclid/field/${pair.field_id}/jwst_png`} alt={`JWST aligned view of ${pair.target_name || "selected field"}`} />
              <figcaption>resampled to Euclid VIS pixels · {pair.alignment.source_units}</figcaption>
            </figure>
          </div>
          <div className="jwst-euclid__meta">
            <Stat k="Euclid view" v="VIS reference grid" />
            <Stat k="JWST view" v={`${pair.jwst_instrument || "imaging"} · ${pair.jwst_filters || "filter not listed"}`} />
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
