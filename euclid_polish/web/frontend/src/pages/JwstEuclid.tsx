import { useEffect, useMemo, useState } from "react";
import { useResource } from "../hooks";
import { JobProgressView, useJob } from "../jobs";
import { CutoutViewer } from "../legacy";
import { Badge, Button, Card, CardBody, CardHead, Empty, Field, Input, Page, PageHead, Segmented, Select, Spinner, Stat } from "../ui";
import "./jwst-euclid.css";

type FieldRow = {
  field_id: string;
  jwst_archive: string;
  jwst_observation_id: string;
  jwst_target_name?: string;
  jwst_proposal_id?: string;
  jwst_instrument?: string;
  jwst_filters?: string;
  jwst_exposure_time_s?: number | string;
  euclid_file_name?: string;
  euclid_tile_index: string;
  euclid_ra_deg?: number | string;
  euclid_dec_deg?: number | string;
  jwst_ra_deg?: number | string;
  jwst_dec_deg?: number | string;
  jwst_distance_deg?: number | string;
  jwst_product_count?: number;
  jwst_row_count?: number;
  footprint_status?: string;
  euclid_coverage_status?: "unchecked" | "covered" | "not_covered" | "error";
  euclid_coverage_tile_count?: number | string;
  euclid_coverage_error?: string;
  available: boolean;
};
type FieldIndex = {
  fields: FieldRow[];
  status: {
    partial?: boolean;
    count: number;
    source_files?: string[];
    source_manifest?: Record<string, unknown>;
    coverage_scan?: {
      checked_count?: number;
      covered_count?: number;
      not_covered_count?: number;
      error_count?: number;
      unique_count?: number;
      updated_utc?: string;
    };
  };
};
type Manifest = {
  field_id: string;
  jwst_archive: string;
  jwst_observation_id: string;
  jwst_product: string;
  euclid_tile_index: string;
  euclid_product?: string;
  target_name?: string;
  jwst_instrument?: string;
  jwst_filters?: string;
  ra_deg: number | string;
  dec_deg: number | string;
  size_arcsec: number | string;
  shape?: [number, number];
  alignment: { method: string; target_grid: string; source_units: string; target_units: string };
  display: { euclid: { display_min: number; display_max: number }; jwst: { display_min: number; display_max: number } };
  files: { euclid?: string; jwst_native?: string };
  euclid_metadata?: PixelMetadata;
  jwst_metadata?: PixelMetadata;
  jwst_bands?: Array<{ key: string; filter: string; product: string; metadata?: PixelMetadata }>;
  inference?: {
    mode: "starfull";
    combiner_kind: string;
    combiner_label: string;
    pixel_scale_arcsec: number;
    shape: number[];
    files: { lr: string; starfull: string };
  };
};
type PixelMetadata = {
  shape?: number[];
  pixel_scale_arcsec?: number[];
  units?: string;
  instrument?: string;
  detector?: string;
  filter?: string;
  pupil?: string;
  exposure_s?: number | string;
};

const asNumber = (value: number | string | undefined): number | null => {
  if (value == null || value === "") return null;
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : null;
};

const formatCoord = (value: number | string | undefined, positive: string, negative: string) => {
  const number = asNumber(value);
  if (number == null) return "—";
  return `${Math.abs(number).toFixed(5)}° ${number >= 0 ? positive : negative}`;
};

const targetName = (row: FieldRow) => row.jwst_target_name?.trim() || "Unnamed JWST field";
const instrumentName = (row: FieldRow) => row.jwst_instrument?.split("/")[0]?.trim() || "JWST imaging";
const filterName = (row: FieldRow) => row.jwst_filters?.trim() || "filter not listed";
const footprintName = (row: FieldRow) => row.footprint_status === "exact_intersection" ? "footprints intersect" : "nearby candidate";
const fieldRa = (row: FieldRow) => row.jwst_ra_deg ?? row.euclid_ra_deg;
const fieldDec = (row: FieldRow) => row.jwst_dec_deg ?? row.euclid_dec_deg;
type FieldView = "all" | "exact" | "saved" | "covered" | "uncovered" | "unchecked";

const coverageStatus = (row: FieldRow) => row.euclid_coverage_status || "unchecked";
const coverageLabel = (row: FieldRow) => {
  const status = coverageStatus(row);
  if (status === "covered") return "Euclid covered";
  if (status === "not_covered") return "No Euclid coverage";
  if (status === "error") return "Check failed";
  return "Needs checking";
};
const coverageTone = (row: FieldRow): "good" | "warn" | undefined => {
  const status = coverageStatus(row);
  if (status === "covered") return "good";
  if (status === "not_covered" || status === "error") return "warn";
  return undefined;
};

export default function JwstEuclidPage() {
  const index = useResource<FieldIndex>("/api/jwst-euclid/fields", [], { ttl: 60_000 });
  const [selectedId, setSelectedId] = useState("");
  const [search, setSearch] = useState("");
  const [instrument, setInstrument] = useState("all");
  const [view, setView] = useState<FieldView>("all");
  const [carouselIndex, setCarouselIndex] = useState(0);
  const [viewerVersion, setViewerVersion] = useState(0);
  const pairJob = useJob();
  const bulkDownloadJob = useJob();
  const coverageJob = useJob();
  const inferenceJob = useJob();
  const nexusJob = useJob();
  const [nexusFilter, setNexusFilter] = useState("F200W");

  const fields = index.data?.fields ?? [];
  const instruments = useMemo(() => Array.from(new Set(fields.map(instrumentName))).sort(), [fields]);
  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();
    return fields.filter((row) => {
      const searchable = [targetName(row), instrumentName(row), filterName(row),
        asNumber(fieldRa(row))?.toFixed(4), asNumber(fieldDec(row))?.toFixed(4)].join(" ").toLowerCase();
      const matchesSearch = !query || searchable.includes(query);
      const matchesInstrument = instrument === "all" || instrumentName(row) === instrument;
      const matchesView = view === "all"
        || (view === "exact" && row.footprint_status === "exact_intersection")
        || (view === "saved" && row.available)
        || (view === "covered" && coverageStatus(row) === "covered")
        || (view === "uncovered" && coverageStatus(row) === "not_covered")
        || (view === "unchecked" && ["unchecked", "error"].includes(coverageStatus(row)));
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
    void pairJob.run("/api/jwst-euclid/download", { field_id: selected.field_id, size_arcsec: 30 }, {
      onDone: () => {
        window.setTimeout(() => index.reload(), 250);
      },
    });
  };

  const runCoverageScan = () => {
    void coverageJob.run("/api/jwst-euclid/scan-coverage", undefined, {
      onDone: () => {
        window.setTimeout(() => index.reload(), 250);
      },
    });
  };

  const runRemainingDownloads = () => {
    void bulkDownloadJob.run("/api/jwst-euclid/download-all", { size_arcsec: 30 }, {
      onDone: () => {
        window.setTimeout(() => index.reload(), 250);
      },
    });
  };

  const runInference = (field: FieldRow | null) => {
    if (!field?.available) return;
    void inferenceJob.run("/api/jwst-euclid/infer", { field_id: field.field_id }, {
      onDone: () => {
        window.setTimeout(() => {
          void index.reload();
          void manifest.reload();
          setViewerVersion((version) => version + 1);
        }, 250);
      },
    });
  };

  const pair = manifest.data;
  const savedFields = fields.filter((row) => row.available);
  const carouselField = savedFields[carouselIndex] ?? savedFields[0] ?? null;
  const canDownload = !!selected && coverageStatus(selected) !== "not_covered" && !pairJob.busy;
  const remainingDownloadCount = fields.filter((row) => !row.available && coverageStatus(row) !== "not_covered").length;
  const sourceStatus = index.data?.status;
  const coverageSummary = sourceStatus?.coverage_scan;
  const downloadNexus = () => {
    void nexusJob.run("/api/jwst-euclid/nexus/download-field", { filter: nexusFilter });
  };

  return (
    <Page>
      <PageHead
        eyebrow="reference imaging · archive bridge"
        title="JWST × Euclid"
        sub="Choose a sky location. Its JWST filters download together beside one Euclid counterpart, while every instrument keeps its native grid."
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
            One card is one sky location, not one archive product. The location groups its JWST filters automatically; Euclid coverage is checked before the shared field is saved.
          </p>
        </div>
        <div className="jwst-euclid__hero-stats">
          <Stat k="fields to browse" v={sourceStatus?.count ?? "—"} />
          <Stat k="saved comparisons" v={fields.filter((row) => row.available).length} />
        </div>
      </section>

      <Card className="jwst-euclid__control-card">
        <CardHead title="NEXUS quick-release comparison"
          sub="Download one full public NEXUS Deep Epoch 05 mosaic, derive the tile centres from its WCS, then fetch all released Euclid VIS, Y, J, and H coverage. Each retained comparison row is exactly 255 × 255 Euclid VIS pixels with a larger native-JWST counterpart; reruns reuse every readable cached tile and only fill missing bands." />
        <CardBody>
          <div className="grid" style={{ gridTemplateColumns: "minmax(220px, 360px)", gap: "var(--s3)" }}>
            <Field label="NEXUS filter"><Select value={nexusFilter} onChange={setNexusFilter} options={[
              { value: "F200W", label: "F200W · 30 mas · 1.0 GB" },
              { value: "F444W", label: "F444W · 60 mas · 252 MB" },
            ]} /></Field>
          </div>
          <div className="row" style={{ marginTop: "var(--s3)", gap: "var(--s3)", alignItems: "center" }}>
            <Button variant="primary" onClick={downloadNexus} disabled={nexusJob.busy}>
              {nexusJob.busy ? "covering NEXUS with four-band Euclid…" : "cache NEXUS mosaic + four-band Euclid"}
            </Button>
            <span className="muted">No coordinate is chosen manually. The NEXUS mosaic WCS sets every tile centre; F200W produces 850 × 850 JWST pixels per Euclid tile, and F444W produces 425 × 425.</span>
          </div>
          {(nexusJob.job || nexusJob.error) && <div style={{ marginTop: "var(--s3)" }}>
            <JobProgressView job={nexusJob.job} error={nexusJob.error} />
          </div>}
        </CardBody>
      </Card>

      <Card className="jwst-euclid__control-card">
        <CardHead title="Pick a sky location" sub="Choose by target, camera, available JWST filters, and sky position. Archive products remain behind the scenes." right={
          <div className="jwst-euclid__control-actions">
            <Button size="sm" variant="primary" onClick={runRemainingDownloads}
              disabled={bulkDownloadJob.busy || pairJob.busy || fields.length === 0 || remainingDownloadCount === 0}>
              {bulkDownloadJob.busy ? "downloading locations…" : `download remaining (${remainingDownloadCount})`}
            </Button>
            <Button size="sm" variant="primary" onClick={runCoverageScan} disabled={coverageJob.busy || fields.length === 0}>
              {coverageJob.busy ? "scanning Euclid…" : "scan Euclid coverage"}
            </Button>
            <Button size="sm" variant="ghost" onClick={index.reload}>↻ refresh index</Button>
          </div>
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
                  { value: "covered", label: "Euclid covered" },
                  { value: "uncovered", label: "No Euclid" },
                  { value: "unchecked", label: "Needs checking" },
                ]} />
              </div>
            </div>
            <div className="jwst-euclid__scan-summary">
              <span><strong>{coverageSummary?.checked_count ?? 0}</strong> / {coverageSummary?.unique_count ?? "—"} unique JWST positions checked</span>
              <span className="jwst-euclid__scan-summary-good">{coverageSummary?.covered_count ?? 0} covered</span>
              <span className="jwst-euclid__scan-summary-warn">{coverageSummary?.not_covered_count ?? 0} without Euclid coverage</span>
              {(coverageSummary?.error_count ?? 0) > 0 && <span className="jwst-euclid__scan-summary-warn">{coverageSummary?.error_count} query errors</span>}
            </div>
            <div className="jwst-euclid__results-head">
              <span>{filtered.length.toLocaleString()} fields shown</span>
              <span className="mono">one location · grouped JWST filters</span>
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
                    <Badge tone={coverageTone(row)}>
                      {coverageLabel(row)}
                    </Badge>
                  </span>
                  <span className="jwst-euclid__field-card-meta">
                    <span>{instrumentName(row)}</span><span>{row.jwst_product_count ?? 1} JWST bands</span>
                  </span>
                  <span className="jwst-euclid__field-card-coords">
                    {formatCoord(fieldRa(row), "N", "S")} · {formatCoord(fieldDec(row), "E", "W")}
                  </span>
                  <span className="jwst-euclid__field-card-action">{isSelected ? "selected" : "choose field"} <span>→</span></span>
                </button>;
              })}
            </div> : <Empty>No fields match these choices. Clear the search or choose a broader instrument/footprint view.</Empty>}
            {selected && <div className="jwst-euclid__selection">
              <div>
                <span className="eyebrow">selected field</span>
                <strong>{targetName(selected)}</strong>
                <span>{instrumentName(selected)} · {selected.jwst_product_count ?? 1} JWST bands · {footprintName(selected)}</span>
                <span className="jwst-euclid__selection-status">
                  {coverageLabel(selected)}{coverageStatus(selected) === "covered" && ` · ${asNumber(selected.euclid_coverage_tile_count) ?? 0} VIS tile${asNumber(selected.euclid_coverage_tile_count) === 1 ? "" : "s"}`}
                </span>
                <div className="jwst-euclid__selection-details">
                  <span><b>JWST filters</b>{filterName(selected)}</span>
                  <span><b>JWST bands</b>{selected.jwst_product_count ?? 1}</span>
                  <span><b>Euclid</b>coverage checked at this location</span>
                </div>
              </div>
              <div className="jwst-euclid__selection-actions">
                <Button variant="primary" onClick={runDownload} disabled={!canDownload}>
                  {pairJob.busy ? "downloading location…" : coverageStatus(selected) === "not_covered" ? "no Euclid coverage" : selected.available ? "refresh grouped field" : "download grouped field"}
                </Button>
                <Button variant="ghost" onClick={() => runInference(selected)} disabled={!selected.available || inferenceJob.busy}>
                  {inferenceJob.busy ? "running STARFULL…" : pair?.inference ? "refresh STARFULL" : "run STARFULL combiner"}
                </Button>
              </div>
            </div>}
            </>
          )}
          <JobProgressView job={pairJob.job} error={pairJob.error} />
          <JobProgressView job={bulkDownloadJob.job} error={bulkDownloadJob.error} />
          <JobProgressView job={coverageJob.job} error={coverageJob.error} />
        </CardBody>
      </Card>

      {savedFields.length > 0 ? (
        <section className="jwst-euclid__viewer" aria-label="JWST and Euclid native-grid field viewer">
          <div className="jwst-euclid__viewer-head">
            <div>
              <div className="eyebrow">saved field carousel</div>
              <h2>Euclid LR × STARFULL × JWST</h2>
              <p>Carousel position 0 is {targetName(savedFields[0])}. Use the previous/next controls to move through every saved sky location, run STARFULL on its Euclid VIS+Y+J+H inputs, then compare LR, SR, and JWST.</p>
            </div>
            <div className="jwst-euclid__viewer-actions">
              <Badge tone="good">{savedFields.length} saved fields</Badge>
              <Badge>field {carouselIndex} · {carouselField ? targetName(carouselField) : "—"}</Badge>
              <Button variant="primary" size="sm" onClick={() => runInference(carouselField)}
                disabled={!carouselField || inferenceJob.busy}>
                {inferenceJob.busy ? "running STARFULL…" : "run STARFULL on Euclid"}
              </Button>
            </div>
          </div>
          <JobProgressView job={inferenceJob.job} error={inferenceJob.error} />
          <CutoutViewer
            key={`jwst-carousel-${savedFields.length}-${viewerVersion}`}
            collection="jwst-euclid"
            initialTiers={["lr", "sr", "jwst"]}
            initialIndex={carouselIndex}
            onChange={(state) => setCarouselIndex((current) => current === state.index ? current : state.index)}
          />
          <div className="jwst-euclid__meta">
            <Stat k="Euclid tier" v="LR · VIS reference" />
            <Stat k="STARFULL tier" v="SR · appears after the active Euclid field is run" />
            <Stat k="JWST tier" v="colour, grayscale native band, or approximate temperature" />
            <Stat k="display stretch" v="one p99.5 scale per panel; native JWST band ratios preserved" />
          </div>
        </section>
      ) : (
        <Card className="jwst-euclid__empty-view"><CardBody><Empty>
          Choose a candidate and download it to add the first field to the LR × JWST colour carousel.
        </Empty></CardBody></Card>
      )}
    </Page>
  );
}
