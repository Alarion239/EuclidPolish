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
const detectorToken = (row: FieldRow) => row.jwst_observation_id.match(/(?:^|[_-])((?:nrc|mir|nis|fgs)[a-z0-9]+)/i)?.[1] || "resolved after download";
const exposureName = (value: number | string | undefined) => {
  const number = asNumber(value);
  return number == null ? "not listed" : `${number.toFixed(1)} s`;
};
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
  const pairJob = useJob();
  const coverageJob = useJob();
  const inferenceJob = useJob();

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

  const runCoverageScan = () => {
    void coverageJob.run("/api/jwst-euclid/scan-coverage", undefined, {
      onDone: () => {
        window.setTimeout(() => index.reload(), 250);
      },
    });
  };

  const runInference = () => {
    if (!selected?.available) return;
    void inferenceJob.run("/api/jwst-euclid/infer", { field_id: selected.field_id }, {
      onDone: () => {
        window.setTimeout(() => {
          void index.reload();
          void manifest.reload();
        }, 250);
      },
    });
  };

  const pair = manifest.data;
  const canDownload = !!selected && coverageStatus(selected) !== "not_covered" && !pairJob.busy;
  const sourceStatus = index.data?.status;
  const coverageSummary = sourceStatus?.coverage_scan;

  return (
    <Page>
      <PageHead
        eyebrow="reference imaging · archive bridge"
        title="JWST × Euclid"
        sub="Download a matched field, retain both instruments on their native grids, and inspect them with the standard tiered field viewer."
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
            Euclid and JWST share one sky field but keep their own pixel grids. The viewer’s tier strip lets you move between Euclid LR, native JWST, and—when requested—the STARFULL output.
          </p>
        </div>
        <div className="jwst-euclid__hero-stats">
          <Stat k="fields to browse" v={sourceStatus?.count ?? "—"} />
          <Stat k="saved comparisons" v={fields.filter((row) => row.available).length} />
        </div>
      </section>

      <Card className="jwst-euclid__control-card">
        <CardHead title="Pick a field" sub="Choose by target, instrument, filters, and sky position. Archive identifiers stay behind the scenes." right={
          <div className="jwst-euclid__control-actions">
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
                    <Badge tone={coverageTone(row)}>
                      {coverageLabel(row)}
                    </Badge>
                  </span>
                  <span className="jwst-euclid__field-card-meta">
                    <span>{instrumentName(row)}</span><span>{filterName(row)}</span>
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
                <span>{instrumentName(selected)} · {filterName(selected)} · {footprintName(selected)}</span>
                <span className="jwst-euclid__selection-status">
                  {coverageLabel(selected)}{coverageStatus(selected) === "covered" && ` · ${asNumber(selected.euclid_coverage_tile_count) ?? 0} VIS tile${asNumber(selected.euclid_coverage_tile_count) === 1 ? "" : "s"}`}
                </span>
                <div className="jwst-euclid__selection-details">
                  <span><b>JWST product</b>{selected.jwst_observation_id}</span>
                  <span><b>detector</b>{detectorToken(selected)}</span>
                  <span><b>exposure</b>{exposureName(selected.jwst_exposure_time_s)}</span>
                  <span><b>proposal</b>{selected.jwst_proposal_id || "not listed"}</span>
                  <span><b>Euclid product</b>{selected.euclid_file_name || selected.euclid_tile_index}</span>
                </div>
              </div>
              <div className="jwst-euclid__selection-actions">
                <Button variant="primary" onClick={runDownload} disabled={!canDownload}>
                  {pairJob.busy ? "preparing field…" : coverageStatus(selected) === "not_covered" ? "no Euclid coverage" : selected.available ? "refresh saved field" : "download field"}
                </Button>
                <Button variant="ghost" onClick={runInference} disabled={!selected.available || inferenceJob.busy}>
                  {inferenceJob.busy ? "running STARFULL…" : pair?.inference ? "refresh STARFULL" : "run STARFULL combiner"}
                </Button>
              </div>
            </div>}
            </>
          )}
          <JobProgressView job={pairJob.job} error={pairJob.error} />
          <JobProgressView job={coverageJob.job} error={coverageJob.error} />
          <JobProgressView job={inferenceJob.job} error={inferenceJob.error} />
        </CardBody>
      </Card>

      {pair ? (
        <section className="jwst-euclid__viewer" aria-label="JWST and Euclid native-grid field viewer">
          <div className="jwst-euclid__viewer-head">
            <div>
              <div className="eyebrow">native-grid field viewer</div>
              <h2>{pair.target_name || "Unnamed field"}</h2>
              <p>{formatCoord(pair.ra_deg, "N", "S")} · {formatCoord(pair.dec_deg, "E", "W")} · {(asNumber(pair.size_arcsec) ?? 0).toFixed(1)}″ field. Use the tier strip for Euclid LR and native JWST pixels.</p>
            </div>
            <div className="jwst-euclid__viewer-actions">
              <Badge tone="good">native WCS grids</Badge>
              <a className="ui-btn ui-btn--sm ui-btn--ghost" href={`/api/jwst-euclid/field/${pair.field_id}/download/jwst_native`}>
                JWST FITS
              </a>
              {pair.inference && <a className="ui-btn ui-btn--sm ui-btn--ghost" href={`/api/jwst-euclid/field/${pair.field_id}/download/starfull`}>
                STARFULL FITS
              </a>}
            </div>
          </div>
          <CutoutViewer
            key={`${pair.field_id}:${pair.inference?.combiner_kind ?? "pending"}`}
            collection="jwst-euclid"
            params={{ field: pair.field_id }}
            initialTiers={["lr", "jwst"]}
          />
          <div className="jwst-euclid__meta">
            <Stat k="Euclid LR" v="VIS native grid" />
            <Stat k="JWST tier" v={`${pair.jwst_instrument || "imaging"} · native pixels`} />
            <Stat k="STARFULL" v={pair.inference?.combiner_label || "run on matching VIS + Y + J + H"} />
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
