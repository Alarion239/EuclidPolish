/* Side-by-side input-domain inspection: one real-Euclid LR tile and one
   synthetic LR record, independently indexed but rendered through the exact
   same client-side colour transfer. */
import { useEffect, useRef, useState } from "react";
import { NavLink } from "react-router-dom";
import { StepById } from "../fasrc";
import { useResource } from "../hooks";
import { JobProgressView, useJob } from "../jobs";
import { CutoutViewer, type ViewerApi } from "../legacy";
import {
  Badge, Button, Card, CardBody, CardHead, Empty, Page, PageHead, Segmented,
  Spinner,
} from "../ui";
import {
  archiveFieldBreakdown,
  archiveOverview,
  archiveSampleProvenance,
  shortArchiveFingerprint,
  type ArchiveCollectionMeta,
} from "./archiveFields";
import "./synthetic-real.css";

type Subset = "test" | "validate" | "train";
type ColorMode = "VIS" | "Y_E" | "J_E" | "H_E" | "lupton" | "temp";
type ViewTransfer = { color: ColorMode; knee: number; gain: number };

type SkyMeta = {
  count: number;
  tier_counts?: Record<string, number>;
};
const COLORS: { value: ColorMode; label: string; title: string }[] = [
  { value: "VIS", label: "VIS", title: "Euclid VIS intensity" },
  { value: "Y_E", label: "Y_E", title: "Euclid Y_E intensity" },
  { value: "J_E", label: "J_E", title: "Euclid J_E intensity" },
  { value: "H_E", label: "H_E", title: "Euclid H_E intensity" },
  { value: "lupton", label: "Lupton", title: "4-band solar-balanced Lupton RGB" },
  { value: "temp", label: "Temp", title: "Per-pixel blackbody-temperature colour" },
];

function LogRange(
  { label, value, min, max, format, onChange }: {
    label: string;
    value: number;
    min: number;
    max: number;
    format: (value: number) => string;
    onChange: (value: number) => void;
  },
) {
  const position = Math.round(
    1000 * (Math.log(value) - Math.log(min)) / (Math.log(max) - Math.log(min)),
  );
  const fromPosition = (positionValue: number) => Math.exp(
    Math.log(min) + (positionValue / 1000) * (Math.log(max) - Math.log(min)),
  );
  return (
    <label className="domain-transfer__slider">
      <span>{label}</span>
      <input type="range" min="0" max="1000" value={position}
        onChange={(event) => onChange(fromPosition(Number(event.target.value)))} />
      <output>{format(value)}</output>
    </label>
  );
}

export default function SyntheticRealPage() {
  const [subset, setSubset] = useState<Subset>("test");
  const [transfer, setTransfer] = useState<ViewTransfer>({
    color: "VIS",
    knee: 100,
    gain: 1,
  });
  const [realIndex, setRealIndex] = useState(0);
  const [syntheticIndex, setSyntheticIndex] = useState(0);
  const archiveSync = useJob();
  const realApi = useRef<ViewerApi | null>(null);
  const syntheticApi = useRef<ViewerApi | null>(null);
  const archiveResource = useResource<ArchiveCollectionMeta>(
    "/viewer/meta/archive-fields",
    [],
    { ttl: 0 },
  );
  const syntheticResource = useResource<SkyMeta>(`/viewer/meta/sky?subset=${subset}`);

  useEffect(() => {
    realApi.current?.setView(transfer);
    syntheticApi.current?.setView(transfer);
  }, [transfer]);

  useEffect(() => setSyntheticIndex(0), [subset]);
  useEffect(() => setRealIndex(0), [archiveResource.data?.archive?.collection_fingerprint]);

  const attach = (
    ref: React.MutableRefObject<ViewerApi | null>,
    api: ViewerApi | null,
  ) => {
    ref.current = api;
    api?.setView(transfer);
  };
  const updateTransfer = (patch: Partial<ViewTransfer>) => {
    setTransfer((current) => ({ ...current, ...patch }));
  };
  const archive = archiveResource.data?.archive;
  const archiveCount = archiveResource.data?.count ?? 0;
  const archiveObject = archiveResource.data?.objects?.[realIndex];
  const syntheticCount = syntheticResource.data?.tier_counts?.dirty ?? 0;
  const bothReady = archiveCount > 0 && syntheticCount > 0;
  const sourcesCurrent = bothReady && archive?.current !== false;
  const archiveStale = !!archive?.valid && !!archive.complete && !archive.current;

  function syncArchiveFields() {
    archiveSync.run("/api/archive-fields/sync", {}, {
      onDone: () => archiveResource.reload(),
    });
  }

  return (
    <Page>
      <PageHead
        eyebrow="diagnostics · input domain"
        title="Synthetic–Real comparison"
        sub="Inspect the LR data the model actually receives. The indices move independently; colour, asinh knee, and brightness are locked together."
        right={<Badge tone={sourcesCurrent ? "good" : "warn"}>
          {archiveStale ? "archive source changed" : !bothReady
            ? "input missing" : "both inputs ready"}
        </Badge>}
      />

      <section className="domain-compare" aria-label="Synthetic and real LR comparison">
        <header className="domain-transfer">
          <div className="domain-transfer__intro">
            <span className="domain-transfer__pulse" aria-hidden />
            <div>
              <div className="eyebrow">shared transfer function</div>
              <p>One renderer, one scale, two independent scan positions.</p>
            </div>
          </div>
          <div className="domain-transfer__controls">
            <div className="domain-transfer__colors" aria-label="Colour mode">
              {COLORS.map((mode) => (
                <button key={mode.value} type="button"
                  className="domain-transfer__color"
                  data-active={transfer.color === mode.value}
                  aria-pressed={transfer.color === mode.value}
                  title={mode.title}
                  onClick={() => updateTransfer({ color: mode.value })}>
                  {mode.label}
                </button>
              ))}
            </div>
            <LogRange label="asinh knee" min={5} max={5000} value={transfer.knee}
              format={(value) => `${Math.round(value)} e⁻`}
              onChange={(knee) => updateTransfer({ knee })} />
            <LogRange label="brightness" min={0.1} max={10} value={transfer.gain}
              format={(value) => `${value.toFixed(2)}×`}
              onChange={(gain) => updateTransfer({ gain })} />
          </div>
        </header>

        <div className="domain-compare__lanes">
          <section className="domain-lane domain-lane--real">
            <header className="domain-lane__head">
              <div>
                <div className="domain-lane__source">REAL EUCLID</div>
                <h2>Multipoint archive · LR</h2>
                <p>{archiveOverview(archive)}</p>
                {archive?.ready && <p title={archive.source_plan_fingerprint ?? undefined}>
                  {archiveFieldBreakdown(archive)} · source plan {shortArchiveFingerprint(archive.source_plan_fingerprint)}
                </p>}
                {archiveObject && <p>
                  parent {archiveObject.parent_id} · RA {archiveObject.ra.toFixed(5)}, Dec {archiveObject.dec.toFixed(5)}
                </p>}
              </div>
              {archiveCount > 0 && <span className="domain-lane__counter">
                {archiveSampleProvenance(archiveObject, realIndex, archiveCount)}
              </span>}
            </header>
            <div className="domain-lane__viewer">
              {archiveResource.loading ? <Empty><Spinner /> loading archive samples…</Empty>
                : archiveCount > 0 ? (
                  <CutoutViewer key={archive?.collection_fingerprint ?? "archive-fields"}
                    collection="archive-fields" initialTier="lr" hideToolbar
                    onReady={(api) => attach(realApi, api)}
                    onChange={(state) => setRealIndex(state.index)} />
                ) : (
                  <Empty>
                    <span>{archiveResource.error
                      ? "The archive-field metadata endpoint is unavailable."
                      : archive?.reasons?.[0] ?? "Synchronize the multipoint archive collection below."}</span>
                  </Empty>
                )}
            </div>
          </section>

          <section className="domain-lane domain-lane--synthetic">
            <header className="domain-lane__head">
              <div>
                <div className="domain-lane__source">SYNTHETIC EUCLID</div>
                <h2>Forward model · LR</h2>
                <p>Dirty TFRecords · detector artifacts and warped PSFs included</p>
              </div>
              <div className="domain-lane__head-right">
                <Segmented<Subset> value={subset} onChange={setSubset} options={[
                  { value: "test", label: "test" },
                  { value: "validate", label: "validate" },
                  { value: "train", label: "train" },
                ]} />
                {syntheticCount > 0 && <span className="domain-lane__counter">
                  record {syntheticIndex} / {syntheticCount - 1}
                </span>}
              </div>
            </header>
            <div className="domain-lane__viewer">
              {syntheticResource.loading ? <Empty><Spinner /> loading {subset} records…</Empty>
                : syntheticCount > 0 ? (
                  <CutoutViewer key={subset} collection="sky" params={{ subset }}
                    initialTier="dirty" hideToolbar
                    onReady={(api) => attach(syntheticApi, api)}
                    onChange={(state) => setSyntheticIndex(state.index)} />
                ) : (
                  <Empty>
                    <span>Sync the {subset} dirty records on the <NavLink to="/sky">Sky page</NavLink> first.</span>
                  </Empty>
                )}
            </div>
          </section>
        </div>

        <footer className="domain-compare__foot">
          <span>LR only</span>
          <span>independent indices + autoplay</span>
          <span>identical browser colour math</span>
          <Button size="sm" variant="ghost" onClick={() => {
            archiveResource.reload();
            syntheticResource.reload();
          }}>↻ refresh sources</Button>
        </footer>
      </section>

      <Card className="domain-workflow">
        <CardHead
          title="Multipoint archive reference"
          sub="Generate the four-band samples on FASRC, then synchronize the manifest and FITS bundles used above."
          right={archiveStale
            ? <Badge tone="warn">source changed</Badge>
            : archive?.ready
            ? <Badge tone={archive.current ? "good" : "warn"}>
              {archive.current ? `${archive.parent_count} pointings ready` : "source changed"}
            </Badge>
            : <Badge tone="warn">not synchronized</Badge>}
        />
        <CardBody>
          <StepById stepId="archive_field_sample" embedded />
          <div className="row" style={{ marginTop: "var(--s4)", gap: "var(--s3)", alignItems: "center" }}>
            <Button variant="primary" disabled={archiveSync.busy} onClick={syncArchiveFields}>
              {archiveSync.busy ? "syncing archive fields…" : "⤓ Sync archive fields from FASRC"}
            </Button>
            {archiveSync.busy && <span className="muted"><Spinner /> pulling manifest and bundles…</span>}
          </div>
          {(archiveSync.job || archiveSync.error) && <div style={{ marginTop: "var(--s3)" }}>
            <JobProgressView job={archiveSync.job} error={archiveSync.error} />
          </div>}
        </CardBody>
      </Card>
    </Page>
  );
}
