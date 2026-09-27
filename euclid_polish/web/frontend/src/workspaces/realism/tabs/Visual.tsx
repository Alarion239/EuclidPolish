/* realism/visual (spec §8.3): synthetic vs real, side by side — one real
   Euclid LR sample of the multipoint archive collection and one synthetic
   dirty LR record, indexed independently, each in the image viewer with its
   own (compact) toolbar, both on ONE colour transfer: a colour / knee /
   brightness edit in either viewer is applied to the other (./visual/
   sync.ts; the lock is `?lock=`, the shared transfer `?c=&k=&g=`, default
   Lupton so NISP-only blackouts stay visible). Only the multipoint archive
   collection is used here (the legacy one-pointing real field is a Sky ›
   Real results tile). */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import { useFasrcStatus } from "../../../app/status";
import { StepById } from "../../../fasrc";
import { formatNumber } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { Badge, Button, Card, CardBody, CardHead, EmptyState, IconButton, JobProgress, Page, Segmented, Skeleton, Switch } from "../../../ui";
import { ImageViewer, type ViewerApi, type ViewerState } from "../../../viewer";
import { useArchiveMeta, useSkyMeta } from "../api";
import { archiveFieldBreakdown, archiveOverview, archiveSampleProvenance, shortArchiveFingerprint } from "../archiveFields";
import { BarGroup, BarSpacer, Info, RealismBar, SkyLink } from "../common";
import { JOB, offlinePolicy, runJob, useRealismJob } from "../jobs";
import { DEFAULT_GAIN, DEFAULT_KNEE, editOf, shows, viewPatch, type Reported, type Transfer } from "../visual/sync";

type Subset = "test" | "validate" | "train";
type Lane = "real" | "syn";
const SUBSETS: { value: Subset; label: string }[] = [
  { value: "test", label: "test" }, { value: "validate", label: "validate" }, { value: "train", label: "train" },
];
export const ARCHIVE_SYNC_URL = "/api/archive-fields/sync";

export const syncArchiveFields = () => runJob({
  url: ARCHIVE_SYNC_URL, label: "Sync archive fields from FASRC",
  question: {
    title: "Sync the archive fields from FASRC?", confirmLabel: "Sync",
    message: "Pulls the multipoint manifest and its four-band FITS bundles (SHA-256 checked) and replaces the local collection.",
  },
});

const reportOf = (s: ViewerState): Reported => ({ color: s.color, knee: s.knee, gain: s.gain });

export default function VisualTab() {
  const [rawSubset, setSubset] = useUrlState("sub", "test");
  const subset: Subset = (["test", "validate", "train"].includes(rawSubset) ? rawSubset : "test") as Subset;
  const [color, setColor] = useUrlState("c", "lupton");
  const [knee, setKnee] = useUrlState("k", 0);
  const [gain, setGain] = useUrlState("g", 0);
  const [lock, setLock] = useUrlState("lock", true);
  const shared: Transfer = useMemo(() => ({ color, knee: knee > 0 ? knee : null, gain: gain > 0 ? gain : null }), [color, knee, gain]);
  const sharedRef = useRef(shared);
  sharedRef.current = shared;
  const lockRef = useRef(lock);
  lockRef.current = lock;
  const apis = useRef<Record<Lane, ViewerApi | null>>({ real: null, syn: null });
  const reported = useRef<Record<Lane, Reported | null>>({ real: null, syn: null });
  const [realIndex, setRealIndex] = useState(0);

  const archive = useArchiveMeta();
  const sky = useSkyMeta(subset);
  const sync = useRealismJob(JOB.archiveSync);
  const info = archive.data?.archive;
  const archiveCount = archive.data?.count ?? 0;
  const object = archive.data?.objects?.[realIndex];
  const synCount = sky.data?.tier_counts?.dirty ?? 0;
  const stale = !!info?.valid && !!info.complete && !info.current;
  // The archive sync is a self-connecting job: enabled offline, and says so.
  const fasrc = useFasrcStatus().data;
  const syncHint = offlinePolicy({ self_connects: true }, fasrc ? !fasrc.ssh_connected : false).hint ?? undefined;

  const apply = useCallback((lane: Lane) => {
    const api = apis.current[lane];
    if (api && !shows(reported.current[lane], sharedRef.current)) api.setView(viewPatch(sharedRef.current));
  }, []);
  // A new shared transfer (a reset, a palette colour, an edit while locked)
  // reaches both viewers; re-locking brings a viewer edited on its own back.
  useEffect(() => { apply("real"); apply("syn"); }, [shared, lock, apply]);

  const onReady = (lane: Lane) => (api: ViewerApi | null) => {
    apis.current[lane] = api;
    reported.current[lane] = null;
    if (api) api.setView(viewPatch(sharedRef.current));
  };
  const onState = (lane: Lane) => (s: ViewerState) => {
    const next = reportOf(s);
    const edit = editOf(reported.current[lane], next);
    reported.current[lane] = next;
    if (lane === "real") setRealIndex(s.index);
    // A viewer that now shows the shared transfer is echoing our setView.
    if (!edit || !lockRef.current || shows(next, sharedRef.current)) return;
    if (edit.color != null) setColor(edit.color);
    // The default knee / brightness is the unset URL value (a clean link).
    if (edit.knee != null) setKnee(Math.abs(edit.knee - DEFAULT_KNEE) < 1e-6 ? 0 : Number(edit.knee.toPrecision(4)));
    if (edit.gain != null) setGain(Math.abs(edit.gain - DEFAULT_GAIN) < 1e-6 ? 0 : Number(edit.gain.toPrecision(3)));
  };
  const resetTransfer = () => { setColor("lupton"); setKnee(0); setGain(0); };
  const inspectCurrent = () => { if (object) openInspector({ kind: "archivefield", id: object.id ?? String(object.sample_id) }); };

  usePageActions([
    { id: "visual-lock", label: lock ? "Unlock the synthetic–real transfer" : "Lock the synthetic–real transfer", group: "Visual", run: () => setLock(!lock) },
    { id: "visual-lupton", label: "Synthetic–real: Lupton colour", group: "Visual", run: () => setColor("lupton") },
    { id: "visual-vis", label: "Synthetic–real: VIS", group: "Visual", run: () => setColor("VIS") },
    { id: "visual-reset", label: "Reset the shared transfer", group: "Visual", run: resetTransfer },
    { id: "visual-inspect", label: "Inspect the current archive field", group: "Visual", disabled: !object, run: inspectCurrent },
    { id: "visual-sync", label: "Sync archive fields from FASRC", group: "Visual", keywords: ["archive", "multipoint"],
      run: () => { void syncArchiveFields(); } },
  ]);

  return (
    <Page className="rl-page">
      <RealismBar label="Synthetic–real controls">
        <BarGroup label="synthetic">
          <Segmented size="sm" aria-label="Synthetic subset" value={subset} onChange={setSubset} options={SUBSETS} />
        </BarGroup>
        <BarGroup label="transfer">
          <Switch size="sm" checked={lock} onChange={setLock}>one transfer</Switch>
          <Badge size="sm" title="The shared colour transfer (edit it in either viewer's toolbar)">
            {color} · knee {formatNumber(shared.knee ?? DEFAULT_KNEE, { sig: 4 })} e⁻{shared.knee == null ? " (default)" : ""} · ×{formatNumber(shared.gain ?? DEFAULT_GAIN, { sig: 3 })}
          </Badge>
          <IconButton size="sm" icon="reset" label="Reset the shared transfer (Lupton, default knee)" onClick={resetTransfer}
            disabled={color === "lupton" && knee === 0 && gain === 0} />
        </BarGroup>
        <BarSpacer />
        <Badge size="sm" dot tone={stale ? "warn" : archiveCount > 0 && synCount > 0 ? "good" : "warn"}>
          {stale ? "archive source changed" : archiveCount > 0 && synCount > 0 ? "both inputs ready" : "input missing"}
        </Badge>
        <Info label="About the synthetic–real view">
          <p>The LR data the model actually receives: a real multipoint archive sample and a synthetic dirty record
            (detector artifacts and warped PSFs included), rendered by the same colour transfer.</p>
          <p>Lupton shows all four bands: NISP-only saturation blackouts are invisible in a single VIS channel.</p>
        </Info>
        <IconButton size="sm" icon="reset" label="Refresh both sources" loading={archive.fetching || sky.fetching}
          onClick={() => { archive.reload(); sky.reload(); }} />
      </RealismBar>

      <div className="rl-lanes">
        <section className="rl-lane" aria-label="Real Euclid LR">
          <header className="rl-lane__head">
            <div className="rl-lane__title">
              <span className="rl-subhead">real Euclid · multipoint archive</span>
              <strong>{archiveCount > 0 ? archiveSampleProvenance(object, archiveCount) : "no samples"}</strong>
              <small title={info?.source_plan_fingerprint ?? undefined}>
                {archiveOverview(info)}{info?.ready ? ` · ${archiveFieldBreakdown(info)} · plan ${shortArchiveFingerprint(info.source_plan_fingerprint)}` : ""}
              </small>
            </div>
            {object && (
              <div className="rl-row">
                <IconButton size="sm" icon="panelRight" label="Inspect this archive field" onClick={inspectCurrent} />
                <SkyLink layers={["q1-tiles:0.2", "archive-fields"]} ra={object.ra} dec={object.dec} fov={0.3}
                  hint={`${object.parent_id} · RA ${object.ra.toFixed(4)}, Dec ${object.dec.toFixed(4)}`}>Sky</SkyLink>
              </div>
            )}
          </header>
          <div className="rl-lane__viewer">
            {archive.loading && !archive.data ? <Skeleton height={320} />
              : archiveCount > 0 ? (
                <ImageViewer key={info?.collection_fingerprint ?? "archive-fields"} collection="archive-fields" id="visual-real"
                  tiers={["lr"]} urlKey="real" toolbar="compact" onReady={onReady("real")} onState={onState("real")} />
              ) : (
                <EmptyState icon="image" title="No multipoint archive samples"
                  action={<Button size="sm" icon="download" loading={sync.busy} title={syncHint} onClick={() => void syncArchiveFields()}>Sync from FASRC</Button>}>
                  {archive.error ? archive.error.message : info?.reasons?.[0] ?? "Synchronize the multipoint archive collection."}
                </EmptyState>
              )}
          </div>
        </section>
        <section className="rl-lane" aria-label="Synthetic LR">
          <header className="rl-lane__head">
            <div className="rl-lane__title">
              <span className="rl-subhead">synthetic · forward model</span>
              <strong>{synCount > 0 ? `${synCount.toLocaleString("en")} dirty ${subset} records` : "no records"}</strong>
              <small>detector artifacts and warped PSFs</small>
            </div>
          </header>
          <div className="rl-lane__viewer">
            {sky.loading && !sky.data ? <Skeleton height={320} />
              : synCount > 0 ? (
                <ImageViewer key={subset} collection="sky" params={{ subset }} id="visual-syn" tiers={["dirty"]}
                  urlKey="syn" toolbar="compact" onReady={onReady("syn")} onState={onState("syn")} />
              ) : (
                <EmptyState icon="image" title={`No ${subset} dirty records`}
                  action={<Button asChild size="sm" iconRight="chevronRight"><Link to="/data/records">Data › Records</Link></Button>}>
                  {sky.error ? sky.error.message : `Sync the ${subset} records first.`}
                </EmptyState>
              )}
          </div>
        </section>
      </div>

      <Card>
        <CardHead title="Multipoint archive reference" sub="generate the four-band samples on FASRC, then sync them here"
          right={stale ? <Badge tone="warn">source changed</Badge>
            : info?.ready ? <Badge tone={info.current ? "good" : "warn"}>{info.current ? `${info.parent_count} pointings ready` : "source changed"}</Badge>
              : <Badge tone="warn">not synchronized</Badge>} />
        <CardBody>
          <div className="rl-row">
            <Button variant="primary" icon="download" loading={sync.busy} title={syncHint} onClick={() => void syncArchiveFields()}>
              Sync archive fields from FASRC
            </Button>
            <SkyLink layers={["q1-tiles:0.2", "archive-fields"]} hint="Every archive field on the sky atlas">All fields on sky</SkyLink>
          </div>
          <JobProgress job={sync.job} error={sync.error} />
          <StepById stepId="archive_field_sample" embedded />
        </CardBody>
      </Card>
    </Page>
  );
}
