/* realism/visual (spec §8.3; image-first pass 2026-09-27): synthetic vs
   real, side by side and as large as fits — one real Euclid LR sample of the
   multipoint archive collection and one synthetic dirty LR record, indexed
   independently, on ONE dark instrument block:

     ┌ shared row: colour · knee · brightness (a Display menu when narrow) ┐
     │             · same transfer · fit both · reset · ⓘ (the viewer keys) │
     ├ real viewer (nav bar only) ─────┬ synthetic viewer (nav bar only) ──┤
     ├ caption: real sample · inspect  ┴ caption: synthetic subset · count ┘

   The shared row is the only colour control: both viewers show its transfer
   (setView), so the two lanes are always rendered alike (./visual/sync.ts;
   the transfer is the URL's `?c=&k=&g=`, default Lupton so NISP-only
   blackouts stay visible). Keys typed in a viewer (Q–Y colours, the
   horizontal-wheel brightness) are edits too: with "Same transfer" on
   (`?lock=`, default) they reach both viewers. Only the multipoint archive
   collection is used here (the legacy one-pointing real field is a Sky ›
   Real results tile). The lanes stay side by side down to a ~400 px block
   (two small frames compare better than one frame and a scroll). */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import { useFasrcStatus } from "../../../app/status";
import { StepById } from "../../../fasrc";
import { useUrlState } from "../../../hooks/useUrlState";
import { Badge, Button, Card, CardBody, CardHead, EmptyState, IconButton, JobProgress, Kbd, Page, Popover, Segmented, Skeleton, Slider, Switch, Tooltip } from "../../../ui";
import { ImageViewer, type ViewerApi, type ViewerState } from "../../../viewer";
import { ZOOM_STEP } from "../../../viewer/controller";
import { VIcon } from "../../../viewer/icons";
import { GAIN_SLIDER_RANGE, KNEE_SLIDER_RANGE, formatSig, parseNumber } from "../../../viewer/barModel";
import { useArchiveMeta, useSkyMeta } from "../api";
import { archiveFieldBreakdown, archiveOverview, archiveSampleProvenance, shortArchiveFingerprint } from "../archiveFields";
import { Info, SkyLink } from "../common";
import { JOB, offlinePolicy, runJob, useRealismJob } from "../jobs";
import { DEFAULT_GAIN, DEFAULT_KNEE, editOf, shows, urlGain, urlKnee, viewPatch, type Reported, type Transfer } from "../visual/sync";

type Subset = "test" | "validate" | "train";
type Lane = "real" | "syn";
const SUBSETS: { value: Subset; label: string }[] = [
  { value: "test", label: "Test" }, { value: "validate", label: "Validate" }, { value: "train", label: "Train" },
];
const COLOURS = [
  { value: "VIS", label: "VIS", title: "VIS (key Q)" }, { value: "Y_E", label: "Y", title: "NISP Y (key W)" },
  { value: "J_E", label: "J", title: "NISP J (key E)" }, { value: "H_E", label: "H", title: "NISP H (key R)" },
  { value: "lupton", label: "Lupton", title: "Four-band Lupton RGB (key T): shows NISP-only blackouts" },
  { value: "temp", label: "Temp", title: "Per-pixel blackbody colour (key Y)" },
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

/** A log knee slider with the exact value typed beside it (Enter / blur
 *  applies; anything but a positive number is ignored). */
function KneeField({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  const [draft, setDraft] = useState<string | null>(null);
  const commit = () => {
    const v = parseNumber(draft ?? "");
    setDraft(null);
    if (v != null && v > 0) onChange(v);
  };
  return (
    <span className="rl-vis__field">
      <span className="rl-vis__label">Knee</span>
      <Slider value={value} min={KNEE_SLIDER_RANGE[0]} max={KNEE_SLIDER_RANGE[1]} scale="log" className="rl-vis__slider"
        format={(v) => `${formatSig(v)} e⁻`} aria-label="Shared knee" onChange={onChange} />
      <input className="ui-input ui-input--sm rl-vis__num" type="text" inputMode="decimal" aria-label="Shared knee (e⁻)"
        value={draft ?? formatSig(value)} onChange={(e) => setDraft(e.target.value)} onBlur={commit}
        onKeyDown={(e) => { if (e.key === "Enter") commit(); if (e.key === "Escape") setDraft(null); }} />
      <span className="rl-vis__unit">e⁻</span>
    </span>
  );
}

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
  const [lensIn, setLensIn] = useState<Record<Lane, boolean>>({ real: false, syn: false });

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
  // A new shared transfer (the shared row, a reset, a palette colour, an edit
  // while locked) reaches both viewers; re-locking brings a viewer edited on
  // its own back.
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
    // the shared lens button reads the lanes' own tool (L in a lane, its menu)
    const lensOn = s.tool === "lens";
    setLensIn((prev) => (prev[lane] === lensOn ? prev : { ...prev, [lane]: lensOn }));
    // A viewer that now shows the shared transfer is echoing our setView.
    if (!edit || !lockRef.current || shows(next, sharedRef.current)) return;
    if (edit.color != null) setColor(edit.color);
    // The default knee / brightness is the unset URL value (a clean link).
    if (edit.knee != null) setKnee(urlKnee(edit.knee));
    if (edit.gain != null) setGain(urlGain(edit.gain));
  };
  const resetTransfer = () => { setColor("lupton"); setKnee(0); setGain(0); };
  const inspectCurrent = () => { if (object) openInspector({ kind: "archivefield", id: object.id ?? String(object.sample_id) }); };
  const isDefault = color === "lupton" && knee === 0 && gain === 0;
  const kneeValue = shared.knee ?? DEFAULT_KNEE;
  const gainValue = shared.gain ?? DEFAULT_GAIN;

  usePageActions([
    { id: "visual-lock", label: lock ? "Unlock the synthetic–real transfer" : "Lock the synthetic–real transfer", group: "Visual", run: () => setLock(!lock) },
    { id: "visual-lupton", label: "Synthetic–real: Lupton colour", group: "Visual", run: () => setColor("lupton") },
    { id: "visual-vis", label: "Synthetic–real: VIS", group: "Visual", run: () => setColor("VIS") },
    { id: "visual-reset", label: "Reset the shared transfer", group: "Visual", run: resetTransfer },
    { id: "visual-inspect", label: "Inspect the current archive field", group: "Visual", disabled: !object, run: inspectCurrent },
    { id: "visual-sync", label: "Sync archive fields from FASRC", group: "Visual", keywords: ["archive", "multipoint"],
      run: () => { void syncArchiveFields(); } },
  ]);

  const fitBoth = () => { apis.current.real?.resetView(); apis.current.syn?.resetView(); };
  // Zoom and the lens in both lanes (the viewers' own + / − / L keys act on the hovered one).
  const zoomBoth = (factor: number) => { apis.current.real?.zoomBy(factor); apis.current.syn?.zoomBy(factor); };
  // pressed while both lanes show the lens; a click turns it on in both, or off in both
  const lens = lensIn.real && lensIn.syn;
  const toggleLens = () => {
    const on = !lens;
    apis.current.real?.setTool(on ? "lens" : "none");
    apis.current.syn?.setTool(on ? "lens" : "none");
  };
  const kneeField = <KneeField value={kneeValue} onChange={(v) => setKnee(urlKnee(v))} />;
  const gainField = (
    <span className="rl-vis__field">
      <span className="rl-vis__label">Brightness</span>
      <Slider value={gainValue} min={GAIN_SLIDER_RANGE[0]} max={GAIN_SLIDER_RANGE[1]} scale="log" showValue className="rl-vis__slider rl-vis__slider--gain"
        format={(v) => `×${formatSig(v)}`} aria-label="Shared brightness" onChange={(v) => setGain(urlGain(v))} />
    </span>
  );

  return (
    <Page className="rl-page">
      <div className="rl-vis">
        <div className="rl-vis__bar" role="toolbar" aria-label="Shared display of both lanes">
          <Segmented size="sm" aria-label="Colour of both lanes" value={COLOURS.some((c) => c.value === color) ? color : "lupton"}
            onChange={setColor} options={COLOURS} />
          {/* Wide: knee and brightness in the row. Narrow: one "Display" menu
              holds them, so the row stays one or two lines. */}
          <span className="rl-vis__inline">{kneeField}{gainField}</span>
          <Popover label="Knee and brightness of both lanes" align="start" className="rl-vis rl-vis__pop" width={320}
            trigger={<Button size="sm" variant="ghost" icon="contrast" iconRight="chevronDown" className="rl-vis__more"
              title={`Knee ${formatSig(kneeValue)} e⁻, brightness ×${formatSig(gainValue)}`}>Display</Button>}>
            {kneeField}{gainField}
          </Popover>
          <Tooltip content={lock ? "Keys typed in one viewer (Q–Y, brightness) change both. Switch off to let them differ."
            : "Keys typed in a viewer change only that viewer; the row above still sets both."}>
            <span className="rl-vis__lock"><Switch size="sm" checked={lock} onChange={setLock}>Same transfer</Switch></span>
          </Tooltip>
          <span className="rl-vis__spacer" />
          <span className="rl-vis__end">
            <IconButton size="sm" icon={<VIcon name="zoomOut" />} label="Zoom out both lanes"
              tooltip={<span className="cv-tip">Zoom out both lanes; <Kbd keys="-" /> zooms the hovered one</span>} onClick={() => zoomBoth(1 / ZOOM_STEP)} />
            <IconButton size="sm" icon={<VIcon name="zoomIn" />} label="Zoom in both lanes"
              tooltip={<span className="cv-tip">Zoom in both lanes; <Kbd keys="+" /> zooms the hovered one</span>} onClick={() => zoomBoth(ZOOM_STEP)} />
            <IconButton size="sm" icon={<VIcon name="lens" />} label="Magnifier lens in both lanes" pressed={lens}
              tooltip={<span className="cv-tip">Magnifier lens in both lanes; <Kbd keys="L" /> in the hovered one</span>} onClick={toggleLens} />
            <Button size="sm" variant="ghost" onClick={fitBoth} title="Show the whole field in both lanes (0 in a viewer)">Fit both</Button>
            <IconButton size="sm" icon="reset" label="Reset the shared transfer (Lupton, default knee)" onClick={resetTransfer} disabled={isDefault} />
            <Info label="About the synthetic–real view">
              <p>The LR data the model actually receives: a real multipoint archive sample and a synthetic dirty record
                (detector artifacts and warped PSFs included), rendered by the same colour transfer.</p>
              <p>Lupton shows all four bands: NISP-only saturation blackouts are invisible in a single VIS channel.</p>
              <p className="rl-subhead">In either viewer</p>
              <ul className="rl-info__defs">
                <li><b>Wheel, + / −</b><span>zoom in and out; drag to pan</span></li>
                <li><b>0</b><span>fit the whole field (the fit button above does both lanes)</span></li>
                <li><b>L</b><span>the lens (magnifier); click to freeze a crop, S saves it</span></li>
                <li><b>Q W E R T Y</b><span>VIS, Y, J, H, Lupton, temperature</span></li>
                <li><b>← / →</b><span>previous / next sample in that lane</span></li>
                <li><b>F</b><span>that lane full screen (⛶); Esc returns</span></li>
              </ul>
            </Info>
          </span>
        </div>

        <div className="rl-vis__lanes">
          <section className="rl-vis__lane" aria-label="Real Euclid LR">
            <div className="rl-vis__viewer">
              {archive.loading && !archive.data ? <Skeleton height={320} />
                : archiveCount > 0 ? (
                  <ImageViewer key={info?.collection_fingerprint ?? "archive-fields"} collection="archive-fields" id="visual-real"
                    tiers={["lr"]} urlKey="real" toolbar="none" nav onReady={onReady("real")} onState={onState("real")} />
                ) : (
                  <EmptyState icon="image" title="No multipoint archive samples"
                    action={<Button size="sm" icon="download" loading={sync.busy} title={syncHint} onClick={() => void syncArchiveFields()}>Sync from FASRC</Button>}>
                    {archive.error ? archive.error.message : info?.reasons?.[0] ?? "Synchronize the multipoint archive collection."}
                  </EmptyState>
                )}
            </div>
            <footer className="rl-vis__caption">
              <span className="rl-vis__name">Real Euclid</span>
              <span className="rl-vis__desc" title={archiveCount > 0 ? archiveSampleProvenance(object, archiveCount) : undefined}>
                {archiveCount > 0 ? archiveSampleProvenance(object, archiveCount) : "no samples"}
              </span>
              {object && <>
                <IconButton size="sm" icon="panelRight" label="Inspect this archive field" onClick={inspectCurrent} />
                <SkyLink layers={["q1-tiles:0.2", "archive-fields"]} ra={object.ra} dec={object.dec} fov={0.3}
                  hint={`${object.parent_id} · RA ${object.ra.toFixed(4)}, Dec ${object.dec.toFixed(4)}`}>Sky</SkyLink>
              </>}
            </footer>
          </section>
          <section className="rl-vis__lane" aria-label="Synthetic LR">
            <div className="rl-vis__viewer">
              {sky.loading && !sky.data ? <Skeleton height={320} />
                : synCount > 0 ? (
                  <ImageViewer key={subset} collection="sky" params={{ subset }} id="visual-syn" tiers={["dirty"]}
                    urlKey="syn" toolbar="none" nav onReady={onReady("syn")} onState={onState("syn")} />
                ) : (
                  <EmptyState icon="image" title={`No ${subset} dirty records`}
                    action={<Button asChild size="sm" iconRight="chevronRight"><Link to="/data/records">Data › Records</Link></Button>}>
                    {sky.error ? sky.error.message : `Sync the ${subset} records first.`}
                  </EmptyState>
                )}
            </div>
            <footer className="rl-vis__caption">
              <span className="rl-vis__name">Synthetic</span>
              <span className="rl-vis__desc">{synCount > 0 ? `${synCount.toLocaleString("en")} dirty records` : "no records"}</span>
              <Segmented size="sm" aria-label="Synthetic subset" value={subset} onChange={setSubset} options={SUBSETS} />
            </footer>
          </section>
        </div>
      </div>

      <Card>
        <CardHead title="Multipoint archive reference"
          sub={info?.ready ? `${archiveOverview(info)} · ${archiveFieldBreakdown(info)} · plan ${shortArchiveFingerprint(info.source_plan_fingerprint)}`
            : "generate the four-band samples on FASRC, then sync them here"}
          right={stale ? <Badge tone="warn" dot>source changed</Badge>
            : info?.ready ? <Badge tone={info.current ? "good" : "warn"} dot>{info.current ? `${info.parent_count} pointings ready` : "source changed"}</Badge>
              : <Badge tone="warn" dot>not synchronized</Badge>} />
        <CardBody>
          <div className="rl-row">
            <Button variant="primary" icon="download" loading={sync.busy} title={syncHint} onClick={() => void syncArchiveFields()}>
              Sync archive fields from FASRC
            </Button>
            <SkyLink layers={["q1-tiles:0.2", "archive-fields"]} hint="Every archive field on the sky atlas">All fields on sky</SkyLink>
            <Button size="sm" variant="ghost" icon="reset" loading={archive.fetching || sky.fetching}
              onClick={() => { archive.reload(); sky.reload(); }}>Refresh both sources</Button>
            <span className="rl-faint">{archiveCount > 0 && synCount > 0 ? "Both inputs ready" : "An input is missing"}</span>
          </div>
          <JobProgress job={sync.job} error={sync.error} />
          <StepById stepId="archive_field_sample" embedded />
        </CardBody>
      </Card>
    </Page>
  );
}
