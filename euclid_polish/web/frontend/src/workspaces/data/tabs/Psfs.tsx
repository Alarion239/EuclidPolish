/* Data › PSFs (spec §8.4): the empirical ePSFs synchronised from the FASRC
 * extraction.
 *
 * Toolbar: each band's state — empirical / no empirical PSF (FASRC has none:
 * the Gaussian fallback) / not cached (never synchronised here) — and the two
 * syncs, both background jobs. Body: the ePSF viewer (one object per spatial
 * cluster, one tier per band; live elastic "training warps"), the per-band
 * table (FWHM, cluster count, kernel, file; inspect the FITS) and the cluster
 * table (RA/Dec, stars, per-band FWHM; row → the `psf` inspector and the
 * viewer; the atlas shows them on the sky). The extraction steps sit at the
 * bottom. The viewer object is in the URL. */
import { useCallback, useEffect, useRef, useState, type MutableRefObject } from "react";
import { useJob } from "../../../api/jobs";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import { StepById } from "../../../fasrc";
import { formatBytes, formatDeg, formatNumber } from "../../../format";
import { useMediaQuery } from "../../../hooks/useMediaQuery";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Badge, Button, Callout, Card, CardBody, CardHead, DataTable, EmptyState, Page, Section, Slider, Switch, Tooltip,
  type DataColumn,
} from "../../../ui";
import { ImageViewer, type ViewerApi, type ViewerState } from "../../../viewer";
import { URLS, usePsfInventory, type PsfBand, type PsfCluster } from "../api";
import { DataBar, Freshness, JobStrip, LoadState, OFFLINE_HINT, SkyButton, Spacer, startDataJob, useFasrcOnline } from "../common";
import { BANDS, PSF_STATE, bandShort, clusterObjectId } from "../model";
import "../register";
import "../data.css";

type ConfigResp = { config?: { psf_warp_alpha_max?: number; psf_warp_sigma?: number } };

function StateBadge({ band }: { band: PsfBand }) {
  const info = PSF_STATE[band.state];
  const extra = band.state === "not_cached" && band.error ? `\nLast sync: ${band.error}` : "";
  return (
    <Tooltip content={<span className="dt-pre">{info.hint + extra}</span>}>
      <span tabIndex={0} className="dt-state">
        <Badge size="sm" tone={info.tone} dot>{bandShort(band.name)} · {info.label}</Badge>
      </span>
    </Tooltip>
  );
}

const BAND_COLUMNS: DataColumn<PsfBand>[] = [
  { id: "name", header: "Band", width: 60, cell: (b) => <strong>{bandShort(b.name)}</strong> },
  { id: "state", header: "State", width: 150, accessor: (b) => PSF_STATE[b.state].label,
    cell: (b) => <Badge size="sm" tone={PSF_STATE[b.state].tone}>{PSF_STATE[b.state].label}</Badge> },
  { id: "fwhm", header: "Fallback FWHM ″", numeric: true, width: 110, cell: (b) => formatNumber(b.fwhm, { digits: 3 }) },
  { id: "measured_fwhm", header: "ePSF FWHM ″", numeric: true, width: 100, accessor: (b) => b.measured_fwhm ?? null,
    cell: (b) => formatNumber(b.measured_fwhm, { digits: 3 }) },
  { id: "n_psf", header: "Clusters", numeric: true, width: 74, accessor: (b) => b.n_psf ?? null },
  { id: "kernel", header: "Kernel", width: 130, accessor: (b) => (b.shape ? b.shape[0] * b.shape[1] : null),
    cell: (b) => (b.shape ? <span className="mono">{b.shape[1]}×{b.shape[0]} @ {formatNumber(b.pixel_scale, { digits: 4 })}″</span> : "—") },
  { id: "oversampling", header: "Oversampling", numeric: true, width: 96, hidden: true, cell: (b) => `${b.oversampling}×` },
  { id: "size_bytes", header: "File", numeric: true, width: 76, accessor: (b) => b.size_bytes ?? null, cell: (b) => formatBytes(b.size_bytes) },
  { id: "synced", header: "Synced", width: 110, accessor: (b) => b.synced_at ?? null, csv: (b) => String(b.synced_at ?? ""),
    cell: (b) => (b.synced_at ? <Freshness at={b.synced_at} label="" stale={30 * 24 * 3600} /> : "—") },
  { id: "inspect", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 70,
    cell: (b) => (b.path ? (
      <Button size="sm" variant="ghost" icon="fileSearch" onClick={() => openInspector({ kind: "fits", id: b.path! })}>FITS</Button>
    ) : null) },
];

function clusterColumns(onView: (c: PsfCluster) => void): DataColumn<PsfCluster>[] {
  return [
    { id: "index", header: "Cluster", numeric: true, width: 70 },
    { id: "ra", header: "RA", numeric: true, width: 96, cell: (c) => <span className="mono">{formatDeg(c.ra, 4)}</span> },
    { id: "dec", header: "Dec", numeric: true, width: 96, cell: (c) => <span className="mono">{formatDeg(c.dec, 4, { signed: true })}</span> },
    { id: "n_stars", header: "Stars", numeric: true, width: 64 },
    ...BANDS.map((b): DataColumn<PsfCluster> => ({
      id: `fwhm_${b}`, header: `FWHM ${bandShort(b)} ″`, numeric: true, width: 88,
      accessor: (c) => c.fwhm_by_band[b] ?? null, cell: (c) => formatNumber(c.fwhm_by_band[b], { digits: 3 }),
    })),
    { id: "view", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 120,
      cell: (c) => (
        <span className="dt-chips">
          <Button size="sm" variant="ghost" icon="image" onClick={() => onView(c)}>View</Button>
          <SkyButton ra={c.ra} dec={c.dec} fov={0.6} layers={["psf-clusters", "q1-tiles:0.3"]} label="Sky" />
        </span>
      ) },
  ];
}

/** Live elastic warps of the PSF preview (the training-time distribution). */
function WarpControls({ api, ready }: { api: MutableRefObject<ViewerApi | null>; ready: boolean }) {
  const config = useResource<ConfigResp>(URLS.config, [], { ttl: 5 * 60_000 });
  const reduceMotion = useMediaQuery("(prefers-reduced-motion: reduce)");
  const [on, setOn] = useState(false);
  const [interval, setIntervalSec] = useState(1.0);
  const [sample, setSample] = useState(0);
  const [loading, setLoading] = useState(false);
  const seed = useRef(Math.floor(Math.random() * 0xffff_ffff));
  const inFlight = useRef(false);
  const alphaMax = config.data?.config?.psf_warp_alpha_max ?? 20;
  const sigma = config.data?.config?.psf_warp_sigma ?? 3;

  const next = useCallback(async () => {
    const viewer = api.current;
    if (!viewer || inFlight.current) return;
    inFlight.current = true;
    setLoading(true);
    seed.current = (seed.current + 1) >>> 0;
    setSample((n) => n + 1);
    try {
      await viewer.setParams({ psf_warp: "1", psf_warp_seed: String(seed.current) });
    } finally {
      inFlight.current = false;
      setLoading(false);
    }
  }, [api]);

  useEffect(() => {
    if (!on || !ready) return;
    let stopped = false;
    let timer: number | undefined;
    const cycle = async () => {
      await next();
      if (!stopped && !reduceMotion) timer = window.setTimeout(cycle, interval * 1000);
    };
    void cycle();
    return () => { stopped = true; if (timer != null) window.clearTimeout(timer); };
  }, [on, ready, interval, reduceMotion, next]);

  const toggle = (value: boolean) => {
    setOn(value);
    if (!value) { setSample(0); void api.current?.setParams({ psf_warp: "0" }); }
  };
  return (
    <div className="dt-warp">
      <Tooltip content="Apply the training-time elastic PSF distribution to this preview (replayable seeds)">
        <span><Switch size="sm" checked={on} disabled={!ready} onChange={toggle}>Live warps</Switch></span>
      </Tooltip>
      <Slider value={interval} onChange={setIntervalSec} min={0.3} max={2} step={0.1} disabled={reduceMotion || !on}
        aria-label="Seconds between warps" showValue format={(v) => `${v.toFixed(1)} s`} />
      <Button size="sm" loading={loading} disabled={!ready || !on} onClick={() => void next()}>New warp</Button>
      <span className="dt-warp__status">
        {on ? `sample ${sample}${reduceMotion ? " · manual" : ""}` : `α∈[0, ${alphaMax}] · σ = ${sigma} px`}
      </span>
    </div>
  );
}

export default function Psfs() {
  const inv = usePsfInventory();
  const data = inv.data;
  const sync = useJob("data:psf-sync");
  const { online } = useFasrcOnline();
  const api = useRef<ViewerApi | null>(null);
  const [ready, setReady] = useState(false);
  const [viewerKey, setViewerKey] = useState(0);
  const [current, setCurrent] = useState<string | null>(null);
  const [steps, setSteps] = useUrlState("steps", false);
  const bump = () => setViewerKey((k) => k + 1);

  const run = (url: string, label: string, message: string) => void startDataJob(sync, url, {}, {
    label, question: { title: `${label}?`, message, confirmLabel: "Sync" }, onDone: bump,
  });
  const syncAll = () => run(URLS.psfSync, "Sync the ePSFs from FASRC",
    "Force-pulls the four band ePSFs (the VIS stack is tens to hundreds of MB) and the cluster metadata.");
  const syncMeta = () => run(URLS.psfSyncMeta, "Sync the cluster metadata",
    "Dumps the cluster centroids, star counts and FWHMs from the ePSF headers on FASRC (seconds) and pulls the small JSON.");
  const view = (c: PsfCluster) => {
    void api.current?.goToId(clusterObjectId(c.index));
    openInspector({ kind: "psf", id: String(c.index) });
  };
  usePageActions([
    { id: "psfs-sync", label: "Sync the ePSFs from FASRC…", group: "PSFs", disabled: !online || sync.busy, run: syncAll },
    { id: "psfs-sync-meta", label: "Sync the PSF cluster metadata…", group: "PSFs", disabled: !online || sync.busy, run: syncMeta },
    { id: "psfs-steps", label: "Extract Euclid PSFs (FASRC)…", group: "PSFs", run: () => setSteps(true) },
  ]);

  const anyCached = data?.bands.some((b) => b.state === "empirical");
  return (
    <Page className="dt-page">
      <DataBar label="PSFs">
        {data ? data.bands.map((b) => <StateBadge key={b.name} band={b} />) : <Badge size="sm">…</Badge>}
        {data?.last_sync != null && <Freshness at={data.last_sync} label="last sync" stale={30 * 24 * 3600} />}
        <Spacer />
        <Tooltip content={online ? "Force-pull the four band ePSFs + cluster metadata (a job)" : OFFLINE_HINT}>
          <span><Button size="sm" icon="download" loading={sync.busy} disabled={!online} onClick={syncAll}>Sync ePSFs</Button></span>
        </Tooltip>
        <Tooltip content={online ? "Cluster centroids, star counts and FWHMs only (kilobytes)" : OFFLINE_HINT}>
          <span><Button size="sm" variant="ghost" disabled={!online || sync.busy} onClick={syncMeta}>Metadata only</Button></span>
        </Tooltip>
        <SkyButton layers={["psf-clusters", "q1-tiles:0.3"]} label="Clusters on sky" />
      </DataBar>
      <JobStrip job={sync} />
      <LoadState loading={inv.loading && !data} error={inv.error} onRetry={() => void inv.reload()}>
        <div className="dt-split">
          <Card className="dt-viewer-card">
            <CardHead title="ePSF viewer" sub={current ?? undefined} right={<WarpControls api={api} ready={ready} />} />
            <CardBody>
              {anyCached ? (
                <ImageViewer key={viewerKey} collection="psfs" urlKey="psf"
                  onReady={(a) => { api.current = a; setReady(a != null); }}
                  onState={(s: ViewerState) => setCurrent(s.id)} />
              ) : (
                <EmptyState icon="image" title="No ePSF cached on this machine"
                  action={<Button variant="primary" icon="download" disabled={!online} onClick={syncAll}>Sync ePSFs</Button>}>
                  {online ? "Pull the FASRC extraction (a background job)." : OFFLINE_HINT}
                </EmptyState>
              )}
            </CardBody>
          </Card>
          <Card>
            <CardHead title="Bands" sub="state, FWHM, clusters, kernel" />
            <CardBody>
              <DataTable rows={data?.bands ?? []} columns={BAND_COLUMNS} rowKey={(b) => b.name} height="auto" dense
                hideToolbar aria-label="ePSF bands" />
              {data?.bands.some((b) => b.state === "no_empirical") && (
                <Callout tone="warn" title="Gaussian fallback in use">
                  {data.bands.filter((b) => b.state === "no_empirical").map((b) => bandShort(b.name)).join(", ")}: FASRC has no
                  empirical ePSF — generation uses the configured Gaussian FWHM.
                </Callout>
              )}
            </CardBody>
          </Card>
        </div>
        <Card>
          <CardHead title="Spatial clusters"
            sub={data?.clusters_source === "metadata" ? "from the synced cluster metadata"
              : data?.clusters_source === "vis_headers" ? "from the cached VIS headers" : "no cluster data yet"}
            right={data?.clusters_meta.synced_at ? <Freshness at={data.clusters_meta.synced_at} label="metadata" /> : undefined} />
          <CardBody>
            <DataTable rows={data?.clusters ?? []} columns={clusterColumns(view)} rowKey={(c) => c.id} dense height={360}
              aria-label="PSF clusters" exportName="psf-clusters" urlKey="pc"
              inspect={(c) => ({ kind: "psf", id: String(c.index) })}
              onRowClick={(c) => { void api.current?.goToId(clusterObjectId(c.index)); }}
              empty={online ? "Sync the cluster metadata to list the clusters." : "No cluster data cached."} />
          </CardBody>
        </Card>
      </LoadState>
      <Section title="Extraction steps (FASRC)" collapsible open={steps} onOpenChange={setSteps}>
        <div className="dt-steps">
          <StepById stepId="extract_euclid_psf" />
          <StepById stepId="psf_rotation_pool" />
        </div>
      </Section>
    </Page>
  );
}
