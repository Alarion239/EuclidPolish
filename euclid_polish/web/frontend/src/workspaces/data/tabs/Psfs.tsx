/* Data › PSFs (spec §8.4): the empirical ePSFs synchronised from the FASRC
 * extraction, image first (docs/superpowers/specs/2026-09-27-image-first-viewer-design.md).
 *
 * One toolbar row: each band's state — empirical / fallback (FASRC has no
 * ePSF: the Gaussian fallback) / not cached (never synchronised here) — the
 * last sync, and the two syncs (background jobs) and the atlas link. Then a
 * caption row — the cluster in the viewer and the live elastic "training
 * warps" of the preview — and the ePSF viewer (one object per spatial
 * cluster, one tier per band) sized so its first frame row is in view, with
 * the spatial clusters beside it when there is room (else below): RA/Dec,
 * stars, per-band FWHM; a row → that cluster in the viewer (and the `psf`
 * inspector from its View button; the atlas shows them on the sky). Below:
 * the per-band table (FWHM, cluster count, kernel, file; inspect the FITS) and
 * the extraction steps (collapsed). The viewer object is in the URL. */
import { useCallback, useEffect, useMemo, useRef, useState, type MutableRefObject } from "react";
import { useJob } from "../../../api/jobs";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import { StepById } from "../../../fasrc";
import { formatBytes, formatDeg, formatNumber } from "../../../format";
import { useMediaQuery } from "../../../hooks/useMediaQuery";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Badge, Button, Callout, DataTable, EmptyState, Page, Section, Slider, Switch, Tooltip, type DataColumn,
} from "../../../ui";
import { ImageViewer, type ViewerApi, type ViewerState } from "../../../viewer";
import type { LayoutMode } from "../../../viewer/fit";
import { URLS, usePsfInventory, type PsfBand, type PsfCluster } from "../api";
import { BarActions, DataBar, Freshness, JobStrip, LoadState, OFFLINE_HINT, SkyButton, Spacer, startDataJob, useFasrcOnline } from "../common";
import { BANDS, PSF_STATE, bandShort, clusterObjectId, psfBandGroups } from "../model";
import { ViewerStage, type AsidePlace } from "../ViewerStage";
import "../register";
import "../data.css";

type ConfigResp = { config?: { psf_warp_alpha_max?: number; psf_warp_sigma?: number } };

/** The cluster list beside the viewer keeps these columns (the rest stay in
 *  its column menu); a row shows that cluster (and opens its inspector card),
 *  so the View / Sky buttons are left to the full-width table. */
const NARROW_CLUSTER_COLUMNS = new Set(["index", "n_stars", "fwhm_VIS"]);

/** The bands that share a state, as one badge ("J, H: not cached"; "All
 *  bands: …"): at most three short badges, so the labelled actions fit. */
function StateBadges({ bands }: { bands: PsfBand[] }) {
  return (
    <>
      {psfBandGroups(bands).map((g) => {
        const info = PSF_STATE[g.state];
        const errors = bands.filter((b) => g.bands.includes(b.name) && b.state === "not_cached" && b.error)
          .map((b) => `\nLast ${bandShort(b.name)} sync: ${b.error}`).join("");
        return (
          <Tooltip key={g.state} content={<span className="dt-pre">{info.hint + errors}</span>}>
            <span tabIndex={0} className="dt-tipbadge">
              <Badge size="sm" tone={info.tone} dot>{g.label}</Badge>
            </span>
          </Tooltip>
        );
      })}
    </>
  );
}

/* Headers are uppercase mono in the kit's table (~7.5 px a character + the
   sort icon): short ones, sized to fit; `headerText` is the full name (sort
   tip, CSV, column menu). In a narrow pane the kernel / file columns start
   hidden (the column menu has them) so the table fits instead of scrolling. */
function bandColumns(narrow: boolean): DataColumn<PsfBand>[] {
  return [
  { id: "name", header: "Band", width: 76, cell: (b) => <strong>{bandShort(b.name)}</strong> },
  { id: "state", header: "State", width: 140, accessor: (b) => PSF_STATE[b.state].label,
    cell: (b) => <Badge size="sm" tone={PSF_STATE[b.state].tone}>{PSF_STATE[b.state].label}</Badge> },
  { id: "measured_fwhm", header: "ePSF ″", headerText: "ePSF FWHM ″", numeric: true, width: 88, accessor: (b) => b.measured_fwhm ?? null,
    cell: (b) => formatNumber(b.measured_fwhm, { digits: 3 }) },
  { id: "fwhm", header: "Gauss ″", headerText: "Fallback (Gaussian) FWHM ″", numeric: true, width: 96, cell: (b) => formatNumber(b.fwhm, { digits: 3 }) },
  { id: "n_psf", header: "Clusters", numeric: true, width: 104, accessor: (b) => b.n_psf ?? null },
  { id: "kernel", header: "Kernel", width: 136, hidden: narrow, accessor: (b) => (b.shape ? b.shape[0] * b.shape[1] : null),
    cell: (b) => (b.shape ? <span className="mono">{b.shape[1]}×{b.shape[0]} at {formatNumber(b.pixel_scale, { digits: 4 })}″</span> : "—") },
  { id: "oversampling", header: "Oversampling", numeric: true, width: 128, hidden: true, cell: (b) => `${b.oversampling}×` },
  { id: "size_bytes", header: "File", numeric: true, width: 80, hidden: narrow, accessor: (b) => b.size_bytes ?? null, cell: (b) => formatBytes(b.size_bytes) },
  { id: "synced", header: "Synced", width: 104, accessor: (b) => b.synced_at ?? null, csv: (b) => String(b.synced_at ?? ""),
    cell: (b) => (b.synced_at ? <Freshness at={b.synced_at} label="" stale={30 * 24 * 3600} /> : "—") },
  { id: "inspect", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 70,
    cell: (b) => (b.path ? (
      <Button size="sm" variant="ghost" icon="fileSearch" onClick={() => openInspector({ kind: "fits", id: b.path! })}>FITS</Button>
    ) : null) },
  ];
}

function clusterColumns(onView: (c: PsfCluster) => void, narrow: boolean): DataColumn<PsfCluster>[] {
  const all: DataColumn<PsfCluster>[] = [
    { id: "index", header: "#", headerText: "Cluster", numeric: true, width: 52 },
    { id: "ra", header: "RA", numeric: true, width: 96, cell: (c) => <span className="mono">{formatDeg(c.ra, 4)}</span> },
    { id: "dec", header: "Dec", numeric: true, width: 96, cell: (c) => <span className="mono">{formatDeg(c.dec, 4, { signed: true })}</span> },
    { id: "n_stars", header: "Stars", numeric: true, width: 76 },
    // the FWHM per band (arcsec): the band is the header, the section says FWHM
    ...BANDS.map((b): DataColumn<PsfCluster> => ({
      id: `fwhm_${b}`, header: `${bandShort(b)} ″`, headerText: `FWHM ${bandShort(b)} ″`, numeric: true, width: bandShort(b).length > 1 ? 80 : 68,
      accessor: (c) => c.fwhm_by_band[b] ?? null, cell: (c) => formatNumber(c.fwhm_by_band[b], { digits: 3 }),
    })),
    { id: "view", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 120,
      cell: (c) => (
        <span className="dt-chips">
          <Button size="sm" variant="ghost" icon="image" onClick={(e) => { e.stopPropagation(); onView(c); }}>View</Button>
          <SkyButton ra={c.ra} dec={c.dec} fov={0.6} layers={["psf-clusters", "q1-tiles:0.3"]} label="Sky" />
        </span>
      ) },
  ];
  if (!narrow) return all;
  // Short headers in the narrow list; headerText keeps the full name (sort tip, CSV, column menu).
  const short: Record<string, Partial<DataColumn<PsfCluster>>> = { index: { width: 44 } };
  return all.filter((col) => col.id !== "view")
    .map((col) => (NARROW_CLUSTER_COLUMNS.has(col.id) ? { ...col, ...short[col.id] } : { ...col, hidden: true }));
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
      {on && (
        <>
          <Slider value={interval} onChange={setIntervalSec} min={0.3} max={2} step={0.1} disabled={reduceMotion}
            aria-label="Seconds between warps" showValue format={(v) => `${v.toFixed(1)} s`} />
          <Button size="sm" loading={loading} disabled={!ready} onClick={() => void next()}>New warp</Button>
        </>
      )}
      <span className="dt-warp__status">
        {on ? `Sample ${sample}${reduceMotion ? " (manual: reduced motion)" : ""}` : `α up to ${alphaMax}, σ ${sigma} px`}
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
  const [layout, setLayout] = useState<LayoutMode>("auto");
  const [tierCount, setTierCount] = useState(1);
  const [steps, setSteps] = useUrlState("steps", false);
  const bump = () => setViewerKey((k) => k + 1);
  // a pane too narrow for every band column (the user's is ~720–800 px)
  const narrowPane = useMediaQuery("(max-width: 1000px)");
  const bandCols = useMemo(() => bandColumns(narrowPane), [narrowPane]);

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
  const clusterSub = data?.clusters_source === "metadata" ? "FWHM in arcsec, from the synced cluster metadata"
    : data?.clusters_source === "vis_headers" ? "FWHM in arcsec, from the cached VIS headers" : "No cluster data yet";
  const clusterTable = (place: AsidePlace | null) => {
    const narrow = !!place?.beside;
    return (
      <div className="dt-clusters">
        <div className="dt-gallery__head">
          <span className="dt-gallery__title">Spatial clusters</span>
          <span className="dt-gallery__count">{clusterSub}</span>
          {data?.clusters_meta.synced_at ? <Freshness at={data.clusters_meta.synced_at} label="metadata" /> : null}
        </div>
        <ClusterTable clusters={data?.clusters ?? []} narrow={narrow}
          height={place?.beside ? "none" : 360} online={online}
          onView={view} onPick={(c) => { void api.current?.goToId(clusterObjectId(c.index)); }} />
      </div>
    );
  };
  const onViewerState = (s: ViewerState) => {
    setCurrent(s.id);
    setLayout(s.layout);
    setTierCount(s.tiers?.length || 1);
  };
  return (
    <Page className="dt-page dt-page--image">
      <DataBar label="PSFs" compactable>
        <div className="dt-bar__status" role="group" aria-label="ePSF state per band">
          {data ? <StateBadges bands={data.bands} /> : <Badge size="sm">…</Badge>}
          {data?.last_sync != null && <Freshness at={data.last_sync} label="last sync" stale={30 * 24 * 3600} />}
        </div>
        <Spacer />
        <BarActions>
          <Tooltip content={online ? "Force-pull the four band ePSFs + cluster metadata (a job)" : OFFLINE_HINT}>
            <span><Button size="sm" icon="download" loading={sync.busy} disabled={!online} onClick={syncAll} aria-label="Sync ePSFs">Sync ePSFs</Button></span>
          </Tooltip>
          <Tooltip content={online ? "Cluster centroids, star counts and FWHMs only (kilobytes)" : OFFLINE_HINT}>
            <span><Button size="sm" variant="ghost" icon="database" disabled={!online || sync.busy} onClick={syncMeta} aria-label="Metadata only">Metadata only</Button></span>
          </Tooltip>
          <SkyButton layers={["psf-clusters", "q1-tiles:0.3"]} label="Clusters on sky" />
        </BarActions>
      </DataBar>
      <LoadState loading={inv.loading && !data} error={inv.error} onRetry={() => void inv.reload()}>
        {anyCached ? (
          <div className="dt-figure">
            <div className="dt-caption" role="group" aria-label="ePSF preview">
              <span className="dt-caption__title">{current ?? "ePSF"}</span>
              <WarpControls api={api} ready={ready} />
            </div>
            <ViewerStage layout={layout} frames={tierCount} asideLabel="Spatial clusters" asideMin={210} asideFill
              aside={(place) => clusterTable(place)}>
              <ImageViewer key={viewerKey} collection="psfs" urlKey="psf"
                onReady={(a) => { api.current = a; setReady(a != null); }}
                onState={onViewerState} />
            </ViewerStage>
          </div>
        ) : (
          <>
            <EmptyState icon="image" title="No ePSF cached on this machine"
              action={<Button variant="primary" icon="download" disabled={!online} onClick={syncAll}>Sync ePSFs</Button>}>
              {online ? "Pull the FASRC extraction (a background job)." : OFFLINE_HINT}
            </EmptyState>
            {clusterTable(null)}
          </>
        )}
        <JobStrip job={sync} />
        <Section title="Bands" sub="state, FWHM, clusters, kernel" collapsible defaultOpen>
          <DataTable rows={data?.bands ?? []} columns={bandCols} rowKey={(b) => b.name} height="auto" dense
            hideToolbar={!narrowPane} aria-label="ePSF bands" />
          {data?.bands.some((b) => b.state === "no_empirical") && (
            <Callout tone="warn" title="Gaussian fallback in use">
              {data.bands.filter((b) => b.state === "no_empirical").map((b) => bandShort(b.name)).join(", ")}: FASRC has no
              empirical ePSF — generation uses the configured Gaussian FWHM.
            </Callout>
          )}
        </Section>
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

function ClusterTable({ clusters, narrow, height, online, onView, onPick }: {
  clusters: PsfCluster[]; narrow: boolean; online: boolean;
  /** The scroll box's max height; "none" beside the viewer (it fills the column). */
  height: number | string;
  onView: (c: PsfCluster) => void; onPick: (c: PsfCluster) => void;
}) {
  // A cluster list beside a narrow viewer keeps the key columns (the column menu has the rest).
  const columns = useMemo(() => clusterColumns(onView, narrow), [onView, narrow]);
  return (
    <DataTable rows={clusters} columns={columns} rowKey={(c) => c.id} dense height={height}
      aria-label="PSF clusters" exportName="psf-clusters" urlKey="pc"
      inspect={(c) => ({ kind: "psf", id: String(c.index) })}
      onRowClick={onPick}
      empty={online ? "Sync the cluster metadata to list the clusters." : "No cluster data cached."} />
  );
}
