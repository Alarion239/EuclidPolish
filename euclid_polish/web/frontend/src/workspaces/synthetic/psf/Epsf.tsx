/* Synthetic › PSF › ePSF: the empirical PSFs synchronised from the FASRC
   extraction, image first.

   The headline line: which kernels synthetic generation convolves with
   (empirical in every band, a Gaussian fallback where FASRC has none, or
   unknown here while nothing is synced). Then the caption row — the cluster
   in the viewer with its FWHM per band, and the live elastic training warps
   (the warp's α and σ in the switch's tooltip) — and the ePSF viewer (one
   object per spatial cluster, one tier per band). Below: the ePSF vs
   Gaussian-fallback FWHM per band (a small comparison table; the kernel,
   file and sync time collapsed), and the cluster FWHM map (RA/Dec coloured
   by one band's FWHM; a click shows that cluster and opens its card; the
   table stays one click away). The syncs and extraction steps are in the
   How-this-is-produced drawer. The viewer object is in the URL. */
import { useCallback, useEffect, useMemo, useRef, useState, type MutableRefObject } from "react";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import Plot from "../../../charts/Plot";
import { C, viridis } from "../../../colors";
import { formatBytes, formatDeg, formatNumber, formatRelative } from "../../../format";
import { useMediaQuery } from "../../../hooks/useMediaQuery";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Button, Caption, DataTable, Details, EmptyState, Section, Segmented, Slider, Switch, Table, Tooltip,
  type Column, type DataColumn,
} from "../../../ui";
import { ImageViewer, type ViewerApi, type ViewerState } from "../../../viewer";
import { URLS, type PsfBand, type PsfCluster, type PsfInventory } from "../dataApi";
import { SkyButton } from "../dataCommon";
import { BANDS, axisDomain, bandShort, clusterObjectId, nearestPoint, parseClusterId } from "../dataModel";
import { clusterMapGroups, generationPsfLine, psfFwhmRows, type FwhmRow } from "./psfModel";

type ConfigResp = { config?: { psf_warp_alpha_max?: number; psf_warp_sigma?: number } };

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
      <Tooltip content={`The training-time elastic PSF warps on this preview: displacement α up to ${alphaMax}, smoothed over σ ${sigma} px (replayable seeds)`}>
        <span><Switch size="sm" checked={on} disabled={!ready} onChange={toggle}>Live warps</Switch></span>
      </Tooltip>
      {on && (
        <>
          <Slider value={interval} onChange={setIntervalSec} min={0.3} max={2} step={0.1} disabled={reduceMotion}
            aria-label="Seconds between warps" showValue format={(v) => `${v.toFixed(1)} s`} />
          <Button size="sm" loading={loading} disabled={!ready} onClick={() => void next()}>New warp</Button>
          <span className="dt-warp__status">{`Sample ${sample}${reduceMotion ? " (manual: reduced motion)" : ""}`}</span>
        </>
      )}
    </div>
  );
}

const FWHM_COLUMNS: Column<FwhmRow>[] = [
  { header: "Band", cell: (r) => <strong>{r.band}</strong> },
  { header: "ePSF FWHM (″)", align: "right", cell: (r) => r.epsf },
  { header: "Gaussian fallback FWHM (″)", align: "right", cell: (r) => r.gaussian },
  { header: "Used", cell: (r) => r.state },
];

const CLUSTER_COLUMNS: DataColumn<PsfCluster>[] = [
  { id: "index", header: "#", headerText: "Cluster", numeric: true, width: 52 },
  { id: "ra", header: "RA", numeric: true, width: 96, cell: (c) => <span className="mono">{formatDeg(c.ra, 4)}</span> },
  { id: "dec", header: "Dec", numeric: true, width: 96, cell: (c) => <span className="mono">{formatDeg(c.dec, 4, { signed: true })}</span> },
  { id: "n_stars", header: "Stars", numeric: true, width: 76 },
  ...BANDS.map((b): DataColumn<PsfCluster> => ({
    id: `fwhm_${b}`, header: `${bandShort(b)} (″)`, headerText: `${bandShort(b)} FWHM (arcsec)`, numeric: true, width: 80,
    accessor: (c) => c.fwhm_by_band[b] ?? null, cell: (c) => formatNumber(c.fwhm_by_band[b], { digits: 3 }),
  })),
];

function ClusterMap({ clusters, onPick }: { clusters: PsfCluster[]; onPick: (c: PsfCluster) => void }) {
  const [band, setBand] = useUrlState("fwb", "VIS");
  const b = (BANDS as readonly string[]).includes(band) ? band : "VIS";
  const { groups, domain } = useMemo(() => clusterMapGroups(clusters, b), [clusters, b]);
  if (!groups.length) return <EmptyState compact icon="globe" title="No cluster positions: sync the cluster metadata (How this is produced)" />;
  const xs = groups.flatMap((g) => g.x), ys = groups.flatMap((g) => g.y);
  const xDom = axisDomain(xs, false), yDom = axisDomain(ys, false);
  const series = groups.map((g) => ({
    x: g.x, y: g.y, mode: "scatter" as const, color: g.t < 0 ? C.muted : viridis(g.t), name: g.label, key: g.key, width: 1.4,
  }));
  const byIndex = new Map(clusters.map((c) => [c.index, c]));
  return (
    <figure className="rl-fig">
      <figcaption className="rl-fig__head">
        <strong>Cluster FWHM map</strong>
        <Segmented size="sm" aria-label="FWHM band" value={b} onChange={setBand} options={BANDS.map((x) => ({ value: x, label: bandShort(x) }))} />
      </figcaption>
      <Plot xDomain={xDom} yDomain={yDom} xLabel="RA (deg)" yLabel="Dec (deg)" series={series}
        aspect={0.5} legend="auto" exportName={`psf-cluster-fwhm-${bandShort(b)}`}
        aria-label={`PSF clusters on the sky coloured by their ${bandShort(b)} FWHM`}
        onPlotClick={(pt) => {
          const id = nearestPoint(groups, pt, { xlog: false, ylog: false, xSpan: xDom[1] - xDom[0], ySpan: yDom[1] - yDom[0] });
          const c = id != null ? byIndex.get(Number(id)) : undefined;
          if (c) onPick(c);
        }} />
      <Caption>
        {`${clusters.length} spatial clusters, each one ePSF · colour: ${bandShort(b)} FWHM quintiles`
          + (domain ? ` (${domain[0].toFixed(3)}–${domain[1].toFixed(3)}″)` : "") + " · click a cluster to show it"}
      </Caption>
    </figure>
  );
}

export function Epsf({ data }: { data: PsfInventory }) {
  const api = useRef<ViewerApi | null>(null);
  const [ready, setReady] = useState(false);
  const [current, setCurrent] = useState<string | null>(null);
  const line = generationPsfLine(data.bands, data.generation);
  const anyCached = data.bands.some((b) => b.state === "empirical");
  const clusterIndex = current ? parseClusterId(current) : null;
  const cluster = clusterIndex != null ? data.clusters.find((c) => c.index === clusterIndex) : undefined;
  const show = (c: PsfCluster) => {
    void api.current?.goToId(clusterObjectId(c.index));
    openInspector({ kind: "psf", id: String(c.index) });
  };
  const clusterFwhm = cluster
    ? BANDS.map((b) => (cluster.fwhm_by_band[b] != null ? `${bandShort(b)} ${formatNumber(cluster.fwhm_by_band[b], { digits: 3 })}″` : null))
      .filter(Boolean).join(" · ")
    : null;
  return (
    <div className="rl-stack">
      {/* The PSF tab's header is the page's one summary line; this is the view's lead sentence. */}
      <p className="syn-sentence" title={data.generation ? undefined
        : "The local records do not record their PSFs (generated before the stamp, or not synced): this is what the synced ePSFs say"}>
        {line.lead}: <span className="syn-tone" data-tone={line.tone === "neutral" ? undefined : line.tone}>{line.text}</span>
      </p>
      {anyCached ? (
        <div className="dt-figure">
          <div className="dt-caption" role="group" aria-label="ePSF preview">
            <span className="dt-caption__title">{cluster ? `Cluster ${cluster.index}` : current ?? "ePSF"}</span>
            {clusterFwhm && <span className="dt-caption__note mono">{`FWHM ${clusterFwhm}`}</span>}
            <WarpControls api={api} ready={ready} />
          </div>
          <ImageViewer collection="psfs" urlKey="psf"
            onReady={(a) => { api.current = a; setReady(a != null); }}
            onState={(s: ViewerState) => setCurrent(s.id)} />
        </div>
      ) : (
        <EmptyState icon="image" title="No ePSF cached on this machine">
          Sync the ePSFs from FASRC (How this is produced) to look at the kernels.
        </EmptyState>
      )}
      <section className="rl-stack" aria-label="ePSF against the Gaussian fallback">
        <h3 className="syn-subtitle">ePSF against the Gaussian fallback</h3>
        <Table columns={FWHM_COLUMNS} rows={psfFwhmRows(data.bands)} rowKey={(r) => r.band} aria-label="ePSF and Gaussian FWHM per band" />
        {data.bands.some((b) => b.path) && (
          <Details summary="Kernels and files" className="rl-details">
            <ul className="syn-cmds">
              {data.bands.filter((b) => b.path).map((b: PsfBand) => (
                <li key={b.name} className="rl-row">
                  <span className="rl-mono">
                    {`${bandShort(b.name)} · ${b.shape ? `${b.shape[1]}×${b.shape[0]} at ${formatNumber(b.pixel_scale, { digits: 4 })}″` : "—"}`
                      + ` · ${b.n_psf ?? "?"} clusters · ${formatBytes(b.size_bytes)}${b.synced_at ? ` · synced ${formatRelative(b.synced_at)}` : ""} · ${b.path}`}
                  </span>
                  <Button size="sm" variant="ghost" icon="fileSearch" aria-label={`Open the ${bandShort(b.name)} ePSF file in the FITS inspector`}
                    onClick={() => openInspector({ kind: "fits", id: b.path! })}>FITS</Button>
                </li>
              ))}
            </ul>
          </Details>
        )}
      </section>
      <ClusterMap clusters={data.clusters} onPick={show} />
      {data.clusters.length > 0 && (
        <Section title="Clusters as a table" collapsible defaultOpen={false}
          right={<SkyButton layers={["psf-clusters", "q1-tiles:0.3"]} label="Clusters on sky" />}>
          <DataTable rows={data.clusters} columns={CLUSTER_COLUMNS} rowKey={(c) => c.id} dense height={360}
            aria-label="PSF clusters" exportName="psf-clusters" urlKey="pc"
            inspect={(c) => ({ kind: "psf", id: String(c.index) })}
            onRowClick={(c) => { void api.current?.goToId(clusterObjectId(c.index)); }} />
        </Section>
      )}
    </div>
  );
}
