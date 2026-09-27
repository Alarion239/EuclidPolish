/* An image HDU (or a 4-band HDU group) of any dimensionality in the viewer
   (collection `fits`), image first: one control row (the page's HDU / view
   picker, then colour/planes stacking, display bin, log render, HDU
   comparison; the plane picker for a cube), the viewer at the full size the
   stage allows, then the plane's statistics + histogram, its sky footprint
   and a PNG thumbnail. */
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useResource } from "../../api/query";
import { formatCount, formatNumber, formatRaDec } from "../../format";
import { useUrlState } from "../../hooks/useUrlState";
import {
  Badge, Button, Callout, Card, CardBody, CardHead, CopyButton, DefList, IconButton, MultiSelect,
  Segmented, Select, Skeleton, Slider,
} from "../../ui";
import { ImageViewer, type ViewerApi, type ViewerState } from "../../viewer";
import { imageStatsUrl, previewUrl, type ImageStats, type InspectResponse, type WcsSummary } from "./api";
import { FitBox } from "../sky/results/FitBox";
import { HistogramPlot } from "./charts";
import { axisName, basename, brightExposure, compareTiers, flatIndex, planeIndex, shapeText, viewerParams, type Selected } from "./model";
import { SkyLink } from "./SkyLink";

/** The viewer shows at most this many tier chips (the rest are hidden, see
 *  viewer_data `_FITS_SHOWN_TIERS`); beyond it a picker reaches every HDU. */
const SHOWN_TIERS = 12;

const BIN_OPTIONS = [
  { value: "auto", label: "Auto" }, { value: "1", label: "1×" }, { value: "2", label: "2×" },
  { value: "4", label: "4×" }, { value: "8", label: "8×" }, { value: "16", label: "16×" },
];

function PlanePicker({ axes, ndim, plane, bands, onPlane }: {
  axes: number[]; ndim: number; plane: number; bands: string[] | null; onPlane: (k: number) => void;
}) {
  const total = axes.reduce((a, b) => a * b, 1);
  const idx = planeIndex(axes, plane);
  return (
    <div className="insp-planes" role="group" aria-label="Plane">
      <IconButton size="sm" icon="chevronLeft" label="Previous plane (←)" disabled={plane <= 0}
        onClick={() => onPlane(plane - 1)} />
      {axes.map((n, a) => (
        <label key={a} className="insp-planes__axis">
          <span className="insp-planes__name">{axisName(ndim, a)}</span>
          <Slider min={0} max={Math.max(0, n - 1)} step={1} value={idx[a]} aria-label={`${axisName(ndim, a)} index`}
            onChange={(v) => { const next = [...idx]; next[a] = v; onPlane(flatIndex(axes, next)); }} />
          <span className="insp-planes__value mono">
            {axes.length === 1 && bands?.[idx[a]] ? bands[idx[a]] : `${idx[a]}/${n - 1}`}
          </span>
        </label>
      ))}
      <IconButton size="sm" icon="chevronRight" label="Next plane (→)" disabled={plane >= total - 1}
        onClick={() => onPlane(plane + 1)} />
      <span className="insp-dim insp-planes__count">{formatCount(plane + 1)} / {formatCount(total)}</span>
    </div>
  );
}

function WcsCard({ wcs }: { wcs: WcsSummary | null | undefined }) {
  if (!wcs) {
    return (
      <Card className="insp-card">
        <CardHead title="Sky" />
        <CardBody><p className="insp-dim">No celestial WCS. The viewer assumes 0.1″ pixels.</p></CardBody>
      </Card>
    );
  }
  const coords = `${wcs.ra.toFixed(6)} ${wcs.dec >= 0 ? "+" : ""}${wcs.dec.toFixed(6)}`;
  return (
    <Card className="insp-card">
      <CardHead title="Sky" right={wcs.constructed ? <Badge size="sm" tone="warn">built from RA/DEC</Badge> : <Badge size="sm" tone="good">WCS</Badge>} />
      <CardBody>
        <DefList dense items={[
          ["centre", <span className="insp-id"><span className="mono">{formatRaDec(wcs.ra, wcs.dec)}</span><CopyButton value={coords} label="Copy RA Dec (degrees)" /></span>],
          ["degrees", <span className="mono">{coords}</span>],
          ["pixel", `${formatNumber(wcs.pixscale_arcsec, { sig: 4 })}″`],
          ["extent", `${formatNumber(wcs.width_arcsec, { sig: 4 })}″ × ${formatNumber(wcs.height_arcsec, { sig: 4 })}″`],
          ["projection", <span className="mono">{wcs.ctype.join(" ")}</span>],
        ]} />
        <SkyLink wcs={wcs} className="insp-card__action" />
      </CardBody>
    </Card>
  );
}

function StatsCard({ stats, loading, error, unit, title }: {
  stats: ImageStats | null; loading: boolean; error: string | null; unit: string; title: string;
}) {
  const u = unit ? ` ${unit}` : "";
  return (
    <Card className="insp-card">
      <CardHead title="Statistics" sub={title}
        right={stats?.sampled ? <Badge size="sm" tone="info">every {stats.sampled}th px</Badge> : undefined} />
      <CardBody>
        {loading && <Skeleton lines={6} />}
        {error && <Callout tone="bad" title="No statistics">{error}</Callout>}
        {stats && (
          <DefList dense items={[
            ["pixels", `${formatCount(stats.n_finite)} finite of ${formatCount(stats.n)}`],
            stats.n_nan + stats.n_posinf + stats.n_neginf > 0
              ? ["non-finite", <Badge size="sm" tone="warn">{formatCount(stats.n_nan)} NaN · {formatCount(stats.n_posinf + stats.n_neginf)} ±∞</Badge>] : null,
            ["min / max", `${formatNumber(stats.min)} / ${formatNumber(stats.max)}${u}`],
            ["mean ± σ", `${formatNumber(stats.mean)} ± ${formatNumber(stats.std)}${u}`],
            ["median", `${formatNumber(stats.median)}${u}`],
            ["σ (MAD)", `${formatNumber(stats.mad_std)}${u}`],
            ["p1 / p99", `${formatNumber(stats.percentiles["1"])} / ${formatNumber(stats.percentiles["99"])}${u}`],
            stats.sum != null ? ["sum", `${formatNumber(stats.sum, { sig: 5 })}${u}`] : null,
            stats.n_negative ? ["negative", formatCount(stats.n_negative)] : null,
          ]} />
        )}
      </CardBody>
    </Card>
  );
}

export function ImagePanel({ fits, summary, sel, unit, head }: {
  fits: string; summary: InspectResponse; sel: Selected; unit: string;
  /** The page's HDU / view picker, the start of this panel's one control row. */
  head?: ReactNode;
}) {
  const [stack, setStack] = useUrlState("stack", "bands");
  const [bin, setBin] = useUrlState("bin", "auto");
  const [render, setRender] = useUrlState("render", "asinh");
  const [band, setBand] = useUrlState("band", 0);
  const api = useRef<ViewerApi | null>(null);
  const [vstate, setVstate] = useState<Pick<ViewerState, "index" | "tiers"> | null>(null);

  const hdu = sel.hdu;
  const group = sel.group;
  const stackable = !!(hdu && hdu.ndim === 3 && hdu.bands);
  const stacked = !!group || (stackable && stack !== "planes");
  const axes = hdu?.plane_axes ?? [];
  const planes = hdu?.planes ?? 1;
  const plane = stacked ? 0 : Math.min(vstate?.index ?? 0, Math.max(0, planes - 1));
  const params = useMemo(() => viewerParams(fits, sel, { stack, bin, render }), [fits, sel, stack, bin, render]);

  // Which plane of which HDU the statistics / thumbnail describe.
  const bands = group?.bands ?? (stackable ? hdu?.bands ?? null : null);
  const bandIdx = bands ? Math.min(Math.max(0, band), bands.length - 1) : 0;
  const statsHdu = group ? group.hdus[bandIdx] : hdu?.index ?? 0;
  const statsPlane = group ? 0 : stacked ? bandIdx : plane;
  const stats = useResource<ImageStats>(imageStatsUrl(fits, statsHdu, statsPlane), [], { ttl: 60_000 });
  const planeBand = hdu?.ndim === 3 ? hdu.bands?.[plane] : undefined;
  const statsTitle = bands && stacked ? bands[bandIdx]
    : planeBand ?? (planes > 1 ? `plane ${formatCount(plane)}` : "whole image");

  const tierOptions = useMemo(() => [
    ...summary.hdus.filter((h) => h.type === "image" && h.viewable)
      .map((h) => ({ value: `h${h.index}`, label: `${h.index} · ${h.name}` })),
    ...summary.band_groups.map((g) => ({ value: g.id, label: g.label })),
  ], [summary]);
  const shownTiers = vstate?.tiers ?? [group ? group.id : `h${hdu?.index ?? 0}`];

  const wcs = group?.wcs ?? hdu?.wcs;
  // A results FITS opens on LR and SR colour side by side (the comparison it is for).
  const firstTiers = useMemo(() => compareTiers(summary, sel), [summary, sel]);
  // A bright target (the poster galaxy's core at VIS 12 AB): scale knee and
  // white to its 99.9th percentile once per HDU, so the core is not blown
  // out. A per-viewer setting; the Display dock resets it.
  const exposure = useRef<{ at: string; view: { knee: number; gain: number } | null }>({ at: "", view: null });
  const p999 = stats.data?.percentiles?.["99.9"];
  const exposureAt = `${fits}|${sel.key}|${render}`;
  useEffect(() => {
    if (render === "log" || p999 == null || exposure.current.at === exposureAt) return;
    const view = brightExposure(p999, 100);
    exposure.current = { at: exposureAt, view };
    if (view) api.current?.setView(view);
  }, [p999, exposureAt, render]);
  return (
    <div className="insp-img">
      <div className="insp-viewbar">
        {head}
        <div className="insp-viewbar__controls" role="toolbar" aria-label="Image controls">
        {stackable && (
          <Segmented size="sm" aria-label="Planes as" value={stack === "planes" ? "planes" : "bands"}
            onChange={(v) => setStack(v)}
            options={[
              { value: "bands", label: "Colour", title: `The ${hdu?.bands?.length} planes as one colour cube${hdu?.bands_assumed ? " (bands assumed VIS, Y, J, H)" : ""}` },
              { value: "planes", label: "Planes", title: "One plane at a time" },
            ]} />
        )}
        <label className="insp-inline">
          <span className="insp-inline__label">Bin</span>
          <Select size="sm" aria-label="Display bin" value={bin} onChange={setBin} options={BIN_OPTIONS} />
        </label>
        <Segmented size="sm" aria-label="Render" value={render === "log" ? "log" : "asinh"} onChange={setRender}
          options={[
            { value: "asinh", label: "asinh", title: "The Display panel's stretch (absolute asinh by default)" },
            { value: "log", label: "log", title: "log₁₀ over 6 decades (kernels, PSFs)" },
          ]} />
        {tierOptions.length > SHOWN_TIERS && (
          <MultiSelect size="sm" aria-label="Compare HDUs" placeholder="Compare…" value={shownTiers}
            options={tierOptions} onChange={(keys) => { if (keys.length) api.current?.setTiers(keys); }} />
        )}
        </div>
        {!stacked && planes > 1 && (
          <PlanePicker axes={axes} ndim={hdu?.ndim ?? 3} plane={plane} bands={hdu?.bands ?? null}
            onPlane={(k) => api.current?.goTo(Math.max(0, Math.min(planes - 1, k)))} />
        )}
      </div>
      {hdu?.bands_assumed && stacked && (
        <p className="insp-note">No BANDS card: the four planes are taken as VIS, Y, J, H.</p>
      )}
      <FitBox className="insp-viewer" label="Image">
        <ImageViewer collection="fits" params={params} urlKey="fits" tiers={firstTiers}
          onReady={(a) => {
            api.current = a;
            const e = exposure.current;
            if (a && e.view && e.at === exposureAt) a.setView(e.view);
          }}
          onState={(s) => setVstate((prev) => (prev && prev.index === s.index && prev.tiers.join() === s.tiers.join()
            ? prev : { index: s.index, tiers: s.tiers }))} />
      </FitBox>
      <div className="insp-cards">
        <StatsCard stats={stats.data} loading={stats.loading} error={stats.error?.message ?? null} unit={unit}
          title={statsTitle} />
        <Card className="insp-card insp-card--wide">
          <CardHead title="Histogram" sub={statsTitle}
            right={bands && stacked ? (
              <Segmented size="sm" aria-label="Band" value={String(bandIdx)} onChange={(v) => setBand(Number(v))}
                options={bands.map((b, i) => ({ value: String(i), label: b.replace("_E", "") }))} />
            ) : undefined} />
          <CardBody>
            {stats.loading && <Skeleton height={190} />}
            {stats.data?.histogram
              ? <HistogramPlot hist={stats.data.histogram} label="value" unit={unit}
                  exportName={`${basename(fits)}_hdu${statsHdu}_p${statsPlane}_hist`}
                  markers={[{ v: stats.data.median, label: "median" }]} />
              : !stats.loading && <p className="insp-dim">No finite pixels.</p>}
          </CardBody>
        </Card>
        <WcsCard wcs={wcs} />
        <Card className="insp-card">
          <CardHead title="Preview" sub={`${shapeText(group ? group.shape : hdu?.shape)} · PNG`}
            right={<Button size="sm" variant="ghost" icon="external"
              href={previewUrl(fits, { hdu: statsHdu, plane: statsPlane, size: 1024 })} target="_blank" rel="noreferrer">1024 px</Button>} />
          <CardBody>
            <img className="insp-thumb" loading="lazy"
              src={previewUrl(fits, { hdu: statsHdu, plane: statsPlane, size: 256 })}
              alt={`Thumbnail of HDU ${statsHdu}, plane ${statsPlane}`} />
          </CardBody>
        </Card>
      </div>
    </div>
  );
}
