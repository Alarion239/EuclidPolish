/* An image HDU (or a 4-band HDU group) of any dimensionality in the viewer
   (collection `fits`), image first: one control row (the page's HDU / view
   picker, then colour/planes stacking, display bin, log render, HDU
   comparison; the plane picker for a cube), the viewer at the full size the
   stage allows, then one statistics table with a row per frame the viewer
   shows, the selected plane's histogram (its clipping in a tooltip) and the
   sky footprint as one caption line. */
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useResource } from "../../api/query";
import { formatCount, formatNumber } from "../../format";
import { useUrlState } from "../../hooks/useUrlState";
import {
  Caption, CopyButton, IconButton, MultiSelect, Segmented, Select, Skeleton, Slider, Tooltip,
} from "../../ui";
import { ImageViewer, type ViewerApi, type ViewerState } from "../../viewer";
import { imageStatsUrl, type ImageStats, type InspectResponse } from "./api";
import { HistogramPlot } from "./charts";
import {
  axisName, basename, brightExposure, compareDisplay, compareTiers, flatIndex, frameStats, frameTargets,
  groupOptionLabel, hduOptionLabel, planeIndex, viewerParams, wcsCaption, type FrameTarget, type Selected,
} from "./model";

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

/** One frame's row: its statistics are its own request (cached per plane). */
function FrameRow({ fits, frame, unit, showUnit, nonFiniteCol, onNonFinite }: {
  fits: string; frame: FrameTarget; unit: string; showUnit: boolean; nonFiniteCol: boolean;
  onNonFinite: (key: string, n: number) => void;
}) {
  const res = useResource<ImageStats>(imageStatsUrl(fits, frame.hdu, frame.plane), [], { ttl: 60_000 });
  const st = res.data ? frameStats(res.data) : null;
  const n = st?.nonFinite ?? 0;
  useEffect(() => { onNonFinite(frame.key, n); }, [frame.key, n, onNonFinite]);
  const u = showUnit && unit ? <span className="insp-stats__u"> {unit}</span> : null;
  const cell = (v: number | null | undefined, sig = 4) => (st ? <>{formatNumber(v, { sig })}{v != null && u}</> : "…");
  return (
    <tr>
      <th scope="row" className="insp-stats__frame">
        {frame.label}
        {res.data?.sampled ? <span className="insp-dim"> · every {res.data.sampled}th px</span> : null}
      </th>
      {res.error ? <td colSpan={nonFiniteCol ? 5 : 4} className="insp-stats__err">{res.error.message}</td> : (
        <>
          <td>{cell(st?.median)}</td>
          <td>{cell(st?.sigma)}</td>
          <td>{cell(st?.p99)}</td>
          <td>{cell(st?.sum, 3)}</td>
          {nonFiniteCol && <td data-tone={n > 0 ? "warn" : undefined}>{st ? formatCount(n) : "…"}</td>}
        </>
      )}
    </tr>
  );
}

/** Median, σ (MAD), p99 and the total flux of every frame the viewer shows,
 *  one row each; a non-finite column only when some frame has such pixels. */
function FrameStatsTable({ fits, frames, unit }: { fits: string; frames: FrameTarget[]; unit: string }) {
  const [nonFinite, setNonFinite] = useState<Record<string, number>>({});
  const onNonFinite = useCallback((key: string, n: number) => {
    setNonFinite((prev) => (prev[key] === n ? prev : { ...prev, [key]: n }));
  }, []);
  const shown = new Set(frames.map((f) => f.key));
  const anyNonFinite = Object.entries(nonFinite).some(([k, n]) => shown.has(k) && n > 0);
  const head = unit ? ` [${unit}]` : "";
  return (
    <div className="ui-table-wrap insp-stats__wrap">
      <table className="ui-table insp-stats__table" aria-label="Statistics of the frames shown">
        <thead>
          <tr>
            <th scope="col">Frame</th>
            <th scope="col">Median{head}</th>
            <th scope="col"><Tooltip content="Robust σ from the median absolute deviation (1.4826 × MAD)"><span tabIndex={0} className="insp-stats__hint">σ (MAD){head}</span></Tooltip></th>
            <th scope="col">p99{head}</th>
            <th scope="col">Σ flux{head}</th>
            {anyNonFinite && <th scope="col">Non-finite px</th>}
          </tr>
        </thead>
        <tbody>
          {frames.map((f) => (
            <FrameRow key={`${f.key}:${f.hdu}:${f.plane}`} fits={fits} frame={f} unit={unit} showUnit={false}
              nonFiniteCol={anyNonFinite} onNonFinite={onNonFinite} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ImagePanel({ fits, summary, sel, unit, head, facts }: {
  fits: string; summary: InspectResponse; sel: Selected; unit: string;
  /** The selected HDU in one line ("PRIMARY (HDU 0): 20 × 16 × 4 · f4 · electron"). */
  facts?: string;
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
      .map((h) => ({ value: `h${h.index}`, label: hduOptionLabel(h) })),
    ...summary.band_groups.map((g) => ({ value: g.id, label: groupOptionLabel(g) })),
  ], [summary]);

  const wcs = group?.wcs ?? hdu?.wcs;

  // A results FITS opens on LR and SR colour side by side (the comparison it is for).
  const firstTiers = useMemo(() => compareTiers(summary, sel), [summary, sel]);
  const firstDisplay = useMemo(() => compareDisplay(firstTiers), [firstTiers]);
  const shownKey = (vstate?.tiers ?? firstTiers ?? [group ? group.id : `h${hdu?.index ?? 0}`]).join(",");
  const shownTiers = useMemo(() => shownKey.split(",").filter(Boolean), [shownKey]);
  // A colour frame gets one row per band; the band chip below picks the histogram only.
  const frames = useMemo(() => frameTargets(summary, sel, shownTiers, { stacked, plane }),
    [summary, sel, shownTiers, stacked, plane]);
  // A bright target (the poster galaxy's core at VIS 12 AB): the server moves
  // the file's white point to the plane's 99.99th percentile; the knee is
  // seeded here from its 99.9th, once per HDU, so the core keeps its
  // structure. A per-viewer setting; the Display row's reset clears it.
  const exposure = useRef<{ at: string; view: { knee: number } | null }>({ at: "", view: null });
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
      <div className="insp-viewer" role="region" aria-label="Image">
        <ImageViewer collection="fits" params={params} urlKey="fits" tiers={firstTiers}
          display={firstDisplay}
          onReady={(a) => {
            api.current = a;
            const e = exposure.current;
            if (a && e.view && e.at === exposureAt) a.setView(e.view);
          }}
          onState={(s) => setVstate((prev) => (prev && prev.index === s.index && prev.tiers.join() === s.tiers.join()
            ? prev : { index: s.index, tiers: s.tiers }))} />
      </div>
      <div className="insp-stats">
        <FrameStatsTable fits={fits} frames={frames} unit={unit} />
        <section className="insp-stats__hist" aria-label={`Histogram, ${statsTitle}`}>
          <div className="insp-stats__histhead">
            <h3 className="insp-stats__title">Histogram · {statsTitle}</h3>
            {bands && stacked && (
              <Segmented size="sm" aria-label="Band" value={String(bandIdx)} onChange={(v) => setBand(Number(v))}
                options={bands.map((b, i) => ({ value: String(i), label: b.replace("_E", "") }))} />
            )}
          </div>
          {stats.loading && <Skeleton height={190} />}
          {stats.error && <p className="insp-dim">{stats.error.message}</p>}
          {stats.data?.histogram
            ? <HistogramPlot hist={stats.data.histogram} label="value" unit={unit}
                exportName={`${basename(fits)}_hdu${statsHdu}_p${statsPlane}_hist`}
                markers={[{ v: stats.data.median, label: "median" }]} />
            : stats.data && <p className="insp-dim">No finite pixels.</p>}
        </section>
      </div>
      <Caption className="insp-wcs">
        {facts ? `${facts}. ` : ""}{wcsCaption(wcs)}
        {wcs && <CopyButton value={`${wcs.ra.toFixed(6)} ${wcs.dec >= 0 ? "+" : ""}${wcs.dec.toFixed(6)}`} label="Copy RA Dec (degrees)" />}
      </Caption>
    </div>
  );
}
