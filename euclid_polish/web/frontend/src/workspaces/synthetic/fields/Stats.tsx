/* Synthetic › Fields › statistics and detection: the LR pixels the model is
   trained on (synthetic test + validate) against the multipoint Euclid
   archive fields, one 255 × 255 centre crop at 0.1″ per field.
   - Stats: six figures (brightness distribution, quantile profile,
     angular-scale power, mean vs field variation, inter-band correlation,
     the wide scale-spectrum similarity with the NISP score table), then the
     median field metrics and the geometry caption. The background vs
     robust noise figure lives on Synthetic › Noise (one place), linked
     from a caption. Bands and samples toggle in the tab's toolbar (?hide=); every
     figure zooms and has exact bounds; a real-field dot opens its archive
     field.
   - Detection: detections and negative islands per field; the galaxy
     completeness is a caption (only the synthetic fields have truth).
   The build is the ONE "Measure" job (confirmed), never run on a visit. */
import type { ReactNode } from "react";
import Plot, { Legend } from "../../../charts/Plot";
import { formatCount, formatNumber, formatPercent } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { linearTicks } from "../../../ticks";
import { Caption, Card, CardBody, CardHead, DefList, EmptyState, Num, Table, toast, type Column } from "../../../ui";
import type { Band, Comparison, FieldComparison, PixelsPayload, SourceDetection } from "../api";
import { domainOf, logDomain, nearestIndex2d, quantile, ticksFor } from "../chartKit";
import { BoundedPlot, Info, Swatch } from "../common";
import { runJob } from "../jobs";
import { powerOff } from "../statusModel";
import {
  BANDS, SAMPLES, SAMPLE_LABEL, bandLabel, bandLegend, correlationSeries, detectionHistogram, detectionStats, histogramSeries,
  perFieldCompleteness, powerSeries, quantileSeries, relationDomains, relationPoints, relationSeries, sampleLegend,
  similaritySeries, visibleFrom, type RelationKey, type Visible,
} from "./model";
import { bandColor, C } from "../../../colors";

const BUILD_URL = "/api/population-comparison/build";

/** The field-statistics build (the Status row's "Rebuild field statistics"
 *  and Fields' Measure share its job key). */
export function buildStatistics(payload: PixelsPayload | null | undefined) {
  const real = payload?.availability.real;
  const n = (payload?.availability.synthetic.fields ?? 0) + (real?.compared_fields ?? real?.fields ?? 0);
  return runJob({
    url: BUILD_URL, label: "Rebuild field statistics",
    question: {
      title: "Measure the field statistics?", confirmLabel: "Measure",
      message: `Streams ${n} synthetic and real fields through the pixel and detection statistics (TensorFlow reads the TFRecords; several minutes). The source FITS and TFRecords are not changed.`,
    },
  });
}

export type Toggle = { hidden: string[]; toggle: (key: string) => void };

export function useToggles(): Toggle & { visible: Visible } {
  const [hidden, setHidden] = useUrlState<string[]>("hide", []);
  const toggle = (key: string) => setHidden(hidden.includes(key) ? hidden.filter((k) => k !== key) : [...hidden, key]);
  return { hidden, toggle, visible: visibleFrom(hidden) };
}

function Legends({ t, kind, bands = true }: { t: Toggle; kind: "line" | "histogram" | "scatter"; bands?: boolean }) {
  return (
    <div className="rl-legends">
      {bands && <Legend items={bandLegend(kind === "histogram")} hidden={t.hidden} onToggle={t.toggle} />}
      <Legend items={sampleLegend(kind)} hidden={t.hidden} onToggle={t.toggle} />
    </div>
  );
}

function FigureCard({ title, sub, info, children, wide }: {
  title: string; sub: string; info?: string; children: ReactNode; wide?: boolean;
}) {
  return (
    <Card className={wide ? "rl-wide" : undefined}>
      <CardHead title={title} sub={sub} right={info ? <Info label={`About: ${title}`}>{info}</Info> : undefined} />
      <CardBody>{children}</CardBody>
    </Card>
  );
}

function RelationFigure({ fields, which, title, sub, t, onRealField }: {
  fields: FieldComparison; which: RelationKey; title: string; sub: string; t: Toggle & { visible: Visible };
  onRealField: (parent: string) => void;
}) {
  const series = relationSeries(fields, which, t.visible);
  const d = relationDomains(fields, which);
  const any = BANDS.map((b) => fields.relations[which][b]).find(Boolean);
  const points = relationPoints(fields, which, t.visible);
  const pick = (p: { x: number; y: number }) => {
    const i = nearestIndex2d(points.map((q) => q.x), points.map((q) => q.y), p, { x: d.x[1] - d.x[0], y: d.y[1] - d.y[0] }, { maxDistance: 0.03 });
    const hit = i >= 0 ? points[i] : null;
    if (!hit) return;
    if (hit.sample === "real" && hit.parent) onRealField(hit.parent);
    else toast.info(`${bandLabel(hit.band)} · ${SAMPLE_LABEL[hit.sample]}${hit.parent ? ` · ${hit.parent}` : ""}`);
  };
  return (
    <FigureCard title={title} sub={sub}>
      <BoundedPlot boundsLabel={title} xDomain={d.x} yDomain={d.y} xTicks={linearTicks(d.x, { count: 5 })}
        yTicks={linearTicks(d.y, { count: 5 })} xLabel={any?.x_label} yLabel={any?.y_label} series={series}
        onPlotClick={pick} aspect={0.66} exportName={`field-${which}`} aria-label={title} />
      <Legends t={t} kind="scatter" />
    </FigureCard>
  );
}

export function PixelFigures({ comparison, t, onRealField }: {
  comparison: Comparison; t: Toggle & { visible: Visible }; onRealField: (parent: string) => void;
}) {
  const fields = comparison.fields;
  const v = t.visible;
  const hist = histogramSeries(fields, v);
  const histX = domainOf(BANDS.flatMap((b) => fields.histograms[b].x));
  const histY = domainOf(hist.flatMap((s) => s.y), 0, true);
  const quant = quantileSeries(fields, v);
  const quantY = domainOf(BANDS.flatMap((b) => [...fields.quantiles[b].synthetic, ...fields.quantiles[b].real]));
  const power = powerSeries(fields, v);
  const powerX = logDomain(power.flatMap((s) => s.x));
  const powerY = logDomain(power.flatMap((s) => s.y));
  const sim = similaritySeries(fields, v);
  const simX = logDomain(sim.flatMap((s) => s.x));
  const simVals = sim.flatMap((s) => [...(s.low ?? []), ...(s.high ?? []), ...s.y]);
  const simY = domainOf([...simVals, 0], 0.1);
  const corr = correlationSeries(fields, v);
  const pairs = fields.band_correlation.pairs;
  const corrX: [number, number] = [-0.35, pairs.length - 0.65];
  const corrY = domainOf(corr.flatMap((s) => [...(s.low ?? []), ...(s.high ?? [])]));
  const unit = fields.histograms.VIS?.x_label ?? "pixel brightness (e⁻ / stack)";
  return (
    <div className="rl-plot-grid">
      <FigureCard title="Brightness distribution" sub="filled bars synthetic, hatched outlines real · the 0 e⁻ bin is hidden"
        info="The exact-zero bin is blanked to expose the signal wings; every histogram uses the same 255×255 centre crop at the native 0.1″ pixel scale.">
        <BoundedPlot boundsLabel="Brightness distribution" xDomain={histX} yDomain={histY}
          xTicks={linearTicks(histX, { count: 6 })} yTicks={linearTicks(histY, { count: 5 })} xLabel={unit}
          yLabel="fraction of sampled pixels / bin" series={hist} aspect={0.6} exportName="field-brightness"
          aria-label="Pixel brightness distribution" />
        <Legends t={t} kind="histogram" />
      </FigureCard>
      <FigureCard title="Pixel quantile profile" sub="0.1st to 99.9th percentile · circles synthetic, diamonds real">
        <BoundedPlot boundsLabel="Pixel quantile profile" xDomain={[0.1, 99.9]} yDomain={quantY}
          xTicks={linearTicks([0.1, 99.9], { count: 6, format: (x) => `${formatNumber(x, { sig: 3 })}%` })}
          yTicks={linearTicks(quantY, { count: 5 })} xLabel="pixel percentile" yLabel={unit} series={quant}
          xFormat={(x) => `${formatNumber(x, { sig: 3 })}%`} aspect={0.6} exportName="field-quantiles"
          aria-label="Pixel quantile profile" />
        <Legends t={t} kind="line" />
      </FigureCard>
      <FigureCard title="Angular-scale power" sub="median mean-subtracted field power">
        <BoundedPlot boundsLabel="Angular-scale power" xDomain={powerX} yDomain={powerY} xScale="log" yScale="log"
          xTicks={ticksFor(powerX, "log")} yTicks={ticksFor(powerY, "log")} xLabel="angular scale (arcsec / cycle)"
          yLabel="mean-subtracted power (e⁻², log scale)" series={power} aspect={0.6} exportName="field-power"
          aria-label="Angular-scale power spectrum" />
        <Legends t={t} kind="line" />
      </FigureCard>
      <RelationFigure fields={fields} which="mean_std" title="Mean brightness vs field variation"
        sub="one marker per field · click a real field to inspect it" t={t} onRealField={onRealField} />
      <FigureCard title="Inter-band pixel correlation" sub="median within-field Pearson r · ribbons 16–84% over fields">
        <BoundedPlot boundsLabel="Inter-band pixel correlation" xDomain={corrX} yDomain={corrY}
          xTicks={pairs.map((label, i) => ({ v: i, label }))} yTicks={linearTicks(corrY, { count: 5 })}
          xLabel="band pair" yLabel="within-field pixel correlation" series={corr} zoomAxes="y"
          xFormat={(x) => pairs[Math.round(x)] ?? ""} aspect={0.6} exportName="field-band-correlation"
          aria-label="Inter-band pixel correlation" />
        <Legends t={t} kind="line" bands={false} />
      </FigureCard>
      <FigureCard wide title="Scale-spectrum similarity" sub="where each population places its fluctuation power · bootstrap 16–84%"
        info="0 means an equal share of the variance at that scale; positive means the synthetic fields place more of their variance there, negative means Euclid does.">
        <BoundedPlot boundsLabel="Scale-spectrum similarity" xDomain={simX} yDomain={simY} xScale="log"
          xTicks={ticksFor(simX, "log")} yTicks={linearTicks(simY, { count: 5 })} guides={[{ axis: "y", v: 0, dash: [4, 4] }]}
          xLabel="angular scale (arcsec / cycle)" yLabel="log₁₀ scale-share ratio (synthetic ÷ real)" series={sim}
          aspect={0.34} exportName="field-scale-similarity" aria-label="Scale-spectrum similarity" />
        <Legend items={bandLegend()} hidden={t.hidden} onToggle={t.toggle} />
        <ScaleScores fields={fields} bands={BANDS.filter((b) => t.visible.bands.includes(b))} />
      </FigureCard>
    </div>
  );
}

const interval2 = (i: { median: number; p16: number; p84: number }) => `${i.median.toFixed(2)} (${i.p16.toFixed(2)}–${i.p84.toFixed(2)})`;

/** The NISP scale scores: one comparison table (median, bootstrap 16–84%);
 *  a power ratio outside ±25% is warn-toned. VIS is the summary line's. */
function ScaleScores({ fields, bands }: { fields: FieldComparison; bands: Band[] }) {
  const columns: Column<Band>[] = [
    { header: "band", cell: (b) => <span className="rl-legend-cell"><Swatch color={bandColor(b)} />{bandLabel(b)}</span> },
    { header: "overlap (16–84%)", align: "right", cell: (b) => interval2(fields.scale_similarity[b].overlap) },
    { header: "power syn / real (16–84%)", align: "right", cell: (b) => {
      const r = fields.scale_similarity[b].variance_ratio;
      return <Num tone={powerOff(r.median) ? "warn" : undefined}>{interval2(r)}</Num>;
    } },
  ];
  const rows = bands.filter((b) => b !== "VIS" && fields.scale_similarity[b]);
  if (!rows.length) return null;
  return <div className="rl-scores"><Table columns={columns} rows={rows} rowKey={(b) => b} aria-label="Scale-spectrum similarity per NISP band" /></div>;
}

/** The field geometry under the figures it qualifies: the analysed centre
 *  crop (255 × 255 of the 256-px tiles) and its area. */
export function geometryText(g: Comparison["geometry"]): string {
  const side = g.analysis_size || g.tile_size;
  const area = (side * g.pixel_scale_arcsec / 60) ** 2;
  const tile = g.analysis_size && g.analysis_size !== g.tile_size ? ` (the centre of each ${g.tile_size}-px tile)` : "";
  return `Each field: ${side} × ${side} px at ${formatNumber(g.pixel_scale_arcsec, { sig: 3 })}″, ${formatNumber(area, { sig: 3 })} arcmin²${tile}`;
}

export function GeometryCaption({ comparison }: { comparison: Comparison }) {
  return <Caption>{geometryText(comparison.geometry)}</Caption>;
}

type LedgerRow = { band: Band };

export function BandLedger({ fields }: { fields: FieldComparison }) {
  const pair = (band: Band, k: "mean" | "std" | "robust_std") =>
    `${formatNumber(fields.summary.synthetic[band][k].median, { sig: 3 })} / ${formatNumber(fields.summary.real[band][k].median, { sig: 3 })}`;
  const pct = (band: Band, k: "zero_fraction" | "negative_fraction") =>
    `${formatPercent(fields.summary.synthetic[band][k].median, 1)} / ${formatPercent(fields.summary.real[band][k].median, 1)}`;
  const columns: Column<LedgerRow>[] = [
    { header: "Band", cell: (r) => <span className="rl-legend-cell"><Swatch color={bandColor(r.band)} />{bandLabel(r.band)}</span> },
    { header: "Mean (e⁻)", align: "right", cell: (r) => pair(r.band, "mean") },
    { header: "Std (e⁻)", align: "right", cell: (r) => pair(r.band, "std") },
    { header: "Robust σ (e⁻)", align: "right", cell: (r) => pair(r.band, "robust_std") },
    { header: "Zero", align: "right", cell: (r) => pct(r.band, "zero_fraction") },
    { header: "Negative", align: "right", cell: (r) => pct(r.band, "negative_fraction") },
  ];
  return (
    <Card>
      <CardHead title="Median field metrics" sub="synthetic / real per band · zero and negative as a share of the pixels" />
      <CardBody><div className="rl-scroll-x"><Table columns={columns} rows={BANDS.map((band) => ({ band }))} rowKey={(r) => r.band} /></div></CardBody>
    </Card>
  );
}

/** The galaxy completeness in words (only the synthetic fields have truth):
 *  the overall share of truth galaxies matched, the per-field median and
 *  16–84% range, and the matched stars. */
export function completenessCaption(detection: SourceDetection): string | null {
  const stats = detectionStats(detection.synthetic);
  const perField = perFieldCompleteness(detection.synthetic);
  if (stats.completeness == null && !perField.length) return null;
  const median = quantile(perField, 0.5), p16 = quantile(perField, 0.16), p84 = quantile(perField, 0.84);
  const pct0 = (v: number) => `${Math.round(100 * v)}%`;
  return [
    `Galaxy completeness, synthetic only (the real fields have no truth): ${formatPercent(stats.completeness, 1)} of the truth galaxies detected`,
    median != null && p16 != null && p84 != null ? `per field ${pct0(median)} (16–84% ${pct0(p16)}–${pct0(p84)})` : null,
    stats.matchedStars ? `${formatCount(stats.matchedStars)} detections matched to stars` : null,
  ].filter(Boolean).join(" · ");
}

export function Detection({ detection, t }: { detection?: SourceDetection; t: Toggle & { visible: Visible } }) {
  if (!detection) {
    return <EmptyState icon="activity" title="No source-detection statistics in this cache">Rebuild the field statistics to measure them.</EmptyState>;
  }
  const s = detection.settings;
  const stats = { synthetic: detectionStats(detection.synthetic), real: detectionStats(detection.real) };
  type Row = { sample: "synthetic" | "real" };
  // Counts per field: whole numbers (a median of two fields can end in .5).
  const q = (r: { median: number | null; p16: number | null; p84: number | null }) =>
    `${formatNumber(r.median, { digits: r.median != null && !Number.isInteger(r.median) ? 1 : 0 })} `
      + `(${formatCount(r.p16)}–${formatCount(r.p84)})`;
  const totalNegative = [...detection.synthetic.negative, ...detection.real.negative]
    .reduce((a, v) => a + (Number.isFinite(v) ? v : 0), 0);
  const anyNegative = totalNegative > 0;
  const columns: Column<Row>[] = [
    { header: "Sample", cell: (r) => <strong>{SAMPLE_LABEL[r.sample]}</strong> },
    { header: "Detections per field (16–84%)", align: "right", cell: (r) => q(stats[r.sample].positive) },
    ...(anyNegative ? [
      { header: "Negative islands per field (16–84%)", align: "right", cell: (r) => q(stats[r.sample].negative) },
      { header: "Negative ÷ positive", align: "right", cell: (r) => formatPercent(stats[r.sample].spurious, 1) },
    ] satisfies Column<Row>[] : []),
  ];
  const completenessText = completenessCaption(detection);
  const hist = (which: "positive" | "negative", title: string) => {
    const series = detectionHistogram(detection, which, t.visible);
    const xs = series.flatMap((x) => x.x);
    const xDomain = domainOf(xs, 1, true);
    const yDomain = domainOf(series.flatMap((x) => x.y), 0, true);
    return (
      <Card>
        <CardHead title={title} sub="fraction of fields per bin" />
        <CardBody>
          <Plot xDomain={xDomain} yDomain={yDomain} xTicks={linearTicks(xDomain, { count: 6 })} yTicks={linearTicks(yDomain, { count: 5 })}
            xLabel={`${which === "positive" ? "detections" : "negative islands"} per field`} yLabel="fraction of fields"
            series={series} aspect={0.6} exportName={`detection-${which}`} aria-label={title} />
          <Legend items={sampleLegend("histogram").map((i) => ({ ...i, color: i.key === "synthetic" ? C.comb : C.mean }))}
            hidden={t.hidden} onToggle={t.toggle} />
        </CardBody>
      </Card>
    );
  };
  return (
    <div className="rl-stack">
      <Card>
        <CardHead title="VIS source detection per field" sub={`${s.threshold_sigma}σ · ≥ ${s.minimum_connected_pixels} connected pixels · deblended`}
          right={<Info label="About the detection settings">
            <DefList dense items={[
              ["band", s.band], ["threshold", `${s.threshold_sigma}σ`], ["min. connected pixels", String(s.minimum_connected_pixels)],
              ["background box", `${s.background_box_pixels} px`], ["deblending", `${s.deblend_levels} levels · contrast ${s.deblend_contrast}`],
              ["negative-image correction", s.negative_image_correction ? "yes" : "no"],
              ["truth match radius", `${s.truth_match_radius_lr_pixels} LR px`],
            ]} />
            <p>Negative islands are detections on the inverted image: a symmetric noise field yields as many as its false positives.
              Completeness matches synthetic detections one-to-one to the truth galaxies of the source catalogue.</p>
          </Info>} />
        <CardBody><div className="rl-scroll-x"><Table columns={columns} rows={SAMPLES.map((sample) => ({ sample }))} rowKey={(r) => r.sample} /></div></CardBody>
      </Card>
      {anyNegative ? (
        <div className="rl-grid rl-grid--2">
          {hist("positive", "Detections per field")}
          {hist("negative", "Negative islands per field")}
        </div>
      ) : hist("positive", "Detections per field")}
      {!anyNegative && <Caption>Neither sample has a negative island (a detection on the inverted image), so no noise peak passes the threshold in either.</Caption>}
      {completenessText && <Caption>{completenessText}</Caption>}
    </div>
  );
}
