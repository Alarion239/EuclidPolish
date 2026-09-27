/* realism/pixels (spec §8.3): field statistics — the LR pixels the model is
   trained on (synthetic test + validation) against the multipoint Euclid
   archive fields, one 255×255 crop at 0.1″ per field. Views (?view=):
   pixels (7 figures + the band ledger), detection (the VIS source detection
   of every field: detections, negative islands, completeness — computed by
   the build and never shown before), census (the generated population) and
   inputs (the FASRC steps behind both samples). Bands and samples toggle in
   the toolbar (?hide=); every figure zooms (drag / Ctrl-wheel) and has exact
   bounds. A real-field dot opens its archive field in the inspector. */
import { useMemo, type ReactNode } from "react";
import { Link } from "react-router-dom";
import Plot, { Legend } from "../../../charts/Plot";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import { StepById } from "../../../fasrc";
import { formatCount, formatDateTime, formatNumber, formatPercent } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { linearTicks } from "../../../ticks";
import {
  Badge, Button, Card, CardBody, CardHead, Chip, DefList, EmptyState, JobProgress, Page, Segmented, Stat, Table,
  toast, type Column,
} from "../../../ui";
import { useArchiveMeta, usePixels, type Band, type Comparison, type FieldComparison, type PixelsPayload, type SourceDetection } from "../api";
import { logDomain, nearestIndex2d, ticksFor, domainOf } from "../chartKit";
import { BarGroup, BarSpacer, BoundedPlot, Info, LoadState, RealismBar, StatStrip, Swatch } from "../common";
import { useIncludeTraining } from "../header";
import { JOB, runJob, useRealismJob } from "../jobs";
import {
  BANDS, SAMPLES, SAMPLE_LABEL, bandLabel, bandLegend, correlationSeries, detectionHistogram, detectionStats,
  histogramSeries, perFieldCompleteness, powerSeries, quantileSeries, relationDomains, relationPoints,
  relationSeries, sampleLegend, similaritySeries, visibleFrom, type RelationKey, type Visible,
} from "../pixels/model";
import { bandColor, C } from "../../../colors";

export const PIXEL_VIEWS = [
  { value: "pixels", label: "Pixels" },
  { value: "detection", label: "Detection" },
  { value: "census", label: "Census" },
  { value: "inputs", label: "Inputs" },
] as const;
type View = (typeof PIXEL_VIEWS)[number]["value"];
const VIEW_SET = new Set<string>(PIXEL_VIEWS.map((v) => v.value));
const BUILD_URL = "/api/population-comparison/build";

export function buildStatistics(payload: PixelsPayload | null | undefined) {
  const n = (payload?.availability.synthetic.fields ?? 0) + (payload?.availability.real.fields ?? 0);
  return runJob({
    url: BUILD_URL, label: "Rebuild field statistics",
    question: {
      title: "Rebuild the field statistics?", confirmLabel: "Rebuild",
      message: `Streams ${n} synthetic and real fields through the pixel and detection statistics (TensorFlow reads the TFRecords; several minutes). The source FITS and TFRecords are not changed.`,
    },
  });
}

type Toggle = { hidden: string[]; toggle: (key: string) => void };

function useToggles(): Toggle & { visible: Visible } {
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

function PixelFigures({ comparison, t, onRealField }: {
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
      <RelationFigure fields={fields} which="median_robust_std" title="Background vs robust noise"
        sub="median and 1.4826 × MAD suppress bright objects" t={t} onRealField={onRealField} />
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
        <div className="rl-scores">
          {BANDS.filter((b) => t.visible.bands.includes(b)).map((band) => {
            const s = fields.scale_similarity[band];
            return (
              <div key={band} className="rl-score" style={{ ["--sw" as string]: bandColor(band) }}>
                <strong>{bandLabel(band)}</strong>
                <span>overlap <b>{s.overlap.median.toFixed(3)}</b> <small>{s.overlap.p16.toFixed(3)}–{s.overlap.p84.toFixed(3)}</small></span>
                <span>power syn / real <b>{s.variance_ratio.median.toFixed(2)}</b> <small>{s.variance_ratio.p16.toFixed(2)}–{s.variance_ratio.p84.toFixed(2)}</small></span>
              </div>
            );
          })}
        </div>
        <BoundedPlot boundsLabel="Scale-spectrum similarity" xDomain={simX} yDomain={simY} xScale="log"
          xTicks={ticksFor(simX, "log")} yTicks={linearTicks(simY, { count: 5 })} guides={[{ axis: "y", v: 0, dash: [4, 4] }]}
          xLabel="angular scale (arcsec / cycle)" yLabel="log₁₀ scale-share ratio (synthetic ÷ real)" series={sim}
          aspect={0.34} exportName="field-scale-similarity" aria-label="Scale-spectrum similarity" />
        <Legend items={bandLegend()} hidden={t.hidden} onToggle={t.toggle} />
      </FigureCard>
    </div>
  );
}

type LedgerRow = { band: Band };

function BandLedger({ fields }: { fields: FieldComparison }) {
  const pair = (band: Band, k: "mean" | "std" | "robust_std") =>
    `${formatNumber(fields.summary.synthetic[band][k].median, { sig: 4 })} / ${formatNumber(fields.summary.real[band][k].median, { sig: 4 })}`;
  const pct = (band: Band, k: "zero_fraction" | "negative_fraction") =>
    `${formatPercent(fields.summary.synthetic[band][k].median, 2)} / ${formatPercent(fields.summary.real[band][k].median, 2)}`;
  const columns: Column<LedgerRow>[] = [
    { header: "band", cell: (r) => <span className="rl-legend-cell"><Swatch color={bandColor(r.band)} />{bandLabel(r.band)}</span> },
    { header: "mean", align: "right", cell: (r) => pair(r.band, "mean") },
    { header: "std", align: "right", cell: (r) => pair(r.band, "std") },
    { header: "robust σ", align: "right", cell: (r) => pair(r.band, "robust_std") },
    { header: "zero", align: "right", cell: (r) => pct(r.band, "zero_fraction") },
    { header: "negative", align: "right", cell: (r) => pct(r.band, "negative_fraction") },
  ];
  return (
    <Card>
      <CardHead title="Median field metrics" sub="synthetic / real · e⁻ per pixel unless marked" />
      <CardBody><div className="rl-scroll-x"><Table columns={columns} rows={BANDS.map((band) => ({ band }))} rowKey={(r) => r.band} /></div></CardBody>
    </Card>
  );
}

function Detection({ detection, t }: { detection?: SourceDetection; t: Toggle & { visible: Visible } }) {
  if (!detection) {
    return <EmptyState icon="activity" title="No source-detection statistics in this cache">Rebuild the field statistics to measure them.</EmptyState>;
  }
  const s = detection.settings;
  const stats = { synthetic: detectionStats(detection.synthetic), real: detectionStats(detection.real) };
  type Row = { sample: "synthetic" | "real" };
  const q = (r: { median: number | null; p16: number | null; p84: number | null }) =>
    `${formatNumber(r.median, { digits: 1 })} (${formatNumber(r.p16, { digits: 0 })}–${formatNumber(r.p84, { digits: 0 })})`;
  const columns: Column<Row>[] = [
    { header: "sample", cell: (r) => <strong>{SAMPLE_LABEL[r.sample]}</strong> },
    { header: "fields", align: "right", cell: (r) => formatCount(stats[r.sample].fields) },
    { header: "detections / field", align: "right", cell: (r) => q(stats[r.sample].positive) },
    { header: "negative islands / field", align: "right", cell: (r) => q(stats[r.sample].negative) },
    { header: "negative ÷ positive", align: "right", cell: (r) => formatPercent(stats[r.sample].spurious, 1) },
    { header: "galaxy completeness", align: "right", cell: (r) => (stats[r.sample].completeness == null ? "no truth" : formatPercent(stats[r.sample].completeness, 1)) },
    { header: "matched stars", align: "right", cell: (r) => (r.sample === "real" ? "—" : formatCount(stats[r.sample].matchedStars)) },
  ];
  const completeness = perFieldCompleteness(detection.synthetic);
  const compHist = (() => {
    const edges = Array.from({ length: 21 }, (_, i) => i / 20);
    const counts = new Array<number>(20).fill(0);
    for (const c of completeness) counts[Math.min(19, Math.floor(c * 20))] += 1;
    return { x: edges.slice(0, -1).map((e) => e + 0.025), y: counts.map((c) => (completeness.length ? c / completeness.length : 0)) };
  })();
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
      <div className="rl-grid rl-grid--3">
        {hist("positive", "Detections per field")}
        {hist("negative", "Negative islands per field")}
        <Card>
          <CardHead title="Galaxy completeness per field" sub={`${formatCount(completeness.length)} synthetic fields with truth`} />
          <CardBody>
            {completeness.length ? (
              <Plot xDomain={[0, 1]} yDomain={domainOf(compHist.y, 0, true)} xTicks={linearTicks([0, 1], { count: 6, format: (x) => `${Math.round(100 * x)}%` })}
                yTicks={linearTicks(domainOf(compHist.y, 0, true), { count: 5 })} xLabel="matched ÷ truth galaxies" yLabel="fraction of fields"
                series={[{ x: compHist.x, y: compHist.y, mode: "histogram", color: C.comb, fillAlpha: 0.24, width: 1.5, name: "synthetic LR" }]}
                aspect={0.6} xFormat={(x) => `${Math.round(100 * x)}%`} exportName="detection-completeness" aria-label="Galaxy completeness per field" />
            ) : <EmptyState compact icon="table" title="No synthetic truth matched" />}
          </CardBody>
        </Card>
      </div>
    </div>
  );
}

function Census({ comparison }: { comparison: Comparison }) {
  const pop = comparison.population.synthetic;
  const kinds = ["galaxy", "star", "unknown"].filter((k) => k in pop.counts);
  return (
    <div className="rl-stack">
      <Card>
        <CardHead title="Synthetic source truth" sub={`${formatNumber(pop.area_arcmin2, { digits: 2 })} arcmin² · ${formatCount(comparison.population.synthetic_field_count)} fields`}
          right={<Badge size="sm" tone={comparison.population.training_included ? "warn" : undefined}>
            {comparison.population.training_included ? "train + test + validation" : "test + validation"}
          </Badge>} />
        <CardBody>
          <StatStrip label="Synthetic census">
            <Stat k="catalogue objects" v={formatCount(pop.objects)} />
            {kinds.map((k) => (
              <Stat key={k} k={`${k === "galaxy" ? "galaxies" : `${k}s`} / arcmin²`} v={formatNumber(pop.density_arcmin2[k] ?? 0, { digits: 2 })}
                sub={`${formatCount(pop.counts[k])} objects`} />
            ))}
          </StatStrip>
          {comparison.population.training_included && (
            <p className="rl-faint">Training truth is in this census only; the pixel and detection statistics stay on test + validation.</p>
          )}
          <div className="rl-row">
            <Button asChild size="sm" variant="ghost" iconRight="chevronRight"><Link to="/realism/galaxies">Galaxy population fits</Link></Button>
            <Button asChild size="sm" variant="ghost" iconRight="chevronRight"><Link to="/realism/stars">Stellar prior</Link></Button>
          </div>
        </CardBody>
      </Card>
    </div>
  );
}

function Inputs() {
  return (
    <div className="rl-stack">
      <StepById stepId="vis_noise_sample" />
      <StepById stepId="archive_field_sample" />
      <div className="rl-row">
        <Button asChild size="sm" variant="ghost" iconRight="chevronRight"><Link to="/realism/visual">Compare the fields visually</Link></Button>
      </div>
    </div>
  );
}

function FieldCensus({ payload }: { payload: PixelsPayload }) {
  const a = payload.availability;
  return (
    <StatStrip label="Compared field census">
      <Stat k="field" v={`${payload.comparison?.geometry.tile_size ?? 256} px × ${payload.comparison?.geometry.pixel_scale_arcsec ?? 0.1}″`}
        sub={`${a.field_area_arcmin2.toFixed(3)} arcmin²`} />
      <Stat k="synthetic LR" v={`${formatCount(payload.comparison?.samples.synthetic.fields ?? a.synthetic.fields)} fields`}
        sub={(payload.comparison?.samples.synthetic.splits ?? ["test", "validate"]).join(" + ")} />
      <Stat k="real Euclid LR" v={`${formatCount(payload.comparison?.samples.real.fields ?? a.real.fields)} fields`}
        sub={`${formatCount(payload.comparison?.samples.real.independent_parents ?? a.real.independent_parents)} Q1 pointings`} />
      <Stat k="built" v={payload.comparison?.provenance?.generated_at ? formatDateTime(payload.comparison.provenance.generated_at) : "—"} />
    </StatStrip>
  );
}

export default function PixelsTab() {
  const [training] = useIncludeTraining();
  const [rawView, setView] = useUrlState("view", "pixels");
  const view: View = (VIEW_SET.has(rawView) ? rawView : "pixels") as View;
  const t = useToggles();
  const resource = usePixels(training);
  const payload = resource.data;
  const build = useRealismJob(JOB.pixelsBuild);
  const archive = useArchiveMeta();
  const comparison = payload?.comparison ?? null;
  const cache = payload?.availability.comparison_cache;
  const realReady = !!payload?.availability.real.ready;
  const byParent = useMemo(() => {
    const map = new Map<string, string>();
    for (const o of archive.data?.objects ?? []) if (!map.has(o.parent_id)) map.set(o.parent_id, o.id ?? String(o.sample_id));
    return map;
  }, [archive.data]);
  const onRealField = (parent: string) => {
    const id = byParent.get(parent);
    if (id) openInspector({ kind: "archivefield", id });
    else toast.info(`Real field from pointing ${parent}`, { description: "Its archive sample is not in the local collection." });
  };
  usePageActions([
    ...PIXEL_VIEWS.map((v) => ({ id: `pixels-view-${v.value}`, label: `Field statistics: ${v.label}`, group: "Field statistics", run: () => setView(v.value) })),
    { id: "pixels-build", label: "Rebuild the field statistics", group: "Field statistics", keywords: ["pixels", "detection", "cache"],
      disabled: !realReady, run: () => { void buildStatistics(payload); } },
    ...BANDS.map((b) => ({ id: `pixels-band-${b}`, label: `${t.hidden.includes(b) ? "Show" : "Hide"} ${bandLabel(b)} in the field statistics`,
      group: "Field statistics", run: () => t.toggle(b) })),
  ]);
  return (
    <Page className="rl-page">
      <RealismBar label="Field statistics controls">
        <BarGroup>
          <Segmented size="sm" aria-label="Field statistics view" value={view} onChange={setView} options={[...PIXEL_VIEWS]} />
        </BarGroup>
        {(view === "pixels" || view === "detection") && (
          <BarGroup label="Show">
            <div className="rl-chips" role="group" aria-label="Bands and samples">
              {view === "pixels" && BANDS.map((b) => (
                <Chip key={b} on={!t.hidden.includes(b)} dot={bandColor(b)} onClick={() => t.toggle(b)}>{bandLabel(b)}</Chip>
              ))}
              {SAMPLES.map((s) => <Chip key={s} on={!t.hidden.includes(s)} onClick={() => t.toggle(s)}>{s}</Chip>)}
            </div>
          </BarGroup>
        )}
        {cache && (
          <Badge size="sm" tone={cache.fresh ? "good" : "warn"} dot title={cache.reason ?? undefined}>
            {cache.fresh ? "cache current" : cache.present ? "cache stale" : "not built"}
          </Badge>
        )}
        <BarSpacer />
        <Button size="sm" variant={comparison && cache?.fresh ? "ghost" : "primary"} icon="reset" loading={build.busy}
          disabled={!realReady} onClick={() => void buildStatistics(payload)}>
          {comparison ? "Rebuild" : "Measure fields"}
        </Button>
      </RealismBar>
      <LoadState loading={resource.loading && !payload} error={resource.error} onRetry={resource.reload}>
        {payload && (
          <div className="rl-stack">
            <JobProgress job={build.job} error={build.error} />
            {view !== "inputs" && <FieldCensus payload={payload} />}
            {view === "inputs" ? <Inputs />
              : !comparison ? (
                <EmptyState icon="table"
                  title={realReady ? "The field-statistics cache has not been built" : "The multipoint Euclid reference is not ready"}
                  action={realReady
                    ? <Button variant="primary" onClick={() => void buildStatistics(payload)}>Measure fields</Button>
                    : <Button onClick={() => setView("inputs")}>Open the inputs</Button>}>
                  {realReady ? cache?.reason ?? "Streams every field once; the sources stay unchanged."
                    : payload.availability.real.unavailable_reason ?? "Generate and synchronize the four-band archive fields first."}
                </EmptyState>
              )
                : view === "pixels" ? <><PixelFigures comparison={comparison} t={t} onRealField={onRealField} /><BandLedger fields={comparison.fields} /></>
                  : view === "detection" ? <Detection detection={comparison.fields.source_detection} t={t} />
                    : <Census comparison={comparison} />}
          </div>
        )}
      </LoadState>
    </Page>
  );
}
