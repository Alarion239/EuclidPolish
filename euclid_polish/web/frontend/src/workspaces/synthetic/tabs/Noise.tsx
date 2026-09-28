/* Synthetic › Noise: how a scene gets its noise, and whether the realised
   noise matches real Euclid LR. Read-only (GET /api/noise, the committed Q1
   MER level table; GET /api/population-comparison for the realised noise).

   Top to bottom: the three sentences of how a scene gets its noise; the sky
   noise level per band (the field legend on top, "294 Q1 positions in
   EDF-N/S/F", each panel's median · p5–p95); the realised background σ,
   synthetic vs real per band (the tab's summary line, a small comparison
   table and the per-field background-vs-robust-noise figure; a stale field
   statistics cache keeps its last result behind a badge, measured on Fields);
   how the bands move together (the pair switch in the card's header, the r
   badge, the 4 × 4 correlation table on demand); the depth steps inside one
   field; the measured positions on demand (the atlas layer is the map); the
   provenance footer. The How-this-is-produced drawer (`?how=1`) holds the
   scene-scale jitter switch, the vis_noise_sample step and the MER noise
   downloader (a local script: its commands, to copy). A point or row opens
   its `noisepos` inspector. */
import { useMemo } from "react";
import { Link, useNavigate } from "react-router-dom";
import Plot, { Legend, type Guide, type LegendItem, type Series } from "../../../charts/Plot";
import { C, bandColor, categorical } from "../../../colors";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import { StepById } from "../../../fasrc";
import { formatCount, formatDate, formatDateTime, formatNumber } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { linearTicks, logTicks, paddedDomain } from "../../../ticks";
import {
  Badge, Button, Caption, Card, CardBody, CardHead, CopyButton, DataTable, Num, Page, Section, Segmented, SummaryLine,
  Switch, Table, Toolbar, ToolbarSpacer, Tooltip, type Column, type DataColumn,
} from "../../../ui";
import { useNoise, usePixels, type NoisePayload } from "../api";
import { binCenters, exceedancePercent, nearestIndex2d } from "../chartKit";
import { Drawer, DrawerButton, Info, LoadState, SkyLink, Swatch, atlasHref, useDrawer, useUrlLegend } from "../common";
import { BANDS, relationDomains, relationSeries, sampleLegend, visibleFrom } from "../fields/model";
import { fieldsLabel, noiseProvenance, noiseSteps, stackedTopsByField } from "../noiseModel";
import { TOLERANCE, backgroundNoise, lastComparison } from "../statusModel";
import "../synthetic.css";

const PAIRS = [
  { value: "Y_E|J_E", label: "Y·J" }, { value: "Y_E|H_E", label: "Y·H" }, { value: "J_E|H_E", label: "J·H" },
  { value: "VIS|Y_E", label: "VIS·Y" }, { value: "VIS|J_E", label: "VIS·J" }, { value: "VIS|H_E", label: "VIS·H" },
];
const bandLabel = (band: string) => band.replace("_E", "");
const fieldColor = (index: number) => categorical(index);
const JITTER_KEY = "scene scale";
const MER_SCRIPT = "scripts/download_mer_noise_levels.py";

/** A σ at 3 significant figures, trailing zeros kept (a column reads alike: "1.50", "20.4"). */
const sigma3 = (v: number) => (Math.abs(v) >= 1000 ? formatNumber(Math.round(v)) : v.toPrecision(3));

const formatLevel = (value: number): string =>
  value >= 10 ? value.toFixed(0) : String(Number(value.toFixed(1)));
const levelTicks = (domain: [number, number]) =>
  logTicks(domain, { space: "log10", maxTicks: 7, format: formatLevel });

type Position = NoisePayload["positions"][number];

/* ── how a scene gets its noise ───────────────────────────────────────── */

function HowScenesUseIt({ payload }: { payload: NoisePayload }) {
  const [first, second, third] = noiseSteps(payload);
  return (
    <section className="syn-howto" aria-labelledby="syn-noise-how">
      <div className="rl-row">
        <h2 id="syn-noise-how" className="syn-group__title">How a scene gets its noise</h2>
        {/* A badge only on a problem: the generator ignoring the measured levels. */}
        {!payload.generator.draws_measured_levels && <Badge size="sm" tone="warn" dot>band medians only</Badge>}
      </div>
      <p className="syn-howto__text">{first} {second} {third}</p>
    </section>
  );
}

/* ── the level histograms ─────────────────────────────────────────────── */

function LevelHistogram({ band, payload, fields, hidden, jitter }: {
  band: string; payload: NoisePayload; fields: string[]; hidden: readonly string[]; jitter: boolean;
}) {
  const histogram = payload.histograms[band];
  const summary = payload.summary[band];
  const edges = histogram.log10_edges;
  const centers = binCenters(edges);
  const { tops, order } = stackedTopsByField(histogram.counts_by_field, fields, hidden, centers.length);
  const series: Series[] = tops.map((top, i) => ({
    x: centers, y: top, color: fieldColor(fields.indexOf(order[i])), mode: "histogram",
    fillAlpha: 0.72, width: 1, name: order[i], key: order[i],
  }));
  const showJitter = jitter && !hidden.includes(JITTER_KEY);
  if (showJitter) {
    series.push({ x: centers, y: histogram.jittered_counts, color: C.comb, width: 2.2, dash: [7, 4], name: "after scene scale", key: JITTER_KEY });
  }
  const yMax = 1.12 * Math.max(1, ...(tops[0] ?? []), ...(showJitter ? histogram.jittered_counts : []));
  const xDomain: [number, number] = [edges[0], edges[edges.length - 1]];
  const guides: Guide[] = [
    { axis: "x", v: Math.log10(summary.p5), dash: [3, 4], label: "p5" },
    { axis: "x", v: Math.log10(summary.median), width: 1.6, label: "median" },
    { axis: "x", v: Math.log10(summary.p95), dash: [3, 4], label: "p95" },
  ];
  return (
    <figure className="rl-fig">
      <figcaption className="rl-fig__head">
        <strong>{bandLabel(band)}</strong>
        <span className="rl-num">median {formatLevel(summary.median)} · p5–p95 {formatLevel(summary.p5)}–{formatLevel(summary.p95)}{"\u00a0"}e⁻</span>
      </figcaption>
      <Plot xDomain={xDomain} yDomain={[0, yMax]} xTicks={levelTicks(xDomain)}
        yTicks={linearTicks([0, yMax], { count: 5 })}
        xLabel="noise level (e⁻ / 0.1″ px, log)" yLabel="positions" series={series} guides={guides}
        xFormat={(v) => `${formatLevel(10 ** v)} e⁻`} aspect={0.58} exportName={`noise-levels-${bandLabel(band)}`}
        aria-label={`${bandLabel(band)} noise-level histogram by field`} />
    </figure>
  );
}

function Histograms({ payload, fields, jitter }: { payload: NoisePayload; fields: string[]; jitter: boolean }) {
  const legend = useUrlLegend("hide");
  const counts = new Map(payload.fields.map((f) => [f.name, f.positions]));
  const items: LegendItem[] = [
    ...fields.map((f, i) => ({ label: `${f} · ${formatCount(counts.get(f) ?? 0)}`, key: f, color: fieldColor(i), histogram: true, filled: true })),
    ...(jitter ? [{ label: "after scene scale", key: JITTER_KEY, color: C.comb, dash: true }] : []),
  ];
  const scale = payload.generator.scene_scale;
  const { source } = payload;
  return (
    <Card>
      <CardHead title="Sky noise level per band" sub={`${formatCount(source.position_count)} Q1 positions in ${fieldsLabel(fields)}`}
        right={<Info label="About the level histograms">
          <p>Each measured Q1 position contributes its four band levels, in {source.units}. Bars stack the fields (click a
            legend entry to hide one). {scale ? `With the jitter on (How this is produced), the dashed curve is the expected histogram after the generator's scene scale ×${scale[0]}–${scale[1]}.` : ""}</p>
          <p>{`${formatCount(source.unobserved_tiles)} of ${formatCount(source.tiles_attempted)} tiles had no coverage in all four bands.`}</p>
          <p>{source.description}</p>
        </Info>} />
      <CardBody>
        <Legend items={items} {...legend.legendProps} />
        <div className="rl-grid rl-grid--2">
          {payload.bands.map((band) => (
            <LevelHistogram key={band} band={band} payload={payload} fields={fields} hidden={legend.hidden} jitter={jitter} />
          ))}
        </div>
      </CardBody>
    </Card>
  );
}

/* ── realised noise: synthetic vs real LR ─────────────────────────────── */

function RealisedNoise() {
  const pixels = usePixels(false);
  const visible = useMemo(() => visibleFrom([]), []);
  const last = lastComparison(pixels.data);
  const comparison = last.comparison;
  if (pixels.loading && !pixels.data) return null;
  if (!comparison) {
    return (
      <Card>
        <CardHead title="Realised noise, synthetic vs real" />
        <CardBody>
          <p className="rl-note">The field statistics have not been measured yet, so the realised noise of the synthetic and real LR fields
            cannot be compared. <Link to="/synthetic/fields?view=stats">Measure them on Fields</Link>.</p>
        </CardBody>
      </Card>
    );
  }
  const rows = backgroundNoise(comparison.fields);
  const off = (ratio: number) => Math.abs(ratio - 1) > TOLERANCE.noise;
  const fields = comparison.fields;
  const series = relationSeries(fields, "median_robust_std", visible);
  const d = relationDomains(fields, "median_robust_std");
  const any = BANDS.map((b) => fields.relations.median_robust_std[b]).find(Boolean);
  const s = comparison.samples;
  const built = comparison.provenance?.generated_at;
  const columns: Column<(typeof rows)[number]>[] = [
    { header: "Band", cell: (r) => <span className="rl-legend-cell"><Swatch color={bandColor(r.band)} />{bandLabel(r.band)}</span> },
    { header: "Real σ (e⁻)", align: "right", cell: (r) => sigma3(r.real) },
    { header: "Synthetic σ (e⁻)", align: "right", cell: (r) => sigma3(r.synthetic) },
    { header: "Synthetic ÷ real", align: "right", cell: (r) => <span data-tone={off(r.ratio) ? "warn" : undefined} className="syn-ratio">{r.ratio.toFixed(2)}</span> },
  ];
  return (
    <Card>
      <CardHead title="Realised noise, synthetic vs real" sub="the background σ of the LR fields the model sees"
        right={last.stale ? (
          <Tooltip content={`The field statistics were measured ${built ? formatDateTime(built) : "earlier"} with an older schema or older inputs; re-measure them on Fields`}>
            <span tabIndex={0}><Badge size="sm" tone="warn" dot>last result</Badge></span>
          </Tooltip>
        ) : undefined} />
      <CardBody>
        {rows.length ? (
          <SummaryLine>
            Background σ, synthetic ÷ real:{" "}
            {rows.map((r, i) => (
              <span key={r.band}>{i ? " · " : ""}{bandLabel(r.band)} <Num tone={off(r.ratio) ? "warn" : undefined}>{r.ratio.toFixed(2)}</Num></span>
            ))}
          </SummaryLine>
        ) : <p className="rl-note">This field-statistics cache has no robust-σ summary.</p>}
        <div className="rl-split">
          <div className="rl-stack">
            <Table columns={columns} rows={rows} rowKey={(r) => r.band} aria-label="Background σ per band, real and synthetic" />
            <Caption>
              {`Median over fields of the robust σ (1.4826 × MAD) of each LR field · ${formatCount(s.synthetic.fields)} synthetic, `
                + `${formatCount(s.real.fields)} real fields (${formatCount(s.real.independent_parents)} pointings)`
                + ` · warn beyond ±${Math.round(100 * TOLERANCE.noise)}%${built ? ` · measured ${formatDate(built)}` : ""}`}
            </Caption>
            {last.stale && <Button asChild size="sm" variant="ghost" iconRight="chevronRight"><Link to="/synthetic/fields?view=stats">Measure on Fields</Link></Button>}
          </div>
          <figure className="rl-fig">
            <figcaption className="rl-fig__head"><strong>Background vs robust noise</strong><small>one marker per field</small></figcaption>
            <Plot xDomain={d.x} yDomain={d.y} xTicks={linearTicks(d.x, { count: 5 })} yTicks={linearTicks(d.y, { count: 5 })}
              xLabel={any?.x_label} yLabel={any?.y_label} series={series} aspect={0.7} exportName="noise-background-vs-robust"
              aria-label="Background versus robust noise per field, synthetic and real" />
            <Legend items={sampleLegend("scatter")} />
          </figure>
        </div>
      </CardBody>
    </Card>
  );
}

/* ── band pairs, depth steps, positions ───────────────────────────────── */

function BandPairs({ payload, pair, onPair }: { payload: NoisePayload; pair: string; onPair: (v: string) => void }) {
  const { bands, positions, log_correlation: corr } = payload;
  const [xBand, yBand] = pair.split("|");
  const xi = Math.max(0, bands.indexOf(xBand)), yi = Math.max(0, bands.indexOf(yBand));
  const xs = positions.map((p) => Math.log10(p.levels_e[xi]));
  const ys = positions.map((p) => Math.log10(p.levels_e[yi]));
  const series: Series[] = payload.fields.map((field, index) => {
    const rows = positions.flatMap((p, row) => (p.field === field.name ? [row] : []));
    return {
      x: rows.map((r) => xs[r]), y: rows.map((r) => ys[r]), color: fieldColor(index), mode: "scatter",
      marker: "filled", width: 1.2, alpha: 0.8, name: field.name,
    };
  });
  const xDomain = paddedDomain(xs, { pad: 0.06, minSpan: 0.02 });
  const yDomain = paddedDomain(ys, { pad: 0.06, minSpan: 0.02 });
  const pick = (point: { x: number; y: number }) => {
    const i = nearestIndex2d(xs, ys, point, { x: xDomain[1] - xDomain[0], y: yDomain[1] - yDomain[0] });
    if (i >= 0) openInspector({ kind: "noisepos", id: positions[i].tile });
  };
  const columns: Column<number>[] = [
    { header: "", cell: (r) => <strong>{bandLabel(bands[r])}</strong> },
    ...bands.map((b, c): Column<number> => ({ header: bandLabel(b), align: "right", cell: (r) => corr[r][c].toFixed(2) })),
  ];
  return (
    <Card>
      <CardHead title="How the bands move together" sub="one dot per measured position · click a dot to inspect it"
        right={(
          <span className="rl-row">
            <Segmented size="sm" aria-label="Band pair" value={pair} onChange={onPair} options={PAIRS} />
            <Badge size="sm">r = {corr[xi][yi].toFixed(2)}</Badge>
          </span>
        )} />
      <CardBody>
        <Plot xDomain={xDomain} yDomain={yDomain} xTicks={levelTicks(xDomain)} yTicks={levelTicks(yDomain)}
          xLabel={`${bandLabel(xBand)} level (e⁻, log)`} yLabel={`${bandLabel(yBand)} level (e⁻, log)`}
          series={series} legend="auto" onPlotClick={pick}
          xFormat={(v) => `${formatLevel(10 ** v)} e⁻`} yFormat={(v) => `${formatLevel(10 ** v)} e⁻`}
          aspect={0.5} exportName={`noise-pair-${bandLabel(xBand)}-${bandLabel(yBand)}`}
          aria-label={`${bandLabel(xBand)} versus ${bandLabel(yBand)} noise level`} />
        <Caption>r is the Pearson correlation of the log levels over the measured positions.</Caption>
        <Section title="Correlation of the log levels" sub="every band pair (4 × 4)" collapsible defaultOpen={false}>
          <Table columns={columns} rows={bands.map((_, r) => r)} rowKey={(r) => bands[r]} aria-label="Correlation of the log levels" />
        </Section>
      </CardBody>
    </Card>
  );
}

function WithinField({ payload }: { payload: NoisePayload }) {
  const within = payload.within_field;
  if (!within) return null;
  const region = payload.generator.region;
  const edges = within.step_edges;
  const series: Series[] = payload.bands.map((band) => ({
    x: [...edges.slice(0, -1), edges[edges.length - 1]],
    y: exceedancePercent(within.bands[band].counts, within.bands[band].fields),
    color: bandColor(band), width: 2, name: bandLabel(band),
  }));
  const xDomain: [number, number] = [edges[0], edges[edges.length - 1]];
  const pct = (v: number) => `${Math.round(100 * v)}%`;
  const columns: Column<string>[] = [
    { header: "Band", cell: (b) => <strong>{bandLabel(b)}</strong> },
    { header: "Seams", align: "right", cell: (b) => `${within.bands[b].seam_count} of ${within.bands[b].fields}` },
    { header: "Rate", align: "right", cell: (b) => pct(within.bands[b].seam_rate) },
    ...(["p50", "p90", "max"] as const).map((k): Column<string> => ({
      header: `Step ${k}`, align: "right", cell: (b) => (within.bands[b].steps ? `×${within.bands[b].steps![k].toFixed(2)}` : "—"),
    })),
  ];
  return (
    <Card>
      <CardHead title="Depth steps inside one field"
        sub={`${within.grid_side}×${within.grid_side} grid of ${within.sub_tile_arcsec}″ sub-tiles per ${within.cutout_arcsec}″ cutout`}
        right={<Info label="About the depth steps">
          The step is the largest straight-line split of a cutout's sub-tile grid (ratio of the two
          sides' median levels). A seam also needs both sides uniform within
          {" "}{Math.round(100 * (within.uniformity_threshold - 1))}%, which separates a pointing boundary from
          a bright source. The curve is an exceedance: the share of fields stepping at least that much.
          {region ? ` The shaded band is the generator's strip step ×${region.step[0]}–${region.step[1]} (${pct(region.probability)} of scenes per band).` : ""}
        </Info>} />
      <CardBody>
        <div className="rl-split">
          <Plot xDomain={xDomain} yDomain={[0, 100]}
            xTicks={linearTicks(xDomain, { count: 7, format: (v) => `×${v.toFixed(1)}` })}
            yTicks={[0, 25, 50, 75, 100].map((v) => ({ v, label: `${v}%` }))}
            xLabel="largest straight-line depth step" yLabel="fields stepping ≥ x" series={series}
            bands={region ? [{ axis: "x", from: region.step[0], to: region.step[1], color: C.comb, alpha: 0.14 }] : []}
            guides={[{ axis: "x", v: within.step_threshold, width: 1.6, dash: [4, 4], label: "seam threshold" }]}
            legend="auto" xFormat={(v) => `×${v.toFixed(3)}`} yFormat={(v) => `${v.toFixed(1)}%`}
            aspect={0.5} exportName="noise-depth-steps" aria-label="Within-field depth step exceedance per band" />
          <Table columns={columns} rows={payload.bands} rowKey={(b) => b} aria-label="Pointing seams per band" />
        </div>
      </CardBody>
    </Card>
  );
}

function Positions({ payload }: { payload: NoisePayload }) {
  const [open, setOpen] = useUrlState("npos", false);
  const columns = useMemo<DataColumn<Position>[]>(() => [
    { id: "tile", header: "Tile", cell: (p) => <span className="rl-mono">{p.tile}</span> },
    { id: "field", header: "Field" },
    { id: "ra", header: "RA", numeric: true, cell: (p) => p.ra.toFixed(4) },
    { id: "dec", header: "Dec", numeric: true, cell: (p) => p.dec.toFixed(4) },
    ...payload.bands.map((b, i): DataColumn<Position> => ({
      id: bandLabel(b), header: `${bandLabel(b)} (e⁻)`, numeric: true,
      accessor: (p) => p.levels_e[i], cell: (p) => formatNumber(p.levels_e[i], { digits: 2 }),
    })),
  ], [payload.bands]);
  return (
    <Section title="Measured positions" sub="one Q1 tile per row · the atlas layer is their map" collapsible open={open} onOpenChange={setOpen}
      right={<SkyLink layers={["q1-tiles:0.3", "noise-positions"]} hint="The measured positions on the sky atlas">On sky</SkyLink>}>
      <DataTable rows={payload.positions} columns={columns} rowKey={(p) => p.tile} aria-label="Noise positions"
        inspect={(p) => ({ kind: "noisepos", id: p.tile })} exportName="noise-positions" urlKey="np" height={360} dense />
    </Section>
  );
}

/* ── the drawer ────────────────────────────────────────────────────────── */

function HowDrawer({ payload, jitter, onJitter }: { payload: NoisePayload; jitter: boolean; onJitter: (on: boolean) => void }) {
  const scale = payload.generator.scene_scale;
  const commands = [`python ${MER_SCRIPT} plan`, `python ${MER_SCRIPT} acquire --max-minutes 55`, `python ${MER_SCRIPT} finalize`];
  return (
    <Drawer flag="how" title="How this is produced" sub="the MER noise-level table, the VIS noise samples and the scene-scale jitter">
      {scale && (
        <Switch size="sm" checked={jitter} onChange={onJitter}>
          {`Show the scene-scale jitter (×${scale[0]}–${scale[1]}) on the level histograms`}
        </Switch>
      )}
      <div className="rl-stack">
        <h3 className="syn-subtitle">MER noise-level table (local script)</h3>
        <p className="rl-note">
          Samples Euclid's Q1 MER noise maps at one random position per extragalactic tile (a scene-sized 25.6″ cutout) and
          writes <code>{payload.source.table_path}</code>. It runs on this machine from the repository root, in stages; a
          stopped run resumes. Commit the table and bump NOISE_MODEL after a re-run.
        </p>
        <ul className="syn-cmds" aria-label="MER noise downloader commands">
          {commands.map((c) => (
            <li key={c} className="syn-cmd"><code>{c}</code><CopyButton value={c} label={`Copy: ${c}`} /></li>
          ))}
        </ul>
      </div>
      <Section title="Real VIS noise fields (FASRC)" sub="vis_noise_sample" collapsible defaultOpen={false}>
        <StepById stepId="vis_noise_sample" embedded />
      </Section>
    </Drawer>
  );
}

export default function NoiseTab() {
  const resource = useNoise();
  const payload = resource.data;
  const [pair, setPair] = useUrlState("pair", PAIRS[0].value);
  const [jitter, setJitter] = useUrlState("jitter", false);
  const fields = payload?.fields.map((f) => f.name) ?? [];
  const navigate = useNavigate();
  const how = useDrawer("how");
  usePageActions([
    { id: "noise-jitter", label: jitter ? "Hide the scene-scale jitter" : "Show the scene-scale jitter", group: "Noise",
      run: () => setJitter(!jitter) },
    { id: "noise-how", label: "Noise: how this is produced (MER table, VIS noise samples)", group: "Noise", run: how.reveal },
    { id: "noise-sky", label: "Open the noise positions on the sky", group: "Noise", keywords: ["atlas"],
      run: () => navigate(atlasHref({ layers: ["q1-tiles:0.3", "noise-positions"] })) },
  ]);
  return (
    <Page className="rl-page syn-page">
      <Toolbar label="Noise controls">
        <ToolbarSpacer />
        <SkyLink layers={["q1-tiles:0.3", "noise-positions"]} hint="The measured positions on the sky atlas">Positions on sky</SkyLink>
        <DrawerButton flag="how" icon="database" hint="The MER noise-level table, the VIS noise samples and the jitter">How this is produced</DrawerButton>
      </Toolbar>
      <LoadState loading={resource.loading && !payload} error={resource.error} onRetry={resource.reload}>
        {payload && (
          <div className="rl-stack">
            <HowScenesUseIt payload={payload} />
            <Histograms payload={payload} fields={fields} jitter={jitter} />
            <RealisedNoise />
            <BandPairs payload={payload} pair={pair} onPair={setPair} />
            <WithinField payload={payload} />
            <Positions payload={payload} />
            <Caption>{noiseProvenance(payload)}</Caption>
            <HowDrawer payload={payload} jitter={jitter} onJitter={setJitter} />
          </div>
        )}
      </LoadState>
    </Page>
  );
}
