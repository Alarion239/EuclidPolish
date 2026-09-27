/* realism/noise: the sky-noise distribution synthetic scenes are drawn from
   (the committed Euclid Q1 MER level table, GET /api/noise; read-only, works
   offline). Per-band level histograms stacked by field (+ the scene-scale
   jitter), the within-field depth steps (pointing seams), how the bands move
   together, and every measured position — a row or a scatter point opens its
   `noisepos` inspector (4×4 sub-tile grid, atlas link). The sample size is the
   histogram card's subtitle, its provenance the card's footer caption and the
   coverage gaps its info popover; each panel keeps its median · p5–p95. */
import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import Plot, { Legend, type Guide, type LegendItem, type Series } from "../../../charts/Plot";
import { C, bandColor, categorical } from "../../../colors";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import { formatCount, formatDate, formatNumber } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { linearTicks, logTicks, paddedDomain } from "../../../ticks";
import {
  Badge, Caption, Card, CardBody, CardHead, DataTable, Page, Segmented, Switch, Table,
  type Column, type DataColumn,
} from "../../../ui";
import { useNoise, type NoisePayload } from "../api";
import { binCenters, exceedancePercent, nearestIndex2d, stackedTops } from "../chartKit";
import { BarGroup, BarSpacer, Info, LoadState, RealismBar, SkyLink, atlasHref, useUrlLegend } from "../common";

const PAIRS = [
  { value: "Y_E|J_E", label: "Y·J" }, { value: "Y_E|H_E", label: "Y·H" }, { value: "J_E|H_E", label: "J·H" },
  { value: "VIS|Y_E", label: "VIS·Y" }, { value: "VIS|J_E", label: "VIS·J" }, { value: "VIS|H_E", label: "VIS·H" },
];
const bandLabel = (band: string) => band.replace("_E", "");
const fieldColor = (index: number) => categorical(index);
const JITTER_KEY = "scene scale";

const formatLevel = (value: number): string =>
  value >= 10 ? value.toFixed(0) : String(Number(value.toFixed(1)));
const levelTicks = (domain: [number, number]) =>
  logTicks(domain, { space: "log10", maxTicks: 7, format: formatLevel });

type Position = NoisePayload["positions"][number];

/** "EDF-N/S/F" for fields sharing a prefix, else the names joined. */
function fieldsLabel(names: readonly string[]): string {
  const parts = names.map((n) => { const i = n.lastIndexOf("-"); return i > 0 ? [n.slice(0, i), n.slice(i + 1)] : [n, ""]; });
  const prefix = parts[0]?.[0];
  if (names.length > 1 && parts.every(([p, rest]) => p === prefix && rest)) return `${prefix}-${parts.map(([, rest]) => rest).join("/")}`;
  return names.join(", ");
}

/** "NOISE_MODEL v5 · Q1_R1 · retrieved 2026-09-19". */
function provenanceText(payload: NoisePayload): string {
  const version = /-v(\d+)$/.exec(payload.generator.noise_model)?.[1];
  return [
    version ? `NOISE_MODEL v${version}` : `NOISE_MODEL ${payload.generator.noise_model}`,
    payload.source.release,
    `retrieved ${formatDate(payload.source.retrieved_last, { utc: true })}`,
  ].join(" · ");
}

function LevelHistogram({ band, payload, fields, hidden, jitter }: {
  band: string; payload: NoisePayload; fields: string[]; hidden: readonly string[]; jitter: boolean;
}) {
  const histogram = payload.histograms[band];
  const summary = payload.summary[band];
  const edges = histogram.log10_edges;
  const centers = binCenters(edges);
  const visible = fields.filter((f) => !hidden.includes(f));
  const tops = stackedTops(visible.map((f) => histogram.counts_by_field[f] ?? []), centers.length);
  const order = [...visible].reverse();
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
  const items: LegendItem[] = [
    ...fields.map((f, i) => ({ label: f, key: f, color: fieldColor(i), histogram: true, filled: true })),
    ...(jitter ? [{ label: "after scene scale", key: JITTER_KEY, color: C.comb, dash: true }] : []),
  ];
  const scale = payload.generator.scene_scale;
  const { source } = payload;
  return (
    <Card>
      <CardHead title="Sky noise level per band" sub={`${formatCount(source.position_count)} Q1 positions in ${fieldsLabel(fields)}`}
        right={<Info label="About the level histograms">
          <p>Each measured Q1 position contributes its four band levels, in {source.units}. Bars stack the fields (click a
            legend entry to hide one). {scale ? `The dashed curve is the expected histogram after the generator's scene scale ×${scale[0]}–${scale[1]}.` : ""}</p>
          <p>{`${formatCount(source.unobserved_tiles)} of ${formatCount(source.tiles_attempted)} tiles without coverage.`}</p>
          <p>{source.description}</p>
        </Info>} />
      <CardBody>
        <div className="rl-grid rl-grid--2">
          {payload.bands.map((band) => (
            <LevelHistogram key={band} band={band} payload={payload} fields={fields} hidden={legend.hidden} jitter={jitter} />
          ))}
        </div>
        <Legend items={items} {...legend.legendProps} />
        <Caption>{provenanceText(payload)}</Caption>
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
    { header: "band", cell: (b) => <strong>{bandLabel(b)}</strong> },
    { header: "seams", align: "right", cell: (b) => `${within.bands[b].seam_count} / ${within.bands[b].fields}` },
    { header: "rate", align: "right", cell: (b) => pct(within.bands[b].seam_rate) },
    ...(["p50", "p90", "max"] as const).map((k): Column<string> => ({
      header: `step ${k}`, align: "right", cell: (b) => (within.bands[b].steps ? `×${within.bands[b].steps![k].toFixed(2)}` : "—"),
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
          <Table columns={columns} rows={payload.bands} rowKey={(b) => b} />
        </div>
      </CardBody>
    </Card>
  );
}

function BandPairs({ payload, pair }: { payload: NoisePayload; pair: string }) {
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
        right={<Badge size="sm">r = {corr[xi][yi].toFixed(2)} (log levels)</Badge>} />
      <CardBody>
        <div className="rl-split">
          <Plot xDomain={xDomain} yDomain={yDomain} xTicks={levelTicks(xDomain)} yTicks={levelTicks(yDomain)}
            xLabel={`${bandLabel(xBand)} level (e⁻, log)`} yLabel={`${bandLabel(yBand)} level (e⁻, log)`}
            series={series} legend="auto" onPlotClick={pick}
            xFormat={(v) => `${formatLevel(10 ** v)} e⁻`} yFormat={(v) => `${formatLevel(10 ** v)} e⁻`}
            aspect={0.7} exportName={`noise-pair-${bandLabel(xBand)}-${bandLabel(yBand)}`}
            aria-label={`${bandLabel(xBand)} versus ${bandLabel(yBand)} noise level`} />
          <div>
            <div className="rl-subhead">Correlation of log levels</div>
            <Table columns={columns} rows={bands.map((_, r) => r)} rowKey={(r) => bands[r]} />
          </div>
        </div>
      </CardBody>
    </Card>
  );
}

function Positions({ payload }: { payload: NoisePayload }) {
  const columns = useMemo<DataColumn<Position>[]>(() => [
    { id: "tile", header: "Tile", cell: (p) => <span className="rl-mono">{p.tile}</span> },
    { id: "field", header: "Field" },
    { id: "ra", header: "RA", numeric: true, cell: (p) => p.ra.toFixed(4) },
    { id: "dec", header: "Dec", numeric: true, cell: (p) => p.dec.toFixed(4) },
    ...payload.bands.map((b, i): DataColumn<Position> => ({
      id: bandLabel(b), header: `${bandLabel(b)} e⁻`, numeric: true,
      accessor: (p) => p.levels_e[i], cell: (p) => formatNumber(p.levels_e[i], { digits: 2 }),
    })),
  ], [payload.bands]);
  return (
    <Card>
      <CardHead title="Measured positions" sub="one Q1 tile per row · click a row to inspect"
        right={<SkyLink layers={["q1-tiles:0.3", "noise-positions"]} hint="Every position on the sky atlas" />} />
      <CardBody>
        <DataTable rows={payload.positions} columns={columns} rowKey={(p) => p.tile} aria-label="Noise positions"
          inspect={(p) => ({ kind: "noisepos", id: p.tile })} exportName="noise-positions" urlKey="np" height={360} dense />
      </CardBody>
    </Card>
  );
}

function HowScenesUseIt({ payload }: { payload: NoisePayload }) {
  const { generator, source } = payload;
  const pct = (v: number) => `${Math.round(100 * v)}%`;
  const scale = generator.scene_scale;
  const region = generator.region;
  return (
    <Card>
      <CardHead title="How a scene gets its noise"
        right={generator.draws_measured_levels
          ? <Badge size="sm" tone="good">measured levels</Badge>
          : <Badge size="sm" tone="warn">band medians only</Badge>} />
      <CardBody>
        <ol className="rl-steps">
          <li><strong>Pick a position</strong> — one of {source.position_count}, uniformly; all four bands take its levels.</li>
          <li><strong>Vary the depth</strong> — {scale ? `scene scale ×${scale[0]}–${scale[1]}` : "no jitter"}
            {region && `; in ${pct(region.probability)} of scenes a ${pct(region.fraction[0])}–${pct(region.fraction[1])} strip steps ×${region.step[0]}–${region.step[1]}`}.</li>
          <li><strong>Draw the noise</strong> — σ = √(level² + signal) × scale on a dithered, resampled unit field.</li>
        </ol>
        <p className="rl-faint">model <code>{generator.noise_model}</code> · <code>{source.table_path}</code></p>
      </CardBody>
    </Card>
  );
}

export default function NoiseTab() {
  const resource = useNoise();
  const payload = resource.data;
  const [pair, setPair] = useUrlState("pair", PAIRS[0].value);
  const [jitter, setJitter] = useUrlState("jitter", false);
  const fields = payload?.fields.map((f) => f.name) ?? [];
  const navigate = useNavigate();
  usePageActions([
    { id: "noise-jitter", label: jitter ? "Hide the scene-scale jitter" : "Show the scene-scale jitter", group: "Noise",
      run: () => setJitter(!jitter) },
    { id: "noise-sky", label: "Open the noise positions on the sky", group: "Noise", keywords: ["atlas"],
      run: () => navigate(atlasHref({ layers: ["q1-tiles:0.3", "noise-positions"] })) },
  ]);
  return (
    <Page className="rl-page">
      <RealismBar label="Noise controls">
        <BarGroup label="Pair">
          <Segmented size="sm" aria-label="Band pair" value={pair} onChange={setPair} options={PAIRS} />
        </BarGroup>
        {payload?.generator.scene_scale && (
          <Switch size="sm" checked={jitter} onChange={setJitter}>scene-scale jitter</Switch>
        )}
        <BarSpacer />
        <SkyLink layers={["q1-tiles:0.3", "noise-positions"]} hint="The measured positions on the sky atlas" />
      </RealismBar>
      <LoadState loading={resource.loading && !payload} error={resource.error} onRetry={resource.reload}>
        {payload && (
          <div className="rl-stack">
            <Histograms payload={payload} fields={fields} jitter={jitter} />
            <BandPairs payload={payload} pair={pair} />
            <WithinField payload={payload} />
            <Positions payload={payload} />
            <HowScenesUseIt payload={payload} />
          </div>
        )}
      </LoadState>
    </Page>
  );
}
