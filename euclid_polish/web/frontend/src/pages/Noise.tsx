/* Noise — the sky-noise distribution synthetic scenes are drawn from.
   Each scene takes all four band levels from one measured Euclid Q1 position,
   then the generator's depth jitter. Read-only: the payload is computed from
   the committed level table, so the tab needs no FASRC connection. */
import { useState } from "react";
import Plot, { Legend, type Guide, type Series, type Tick } from "../charts/Plot";
import { C, categorical } from "../colors";
import { useResource } from "../hooks";
import {
  Badge, Card, CardBody, CardHead, Checkbox, Empty, Page, PageHead,
  Segmented, Spinner, Stat, Table, type Column,
} from "../ui";
import "./noise.css";

type Quantiles = {
  count: number; min: number; p5: number; p16: number;
  median: number; p84: number; p95: number; max: number;
};
type BandSummary = Quantiles & { pixel_scatter_ratio: number };
type Histogram = {
  log10_edges: number[];
  counts_by_field: Record<string, number[]>;
  jittered_counts: number[];
};
type Position = { field: string; tile: string; ra: number; dec: number; levels_e: number[] };
type FieldSummary = { name: string; positions: number; bands: Record<string, Quantiles> };
type NoisePayload = {
  bands: string[];
  source: {
    release: string; archive: string; description: string; units: string;
    retrieved_first: string; retrieved_last: string;
    tiles_attempted: number; position_count: number; unobserved_tiles: number;
    table_path: string;
  };
  generator: {
    noise_model: string;
    draws_measured_levels: boolean;
    scene_scale: [number, number] | null;
    region: null | { probability: number; fraction: [number, number]; step: [number, number] };
  };
  summary: Record<string, BandSummary>;
  fields: FieldSummary[];
  histograms: Record<string, Histogram>;
  log_correlation: number[][];
  positions: Position[];
};

const PAIRS = [
  { value: "Y_E|J_E", label: "Y · J" },
  { value: "Y_E|H_E", label: "Y · H" },
  { value: "J_E|H_E", label: "J · H" },
  { value: "VIS|Y_E", label: "VIS · Y" },
  { value: "VIS|J_E", label: "VIS · J" },
  { value: "VIS|H_E", label: "VIS · H" },
];

const bandLabel = (band: string) => band.replace("_E", "");
const fieldColor = (index: number) => categorical(index);

function formatLevel(value: number): string {
  return value >= 10 ? value.toFixed(0) : String(Number(value.toFixed(1)));
}

/* Round levels on a log10 axis, coarsened until at most seven fit. */
function levelTicks([lo, hi]: [number, number]): Tick[] {
  const at = (mantissas: number[]) => {
    const ticks: Tick[] = [];
    for (let exponent = Math.floor(lo) - 1; exponent <= Math.ceil(hi); exponent++) {
      for (const mantissa of mantissas) {
        const value = mantissa * 10 ** exponent;
        const x = Math.log10(value);
        if (x >= lo && x <= hi) ticks.push({ v: x, label: formatLevel(value) });
      }
    }
    return ticks;
  };
  const fine = at([1, 1.5, 2, 3, 5, 7]);
  return fine.length <= 7 ? fine : at([1, 2, 5]);
}

function countTicks(max: number): Tick[] {
  const raw = Math.max(1, max / 4);
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = Math.max(1, [1, 2, 5, 10].map((m) => m * power).find((s) => s >= raw)!);
  const ticks: Tick[] = [];
  for (let v = 0; v <= max; v += step) ticks.push({ v, label: String(v) });
  return ticks;
}

function padded(values: number[]): [number, number] {
  const lo = Math.min(...values), hi = Math.max(...values);
  const pad = Math.max(0.01, (hi - lo) * 0.06);
  return [lo - pad, hi + pad];
}

function spread(q: Quantiles, digits = 1): string {
  return `${q.median.toFixed(digits + 1)} (${q.p5.toFixed(digits)}–${q.p95.toFixed(digits)})`;
}

function LevelHistogram({ band, histogram, summary, fields, showJitter }: {
  band: string; histogram: Histogram; summary: BandSummary;
  fields: string[]; showJitter: boolean;
}) {
  const edges = histogram.log10_edges;
  const centers = edges.slice(0, -1).map((lo, bin) => (lo + edges[bin + 1]) / 2);
  // Stacked by field: the tallest cumulative layer is drawn first so every
  // field's slice stays visible above the fields below it.
  const layers: { color: string; top: number[] }[] = [];
  let total = centers.map(() => 0);
  fields.forEach((field, index) => {
    const counts = histogram.counts_by_field[field] ?? [];
    total = total.map((value, bin) => value + (counts[bin] ?? 0));
    layers.push({ color: fieldColor(index), top: total });
  });
  const series: Series[] = layers.reverse().map(({ color, top }) => ({
    x: centers, y: top, color, mode: "histogram", fillAlpha: 0.72, width: 1,
  }));
  if (showJitter) {
    series.push({
      x: centers, y: histogram.jittered_counts, color: C.comb, width: 2.2, dash: [7, 4],
    });
  }
  const yMax = 1.12 * Math.max(1, ...total, ...(showJitter ? histogram.jittered_counts : []));
  const xDomain: [number, number] = [edges[0], edges[edges.length - 1]];
  const guides: Guide[] = [
    { axis: "x", v: Math.log10(summary.p5), dash: [3, 4], label: "p5" },
    { axis: "x", v: Math.log10(summary.median), width: 1.6, label: "median" },
    { axis: "x", v: Math.log10(summary.p95), dash: [3, 4], label: "p95" },
  ];
  return (
    <div className="noise-hist">
      <div className="noise-hist__head">
        <strong>{bandLabel(band)}</strong>
        <span>median {summary.median.toFixed(2)} · p5–p95 {summary.p5.toFixed(1)}–{summary.p95.toFixed(1)} · max {summary.max.toFixed(1)}</span>
      </div>
      <Plot xDomain={xDomain} yDomain={[0, yMax]}
        xTicks={levelTicks(xDomain)} yTicks={countTicks(yMax)}
        xLabel="sky noise level (e⁻ per 0.1″ pixel, log scale)" yLabel="positions"
        series={series} guides={guides} aspect={0.55} />
    </div>
  );
}

function BandPairs({ payload }: { payload: NoisePayload }) {
  const [pair, setPair] = useState(PAIRS[0].value);
  const { bands, positions, log_correlation: correlation } = payload;
  const [xBand, yBand] = pair.split("|");
  const xi = bands.indexOf(xBand), yi = bands.indexOf(yBand);
  const xs = positions.map((position) => Math.log10(position.levels_e[xi]));
  const ys = positions.map((position) => Math.log10(position.levels_e[yi]));
  const series: Series[] = payload.fields.map((field, index) => {
    const rows = positions.flatMap((position, row) => (position.field === field.name ? [row] : []));
    return {
      x: rows.map((row) => xs[row]), y: rows.map((row) => ys[row]),
      color: fieldColor(index), mode: "scatter", marker: "filled", width: 1.2, alpha: 0.8,
    };
  });
  const xDomain = padded(xs), yDomain = padded(ys);
  const columns: Column<number>[] = [
    { header: "", cell: (row) => <strong>{bandLabel(bands[row])}</strong> },
    ...bands.map((band, col): Column<number> => ({
      header: bandLabel(band), align: "right",
      cell: (row) => correlation[row][col].toFixed(2),
    })),
  ];
  return (
    <Card className="noise-card">
      <CardHead title="How the bands move together"
        sub="Each dot is one measured position. Scenes use all four levels of one dot, so these relations carry into training."
        right={<Badge>r = {correlation[xi][yi].toFixed(2)} in log level</Badge>} />
      <CardBody>
        <div className="noise-pairs">
          <div>
            <Segmented value={pair} options={PAIRS} onChange={setPair} />
            <Plot xDomain={xDomain} yDomain={yDomain}
              xTicks={levelTicks(xDomain)} yTicks={levelTicks(yDomain)}
              xLabel={`${bandLabel(xBand)} level (e⁻, log scale)`}
              yLabel={`${bandLabel(yBand)} level (e⁻, log scale)`}
              series={series} aspect={0.62} />
          </div>
          <div>
            <div className="noise-subhead">Correlation of log levels</div>
            <Table columns={columns} rows={bands.map((_, row) => row)}
              rowKey={(row) => bands[row]} />
          </div>
        </div>
      </CardBody>
    </Card>
  );
}

function HowScenesUseIt({ payload }: { payload: NoisePayload }) {
  const { generator, source } = payload;
  const percent = (value: number) => `${Math.round(100 * value)}%`;
  const scale = generator.scene_scale;
  const region = generator.region;
  return (
    <Card className="noise-card">
      <CardHead title="How a scene gets its noise"
        right={generator.draws_measured_levels
          ? <Badge tone="good">measured levels on</Badge>
          : <Badge tone="warn">band medians only</Badge>} />
      <CardBody>
        <ol className="noise-steps">
          <li>
            <strong>Pick a sky position.</strong> One of the {source.position_count} measured
            positions, uniformly at random. All four bands take their level from it, so VIS and
            NISP depths stay paired as on the real sky.
          </li>
          <li>
            <strong>Vary the depth.</strong>{" "}
            {scale
              ? `The level is multiplied by a scene scale drawn uniformly from ${scale[0]}–${scale[1]}.`
              : "No depth jitter is applied."}
            {region && ` Per band, in ${percent(region.probability)} of scenes a straight-edged strip covering ${percent(region.fraction[0])}–${percent(region.fraction[1])} of the cutout steps ×${region.step[0]}–${region.step[1]} deeper or shallower — a pointing seam, measured in 7–12% of real Q1 fields and independent between bands.`}
          </li>
          <li>
            <strong>Draw the noise.</strong> Per pixel σ = √(level² + signal) × scale, applied to a
            unit field built like a MER stack: each band's exposures on its detector grid with
            stratified sub-pixel shifts, bilinearly resampled to 0.1″ and averaged. Sums over large
            areas match the Euclid noise map; single pixels scatter less, by the pixel ratio in
            the table below.
          </li>
        </ol>
        <p className="noise-note">
          Noise model <code>{generator.noise_model}</code> · levels from <code>{source.table_path}</code>
        </p>
      </CardBody>
    </Card>
  );
}

export default function NoisePage() {
  const [showJitter, setShowJitter] = useState(false);
  const resource = useResource<NoisePayload>("/api/noise");
  const payload = resource.data;
  if (resource.loading && !payload) {
    return <Page><Empty><Spinner /> reading the noise-level table…</Empty></Page>;
  }
  if (!payload) {
    return <Page><Empty>The noise-level table is unavailable.</Empty></Page>;
  }
  const { bands, source, summary, generator } = payload;
  const fields = payload.fields.map((field) => field.name);
  const scale = generator.scene_scale;

  const summaryColumns: Column<string>[] = [
    { header: "band", cell: (band) => <strong>{bandLabel(band)}</strong> },
    ...(["p5", "p16", "median", "p84", "p95", "max"] as const).map((key): Column<string> => ({
      header: key, align: "right", cell: (band) => summary[band][key].toFixed(2),
    })),
    { header: "pixel σ ÷ level", align: "right", cell: (band) => summary[band].pixel_scatter_ratio.toFixed(2) },
    {
      header: "median pixel σ", align: "right",
      cell: (band) => (summary[band].median * summary[band].pixel_scatter_ratio).toFixed(2),
    },
  ];
  const fieldColumns: Column<FieldSummary>[] = [
    {
      header: "field",
      cell: (field) => (
        <span className="noise-field">
          <i style={{ background: fieldColor(fields.indexOf(field.name)) }} />{field.name}
        </span>
      ),
    },
    { header: "positions", align: "right", cell: (field) => field.positions },
    ...bands.map((band): Column<FieldSummary> => ({
      header: `${bandLabel(band)} median (p5–p95)`, align: "right",
      cell: (field) => spread(field.bands[band]),
    })),
  ];

  return (
    <Page>
      <PageHead eyebrow="observation model · sky noise"
        title="Noise"
        sub="The sky noise levels synthetic scenes are drawn from, measured from Euclid Q1 MER noise maps." />

      <div className="noise-stack">
        <Card className="noise-card">
          <CardBody>
            <div className="noise-stats">
              <Stat k="measured positions" v={source.position_count} />
              <Stat k="tiles without coverage" v={`${source.unobserved_tiles} of ${source.tiles_attempted}`} />
              <Stat k="fields" v={fields.join(" · ")} />
              <Stat k="release" v={source.release} />
              <Stat k="retrieved" v={source.retrieved_last.slice(0, 10)} />
            </div>
            <p className="noise-note">{source.description}</p>
          </CardBody>
        </Card>

        <Card className="noise-card">
          <CardHead title="Sky noise level per band"
            sub={`Stacked by field. Units: ${source.units}.`}
            right={scale && (
              <Checkbox checked={showJitter} onChange={setShowJitter}>
                show with scene scale ×{scale[0]}–{scale[1]}
              </Checkbox>
            )} />
          <CardBody>
            <div className="noise-hist-grid">
              {bands.map((band) => (
                <LevelHistogram key={band} band={band} histogram={payload.histograms[band]}
                  summary={summary[band]} fields={fields} showJitter={showJitter} />
              ))}
            </div>
            <Legend items={[
              ...fields.map((field, index) => ({
                label: field, color: fieldColor(index), histogram: true, filled: true,
              })),
              ...(showJitter ? [{ label: "expected after scene scale", color: C.comb, dash: true }] : []),
            ]} />
          </CardBody>
        </Card>

        <Card className="noise-card">
          <CardHead title="Summary"
            sub="Noise-map level quantiles over all positions. Pixel σ ÷ level is the single-pixel scatter of the generated unit field." />
          <CardBody>
            <Table columns={summaryColumns} rows={bands} rowKey={(band) => band} />
          </CardBody>
        </Card>

        <Card className="noise-card">
          <CardHead title="By field" sub="Deep-field depth differs between EDF-N, EDF-S and EDF-F." />
          <CardBody>
            <Table columns={fieldColumns} rows={payload.fields} rowKey={(field) => field.name} />
          </CardBody>
        </Card>

        <BandPairs payload={payload} />
        <HowScenesUseIt payload={payload} />
      </div>
    </Page>
  );
}
