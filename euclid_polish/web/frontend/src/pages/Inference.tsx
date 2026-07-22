/* Real Euclid inference workspace. Archive data is cached as one field and
   viewed as its fixed 10x10 grid; cached synthetic evaluation diagnostics are
   shown beside the real-field diagnostics for direct comparison. */
import { useState, type ReactNode } from "react";
import Plot, { Legend } from "../charts/Plot";
import { useResource } from "../hooks";
import { useJob, JobProgressView } from "../jobs";
import { CutoutViewer } from "../legacy";
import type { Evals } from "./Ensemble";
import {
  Badge, Button, Card, CardBody, CardHead, Empty, Field, Input, Page,
  PageHead, Spinner, Stat,
} from "../ui";

type FieldManifest = {
  field_id: string; ra: number; dec: number; field_size: number;
  tile_size: number; count: number; member_labels: string[];
  combiner_kinds: string[];
};
type FieldStatus = { field: FieldManifest | null; field_size: number };
type ModelPower = {
  k: number[]; r_pairs: (number | null)[][]; r_cross: (number | null)[];
  pair_indices: [number, number][]; samples: number; pixel_scale_arcsec: number;
};
type Diagnostics = {
  version: number; member_labels: string[]; model_power: ModelPower;
  std_brightness: { x_edges: number[]; y_edges: number[]; counts: number[][]; x_label: string; y_label: string };
  combiners: Record<string, { mode: "histogram" | "heat"; x_edges: number[]; y_edges?: number[];
    counts: number[] | number[][]; x_label: string; y_label?: string; pixel_count: number }>;
};

type ComparisonCardProps = {
  title: string; sub: string; real: ReactNode; synthetic: ReactNode;
};

const finiteEdges = (values: (number | null)[]) =>
  values.filter((value): value is number => value != null && isFinite(value));

function ComparisonCard({ title, sub, real, synthetic }: ComparisonCardProps) {
  return <Card>
    <CardHead title={title} sub={sub} />
    <CardBody>
      <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))", gap: "var(--s5)" }}>
        <div style={{ minWidth: 0 }}>
          <div className="eyebrow" style={{ marginBottom: "var(--s2)" }}>real Euclid field</div>
          {real}
        </div>
        <div style={{ minWidth: 0 }}>
          <div className="eyebrow" style={{ marginBottom: "var(--s2)" }}>synthetic STARFULL evaluation</div>
          {synthetic}
        </div>
      </div>
    </CardBody>
  </Card>;
}

function MissingSyntheticPlot({ loading, error }: { loading: boolean; error: boolean }) {
  return <p className="muted" style={{ margin: "var(--s4) 0", fontSize: 12 }}>
    {loading ? "Loading cached synthetic evaluation…"
      : error ? "Synthetic evaluation data is unavailable."
        : "This synthetic evaluation does not contain an analogous plot."}
  </p>;
}

function transformedSeries(valuesX: (number | null)[], valuesY: (number | null)[], transform: (value: number) => number) {
  return valuesX.map((value, index) => ({
    x: value == null || !isFinite(value) || value <= 0 ? NaN : transform(value),
    y: valuesY[index] ?? null,
  })).filter((point) => isFinite(point.x) && point.y != null && isFinite(point.y))
    .sort((a, b) => a.x - b.x);
}

function sharedDomain(...arrays: Array<readonly (number | null)[]>): [number, number] {
  const values = arrays.flatMap((array) => array.filter(
    (value): value is number => value != null && isFinite(value),
  ));
  if (!values.length) return [0, 1];
  const lo = Math.min(...values), hi = Math.max(...values);
  return lo === hi ? [lo, lo + 1] : [lo, hi];
}

function displayEdges(domain: [number, number], bins: number) {
  return Array.from({ length: bins + 1 }, (_, index) =>
    domain[0] + (domain[1] - domain[0]) * index / bins);
}

function edgeBin(edges: number[], value: number) {
  if (value < edges[0] || value > edges[edges.length - 1]) return -1;
  if (value === edges[edges.length - 1]) return edges.length - 2;
  const index = Math.floor((value - edges[0]) / (edges[1] - edges[0]));
  return index >= 0 && index < edges.length - 1 ? index : -1;
}

function rebinHeat(z: number[][], sourceXEdges: number[], sourceYEdges: number[], targetXEdges: number[], targetYEdges: number[]) {
  const output = Array.from({ length: targetXEdges.length - 1 }, () =>
    Array(targetYEdges.length - 1).fill(0));
  for (let i = 0; i < z.length; i++) {
    const x = (sourceXEdges[i] + sourceXEdges[i + 1]) / 2;
    const targetI = edgeBin(targetXEdges, x);
    if (targetI < 0) continue;
    for (let j = 0; j < z[i].length; j++) {
      const count = z[i][j];
      const y = (sourceYEdges[j] + sourceYEdges[j + 1]) / 2;
      const targetJ = edgeBin(targetYEdges, y);
      if (targetJ >= 0 && isFinite(count)) output[targetI][targetJ] += count;
    }
  }
  return output;
}

const CROSS_X_DOMAIN: [number, number] = [0.05, 10];
const CROSS_X_TICKS = [
  { v: 0.05, label: "0.05" }, { v: 0.1, label: "0.1" }, { v: 0.2, label: "0.2" },
  { v: 0.5, label: "0.5" }, { v: 1, label: "1" }, { v: 2, label: "2" },
  { v: 5, label: "5" }, { v: 10, label: "10" },
];
const CROSS_Y_DOMAIN: [number, number] = [0, 1.05];
const CROSS_Y_TICKS = [
  { v: 0, label: "0" }, { v: 0.5, label: "0.5" }, { v: 1, label: "1" },
];

function SyntheticPowerPlot({ evals }: { evals: Evals }) {
  const power = evals.ps;
  if (!power?.theta?.length || !power.r_cross?.length) return null;
  const cross = transformedSeries(power.theta, power.r_cross, (value) => value);
  if (!cross.length) return null;
  const pairs = (power.r_pairs ?? []).map((row) => transformedSeries(power.theta, row, (value) => value));
  return <>
    <Plot title="cross-correlation rᵢⱼ(d)  ·  synthetic test fields"
      xScale="log" xDomain={CROSS_X_DOMAIN} yDomain={CROSS_Y_DOMAIN} xTicks={CROSS_X_TICKS}
      yTicks={CROSS_Y_TICKS}
      xLabel="angular distance d [arcsec]"
      yLabel="rᵢⱼ(d)" series={[
        ...pairs.filter((row) => row.length).map((row) => ({ x: row.map((point) => point.x), y: row.map((point) => point.y), color: "#708096", width: 0.8, alpha: 0.22 })),
        { x: cross.map((point) => point.x), y: cross.map((point) => point.y), color: "#d47f34", width: 2.4, dash: [6, 3] },
      ]} guides={[{ axis: "y", v: 1, color: "#8c98a8", dash: [2, 3] }]} height={430} />
    <Legend items={[{ label: "individual model pairs", color: "#708096" }, { label: "median rᵢⱼ(d)", color: "#d47f34", dash: true }]} />
  </>;
}

function SyntheticBrightnessPlot({ evals, xDomain, yDomain, xTicks, yTicks, xEdges: targetXEdges, yEdges: targetYEdges }: {
  evals: Evals; xDomain: [number, number]; yDomain: [number, number];
  xTicks: { v: number; label: string }[]; yTicks: { v: number; label: string }[];
  xEdges: number[]; yEdges: number[];
}) {
  const bright = evals.bright_std;
  const sourceXEdges = finiteEdges(bright?.bright_edges ?? []);
  const sourceYEdges = finiteEdges(bright?.std_edges ?? []);
  if (!bright || sourceXEdges.length < 2 || sourceYEdges.length < 2 || !bright.hist.length) return null;
  return <Plot title="member STD versus brightness  ·  synthetic test fields"
    xDomain={xDomain} yDomain={yDomain} xTicks={xTicks} yTicks={yTicks}
    xLabel="mean brightness (asinh)" yLabel="log10(member std)"
    heat={{ z: rebinHeat(bright.hist, sourceXEdges, sourceYEdges, targetXEdges, targetYEdges), xEdges: targetXEdges, yEdges: targetYEdges, colorLabel: "synthetic pixels" }} series={[]} height={360} />;
}

function SyntheticCombinerPlot({ evals, kind, xDomain, yDomain, xTicks, yTicks, xEdges: targetXEdges, yEdges: targetYEdges }: {
  evals: Evals; kind: string; xDomain: [number, number]; yDomain: [number, number];
  xTicks: { v: number; label: string }[]; yTicks: { v: number; label: string }[];
  xEdges: number[]; yEdges: number[];
}) {
  const axis = evals.combiner_feature_error?.axes?.min_max;
  const model = axis?.models?.[kind];
  const sourceXEdges = finiteEdges(axis?.edges?.[0] ?? []);
  const sourceYEdges = finiteEdges(axis?.edges?.[1] ?? []);
  if (!model || sourceXEdges.length < 2 || sourceYEdges.length < 2 || !model.counts.length) return null;
  return <Plot title="combiner feature occupancy  ·  synthetic test fields"
    xDomain={xDomain} yDomain={yDomain} xTicks={xTicks} yTicks={yTicks}
    xLabel={axis?.axis_names?.[0] ?? "min member"} yLabel={axis?.axis_names?.[1] ?? "max member"}
    heat={{ z: rebinHeat(model.counts, sourceXEdges, sourceYEdges, targetXEdges, targetYEdges), xEdges: targetXEdges, yEdges: targetYEdges, colorLabel: "synthetic pixels" }} series={[]} height={360} />;
}

const ticks = (lo: number, hi: number, n = 4) => Array.from({ length: n + 1 }, (_, i) => {
  const v = lo + (hi - lo) * i / n;
  return { v, label: Number.isInteger(v) ? String(v) : v.toFixed(1) };
});

function InferenceDiagnostics({ data, synthetic, syntheticLoading, syntheticError }: {
  data: Diagnostics; synthetic: Evals | null; syntheticLoading: boolean; syntheticError: boolean;
}) {
  const labels = data.member_labels.map((label) => label.replace("·psnr", ""));
  const power = data.model_power;
  const std = data.std_brightness;
  const syntheticBright = synthetic?.bright_std;
  const brightnessXDomain = sharedDomain(std.x_edges, syntheticBright?.bright_edges ?? []);
  const brightnessYDomain = sharedDomain(std.y_edges, syntheticBright?.std_edges ?? []);
  const brightnessXTicks = ticks(brightnessXDomain[0], brightnessXDomain[1]);
  const brightnessYTicks = ticks(brightnessYDomain[0], brightnessYDomain[1]);
  const brightnessXEdges = displayEdges(brightnessXDomain,
    Math.max(std.x_edges.length - 1, (syntheticBright?.bright_edges.length ?? 2) - 1));
  const brightnessYEdges = displayEdges(brightnessYDomain,
    Math.max(std.y_edges.length - 1, (syntheticBright?.std_edges.length ?? 2) - 1));
  const syntheticAxis = synthetic?.combiner_feature_error?.axes?.min_max;
  return <>
    <ComparisonCard title="Model–model angular cross-correlation"
      sub={`${labels.length * (labels.length - 1) / 2} real-field member pairs · no HR reference · matched to synthetic STARFULL evaluation`}
      real={<>
        <Plot title="cross-correlation rᵢⱼ(d)  ·  1 = identical Fourier structure"
          xScale="log" xDomain={CROSS_X_DOMAIN} yDomain={CROSS_Y_DOMAIN}
          xTicks={CROSS_X_TICKS}
          yTicks={CROSS_Y_TICKS}
          xLabel={`angular distance d [arcsec] · ${power.pixel_scale_arcsec.toFixed(2)}″ pixels`}
          yLabel="rᵢⱼ(d)" series={[
            ...power.r_pairs.map((row) => { const points = transformedSeries(power.k, row, (value) => 1 / value); return { x: points.map((point) => point.x), y: points.map((point) => point.y), color: "#708096", width: 0.8, alpha: 0.22 }; }),
            (() => { const points = transformedSeries(power.k, power.r_cross, (value) => 1 / value); return { x: points.map((point) => point.x), y: points.map((point) => point.y), color: "#4c9ffe", width: 2.4, dash: [6, 3] }; })(),
          ]} guides={[{ axis: "y", v: 1, color: "#8c98a8", dash: [2, 3] }]} height={430} />
        <Legend items={[{ label: "individual model pairs", color: "#708096" }, { label: "median rᵢⱼ(d)", color: "#4c9ffe", dash: true }]} />
        <p className="muted" style={{ margin: "var(--s3) 0 0", fontSize: 12 }}>
          Each curve compares two cached STARFULL predictions after mean subtraction and windowed 2-D FFTs; the bands and tiles are combined by median.
        </p>
      </>}
      synthetic={synthetic ? <SyntheticPowerPlot evals={synthetic} />
        : <MissingSyntheticPlot loading={syntheticLoading} error={syntheticError} />} />

    <ComparisonCard title="Member STD versus brightness"
      sub="Density of pixels in each domain. Brightness and spread are in asinh-space; this is ensemble variation, not error."
      real={<>
        <Plot xDomain={brightnessXDomain} yDomain={brightnessYDomain}
          xTicks={brightnessXTicks} yTicks={brightnessYTicks}
          xLabel={std.x_label} yLabel={std.y_label}
          heat={{ z: rebinHeat(std.counts, std.x_edges, std.y_edges, brightnessXEdges, brightnessYEdges), xEdges: brightnessXEdges, yEdges: brightnessYEdges, colorLabel: "sampled pixels" }}
          series={[]} height={360} />
      </>}
      synthetic={synthetic ? <SyntheticBrightnessPlot evals={synthetic}
        xDomain={brightnessXDomain} yDomain={brightnessYDomain}
        xTicks={brightnessXTicks} yTicks={brightnessYTicks}
        xEdges={brightnessXEdges} yEdges={brightnessYEdges} />
        : <MissingSyntheticPlot loading={syntheticLoading} error={syntheticError} />} />

    {Object.entries(data.combiners)
      .filter(([kind]) => kind === "raw_incremental_minmeanmax_rbf")
      .map(([kind, comb]) => {
      const title = "minibatched convex all-asinh RBF";
      const combinerXDomain = sharedDomain(comb.x_edges, syntheticAxis?.edges?.[0] ?? []);
      const combinerYDomain = sharedDomain(comb.y_edges ?? [], syntheticAxis?.edges?.[1] ?? []);
      const combinerXTicks = ticks(combinerXDomain[0], combinerXDomain[1]);
      const combinerYTicks = ticks(combinerYDomain[0], combinerYDomain[1]);
      const combinerXEdges = displayEdges(combinerXDomain,
        Math.max(comb.x_edges.length - 1, (syntheticAxis?.edges?.[0]?.length ?? 2) - 1));
      const combinerYEdges = displayEdges(combinerYDomain,
        Math.max(comb.y_edges?.length ? comb.y_edges.length - 1 : 1, (syntheticAxis?.edges?.[1]?.length ?? 2) - 1));
      if (comb.mode === "histogram") {
        const counts = comb.counts as number[];
        const centers = counts.map((_, i) => (comb.x_edges[i] + comb.x_edges[i + 1]) / 2);
        const ys = counts.map((count) => Math.log10(count + 1));
        return <ComparisonCard key={kind} title={`${title} · pixel occupancy`}
          sub={`${comb.pixel_count.toLocaleString()} real pixels across four bands · no error or HR target · matched to synthetic gate occupancy`}
          real={<Plot xDomain={[comb.x_edges[0], comb.x_edges[comb.x_edges.length - 1]]} yDomain={[0, Math.max(...ys, 1)]}
            xTicks={ticks(comb.x_edges[0], comb.x_edges[comb.x_edges.length - 1])} yTicks={ticks(0, Math.max(...ys, 1))}
            xLabel={comb.x_label} yLabel="log10(pixel count + 1)"
            series={[{ x: centers, y: ys, color: "#4c9ffe", width: 2 }]} height={300} />}
          synthetic={synthetic ? <SyntheticCombinerPlot evals={synthetic} kind={kind}
            xDomain={combinerXDomain} yDomain={combinerYDomain}
            xTicks={combinerXTicks} yTicks={combinerYTicks}
            xEdges={combinerXEdges} yEdges={combinerYEdges} />
            : <MissingSyntheticPlot loading={syntheticLoading} error={syntheticError} />} />;
      }
      const counts = comb.counts as number[][];
      return <ComparisonCard key={kind} title={`${title} · pixel occupancy`}
        sub={`${comb.pixel_count.toLocaleString()} real pixels across four bands · no error or HR target · matched to synthetic gate occupancy`}
        real={<Plot xDomain={combinerXDomain} yDomain={combinerYDomain}
          xTicks={combinerXTicks} yTicks={combinerYTicks}
          xLabel={comb.x_label} yLabel={comb.y_label ?? "feature"}
          heat={{ z: rebinHeat(counts, comb.x_edges, comb.y_edges!, combinerXEdges, combinerYEdges), xEdges: combinerXEdges, yEdges: combinerYEdges, colorLabel: "real pixel count" }} series={[]} height={360} />}
        synthetic={synthetic ? <SyntheticCombinerPlot evals={synthetic} kind={kind}
          xDomain={combinerXDomain} yDomain={combinerYDomain}
          xTicks={combinerXTicks} yTicks={combinerYTicks}
          xEdges={combinerXEdges} yEdges={combinerYEdges} />
          : <MissingSyntheticPlot loading={syntheticLoading} error={syntheticError} />} />;
    })}
  </>;
}

export default function InferencePage() {
  const { data, loading, reload } = useResource<FieldStatus>("/api/inference/field.json");
  const diagnostics = useResource<{ diagnostics: Diagnostics | null }>("/api/inference/diagnostics.json");
  const synthetic = useResource<Evals>("/ensemble/evals.json?mode=starfull");
  const [ra, setRa] = useState("267.4229");
  const [dec, setDec] = useState("64.8873");
  const job = useJob();
  const refreshJob = useJob();
  const raNum = Number(ra), decNum = Number(dec);
  const valid = ra.trim() !== "" && dec.trim() !== "" && Number.isFinite(raNum)
    && Number.isFinite(decNum) && raNum >= 0 && raNum < 360 && decNum >= -90 && decNum <= 90;
  const field = data?.field ?? null;
  const cache = () => {
    if (!valid) return;
    job.run("/inference/cache-real-field", { ra: raNum, dec: decNum }, { onDone: () => { reload(); diagnostics.reload(); } });
  };
  const refreshCombiners = () => {
    if (!field) return;
    refreshJob.run("/inference/refresh-combiners", {}, {
      onDone: () => { reload(); diagnostics.reload(); },
    });
  };

  return (
    <Page>
      <PageHead
        eyebrow="data · real Euclid"
        title="Inference"
        sub="A field workspace: download one 2560 × 2560 archive cutout, cache 100 tiles for every active STARFULL member and fitted combiner, then inspect them directly."
        right={field ? <Badge tone="good">{field.count} cached tiles</Badge> : <Badge tone="warn">no field cached</Badge>}
      />

      <div className="grid" style={{ gridTemplateColumns: "1fr", gap: "var(--s4)" }}>
        <Card>
          <CardHead
            title="Field centre"
            sub="The initial coordinates point at a Euclid field. Archive cutouts and derived cubes are reused when you return to the same centre."
          />
          <CardBody>
            <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "var(--s3)" }}>
              <Field label="RA (deg)"><Input value={ra} onChange={setRa} onEnter={cache} /></Field>
              <Field label="Dec (deg)"><Input value={dec} onChange={setDec} onEnter={cache} /></Field>
              <Stat k="archive field" v={`${data?.field_size ?? 2560} px`} />
              <Stat k="viewer tiles" v="100 × 256 px" />
            </div>
            {!valid && <p className="muted" style={{ marginTop: "var(--s2)", fontSize: 12 }}>
              Enter RA in [0, 360) and Dec in [-90, 90].
            </p>}
            <div className="row" style={{ marginTop: "var(--s3)", gap: "var(--s3)", alignItems: "center" }}>
              <Button variant="primary" disabled={!valid || job.busy} onClick={cache}>
                Download field & cache cubes
              </Button>
              <Button disabled={!field || refreshJob.busy || job.busy}
                onClick={refreshCombiners}>
                Refresh inference with latest combiner
              </Button>
              <span className="muted">STARFULL only; cached members are reused when current, otherwise rebuilt from the cached archive field.</span>
            </div>
            {(job.job || job.error) && <div style={{ marginTop: "var(--s3)" }}>
              <JobProgressView job={job.job} error={job.error} />
            </div>}
            {(refreshJob.job || refreshJob.error) && <div style={{ marginTop: "var(--s3)" }}>
              <JobProgressView job={refreshJob.job} error={refreshJob.error} />
            </div>}
          </CardBody>
        </Card>

        <Card>
          <CardHead
            title="Tile viewer"
            sub={field
              ? `RA ${field.ra.toFixed(5)}, Dec ${field.dec.toFixed(5)} · ${field.member_labels.length} STARFULL members · ${field.combiner_kinds.length} fitted combiners`
              : "Cache a field above to enable LR, mean SR, member disagreement, and combiner reconstructions."}
            right={<Button size="sm" variant="ghost" onClick={reload}>↻</Button>}
          />
          <CardBody>
            {loading ? <Empty><Spinner /> loading field cache…</Empty>
              : field ? <CutoutViewer key={`${field.field_id}:${field.combiner_kinds.join(",")}`} collection="real-field" params={{ field: field.field_id }} />
                : <Empty>No real Euclid field is cached yet.</Empty>}
          </CardBody>
        </Card>

        {field && diagnostics.data?.diagnostics && <InferenceDiagnostics data={diagnostics.data.diagnostics}
          synthetic={synthetic.data} syntheticLoading={synthetic.loading} syntheticError={synthetic.error} />}
        {field && !diagnostics.loading && !diagnostics.data?.diagnostics && (
          <Card><CardHead title="Field diagnostics" sub="Rerun this exact cached field once to derive its model–model spectral and gate-occupancy plots." /></Card>
        )}
      </div>
    </Page>
  );
}
