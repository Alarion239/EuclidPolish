/* Real Euclid inference workspace.  Archive data is cached as one field and
   viewed as its fixed 10x10 grid; synthetic generation and run galleries live
   elsewhere and are intentionally not part of this page. */
import { useState } from "react";
import Plot, { Legend } from "../charts/Plot";
import { useResource } from "../hooks";
import { useJob, JobProgressView } from "../jobs";
import { CutoutViewer } from "../legacy";
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

const ticks = (lo: number, hi: number, n = 4) => Array.from({ length: n + 1 }, (_, i) => {
  const v = lo + (hi - lo) * i / n;
  return { v, label: Number.isInteger(v) ? String(v) : v.toFixed(1) };
});

function InferenceDiagnostics({ data }: { data: Diagnostics }) {
  const labels = data.member_labels.map((label) => label.replace("·psnr", ""));
  const power = data.model_power;
  const kMin = power.k[0] ?? 0.2;
  const kMax = power.k[power.k.length - 1] ?? 10;
  const kTicks = [0.2, 0.5, 1, 2, 5, 10]
    .filter((value) => value >= kMin && value <= kMax)
    .map((value) => ({ v: value, label: String(value) }));
  const std = data.std_brightness;
  return <>
    <Card>
      <CardHead title="Model–model angular cross-correlation"
        sub={`${labels.length * (labels.length - 1) / 2} member pairs · median across ${power.samples.toLocaleString()} tile-band spectra · no HR reference`} />
      <CardBody>
        <Plot title="cross-correlation rᵢⱼ(k)  ·  1 = identical Fourier structure"
          xScale="log" xDomain={[kMin, kMax]} yDomain={[-1, 1.05]}
          xTicks={kTicks}
          yTicks={[{ v: -1, label: "−1" }, { v: -0.5, label: "−0.5" }, { v: 0, label: "0" }, { v: 0.5, label: "0.5" }, { v: 1, label: "1" }]}
          xLabel={`angular frequency k [cycles / arcsec] · ${power.pixel_scale_arcsec.toFixed(2)}″ pixels`}
          yLabel="rᵢⱼ(k)" series={[
            ...power.r_pairs.map((row) => ({ x: power.k, y: row, color: "#708096", width: 0.8, alpha: 0.22 })),
            { x: power.k, y: power.r_cross, color: "#4c9ffe", width: 2.4, dash: [6, 3] },
          ]} guides={[{ axis: "y", v: 1, color: "#8c98a8", dash: [2, 3] }]} height={430} />
        <Legend items={[{ label: "individual model pairs", color: "#708096" }, { label: "median rᵢⱼ(k)", color: "#4c9ffe", dash: true }]} />
        <p className="muted" style={{ margin: "var(--s3) 0 0", fontSize: 12 }}>
          Each curve compares two cached STARFULL predictions after mean subtraction and windowed 2-D FFTs; the bands and tiles are combined by median.
        </p>
      </CardBody>
    </Card>

    <Card>
      <CardHead title="Member STD versus brightness"
        sub="Density of cached real-field pixels. Brightness and spread are in asinh-space; this is ensemble variation, not error." />
      <CardBody>
        <Plot xDomain={[std.x_edges[0], std.x_edges.at(-1)!]} yDomain={[std.y_edges[0], std.y_edges.at(-1)!]}
          xTicks={ticks(std.x_edges[0], std.x_edges.at(-1)!)} yTicks={ticks(std.y_edges[0], std.y_edges.at(-1)!)}
          xLabel={std.x_label} yLabel={std.y_label}
          heat={{ z: std.counts, xEdges: std.x_edges, yEdges: std.y_edges, colorLabel: "sampled pixels" }}
          series={[]} height={360} />
      </CardBody>
    </Card>

    {Object.entries(data.combiners).map(([kind, comb]) => {
      const title = kind === "rbf_gate" ? "max RBF"
        : kind === "stats_rbf_gate" ? "mean + std RBF"
          : kind === "stacked_rbf_gate" ? "stacked RBF" : "min + max RBF";
      if (comb.mode === "histogram") {
        const counts = comb.counts as number[];
        const centers = counts.map((_, i) => (comb.x_edges[i] + comb.x_edges[i + 1]) / 2);
        const ys = counts.map((count) => Math.log10(count + 1));
        return <Card key={kind}>
          <CardHead title={`${title} · pixel occupancy`} sub={`${comb.pixel_count.toLocaleString()} real pixels across four bands · no error or HR target`} />
          <CardBody><Plot xDomain={[comb.x_edges[0], comb.x_edges.at(-1)!]} yDomain={[0, Math.max(...ys, 1)]}
            xTicks={ticks(comb.x_edges[0], comb.x_edges.at(-1)!)} yTicks={ticks(0, Math.max(...ys, 1))}
            xLabel={comb.x_label} yLabel="log10(pixel count + 1)"
            series={[{ x: centers, y: ys, color: "#4c9ffe", width: 2 }]} height={300} /></CardBody>
        </Card>;
      }
      const counts = comb.counts as number[][];
      return <Card key={kind}>
        <CardHead title={`${title} · pixel occupancy`} sub={`${comb.pixel_count.toLocaleString()} real pixels across four bands · no error or HR target`} />
        <CardBody><Plot xDomain={[comb.x_edges[0], comb.x_edges.at(-1)!]} yDomain={[comb.y_edges![0], comb.y_edges!.at(-1)!]}
          xTicks={ticks(comb.x_edges[0], comb.x_edges.at(-1)!)} yTicks={ticks(comb.y_edges![0], comb.y_edges!.at(-1)!)}
          xLabel={comb.x_label} yLabel={comb.y_label ?? "feature"}
          heat={{ z: counts, xEdges: comb.x_edges, yEdges: comb.y_edges!, colorLabel: "pixel count" }} series={[]} height={360} /></CardBody>
      </Card>;
    })}
  </>;
}

export default function InferencePage() {
  const { data, loading, reload } = useResource<FieldStatus>("/api/inference/field.json");
  const diagnostics = useResource<{ diagnostics: Diagnostics | null }>("/api/inference/diagnostics.json");
  const [ra, setRa] = useState("267.4229");
  const [dec, setDec] = useState("64.8873");
  const job = useJob();
  const raNum = Number(ra), decNum = Number(dec);
  const valid = ra.trim() !== "" && dec.trim() !== "" && Number.isFinite(raNum)
    && Number.isFinite(decNum) && raNum >= 0 && raNum < 360 && decNum >= -90 && decNum <= 90;
  const field = data?.field ?? null;
  const cache = () => {
    if (!valid) return;
    job.run("/inference/cache-real-field", { ra: raNum, dec: decNum }, { onDone: () => { reload(); diagnostics.reload(); } });
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
              <span className="muted">STARFULL members only; fitted combiners are cached alongside the mean.</span>
            </div>
            {(job.job || job.error) && <div style={{ marginTop: "var(--s3)" }}>
              <JobProgressView job={job.job} error={job.error} />
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

        {field && diagnostics.data?.diagnostics && <InferenceDiagnostics data={diagnostics.data.diagnostics} />}
        {field && !diagnostics.loading && !diagnostics.data?.diagnostics && (
          <Card><CardHead title="Field diagnostics" sub="Rerun this exact cached field once to derive its model–model spectral and gate-occupancy plots." /></Card>
        )}
      </div>
    </Page>
  );
}
