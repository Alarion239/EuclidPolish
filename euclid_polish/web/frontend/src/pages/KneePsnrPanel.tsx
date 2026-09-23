/* PSNR vs scoring knee: a knee-independent view of every model. Each curve is
   a model's PSNR when scored with a given asinh knee (the knee decides which
   brightnesses count); the integrated PSNR averages it over log(knee). */
import { useMemo, useState } from "react";
import { useResource } from "../hooks";
import { asArray } from "../data";
import { useJob, JobProgressView } from "../jobs";
import { C } from "../colors";
import Plot, { Legend, type Series, type Tick } from "../charts/Plot";
import {
  Badge, Button, Empty, Segmented, Select, Spinner, Table, type Column,
} from "../ui";

type KneeModel = {
  id: string; kind: "member" | "mean" | "combiner"; label: string;
  loss?: string | null; asinh_knee?: number | null; blocks?: number | null;
  psnr: number[][];            // [knee][band]
  integrated: number[];        // [band]
};
type KneePayload = {
  available?: boolean; stale?: boolean; reason?: string; n_fields?: number;
  knees?: number[]; bands?: string[]; models?: KneeModel[];
  integration?: { from_e: number; to_e: number };
};
type View = "relative" | "absolute";

/* Single-hue ordinal ramp by training knee: low knee light → high knee dark
   (light theme); on the dark theme the steps stay clear of the surface. */
const RAMP_LIGHT = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#1c5cab", "#184f95", "#0d366b"];
const RAMP_DARK = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6"];
const DEFAULT_KNEE = 100;
const X_TICKS: Tick[] = [
  { v: 0.1, label: "0.1" }, { v: 1, label: "1" }, { v: 10, label: "10" },
  { v: 100, label: "100" }, { v: 1000, label: "10³" }, { v: 10000, label: "10⁴" },
];
const BAND_TITLE: Record<string, string> = { VIS: "VIS", Y_E: "Y", J_E: "J", H_E: "H" };
const kneeOf = (m: KneeModel) => (m.asinh_knee == null ? DEFAULT_KNEE : Number(m.asinh_knee));
const fmt = (v: number | undefined) => (v == null || !isFinite(v) ? "—" : v.toFixed(2));

function yAxis(values: number[], view: View): { domain: [number, number]; ticks: Tick[] } {
  const finite = values.filter((v) => isFinite(v)).sort((a, b) => a - b);
  if (!finite.length) return { domain: [0, 1], ticks: [] };
  const hi = finite[finite.length - 1];
  let lo: number, step: number;
  if (view === "relative") {
    lo = Math.max(finite[Math.floor(0.03 * (finite.length - 1))], -3) - 0.2;
    step = 1;
  } else {
    lo = finite[0];
    step = hi - lo > 20 ? 10 : 5;
  }
  const top = hi + (view === "relative" ? 0.3 : 1);
  const ticks: Tick[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= top; v += step) {
    ticks.push({ v, label: v < 0 ? `−${Math.abs(v)}` : String(v) });
  }
  return { domain: [lo, top], ticks };
}

export function KneePsnrPanel(
  { mode, theme, colorOf }:
  { mode: string; theme: string; colorOf: (kind: string) => string },
) {
  const res = useResource<KneePayload>(`/ensemble/knee-psnr.json?mode=${mode}`, [mode]);
  const job = useJob();
  const [view, setView] = useState<View>("relative");
  const [atKnee, setAtKnee] = useState("none");
  const data = res.data;
  const knees = asArray<number>(data?.knees);
  const bands = asArray<string>(data?.bands);
  const models = asArray<KneeModel>(data?.models);
  const mean = models.find((m) => m.kind === "mean");
  const ramp = theme === "dark" ? RAMP_DARK : RAMP_LIGHT;

  const memberKnees = useMemo(
    () => [...new Set(models.filter((m) => m.kind === "member").map(kneeOf))].sort((a, b) => a - b),
    [models]);
  const rampColor = (knee: number) => {
    if (memberKnees.length <= 1) return ramp[4];
    const lo = Math.log10(memberKnees[0]), hi = Math.log10(memberKnees[memberKnees.length - 1]);
    const t = (Math.log10(knee) - lo) / Math.max(hi - lo, 1e-9);
    return ramp[Math.round(Math.min(1, Math.max(0, t)) * (ramp.length - 1))];
  };

  const panels = useMemo(() => bands.map((band, b) => {
    const ref = (k: number) => (view === "relative" && mean ? mean.psnr[k][b] : 0);
    const curve = (m: KneeModel) => knees.map((_, k) => m.psnr[k][b] - ref(k));
    const series: Series[] = [];
    for (const m of models.filter((x) => x.kind === "member")) {
      series.push({ x: knees, y: curve(m), color: rampColor(kneeOf(m)), width: 1,
        dash: m.loss === "l1" ? [5, 4] : undefined });
    }
    if (view === "absolute" && mean) {
      series.push({ x: knees, y: curve(mean), color: C.cross, width: 1.4, dash: [6, 4] });
    }
    for (const m of models.filter((x) => x.kind === "combiner")) {
      series.push({ x: knees, y: curve(m), color: colorOf(m.id), width: 2.4 });
    }
    const axis = yAxis(series.flatMap((s) => s.y as number[]), view);
    return { band, series, axis };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [bands, knees, models, mean, view, theme, memberKnees]);

  const legend = [
    ...memberKnees.map((k) => ({ label: `trained knee ${k}`, color: rampColor(k) })),
    { label: "L2 member", color: C.cross },
    { label: "L1 member", color: C.cross, dash: true },
    ...models.filter((m) => m.kind === "combiner").map((m) => ({ label: m.label, color: colorOf(m.id) })),
    { label: view === "relative" ? "ensemble mean (zero line)" : "ensemble mean", color: C.cross, dash: true },
  ];

  const kneeIndex = atKnee === "none" ? -1 : knees.findIndex((k) => String(k) === atKnee);
  const rows = [...models].sort((a, b) => (b.integrated[0] ?? -Infinity) - (a.integrated[0] ?? -Infinity));
  const columns: Column<KneeModel>[] = [
    { header: "model", cell: (m) => (m.kind === "combiner" ? <b>{m.label}</b> : m.label) },
    { header: "loss", cell: (m) => m.loss ?? "—" },
    { header: "trained knee", align: "right", cell: (m) => (m.kind === "member" ? kneeOf(m) : "—") },
    ...bands.map((band, b) => ({
      header: `${BAND_TITLE[band] ?? band} integrated`, align: "right" as const,
      cell: (m: KneeModel) => fmt(m.integrated[b]),
    })),
    ...(kneeIndex >= 0 ? bands.map((band, b) => ({
      header: `${BAND_TITLE[band] ?? band} @ ${knees[kneeIndex]} e⁻`, align: "right" as const,
      cell: (m: KneeModel) => fmt(m.psnr[kneeIndex][b]),
    })) : []),
  ];

  const compute = () => job.run(`/ensemble/knee-psnr?mode=${mode}`, {}, { onDone: () => res.reload() });
  if (res.loading) return <Empty><Spinner /> loading…</Empty>;
  if (!data?.available) {
    return (
      <div>
        <Empty>{data?.reason ?? "PSNR-vs-knee curves not computed yet."} They are computed after each evaluation, or now:</Empty>
        <Button variant="primary" disabled={job.busy} onClick={compute}>Compute PSNR vs knee</Button>
        <JobProgressView job={job.job} error={job.error} />
      </div>
    );
  }
  const range = data.integration;
  return (
    <div>
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 8, gap: 8, flexWrap: "wrap" }}>
        <Segmented<View> value={view} onChange={setView}
          options={[{ value: "relative", label: "relative to mean" }, { value: "absolute", label: "absolute" }]} />
        <div className="row" style={{ gap: 8, alignItems: "center" }}>
          {data.stale && <Badge tone="warn">stale — cubes or combiners changed</Badge>}
          <Button size="sm" variant="ghost" disabled={job.busy} onClick={compute}>Recompute</Button>
        </div>
      </div>
      <JobProgressView job={job.job} error={job.error} />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "var(--s3)" }}>
        {panels.map(({ band, series, axis }) => (
          <Plot key={band} title={BAND_TITLE[band] ?? band} xScale="log"
            xDomain={[knees[0], knees[knees.length - 1]]} yDomain={axis.domain}
            xTicks={X_TICKS} yTicks={axis.ticks}
            xLabel="knee used for scoring (e⁻)"
            yLabel={view === "relative" ? "PSNR − ensemble mean (dB)" : "PSNR (dB)"}
            series={series} aspect={0.72}
            guides={view === "relative" ? [{ axis: "y", v: 0, color: C.cross, dash: [6, 4] }] : []} />
        ))}
      </div>
      <Legend items={legend} />
      <div className="muted" style={{ fontSize: 12, margin: "8px 0 var(--s4)" }}>
        {data.n_fields} test fields. Each curve is a model's PSNR when the stretch uses the given asinh knee — the knee
        decides which brightnesses count. Integrated PSNR = mean over log(knee)
        from {range?.from_e ?? 0.1} to {range?.to_e ?? 10000} e⁻ (trapezoid rule).
      </div>
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 8, gap: 8 }}>
        <div className="eyebrow">integrated PSNR (dB) · sorted by VIS</div>
        <Select<string> value={atKnee} onChange={setAtKnee}
          options={[{ value: "none", label: "also show PSNR at knee…" },
            ...knees.map((k) => ({ value: String(k), label: `at ${k} e⁻` }))]} />
      </div>
      <Table columns={columns} rows={rows} rowKey={(m) => m.id} />
    </div>
  );
}
