/* Spatial gate combiner card: fit it on validate, then show what it learned —
   the held-out loss trajectory and how much weight each member receives per
   band (overall, on sources, and by brightness). */
import { useEffect, useMemo, useRef, useState } from "react";
import { asArray } from "../data";
import { useJob, useTrackedJob, JobProgressView } from "../jobs";
import { C } from "../colors";
import Plot from "../charts/Plot";
import {
  Badge, Button, Card, CardBody, CardHead, DefList, Empty, Field, Input,
  NumberField, Segmented, Spinner, Table, type Column,
} from "../ui";

export const SPATIAL_GATE_KIND = "spatial_gate";

type HistoryRow = { step: number; loss: number; vis_psnr: number; train_loss?: number | null };
type GateDiagnostics = {
  available?: boolean; reason?: string; n_fields?: number; n_pixels?: number;
  brightness_names?: string[];
  usage?: Record<string, number[]>;
  usage_source?: Record<string, number[]>;
  usage_by_brightness?: Record<string, number[][]>;
};
export type SpatialGate = {
  available?: boolean; stale?: boolean; reason?: string; kind?: string;
  member_labels?: string[]; band_names?: string[]; n_parameters?: number; use_lr?: boolean;
  surviving?: { source?: boolean[] };
  gate_diagnostics?: GateDiagnostics;
  fit_meta?: {
    history?: HistoryRow[]; baseline_holdout?: HistoryRow; selected?: HistoryRow;
    best_member_per_band?: string[]; train_fields?: number[]; holdout_fields?: number[];
    blackout_fields?: number; steps?: number; fit_seconds?: number; num_images?: number;
  };
};

type UsageRow = { i: number; label: string; all: number; source: number; bins: number[] };

const pct = (v: number | undefined) => (v == null || !isFinite(v) ? "—" : `${(100 * v).toFixed(1)}%`);

export function SpatialGateCard(
  { gate, loading, mode, evalReady, targetFwhm, onFit }:
  { gate: SpatialGate | null; loading: boolean; mode: string; evalReady: boolean;
    targetFwhm: string; onFit: () => void },
) {
  const fitJob = useJob();
  const [nImg, setNImg] = useState("100");
  const [members, setMembers] = useState("");
  const [band, setBand] = useState("");
  const trackedFit = useTrackedJob(`combiner: fit ${mode} on validate`);
  const seen = useRef<string | null>(null);
  const visibleJob = fitJob.job ?? (trackedFit?.label?.includes("spatial gate") ? trackedFit : null);
  const running = visibleJob?.status === "running";

  useEffect(() => {
    if (!fitJob.job && visibleJob && visibleJob.status !== "running" && seen.current !== visibleJob.job_id) {
      seen.current = visibleJob.job_id;
      onFit();
    }
  }, [fitJob.job, onFit, visibleJob]);

  const bands = asArray<string>(gate?.band_names);
  const activeBand = band || bands[0] || "";
  const diag = gate?.gate_diagnostics;
  const binNames = asArray<string>(diag?.brightness_names);
  const history = asArray<HistoryRow>(gate?.fit_meta?.history);
  const baseline = gate?.fit_meta?.baseline_holdout;
  const selected = gate?.fit_meta?.selected;

  const rows = useMemo<UsageRow[]>(() => {
    if (!diag?.available || !activeBand) return [];
    const labels = asArray<string>(gate?.member_labels);
    return labels.map((label, i) => ({
      i, label,
      all: diag.usage?.[activeBand]?.[i] ?? NaN,
      source: diag.usage_source?.[activeBand]?.[i] ?? NaN,
      bins: asArray<number[]>(diag.usage_by_brightness?.[activeBand]).map((row) => row[i] ?? NaN),
    })).sort((a, b) => (b.source || 0) - (a.source || 0));
  }, [diag, activeBand, gate?.member_labels]);

  const columns = useMemo<Column<UsageRow>[]>(() => [
    { header: "member", cell: (r) => r.label },
    { header: "all pixels", align: "right", cell: (r) => pct(r.all) },
    { header: "source pixels", align: "right", cell: (r) => pct(r.source) },
    ...binNames.map((name, b) => ({
      header: name, align: "right" as const, cell: (r: UsageRow) => pct(r.bins[b]),
    })),
  ], [binNames]);

  const lossPlot = useMemo(() => {
    const pts = history.filter((h) => isFinite(h.loss));
    if (pts.length < 2) return null;
    const xs = pts.map((h) => h.step);
    const ys = pts.map((h) => h.loss);
    const lo = Math.min(...ys, 1), hi = Math.max(...ys, 1);
    const pad = Math.max(0.01, 0.1 * (hi - lo));
    return (
      <Plot xDomain={[0, Math.max(...xs)]} yDomain={[lo - pad, hi + pad]}
        xLabel="training step" yLabel="held-out loss (1 = best member)" height={220}
        series={[{ x: xs, y: ys, color: C.mean, mode: "line", dots: true }]}
        guides={[{ axis: "y", v: 1, label: "best member", color: "#999", dash: [4, 4] }]} />
    );
  }, [history]);

  return (
    <Card>
      <CardHead title={`Combiner · spatial gate · ${mode}`}
        sub="convolutional member gate · weighs members from each pixel's neighbourhood · convex output"
        right={gate?.available && <Badge tone={gate.stale ? "warn" : "good"}>{gate.stale ? "stale" : "fitted"}</Badge>} />
      <CardBody>
        <div className="row" style={{ alignItems: "flex-end", gap: "var(--s3)" }}>
          <NumberField label="validate fields" value={nImg} onChange={setNImg} min={2} max={2000} />
          <Field label="members (optional)">
            <Input value={members} onChange={setMembers} placeholder="all, or e.g. 170,180,184"
              style={{ width: 200 }} />
          </Field>
          <Button variant="primary" disabled={fitJob.busy || running || !evalReady}
            title={evalReady ? undefined : `evaluate ${mode} on the test set first`}
            onClick={() => fitJob.run("/ensemble/combiner/fit", {
              num_images: nImg, model_kind: SPATIAL_GATE_KIND, mode, members,
              target_psf_fwhm_arcsec: targetFwhm,
            }, { onDone: onFit })}>
            Fit spatial gate
          </Button>
        </div>
        <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
          Local fit (~20 min on CPU). The first fit also runs the members once on blackout-augmented copies of up to 40 validate fields; later fits reuse them.
          {" "}Listing members fits a pruned gate that reads only those (pick them from the usage table below) — inference then runs just those members.
        </div>
        <JobProgressView job={visibleJob} error={fitJob.error} />

        {loading ? <Empty><Spinner /> loading…</Empty>
          : !gate?.available ? <Empty>{gate?.reason ?? `no spatial gate fitted for ${mode} yet.`}</Empty> : (
          <div style={{ marginTop: "var(--s4)" }}>
            <DefList items={[
              ["members read", `${asArray<boolean>(gate.surviving?.source).filter(Boolean).length || asArray<string>(gate.member_labels).length} of ${asArray<string>(gate.member_labels).length}`],
              ["inputs", gate.use_lr ? "member neighbourhoods + LR + blackout mask" : "member neighbourhoods"],
              ["parameters", gate.n_parameters?.toLocaleString() ?? "—"],
              ["starts from", asArray<string>(gate.fit_meta?.best_member_per_band).join(" / ") || "—"],
              ["training fields", `${asArray(gate.fit_meta?.train_fields).length} (${gate.fit_meta?.blackout_fields ?? 0} blackout copies)`],
              ["held-out loss", baseline && selected ? `${baseline.loss.toFixed(3)} → ${selected.loss.toFixed(3)} (step ${selected.step})` : "—"],
              ["held-out VIS PSNR", baseline && selected ? `${baseline.vis_psnr.toFixed(3)} → ${selected.vis_psnr.toFixed(3)} dB` : "—"],
              ["fit time", gate.fit_meta?.fit_seconds != null ? `${(gate.fit_meta.fit_seconds / 60).toFixed(1)} min` : "—"],
            ]} />
            {lossPlot && <div style={{ marginTop: "var(--s4)" }}>{lossPlot}</div>}
            {diag?.available ? (
              <div style={{ marginTop: "var(--s4)" }}>
                <div className="row" style={{ justifyContent: "space-between", marginBottom: 8, gap: "var(--s3)" }}>
                  <div className="eyebrow">member usage · mean gate weight on held-out validate fields</div>
                  <Segmented<string> value={activeBand} onChange={setBand}
                    options={bands.map((b) => ({ value: b, label: b }))} />
                </div>
                <Table columns={columns} rows={rows} rowKey={(r) => r.i} />
                <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                  {diag.n_fields} fields, {diag.n_pixels?.toLocaleString()} pixels. Brightness bins use the member-mean asinh level of the selected band; source pixels have VIS level above 0.1.
                </div>
              </div>
            ) : <Empty>{diag?.reason ?? "no usage diagnostics"}</Empty>}
          </div>
        )}
      </CardBody>
    </Card>
  );
}
