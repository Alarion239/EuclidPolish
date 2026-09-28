/* Inspector kind `member` (`member:member_196`, also `196` / `196·psnr`):
   one ensemble member — status and recipe (origin.json), its training curves
   (per-band PSNR, loss, gradient norm, step time), its PSNR-vs-knee curve
   against the plain mean and the production gate, and how much weight the
   production gate gives it (overall and by brightness). Active and archived
   members (restore from the archive zip). GET /ensemble/member/<name>.json. */
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import Plot, { type Series } from "../../charts/Plot";
import { C, bandColor, categorical } from "../../colors";
import { useJob } from "../../api/jobs";
import { useResource } from "../../api/query";
import { openInspector, type InspectorProps } from "../../app/inspector";
import { pagePath } from "../../app/nav";
import { formatDateTime, formatDuration, formatRelative } from "../../format";
import { logTicks } from "../../ticks";
import {
  Badge, Button, DefList, EmptyState, JobProgress, JsonTree, Section, Segmented, Skeleton, confirm,
} from "../../ui";
import { BAND_SHORT, BANDS, url, type MemberDetail } from "./api";
import { JOB, useOnJobEnd } from "./jobs";
import { db, dbDelta, kneeText, memberName, memberNumber, relativeTo, stepsText, xy } from "./model";
import "./models.css";

type CurveMetric = "bands" | "loss" | "gnorm" | "time";
const kfmt = (v: number) => (Math.abs(v) >= 1000 ? `${+(v / 1000).toFixed(1)}k` : String(Math.round(v)));

function CurvesPlot({ d }: { d: MemberDetail }) {
  const [metric, setMetric] = useState<CurveMetric>("bands");
  const c = d.curves;
  if (!c) return <EmptyState compact icon="activity" title="No training log" />;
  let series: Series[];
  let yLabel = "PSNR [dB]";
  let log = false;
  if (metric === "bands") {
    series = [
      { ...xy(c.psnr), color: C.mean, width: 2.2, name: "joint (asinh)" },
      ...BANDS.map((b) => ({ ...xy(c.band_psnr?.[b]), color: bandColor(b), width: 1.3, name: BAND_SHORT[b] })),
    ];
  } else if (metric === "loss") {
    series = [{ ...xy(c.loss_series), color: C.comb, width: 1.8, name: "combined loss" },
      { ...xy(c.train_loss), color: C.muted, width: 1.2, name: "loss column" }];
    yLabel = "loss"; log = true;
  } else if (metric === "gnorm") {
    series = [{ ...xy(c.gnorm), color: categorical(3), width: 1.8, name: "mean ‖g‖" },
      { ...xy(c.gnorm_max), color: categorical(3), width: 1, alpha: 0.5, dash: [4, 3], name: "max ‖g‖" }];
    yLabel = "gradient norm"; log = true;
  } else {
    series = [{ ...xy(c.step_time), color: categorical(1), width: 1.8, name: "s / 1k steps" }];
    yLabel = "s / 1000 steps";
  }
  const ys = series.flatMap((s) => s.y.filter((v): v is number => v != null && Number.isFinite(v) && (!log || v > 0)));
  const xs = series.flatMap((s) => s.x);
  const lo = Math.min(...ys), hi = Math.max(...ys);
  const yDomain: [number, number] = !ys.length ? [0, 1] : log ? [lo / 1.2, hi * 1.2] : [lo - (hi - lo) * 0.06 - 0.01, hi + (hi - lo) * 0.06 + 0.01];
  return (
    <div className="mdl-stack" style={{ gap: "var(--s2)" }}>
      <Segmented<CurveMetric> size="sm" aria-label="Curve" value={metric} onChange={setMetric}
        options={[{ value: "bands", label: "PSNR" }, { value: "loss", label: "loss" }, { value: "gnorm", label: "‖g‖" }, { value: "time", label: "time" }]} />
      <Plot xDomain={[0, Math.max(1, ...xs)]} yDomain={yDomain} yScale={log ? "log" : "linear"} series={series}
        xLabel="step" yLabel={yLabel} aspect={0.62} legend="auto" xFormat={kfmt} zoomAxes="x"
        exportName={`${d.name}-${metric}`} aria-label={`${d.name} training ${metric}`} />
    </div>
  );
}

function KneePlot({ d }: { d: MemberDetail }) {
  const [band, setBand] = useState(0);
  const k = d.knee;
  const me = k?.models.find((m) => m.kind === "member" && m.label === d.label);
  if (!k || !me) return <EmptyState compact icon="activity" title="No PSNR-vs-knee curve" />;
  const mean = k.models.find((m) => m.kind === "mean");
  const gate = k.models.find((m) => m.id === "spatial_gate");
  const rel = (p: number[][]) => relativeTo(p, mean?.psnr).map((r) => r[band]);
  const series: Series[] = [
    { x: k.knees, y: rel(me.psnr), color: categorical(0), width: 2.4, dots: true, name: `#${memberNumber(d.name)}` },
    ...(gate ? [{ x: k.knees, y: rel(gate.psnr), color: C.comb, width: 2, name: "production gate" }] : []),
    { x: k.knees, y: k.knees.map(() => 0), color: C.mean, width: 1.2, dash: [4, 3], name: "plain mean" },
  ];
  const ys = series.flatMap((s) => s.y.filter((v): v is number => v != null && Number.isFinite(v)));
  const lo = Math.min(...ys, 0), hi = Math.max(...ys, 0);
  const pad = (hi - lo) * 0.1 || 0.3;
  return (
    <div className="mdl-stack" style={{ gap: "var(--s2)" }}>
      <Segmented size="sm" aria-label="Band" value={String(band)} onChange={(v) => setBand(Number(v))}
        options={k.bands.map((b, i) => ({ value: String(i), label: BAND_SHORT[b] ?? b }))} />
      <Plot xScale="log" xDomain={[k.knees[0], k.knees[k.knees.length - 1]]} xTicks={logTicks([k.knees[0], k.knees[k.knees.length - 1]])}
        yDomain={[lo - pad, hi + pad]} series={series} xLabel="scoring knee [e⁻]" yLabel="PSNR − mean [dB]"
        aspect={0.62} legend="auto" exportName={`${d.name}-knee`} aria-label={`${d.name} PSNR vs knee`} />
      {k.stale && <Badge tone="warn">knee curves stale</Badge>}
    </div>
  );
}

function GatePlot({ d }: { d: MemberDetail }) {
  const g = d.gate;
  if (!g) return <EmptyState compact icon="layers" title="Not read by the production gate" />;
  const names = g.brightness_names;
  const series: Series[] = [
    ...g.bands.map((b) => ({ x: names.map((_, i) => i), y: (g.by_brightness[b] ?? []).map((v) => v ?? null), color: bandColor(b),
      width: 1.8, dots: true, name: BAND_SHORT[b] ?? b })),
    { x: [0, names.length - 1], y: [g.uniform, g.uniform], color: C.guide, dash: [4, 3], width: 1.2, name: "uniform share" },
  ];
  const ys = series.flatMap((s) => s.y.filter((v): v is number => v != null && Number.isFinite(v)));
  return (
    <div className="mdl-stack" style={{ gap: "var(--s2)" }}>
      <DefList dense items={g.bands.map((b) => [`${BAND_SHORT[b] ?? b} weight`,
        `${g.usage[b] != null ? (100 * (g.usage[b] as number)).toFixed(2) : "—"}% · sources ${g.usage_source[b] != null ? (100 * (g.usage_source[b] as number)).toFixed(2) : "—"}%`])} />
      <Plot xDomain={[-0.3, names.length - 0.7]} yDomain={[0, Math.max(g.uniform * 1.5, ...ys) * 1.1]}
        xTicks={names.map((n, i) => ({ v: i, label: n }))} series={series} xLabel="member-mean brightness" yLabel="mean gate weight"
        aspect={0.6} legend="auto" zoom={false} exportName={`${d.name}-gate`} aria-label={`${d.name} gate weight by brightness`} />
      {g.stale && <Badge tone="warn">from a stale gate</Badge>}
    </div>
  );
}

export default function MemberInspector({ id }: InspectorProps) {
  const name = memberName(id) ?? id;
  const res = useResource<MemberDetail>(url.member(name), [name], { ttl: 30_000 });
  const navigate = useNavigate();
  const restore = useJob(JOB.restore);
  const archive = useJob(JOB.archive);
  useOnJobEnd(restore.job, () => void res.reload());
  useOnJobEnd(archive.job, () => void res.reload());
  if (res.loading) return <Skeleton lines={6} />;
  if (res.error || !res.data) {
    return <EmptyState icon="warn" title={res.error?.status === 404 ? `No member ${name}` : "Could not load the member"}>
      <span className="mdl-mono">{res.error?.message}</span></EmptyState>;
  }
  const d = res.data;
  const r = d.row;
  const num = memberNumber(d.name);
  const tab = (t: string, q = "") => navigate(`${pagePath("models", { tab: t, params: { mode: d.regime } })}${q}`);
  const k = r ? kneeText(r) : null;
  const meanModel = d.knee?.models.find((x) => x.kind === "mean");
  const meanVals = (meanModel?.integrated ?? []).filter((x) => Number.isFinite(x));
  const meanKnee = meanVals.length ? meanVals.reduce((a, b) => a + b, 0) / meanVals.length : null;
  return (
    <div className="mdl-insp">
      <div className="mdl-insp__head">
        <span className="mdl-insp__title">#{num}</span>
        <Badge>{d.regime}</Badge>
        {!d.active && <Badge tone="warn">archived</Badge>}
        {r?.status === "timeout" && <Badge tone="warn">TIMEOUT</Badge>}
        {r?.status === "running" && <Badge tone="info" dot>training</Badge>}
        {r && <Badge>{r.loss.toUpperCase()}</Badge>}
        {k && <Badge tone={k.kind === "multi" ? "accent" : undefined}>{k.text}</Badge>}
      </div>
      {d.active && (
        <div className="mdl-row">
          <Button size="sm" onClick={() => tab("train", `?mode=continue&members=${d.name}`)}>Continue</Button>
          <Button size="sm" onClick={() => tab("train", `?mode=fork&member=${d.name}`)}>Fork</Button>
          <Button size="sm" variant="ghost" onClick={() => tab("members", `?view=curves&sel=${num}`)}>Curves</Button>
          <Button size="sm" variant="ghost" onClick={() => tab("images", `?sel=${num}`)}>Viewer</Button>
          <Button size="sm" variant="danger" loading={archive.busy} onClick={async () => {
            if (await confirm({ title: `Archive ${d.name}?`, message: "Zipped to the tracking campaign, tombstoned and deleted (also on FASRC when connected). Restorable from the zip.", tone: "danger", confirmLabel: "Archive" })) {
              await archive.run("/ensemble/archive-member", { member: d.name });
            }
          }}>Archive</Button>
        </div>
      )}
      {!d.active && d.archived && (
        <div className="mdl-stack" style={{ gap: "var(--s2)" }}>
          <DefList dense items={[
            ["archived", d.archived.archived_at ? `${formatRelative(d.archived.archived_at)} · ${formatDateTime(d.archived.archived_at)}` : "—"],
            ["commit", <code key="c">{d.archived.commit ?? "—"}</code>],
            ["zip", d.archived.zip_found ? `${d.archived.zip} (${d.archived.campaign})` : `${d.archived.zip ?? "—"} — not found`],
          ]} />
          <Button size="sm" disabled={!d.archived.zip_found} loading={restore.busy} onClick={async () => {
            if (await confirm({ title: `Restore ${d.name}?`, message: "Unzips the archive into the ensemble and makes it active again; evaluation and gate then read stale.", confirmLabel: "Restore" })) {
              await restore.run("/ensemble/restore-member", { member: d.name });
            }
          }}>Restore from zip</Button>
        </div>
      )}
      <JobProgress job={archive.job} error={archive.error} />
      <JobProgress job={restore.job} error={restore.error} />
      {r && (
        <DefList dense items={[
          ["steps", `${stepsText(r.step, r.target_steps)}${r.status === "timeout" ? " · stopped short" : ""}`],
          ["∫PSNR", r.knee_integrated ? `${db(r.knee_integrated.mean)} dB (rank ${r.knee_rank ?? "—"}) · ${BANDS.map((b) => `${BAND_SHORT[b]} ${db(r.knee_integrated?.[b])}`).join(" · ")}` : "—"],
          ["test VIS", r.vis_psnr != null ? `${db(r.vis_psnr, 3)} dB · VIS asinh, last evaluation` : "—"],
          ["test 4-band", r.psnr != null ? `${db(r.psnr, 3)} dB (rank ${r.psnr_rank ?? "—"}) · joint asinh, member-PSNR cache` : "—"],
          ["coherence", r.coherence ? `all ${db(r.coherence.overall, 3)} · SR ${db(r.coherence.sr, 3)}` : "—"],
          ["knee", k ? k.title : "—"],
          ["depth", r.blocks != null ? `${r.blocks} blocks` : "—"],
          ["bootstrap · noise", `${r.bootstrap ?? "off"} · ${r.noise_aug ?? 0} RN`],
          ["ICNR", r.icnr ? "yes" : "no"],
          r.forked_from ? ["forked from", r.forked_from] : null,
          ["seed", r.seed != null ? String(r.seed) : "—"],
          ["commit", <code key="c">{r.commit ?? "—"}</code>],
          ["created", r.created_at ? formatDateTime(r.created_at) : "—"],
          r.job ? ["job", <Button key="j" size="sm" variant="ghost" onClick={() => openInspector({ kind: "job", id: `slurm/${r.job?.jobid}` })}>
            {r.job.jobid} · {r.job.state ?? "?"}{r.job.elapsed_seconds ? ` · ${formatDuration(r.job.elapsed_seconds)}` : ""}{r.job.gpu_util_mean != null ? ` · GPU ${Math.round(r.job.gpu_util_mean)}%` : ""}</Button>] : null,
          ["noise model", r.noise_model ?? "—"],
        ]} />
      )}
      {d.active && <>
        <Section title="Training curves"><CurvesPlot d={d} /></Section>
        <Section title="PSNR vs knee"><KneePlot d={d} /></Section>
        <Section title="Production-gate usage"><GatePlot d={d} /></Section>
        {r?.origin && <Section title="origin.json" collapsible defaultOpen={false}><JsonTree data={r.origin} expandDepth={1} /></Section>}
      </>}
      {r && r.knee_integrated?.mean != null && meanKnee != null && (
        <span className="mdl-faint">vs plain mean: {dbDelta((r.knee_integrated.mean as number) - meanKnee)} dB knee-integrated</span>
      )}
    </div>
  );
}
