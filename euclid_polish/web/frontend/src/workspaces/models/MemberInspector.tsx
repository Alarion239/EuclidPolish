/* Inspector kind `member` (`member:member_257`, also `257` / `257·psnr`):
   one universal model — status and recipe (origin.json), its training
   curves (per-band PSNR, loss, gradient norm, step time), Models ›
   Universal, Sky › Compare and Archive (a stale ∫PSNR marked); an archived
   one restores from its zip as a universal model. The production model
   (the catalogue's `production`, GET /api/models) is badged and cannot be
   archived. GET /ensemble/member/<name>.json. */
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import Plot, { type Series } from "../../charts/Plot";
import { C, bandColor, categorical } from "../../colors";
import { useJob } from "../../api/jobs";
import { useResource } from "../../api/query";
import { openInspector, type InspectorProps } from "../../app/inspector";
import { pagePath } from "../../app/nav";
import { formatDateTime, formatDuration, formatRelative } from "../../format";
import {
  Badge, Button, DefList, EmptyState, JobProgress, JsonTree, Section, Segmented, Skeleton, confirm,
} from "../../ui";
import { BAND_SHORT, BANDS, url, useModelCatalog, type MemberDetail } from "./api";
import { JOB, useOnJobEnd } from "./jobs";
import { db, kneeText, memberName, memberNumber, restoreText, stepsText, xy } from "./model";
import { compareHref } from "./universal";
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

export default function MemberInspector({ id }: InspectorProps) {
  const name = memberName(id) ?? id;
  const res = useResource<MemberDetail>(url.member(name), [name], { ttl: 30_000 });
  const navigate = useNavigate();
  const restore = useJob(JOB.restore);
  const archive = useJob(JOB.archive);
  const catalog = useModelCatalog();
  const production = catalog.data?.production?.kind === "universal" ? catalog.data.production : null;
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
  const tab = (t: string, q = "") => navigate(`${pagePath("models", { tab: t })}${q}`);
  const k = r ? kneeText(r) : null;
  const universal = !d.archived;
  const stale = universal && r?.score_stale === true;
  const isProduction = universal && production?.name === d.name;
  const doArchive = async () => {
    if (await confirm({ title: `Archive ${d.name}?`, tone: "danger", confirmLabel: "Archive",
      message: `Zipped to the tracking campaign, tombstoned and deleted (also on FASRC when connected). Restorable from the zip${universal ? ", as a universal model again" : ""}.` })) {
      await archive.run("/ensemble/archive-member", { member: d.name });
    }
  };
  return (
    <div className="mdl-insp">
      <div className="mdl-insp__head">
        <span className="mdl-insp__title">#{num}</span>
        {isProduction && <Badge tone="accent">production</Badge>}
        {universal ? <Badge tone="accent">universal</Badge> : <Badge tone="warn">archived</Badge>}
        {r?.status === "timeout" && <Badge tone="warn">TIMEOUT</Badge>}
        {r?.status === "running" && <Badge tone="info" dot>training</Badge>}
        {r && <Badge>{r.loss.toUpperCase()}</Badge>}
        {k && <Badge tone={k.kind === "multi" ? "accent" : undefined}>{k.text}</Badge>}
      </div>
      {universal && (
        <div className="mdl-row">
          <Button size="sm" onClick={() => tab("universal")}>Universal</Button>
          <Button size="sm" variant="ghost" onClick={() => navigate(compareHref([`universal:${d.name}`], production?.spec ?? null))}>Compare on real fields</Button>
          <Button size="sm" variant="danger" loading={archive.busy} onClick={doArchive} disabled={isProduction}
            title={isProduction ? "The production model: make another model production first (Models › Universal)" : undefined}>Archive</Button>
        </div>
      )}
      {d.archived && (
        <div className="mdl-stack" style={{ gap: "var(--s2)" }}>
          <DefList dense items={[
            ["archived", d.archived.archived_at ? `${formatRelative(d.archived.archived_at)} · ${formatDateTime(d.archived.archived_at)}` : "—"],
            ["commit", <code key="c">{d.archived.commit ?? "—"}</code>],
            ["zip", d.archived.zip_found ? `${d.archived.zip} (${d.archived.campaign})` : `${d.archived.zip ?? "—"} — not found`],
          ]} />
          <Button size="sm" disabled={!d.archived.zip_found} loading={restore.busy} onClick={async () => {
            if (await confirm({ title: `Restore ${d.name}?`, message: restoreText({ class: "universal" }), confirmLabel: "Restore" })) {
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
          ["∫PSNR", r.knee_integrated ? <span key="k" className={stale ? "mdl-faint" : undefined}>
            {`${db(r.knee_integrated.mean)} dB · ${BANDS.map((b) => `${BAND_SHORT[b]} ${db(r.knee_integrated?.[b])}`).join(" · ")}`}
            {stale && <> <Badge tone="warn" size="sm">stale score</Badge></>}
          </span> : "—"],
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
      {universal && <>
        <Section title="Training curves"><CurvesPlot d={d} /></Section>
        {r?.origin && <Section title="origin.json" collapsible defaultOpen={false}><JsonTree data={r.origin} expandDepth={1} /></Section>}
      </>}
    </div>
  );
}
