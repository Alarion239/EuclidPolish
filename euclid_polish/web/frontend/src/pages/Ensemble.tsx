/* Ensemble — the deep-ensemble control room. Members table (+ archive), live
   training curves, evaluate/pull/PSNR controls, the power-spectrum + pixel
   diagnostics, the per-band combiner (fit + gate curves), and the disagreement
   cutout viewer. Full parity with the classic page, drawn from the JSON
   endpoints (status.json / evals.json / combiner.json / training-curves.json). */
import { useMemo, useState } from "react";
import { useResource } from "../hooks";
import { useJob, JobProgressView } from "../jobs";
import { useThemeValue } from "../theme";
import { CutoutViewer } from "../legacy";
import { C, LOSS_COLOR, categorical } from "../colors";
import Plot, { Legend, type Series, type Guide, type Tick } from "../charts/Plot";
import {
  Badge, Button, Card, CardBody, CardHead, Checkbox, DefList, Empty, Field,
  Input, NumberField, Page, PageHead, Segmented, Select, Spinner, Stat, Table,
  Tabs, type Column,
} from "../ui";

type Mode = "starfull" | "starless";
type ColorBy = "uniform" | "loss" | "depth";

/* ── status.json ─────────────────────────────────────────────────────────── */
type Member = {
  name: string; seed?: number | null; size_mb?: number; step?: number | null;
  blocks?: number; loss?: string; psnr?: number | null; psnr_rank?: number | null;
  starless?: boolean; has_loss_best?: boolean;
};
type EvalSummary = { ensemble_psnr?: number; mean_member_psnr?: number; ensemble_gain_db?: number } | null;
type Status = {
  base_dir: string; members: Member[]; archived: string[];
  n_members: number; n_models: number; records_dir?: string | null;
  eval_subset?: string; test_present: boolean; psnr_fields: number;
  evaluations_available: boolean; eval_summary: EvalSummary; eval_summary_stale?: boolean;
};

/* ── evals.json ──────────────────────────────────────────────────────────── */
type PS = {
  theta: (number | null)[]; r: (number | null)[]; r_lr?: (number | null)[];
  r_comb?: (number | null)[]; r_members?: (number | null)[][];
  r_pairs?: (number | null)[][]; r_cross?: (number | null)[];
};
type Evals = {
  ps: PS | null; guides?: { theta_min?: number; lr_scale?: number; vis_fwhm?: number };
  members?: { label: string; loss?: string; blocks?: number }[];
  n_fields?: number; n_members?: number;
  combiner?: { available?: boolean; psnr?: number | null; ensemble_mean_psnr?: number | null; best_member_psnr?: number | null } | null;
};

/* ── combiner.json ───────────────────────────────────────────────────────── */
type EffW = { brightness_asinh?: (number | null)[]; brightness_e?: (number | null)[]; jacobian?: (number | null)[][] };
type Combiner = {
  available?: boolean; stale?: boolean; regime?: string;
  member_labels: string[];
  members?: { label: string; loss?: string; blocks?: number; step?: number | null; psnr?: number | null }[];
  n_kernels?: number; min_usage?: number; val_l1?: number | null;
  band_names?: string[]; surviving?: Record<string, boolean[]>;
  eff_weights?: Record<string, EffW>;
};

/* ── training-curves.json ────────────────────────────────────────────────── */
/* Note: the payload overwrites the loss *series* with the loss-*norm* string,
   so only the PSNR series is chartable; `loss` here is the norm ("l1"…). */
type Curve = { name: string; psnr: [number, number][]; blocks?: number; test_psnr?: number | null; loss?: string; starless?: boolean };

const XTICKS = [0.05, 0.1, 0.2, 0.5, 1, 2, 5];
const hasData = (a?: (number | null)[]) => !!a && a.some((v) => v != null && isFinite(v as number));
const finite = (a: [number, number][], i: 0 | 1) => a.map((p) => p[i]);

const DIAG_TABS = [
  { id: "power-spectrum", label: "power spectrum" },
  { id: "std-error", label: "std vs error" },
  { id: "std-brightness", label: "std vs brightness" },
  { id: "calibration", label: "calibration" },
] as const;
type DiagTab = typeof DIAG_TABS[number]["id"];

export default function EnsemblePage() {
  const theme = useThemeValue();
  const [mode, setMode] = useState<Mode>("starfull");
  const starless = mode === "starless";

  const status = useResource<Status>("/ensemble/status.json", [mode]);
  const evals = useResource<Evals>(`/ensemble/evals.json?mode=${mode}`, [mode]);
  const comb = useResource<Combiner>(`/ensemble/combiner.json?mode=${mode}`, [mode]);
  const curves = useResource<{ members: Curve[] }>("/ensemble/training-curves.json");

  const evalJob = useJob();
  const opJob = useJob();
  const fitJob = useJob();

  const reloadAll = () => { status.reload(); evals.reload(); comb.reload(); curves.reload(); };

  return (
    <Page>
      <PageHead eyebrow="model · ensemble" title="Ensemble"
        sub="The seed-diverse WDSR deep ensemble: members, training, evaluation, the per-band combiner, and where the members disagree."
        right={
          <div className="row" style={{ gap: 8 }}>
            <Segmented<Mode> value={mode} onChange={setMode}
              options={[{ value: "starfull", label: "starfull" }, { value: "starless", label: "starless" }]} />
          </div>
        } />

      <div className="grid" style={{ gridTemplateColumns: "1fr", gap: "var(--s4)" }}>
        <Controls status={status.data} mode={mode} evalJob={evalJob} opJob={opJob} onDone={reloadAll} />
        <Members status={status.data} loading={status.loading} starless={starless} opJob={opJob} onArchived={reloadAll} />
        <TrainingCurves curves={curves.data?.members ?? []} starless={starless} />
        <Evaluations evals={evals.data} loading={evals.loading} mode={mode} theme={theme} />
        <CombinerCard comb={comb.data} loading={comb.loading} mode={mode} theme={theme} fitJob={fitJob} onFit={reloadAll} />
        <Card>
          <CardHead title="Disagreement viewer" sub="per-field members · mean · std — where the ensemble hallucinates" />
          <CardBody><CutoutViewer collection="ensemble" params={{ mode }} /></CardBody>
        </Card>
      </div>
    </Page>
  );
}

/* ── controls: evaluate / pull / psnr ────────────────────────────────────── */
function Controls(
  { status, mode, evalJob, opJob, onDone }:
  { status: Status | null; mode: Mode; evalJob: ReturnType<typeof useJob>; opJob: ReturnType<typeof useJob>; onDone: () => void },
) {
  const [n, setN] = useState("100");
  const s = status;
  return (
    <Card>
      <CardHead title="Run"
        sub={s ? `${s.n_members} members · ${s.eval_subset ?? "test"} set${s.records_dir ? "" : " · no records synced"}` : undefined}
        right={s?.eval_summary && (
          <div className="row" style={{ gap: "var(--s4)" }}>
            <Stat k="ensemble PSNR" v={s.eval_summary.ensemble_psnr != null ? `${s.eval_summary.ensemble_psnr.toFixed(2)} dB` : "—"} />
            <Stat k="gain vs mean member" v={s.eval_summary.ensemble_gain_db != null ? `${s.eval_summary.ensemble_gain_db >= 0 ? "+" : ""}${s.eval_summary.ensemble_gain_db.toFixed(2)} dB` : "—"} />
          </div>
        )} />
      <CardBody>
        <div className="row" style={{ alignItems: "flex-end", gap: "var(--s3)" }}>
          <NumberField label="test fields" value={n} onChange={setN} min={1} max={2000} />
          <Button variant="primary" disabled={evalJob.busy || !s?.test_present}
            onClick={() => evalJob.run("/ensemble/evaluate", { num_images: n, mode }, { onDone })}>
            Evaluate on test set
          </Button>
          <Button disabled={opJob.busy}
            onClick={() => opJob.run("/ensemble/member-psnr", {}, { onDone })}>↻ Refresh member PSNR</Button>
          <Button disabled={opJob.busy}
            onClick={() => opJob.run("/ensemble/pull", {}, { onDone })}>⬇ Pull from FASRC</Button>
          {s?.eval_summary_stale && <Badge tone="warn">summary stale — re-evaluate</Badge>}
        </div>
        <JobProgressView job={evalJob.job} error={evalJob.error} />
        <JobProgressView job={opJob.job} error={opJob.error} />
      </CardBody>
    </Card>
  );
}

/* ── members table ───────────────────────────────────────────────────────── */
function Members(
  { status, loading, starless, opJob, onArchived }:
  { status: Status | null; loading: boolean; starless: boolean; opJob: ReturnType<typeof useJob>; onArchived: () => void },
) {
  const rows = (status?.members ?? []).filter((m) => !!m.starless === starless);
  const cols: Column<Member>[] = [
    { header: "#", cell: (m) => m.psnr_rank ? <b>{m.psnr_rank}</b> : <span className="muted">—</span>, width: 40 },
    { header: "member", cell: (m) => <code className="mono">{m.name}</code> },
    { header: "PSNR", cell: (m) => m.psnr != null ? `${m.psnr.toFixed(3)} dB` : <span className="muted">—</span>, align: "right" },
    { header: "loss", cell: (m) => <Badge>{(m.loss ?? "l1").toUpperCase()}</Badge> },
    { header: "depth", cell: (m) => m.blocks ?? "—", align: "right" },
    { header: "step", cell: (m) => m.step ? m.step.toLocaleString() : "—", align: "right" },
    { header: "seed", cell: (m) => <span className="muted mono">{m.seed ?? "—"}</span> },
    { header: "size", cell: (m) => m.size_mb != null ? `${m.size_mb} MB` : "—", align: "right" },
    { header: "loss-best", cell: (m) => m.has_loss_best ? "✓" : "—", align: "center" },
    { header: "", align: "right", cell: (m) => (
      <Button size="sm" variant="ghost" disabled={opJob.busy}
        title="zip → tracking, tombstone, delete, purge caches"
        onClick={() => { if (window.confirm(`Archive ${m.name}? This retires it from the ensemble.`)) opJob.run("/ensemble/archive-member", { member: m.name }, { onDone: onArchived }); }}>
        📦 archive
      </Button>
    ) },
  ];
  return (
    <Card>
      <CardHead title="Members" sub={`${rows.length} ${starless ? "starless" : "starfull"} member(s)${status?.archived?.length ? ` · ${status.archived.length} archived` : ""}`} />
      <CardBody>
        {loading ? <Empty><Spinner /> loading…</Empty>
          : <Table columns={cols} rows={rows} rowKey={(m) => m.name}
              empty={`no ${starless ? "starless" : "starfull"} members — train some, then ⬇ pull from FASRC`} />}
      </CardBody>
    </Card>
  );
}

/* ── training curves ─────────────────────────────────────────────────────── */
function TrainingCurves({ curves, starless }: { curves: Curve[]; starless: boolean }) {
  const theme = useThemeValue();
  const [colorBy, setColorBy] = useState<ColorBy>("loss");
  const rows = curves.filter((c) => !!c.starless === starless);

  const chart = useMemo(() => {
    const series: Series[] = [];
    let xMax = 1, yMin = Infinity, yMax = -Infinity;
    const depths = [...new Set(rows.map((c) => c.blocks ?? 0))].sort((a, b) => a - b);
    rows.forEach((c) => {
      if (!c.psnr?.length) return;
      const x = finite(c.psnr, 0), y = finite(c.psnr, 1);
      xMax = Math.max(xMax, ...x);
      for (const v of y) { if (isFinite(v)) { yMin = Math.min(yMin, v); yMax = Math.max(yMax, v); } }
      const color = colorBy === "loss" ? (LOSS_COLOR[c.loss ?? "l1"])
        : colorBy === "depth" ? categorical(depths.indexOf(c.blocks ?? 0))
        : C.mean;
      series.push({ x, y, color, width: 1.4, alpha: 0.85 });
    });
    if (!series.length || !isFinite(yMin)) return null;
    const pad = (yMax - yMin) * 0.06 || 1;
    const yDomain: [number, number] = [yMin - pad, yMax + pad];
    const xTicks: Tick[] = [0, 0.25, 0.5, 0.75, 1].map((f) => ({ v: f * xMax, label: `${Math.round((f * xMax) / 1000)}k` }));
    const yTicks: Tick[] = [0, 0.25, 0.5, 0.75, 1].map((f) => { const v = yDomain[0] + f * (yDomain[1] - yDomain[0]); return { v, label: v.toFixed(1) }; });
    return { series, xDomain: [0, xMax] as [number, number], yDomain, xTicks, yTicks };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows, colorBy, theme]);

  return (
    <Card>
      <CardHead title="Training curves" sub={`${rows.length} member(s) · validation PSNR, rollback-deduped`}
        right={
          <Select<ColorBy> value={colorBy} onChange={setColorBy}
            options={[{ value: "loss", label: "by loss" }, { value: "depth", label: "by depth" }, { value: "uniform", label: "uniform" }]} />
        } />
      <CardBody>
        {!chart ? <Empty>no training logs yet</Empty> : (
          <Plot title="validation PSNR (asinh) vs step"
            xDomain={chart.xDomain} yDomain={chart.yDomain} xTicks={chart.xTicks} yTicks={chart.yTicks}
            xLabel="training step" yLabel="PSNR [dB]" series={chart.series} aspect={0.4} />
        )}
      </CardBody>
    </Card>
  );
}

/* ── evaluations: power spectrum + diagnostics ───────────────────────────── */
function Evaluations(
  { evals, loading, mode, theme }:
  { evals: Evals | null; loading: boolean; mode: Mode; theme: string },
) {
  const [tab, setTab] = useState<DiagTab>("power-spectrum");
  const [colorBy, setColorBy] = useState<ColorBy>("uniform");
  const ps = evals?.ps ?? null;
  const members = evals?.members ?? [];

  const chart = useMemo(() => {
    if (!ps || !hasData(ps.theta)) return null;
    const theta = ps.theta.map((v) => (v == null ? NaN : v)) as number[];
    const xs = theta.filter((v) => isFinite(v));
    const g = evals?.guides ?? {};
    const xDomain: [number, number] = [g.theta_min ?? 0.05, Math.max(...xs)];
    const xTicks: Tick[] = XTICKS.filter((v) => v >= xDomain[0] && v <= xDomain[1]).map((v) => ({ v, label: String(v) }));
    const yTicks: Tick[] = [0, 0.25, 0.5, 0.75, 1].map((v) => ({ v, label: String(v) }));
    const memberColor = (i: number) => colorBy === "loss" ? (LOSS_COLOR[members[i]?.loss ?? "l1"] ?? C.muted) : C.muted;
    const series: Series[] = [];
    for (const pair of ps.r_pairs ?? []) series.push({ x: theta, y: pair, color: C.muted, width: 0.7, alpha: 0.18 });
    (ps.r_members ?? []).forEach((row, i) => series.push({ x: theta, y: row, color: memberColor(i), width: 1, alpha: colorBy === "loss" ? 0.6 : 0.4 }));
    if (hasData(ps.r_cross)) series.push({ x: theta, y: ps.r_cross!, color: C.cross, width: 2, dash: [6, 3] });
    if (hasData(ps.r_lr)) series.push({ x: theta, y: ps.r_lr!, color: C.baseline, width: 2.5, dash: [7, 4] });
    series.push({ x: theta, y: ps.r, color: C.mean, width: 2.6, dots: true });
    if (hasData(ps.r_comb)) series.push({ x: theta, y: ps.r_comb!, color: C.comb, width: 2.2, dots: true });
    const guides: Guide[] = [
      { axis: "y", v: 1, color: C.guide, dash: [2, 3] },
      { axis: "x", v: g.lr_scale ?? 0.1, color: C.guide, width: 1.3, dash: [6, 3] },
      { axis: "x", v: g.vis_fwhm ?? 0.16, color: C.visfwhm, alpha: 0.5, width: 1.5, dash: [5, 2] },
    ];
    const legend = [
      ...(hasData(ps.r_lr) ? [{ label: "LR baseline", color: C.baseline, dash: true }] : []),
      { label: "ensemble mean", color: C.mean },
      ...(hasData(ps.r_comb) ? [{ label: "combiner", color: C.comb }] : []),
      { label: "individual models", color: C.muted },
      { label: "model–model r̃(k)", color: C.cross, dash: true },
    ];
    return { series, guides, xDomain, yDomain: [0, 1.05] as [number, number], xTicks, yTicks, legend };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ps, members, colorBy, evals, theme]);

  const cb = evals?.combiner;
  return (
    <Card>
      <CardHead title="Evaluations"
        sub={evals ? `${evals.n_fields ?? 0} fields · ${evals.n_members ?? 0} members · VIS` : "VIS band"}
        right={<Tabs<DiagTab> value={tab} tabs={DIAG_TABS.map((t) => ({ id: t.id, label: t.label }))} onChange={setTab} />} />
      <CardBody>
        {loading ? <Empty><Spinner /> loading…</Empty>
          : tab === "power-spectrum" ? (
            !chart ? <Empty>no evaluation cached for <b>{mode}</b> — run “Evaluate on test set”.</Empty> : (
              <>
                <div className="row" style={{ justifyContent: "flex-end", marginBottom: 8 }}>
                  <Select<ColorBy> value={colorBy} onChange={setColorBy}
                    options={[{ value: "uniform", label: "uniform" }, { value: "loss", label: "by loss" }]} />
                </div>
                <Plot title="cross-correlation r(k) vs HR   (1 = perfect)" xScale="log"
                  xDomain={chart.xDomain} yDomain={chart.yDomain} xTicks={chart.xTicks} yTicks={chart.yTicks}
                  xLabel="angular scale θ = 1/2k [arcsec]" yLabel="r(k) [VIS]" series={chart.series} guides={chart.guides} aspect={0.46} />
                <Legend items={chart.legend} />
                <div className="row" style={{ gap: "var(--s5)", marginTop: "var(--s4)" }}>
                  <Stat k="fields" v={evals?.n_fields ?? "—"} />
                  <Stat k="members" v={evals?.n_members ?? "—"} />
                  {cb?.available && <Stat k="combiner PSNR" v={cb.psnr != null ? `${cb.psnr.toFixed(2)} dB` : "—"} />}
                  {cb?.available && cb.psnr != null && cb.ensemble_mean_psnr != null &&
                    <Badge tone={cb.psnr >= cb.ensemble_mean_psnr ? "good" : "warn"}>
                      {cb.psnr >= cb.ensemble_mean_psnr ? "+" : ""}{(cb.psnr - cb.ensemble_mean_psnr).toFixed(2)} vs mean
                    </Badge>}
                </div>
              </>
            )
          ) : (
            <div className="ui-figure__paper" style={{ minHeight: 300 }}>
              <img src={`/ensemble/eval-plot/${tab}.png?mode=${mode}`} alt={tab} loading="lazy" style={{ maxWidth: "100%" }} />
            </div>
          )}
      </CardBody>
    </Card>
  );
}

/* ── combiner ────────────────────────────────────────────────────────────── */
function CombinerCard(
  { comb, loading, mode, theme, fitJob, onFit }:
  { comb: Combiner | null; loading: boolean; mode: Mode; theme: string; fitJob: ReturnType<typeof useJob>; onFit: () => void },
) {
  const [nImg, setNImg] = useState("100");
  const [nKernels, setNKernels] = useState("12");
  const [minUsage, setMinUsage] = useState("0");
  const [band, setBand] = useState<string>("");

  const bands = comb?.band_names ?? [];
  const activeBand = band || bands[0] || "";

  const importance = useMemo(() => {
    if (!comb?.eff_weights || !comb.member_labels) return [];
    const m = comb.member_labels.length;
    const cum = new Array(m).fill(0);
    for (const b of Object.values(comb.eff_weights)) {
      const jac = b.jacobian ?? [];
      const L = jac.length || 1;
      for (const row of jac) row?.forEach((w, i) => { if (w != null && isFinite(w)) cum[i] += w / L; });
    }
    const surv = comb.member_labels.map((_, i) => Object.values(comb.surviving ?? {}).some((arr) => arr[i] !== false));
    return comb.member_labels
      .map((label, i) => ({ label, total: cum[i], kept: surv[i], meta: comb.members?.[i] }))
      .filter((r) => r.kept)
      .sort((a, b2) => b2.total - a.total);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [comb, theme]);

  const gate = useMemo(() => {
    const ew = comb?.eff_weights?.[activeBand];
    if (!ew?.jacobian?.length) return null;
    const bx = (ew.brightness_e ?? []).map((e) => (e == null ? NaN : Math.asinh(e / 100))) as number[];
    const surv = comb?.surviving?.[activeBand] ?? [];
    const M = comb?.member_labels.length ?? 0;
    const xs = bx.filter((v) => isFinite(v));
    const xDomain: [number, number] = [Math.min(...xs), Math.max(...xs)];
    const series: Series[] = [];
    for (let m = 0; m < M; m++) {
      if (surv[m] === false) continue;
      const y = ew.jacobian.map((row) => (row?.[m] ?? null));
      series.push({ x: bx, y, color: LOSS_COLOR[comb?.members?.[m]?.loss ?? "l1"] ?? C.muted, width: 1.8 });
    }
    const xTicks: Tick[] = [0, 0.25, 0.5, 0.75, 1].map((f) => { const v = xDomain[0] + f * (xDomain[1] - xDomain[0]); return { v, label: Math.round(100 * Math.sinh(v)).toString() }; });
    const yTicks: Tick[] = [0, 0.5, 1].map((v) => ({ v, label: String(v) }));
    return { series, xDomain, yDomain: [0, 1.02] as [number, number], xTicks, yTicks };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [comb, activeBand, theme]);

  return (
    <Card>
      <CardHead title={`Combiner · ${mode}`}
        sub="a per-band brightness gate fusing members — fit locally on validate, scored on test"
        right={comb?.available && <Badge tone={comb.stale ? "warn" : "good"}>{comb.stale ? "stale" : "fitted"}</Badge>} />
      <CardBody>
        <div className="row" style={{ alignItems: "flex-end", gap: "var(--s3)" }}>
          <NumberField label="validate fields" value={nImg} onChange={setNImg} min={1} max={2000} />
          <NumberField label="kernels (K)" value={nKernels} onChange={setNKernels} min={2} max={32} />
          <NumberField label="prune (min importance)" value={minUsage} onChange={setMinUsage} min={0} max={0.5} step={0.01} />
          <Button variant="primary" disabled={fitJob.busy}
            onClick={() => fitJob.run("/ensemble/combiner/fit", { num_images: nImg, n_kernels: nKernels, min_usage: minUsage, mode }, { onDone: onFit })}>
            Fit combiner
          </Button>
        </div>
        <JobProgressView job={fitJob.job} error={fitJob.error} />

        {loading ? <Empty><Spinner /> loading…</Empty>
          : !comb?.available ? <Empty>no combiner fitted for <b>{mode}</b> yet — set knobs above and fit.</Empty> : (
          <div style={{ marginTop: "var(--s4)" }}>
            <DefList items={[
              ["members", `${comb.member_labels.length} (${importance.length} surviving)`],
              ["kernels", String(comb.n_kernels ?? "—")],
              ["prune ≥", String(comb.min_usage ?? 0)],
              ["validate L1", comb.val_l1 != null ? comb.val_l1.toFixed(4) : "—"],
            ]} />

            <div style={{ marginTop: "var(--s4)" }}>
              <div className="eyebrow" style={{ marginBottom: 8 }}>member importance (cumulative gate weight)</div>
              {importance.map((r) => (
                <div key={r.label} className="ens-imp">
                  <span className="ens-imp__label mono">{r.label}</span>
                  <div className="ens-imp__bar"><div className="ens-imp__fill" style={{ width: `${Math.min(100, (r.total / (importance[0]?.total || 1)) * 100)}%`, background: LOSS_COLOR[r.meta?.loss ?? "l1"] ?? "var(--accent)" }} /></div>
                  <span className="ens-imp__val mono">{r.total.toFixed(2)}</span>
                  <span className="ens-imp__meta muted">{[r.meta?.loss, r.meta?.blocks && `${r.meta.blocks}b`, r.meta?.psnr != null && `${r.meta.psnr.toFixed(1)}dB`].filter(Boolean).join(" · ")}</span>
                </div>
              ))}
            </div>

            {bands.length > 0 && (
              <div style={{ marginTop: "var(--s4)" }}>
                <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
                  <div className="eyebrow">gate weight vs brightness</div>
                  <Segmented<string> value={activeBand} onChange={setBand} options={bands.map((b) => ({ value: b, label: b }))} />
                </div>
                {!gate ? <Empty>no gate data for {activeBand}</Empty> : (
                  <Plot title={`${activeBand} — convex member weights`} xDomain={gate.xDomain} yDomain={gate.yDomain}
                    xTicks={gate.xTicks} yTicks={gate.yTicks} xLabel="pixel brightness [e⁻]" yLabel="gate weight"
                    series={gate.series} aspect={0.4} />
                )}
              </div>
            )}
          </div>
        )}
      </CardBody>
    </Card>
  );
}
