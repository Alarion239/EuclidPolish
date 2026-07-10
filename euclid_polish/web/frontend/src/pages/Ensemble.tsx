/* Ensemble — the deep-ensemble control room. Members table (+ archive), live
   training curves, evaluate/pull/PSNR controls, the power-spectrum + pixel
   diagnostics, the per-band combiner (fit + gate curves), and the disagreement
   cutout viewer. Full parity with the classic page, drawn from the JSON
   endpoints (status.json / evals.json / combiner.json / training-curves.json). */
import { useCallback, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useResource } from "../hooks";
import { useJob, JobProgressView } from "../jobs";
import { StepById } from "../fasrc";
import { useThemeValue } from "../theme";
import { CutoutViewer, type ViewerApi } from "../legacy";
import { C, LOSS_COLOR, categorical, viridis } from "../colors";
import Plot, { Legend, type Series, type Guide, type Tick, type Heat } from "../charts/Plot";
import {
  Badge, Button, Card, CardBody, CardHead, Checkbox, DefList, Empty, Field,
  Input, NumberField, Page, PageHead, Segmented, Select, Spinner, Stat, Table,
  Tabs, type Column,
} from "../ui";

type Mode = "starfull" | "starless";
type ColorBy = "uniform" | "loss" | "depth" | "knee";

/* Per-member asinh stretch knee (electrons). `null`/absent → the per-band
   config default, which every pre-knob member trained under → render as 100. */
const kneeOf = (k?: number | null): number => (k == null ? 100 : k);
const kneeTag = (k?: number | null): string => `${kneeOf(k)}e`;

/* ── status.json ─────────────────────────────────────────────────────────── */
type Member = {
  name: string; seed?: number | null; size_mb?: number; step?: number | null;
  blocks?: number; loss?: string; asinh_knee?: number | null; psnr?: number | null;
  psnr_rank?: number | null; starless?: boolean; has_loss_best?: boolean;
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
/* Pixel diagnostics (VIS, electrons). All *_edges / med_* arrays are already in
   the axis units the frontend plots: std_err in log10(e⁻) on both axes;
   bright_std x is asinh(HR/stretch), y is log10(e⁻). hist[i][j] = pixel count. */
type StdErr = {
  edges: (number | null)[]; hist: number[][];
  med_std: (number | null)[]; med_err: (number | null)[];
};
type BrightStd = {
  bright_edges: (number | null)[]; std_edges: (number | null)[]; hist: number[][];
  bright: (number | null)[]; lo: (number | null)[]; med: (number | null)[]; hi: (number | null)[];
  stretch: number;
};
type Evals = {
  ps: PS | null; guides?: { theta_min?: number; lr_scale?: number; vis_fwhm?: number };
  members?: { label: string; loss?: string; blocks?: number; asinh_knee?: number | null }[];
  n_fields?: number; n_members?: number;
  std_err?: StdErr | null; bright_std?: BrightStd | null;
  combiner?: { available?: boolean; psnr?: number | null; ensemble_mean_psnr?: number | null; best_member_psnr?: number | null } | null;
};

/* ── combiner.json ───────────────────────────────────────────────────────── */
type EffW = { brightness_asinh?: (number | null)[]; brightness_e?: (number | null)[]; jacobian?: (number | null)[][] };
type Combiner = {
  available?: boolean; stale?: boolean; regime?: string;
  member_labels: string[];
  members?: { label: string; loss?: string; blocks?: number; asinh_knee?: number | null; step?: number | null; psnr?: number | null }[];
  n_kernels?: number; min_usage?: number; val_l1?: number | null;
  band_names?: string[]; surviving?: Record<string, boolean[]>;
  eff_weights?: Record<string, EffW>;
};

/* ── training-curves.json ────────────────────────────────────────────────── */
/* Note: the payload overwrites the loss *series* with the loss-*norm* string,
   so only the PSNR series is chartable; `loss` here is the norm ("l1"…). */
type Curve = { name: string; psnr: [number, number][]; blocks?: number; test_psnr?: number | null; loss?: string; asinh_knee?: number | null; starless?: boolean };

const XTICKS = [0.05, 0.1, 0.2, 0.5, 1, 2, 5];
const hasData = (a?: (number | null)[]) => !!a && a.some((v) => v != null && isFinite(v as number));
const finite = (a: [number, number][], i: 0 | 1) => a.map((p) => p[i]);
const num = (a?: (number | null)[]) => (a ?? []).map((v) => (v == null ? NaN : v));

const SUP: Record<string, string> = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
const sup = (n: number) => String(n).split("").map((c) => SUP[c] ?? c).join("");
/* Integer-decade ticks on a log10-valued (linear-drawn) axis → 10ⁿ labels. */
const logDecadeTicks = (lo: number, hi: number, step = 2): Tick[] => {
  const out: Tick[] = [];
  for (let e = Math.ceil(lo); e <= Math.floor(hi); e += step) out.push({ v: e, label: `10${sup(e)}` });
  return out;
};
/* Brightness axis: asinh(e⁻/stretch) tick positions labelled in electrons. */
const brightTicks = (stretch: number): Tick[] =>
  [0, 100, 1e3, 1e4, 1e5, 1e6].map((e, i) => ({
    v: Math.asinh(e / stretch), label: i === 0 ? "0" : i === 1 ? "100" : `10${sup([0, 0, 3, 4, 5, 6][i])}`,
  }));

const DIAG_TABS = [
  { id: "power-spectrum", label: "power spectrum" },
  { id: "std-error", label: "std vs error" },
  { id: "std-brightness", label: "std vs brightness" },
] as const;
type DiagTab = typeof DIAG_TABS[number]["id"];

export default function EnsemblePage() {
  const theme = useThemeValue();
  // Star regime lives in the URL (/ensemble/starfull | /ensemble/starless).
  const navigate = useNavigate();
  const { mode: modeParam } = useParams<{ mode: string }>();
  const mode: Mode = modeParam === "starless" ? "starless" : "starfull";
  const setMode = (m: Mode) => navigate(`/ensemble/${m}`);
  const starless = mode === "starless";

  const status = useResource<Status>("/ensemble/status.json", [mode]);
  const evals = useResource<Evals>(`/ensemble/evals.json?mode=${mode}`, [mode]);
  const comb = useResource<Combiner>(`/ensemble/combiner.json?mode=${mode}`, [mode]);
  const curves = useResource<{ members: Curve[] }>("/ensemble/training-curves.json");

  const evalJob = useJob();
  const opJob = useJob();
  const fitJob = useJob();
  // Train-card prefill (continue/fork buttons in the members table set these).
  const [train, setTrain] = useState<{ mode: "add" | "continue" | "fork"; member: string }>({ mode: "add", member: "" });

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
        <Members status={status.data} loading={status.loading} starless={starless} opJob={opJob}
          onArchived={reloadAll}
          onContinue={(m) => setTrain({ mode: "continue", member: m })}
          onFork={(m) => setTrain({ mode: "fork", member: m })} />
        <TrainMembers starlessDefault={starless} prefill={train} onPrefill={setTrain} />
        <TrainingCurves curves={curves.data?.members ?? []} starless={starless} />
        <Evaluations evals={evals.data} loading={evals.loading} mode={mode} theme={theme} />
        <CombinerCard comb={comb.data} loading={comb.loading} mode={mode} theme={theme} fitJob={fitJob} onFit={reloadAll} />
        <DisagreementCard key={mode} mode={mode} members={status.data?.members ?? []} />
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
  { status, loading, starless, opJob, onArchived, onContinue, onFork }:
  { status: Status | null; loading: boolean; starless: boolean; opJob: ReturnType<typeof useJob>;
    onArchived: () => void; onContinue: (m: string) => void; onFork: (m: string) => void },
) {
  const rows = (status?.members ?? []).filter((m) => !!m.starless === starless);
  const cols: Column<Member>[] = [
    { header: "#", cell: (m) => m.psnr_rank ? <b>{m.psnr_rank}</b> : <span className="muted">—</span>, width: 40 },
    { header: "member", cell: (m) => <code className="mono">{m.name}</code> },
    { header: "PSNR", cell: (m) => m.psnr != null ? `${m.psnr.toFixed(3)} dB` : <span className="muted">—</span>, align: "right" },
    { header: "loss", cell: (m) => <Badge>{(m.loss ?? "l1").toUpperCase()}</Badge> },
    { header: "depth", cell: (m) => m.blocks ?? "—", align: "right" },
    { header: "knee", cell: (m) => <span className="mono">{kneeTag(m.asinh_knee)}</span>, align: "right" },
    { header: "step", cell: (m) => m.step ? m.step.toLocaleString() : "—", align: "right" },
    { header: "", align: "right", cell: (m) => (
      <div className="row" style={{ gap: 4, justifyContent: "flex-end" }}>
        <Button size="sm" variant="ghost" title="continue training this member"
          onClick={() => onContinue(m.name)}>▶ continue</Button>
        <Button size="sm" variant="ghost" title="fork a new member from this one"
          onClick={() => onFork(m.name)}>⑂ fork</Button>
        <Button size="sm" variant="ghost" disabled={opJob.busy}
          title="zip → tracking, tombstone, delete, purge caches"
          onClick={() => { if (window.confirm(`Archive ${m.name}? This retires it from the ensemble.`)) opJob.run("/ensemble/archive-member", { member: m.name }, { onDone: onArchived }); }}>
          📦
        </Button>
      </div>
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

/* ── train members (add / continue / fork via the ensemble_train step) ────── */
function TrainMembers(
  { starlessDefault, prefill, onPrefill }:
  { starlessDefault: boolean; prefill: { mode: "add" | "continue" | "fork"; member: string }; onPrefill: (p: { mode: "add" | "continue" | "fork"; member: string }) => void },
) {
  const mode = prefill.mode;
  const setMode = (m: "add" | "continue" | "fork") => onPrefill({ mode: m, member: prefill.member });
  const [count, setCount] = useState("5");
  const [depth, setDepth] = useState("16");
  const [loss, setLoss] = useState("l1");
  const [knee, setKnee] = useState("100");
  const [steps, setSteps] = useState("50000");
  const [extraSteps, setExtraSteps] = useState("50000");
  const [starless, setStarless] = useState(starlessDefault);
  const [icnr, setIcnr] = useState(true);
  const [bootstrap, setBootstrap] = useState("0.7");
  const [forwardOtf, setForwardOtf] = useState(true);

  const extra = useMemo(() => {
    const p: Record<string, string | number> = { mode };
    if (mode === "add") Object.assign(p, {
      count, num_res_blocks: depth, loss, asinh_knee: knee, steps, starless: starless ? 1 : 0,
      icnr: icnr ? 1 : 0, bootstrap, forward_onthefly: forwardOtf ? 1 : 0,
    });
    else if (mode === "continue") Object.assign(p, { members: prefill.member, extra_steps: extraSteps });
    else Object.assign(p, { fork_from: prefill.member, count, steps });
    return p;
  }, [mode, count, depth, loss, knee, steps, extraSteps, starless, icnr, bootstrap, forwardOtf, prefill.member]);

  return (
    <Card>
      <CardHead title="Train members" sub="submit an ensemble_train SLURM job — add fresh members, or continue/fork an existing one"
        right={<Segmented<"add" | "continue" | "fork"> value={mode} onChange={setMode}
          options={[{ value: "add", label: "add" }, { value: "continue", label: "continue" }, { value: "fork", label: "fork" }]} />} />
      <CardBody>
        <div className="fasrc-step__res">
          {mode === "add" && <>
            <NumberField label="count" value={count} onChange={setCount} min={1} max={20} />
            <NumberField label="depth (blocks)" value={depth} onChange={setDepth} min={4} max={64} />
            <Field label="loss"><Select value={loss} onChange={setLoss}
              options={["l1", "l2", "l3", "berhu"].map((v) => ({ value: v, label: v }))} /></Field>
            <NumberField label="asinh knee [e⁻]" value={knee} onChange={setKnee} min={1} max={100000} step={10} />
            <NumberField label="steps" value={steps} onChange={setSteps} min={1000} max={500000} step={1000} />
            <NumberField label="bootstrap" value={bootstrap} onChange={setBootstrap} min={0} max={1} step={0.05} />
            <Checkbox checked={starless} onChange={setStarless}>starless</Checkbox>
            <Checkbox checked={icnr} onChange={setIcnr}>ICNR init</Checkbox>
            <Checkbox checked={forwardOtf} onChange={setForwardOtf}>on-the-fly forward</Checkbox>
          </>}
          {mode === "continue" && <>
            <Field label="member"><Input value={prefill.member} onChange={(m) => onPrefill({ mode, member: m })} placeholder="member_00" /></Field>
            <NumberField label="extra steps" value={extraSteps} onChange={setExtraSteps} min={1000} max={500000} step={1000} />
          </>}
          {mode === "fork" && <>
            <Field label="fork from"><Input value={prefill.member} onChange={(m) => onPrefill({ mode, member: m })} placeholder="member_02" /></Field>
            <NumberField label="new members" value={count} onChange={setCount} min={1} max={10} />
            <NumberField label="steps" value={steps} onChange={setSteps} min={1000} max={500000} step={1000} />
          </>}
        </div>
        <div style={{ marginTop: "var(--s3)" }}>
          <StepById stepId="ensemble_train" extraParams={extra} />
        </div>
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
    const knees = [...new Set(rows.map((c) => kneeOf(c.asinh_knee)))].sort((a, b) => a - b);
    rows.forEach((c) => {
      if (!c.psnr?.length) return;
      const x = finite(c.psnr, 0), y = finite(c.psnr, 1);
      xMax = Math.max(xMax, ...x);
      for (const v of y) { if (isFinite(v)) { yMin = Math.min(yMin, v); yMax = Math.max(yMax, v); } }
      const color = colorBy === "loss" ? (LOSS_COLOR[c.loss ?? "l1"])
        : colorBy === "depth" ? categorical(depths.indexOf(c.blocks ?? 0))
        : colorBy === "knee" ? categorical(knees.indexOf(kneeOf(c.asinh_knee)))
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
            options={[{ value: "loss", label: "by loss" }, { value: "depth", label: "by depth" }, { value: "knee", label: "by knee" }, { value: "uniform", label: "uniform" }]} />
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
    const depths = [...new Set(members.map((mm) => mm?.blocks ?? 0))].sort((a, b) => a - b);
    const knees = [...new Set(members.map((mm) => kneeOf(mm?.asinh_knee)))].sort((a, b) => a - b);
    const memberColor = (i: number) => {
      const mm = members[i];
      if (colorBy === "loss") return LOSS_COLOR[mm?.loss ?? "l1"] ?? C.muted;
      if (colorBy === "depth") return categorical(depths.indexOf(mm?.blocks ?? 0));
      if (colorBy === "knee") return categorical(knees.indexOf(kneeOf(mm?.asinh_knee)));
      return C.muted;
    };
    const grouped = colorBy !== "uniform";
    const series: Series[] = [];
    for (const pair of ps.r_pairs ?? []) series.push({ x: theta, y: pair, color: C.muted, width: 0.7, alpha: 0.18 });
    (ps.r_members ?? []).forEach((row, i) => series.push({ x: theta, y: row, color: memberColor(i), width: 1, alpha: grouped ? 0.6 : 0.4 }));
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

  // std vs error — density cloud + median-|error|-per-std curve, both axes
  // log10(e⁻). Diagonal reference lines (|err|=σ, 0.674σ) ride as 2-pt series.
  const stdErr = useMemo(() => {
    const d = evals?.std_err;
    if (!d?.hist?.length) return null;
    const edges = num(d.edges);
    const lo = edges[0], hi = edges[edges.length - 1];
    const heat: Heat = { z: d.hist, xEdges: edges, yEdges: edges };
    const g = Math.log10(0.6745);
    const series: Series[] = [
      { x: [lo, hi], y: [lo, hi], color: C.guide, width: 1.3, dash: [6, 3] },
      { x: [lo, hi], y: [lo + g, hi + g], color: C.guide, width: 1.3, dash: [2, 3] },
      { x: num(d.med_std), y: num(d.med_err), color: C.baseline, width: 2.6, dots: true },
    ];
    const ticks = logDecadeTicks(lo, hi);
    const legend = [
      { label: "median |error| per σ bin", color: C.baseline },
      { label: "|error| = σ", color: C.guide, dash: true },
      { label: "0.674·σ (calibrated Gaussian)", color: C.guide, dash: true },
    ];
    return { heat, series, domain: [lo, hi] as [number, number], ticks, legend };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [evals, theme]);

  // std vs brightness — density cloud + median σ (with 16–84%) per HR-brightness
  // bin. x = asinh(HR/stretch), y = log10 σ(e⁻).
  const brightStd = useMemo(() => {
    const d = evals?.bright_std;
    if (!d?.hist?.length) return null;
    const bx = num(d.bright_edges), sy = num(d.std_edges);
    const heat: Heat = { z: d.hist, xEdges: bx, yEdges: sy };
    const bright = num(d.bright);
    const series: Series[] = [
      { x: bright, y: num(d.lo), color: C.baseline, width: 1, alpha: 0.5, dash: [4, 3] },
      { x: bright, y: num(d.hi), color: C.baseline, width: 1, alpha: 0.5, dash: [4, 3] },
      { x: bright, y: num(d.med), color: C.baseline, width: 2.6, dots: true },
    ];
    const legend = [
      { label: "median σ per brightness bin", color: C.baseline },
      { label: "16–84%", color: C.baseline, dash: true },
    ];
    return {
      heat, series,
      xDomain: [bx[0], bx[bx.length - 1]] as [number, number],
      yDomain: [sy[0], sy[sy.length - 1]] as [number, number],
      xTicks: brightTicks(d.stretch), yTicks: logDecadeTicks(sy[0], sy[sy.length - 1]), legend,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [evals, theme]);

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
                    options={[{ value: "uniform", label: "uniform" }, { value: "loss", label: "by loss" }, { value: "depth", label: "by depth" }, { value: "knee", label: "by knee" }]} />
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
          ) : tab === "std-error" ? (
            !stdErr ? <Empty>no evaluation cached for <b>{mode}</b> — run “Evaluate on test set”.</Empty> : (
              <>
                <Plot title="Does disagreement predict error?  (VIS, per pixel)"
                  xDomain={stdErr.domain} yDomain={stdErr.domain} xTicks={stdErr.ticks} yTicks={stdErr.ticks}
                  xLabel="cross-member per-pixel σ  [e⁻]" yLabel="|ensemble mean − HR|  [e⁻]"
                  heat={stdErr.heat} series={stdErr.series} aspect={0.62} />
                <Legend items={stdErr.legend} />
              </>
            )
          ) : (
            !brightStd ? <Empty>no evaluation cached for <b>{mode}</b> — run “Evaluate on test set”.</Empty> : (
              <>
                <Plot title="Where does disagreement live?  (VIS, per pixel)"
                  xDomain={brightStd.xDomain} yDomain={brightStd.yDomain} xTicks={brightStd.xTicks} yTicks={brightStd.yTicks}
                  xLabel="HR pixel brightness  [e⁻]  (asinh axis)" yLabel="cross-member per-pixel σ  [e⁻]"
                  heat={brightStd.heat} series={brightStd.series} aspect={0.62} />
                <Legend items={brightStd.legend} />
              </>
            )
          )}
      </CardBody>
    </Card>
  );
}

/* ── combiner ────────────────────────────────────────────────────────────── */
type GateColorBy = "loss" | "psnr" | "depth" | "knee";

function CombinerCard(
  { comb, loading, mode, theme, fitJob, onFit }:
  { comb: Combiner | null; loading: boolean; mode: Mode; theme: string; fitJob: ReturnType<typeof useJob>; onFit: () => void },
) {
  const [nImg, setNImg] = useState("100");
  const [nKernels, setNKernels] = useState("12");
  const [minUsage, setMinUsage] = useState("0");
  const [band, setBand] = useState<string>("");
  const [colorBy, setColorBy] = useState<GateColorBy>("loss");

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
    const members = comb?.members ?? [];
    const M = comb?.member_labels.length ?? 0;
    const survIdx = Array.from({ length: M }, (_, m) => m).filter((m) => surv[m] !== false);

    // Facet colorer. depth/knee → categorical over the ensemble's distinct
    // depths/knees; psnr → viridis over the surviving members' PSNR range;
    // loss → loss token.
    const depths = [...new Set(members.map((mm) => mm?.blocks ?? 0))].sort((a, b) => a - b);
    const knees = [...new Set(members.map((mm) => kneeOf(mm?.asinh_knee)))].sort((a, b) => a - b);
    const psnrs = survIdx.map((m) => members[m]?.psnr).filter((p): p is number => p != null && isFinite(p));
    const pMin = psnrs.length ? Math.min(...psnrs) : 0;
    const pMax = psnrs.length ? Math.max(...psnrs) : 1;
    const memberColor = (m: number): string => {
      const meta = members[m];
      if (colorBy === "loss") return LOSS_COLOR[meta?.loss ?? "l1"] ?? C.muted;
      if (colorBy === "depth") return categorical(depths.indexOf(meta?.blocks ?? 0));
      if (colorBy === "knee") return categorical(knees.indexOf(kneeOf(meta?.asinh_knee)));
      const p = meta?.psnr;
      return p == null || pMax === pMin ? C.mean : viridis((p - pMin) / (pMax - pMin));
    };

    const xs = bx.filter((v) => isFinite(v));
    const xDomain: [number, number] = [Math.min(...xs), Math.max(...xs)];
    const series: Series[] = survIdx.map((m) => ({
      x: bx, y: ew.jacobian!.map((row) => (row?.[m] ?? null)), color: memberColor(m), width: 1.8,
    }));
    const legend = survIdx.map((m) => {
      const meta = members[m];
      const tag = colorBy === "loss" ? (meta?.loss ?? "l1")
        : colorBy === "depth" ? `${meta?.blocks ?? "?"}b`
        : colorBy === "knee" ? kneeTag(meta?.asinh_knee)
        : (meta?.psnr != null ? `${meta.psnr.toFixed(1)}dB` : "—");
      return { label: `${comb?.member_labels[m]} · ${tag}`, color: memberColor(m) };
    });
    const xTicks: Tick[] = [0, 0.25, 0.5, 0.75, 1].map((f) => { const v = xDomain[0] + f * (xDomain[1] - xDomain[0]); return { v, label: Math.round(100 * Math.sinh(v)).toString() }; });
    const yTicks: Tick[] = [0, 0.5, 1].map((v) => ({ v, label: String(v) }));
    return { series, legend, xDomain, yDomain: [0, 1.02] as [number, number], xTicks, yTicks };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [comb, activeBand, colorBy, theme]);

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
                  <span className="ens-imp__meta muted">{[r.meta?.loss, r.meta?.blocks && `${r.meta.blocks}b`, kneeTag(r.meta?.asinh_knee), r.meta?.psnr != null && `${r.meta.psnr.toFixed(1)}dB`].filter(Boolean).join(" · ")}</span>
                </div>
              ))}
            </div>

            {bands.length > 0 && (
              <div style={{ marginTop: "var(--s4)" }}>
                <div className="row" style={{ justifyContent: "space-between", marginBottom: 8, gap: "var(--s3)" }}>
                  <div className="eyebrow">gate weight vs brightness</div>
                  <div className="row" style={{ gap: 8 }}>
                    <Select<GateColorBy> value={colorBy} onChange={setColorBy}
                      options={[{ value: "loss", label: "by loss" }, { value: "psnr", label: "by PSNR" }, { value: "depth", label: "by depth" }, { value: "knee", label: "by knee" }]} />
                    <Segmented<string> value={activeBand} onChange={setBand} options={bands.map((b) => ({ value: b, label: b }))} />
                  </div>
                </div>
                {!gate ? <Empty>no gate data for {activeBand}</Empty> : (
                  <>
                    <Plot title={`${activeBand} — convex member weights`} xDomain={gate.xDomain} yDomain={gate.yDomain}
                      xTicks={gate.xTicks} yTicks={gate.yTicks} xLabel="pixel brightness [e⁻]" yLabel="gate weight"
                      series={gate.series} aspect={0.4} />
                    <Legend items={gate.legend} />
                  </>
                )}
              </div>
            )}
          </div>
        )}
      </CardBody>
    </Card>
  );
}

/* ── disagreement viewer + member panel ──────────────────────────────────────
   The tier row is trimmed to LR · SR (mean) · combiner · HR · Movie. The 22
   individual member SRs live here instead — a searchable, sortable panel that
   (a) toggles a member's SR into the viewer as a frame, and (b) selects the
   member SUBSET the disagreement movie decomposes (PCA recomputed on the fly,
   server-side, over just the checked members). */
type ViewMeta = { member_labels: string[]; count?: number; pca_max?: number };
type MemRow = {
  i: number; key: string; label: string; num: string;
  psnr: number | null; loss: string; depth: number | null; knee: number | null;
};
type SortBy = "index" | "psnr" | "loss" | "depth";

function DisagreementCard({ mode, members }: { mode: Mode; members: Member[] }) {
  const meta = useResource<ViewMeta>(`/viewer/meta/ensemble?mode=${mode}`, [mode]);
  const apiRef = useRef<ViewerApi | null>(null);
  // One selection drives the viewer: 0 → just the base tiers, 1 → that member's
  // SR as a STILL, 2+ → the disagreement MOVIE over the picked members. No
  // separate movie/view controls — the count decides.
  const [selection, setSelection] = useState<Set<number>>(new Set());
  const [q, setQ] = useState("");
  const [sortBy, setSortBy] = useState<SortBy>("index");

  // Join the viewer's member_labels (cube stack order, "NN·psnr") onto the
  // status members (facets) by dir name member_NN.
  const rows: MemRow[] = useMemo(() => {
    const byName = new Map(members.map((m) => [m.name, m]));
    return (meta.data?.member_labels ?? []).map((label, i) => {
      // Labels are "NN·psnr" (the PSNR-best checkpoint fingerprint); show just NN.
      const num = label.split("·")[0];
      const m = byName.get(`member_${num}`);
      return {
        i, key: `member${i}`, label, num,
        psnr: m?.psnr ?? null, loss: m?.loss ?? "l1",
        depth: m?.blocks ?? null, knee: m?.asinh_knee ?? null,
      };
    });
  }, [meta.data, members]);

  const shown = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const filtered = rows.filter((x) => !needle
      || x.label.toLowerCase().includes(needle) || x.loss.includes(needle));
    const cmp: Record<SortBy, (a: MemRow, b: MemRow) => number> = {
      index: (a, b) => a.i - b.i,
      psnr: (a, b) => (b.psnr ?? -Infinity) - (a.psnr ?? -Infinity),
      loss: (a, b) => a.loss.localeCompare(b.loss) || a.i - b.i,
      depth: (a, b) => (a.depth ?? 0) - (b.depth ?? 0) || a.i - b.i,
    };
    return [...filtered].sort(cmp[sortBy]);
  }, [rows, q, sortBy]);

  const csv = (s: Set<number>) => [...s].sort((a, b) => a - b).join(",");
  // Map the selection onto the viewer, preserving the user's base chip picks
  // (lr/sr/comb/hr): 2+ → the movie over the subset; 1 → that member's still;
  // 0 → base only.
  const apply = useCallback((sel: Set<number>) => {
    const api = apiRef.current;
    if (!api) return;
    const cur = api.getState().tiers;
    const base = cur.filter((t) => !/^member\d+$/.test(t) && t !== "morph");
    const arr = [...sel].sort((a, b) => a - b);
    if (arr.length >= 2) {
      api.setMorphMembers(csv(sel));
      api.setTiers([...base, "morph"]);
    } else if (arr.length === 1) {
      api.setMorphMembers(null);
      api.setTiers([...base, `member${arr[0]}`]);
    } else {
      api.setMorphMembers(null);
      api.setTiers(base.length ? base : ["sr"]);
    }
  }, []);

  const commit = (next: Set<number>) => { setSelection(next); apply(next); };
  const toggle = (i: number) => {
    const next = new Set(selection);
    if (next.has(i)) next.delete(i); else next.add(i);
    commit(next);
  };
  const topByPsnr = (k: number) => commit(new Set(
    [...rows].filter((r) => r.psnr != null)
      .sort((a, b) => b.psnr! - a.psnr!).slice(0, k).map((r) => r.i)));

  const nSel = selection.size;
  const status = nSel >= 2 ? `disagreement movie · ${nSel} members`
    : nSel === 1 ? `showing member #${rows.find((r) => selection.has(r.i))?.num} (still)`
    : "click a member — one shows a still, two+ play the movie";
  return (
    <Card>
      <CardHead title="Disagreement viewer"
        sub="LR · mean · combiner · HR · movie — pick members to see where those reconstructions disagree" />
      <CardBody>
        <CutoutViewer collection="ensemble" params={{ mode }}
          onReady={(api) => { apiRef.current = api; }} />
        <div style={{ marginTop: "var(--s4)" }}>
          <div className="row" style={{ justifyContent: "space-between", gap: "var(--s3)", marginBottom: 8, flexWrap: "wrap" }}>
            <div className="eyebrow">members ({rows.length}) · {status}</div>
            <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
              <Input value={q} onChange={setQ} placeholder="filter…" style={{ width: 120 }} />
              <Select<SortBy> value={sortBy} onChange={setSortBy}
                options={[{ value: "index", label: "by index" }, { value: "psnr", label: "by PSNR" },
                { value: "loss", label: "by loss" }, { value: "depth", label: "by depth" }]} />
              <Button size="sm" onClick={() => topByPsnr(5)} disabled={!rows.some((r) => r.psnr != null)}
                title="movie over the 5 highest-PSNR members">top 5</Button>
              <Button size="sm" onClick={() => commit(new Set())} disabled={nSel === 0}>clear</Button>
            </div>
          </div>
          {!shown.length
            ? <Empty>no members in this regime's cube cache — run an evaluation first</Empty>
            : (
              <div className="ens-mpanel"><div className="ens-mgrid">
                {shown.map((r) => (
                  <button key={r.key} type="button" className="ens-mcard" data-on={selection.has(r.i)}
                    onClick={() => toggle(r.i)}
                    title="click to add — one member shows a still, two or more play the movie">
                    <div className="ens-mcard__hd">
                      <span className="ens-mcard__num mono">#{r.num}</span>
                      <span className="ens-mcard__psnr mono">{r.psnr != null ? `${r.psnr.toFixed(1)} dB` : "—"}</span>
                    </div>
                    <div className="ens-mcard__meta mono">
                      <span style={{ color: LOSS_COLOR[r.loss] ?? "inherit" }}>{r.loss}</span>
                      {" · "}{r.depth != null ? `${r.depth}b` : "—"}{" · "}{kneeTag(r.knee)}
                    </div>
                  </button>
                ))}
              </div></div>
            )}
        </div>
      </CardBody>
    </Card>
  );
}
