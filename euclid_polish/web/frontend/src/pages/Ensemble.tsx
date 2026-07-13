/* Ensemble — the deep-ensemble control room. Members table (+ archive), live
   training curves, evaluate/pull/PSNR controls, the power-spectrum + pixel
   diagnostics, the per-band combiner (fit + gate curves), and the disagreement
   cutout viewer. Full parity with the classic page, drawn from the JSON
   endpoints (status.json / evals.json / combiner.json / training-curves.json). */
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useResource } from "../hooks";
import { asArray } from "../data";
import { useJob, JobProgressView } from "../jobs";
import { useThemeValue } from "../theme";
import { CutoutViewer, loadColorEngine, type ViewerApi, type ColorMeta, type RenderOpts } from "../legacy";
import { C, LOSS_COLOR, categorical, viridis } from "../colors";
import Plot, { Legend, type Series, type Guide, type Tick, type Heat } from "../charts/Plot";
import {
  Badge, Button, Card, CardBody, CardHead, Chip, DefList, Empty,
  NumberField, Page, PageHead, Segmented, Select, Spinner, Stat, Table,
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
  evaluations_ready?: boolean;
};

/* ── evals.json ──────────────────────────────────────────────────────────── */
type PS = {
  theta: (number | null)[]; r: (number | null)[]; r_lr?: (number | null)[];
  r_comb?: (number | null)[]; r_members?: (number | null)[][];
  r_combined?: (number | null)[];
  r_pairs?: (number | null)[][]; r_cross?: (number | null)[];
};
/* Pixel diagnostics (VIS, electrons). All *_edges / med_* arrays are already in
   the axis units the frontend plots: std_err in log10(e⁻) on both axes;
   bright_std x is asinh(HR/stretch), y is log10(e⁻). hist[i][j] = pixel count. */
type StdErr = {
  edges: (number | null)[]; hist: number[][];
  med_std: (number | null)[]; med_err: (number | null)[];
  pred?: string;   // point estimate the error is measured against ("combiner" | "ensemble mean")
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
  combined_combiner?: { available?: boolean; psnr?: number | null; ensemble_mean_psnr?: number | null; best_member_psnr?: number | null } | null;
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
  source_starless?: boolean[]; reason?: string;
};

/* ── training-curves.json ────────────────────────────────────────────────── */
/* Note: the payload overwrites the loss *series* with the loss-*norm* string,
   so only the PSNR series is chartable; `loss` here is the norm ("l1"…). */
type Curve = { name: string; psnr: [number, number][]; blocks?: number; test_psnr?: number | null; loss?: string; asinh_knee?: number | null; starless?: boolean };

const XTICKS = [0.05, 0.1, 0.2, 0.5, 1, 2, 5];
const hasData = (a: unknown) => asArray<number | null>(a).some((v) => v != null && isFinite(v));
const finite = (a: unknown, i: 0 | 1) => asArray<[number, number]>(a).map((p) => p[i]);
const num = (a: unknown) => asArray<number | null>(a).map((v) => (v == null ? NaN : v));

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

/* Legend for the member-line coloring facet: maps each distinct loss / depth /
   knee to its swatch, so "color by X" always says what each colour means. The
   depth/knee colours are categorical() over the SORTED distinct values, matching
   how the charts pick memberColor (categorical(values.indexOf(v))). */
function facetLegend(
  colorBy: ColorBy, losses: string[], depths: number[], knees: number[],
): { label: string; color: string }[] {
  if (colorBy === "loss") return losses.map((l) => ({ label: l, color: LOSS_COLOR[l] ?? C.muted }));
  if (colorBy === "depth") return depths.map((d, i) => ({ label: `${d}b`, color: categorical(i) }));
  if (colorBy === "knee") return knees.map((k, i) => ({ label: kneeTag(k), color: categorical(i) }));
  return [];
}

const DIAG_TABS = [
  { id: "power-spectrum", label: "power spectrum" },
  { id: "std-error", label: "std vs error" },
  { id: "std-brightness", label: "std vs brightness" },
] as const;
type DiagTab = typeof DIAG_TABS[number]["id"];

/* ── pixel back-tracing (click a heatmap cell → real image stamps) ─────────── */
type PickDiag = "std_err" | "bright_std";
/* Stamps arrive as base64 little-endian float32 blobs (one per tier): the
   full N-band LR/HR/SR cubes (rendered with the field viewer's colour engine)
   and the single-band σ. */
type Stamp = {
  field: number; y: number; x: number; center: number; sr_is_combiner: boolean;
  lr?: string; hr: string; sr: string; std: string;
  hr_val: number; sr_val: number; std_val: number; err_val: number; bright_asinh: number;
};
type Trace = {
  diag: string; i: number; j: number; half: number; size: number;
  bands: string[]; stretch: number; stamps: Stamp[];
};
type RenderFn = (rec: { data: Float32Array; h: number; w: number; c: number },
  colorMeta: ColorMeta, opts: RenderOpts) => ImageData;
type ViewerMetaColor = { color?: ColorMeta };

const COLOR_OPTS = [
  { value: "VIS", label: "VIS" }, { value: "Y_E", label: "Y_E" },
  { value: "J_E", label: "J_E" }, { value: "H_E", label: "H_E" },
  { value: "lupton", label: "Lupton" }, { value: "temp", label: "Temp" },
];

/* Compact electron formatter for the stamp captions / cell ranges. */
const fmtE = (v: number): string =>
  !isFinite(v) ? "—" : Math.abs(v) >= 1000 || (Math.abs(v) > 0 && Math.abs(v) < 0.01)
    ? v.toExponential(1) : String(Number(v.toPrecision(3)));
/* "10^lo–10^hi" range for cell k of a log10-valued edge array. */
const logRange = (edges: number[], k: number): string =>
  `${fmtE(10 ** edges[k])}–${fmtE(10 ** edges[k + 1])}`;

function b64ToF32(b64: string): Float32Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

/* Draw one S×S ImageData scaled up ×px (pixelated) with the center pixel ringed. */
function blitStamp(cv: HTMLCanvasElement, img: ImageData, size: number, center: number, px: number) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  cv.width = size * px * dpr; cv.height = size * px * dpr;
  cv.style.width = size * px + "px"; cv.style.height = size * px + "px";
  const off = document.createElement("canvas");
  off.width = size; off.height = size;
  off.getContext("2d")!.putImageData(img, 0, 0);
  const ctx = cv.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, size * px, size * px);
  ctx.drawImage(off, 0, 0, size * px, size * px);
  ctx.strokeStyle = "#ff3b6b"; ctx.lineWidth = 1.4;
  ctx.strokeRect(center * px - 0.7, center * px - 0.7, px + 1.4, px + 1.4);
}

/* An LR/HR/SR image stamp, rendered with the field viewer's colour/knee/gain. */
function ImageStamp(
  { b64, size, center, bands, colorMeta, opts, render, px, label, badge }:
  { b64?: string; size: number; center: number; bands: string[]; colorMeta?: ColorMeta;
    opts: RenderOpts; render: RenderFn | null; px: number; label: string; badge?: string },
) {
  const c = bands?.length || 0;
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv || !b64 || !render || !colorMeta || !c) return;
    const data = b64ToF32(b64);
    const img = render({ data, h: size, w: size, c }, colorMeta, opts);
    blitStamp(cv, img, size, center, px);
  }, [b64, size, center, c, colorMeta, opts, render, px]);
  return (
    <div style={{ textAlign: "center" }}>
      {b64 ? <canvas ref={ref} style={{ imageRendering: "pixelated", borderRadius: 3, display: "block" }} />
        : <div className="ens-trace__na" style={{ width: size * px, height: size * px }}>n/a</div>}
      <div className="mono ens-trace__tier">{label}{badge && <b> · {badge}</b>}</div>
    </div>
  );
}

/* The σ (disagreement) stamp — a scalar map in fixed viridis, asinh + per-stamp
   normalized (unaffected by the colour/knee controls: it IS the thing shown). */
function SigmaStamp(
  { b64, size, center, stretch, px, label }:
  { b64: string; size: number; center: number; stretch: number; px: number; label: string },
) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const a = b64ToF32(b64);
    const t = new Float32Array(a.length);
    let amin = Infinity, amax = -Infinity;
    for (let i = 0; i < a.length; i++) { const v = Math.asinh(a[i] / stretch); t[i] = v; if (v < amin) amin = v; if (v > amax) amax = v; }
    const span = amax - amin || 1;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = size * px * dpr; cv.height = size * px * dpr;
    cv.style.width = size * px + "px"; cv.style.height = size * px + "px";
    const ctx = cv.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      ctx.fillStyle = viridis((t[y * size + x] - amin) / span);
      ctx.fillRect(x * px, y * px, px + 0.5, px + 0.5);
    }
    ctx.strokeStyle = "#ff3b6b"; ctx.lineWidth = 1.4;
    ctx.strokeRect(center * px - 0.7, center * px - 0.7, px + 1.4, px + 1.4);
  }, [b64, size, center, stretch, px]);
  return (
    <div style={{ textAlign: "center" }}>
      <canvas ref={ref} style={{ imageRendering: "pixelated", borderRadius: 3, display: "block" }} />
      <div className="mono ens-trace__tier">{label}</div>
    </div>
  );
}

function TraceHint() {
  return (
    <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
      Click any cell to back-trace it — a handful of the real pixels that landed there, as
      stamps (LR · HR · SR/combiner · σ) from across the test fields, with the field viewer's colour controls.
    </div>
  );
}

const STAMP_PX = 3;

function PixelTrace(
  { mode, pick, cellLabel, colorMeta, onClose }:
  { mode: Mode; pick: { diag: PickDiag; i: number; j: number }; cellLabel: string;
    colorMeta?: ColorMeta; onClose: () => void },
) {
  const url = `/ensemble/pixel-trace.json?mode=${mode}&diag=${pick.diag}&i=${pick.i}&j=${pick.j}`;
  const trace = useResource<Trace>(url, [mode, pick.diag, pick.i, pick.j]);
  const t = trace.data;
  const stamps = asArray<Stamp>(t?.stamps);
  const traceBands = asArray<string>(t?.bands);

  // Load the viewer's colour engine once; render null until it's ready.
  const [render, setRender] = useState<RenderFn | null>(null);
  useEffect(() => { let live = true; loadColorEngine().then((fn) => { if (live) setRender(() => fn as RenderFn); }); return () => { live = false; }; }, []);

  // Viewer-parity controls: colour mode, asinh knee (K), brightness (gain). K0
  // (the white reference) is pinned to the served default, as the viewer does.
  const K0 = colorMeta?.default_asinh ?? 100;
  const [color, setColor] = useState("VIS");
  const [knee, setKnee] = useState(String(K0));
  const [gain, setGain] = useState("1");
  const opts: RenderOpts = useMemo(
    () => ({ color, knee: Number(knee) || K0, gain: Number(gain) || 1, K0 }),
    [color, knee, gain, K0]);

  return (
    <div className="ens-trace">
      <div className="row" style={{ justifyContent: "space-between", alignItems: "baseline", marginBottom: 8, flexWrap: "wrap", gap: 8 }}>
        <div className="eyebrow">back-trace · <span className="mono" style={{ textTransform: "none" }}>{cellLabel}</span></div>
        <div className="row" style={{ gap: 8, alignItems: "flex-end", flexWrap: "wrap" }}>
          <Select value={color} onChange={setColor} options={COLOR_OPTS} />
          <NumberField label="asinh knee [e⁻]" value={knee} onChange={setKnee} min={5} max={5000} />
          <NumberField label="brightness ×" value={gain} onChange={setGain} min={0.1} max={10} step={0.1} />
          <Button size="sm" variant="ghost" onClick={onClose}>✕ close</Button>
        </div>
      </div>
      {trace.loading ? <Empty><Spinner /> pulling stamps…</Empty>
        : !t || !stamps.length
          ? <Empty>no pixels were sampled in this cell — try a denser (brighter-colored) one</Empty>
          : (
            <div className="ens-trace__grid">
              {stamps.map((s, k) => (
                <div key={k} className="ens-trace__card">
                  <div className="ens-trace__stamps">
                    <ImageStamp b64={s.lr} size={t.size} center={s.center} bands={traceBands}
                      colorMeta={colorMeta} opts={opts} render={render} px={STAMP_PX} label="LR" />
                    <ImageStamp b64={s.hr} size={t.size} center={s.center} bands={traceBands}
                      colorMeta={colorMeta} opts={opts} render={render} px={STAMP_PX} label="HR" />
                    <ImageStamp b64={s.sr} size={t.size} center={s.center} bands={traceBands}
                      colorMeta={colorMeta} opts={opts} render={render} px={STAMP_PX}
                      label="SR" badge={s.sr_is_combiner ? "combiner" : "mean"} />
                    <SigmaStamp b64={s.std} size={t.size} center={s.center} stretch={t.stretch} px={STAMP_PX} label="σ" />
                  </div>
                  <div className="mono ens-trace__nums">
                    #{s.field} · σ={fmtE(s.std_val)} · |err|={fmtE(s.err_val)} · HR={fmtE(s.hr_val)} e⁻
                  </div>
                </div>
              ))}
            </div>
          )}
    </div>
  );
}

export default function EnsemblePage() {
  const theme = useThemeValue();
  // Star regime lives in the URL (/ensemble/starfull | /ensemble/starless).
  const navigate = useNavigate();
  const { mode: modeParam } = useParams<{ mode: string }>();
  const mode: Mode = modeParam === "starless" ? "starless" : "starfull";
  const setMode = (m: Mode) => navigate(`/ensemble/${m}`);
  const starless = mode === "starless";

  const status = useResource<Status>(`/ensemble/status.json?mode=${mode}`, [mode]);
  const evals = useResource<Evals>(`/ensemble/evals.json?mode=${mode}`, [mode]);
  const comb = useResource<Combiner>(`/ensemble/combiner.json?mode=${mode}`, [mode]);
  const combined = useResource<Combiner>(`/ensemble/combined-combiner.json?mode=${mode}`, [mode]);
  const curves = useResource<{ members?: Curve[] }>("/ensemble/training-curves.json");

  const evalJob = useJob();
  const opJob = useJob();
  const fitJob = useJob();
  const combinedFitJob = useJob();
  // Continue/fork buttons in the members table jump to the Train members page,
  // prefilled via query params.
  const toTrain = (m: "continue" | "fork", member: string) =>
    navigate(`/train-members?mode=${m}&member=${encodeURIComponent(member)}`);

  const reloadAll = () => { status.reload(); evals.reload(); comb.reload(); combined.reload(); curves.reload(); };

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
          onContinue={(m) => toTrain("continue", m)}
          onFork={(m) => toTrain("fork", m)} />
        <TrainingCurves curves={asArray<Curve>(curves.data?.members)} starless={starless} />
        <Evaluations evals={evals.data} loading={evals.loading} mode={mode} theme={theme} />
        <CombinerCard comb={comb.data} loading={comb.loading} mode={mode} theme={theme} fitJob={fitJob} onFit={reloadAll}
          evalReady={status.data?.evaluations_ready ?? false} />
        <DisagreementCard key={mode} mode={mode} members={asArray<Member>(status.data?.members)} />
        <CombinerCard comb={combined.data} loading={combined.loading} mode={mode} theme={theme} fitJob={combinedFitJob} onFit={reloadAll}
          evalReady={status.data?.evaluations_ready ?? false} combined />
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
  const rows = asArray<Member>(status?.members).filter((m) => !!m.starless === starless);
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
    const losses = [...new Set(rows.map((c) => c.loss ?? "l1"))].sort();
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
    return { series, xDomain: [0, xMax] as [number, number], yDomain, xTicks, yTicks,
             legend: facetLegend(colorBy, losses, depths, knees) };
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
          <>
            <Plot title="validation PSNR (asinh) vs step"
              xDomain={chart.xDomain} yDomain={chart.yDomain} xTicks={chart.xTicks} yTicks={chart.yTicks}
              xLabel="training step" yLabel="PSNR [dB]" series={chart.series} aspect={0.4} />
            {chart.legend.length > 0 && <Legend items={chart.legend} />}
          </>
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
  // Which overlay curves to draw on the power spectrum.
  const [show, setShow] = useState({ cross: true, mean: true, comb: true, combined: true });
  const toggle = (k: keyof typeof show) => setShow((s) => ({ ...s, [k]: !s[k] }));
  // Back-traced heatmap cell (click-to-inspect). Cleared when the tab/regime
  // changes — a cell only means something within one diagnostic.
  const [pick, setPick] = useState<{ diag: PickDiag; i: number; j: number } | null>(null);
  useEffect(() => setPick(null), [tab, mode]);
  // Viewer colour meta (band constants) so back-trace stamps colour like the viewer.
  const vmeta = useResource<ViewerMetaColor>(`/viewer/meta/ensemble?mode=${mode}`, [mode]);
  const colorMeta = vmeta.data?.color;
  const ps = evals?.ps ?? null;
  const members = asArray<NonNullable<Evals["members"]>[number]>(evals?.members);

  const chart = useMemo(() => {
    if (!ps || !hasData(ps.theta)) return null;
    const theta = num(ps.theta);
    const xs = theta.filter((v) => isFinite(v));
    const g = evals?.guides ?? {};
    const xDomain: [number, number] = [g.theta_min ?? 0.05, Math.max(...xs)];
    const xTicks: Tick[] = XTICKS.filter((v) => v >= xDomain[0] && v <= xDomain[1]).map((v) => ({ v, label: String(v) }));
    const yTicks: Tick[] = [0, 0.25, 0.5, 0.75, 1].map((v) => ({ v, label: String(v) }));
    const depths = [...new Set(members.map((mm) => mm?.blocks ?? 0))].sort((a, b) => a - b);
    const knees = [...new Set(members.map((mm) => kneeOf(mm?.asinh_knee)))].sort((a, b) => a - b);
    const losses = [...new Set(members.map((mm) => mm?.loss ?? "l1"))].sort();
    const memberColor = (i: number) => {
      const mm = members[i];
      if (colorBy === "loss") return LOSS_COLOR[mm?.loss ?? "l1"] ?? C.muted;
      if (colorBy === "depth") return categorical(depths.indexOf(mm?.blocks ?? 0));
      if (colorBy === "knee") return categorical(knees.indexOf(kneeOf(mm?.asinh_knee)));
      return C.muted;
    };
    const grouped = colorBy !== "uniform";
    const series: Series[] = [];
    // Member↔member cross-correlation (pairwise cloud + the model–model r̃(k)).
    if (show.cross) {
      for (const pair of asArray<(number | null)[]>(ps.r_pairs)) series.push({ x: theta, y: num(pair), color: C.muted, width: 0.7, alpha: 0.18 });
      if (hasData(ps.r_cross)) series.push({ x: theta, y: ps.r_cross!, color: C.cross, width: 2, dash: [6, 3] });
    }
    asArray<(number | null)[]>(ps.r_members).forEach((row, i) => series.push({ x: theta, y: num(row), color: memberColor(i), width: 1, alpha: grouped ? 0.6 : 0.4 }));
    if (hasData(ps.r_lr)) series.push({ x: theta, y: ps.r_lr!, color: C.baseline, width: 2.5, dash: [7, 4] });
    if (show.mean && hasData(ps.r)) series.push({ x: theta, y: num(ps.r), color: C.mean, width: 2.6, dots: true });
    if (show.comb && hasData(ps.r_comb)) series.push({ x: theta, y: ps.r_comb!, color: C.comb, width: 2.2, dots: true });
    if (show.combined && hasData(ps.r_combined)) series.push({ x: theta, y: ps.r_combined!, color: "#9d6cff", width: 2.2, dots: true });
    const guides: Guide[] = [
      { axis: "y", v: 1, color: C.guide, dash: [2, 3] },
      { axis: "x", v: g.lr_scale ?? 0.1, color: C.guide, width: 1.3, dash: [6, 3] },
      { axis: "x", v: g.vis_fwhm ?? 0.16, color: C.visfwhm, alpha: 0.5, width: 1.5, dash: [5, 2] },
    ];
    // When coloured by a facet, the legend spells out each loss/depth/knee
    // swatch instead of a single grey "individual models" entry.
    const facet = facetLegend(colorBy, losses, depths, knees);
    const legend = [
      ...(hasData(ps.r_lr) ? [{ label: "LR baseline", color: C.baseline, dash: true }] : []),
      ...(show.mean ? [{ label: "ensemble mean", color: C.mean }] : []),
      ...(show.comb && hasData(ps.r_comb) ? [{ label: "combiner", color: C.comb }] : []),
      ...(show.combined && hasData(ps.r_combined) ? [{ label: "combined combiner", color: "#9d6cff" }] : []),
      ...(facet.length ? facet : [{ label: "individual models", color: C.muted }]),
      ...(show.cross && hasData(ps.r_cross) ? [{ label: "model–model r̃(k)", color: C.cross, dash: true }] : []),
    ];
    return { series, guides, xDomain, yDomain: [0, 1.05] as [number, number], xTicks, yTicks, legend };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ps, members, colorBy, show, evals, theme]);

  // std vs error — density cloud + median-|error|-per-std curve, both axes
  // log10(e⁻). The |err|=σ diagonal reference rides as a 2-pt series.
  const stdErr = useMemo(() => {
    const d = evals?.std_err;
    const hist = asArray<number[]>(d?.hist);
    if (!d || !hist.length) return null;
    const edges = num(d.edges);
    if (edges.length < 2) return null;
    const lo = edges[0], hi = edges[edges.length - 1];
    const heat: Heat = { z: hist, xEdges: edges, yEdges: edges };
    const series: Series[] = [
      { x: [lo, hi], y: [lo, hi], color: C.guide, width: 1.3, dash: [6, 3] },
      { x: num(d.med_std), y: num(d.med_err), color: C.baseline, width: 2.6, dots: true },
    ];
    const ticks = logDecadeTicks(lo, hi);
    const legend = [
      { label: "median |error| per σ bin", color: C.baseline },
      { label: "|error| = σ", color: C.guide, dash: true },
    ];
    // Human range for a clicked cell (both axes log10 e⁻).
    const describe = (c: { i: number; j: number }) =>
      `σ ${logRange(edges, c.i)} e⁻ · |err| ${logRange(edges, c.j)} e⁻`;
    const pred = d.pred || "ensemble mean";
    const yLabel = `|${pred} − HR|  [e⁻]`;
    return { heat, series, domain: [lo, hi] as [number, number], ticks, legend, describe, yLabel };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [evals, theme]);

  // std vs brightness — density cloud + median σ (with 16–84%) per HR-brightness
  // bin. x = asinh(HR/stretch), y = log10 σ(e⁻).
  const brightStd = useMemo(() => {
    const d = evals?.bright_std;
    const hist = asArray<number[]>(d?.hist);
    if (!d || !hist.length) return null;
    const bx = num(d.bright_edges), sy = num(d.std_edges);
    if (bx.length < 2 || sy.length < 2) return null;
    const heat: Heat = { z: hist, xEdges: bx, yEdges: sy };
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
    // Human range for a clicked cell: x = HR brightness (asinh), y = σ (log10).
    const st = d.stretch;
    const describe = (c: { i: number; j: number }) =>
      `HR ${fmtE(st * Math.sinh(bx[c.i]))}–${fmtE(st * Math.sinh(bx[c.i + 1]))} e⁻` +
      ` · σ ${logRange(sy, c.j)} e⁻`;
    return {
      heat, series,
      xDomain: [bx[0], bx[bx.length - 1]] as [number, number],
      yDomain: [sy[0], sy[sy.length - 1]] as [number, number],
      xTicks: brightTicks(d.stretch), yTicks: logDecadeTicks(sy[0], sy[sy.length - 1]), legend, describe,
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
                <div className="row" style={{ justifyContent: "space-between", marginBottom: 8, gap: 8, flexWrap: "wrap" }}>
                  <div className="row" style={{ gap: 6, flexWrap: "wrap" }}>
                    <Chip on={show.cross} dot={C.cross} onClick={() => toggle("cross")}
                      title="member↔member cross-correlation curves">cross-corr</Chip>
                    <Chip on={show.mean} dot={C.mean} onClick={() => toggle("mean")}
                      title="ensemble-mean r(k)">mean</Chip>
                    {hasData(ps?.r_comb) && <Chip on={show.comb} dot={C.comb} onClick={() => toggle("comb")}
                      title="combiner r(k)">combiner</Chip>}
                    {hasData(ps?.r_combined) && <Chip on={show.combined} dot="#9d6cff" onClick={() => toggle("combined")}
                      title="cross-regime combined-combiner r(k)">combined combiner</Chip>}
                  </div>
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
                  xLabel="cross-member per-pixel σ  [e⁻]" yLabel={stdErr.yLabel}
                  heat={stdErr.heat} series={stdErr.series} aspect={0.62}
                  onHeatClick={(c) => setPick({ diag: "std_err", ...c })}
                  highlight={pick?.diag === "std_err" ? pick : null} />
                <Legend items={stdErr.legend} />
                <TraceHint />
                {pick?.diag === "std_err" &&
                  <PixelTrace mode={mode} pick={pick} cellLabel={stdErr.describe(pick)}
                    colorMeta={colorMeta} onClose={() => setPick(null)} />}
              </>
            )
          ) : (
            !brightStd ? <Empty>no evaluation cached for <b>{mode}</b> — run “Evaluate on test set”.</Empty> : (
              <>
                <Plot title="Where does disagreement live?  (VIS, per pixel)"
                  xDomain={brightStd.xDomain} yDomain={brightStd.yDomain} xTicks={brightStd.xTicks} yTicks={brightStd.yTicks}
                  xLabel="HR pixel brightness  [e⁻]  (asinh axis)" yLabel="cross-member per-pixel σ  [e⁻]"
                  heat={brightStd.heat} series={brightStd.series} aspect={0.62}
                  onHeatClick={(c) => setPick({ diag: "bright_std", ...c })}
                  highlight={pick?.diag === "bright_std" ? pick : null} />
                <Legend items={brightStd.legend} />
                <TraceHint />
                {pick?.diag === "bright_std" &&
                  <PixelTrace mode={mode} pick={pick} cellLabel={brightStd.describe(pick)}
                    colorMeta={colorMeta} onClose={() => setPick(null)} />}
              </>
            )
          )}
      </CardBody>
    </Card>
  );
}

/* ── combiner ────────────────────────────────────────────────────────────── */
type GateColorBy = "loss" | "psnr" | "depth" | "knee" | "regime";

function CombinerCard(
  { comb, loading, mode, theme, fitJob, onFit, evalReady, combined = false }:
  { comb: Combiner | null; loading: boolean; mode: Mode; theme: string; fitJob: ReturnType<typeof useJob>; onFit: () => void; evalReady: boolean; combined?: boolean },
) {
  const [nImg, setNImg] = useState("100");
  const [nKernels, setNKernels] = useState("12");
  const [minUsage, setMinUsage] = useState("0");
  const [band, setBand] = useState<string>("");
  const [colorBy, setColorBy] = useState<GateColorBy>("loss");

  const bands = asArray<string>(comb?.band_names);
  const activeBand = band || bands[0] || "";

  const importance = useMemo(() => {
    if (!comb?.eff_weights) return [];
    const labels = asArray<string>(comb.member_labels);
    const memberMeta = asArray<NonNullable<Combiner["members"]>[number]>(comb.members);
    const m = labels.length;
    const cum = new Array(m).fill(0);
    for (const b of Object.values(comb.eff_weights)) {
      const jac = asArray<(number | null)[]>(b?.jacobian);
      const L = jac.length || 1;
      for (const row of jac) row?.forEach((w, i) => { if (w != null && isFinite(w)) cum[i] += w / L; });
    }
    const surv = labels.map((_, i) => Object.values(comb.surviving ?? {}).some((arr) => asArray<boolean>(arr)[i] !== false));
    return labels
      .map((label, i) => ({ i, label, total: cum[i], kept: surv[i], meta: memberMeta[i] }))
      .filter((r) => r.kept)
      .sort((a, b2) => b2.total - a.total);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [comb, theme]);

  // Shared facet colorer (member index → colour by `colorBy`) so the importance
  // bars and the gate-weight curves colour a member the SAME way — and both
  // recolour when the characteristic changes. depth/knee → categorical over the
  // ensemble's distinct values; psnr → viridis over the surviving members' PSNR
  // range; loss → loss token.
  const memberColor = useMemo(() => {
    const members = asArray<NonNullable<Combiner["members"]>[number]>(comb?.members);
    const M = asArray<string>(comb?.member_labels).length;
    const survIdx = Array.from({ length: M }, (_, m) => m)
      .filter((m) => Object.values(comb?.surviving ?? {}).some((arr) => asArray<boolean>(arr)[m] !== false));
    const depths = [...new Set(members.map((mm) => mm?.blocks ?? 0))].sort((a, b) => a - b);
    const knees = [...new Set(members.map((mm) => kneeOf(mm?.asinh_knee)))].sort((a, b) => a - b);
    const psnrs = survIdx.map((m) => members[m]?.psnr).filter((p): p is number => p != null && isFinite(p));
    const pMin = psnrs.length ? Math.min(...psnrs) : 0;
    const pMax = psnrs.length ? Math.max(...psnrs) : 1;
    return (m: number): string => {
      const meta = members[m];
      if (colorBy === "loss") return LOSS_COLOR[meta?.loss ?? "l1"] ?? C.muted;
      if (colorBy === "depth") return categorical(depths.indexOf(meta?.blocks ?? 0));
      if (colorBy === "knee") return categorical(knees.indexOf(kneeOf(meta?.asinh_knee)));
      if (colorBy === "regime") return comb?.source_starless?.[m] ? "#9d6cff" : "#ee7733";
      const p = meta?.psnr;
      return p == null || pMax === pMin ? C.mean : viridis((p - pMin) / (pMax - pMin));
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [comb, colorBy, theme]);

  const gate = useMemo(() => {
    const ew = comb?.eff_weights?.[activeBand];
    const jacobian = asArray<(number | null)[]>(ew?.jacobian);
    if (!ew || !jacobian.length) return null;
    const bx = asArray<number | null>(ew.brightness_e).map((e) => (e == null ? NaN : Math.asinh(e / 100)));
    const surv = asArray<boolean>(comb?.surviving?.[activeBand]);
    const members = asArray<NonNullable<Combiner["members"]>[number]>(comb?.members);
    const labels = asArray<string>(comb?.member_labels);
    const M = labels.length;
    const survIdx = Array.from({ length: M }, (_, m) => m).filter((m) => surv[m] !== false);

    const xs = bx.filter((v) => isFinite(v));
    if (!xs.length) return null;
    const xDomain: [number, number] = [Math.min(...xs), Math.max(...xs)];
    const series: Series[] = survIdx.map((m) => ({
      x: bx, y: jacobian.map((row) => (asArray<number | null>(row)[m] ?? null)), color: memberColor(m), width: 1.8,
    }));
    const legend = survIdx.map((m) => {
      const meta = members[m];
      const tag = colorBy === "loss" ? (meta?.loss ?? "l1")
        : colorBy === "depth" ? `${meta?.blocks ?? "?"}b`
        : colorBy === "knee" ? kneeTag(meta?.asinh_knee)
        : colorBy === "regime" ? (comb?.source_starless?.[m] ? "starless" : "starfull")
        : (meta?.psnr != null ? `${meta.psnr.toFixed(1)}dB` : "—");
      return { label: `${labels[m]} · ${tag}`, color: memberColor(m) };
    });
    const xTicks: Tick[] = [0, 0.25, 0.5, 0.75, 1].map((f) => { const v = xDomain[0] + f * (xDomain[1] - xDomain[0]); return { v, label: Math.round(100 * Math.sinh(v)).toString() }; });
    const yTicks: Tick[] = [0, 0.5, 1].map((v) => ({ v, label: String(v) }));
    return { series, legend, xDomain, yDomain: [0, 1.02] as [number, number], xTicks, yTicks };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [comb, activeBand, colorBy, memberColor, theme]);

  return (
    <Card>
      <CardHead title={`${combined ? "Combined combiner" : "Combiner"} · ${mode}`}
        sub={combined ? "experimental all-member brightness gate — fit locally on validate, scored on test" : "a per-band brightness gate fusing members — fit locally on validate, scored on test"}
        right={comb?.available && <Badge tone={comb.stale ? "warn" : "good"}>{comb.stale ? "stale" : "fitted"}</Badge>} />
      <CardBody>
        <div className="row" style={{ alignItems: "flex-end", gap: "var(--s3)" }}>
          <NumberField label="validate fields" value={nImg} onChange={setNImg} min={1} max={2000} />
          <NumberField label="kernels (K)" value={nKernels} onChange={setNKernels} min={2} max={32} />
          <NumberField label="prune (min importance)" value={minUsage} onChange={setMinUsage} min={0} max={0.5} step={0.01} />
          <Button variant="primary" disabled={fitJob.busy || !evalReady}
            title={evalReady ? undefined : `evaluate ${mode} on the test set first — its evaluation must match the current members`}
            onClick={() => fitJob.run(combined ? "/ensemble/combined-combiner/fit" : "/ensemble/combiner/fit", { num_images: nImg, n_kernels: nKernels, min_usage: minUsage, mode }, { onDone: onFit })}>
            Fit {combined ? "combined combiner" : "combiner"}
          </Button>
        </div>
        {!evalReady && (
          <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
            Evaluations for <b>{mode}</b> aren’t ready — run “Evaluate on test set” (for the current members) before fitting a combiner.
          </div>
        )}
        <JobProgressView job={fitJob.job} error={fitJob.error} />

        {loading ? <Empty><Spinner /> loading…</Empty>
          : !comb?.available ? <Empty>{comb?.reason ?? `no ${combined ? "combined " : ""}combiner fitted for ${mode} yet — set knobs above and fit.`}</Empty> : (
          <div style={{ marginTop: "var(--s4)" }}>
            <DefList items={[
              ["members", `${asArray<string>(comb.member_labels).length} (${importance.length} surviving)`],
              ["kernels", String(comb.n_kernels ?? "—")],
              ["prune ≥", String(comb.min_usage ?? 0)],
              ["validate L1", comb.val_l1 != null ? comb.val_l1.toFixed(4) : "—"],
            ]} />

            <div style={{ marginTop: "var(--s4)" }}>
              <div className="row" style={{ justifyContent: "space-between", marginBottom: 8, gap: "var(--s3)" }}>
                <div className="eyebrow">member importance (cumulative gate weight)</div>
                <Select<GateColorBy> value={colorBy} onChange={setColorBy}
                  options={[{ value: "loss", label: "by loss" }, { value: "psnr", label: "by PSNR" }, { value: "depth", label: "by depth" }, { value: "knee", label: "by knee" }, ...(combined ? [{ value: "regime" as GateColorBy, label: "by star regime" }] : [])]} />
              </div>
              {importance.map((r) => (
                <div key={r.label} className="ens-imp">
                  <span className="ens-imp__label mono">{r.label}</span>
                  <div className="ens-imp__bar"><div className="ens-imp__fill" style={{ width: `${Math.min(100, (r.total / (importance[0]?.total || 1)) * 100)}%`, background: memberColor(r.i) }} /></div>
                  <span className="ens-imp__val mono">{r.total.toFixed(2)}</span>
                  <span className="ens-imp__meta muted">{[r.meta?.loss, r.meta?.blocks && `${r.meta.blocks}b`, kneeTag(r.meta?.asinh_knee), r.meta?.psnr != null && `${r.meta.psnr.toFixed(1)}dB`].filter(Boolean).join(" · ")}</span>
                </div>
              ))}
            </div>

            {bands.length > 0 && (
              <div style={{ marginTop: "var(--s4)" }}>
                <div className="row" style={{ justifyContent: "space-between", marginBottom: 8, gap: "var(--s3)" }}>
                  <div className="eyebrow">gate weight vs brightness</div>
                  <Segmented<string> value={activeBand} onChange={setBand} options={bands.map((b) => ({ value: b, label: b }))} />
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
  // ttl:0 → always background-revalidate. The cube cache is wiped+rebuilt during
  // an evaluation, so a meta cached mid-run is briefly empty; revalidating on
  // every mount lets the panel self-heal (shows cached instantly, then refreshes
  // to the real membership) instead of staying stuck on "no members".
  const meta = useResource<ViewMeta>(`/viewer/meta/ensemble?mode=${mode}`, [mode], { ttl: 0 });
  const apiRef = useRef<ViewerApi | null>(null);
  // One selection drives the viewer: 0 → just the base tiers, 1 → that member's
  // SR as a STILL, 2+ → the disagreement MOVIE over the picked members. No
  // separate movie/view controls — the count decides.
  const [selection, setSelection] = useState<Set<number>>(new Set());
  const [lossFilter, setLossFilter] = useState<Set<string>>(new Set());
  const [sortBy, setSortBy] = useState<SortBy>("index");

  // Join the viewer's member_labels (cube stack order, "NN·psnr") onto the
  // status members (facets) by dir name member_NN.
  const rows: MemRow[] = useMemo(() => {
    const byName = new Map(members.map((m) => [m.name, m]));
    return asArray<string>(meta.data?.member_labels).map((label, i) => {
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

  // Distinct loss norms present, for the filter tags.
  const losses = useMemo(
    () => [...new Set(rows.map((r) => r.loss))].sort(), [rows]);

  const shown = useMemo(() => {
    const filtered = rows.filter((x) => !lossFilter.size || lossFilter.has(x.loss));
    const cmp: Record<SortBy, (a: MemRow, b: MemRow) => number> = {
      index: (a, b) => a.i - b.i,
      psnr: (a, b) => (b.psnr ?? -Infinity) - (a.psnr ?? -Infinity),
      loss: (a, b) => a.loss.localeCompare(b.loss) || a.i - b.i,
      depth: (a, b) => (a.depth ?? 0) - (b.depth ?? 0) || a.i - b.i,
    };
    return [...filtered].sort(cmp[sortBy]);
  }, [rows, lossFilter, sortBy]);

  const toggleLoss = (l: string) => setLossFilter((s) => {
    const n = new Set(s);
    if (n.has(l)) n.delete(l); else n.add(l);
    return n;
  });

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
              {losses.length > 1 && losses.map((l) => (
                <Chip key={l} on={lossFilter.has(l)} dot={LOSS_COLOR[l]} onClick={() => toggleLoss(l)}
                  title={`show only ${l} members`}>{l}</Chip>
              ))}
              <Select<SortBy> value={sortBy} onChange={setSortBy}
                options={[{ value: "index", label: "by index" }, { value: "psnr", label: "by PSNR" },
                { value: "loss", label: "by loss" }, { value: "depth", label: "by depth" }]} />
              <Button size="sm" onClick={() => topByPsnr(5)} disabled={!rows.some((r) => r.psnr != null)}
                title="movie over the 5 highest-PSNR members">top 5</Button>
              <Button size="sm" onClick={() => commit(new Set())} disabled={nSel === 0}>clear</Button>
              <Button size="sm" onClick={() => { meta.reload(); apiRef.current?.reload(); }}
                title="re-pull the cube cache (after an evaluation finishes)">↻</Button>
            </div>
          </div>
          {!shown.length
            ? <Empty>no members in this regime's cube cache — run an evaluation first</Empty>
            : (
              <div className="ens-mpanel"><div className="ens-mgrid">
                {shown.map((r) => (
                  <button key={r.key} type="button" className="ens-mcard" data-on={selection.has(r.i)}
                    style={{ "--lc": LOSS_COLOR[r.loss] } as CSSProperties}
                    onClick={() => toggle(r.i)}
                    title="click to add — one member shows a still, two or more play the movie">
                    <div className="ens-mcard__hd">
                      <span className="ens-mcard__num mono">#{r.num}</span>
                      <span className="ens-mcard__psnr mono">{r.psnr != null ? `${r.psnr.toFixed(1)} dB` : "—"}</span>
                    </div>
                    <div className="ens-mcard__meta mono">
                      <span>{r.loss}</span>
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
