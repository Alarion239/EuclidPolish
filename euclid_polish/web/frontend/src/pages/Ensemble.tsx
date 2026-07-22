/* Ensemble — the deep-ensemble control room. Members table (+ archive), live
   training curves, evaluate/pull/PSNR controls, the power-spectrum + pixel
   diagnostics, the per-band combiner (fit + gate curves), and the disagreement
   cutout viewer. Full parity with the classic page, drawn from the JSON
   endpoints (status.json / evals.json / combiner.json / training-curves.json). */
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useResource } from "../hooks";
import { asArray } from "../data";
import { useJob, useTrackedJob, JobProgressView } from "../jobs";
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
export type Member = {
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
  r_stats_rbf_comb?: (number | null)[];
  r_combined?: (number | null)[];
  r_pairs?: (number | null)[][]; r_cross?: (number | null)[];
  model_combiners?: Record<string, { r?: (number | null)[]; T?: (number | null)[] }>;
};
type CoherenceScore = {
  id: string; label: string;
  overall: number | null; sr: number | null;
  overall_lo?: number | null; overall_hi?: number | null;
  sr_lo?: number | null; sr_hi?: number | null;
  n_fields?: number;
};
type Coherence = {
  metric?: string; measure?: string;
  domains?: { overall?: { k_min?: number; k_max?: number; theta_min?: number; theta_max?: number };
    sr?: { k_min?: number; k_max?: number; theta_min?: number; theta_max?: number } };
  scores?: CoherenceScore[];
};
/* Pixel diagnostics (VIS, electrons). All *_edges / med_* arrays are already in
   the axis units the frontend plots: std_err in log10(e⁻) on both axes;
   bright_std x is asinh(HR/stretch), y is log10(e⁻). hist[i][j] = pixel count. */
type StdErrModel = {
  edges: (number | null)[]; hist: number[][];
  med_std: (number | null)[]; med_err: (number | null)[];
  n_fields?: number;
};
type StdErr = StdErrModel & {
  pred?: string;
  primary?: string;
  models?: Record<string, StdErrModel>;
};
type BrightStd = {
  bright_edges: (number | null)[]; std_edges: (number | null)[]; hist: number[][];
  bright: (number | null)[]; lo: (number | null)[]; med: (number | null)[]; hi: (number | null)[];
  stretch: number;
};
type CombinerFeatureErrorModel = {
  median_log_error: (number | null)[][]; counts: number[][];
};
type CombinerFeatureErrorAxis = {
  axis_names: string[]; edges: number[][];
  models: Record<string, CombinerFeatureErrorModel>;
};
type CombinerFeatureError = {
  axes: Record<string, CombinerFeatureErrorAxis>;
  color_range: number[]; error_unit?: string;
};
export type Evals = {
  ps: PS | null; guides?: { theta_min?: number; lr_scale?: number; vis_fwhm?: number };
  coherence?: Coherence | null;
  members?: { label: string; loss?: string; blocks?: number; asinh_knee?: number | null }[];
  n_fields?: number; n_members?: number;
  std_err?: StdErr | null; bright_std?: BrightStd | null;
  combiner_feature_error?: CombinerFeatureError;
  combiner?: CombinerMetrics | null;
  model_combiners?: Record<string, CombinerMetrics | null>;
};
type CombinerMetrics = {
  available?: boolean;
  psnr?: number | null; asinh_l1?: number | null;
  ensemble_mean_psnr?: number | null; ensemble_mean_asinh_l1?: number | null;
  best_member_psnr?: number | null; best_member_label?: string | null;
  best_member_asinh_l1?: number | null; best_member_l1_label?: string | null;
  coherence_overall?: number | null; coherence_sr?: number | null;
};

/* ── combiner.json ───────────────────────────────────────────────────────── */
type EffW = { brightness_asinh?: (number | null)[]; brightness_e?: (number | null)[]; jacobian?: (number | null)[][] };
type FeatureGrid = {
  mean_asinh?: (number | null)[]; std_asinh?: (number | null)[];
  std_log?: (number | null)[];
  center_mean_asinh?: (number | null)[]; center_std_asinh?: (number | null)[];
  center_std_log?: (number | null)[];
  x_label?: string; y_label?: string; y_is_log?: boolean; conditioning_note?: string;
  z_label?: string;
  weights?: (number | null)[][][];
};
type PCAWeightSurface = FeatureGrid & {
  available?: boolean; reason?: string; n_pixels?: number; n_fields?: number;
  feature_space?: string; conditioning_note?: string;
  explained_variance_ratio?: (number | null)[];
  feature_names?: string[]; loadings?: (number | null)[][];
  surface_labels?: string[];
  projection_method?: string; integration_neighbors?: number;
  integrated_weights?: (number | null)[]; peak_weights?: (number | null)[];
};
type PredictiveAxisMetrics = {
  soft_route_r2?: number; top_route_accuracy?: number;
  mean_selected_regret?: number; soft_expected_regret?: number;
};
type PredictiveAxes = {
  available?: boolean; stale?: boolean; reason?: string; regime?: string; target?: string;
  method?: string; feature_space?: string; sampling?: string; split?: string;
  n_fields?: number; n_pixels?: number; n_train?: number; n_holdout?: number;
  oracle_temperature?: number; member_labels?: string[]; route_labels?: string[];
  feature_names?: string[]; axis_loadings?: (number | null)[][];
  axis1?: (number | null)[]; axis2?: (number | null)[];
  oracle_weights?: (number | null)[][][]; density?: (number | null)[][];
  route_fraction?: (number | null)[]; conditioning_note?: string;
  metrics?: {
    two_axis?: PredictiveAxisMetrics; full_input_linear?: PredictiveAxisMetrics;
    ensemble_mean_regret?: number; r2_retention?: number | null;
  };
};
type HRWeightBand = {
  brightness_asinh?: (number | null)[]; brightness_e?: (number | null)[];
  mean?: (number | null)[][]; p16?: (number | null)[][]; p84?: (number | null)[][];
  counts?: number[];
};
type HRWeights = {
  available?: boolean; reason?: string; subset?: string; target?: string;
  member_labels?: string[]; bands?: Record<string, HRWeightBand>;
  n_fields?: number; n_pixels?: number;
};
type CenterStage = {
  stage?: number; epoch?: number; n_centers?: number; val_l1?: number; val_vis_asinh_psnr?: number;
  train_pixels?: number; train_l1?: number; val_pixels?: number;
  added_centers?: number; candidate_pixels?: number;
  candidate_mean_achievable_gain?: number; candidate_max_achievable_gain?: number;
  train_mean_residual?: number; train_max_residual?: number;
  train_mean_minimum_possible_l1?: number;
  center_weight_mean_recoverable_l1?: number;
  center_weight_max_recoverable_l1?: number;
  train_mean_recoverable_l1?: number; train_max_recoverable_l1?: number;
  target_nonzero_mean_residual?: number;
  aborted_for_small_residual?: boolean;
  optimizer_converged?: boolean; optimizer_iterations?: number;
  optimizer_progress?: boolean; parameter_delta_norm?: number; new_block_norm?: number;
  optimizer_gradient_inf?: number; optimizer_message?: string;
};
export type Combiner = {
  available?: boolean; stale?: boolean; regime?: string;
  member_labels: string[];
  members?: { label: string; loss?: string; blocks?: number; asinh_knee?: number | null; step?: number | null; psnr?: number | null }[];
  kind?: "raw_incremental_minmeanmax_rbf"; n_kernels?: number;
  min_usage?: number; val_l1?: number | null;
  band_names?: string[];
  member_weight_peaks?: Record<string, number[]>;
  member_weight_integrals?: Record<string, number[]>;
  eff_weights?: Record<string, EffW>;
  feature_grid?: Record<string, FeatureGrid>;
  pca_weight_surface?: PCAWeightSurface | null;
  hr_weights?: HRWeights;
  source_starless?: boolean[]; reason?: string;
  source_member_labels?: string[];
  fit_meta?: { experts?: string[]; parents_fitted_together?: boolean; preview?: boolean; num_images?: number;
    center_history?: CenterStage[]; center_abort_reason?: string; residual_abort_threshold?: number;
    baseline_member_index?: number; baseline_member_label?: string; baseline_selection_metric?: string;
    training_mode?: string; training_fields?: number; validation_fields?: number;
    training_pixels_per_epoch?: number; validation_pixels_per_epoch?: number;
    batch_rows?: number; selected_epoch?: number; selected_stage?: number;
    initial_best_member_label?: string; increment_size?: number };
};

/* ── training-curves.json ────────────────────────────────────────────────── */
/* Note: the payload overwrites the loss *series* with the loss-*norm* string,
   so only the PSNR series is chartable; `loss` here is the norm ("l1"…). */
export type Curve = { name: string; psnr: [number, number][]; blocks?: number; test_psnr?: number | null; loss?: string; asinh_knee?: number | null; starless?: boolean };

const XTICKS = [0.05, 0.1, 0.2, 0.5, 1, 2, 5];
const COMBINER_META: Record<string, { label: string; color: string }> = {
  ensemble_mean: { label: "ensemble mean", color: C.mean },
  raw_incremental_minmeanmax_rbf: { label: "minibatched convex all-asinh RBF", color: "#4f9d69" },
};
const combinerMeta = (kind: string) => COMBINER_META[kind] ?? {
  label: kind.replace(/_/g, " "), color: categorical(kind.length),
};
const activeCombinerKind = (kind: string) => kind === "raw_incremental_minmeanmax_rbf";
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
  { id: "coherence", label: "coherence score" },
  { id: "std-error", label: "std vs error" },
  { id: "combiner-error", label: "combiner axes vs error" },
  { id: "std-brightness", label: "std vs brightness" },
] as const;
type DiagTab = typeof DIAG_TABS[number]["id"];

/* ── pixel back-tracing (click a heatmap cell → real image stamps) ─────────── */
type PickDiag = "std_err" | "bright_std" | "combiner_feature_error";
/* Stamps arrive as base64 little-endian float32 blobs (one per tier): the
   full N-band LR/HR/SR cubes (rendered with the field viewer's colour engine)
   and the single-band σ. */
type Stamp = {
  field: number; y: number; x: number; center: number; sr_is_combiner: boolean;
  lr?: string; hr: string; sr: string; std: string;
  hr_val: number; sr_val: number; std_val: number; err_val: number; bright_asinh: number;
  model_kind?: string;
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

function TraceHint({ targetLabel = "HR" }: { targetLabel?: string }) {
  return (
    <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
      Click any cell to back-trace it — a handful of the real pixels that landed there, as
      stamps (LR · {targetLabel} · SR/combiner · σ) from across the test fields, with the field viewer's colour controls.
    </div>
  );
}

const STAMP_PX = 3;

function PixelTrace(
  { context, traceBase, targetLabel, pick, modelKind, axisMode, cellLabel, colorMeta, onClose }:
  { context: string; traceBase: string; pick: { diag: PickDiag; i: number; j: number }; cellLabel: string;
    modelKind?: string; axisMode?: string; colorMeta?: ColorMeta; targetLabel: string; onClose: () => void },
) {
  const separator = traceBase.includes("?") ? "&" : "?";
  const modelQuery = modelKind ? `&model=${encodeURIComponent(modelKind)}` : "";
  const axisQuery = axisMode ? `&axis=${encodeURIComponent(axisMode)}` : "";
  const url = `${traceBase}${separator}diag=${pick.diag}&i=${pick.i}&j=${pick.j}${modelQuery}${axisQuery}`;
  const trace = useResource<Trace>(url, [context, traceBase, pick.diag, pick.i, pick.j, modelKind, axisMode]);
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
                      colorMeta={colorMeta} opts={opts} render={render} px={STAMP_PX} label={targetLabel} />
                    <ImageStamp b64={s.sr} size={t.size} center={s.center} bands={traceBands}
                      colorMeta={colorMeta} opts={opts} render={render} px={STAMP_PX}
                      label="SR" badge={combinerMeta(s.model_kind ?? (s.sr_is_combiner ? "rbf_gate" : "ensemble_mean")).label} />
                    <SigmaStamp b64={s.std} size={t.size} center={s.center} stretch={t.stretch} px={STAMP_PX} label="σ" />
                  </div>
                  <div className="mono ens-trace__nums">
                    #{s.field} · σ={fmtE(s.std_val)} · |err|={fmtE(s.err_val)} · {targetLabel}={fmtE(s.hr_val)} e⁻
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
  const combinerKind: CombinerModelKind = "raw_incremental_minmeanmax_rbf";

  const status = useResource<Status>(`/ensemble/status.json?mode=${mode}`, [mode]);
  const evals = useResource<Evals>(`/ensemble/evals.json?mode=${mode}`, [mode]);
  const comb = useResource<Combiner>(`/ensemble/combiner.json?mode=${mode}&model_kind=${combinerKind}`, [mode, combinerKind]);
  const curves = useResource<{ members?: Curve[] }>("/ensemble/training-curves.json");

  const evalJob = useJob();
  const opJob = useJob();
  const fitJob = useJob();
  // Continue/fork buttons in the members table jump to the Train members page,
  // prefilled via query params.
  const toTrain = (m: "continue" | "fork", member: string) =>
    navigate(`/train-members?mode=${m}&member=${encodeURIComponent(member)}`);

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
          onContinue={(m) => toTrain("continue", m)}
          onFork={(m) => toTrain("fork", m)} />
        <TrainingCurves curves={asArray<Curve>(curves.data?.members)} starless={starless} />
        <Evaluations evals={evals.data} loading={evals.loading} mode={mode} theme={theme}
          targetLabel={starless ? "clean goal" : "HR"} />
        <CombinerCard comb={comb.data} loading={comb.loading} mode={mode} theme={theme} fitJob={fitJob} onFit={reloadAll}
          evalReady={status.data?.evaluations_ready ?? false}
          modelKind={combinerKind} />
        <DisagreementCard key={mode} mode={mode} members={asArray<Member>(status.data?.members)}
          targetLabel={starless ? "clean goal" : "HR"} />
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
type MemberSortKey = "rank" | "name" | "psnr" | "loss" | "depth" | "knee" | "step";
type MemberSort = { key: MemberSortKey; direction: "asc" | "desc" };

function memberSortValue(member: Member, key: MemberSortKey): number | string | null {
  switch (key) {
    case "rank": return member.psnr_rank ?? null;
    case "name": return member.name;
    case "psnr": return member.psnr ?? null;
    case "loss": return member.loss ?? "l1";
    case "depth": return member.blocks ?? null;
    case "knee": return kneeOf(member.asinh_knee);
    case "step": return member.step ?? null;
  }
}

function compareMemberValues(a: number | string | null, b: number | string | null): number {
  if (a == null && b == null) return 0;
  if (a == null) return 1;
  if (b == null) return -1;
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: "base" });
}

function Members(
  { status, loading, starless, opJob, onArchived, onContinue, onFork }:
  { status: Status | null; loading: boolean; starless: boolean; opJob: ReturnType<typeof useJob>;
    onArchived: () => void; onContinue: (m: string) => void; onFork: (m: string) => void },
) {
  const rows = asArray<Member>(status?.members).filter((m) => !!m.starless === starless);
  const [sort, setSort] = useState<MemberSort | null>(null);
  const shownRows = useMemo(() => {
    if (!sort) return rows;
    const direction = sort.direction === "asc" ? 1 : -1;
    return [...rows].sort((a, b) => {
      const aValue = memberSortValue(a, sort.key);
      const bValue = memberSortValue(b, sort.key);
      // Keep unscored/unavailable values at the bottom in either direction.
      if (aValue == null || bValue == null) return aValue == null && bValue == null ? 0 : aValue == null ? 1 : -1;
      const compared = compareMemberValues(aValue, bValue);
      return compared * direction || a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: "base" }) * direction;
    });
  }, [rows, sort]);

  const sortHeader = (key: MemberSortKey, label: string, align?: "right") => {
    const active = sort?.key === key;
    return (
      <button type="button" className={`ui-table__sort${align === "right" ? " ui-table__sort--right" : ""}`}
        onClick={() => setSort((previous) => previous?.key === key
          ? { key, direction: previous.direction === "asc" ? "desc" : "asc" }
          : { key, direction: "asc" })}
        aria-label={`Sort by ${label}`}
        title={`Sort by ${label}${active ? ` (${sort.direction === "asc" ? "ascending" : "descending"})` : ""}`}>
        <span>{label}</span>
        <span className="ui-table__sort-icon" aria-hidden>{active ? (sort.direction === "asc" ? "↑" : "↓") : "↕"}</span>
      </button>
    );
  };

  const cols: Column<Member>[] = [
    { header: sortHeader("rank", "#", "right"), cell: (m) => m.psnr_rank ? <b>{m.psnr_rank}</b> : <span className="muted">—</span>, width: "5%", align: "right" },
    { header: sortHeader("name", "member"), cell: (m) => <code className="mono">{m.name}</code>, width: "17%" },
    { header: sortHeader("psnr", "PSNR", "right"), cell: (m) => m.psnr != null ? `${m.psnr.toFixed(3)} dB` : <span className="muted">—</span>, width: "13%", align: "right" },
    { header: sortHeader("loss", "loss"), cell: (m) => <Badge>{(m.loss ?? "l1").toUpperCase()}</Badge>, width: "9%" },
    { header: sortHeader("depth", "depth", "right"), cell: (m) => m.blocks ?? "—", width: "8%", align: "right" },
    { header: sortHeader("knee", "knee", "right"), cell: (m) => <span className="mono">{kneeTag(m.asinh_knee)}</span>, width: "10%", align: "right" },
    { header: sortHeader("step", "step", "right"), cell: (m) => m.step ? m.step.toLocaleString() : "—", width: "12%", align: "right" },
    { header: "", align: "right", width: "26%", cell: (m) => (
      <div className="row" style={{ gap: 4, justifyContent: "flex-end", flexWrap: "nowrap" }}>
        <Button size="sm" variant="ghost" title="continue training this member"
          onClick={() => onContinue(m.name)}>▶ continue</Button>
        <Button size="sm" variant="ghost" title="fork a new member from this one"
          onClick={() => onFork(m.name)}>⑂ fork</Button>
        <Button size="sm" variant="ghost" disabled={opJob.busy}
          title="zip → tracking, tombstone, delete; rebuild cached cubes on next evaluation"
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
          : <Table className="ens-members-table" columns={cols} rows={shownRows} rowKey={(m) => m.name}
              empty={`no ${starless ? "starless" : "starfull"} members — train some, then ⬇ pull from FASRC`} />}
      </CardBody>
    </Card>
  );
}

/* ── training curves ─────────────────────────────────────────────────────── */
export function TrainingCurves({ curves, starless }: { curves: Curve[]; starless?: boolean }) {
  const theme = useThemeValue();
  const [colorBy, setColorBy] = useState<ColorBy>("loss");
  const rows = starless == null ? curves : curves.filter((c) => !!c.starless === starless);

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
export function Evaluations(
  { evals, loading, mode, theme, viewerCollection = "ensemble", traceBase, targetLabel = "HR" }:
  { evals: Evals | null; loading: boolean; mode: string; theme: string;
    viewerCollection?: string; traceBase?: string; targetLabel?: string },
) {
  const [tab, setTab] = useState<DiagTab>("power-spectrum");
  const [colorBy, setColorBy] = useState<ColorBy>("uniform");
  // Which overlay curves to draw on the power spectrum.
  const [show, setShow] = useState({ cross: true, mean: true });
  const [showModels, setShowModels] = useState<Record<string, boolean>>({});
  const [stdErrKind, setStdErrKind] = useState("raw_incremental_minmeanmax_rbf");
  const [combinerErrorKind, setCombinerErrorKind] = useState("raw_incremental_minmeanmax_rbf");
  const [combinerAxisMode, setCombinerAxisMode] = useState("mean_std");
  const toggle = (k: keyof typeof show) => setShow((s) => ({ ...s, [k]: !s[k] }));
  const modelShown = (kind: string) => showModels[kind] ?? true;
  const toggleModel = (kind: string) => setShowModels((s) => ({ ...s, [kind]: !(s[kind] ?? true) }));
  // Back-traced heatmap cell (click-to-inspect). Cleared when the tab/regime
  // changes — a cell only means something within one diagnostic.
  const [pick, setPick] = useState<{ diag: PickDiag; i: number; j: number } | null>(null);
  useEffect(() => setPick(null), [tab, mode, stdErrKind, combinerErrorKind, combinerAxisMode]);
  // Viewer colour meta (band constants) so back-trace stamps colour like the viewer.
  const viewerQuery = viewerCollection === "ensemble" ? `?mode=${mode}` : "";
  const vmeta = useResource<ViewerMetaColor>(`/viewer/meta/${viewerCollection}${viewerQuery}`, [viewerCollection, mode]);
  const pixelTraceBase = traceBase ?? `/ensemble/pixel-trace.json?mode=${mode}`;
  const colorMeta = vmeta.data?.color;
  const ps = evals?.ps ?? null;
  const members = asArray<NonNullable<Evals["members"]>[number]>(evals?.members);
  const availableModelCurves = Object.entries(ps?.model_combiners ?? {})
    .filter(([kind, curve]) => activeCombinerKind(kind) && hasData(curve?.r));
  const availableStdErrModels = Object.entries(evals?.std_err?.models ?? {})
    .filter(([kind, diagnostic]) => activeCombinerKind(kind)
      && asArray<number[]>(diagnostic?.hist).length > 0);
  const availableCombinerErrorAxes = Object.entries(evals?.combiner_feature_error?.axes ?? {});
  const selectedCombinerErrorAxis = evals?.combiner_feature_error?.axes?.[combinerAxisMode]
    ?? availableCombinerErrorAxes[0]?.[1];
  const availableCombinerErrorModels = Object.keys(selectedCombinerErrorAxis?.models ?? {})
    .filter(activeCombinerKind);

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
    const modelCurves = Object.entries(ps.model_combiners ?? {})
      .filter(([kind, curve]) => activeCombinerKind(kind) && hasData(curve?.r));
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
    modelCurves.forEach(([kind, curve]) => {
      if (modelShown(kind)) series.push({ x: theta, y: curve.r!, color: combinerMeta(kind).color, width: 2.2, dots: true });
    });
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
      ...(show.mean ? [{ label: "displayed ensemble mean", color: C.mean }] : []),
      ...modelCurves.filter(([kind]) => modelShown(kind)).map(([kind]) => ({ label: combinerMeta(kind).label, color: combinerMeta(kind).color })),
      ...(facet.length ? facet : [{ label: "individual models", color: C.muted }]),
      ...(show.cross && hasData(ps.r_cross) ? [{ label: "model–model r̃(k)", color: C.cross, dash: true }] : []),
    ];
    return { series, guides, xDomain, yDomain: [0, 1.05] as [number, number], xTicks, yTicks, legend };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ps, members, colorBy, show, showModels, evals, theme]);

  const coherenceChart = useMemo(() => {
    const rows = asArray<CoherenceScore>(evals?.coherence?.scores)
      .filter((r) => r && r.id !== "combiner" && (r.overall != null || r.sr != null));
    if (!rows.length) return null;
    const shortLabel = (r: CoherenceScore, i: number) => {
      if (r.id === "ensemble_mean") return "mean";
      if (r.id === "lr_baseline") return "LR";
      if (r.id === "combiner") return "comb";
      if (r.id === "raw_incremental_minmeanmax_rbf_combiner") return "raw RBF";
      if (r.id === "model_agreement") return "agree";
      return `m${String(i).padStart(2, "0")}`;
    };
    const series: Series[] = [];
    rows.forEach((r, i) => {
      if (r.overall != null) {
        if (r.overall_lo != null && r.overall_hi != null)
          series.push({ x: [i, i], y: [r.overall_lo, r.overall_hi], color: C.mean, width: 1, alpha: 0.45 });
        series.push({ x: [i], y: [r.overall], color: C.mean, width: 2.8, dots: true });
      }
      if (r.sr != null) {
        if (r.sr_lo != null && r.sr_hi != null)
          series.push({ x: [i, i], y: [r.sr_lo, r.sr_hi], color: C.comb, width: 1, alpha: 0.45 });
        series.push({ x: [i], y: [r.sr], color: C.comb, width: 2.8, dots: true });
      }
    });
    return {
      rows, series,
      xDomain: [-0.5, Math.max(0.5, rows.length - 0.5)] as [number, number],
      xTicks: rows.map((r, i) => ({ v: i, label: shortLabel(r, i) })),
      legend: [
        { label: "overall log-k mean", color: C.mean },
        { label: "super-resolution range", color: C.comb },
        { label: "16–84% field spread", color: C.guide, dash: true },
      ],
    };
  }, [evals, theme]);

  // std vs error — density cloud + median-|error|-per-std curve, both axes
  // log10(e⁻). The |err|=σ diagonal reference rides as a 2-pt series.
  const stdErr = useMemo(() => {
    const d = evals?.std_err;
    const fallbackKind = Object.keys(d?.models ?? {}).find(activeCombinerKind)
      ?? "ensemble_mean";
    const chosenKind = d?.models?.[stdErrKind] ? stdErrKind : fallbackKind;
    const selected = d?.models?.[chosenKind] ?? d;
    const hist = asArray<number[]>(selected?.hist);
    if (!d || !selected || !hist.length) return null;
    const edges = num(selected.edges);
    if (edges.length < 2) return null;
    const lo = edges[0], hi = edges[edges.length - 1];
    const heat: Heat = { z: hist, xEdges: edges, yEdges: edges };
    const series: Series[] = [
      { x: [lo, hi], y: [lo, hi], color: C.guide, width: 1.3, dash: [6, 3] },
      { x: num(selected.med_std), y: num(selected.med_err), color: combinerMeta(chosenKind).color, width: 2.6, dots: true },
    ];
    const ticks = logDecadeTicks(lo, hi);
    const legend = [
      { label: `median |error| · ${combinerMeta(chosenKind).label}`, color: combinerMeta(chosenKind).color },
      { label: "|error| = σ", color: C.guide, dash: true },
    ];
    // Human range for a clicked cell (both axes log10 e⁻).
    const describe = (c: { i: number; j: number }) =>
      `σ ${logRange(edges, c.i)} e⁻ · |err| ${logRange(edges, c.j)} e⁻`;
    const yLabel = `|${combinerMeta(chosenKind).label} − ${targetLabel}|  [e⁻]`;
    return { heat, series, domain: [lo, hi] as [number, number], ticks, legend, describe, yLabel,
      chosenKind };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [evals, stdErrKind, theme]);

  const combinerError = useMemo(() => {
    const diagnostic = evals?.combiner_feature_error;
    const axes = diagnostic?.axes ?? {};
    const chosenAxisMode = axes[combinerAxisMode]
      ? combinerAxisMode : (Object.keys(axes)[0] ?? "mean_std");
    const axis = axes[chosenAxisMode];
    if (!axis || !axis.edges?.[0]?.length) return null;
    const modelKinds = Object.keys(axis.models ?? {}).filter(activeCombinerKind);
    const chosenKind = axis.models?.[combinerErrorKind]
      ? combinerErrorKind : (modelKinds[0] ?? "ensemble_mean");
    const model = axis.models?.[chosenKind];
    if (!model) return null;
    const xEdges = num(axis.edges[0]);
    const yEdges = num(axis.edges[1]);
    const xDomain: [number, number] = [xEdges[0], xEdges[xEdges.length - 1]];
    const featureTicks = (edges: number[]) => {
      const lo = edges[0], hi = edges[edges.length - 1];
      const values = [-1, 0, 2, 4, 6, 8, 10, 12].filter((v) => v >= lo && v <= hi);
      return values.map((v) => ({ v, label: String(v) }));
    };
    const z = asArray<(number | null)[]>(model.median_log_error).map(num);
    const finiteZ = z.flat().filter(Number.isFinite);
    if (!finiteZ.length) return null;
    const sharedRange = num(diagnostic?.color_range);
    const zlo = sharedRange[0] ?? Math.floor(Math.min(...finiteZ));
    const zhi = sharedRange[1] ?? Math.ceil(Math.max(...finiteZ));
    const colorTicks = [];
    for (let e = zlo; e <= zhi; e += Math.max(1, Math.ceil((zhi - zlo) / 4)))
      colorTicks.push({ v: e, label: `10${sup(e)}` });
    const describe = (c: { i: number; j: number }) =>
      `${axis.axis_names[0]} ${xEdges[c.i].toFixed(2)}–${xEdges[c.i + 1].toFixed(2)} · ` +
      `${axis.axis_names[1]} ${yEdges[c.j].toFixed(2)}–${yEdges[c.j + 1].toFixed(2)} asinh · ` +
      `median |error| ${fmtE(10 ** z[c.i][c.j])} e⁻`;
    return { chosenKind, chosenAxisMode, xEdges, yEdges, xDomain,
      yDomain: [yEdges[0], yEdges[yEdges.length - 1]] as [number, number],
      xTicks: featureTicks(xEdges), yTicks: featureTicks(yEdges),
      heat: { z, xEdges, yEdges, scale: "linear" as const, min: zlo, max: zhi,
        colorTicks, colorLabel: "median |error| [e⁻]" },
      xLabel: `${axis.axis_names[0]}  [asinh member space]`,
      yLabel: `${axis.axis_names[1]}  [asinh member space]`, describe };
  }, [evals, combinerErrorKind, combinerAxisMode, theme]);

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
      `${targetLabel} ${fmtE(st * Math.sinh(bx[c.i]))}–${fmtE(st * Math.sinh(bx[c.i + 1]))} e⁻` +
      ` · σ ${logRange(sy, c.j)} e⁻`;
    return {
      heat, series,
      xDomain: [bx[0], bx[bx.length - 1]] as [number, number],
      yDomain: [sy[0], sy[sy.length - 1]] as [number, number],
      xTicks: brightTicks(d.stretch), yTicks: logDecadeTicks(sy[0], sy[sy.length - 1]), legend, describe,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [evals, theme]);

  const modelMetrics = Object.entries(evals?.model_combiners ?? {})
    .filter(([kind, metric]) => activeCombinerKind(kind) && Boolean(metric?.available));
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
                    {availableModelCurves.length ? availableModelCurves.map(([kind]) => (
                      <Chip key={kind} on={modelShown(kind)} dot={combinerMeta(kind).color} onClick={() => toggleModel(kind)}
                        title={`${combinerMeta(kind).label} combiner r(k)`}>{combinerMeta(kind).label}</Chip>
                    )) : null}
                  </div>
                  <Select<ColorBy> value={colorBy} onChange={setColorBy}
                    options={[{ value: "uniform", label: "uniform" }, { value: "loss", label: "by loss" }, { value: "depth", label: "by depth" }, { value: "knee", label: "by knee" }]} />
                </div>
                <Plot title={`cross-correlation r(k) vs ${targetLabel}   (1 = perfect)`} xScale="log"
                  xDomain={chart.xDomain} yDomain={chart.yDomain} xTicks={chart.xTicks} yTicks={chart.yTicks}
                  xLabel="angular scale θ = 1/2k [arcsec]" yLabel="r(k) [VIS]" series={chart.series} guides={chart.guides} aspect={0.46} />
                <Legend items={chart.legend} />
                <div className="row" style={{ gap: "var(--s5)", marginTop: "var(--s4)" }}>
                  <Stat k="fields" v={evals?.n_fields ?? "—"} />
                  <Stat k="members" v={evals?.n_members ?? "—"} />
                  {modelMetrics.flatMap(([kind, metric]) => [
                    <Stat key={`${kind}-l1`} k={`${combinerMeta(kind).label} · asinh L1`}
                      v={metric?.asinh_l1 != null ? metric.asinh_l1.toFixed(5) : "—"} />,
                    <Stat key={`${kind}-psnr`} k={`${combinerMeta(kind).label} · PSNR`}
                      v={metric?.psnr != null ? `${metric.psnr.toFixed(2)} dB` : "—"} />,
                    <Stat key={`${kind}-coherence`} k={`${combinerMeta(kind).label} · coherence overall / SR`}
                      v={metric?.coherence_overall != null || metric?.coherence_sr != null
                        ? `${metric?.coherence_overall?.toFixed(3) ?? "—"} / ${metric?.coherence_sr?.toFixed(3) ?? "—"}` : "—"} />,
                  ])}
                </div>
              </>
            )
          ) : tab === "coherence" ? (
            !coherenceChart ? <Empty>no spectral coherence cached for <b>{mode}</b> — refresh the evaluation from cached cubes.</Empty> : (
              <>
                <Plot title="one-number spectral coherence  (median across fields)"
                  xDomain={coherenceChart.xDomain} yDomain={[-1, 1.05]} xTicks={coherenceChart.xTicks}
                  yTicks={[{ v: -1, label: "−1" }, { v: 0, label: "0" }, { v: 0.5, label: "0.5" }, { v: 1, label: "1" }]}
                  xLabel="reconstruction / reference" yLabel="mean r(k) over d log(k)"
                  series={coherenceChart.series}
                  guides={[{ axis: "y", v: 1, color: C.guide, dash: [2, 3] }]}
                  aspect={0.42} />
                <Legend items={coherenceChart.legend} />
                <div className="muted" style={{ marginTop: "var(--s3)", lineHeight: 1.5 }}>
                  Equal weight per spatial-frequency octave. “Overall” spans the measured spectrum;
                  “super-resolution” is restricted to scales finer than the LR sampling limit.
                  Whiskers show the 16–84% field spread.
                </div>
                <div className="row" style={{ gap: "var(--s5)", marginTop: "var(--s4)", flexWrap: "wrap" }}>
                  {coherenceChart.rows
                    .filter((r) => r.id === "ensemble_mean" || r.id.endsWith("_combiner"))
                    .map((r) => (
                      <Stat key={r.id} k={`${r.label} · overall / SR`}
                        v={`${r.overall?.toFixed(3) ?? "—"} / ${r.sr?.toFixed(3) ?? "—"}`} />
                    ))}
                </div>
              </>
            )
          ) : tab === "std-error" ? (
            !stdErr ? <Empty>no evaluation cached for <b>{mode}</b> — run “Evaluate on test set”.</Empty> : (
              <>
                {availableStdErrModels.length > 0 && <div className="row" style={{ gap: 6, flexWrap: "wrap", marginBottom: 8 }}>
                  {availableStdErrModels.map(([kind]) => (
                    <Chip key={kind} on={stdErr.chosenKind === kind} dot={combinerMeta(kind).color}
                      onClick={() => setStdErrKind(kind)} title={`error distribution of ${combinerMeta(kind).label}`}>
                      {combinerMeta(kind).label}
                    </Chip>
                  ))}
                </div>}
                <Plot title="Does disagreement predict error?  (VIS, per pixel)"
                  xDomain={stdErr.domain} yDomain={stdErr.domain} xTicks={stdErr.ticks} yTicks={stdErr.ticks}
                  xLabel="cross-member per-pixel σ  [e⁻]" yLabel={stdErr.yLabel}
                  heat={stdErr.heat} series={stdErr.series} aspect={0.62}
                  onHeatClick={(c) => setPick({ diag: "std_err", ...c })}
                  highlight={pick?.diag === "std_err" ? pick : null} />
                <Legend items={stdErr.legend} />
                <TraceHint targetLabel={targetLabel} />
                {pick?.diag === "std_err" &&
                  <PixelTrace context={mode} traceBase={pixelTraceBase} targetLabel={targetLabel} pick={pick} cellLabel={stdErr.describe(pick)}
                    modelKind={stdErr.chosenKind} colorMeta={colorMeta} onClose={() => setPick(null)} />}
              </>
            )
          ) : tab === "combiner-error" ? (
            !combinerError ? <Empty>no combiner-coordinate diagnostic cached for <b>{mode}</b> — refresh the evaluation from cached cubes.</Empty> : (
              <>
                <div style={{ display: "grid", gap: 7, marginBottom: 10 }}>
                  <div className="row" style={{ gap: 6, flexWrap: "wrap" }}>
                    <span className="mono muted" style={{ width: 70, fontSize: 11 }}>axes</span>
                    {availableCombinerErrorAxes.map(([axisMode]) => (
                      <Chip key={axisMode} on={combinerError.chosenAxisMode === axisMode}
                        onClick={() => setCombinerAxisMode(axisMode)}
                        title={`project every model error onto ${axisMode === "mean_std" ? "mean–std" : "min–max"} coordinates`}>
                        {axisMode === "mean_std" ? "mean – std" : "min – max"}
                      </Chip>
                    ))}
                  </div>
                  <div className="row" style={{ gap: 6, flexWrap: "wrap" }}>
                    <span className="mono muted" style={{ width: 70, fontSize: 11 }}>error of</span>
                    {availableCombinerErrorModels.map((kind) => (
                    <Chip key={kind} on={combinerError.chosenKind === kind} dot={combinerMeta(kind).color}
                      onClick={() => setCombinerErrorKind(kind)} title={`absolute error of ${combinerMeta(kind).label}`}>
                      {combinerMeta(kind).label}
                    </Chip>
                    ))}
                  </div>
                </div>
                <Plot title={`${combinerMeta(combinerError.chosenKind).label} error on ${combinerError.chosenAxisMode === "mean_std" ? "mean–std" : "min–max"} coordinates  (VIS, per pixel)`}
                  xDomain={combinerError.xDomain} yDomain={combinerError.yDomain}
                  xTicks={combinerError.xTicks} yTicks={combinerError.yTicks}
                  xLabel={combinerError.xLabel} yLabel={combinerError.yLabel}
                  heat={combinerError.heat} series={[]} aspect={0.62}
                  onHeatClick={(c) => setPick({ diag: "combiner_feature_error", ...c })}
                  highlight={pick?.diag === "combiner_feature_error" ? pick : null} />
                <TraceHint targetLabel={targetLabel} />
                {pick?.diag === "combiner_feature_error" &&
                  <PixelTrace context={mode} traceBase={pixelTraceBase} targetLabel={targetLabel} pick={pick}
                    modelKind={combinerError.chosenKind} axisMode={combinerError.chosenAxisMode}
                    cellLabel={combinerError.describe(pick)}
                    colorMeta={colorMeta} onClose={() => setPick(null)} />}
              </>
            )
          ) : (
            !brightStd ? <Empty>no evaluation cached for <b>{mode}</b> — run “Evaluate on test set”.</Empty> : (
              <>
                <Plot title="Where does disagreement live?  (VIS, per pixel)"
                  xDomain={brightStd.xDomain} yDomain={brightStd.yDomain} xTicks={brightStd.xTicks} yTicks={brightStd.yTicks}
                  xLabel={`${targetLabel} pixel brightness  [e⁻]  (asinh axis)`} yLabel="cross-member per-pixel σ  [e⁻]"
                  heat={brightStd.heat} series={brightStd.series} aspect={0.62}
                  onHeatClick={(c) => setPick({ diag: "bright_std", ...c })}
                  highlight={pick?.diag === "bright_std" ? pick : null} />
                <Legend items={brightStd.legend} />
                <TraceHint targetLabel={targetLabel} />
                {pick?.diag === "bright_std" &&
                  <PixelTrace context={mode} traceBase={pixelTraceBase} targetLabel={targetLabel} pick={pick} cellLabel={brightStd.describe(pick)}
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
type CombinerModelKind = "raw_incremental_minmeanmax_rbf";
type SurfaceView = { yaw: number; pitch: number; zoom: number };

function WeightSurface3D(
  { grid, member, memberLabel, theme, stdAxis, view, resetView, onViewChange, onMemberStep }:
  { grid: FeatureGrid; member: number; memberLabel: string; theme: string; stdAxis: "raw" | "log";
    view: SurfaceView; resetView?: SurfaceView; onViewChange: (view: SurfaceView) => void;
    onMemberStep?: (step: -1 | 1) => void },
) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragRef = useRef<{ x: number; y: number; yaw: number; pitch: number } | null>(null);
  const [size, setSize] = useState({ width: 640, height: 480 });

  const mean = num(grid.mean_asinh).filter(isFinite);
  const std = num(grid.std_asinh).filter(isFinite);
  const savedLogStd = num(grid.std_log).filter(isFinite);
  // Existing fitted payloads predate std_log. Keep their log view usable until
  // the endpoint reserializes them with the exact model-coordinate samples.
  const useLogAxis = stdAxis === "log" && grid.y_is_log !== false;
  const stdCoord = useLogAxis
    ? (savedLogStd.length === std.length ? savedLogStd : std.map((v) => Math.log(v + 0.005)))
    : std;
  const weights = grid.weights ?? [];
  const values = std.flatMap((_, i) => mean.map((_, j) => weights[i]?.[j]?.[member]))
    .filter((w): w is number => w != null && isFinite(w));
  const maxWeight = values.length ? Math.max(...values) : 0;

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return undefined;
    const resize = () => setSize({ width: Math.max(320, host.clientWidth), height: 480 });
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || mean.length < 2 || std.length < 2 || stdCoord.length !== std.length || !maxWeight) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(size.width * dpr);
    canvas.height = Math.round(size.height * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.width, size.height);
    const css = getComputedStyle(document.documentElement);
    const text = css.getPropertyValue("--text").trim() || "#dce6f2";
    const muted = css.getPropertyValue("--muted").trim() || C.muted;
    const guide = css.getPropertyValue("--guide").trim() || C.guide;
    const cy = Math.cos(view.yaw), sy = Math.sin(view.yaw);
    const cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
    const scale = Math.min(size.width * 0.35, size.height * 0.42) * view.zoom;
    const stdLo = stdCoord[0], stdHi = stdCoord[stdCoord.length - 1];
    const project = (x: number, y: number, z: number) => {
      const rx = x * cy - y * sy;
      const ry = x * sy + y * cy;
      const py = ry * cp - z * sp;
      const depth = ry * sp + z * cp;
      const perspective = 0.82 + (depth + 1.8) * 0.08;
      return { x: size.width * 0.5 + rx * scale * perspective,
        y: size.height * 0.66 - py * scale * perspective, depth };
    };
    const point = (i: number, j: number) => {
      const w = weights[i]?.[j]?.[member];
      const value = typeof w === "number" && isFinite(w) ? w : 0;
      const y = stdHi === stdLo ? 0 : -1 + 2 * (stdCoord[i] - stdLo) / (stdHi - stdLo);
      return project(-1 + 2 * j / (mean.length - 1), y, Math.max(0, Math.min(1, value)));
    };
    const cells: { q: ReturnType<typeof point>[]; depth: number; value: number }[] = [];
    for (let i = 0; i < std.length - 1; i += 1) for (let j = 0; j < mean.length - 1; j += 1) {
      const q = [point(i, j), point(i, j + 1), point(i + 1, j + 1), point(i + 1, j)];
      const value = [weights[i]?.[j]?.[member], weights[i]?.[j + 1]?.[member],
        weights[i + 1]?.[j + 1]?.[member], weights[i + 1]?.[j]?.[member]]
        .filter((w): w is number => typeof w === "number" && isFinite(w))
        .reduce((sum, w, _, a) => sum + w / a.length, 0);
      cells.push({ q, depth: q.reduce((sum, p) => sum + p.depth, 0) / 4, value });
    }
    for (const cell of cells.sort((a, b) => a.depth - b.depth)) {
      ctx.beginPath();
      cell.q.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
      ctx.closePath();
      ctx.fillStyle = viridis(Math.max(0, Math.min(1, cell.value)));
      ctx.globalAlpha = 0.8;
      ctx.fill();
      ctx.strokeStyle = guide;
      ctx.globalAlpha = 0.28;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    const centerMean = num(grid.center_mean_asinh);
    const centerStd = (useLogAxis ? num(grid.center_std_log) : num(grid.center_std_asinh));
    for (let i = 0; i < Math.min(centerMean.length, centerStd.length); i += 1) {
      if (!isFinite(centerMean[i]) || !isFinite(centerStd[i])) continue;
      const x = -1 + 2 * (centerMean[i] - mean[0]) / (mean[mean.length - 1] - mean[0]);
      const y = stdHi === stdLo ? 0 : -1 + 2 * (centerStd[i] - stdLo) / (stdHi - stdLo);
      const p = project(x, y, 0.025);
      ctx.beginPath(); ctx.arc(p.x, p.y, 3.8, 0, Math.PI * 2);
      ctx.fillStyle = text; ctx.fill(); ctx.strokeStyle = guide; ctx.lineWidth = 1.3; ctx.stroke();
    }
    ctx.lineWidth = 1.2;
    const axis = (a: [number, number, number], b: [number, number, number], label: string) => {
      const p = project(...a), q = project(...b);
      ctx.strokeStyle = muted;
      ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
      ctx.fillStyle = text; ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.fillText(label, q.x + 6, q.y - 4);
    };
    axis([-1, -1, 0], [1, -1, 0], grid.x_label ?? "mean");
    axis([-1, -1, 0], [-1, 1, 0], useLogAxis ? `log ${grid.y_label ?? "std"}` : (grid.y_label ?? "std"));
    axis([-1, -1, 0], [-1, -1, 1], grid.z_label ?? "relative weight [0–1]");
    const fmtTick = (v: number) => Math.abs(v) >= 10 ? v.toFixed(0)
      : Math.abs(v) >= 1 ? v.toFixed(1) : v.toPrecision(2);
    const tick = (p: ReturnType<typeof project>, q: ReturnType<typeof project>, label: string) => {
      ctx.strokeStyle = muted; ctx.fillStyle = muted; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
      ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.fillText(label, q.x + 4, q.y + 3);
    };
    for (const f of [0, 0.25, 0.5, 0.75, 1]) {
      const x = -1 + 2 * f;
      const meanValue = mean[0] + f * (mean[mean.length - 1] - mean[0]);
      tick(project(x, -1, 0), project(x, -0.93, 0), fmtTick(meanValue));
      const i = Math.round(f * (std.length - 1));
      const y = stdHi === stdLo ? 0 : -1 + 2 * (stdCoord[i] - stdLo) / (stdHi - stdLo);
      tick(project(-1, y, 0), project(-0.93, y, 0), fmtTick(std[i]));
    }
  }, [grid, maxWeight, mean, member, size, std, stdAxis, stdCoord, theme, useLogAxis, view, weights]);

  return (
    <div ref={hostRef} style={{ border: "1px solid var(--guide)", borderRadius: 8, overflow: "hidden" }}>
      <canvas ref={canvasRef} width={size.width} height={size.height}
        aria-label={`Interactive ${grid.x_label ?? "mean"} and ${grid.y_label ?? "std"} gate-weight surface for ${memberLabel}`}
        style={{ display: "block", width: "100%", height: size.height, cursor: dragRef.current ? "grabbing" : "grab", touchAction: "none" }}
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          dragRef.current = { x: event.clientX, y: event.clientY, yaw: view.yaw, pitch: view.pitch };
        }}
        onPointerMove={(event) => {
          const drag = dragRef.current;
          if (!drag) return;
          onViewChange({ ...view, yaw: drag.yaw + (event.clientX - drag.x) * 0.012,
            pitch: Math.max(-1.35, Math.min(1.35, drag.pitch + (event.clientY - drag.y) * 0.01)) });
        }}
        onPointerUp={(event) => { dragRef.current = null; event.currentTarget.releasePointerCapture(event.pointerId); }}
        onPointerCancel={() => { dragRef.current = null; }}
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
          event.preventDefault();
          onMemberStep?.(event.key === "ArrowLeft" ? -1 : 1);
        }}
        onWheel={(event) => {
          event.preventDefault();
          onViewChange({ ...view,
            zoom: Math.max(0.55, Math.min(2.5, view.zoom * (event.deltaY > 0 ? 0.9 : 1.1))),
          });
        }} />
      <div className="row muted" style={{ padding: "6px 10px", justifyContent: "space-between", fontSize: 12, gap: "var(--s3)" }}>
        <span>{memberLabel} · {grid.z_label?.includes("suppression") ? "max suppression" : "max weight"} {(maxWeight * 100).toFixed(1)}%</span>
      <Button variant="ghost" onClick={() => onViewChange(resetView ?? { yaw: -0.72, pitch: 0.58, zoom: 1 })}>reset view</Button>
      </div>
    </div>
  );
}

function PredictiveAxesCard(
  { axes, loading, mode, theme, fitJob, onFit }:
  { axes: PredictiveAxes | null; loading: boolean; mode: string; theme: string;
    fitJob: ReturnType<typeof useJob>; onFit: () => void },
) {
  const [nImg, setNImg] = useState("100");
  const [route, setRoute] = useState("0");
  const defaultView: SurfaceView = { yaw: -0.72, pitch: 0.58, zoom: 0.62 };
  const [view, setView] = useState<SurfaceView>(defaultView);
  const labels = asArray<string>(axes?.route_labels);
  const densitySelected = route === "density";
  const selected = densitySelected ? 0
    : Math.max(0, Math.min(labels.length - 1, Number(route) || 0));
  useEffect(() => {
    if (!densitySelected && selected >= labels.length) setRoute("0");
  }, [densitySelected, labels.length, selected]);
  const grid = useMemo<FeatureGrid | null>(() => {
    const axis1 = asArray<number | null>(axes?.axis1);
    const axis2 = asArray<number | null>(axes?.axis2);
    let weights = asArray<(number | null)[][]>(axes?.oracle_weights);
    if (axis1.length < 2 || axis2.length < 2 || !weights.length) return null;
    if (densitySelected) {
      const density = asArray<(number | null)[]>(axes?.density);
      const maxDensity = Math.max(1, ...density.flat().map((value) => Number(value) || 0));
      weights = density.map((row) => row.map((value) => [
        Math.log1p(Number(value) || 0) / Math.log1p(maxDensity),
      ]));
    }
    return {
      mean_asinh: axis1,
      std_asinh: axis2,
      std_log: axis2,
      x_label: "predictive axis 1",
      y_label: "predictive axis 2",
      y_is_log: false,
      z_label: densitySelected ? "log validation density [0–1]"
        : "soft oracle responsibility [0–1]",
      weights,
    };
  }, [axes, densitySelected]);
  const two = axes?.metrics?.two_axis;
  const full = axes?.metrics?.full_input_linear;
  const loadings = asArray<(number | null)[]>(axes?.axis_loadings);
  const names = asArray<string>(axes?.feature_names);
  const loadingSummary = (axis: number) => num(loadings[axis]).map((value, i) => ({
    name: names[i] ?? `feature ${i + 1}`, value,
  })).filter((item) => isFinite(item.value))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 5)
    .map((item) => `${item.value >= 0 ? "+" : "−"}${item.name}`).join(" · ");
  const pct = (value?: number | null) => value == null || !isFinite(value)
    ? "—" : `${(100 * value).toFixed(1)}%`;
  const regret = (value?: number | null) => value == null || !isFinite(value)
    ? "—" : value.toFixed(5);

  return (
    <Card>
      <CardHead title={`Predictive routing axes · ${mode}`}
        sub="Two supervised PLS axes fitted from ordered member-band pixels to soft oracle expert responsibilities. This is a diagnostic, not an image combiner."
        right={axes?.available && <Badge tone={axes.stale ? "warn" : "good"}>{axes.stale ? "stale" : "fitted"}</Badge>} />
      <CardBody>
        <div className="row" style={{ alignItems: "flex-end", gap: "var(--s3)" }}>
          <NumberField label="validate fields" value={nImg} onChange={setNImg} min={1} max={2000} />
          <Button variant="primary" disabled={fitJob.busy}
            onClick={() => fitJob.run("/ensemble/predictive-axes/fit", {
              num_images: nImg, mode,
            }, { onDone: onFit })}>
            Fit two predictive axes
          </Button>
        </div>
        <JobProgressView job={fitJob.job} error={fitJob.error} />
        {loading ? <Empty><Spinner /> loading…</Empty>
          : !axes?.available ? <Empty>{axes?.reason ?? "predictive axes have not been fitted"}</Empty>
          : <div style={{ marginTop: "var(--s4)" }}>
            <DefList items={[
              ["fit", `${axes.n_fields ?? "—"} fields · ${(axes.n_train ?? 0).toLocaleString()} train · ${(axes.n_holdout ?? 0).toLocaleString()} held out`],
              ["two-axis soft-route R²", two?.soft_route_r2 == null ? "—" : two.soft_route_r2.toFixed(3)],
              ["full-input linear R²", full?.soft_route_r2 == null ? "—" : full.soft_route_r2.toFixed(3)],
              ["R² retained in 2D", pct(axes.metrics?.r2_retention)],
              ["two-axis top-route accuracy", pct(two?.top_route_accuracy)],
              ["full-input top-route accuracy", pct(full?.top_route_accuracy)],
              ["two-axis selected regret", regret(two?.mean_selected_regret)],
              ["ensemble-mean regret", regret(axes.metrics?.ensemble_mean_regret)],
            ]} />
            <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
              Regret is equal-band asinh L1 above the best available expert on held-out validation pixels.
              The split is by sampled pixel, so this is a separability diagnostic rather than a field-level generalization claim.
            </div>
            {grid && labels.length > 0 && <div style={{ marginTop: "var(--s4)" }}>
              <div className="row" style={{ justifyContent: "space-between", marginBottom: 8, gap: "var(--s3)" }}>
                <div className="eyebrow">soft oracle responsibility · supervised axis 1 × axis 2</div>
                <Select<string> value={route} onChange={setRoute}
                  options={[
                    ...labels.map((label, index) => ({ value: String(index), label })),
                    { value: "density", label: "validation density" },
                  ]} />
              </div>
              <WeightSurface3D grid={grid} member={selected}
                memberLabel={densitySelected ? "validation density"
                  : labels[selected] ?? `route ${selected + 1}`} theme={theme}
                stdAxis="raw" view={view} resetView={defaultView} onViewChange={setView}
                onMemberStep={densitySelected ? undefined : (step) => setRoute(String(
                  Math.max(0, Math.min(labels.length - 1, selected + step))))} />
              <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                {axes.conditioning_note} Dominant inputs: axis 1 {loadingSummary(0) || "—"}; axis 2 {loadingSummary(1) || "—"}.
              </div>
            </div>}
          </div>}
      </CardBody>
    </Card>
  );
}

export function CombinerCard(
  { comb, loading, mode, theme, fitJob, onFit, evalReady,
    fitUrl = "/ensemble/combiner/fit", title, modelKind: controlledKind }:
  { comb: Combiner | null; loading: boolean; mode: string; theme: string; fitJob: ReturnType<typeof useJob>; onFit: () => void; evalReady: boolean;
    fitUrl?: string; title?: string; modelKind?: CombinerModelKind },
) {
  const [nImg, setNImg] = useState("100");
  const [nKernels, setNKernels] = useState("128");
  const [localModelKind] = useState<CombinerModelKind>("raw_incremental_minmeanmax_rbf");
  const [band, setBand] = useState<string>("");
  const [colorBy, setColorBy] = useState<GateColorBy>("loss");
  const [surfaceMember, setSurfaceMember] = useState("0");
  const [surfaceStdAxis, setSurfaceStdAxis] = useState<"raw" | "log">("log");
  const [surfaceView, setSurfaceView] = useState<SurfaceView>({ yaw: -0.72, pitch: 0.58, zoom: 1 });
  const pcaDefaultView: SurfaceView = { yaw: -0.72, pitch: 0.58, zoom: 0.5 };
  const [pcaSurfaceView, setPcaSurfaceView] = useState<SurfaceView>(pcaDefaultView);
  const [focusedSurfaceBand, setFocusedSurfaceBand] = useState<string | null>(null);
  const trackedFitJob = useTrackedJob(`combiner: fit ${mode} `);
  const trackedCompletion = useRef<string | null>(null);

  const modelKind = controlledKind ?? localModelKind;
  const visibleFitJob = fitJob.job ?? trackedFitJob;
  const backgroundFitRunning = trackedFitJob?.status === "running";

  useEffect(() => {
    if (!fitJob.job && trackedFitJob && trackedFitJob.status !== "running"
        && trackedCompletion.current !== trackedFitJob.job_id) {
      trackedCompletion.current = trackedFitJob.job_id;
      onFit();
    }
  }, [fitJob.job, onFit, trackedFitJob]);

  const bands = asArray<string>(comb?.band_names);
  const activeBand = band || bands[0] || "";
  const fittedConvexGate = comb?.kind === "raw_incremental_minmeanmax_rbf";
  const centerHistory = asArray<CenterStage>(comb?.fit_meta?.center_history);
  const minibatchedFit = comb?.fit_meta?.training_mode === "all_validation_pixels_minibatch";
  const fittedModelLabel = "minibatched convex all-asinh RBF";
  const surfaceLabels = asArray<string>(comb?.pca_weight_surface?.surface_labels);
  const selectedSurfaceMember = Math.max(0, Math.min(surfaceLabels.length - 1, Number(surfaceMember) || 0));
  const surfaceBands = bands.filter((b) => comb?.feature_grid?.[b]);
  const pcaDiagnostic = comb?.pca_weight_surface?.available
    ? comb.pca_weight_surface : null;
  const pcaSurface = pcaDiagnostic;
  const pcaVariance = num(pcaSurface?.explained_variance_ratio);
  const pcaLoadingSummary = (component: number) => {
    const names = asArray<string>(pcaSurface?.feature_names);
    const values = num(asArray<(number | null)[]>(pcaSurface?.loadings)[component]);
    return values.map((value, i) => ({ name: names[i] ?? `feature ${i + 1}`, value }))
      .filter((item) => isFinite(item.value))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 3)
      .map((item) => `${item.value >= 0 ? "+" : "−"}${item.name}`).join(" · ");
  };

  useEffect(() => {
    if (selectedSurfaceMember >= surfaceLabels.length) setSurfaceMember("0");
  }, [selectedSurfaceMember, surfaceLabels.length]);

  const [importanceSort, setImportanceSort] = useState<"index" | "integrated">("index");
  const [importanceSortDescending, setImportanceSortDescending] = useState(true);

  const importance = useMemo(() => {
    if (!comb) return [];
    const labels = asArray<string>(comb.member_labels);
    const memberMeta = asArray<NonNullable<Combiner["members"]>[number]>(comb.members);
    const sharedBand = bands[0];
    const rows = labels
      .map((label, i) => {
        const peak = sharedBand ? comb.member_weight_peaks?.[sharedBand]?.[i] ?? null : null;
        const integrated = sharedBand ? comb.member_weight_integrals?.[sharedBand]?.[i] ?? null : null;
        return { i, label, meta: memberMeta[i],
          peak, integrated,
          total: integrated == null || !Number.isFinite(integrated) ? 0 : integrated };
      });
    return rows.sort((a, b2) => importanceSort === "integrated"
      ? (importanceSortDescending ? b2.total - a.total : a.total - b2.total) || a.i - b2.i
      : a.i - b2.i);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bands, comb, importanceSort, importanceSortDescending, theme]);

  const weightColumns = useMemo<Column<(typeof importance)[number]>[]>(() => [
    { header: "member", cell: (row) => <code>{row.label}</code> },
    { header: <button type="button" className="ui-table__sort-button"
        onClick={() => { setImportanceSort("integrated"); setImportanceSortDescending((v) => importanceSort === "integrated" ? !v : true); }}>
        integrated weight{importanceSort === "integrated" ? (importanceSortDescending ? " ↓" : " ↑") : ""}
      </button>, align: "right",
      cell: (row) => row.integrated == null ? "—" : `${(100 * row.integrated).toFixed(2)}%` },
    { header: "peak weight", align: "right",
      cell: (row) => row.peak == null ? "—" : `${(100 * row.peak).toFixed(2)}%` },
    { header: "PSNR", align: "right",
      cell: (row) => row.meta?.psnr == null ? "—" : `${row.meta.psnr.toFixed(2)} dB` },
    { header: "loss", cell: (row) => (row.meta?.loss ?? "—").toUpperCase() },
    { header: "depth", align: "right", cell: (row) => row.meta?.blocks ?? "—" },
  ], [importance, importanceSort, importanceSortDescending]);

  const surfaceMembers = importance;
  const surfaceMemberPosition = surfaceMembers.findIndex((r) => r.i === selectedSurfaceMember);
  const moveSurfaceMember = (step: -1 | 1) => {
    const next = surfaceMemberPosition + step;
    if (surfaceMemberPosition >= 0 && next >= 0 && next < surfaceMembers.length) {
      setSurfaceMember(String(surfaceMembers[next].i));
    }
  };
  useEffect(() => {
    if (fittedConvexGate && surfaceMembers.length && surfaceMemberPosition < 0) {
      setSurfaceMember(String(surfaceMembers[0].i));
    }
  }, [fittedConvexGate, selectedSurfaceMember, surfaceMemberPosition, surfaceMembers]);

  // Shared facet colorer (member index → colour by `colorBy`) so the importance
  // bars and the gate-weight curves colour a member the SAME way — and both
  // recolour when the characteristic changes. depth/knee → categorical over the
  // ensemble's distinct values; psnr → viridis over the members' PSNR
  // range; loss → loss token.
  const memberColor = useMemo(() => {
    const members = asArray<NonNullable<Combiner["members"]>[number]>(comb?.members);
    const M = asArray<string>(comb?.member_labels).length;
    const memberIdx = Array.from({ length: M }, (_, m) => m);
    const depths = [...new Set(members.map((mm) => mm?.blocks ?? 0))].sort((a, b) => a - b);
    const knees = [...new Set(members.map((mm) => kneeOf(mm?.asinh_knee)))].sort((a, b) => a - b);
    const psnrs = memberIdx.map((m) => members[m]?.psnr).filter((p): p is number => p != null && isFinite(p));
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
    const members = asArray<NonNullable<Combiner["members"]>[number]>(comb?.members);
    const labels = asArray<string>(comb?.member_labels);
    const M = labels.length;
    const memberIdx = Array.from({ length: M }, (_, m) => m);

    const xs = bx.filter((v) => isFinite(v));
    if (!xs.length) return null;
    const xDomain: [number, number] = [Math.min(...xs), Math.max(...xs)];
    const series: Series[] = memberIdx.map((m) => ({
      x: bx, y: jacobian.map((row) => (asArray<number | null>(row)[m] ?? null)), color: memberColor(m), width: 1.8,
    }));
    const legend = memberIdx.map((m) => {
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

  const hrGate = useMemo(() => {
    const d = comb?.hr_weights?.bands?.[activeBand];
    const xs = asArray<number | null>(d?.brightness_asinh);
    const mean = asArray<(number | null)[]>(d?.mean);
    const counts = asArray<number>(d?.counts);
    if (!d || !xs.length || !mean.length) return null;
    const valid = xs.map((x, i) => x != null && isFinite(x) && (counts[i] ?? 0) > 0);
    const finiteX = xs.filter((x, i) => valid[i]).map((x) => x as number);
    if (!finiteX.length) return null;
    const labels = asArray<string>(comb?.member_labels);
    const members = asArray<NonNullable<Combiner["members"]>[number]>(comb?.members);
    const M = labels.length;
    const memberIdx = Array.from({ length: M }, (_, m) => m);
    const plotX = xs.map((x) => x == null ? NaN : x);
    const series: Series[] = memberIdx.map((m) => ({
      x: plotX,
      y: mean.map((row) => asArray<number | null>(row)[m] ?? null),
      color: memberColor(m), width: 1.8,
    }));
    const legend = memberIdx.map((m) => ({
      label: labels[m] || members[m]?.label || String(m), color: memberColor(m),
    }));
    const xDomain: [number, number] = [Math.min(...finiteX), Math.max(...finiteX)];
    const xTicks: Tick[] = [0, 0.25, 0.5, 0.75, 1].map((f) => {
      const v = xDomain[0] + f * (xDomain[1] - xDomain[0]);
      return { v, label: Math.round(100 * Math.sinh(v)).toString() };
    });
    return {
      series, legend, xDomain, xTicks, yDomain: [0, 1.02] as [number, number],
      nFields: comb.hr_weights?.n_fields ?? 0,
      nPixels: comb.hr_weights?.n_pixels ?? 0,
      target: comb.hr_weights?.target ?? "HR",
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [comb, activeBand, memberColor, theme]);

  return (
    <Card>
      <CardHead title={title ?? `Combiner · ${mode}`}
        sub="all validation pixels · field-separated minibatch fit · scored only afterward on test"
        right={comb?.available && <Badge tone={comb.stale || comb.fit_meta?.preview ? "warn" : "good"}>
          {comb.stale ? "stale" : comb.fit_meta?.preview ? `preview · ${comb.fit_meta.num_images ?? "?"} fields` : "fitted"}
        </Badge>} />
      <CardBody>
        <div className="row" style={{ alignItems: "flex-end", gap: "var(--s3)" }}>
          <NumberField label="validate fields" value={nImg} onChange={setNImg} min={1} max={2000} />
          <NumberField label="kernels (K)" value={nKernels} onChange={setNKernels} min={2} />
          <Button variant="primary" disabled={fitJob.busy || backgroundFitRunning || !evalReady}
            title={evalReady ? undefined : `evaluate ${mode} on the test set first — its evaluation must match the current members`}
            onClick={() => fitJob.run(fitUrl, {
              num_images: nImg, n_kernels: nKernels,
              model_kind: modelKind, mode,
            }, { onDone: onFit })}>
            Fit convex all-asinh RBF
          </Button>
        </div>
        {!evalReady && (
          <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
            Evaluations for <b>{mode}</b> aren’t ready — run “Evaluate on test set” (for the current members) before fitting a combiner.
          </div>
        )}
        <JobProgressView job={visibleFitJob} error={fitJob.error} />

        {loading ? <Empty><Spinner /> loading…</Empty>
          : !comb?.available ? <Empty>{comb?.reason ?? `no combiner fitted for ${mode} yet — set knobs above and fit.`}</Empty> : (
          <div style={{ marginTop: "var(--s4)" }}>
            <DefList items={[
              ["members", String(asArray<string>(comb.source_member_labels).length)] as [string, string],
              ["initialization", comb.fit_meta?.initial_best_member_label
                ? `99% ${comb.fit_meta.initial_best_member_label} (best PSNR)`
                : "best-PSNR member prior"] as [string, string],
              ["weights", "one shared softmax vector for all bands"] as [string, string],
              ["model", fittedModelLabel] as [string, string],
              ["kernels", String(comb.n_kernels ?? "—")] as [string, string],
              ...(comb.fit_meta?.preview ? [["fit scope", `preview · ${comb.fit_meta.num_images ?? "?"} validation fields`] as [string, string]] : []),
              ["validate asinh L1", comb.val_l1 != null ? comb.val_l1.toFixed(4) : "—"] as [string, string],
            ]} />

            {fittedConvexGate && centerHistory.length > 0 && (
              <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
                {minibatchedFit ? <>
                  16-kernel staged refits: {centerHistory.map((stage) => stage.n_centers ?? "?").join(" → ")} centers.
                  {" "}Training asinh L1: {centerHistory.map((stage) => stage.train_l1?.toFixed(5) ?? "—").join(" → ")}.
                  {" "}Mean achievable placement gain: {centerHistory.map((stage) => stage.candidate_mean_achievable_gain?.toFixed(5) ?? "—").join(" → ")}.
                  {" "}{comb.fit_meta?.training_fields ?? "?"} training fields ({comb.fit_meta?.training_pixels_per_epoch?.toLocaleString() ?? "?"} pixels/epoch)
                  {" "}and {comb.fit_meta?.validation_fields ?? "?"} disjoint validation fields ({comb.fit_meta?.validation_pixels_per_epoch?.toLocaleString() ?? "?"} pixels).
                  {" "}Saved stage: {comb.fit_meta?.selected_stage ?? 0}; minibatch size: {comb.fit_meta?.batch_rows?.toLocaleString() ?? "?"}.
                </> : <>
                  Recoverable-error center growth: {centerHistory.map((stage) => stage.n_centers ?? "?").join(" → ")} requested centers.
                  {" "}Mean recoverable asinh L1 used for placement: {centerHistory.map((stage) => stage.center_weight_mean_recoverable_l1?.toFixed(4) ?? "—").join(" → ")}.
                </>}
                {" "}Validation asinh L1: {centerHistory.map((stage) => stage.val_l1?.toFixed(3) ?? "—").join(" → ")}.
                {" "}Validation VIS asinh PSNR: {centerHistory.map((stage) => stage.val_vis_asinh_psnr?.toFixed(3) ?? "—").join(" → ")} dB.
                {" "}Optimizer minibatches: {centerHistory.map((stage) => stage.optimizer_iterations ?? "—").join(" → ")}.
              </div>
            )}

            {fittedConvexGate && importance.length > 0 && (
              <div style={{ marginTop: "var(--s4)" }}>
                <div className="eyebrow" style={{ marginBottom: 8 }}>
                  member importance · validation-distribution integrated shared weight
                </div>
                <Table columns={weightColumns} rows={importance}
                  rowKey={(row) => row.i} />
                <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                  Integrated weight is the mean learned shared softmax weight over sampled full-dimensional validation pixels; values sum to 100%.
                </div>
              </div>
            )}

            {bands.length > 0 && (
              <div style={{ marginTop: "var(--s4)" }}>
                <div className="row" style={{ justifyContent: "space-between", marginBottom: 8, gap: "var(--s3)" }}>
                  <div className="eyebrow">shared convex-weight diagnostic band</div>
                  <Segmented<string> value={activeBand} onChange={setBand} options={bands.map((b) => ({ value: b, label: b }))} />
                </div>
                {fittedConvexGate && pcaSurface && surfaceMembers.length > 0 && (
                  <div style={{ marginTop: "var(--s4)" }}>
                    <div className="row" style={{ justifyContent: "space-between", marginBottom: 8, gap: "var(--s3)" }}>
                      <div className="eyebrow">shared convex member weight on validation PCA · PC1 × PC2</div>
                      <div className="row" style={{ gap: 4 }}>
                        <Button size="sm" variant="ghost" title="previous member"
                          onClick={() => moveSurfaceMember(-1)}
                          disabled={surfaceMemberPosition <= 0}>←</Button>
                        <Select<string> value={surfaceMember} onChange={setSurfaceMember}
                          options={surfaceMembers.map((r) => ({ value: String(r.i), label: r.label }))} />
                        <Button size="sm" variant="ghost" title="next member"
                          onClick={() => moveSurfaceMember(1)}
                          disabled={surfaceMemberPosition < 0 || surfaceMemberPosition >= surfaceMembers.length - 1}>→</Button>
                      </div>
                    </div>
                    <WeightSurface3D grid={pcaSurface} member={selectedSurfaceMember}
                      memberLabel={surfaceLabels[selectedSurfaceMember] ?? `member ${selectedSurfaceMember}`} theme={theme}
                      stdAxis="raw" view={pcaSurfaceView} resetView={pcaDefaultView}
                      onViewChange={setPcaSurfaceView}
                      onMemberStep={moveSurfaceMember} />
                    <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                      PCA uses {pcaSurface.n_pixels?.toLocaleString() ?? "—"} sampled validation pixels in the model’s scale-normalized full member-inference feature space.
                      {" "}PC1 explains {isFinite(pcaVariance[0]) ? `${(100 * pcaVariance[0]).toFixed(1)}%` : "—"}; PC2 explains {isFinite(pcaVariance[1]) ? `${(100 * pcaVariance[1]).toFixed(1)}%` : "—"}.
                      {" "}Weights are evaluated in the complete input space and locally integrated after projection using {pcaSurface.integration_neighbors ?? "—"} neighboring validation pixels; omitted PCs are not fixed to their mean.
                      {" "}Dominant loadings: PC1 {pcaLoadingSummary(0) || "—"}; PC2 {pcaLoadingSummary(1) || "—"}.
                    </div>
                  </div>
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

export function DisagreementCard(
  { mode, members, collection = "ensemble", targetLabel = "HR" }:
  { mode: string; members: Member[]; collection?: string; targetLabel?: string },
) {
  // ttl:0 → always background-revalidate. The cube cache is wiped+rebuilt during
  // an evaluation, so a meta cached mid-run is briefly empty; revalidating on
  // every mount lets the panel self-heal (shows cached instantly, then refreshes
  // to the real membership) instead of staying stuck on "no members".
  const viewerQuery = collection === "ensemble" ? `?mode=${mode}` : "";
  const meta = useResource<ViewMeta>(`/viewer/meta/${collection}${viewerQuery}`, [collection, mode], { ttl: 0 });
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
        sub={`LR · mean · mean+std RBF · min+max RBF · stacked RBF · ${targetLabel} · movie — pick members to see where those reconstructions disagree`} />
      <CardBody>
        <CutoutViewer collection={collection} params={collection === "ensemble" ? { mode } : {}}
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
