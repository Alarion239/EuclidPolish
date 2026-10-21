/* Models › Universal (`/models/universal`): the universal models — members
   trained at several knees at once with ONE output image (a fixed or a
   learned output knee) — as first-class models, set against the combiner
   by training time. From the top:
   - one summary line: how many, the best and its training time, against the
     production gate and its weighted training time; its caption defines ∫PSNR
     and the training time;
   - callouts with their confirmed jobs: "Score N models on the 100 test
     fields" (those without a current score), "Measure gate weights" (gates
     whose shares are measurable but not measured: their hours are unknown),
     "Pull N models from FASRC" (the universal models that finished there
     COMPLETED / TIMEOUT; needs FASRC; the failed ones and the registered
     ones whose pull left no checkpoint are named, each pulled from its row
     of the FASRC table), a stale production score → the Leaderboard; the
     payload's own notes;
   - the toolbar (URL state): x axis training time | steps (?x=), the metric
     ∫PSNR mean | VIS | Y | J | H (?y=), the combiner weighting max weight |
     mean weight | every member (?w=), "Compare all on real fields" (Sky ›
     Compare's new comparison with production and every local universal
     spec), Log to notebook and the Run menu;
   - the plot by training time (log hours: a dot per model coloured by depth,
     a ring for a MIG slice of gpu_test, faded for a stale score at the hours
     it was scored; the combiners as diamonds at their weighted hours,
     production in the combiner colour, a thin diamond at a lower bound when
     a member it reads has no hours; production and the plain mean as lines)
     or by training length (log steps: a line per depth through its models);
     a click opens the nearest model;
   - the models table (per band ∫PSNR hidden; Δ vs production and vs the
     time-matched combiner, named — only against a combiner scored on the
     same fields with a current score, a stale model's Δ marked; real holes /
     R̃ from the newest Sky › Compare run that scored THIS model; a row opens
     the member inspector; "Compare on real fields", "Make production…"), the
     combiners table ("≥" hours for a partial one) and the universal models on
     FASRC (Pull, Try pull, Pull again per row).
   Production is the production gate, or one of these models (the payload's
   `production_model`, its row badged "production"): then every "production"
   reference — the summary, the line across, Δ vs production — is that
   model's own score, and the gate is the former production gate.
   Opening the page reads caches only (GETs: the payload, the Sky › Compare
   runs and their records, the catalogue); every job and the production
   switch ask first. */
import { useMemo, useState, type ReactNode } from "react";
import { Link, useNavigate } from "react-router-dom";
import Plot, { Legend, useLegend, type Guide, type LegendItem, type Series } from "../../../charts/Plot";
import { C, categorical } from "../../../colors";
import { useJob } from "../../../api/jobs";
import { openInspector } from "../../../app/inspector";
import { pagePath } from "../../../app/nav";
import { usePageActions } from "../../../app/palette";
import { useFasrcStatus } from "../../../app/status";
import { formatDate, formatRelative } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { logTicks, paddedDomain } from "../../../ticks";
import {
  Badge, Button, Callout, Caption, DataTable, EmptyState, IconButton, JobProgress, Menu, Num, Page, Segmented, SummaryLine, Table, Toolbar,
  ToolbarGroup, ToolbarSpacer, Tooltip, type Column, type DataColumn,
} from "../../../ui";
import {
  BAND_SHORT, BANDS, useExperimentDetails, useExperiments, useModelCatalog, useUniversal,
  type UniversalCombiner, type UniversalFasrcRow,
} from "../api";
import { LoadState, openMember } from "../common";
import { JOB, measureGateWeights, pullUniversal, scoreUniversal, setProduction, useOnJobEnd } from "../jobs";
import { compareRunPath, db, dbDelta, kneeNum, memberNumber } from "../model";
import { universalNote } from "../notes";
import { gateName, universalLabel } from "../production";
import {
  METRICS, WEIGHTINGS, X_AXES, classText, combinerComparable, combinerHours, combinerHoursText, combinerName, combinerPoints, compareHref, depthKey,
  depthLabel,
  depthOrder, depthSlot, fasrcKind, fasrcSplit, hardwareText, hoursText, isComparable, isPartial, kneeWords, learnedKneesText, lengthLines,
  logDomain, nearestPoint, numbersText, parseMetric, parseWeighting, parseXAxis, pendingWeights, preferCurrent, productionLabel, productionReference,
  metricValue, realPicks, scoredFields, shapeRows, stepImplText, stepsShort, STATUS_TEXT, timePoints, toScore, universalSummary, type Metric, type ModelPoint,
  type RealPick, type ShapedRow, type Weighting, type XAxis,
} from "../universal";
import { LogToNotebookButton } from "../../shared/LogToNotebook";
import "../models.css";

const tabPath = (tab: string) => pagePath("models", { tab });
const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;
const weightText = (w: Weighting) => WEIGHTINGS.find((x) => x.value === w)?.text ?? w;
const hoursNum = (h: number) => hoursText(h).replace(/ h$/, "");
/** A server note as a sentence: capitalised, with a full stop. */
const sentence = (t: string) => {
  const s = t.trim();
  return s ? `${s[0].toUpperCase()}${s.slice(1)}${/[.!?]$/.test(s) ? "" : "."}` : s;
};

/* ── summary ───────────────────────────────────────────────────────────── */

/** The best model against production: the production gate (its hours under
 *  the weighting) or the universal model that is production (its own score
 *  and hours; "is production" when it is the best). */
function Summary({ s, weighting }: { s: ReturnType<typeof universalSummary>; weighting: Weighting }) {
  if (!s.n) return null;
  const b = s.best;
  if (!b || b.mean == null) {
    return <SummaryLine>{plural(s.n, "universal model")}, none scored on the test fields yet.</SummaryLine>;
  }
  const prod = s.production;
  const gain = s.gain;
  const isProd = !!prod?.name && b.name === prod.name;
  const prodHours = prod?.hours == null ? ""
    : prod.label ? `; ${hoursText(prod.hours)}` : `; ${prod.partial ? "≥ " : ""}${hoursText(prod.hours)} ${weightText(weighting)}`;
  const prodText = prod?.mean != null ? ` (${db(prod.mean)} dB${prodHours})` : "";
  const what = prod?.label ? `production, ${prod.label}` : "the production gate";
  return (
    <SummaryLine>
      {plural(s.n, "universal model")}; the best, <Num>#{b.num}</Num>, {isProd ? "is production and scores" : "scores"}{" "}
      <Num unit="dB">{db(b.mean)}</Num> integrated PSNR
      {b.hours != null && <> after <Num unit="h">{hoursNum(b.hours)}</Num> of training</>}
      {!isProd && gain != null && (gain >= 0
        ? <>, <Num>{dbDelta(gain)}</Num> dB over {what}{prodText}</>
        : <>, <Num tone="warn">{db(-gain)}</Num> dB under {what}{prodText}</>)}
      {s.bestStale && " (a stale score)"}
    </SummaryLine>
  );
}

/* ── real-field cells ──────────────────────────────────────────────────── */

function RealHoles({ pick }: { pick: RealPick | undefined }) {
  if (!pick || pick.state === "none") return <span className="mdl-faint">—</span>;
  if (pick.state === "loading") return <span className="mdl-faint">checking…</span>;
  if (pick.state === "earlier") {
    return (
      <Tooltip content={`The newest Sky › Compare run of this model${pick.created ? ` (${formatDate(pick.created)})` : ""} scored an earlier checkpoint. Compare it on real fields again.`}>
        <span tabIndex={0} className="mdl-faint">earlier fit</span>
      </Tooltip>
    );
  }
  const b = pick.bench;
  const w = b.worst;
  const per = b.bands.map((x) => `${x.short} ${x.pct == null ? "—" : x.pct.toFixed(1)}`).join(" · ");
  return (
    <Tooltip content={`Holes per band: ${per || "—"} % · from the Sky › Compare run ${pick.label ? `“${pick.label}”` : pick.expId} on ${pick.tileSet}${pick.created ? ` (${formatDate(pick.created)})` : ""}`}>
      <Link className="mdl-tnum" to={compareRunPath(pick.expId)}>
        {w?.pct != null ? `${w.short} ${w.pct.toFixed(0)}` : b.holeMean != null ? b.holeMean.toFixed(0) : "—"}
      </Link>
    </Tooltip>
  );
}

function statusCell(r: ShapedRow): ReactNode {
  if (r.status === "stale") return <Badge tone="warn" size="sm">{STATUS_TEXT.stale}</Badge>;
  if (r.status === "scored") return <span className="mdl-muted">{STATUS_TEXT.scored}</span>;
  return <span className="mdl-faint">{STATUS_TEXT[r.status]}</span>;
}

/* ── FASRC rows ────────────────────────────────────────────────────────── */

const stateText = (r: UniversalFasrcRow) => (r.state ?? (r.finished ? "finished" : "running")).toLowerCase();

function fasrcState(r: UniversalFasrcRow): ReactNode {
  const kind = fasrcKind(r);
  if (kind === "training") return <Badge tone="info" size="sm" dot>{stateText(r)}</Badge>;
  if (kind === "try") return <Badge tone="warn" size="sm">{stateText(r)}</Badge>;
  if (kind === "again") {
    return (
      <Tooltip content="Registered as a universal model, but its last pull left no checkpoint here">
        <span tabIndex={0} className="mdl-warn">{stateText(r)} · not pulled</span>
      </Tooltip>
    );
  }
  return <span className="mdl-muted">{stateText(r)}</span>;
}

const FASRC_COLUMNS: Column<UniversalFasrcRow>[] = [
  { header: "Model", cell: (r) => <span className="mdl-member">#{r.name.replace(/^member_/, "")}</span> },
  { header: "Depth", align: "right", cell: (r) => <span className="mdl-tnum">{r.num_res_blocks ?? "—"}</span> },
  { header: "Steps", align: "right", cell: (r) => <span className="mdl-tnum">{stepsShort(r.steps)}</span> },
  { header: "Knee", cell: (r) => r.knee ?? "—" },
  { header: "State", cell: fasrcState },
  { header: "Partition", cell: (r) => r.partition ?? "—" },
  { header: "Jobs", cell: (r) => <span className="mdl-mono">{r.jobids.join(", ") || "—"}</span> },
  { header: "Submitted", cell: (r) => (r.submitted_at ? formatRelative(r.submitted_at) : "—") },
];

/** "A stale score: …" — the tooltip of a stale model's Δ cells. */
const STALE_DELTA = "A stale score (an earlier checkpoint or other test fields): re-score the model before reading this Δ";

/* ── the tab ───────────────────────────────────────────────────────────── */

type Drawn = ModelPoint & { inspect: { kind: "member" | "combiner"; id: string } };

export default function Universal() {
  const navigate = useNavigate();
  const res = useUniversal();
  const fasrc = useFasrcStatus().data;
  const offline = fasrc ? !fasrc.ssh_connected : false;
  const exps = useExperiments();
  const catalog = useModelCatalog();
  const scoreJob = useJob(JOB.universalScore);
  const pullJob = useJob(JOB.universalPull);
  useOnJobEnd(scoreJob.job);
  useOnJobEnd(pullJob.job);
  const [x, setX] = useUrlState<XAxis>("x", "time", { parse: parseXAxis });
  const [metric, setMetric] = useUrlState<Metric>("y", "mean", { parse: parseMetric });
  const [weighting, setWeighting] = useUrlState<Weighting>("w", "max", { parse: parseWeighting });
  const lg = useLegend();

  const data = res.data;
  const bands = useMemo(() => (data?.bands?.length ? data.bands : [...BANDS]), [data]);
  const rows = useMemo(() => (data ? shapeRows(data, { weighting, metric }) : []), [data, weighting, metric]);
  const order = useMemo(() => depthOrder(rows), [rows]);
  const colorOf = (d: number | null) => categorical(depthSlot(order, d));
  const summary = useMemo(() => (data ? universalSummary(rows, data, weighting) : null), [data, rows, weighting]);
  const nFields = scoredFields(data);
  const scoreNames = useMemo(() => toScore(data?.models ?? []), [data]);
  const split = useMemo(() => fasrcSplit(data?.fasrc ?? []), [data]);
  const localSpecs = useMemo(() => rows.filter((r) => r.fingerprint).map((r) => r.spec), [rows]);
  const metricInfo = METRICS.find((m) => m.value === metric) ?? METRICS[0];
  // Production: the gate, or one of these models (its own score is then the reference).
  const prodName = data?.production_model ?? null;
  const prodLabel = productionLabel(data);
  const prodSpec = prodName ? `universal:${prodName}` : null;
  const gateWord = gateName(prodName);
  const prodRow = prodName ? rows.find((r) => r.name === prodName) ?? null : null;
  const prodRef = data ? productionReference(data) : null;
  const prodValue = metricValue(prodRef, metric, bands);
  const meanValue = metricValue(data?.mean_model, metric, bands);
  const comparable = data ? isComparable(data) : true;
  const combs = useMemo(() => (data ? combinerPoints(data.combiners ?? [], weighting, metric, bands, comparable, data.production_model ?? null) : []),
    [data, weighting, metric, bands, comparable]);
  const pending = useMemo(() => pendingWeights(data), [data]);
  const pendingIds = useMemo(() => new Set(pending.map((g) => g.id)), [pending]);

  /* real fields: each spec's newest Sky › Compare run that scored THIS model
     (the walk names the run records it needs; it grows as they arrive); the
     production model's row also takes a run of `production`, which runs it */
  const specs = useMemo(() => [...rows.map((r) => r.spec), ...(prodName ? ["production"] : [])], [rows, prodName]);
  const [ids, setIds] = useState<string[]>([]);
  const records = useExperimentDetails(ids);
  // An unreadable catalogue leaves every run "earlier" (nothing proven current), never "checking…" forever.
  const catalogFailed = !!catalog.error && !catalog.data;
  const real = useMemo(() => realPicks(specs, exps.data?.experiments, records.details,
    catalog.data ?? (catalogFailed ? { models: [] } : null), undefined, records.failed), [specs, exps.data, records, catalog.data, catalogFailed]);
  if (real.ids.join(",") !== ids.join(",")) setIds(real.ids); // state derived from the previous render
  const pickOf = (spec: string): RealPick | undefined => (!exps.data && exps.loading ? { state: "loading" }
    : spec === prodSpec ? preferCurrent(real.picks.get(spec), real.picks.get("production")) : real.picks.get(spec));

  /* jobs */
  const score = () => void scoreUniversal(scoreNames, nFields);
  const rescore = () => void scoreUniversal([], nFields, true);
  const pull = () => void pullUniversal(split.finished);
  const pullAgain = () => void pullUniversal(split.again, { again: true });
  const measure = () => void measureGateWeights(pending.map((g) => g.name), scoreNames, nFields);
  const pullRow = (r: UniversalFasrcRow) => {
    const kind = fasrcKind(r);
    void pullUniversal([r.name], kind === "again" ? { again: true } : kind === "try" ? { state: r.state ?? "unfinished" } : {});
  };
  /** Only a registered universal model with a checkpoint here can be production. */
  const canProduce = (r: ShapedRow) => r.class === "universal" && !!r.fingerprint;
  const makeProduction = (r: ShapedRow) => void setProduction({ spec: r.spec, label: universalLabel(r.name) }, prodLabel ?? "the production gate");
  const compareAll = compareHref(localSpecs, prodSpec);
  usePageActions([
    { id: "uni-score", label: "Universal: score the models without a current score…", group: "Models", keywords: ["universal", "knee", "psnr"],
      disabled: !scoreNames.length, run: score },
    { id: "uni-rescore", label: "Universal: re-score every universal model…", group: "Models", keywords: ["universal", "force"], disabled: !rows.some((r) => r.fingerprint), run: rescore },
    { id: "uni-pull", label: "Universal: pull the finished models from FASRC…", group: "Models", keywords: ["universal", "download", "rsync"],
      disabled: offline || !split.finished.length, run: pull },
    { id: "uni-pull-again", label: "Universal: pull again the registered models with no checkpoint here…", group: "Models",
      keywords: ["universal", "download", "rsync", "retry"], disabled: offline || !split.again.length, run: pullAgain },
    { id: "uni-weights", label: "Universal: measure the gate weights…", group: "Models", keywords: ["universal", "gate", "share", "hours"],
      disabled: !pending.length, run: measure },
    { id: "uni-compare", label: "Universal: compare every universal model on real fields", group: "Models", keywords: ["sky", "real", "holes"],
      disabled: !localSpecs.length, run: () => navigate(compareAll) },
    { id: "uni-x-time", label: "Universal: plot by training time", group: "Models", disabled: x === "time", run: () => setX("time") },
    { id: "uni-x-steps", label: "Universal: plot by training length (steps)", group: "Models", disabled: x === "steps", run: () => setX("steps") },
  ]);

  /* the plot */
  const plot = useMemo(() => {
    const series: Series[] = [];
    const drawn: Drawn[] = [];
    const items: LegendItem[] = [];
    const seenDepth = new Set<string>();
    const depthItem = (d: number | null, line: boolean) => {
      if (seenDepth.has(depthKey(d))) return;
      seenDepth.add(depthKey(d));
      items.push({ label: depthLabel(d), key: depthKey(d), color: colorOf(d), marker: "filled", line });
    };
    const point = (p: ModelPoint) => {
      series.push({ x: [p.x], y: [p.y], mode: "scatter", marker: p.mig ? "ring" : "filled", color: colorOf(p.depth), width: 2.6,
        alpha: p.stale ? 0.35 : undefined, name: p.label, key: p.key });
      drawn.push({ ...p, inspect: { kind: "member", id: p.name } });
    };
    let xs: number[] = [];
    let missing: { unscored: string[]; unplaced: string[] };
    if (x === "time") {
      const t = timePoints(rows);
      missing = { unscored: t.unscored, unplaced: t.untimed };
      for (const d of order) if (t.points.some((p) => p.depth === d)) depthItem(d, false);
      t.points.forEach(point);
      xs = t.points.map((p) => p.x);
      for (const c of combs) {
        const key = c.production ? "comb:production" : "comb:variants";
        // partial: a thin diamond at a lower bound (a member it reads has no hours)
        series.push({ x: [c.hours], y: [c.value], mode: "scatter", marker: "diamond", color: c.production ? C.comb : C.muted,
          width: c.partial ? 1.2 : 3, alpha: c.stale ? 0.45 : undefined,
          name: `${c.name} · ${c.partial ? "≥ " : ""}${hoursText(c.hours)} ${weightText(weighting)}${c.stale ? " (stale score)" : ""}`, key });
        drawn.push({ name: c.combiner.name, num: c.name, key, depth: null, x: c.hours, y: c.value, mig: false, stale: c.stale, label: c.name,
          inspect: { kind: "combiner", id: c.combiner.name } });
        xs.push(c.hours);
      }
      if (combs.some((c) => c.production)) items.push({ label: gateWord, key: "comb:production", color: C.comb, marker: "diamond" });
      if (combs.some((c) => !c.production)) items.push({ label: "other gates", key: "comb:variants", color: C.muted, marker: "diamond" });
    } else {
      const lines = lengthLines(rows);
      missing = { unscored: rows.filter((r) => r.value == null).map((r) => r.num),
        unplaced: rows.filter((r) => r.value != null && r.stepsDone == null).map((r) => r.num) };
      for (const l of lines) {
        depthItem(l.depth, true);
        if (l.points.length > 1) {
          series.push({ x: l.points.map((p) => p.x), y: l.points.map((p) => p.y), color: colorOf(l.depth), width: 1.6, alpha: 0.8,
            name: depthLabel(l.depth), key: l.key });
        }
        l.points.forEach(point);
        xs = xs.concat(l.points.map((p) => p.x));
      }
    }
    const guides: Guide[] = [];
    // production's line: the gate's (the combiner colour), or the universal model's (its depth colour, as its dot)
    if (prodValue != null) {
      guides.push({ axis: "y", v: prodValue, color: prodLabel ? colorOf(prodRow?.depth ?? null) : C.comb, dash: [6, 3], width: 1.4,
        label: prodLabel ? `production · ${prodLabel}` : "production gate" });
    }
    if (meanValue != null) guides.push({ axis: "y", v: meanValue, color: C.mean, dash: [2, 3], width: 1.2, label: "plain mean" });
    const xDomain = logDomain(xs, x === "time" ? [0.1, 100] : [1e4, 1e7]);
    const yDomain = paddedDomain([...series.flatMap((s) => s.y), ...guides.map((g) => g.v)], { pad: 0.08, minSpan: 0.2 });
    return { series, drawn, items, guides, xDomain, yDomain, missing };
  }, [rows, order, combs, x, weighting, prodValue, meanValue, prodLabel, gateWord]); // eslint-disable-line react-hooks/exhaustive-deps

  const onPlotClick = (at: { x: number; y: number }) => {
    const hit = nearestPoint(plot.drawn, at, plot.xDomain, plot.yDomain, true);
    if (hit) openInspector(hit.inspect);
  };

  /* tables */
  const columns = useMemo<DataColumn<ShapedRow>[]>(() => [
    { id: "model", header: "Model", accessor: (r) => Number(r.num), width: prodName ? 156 : 86,
      cell: (r) => (
        <span className="mdl-member">
          <span className="mdl-swatch" data-ring={r.mig || undefined} style={{ ["--sw" as string]: colorOf(r.depth) }} />#{r.num}
          {r.name === prodName && <Badge tone="accent" size="sm">production</Badge>}
        </span>
      ) },
    { id: "class", header: "Class", accessor: (r) => classText(r.class), width: 128, priority: 5 },
    { id: "depth", header: "Depth", numeric: true, accessor: (r) => r.depth, width: 70 },
    { id: "steps", header: "Steps", numeric: true, accessor: (r) => r.stepsDone, width: 80,
      cell: (r) => (
        <Tooltip content={[r.served_step != null ? `served checkpoint at step ${r.served_step.toLocaleString()}` : null,
          r.target_steps != null ? `target ${r.target_steps.toLocaleString()}` : null].filter(Boolean).join(" · ") || "steps trained"}>
          <span tabIndex={0}>{stepsShort(r.stepsDone)}</span>
        </Tooltip>
      ) },
    { id: "batch", header: "Batch", numeric: true, accessor: (r) => r.batch_size ?? null, width: 70, priority: 4 },
    { id: "crops", header: "Crops/field", numeric: true, accessor: (r) => r.crops_per_field ?? null, width: 92, priority: 4 },
    { id: "knee", header: "Knee", accessor: (r) => kneeWords(r), width: 100, priority: 3,
      cell: (r) => {
        const learned = learnedKneesText(r, bands);
        return learned ? <Tooltip content={`Learned output knee: ${learned}`}><span tabIndex={0}>{kneeWords(r)}</span></Tooltip> : kneeWords(r);
      } },
    { id: "impl", header: "Step", headerText: "how the training step ran (XLA / bf16)", accessor: (r) => stepImplText(r), width: 96, priority: 4 },
    { id: "hardware", header: "Hardware", accessor: (r) => hardwareText(r.hardware), width: 130, priority: 3,
      cell: (r) => (
        <Tooltip content={`${r.hardware.partition ? `partition ${r.hardware.partition}` : "partition unknown"}${r.hardware.jobs.length ? ` · jobs ${r.hardware.jobs.join(", ")}` : ""}`}>
          <span tabIndex={0}>{hardwareText(r.hardware)}</span>
        </Tooltip>
      ) },
    { id: "hours", header: "Train h", headerText: "training time in hours", numeric: true, width: 80,
      accessor: (r) => (r.hours == null ? null : Number(r.hours.toFixed(3))), cell: (r) => (r.hours == null ? "—" : hoursNum(r.hours)) },
    { id: "value", header: metric === "mean" ? "∫PSNR" : `∫${BAND_SHORT[metric] ?? metric}`,
      headerText: metric === "mean" ? "∫PSNR, mean of the bands [dB]" : metricInfo.axis, numeric: true, width: 84,
      accessor: (r) => (r.value == null ? null : Number(r.value.toFixed(4))),
      cell: (r) => <span className={r.status === "stale" ? "mdl-faint" : undefined}><b>{db(r.value)}</b></span> },
    ...bands.map((b, i): DataColumn<ShapedRow> => ({
      id: `b_${b}`, header: `∫${BAND_SHORT[b] ?? b}`, numeric: true, hidden: true, width: 76,
      accessor: (r) => (r.bandValues[i] == null ? null : Number((r.bandValues[i] as number).toFixed(4))), cell: (r) => db(r.bandValues[i]),
    })),
    { id: "vsProduction", header: "vs production",
      headerText: `∫PSNR minus ${prodLabel ? `production's (${prodLabel})` : "the production gate's"} (dB)`, numeric: true, width: 104,
      accessor: (r) => r.vsProduction,
      cell: (r) => (r.stale && r.vsProduction != null ? (
        <Tooltip content={STALE_DELTA}><span tabIndex={0} className="mdl-faint">{dbDelta(r.vsProduction)}*</span></Tooltip>
      ) : <span className={r.vsProduction != null && r.vsProduction > 0 ? "mdl-good" : undefined}>{dbDelta(r.vsProduction)}</span>) },
    { id: "vsMatch", header: "vs time-matched combiner", headerText: `∫PSNR minus the scored combiner nearest in training hours (${weightText(weighting)})`,
      numeric: true, width: 200, accessor: (r) => r.vsMatch,
      csv: (r) => (!r.match ? "" : r.vsMatch != null ? `${dbDelta(r.vsMatch)}${r.stale ? " (stale)" : ""} vs ${r.match.name} (${hoursText(r.match.hours)})`
        : r.matchBlock ? `— (${r.match.name}: ${r.matchBlock === "stale" ? "stale score" : "other fields"})` : ""),
      cell: (r) => (r.match && r.vsMatch != null ? (
        <Tooltip content={`${r.match.name}: ${db(r.match.value)} dB at ${hoursText(r.match.hours)} ${weightText(weighting)}; #${r.num}: ${db(r.value)} dB at ${hoursText(r.scoreHours)}${r.stale ? ` — ${STALE_DELTA}` : ""}`}>
          <span tabIndex={0}><span className={r.stale ? "mdl-faint" : r.vsMatch > 0 ? "mdl-good" : undefined}>{dbDelta(r.vsMatch)}{r.stale ? "*" : ""}</span>
            <span className="mdl-faint"> vs {r.match.name}</span></span>
        </Tooltip>
      ) : r.match && r.matchBlock ? (
        <Tooltip content={r.matchBlock === "stale"
          ? `${r.match.name}'s ∫PSNR is stale (its compare report or knee curves predate its fit or its members' checkpoints): no Δ until it is scored again`
          : `${r.match.name}'s ∫PSNR was scored on other test fields, records or target than the universal models: no Δ`}>
          <span tabIndex={0} className="mdl-faint">{r.matchBlock === "stale" ? "stale gate score" : "other fields"} · {r.match.name}</span>
        </Tooltip>
      ) : <span className="mdl-faint">—</span>) },
    { id: "holes", header: "Real holes [%]", headerText: "real-data holes in the worst band (%), newest Sky › Compare run of this model", numeric: true, width: 110,
      accessor: (r) => { const p = pickOf(r.spec); return p?.state === "current" ? p.bench.worst?.pct ?? p.bench.holeMean : null; },
      cell: (r) => <RealHoles pick={pickOf(r.spec)} /> },
    { id: "medianR", header: "R̃", headerText: "median SR/LR peak flux ratio on real fields", numeric: true, width: 64,
      accessor: (r) => { const p = pickOf(r.spec); return p?.state === "current" ? p.bench.medianR : null; },
      cell: (r) => { const p = pickOf(r.spec); const v = p?.state === "current" ? p.bench.medianR : null; return v == null ? "—" : v.toFixed(2); } },
    { id: "status", header: "Status", accessor: (r) => STATUS_TEXT[r.status], width: 104, cell: statusCell },
    { id: "actions", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 48,
      cell: (r) => (
        <Menu label={`#${r.num} actions`} trigger={<IconButton size="sm" icon="more" label={`#${r.num} actions`} />} items={[
          { label: "Inspect", onSelect: () => openMember(r.name) },
          { label: "Compare on real fields", disabled: !r.fingerprint, onSelect: () => navigate(compareHref([r.spec], prodSpec)) },
          ...(r.fingerprint ? [{ label: r.status === "scored" ? "Re-score…" : "Score…",
            onSelect: () => void scoreUniversal([r.name], nFields, r.status === "scored") }] : []),
          // a registered universal model (not an ensemble member) with a checkpoint here
          ...(r.class === "universal" && r.name !== prodName ? [{ type: "separator" as const },
            { label: "Make production…", disabled: !canProduce(r), onSelect: () => makeProduction(r) }] : []),
        ]} />
      ) },
  ], [bands, metric, metricInfo, weighting, real, exps.data, exps.loading, order, nFields, prodName, prodLabel]); // eslint-disable-line react-hooks/exhaustive-deps

  const combColumns = useMemo<DataColumn<UniversalCombiner>[]>(() => {
    const hoursCol = (id: "sum" | "max" | "mean", header: string, headerText: string): DataColumn<UniversalCombiner> => ({
      id: `h_${id}`, header, headerText, numeric: true, width: id === "sum" ? 90 : 132,
      accessor: (c) => { const v = combinerHours(c, id); return v == null ? null : Number(v.toFixed(3)); },
      cell: (c) => {
        const text = combinerHoursText(c, id, false);
        return isPartial(c)
          ? (
            <Tooltip content={`A lower bound: no training time for ${numbersText(c.missing_hours)}. Never time-matched.`}>
              <span tabIndex={0} className="mdl-warn">{text}</span>
            </Tooltip>
          )
          : <span className={id === weighting ? undefined : "mdl-muted"}>{text}</span>;
      },
    });
    return [
      { id: "gate", header: "Gate", accessor: (c) => combinerName(c, prodName), width: prodName ? 190 : 150,
        cell: (c) => (c.production ? <strong>{combinerName(c, prodName)}</strong> : combinerName(c, prodName)) },
      { id: "reads", header: "Members read", numeric: true, accessor: (c) => c.n_reads, width: 110 },
      hoursCol("sum", "Σ hours", "training hours of every member it reads"),
      hoursCol("max", "Max-weighted h", "Σ hours × the member's peak gate share"),
      hoursCol("mean", "Mean-weighted h", "Σ hours × the member's mean gate share"),
      { id: "integrated", header: "∫PSNR", numeric: true, width: 84, accessor: (c) => (c.mean == null ? null : Number(c.mean.toFixed(4))),
        cell: (c) => (c.stale && c.mean != null ? (
          <Tooltip content={`A stale score (${c.score_source === "compare" ? "the compare report predates its fit or its members' checkpoints" : "the knee curves predate the current cubes"})`}>
            <span tabIndex={0} className="mdl-faint"><b>{db(c.mean)}</b>*</span>
          </Tooltip>
        ) : c.mean != null && !combinerComparable(c, comparable) ? (
          <Tooltip content="Scored on other test fields, records or target than the universal models: no Δ is taken against it">
            <span tabIndex={0} className="mdl-faint"><b>{db(c.mean)}</b>†</span>
          </Tooltip>
        ) : <b>{db(c.mean)}</b>) },
      { id: "weights", header: "Weights from", accessor: (c) => c.weights_source ?? c.weights_reason ?? null, width: 200,
        cell: (c) => (c.weights_source ? (c.weights_source === "production" ? `the ${gateWord}'s diagnostic` : "the variant's own diagnostic")
          : pendingIds.has(c.id) ? <span className="mdl-warn">not measured yet: Measure gate weights</span>
            : <span className="mdl-faint">{c.weights_reason ?? "—"}</span>) },
      { id: "fit", header: "Fit h", headerText: "the gate's own fit time (not counted above)", numeric: true, hidden: true, width: 70,
        accessor: (c) => c.fit_hours, cell: (c) => (c.fit_hours == null ? "—" : hoursNum(c.fit_hours)) },
    ];
  }, [weighting, pendingIds, comparable, prodName, gateWord]);

  const note = () => (data ? universalNote(shapeRows(data, { weighting, metric: "mean" }), data.combiners ?? [],
    { weighting, nFields, integration: data.integration, production: data.production_model ?? null }) : "");
  const integration = data?.integration;
  const kneeRange = `${kneeNum(integration?.from_e ?? 0.1)}–${kneeNum(integration?.to_e ?? 1e4)} e⁻`;
  const runMenu = (
    <Menu label="Run" trigger={<Button size="sm" iconRight="chevronDown" loading={scoreJob.busy || pullJob.busy}>Run</Button>} items={[
      { label: scoreNames.length ? `Score ${plural(scoreNames.length, "model")}…` : "Every model has a current score", disabled: !scoreNames.length, onSelect: score },
      { label: "Re-score every model…", disabled: !rows.some((r) => r.fingerprint), onSelect: rescore },
      { label: pending.length ? `Measure the weights of ${plural(pending.length, "gate")}…` : "Every gate's weights are measured",
        disabled: !pending.length, onSelect: measure },
      { type: "separator" },
      { label: split.finished.length ? `Pull ${plural(split.finished.length, "model")} from FASRC…` : "Nothing to pull from FASRC",
        disabled: offline || !split.finished.length, onSelect: pull },
      ...(split.again.length ? [{ label: `Pull ${plural(split.again.length, "model")} again…`, disabled: offline, onSelect: pullAgain }] : []),
    ]} />
  );
  const jobs = (scoreJob.job || pullJob.job || scoreJob.error || pullJob.error) && (
    <div className="mdl-jobs">
      <JobProgress job={scoreJob.job} error={scoreJob.error} />
      <JobProgress job={pullJob.job} error={pullJob.error} />
    </div>
  );
  const failedText = (data?.fasrc ?? []).filter((r) => fasrcKind(r) === "try")
    .map((r) => `${numbersText([r.name])} ${stateText(r)}`).join(", ");
  const fasrcText = [
    split.finished.length ? `${plural(split.finished.length, "universal model")} finished on FASRC and ${split.finished.length === 1 ? "is" : "are"} not here: ${numbersText(split.finished)}` : null,
    split.again.length ? `${plural(split.again.length, "model")} registered here with no checkpoint (the last pull failed): ${numbersText(split.again)}` : null,
    split.failed.length ? `${plural(split.failed.length, "model")} ended without a clean finish (${failedText}): try ${split.failed.length === 1 ? "it" : "each"} from the table below` : null,
    split.training.length ? `${plural(split.training.length, "model")} still training: ${numbersText(split.training)}` : null,
  ].filter(Boolean).join("; ");
  const offlineTip = `FASRC offline: ${fasrc?.last_error ?? "not connected"}`;
  const fasrcColumns: Column<UniversalFasrcRow>[] = [...FASRC_COLUMNS, {
    header: "", cell: (r) => {
      const kind = fasrcKind(r);
      if (kind === "training") return null;
      const label = kind === "again" ? "Pull again" : kind === "try" ? "Try pull" : "Pull";
      return (
        <Tooltip content={offline ? offlineTip : kind === "try" ? `Its job ended ${stateText(r)}: FASRC may hold no checkpoint (then nothing is registered)`
          : kind === "again" ? "Copies its checkpoint again (it is registered as a universal model)" : "Registers it as a universal model and copies its checkpoint"}>
          <span><Button size="sm" variant="ghost" disabled={offline || pullJob.busy} aria-label={`${label} #${memberNumber(r.name) ?? r.name}`}
            onClick={() => pullRow(r)}>{label}</Button></span>
        </Tooltip>
      );
    },
  }];
  const empty = data && !data.models.length && !data.fasrc.length;
  const { unscored, unplaced } = plot.missing;

  return (
    <Page className="mdl-page">
      <LoadState loading={res.loading} error={res.error} onRetry={res.reload}
        empty={empty && (
          <EmptyState icon="layers" title="No universal models yet"
            action={<Button variant="primary" asChild><Link to={tabPath("train")}>Train</Link></Button>}>
            A universal model is trained at several knees at once and outputs ONE image (a fixed output knee or one learned per band).
            Train one in Models › Train; when it finishes on FASRC, pull it here.
            {data?.notes.length ? <> {data.notes.map((n) => sentence(n)).join(" ")}</> : null}
          </EmptyState>
        )}>
        {data && (
          <div className="mdl-stack">
            <section className="mdl-result" aria-label="Universal models">
              {summary && <Summary s={summary} weighting={weighting} />}
              <Caption>
                ∫PSNR: PSNR averaged over log knee {kneeRange} on {nFields} test fields (mean of VIS · Y · J · H), the{" "}
                <Link to={tabPath("leaderboard")}>Leaderboard</Link>'s metric. Training time: the training log's windows summed
                (rolled-back ones included), in hours on the hardware named; a combiner's is the hours of the members it reads,
                each × its peak gate share (max weight), its mean share (mean weight), or not weighted (every member).
              </Caption>
            </section>
            {scoreNames.length > 0 && (
              <Callout tone="info" dense action={(
                <Button size="sm" variant="primary" loading={scoreJob.busy} onClick={score}>
                  Score {plural(scoreNames.length, "model")} on the {nFields} test fields
                </Button>
              )}>
                {`${plural(scoreNames.length, "universal model")} ${scoreNames.length === 1 ? "has" : "have"} no current score: ${numbersText(scoreNames)}.`}
              </Callout>
            )}
            {pending.length > 0 && (
              <Callout tone="info" dense action={(
                <Button size="sm" loading={scoreJob.busy} onClick={measure}>Measure gate weights</Button>
              )}>
                {`${plural(pending.length, "gate")} ${pending.length === 1 ? "has" : "have"} no measured member shares yet (${pending.map((g) => g.name).join(", ")}), so ${pending.length === 1 ? "its" : "their"} weighted training hours are unknown; the validate cubes can measure them now.`}
              </Callout>
            )}
            {fasrcText && (
              <Callout tone={split.again.length || split.failed.length ? "warn" : "info"} dense action={(split.finished.length > 0 || split.again.length > 0) && (
                <span className="mdl-row">
                  {split.finished.length > 0 && (
                    <Tooltip content={offline ? offlineTip : "Registers them as universal models (never ensemble members) and copies their checkpoints"}>
                      <span><Button size="sm" icon="download" disabled={offline || pullJob.busy} loading={pullJob.busy} onClick={pull}>
                        Pull {plural(split.finished.length, "model")} from FASRC
                      </Button></span>
                    </Tooltip>
                  )}
                  {split.again.length > 0 && (
                    <Tooltip content={offline ? offlineTip : "Copies their checkpoints again: they are registered, but the last pull left none here"}>
                      <span><Button size="sm" icon="download" disabled={offline || pullJob.busy} onClick={pullAgain}>
                        Pull {plural(split.again.length, "model")} again
                      </Button></span>
                    </Tooltip>
                  )}
                </span>
              )}>
                {fasrcText}.
              </Callout>
            )}
            {data.production?.stale && (prodLabel ? (
              <Callout tone="warn" dense>
                Production's ∫PSNR ({prodLabel}) is stale (an earlier checkpoint or other test fields): Δ vs production may be off
                until it is scored again.
              </Callout>
            ) : (
              <Callout tone="warn" dense action={<Button size="sm" asChild><Link to={tabPath("leaderboard")}>Leaderboard</Link></Button>}>
                The production gate's ∫PSNR is stale (its knee curves predate the current cubes): Δ vs production may be off until
                the Leaderboard's Evaluate and Knee PSNR run again.
              </Callout>
            ))}
            {data.notes.length > 0 && (
              <Caption>{data.notes.map((n, i) => <span key={i}>{sentence(n)} </span>)}</Caption>
            )}
            {jobs}
            <Toolbar label="Universal plot">
              <ToolbarGroup label="X axis">
                <Segmented<XAxis> size="sm" aria-label="X axis" value={x} onChange={setX} options={[...X_AXES]} />
              </ToolbarGroup>
              <ToolbarGroup label="Metric">
                <Segmented<Metric> size="sm" aria-label="Metric" value={metric} onChange={setMetric}
                  options={METRICS.map(({ value, label }) => ({ value, label }))} />
              </ToolbarGroup>
              <ToolbarGroup label="Combiner hours">
                <Segmented<Weighting> size="sm" aria-label="Combiner weighting" value={weighting} onChange={setWeighting}
                  options={WEIGHTINGS.map(({ value, label, text }) => ({ value, label, title: `Combiner training time ${text}` }))} />
              </ToolbarGroup>
              <ToolbarSpacer />
              {localSpecs.length
                ? (
                  <Tooltip content="Sky › Compare: a new comparison of production and every local universal model on real tiles">
                    <Button size="sm" icon="external" asChild><Link to={compareAll}>Compare all on real fields</Link></Button>
                  </Tooltip>
                )
                : <Button size="sm" icon="external" disabled>Compare all on real fields</Button>}
              <LogToNotebookButton from="Models › Universal" disabled={!rows.length} note={note}
                title="Open Notebook › Log with the models and combiners table; you edit it there first" />
              {runMenu}
            </Toolbar>
            <section className="mdl-stack mdl-stack--tight" aria-label={x === "time" ? "By training time" : "By training length"}>
              {plot.drawn.length ? (
                <>
                  <Legend items={plot.items} {...lg.legendProps} />
                  <Plot {...lg.plotProps} xScale="log" xDomain={plot.xDomain} xTicks={logTicks(plot.xDomain)} yDomain={plot.yDomain}
                    xLabel={x === "time" ? "training time [h]" : "training steps"} yLabel={metricInfo.axis}
                    series={plot.series} guides={plot.guides} aspect={0.45} onPlotClick={onPlotClick}
                    xFormat={(v) => (x === "time" ? hoursText(v) : stepsShort(v))} yFormat={(v) => v.toFixed(2)}
                    exportName={x === "time" ? "universal-by-time" : "universal-by-steps"}
                    aria-label={x === "time" ? `${metricInfo.label} by training time` : `${metricInfo.label} by training steps`} />
                </>
              ) : <EmptyState compact icon="activity" title="Nothing to draw yet">Score the models on the test fields to place them.</EmptyState>}
              {(plot.drawn.length > 0 || unscored.length > 0 || unplaced.length > 0) && (
                <Caption>
                  {plot.drawn.length > 0 && <>
                    {x === "time"
                      ? <>One dot per model, coloured by depth (a ring: trained on a MIG slice of gpu_test, a fraction of an A100;
                        faded: a stale score, at the hours it was scored){combs.length > 0 && <>; the diamonds are the combiners at
                        their {weightText(weighting)} training hours (thin: a lower bound, a member it reads has no hours)</>}.
                        Click a point to open it.</>
                      : <>One line per depth through its models by steps trained; the dots are the models (a ring: a MIG slice;
                        faded: a stale score, at the steps it was scored).</>}
                    {" "}The lines across are {prodLabel ? `production (${prodLabel})` : "the production gate"} and the plain mean.{" "}
                  </>}
                  {unscored.length > 0 && `Not drawn: ${unscored.map((n) => `#${n}`).join(", ")} (no score). `}
                  {unplaced.length > 0 && `Not drawn: ${unplaced.map((n) => `#${n}`).join(", ")} (${x === "time" ? "no training time" : "no step count"}).`}
                </Caption>
              )}
            </section>
            <section className="mdl-stack mdl-stack--tight" aria-labelledby="uni-models-title">
              <h2 id="uni-models-title" className="mdl-h2">Models</h2>
              <DataTable rows={rows} columns={columns} rowKey={(r) => r.name} aria-label="Universal models" urlKey="u"
                defaultSort={[{ id: "value", desc: true }]} exportName="universal-models" height="auto"
                inspect={(r) => ({ kind: "member", id: r.name })} empty="No universal model here yet." />
              <Caption>
                vs time-matched combiner: against the scored combiner nearest in training hours ({weightText(weighting)}), only when
                its score is current and on the universal models' test fields; a gate with members of unknown hours is never matched.
                * a stale model score (an earlier checkpoint): its Δs are not current. Real holes
                (worst band) and R̃ come from the newest Sky › Compare run that scored the current checkpoint;{" "}
                “earlier fit” when it scored an older one. “Compare on real fields” opens Sky › Compare with production and the model.
              </Caption>
            </section>
            {data.combiners.length > 0 && (
              <section className="mdl-stack mdl-stack--tight" aria-labelledby="uni-comb-title">
                <h2 id="uni-comb-title" className="mdl-h2">Combiners</h2>
                <DataTable rows={data.combiners} columns={combColumns} rowKey={(c) => c.id} aria-label="Combiners by training time"
                  urlKey="uc" defaultSort={[{ id: "integrated", desc: true }]} exportName="universal-combiners" height="auto"
                  inspect={(c) => ({ kind: "combiner", id: c.name })} />
                <Caption>
                  A gate's training time is that of the members it reads; max weight (the default) weights each by its largest
                  share of the gate over the bands and brightness bins, so a member the gate leans on in a few cores counts.
                  ∫PSNR: {prodLabel ? `the ${gateWord}'s` : "production's"} from the Leaderboard, a variant's from its newest compare report (* a stale score; † scored
                  on other test fields than the universal models). ≥: a lower bound, a member it reads has no training time.
                  The gate's own fit time is the hidden Fit h column.{prodLabel ? ` Production is ${prodLabel}: no gate runs in production.` : ""}
                </Caption>
              </section>
            )}
            {data.fasrc.length > 0 && (
              <section className="mdl-stack mdl-stack--tight" aria-labelledby="uni-fasrc-title">
                <h2 id="uni-fasrc-title" className="mdl-h2">On FASRC, not here</h2>
                <Caption>
                  Pull: finished COMPLETED or TIMEOUT. Try pull: the job ended otherwise, so FASRC may hold no checkpoint (the pull
                  then refuses and registers nothing). Pull again: registered here, but the last pull left no checkpoint.
                </Caption>
                <Table className="mdl-table-mini" aria-label="Universal models on FASRC" columns={fasrcColumns} rows={data.fasrc}
                  rowKey={(r) => r.name} />
              </section>
            )}
          </div>
        )}
      </LoadState>
    </Page>
  );
}
