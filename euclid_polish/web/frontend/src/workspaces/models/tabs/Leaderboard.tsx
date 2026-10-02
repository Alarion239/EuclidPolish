/* Models › Leaderboard (`/models/leaderboard`; absorbs the old
   Ensemble Overview and Knee PSNR tabs). From the top:
   - one status line: "All current", or each failing staleness check once with
     its confirmed fix (Evaluate N fields, Member PSNR, Knee PSNR, Combiner);
   - the verdict (the production gate against the best member and the plain
     mean on ONE metric, the knee-integrated PSNR) and its caption;
   - the comparison table gate / plain mean / best member × ∫PSNR, ∫VIS/Y/J/H
     and, from the newest Sky › Compare run of THIS production, the real holes
     (worst band) and R̃ — or the line saying there is no real benchmark for
     this membership (model.ts leaderboardBenchmark);
   - one alert line for the members that stopped short (TIMEOUT) · Continue;
   - the knee curves' toolbar (view, band, colour, members, the integration
     range, Log to notebook, the run menu), the PSNR-vs-knee curves per band;
   - the leaderboard table — the one ranking Members, Combiner, Images and
     Home link to — with Test VIS @100 e⁻ and Test 4b as hidden columns. A row
     opens the member or combiner inspector.
   Opening the page reads caches only; every job asks first. */
import { useMemo, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import Plot, { Legend, useLegend, type Band as PlotBand, type LegendItem, type Series } from "../../../charts/Plot";
import { C, LOSS_COLOR } from "../../../colors";
import { useJob } from "../../../api/jobs";
import { pagePath } from "../../../app/nav";
import { usePageActions } from "../../../app/palette";
import { formatDate, formatRelative } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { logTicks } from "../../../ticks";
import {
  Badge, Button, Callout, Caption, Checkbox, Chip, DataTable, Dialog, EmptyState, IconButton, JobProgress, Menu, Num, NumberField, Page, RangeSlider, Segmented,
  SummaryLine, Table, Toolbar, ToolbarGroup, ToolbarSpacer, Tooltip, type Column, type DataColumn,
} from "../../../ui";
import {
  BAND_SHORT, BANDS, useExperiment, useExperiments, useKnee, useMembers, useModelCatalog, useOverview,
  type KneeModel, type MemberRow, type Overview,
} from "../api";
import { kneeColor, kneeOrderOf, LoadState } from "../common";
import { JOB, computeKnee, evaluate, refreshMemberPsnr, useOnJobEnd } from "../jobs";
import {
  compareRunPath, db, dbDelta, kneeLeaderboard, kneeModelName, kneeNum, kneeText, leaderboardBenchmark, memberNumber, overviewComparison,
  productionRun, relativeTo, statusChecks, type Comparison, type ComparisonRow, type LeaderBenchmark, type LeaderRow, type StatusCheck,
} from "../model";
import { evaluationNote, kneeNote } from "../notes";
import { FreezeStudyDialog } from "../../shared/FreezeStudyDialog";
import { LogToNotebookButton } from "../../shared/LogToNotebook";
import "../models.css";

const tabPath = (tab: string) => pagePath("models", { tab });

/* ── prose helpers ─────────────────────────────────────────────────────── */

const SUP: Record<string, string> = { 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
/** A knee in e⁻ as prose reads it: 0.1, 100, 10⁴. */
function kneeE(v: number): string {
  const p = Math.log10(v);
  return v >= 1000 && Number.isInteger(p) ? `10${String(p).split("").map((d) => SUP[d]).join("")}` : String(v);
}
/** Non-breaking spaces: "2 d ago" and "10⁴ e⁻" never wrap inside. */
const nb = (t: string) => t.replace(/ /g, " ");

/** "+1.02 dB over …"; a loss reads "0.30 dB under …" in warn. */
function gain(v: number | null, what: string): ReactNode {
  if (v == null || !Number.isFinite(v)) return null;
  return v >= 0 ? <><Num>{dbDelta(v)}</Num>{" "}dB over {what}</> : <><Num tone="warn">{db(-v)}</Num>{" "}dB under {what}</>;
}

/* ── status line ───────────────────────────────────────────────────────── */

/** One quiet line when every check passes; else each failing check once,
 *  with its confirmed fix. */
function StatusLine({ checks, o, nFields, onEvaluate }: {
  checks: StatusCheck[]; o: Overview; nFields: number; onEvaluate: () => void;
}) {
  if (!checks.length) {
    return <p className="mdl-status" data-tone="good"><span className="mdl-dot" aria-hidden />All current</p>;
  }
  const fix = (c: StatusCheck) => {
    if (c.fix === "evaluate") {
      return (
        <Tooltip content={o.test_present ? "Evaluate the members, the mean and the combiners on the test records"
          : "No local test records: sync them in Synthetic › Records"}>
          <span><Button size="sm" onClick={onEvaluate} disabled={!o.test_present}>Evaluate {nFields} fields…</Button></span>
        </Tooltip>
      );
    }
    if (c.fix === "knee") return <Button size="sm" onClick={() => void computeKnee()}>Knee PSNR</Button>;
    if (c.fix === "member-psnr") return <Button size="sm" onClick={() => void refreshMemberPsnr()}>Member PSNR</Button>;
    if (c.fix === "combiners") return <Button size="sm" asChild><Link to={tabPath("combiner")}>Combiner</Link></Button>;
    return <span />;
  };
  return (
    <ul className="mdl-checks" aria-label="Staleness">
      {checks.map((c) => (
        <li key={c.id} className="mdl-check" data-tone={c.tone}>
          <span className="mdl-dot" aria-hidden />
          <span><span className="mdl-check__title">{c.title}</span>{" "}<span className="mdl-check__detail">{c.detail}</span></span>
          {fix(c)}
        </li>
      ))}
    </ul>
  );
}

/* ── evaluate dialog ───────────────────────────────────────────────────── */

const MAX_FIELDS = 2000;

/** Evaluate the ensemble: how many test fields (default: the last run's
 *  count, so the scores stay comparable) and whether to re-run every member
 *  even when an identical evaluation is cached. The dialog is the confirm. */
function EvaluateDialog({ open, onOpenChange, defaultN, lastN, defaultForce, onStart }: {
  open: boolean; onOpenChange: (v: boolean) => void; defaultN: number; lastN: number | null; defaultForce: boolean;
  onStart: (n: number, force: boolean) => void;
}) {
  const [n, setN] = useState(String(defaultN));
  const [force, setForce] = useState(defaultForce);
  const count = Number(n);
  const bad = !Number.isInteger(count) || count < 1 || count > MAX_FIELDS ? `a whole number from 1 to ${MAX_FIELDS}` : null;
  return (
    <Dialog open={open} onOpenChange={onOpenChange} title="Evaluate the ensemble"
      description="Runs every active member, the plain mean and the combiners on the local test records (a local TensorFlow job, several minutes)."
      footer={<>
        <Button variant="ghost" onClick={() => onOpenChange(false)}>Cancel</Button>
        <Button variant="primary" disabled={!!bad} onClick={() => { onStart(count, force); onOpenChange(false); }}>
          Evaluate {bad ? "" : `${count} field${count === 1 ? "" : "s"}`}
        </Button>
      </>}>
      <div className="mdl-stack mdl-stack--tight">
        <NumberField label="Test fields" value={n} onChange={setN} min={1} max={MAX_FIELDS} step={1}
          hint={bad ?? (lastN == null ? "No evaluation yet." : count === lastN ? "The count of the last evaluation, so the scores stay comparable."
            : `The last evaluation scored ${lastN}; a different count gives scores that are not comparable with it.`)} />
        <Checkbox checked={force} onChange={setForce}>Re-run every member, even when an identical evaluation is cached</Checkbox>
      </div>
    </Dialog>
  );
}

/* ── verdict and comparison table ──────────────────────────────────────── */

/** The production gate (else the plain mean) against the best member and the
 *  plain mean, on the comparison table's own numbers. */
function Verdict({ cmp }: { cmp: Comparison }) {
  const row = (id: ComparisonRow["id"]) => cmp.rows.find((r) => r.id === id)?.integrated ?? null;
  const gate = row("gate"), mean = row("mean"), best = row("best");
  const head = gate ?? mean;
  if (head == null) return null;
  const bestWhat = cmp.bestNumber ? `the best member (#${cmp.bestNumber})` : "the best member";
  const clauses = [
    best != null ? gain(head - best, bestWhat) : null,
    gate != null && mean != null ? gain(gate - mean, "the plain mean") : null,
  ].filter((c) => c != null);
  return (
    <SummaryLine>
      {gate != null ? "Production gate" : "Plain mean"} <Num>{db(head)}</Num>{" "}dB integrated PSNR
      {clauses.length > 0 && ", "}
      {clauses.map((c, i) => <span key={i}>{i > 0 && " and "}{c}</span>)}
    </SummaryLine>
  );
}

const SPEC: Record<ComparisonRow["id"], (best: string | null) => string | null> = {
  gate: () => "production", mean: () => "mean", best: (n) => (n ? `member:member_${n}` : null),
};

function ComparisonTable({ cmp, bands, bench }: { cmp: Comparison; bands: readonly string[]; bench: LeaderBenchmark }) {
  const real = bench.state === "current" ? bench : null;
  const facts = (r: ComparisonRow) => {
    const spec = SPEC[r.id](cmp.bestNumber);
    return real && spec ? real.facts(spec) : null;
  };
  const columns: Column<ComparisonRow>[] = [
    { header: "", cell: (r) => (r.id === "gate" ? <strong>{r.label}</strong> : r.label) },
    { header: "∫PSNR [dB]", align: "right", cell: (r) => <span className="mdl-tnum">{db(r.integrated)}</span> },
    ...bands.map((b, i): Column<ComparisonRow> => ({
      header: `∫${BAND_SHORT[b] ?? b} [dB]`, align: "right", cell: (r) => <span className="mdl-tnum">{db(r.bands[i])}</span>,
    })),
    ...(real ? [
      { header: "Real holes, worst band [%]", align: "right" as const, cell: (r: ComparisonRow) => {
        const w = facts(r)?.worst;
        return <span className="mdl-tnum">{w?.pct != null ? `${w.short} ${w.pct.toFixed(0)}` : "—"}</span>;
      } },
      { header: "R̃", align: "right" as const, cell: (r: ComparisonRow) => {
        const v = facts(r)?.medianR;
        return <span className="mdl-tnum">{v != null ? v.toFixed(2) : "—"}</span>;
      } },
    ] : []),
  ];
  return <Table className="mdl-compare" aria-label="Production vs references" columns={columns} rows={cmp.rows} rowKey={(r) => r.id} />;
}

/** Where the real columns come from, or why there are none. */
function RealSource({ bench }: { bench: LeaderBenchmark }) {
  if (bench.state === "loading") return <Caption>Reading the real benchmark…</Caption>;
  if (bench.state === "none") {
    return (
      <Caption>
        Real data: {bench.reason}{bench.last?.created ? ` (${formatDate(bench.last.created)})` : ""}.{" "}
        <Link to={bench.last ? compareRunPath(bench.last.expId) : "/sky/compare"}>Sky › Compare</Link>
      </Caption>
    );
  }
  return (
    <Caption>
      Real columns from the Sky › Compare run{" "}
      <Link to={compareRunPath(bench.expId)}>{bench.label ? `“${bench.label}”` : bench.expId}</Link>
      {" "}({bench.tileSet}{bench.created ? `, ${formatDate(bench.created)}` : ""}): holes = % of bright LR pixels the SR
      blanks, in its worst band; R̃ = median SR/LR peak flux ratio. A row the run did not score for the current model is blank.
    </Caption>
  );
}

/* ── knee curves ───────────────────────────────────────────────────────── */

type View = "relative" | "absolute";
type Colour = "knee" | "loss" | "multi";
const LOSS_ORDER = ["l1", "l2", "l3", "mse", "berhu"];
const parseRange = (raw: string): [number, number] | undefined => {
  const [a, b] = raw.split(",").map(Number);
  return a > 0 && b > 0 ? [Math.min(a, b), Math.max(a, b)] : undefined;
};

function modelColor(m: KneeModel, by: Colour, kneeOrder: number[]): string {
  if (m.kind === "mean") return C.mean;
  if (m.kind === "combiner") return C.comb;
  if (by === "loss") return LOSS_COLOR[(m.loss ?? "l1").toLowerCase()];
  return kneeColor(m, kneeOrder, by);
}

/** The legend entry a member line belongs to under the active colouring:
 *  one toggle per knee / loss / multi-knee kind (each line keeps its own
 *  name for hover identification). `rank` orders the entries. */
type Facet = LegendItem & { key: string; rank: number };
function memberFacet(m: KneeModel, by: Colour, kneeOrder: number[]): Facet {
  const k = kneeText(m);
  const color = modelColor(m, by, kneeOrder);
  if (by === "loss") {
    const l = (m.loss ?? "l1").toLowerCase();
    return { key: `members:loss:${l}`, label: l.toUpperCase(), color, rank: LOSS_ORDER.indexOf(l) + 1 || 99 };
  }
  if (k.kind === "multi") {
    return m.output_knee != null
      ? { key: "members:multi:image", label: "multi-knee → 1 image", color, rank: 1e7 }
      : { key: "members:multi:heads", label: "multi-knee, heads", color, dash: true, rank: 1e7 + 1 };
  }
  if (by === "multi") return { key: "members:single", label: "single knee", color, rank: 0 };
  return { key: `members:knee:${k.sort}`, label: k.text, color, rank: k.sort };
}

type BoardRow = LeaderRow & { testVis: number | null; test4b: number | null };

/* ── the tab ───────────────────────────────────────────────────────────── */

export default function Leaderboard() {
  const ov = useOverview();
  const members = useMembers();
  const kneeRes = useKnee();
  const exps = useExperiments();
  const catalog = useModelCatalog();
  const run = productionRun(exps.data?.experiments);
  const detail = useExperiment(run?.id);
  const evalJob = useJob(JOB.evaluate);
  const kneeJob = useJob(JOB.knee);
  const psnrJob = useJob(JOB.memberPsnr);
  useOnJobEnd(evalJob.job);
  useOnJobEnd(kneeJob.job);
  useOnJobEnd(psnrJob.job);
  const [view, setView] = useUrlState<View>("view", "relative");
  const [band, setBand] = useUrlState("band", "all");
  const [colorBy, setColorBy] = useUrlState<Colour>("color", "knee");
  const [rangeRaw, setRangeRaw] = useUrlState("range", "");
  const [showMembers, setShowMembers] = useUrlState("members", true);
  const lg = useLegend();

  const o = ov.data;
  const kneeData = kneeRes.data;
  const knees = useMemo(() => kneeData?.knees ?? [], [kneeData]);
  const bands = useMemo(() => (kneeData?.available && kneeData.bands?.length ? kneeData.bands : [...BANDS]), [kneeData]);
  const models = useMemo(() => (kneeData?.available ? kneeData.models ?? [] : []), [kneeData]);
  const full: [number, number] = knees.length ? [knees[0], knees[knees.length - 1]] : [0.1, 1e4];
  const range = parseRange(rangeRaw) ?? full;
  const isFull = range[0] <= full[0] && range[1] >= full[1];
  const nFields = o?.headline.n_scored ?? 100;
  // The old Overview kept its chosen field count in ?n=: it seeds the dialog.
  const [legacyN] = useUrlState("n", 0);
  const askN = legacyN > 0 ? Math.min(MAX_FIELDS, Math.round(legacyN)) : nFields;
  const [evalAsk, setEvalAsk] = useState<null | { force: boolean }>(null);
  const [freezeOpen, setFreezeOpen] = useState(false);
  const memberRows = members.data?.members;
  const checks = useMemo(() => (o ? statusChecks(o.checks, memberRows) : []), [o, memberRows]);
  const kneeOffered = checks.some((c) => c.fix === "knee" || c.action === "knee");
  const cmp = useMemo(() => overviewComparison(kneeData, o?.headline.knee), [kneeData, o]);
  const bench = leaderboardBenchmark(exps.data?.experiments, detail.data, catalog.data);
  const timeouts = (memberRows ?? []).filter((m) => m.timeout);

  const run_ = {
    evaluate: () => setEvalAsk({ force: false }),
    force: () => setEvalAsk({ force: true }),
    knee: () => void computeKnee(),
    psnr: () => void refreshMemberPsnr(),
  };
  usePageActions([
    { id: "mdl-evaluate", label: "Evaluate the ensemble", group: "Models", keywords: ["test", "psnr"], shortcut: "Shift+E", run: run_.evaluate },
    { id: "mdl-evaluate-force", label: "Evaluate the ensemble (force re-inference)", group: "Models", run: run_.force },
    { id: "mdl-knee", label: "Compute PSNR vs knee", group: "Models", keywords: ["integrated"], run: run_.knee },
    { id: "mdl-member-psnr", label: "Refresh member PSNR", group: "Models", run: run_.psnr },
    { id: "knee-full", label: "Leaderboard: integrate over the full knee range", group: "Models", disabled: isFull, run: () => setRangeRaw("") },
    { id: "knee-relative", label: "Leaderboard: curves relative to the plain mean", group: "Models", run: () => setView("relative") },
    { id: "study-freeze", label: "Freeze a study of the ensemble…", group: "Models", keywords: ["study", "paper", "figures", "snapshot"], run: () => setFreezeOpen(true) },
  ]);

  /* curves */
  const kneeOrder = useMemo(() => kneeOrderOf(models.filter((m) => m.kind === "member")), [models]);
  const mean = models.find((m) => m.kind === "mean");
  const bandIdx = band === "all" ? bands.map((_, i) => i) : [Math.max(0, bands.indexOf(band))];
  const panels = useMemo(() => bandIdx.map((c) => {
    const series: Series[] = [];
    let lo = Infinity, hi = -Infinity;
    for (const m of models) {
      if (m.kind === "member" && !showMembers) continue;
      const curve = view === "relative" ? relativeTo(m.psnr, mean?.psnr) : m.psnr;
      const y = curve.map((row) => row[c]);
      for (const v of y) if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      const main = m.kind !== "member";
      const k = kneeText(m);
      series.push({
        x: knees, y, color: modelColor(m, colorBy, kneeOrder), width: main ? 2.6 : 1.2, alpha: main ? 1 : 0.8,
        dots: main, dash: k.kind === "multi" && m.output_knee == null ? [5, 3] : undefined,
        name: `${kneeModelName(m)}${m.kind === "member" ? ` · ${k.text}` : ""}`,
        key: m.kind === "member" ? memberFacet(m, colorBy, kneeOrder).key : kneeModelName(m),
      });
    }
    if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
    const pad = (hi - lo) * 0.06 || 0.5;
    const shade: PlotBand[] = isFull ? [] : [{ axis: "x", from: range[0], to: range[1], color: C.guide, alpha: 0.18, label: "integration range" }];
    return { c, name: BAND_SHORT[bands[c]] ?? bands[c], series, yDomain: [lo - pad, hi + pad] as [number, number], bands: shade };
  }), [bandIdx.join(","), models, showMembers, view, mean, knees, colorBy, kneeOrder, isFull, range[0], range[1], bands]); // eslint-disable-line react-hooks/exhaustive-deps

  const legend = useMemo<LegendItem[]>(() => {
    const main: LegendItem[] = [];
    const facets = new Map<string, Facet>();
    for (const m of models) {
      if (m.kind === "member") {
        if (!showMembers) continue;
        const f = memberFacet(m, colorBy, kneeOrder);
        if (!facets.has(f.key)) facets.set(f.key, f);
      } else if (!main.some((it) => it.label === kneeModelName(m))) {
        main.push({ label: kneeModelName(m), key: kneeModelName(m), color: modelColor(m, colorBy, kneeOrder), line: true });
      }
    }
    main.sort((a, b) => Number(b.label === "production gate") - Number(a.label === "production gate"));
    const facetItems = [...facets.values()].sort((a, b) => a.rank - b.rank).map(({ key, label, color, dash }) => ({ key, label, color, dash }));
    return [...main, ...facetItems];
  }, [models, showMembers, colorBy, kneeOrder]);

  /* leaderboard */
  const byLabel = useMemo(() => new Map((memberRows ?? []).map((m) => [memberNumber(m.name) ?? m.name, m])), [memberRows]);
  const board = useMemo<BoardRow[]>(() => (knees.length
    ? kneeLeaderboard(models, knees, range, band === "all" ? null : bands.map((b) => b === band)).map((r) => {
      const m: MemberRow | undefined = r.kind === "member" ? byLabel.get(memberNumber(r.label) ?? "") : undefined;
      const testVis = r.kind === "member" ? m?.vis_psnr ?? null
        : r.kind === "mean" ? o?.headline.mean.psnr ?? null
          : r.id === "spatial_gate" ? o?.headline.production.psnr ?? null : null;
      return { ...r, testVis, test4b: m?.psnr ?? null };
    }) : []), [models, knees, range[0], range[1], band, bands, byLabel, o]); // eslint-disable-line react-hooks/exhaustive-deps

  const kneeAt = o?.headline.knee_e ?? 100;
  const columns = useMemo<DataColumn<BoardRow>[]>(() => [
    { id: "rank", header: "#", numeric: true, width: 48 },
    { id: "rankDelta", header: "Δ", headerText: "rank change vs full range", numeric: true, width: 52, priority: 2,
      cell: (r) => (r.rankDelta == null || r.rankDelta === 0 || isFull ? <span className="mdl-faint">·</span>
        : <span className={r.rankDelta > 0 ? "mdl-good" : "mdl-bad"}>{r.rankDelta > 0 ? `▲${r.rankDelta}` : `▼${-r.rankDelta}`}</span>) },
    { id: "label", header: "Model", accessor: (r) => kneeModelName(r.model),
      cell: (r) => <span className="mdl-member"><span className="mdl-swatch" style={{ ["--sw" as string]: modelColor(r.model, colorBy, kneeOrder) }} />{kneeModelName(r.model)}</span> },
    { id: "knee", header: "Trained", accessor: (r) => (r.kind === "member" ? kneeText(r.model).text : r.kind), width: 128, priority: 4 },
    { id: "loss", header: "Loss", accessor: (r) => r.model.loss ?? null, hidden: true },
    ...bands.map((b, i): DataColumn<BoardRow> => ({
      id: `b_${b}`, header: `∫${BAND_SHORT[b] ?? b}`, numeric: true, priority: 3,
      accessor: (r) => (r.bands[i] == null ? null : Number((r.bands[i] as number).toFixed(4))), cell: (r) => db(r.bands[i]),
    })),
    { id: "mean", header: band === "all" ? "∫ mean" : `∫ ${BAND_SHORT[band] ?? band}`, numeric: true,
      accessor: (r) => (r.mean == null ? null : Number(r.mean.toFixed(4))), cell: (r) => <b>{db(r.mean)}</b> },
    { id: "vsMean", header: "vs mean", numeric: true, accessor: (r) => r.vsMean,
      cell: (r) => <span className={r.vsMean != null && r.vsMean > 0 ? "mdl-good" : undefined}>{dbDelta(r.vsMean)}</span> },
    { id: "testVis", header: `Test VIS @${kneeNum(kneeAt)} e⁻`, headerText: `test PSNR VIS at the ${kneeAt} e⁻ knee`, numeric: true, hidden: true,
      accessor: (r) => r.testVis, cell: (r) => db(r.testVis) },
    { id: "test4b", header: "Test 4b", headerText: "test PSNR, joint 4-band asinh (member-PSNR cache)", numeric: true, hidden: true,
      accessor: (r) => r.test4b, cell: (r) => db(r.test4b) },
  ], [bands, band, colorBy, kneeOrder, isFull, kneeAt]);

  const integration = kneeData?.integration ?? o?.headline.knee.integration ?? null;
  const nKneeFields = kneeData?.n_fields ?? o?.headline.knee.n_fields ?? null;
  const caption = o ? [
    `${o.n_members} member${o.n_members === 1 ? "" : "s"}`,
    o.production_gate.available && o.production_gate.fitted_at
      ? `production gate fitted ${nb(formatRelative(o.production_gate.fitted_at))}${o.production_gate.mix_space ? ` (${o.production_gate.mix_space} mix)` : ""}` : null,
    cmp.rows.length ? `∫ = PSNR averaged over log knee ${nb(`${kneeE(integration?.from_e ?? 0.1)}–${kneeE(integration?.to_e ?? 1e4)} e⁻`)}${nKneeFields ? ` on ${nKneeFields} test fields` : ""}` : null,
    o.evaluated_at ? `evaluated ${nb(formatRelative(o.evaluated_at))}` : "not evaluated yet",
  ].filter(Boolean).join(" · ") : "";

  const note = () => [o ? evaluationNote(o) : "",
    board.length ? kneeNote(board, { range, full, band, bands, nFields: kneeData?.n_fields }) : ""].filter(Boolean).join("\n\n");
  const runMenu = (
    <Menu label="Run" trigger={<Button size="sm" iconRight="chevronDown" loading={evalJob.busy || kneeJob.busy || psnrJob.busy}>Run</Button>} items={[
      { label: "Evaluate test fields…", onSelect: run_.evaluate, disabled: o ? !o.test_present : true },
      { label: "Evaluate, forcing re-inference…", onSelect: run_.force, disabled: o ? !o.test_present : true },
      { type: "separator" },
      { label: "Member PSNR…", onSelect: run_.psnr },
      { label: "Knee PSNR…", onSelect: run_.knee },
    ]} />
  );
  const jobs = (evalJob.job || kneeJob.job || psnrJob.job || evalJob.error) && (
    <div className="mdl-jobs">
      <JobProgress job={evalJob.job} error={evalJob.error} />
      <JobProgress job={kneeJob.job} error={kneeJob.error} />
      <JobProgress job={psnrJob.job} error={psnrJob.error} />
    </div>
  );

  return (
    <Page className="mdl-page">
      <LoadState loading={ov.loading} error={ov.error} onRetry={ov.reload}>
        {o && (
          <div className="mdl-stack">
            <StatusLine checks={checks} o={o} nFields={nFields} onEvaluate={run_.evaluate} />
            {members.error && !members.data && <p className="mdl-note">Could not read the members: {members.error.message}</p>}
            {jobs}
            <section className="mdl-result" aria-label="Production model">
              <Verdict cmp={cmp} />
              <Caption>{caption}</Caption>
              {cmp.rows.length > 0 && <ComparisonTable cmp={cmp} bands={bands} bench={bench} />}
              {cmp.rows.length > 0 && <RealSource bench={bench} />}
            </section>
            {timeouts.length > 0 && (
              <Callout tone="warn" dense action={(
                <Button size="sm" asChild>
                  <Link to={`${tabPath("train")}?mode=continue&members=${timeouts.map((m) => m.name).join(",")}`}>Continue</Link>
                </Button>
              )}>
                {`${timeouts.length} member${timeouts.length === 1 ? "" : "s"} stopped short of the target steps: ${timeouts.map((m) => memberNumber(m.name)).join(", ")}`}
              </Callout>
            )}
          </div>
        )}
      </LoadState>
      <section className="mdl-stack mdl-knee" aria-label="PSNR vs knee">
        <Toolbar label="Knee curves">
          <ToolbarGroup label="View">
            <Segmented<View> size="sm" aria-label="Curve view" value={view} onChange={setView}
              options={[{ value: "relative", label: "vs mean" }, { value: "absolute", label: "absolute" }]} />
          </ToolbarGroup>
          <ToolbarGroup label="Band">
            <Segmented size="sm" aria-label="Band" value={band} onChange={setBand}
              options={[{ value: "all", label: "All" }, ...bands.map((b) => ({ value: b, label: BAND_SHORT[b] ?? b }))]} />
          </ToolbarGroup>
          <ToolbarGroup label="Colour">
            <Segmented<Colour> size="sm" aria-label="Colour members by" value={colorBy} onChange={setColorBy}
              options={[{ value: "knee", label: "knee" }, { value: "loss", label: "loss" }, { value: "multi", label: "multi" }]} />
          </ToolbarGroup>
          <Chip on={showMembers} onClick={() => setShowMembers(!showMembers)}>members</Chip>
          <ToolbarGroup label="Integrate">
            <RangeSlider className="mdl-range" aria-label="Knee integration range" scale="log"
              min={full[0]} max={full[1]} value={range} format={(v) => `${kneeNum(v)} e⁻`} showValue
              onChange={(v) => setRangeRaw(`${+v[0].toPrecision(3)},${+v[1].toPrecision(3)}`)} />
            {!isFull && <IconButton size="sm" icon="reset" label="Integrate over the full range" onClick={() => setRangeRaw("")} />}
          </ToolbarGroup>
          <ToolbarSpacer />
          {kneeData?.stale && <Badge tone="warn">curves stale</Badge>}
          <LogToNotebookButton from="Models › Leaderboard" disabled={!o} note={note}
            title="Open Notebook › Log with the evaluation summary and this leaderboard (range and band); you edit it there first" />
          <Button size="sm" onClick={() => setFreezeOpen(true)}
            title="Freeze the whole ensemble (members, gate, comparisons and up to 10 fields) into a study in Figures › Studies">Freeze study…</Button>
          {runMenu}
        </Toolbar>
        <LoadState loading={kneeRes.loading} error={kneeRes.error} onRetry={kneeRes.reload}
          empty={kneeData && !kneeData.available && (kneeOffered
            // The status line already names this failure and offers its fix: one quiet line here.
            ? <p className="mdl-note">No curves to draw yet{checks.some((c) => c.fix === "evaluate") ? ": evaluate, then compute Knee PSNR (above)" : ": compute Knee PSNR (above)"}.</p>
            : (
              <EmptyState icon="activity" title="No PSNR-vs-knee curves" action={<Button variant="primary" onClick={run_.knee}>Knee PSNR</Button>}>
                {kneeData.reason ?? "Compute them from the cached test cubes (evaluate the ensemble first)."}
              </EmptyState>
            ))}>
          <Legend items={legend} {...lg.legendProps} />
          <div className={panels.length > 1 ? "mdl-charts" : undefined}>
            {panels.map((p) => (
              <div key={p.c} className="mdl-chart">
                <h3 className="mdl-chart__title">{p.name}</h3>
                <Plot {...lg.plotProps} xScale="log" xDomain={full} yDomain={p.yDomain} xTicks={logTicks(full)}
                  xLabel="scoring knee [e⁻]" yLabel={view === "relative" ? "PSNR − mean [dB]" : "PSNR [dB]"}
                  series={p.series} bands={p.bands} aspect={panels.length > 1 ? 0.62 : 0.45} syncKey="mdl-knee"
                  xFormat={(v) => `${kneeNum(v)} e⁻`} yFormat={(v) => v.toFixed(2)}
                  exportName={`knee-psnr-${p.name}`} aria-label={`PSNR vs knee, ${p.name}`} />
              </div>
            ))}
          </div>
          <DataTable rows={board} columns={columns} rowKey={(r) => r.id} aria-label="Knee-integrated PSNR leaderboard"
            urlKey="k" defaultSort={[{ id: "rank", desc: false }]} exportName="knee-leaderboard" height={480}
            caption={`${cmp.rows.length && nKneeFields ? "" : `${kneeData?.n_fields ?? "?"} test fields · `}∫ uniform in log knee over ${kneeNum(range[0])}–${kneeNum(range[1])} e⁻`}
            inspect={(r) => (r.kind === "member" ? { kind: "member", id: `member_${memberNumber(r.label)}` }
              : r.kind === "combiner" ? { kind: "combiner", id: "spatial_gate_combiner" } : null)} />
        </LoadState>
      </section>
      {freezeOpen && <FreezeStudyDialog onClose={() => setFreezeOpen(false)} />}
      {evalAsk && (
        <EvaluateDialog open onOpenChange={(v) => { if (!v) setEvalAsk(null); }}
          defaultN={askN} lastN={o?.headline.n_scored ?? null} defaultForce={evalAsk.force}
          onStart={(n, force) => void evaluate(n, force, { asked: true })} />
      )}
    </Page>
  );
}
