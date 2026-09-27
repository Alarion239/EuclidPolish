/* ensemble/knee (spec §8.2): PSNR as a function of the asinh knee for every
   member, the plain mean and each combiner (GET /ensemble/knee-psnr.json),
   per band, absolute or relative to the plain mean; and the sortable
   leaderboard of knee-integrated PSNR per band + mean over a SELECTABLE
   integration range (rank and its change vs the full 0.1–10⁴ e⁻ range),
   CSV export and the compute job. No 100 e⁻ reference line (user). */
import { useMemo } from "react";
import Plot, { Legend, useLegend, type Band as PlotBand, type LegendItem, type Series } from "../../../charts/Plot";
import { C, categorical, LOSS_COLOR, viridis } from "../../../colors";
import { useJob } from "../../../api/jobs";
import { usePageActions } from "../../../app/palette";
import { startJob } from "../../../app/RunActions";
import { useUrlState } from "../../../hooks/useUrlState";
import { logTicks } from "../../../ticks";
import {
  Badge, Button, Chip, DataTable, EmptyState, JobProgress, Page, RangeSlider, Segmented, type DataColumn,
} from "../../../ui";
import { BAND_SHORT, useKnee, useMode, type KneeModel } from "../api";
import { BarGroup, EnsBar, LoadState } from "../common";
import { JOB, useOnJobEnd } from "../jobs";
import { db, dbDelta, kneeLeaderboard, kneeText, memberNumber, relativeTo, type LeaderRow } from "../model";
import "../ensemble.css";

type View = "relative" | "absolute";
type Colour = "knee" | "loss" | "multi";

const kfmt = (v: number) => (v >= 1000 ? `${+(v / 1000).toPrecision(3)}k` : `${+v.toPrecision(3)}`);
const parseRange = (raw: string): [number, number] | undefined => {
  const [a, b] = raw.split(",").map(Number);
  return a > 0 && b > 0 ? [Math.min(a, b), Math.max(a, b)] : undefined;
};

function modelColor(m: KneeModel, by: Colour, kneeOrder: number[]): string {
  if (m.kind === "mean") return C.mean;
  if (m.kind === "combiner") return C.comb;
  const k = kneeText(m);
  if (by === "loss") return LOSS_COLOR[(m.loss ?? "l1").toLowerCase()];
  if (by === "multi") return k.kind === "multi" ? categorical(m.output_knee != null ? 1 : 3) : C.muted;
  if (k.kind === "multi") return categorical(m.output_knee != null ? 1 : 3);
  const i = kneeOrder.indexOf(k.sort);
  return viridis(kneeOrder.length > 1 ? 0.1 + (0.8 * i) / (kneeOrder.length - 1) : 0.5);
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
const LOSS_ORDER = ["l1", "l2", "l3", "mse", "berhu"];

const displayName = (m: KneeModel) => (m.kind === "member" ? `#${memberNumber(m.label) ?? m.label}`
  : m.kind === "mean" ? "plain mean" : m.id === "spatial_gate" ? "production gate" : m.label);

export default function Knee() {
  const mode = useMode();
  const res = useKnee(mode);
  const job = useJob(JOB.knee);
  useOnJobEnd(job.job);
  const [view, setView] = useUrlState<View>("view", "relative");
  const [band, setBand] = useUrlState("band", "all");
  const [colorBy, setColorBy] = useUrlState<Colour>("color", "knee");
  const [rangeRaw, setRangeRaw] = useUrlState("range", "");
  const [showMembers, setShowMembers] = useUrlState("members", true);
  const lg = useLegend();
  const data = res.data;
  const knees = useMemo(() => data?.knees ?? [], [data]);
  const bands = useMemo(() => data?.bands ?? [], [data]);
  const models = useMemo(() => data?.models ?? [], [data]);
  const full: [number, number] = knees.length ? [knees[0], knees[knees.length - 1]] : [0.1, 1e4];
  const range = parseRange(rangeRaw) ?? full;
  const isFull = range[0] <= full[0] && range[1] >= full[1];

  const compute = () => void startJob({
    key: JOB.knee, url: "/ensemble/knee-psnr", label: `PSNR vs knee (${mode})`, data: { mode },
    question: { title: "Recompute PSNR vs knee?", message: "Scores every member, the mean and each combiner at every knee from the cached test cubes.", confirmLabel: "Compute" },
  });
  usePageActions([
    { id: "knee-compute", label: "Compute PSNR vs knee", group: "Knee", run: compute },
    { id: "knee-full", label: "Knee: integrate over the full range", group: "Knee", disabled: isFull, run: () => setRangeRaw("") },
    { id: "knee-relative", label: "Knee: curves relative to the plain mean", group: "Knee", run: () => setView("relative") },
  ]);

  const kneeOrder = useMemo(() => [...new Set(models.filter((m) => m.kind === "member" && kneeText(m).kind !== "multi")
    .map((m) => kneeText(m).sort))].sort((a, b) => a - b), [models]);
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
        name: `${displayName(m)}${m.kind === "member" ? ` · ${k.text}` : ""}`,
        key: m.kind === "member" ? memberFacet(m, colorBy, kneeOrder).key : displayName(m),
      });
    }
    if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
    const pad = (hi - lo) * 0.06 || 0.5;
    const bandsShade: PlotBand[] = isFull ? [] : [{ axis: "x", from: range[0], to: range[1], color: C.guide, alpha: 0.18, label: "integration range" }];
    return { c, name: BAND_SHORT[bands[c]] ?? bands[c], series, yDomain: [lo - pad, hi + pad] as [number, number], bands: bandsShade };
  }), [bandIdx.join(","), models, showMembers, view, mean, knees, colorBy, kneeOrder, isFull, range[0], range[1], bands]); // eslint-disable-line react-hooks/exhaustive-deps

  const legend = useMemo<LegendItem[]>(() => {
    const main: LegendItem[] = [];
    const facets = new Map<string, Facet>();
    for (const m of models) {
      if (m.kind === "member") {
        if (!showMembers) continue;
        const f = memberFacet(m, colorBy, kneeOrder);
        if (!facets.has(f.key)) facets.set(f.key, f);
      } else if (!main.some((it) => it.label === displayName(m))) {
        main.push({ label: displayName(m), key: displayName(m), color: modelColor(m, colorBy, kneeOrder), line: true });
      }
    }
    main.sort((a, b) => Number(b.label === "production gate") - Number(a.label === "production gate"));
    const members = [...facets.values()].sort((a, b) => a.rank - b.rank)
      .map(({ key, label, color, dash }) => ({ key, label, color, dash }));
    return [...main, ...members];
  }, [models, showMembers, colorBy, kneeOrder]);

  const board = useMemo(() => (knees.length ? kneeLeaderboard(models, knees, range, band === "all" ? null : bands.map((b) => b === band)) : []),
    [models, knees, range[0], range[1], band, bands]); // eslint-disable-line react-hooks/exhaustive-deps

  const columns = useMemo<DataColumn<LeaderRow>[]>(() => [
    { id: "rank", header: "#", numeric: true, width: 48 },
    { id: "rankDelta", header: "Δ", headerText: "rank change vs full range", numeric: true, width: 52,
      cell: (r) => (r.rankDelta == null || r.rankDelta === 0 || isFull ? <span className="ens-faint">·</span>
        : <span className={r.rankDelta > 0 ? "ens-good" : "ens-bad"}>{r.rankDelta > 0 ? `▲${r.rankDelta}` : `▼${-r.rankDelta}`}</span>) },
    { id: "label", header: "Model", accessor: (r) => displayName(r.model),
      cell: (r) => <span className="ens-member"><span className="ens-swatch" style={{ ["--sw" as string]: modelColor(r.model, colorBy, kneeOrder) }} />{displayName(r.model)}</span> },
    { id: "knee", header: "Trained", accessor: (r) => (r.kind === "member" ? kneeText(r.model).text : r.kind), width: 128 },
    { id: "loss", header: "Loss", accessor: (r) => r.model.loss ?? null, hidden: true },
    ...bands.map((b, i): DataColumn<LeaderRow> => ({
      id: `b_${b}`, header: `∫${BAND_SHORT[b] ?? b}`, numeric: true,
      accessor: (r) => (r.bands[i] == null ? null : Number((r.bands[i] as number).toFixed(4))), cell: (r) => db(r.bands[i]),
    })),
    { id: "mean", header: band === "all" ? "∫ mean" : `∫ ${BAND_SHORT[band] ?? band}`, numeric: true,
      accessor: (r) => (r.mean == null ? null : Number(r.mean.toFixed(4))), cell: (r) => <b>{db(r.mean)}</b> },
    { id: "vsMean", header: "vs mean", numeric: true, accessor: (r) => r.vsMean,
      cell: (r) => <span className={r.vsMean != null && r.vsMean > 0 ? "ens-good" : undefined}>{dbDelta(r.vsMean)}</span> },
  ], [bands, band, colorBy, kneeOrder, isFull]);

  const xTicks = logTicks(full);
  return (
    <Page>
      <EnsBar label="Knee controls">
        <BarGroup label="View">
          <Segmented<View> size="sm" aria-label="Curve view" value={view} onChange={setView}
            options={[{ value: "relative", label: "vs mean" }, { value: "absolute", label: "absolute" }]} />
        </BarGroup>
        <BarGroup label="Band">
          <Segmented size="sm" aria-label="Band" value={band} onChange={setBand}
            options={[{ value: "all", label: "All" }, ...bands.map((b) => ({ value: b, label: BAND_SHORT[b] ?? b }))]} />
        </BarGroup>
        <BarGroup label="Colour">
          <Segmented<Colour> size="sm" aria-label="Colour members by" value={colorBy} onChange={setColorBy}
            options={[{ value: "knee", label: "knee" }, { value: "loss", label: "loss" }, { value: "multi", label: "multi" }]} />
        </BarGroup>
        <Chip on={showMembers} onClick={() => setShowMembers(!showMembers)}>members</Chip>
        <span className="ens-bar__spacer" />
        {data?.stale && <Badge tone="warn">stale</Badge>}
        <Button size="sm" loading={job.busy} onClick={compute}>Compute</Button>
      </EnsBar>
      <LoadState loading={res.loading} error={res.error} onRetry={res.reload}
        empty={data && !data.available && (
          <EmptyState icon="activity" title="No PSNR-vs-knee curves" action={<Button variant="primary" onClick={compute}>Compute</Button>}>
            {data.reason ?? "Compute them from the cached test cubes (evaluate the ensemble first)."}
          </EmptyState>
        )}>
        <div className="ens-stack">
          <div className="ens-row">
            <span className="ens-bar__label">Integrate</span>
            <RangeSlider style={{ flex: "1 1 240px", maxWidth: 420 }} aria-label="Knee integration range" scale="log"
              min={full[0]} max={full[1]} value={range} format={(v) => `${kfmt(v)} e⁻`} showValue
              onChange={(v) => setRangeRaw(`${+v[0].toPrecision(3)},${+v[1].toPrecision(3)}`)} />
            {!isFull && <Button size="sm" variant="ghost" onClick={() => setRangeRaw("")}>full range</Button>}
            <span className="ens-faint">{data?.n_fields ?? "?"} test fields · uniform in log knee</span>
          </div>
          <Legend items={legend} {...lg.legendProps} />
          <div className={panels.length > 1 ? "ens-charts" : undefined}>
            {panels.map((p) => (
              <div key={p.c} className="ens-chart">
                <h3 className="ens-chart__title">{p.name}</h3>
                <Plot {...lg.plotProps} xScale="log" xDomain={full} yDomain={p.yDomain} xTicks={xTicks}
                  xLabel="scoring knee [e⁻]" yLabel={view === "relative" ? "PSNR − mean [dB]" : "PSNR [dB]"}
                  series={p.series} bands={p.bands} aspect={panels.length > 1 ? 0.62 : 0.45} syncKey="ens-knee"
                  xFormat={(v) => `${kfmt(v)} e⁻`} yFormat={(v) => v.toFixed(2)}
                  exportName={`knee-psnr-${mode}-${p.name}`} aria-label={`PSNR vs knee, ${p.name}`} />
              </div>
            ))}
          </div>
          <DataTable rows={board} columns={columns} rowKey={(r) => r.id} aria-label="Knee-integrated PSNR leaderboard"
            urlKey="k" defaultSort={[{ id: "rank", desc: false }]} exportName={`knee-leaderboard-${mode}`} height={480}
            inspect={(r) => (r.kind === "member" ? { kind: "member", id: `member_${memberNumber(r.label)}` }
              : r.kind === "combiner" ? { kind: "combiner", id: `${mode}/spatial_gate_combiner` } : null)} />
          <JobProgress job={job.job} error={job.error} />
        </div>
      </LoadState>
    </Page>
  );
}
