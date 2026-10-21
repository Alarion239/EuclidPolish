/* Models › Spectra (`/models/spectra`): the power spectra of every universal
   model on the synthetic test fields, side by side (measured by the
   universal score job; shaped in ../spectra.ts). From the top:
   - callouts: the models without current spectra with "Compute spectra" (a
     confirmed local job, then its progress; a score job started anywhere is
     followed too, the page reading again as it writes and when it ends),
     test records that are not local (→ Synthetic › Records), the server's
     notes and the models whose spectra cannot be measured (why);
   - the toolbar (URL state): the section (?s= power | coherence |
     direction), the band (?band=) and the models shown (?m=, at most eight,
     in pick order, each keeping one colour in every figure; by default the
     best ∫PSNR, the newest trained without the isotropy rule and the
     newest);
   - Power: each output's radial power against the truth's by period
     (log-log, fine periods left) and the transfer √(P_out / P_truth) (log,
     about 1), the HR (2 px) and LR-pixel (4 px) Nyquist periods marked;
   - Coherence: r by period, and where each falls to ½;
   - Direction, at a period band (?pb=, a toolbar choice): which models
     show a pixel-axis grid there (axis / diagonal ≥ 1.2; one band at a time,
     as 2–3 px exceeds it for every model), one model's 2-D error map (?map=,
     fetched on demand; log colour about 1 with the period rings; a click
     reads a cell), the angular profiles, and every model's axis / diagonal
     ratio per period band.
   Opening the page reads only (GETs); the one job starts after a confirm. */
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import Plot, { Legend, useLegend, type Cell, type Guide, type Heat, type LegendItem, type Series } from "../../../charts/Plot";
import { C, categorical } from "../../../colors";
import { useJob, useTrackedJob } from "../../../api/jobs";
import { pagePath } from "../../../app/nav";
import { usePageActions } from "../../../app/palette";
import { formatNumber } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { logTicks } from "../../../ticks";
import {
  Badge, Button, Callout, Caption, DataTable, EmptyState, JobProgress, MultiSelect, Num, Page, Segmented, Select, Skeleton, SummaryLine,
  Toolbar, ToolbarGroup, Tooltip, type DataColumn,
} from "../../../ui";
import { BANDS, type Band } from "../api";
import { LoadState } from "../common";
import { JOB, UNIVERSAL_SCORE_LABEL, computeSpectra, useOnJobEnd } from "../jobs";
import { db } from "../model";
import {
  ANGLE_DOMAIN, ANGLE_TICKS, DEFAULT_PB, GRID_RATIO, HR_NYQUIST_PX, LR_NYQUIST_PX, MAP_COLOR_TICKS, MAP_LOG2_RANGE, MAX_SHOWN,
  PERIOD_DOMAIN, PERIOD_TICKS, SECTIONS, angularCurve, angularKeys, bandText, byNumberOf, cellText, coherenceDomain, coherencePeriod,
  coherenceTicks, colourSlots, directionRows, divergingColor, effectivePb, fieldsOf, freshNotes, gridVerdict, hasAngular, hasDirection,
  hasRadial, hasValues, isGrid, limitPick, mapCell, mapLayer, mapTicks, modelHint, modelLabel, normaliseRecord, normaliseSpectra, notDrawn,
  parseBand, parseSection, pbLabel, periodBandKeys, periodCurve, periodText, pickable, radialRow, ratioDomain, reasonGroups, ringCurves,
  shownModels, spectraEstimate, truthRow, type AxisDiagCell, type Curve, type DirectionRow, type RadialKind, type Section,
} from "../spectra";
import { useSpectra, useSpectraRecord, type SpectraModel, type SpectraPayload } from "../spectraApi";
import { DEFAULT_FIELDS, logDomain, numbersText, stepsShort } from "../universal";
import "../models.css";

/** The summary is read again this often while a score job writes it. */
const POLL = 10_000;
const TRUTH = "truth";
const PERIOD_LABEL = "period [px], fine → coarse";
const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;
/** A server note as a sentence: capitalised, with a full stop. */
const sentence = (t: string) => {
  const s = t.trim();
  return s ? `${s[0].toUpperCase()}${s.slice(1)}${/[.!?]$/.test(s) ? "" : "."}` : s;
};
const knee = (e: number) => formatNumber(e, { sig: 3 });
/** "#195, #196 and 4 more". */
const labelsText = (labels: readonly string[], max = 6) =>
  (labels.length > max ? `${labels.slice(0, max).join(", ")} and ${labels.length - max} more` : labels.join(", "));

type LineStyle = Pick<Series, "color" | "alpha">;
type SectionProps = {
  data: SpectraPayload; models: readonly SpectraModel[]; band: Band; nFields: number;
  styleOf: (m: SpectraModel) => LineStyle;
};

const legendLabel = (m: SpectraModel) => `${modelLabel(m)}${m.status === "stale" ? " (stale)" : ""}`;
const modelLine = (m: SpectraModel, c: Curve, style: LineStyle): Series => ({ ...c, ...style, width: 1.9, name: legendLabel(m), key: m.name });
const modelItems = (models: readonly SpectraModel[], styleOf: (m: SpectraModel) => LineStyle): LegendItem[] =>
  models.map((m) => ({ label: legendLabel(m), color: styleOf(m).color, key: m.name }));

function nyquistGuides(): Guide[] {
  const g = { axis: "x" as const, color: C.cross, dash: [6, 3], alpha: 0.7 };
  return [{ ...g, v: HR_NYQUIST_PX, label: "HR Nyquist" }, { ...g, v: LR_NYQUIST_PX, label: "LR Nyquist" }];
}
const levelGuide = (v: number, dash = [2, 3]): Guide => ({ axis: "y", v, color: C.cross, dash, alpha: 0.8 });

/** The shown models a figure cannot draw, with the server's reason. */
function NotDrawn({ models, has, what }: { models: readonly SpectraModel[]; has: (m: SpectraModel) => boolean; what: string }) {
  const off = notDrawn(models, has);
  return off.length ? <p className="mdl-note">No {what} for {off.join(", ")}.</p> : null;
}

/* ── Power ─────────────────────────────────────────────────────────────── */

function PowerSection({ data, models, band, nFields, styleOf }: SectionProps) {
  const lg = useLegend();
  const drawn = models.filter(hasRadial);
  if (!drawn.length) {
    return (
      <>
        <EmptyState compact icon="activity" title="No radial spectra for the models shown" />
        <NotDrawn models={models} has={hasRadial} what="radial spectra" />
      </>
    );
  }
  const b = bandText(band);
  const truth = periodCurve(data, truthRow(data, band));
  const withTruth = hasValues(truth);
  const line = (m: SpectraModel, kind: RadialKind) => modelLine(m, periodCurve(data, radialRow(data, m, kind, band)), styleOf(m));
  const power: Series[] = [
    ...(withTruth ? [{ ...truth, color: C.cross, width: 3.2, alpha: 0.7, name: "truth", key: TRUTH }] : []),
    ...drawn.map((m) => line(m, "power_sr")),
  ];
  const transfer = drawn.map((m) => line(m, "transfer"));
  const powerDomain = logDomain(power.flatMap((s) => s.y), [1e-4, 1e2], 1.5);
  const transferDomain = ratioDomain(transfer.flatMap((s) => s.y));
  const items: LegendItem[] = [...(withTruth ? [{ label: "truth", color: C.cross, key: TRUTH }] : []), ...modelItems(drawn, styleOf)];
  const axes = {
    xScale: "log" as const, yScale: "log" as const, xDomain: PERIOD_DOMAIN, xTicks: PERIOD_TICKS, xLabel: PERIOD_LABEL,
    aspect: 0.42, syncKey: "spectra-period", xFormat: (v: number) => periodText(v, data.pixel_scale_arcsec),
  };
  return (
    <div className="mdl-stack">
      <Legend items={items} {...lg.legendProps} />
      <section className="mdl-stack mdl-stack--tight" aria-labelledby="sp-power-title">
        <h2 id="sp-power-title" className="mdl-h2">Radial power</h2>
        <Plot {...lg.plotProps} {...axes} yDomain={powerDomain} yTicks={logTicks(powerDomain, { maxTicks: 8 })}
          yLabel={`power of asinh(x / ${knee(data.radial.stretch_knee_e)} e⁻), ${b}`} series={power} guides={nyquistGuides()}
          yFormat={(v) => formatNumber(v, { sig: 3 })} exportName={`spectra-power-${band}`} aria-label={`Radial power by period, ${b}`} />
        <Caption>
          Each output's power at each period, ring-averaged over {data.radial.window_px}-px round-Hann windows of the {nFields} test
          fields (white noise of variance σ² would lie flat at σ²){withTruth ? ", against the truth's (the thick grey line)" : ""}. Above
          the truth a model adds structure, below it smooths. Finer than the LR Nyquist (4 px) every detail is super-resolved; 2 px is
          the HR Nyquist.
        </Caption>
      </section>
      <section className="mdl-stack mdl-stack--tight" aria-labelledby="sp-transfer-title">
        <h2 id="sp-transfer-title" className="mdl-h2">Transfer</h2>
        <Plot {...lg.plotProps} {...axes} yDomain={transferDomain} yTicks={logTicks(transferDomain)}
          yLabel={`transfer √(P_out / P_truth), ${b}`} series={transfer} guides={[...nyquistGuides(), levelGuide(1)]}
          yFormat={(v) => v.toFixed(3)} exportName={`spectra-transfer-${band}`} aria-label={`Transfer by period, ${b}`} />
        <Caption>
          The output's amplitude over the truth's at each period, √(P_out / P_truth): 1 is the right amount of structure, under 1 too
          smooth, over 1 invented structure.
        </Caption>
      </section>
      <NotDrawn models={models} has={hasRadial} what="radial spectra" />
    </div>
  );
}

/* ── Coherence ─────────────────────────────────────────────────────────── */

function CoherenceSection({ data, models, band, nFields, styleOf }: SectionProps) {
  const lg = useLegend();
  const drawn = models.filter(hasRadial);
  if (!drawn.length) {
    return (
      <>
        <EmptyState compact icon="activity" title="No radial spectra for the models shown" />
        <NotDrawn models={models} has={hasRadial} what="radial spectra" />
      </>
    );
  }
  const b = bandText(band);
  const curves = drawn.map((m) => ({ m, c: periodCurve(data, radialRow(data, m, "coherence", band)) }));
  const series = curves.map(({ m, c }) => modelLine(m, c, styleOf(m)));
  const yDomain = coherenceDomain(series.flatMap((s) => s.y));
  const halves = curves.flatMap(({ m, c }) => { const h = coherencePeriod(c); return h ? [{ m, ...h }] : []; });
  const falls = halves.filter((h) => !h.holds).sort((a, z) => a.period - z.period);
  const holds = halves.filter((h) => h.holds);
  const guides = [...nyquistGuides(), levelGuide(1), levelGuide(0.5, [1, 4]), ...(yDomain[0] < 0 ? [levelGuide(0, [])] : [])];
  return (
    <div className="mdl-stack">
      {halves.length > 0 && (
        <SummaryLine>
          In {b},{" "}
          {falls.length > 0 && <>
            the coherence falls to ½ at{" "}
            {falls.map((h, i) => (
              <span key={h.m.name}>{i > 0 ? ", " : ""}<Num unit="px">{+h.period.toPrecision(3)}</Num> for {modelLabel(h.m)}</span>
            ))}
          </>}
          {holds.length > 0 && <>
            {falls.length > 0 ? "; " : ""}{holds.map((h) => modelLabel(h.m)).join(", ")} {holds.length === 1 ? "stays" : "stay"} above ½
            down to {periodText(holds[0].period)}
          </>}
          : the finer that period, the further the output follows the truth.
        </SummaryLine>
      )}
      <Legend items={modelItems(drawn, styleOf)} {...lg.legendProps} />
      <section className="mdl-stack mdl-stack--tight" aria-labelledby="sp-coherence-title">
        <h2 id="sp-coherence-title" className="mdl-h2">Coherence with the truth</h2>
        <Plot {...lg.plotProps} xScale="log" xDomain={PERIOD_DOMAIN} xTicks={PERIOD_TICKS} xLabel={PERIOD_LABEL}
          yDomain={yDomain} yTicks={coherenceTicks(yDomain)} yLabel={`coherence r, ${b}`} series={series} guides={guides}
          aspect={0.42} syncKey="spectra-period" xFormat={(v) => periodText(v, data.pixel_scale_arcsec)} yFormat={(v) => v.toFixed(3)}
          exportName={`spectra-coherence-${band}`} aria-label={`Coherence by period, ${b}`} />
        <Caption>
          The normalised cross-spectrum of output and truth, Re⟨S T*⟩ / √(⟨|S|²⟩⟨|T|²⟩), over the {nFields} test fields: 1 means the
          output follows the truth at that period, 0 that it is unrelated (below 0, opposed). It falls toward the fine periods, where
          the input holds less; the dotted line is ½.
        </Caption>
      </section>
      <NotDrawn models={models} has={hasRadial} what="radial spectra" />
    </div>
  );
}

/* ── Direction ─────────────────────────────────────────────────────────── */

function RatioCell({ c }: { c: AxisDiagCell | null | undefined }) {
  if (c?.ratio == null) return <span className="mdl-faint">—</span>;
  const grid = isGrid(c.ratio);
  const title = `axis ${c.axis?.toFixed(3) ?? "—"} · diagonal ${c.diag?.toFixed(3) ?? "—"} (error power over round)${grid ? ": a grid" : ""}`;
  return (
    <span className={grid ? "mdl-warn" : undefined} data-grid={grid || undefined} title={title}>
      {grid ? <b>{c.ratio.toFixed(2)}</b> : c.ratio.toFixed(2)}
    </span>
  );
}

/** One model's 2-D error map in `band`, read on demand (?map=: one of `choices`). */
function ErrorMap({ data, choices, band }: { data: SpectraPayload; choices: readonly SpectraModel[]; band: Band }) {
  const [param, setParam] = useUrlState("map", "");
  const model = choices.find((m) => m.name === param) ?? choices[0] ?? null;
  const res = useSpectraRecord(model?.name, model?.computed_at);
  const rec = useMemo(() => normaliseRecord(res.data), [res.data]);
  const layer = useMemo(() => mapLayer(rec, band), [rec, band]);
  const heat = useMemo<Heat | null>(() => (layer ? {
    z: layer.z, xEdges: layer.xEdges, yEdges: layer.yEdges, scale: "linear", min: MAP_LOG2_RANGE[0], max: MAP_LOG2_RANGE[1],
    colorTicks: MAP_COLOR_TICKS, colorLabel: "error power / round", color: divergingColor,
  } : null), [layer]);
  const [cell, setCell] = useState<Cell | null>(null);
  const b = bandText(band);
  if (!model) return <EmptyState compact icon="activity" title="No direction diagnostics for the models shown" />;
  const label = modelLabel(model);
  const rings = layer ? ringCurves(layer.periodLines, layer.size, layer.extent) : [];
  const info = layer && cell ? mapCell(layer, cell) : null;
  const ringText = [...rings].sort((a, z) => z.period - a.period).map((r) => `${+r.period.toPrecision(3)}`).join(" and ");
  let body;
  if (res.loading) body = <Skeleton height={320} />;
  else if (res.error) {
    body = (
      <EmptyState compact icon="warn" title={res.error.status === 404 ? `No error map for ${label} yet` : "Could not read the error map"}>
        <span className="mdl-mono">{res.error.message}</span>
      </EmptyState>
    );
  } else if (!layer || !heat) body = <EmptyState compact icon="activity" title={`No error map in ${b} for ${label}`} />;
  else {
    const domain: [number, number] = [-layer.extent, layer.extent];
    const ticks = mapTicks(layer.extent);
    body = (
      <>
        <Plot xDomain={domain} yDomain={domain} xTicks={ticks} yTicks={ticks} tooltip={false} aspect={0.9}
          xLabel={`kx [cycles per ${layer.size} px]`} yLabel={`ky [cycles per ${layer.size} px]`} heat={heat}
          series={rings.map((r) => ({ x: r.x, y: r.y, color: C.cross, width: 1.4, dash: [5, 3], name: `${r.period} px`, key: `ring:${r.period}` }))}
          onHeatClick={setCell} highlight={cell} exportName={`error-map-${model.name}-${band}`} aria-label={`Error map, ${label}, ${b}`} />
        <p className="mdl-note" aria-live="polite">{info ? cellText(info) : "Click a cell to read it."}</p>
      </>
    );
  }
  return (
    <div className="mdl-chart mdl-stack mdl-stack--tight">
      <div className="mdl-row">
        <h3 className="mdl-chart__title">Error map · {label} · {b}</h3>
        {choices.length > 1 && (
          <Select size="sm" aria-label="Error map of" value={model.name} onChange={setParam}
            options={choices.map((m) => ({ value: m.name, label: legendLabel(m) }))} />
        )}
      </div>
      {body}
      <Caption>
        The error (output − truth in asinh(x / {knee(data.direction.stretch_knee_e)} e⁻)) as a 2-D power spectrum, each frequency over
        its ring's mean{layer?.nWindows ? ` (${formatNumber(layer.nWindows, { grouping: true })} windows)` : ""}: white is no preferred
        direction, red more error than the ring's average, blue less (log colour, ½ to 2). A cross on the kx = 0 and ky = 0 axes is the
        pixel-axis grid.{ringText ? ` The dashed rings mark periods of ${ringText} px (period = ${layer?.size ?? data.direction.window_px} px / |k|).` : ""}
      </Caption>
    </div>
  );
}

/** The Direction section's one-line answer: which models show a grid at the
 *  period band `pb` (the smaller group named: the grid ones, or the round). */
function GridVerdict({ rows, pb, band }: { rows: readonly DirectionRow[]; pb: string; band: Band }) {
  const { measured, grid, round } = gridVerdict(rows, pb);
  const n = measured.length;
  if (!n) return null;
  const g = grid.length;
  const rule = `axis / diagonal ≥ ${GRID_RATIO}`;
  return (
    <SummaryLine>
      In {bandText(band)} at {pbLabel(pb)},{" "}
      {g === 0 ? <>no model shows a pixel-axis grid: {n === 1 ? "the one model measured stays" : `all ${n} models measured stay`} under {GRID_RATIO} axis / diagonal</>
        : !round.length ? <>{n === 1 ? "the one model measured shows" : `every one of the ${n} models measured shows`} a pixel-axis grid ({rule})</>
          : g <= round.length
            ? <><Num tone="warn">{g}</Num> of the {plural(n, "model")} measured {g === 1 ? "shows" : "show"} a pixel-axis grid ({rule}):{" "}
              {labelsText(grid.map((r) => r.label))}</>
            : <><Num tone="warn">{g}</Num> of the {plural(n, "model")} measured show a pixel-axis grid ({rule}); the{" "}
              {plural(round.length, "round one")}: {labelsText(round.map((r) => r.label))}</>}.
    </SummaryLine>
  );
}

function DirectionSection({ data, models, band, styleOf, slots, pb }: SectionProps & { slots: ReadonlyMap<string, number>; pb: string }) {
  const lg = useLegend();
  const b = bandText(band);
  const allKeys = useMemo(() => periodBandKeys(data), [data]);
  const rows = useMemo(() => directionRows(data, band), [data, band]);
  const drawn = models.filter(hasAngular);
  const series = drawn.map((m) => modelLine(m, angularCurve(data, m, pb, band), styleOf(m)));
  const yDomain = ratioDomain(series.flatMap((s) => s.y));
  const columns = useMemo<DataColumn<DirectionRow>[]>(() => [
    { id: "model", header: "Model", accessor: (r) => r.num, csv: (r) => r.label, width: 108,
      cell: (r) => (
        <span className="mdl-member">
          <span className="mdl-swatch" style={slots.has(r.name) ? { ["--sw" as string]: `var(--cat-${slots.get(r.name)})` } : undefined} />{r.label}
          {r.status === "stale" && <Badge tone="warn" size="sm">stale</Badge>}
        </span>
      ) },
    { id: "steps", header: "Steps", headerText: "the checkpoint step the spectra were computed at", numeric: true, width: 76,
      accessor: (r) => r.served_step ?? r.steps, cell: (r) => stepsShort(r.served_step ?? r.steps) },
    { id: "psnr", header: "∫PSNR", headerText: "knee-integrated PSNR [dB]", numeric: true, width: 78,
      accessor: (r) => (r.mean_psnr == null ? null : Number(r.mean_psnr.toFixed(4))), cell: (r) => db(r.mean_psnr) },
    { id: "isotropy", header: "Isotropy", headerText: "the isotropy rule's weight in training (— trained without it)", numeric: true, width: 84,
      accessor: (r) => (r.isotropy != null && r.isotropy > 0 ? r.isotropy : null),
      cell: (r) => (r.isotropy != null && r.isotropy > 0 ? formatNumber(r.isotropy, { sig: 3 }) : <span className="mdl-faint">—</span>) },
    { id: "depth", header: "Depth", numeric: true, hidden: true, width: 70, accessor: (r) => r.depth },
    ...allKeys.map((k): DataColumn<DirectionRow> => ({
      id: `pb_${k}`, header: pbLabel(k), headerText: `axis / diagonal, ${pbLabel(k)}`, numeric: true, width: 86,
      accessor: (r) => { const v = r.cells[k]?.ratio; return v == null ? null : Number(v.toFixed(3)); },
      cell: (r) => <RatioCell c={r.cells[k]} />,
    })),
  ], [allKeys, slots]);

  return (
    <div className="mdl-stack">
      <GridVerdict rows={rows} pb={pb} band={band} />
      <div className="mdl-charts">
        <ErrorMap data={data} choices={models.filter(hasDirection)} band={band} />
        <div className="mdl-chart mdl-stack mdl-stack--tight">
          <h3 className="mdl-chart__title">By direction · {pbLabel(pb)} · {b}</h3>
          {drawn.length ? (
            <>
              <Legend items={modelItems(drawn, styleOf)} {...lg.legendProps} />
              <Plot {...lg.plotProps} xDomain={ANGLE_DOMAIN} xTicks={ANGLE_TICKS} yScale="log" yDomain={yDomain} yTicks={logTicks(yDomain)}
                xLabel="wave-vector direction (0° and 90°: the pixel axes)" yLabel="error power / round" series={series}
                guides={[levelGuide(1)]} aspect={0.9} xFormat={(v) => `${Math.round(v)}°`} yFormat={(v) => v.toFixed(2)}
                exportName={`spectra-direction-${pb}-${band}`} aria-label={`Error power by direction, ${pbLabel(pb)}, ${b}`} />
            </>
          ) : <EmptyState compact icon="activity" title="No angular profiles for the models shown" />}
          <Caption>
            The error power by the direction of its wave vector, over periods of {pbLabel(pb)}, each over its round expectation: flat at 1
            is round; peaks at 0°, 90° and 180° are the pixel axes (a grid), at 45° and 135° the diagonals. An empty angle bin is a gap.
          </Caption>
          <NotDrawn models={models} has={hasAngular} what="angular profiles" />
        </div>
      </div>
      <section className="mdl-stack mdl-stack--tight" aria-labelledby="sp-axis-title">
        <h2 id="sp-axis-title" className="mdl-h2">Axis / diagonal by period band</h2>
        <DataTable rows={rows} columns={columns} rowKey={(r) => r.name} aria-label={`Axis over diagonal error power, ${b}`} urlKey="sp"
          defaultSort={[{ id: "model", desc: true }]} exportName={`spectra-axis-diagonal-${band}`} height="auto"
          inspect={(r) => ({ kind: "member", id: r.name })} empty="No model has direction diagnostics yet." />
        <Caption>
          The error power along the pixel axes over that along the diagonals in each period band ({b}, every model with spectra):
          about 1 is round, {GRID_RATIO} or more (bold) a pixel-axis grid. Isotropy: the weight of the isotropy rule in training (— trained
          without it). A coloured swatch marks a model in the figures; a row opens the model.
        </Caption>
      </section>
    </div>
  );
}

/* ── the tab ───────────────────────────────────────────────────────────── */

export default function Spectra() {
  const job = useJob(JOB.universalScore);
  const tracked = useTrackedJob(UNIVERSAL_SCORE_LABEL);
  const live = [job.job, tracked].find((j) => j?.status === "running") ?? null;
  const running = job.busy || !!live;
  const res = useSpectra(running ? POLL : undefined);
  const data = useMemo(() => normaliseSpectra(res.data), [res.data]);
  useOnJobEnd(job.job);
  useOnJobEnd(tracked);
  const [section, setSection] = useUrlState<Section>("s", "power", { parse: parseSection });
  const [band, setBand] = useUrlState<Band>("band", "VIS", { parse: parseBand });
  const [picked, setPicked] = useUrlState<string[]>("m", []);
  const [pbParam, setPb] = useUrlState("pb", DEFAULT_PB);

  const models = useMemo(() => data?.models ?? [], [data]);
  const pbKeys = useMemo(() => (data ? angularKeys(data) : []), [data]);
  const pb = effectivePb(pbParam, pbKeys);
  const list = useMemo(() => pickable(models), [models]);
  const shown = useMemo(() => shownModels(models, picked), [models, picked]);
  const slots = useMemo(() => colourSlots(shown), [shown]);
  const drawn = useMemo(() => byNumberOf(models, shown), [models, shown]);
  const styleOf = (m: SpectraModel): LineStyle =>
    ({ color: categorical(slots.get(m.name) ?? 0), alpha: m.status === "stale" ? 0.55 : undefined });
  const missing = data?.missing ?? [];
  const recordsLocal = data?.records_local ?? true;
  const nFields = fieldsOf(data) ?? DEFAULT_FIELDS;
  const compute = () => void computeSpectra(missing, nFields, spectraEstimate(missing.length));

  usePageActions([
    { id: "spectra-compute", label: "Spectra: compute the missing spectra…", group: "Models",
      keywords: ["spectra", "power spectrum", "isotropy", "grid", "score"], disabled: !missing.length || !recordsLocal || running, run: compute },
    ...SECTIONS.map((s) => ({
      id: `spectra-${s.value}`, label: `Spectra: ${s.title}`, group: "Models", disabled: section === s.value, run: () => setSection(s.value),
    })),
  ]);

  const progress = live ?? job.job ?? tracked;
  const notes = data ? freshNotes(data.notes, data.missing) : [];
  const reasons = reasonGroups(models);
  const reasonText = reasons.map((g) => `${g.labels.join(", ")}: ${g.reason}`).join("; ");
  const computeButton = recordsLocal
    ? <Button size="sm" variant="primary" loading={running} onClick={compute}>Compute spectra</Button>
    : (
      <Tooltip content="Needs the synthetic test records on this machine (Synthetic › Records)">
        <span><Button size="sm" variant="primary" disabled>Compute spectra</Button></span>
      </Tooltip>
    );
  const sectionOf = (d: SpectraPayload) => {
    const props: SectionProps = { data: d, models: drawn, band, nFields, styleOf };
    if (section === "power") return <PowerSection {...props} />;
    if (section === "coherence") return <CoherenceSection {...props} />;
    return <DirectionSection {...props} slots={slots} pb={pb} />;
  };

  return (
    <Page className="mdl-page">
      <LoadState loading={res.loading} error={res.error} onRetry={res.reload}
        empty={res.data != null && !data && (
          <EmptyState icon="warn" title="Could not read the spectra">The server's reply is not a spectra payload.</EmptyState>
        )}>
        {data && (
          <div className="mdl-stack">
            {!recordsLocal && (
              <Callout tone="warn" dense action={<Button size="sm" asChild><Link to={pagePath("synthetic", { tab: "records" })}>Synthetic › Records</Link></Button>}>
                The synthetic test records are not on this machine, so no spectra can be computed here.
              </Callout>
            )}
            {missing.length > 0 && (
              <Callout tone="info" dense action={computeButton}>
                {`${plural(missing.length, "universal model")} ${missing.length === 1 ? "has" : "have"} no spectra for the current test records: ${numbersText(missing)}.`}
                {running && " The score job is computing them: they appear here as each model finishes."}
              </Callout>
            )}
            {(progress || job.error) && <div className="mdl-jobs"><JobProgress job={progress} error={job.error} /></div>}
            {notes.length > 0 && <Caption>{notes.map((n, i) => <span key={i}>{sentence(n)} </span>)}</Caption>}
            {list.length > 0 && reasons.length > 0 && <p className="mdl-note">No spectra for {reasonText}.</p>}
            <Toolbar label="Spectra view">
              <ToolbarGroup label="Section">
                <Segmented<Section> size="sm" aria-label="Section" value={section} onChange={setSection}
                  options={SECTIONS.map(({ value, label, title }) => ({ value, label, title }))} />
              </ToolbarGroup>
              <ToolbarGroup label="Band">
                <Segmented<Band> size="sm" aria-label="Band" value={band} onChange={setBand}
                  options={BANDS.map((b) => ({ value: b, label: bandText(b) }))} />
              </ToolbarGroup>
              {section === "direction" && list.length > 0 && (
                <ToolbarGroup label="Period [px]">
                  <Segmented size="sm" aria-label="Period band" value={pb} onChange={setPb}
                    options={pbKeys.map((k) => ({ value: k, label: k.replace("-", "–"), title: `Periods of ${pbLabel(k)}` }))} />
                </ToolbarGroup>
              )}
              {list.length > 0 && (
                <ToolbarGroup label="Models">
                  <MultiSelect size="sm" aria-label="Models shown" value={shown} maxSummary={4} placeholder="Default pick"
                    onChange={(v) => setPicked(limitPick(v))}
                    options={list.map((m) => ({ value: m.name, label: legendLabel(m), hint: modelHint(m) }))} />
                </ToolbarGroup>
              )}
            </Toolbar>
            {!list.length ? (
              <EmptyState icon="activity" title={models.length ? "No spectra yet" : "No universal models yet"}
                action={models.length ? undefined : <Button variant="primary" asChild><Link to={pagePath("models", { tab: "universal" })}>Universal models</Link></Button>}>
                {!models.length
                  ? "The spectra compare the universal models: pull or train one first."
                  : reasons.length
                    ? `The spectra cannot be measured here: ${reasonText}.`
                    : running
                      ? "The score job is computing them: each model appears here as it finishes (about 25 s per model)."
                      : `The universal score job measures them on the ${nFields} synthetic test fields, about 25 s per model: Compute spectra above.`}
              </EmptyState>
            ) : sectionOf(data)}
            {list.length > MAX_SHOWN && (
              <Caption>
                {plural(list.length, "model")} have spectra; {MAX_SHOWN} are shown at most (pick them in Models above; past {MAX_SHOWN} the
                earliest picked drops). A model keeps its colour in every figure while others are added.
              </Caption>
            )}
          </div>
        )}
      </LoadState>
    </Page>
  );
}
