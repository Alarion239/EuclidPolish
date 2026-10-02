/* Models › Diagnostics (`/models/diagnostics`; absorbs the old
   Ensemble Diagnostics tab, the Sky › Results field diagnostics (?diag=1)
   and the Sky › Catalog eval figures). One section at a time (?d=), each
   drawn from the caches:
   - spectrum r(k) and transfer T(k) (GET /ensemble/evals.json), x focused on
     the SR scales θ < 0.5″ (a chip shows every scale) with the LR-pixel and
     VIS-FWHM labels on opposite sides of their lines;
   - coherence, a sorted horizontal dot plot (16–84% intervals as whiskers);
   - spread: the one-sentence answer (is cross-member σ an error bar?), the
     |z| coverage table observed vs Gaussian, σ vs error and σ vs brightness
     (a cell click back-traces its pixels), the z-pdf on a y domain that fits
     its peak with the coverage guides, and σ vs RMSE per field;
   - real-field: the legacy real field beside its synthetic twin
     (diagnostics/FieldSection.tsx, ?fd=cross|brightness);
   - recovery: SR → HR recovery of the synthetic stamps and their flux
     conservation, drawn from the evaluation rows; the per-band angular power
     spectrum figure on request.
   A band switch (?band=VIS|Y_E|J_E|H_E) picks the band of the evaluation
   sections: VIS is GET /ensemble/evals.json, the NISP bands are the same
   payload per band (?band=), computed from the cached test cubes by
   Evaluate or by a confirmed local job here (never on open); a payload made
   for an earlier evaluation says so. Recovery's angular power spectrum
   follows the same switch. The real field is the legacy VIS field. The old
   d=stderr|brightness|calibration land on spread (the redirect rules);
   d=axes and the RBF are gone. State in the URL. */
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import Plot, { useLegend, type Guide, type Heat, type LegendItem, type Series } from "../../../charts/Plot";
import { C, bandColor, categorical } from "../../../colors";
import { apiPost } from "../../../api/client";
import { useJob } from "../../../api/jobs";
import { useResource } from "../../../api/query";
import { pagePath } from "../../../app/nav";
import { usePageActions } from "../../../app/palette";
import { formatCount } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { decadeTicks, extent, logTicks } from "../../../ticks";
import {
  Button, Callout, Caption, Chip, EmptyState, JobProgress, Num, Page, Segmented, SummaryLine, Table, Toolbar, ToolbarGroup, ToolbarSpacer,
  confirm, toast, type Column,
} from "../../../ui";
import { EVAL_BANDS, url, useEvalRuns, type Aps, type EvalBand, type Evals, type NumArr } from "../api";
import { ColorBySelect, LoadState, useFacetColors } from "../common";
import { JOB, computeBandEvals, useOnJobEnd } from "../jobs";
import {
  bandLabel, coherenceLabel, coherenceOrder, coverageRows, db, facetOf, fieldErrorRatio, formatE, memberNumber, pdfYDomain, spectrumDomain,
  spectrumGuides, spreadVerdict, stampSets, type ColorBy,
} from "../model";
import { PixelTrace, type Pick } from "../PixelTrace";
import { FieldSection } from "../diagnostics/FieldSection";
import { unitTicks } from "../../plotTicks";
import "../models.css";

type Section = "spectrum" | "transfer" | "coherence" | "spread" | "real-field" | "recovery";
const SECTIONS: { value: Section; label: string; title: string }[] = [
  { value: "spectrum", label: "r(k)", title: "Cross-correlation with the target per angular scale" },
  { value: "transfer", label: "T(k)", title: "Transfer function √(P_SR / P_target)" },
  { value: "coherence", label: "Coherence", title: "One-number spectral coherence per model" },
  { value: "spread", label: "Spread", title: "Is the cross-member σ an error bar?" },
  { value: "real-field", label: "Real field", title: "The legacy real field beside its synthetic twin" },
  { value: "recovery", label: "Recovery", title: "SR → HR recovery of the synthetic stamps" },
];
const LEGACY: Record<string, Section> = { stderr: "spread", brightness: "spread", calibration: "spread", axes: "spread" };
const parseSection = (r: string): Section | undefined => (SECTIONS.some((s) => s.value === r) ? r as Section : LEGACY[r]);
const MODEL_LABEL: Record<string, string> = { ensemble_mean: "plain mean", spatial_gate: "production gate" };
/** The RBF combiner is legacy: an old payload may still carry its blocks. */
const isRbf = (kind: string) => /rbf/i.test(kind);
const pct = (v: number | null | undefined) => (v == null || !Number.isFinite(v) ? "—" : `${(100 * v).toFixed(1)}%`);
type CoverRow = { k: number; observed: number | null; gaussian: number };
const COVER_COLUMNS: Column<CoverRow>[] = [
  { header: "", cell: (r) => `|z| < ${r.k}` },
  { header: "Observed", align: "right", cell: (r) => <span className="mdl-tnum">{pct(r.observed)}</span> },
  { header: "Gaussian", align: "right", cell: (r) => <span className="mdl-tnum">{pct(r.gaussian)}</span> },
];
const modelColor = (kind: string) => (kind === "ensemble_mean" ? C.mean : kind === "spatial_gate" ? C.comb : categorical(5));
const num = (a: NumArr | undefined | null) => (a ?? []).map((v) => (v == null ? NaN : v));
const has = (a: NumArr | undefined | null) => (a ?? []).some((v) => v != null && Number.isFinite(v));
const range = (edges: number[], k: number) => `${formatE(10 ** edges[k])}–${formatE(10 ** edges[k + 1])}`;
const X_TICKS = [0.05, 0.1, 0.2, 0.5, 1, 2, 5].map((v) => ({ v, label: String(v) }));

function parseCell(raw: string): Pick | undefined {
  const [diag, i, j] = raw.split(",");
  if (!["std_err", "bright_std"].includes(diag)) return undefined;
  const ii = Number(i), jj = Number(j);
  return Number.isInteger(ii) && Number.isInteger(jj) ? { diag: diag as Pick["diag"], i: ii, j: jj } : undefined;
}

const parseBand = (r: string): EvalBand | undefined => (EVAL_BANDS as readonly string[]).includes(r) ? r as EvalBand : undefined;
const BAND_OPTIONS = EVAL_BANDS.map((b) => ({ value: b, label: bandLabel(b) }));
const REAL_FIELD_BANDS_TITLE = "VIS only: the legacy real field has no Y, J or H diagnostics";

export default function Diagnostics() {
  const [band, setBand] = useUrlState<EvalBand>("band", "VIS", { parse: parseBand });
  const res = useResource<Evals>(url.evals(band), [band], { ttl: 5 * 60_000 });
  const bandJob = useJob(JOB.bandEvals);
  useOnJobEnd(bandJob.job, () => void res.reload());
  const [section, setSection] = useUrlState<Section>("d", "spectrum", { parse: parseSection });
  const [colorBy, setColorBy] = useUrlState<ColorBy>("color", "uniform");
  const [model, setModel] = useUrlState("model", "spatial_gate");
  const [pairs, setPairs] = useUrlState("pairs", false);
  const [allScales, setAllScales] = useUrlState("scales", false);
  const [cellRaw, setCellRaw] = useUrlState("cell", "");
  const cell = parseCell(cellRaw) ?? null;
  const setPick = (p: Pick | null) => setCellRaw(p ? `${p.diag},${p.i},${p.j}` : "");
  const lg = useLegend();
  const e = res.data;
  const members = useMemo(() => e?.members ?? [], [e]);
  const colors = useFacetColors(members, colorBy);
  const memberColor = (i: number) => (colorBy === "uniform" ? C.muted : colors.of(members[i] ?? { loss: "l1" }));

  usePageActions(SECTIONS.map((s) => ({
    id: `diag-${s.value}`, label: `Diagnostics: ${s.title}`, group: "Diagnostics", run: () => setSection(s.value),
  })));

  /* power spectrum r(k) / T(k) */
  const spectrum = useMemo(() => {
    const ps = e?.ps;
    if (!ps || !has(ps.theta)) return null;
    const theta = num(ps.theta);
    const g = e?.guides ?? {};
    const xDomain = spectrumDomain(ps.theta, g.theta_min, !allScales);
    const isT = section === "transfer";
    const series: Series[] = [];
    // ONE legend entry per member facet (key members / members:<facet>);
    // each line keeps its own name for hover identification.
    const legend: LegendItem[] = [];
    const push = (sr: Series, item?: LegendItem) => {
      series.push(sr);
      const it = item ?? (sr.name ? { label: sr.name, key: sr.key ?? sr.name, color: sr.color, dash: !!sr.dash?.length, line: !!sr.dots } : null);
      if (it && !legend.some((l) => (l.key ?? l.label) === (it.key ?? it.label))) legend.push(it);
    };
    if (!isT && pairs) for (const p of ps.r_pairs ?? []) {
      push({ x: theta, y: num(p), color: C.muted, width: 0.6, alpha: 0.15, key: "member pairs" }, { label: "member pairs", key: "member pairs", color: C.muted });
    }
    if (!isT && has(ps.r_cross)) push({ x: theta, y: num(ps.r_cross), color: C.cross, width: 2, dash: [6, 3], name: "member–member r̃(k)" });
    const memberRows = (isT ? ps.T_members : ps.r_members) ?? [];
    if (memberRows.length) {
      if (colorBy === "uniform") legend.push({ label: `members ${memberRows.length}`, key: "members", color: C.muted });
      else for (const it of colors.legend) legend.push({ label: it.label, key: `members:${it.key}`, color: it.color });
    }
    memberRows.forEach((row, i) => {
      const facet = colorBy === "uniform" || !members[i] ? "members" : facetOf(members[i], colorBy);
      const key = colorBy === "uniform" ? "members" : `members:${facet}`;
      push({ x: theta, y: num(row), color: memberColor(i), width: 1, alpha: 0.55, name: `#${memberNumber(members[i]?.label ?? "") ?? i}`, key },
        { label: facet, key, color: memberColor(i) });
    });
    if (!isT && has(ps.r_lr)) push({ x: theta, y: num(ps.r_lr), color: C.baseline, width: 2.2, dash: [7, 4], name: "LR (bicubic)" });
    const mean = isT ? ps.T : ps.r;
    if (has(mean)) push({ x: theta, y: num(mean), color: C.mean, width: 2.6, dots: true, name: "plain mean" });
    for (const [kind, c] of Object.entries(ps.model_combiners ?? {})) {
      if (isRbf(kind)) continue;
      const y = isT ? c.T : c.r;
      if (has(y)) push({ x: theta, y: num(y), color: modelColor(kind), width: 2.4, dots: true, name: MODEL_LABEL[kind] ?? kind });
    }
    const guides: Guide[] = [
      { axis: "y", v: 1, color: C.guide, dash: [2, 3] },
      ...spectrumGuides(g, xDomain).map((x): Guide => ({
        axis: "x", v: x.v, label: x.label, labelSide: x.side,
        ...(x.kind === "lr" ? { color: C.guide, dash: [6, 3] } : { color: C.visfwhm, dash: [5, 2], alpha: 0.6 }),
      })),
    ];
    const inView = series.flatMap((s) => s.y.filter((_, i) => theta[i] >= xDomain[0] && theta[i] <= xDomain[1]));
    const yTop = Math.max(1, extent(inView)?.[1] ?? 1);
    const yDomain: [number, number] = isT ? [0, Math.max(1.2, Math.min(2, yTop * 1.05))] : [0, 1.05];
    const xTicks = X_TICKS.filter((t) => t.v >= xDomain[0] * 0.999 && t.v <= xDomain[1] * 1.001);
    return { series, legend, guides, xDomain, yDomain, xTicks };
  }, [e, section, pairs, colorBy, members, colors.values, allScales]); // eslint-disable-line react-hooks/exhaustive-deps

  /* σ vs error */
  const stdErr = useMemo(() => {
    const d = e?.std_err;
    if (!d) return null;
    const kinds = Object.keys(d.models ?? {}).filter((k) => !isRbf(k));
    const kind = kinds.includes(model) ? model : kinds.includes("spatial_gate") ? "spatial_gate" : kinds[0] ?? "ensemble_mean";
    const m = d.models?.[kind] ?? d;
    if (!m.hist?.length) return null;
    const edges = num(m.edges);
    const lo = edges[0], hi = edges[edges.length - 1];
    return {
      kind, kinds,
      heat: { z: m.hist, xEdges: edges, yEdges: edges } as Heat,
      series: [
        { x: [lo, hi], y: [lo, hi], color: C.guide, width: 1.3, dash: [6, 3], name: "|error| = σ" },
        { x: num(m.med_std), y: num(m.med_err), color: modelColor(kind), width: 2.6, dots: true, name: `median |error| · ${MODEL_LABEL[kind] ?? kind}` },
      ] as Series[],
      domain: [lo, hi] as [number, number], ticks: decadeTicks([lo, hi], { space: "log10" }),
      describe: (p: Pick) => `σ ${range(edges, p.i)} e⁻ · |err| ${range(edges, p.j)} e⁻`,
    };
  }, [e, model]);

  /* σ vs brightness */
  const bright = useMemo(() => {
    const d = e?.bright_std;
    if (!d?.hist?.length) return null;
    const bx = num(d.bright_edges), sy = num(d.std_edges);
    const b = num(d.bright);
    const st = d.stretch;
    return {
      heat: { z: d.hist, xEdges: bx, yEdges: sy } as Heat,
      series: [
        { x: b, y: num(d.lo), color: C.baseline, width: 1, alpha: 0.6, dash: [4, 3], name: "16th percentile" },
        { x: b, y: num(d.hi), color: C.baseline, width: 1, alpha: 0.6, dash: [4, 3], name: "84th percentile" },
        { x: b, y: num(d.med), color: C.baseline, width: 2.6, dots: true, name: "median σ" },
      ] as Series[],
      xDomain: [bx[0], bx[bx.length - 1]] as [number, number], yDomain: [sy[0], sy[sy.length - 1]] as [number, number],
      xTicks: [0, 100, 1e3, 1e4, 1e5, 1e6].map((v) => ({ v: Math.asinh(v / st), label: v === 0 ? "0" : formatE(v) })),
      yTicks: decadeTicks([sy[0], sy[sy.length - 1]], { space: "log10" }),
      describe: (p: Pick) => `HR ${formatE(st * Math.sinh(bx[p.i]))}–${formatE(st * Math.sinh(bx[p.i + 1]))} e⁻ · σ ${range(sy, p.j)} e⁻`,
    };
  }, [e]);

  /* coherence: a sorted horizontal dot plot, highest at the top */
  const coherence = useMemo(() => {
    const rows = coherenceOrder((e?.coherence?.scores ?? []).filter((r) => !isRbf(r.id) && (r.overall != null || r.sr != null)));
    if (!rows.length) return null;
    const n = rows.length;
    const y = rows.map((_, i) => n - 1 - i);
    const whiskers = (lo: (r: typeof rows[number]) => number | null | undefined, hi: (r: typeof rows[number]) => number | null | undefined, color: string, key: string, dy: number): Series[] =>
      rows.flatMap((r, i) => {
        const a = lo(r), b = hi(r);
        return a != null && b != null && Number.isFinite(a) && Number.isFinite(b)
          ? [{ x: [a, b], y: [y[i] + dy, y[i] + dy], color, width: 1.2, alpha: 0.6, key }] : [];
      });
    const series: Series[] = [
      ...whiskers((r) => r.overall_lo, (r) => r.overall_hi, C.mean, "overall", 0.12),
      ...whiskers((r) => r.sr_lo, (r) => r.sr_hi, C.comb, "sr", -0.12),
      { x: rows.map((r) => r.overall ?? NaN), y: y.map((v) => v + 0.12), mode: "scatter", color: C.mean, name: "all scales", key: "overall" },
      { x: rows.map((r) => r.sr ?? NaN), y: y.map((v) => v - 0.12), mode: "scatter", marker: "diamond", color: C.comb, name: "SR scales (θ < 0.1″)", key: "sr" },
    ];
    const xs = rows.flatMap((r) => [r.overall, r.sr, r.overall_lo, r.sr_lo, r.overall_hi, r.sr_hi]);
    const ext = extent(xs) ?? [0, 1];
    return {
      rows, series, n,
      xDomain: [Math.max(-1, ext[0] - 0.05), Math.min(1.05, Math.max(1, ext[1]) + 0.02)] as [number, number],
      ticks: rows.map((r, i) => ({ v: y[i], label: coherenceLabel(r) })),
    };
  }, [e]);

  /* spread: calibration */
  const calib = useMemo(() => {
    const c = e?.calibration;
    if (!c || !has(c.pdf)) return null;
    const edges = num(c.z_edges);
    const cen = edges.slice(0, -1).map((v, i) => 0.5 * (v + edges[i + 1]));
    const gauss = cen.map((z) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI));
    const std = num(c.field_std), rmse = num(c.field_rmse);
    const hi = Math.max(extent([...std, ...rmse])?.[1] ?? 1, 1);
    const lo = Math.max(1e-3, extent([...std, ...rmse].filter((v) => v > 0))?.[0] ?? 1e-3);
    const ratio = fieldErrorRatio(std.map((v) => (Number.isFinite(v) ? v : null)), rmse.map((v) => (Number.isFinite(v) ? v : null)));
    return {
      ratio, verdict: spreadVerdict(ratio.ratio),
      cover: coverageRows(c.stats),
      pdf: [
        { x: cen, y: num(c.pdf), color: C.mean, mode: "histogram", fillAlpha: 0.35, name: "z = (SR − target) / σ" },
        { x: cen, y: gauss, color: C.baseline, width: 2, dash: [6, 3], name: "N(0, 1)" },
      ] as Series[],
      pdfDomain: pdfYDomain(c.pdf, gauss),
      zGuides: [-3, -2, -1, 1, 2, 3].map((v): Guide => ({ axis: "x", v, color: C.guide, dash: [2, 3], alpha: 0.7 })),
      scatter: [
        { x: [lo, hi], y: [lo, hi], color: C.guide, dash: [6, 3], width: 1.3, name: "RMSE = σ" },
        { x: std, y: rmse, color: C.comb, mode: "scatter", name: `one test field (${formatCount(std.length)})` },
      ] as Series[],
      sdom: [lo / 1.2, hi * 1.2] as [number, number],
    };
  }, [e]);

  const trace = (describe: (p: Pick) => string, diag: Pick["diag"], extra: { model?: string } = {}) =>
    cell?.diag === diag ? (
      <PixelTrace pick={cell} model={extra.model} band={band} cellLabel={describe(cell)} onClose={() => setPick(null)} />
    ) : null;

  const evalSection = section !== "real-field" && section !== "recovery";
  return (
    <Page className="mdl-page">
      <Toolbar label="Diagnostics controls">
        <Segmented<Section> size="sm" aria-label="Diagnostic" value={section} onChange={(v) => { setSection(v); setPick(null); }}
          options={SECTIONS.map((s) => ({ value: s.value, label: s.label, title: s.title }))} />
        {section !== "real-field" ? (
          <Segmented<EvalBand> size="sm" aria-label="Band" value={band} onChange={(v) => { setBand(v); setPick(null); }} options={BAND_OPTIONS} />
        ) : (
          /* Same place in every section; the legacy real field has VIS diagnostics only. */
          <span title={REAL_FIELD_BANDS_TITLE}>
            <Segmented<EvalBand> size="sm" aria-label="Band" value="VIS" onChange={() => undefined}
              options={BAND_OPTIONS.map((o) => (o.value === "VIS" ? o : { ...o, disabled: true, title: REAL_FIELD_BANDS_TITLE }))} />
          </span>
        )}
        <ToolbarSpacer />
        {(section === "spectrum" || section === "transfer") && <>
          <Chip on={allScales} onClick={() => setAllScales(!allScales)} title="Show every measured scale (default: the SR scales, θ < 0.5″)">all scales</Chip>
          <ToolbarGroup label="Members"><ColorBySelect value={colorBy} onChange={setColorBy} /></ToolbarGroup>
          {section === "spectrum" && !!e?.ps?.r_pairs?.length && <Chip on={pairs} onClick={() => setPairs(!pairs)} title="Every member pair's cross-correlation">pairs</Chip>}
        </>}
        {section === "spread" && stdErr && (
          <ToolbarGroup label="Error of">
            {stdErr.kinds.map((k) => <Chip key={k} on={stdErr.kind === k} dot={modelColor(k)} onClick={() => setModel(k)}>{MODEL_LABEL[k] ?? k}</Chip>)}
          </ToolbarGroup>
        )}
      </Toolbar>
      {section === "real-field" && <FieldSection />}
      {section === "recovery" && <Recovery band={band} />}
      {evalSection && band !== "VIS" && <JobProgress job={bandJob.job} error={bandJob.error} />}
      {evalSection && band !== "VIS" && res.error?.status === 404 ? (
        <EmptyState icon="activity" title={`The ${bandLabel(band)} diagnostics are not computed yet`}
          action={<Button size="sm" variant="primary" loading={bandJob.busy} onClick={() => void computeBandEvals()}>Compute Y, J and H…</Button>}>
          They are measured from the cached test cubes of the last evaluation (no model runs). Evaluate keeps them current afterwards.
        </EmptyState>
      ) : evalSection && (
        <LoadState loading={res.loading} error={res.error} onRetry={res.reload}>
          {e && (
            <div className="mdl-stack">
              {e.stale && (
                <Callout tone="warn" title={`These ${bandLabel(band)} diagnostics belong to an earlier evaluation`}
                  action={<Button size="sm" loading={bandJob.busy} onClick={() => void computeBandEvals()}>Recompute…</Button>}>
                  The members, the test fields or the gate changed since; VIS is current.
                </Callout>
              )}
              {(section === "spectrum" || section === "transfer") && (spectrum ? (
                <div className="mdl-chart">
                  <Plot {...lg.plotProps} xScale="log" xDomain={spectrum.xDomain} yDomain={spectrum.yDomain} xTicks={spectrum.xTicks}
                    yTicks={unitTicks(spectrum.yDomain[1])}
                    xLabel="angular scale θ = 1/2k [arcsec]" yLabel={section === "transfer" ? `T(k) [${bandLabel(band)}]` : `r(k) vs HR [${bandLabel(band)}]`}
                    series={spectrum.series} guides={spectrum.guides} aspect={0.46} legend={spectrum.legend} exportName={`ensemble-${section}-${band}`}
                    xFormat={(v) => `${v.toPrecision(2)}″`} yFormat={(v) => v.toFixed(3)} aria-label={section === "transfer" ? "Transfer function" : "Cross-correlation r(k)"} />
                </div>
              ) : <EmptyState icon="activity" title="No power spectrum cached">Evaluate the ensemble (Leaderboard).</EmptyState>)}
              {section === "coherence" && (coherence ? (
                <Plot xDomain={coherence.xDomain} yDomain={[-0.7, coherence.n - 0.3]} yTicks={coherence.ticks}
                  xLabel="normalised mean r(k) over d log k" yLabel="" series={coherence.series} height={Math.max(260, 17 * coherence.n + 70)}
                  guides={[{ axis: "x", v: 1, color: C.guide, dash: [2, 3] }]} legend="auto" zoomAxes="x"
                  xFormat={(v) => v.toFixed(2)} exportName="ensemble-coherence" aria-label="Spectral coherence per model, highest first" />
              ) : <EmptyState icon="activity" title="No coherence cached">Re-evaluate to compute it.</EmptyState>)}
              {section === "spread" && (calib || stdErr || bright ? <>
                {calib?.verdict && calib.ratio.ratio != null && (
                  <SummaryLine>
                    {calib.verdict.text}: per test field, the RMSE is{" "}
                    <Num tone={calib.verdict.warn ? "warn" : undefined}>≈{Number(calib.ratio.ratio.toPrecision(2))}×</Num> the mean σ
                    {" "}(median over {calib.ratio.n} field{calib.ratio.n === 1 ? "" : "s"}).
                  </SummaryLine>
                )}
                {calib && <Table className="mdl-compare" aria-label="Coverage of |z|" columns={COVER_COLUMNS} rows={calib.cover} rowKey={(r) => r.k} />}
                <div className="mdl-charts">
                  {stdErr && (
                    <div className="mdl-chart">
                      <h3 className="mdl-chart__title">σ vs error ({MODEL_LABEL[stdErr.kind] ?? stdErr.kind})</h3>
                      <Plot xDomain={stdErr.domain} yDomain={stdErr.domain} xTicks={stdErr.ticks} yTicks={stdErr.ticks}
                        xLabel="cross-member σ per pixel [e⁻]" yLabel={`|${MODEL_LABEL[stdErr.kind] ?? stdErr.kind} − HR| [e⁻]`}
                        heat={stdErr.heat} series={stdErr.series} aspect={0.8} legend="auto"
                        onHeatClick={(c) => setPick({ diag: "std_err", ...c })} highlight={cell?.diag === "std_err" ? cell : null}
                        exportName="ensemble-std-error" aria-label="Disagreement vs error" />
                    </div>
                  )}
                  {bright && (
                    <div className="mdl-chart">
                      <h3 className="mdl-chart__title">σ vs brightness</h3>
                      <Plot xDomain={bright.xDomain} yDomain={bright.yDomain} xTicks={bright.xTicks} yTicks={bright.yTicks}
                        xLabel="HR brightness [e⁻] (asinh axis)" yLabel="cross-member σ [e⁻]"
                        heat={bright.heat} series={bright.series} aspect={0.8} legend="auto"
                        onHeatClick={(c) => setPick({ diag: "bright_std", ...c })} highlight={cell?.diag === "bright_std" ? cell : null}
                        exportName="ensemble-std-brightness" aria-label="Disagreement vs brightness" />
                    </div>
                  )}
                </div>
                {stdErr && trace(stdErr.describe, "std_err", { model: stdErr.kind })}
                {bright && trace(bright.describe, "bright_std")}
                {!cell && (stdErr || bright) && <Caption>Click a heat-map cell to see the real pixels that landed in it.</Caption>}
                {calib && (
                  <div className="mdl-charts">
                    <div className="mdl-chart">
                      <h3 className="mdl-chart__title">z-score distribution</h3>
                      <Plot xDomain={[-6, 6]} yDomain={calib.pdfDomain} xLabel="z" yLabel="pdf" series={calib.pdf} guides={calib.zGuides}
                        aspect={0.62} legend="auto" exportName="ensemble-z-pdf" aria-label="z-score distribution" />
                      <Caption>Dashed lines: |z| = 1, 2, 3 (the coverage table above).</Caption>
                    </div>
                    <div className="mdl-chart">
                      <h3 className="mdl-chart__title">Per field: mean σ vs RMSE</h3>
                      <Plot xScale="log" yScale="log" xDomain={calib.sdom} yDomain={calib.sdom}
                        xTicks={logTicks(calib.sdom)} yTicks={logTicks(calib.sdom)}
                        xLabel="mean cross-member σ [e⁻]" yLabel="RMSE vs HR [e⁻]" series={calib.scatter} aspect={0.62}
                        legend="auto" exportName="ensemble-sigma-rmse" aria-label="Per-field sigma vs RMSE" />
                    </div>
                  </div>
                )}
              </> : <EmptyState icon="activity" title="No spread diagnostics cached">Evaluate the ensemble (Leaderboard).</EmptyState>)}
              <Caption>{e.n_fields ?? 0} {e.subset ?? "test"} fields · {e.n_members ?? 0} members · {bandLabel(e.band ?? band)} band</Caption>
            </div>
          )}
        </LoadState>
      )}
    </Page>
  );
}

/* ── recovery: SR → HR on the synthetic stamps ─────────────────────────── */
const GROUP_COLOR: Record<string, number> = { "syn-lens": 2, "syn-gal": 0 };

function Recovery({ band }: { band: EvalBand }) {
  const runs = useEvalRuns();
  const sets = useMemo(() => stampSets(runs.data?.rows ?? []), [runs.data]);
  const plot = useMemo(() => {
    const pts = sets.flatMap((s) => s.points);
    if (!pts.length) return null;
    const ext = extent(pts.flatMap((p) => [p.x, p.y])) ?? [0, 1];
    const pad = (ext[1] - ext[0]) * 0.05 || 1;
    const dom: [number, number] = [ext[0] - pad, ext[1] + pad];
    const series: Series[] = [
      { x: dom, y: dom, color: C.guide, width: 1.3, dash: [6, 3], name: "no change" },
      ...sets.map((s): Series => ({
        x: s.points.map((p) => p.x), y: s.points.map((p) => p.y), mode: "scatter", color: categorical(GROUP_COLOR[s.grade] ?? 4),
        name: `${s.label} ${s.points.length}`,
      })),
    ];
    const flux: Series[] = sets.map((s, gi): Series => {
      const f = s.points.filter((p) => p.flux != null);
      return {
        x: f.map((_, i) => gi + ((i * 0.618) % 1 - 0.5) * 0.5), y: f.map((p) => p.flux as number), mode: "scatter",
        color: categorical(GROUP_COLOR[s.grade] ?? 4), name: `${s.label} ${f.length}`,
      };
    });
    // The axis spans the central 96% of the ratios (a few stamps with a
    // near-zero LR flux reach −4 or +5), never below 0; the caption counts the rest.
    const ratios = flux.flatMap((s) => s.y.filter((v): v is number => v != null && Number.isFinite(v))).sort((a, b) => a - b);
    const q = (p: number) => ratios[Math.min(ratios.length - 1, Math.max(0, Math.round(p * (ratios.length - 1))))];
    const fluxDom: [number, number] = ratios.length ? [Math.max(0, Math.min(0.8, q(0.02) - 0.05)), Math.max(1.2, q(0.98) + 0.05)] : [0.5, 1.5];
    const outside = ratios.filter((v) => v < fluxDom[0] || v > fluxDom[1]).length;
    return { series, dom, flux, fluxDom, outside };
  }, [sets]);
  const imagesPath = `${pagePath("models", { tab: "images" })}?set=stamps`;
  const total = sets.reduce((n, s) => n + s.points.length, 0);
  const improved = sets.reduce((n, s) => n + s.improved, 0);
  return (
    <div className="mdl-stack">
      {runs.error ? (
        <EmptyState icon="warn" title="The catalogue evaluation is not readable" action={<Button size="sm" onClick={() => void runs.reload()}>Retry</Button>}>
          <span className="mdl-mono">{runs.error.message}</span>
        </EmptyState>
      ) : runs.loading && !runs.data ? <EmptyState compact icon="activity" title="Reading the synthetic stamps…" /> : !plot ? (
        <EmptyState icon="activity" title="No synthetic stamps with HR truth">
          The recovery needs the source-centred synthetic lenses and galaxies of the grouped analysis (Sky › Targets › Sources).
        </EmptyState>
      ) : <>
        <SummaryLine>
          SR is closer to the HR truth than LR on <Num>{formatCount(improved)}</Num> of <Num>{formatCount(total)}</Num> synthetic stamps
          {sets.map((s) => <span key={s.grade}>; {s.label} median <Num>{db(s.medianLr)}</Num> → <Num>{db(s.medianSr)}</Num> dB</span>)}.
        </SummaryLine>
        <div className="mdl-charts">
          <div className="mdl-chart">
            <h3 className="mdl-chart__title">SR → HR recovery</h3>
            <Plot xDomain={plot.dom} yDomain={plot.dom} xLabel="PSNR, LR vs HR [dB]" yLabel="PSNR, SR vs HR [dB]" series={plot.series}
              legend="auto" aspect={0.8} xFormat={(v) => v.toFixed(1)} yFormat={(v) => v.toFixed(1)}
              exportName="sr-hr-recovery" aria-label="SR vs HR recovery of the synthetic stamps" />
            <Caption>Above the dashed line SR is closer to the truth than LR.</Caption>
          </div>
          <div className="mdl-chart">
            <h3 className="mdl-chart__title">Flux conservation</h3>
            <Plot xDomain={[-0.6, sets.length - 0.4]} yDomain={plot.fluxDom} xTicks={sets.map((s, i) => ({ v: i, label: s.label }))}
              xLabel="" yLabel="total flux SR / LR (1 = conserved)" series={plot.flux} guides={[{ axis: "y", v: 1, color: C.guide, dash: [2, 3] }]}
              legend="auto" aspect={0.8} yFormat={(v) => v.toFixed(2)} exportName="sr-flux-ratio" aria-label="SR over LR total flux per stamp" />
            {plot.outside > 0 && <Caption>{formatCount(plot.outside)} stamp{plot.outside === 1 ? " lies" : "s lie"} outside the axis (a ratio below {Number(plot.fluxDom[0].toFixed(2))} or above {Number(plot.fluxDom[1].toFixed(2))}; all are in the CSV).</Caption>}
          </div>
        </div>
        <Caption>One point per stamp; the stamps themselves are in <Link to={imagesPath}>Images › Stamps</Link>.</Caption>
      </>}
      <AngularSpectrum band={band} />
    </div>
  );
}

/* ── recovery: the angular power spectrum, HR vs SR per band ───────────── */
type Space = "asinh" | "linear";
const SPACES: { value: Space; label: string; title: string }[] = [
  { value: "asinh", label: "asinh", title: "Measured on asinh-stretched images (bright stars compressed)" },
  { value: "linear", label: "linear", title: "Measured on the images in electrons" },
];

function AngularSpectrum({ band }: { band: EvalBand }) {
  const aps = useResource<Aps>(url.recoverySpectrum(), [], { ttl: 5 * 60_000 });
  const [space, setSpace] = useUrlState<Space>("space", "asinh");
  const [allBands, setAllBands] = useUrlState("bands", false);
  const [measuring, setMeasuring] = useState(false);
  const lg = useLegend();
  const measure = async () => {
    const ok = await confirm({
      title: "Measure the angular power spectrum?",
      message: "Compares HR with the production SR of the synced validation records, per band (a few seconds to minutes; nothing is trained or re-run).",
      confirmLabel: "Measure",
    });
    if (!ok) return;
    setMeasuring(true);
    try {
      await apiPost(url.recoveryFigure());
      await aps.reload();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err));
    } finally {
      setMeasuring(false);
    }
  };
  const a = aps.data;
  const charts = useMemo(() => {
    if (!a) return null;
    const entry = a.bands[band];
    const curves = entry?.[space];
    if (!curves || !has(curves.theta)) return null;
    const theta = num(curves.theta);
    const lo = a.pixel_scale ?? 0.05, hi = a.theta_max ?? 5;
    const xDomain: [number, number] = [lo, hi];
    const guidesFor = (): Guide[] => [
      { axis: "y", v: 1, color: C.guide, dash: [2, 3] },
      ...spectrumGuides({ lr_scale: a.lr_scale, psf_fwhm: entry?.psf_fwhm, band }, xDomain).map((x): Guide => ({
        axis: "x", v: x.v, label: x.label, labelSide: x.side,
        ...(x.kind === "lr" ? { color: C.guide, dash: [6, 3] } : { color: bandColor(band), dash: [5, 2], alpha: 0.6 }),
      })),
    ];
    const make = (key: "T" | "r"): Series[] => {
      const out: Series[] = [];
      if (allBands) {
        for (const b of a.band_names ?? Object.keys(a.bands)) {
          const c = a.bands[b]?.[space];
          if (!c || !has(c[key])) continue;
          out.push({ x: num(c.theta), y: num(c[key]), color: bandColor(b), width: b === band ? 2.6 : 1.6, dots: b === band,
            alpha: b === band ? 1 : 0.8, name: bandLabel(b) });
        }
        return out;
      }
      out.push({ x: theta, y: num(curves[key]), low: num(curves[`${key}_lo`]), high: num(curves[`${key}_hi`]),
        color: bandColor(band), width: 2.6, dots: true, fillAlpha: 0.16, name: `${bandLabel(band)}, median of ${formatCount(a.n_fields)} fields` });
      return out;
    };
    const xTicks = X_TICKS.filter((t) => t.v >= lo * 0.999 && t.v <= hi * 1.001);
    return { T: make("T"), r: make("r"), guides: guidesFor(), xDomain, xTicks };
  }, [a, band, space, allBands]);
  return (
    <section className="mdl-stack mdl-stack--tight" aria-labelledby="mdl-aps-title">
      <div className="mdl-row">
        <h2 id="mdl-aps-title" className="mdl-h2">Angular power spectrum</h2>
        <span className="mdl-muted">HR vs the production SR of the synced validation records</span>
        <span className="mdl-grow" />
        {a && <>
          <Segmented<Space> size="sm" aria-label="Measured on" value={space} onChange={setSpace} options={SPACES} />
          <Chip on={allBands} onClick={() => setAllBands(!allBands)} title="Every band's median on the same axes">all bands</Chip>
          <Button size="sm" variant="ghost" icon="download" href={url.recoveryFigure()} download="angular_power_spectrum.png">PNG</Button>
        </>}
        <Button size="sm" icon="reset" loading={measuring} onClick={() => void measure()}>{a ? "Re-measure…" : "Measure…"}</Button>
      </div>
      {aps.loading && !a ? <EmptyState compact icon="activity" title="Reading the angular power spectrum…" />
        : aps.error && aps.error.status !== 404 ? (
          <EmptyState compact icon="warn" title="The angular power spectrum is not readable" action={<Button size="sm" onClick={() => void aps.reload()}>Retry</Button>}>
            <span className="mdl-mono">{aps.error.message}</span>
          </EmptyState>
        ) : !a ? (
          <EmptyState compact icon="activity" title="Not measured yet">
            It needs the synced validation records and their production SR (Images › Generate SR over local records); then Measure.
          </EmptyState>
        ) : !charts ? (
          <EmptyState compact icon="activity" title={`No ${bandLabel(band)} curves in this measurement`}>Re-measure to include every band.</EmptyState>
        ) : <>
          <div className="mdl-charts">
            <div className="mdl-chart">
              <h3 className="mdl-chart__title">Transfer T(k) = √(P_SR / P_HR)</h3>
              <Plot {...lg.plotProps} xScale="log" xDomain={charts.xDomain} yDomain={[0, 1.45]} xTicks={charts.xTicks} yTicks={unitTicks(1.45)}
                xLabel="angular scale θ = 1/2k [arcsec]" yLabel={`T(k) [${space}]`} series={charts.T} guides={charts.guides}
                aspect={0.62} legend="auto" xFormat={(v) => `${v.toPrecision(2)}″`} yFormat={(v) => v.toFixed(3)}
                exportName={`aps-transfer-${band}-${space}`} aria-label={`Transfer function, ${bandLabel(band)}`} />
            </div>
            <div className="mdl-chart">
              <h3 className="mdl-chart__title">Cross-correlation r(k)</h3>
              <Plot {...lg.plotProps} xScale="log" xDomain={charts.xDomain} yDomain={[0, 1.05]} xTicks={charts.xTicks} yTicks={unitTicks(1.05)}
                xLabel="angular scale θ = 1/2k [arcsec]" yLabel={`r(k) [${space}]`} series={charts.r} guides={charts.guides}
                aspect={0.62} legend="auto" xFormat={(v) => `${v.toPrecision(2)}″`} yFormat={(v) => v.toFixed(3)}
                exportName={`aps-r-${band}-${space}`} aria-label={`Cross-correlation, ${bandLabel(band)}`} />
            </div>
          </div>
          <Caption>
            {formatCount(a.n_fields)} {a.subset ?? "validation"} fields{a.field_n ? ` of ${a.field_n}×${a.field_n} px at 0.05″` : ""}
            {allBands ? " · each band's per-field median" : " · per-field median, shaded 16–84% of fields"} · finer scales to the left.
          </Caption>
        </>}
    </section>
  );
}
