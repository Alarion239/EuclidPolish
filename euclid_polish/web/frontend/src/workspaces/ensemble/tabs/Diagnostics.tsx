/* ensemble/diagnostics (spec §8.2), from GET /ensemble/evals.json: the VIS
   power spectrum (cross-correlation r(k) and transfer function T(k)),
   spectral coherence, std-vs-error, combiner axes vs error, std-vs-
   brightness and the calibration (z-pdf, coverage, per-field σ vs RMSE —
   computed by every evaluation, shown here for the first time). A click on
   a heat cell back-traces it to real image stamps. State in the URL. */
import { useMemo } from "react";
import Plot, { useLegend, type Guide, type Heat, type LegendItem, type Series } from "../../../charts/Plot";
import { C, categorical } from "../../../colors";
import { useResource } from "../../../api/query";
import { usePageActions } from "../../../app/palette";
import { useUrlState } from "../../../hooks/useUrlState";
import { decadeTicks, logTicks } from "../../../ticks";
import { Chip, EmptyState, Kpi, Page, Segmented } from "../../../ui";
import { url, useMode, type Evals, type NumArr } from "../api";
import { BarGroup, ColorBySelect, EnsBar, LoadState, useFacetColors } from "../common";
import { facetOf, memberNumber, type ColorBy } from "../model";
import { PixelTrace, type Pick } from "../PixelTrace";
import "../ensemble.css";

type Section = "spectrum" | "transfer" | "coherence" | "stderr" | "axes" | "brightness" | "calibration";
const SECTIONS: { value: Section; label: string; title: string }[] = [
  { value: "spectrum", label: "r(k)", title: "Cross-correlation with the target per angular scale" },
  { value: "transfer", label: "T(k)", title: "Transfer function √(P_SR / P_target)" },
  { value: "coherence", label: "Coherence", title: "One-number spectral coherence per model" },
  { value: "stderr", label: "σ vs error", title: "Does member disagreement predict the error?" },
  { value: "axes", label: "Combiner axes", title: "Error on the combiner's input coordinates" },
  { value: "brightness", label: "σ vs brightness", title: "Where does disagreement live?" },
  { value: "calibration", label: "Calibration", title: "Is σ a calibrated error bar?" },
];
const MODEL_LABEL: Record<string, string> = {
  ensemble_mean: "plain mean", spatial_gate: "production gate",
  raw_incremental_minmeanmax_rbf: "RBF", raw_incremental_frozen_minmeanmax_rbf: "frozen RBF",
};
const modelColor = (kind: string) => (kind === "ensemble_mean" ? C.mean : kind === "spatial_gate" ? C.comb : categorical(5));
const num = (a: NumArr | undefined | null) => (a ?? []).map((v) => (v == null ? NaN : v));
const has = (a: NumArr | undefined | null) => (a ?? []).some((v) => v != null && Number.isFinite(v));
const fmtE = (v: number) => (!Number.isFinite(v) ? "—" : Math.abs(v) >= 1000 || (Math.abs(v) > 0 && Math.abs(v) < 0.01)
  ? v.toExponential(1) : String(Number(v.toPrecision(3))));
const range = (edges: number[], k: number) => `${fmtE(10 ** edges[k])}–${fmtE(10 ** edges[k + 1])}`;
const X_TICKS = [0.05, 0.1, 0.2, 0.5, 1, 2, 5].map((v) => ({ v, label: String(v) }));

function parseCell(raw: string): Pick | undefined {
  const [diag, i, j] = raw.split(",");
  if (!["std_err", "bright_std", "combiner_feature_error"].includes(diag)) return undefined;
  const ii = Number(i), jj = Number(j);
  return Number.isInteger(ii) && Number.isInteger(jj) ? { diag: diag as Pick["diag"], i: ii, j: jj } : undefined;
}

export default function Diagnostics() {
  const mode = useMode();
  const res = useResource<Evals>(url.evals(mode), [mode], { ttl: 5 * 60_000 });
  const [section, setSection] = useUrlState<Section>("d", "spectrum");
  const [colorBy, setColorBy] = useUrlState<ColorBy>("color", "uniform");
  const [model, setModel] = useUrlState("model", "spatial_gate");
  const [axis, setAxis] = useUrlState("axis", "mean_std");
  const [pairs, setPairs] = useUrlState("pairs", false);
  const [cellRaw, setCellRaw] = useUrlState("cell", "");
  const cell = parseCell(cellRaw) ?? null;
  const setPick = (p: Pick | null) => setCellRaw(p ? `${p.diag},${p.i},${p.j}` : "");
  const lg = useLegend();
  const e = res.data;
  const targetLabel = mode === "starless" ? "clean target" : "HR";
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
    const xDomain: [number, number] = [g.theta_min ?? 0.05, Math.max(...theta.filter(Number.isFinite))];
    const isT = section === "transfer";
    const series: Series[] = [];
    // Explicit legend: ONE entry per member facet (key members / members:<facet>,
    // toggling every member line of it) while each line keeps its own name for
    // hover identification; the other curves list themselves.
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
    if (memberRows.length) {                 // facet entries in facet order (knees numerically)
      if (colorBy === "uniform") legend.push({ label: "members", key: "members", color: C.muted });
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
      const y = isT ? c.T : c.r;
      if (has(y)) push({ x: theta, y: num(y), color: modelColor(kind), width: 2.4, dots: true, name: MODEL_LABEL[kind] ?? kind });
    }
    const guides: Guide[] = [
      { axis: "y", v: 1, color: C.guide, dash: [2, 3] },
      { axis: "x", v: g.lr_scale ?? 0.1, color: C.guide, dash: [6, 3], label: "LR pixel" },
      { axis: "x", v: g.vis_fwhm ?? 0.16, color: C.visfwhm, dash: [5, 2], alpha: 0.6, label: "VIS FWHM" },
    ];
    const ys = series.flatMap((s) => s.y.filter((v): v is number => v != null && Number.isFinite(v)));
    const yDomain: [number, number] = isT ? [0, Math.max(1.2, Math.min(2, Math.max(...ys, 1) * 1.05))] : [0, 1.05];
    return { series, legend, guides, xDomain, yDomain };
  }, [e, section, pairs, colorBy, members, colors.values]); // eslint-disable-line react-hooks/exhaustive-deps

  /* std vs error */
  const stdErr = useMemo(() => {
    const d = e?.std_err;
    if (!d) return null;
    const kinds = Object.keys(d.models ?? {});
    const kind = d.models?.[model] ? model : kinds.includes("spatial_gate") ? "spatial_gate" : kinds[0] ?? "ensemble_mean";
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

  /* combiner axes vs error */
  const axes = useMemo(() => {
    const d = e?.combiner_feature_error;
    const a = d?.axes?.[axis] ?? Object.values(d?.axes ?? {})[0];
    if (!d || !a) return null;
    const kinds = Object.keys(a.models ?? {});
    const kind = a.models?.[model] ? model : kinds.includes("spatial_gate") ? "spatial_gate" : kinds[0];
    const m = kind ? a.models[kind] : null;
    if (!m) return null;
    const xEdges = num(a.edges[0]), yEdges = num(a.edges[1]);
    const z = m.median_log_error.map(num);
    const [zlo, zhi] = d.color_range?.length === 2 ? d.color_range : [-3, 3];
    return {
      kind, kinds, axisNames: a.axis_names,
      heat: { z, xEdges, yEdges, scale: "linear", min: zlo, max: zhi, colorLabel: "median |error| [e⁻]",
        colorTicks: decadeTicks([zlo, zhi], { space: "log10" }) } as Heat,
      xDomain: [xEdges[0], xEdges[xEdges.length - 1]] as [number, number],
      yDomain: [yEdges[0], yEdges[yEdges.length - 1]] as [number, number],
      describe: (p: Pick) => `${a.axis_names[0]} ${xEdges[p.i].toFixed(2)}–${xEdges[p.i + 1].toFixed(2)} · ${a.axis_names[1]} ${yEdges[p.j].toFixed(2)}–${yEdges[p.j + 1].toFixed(2)} · median |err| ${fmtE(10 ** (z[p.i]?.[p.j] ?? NaN))} e⁻`,
    };
  }, [e, axis, model]);

  /* std vs brightness */
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
      xTicks: [0, 100, 1e3, 1e4, 1e5, 1e6].map((v) => ({ v: Math.asinh(v / st), label: v === 0 ? "0" : fmtE(v) })),
      yTicks: decadeTicks([sy[0], sy[sy.length - 1]], { space: "log10" }),
      describe: (p: Pick) => `${targetLabel} ${fmtE(st * Math.sinh(bx[p.i]))}–${fmtE(st * Math.sinh(bx[p.i + 1]))} e⁻ · σ ${range(sy, p.j)} e⁻`,
    };
  }, [e, targetLabel]);

  /* coherence */
  const coherence = useMemo(() => {
    const rows = (e?.coherence?.scores ?? []).filter((r) => r.overall != null || r.sr != null);
    if (!rows.length) return null;
    const short = (r: { id: string; label: string }) => (r.id === "ensemble_mean" ? "mean" : r.id === "lr_baseline" ? "LR"
      : r.id === "spatial_gate_combiner" ? "gate" : r.id === "model_agreement" ? "agree" : r.id.startsWith("member_") ? `#${memberNumber(r.label) ?? r.label}` : r.id.replace(/_combiner$/, "").slice(0, 8));
    const series: Series[] = [
      { x: rows.map((_, i) => i), y: rows.map((r) => r.overall), errorLow: rows.map((r) => r.overall_lo ?? null), errorHigh: rows.map((r) => r.overall_hi ?? null),
        mode: "scatter", color: C.mean, name: "overall (all scales)" },
      { x: rows.map((_, i) => i), y: rows.map((r) => r.sr), errorLow: rows.map((r) => r.sr_lo ?? null), errorHigh: rows.map((r) => r.sr_hi ?? null),
        mode: "scatter", marker: "diamond", color: C.comb, name: "super-resolution scales" },
    ];
    return { rows, series, ticks: rows.map((r, i) => ({ v: i, label: short(r) })) };
  }, [e]);

  /* calibration */
  const calib = useMemo(() => {
    const c = e?.calibration;
    if (!c || !has(c.pdf)) return null;
    const edges = num(c.z_edges);
    const cen = edges.slice(0, -1).map((v, i) => 0.5 * (v + edges[i + 1]));
    const gauss = cen.map((z) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI));
    const std = num(c.field_std), rmse = num(c.field_rmse);
    const hi = Math.max(...std.filter(Number.isFinite), ...rmse.filter(Number.isFinite), 1);
    const lo = Math.max(1e-3, Math.min(...std.filter((v) => v > 0), ...rmse.filter((v) => v > 0)));
    return {
      stats: c.stats,
      pdf: [
        { x: cen, y: num(c.pdf), color: C.mean, mode: "histogram", fillAlpha: 0.35, name: "z = (SR − target) / σ" },
        { x: cen, y: gauss, color: C.baseline, width: 2, dash: [6, 3], name: "N(0, 1)" },
      ] as Series[],
      scatter: [
        { x: [lo, hi], y: [lo, hi], color: C.guide, dash: [6, 3], width: 1.3, name: "RMSE = σ" },
        { x: std, y: rmse, color: C.comb, mode: "scatter", name: "one test field" },
      ] as Series[],
      sdom: [lo / 1.2, hi * 1.2] as [number, number],
    };
  }, [e]);

  const modelChips = (kinds: string[]) => (
    <BarGroup label="Error of">
      {kinds.map((k) => <Chip key={k} on={model === k} dot={modelColor(k)} onClick={() => setModel(k)}>{MODEL_LABEL[k] ?? k}</Chip>)}
    </BarGroup>
  );
  const trace = (describe: (p: Pick) => string, diag: Pick["diag"], extra: { model?: string; axis?: string } = {}) =>
    cell?.diag === diag ? (
      <PixelTrace mode={mode} pick={cell} model={extra.model} axis={extra.axis} cellLabel={describe(cell)}
        targetLabel={targetLabel} onClose={() => setPick(null)} />
    ) : <span className="ens-faint">Click a cell to back-trace it to real image stamps.</span>;

  return (
    <Page>
      <EnsBar label="Diagnostics controls">
        <Segmented<Section> size="sm" aria-label="Diagnostic" value={section} onChange={(v) => { setSection(v); setPick(null); }}
          options={SECTIONS.map((s) => ({ value: s.value, label: s.label, title: s.title }))} />
        <span className="ens-bar__spacer" />
        {(section === "spectrum" || section === "transfer") && <>
          <BarGroup label="Members"><ColorBySelect value={colorBy} onChange={setColorBy} /></BarGroup>
          {section === "spectrum" && <Chip on={pairs} onClick={() => setPairs(!pairs)} title="Every member pair's cross-correlation">pairs</Chip>}
        </>}
        {section === "stderr" && stdErr && modelChips(stdErr.kinds)}
        {section === "axes" && axes && <>
          <Segmented size="sm" aria-label="Combiner axes" value={axis} onChange={setAxis}
            options={[{ value: "mean_std", label: "mean–std" }, { value: "min_max", label: "min–max" }]} />
          {modelChips(axes.kinds)}
        </>}
      </EnsBar>
      <LoadState loading={res.loading} error={res.error} onRetry={res.reload}>
        {e && (
          <div className="ens-stack">
            <span className="ens-faint">{e.n_fields ?? 0} {e.subset ?? "test"} fields · {e.n_members ?? 0} members · VIS band</span>
            {(section === "spectrum" || section === "transfer") && (spectrum ? (
              <div className="ens-chart">
                <Plot {...lg.plotProps} xScale="log" xDomain={spectrum.xDomain} yDomain={spectrum.yDomain} xTicks={X_TICKS}
                  xLabel="angular scale θ = 1/2k [arcsec]" yLabel={section === "transfer" ? "T(k) [VIS]" : `r(k) vs ${targetLabel} [VIS]`}
                  series={spectrum.series} guides={spectrum.guides} aspect={0.46} legend={spectrum.legend} exportName={`ensemble-${section}-${mode}`}
                  xFormat={(v) => `${v.toPrecision(2)}″`} yFormat={(v) => v.toFixed(3)} aria-label={section === "transfer" ? "Transfer function" : "Cross-correlation r(k)"} />
              </div>
            ) : <EmptyState icon="activity" title="No power spectrum cached">Evaluate the ensemble (Overview).</EmptyState>)}
            {section === "coherence" && (coherence ? (
              <Plot xDomain={[-0.5, coherence.rows.length - 0.5]} yDomain={[-1, 1.05]} xTicks={coherence.ticks}
                yTicks={[{ v: -1, label: "−1" }, { v: 0, label: "0" }, { v: 0.5, label: "0.5" }, { v: 1, label: "1" }]}
                xLabel="model" yLabel="mean r(k) over d log k" series={coherence.series}
                guides={[{ axis: "y", v: 1, color: C.guide, dash: [2, 3] }]} aspect={0.42} legend="auto" zoomAxes="x"
                exportName={`ensemble-coherence-${mode}`} aria-label="Spectral coherence per model" />
            ) : <EmptyState icon="activity" title="No coherence cached">Re-evaluate to compute it.</EmptyState>)}
            {section === "stderr" && (stdErr ? <>
              <Plot xDomain={stdErr.domain} yDomain={stdErr.domain} xTicks={stdErr.ticks} yTicks={stdErr.ticks}
                xLabel="cross-member σ per pixel [e⁻]" yLabel={`|${MODEL_LABEL[stdErr.kind] ?? stdErr.kind} − ${targetLabel}| [e⁻]`}
                heat={stdErr.heat} series={stdErr.series} aspect={0.62} legend="auto"
                onHeatClick={(c) => setPick({ diag: "std_err", ...c })} highlight={cell?.diag === "std_err" ? cell : null}
                exportName={`ensemble-std-error-${mode}`} aria-label="Disagreement vs error" />
              {trace(stdErr.describe, "std_err", { model: stdErr.kind })}
            </> : <EmptyState icon="activity" title="No σ-vs-error diagnostic cached" />)}
            {section === "axes" && (axes ? <>
              <Plot xDomain={axes.xDomain} yDomain={axes.yDomain} xLabel={`${axes.axisNames[0]} [asinh]`} yLabel={`${axes.axisNames[1]} [asinh]`}
                heat={axes.heat} series={[]} aspect={0.62}
                onHeatClick={(c) => setPick({ diag: "combiner_feature_error", ...c })}
                highlight={cell?.diag === "combiner_feature_error" ? cell : null}
                exportName={`ensemble-combiner-axes-${mode}`} aria-label="Combiner axes vs error" />
              {trace(axes.describe, "combiner_feature_error", { model: axes.kind, axis })}
            </> : <EmptyState icon="activity" title="No combiner-axes diagnostic cached" />)}
            {section === "brightness" && (bright ? <>
              <Plot xDomain={bright.xDomain} yDomain={bright.yDomain} xTicks={bright.xTicks} yTicks={bright.yTicks}
                xLabel={`${targetLabel} brightness [e⁻] (asinh axis)`} yLabel="cross-member σ [e⁻]"
                heat={bright.heat} series={bright.series} aspect={0.62} legend="auto"
                onHeatClick={(c) => setPick({ diag: "bright_std", ...c })} highlight={cell?.diag === "bright_std" ? cell : null}
                exportName={`ensemble-std-brightness-${mode}`} aria-label="Disagreement vs brightness" />
              {trace(bright.describe, "bright_std")}
            </> : <EmptyState icon="activity" title="No σ-vs-brightness diagnostic cached" />)}
            {section === "calibration" && (calib ? (
              <div className="ens-stack">
                <div className="ens-kpis">
                  <Kpi label="σ(z)" value={calib.stats.sigma_z?.toFixed(2) ?? "—"} footer="1.00 when calibrated"
                    tone={calib.stats.sigma_z != null && Math.abs(calib.stats.sigma_z - 1) > 0.25 ? "warn" : undefined}
                    hint="Width of the z = (SR − target)/σ distribution: > 1 means σ under-states the error." />
                  <Kpi label="|z| < 1" value={calib.stats.cover1 != null ? `${(100 * calib.stats.cover1).toFixed(1)}%` : "—"} footer="68.3% expected" />
                  <Kpi label="|z| < 2" value={calib.stats.cover2 != null ? `${(100 * calib.stats.cover2).toFixed(1)}%` : "—"} footer="95.4% expected" />
                  <Kpi label="|z| < 3" value={calib.stats.cover3 != null ? `${(100 * calib.stats.cover3).toFixed(1)}%` : "—"} footer="99.7% expected" />
                </div>
                <div className="ens-charts">
                  <div className="ens-chart">
                    <h3 className="ens-chart__title">z-score distribution</h3>
                    <Plot xDomain={[-6, 6]} yDomain={[0, 0.6]} xLabel="z" yLabel="pdf" series={calib.pdf} aspect={0.62}
                      legend="auto" exportName={`ensemble-z-pdf-${mode}`} aria-label="z-score distribution" />
                  </div>
                  <div className="ens-chart">
                    <h3 className="ens-chart__title">Per field: mean σ vs RMSE</h3>
                    <Plot xScale="log" yScale="log" xDomain={calib.sdom} yDomain={calib.sdom}
                      xTicks={logTicks(calib.sdom)} yTicks={logTicks(calib.sdom)}
                      xLabel="mean cross-member σ [e⁻]" yLabel={`RMSE vs ${targetLabel} [e⁻]`} series={calib.scatter} aspect={0.62}
                      legend="auto" exportName={`ensemble-sigma-rmse-${mode}`} aria-label="Per-field sigma vs RMSE" />
                  </div>
                </div>
              </div>
            ) : <EmptyState icon="activity" title="No calibration cached">Evaluate the ensemble (Overview).</EmptyState>)}
          </div>
        )}
      </LoadState>
    </Page>
  );
}
