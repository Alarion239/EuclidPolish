/* Models › Knees › Pixel types, drawn natively (the data: ../pixelViews.ts).
   One renderer per figure of an analysis — the knee map (density, per-decade
   quantiles, chain-rule baseline), the brightness-controlled groups (median
   stamps on canvas, with the alignment-free radial occupancy), the group
   measures, the knee types, the type mix per decade, the look-around and the
   chain-rule density — and per figure of a comparison or a model series:
   the curves, the knee-share difference, the look-around and the same
   pixels' stamps. Stamps keep the PNGs' orientation: each is turned so the
   scene's brighter side is up (then right), so the bright light sits above
   and right of the centre by construction; distances and the radial
   profiles do not depend on that turn. */
import { useEffect, useMemo, useRef } from "react";
import Plot, { type Guide, type Heat, type Series } from "../../../charts/Plot";
import { C, categorical, viridis } from "../../../colors";
import { formatCount } from "../../../format";
import { useResolvedTheme } from "../../../state/prefs";
import { linearTicks, logTicks } from "../../../ticks";
import { Caption, EmptyState } from "../../../ui";
import { BAND_LABEL, KNEE_BANDS, KNEE_RAMP, kneeLabel, kneeRamp, modelShort, type FieldSet, type KneeBand } from "../knees";
import {
  CIVIDIS, decodeStamp, lut, MAGMA, normText, quantSeries, sigmaText, stampRgba,
  type AnalysisBandView, type AnalysisView, type Basis, type CompareView, type PairsView, type StampCell, type StampsView,
} from "../pixelViews";

const MAGMA_LUT = lut(MAGMA);
const CIVIDIS_LUT = lut(CIVIDIS);
const GROUP_LABEL: Record<string, string> = { lower: "Lower knee", usual: "Usual", higher: "Higher knee" };
/** The groups' colours, read from the theme at draw time. */
const groupColor = (g: string) => (g === "lower" ? categorical(0) : g === "higher" ? categorical(6) : C.muted);
/** The density's colour: grey with the count (theme-neutral). */
const DENSITY = (t: number) => `rgba(110, 118, 135, ${(0.12 + 0.78 * t).toFixed(3)})`;
const X_DOMAIN: [number, number] = [-3, 5];
const X_TICKS = logTicks(X_DOMAIN, { space: "log10" }).map((t) => ({ ...t, label: `${t.label}σ` }));
const BRIGHT_LABEL = "pixel brightness / σ";

/* ── stamps ────────────────────────────────────────────────────────────── */

function Stamp({ b64, side, map = "magma", title }: { b64: string; side: number; map?: "magma" | "cividis"; title: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const ctx = ref.current?.getContext("2d");
    if (!ctx) return;
    const bytes = decodeStamp(b64);
    if (bytes.length !== side * side) return;
    ctx.putImageData(new ImageData(stampRgba(bytes, map === "magma" ? MAGMA_LUT : CIVIDIS_LUT), side, side), 0, 0);
  }, [b64, side, map]);
  return (
    <span className="kn-stamp" title={title}>
      <canvas ref={ref} width={side} height={side} role="img" aria-label={title} />
      <span className="kn-stamp__cross" aria-hidden />
    </span>
  );
}

/** The radial occupancy of a row's cells (alignment-free). */
function RadialPlot({ cells, order, label }: { cells: Record<string, { radial: (number | null)[] }>; order: string[]; label: string }) {
  const series: Series[] = order.filter((g) => cells[g]).map((g) => ({
    x: cells[g].radial.map((_, i) => i), y: cells[g].radial, color: groupColor(g), width: 1.6,
    name: GROUP_LABEL[g] ?? g,
  }));
  return (
    <Plot xDomain={[0, 20]} yDomain={[0, 1]} series={series} height={118} zoom={false} xLabel="distance (px)"
      yTicks={linearTicks([0, 1], { count: 3 })} aria-label={`Source occupancy vs distance, ${label}`} />
  );
}

export function GroupGrid({ stamps, basis, kind }: { stamps: StampsView; basis: Basis; kind: FieldSet }) {
  const rows = stamps.groups[basis] ?? [];
  const refName = kind === "synthetic" ? "truth" : "LR 3×3";
  if (!rows.length) return <EmptyState compact icon="activity" title="No groups">Too few pixels in every brightness decade.</EmptyState>;
  return (
    <div className="kn-grid" role="table" aria-label="Pixel groups">
      <div className="kn-grid__head" role="row">
        <span role="columnheader">brightness</span>
        {["lower", "usual", "higher"].map((g) => (
          <span key={g} role="columnheader" style={{ color: groupColor(g) }}>{GROUP_LABEL[g]}: SR · {refName} · sources</span>
        ))}
        <span role="columnheader">sources vs distance</span>
      </div>
      {rows.map((row) => (
        <div key={row.bin} className="kn-grid__row" role="row">
          <span className="kn-grid__label" role="rowheader">
            <strong>{row.label}</strong>
            <span className="mdl-faint">{normText(row.norm)}</span>
          </span>
          {["lower", "usual", "higher"].map((g) => {
            const c = row.cells[g];
            return (
              <span key={g} className="kn-grid__cell" role="cell">
                {c ? (
                  <>
                    <span className="kn-grid__stamps">
                      <Stamp b64={c.sr} side={stamps.side} title={`${row.label}, ${GROUP_LABEL[g]}: SR`} />
                      <Stamp b64={c.ref} side={stamps.side} title={`${row.label}, ${GROUP_LABEL[g]}: ${refName}`} />
                      <Stamp b64={c.occ} side={stamps.side} map="cividis" title={`${row.label}, ${GROUP_LABEL[g]}: where sources sit`} />
                    </span>
                    <span className="kn-grid__facts">
                      n = {formatCount(c.n ?? 0)} · q̄ = {sigmaText(c.log_q)} ({(c.learned ?? 0) >= 0 ? "+" : ""}{(c.learned ?? 0).toFixed(2)} dex) · d = {c.dist ?? "—"} px
                    </span>
                  </>
                ) : <span className="mdl-faint">—</span>}
              </span>
            );
          })}
          <span className="kn-grid__radial" role="cell"><RadialPlot cells={row.cells} order={["lower", "usual", "higher"]} label={row.label} /></span>
        </div>
      ))}
    </div>
  );
}

/* ── the knee map and its relatives ───────────────────────────────────── */

const heatOf = (z: (number | null)[][], xEdges: number[], yEdges: number[]): Heat => ({
  z: z.map((row) => row.map((v) => v ?? 0)), xEdges, yEdges, scale: "log", color: DENSITY,
});

function kneeGuides(b: AnalysisBandView, knees: number[]): Guide[] {
  return b.logk.map((lk, i) => (lk == null ? null : { axis: "y" as const, v: lk, alpha: 0.35, label: kneeLabel(knees[i]), labelSide: "before" as const }))
    .filter(Boolean) as Guide[];
}

export function KneeMap({ view }: { view: AnalysisView }) {
  useResolvedTheme();                                  // themed series colours redraw on a theme flip
  const panels = KNEE_BANDS.filter((b) => view.bands[b]);
  return (
    <div className="kn-pt__multiples">
      {panels.map((band) => {
        const b = view.bands[band] as AnalysisBandView;
        const lk = b.logk.filter((v): v is number => v != null);
        const yDomain: [number, number] = [Math.min(...lk) - 0.15, Math.max(...lk) + 0.15];
        const q = b.curve.map((c) => (c ? [c.q20, c.median, c.q80] : null));
        const s = quantSeries(q);
        const series: Series[] = [
          { x: view.centres_log, ...s, color: C.comb, width: 2, dots: true, name: "median knee (20–80 %)" },
          { x: view.centres_log, y: b.curve.map((c) => c?.mech ?? null), color: categorical(3), width: 1.4, dash: [5, 3], name: "chain-rule baseline" },
          { x: [-3, 5], y: [-3, 5], color: C.mean, width: 1, dash: [3, 3], name: "knee = brightness" },
        ];
        return (
          <figure key={band} className="kn-pt__panel">
            <figcaption className="mdl-chart__title">
              {BAND_LABEL[band]} · σ = {b.sigma?.toPrecision(3)} e⁻ · n = {formatCount(b.n)}
            </figcaption>
            <Plot xDomain={X_DOMAIN} yDomain={yDomain} xTicks={X_TICKS} yTicks={logTicks(yDomain, { space: "log10" }).map((t) => ({ ...t, label: `${t.label}σ` }))}
              xLabel={BRIGHT_LABEL} yLabel="effective knee / σ" heat={heatOf(b.heat.z, b.heat.x_edges, b.heat.y_edges)}
              series={series} guides={[...kneeGuides(b, view.knees), { axis: "x", v: 0, dash: [2, 3], label: "1σ" }]}
              legend={band === panels[0] ? "auto" : undefined} aspect={0.8} syncKey="pt-map"
              exportName={`knee-map-${view.kind}-${band}`} aria-label={`Effective knee vs brightness, ${BAND_LABEL[band]}`} />
          </figure>
        );
      })}
    </div>
  );
}

export function Mechanics({ b, view, band }: { b: AnalysisBandView; view: AnalysisView; band: KneeBand }) {
  const e = b.mech.edges;
  const dom: [number, number] = [e[0], e[e.length - 1]];
  const ticks = logTicks(dom, { space: "log10" }).map((t) => ({ ...t, label: `${t.label}σ` }));
  const [lo, med, hi] = b.mech.gap;
  return (
    <figure className="kn-pt__panel kn-pt__panel--square">
      <figcaption className="mdl-chart__title">
        {BAND_LABEL[band]}: network − chain rule = {(med ?? 0) >= 0 ? "+" : ""}{med?.toFixed(2)} dex (20–80 %: {lo?.toFixed(2)} … {hi?.toFixed(2)})
      </figcaption>
      <Plot xDomain={dom} yDomain={dom} xTicks={ticks} yTicks={ticks} heat={heatOf(b.mech.z, e, e)}
        series={[{ x: dom, y: dom, color: C.mean, dash: [4, 3], width: 1.2, name: "network = chain rule" }]}
        legend="auto" xLabel="baseline knee / σ (same gradient on every knee)" yLabel="effective knee / σ" aspect={0.85}
        exportName={`chain-rule-${view.kind}-${band}`} aria-label={`Effective knee vs the chain-rule baseline, ${BAND_LABEL[band]}`} />
    </figure>
  );
}

export function Features({ b, view, basis, band }: { b: AnalysisBandView; view: AnalysisView; basis: Basis; band: KneeBand }) {
  const per = b.features[basis] ?? {};
  return (
    <div className="kn-pt__multiples kn-pt__multiples--small">
      {view.features.map((f) => {
        const series: Series[] = view.groups.map((g) => ({
          x: view.centres_log, y: per[g]?.[f.key] ?? [], color: groupColor(g), width: 1.6, dots: true, name: GROUP_LABEL[g] ?? g,
        }));
        const ys = series.flatMap((s) => s.y.filter((v): v is number => v != null && Number.isFinite(v)));
        const positive = ys.length > 0 && ys.every((v) => v > 0);
        const log = f.log && positive;
        const lo = ys.length ? Math.min(...ys) : 0, hi = ys.length ? Math.max(...ys) : 1;
        const yDomain: [number, number] = log ? [lo / 1.5, hi * 1.5] : [Math.min(0, lo), hi > lo ? hi + (hi - lo) * 0.08 : lo + 1];
        return (
          <figure key={f.key} className="kn-pt__panel">
            <figcaption className="mdl-chart__title">{f.label}</figcaption>
            <Plot xDomain={X_DOMAIN} xTicks={X_TICKS} yDomain={yDomain} yScale={log ? "log" : "linear"} series={series}
              legend={f === view.features[0] ? "auto" : undefined} aspect={0.62} syncKey="pt-features" xLabel={BRIGHT_LABEL}
              exportName={`group-${f.key}-${view.kind}-${band}`} aria-label={`${f.label} by group, ${BAND_LABEL[band]}`} />
          </figure>
        );
      })}
    </div>
  );
}

export function LookAround({ b, view, band }: { b: AnalysisBandView; view: AnalysisView; band: KneeBand }) {
  const n = view.bin_labels.length;
  return (
    <div className="kn-pt__multiples">
      {(["raw", "learned"] as const).map((basis) => {
        const per = b.look[basis] ?? {};
        const series: Series[] = Object.entries(per).map(([bin, rho]) => ({
          x: view.radii, y: rho, color: viridis(Number(bin) / Math.max(1, n - 1) * 0.92), width: 1.6, dots: true, name: view.bin_labels[Number(bin)],
        }));
        return (
          <figure key={basis} className="kn-pt__panel">
            <figcaption className="mdl-chart__title">{basis === "raw" ? "Effective knee" : "Network's own preference"}</figcaption>
            <Plot xDomain={[0, Math.max(...view.radii)]} yDomain={[-0.3, 1]} series={series} legend="auto" aspect={0.62}
              guides={[{ axis: "y", v: 0, alpha: 0.5 }]} xLabel="radius r (px)" yLabel="Spearman ρ with the brightest within r"
              syncKey="pt-look" exportName={`look-around-${basis}-${view.kind}-${band}`} aria-label={`Look-around, ${basis}, ${BAND_LABEL[band]}`} />
          </figure>
        );
      })}
    </div>
  );
}

function ShareBars({ shares, knees }: { shares: (number | null)[]; knees: number[] }) {
  const theme = useResolvedTheme();
  const color = useMemo(() => kneeRamp(knees, KNEE_RAMP[theme]), [knees, theme]);
  return (
    <span className="kn-type__bars" role="img" aria-label={`knee shares: ${knees.map((q, k) => `${kneeLabel(q)} ${Math.round((shares[k] ?? 0) * 100)}%`).join(", ")}`}>
      {knees.map((q, k) => (
        <span key={q} className="kn-type__bar" title={`${kneeLabel(q)}: ${Math.round((shares[k] ?? 0) * 100)}%`}>
          <span style={{ height: `${Math.round((shares[k] ?? 0) * 100)}%`, background: color(q) }} />
        </span>
      ))}
    </span>
  );
}

function Hist({ x, y, color }: { x: number[]; y: (number | null)[]; color: string }) {
  const max = Math.max(1e-9, ...y.map((v) => v ?? 0));
  return (
    <svg className="kn-type__hist" viewBox={`0 0 ${y.length} 20`} preserveAspectRatio="none" role="img"
      aria-label={`brightness from ${sigmaText(x[0])} to ${sigmaText(x[x.length - 1])}`}>
      {y.map((v, i) => <rect key={i} x={i + 0.08} width={0.84} y={20 - ((v ?? 0) / max) * 20} height={((v ?? 0) / max) * 20} fill={color} />)}
      <line x1={(3 / 8) * y.length} x2={(3 / 8) * y.length} y1={0} y2={20} className="kn-type__sigma" />
    </svg>
  );
}

export function Types({ b, view, stamps, band }: { b: AnalysisBandView; view: AnalysisView; stamps: StampsView | null; band: KneeBand }) {
  const refName = view.kind === "synthetic" ? "truth" : "LR 3×3";
  const n = b.types.length;
  return (
    <div className="kn-types" role="list" aria-label={`Knee types, ${BAND_LABEL[band]}`}>
      {b.types.map((t, j) => {
        const st: StampCell | undefined = stamps?.types[j];
        const color = viridis(n > 1 ? (j / (n - 1)) * 0.9 + 0.05 : 0.5);
        return (
          <div key={j} className="kn-type" role="listitem">
            <div className="kn-type__head">
              <span className="kn-type__swatch" style={{ background: color }} />
              <strong>Type {j}</strong> · {((t.prevalence ?? 0) * 100).toFixed(t.prevalence != null && t.prevalence < 0.01 ? 2 : 1)}% of pixels
            </div>
            <div className="mdl-faint kn-type__facts">
              q̄ = {sigmaText(t.log_q)} · {t.bright_median != null && t.bright_median > 0.01 ? `median ${sigmaText(Math.log10(t.bright_median))}` : "empty sky"} ·
              {" "}{Object.entries(t.classes).map(([k, v]) => `${k} ${Math.round((v ?? 0) * 100)}%`).join(", ")}
            </div>
            <ShareBars shares={t.mean_share} knees={view.knees} />
            {st && (
              <span className="kn-grid__stamps">
                <Stamp b64={st.sr} side={stamps!.side} title={`Type ${j}: SR`} />
                <Stamp b64={st.ref} side={stamps!.side} title={`Type ${j}: ${refName}`} />
                <Stamp b64={st.occ} side={stamps!.side} map="cividis" title={`Type ${j}: where sources sit`} />
              </span>
            )}
            <Hist x={view.type_hist_x} y={t.hist} color={color} />
            <span className="kn-type__axis mdl-faint"><span>10⁻³σ</span><span>1σ</span><span>10⁵σ</span></span>
          </div>
        );
      })}
    </div>
  );
}

export function TypeMix({ b, view, band }: { b: AnalysisBandView; view: AnalysisView; band: KneeBand }) {
  const n = b.types.length;
  return (
    <div className="kn-mix" role="table" aria-label={`Knee-type mix per brightness decade, ${BAND_LABEL[band]}`}>
      {b.type_mix.map((fr, i) => (
        <div key={i} className="kn-mix__row" role="row">
          <span className="kn-mix__label" role="rowheader">{view.bin_labels[i]}</span>
          <span className="kn-mix__bar" role="cell">
            {fr ? fr.map((v, j) => (v ? (
              <span key={j} title={`type ${j}: ${(v * 100).toFixed(1)}%`}
                style={{ width: `${v * 100}%`, background: viridis(n > 1 ? (j / (n - 1)) * 0.9 + 0.05 : 0.5) }} />
            ) : null)) : <span className="mdl-faint">no pixels</span>}
          </span>
        </div>
      ))}
      <div className="kn-mix__legend">
        {b.types.map((_, j) => (
          <span key={j}><span className="kn-type__swatch" style={{ background: viridis(n > 1 ? (j / (n - 1)) * 0.9 + 0.05 : 0.5) }} /> type {j}</span>
        ))}
      </div>
    </div>
  );
}

/* ── comparisons and series ────────────────────────────────────────────── */

export function CompareCurves({ view, band }: { view: CompareView; band: KneeBand }) {
  const per = view.curves[band] ?? {};
  const names = view.models.filter((m) => per[m]);
  const build = (key: "knee" | "preference"): Series[] => names.flatMap((m, i) => {
    const s = quantSeries(per[m][key]);
    const out: Series[] = [{ x: view.centres_log, ...s, color: categorical(i), width: 2, dots: true, name: modelShort(m) }];
    if (key === "knee") out.push({ x: view.centres_log, y: per[m].mech, color: categorical(i), width: 1, dash: [2, 3], key: modelShort(m) });
    return out;
  });
  const all = names.flatMap((m) => per[m].knee.flatMap((q) => (q ? q.filter((v): v is number => v != null) : [])));
  const kDom: [number, number] = all.length ? [Math.min(...all) - 0.3, Math.max(...all) + 0.3] : [-3, 3];
  const pAll = names.flatMap((m) => per[m].preference.flatMap((q) => (q ? q.filter((v): v is number => v != null) : [])));
  const pDom: [number, number] = pAll.length ? [Math.min(0, ...pAll) - 0.1, Math.max(...pAll) + 0.1] : [0, 2];
  if (!names.length) return <EmptyState compact icon="activity" title={`No ${BAND_LABEL[band]} curves`}>This band has no samples.</EmptyState>;
  return (
    <div className="kn-pt__multiples">
      <figure className="kn-pt__panel">
        <figcaption className="mdl-chart__title">{BAND_LABEL[band]}: effective knee (dotted: chain-rule baseline)</figcaption>
        <Plot xDomain={X_DOMAIN} xTicks={X_TICKS} yDomain={kDom} yTicks={logTicks(kDom, { space: "log10" }).map((t) => ({ ...t, label: `${t.label}σ` }))}
          series={[...build("knee"), { x: [-3, 5], y: [-3, 5], color: C.muted, dash: [3, 3], width: 1 }]} legend="auto"
          xLabel={BRIGHT_LABEL} yLabel="effective knee / σ" aspect={0.7} syncKey="pt-compare"
          exportName={`compare-knee-${view.kind}-${band}`} aria-label={`Effective knee per model, ${BAND_LABEL[band]}`} />
      </figure>
      <figure className="kn-pt__panel">
        <figcaption className="mdl-chart__title">{BAND_LABEL[band]}: the network's own preference over the baseline</figcaption>
        <Plot xDomain={X_DOMAIN} xTicks={X_TICKS} yDomain={pDom} series={build("preference")} legend="auto"
          xLabel={BRIGHT_LABEL} yLabel="dex over the chain-rule baseline" aspect={0.7} syncKey="pt-compare"
          exportName={`compare-preference-${view.kind}-${band}`} aria-label={`Preference per model, ${BAND_LABEL[band]}`} />
      </figure>
    </div>
  );
}

export function ShareDiff({ view, band }: { view: CompareView; band: KneeBand }) {
  const d = view.share_diff?.[band];
  if (!d) return <EmptyState compact icon="activity" title="No share difference">This comparison has no {BAND_LABEL[band]} difference.</EmptyState>;
  const [a, b] = view.models;
  const flat = d.matrix.flat().filter((v): v is number => v != null);
  const lim = Math.max(0.05, ...flat.map(Math.abs));
  const order = d.knees.map((_, k) => k).reverse();
  return (
    <div className="kn-diff-wrap">
      <table className="kn-diff" aria-label={`Mean knee share, ${modelShort(b)} − ${modelShort(a)}, ${BAND_LABEL[band]}`}>
        <thead>
          <tr><th scope="col">knee</th>{view.bin_labels.map((l) => <th key={l} scope="col">{l}</th>)}</tr>
        </thead>
        <tbody>
          {order.map((k) => (
            <tr key={k}>
              <th scope="row">{kneeLabel(d.knees[k])}</th>
              {d.matrix[k].map((v, i) => (
                <td key={i} data-sign={v == null ? undefined : v >= 0 ? "pos" : "neg"}
                  style={{ ["--mix" as string]: `${v == null ? 0 : Math.round((Math.abs(v) / lim) * 85)}%` }}>
                  {v == null ? "" : Math.abs(v) >= 0.01 ? `${v > 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}` : ""}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <Caption>Red: {modelShort(b)} leans on that knee more than {modelShort(a)} at that brightness; blue: less.</Caption>
    </div>
  );
}

export function CompareLook({ view, band }: { view: CompareView; band: KneeBand }) {
  const per = view.look?.[band] ?? {};
  const names = view.models.filter((m) => per[m]);
  const n = view.bin_labels.length;
  const dashes = [undefined, [6, 3], [2, 3], [8, 3, 2, 3]];
  const series: Series[] = names.flatMap((m, i) => Object.entries(per[m]).map(([label, rho]) => ({
    x: view.radii, y: rho, color: viridis((Math.max(0, view.bin_labels.indexOf(label)) / Math.max(1, n - 1)) * 0.92), width: 1.5,
    dash: dashes[i % dashes.length], name: `${modelShort(m)} · ${label}`, key: `${m}|${label}`,
  })));
  if (!series.length) return <EmptyState compact icon="activity" title="No look-around">This comparison has no {BAND_LABEL[band]} look-around.</EmptyState>;
  return (
    <figure className="kn-pt__panel">
      <figcaption className="mdl-chart__title">{BAND_LABEL[band]}: {names.map((m, i) => `${i === 0 ? "solid" : "dashed"} ${modelShort(m)}`).join(", ")}</figcaption>
      <Plot xDomain={[0, Math.max(...view.radii)]} yDomain={[-0.3, 1]} series={series} legend="auto" aspect={0.5}
        guides={[{ axis: "y", v: 0, alpha: 0.5 }]} xLabel="radius r (px)" yLabel="Spearman ρ with the brightest within r"
        exportName={`compare-look-${view.kind}-${band}`} aria-label={`Look-around per model, ${BAND_LABEL[band]}`} />
    </figure>
  );
}

export function PairsGrid({ pairs, kind }: { pairs: PairsView; kind: FieldSet }) {
  const [a, b] = pairs.models;
  const refName = kind === "synthetic" ? "truth" : "LR 3×3";
  const label: Record<string, string> = { lower: `${b} lower`, same: "about the same", higher: `${b} higher` };
  if (!pairs.rows.length) return <EmptyState compact icon="activity" title="No paired groups">Too few shared pixels per decade.</EmptyState>;
  return (
    <div className="kn-grid kn-grid--pairs" role="table" aria-label="The same pixels in both models">
      <div className="kn-grid__head" role="row">
        <span role="columnheader">brightness · {formatCount(pairs.n)} paired</span>
        {["lower", "same", "higher"].map((g) => <span key={g} role="columnheader">{label[g]}: {a} · {b} · {refName}</span>)}
      </div>
      {pairs.rows.map((row) => (
        <div key={row.bin} className="kn-grid__row" role="row">
          <span className="kn-grid__label" role="rowheader"><strong>{row.label}</strong><span className="mdl-faint">{normText(row.norm)}</span></span>
          {["lower", "same", "higher"].map((g) => {
            const c = row.cells[g];
            return (
              <span key={g} className="kn-grid__cell" role="cell">
                {c ? (
                  <>
                    <span className="kn-grid__stamps">
                      <Stamp b64={c.sr_a} side={pairs.side} title={`${row.label}: ${a} SR`} />
                      <Stamp b64={c.sr_b} side={pairs.side} title={`${row.label}: ${b} SR`} />
                      <Stamp b64={c.ref} side={pairs.side} title={`${row.label}: ${refName}`} />
                    </span>
                    <span className="kn-grid__facts">n = {formatCount(c.n)} · Δ = {(c.delta ?? 0) >= 0 ? "+" : ""}{c.delta?.toFixed(2)} dex</span>
                  </>
                ) : <span className="mdl-faint">—</span>}
              </span>
            );
          })}
        </div>
      ))}
    </div>
  );
}
