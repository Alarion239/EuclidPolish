/* ensemble/curves (spec §8.2): every active member's training curves —
   validation PSNR (joint or per band), the training loss, the gradient norm
   and the wall time per 1000 steps — on linked plots (one crosshair), with
   hover identification, colour by loss / depth / knee / multi-knee (the
   legend toggles a group everywhere), the members picked on the Members tab
   highlighted (?sel=), smoothing and log-y. Everything is in the URL. */
import { useMemo } from "react";
import Plot, { Legend, useLegend, type Guide, type Series } from "../../../charts/Plot";
import { C } from "../../../colors";
import { usePageActions } from "../../../app/palette";
import { useUrlState } from "../../../hooks/useUrlState";
import { Button, Chip, EmptyState, Page, Segmented, Select, Switch } from "../../../ui";
import { BAND_SHORT, BANDS, useCurves, useMode, type Curve } from "../api";
import { BarGroup, ColorBySelect, EnsBar, LoadState, openMember, useFacetColors } from "../common";
import { facetOf, kneeText, memberNumber, smooth, xy, type ColorBy } from "../model";
import "../ensemble.css";

type Layout = "grid" | "bands" | "psnr" | "loss" | "gnorm" | "time";
type MetricKey = "psnr" | "loss" | "gnorm" | "time" | `band:${string}`;

const METRICS: Record<string, { title: string; y: string; get: (c: Curve) => [number, number][]; log?: boolean }> = {
  psnr: { title: "Validation PSNR (asinh)", y: "PSNR [dB]", get: (c) => c.psnr },
  loss: { title: "Training loss", y: "combined loss", get: (c) => c.loss_series, log: true },
  gnorm: { title: "Gradient norm (mean)", y: "‖g‖", get: (c) => c.gnorm, log: true },
  time: { title: "Wall time per 1000 steps", y: "s / 1k steps", get: (c) => c.step_time },
  ...Object.fromEntries(BANDS.map((b) => [`band:${b}`, {
    title: `Validation PSNR · ${BAND_SHORT[b]}`, y: "PSNR [dB]", get: (c: Curve) => c.band_psnr?.[b] ?? [],
  }])),
};

const LAYOUTS: Record<Layout, MetricKey[]> = {
  grid: ["psnr", "loss", "gnorm", "time"],
  bands: BANDS.map((b) => `band:${b}` as MetricKey),
  psnr: ["psnr"], loss: ["loss"], gnorm: ["gnorm"], time: ["time"],
};

const kfmt = (v: number) => (Math.abs(v) >= 1000 ? `${+(v / 1000).toFixed(1)}k` : String(Math.round(v)));

function domainOf(series: Series[], log: boolean): [number, number] {
  let lo = Infinity, hi = -Infinity;
  for (const s of series) for (const v of s.y) {
    if (v == null || !Number.isFinite(v) || (log && v <= 0)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo)) return log ? [0.1, 1] : [0, 1];
  if (log) return [lo / 1.15, hi * 1.15];
  const pad = (hi - lo) * 0.06 || 1;
  return [lo - pad, hi + pad];
}

export default function Curves() {
  const mode = useMode();
  const res = useCurves();
  const [layout, setLayout] = useUrlState<Layout>("layout", "grid");
  const [colorBy, setColorBy] = useUrlState<ColorBy>("color", "loss");
  const [selRaw, setSelRaw] = useUrlState("sel", "");
  const [onlySel, setOnlySel] = useUrlState("only", false);
  const [smoothN, setSmoothN] = useUrlState("smooth", "1");
  const [logY, setLogY] = useUrlState("logy", true);
  const lg = useLegend();

  const all = useMemo(() => (res.data?.members ?? []).filter((c) => c.starless === (mode === "starless")), [res.data, mode]);
  const sel = useMemo(() => new Set(selRaw.split(",").map((s) => memberNumber(s)).filter(Boolean) as string[]), [selRaw]);
  const shown = onlySel && sel.size ? all.filter((c) => sel.has(memberNumber(c.name) ?? "")) : all;
  const colors = useFacetColors(shown, colorBy);
  const w = Math.max(1, Number(smoothN) || 1);

  const plots = useMemo(() => LAYOUTS[layout].map((key) => {
    const spec = METRICS[key];
    const log = !!spec.log && logY;
    let xMax = 1;
    const series: Series[] = [];
    for (const c of shown) {
      const { x, y } = xy(spec.get(c));
      if (!x.length) continue;
      xMax = Math.max(xMax, x[x.length - 1]);
      const picked = sel.has(memberNumber(c.name) ?? "");
      const k = kneeText(c);
      series.push({
        x, y: smooth(y, w), color: colors.of(c),
        width: picked ? 2.4 : 1.2, alpha: sel.size && !picked ? 0.35 : 0.9,
        name: `#${memberNumber(c.name)} · ${c.loss_norm.toUpperCase()} · ${k.text}`,
        key: facetOf(c, colorBy),
      });
    }
    const targets = shown.map((c) => c.target_steps).filter((t): t is number => !!t);
    const target = targets.length ? targets.sort((a, b) => targets.filter((t) => t === b).length - targets.filter((t) => t === a).length)[0] : null;
    const guides: Guide[] = target ? [{ axis: "x", v: target, color: C.guide, dash: [4, 3], label: `${kfmt(target)} target` }] : [];
    return { key, spec, log, series, guides, xDomain: [0, xMax] as [number, number], yDomain: domainOf(series, log) };
  }), [layout, shown, sel, colors, colorBy, w, logY]);

  usePageActions([
    { id: "curves-grid", label: "Curves: PSNR, loss, gnorm and step time", group: "Curves", run: () => setLayout("grid") },
    { id: "curves-bands", label: "Curves: per-band PSNR", group: "Curves", run: () => setLayout("bands") },
    { id: "curves-clear", label: "Curves: clear the member highlight", group: "Curves", disabled: !sel.size, run: () => setSelRaw("") },
  ]);

  const members = [...sel];
  return (
    <Page>
      <EnsBar label="Curve controls">
        <BarGroup label="Show">
          <Segmented<Layout> size="sm" aria-label="Curves layout" value={layout} onChange={setLayout} options={[
            { value: "grid", label: "All", title: "PSNR, loss, gradient norm and step time" },
            { value: "bands", label: "Bands", title: "Validation PSNR per band" },
            { value: "psnr", label: "PSNR" }, { value: "loss", label: "Loss" },
            { value: "gnorm", label: "‖g‖" }, { value: "time", label: "Time" },
          ]} />
        </BarGroup>
        <BarGroup label="Colour">
          <ColorBySelect value={colorBy} onChange={setColorBy} />
        </BarGroup>
        <BarGroup label="Smooth">
          <Select size="sm" aria-label="Smoothing window" value={smoothN} onChange={setSmoothN}
            options={[{ value: "1", label: "raw" }, { value: "3", label: "3 pts" }, { value: "5", label: "5 pts" }, { value: "9", label: "9 pts" }]} />
        </BarGroup>
        <Switch size="sm" checked={logY} onChange={setLogY}>log loss</Switch>
        <span className="ens-bar__spacer" />
        {sel.size > 0 && <>
          <Switch size="sm" checked={onlySel} onChange={setOnlySel}>only selected</Switch>
          <Button size="sm" variant="ghost" icon="close" onClick={() => setSelRaw("")}>{sel.size} highlighted</Button>
        </>}
      </EnsBar>
      <LoadState loading={res.loading} error={res.error} onRetry={res.reload}
        empty={!res.loading && !shown.length && (
          <EmptyState icon="activity" title={`No ${mode} training logs`}>Pull members from FASRC (Overview) to see their curves.</EmptyState>
        )}>
        <div className="ens-stack">
          <div className="ens-row">
            <Legend items={colors.legend} {...lg.legendProps} />
          </div>
          <div className={plots.length > 1 ? "ens-charts" : undefined}>
            {plots.map((p) => (
              <div key={p.key} className="ens-chart">
                <h3 className="ens-chart__title">{p.spec.title}</h3>
                <Plot {...lg.plotProps} xDomain={p.xDomain} yDomain={p.yDomain} yScale={p.log ? "log" : "linear"}
                  xLabel="training step" yLabel={p.spec.y} series={p.series} guides={p.guides}
                  aspect={plots.length > 1 ? 0.62 : 0.42} syncKey="ens-curves" zoomAxes="x"
                  xFormat={kfmt} yFormat={(v) => (p.log ? v.toPrecision(3) : v.toFixed(2))}
                  exportName={`ensemble-${p.key.replace(":", "-")}`} aria-label={p.spec.title} />
              </div>
            ))}
          </div>
          {members.length > 0 && (
            <div className="ens-row">
              <span className="ens-faint">Highlighted:</span>
              {members.map((n) => (
                <Chip key={n} onClick={() => openMember(`member_${n}`)} title="Open in the inspector">#{n}</Chip>
              ))}
            </div>
          )}
        </div>
      </LoadState>
    </Page>
  );
}
