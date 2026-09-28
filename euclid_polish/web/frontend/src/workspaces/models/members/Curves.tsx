/* Models › Members, curves view: every active member's training curves on
   linked plots (one crosshair): validation PSNR, the training loss FACETED
   by loss type (L1 and L2 losses live on different scales, so each type gets
   its own plot) and the gradient norm, with the target-steps guide (70k).
   Coloured by training knee by default (loss / depth / multi-knee /
   uniform); the legend toggles a group everywhere; the members picked on the
   roster are highlighted (?sel=). Wall time per step lives in Runs ›
   History. Everything is in the URL. */
import { useMemo } from "react";
import Plot, { Legend, useLegend, type Guide, type Series } from "../../../charts/Plot";
import { C } from "../../../colors";
import { usePageActions } from "../../../app/palette";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Button, Chip, EmptyState, Segmented, Select, Switch, Toolbar, ToolbarGroup, ToolbarSpacer,
} from "../../../ui";
import { BAND_SHORT, BANDS, useCurves, type Curve, type Mode } from "../api";
import { ColorBySelect, LoadState, openMember, useFacetColors } from "../common";
import { facetOf, kneeText, lossFacets, memberNumber, smooth, xy, type ColorBy } from "../model";

type Show = "all" | "bands" | "psnr" | "loss" | "gnorm";
type Panel = { key: string; title: string; y: string; log: boolean; curves: Curve[]; get: (c: Curve) => [number, number][] };

const SHOWS: readonly Show[] = ["all", "bands", "psnr", "loss", "gnorm"];
const parseShow = (raw: string): Show | undefined => (SHOWS.includes(raw as Show) ? raw as Show : raw ? "all" : undefined);

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

/** The most common target step count of the shown members (the guide). */
function commonTarget(curves: readonly Curve[]): number | null {
  const counts = new Map<number, number>();
  for (const c of curves) if (c.target_steps) counts.set(c.target_steps, (counts.get(c.target_steps) ?? 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? null;
}

export function Curves({ mode }: { mode: Mode }) {
  const res = useCurves();
  // ?layout= as the old Curves tab wrote it (its "grid" and "time" read as All).
  const [show, setShow] = useUrlState<Show>("layout", "all", { parse: parseShow });
  const [colorBy, setColorBy] = useUrlState<ColorBy>("color", "knee");
  const [selRaw, setSelRaw] = useUrlState("sel", "");
  const [onlySel, setOnlySel] = useUrlState("only", false);
  const [smoothN, setSmoothN] = useUrlState("smooth", "1");
  const [logY, setLogY] = useUrlState("logy", true);
  const lg = useLegend();

  const all = useMemo(() => (res.data?.members ?? []).filter((c) => c.starless === (mode === "starless")), [res.data, mode]);
  const sel = useMemo(() => new Set(selRaw.split(",").map((s) => memberNumber(s)).filter(Boolean) as string[]), [selRaw]);
  const shown = useMemo(() => (onlySel && sel.size ? all.filter((c) => sel.has(memberNumber(c.name) ?? "")) : all), [all, onlySel, sel]);
  const colors = useFacetColors(shown, colorBy);
  const w = Math.max(1, Number(smoothN) || 1);

  const panels = useMemo<Panel[]>(() => {
    const psnr: Panel = { key: "psnr", title: "Validation PSNR (asinh)", y: "PSNR [dB]", log: false, curves: shown, get: (c) => c.psnr };
    const loss = lossFacets(shown).map((f): Panel => ({
      key: `loss:${f.loss}`, title: `Training loss · ${f.loss.toUpperCase()}`, y: `${f.loss.toUpperCase()} loss`, log: logY,
      curves: f.curves, get: (c) => c.loss_series,
    }));
    const gnorm: Panel = { key: "gnorm", title: "Gradient norm (mean)", y: "‖g‖", log: logY, curves: shown, get: (c) => c.gnorm };
    const bands = BANDS.map((b): Panel => ({
      key: `band:${b}`, title: `Validation PSNR · ${BAND_SHORT[b]}`, y: "PSNR [dB]", log: false, curves: shown, get: (c) => c.band_psnr?.[b] ?? [],
    }));
    if (show === "bands") return bands;
    if (show === "psnr") return [psnr];
    if (show === "loss") return loss;
    if (show === "gnorm") return [gnorm];
    return [psnr, gnorm, ...loss];
  }, [show, shown, logY]);

  const target = commonTarget(shown);
  const plots = useMemo(() => panels.map((p) => {
    let xMax = 1;
    const series: Series[] = [];
    for (const c of p.curves) {
      const { x, y } = xy(p.get(c));
      if (!x.length) continue;
      xMax = Math.max(xMax, x[x.length - 1]);
      const picked = sel.has(memberNumber(c.name) ?? "");
      series.push({
        x, y: smooth(y, w), color: colors.of(c),
        width: picked ? 2.4 : 1.2, alpha: sel.size && !picked ? 0.35 : 0.9,
        name: `#${memberNumber(c.name)} · ${c.loss_norm.toUpperCase()} · ${kneeText(c).text}`,
        key: facetOf(c, colorBy),
      });
    }
    const guides: Guide[] = target ? [{ axis: "x", v: target, color: C.guide, dash: [4, 3], label: `${kfmt(target)} target` }] : [];
    return { ...p, series, guides, xDomain: [0, Math.max(xMax, target ?? 0)] as [number, number], yDomain: domainOf(series, p.log) };
  }), [panels, sel, colors, colorBy, w, target]);

  usePageActions([
    { id: "curves-all", label: "Curves: PSNR, gradient norm and loss by type", group: "Members", run: () => setShow("all") },
    { id: "curves-bands", label: "Curves: per-band PSNR", group: "Members", run: () => setShow("bands") },
    { id: "curves-clear", label: "Curves: clear the member highlight", group: "Members", disabled: !sel.size, run: () => setSelRaw("") },
  ]);

  const members = [...sel];
  return (
    <div className="mdl-stack">
      <Toolbar label="Curve controls">
        <ToolbarGroup label="Show">
          <Segmented<Show> size="sm" aria-label="Curves shown" value={show} onChange={setShow} options={[
            { value: "all", label: "All", title: "Validation PSNR, gradient norm and the loss by loss type" },
            { value: "bands", label: "Bands", title: "Validation PSNR per band" },
            { value: "psnr", label: "PSNR" }, { value: "loss", label: "Loss" }, { value: "gnorm", label: "‖g‖" },
          ]} />
        </ToolbarGroup>
        <ToolbarGroup label="Colour"><ColorBySelect value={colorBy} onChange={setColorBy} /></ToolbarGroup>
        <ToolbarGroup label="Smooth">
          <Select size="sm" aria-label="Smoothing window" value={smoothN} onChange={setSmoothN}
            options={[{ value: "1", label: "raw" }, { value: "3", label: "3 pts" }, { value: "5", label: "5 pts" }, { value: "9", label: "9 pts" }]} />
        </ToolbarGroup>
        <Switch size="sm" checked={logY} onChange={setLogY}>log loss and ‖g‖</Switch>
        <ToolbarSpacer />
        {sel.size > 0 && <>
          <Switch size="sm" checked={onlySel} onChange={setOnlySel}>only selected</Switch>
          <Button size="sm" variant="ghost" icon="close" onClick={() => setSelRaw("")}>{sel.size} highlighted</Button>
        </>}
      </Toolbar>
      <LoadState loading={res.loading} error={res.error} onRetry={res.reload}
        empty={!res.loading && !shown.length && (
          <EmptyState icon="activity" title={`No ${mode} training logs`}>Pull members from FASRC to see their curves.</EmptyState>
        )}>
        <div className="mdl-stack">
          <Legend items={colors.legend} {...lg.legendProps} />
          <div className={plots.length > 1 ? "mdl-charts" : undefined}>
            {plots.map((p) => (
              <div key={p.key} className="mdl-chart">
                <h3 className="mdl-chart__title">{p.title}</h3>
                <Plot {...lg.plotProps} xDomain={p.xDomain} yDomain={p.yDomain} yScale={p.log ? "log" : "linear"}
                  xLabel="training step" yLabel={p.y} series={p.series} guides={p.guides}
                  aspect={plots.length > 1 ? 0.62 : 0.42} syncKey="mdl-curves" zoomAxes="x"
                  xFormat={kfmt} yFormat={(v) => (p.log ? v.toPrecision(3) : v.toFixed(2))}
                  exportName={`ensemble-${p.key.replace(":", "-")}`} aria-label={p.title} />
              </div>
            ))}
          </div>
          {members.length > 0 && (
            <div className="mdl-row">
              <span className="mdl-faint">Highlighted:</span>
              {members.map((n) => (
                <Chip key={n} onClick={() => openMember(`member_${n}`)} title="Open in the inspector">#{n}</Chip>
              ))}
            </div>
          )}
        </div>
      </LoadState>
    </div>
  );
}
