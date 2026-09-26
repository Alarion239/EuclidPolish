/* Histogram of the visible region (the current pan/zoom crop, or the whole
 * image) of one tier and band, in native units, log counts. The draggable
 * black and white handles are opt-in manual cuts: they edit the frame's
 * transfer group — black point, and the gain that puts white at the handle
 * (white = black + 30·K0/gain, in the transfer's display units = native ×
 * display scale) — through the same path as the toolbar sliders (the Display
 * panel when linked, this viewer's override otherwise). */
import { useMemo, useRef, useState } from "react";
import { linearTicks } from "../ticks";
import { Button, Select } from "../ui";
import { colormapGradient } from "./colormaps";
import { useController, useSettings, useViewer } from "./hooks";
import { formatValue, unitLabel } from "./readout";
import { finiteSorted, histogram, percentile, regionValues, robustStats } from "./stats";

const W = 640, H = 150, PAD = 28;
const ANCHORED = new Set(["asinh-abs", "linear", "sqrt", "log"]);

export function HistogramPanel() {
  const ctrl = useController();
  const shown = useViewer((s) => s.shown);
  const view = useViewer((s) => s.view);
  const readoutTier = useViewer((s) => s.readout?.tier ?? null);
  const settings = useSettings();
  const keys = ctrl.frameKeys().filter((k) => shown[k]?.kind === "cube" && k !== "morph");
  const [picked, setPicked] = useState<string | null>(null);
  const tier = picked && keys.includes(picked) ? picked : (readoutTier && keys.includes(readoutTier) ? readoutTier : keys[0]);
  const entry = tier ? shown[tier] : undefined;
  const rec = entry?.kind === "cube" ? entry.rec : null;
  const bands = rec ? (rec.bands.length === rec.c ? rec.bands : ctrl.s.meta?.band_names.slice(0, rec.c) ?? []) : [];
  const [bandPick, setBandPick] = useState<string | null>(null);
  const band = bandPick && bands.includes(bandPick) ? bandPick : bands.includes(settings.color) ? settings.color : bands[0];
  const k = Math.max(0, bands.indexOf(band));
  const svgRef = useRef<SVGSVGElement>(null);
  const dragging = useRef<"black" | "white" | null>(null);

  const data = useMemo(() => {
    if (!rec || !tier) return null;
    const crop = view ? ctrl.cropOf(tier, view) : null;
    const region = crop ? { x: crop.x, y: crop.y, w: crop.side, h: crop.side } : { x: 0, y: 0, w: rec.w, h: rec.h };
    const values = regionValues(rec, k, region);
    const sorted = finiteSorted(values, 200000);
    if (!sorted.length) return { values, hist: null, lo: 0, hi: 1, stats: robustStats(values) };
    let lo = percentile(sorted, 0.1), hi = percentile(sorted, 99.9);
    if (!(hi > lo)) { lo -= 0.5 * (Math.abs(lo) || 1); hi = lo + (Math.abs(lo) || 1); }
    return { values, hist: histogram(values, lo, hi, 128), lo, hi, stats: robustStats(values) };
  }, [ctrl, rec, tier, view, k]);

  if (!rec || !tier || !data) return <div className="cv-panel cv-hist"><p className="cv-panel__empty">No image loaded.</p></div>;
  const group = ctrl.groupOf(rec);
  const t = ctrl.transfer(group, settings);
  const ds = rec.displayScale || 1;
  const K0 = ctrl.K0();
  const anchored = ANCHORED.has(settings.stretch) && ctrl.s.meta?.render_mode !== "log";
  const blackN = t.black / ds;
  const whiteN = (t.black + (30 * K0) / Math.max(t.gain, 1e-30)) / ds;
  const { lo, hi, hist } = data;
  const x = (v: number) => PAD + ((v - lo) / (hi - lo)) * (W - 2 * PAD);
  const valueAt = (clientX: number) => {
    const r = svgRef.current?.getBoundingClientRect();
    if (!r || !(r.width > 0)) return lo;
    const px = ((clientX - r.left) / r.width) * W;
    return lo + ((px - PAD) / (W - 2 * PAD)) * (hi - lo);
  };
  const maxLog = hist ? Math.max(1, ...hist.counts.map((c) => Math.log10(1 + c))) : 1;
  const unit = unitLabel(rec.unit || ctrl.tierMeta(tier)?.unit || "");
  const clampX = (v: number) => Math.max(PAD, Math.min(W - PAD, x(v)));

  const onMove = (e: React.PointerEvent) => {
    const which = dragging.current;
    if (!which) return;
    const v = valueAt(e.clientX) * ds;
    if (which === "black") {
      const white = t.black + (30 * K0) / t.gain;
      if (white > v) ctrl.setTransfer(group, { black: v, gain: Math.max(0.01, (30 * K0) / (white - v)) });
    } else if (v > t.black) {
      ctrl.setTransfer(group, { gain: Math.max(0.01, Math.min(1000, (30 * K0) / (v - t.black))) });
    }
  };

  return (
    <div className="cv-panel cv-hist">
      <div className="cv-panel__head">
        <span className="cv-panel__title">Histogram · {view ? "visible region" : "whole image"}</span>
        <Select value={tier} onChange={setPicked} aria-label="Histogram tier" options={keys.map((kk) => ({ value: kk, label: ctrl.tierLabel(kk) }))} />
        {bands.length > 1 && <Select value={band} onChange={setBandPick} aria-label="Histogram band" options={bands.map((b) => ({ value: b, label: b }))} />}
        <span className="cv-panel__stats mono">
          n {data.stats.n} · median {formatValue(data.stats.median)} · σ(MAD) {formatValue(data.stats.sigma)} · {formatValue(data.stats.min)} … {formatValue(data.stats.max)} {unit}
        </span>
        {anchored && <Button size="sm" variant="ghost" onClick={() => ctrl.setTransfer(group, { black: 0, gain: 1 })}>reset cuts</Button>}
      </div>
      <svg ref={svgRef} className="cv-hist__svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img"
        aria-label={`Histogram of ${ctrl.tierLabel(tier)} ${band}`}
        onPointerMove={onMove} onPointerUp={() => { dragging.current = null; }} onPointerLeave={() => { dragging.current = null; }}>
        {hist && hist.counts.map((c, i) => {
          const h = (Math.log10(1 + c) / maxLog) * (H - 34);
          const x0 = x(hist.edges[i]), x1 = x(hist.edges[i + 1]);
          return <rect key={i} className="cv-hist__bar" x={x0} y={H - 22 - h} width={Math.max(0.5, x1 - x0 - 0.3)} height={h} />;
        })}
        <line className="cv-hist__axis" x1={PAD} y1={H - 22} x2={W - PAD} y2={H - 22} />
        {linearTicks([lo, hi], { count: 6 }).map((tk) => (
          <g key={tk.v}><line className="cv-hist__axis" x1={x(tk.v)} y1={H - 22} x2={x(tk.v)} y2={H - 18} />
            <text className="cv-hist__tick" x={x(tk.v)} y={H - 6} textAnchor="middle">{tk.label}</text></g>
        ))}
        {anchored && (["black", "white"] as const).map((which) => {
          const v = which === "black" ? blackN : whiteN;
          const cx = clampX(v);
          return (
            <g key={which} className={`cv-hist__handle cv-hist__handle--${which}`} role="slider" tabIndex={0}
              aria-label={`${which} point`} aria-valuenow={v} aria-valuetext={`${formatValue(v)} ${unit}`}
              onPointerDown={(e) => { dragging.current = which; (e.currentTarget.ownerSVGElement as SVGSVGElement).setPointerCapture(e.pointerId); }}
              onKeyDown={(e) => {
                if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
                e.preventDefault(); e.stopPropagation();
                const step = ((hi - lo) / 100) * (e.key === "ArrowLeft" ? -1 : 1) * ds;
                if (which === "black") {
                  const white = t.black + (30 * K0) / t.gain, b = t.black + step;
                  if (white > b) ctrl.setTransfer(group, { black: b, gain: (30 * K0) / (white - b) });
                } else {
                  const w = t.black + (30 * K0) / t.gain + step;
                  if (w > t.black) ctrl.setTransfer(group, { gain: (30 * K0) / (w - t.black) });
                }
              }}>
              <line x1={cx} y1={6} x2={cx} y2={H - 22} />
              <rect x={cx - 5} y={2} width={10} height={12} rx={2} />
              <text x={cx} y={24} textAnchor={which === "black" ? "start" : "end"} dx={which === "black" ? 6 : -6}>{which} {formatValue(v)}</text>
            </g>
          );
        })}
      </svg>
      <div className="cv-hist__cmap" style={{ background: colormapGradient(settings.colormap) }} aria-hidden="true" />
      {!anchored && <p className="cv-panel__note">The {settings.stretch} stretch sets its own limits from the image; manual cuts apply to the absolute stretches.</p>}
    </div>
  );
}
