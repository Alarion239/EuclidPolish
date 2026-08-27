/* Plot — one reusable canvas line-chart. Every figure in the app is expressed
   as {domain, ticks, bands, series, guides}; no page hand-rolls axes again. Retina
   crisp, resizes with its container, themed from CSS tokens. */
import { useEffect, useRef, type MouseEvent } from "react";
import { viridis } from "../colors";

export type Series = {
  x: number[];
  y: (number | null)[];
  low?: (number | null)[];
  high?: (number | null)[];
  color: string;
  mode?: "line" | "histogram" | "scatter";
  marker?: "filled" | "ring" | "diamond";
  width?: number;
  dash?: number[];
  dots?: boolean;
  markerEvery?: number;
  hatch?: boolean;
  alpha?: number;
  fillAlpha?: number;
  label?: string;
  labelAt?: number;
};
export type Guide = {
  axis: "x" | "y";
  v: number;
  color?: string;
  dash?: number[];
  width?: number;
  alpha?: number;
  label?: string;
  labelSide?: "before" | "after";
};
export type Band = {
  axis: "x" | "y";
  from: number;
  to: number;
  color: string;
  alpha?: number;
  hatch?: boolean;
  label?: string;
};
export type Tick = { v: number; label: string };

/* Optional density layer drawn UNDER the grid/guides/series — a 2D histogram
   painted as a viridis heatmap (log-count normalized, empty cells transparent).
   Edges are in the same data units as the axes (pre-transform your log axes
   into log10 values and keep the plot linear). Lets the std-vs-error /
   std-vs-brightness diagnostics render fully client-side, cloud and all. */
export type Heat = {
  z: number[][];        // z[i][j] = count in x-bin i, y-bin j
  xEdges: number[];     // length z.length + 1
  yEdges: number[];     // length z[0].length + 1
  max?: number;         // log-norm ceiling (default = max positive count)
  min?: number;         // linear-norm floor (default = min finite value)
  scale?: "log" | "linear"; // default log (pixel-density diagnostics)
  colorTicks?: Tick[];  // values/labels in z units
  colorLabel?: string;
  color?: (t: number) => string;  // t∈[0,1] → css color (default viridis)
};

/* A histogram cell selected in a heat plot — reported by onHeatClick, outlined
   by `highlight`. */
export type Cell = { i: number; j: number };

export type PlotProps = {
  xDomain: [number, number];
  yDomain: [number, number];
  xScale?: "linear" | "log";
  xTicks?: Tick[];
  yTicks?: Tick[];
  xLabel?: string;
  yLabel?: string;
  title?: string;
  series: Series[];
  bands?: Band[];
  guides?: Guide[];
  heat?: Heat;
  /* When set (heat plots only) a click reports the histogram cell under the
     cursor; `highlight` outlines a picked cell. Powers click-to-inspect. */
  onHeatClick?: (cell: Cell) => void;
  onPlotClick?: (point: { x: number; y: number }) => void;
  highlight?: Cell | null;
  height?: number;      /* fixed px; omit → aspect-driven */
  aspect?: number;      /* height = width * aspect (default 0.5) */
};

function css(name: string, fallback: string) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

/* Plot geometry captured on every draw so the click handler can invert screen
   px → data coords → histogram cell without re-deriving margins. */
type Geo = { m: { l: number; r: number; t: number; b: number }; iw: number; ih: number;
  logx: boolean; xDomain: [number, number]; yDomain: [number, number] };

/* Largest k with edges[k] <= v (edges strictly increasing); -1 if out of range. */
function binOf(edges: number[], v: number): number {
  if (!(v >= edges[0]) || !(v <= edges[edges.length - 1])) return -1;
  let lo = 0, hi = edges.length - 1;
  while (lo < hi - 1) { const mid = (lo + hi) >> 1; if (edges[mid] <= v) lo = mid; else hi = mid; }
  return lo;
}

function drawMarker(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  radius: number,
  marker: Series["marker"] = "filled",
) {
  ctx.beginPath();
  if (marker === "diamond") {
    ctx.moveTo(x, y - radius);
    ctx.lineTo(x + radius, y);
    ctx.lineTo(x, y + radius);
    ctx.lineTo(x - radius, y);
    ctx.closePath();
    ctx.stroke();
  } else {
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    if (marker === "ring") ctx.stroke();
    else ctx.fill();
  }
}

export default function Plot(p: PlotProps) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const geo = useRef<Geo | null>(null);

  useEffect(() => {
    const el = wrap.current, cv = canvas.current;
    if (!el || !cv) return;
    const draw = () => {
      const cssW = el.clientWidth || 640;
      const cssH = p.height ?? Math.round(cssW * (p.aspect ?? 0.5));
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      cv.width = Math.round(cssW * dpr);
      cv.height = Math.round(cssH * dpr);
      cv.style.width = cssW + "px";
      cv.style.height = cssH + "px";
      const ctx = cv.getContext("2d")!;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      geo.current = render(ctx, cssW, cssH, p);
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(el);
    return () => ro.disconnect();
  });

  const onClick = (e: MouseEvent<HTMLCanvasElement>) => {
    const g = geo.current, heat = p.heat;
    if ((!p.onHeatClick && !p.onPlotClick) || !g) return;
    const r = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    const [xa, xb] = g.xDomain, [ya, yb] = g.yDomain;
    const fx = (px - g.m.l) / g.iw, fy = 1 - (py - g.m.t) / g.ih;
    if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return;
    const dataX = g.logx ? 10 ** (Math.log10(xa) + fx * (Math.log10(xb) - Math.log10(xa))) : xa + fx * (xb - xa);
    const dataY = ya + fy * (yb - ya);
    if (p.onHeatClick && heat) {
      const i = binOf(heat.xEdges, dataX), j = binOf(heat.yEdges, dataY);
      if (i >= 0 && j >= 0) p.onHeatClick({ i, j });
    } else p.onPlotClick?.({ x: dataX, y: dataY });
  };

  return (
    <div ref={wrap} style={{ width: "100%" }}>
      <canvas ref={canvas} onClick={onClick}
        style={p.onHeatClick || p.onPlotClick ? { cursor: "crosshair" } : undefined} />
    </div>
  );
}

function render(ctx: CanvasRenderingContext2D, W: number, H: number, p: PlotProps): Geo {
  const ink = css("--text", "#e5edf7");
  const dim = css("--text-dim", "#9aa8bd");
  const faint = css("--text-faint", "#64728a");
  const grid = css("--border", "#26303f");
  const gridS = css("--border-strong", "#33415a");

  ctx.clearRect(0, 0, W, H);
  ctx.font = '11px "IBM Plex Mono", monospace';
  const widestYTick = Math.max(
    0,
    ...(p.yTicks ?? []).map((tick) => ctx.measureText(tick.label).width),
  );
  const lastXTickWidth = p.xTicks?.length
    ? ctx.measureText(p.xTicks[p.xTicks.length - 1].label).width
    : 0;
  // A heat plot reserves the right margin for its density colorbar.
  // Tick text and axis titles have separate lanes.  The old fixed 58×44
  // margin let longer scientific-notation ticks collide with both titles.
  const m = {
    l: Math.max(p.yLabel ? 76 : 42, Math.ceil(widestYTick) + (p.yLabel ? 38 : 16)),
    r: p.heat ? 74 : Math.max(22, Math.ceil(lastXTickWidth / 2) + 7),
    t: p.title ? 30 : 12,
    b: p.xLabel ? 58 : 38,
  };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;

  const logx = p.xScale === "log";
  const tx = (v: number) => {
    const [a, b] = p.xDomain;
    const f = logx
      ? (Math.log10(v) - Math.log10(a)) / (Math.log10(b) - Math.log10(a))
      : (v - a) / (b - a);
    return m.l + f * iw;
  };
  const ty = (v: number) => {
    const [a, b] = p.yDomain;
    return m.t + (1 - (v - a) / (b - a)) * ih;
  };

  // title
  if (p.title) {
    ctx.fillStyle = ink;
    ctx.font = '600 13px "IBM Plex Sans", sans-serif';
    ctx.textAlign = "left";
    ctx.fillText(p.title, m.l, 16);
  }

  // density heatmap (under everything) — clipped to the plot area
  let heatNorm: { zmin: number; zmax: number; denom: number; scale: "log" | "linear";
    color: (t: number) => string; ticks?: Tick[]; label?: string } | null = null;
  if (p.heat && p.heat.z.length && p.heat.z[0]?.length) {
    const { z, xEdges, yEdges } = p.heat;
    const color = p.heat.color ?? viridis;
    const scale = p.heat.scale ?? "log";
    let zmin = p.heat.min ?? Infinity, zmax = p.heat.max ?? -Infinity;
    for (const row of z) for (const v of row) if (isFinite(v)) {
      if (p.heat.min == null && v < zmin) zmin = v;
      if (p.heat.max == null && v > zmax) zmax = v;
    }
    if (!isFinite(zmin)) zmin = 0;
    if (!isFinite(zmax)) zmax = scale === "log" ? 1 : zmin + 1;
    const denom = scale === "log" ? Math.log10(Math.max(zmax, 2)) : Math.max(zmax - zmin, 1e-12);
    heatNorm = { zmin, zmax, denom, scale, color, ticks: p.heat.colorTicks, label: p.heat.colorLabel };
    ctx.save();
    ctx.beginPath(); ctx.rect(m.l, m.t, iw, ih); ctx.clip();
    for (let i = 0; i < z.length; i++) {
      const x0 = tx(xEdges[i]), x1 = tx(xEdges[i + 1]);
      for (let j = 0; j < z[i].length; j++) {
        const c = z[i][j];
        if (!isFinite(c) || (scale === "log" && !(c > 0))) continue;
        const t = scale === "log"
          ? (denom > 0 ? Math.min(1, Math.log10(c) / denom) : 1)
          : Math.max(0, Math.min(1, (c - zmin) / denom));
        const y0 = ty(yEdges[j]), y1 = ty(yEdges[j + 1]);
        ctx.fillStyle = color(t);
        // +0.6 overlap kills seams between adjacent cells on retina.
        ctx.fillRect(x0, Math.min(y0, y1), x1 - x0 + 0.6, Math.abs(y1 - y0) + 0.6);
      }
    }
    ctx.restore();
  }

  // Trust/selection bands sit below the grid and data. Optional hatching keeps
  // unsupported regions legible without relying on colour alone.
  for (const band of p.bands ?? []) {
    const a = band.axis === "x" ? tx(band.from) : ty(band.from);
    const b = band.axis === "x" ? tx(band.to) : ty(band.to);
    const x = band.axis === "x" ? Math.min(a, b) : m.l;
    const y = band.axis === "y" ? Math.min(a, b) : m.t;
    const w = band.axis === "x" ? Math.abs(b - a) : iw;
    const h = band.axis === "y" ? Math.abs(b - a) : ih;
    ctx.save();
    ctx.beginPath(); ctx.rect(m.l, m.t, iw, ih); ctx.clip();
    ctx.fillStyle = band.color;
    ctx.globalAlpha = band.alpha ?? 0.08;
    ctx.fillRect(x, y, w, h);
    if (band.hatch && w > 0 && h > 0) {
      ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();
      ctx.strokeStyle = band.color;
      ctx.globalAlpha = Math.max(0.15, (band.alpha ?? 0.08) * 2.2);
      ctx.lineWidth = 0.8;
      const step = 9;
      for (let offset = -h; offset < w; offset += step) {
        ctx.beginPath();
        ctx.moveTo(x + offset, y + h);
        ctx.lineTo(x + offset + h, y);
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  // grid + ticks
  ctx.font = '11px "IBM Plex Mono", monospace';
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const t of p.xTicks ?? []) {
    const x = tx(t.v);
    if (x < m.l - 1 || x > W - m.r + 1) continue;
    ctx.strokeStyle = grid;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, m.t); ctx.lineTo(x, m.t + ih); ctx.stroke();
    ctx.fillStyle = faint;
    ctx.fillText(t.label, x, m.t + ih + 7);
  }
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const t of p.yTicks ?? []) {
    const y = ty(t.v);
    ctx.strokeStyle = grid;
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + iw, y); ctx.stroke();
    if (t.label) { ctx.fillStyle = faint; ctx.fillText(t.label, m.l - 8, y); }
  }

  // guide lines
  for (const g of p.guides ?? []) {
    ctx.save();
    ctx.globalAlpha = g.alpha ?? 1;
    ctx.strokeStyle = g.color ?? gridS;
    ctx.lineWidth = g.width ?? 1;
    ctx.setLineDash(g.dash ?? []);
    ctx.beginPath();
    if (g.axis === "x") { const x = tx(g.v); ctx.moveTo(x, m.t); ctx.lineTo(x, m.t + ih); }
    else { const y = ty(g.v); ctx.moveTo(m.l, y); ctx.lineTo(m.l + iw, y); }
    ctx.stroke();
    ctx.restore();
  }

  // Short in-plot labels make scientific limits identifiable without forcing
  // the caller to duplicate them in a separate legend.
  for (const band of (p.bands ?? []).filter((candidate) => candidate.label)) {
    const a = band.axis === "x" ? tx(band.from) : ty(band.from);
    const b = band.axis === "x" ? tx(band.to) : ty(band.to);
    ctx.save();
    ctx.beginPath(); ctx.rect(m.l, m.t, iw, ih); ctx.clip();
    ctx.font = '600 9px "IBM Plex Mono", monospace';
    ctx.fillStyle = band.color;
    ctx.globalAlpha = 0.9;
    ctx.textBaseline = "top";
    if (band.axis === "x") {
      ctx.textAlign = "center";
      ctx.fillText(band.label!, (a + b) / 2, m.t + 7);
    } else {
      ctx.textAlign = "left";
      ctx.fillText(band.label!, m.l + 7, Math.min(a, b) + 7);
    }
    ctx.restore();
  }
  for (const g of (p.guides ?? []).filter((candidate) => candidate.label)) {
    const side = g.labelSide ?? "after";
    ctx.save();
    ctx.beginPath(); ctx.rect(m.l, m.t, iw, ih); ctx.clip();
    ctx.font = '600 9px "IBM Plex Mono", monospace';
    ctx.fillStyle = g.color ?? gridS;
    ctx.globalAlpha = Math.max(0.8, g.alpha ?? 1);
    ctx.textBaseline = "top";
    if (g.axis === "x") {
      const x = tx(g.v);
      ctx.textAlign = side === "before" ? "right" : "left";
      ctx.fillText(g.label!, x + (side === "before" ? -6 : 6), m.t + 20);
    } else {
      const y = ty(g.v);
      ctx.textAlign = "left";
      ctx.fillText(g.label!, m.l + 7, y + (side === "before" ? -15 : 5));
    }
    ctx.restore();
  }

  // clip to plot area for series
  ctx.save();
  ctx.beginPath(); ctx.rect(m.l, m.t, iw, ih); ctx.clip();
  for (const s of p.series) {
    if (!s.low || !s.high || s.mode === "histogram" || s.mode === "scatter") continue;
    ctx.fillStyle = s.color;
    ctx.globalAlpha = s.fillAlpha ?? 0.1;
    let start = 0;
    while (start < s.x.length) {
      while (start < s.x.length && (
        s.low[start] == null || !isFinite(s.low[start]!)
        || s.high[start] == null || !isFinite(s.high[start]!)
      )) start++;
      if (start >= s.x.length) break;
      let end = start + 1;
      while (end < s.x.length
        && s.low[end] != null && isFinite(s.low[end]!)
        && s.high[end] != null && isFinite(s.high[end]!)) end++;
      ctx.beginPath();
      ctx.moveTo(tx(s.x[start]), ty(s.high[start]!));
      for (let i = start + 1; i < end; i++) {
        ctx.lineTo(tx(s.x[i]), ty(s.high[i]!));
      }
      for (let i = end - 1; i >= start; i--) {
        ctx.lineTo(tx(s.x[i]), ty(s.low[i]!));
      }
      ctx.closePath();
      ctx.fill();
      start = end;
    }
  }
  for (const s of p.series) {
    ctx.globalAlpha = s.alpha ?? 1;
    ctx.strokeStyle = s.color;
    ctx.fillStyle = s.color;
    ctx.lineWidth = s.width ?? 2;
    ctx.setLineDash(s.dash ?? []);
    ctx.lineJoin = "round"; ctx.lineCap = "round";
    if (s.mode === "histogram") {
      const baseline = ty(Math.max(0, p.yDomain[0]));
      for (let i = 0; i < s.x.length; i++) {
        const yv = s.y[i];
        if (yv == null || !isFinite(yv)) continue;
        const left = i > 0
          ? (s.x[i - 1] + s.x[i]) / 2
          : s.x[i] - ((s.x[i + 1] ?? p.xDomain[1]) - s.x[i]) / 2;
        const right = i < s.x.length - 1
          ? (s.x[i] + s.x[i + 1]) / 2
          : s.x[i] + (s.x[i] - (s.x[i - 1] ?? p.xDomain[0])) / 2;
        const x0 = tx(left), x1 = tx(right), top = ty(yv);
        const inset = Math.min(0.6, Math.max(0, (x1 - x0) * 0.08));
        const barX = x0 + inset;
        const barW = Math.max(0, x1 - x0 - 2 * inset);
        const barH = baseline - top;
        ctx.globalAlpha = s.fillAlpha ?? 0.18;
        ctx.fillRect(barX, top, barW, barH);
        if (s.hatch && barW > 0 && barH > 0) {
          ctx.save();
          ctx.beginPath();
          ctx.rect(barX, top, barW, barH);
          ctx.clip();
          ctx.globalAlpha = Math.max(0.26, s.alpha ?? 1);
          ctx.lineWidth = 0.8;
          ctx.setLineDash([]);
          const step = 6;
          for (let x = barX - barH; x < barX + barW; x += step) {
            ctx.beginPath();
            ctx.moveTo(x, baseline);
            ctx.lineTo(x + barH, top);
            ctx.stroke();
          }
          ctx.restore();
        }
        ctx.globalAlpha = s.alpha ?? 1;
        ctx.setLineDash(s.dash ?? []);
        ctx.strokeRect(barX, top, barW, barH);
      }
      continue;
    }
    if (s.mode === "scatter") {
      ctx.setLineDash([]);
      for (let i = 0; i < s.x.length; i++) {
        const yv = s.y[i];
        if (yv == null || !isFinite(yv) || !isFinite(s.x[i])) continue;
        ctx.lineWidth = Math.max(1.2, s.width ?? 1.5);
        drawMarker(
          ctx,
          tx(s.x[i]),
          ty(yv),
          (s.width ?? 2) + (s.marker === "diamond" ? 2 : 0.9),
          s.marker,
        );
      }
      continue;
    }
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < s.x.length; i++) {
      const yv = s.y[i];
      if (yv == null || !isFinite(yv)) { started = false; continue; }
      const X = tx(s.x[i]), Y = ty(yv);
      if (!started) { ctx.moveTo(X, Y); started = true; } else ctx.lineTo(X, Y);
    }
    ctx.stroke();
    if (s.dots) {
      ctx.setLineDash([]);
      const markerEvery = Math.max(1, s.markerEvery ?? 1);
      for (let i = 0; i < s.x.length; i++) {
        const yv = s.y[i];
        if (yv == null || !isFinite(yv)
          || (i % markerEvery !== 0 && i !== s.x.length - 1)) continue;
        ctx.lineWidth = Math.max(1.2, s.width ?? 1.5);
        drawMarker(
          ctx,
          tx(s.x[i]),
          ty(yv),
          (s.width ?? 2) + (s.marker === "diamond" ? 1.7 : 0.6),
          s.marker,
        );
      }
    }
  }

  // Labels sit directly on their line, matching conventional labeled
  // scientific contours. Erasing the short line segment beneath the text
  // exposes the card background and therefore works in both page themes.
  for (const s of p.series.filter((candidate) => candidate.label)) {
    const points = s.x.flatMap((x, index) => {
      const y = s.y[index];
      return Number.isFinite(x) && y != null && Number.isFinite(y)
        ? [{ x: tx(x), y: ty(y) }]
        : [];
    });
    if (points.length < 2) continue;
    const lengths = points.slice(1).map((point, index) => Math.hypot(
      point.x - points[index].x,
      point.y - points[index].y,
    ));
    const totalLength = lengths.reduce((total, length) => total + length, 0);
    if (!(totalLength > 0)) continue;
    const target = totalLength * Math.max(
      0.05, Math.min(0.95, s.labelAt ?? 0.55),
    );
    let cumulative = 0;
    let segment = lengths.length - 1;
    for (let index = 0; index < lengths.length; index++) {
      if (cumulative + lengths[index] >= target) {
        segment = index;
        break;
      }
      cumulative += lengths[index];
    }
    const start = points[segment], end = points[segment + 1];
    const fraction = lengths[segment] > 0
      ? (target - cumulative) / lengths[segment]
      : 0.5;
    const x = start.x + fraction * (end.x - start.x);
    const y = start.y + fraction * (end.y - start.y);
    let angle = Math.atan2(end.y - start.y, end.x - start.x);
    if (angle > Math.PI / 2) angle -= Math.PI;
    if (angle < -Math.PI / 2) angle += Math.PI;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.font = '600 9px "IBM Plex Mono", monospace';
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const width = ctx.measureText(s.label!).width;
    ctx.globalCompositeOperation = "destination-out";
    ctx.globalAlpha = 1;
    ctx.fillRect(-width / 2 - 3, -6, width + 6, 12);
    ctx.globalCompositeOperation = "source-over";
    ctx.fillStyle = s.color;
    ctx.fillText(s.label!, 0, 0);
    ctx.restore();
  }
  ctx.restore();

  // picked-cell outline (heat plots): highlight the back-traced histogram cell.
  if (p.heat && p.highlight) {
    const { xEdges, yEdges } = p.heat;
    const { i, j } = p.highlight;
    if (i >= 0 && i < xEdges.length - 1 && j >= 0 && j < yEdges.length - 1) {
      const x0 = tx(xEdges[i]), x1 = tx(xEdges[i + 1]);
      const y0 = ty(yEdges[j]), y1 = ty(yEdges[j + 1]);
      ctx.save();
      ctx.beginPath(); ctx.rect(m.l, m.t, iw, ih); ctx.clip();
      ctx.strokeStyle = css("--text", "#e5edf7");
      ctx.lineWidth = 1.5;
      ctx.setLineDash([]);
      const pad = 1.5;
      ctx.strokeRect(Math.min(x0, x1) - pad, Math.min(y0, y1) - pad,
        Math.abs(x1 - x0) + 2 * pad, Math.abs(y1 - y0) + 2 * pad);
      ctx.restore();
    }
  }

  // axis frame
  ctx.globalAlpha = 1;
  ctx.setLineDash([]);
  ctx.strokeStyle = gridS;
  ctx.lineWidth = 1;
  ctx.strokeRect(m.l, m.t, iw, ih);

  // density colorbar (heat plots) — a log-count viridis strip in the right
  // margin, ticked at decades of pixel count so a cell's colour is readable.
  if (heatNorm) {
    const bx = W - m.r + 14, bw = 12, bt = m.t, bh = ih;
    const steps = 48;
    for (let s = 0; s < steps; s++) {
      const t = s / (steps - 1);                 // 0 (bottom) … 1 (top)
      ctx.fillStyle = heatNorm.color(t);
      const yy = bt + (1 - t) * bh;
      ctx.fillRect(bx, yy - bh / steps - 0.5, bw, bh / steps + 1);
    }
    ctx.strokeStyle = gridS; ctx.lineWidth = 1; ctx.strokeRect(bx, bt, bw, bh);
    // Default density ticks are decades; value heatmaps provide explicit ticks.
    ctx.font = '10px "IBM Plex Mono", monospace';
    ctx.fillStyle = faint; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.strokeStyle = faint;
    const colorTicks = heatNorm.ticks ?? (() => {
      const ticks: Tick[] = [];
      for (let e = 0; 10 ** e <= heatNorm.zmax + 0.5; e++)
        ticks.push({ v: 10 ** e, label: e === 0 ? "1" : `10${supN(e)}` });
      return ticks;
    })();
    for (const tick of colorTicks) {
      const t = heatNorm.scale === "log"
        ? (heatNorm.denom > 0 ? Math.log10(Math.max(tick.v, 1)) / heatNorm.denom : 1)
        : (tick.v - heatNorm.zmin) / heatNorm.denom;
      if (t < 0 || t > 1) continue;
      const yy = bt + (1 - t) * bh;
      ctx.beginPath(); ctx.moveTo(bx + bw, yy); ctx.lineTo(bx + bw + 3, yy); ctx.stroke();
      ctx.fillText(tick.label, bx + bw + 5, yy);
    }
    ctx.save();
    ctx.translate(bx + bw + 34, bt + bh / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = dim; ctx.font = '500 11px "IBM Plex Mono", monospace';
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(heatNorm.label ?? "pixels / cell", 0, 0);
    ctx.restore();
  }

  // axis labels
  ctx.fillStyle = dim;
  ctx.font = '500 11.5px "IBM Plex Mono", monospace';
  if (p.xLabel) { ctx.textAlign = "center"; ctx.textBaseline = "bottom"; ctx.fillText(p.xLabel, m.l + iw / 2, H - 6); }
  if (p.yLabel) {
    ctx.save(); ctx.translate(14, m.t + ih / 2); ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillText(p.yLabel, 0, 0); ctx.restore();
  }
  return { m, iw, ih, logx, xDomain: p.xDomain, yDomain: p.yDomain };
}

/* Superscript a small non-negative integer for the colorbar decade labels. */
const SUP_DIGITS = ["⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹"];
function supN(n: number): string {
  return String(n).split("").map((d) => SUP_DIGITS[+d] ?? d).join("");
}

export function Legend({ items }: {
  items: {
    label: string;
    color: string;
    dash?: boolean;
    histogram?: boolean;
    filled?: boolean;
    hatch?: boolean;
    line?: boolean;
    marker?: "filled" | "ring" | "diamond";
  }[];
}) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 16px", padding: "10px 2px 2px", fontSize: 12 }}>
      {items.map((it, i) => (
        <span key={i} style={{ display: "inline-flex", alignItems: "center", gap: 7, color: "var(--text-dim)" }}>
          <span style={it.line ? {
            position: "relative",
            width: 22,
            height: 12,
          } : it.marker ? {
            width: 9,
            height: 9,
            boxSizing: "border-box",
            borderRadius: it.marker === "diamond" ? 1 : "50%",
            border: `2px solid ${it.color}`,
            background: it.marker === "filled" ? it.color : "transparent",
            transform: it.marker === "diamond" ? "rotate(45deg)" : undefined,
          } : it.histogram ? {
            width: 18,
            height: 9,
            boxSizing: "border-box",
            border: `2px ${it.dash ? "dashed" : "solid"} ${it.color}`,
            background: it.hatch
              ? `repeating-linear-gradient(135deg, transparent 0 3px, ${it.color} 3px 4px)`
              : it.filled ? it.color : "transparent",
            opacity: it.filled && !it.hatch ? 0.55 : 1,
          } : {
            width: 18, height: 0,
            borderTop: `${it.dash ? "2.5px dashed" : "3px solid"} ${it.color}`,
          }}>
            {it.line && <>
              <span style={{
                position: "absolute",
                left: 0,
                right: 0,
                top: 5,
                borderTop: `${it.dash ? "2.5px dashed" : "3px solid"} ${it.color}`,
              }} />
              <span style={{
                position: "absolute",
                left: 8,
                top: 2,
                width: 7,
                height: 7,
                boxSizing: "border-box",
                borderRadius: it.marker === "diamond" ? 1 : "50%",
                border: `2px solid ${it.color}`,
                background: it.marker === "filled" ? it.color : "var(--surface-1)",
                transform: it.marker === "diamond" ? "rotate(45deg)" : undefined,
              }} />
            </>}
          </span>
          {it.label}
        </span>
      ))}
    </div>
  );
}
