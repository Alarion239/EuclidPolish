/* Plot — one reusable canvas line-chart. Every figure in the app is expressed
   as {domain, ticks, series, guides}; no page hand-rolls axes again. Retina
   crisp, resizes with its container, themed from CSS tokens. */
import { useEffect, useRef, type MouseEvent } from "react";
import { viridis } from "../colors";

export type Series = {
  x: number[];
  y: (number | null)[];
  color: string;
  width?: number;
  dash?: number[];
  dots?: boolean;
  alpha?: number;
};
export type Guide = {
  axis: "x" | "y";
  v: number;
  color?: string;
  dash?: number[];
  width?: number;
  alpha?: number;
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
  guides?: Guide[];
  heat?: Heat;
  /* When set (heat plots only) a click reports the histogram cell under the
     cursor; `highlight` outlines a picked cell. Powers click-to-inspect. */
  onHeatClick?: (cell: Cell) => void;
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
    if (!p.onHeatClick || !g || !heat) return;
    const r = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    const [xa, xb] = g.xDomain, [ya, yb] = g.yDomain;
    const fx = (px - g.m.l) / g.iw, fy = 1 - (py - g.m.t) / g.ih;
    if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return;
    const dataX = g.logx ? 10 ** (Math.log10(xa) + fx * (Math.log10(xb) - Math.log10(xa))) : xa + fx * (xb - xa);
    const dataY = ya + fy * (yb - ya);
    const i = binOf(heat.xEdges, dataX), j = binOf(heat.yEdges, dataY);
    if (i >= 0 && j >= 0) p.onHeatClick({ i, j });
  };

  return (
    <div ref={wrap} style={{ width: "100%" }}>
      <canvas ref={canvas} onClick={onClick}
        style={p.onHeatClick && p.heat ? { cursor: "crosshair" } : undefined} />
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
  // A heat plot reserves the right margin for its density colorbar.
  const m = { l: 58, r: p.heat ? 74 : 16, t: p.title ? 30 : 12, b: 44 };
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
  let heatNorm: { zmax: number; denom: number; color: (t: number) => string } | null = null;
  if (p.heat && p.heat.z.length && p.heat.z[0]?.length) {
    const { z, xEdges, yEdges } = p.heat;
    const color = p.heat.color ?? viridis;
    let zmax = p.heat.max ?? 0;
    if (!p.heat.max) for (const row of z) for (const v of row) if (v > zmax) zmax = v;
    const denom = Math.log10(Math.max(zmax, 2));
    heatNorm = { zmax, denom, color };
    ctx.save();
    ctx.beginPath(); ctx.rect(m.l, m.t, iw, ih); ctx.clip();
    for (let i = 0; i < z.length; i++) {
      const x0 = tx(xEdges[i]), x1 = tx(xEdges[i + 1]);
      for (let j = 0; j < z[i].length; j++) {
        const c = z[i][j];
        if (!(c > 0)) continue;
        const t = denom > 0 ? Math.min(1, Math.log10(c) / denom) : 1;
        const y0 = ty(yEdges[j]), y1 = ty(yEdges[j + 1]);
        ctx.fillStyle = color(t);
        // +0.6 overlap kills seams between adjacent cells on retina.
        ctx.fillRect(x0, Math.min(y0, y1), x1 - x0 + 0.6, Math.abs(y1 - y0) + 0.6);
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

  // clip to plot area for series
  ctx.save();
  ctx.beginPath(); ctx.rect(m.l, m.t, iw, ih); ctx.clip();
  for (const s of p.series) {
    ctx.globalAlpha = s.alpha ?? 1;
    ctx.strokeStyle = s.color;
    ctx.fillStyle = s.color;
    ctx.lineWidth = s.width ?? 2;
    ctx.setLineDash(s.dash ?? []);
    ctx.lineJoin = "round"; ctx.lineCap = "round";
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
      for (let i = 0; i < s.x.length; i++) {
        const yv = s.y[i];
        if (yv == null || !isFinite(yv)) continue;
        ctx.beginPath(); ctx.arc(tx(s.x[i]), ty(yv), (s.width ?? 2) + 0.6, 0, 7); ctx.fill();
      }
    }
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
    // decade ticks: count 10^e sits at t = e / denom on the log-norm strip.
    ctx.font = '10px "IBM Plex Mono", monospace';
    ctx.fillStyle = faint; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.strokeStyle = faint;
    for (let e = 0; 10 ** e <= heatNorm.zmax + 0.5; e++) {
      const t = heatNorm.denom > 0 ? e / heatNorm.denom : 1;
      if (t < 0 || t > 1) continue;
      const yy = bt + (1 - t) * bh;
      ctx.beginPath(); ctx.moveTo(bx + bw, yy); ctx.lineTo(bx + bw + 3, yy); ctx.stroke();
      ctx.fillText(e === 0 ? "1" : `10${supN(e)}`, bx + bw + 5, yy);
    }
    ctx.save();
    ctx.translate(bx + bw + 34, bt + bh / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = dim; ctx.font = '500 11px "IBM Plex Mono", monospace';
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText("pixels / cell", 0, 0);
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

export function Legend({ items }: { items: { label: string; color: string; dash?: boolean }[] }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 16px", padding: "10px 2px 2px", fontSize: 12 }}>
      {items.map((it, i) => (
        <span key={i} style={{ display: "inline-flex", alignItems: "center", gap: 7, color: "var(--text-dim)" }}>
          <span style={{
            width: 18, height: 0,
            borderTop: `${it.dash ? "2.5px dashed" : "3px solid"} ${it.color}`,
          }} />
          {it.label}
        </span>
      ))}
    </div>
  );
}
