/* Plot — one reusable canvas line-chart. Every figure in the app is expressed
   as {domain, ticks, series, guides}; no page hand-rolls axes again. Retina
   crisp, resizes with its container, themed from CSS tokens. */
import { useEffect, useRef } from "react";

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
  height?: number;      /* fixed px; omit → aspect-driven */
  aspect?: number;      /* height = width * aspect (default 0.5) */
};

function css(name: string, fallback: string) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

export default function Plot(p: PlotProps) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);

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
      render(ctx, cssW, cssH, p);
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(el);
    return () => ro.disconnect();
  });

  return (
    <div ref={wrap} style={{ width: "100%" }}>
      <canvas ref={canvas} />
    </div>
  );
}

function render(ctx: CanvasRenderingContext2D, W: number, H: number, p: PlotProps) {
  const ink = css("--text", "#e5edf7");
  const dim = css("--text-dim", "#9aa8bd");
  const faint = css("--text-faint", "#64728a");
  const grid = css("--border", "#26303f");
  const gridS = css("--border-strong", "#33415a");

  ctx.clearRect(0, 0, W, H);
  const m = { l: 58, r: 16, t: p.title ? 30 : 12, b: 44 };
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

  // axis frame
  ctx.globalAlpha = 1;
  ctx.setLineDash([]);
  ctx.strokeStyle = gridS;
  ctx.lineWidth = 1;
  ctx.strokeRect(m.l, m.t, iw, ih);

  // axis labels
  ctx.fillStyle = dim;
  ctx.font = '500 11.5px "IBM Plex Mono", monospace';
  if (p.xLabel) { ctx.textAlign = "center"; ctx.textBaseline = "bottom"; ctx.fillText(p.xLabel, m.l + iw / 2, H - 6); }
  if (p.yLabel) {
    ctx.save(); ctx.translate(14, m.t + ih / 2); ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillText(p.yLabel, 0, 0); ctx.restore();
  }
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
