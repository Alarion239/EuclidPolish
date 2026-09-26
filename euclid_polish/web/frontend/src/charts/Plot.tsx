/* Plot v2 — the one reusable canvas chart. Every figure in the app is
   {domain, ticks, bands, series, guides, heat}; no page hand-rolls axes.

   v2 adds, on top of the unchanged v1 props and look: hover crosshair +
   tooltip (nearest series + every line at that x), legend click-to-toggle
   and hover-emphasis, box-zoom / wheel-zoom / pan / reset (mouse and
   keyboard), log y, linked cursors (`syncKey`), PNG/CSV export, an ARIA
   figure with a text summary, and redraws only when the drawn inputs change
   (structural compare, so inline `series={[…]}` literals are free).
   The pure maths lives in plotModel.ts. */
import {
  useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState,
  type CSSProperties, type FocusEvent, type KeyboardEvent, type MouseEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { viridis } from "../colors";
import { useResolvedTheme } from "../state/prefs";
import { downloadBlob, downloadText, safeFileName } from "../ui/download";
import { Icon } from "../ui/icons";
import {
  axis, clampView, drawable, formatValue, nearestIndex, nearestPoint, panDomain, readoutAt, sameInputs,
  sameValue, seriesKey, seriesName, seriesToCSV, stepDrawable, tooltipReadout, viewTicks, zoomDomain, type Hit,
  type PlotGeometry,
} from "./plotModel";
import type { Band, Heat, LegendItem, PlotProps, PlotView, Series, Tick } from "./types";
import "./plot.css";

export type {
  AxisScale, Band, Cell, Guide, Heat, LegendItem, PlotProps, PlotView, Series, Tick,
} from "./types";

/* ─── theme + text measurement ────────────────────────────────────────────── */

function cssVar(name: string, fallback: string): string {
  if (typeof document === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

type Fonts = { mono: string; sans: string };
const fonts = (): Fonts => ({
  mono: cssVar("--font-mono", "\"IBM Plex Mono\", ui-monospace, monospace"),
  sans: cssVar("--font-sans", "\"IBM Plex Sans\", system-ui, sans-serif"),
});

let measureCtx: CanvasRenderingContext2D | null | undefined;
function measurer(): (text: string, font: string) => number {
  if (measureCtx === undefined) {
    measureCtx = typeof document === "undefined" ? null : document.createElement("canvas").getContext("2d");
  }
  const ctx = measureCtx;
  if (!ctx) return (text) => text.length * 6.6;   // no canvas (tests / SSR): monospace estimate
  return (text, font) => { ctx.font = font; return ctx.measureText(text).width; };
}

/* ─── layout ──────────────────────────────────────────────────────────────── */

type Layout = PlotGeometry & { W: number; H: number };

function computeLayout(W: number, H: number, p: PlotProps, f: Fonts): Layout {
  const measure = measurer();
  const tickFont = `11px ${f.mono}`;
  const widestYTick = Math.max(0, ...(p.yTicks ?? []).map((t) => measure(t.label, tickFont)));
  const lastXTickWidth = p.xTicks?.length ? measure(p.xTicks[p.xTicks.length - 1].label, tickFont) : 0;
  // Tick text and axis titles have separate lanes; a heat plot reserves the
  // right margin for its colorbar (v1 margins, unchanged).
  const m = {
    l: Math.max(p.yLabel ? 76 : 42, Math.ceil(widestYTick) + (p.yLabel ? 38 : 16)),
    r: p.heat ? 74 : Math.max(22, Math.ceil(lastXTickWidth / 2) + 7),
    t: p.title ? 30 : 12,
    b: p.xLabel ? 58 : 38,
  };
  const iw = Math.max(1, W - m.l - m.r), ih = Math.max(1, H - m.t - m.b);
  return {
    W, H, m, iw, ih,
    x: axis(p.xDomain, p.xScale ?? "linear", m.l, m.l + iw),
    y: axis(p.yDomain, p.yScale ?? "linear", m.t + ih, m.t),
  };
}

/* ─── painting (v1 renderer, now scale-aware) ─────────────────────────────── */

function drawMarker(ctx: CanvasRenderingContext2D, x: number, y: number, radius: number,
  marker: Series["marker"] = "filled") {
  ctx.beginPath();
  if (marker === "diamond") {
    ctx.moveTo(x, y - radius); ctx.lineTo(x + radius, y); ctx.lineTo(x, y + radius); ctx.lineTo(x - radius, y);
    ctx.closePath(); ctx.stroke();
  } else {
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    if (marker === "ring") ctx.stroke(); else ctx.fill();
  }
}

const SUP_DIGITS = ["⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹"];
const supN = (n: number) => String(n).split("").map((d) => SUP_DIGITS[+d] ?? d).join("");
const fin = (v: number | null | undefined): v is number => v != null && Number.isFinite(v);

type PaintOpts = { background?: string; fonts: Fonts; dim?: ReadonlySet<number> };

function paint(ctx: CanvasRenderingContext2D, L: Layout, p: PlotProps, series: Series[], o: PaintOpts) {
  const { W, H, m, iw, ih } = L;
  const tx = L.x.toPx, ty = L.y.toPx;
  const okx = (v: number) => Number.isFinite(tx(v)), oky = (v: number) => Number.isFinite(ty(v));
  const ink = cssVar("--text", "#17202e");
  const dim = cssVar("--text-dim", "#4a566a");
  const faint = cssVar("--text-faint", "#5f6a79");
  const grid = cssVar("--border", "#e2e2e6");
  const gridS = cssVar("--border-strong", "#c9c9d0");
  const mono = o.fonts.mono, sans = o.fonts.sans;
  const clipPlot = () => { ctx.beginPath(); ctx.rect(m.l, m.t, iw, ih); ctx.clip(); };

  if (o.background) { ctx.fillStyle = o.background; ctx.fillRect(0, 0, W, H); }
  else ctx.clearRect(0, 0, W, H);

  if (p.title) {
    ctx.fillStyle = ink;
    ctx.font = `600 13px ${sans}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillText(p.title, m.l, 16);
  }

  // density heatmap (under everything)
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
    ctx.save(); clipPlot();
    for (let i = 0; i < z.length; i++) {
      const x0 = tx(xEdges[i]), x1 = tx(xEdges[i + 1]);
      if (!Number.isFinite(x0) || !Number.isFinite(x1)) continue;
      for (let j = 0; j < z[i].length; j++) {
        const c = z[i][j];
        if (!isFinite(c) || (scale === "log" && !(c > 0))) continue;
        const t = scale === "log"
          ? (denom > 0 ? Math.min(1, Math.log10(c) / denom) : 1)
          : Math.max(0, Math.min(1, (c - zmin) / denom));
        const y0 = ty(yEdges[j]), y1 = ty(yEdges[j + 1]);
        if (!Number.isFinite(y0) || !Number.isFinite(y1)) continue;
        ctx.fillStyle = color(t);
        // +0.6 overlap kills seams between adjacent cells on retina.
        ctx.fillRect(x0, Math.min(y0, y1), x1 - x0 + 0.6, Math.abs(y1 - y0) + 0.6);
      }
    }
    ctx.restore();
  }

  // bands (below grid and data); hatching keeps them legible without colour
  const bandRect = (band: Band) => {
    const a = band.axis === "x" ? tx(band.from) : ty(band.from);
    const b = band.axis === "x" ? tx(band.to) : ty(band.to);
    if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
    return {
      a, b,
      x: band.axis === "x" ? Math.min(a, b) : m.l, y: band.axis === "y" ? Math.min(a, b) : m.t,
      w: band.axis === "x" ? Math.abs(b - a) : iw, h: band.axis === "y" ? Math.abs(b - a) : ih,
    };
  };
  for (const band of p.bands ?? []) {
    const r = bandRect(band);
    if (!r) continue;
    ctx.save(); clipPlot();
    ctx.fillStyle = band.color;
    ctx.globalAlpha = band.alpha ?? 0.08;
    ctx.fillRect(r.x, r.y, r.w, r.h);
    if (band.hatch && r.w > 0 && r.h > 0) {
      ctx.beginPath(); ctx.rect(r.x, r.y, r.w, r.h); ctx.clip();
      ctx.strokeStyle = band.color;
      ctx.globalAlpha = Math.max(0.15, (band.alpha ?? 0.08) * 2.2);
      ctx.lineWidth = 0.8;
      for (let offset = -r.h; offset < r.w; offset += 9) {
        ctx.beginPath(); ctx.moveTo(r.x + offset, r.y + r.h); ctx.lineTo(r.x + offset + r.h, r.y); ctx.stroke();
      }
    }
    ctx.restore();
  }

  // grid + ticks
  ctx.font = `11px ${mono}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const t of p.xTicks ?? []) {
    const x = tx(t.v);
    if (!Number.isFinite(x) || x < m.l - 1 || x > W - m.r + 1) continue;
    ctx.strokeStyle = grid; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, m.t); ctx.lineTo(x, m.t + ih); ctx.stroke();
    ctx.fillStyle = faint;
    ctx.fillText(t.label, x, m.t + ih + 7);
  }
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const t of p.yTicks ?? []) {
    const y = ty(t.v);
    if (!Number.isFinite(y) || y < m.t - 1 || y > m.t + ih + 1) continue;
    ctx.strokeStyle = grid;
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + iw, y); ctx.stroke();
    if (t.label) { ctx.fillStyle = faint; ctx.fillText(t.label, m.l - 8, y); }
  }

  // guide lines
  for (const g of p.guides ?? []) {
    const v = g.axis === "x" ? tx(g.v) : ty(g.v);
    if (!Number.isFinite(v)) continue;
    ctx.save(); clipPlot();
    ctx.globalAlpha = g.alpha ?? 1;
    ctx.strokeStyle = g.color ?? gridS;
    ctx.lineWidth = g.width ?? 1;
    ctx.setLineDash(g.dash ?? []);
    ctx.beginPath();
    if (g.axis === "x") { ctx.moveTo(v, m.t); ctx.lineTo(v, m.t + ih); }
    else { ctx.moveTo(m.l, v); ctx.lineTo(m.l + iw, v); }
    ctx.stroke();
    ctx.restore();
  }

  // in-plot band and guide labels
  for (const band of (p.bands ?? []).filter((b) => b.label)) {
    const r = bandRect(band);
    if (!r) continue;
    ctx.save(); clipPlot();
    ctx.font = `600 9px ${mono}`;
    ctx.fillStyle = band.color;
    ctx.globalAlpha = 0.9;
    ctx.textBaseline = "top";
    if (band.axis === "x") { ctx.textAlign = "center"; ctx.fillText(band.label!, (r.a + r.b) / 2, m.t + 7); }
    else { ctx.textAlign = "left"; ctx.fillText(band.label!, m.l + 7, Math.min(r.a, r.b) + 7); }
    ctx.restore();
  }
  for (const g of (p.guides ?? []).filter((c) => c.label)) {
    const side = g.labelSide ?? "after";
    const v = g.axis === "x" ? tx(g.v) : ty(g.v);
    if (!Number.isFinite(v)) continue;
    ctx.save(); clipPlot();
    ctx.font = `600 9px ${mono}`;
    ctx.fillStyle = g.color ?? gridS;
    ctx.globalAlpha = Math.max(0.8, g.alpha ?? 1);
    ctx.textBaseline = "top";
    if (g.axis === "x") {
      ctx.textAlign = side === "before" ? "right" : "left";
      ctx.fillText(g.label!, v + (side === "before" ? -6 : 6), m.t + 20);
    } else {
      ctx.textAlign = "left";
      ctx.fillText(g.label!, m.l + 7, v + (side === "before" ? -15 : 5));
    }
    ctx.restore();
  }

  // series (clipped to the plot area)
  ctx.save(); clipPlot();
  const alphaOf = (s: Series, si: number, base: number) => (o.dim?.has(si) ? base * 0.22 : base);
  series.forEach((s, si) => {
    if (!s.low || !s.high || s.mode === "histogram" || s.mode === "scatter") return;
    ctx.fillStyle = s.color;
    ctx.globalAlpha = alphaOf(s, si, s.fillAlpha ?? 0.1);
    const ok = (i: number) => fin(s.low![i]) && fin(s.high![i]) && okx(s.x[i]) && oky(s.low![i]!) && oky(s.high![i]!);
    let start = 0;
    while (start < s.x.length) {
      while (start < s.x.length && !ok(start)) start++;
      if (start >= s.x.length) break;
      let end = start + 1;
      while (end < s.x.length && ok(end)) end++;
      ctx.beginPath();
      ctx.moveTo(tx(s.x[start]), ty(s.high[start]!));
      for (let i = start + 1; i < end; i++) ctx.lineTo(tx(s.x[i]), ty(s.high[i]!));
      for (let i = end - 1; i >= start; i--) ctx.lineTo(tx(s.x[i]), ty(s.low[i]!));
      ctx.closePath(); ctx.fill();
      start = end;
    }
  });
  series.forEach((s, si) => {
    const baseAlpha = alphaOf(s, si, s.alpha ?? 1);
    ctx.globalAlpha = baseAlpha;
    ctx.strokeStyle = s.color;
    ctx.fillStyle = s.color;
    ctx.lineWidth = s.width ?? 2;
    ctx.setLineDash(s.dash ?? []);
    ctx.lineJoin = "round"; ctx.lineCap = "round";
    if (s.mode === "histogram") {
      const floor = L.y.domain[0];
      const baseline = ty(L.y.scale === "log" ? floor : Math.max(0, floor));
      for (let i = 0; i < s.x.length; i++) {
        const yv = s.y[i];
        if (!fin(yv) || !oky(yv)) continue;
        const left = i > 0 ? (s.x[i - 1] + s.x[i]) / 2 : s.x[i] - ((s.x[i + 1] ?? L.x.domain[1]) - s.x[i]) / 2;
        const right = i < s.x.length - 1 ? (s.x[i] + s.x[i + 1]) / 2 : s.x[i] + (s.x[i] - (s.x[i - 1] ?? L.x.domain[0])) / 2;
        const x0 = tx(left), x1 = tx(right), top = ty(yv);
        if (!Number.isFinite(x0) || !Number.isFinite(x1)) continue;
        const inset = Math.min(0.6, Math.max(0, (x1 - x0) * 0.08));
        const barX = x0 + inset, barW = Math.max(0, x1 - x0 - 2 * inset), barH = baseline - top;
        ctx.globalAlpha = alphaOf(s, si, s.fillAlpha ?? 0.18);
        ctx.fillRect(barX, top, barW, barH);
        if (s.hatch && barW > 0 && barH > 0) {
          ctx.save();
          ctx.beginPath(); ctx.rect(barX, top, barW, barH); ctx.clip();
          ctx.globalAlpha = alphaOf(s, si, Math.max(0.26, s.alpha ?? 1));
          ctx.lineWidth = 0.8;
          ctx.setLineDash([]);
          for (let x = barX - barH; x < barX + barW; x += 6) {
            ctx.beginPath(); ctx.moveTo(x, baseline); ctx.lineTo(x + barH, top); ctx.stroke();
          }
          ctx.restore();
        }
        ctx.globalAlpha = baseAlpha;
        ctx.setLineDash(s.dash ?? []);
        ctx.strokeRect(barX, top, barW, barH);
      }
      return;
    }
    if (s.mode === "scatter") {
      ctx.setLineDash([]);
      const cap = 3.5;
      if (s.errorLow && s.errorHigh) {
        ctx.lineWidth = Math.max(1, (s.width ?? 1.5) * 0.8);
        for (let i = 0; i < s.x.length; i++) {
          const low = s.errorLow[i], high = s.errorHigh[i];
          if (!fin(low) || !fin(high) || !fin(s.x[i]) || !okx(s.x[i])) continue;
          const x = tx(s.x[i]), yLow = ty(low), yHigh = ty(high);
          if (!Number.isFinite(yLow) || !Number.isFinite(yHigh)) continue;
          ctx.beginPath();
          ctx.moveTo(x, yLow); ctx.lineTo(x, yHigh);
          ctx.moveTo(x - cap, yLow); ctx.lineTo(x + cap, yLow);
          ctx.moveTo(x - cap, yHigh); ctx.lineTo(x + cap, yHigh);
          ctx.stroke();
        }
      }
      for (let i = 0; i < s.x.length; i++) {
        const yv = s.y[i];
        if (!fin(yv) || !fin(s.x[i]) || !okx(s.x[i]) || !oky(yv)) continue;
        ctx.lineWidth = Math.max(1.2, s.width ?? 1.5);
        drawMarker(ctx, tx(s.x[i]), ty(yv), (s.width ?? 2) + (s.marker === "diamond" ? 2 : 0.9), s.marker);
      }
      return;
    }
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < s.x.length; i++) {
      const yv = s.y[i];
      if (!fin(yv) || !okx(s.x[i]) || !oky(yv)) { started = false; continue; }
      const X = tx(s.x[i]), Y = ty(yv);
      if (!started) { ctx.moveTo(X, Y); started = true; } else ctx.lineTo(X, Y);
    }
    ctx.stroke();
    if (s.dots) {
      ctx.setLineDash([]);
      const every = Math.max(1, s.markerEvery ?? 1);
      for (let i = 0; i < s.x.length; i++) {
        const yv = s.y[i];
        if (!fin(yv) || !okx(s.x[i]) || !oky(yv) || (i % every !== 0 && i !== s.x.length - 1)) continue;
        ctx.lineWidth = Math.max(1.2, s.width ?? 1.5);
        drawMarker(ctx, tx(s.x[i]), ty(yv), (s.width ?? 2) + (s.marker === "diamond" ? 1.7 : 0.6), s.marker);
      }
    }
  });

  // contour-style labels sitting on their line (erase the segment beneath)
  series.forEach((s) => {
    if (!s.label) return;
    const points = s.x.flatMap((x, i) => {
      const y = s.y[i];
      return fin(x) && fin(y) && okx(x) && oky(y) ? [{ x: tx(x), y: ty(y) }] : [];
    });
    if (points.length < 2) return;
    const lengths = points.slice(1).map((pt, i) => Math.hypot(pt.x - points[i].x, pt.y - points[i].y));
    const total = lengths.reduce((a, b) => a + b, 0);
    if (!(total > 0)) return;
    const target = total * Math.max(0.05, Math.min(0.95, s.labelAt ?? 0.55));
    let cum = 0, seg = lengths.length - 1;
    for (let i = 0; i < lengths.length; i++) {
      if (cum + lengths[i] >= target) { seg = i; break; }
      cum += lengths[i];
    }
    const a = points[seg], b = points[seg + 1];
    const f = lengths[seg] > 0 ? (target - cum) / lengths[seg] : 0.5;
    const x = a.x + f * (b.x - a.x), y = a.y + f * (b.y - a.y);
    let angle = Math.atan2(b.y - a.y, b.x - a.x);
    if (angle > Math.PI / 2) angle -= Math.PI;
    if (angle < -Math.PI / 2) angle += Math.PI;
    ctx.save();
    ctx.translate(x, y); ctx.rotate(angle);
    ctx.font = `600 9px ${mono}`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    const w = ctx.measureText(s.label).width;
    ctx.globalCompositeOperation = "destination-out";
    ctx.globalAlpha = 1;
    ctx.fillRect(-w / 2 - 3, -6, w + 6, 12);
    ctx.globalCompositeOperation = "source-over";
    ctx.fillStyle = s.color;
    ctx.fillText(s.label, 0, 0);
    ctx.restore();
  });
  ctx.restore();

  // picked-cell outline (heat plots)
  if (p.heat && p.highlight) {
    const { xEdges, yEdges } = p.heat;
    const { i, j } = p.highlight;
    if (i >= 0 && i < xEdges.length - 1 && j >= 0 && j < yEdges.length - 1) {
      const x0 = tx(xEdges[i]), x1 = tx(xEdges[i + 1]), y0 = ty(yEdges[j]), y1 = ty(yEdges[j + 1]);
      ctx.save(); clipPlot();
      ctx.strokeStyle = ink; ctx.lineWidth = 1.5; ctx.setLineDash([]);
      const pad = 1.5;
      ctx.strokeRect(Math.min(x0, x1) - pad, Math.min(y0, y1) - pad, Math.abs(x1 - x0) + 2 * pad, Math.abs(y1 - y0) + 2 * pad);
      ctx.restore();
    }
  }

  // axis frame
  ctx.globalAlpha = 1;
  ctx.setLineDash([]);
  ctx.strokeStyle = gridS;
  ctx.lineWidth = 1;
  ctx.strokeRect(m.l, m.t, iw, ih);

  // density colorbar
  if (heatNorm) {
    const bx = W - m.r + 14, bw = 12, bt = m.t, bh = ih, steps = 48;
    for (let s = 0; s < steps; s++) {
      const t = s / (steps - 1);
      ctx.fillStyle = heatNorm.color(t);
      const yy = bt + (1 - t) * bh;
      ctx.fillRect(bx, yy - bh / steps - 0.5, bw, bh / steps + 1);
    }
    ctx.strokeStyle = gridS; ctx.lineWidth = 1; ctx.strokeRect(bx, bt, bw, bh);
    ctx.font = `10px ${mono}`;
    ctx.fillStyle = faint; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.strokeStyle = faint;
    const colorTicks = heatNorm.ticks ?? (() => {
      const ticks: Tick[] = [];
      for (let e = 0; 10 ** e <= heatNorm!.zmax + 0.5; e++) ticks.push({ v: 10 ** e, label: e === 0 ? "1" : `10${supN(e)}` });
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
    ctx.fillStyle = dim; ctx.font = `500 11px ${mono}`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(heatNorm.label ?? "pixels / cell", 0, 0);
    ctx.restore();
  }

  // axis labels
  ctx.fillStyle = dim;
  ctx.font = `500 11.5px ${mono}`;
  if (p.xLabel) { ctx.textAlign = "center"; ctx.textBaseline = "bottom"; ctx.fillText(p.xLabel, m.l + iw / 2, H - 6); }
  if (p.yLabel) {
    ctx.save(); ctx.translate(14, m.t + ih / 2); ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillText(p.yLabel, 0, 0); ctx.restore();
  }
}

/* ─── linked cursors ──────────────────────────────────────────────────────── */

type SyncFn = (x: number | null, from: symbol) => void;
const SYNC = new Map<string, Set<SyncFn>>();
function publishCursor(key: string, x: number | null, from: symbol) {
  for (const fn of SYNC.get(key) ?? []) fn(x, from);
}
function subscribeCursor(key: string, fn: SyncFn): () => void {
  let set = SYNC.get(key);
  if (!set) { set = new Set(); SYNC.set(key, set); }
  set.add(fn);
  return () => { set!.delete(fn); if (!set!.size) SYNC.delete(key); };
}

/* ─── legend ──────────────────────────────────────────────────────────────── */

const legendKey = (it: LegendItem) => it.key ?? it.label;

function Swatch({ it }: { it: LegendItem }) {
  return (
    <span aria-hidden="true" style={it.line ? { position: "relative", width: 22, height: 12 } : it.marker ? {
      width: 9, height: 9, boxSizing: "border-box", borderRadius: it.marker === "diamond" ? 1 : "50%",
      border: `2px solid ${it.color}`, background: it.marker === "filled" ? it.color : "transparent",
      transform: it.marker === "diamond" ? "rotate(45deg)" : undefined,
    } : it.histogram ? {
      width: 18, height: 9, boxSizing: "border-box", border: `2px ${it.dash ? "dashed" : "solid"} ${it.color}`,
      background: it.hatch ? `repeating-linear-gradient(135deg, transparent 0 3px, ${it.color} 3px 4px)`
        : it.filled ? it.color : "transparent",
      opacity: it.filled && !it.hatch ? 0.55 : 1,
    } : { width: 18, height: 0, borderTop: `${it.dash ? "2.5px dashed" : "3px solid"} ${it.color}` }}>
      {it.line && <>
        <span style={{ position: "absolute", left: 0, right: 0, top: 5, borderTop: `${it.dash ? "2.5px dashed" : "3px solid"} ${it.color}` }} />
        <span style={{
          position: "absolute", left: 8, top: 2, width: 7, height: 7, boxSizing: "border-box",
          borderRadius: it.marker === "diamond" ? 1 : "50%", border: `2px solid ${it.color}`,
          background: it.marker === "filled" ? it.color : "var(--surface-1)",
          transform: it.marker === "diamond" ? "rotate(45deg)" : undefined,
        }} />
      </>}
    </span>
  );
}

/** Legend. Static by default (v1). With `onToggle`, entries are toggle
 *  buttons (aria-pressed = shown); `onHover` reports the hovered/focused key
 *  (for emphasis). Keys default to labels. */
export function Legend(
  { items, hidden, onToggle, onHover, className }: {
    items: LegendItem[]; hidden?: readonly string[]; onToggle?: (key: string) => void;
    onHover?: (key: string | null) => void; className?: string;
  },
) {
  return (
    <div className={`plot-legend${className ? ` ${className}` : ""}`}>
      {items.map((it, i) => {
        const key = legendKey(it);
        const off = hidden?.includes(key) ?? false;
        const body = <><Swatch it={it} /><span className="plot-legend__label">{it.label}</span></>;
        if (!onToggle) {
          return (
            <span key={i} className="plot-legend__item"
              onMouseEnter={onHover ? () => onHover(key) : undefined}
              onMouseLeave={onHover ? () => onHover(null) : undefined}>{body}</span>
          );
        }
        return (
          <button key={i} type="button" className="plot-legend__item plot-legend__item--toggle"
            aria-pressed={!off} data-off={off || undefined} title={off ? `Show ${it.label}` : `Hide ${it.label}`}
            onClick={() => onToggle(key)}
            onMouseEnter={() => onHover?.(key)} onMouseLeave={() => onHover?.(null)}
            onFocus={() => onHover?.(key)} onBlur={() => onHover?.(null)}>
            {body}
          </button>
        );
      })}
    </div>
  );
}

/** Wire an external <Legend> to one or more <Plot>s:
 *    const lg = useLegend();
 *    <Plot {...props} {...lg.plotProps} />  <Legend items={…} {...lg.legendProps} /> */
export function useLegend(initialHidden: string[] = []) {
  const [hidden, setHidden] = useState<string[]>(initialHidden);
  const [emphasis, setEmphasis] = useState<string | null>(null);
  const toggle = useCallback((key: string) =>
    setHidden((h) => (h.includes(key) ? h.filter((k) => k !== key) : [...h, key])), []);
  return {
    hidden, setHidden, toggle, emphasis, setEmphasis,
    plotProps: { hidden, onHiddenChange: setHidden, emphasis },
    legendProps: { hidden, onToggle: toggle, onHover: setEmphasis },
  };
}

function autoLegend(series: Series[]): LegendItem[] {
  const seen = new Set<string>();
  const out: LegendItem[] = [];
  series.forEach((s, i) => {
    if (!s.name) return;
    const key = seriesKey(s, i);
    if (seen.has(key)) return;
    seen.add(key);
    out.push({
      key, label: s.name, color: s.color, dash: !!s.dash?.length,
      histogram: s.mode === "histogram", filled: s.mode === "histogram" && !s.hatch, hatch: s.hatch,
      marker: s.mode === "scatter" ? (s.marker ?? "filled") : s.dots ? s.marker : undefined,
      line: s.mode !== "scatter" && s.mode !== "histogram" && !!s.dots,
    });
  });
  return out;
}

/* ─── the component ───────────────────────────────────────────────────────── */

/** Bumps when the drawn inputs change (structural compare, handlers ignored). */
function useInputVersion(p: PlotProps): number {
  const ref = useRef<{ p: PlotProps; v: number }>({ p, v: 0 });
  if (ref.current.p !== p && !sameInputs(ref.current.p, p)) ref.current = { p, v: ref.current.v + 1 };
  else ref.current.p = p;
  return ref.current.v;
}

/** The previous value while the new one is structurally equal (a stable
 *  identity for memo deps, e.g. tick arrays rebuilt on every render). */
function useStable<T>(value: T): T {
  const ref = useRef(value);
  if (ref.current !== value && !sameValue(ref.current, value)) ref.current = value;
  return ref.current;
}

/** Tooltip rows before "+N more". */
const TIP_ROWS = 8;

type Drag = { mode: "box" | "pan"; x0: number; y0: number; x1: number; y1: number; view: PlotView };
const DRAG_PX = 4;

export default function Plot(p: PlotProps) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const me = useRef<symbol>(Symbol("plot"));
  const theme = useResolvedTheme();
  const version = useInputVersion(p);
  const tooltipOn = p.tooltip !== false;
  const zoomOn = p.zoom !== false;
  const zoomAxes = p.zoomAxes ?? "xy";
  const xScale = p.xScale ?? "linear", yScale = p.yScale ?? "linear";

  /* size (one ResizeObserver for the component's life) */
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const el = wrap.current;
    if (!el) return;
    setWidth(el.clientWidth);
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver((entries) => {
      const w = Math.round(entries[0]?.contentRect.width ?? el.clientWidth);
      setWidth((old) => (w > 0 && w !== old ? w : old));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const W = width || 640;
  const H = p.height ?? Math.round(W * (p.aspect ?? 0.5));

  /* hidden series (controlled or not) */
  const [hiddenInner, setHiddenInner] = useState<string[]>([]);
  const hiddenList = p.hidden ?? hiddenInner;
  const hiddenSig = hiddenList.join("\u0000");
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const hidden = useMemo(() => new Set(hiddenList), [hiddenSig]);
  const toggleKey = (key: string) => {
    const next = hidden.has(key) ? hiddenList.filter((k) => k !== key) : [...hiddenList, key];
    if (p.hidden === undefined) setHiddenInner(next);
    p.onHiddenChange?.(next);
  };

  /* zoom view (controlled or not); an uncontrolled zoom resets when the domains change */
  const [viewInner, setViewInner] = useState<PlotView | null>(null);
  const view = p.view !== undefined ? p.view : viewInner;
  const domSig = `${p.xDomain[0]},${p.xDomain[1]},${p.yDomain[0]},${p.yDomain[1]},${xScale},${yScale}`;
  useEffect(() => { setViewInner(null); }, [domSig]);
  const setView = (v: PlotView | null) => {
    const norm = v && (v.x || v.y) ? v : null;
    if (p.view === undefined) setViewInner(norm);
    p.onViewChange?.(norm);
  };
  const xDom = view?.x ?? p.xDomain, yDom = view?.y ?? p.yDomain;

  /* hover / keyboard readout / linked cursor */
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null);
  const [kb, setKb] = useState<{ series: number; index: number } | null>(null);
  const [remoteX, setRemoteX] = useState<number | null>(null);
  const [drag, setDragState] = useState<Drag | null>(null);
  // The live drag also sits in a ref: pointer events can arrive faster than
  // React re-renders, and a handler must never see a stale (null) drag.
  const dragRef = useRef<Drag | null>(null);
  const setDrag = (d: Drag | null) => { dragRef.current = d; setDragState(d); };
  const [legendEmph, setLegendEmph] = useState<string | null>(null);
  const suppressClick = useRef(false);
  useEffect(() => {
    if (!p.syncKey) return;
    return subscribeCursor(p.syncKey, (x, from) => { if (from !== me.current) setRemoteX(x); });
  }, [p.syncKey]);

  /* effective (drawn) props + layout. Zoomed ticks are generated and
     labelled by xFormat/yFormat; they are compared by their output, so an
     inline formatter costs nothing and a changed one redraws. */
  const emphasis = p.emphasis ?? legendEmph;
  const xTicks = useStable(view?.x ? viewTicks(p.xTicks, xDom, xScale, p.xFormat) : p.xTicks);
  const yTicks = useStable(view?.y ? viewTicks(p.yTicks, yDom, yScale, p.yFormat) : p.yTicks);
  const eff = useMemo<PlotProps>(() => ({ ...p, xDomain: xDom, yDomain: yDom, xTicks, yTicks }),
  // `version` stands for every drawn prop (structural compare)
  // eslint-disable-next-line react-hooks/exhaustive-deps
    [version, xDom[0], xDom[1], yDom[0], yDom[1], xTicks, yTicks]);
  const layout = useMemo(() => computeLayout(W, H, eff, fonts()),
    // theme: token fonts could differ per theme; cheap either way
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [W, H, eff, theme]);
  const visibleEntries = useMemo(
    () => p.series.map((s, i) => ({ s, key: seriesKey(s, i) })).filter((e) => !hidden.has(e.key)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [version, hidden]);
  const visible = useMemo(() => visibleEntries.map((e) => e.s), [visibleEntries]);
  // Emphasis (legend hover / `emphasis` prop): dim every other visible series.
  const dimSet = useMemo(() => {
    if (!emphasis || !visibleEntries.some((e) => e.key === emphasis)) return undefined;
    const out = new Set<number>();
    visibleEntries.forEach((e, i) => { if (e.key !== emphasis) out.add(i); });
    return out;
  }, [visibleEntries, emphasis]);

  /* paint: only when layout / drawn inputs / visibility / emphasis / theme change */
  useEffect(() => {
    const cv = canvas.current;
    if (!cv) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = Math.round(layout.W * dpr);
    cv.height = Math.round(layout.H * dpr);
    cv.style.width = `${layout.W}px`;
    cv.style.height = `${layout.H}px`;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    paint(ctx, layout, eff, visible, { fonts: fonts(), dim: dimSet });
  }, [layout, eff, visible, dimSet, theme]);

  /* geometry helpers */
  const local = (e: { clientX: number; clientY: number }) => {
    const r = canvas.current?.getBoundingClientRect();
    return { x: e.clientX - (r?.left ?? 0), y: e.clientY - (r?.top ?? 0) };
  };
  const inside = (pt: { x: number; y: number }) =>
    pt.x >= layout.m.l && pt.x <= layout.m.l + layout.iw && pt.y >= layout.m.t && pt.y <= layout.m.t + layout.ih;
  const currentView = (): PlotView => ({ x: view?.x ?? null, y: view?.y ?? null });
  const zoomAbout = (fx: number, fy: number, factor: number) => {
    const v = currentView();
    const nx = zoomAxes !== "y" ? clampView(zoomDomain(xDom, xScale, fx, factor), xScale) : v.x;
    const ny = zoomAxes !== "x" ? clampView(zoomDomain(yDom, yScale, fy, factor), yScale) : v.y;
    setView({ x: nx ?? v.x, y: ny ?? v.y });
  };
  const panBy = (fx: number, fy: number) => {
    const v = currentView();
    setView({
      x: zoomAxes !== "y" && fx ? clampView(panDomain(xDom, xScale, fx), xScale) ?? v.x : v.x,
      y: zoomAxes !== "x" && fy ? clampView(panDomain(yDom, yScale, fy), yScale) ?? v.y : v.y,
    });
  };

  /* pointer interaction */
  const onPointerDown = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!zoomOn || e.button !== 0) return;
    const pt = local(e);
    if (!inside(pt)) return;
    try { e.currentTarget.setPointerCapture?.(e.pointerId); } catch { /* not an active pointer */ }
    setDrag({ mode: e.shiftKey ? "pan" : "box", x0: pt.x, y0: pt.y, x1: pt.x, y1: pt.y, view: currentView() });
  };
  const onPointerMove = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    const pt = local(e);
    setKb(null);
    const drag = dragRef.current;
    if (drag) {
      if (drag.mode === "pan") {
        const dx = (pt.x - drag.x0) / layout.iw, dy = (pt.y - drag.y0) / layout.ih;
        const baseX = drag.view.x ?? p.xDomain, baseY = drag.view.y ?? p.yDomain;
        setView({
          x: zoomAxes !== "y" ? clampView(panDomain(baseX, xScale, -dx), xScale) : drag.view.x,
          y: zoomAxes !== "x" ? clampView(panDomain(baseY, yScale, dy), yScale) : drag.view.y,
        });
      }
      setDrag({ ...drag, x1: pt.x, y1: pt.y });
      return;
    }
    if (!tooltipOn && !p.syncKey) return;
    if (inside(pt)) {
      setHover(pt);
      if (p.syncKey) publishCursor(p.syncKey, layout.x.fromPx(pt.x), me.current);
    } else if (hover) {
      setHover(null);
      if (p.syncKey) publishCursor(p.syncKey, null, me.current);
    }
  };
  const endDrag = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    if (!drag) return;
    const pt = local(e);
    const moved = Math.abs(pt.x - drag.x0) > DRAG_PX || Math.abs(pt.y - drag.y0) > DRAG_PX;
    if (moved) suppressClick.current = true;
    if (drag.mode === "box" && moved) {
      const x = zoomAxes !== "y" && Math.abs(pt.x - drag.x0) > DRAG_PX
        ? clampView([layout.x.fromPx(drag.x0), layout.x.fromPx(pt.x)], xScale) : drag.view.x;
      const y = zoomAxes !== "x" && Math.abs(pt.y - drag.y0) > DRAG_PX
        ? clampView([layout.y.fromPx(drag.y0), layout.y.fromPx(pt.y)], yScale) : drag.view.y;
      setView({ x, y });
    }
    setDrag(null);
  };
  const onPointerLeave = () => {
    setHover(null);
    if (p.syncKey) publishCursor(p.syncKey, null, me.current);
  };
  const onClick = (e: MouseEvent<HTMLCanvasElement>) => {
    if (suppressClick.current) { suppressClick.current = false; return; }
    if (!p.onHeatClick && !p.onPlotClick) return;
    const pt = local(e);
    if (!inside(pt)) return;
    const dataX = layout.x.fromPx(pt.x), dataY = layout.y.fromPx(pt.y);
    if (p.onHeatClick && p.heat) {
      const i = binOf(p.heat.xEdges, dataX), j = binOf(p.heat.yEdges, dataY);
      if (i >= 0 && j >= 0) p.onHeatClick({ i, j });
    } else p.onPlotClick?.({ x: dataX, y: dataY });
  };

  /* wheel zoom: Ctrl/⌘ held, or the plot focused FROM THE KEYBOARD (Tab, or
     a key the plot acts on pressed since the last pointer-down). A mouse click
     also focuses the tabIndex=0 frame; that must not turn the page wheel into
     zoom, and neither may the Ctrl of a Ctrl-wheel or a key the plot ignores. */
  const kbFocus = useRef(false);
  const pointerFocus = useRef(false);
  const onFramePointerDown = () => { pointerFocus.current = true; kbFocus.current = false; };
  const onFrameFocus = (e: FocusEvent<HTMLDivElement>) => {
    // Any focus inside the frame (the frame, or a tool button) consumes the
    // pointer-down flag, so a later Tab into the plot reads as keyboard focus.
    const fromPointer = pointerFocus.current;
    pointerFocus.current = false;
    if (e.target === e.currentTarget) kbFocus.current = !fromPointer;
  };
  const onFrameBlur = (e: FocusEvent<HTMLDivElement>) => {
    if (e.target !== e.currentTarget) return;
    kbFocus.current = false;
    pointerFocus.current = false;
  };
  const wheelRef = useRef<(e: WheelEvent) => void>(() => {});
  wheelRef.current = (e: WheelEvent) => {
    if (!zoomOn) return;
    const focused = kbFocus.current && document.activeElement === wrap.current;
    if (!(e.ctrlKey || e.metaKey || focused)) return;
    const pt = local(e);
    if (!inside(pt)) return;
    e.preventDefault();
    const factor = Math.exp(Math.max(-1, Math.min(1, e.deltaY * 0.0025)));
    zoomAbout((pt.x - layout.m.l) / layout.iw, 1 - (pt.y - layout.m.t) / layout.ih, factor);
  };
  useEffect(() => {
    const cv = canvas.current;
    if (!cv) return;
    const fn = (e: WheelEvent) => wheelRef.current(e);
    cv.addEventListener("wheel", fn, { passive: false });
    return () => cv.removeEventListener("wheel", fn);
  }, []);

  /* keyboard: +/- zoom, shift+arrows pan, 0/Esc reset, arrows step the readout */
  /* The readout visits drawable points only: a gap (null, or ≤ 0 on a log
     axis) has no pixel position for the marker and the tooltip. */
  const stepReadout = (dir: 1 | -1, dSeries = 0) => {
    const vis = p.series.map((_, i) => i)
      .filter((i) => !hidden.has(seriesKey(p.series[i], i)) && stepDrawable(p.series[i], 0, 0, layout) >= 0);
    if (!vis.length) return;
    setHover(null);
    if (!kb || !vis.includes(kb.series)) {
      setKb({ series: vis[0], index: stepDrawable(p.series[vis[0]], 0, 0, layout) });
      return;
    }
    if (dSeries) {
      const next = vis[(vis.indexOf(kb.series) + dSeries + vis.length) % vis.length];
      const at = nearestIndex(p.series[next].x, p.series[kb.series].x[kb.index]);
      setKb({ series: next, index: stepDrawable(p.series[next], Math.max(0, at), 0, layout) });
      return;
    }
    const i = stepDrawable(p.series[kb.series], kb.index, dir, layout);
    if (i >= 0) setKb({ series: kb.series, index: i });
  };
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.target !== e.currentTarget) return;
    // Ctrl/⌘/Alt combinations are browser shortcuts (Ctrl + = − 0 zoom the page).
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    // Only a key the plot acts on makes this keyboard focus (plain wheel zooms).
    const take = () => { e.preventDefault(); kbFocus.current = true; };
    const zoomed = !!(view?.x || view?.y);
    if (zoomOn && (e.key === "+" || e.key === "=")) { take(); zoomAbout(0.5, 0.5, 0.8); return; }
    if (zoomOn && (e.key === "-" || e.key === "_")) { take(); zoomAbout(0.5, 0.5, 1.25); return; }
    if (e.key === "0" || (e.key === "Escape" && (zoomed || kb))) {
      take();
      if (zoomed) setView(null);
      setKb(null);
      return;
    }
    if (e.key.startsWith("Arrow")) {
      if (e.shiftKey && zoomOn) {
        take();
        const d = 0.1;
        if (e.key === "ArrowRight") panBy(d, 0);
        if (e.key === "ArrowLeft") panBy(-d, 0);
        if (e.key === "ArrowUp") panBy(0, d);
        if (e.key === "ArrowDown") panBy(0, -d);
        return;
      }
      if (!tooltipOn) return;
      take();
      if (e.key === "ArrowRight") stepReadout(1);
      if (e.key === "ArrowLeft") stepReadout(-1);
      if (e.key === "ArrowUp") stepReadout(1, -1);
      if (e.key === "ArrowDown") stepReadout(1, 1);
    }
  };

  /* hit + readout for the overlay */
  let hit: Hit | null = null;
  if (kb) {
    const s = p.series[kb.series];
    const y = s?.y[kb.index];
    if (s && y != null && drawable(s, kb.index, layout)) {
      const x = s.x[kb.index];
      hit = { series: kb.series, index: kb.index, x, y, px: layout.x.toPx(x), py: layout.y.toPx(y), dist: 0 };
    }
  } else if (hover && tooltipOn) {
    hit = nearestPoint(p.series, hidden, hover, layout);
  }
  const cursorX = hit ? hit.x : hover ? layout.x.fromPx(hover.x) : null;
  // The hovered series is always a row; the others are the lines nearest the cursor.
  const readout = cursorX != null && tooltipOn
    ? tooltipReadout(p.series, hidden, cursorX, {
      limit: TIP_ROWS, y: layout.y, first: hit?.series ?? null,
      cursorPy: hover ? hover.y : hit ? hit.py : layout.m.t + layout.ih / 2,
    })
    : { rows: [], more: 0 };
  const heatCell = hover && tooltipOn && p.heat ? heatAt(p.heat, layout.x.fromPx(hover.x), layout.y.fromPx(hover.y)) : null;
  const remoteMarks = remoteX != null && !hover ? readoutAt(p.series, hidden, remoteX, 12) : [];
  const fx = p.xFormat ?? formatValue, fy = p.yFormat ?? formatValue;

  const crossX = hover ? (hit && hit.series >= 0 && p.series[hit.series]?.mode !== "scatter" ? hit.px : hover.x)
    : kb && hit ? hit.px : remoteX != null ? layout.x.toPx(remoteX) : null;
  const showTip = tooltipOn && (hit || heatCell) && (hover || kb);
  const tipAt = hover ?? (hit ? { x: hit.px, y: hit.py } : null);

  const legendItems = p.legend === "auto" ? autoLegend(p.series) : p.legend;
  const zoomed = !!(view?.x || view?.y);
  const label = p["aria-label"] ?? p.title ?? "Chart";
  const summary = `${visible.length} series${p.xLabel ? `; x: ${p.xLabel}` : ""} ${fx(xDom[0])}–${fx(xDom[1])}`
    + `${p.yLabel ? `; y: ${p.yLabel}` : ""} ${fy(yDom[0])}–${fy(yDom[1])}${zoomed ? " (zoomed)" : ""}.`;
  const interactive = tooltipOn || zoomOn;

  const exportPng = () => {
    const off = document.createElement("canvas");
    const scale = 2;
    off.width = layout.W * scale; off.height = layout.H * scale;
    const ctx = off.getContext("2d");
    if (!ctx) return;
    ctx.scale(scale, scale);
    paint(ctx, layout, eff, visible, { fonts: fonts(), background: cssVar("--surface-1", "#ffffff") });
    off.toBlob((b) => { if (b) downloadBlob(`${safeFileName(p.exportName ?? "plot")}.png`, b); }, "image/png");
  };
  const exportCsv = () => downloadText(`${safeFileName(p.exportName ?? "plot")}.csv`, seriesToCSV(visible), "text/csv;charset=utf-8");

  const kbText = kb && hit ? `${seriesName(p.series[hit.series], hit.series)}: x ${fx(hit.x)}, y ${fy(hit.y)}` : "";

  return (
    <div className="plot" data-zoomed={zoomed || undefined}>
      <div ref={wrap} className="plot__frame" role="figure" aria-label={label}
        tabIndex={interactive ? 0 : undefined} onKeyDown={onKeyDown}
        onPointerDownCapture={onFramePointerDown} onFocus={onFrameFocus} onBlur={onFrameBlur}>
        <canvas ref={canvas} aria-hidden="true"
          style={interactive || p.onHeatClick || p.onPlotClick ? { cursor: zoomOn ? "crosshair" : "pointer" } : undefined}
          onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={endDrag}
          onPointerCancel={() => setDrag(null)} onPointerLeave={onPointerLeave}
          onClick={onClick} onDoubleClick={() => { if (zoomOn && zoomed) setView(null); }} />
        <span className="sr-only">{summary}{interactive ? " Arrow keys read values; + and − zoom; shift+arrows pan; 0 resets." : ""}</span>
        <span className="sr-only" role="status" aria-live="polite">{kbText}</span>

        {crossX != null && Number.isFinite(crossX) && (
          <span className="plot__cross-x" style={{ left: crossX, top: layout.m.t, height: layout.ih }} />
        )}
        {hover && !drag && (
          <span className="plot__cross-y" style={{ top: hover.y, left: layout.m.l, width: layout.iw }} />
        )}
        {hit && Number.isFinite(hit.px) && Number.isFinite(hit.py) && (
          <span className="plot__mark" style={{ left: hit.px, top: hit.py, borderColor: p.series[hit.series]?.color }} />
        )}
        {remoteMarks.map((r) => {
          const x = layout.x.toPx(r.x), y = layout.y.toPx(r.y);
          return Number.isFinite(x) && Number.isFinite(y)
            ? <span key={r.series} className="plot__mark plot__mark--remote" style={{ left: x, top: y, borderColor: p.series[r.series].color }} />
            : null;
        })}
        {drag && drag.mode === "box" && (
          <span className="plot__box" style={{
            left: zoomAxes === "y" ? layout.m.l : Math.min(drag.x0, drag.x1),
            top: zoomAxes === "x" ? layout.m.t : Math.min(drag.y0, drag.y1),
            width: zoomAxes === "y" ? layout.iw : Math.abs(drag.x1 - drag.x0),
            height: zoomAxes === "x" ? layout.ih : Math.abs(drag.y1 - drag.y0),
          }} />
        )}
        {showTip && tipAt && (
          <div className="plot__tooltip" aria-hidden="true"
            style={tipPosition(tipAt, layout.W, layout.H)}>
            {heatCell && !hit ? (
              <>
                <div className="plot__tip-head">{fx(heatCell.x0)}–{fx(heatCell.x1)} × {fy(heatCell.y0)}–{fy(heatCell.y1)}</div>
                <div className="plot__tip-row"><span>{p.heat?.colorLabel ?? "count"}</span><b>{formatValue(heatCell.z)}</b></div>
              </>
            ) : hit && p.series[hit.series]?.mode === "scatter" ? (
              <>
                <div className="plot__tip-head">{seriesName(p.series[hit.series], hit.series)}</div>
                <ul className="plot__tip-list">
                  <li data-nearest="true"><span className="plot__tip-sw" style={{ background: p.series[hit.series].color }} />
                    <span>x {fx(hit.x)}</span><b>y {fy(hit.y)}</b></li>
                </ul>
              </>
            ) : (
              <>
                <div className="plot__tip-head">{p.xLabel ? `${p.xLabel} ` : "x "}{fx(hit?.x ?? cursorX ?? 0)}</div>
                <ul className="plot__tip-list">
                  {readout.rows.map((r) => (
                    <li key={r.series} data-nearest={hit?.series === r.series || undefined}>
                      <span className="plot__tip-sw" style={{ background: p.series[r.series].color }} />
                      <span className="plot__tip-name">{seriesName(p.series[r.series], r.series)}</span>
                      <b>{fy(r.y)}</b>
                    </li>
                  ))}
                </ul>
                {readout.more > 0 && <div className="plot__tip-more">+{readout.more} more</div>}
              </>
            )}
          </div>
        )}
        {(zoomed || p.exportName) && (
          <div className="plot__tools">
            {zoomed && (
              <button type="button" className="plot__tool" aria-label="Reset zoom" title="Reset zoom (double-click, 0)"
                onClick={() => setView(null)}><Icon name="zoomOut" size={14} /></button>
            )}
            {p.exportName && <>
              <button type="button" className="plot__tool" aria-label="Download PNG" title="Download PNG" onClick={exportPng}>
                <Icon name="image" size={14} />
              </button>
              <button type="button" className="plot__tool" aria-label="Download CSV" title="Download CSV (visible series)" onClick={exportCsv}>
                <Icon name="table" size={14} />
              </button>
            </>}
          </div>
        )}
      </div>
      {legendItems && legendItems.length > 0 && (
        <Legend items={legendItems} hidden={hiddenList}
          onToggle={p.legendToggle === false ? undefined : toggleKey} onHover={setLegendEmph} />
      )}
    </div>
  );
}

/* ─── small helpers ───────────────────────────────────────────────────────── */

/* Largest k with edges[k] <= v (edges strictly increasing); -1 if out of range. */
function binOf(edges: number[], v: number): number {
  if (!(v >= edges[0]) || !(v <= edges[edges.length - 1])) return -1;
  let lo = 0, hi = edges.length - 1;
  while (lo < hi - 1) { const mid = (lo + hi) >> 1; if (edges[mid] <= v) lo = mid; else hi = mid; }
  return lo;
}

function heatAt(heat: Heat, x: number, y: number) {
  const i = binOf(heat.xEdges, x), j = binOf(heat.yEdges, y);
  if (i < 0 || j < 0) return null;
  const z = heat.z[i]?.[j];
  if (z == null || !Number.isFinite(z)) return null;
  return { i, j, z, x0: heat.xEdges[i], x1: heat.xEdges[i + 1], y0: heat.yEdges[j], y1: heat.yEdges[j + 1] };
}

/** Keep the tooltip inside the plot: flip left/up near the right/bottom edge. */
function tipPosition(pt: { x: number; y: number }, W: number, H: number): CSSProperties {
  const right = pt.x > W * 0.6, below = pt.y < H * 0.35;
  return {
    left: right ? undefined : pt.x + 14, right: right ? W - pt.x + 14 : undefined,
    top: below ? pt.y + 14 : undefined, bottom: below ? undefined : H - pt.y + 14,
  };
}
