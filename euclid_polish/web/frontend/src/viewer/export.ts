/* Exports of the image viewer (ported from the pre-rework engine):
 *   ⬇ PNG     — a composite of the visible frames as shown (pan/zoom, overlay
 *               labels) at up to 2× device pixels;
 *   ⬇ Figure  — a white-paper publication plate (≤ 4800 px wide): panel
 *               titles, literal pixels (no smoothing), an arcsec scale bar and
 *               a heat bar per panel; a frozen crop exports as matched crops;
 *   ⏺ video   — the visible frames recorded at 30 fps as .webm.
 * The heat bar is unit-aware (WP-B1b handoff): it names the tier's unit
 * (X-Cube-Unit, else the meta tier's `unit`; Euclid tiers without one are
 * e⁻), and for a non-e⁻ unit or a display scale ≠ 1 the knee text and the
 * tick values are divided by the display scale (the transfer runs on
 * value × displayScale), so a JWST panel reads MJy/sr, not e⁻.
 * It also follows the frame's display settings (render.ts): the stretch
 * (ticks placed by that stretch's transfer, anchored at the black point; the
 * auto stretches from the frame's own limits), the black point, and the
 * colormap / invert of the gradient. */
import type { Colormap, Stretch } from "../state/display";
import { downloadBlob } from "../ui";
import { colormapLut } from "./colormaps";
import type { Crop } from "./selection";

export const PUBLICATION_MAX_WIDTH = 4800;
export const PUBLICATION_PAPER = "#ffffff";
export const PUBLICATION_INK = "#111111";
/** Frame background of every exported image (astronomy reads on near-black). */
export const FRAME_INK = "#05070d";

export function publicationUnitLabel(unit: string | null | undefined): string {
  const u = String(unit || "").trim();
  if (!u || u === "e-" || u === "e⁻" || u.toLowerCase() === "electron") return "e⁻";
  if (u === "arb") return "arb. units";
  return u;
}

export function publicationElectronLabel(value: number): string {
  if (value >= 1000) return `${(value / 1000).toFixed(value >= 10000 ? 0 : 1)}k`;
  if (value >= 10) return `${Math.round(value)}`;
  if (value >= 1) return value.toFixed(1);
  if (value > 0 && value < 0.01) return value.toExponential(1).replace("e+", "e");
  return value.toFixed(2);
}

/** A tick value: publicationElectronLabel with a proper minus sign. */
export function publicationSignedLabel(value: number): string {
  return value < 0 ? `−${publicationElectronLabel(-value)}` : publicationElectronLabel(value);
}

export type HeatbarInfo = {
  band: string;
  knee: number;
  gain: number;
  /** A log-mode (PSF) panel. */
  log: boolean;
  /** publicationUnitLabel of the tier's unit. */
  unit: string;
  /** The cube's display scale (1 when none). */
  scale: number;
  /** The frame's stretch (default "asinh-abs", the locked default). */
  stretch?: Stretch;
  /** Black point in the transfer's e⁻ units (default 0). */
  black?: number;
  /** asinh-auto / zscale: the frame's own limits (render.ts frameAutoStats)
   *  in the transfer's e⁻ units (the prepared values ÷ their factor). */
  auto?: { lo: number; hi: number; knee?: number };
  /** Gradient stops of the bar (heatbarStops; default black → white). */
  stops?: string[];
  /** A residual tier: a diverging bar over ±range in its own scale. */
  signed?: { scale: "asinh" | "linear"; knee: number; range: number; label: string; stops: string[] };
};

/** The bar's gradient: the frame's colormap for a single-band frame
 *  (reversed when inverted, as render.ts flips the lookup index); a colour
 *  composite (Lupton, Temp, direct RGB) keeps its colours, so its bar is the
 *  luminance ramp (inverted too when the frame is). */
export function heatbarStops(colormap: Colormap, invert: boolean, mode: string, n = 17): string[] {
  const lut = colormapLut(mode === "gray" || mode === "gray-log" ? colormap : "gray");
  const out = Array.from({ length: n }, (_, i) => {
    const k = Math.round((i / (n - 1)) * 255) * 3;
    return `rgb(${lut[k]}, ${lut[k + 1]}, ${lut[k + 2]})`;
  });
  return invert ? out.reverse() : out;
}

export type HeatbarModel = { parameterText: string; ticks: { fraction: number; label: string }[]; signalLabel: string };

/** The text and ticks of one panel's heat bar (K0 = meta.color.default_asinh). */
export function heatbarModel(info: HeatbarInfo, K0: number): HeatbarModel {
  if (info.signed) {
    const sg = info.signed;
    const norm = Math.asinh(sg.range / Math.max(sg.knee, 1e-30));
    const ticks = [0, 0.25, 0.5, 0.75, 1].map((fraction) => {
      const s = 2 * fraction - 1;
      const v = sg.scale === "asinh" ? Math.sign(s) * sg.knee * Math.sinh(Math.abs(s) * norm) : s * sg.range;
      return { fraction, label: `${v < 0 ? "−" : ""}${publicationElectronLabel(Math.abs(v))}` };
    });
    const how = sg.scale === "asinh" ? `asinh, knee ${publicationElectronLabel(sg.knee)} ${info.unit}` : `linear ±${publicationElectronLabel(sg.range)} ${info.unit}`;
    return { parameterText: `Band: ${info.band}  ·  ${sg.label}  ·  ${how}`, ticks, signalLabel: `Residual (${info.unit})` };
  }
  const signalLabel = info.log ? "relative display intensity"
    : info.unit === "e⁻" ? "Pixel signal (e⁻)" : `Pixel signal (${info.unit})`;
  const fractions = [0, 0.25, 0.5, 0.75, 1];
  if (info.log) {
    // A log-mode (PSF) frame ignores the stretch: relative intensity.
    return { parameterText: `Band: ${info.band}  ·  logarithmic display`, ticks: fractions.map((fraction) => ({ fraction, label: fraction.toFixed(2) })), signalLabel };
  }
  const g = Math.max(info.gain, 1e-30);
  const black = Number.isFinite(info.black) ? info.black as number : 0;
  const native = (e: number) => e / info.scale;             // transfer e⁻ → the tier's unit
  const text = (e: number) => publicationSignedLabel(native(e));
  const unit = info.unit;
  const stretch = info.stretch ?? "asinh-abs";
  const W = 30 * K0;                                          // white of the absolute stretches (gain 1)
  // value(f): the pixel value (transfer e⁻) the stretch maps to bar fraction f
  // — the inverse of render.ts luminance(), so the bar reads what the frame shows.
  let value: (f: number) => number;
  let how: string;
  if ((stretch === "asinh-auto" || stretch === "zscale") && info.auto) {
    const { lo, hi } = info.auto;
    const span = hi > lo ? hi - lo : Math.max(info.auto.knee ?? 1, 1e-30);
    if (stretch === "asinh-auto") {
      const kn = Math.max(info.auto.knee ?? span, 1e-30);
      const den = Math.asinh(span / kn);
      value = (f) => lo + (Math.sinh(f * den) * kn) / g;
      how = "asinh (auto limits)";
    } else {
      value = (f) => lo + (f * span) / g;
      how = "zscale";
    }
    how = `${how}: ${text(value(0))} – ${text(value(1))} ${unit}`;
  } else if (stretch === "linear" || stretch === "sqrt" || stretch === "log") {
    const x = stretch === "linear" ? (f: number) => f
      : stretch === "sqrt" ? (f: number) => f * f
        : (f: number) => (Math.pow(1001, f) - 1) / 1000;
    value = (f) => black + (x(f) * W) / g;
    how = `${stretch} stretch: ${text(value(0))} – ${text(value(1))} ${unit}`;
  } else {
    // asinh-abs (and an auto stretch whose limits are unknown): the locked
    // default, t = asinh((I − black)·gain / knee) / asinh(30·K0 / knee).
    const norm = Math.max(Math.asinh(W / Math.max(info.knee, 1e-30)), 1e-6);
    value = (f) => black + (Math.sinh(f * norm) * info.knee) / g;
    const electrons = unit === "e⁻" && info.scale === 1;
    how = electrons ? `asinh knee: ${Math.round(info.knee)} e⁻` : `asinh knee: ${publicationElectronLabel(info.knee / info.scale)} ${unit}`;
    if (black !== 0) how += `  ·  black ${text(black)} ${unit}`;
  }
  const ticks = fractions.map((fraction) => ({ fraction, label: text(value(fraction)) }));
  return { parameterText: `Band: ${info.band}  ·  ${how}`, ticks, signalLabel };
}

/** Largest nice scale-bar length ≤ 20% of the displayed side (arcsec). */
export function niceAngularScale(fullSideArcsec: number): number | null {
  if (!(fullSideArcsec > 0)) return null;
  const target = fullSideArcsec * 0.2;
  const candidates = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200];
  let best = candidates[0];
  for (const value of candidates) if (value <= target) best = value;
  return best;
}

export function publicationPanelName(tier: string, label: string | undefined): string {
  const t = String(tier || "").toLowerCase();
  const l = String(label || tier || "Image");
  const normalized = l.trim().toLowerCase();
  if (t === "lr" || t === "dirty" || normalized === "lr") return "Euclid Image";
  if (t === "sr" || normalized === "sr") return "Super-resolved Image";
  return l;
}

export function exportStem(collection: string, index: number, tiers: string[], color: string): string {
  const safe = (s: string) => s.replace(/[^\w.-]+/g, "_");
  return safe(`${collection || "cutout"}_idx${index}_${tiers.join("-") || "view"}_${color}`);
}

export type PublicationLayout = { columns: number; rows: number; side: number; width: number; height: number; outer: number; gap: number; header: number; footer: number; rowGap: number };

export function publicationLayout(count: number, layout: "one-row" | "two-rows"): PublicationLayout {
  const columns = layout === "two-rows" ? Math.ceil(count / 2) : count;
  const rows = Math.ceil(count / Math.max(1, columns));
  const outer = 28, gap = 14, header = 64, footer = 184, rowGap = 22;
  const preferred = count === 1 ? 1600 : count === 2 ? 1380 : 1260;
  const widthBound = Math.floor((PUBLICATION_MAX_WIDTH - 2 * outer - gap * (columns - 1)) / Math.max(1, columns));
  const side = Math.max(640, Math.min(preferred, widthBound));
  return {
    columns, rows, side, outer, gap, header, footer, rowGap,
    width: 2 * outer + columns * side + (columns - 1) * gap,
    height: 2 * outer + rows * (header + side + footer) + (rows - 1) * rowGap,
  };
}

function drawHeatbar(ctx: CanvasRenderingContext2D, model: HeatbarModel, x: number, imageBottom: number, side: number, stops?: string[]): void {
  const parameterSize = Math.max(23, side * 0.022);
  ctx.fillStyle = PUBLICATION_INK;
  ctx.font = `400 ${parameterSize}px Arial, Helvetica, sans-serif`;
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  ctx.fillText(model.parameterText, x, imageBottom + side * 0.034);

  const barWidth = side * 0.56;
  const barHeight = Math.max(20, side * 0.018);
  const barX = x + (side - barWidth) / 2;
  const barY = imageBottom + side * 0.060;
  const gradient = ctx.createLinearGradient(barX, 0, barX + barWidth, 0);
  if (stops && stops.length > 1) stops.forEach((c, i) => gradient.addColorStop(i / (stops.length - 1), c));
  else { gradient.addColorStop(0, "#000000"); gradient.addColorStop(1, "#ffffff"); }
  ctx.fillStyle = gradient;
  ctx.fillRect(barX, barY, barWidth, barHeight);
  ctx.strokeStyle = PUBLICATION_INK;
  ctx.lineWidth = 1;
  ctx.strokeRect(barX, barY, barWidth, barHeight);

  const tickSize = Math.max(18, side * 0.017);
  ctx.font = `400 ${tickSize}px Arial, Helvetica, sans-serif`;
  ctx.fillStyle = PUBLICATION_INK;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const tick of model.ticks) {
    const tickX = barX + tick.fraction * barWidth;
    ctx.beginPath();
    ctx.moveTo(tickX, barY + barHeight);
    ctx.lineTo(tickX, barY + barHeight + side * 0.009);
    ctx.stroke();
    ctx.fillText(tick.label, tickX, barY + barHeight + side * 0.013);
  }
  ctx.fillText(model.signalLabel, barX + barWidth / 2, barY + barHeight + side * 0.045);
}

/** One panel of a publication plate: the tier's natural-resolution canvas
 *  (the rendered cube), the frozen crop (or null for the whole image). */
export type FigurePanel = {
  source: CanvasImageSource;
  width: number;
  height: number;
  crop: Crop | null;
  name: string;
  pixscale: number | null;
  heatbar: HeatbarInfo;
};

function drawPanel(ctx: CanvasRenderingContext2D, p: FigurePanel, K0: number, x: number, y: number, side: number): void {
  ctx.save();
  ctx.fillStyle = PUBLICATION_INK;
  ctx.font = `600 ${Math.max(32, side * 0.030)}px Arial, Helvetica, sans-serif`;
  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  ctx.fillText(p.name, x, y - side * 0.014);

  ctx.fillStyle = FRAME_INK;
  ctx.fillRect(x, y, side, side);
  // Keep detector/reconstruction pixels literal: interpolation would make an
  // upscaled science image look smoother than the sampled data.
  ctx.imageSmoothingEnabled = false;
  let displayedSidePixels: number;
  if (p.crop) {
    ctx.drawImage(p.source, p.crop.x, p.crop.y, p.crop.side, p.crop.side, x, y, side, side);
    displayedSidePixels = p.crop.side;
  } else {
    const s = side / Math.max(p.width, p.height);
    const dw = p.width * s, dh = p.height * s;
    ctx.drawImage(p.source, 0, 0, p.width, p.height, x + (side - dw) / 2, y + (side - dh) / 2, dw, dh);
    displayedSidePixels = Math.max(p.width, p.height);
  }

  if (p.pixscale && p.pixscale > 0) {
    const displayedSideArcsec = displayedSidePixels * p.pixscale;
    const barArcsec = niceAngularScale(displayedSideArcsec) as number;
    const barWidth = (barArcsec / displayedSideArcsec) * side;
    const right = x + side - side * 0.035;
    const bottom = y + side - side * 0.035;
    ctx.strokeStyle = "rgba(0, 0, 0, .9)";
    ctx.lineWidth = Math.max(6, side * 0.007);
    ctx.beginPath(); ctx.moveTo(right - barWidth, bottom); ctx.lineTo(right, bottom); ctx.stroke();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = Math.max(2.5, side * 0.003);
    ctx.beginPath(); ctx.moveTo(right - barWidth, bottom); ctx.lineTo(right, bottom); ctx.stroke();
    ctx.font = `600 ${Math.max(20, side * 0.020)}px Arial, Helvetica, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.lineWidth = Math.max(3, side * 0.0035);
    ctx.strokeStyle = "rgba(0, 0, 0, .9)";
    const shown = barArcsec < 0.1 ? barArcsec.toFixed(2) : barArcsec < 1 ? barArcsec.toFixed(1) : barArcsec.toFixed(0);
    ctx.strokeText(`${shown}″`, right - barWidth / 2, bottom - side * 0.009);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(`${shown}″`, right - barWidth / 2, bottom - side * 0.009);
  }

  ctx.strokeStyle = PUBLICATION_INK;
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, side, side);
  drawHeatbar(ctx, heatbarModel(p.heatbar, K0), x, y + side, side, p.heatbar.signed?.stops ?? p.heatbar.stops);
  ctx.restore();
}

/** The whole publication plate. */
export function publicationFigureCanvas(panels: FigurePanel[], layout: "one-row" | "two-rows", K0: number): HTMLCanvasElement | null {
  if (!panels.length) return null;
  const L = publicationLayout(panels.length, layout);
  const out = document.createElement("canvas");
  out.width = L.width;
  out.height = L.height;
  const ctx = out.getContext("2d");
  if (!ctx) return null;
  ctx.fillStyle = PUBLICATION_PAPER;
  ctx.fillRect(0, 0, L.width, L.height);
  panels.forEach((p, i) => {
    const col = i % L.columns;
    const row = Math.floor(i / L.columns);
    drawPanel(ctx, p, K0,
      L.outer + col * (L.side + L.gap),
      L.outer + L.header + row * (L.header + L.side + L.footer + L.rowGap), L.side);
  });
  return out;
}

function drawLabel(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, maxW: number, background = "rgba(6, 9, 16, 0.68)"): void {
  if (!text) return;
  ctx.save();
  ctx.font = '11px "IBM Plex Mono", Menlo, monospace';
  const padX = 8, padY = 4, lineH = 15;
  let label = text;
  const limit = Math.max(20, maxW - 18);
  while (label.length > 1 && ctx.measureText(label).width + 2 * padX > limit) label = `${label.slice(0, -2)}…`;
  const w = Math.min(limit, ctx.measureText(label).width + 2 * padX);
  ctx.fillStyle = background;
  ctx.fillRect(x, y, w, lineH);
  ctx.fillStyle = "#cdd6e6";
  ctx.fillText(label, x + padX, y + lineH - padY);
  ctx.restore();
}

/** A visible frame for the PNG / video composite. */
export type CompositeFrame = { canvas: HTMLCanvasElement; rect: DOMRect | { left: number; top: number; right: number; bottom: number; width: number; height: number }; label: string; message: string };

/** Composite the visible frames as laid out on screen (≤ 2× device pixels). */
export function compositeFrames(frames: CompositeFrame[], target?: HTMLCanvasElement): HTMLCanvasElement | null {
  const shown = frames.filter((f) => f.rect.width > 1 && f.rect.height > 1);
  if (!shown.length) return null;
  const left = Math.min(...shown.map((f) => f.rect.left));
  const top = Math.min(...shown.map((f) => f.rect.top));
  const right = Math.max(...shown.map((f) => f.rect.right));
  const bottom = Math.max(...shown.map((f) => f.rect.bottom));
  const cssW = Math.max(1, right - left), cssH = Math.max(1, bottom - top);
  const scale = Math.max(1, Math.min(typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1, 2));
  const out = target || document.createElement("canvas");
  const pxW = Math.max(1, Math.round(cssW * scale)), pxH = Math.max(1, Math.round(cssH * scale));
  if (out.width !== pxW || out.height !== pxH) { out.width = pxW; out.height = pxH; }
  const ctx = out.getContext("2d");
  if (!ctx) return null;
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  ctx.fillStyle = FRAME_INK;
  ctx.fillRect(0, 0, cssW, cssH);
  ctx.imageSmoothingEnabled = false;
  for (const f of shown) {
    const x = f.rect.left - left, y = f.rect.top - top;
    ctx.fillStyle = FRAME_INK;
    ctx.fillRect(x, y, f.rect.width, f.rect.height);
    if (f.canvas.width > 1) ctx.drawImage(f.canvas, x, y, f.rect.width, f.rect.height);
    drawLabel(ctx, f.label, x + 9, y + 8, f.rect.width);
    if (f.message) drawLabel(ctx, f.message, x + 20, y + f.rect.height / 2 - 8, f.rect.width - 40, "rgba(6, 9, 16, 0.82)");
  }
  return out;
}

export function saveCanvasPng(canvas: HTMLCanvasElement, name: string): void {
  canvas.toBlob((b) => { if (b) downloadBlob(name, b); }, "image/png");
}

export type Recording = { stop: () => void };

/** Record `paint(target)` at 30 fps until stop(); downloads `<name>.webm`.
 *  Null when MediaRecorder is unavailable. */
export function recordCanvas(paint: (target: HTMLCanvasElement) => void, name: string, onStop?: () => void): Recording | null {
  if (typeof MediaRecorder === "undefined") return null;
  const out = document.createElement("canvas");
  let raf: number | null = null;
  const loop = () => { paint(out); raf = requestAnimationFrame(loop); };
  loop();
  const stream = (out as HTMLCanvasElement & { captureStream(fps: number): MediaStream }).captureStream(30);
  const mime = ["video/webm;codecs=vp9", "video/webm"].find((m) => MediaRecorder.isTypeSupported(m));
  const chunks: Blob[] = [];
  const recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  recorder.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
  recorder.onstop = () => {
    if (raf != null) cancelAnimationFrame(raf);
    stream.getTracks().forEach((t) => t.stop());
    downloadBlob(`${name}.webm`, new Blob(chunks, { type: "video/webm" }));
    onStop?.();
  };
  recorder.start();
  return { stop: () => { if (recorder.state !== "inactive") recorder.stop(); } };
}
