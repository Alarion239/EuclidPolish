/* Back-trace a diagnostic heatmap cell to the real pixels that landed in it
   (GET /ensemble/pixel-trace.json): one row per sampled pixel with its LR ·
   target · SR · σ stamps, coloured through the viewer's own colour core
   (viewer/color.ts) on the viewer's neutral dark light table. The traced
   pixels sit at a few e⁻, so each ROW gets its own asinh knee from the traced
   pixel's level (`stampKnee`: its |target|, σ and |err|; shared by the row's
   LR / target / SR stamps so they stay comparable; white at 30× the knee),
   named in the row's header line — the Display panel's fixed 100 e⁻ knee
   left the target and SR stamps black. "Display panel knee" switches back.
   The table is the viewer's light table (`.cv-root .cv-table`: its scoped dark
   palette, one definition in viewer.css). The colour and brightness follow the
   Display panel. Stamps fill the row (≤ 240 px each), each backing store at
   the exact device size (viewer/draw.ts: equal-sized pixels, no dropped
   rows) with the sampled pixel ringed; a picked cell scrolls the trace into
   view; each row links to its field in the Models › Images viewer. */
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useResource } from "../../api/query";
import { pagePath } from "../../app/nav";
import { viridis } from "../../colors";
import { transferFor, useDisplay } from "../../state/display";
import { Button, EmptyState, IconButton, Segmented, Skeleton } from "../../ui";
import { renderCubeImageData, type ColorMeta, type RenderOpts } from "../../viewer";
import { drawFrame } from "../../viewer/draw";
import { REGIME, url, type EvalBand } from "./api";
import { bandLabel, formatE, stampBacking, stampKnee } from "./model";

export type Pick = { diag: "std_err" | "bright_std"; i: number; j: number };
type Stamp = {
  field: number; y: number; x: number; center: number; sr_is_combiner: boolean;
  lr?: string; hr: string; sr: string; std: string;
  hr_val: number; sr_val: number; std_val: number; err_val: number; bright_asinh: number; model_kind?: string;
};
type Trace = { diag: string; i: number; j: number; half: number; size: number; bands: string[]; stretch: number; stamps: Stamp[] };

/** Backing-store upscale before a stamp has been laid out (then: its exact device size). */
const UPSCALE = 8;
const RENDERABLE = new Set(["VIS", "Y_E", "J_E", "H_E", "lupton", "temp"]);

function b64ToF32(b64: string): Float32Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

/** The stamp box's laid-out width in CSS px (fractional: the grid column). */
function boxWidth(cv: HTMLCanvasElement): number {
  return (cv.parentElement ?? cv).getBoundingClientRect().width;
}

/** Paint natural-size pixels into the canvas at its exact displayed device
 *  size: the backing is the whole number of device pixels that fits the
 *  stamp box (`stampBacking`) and the canvas's CSS box snaps to it, so the
 *  pixelated canvas is never rescaled (nearest-neighbour at an integer scale,
 *  else sharp bilinear, viewer/draw.ts) — a 284 px backing shown in a
 *  141.5 css px cell (283 device px) dropped a row. */
function paint(cv: HTMLCanvasElement, img: ImageData) {
  const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
  const fit = stampBacking(boxWidth(cv), dpr);
  const side = fit?.side ?? img.width * UPSCALE;
  if (cv.width !== side || cv.height !== side) { cv.width = side; cv.height = side; }
  const css = fit ? `${fit.css}px` : "";
  if (cv.style.width !== css) { cv.style.width = css; cv.style.height = css; }
  const ctx = cv.getContext("2d");
  if (!ctx) return;
  const off = document.createElement("canvas");
  off.width = img.width; off.height = img.height;
  off.getContext("2d")?.putImageData(img, 0, 0);
  ctx.clearRect(0, 0, side, side);
  drawFrame(ctx, off, { sx: 0, sy: 0, sw: img.width, sh: img.height, dx: 0, dy: 0, dw: side, dh: side }, 1, document.createElement("canvas"));
}

/** Repaint when the stamp box's displayed size changes (the box, not the
 *  canvas: the canvas's own CSS size is pinned to its backing). */
function useStampPaint(ref: React.RefObject<HTMLCanvasElement>, render: (() => ImageData | null) | null, deps: unknown[]) {
  useLayoutEffect(() => {
    const cv = ref.current;
    if (!cv || !render) return;
    const img = render();
    if (!img) return;
    paint(cv, img);
    if (typeof ResizeObserver === "undefined") return;
    let last = boxWidth(cv);
    const ro = new ResizeObserver(() => { const w = boxWidth(cv); if (w !== last) { last = w; paint(cv, img); } });
    ro.observe(cv.parentElement ?? cv);
    return () => ro.disconnect();
    // the caller lists what the rendered image depends on
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}

/** The ring around the sampled pixel, in % of the stamp (any displayed size). */
function Mark({ size, center }: { size: number; center: number }) {
  const pct = (v: number) => `${(100 * v) / size}%`;
  return <span className="mdl-trace__mark" aria-hidden style={{ left: pct(center), top: pct(center), width: pct(1), height: pct(1) }} />;
}

function ImageStamp({ b64, size, center, bands, meta, opts, label }: {
  b64?: string; size: number; center: number; bands: string[]; meta?: ColorMeta; opts: RenderOpts; label: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useStampPaint(ref, b64 && meta && bands.length
    ? () => renderCubeImageData({ data: b64ToF32(b64), h: size, w: size, c: bands.length, bands }, meta, opts)
    : null, [b64, size, center, bands, meta, opts]);
  return (
    <div className="mdl-trace__stamp">
      {b64 ? <><canvas ref={ref} aria-label={`${label} stamp`} role="img" /><Mark size={size} center={center} /></>
        : <span className="mdl-trace__na">No {label}</span>}
    </div>
  );
}

/** σ is the thing shown: fixed viridis, asinh, normalised per stamp. */
function SigmaStamp({ b64, size, center, stretch }: { b64: string; size: number; center: number; stretch: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useStampPaint(ref, () => {
    const a = b64ToF32(b64);
    const t = new Float32Array(a.length);
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < a.length; i++) { const v = Math.asinh(a[i] / stretch); t[i] = v; lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const span = hi - lo || 1;
    const img = new ImageData(size, size);
    for (let k = 0; k < size * size && k < t.length; k++) {
      const hex = viridis((t[k] - lo) / span);          // "#rrggbb"
      img.data[4 * k] = parseInt(hex.slice(1, 3), 16);
      img.data[4 * k + 1] = parseInt(hex.slice(3, 5), 16);
      img.data[4 * k + 2] = parseInt(hex.slice(5, 7), 16);
      img.data[4 * k + 3] = 255;
    }
    return img;
  }, [b64, size, center, stretch]);
  return (
    <div className="mdl-trace__stamp">
      <canvas ref={ref} aria-label="σ stamp" role="img" /><Mark size={size} center={center} />
    </div>
  );
}

/** The Images viewer on this stamp's test field (`test:<index>`). */
const fieldHref = (field: number) =>
  `${pagePath("models", { tab: "images" })}?${new URLSearchParams({ "v.ens.id": `test:${field}` }).toString()}`;

export function PixelTrace({ pick, model, band = "VIS", cellLabel, onClose }: {
  pick: Pick; model?: string; band?: EvalBand; cellLabel: string; onClose: () => void;
}) {
  const q = new URLSearchParams({ mode: REGIME, diag: pick.diag, i: String(pick.i), j: String(pick.j) });
  if (model) q.set("model", model);
  if (band !== "VIS") q.set("band", band);
  const trace = useResource<Trace>(`/ensemble/pixel-trace.json?${q.toString()}`, [q.toString()]);
  const meta = useResource<{ color?: ColorMeta }>(url.viewerMeta(), [], { ttl: 5 * 60_000 });
  const display = useDisplay();
  const colorMeta = meta.data?.color;
  const K0 = colorMeta?.default_asinh ?? 100;
  const g = transferFor(display, "default");
  const color = RENDERABLE.has(display.color) ? display.color : "VIS";
  const [kneeFrom, setKneeFrom] = useState<"pixel" | "display">("pixel");
  const panelOpts: RenderOpts = useMemo(() => ({ color, knee: g.knee || K0, gain: g.gain || 1, K0 }), [color, g.knee, g.gain, K0]);
  /** One knee per row (LR / target / SR comparable), white at 30× it. */
  const rowOpts = (s: Stamp): RenderOpts => {
    if (kneeFrom === "display") return panelOpts;
    const k = stampKnee(s);
    return { color, knee: k, gain: g.gain || 1, K0: k };
  };
  const t = trace.data;
  // A picked cell scrolls its trace into view (it sits under the heatmap)
  // once its rows are in (the skeleton alone is a few px tall).
  const box = useRef<HTMLElement>(null);
  const loaded = !!t && !trace.loading;
  useEffect(() => {
    if (!loaded) return;
    const reduce = typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    box.current?.scrollIntoView?.({ block: "nearest", behavior: reduce ? "auto" : "smooth" });
  }, [pick.diag, pick.i, pick.j, loaded]);
  const srLabel = (s: Stamp) => (s.model_kind === "spatial_gate" ? "SR · gate" : s.sr_is_combiner ? "SR · combiner" : "SR · mean");
  const colourName = color === "lupton" ? "Lupton" : color === "temp" ? "Temp" : color.replace(/_E$/, "");
  return (
    <section ref={box} className="mdl-trace" aria-live="polite" aria-label="Pixels in the clicked cell">
      <header className="mdl-trace__head">
        <h3 className="mdl-trace__title">Pixels in this cell</h3>
        <span className="mdl-trace__cell">{cellLabel}</span>
        <span className="mdl-bar__spacer" />
        <Segmented size="sm" aria-label="Stamp knee" value={kneeFrom} onChange={setKneeFrom}
          options={[{ value: "pixel", label: "Knee from the pixel" }, { value: "display", label: "Display panel knee" }]} />
        <IconButton icon="close" label="Close the back-trace" size="sm" onClick={onClose} />
      </header>
      {trace.loading ? <Skeleton lines={3} />
        : trace.error ? <EmptyState compact icon="warn" title="Could not back-trace"
            action={<Button size="sm" onClick={trace.reload}>Retry</Button>}><span className="mdl-mono">{trace.error.message}</span></EmptyState>
          : !t || !t.stamps.length ? <EmptyState compact icon="search" title="No pixels sampled in this cell">Try a denser cell.</EmptyState>
            : (
              <>
                <div className="mdl-trace__table cv-root cv-table" role="list" aria-label={`${t.stamps.length} sampled pixels`}>
                  <div className="mdl-trace__cols" aria-hidden>
                    <span>LR</span><span>HR</span><span>{srLabel(t.stamps[0])}</span><span>σ (members)</span><span>Pixel</span>
                  </div>
                  {t.stamps.map((s, k) => {
                    const opts = rowOpts(s);
                    return (
                      <div key={k} className="mdl-trace__row" role="listitem" aria-label={`Field ${s.field}, x ${s.x}, y ${s.y}`}>
                        <div className="mdl-trace__rowhead">
                          <strong>Field {s.field}</strong>
                          <span>x {s.x}, y {s.y}</span>
                          <span className="mdl-trace__knee" title={kneeFrom === "pixel"
                            ? "One asinh knee for this row's LR, HR and SR stamps: the traced pixel's own level (white at 30× it)"
                            : "The Display panel's knee"}>knee {formatE(opts.knee ?? K0)} e⁻</span>
                        </div>
                        <ImageStamp b64={s.lr} size={t.size} center={s.center} bands={t.bands} meta={colorMeta} opts={opts} label="LR" />
                        <ImageStamp b64={s.hr} size={t.size} center={s.center} bands={t.bands} meta={colorMeta} opts={opts} label="HR" />
                        <ImageStamp b64={s.sr} size={t.size} center={s.center} bands={t.bands} meta={colorMeta} opts={opts} label={srLabel(s)} />
                        <SigmaStamp b64={s.std} size={t.size} center={s.center} stretch={t.stretch} />
                        <div className="mdl-trace__nums">
                          <span>σ {formatE(s.std_val)} e⁻</span>
                          <span>|err| {formatE(s.err_val)} e⁻</span>
                          <span>HR {formatE(s.hr_val)} e⁻</span>
                          <Link to={fieldHref(s.field)}>Open field {s.field} in the viewer</Link>
                        </div>
                      </div>
                    );
                  })}
                </div>
                <p className="mdl-trace__note">
                  {t.size} × {t.size} px around each sampled pixel (ringed), in {colourName}
                  {kneeFrom === "pixel" ? " with a knee at the traced pixel's own level (one per row, white at 30× it)" : " with the Display panel's knee"} and the
                  Display panel&apos;s brightness; σ is the cross-member spread (viridis, asinh, scaled per stamp). Numbers are {bandLabel(band)}.
                </p>
              </>
            )}
    </section>
  );
}
