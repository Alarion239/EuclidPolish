/* Back-trace a diagnostic heatmap cell to the real pixels that landed in it
   (GET /ensemble/pixel-trace.json): LR · target · SR · σ stamps from across
   the test fields, coloured by the Display panel (colour mode, knee, gain)
   through the viewer's own colour core (viewer/color.ts). */
import { useEffect, useMemo, useRef } from "react";
import { useResource } from "../../api/query";
import { viridis } from "../../colors";
import { transferFor, useDisplay } from "../../state/display";
import { Button, EmptyState, IconButton, Skeleton } from "../../ui";
import { renderCubeImageData, type ColorMeta, type RenderOpts } from "../../viewer";
import { url, type Mode } from "./api";

export type Pick = { diag: "std_err" | "bright_std" | "combiner_feature_error"; i: number; j: number };
type Stamp = {
  field: number; y: number; x: number; center: number; sr_is_combiner: boolean;
  lr?: string; hr: string; sr: string; std: string;
  hr_val: number; sr_val: number; std_val: number; err_val: number; bright_asinh: number; model_kind?: string;
};
type Trace = { diag: string; i: number; j: number; half: number; size: number; bands: string[]; stretch: number; stamps: Stamp[] };

const STAMP_PX = 3;
const RENDERABLE = new Set(["VIS", "Y_E", "J_E", "H_E", "lupton", "temp"]);
const fmtE = (v: number) => (!Number.isFinite(v) ? "—" : Math.abs(v) >= 1000 || (Math.abs(v) > 0 && Math.abs(v) < 0.01)
  ? v.toExponential(1) : String(Number(v.toPrecision(3))));

function b64ToF32(b64: string): Float32Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function ringColor(): string {
  return getComputedStyle(document.documentElement).getPropertyValue("--bad").trim() || "currentColor";
}

function paint(cv: HTMLCanvasElement, size: number, center: number, draw: (ctx: CanvasRenderingContext2D) => void) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  cv.width = size * STAMP_PX * dpr; cv.height = size * STAMP_PX * dpr;
  cv.style.width = `${size * STAMP_PX}px`; cv.style.height = `${size * STAMP_PX}px`;
  const ctx = cv.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = false;
  draw(ctx);
  ctx.strokeStyle = ringColor(); ctx.lineWidth = 1.4;
  ctx.strokeRect(center * STAMP_PX - 0.7, center * STAMP_PX - 0.7, STAMP_PX + 1.4, STAMP_PX + 1.4);
}

function ImageStamp({ b64, size, center, bands, meta, opts, label }: {
  b64?: string; size: number; center: number; bands: string[]; meta?: ColorMeta; opts: RenderOpts; label: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv || !b64 || !meta || !bands.length) return;
    const img = renderCubeImageData({ data: b64ToF32(b64), h: size, w: size, c: bands.length, bands }, meta, opts);
    paint(cv, size, center, (ctx) => {
      const off = document.createElement("canvas");
      off.width = size; off.height = size;
      off.getContext("2d")?.putImageData(img, 0, 0);
      ctx.drawImage(off, 0, 0, size * STAMP_PX, size * STAMP_PX);
    });
  }, [b64, size, center, bands, meta, opts]);
  return (
    <div>
      {b64 ? <canvas ref={ref} aria-label={`${label} stamp`} role="img" />
        : <div className="ens-trace__na" style={{ width: size * STAMP_PX, height: size * STAMP_PX }}>n/a</div>}
      <div className="ens-trace__tier">{label}</div>
    </div>
  );
}

/** σ is the thing shown: fixed viridis, asinh, normalised per stamp. */
function SigmaStamp({ b64, size, center, stretch }: { b64: string; size: number; center: number; stretch: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const a = b64ToF32(b64);
    const t = new Float32Array(a.length);
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < a.length; i++) { const v = Math.asinh(a[i] / stretch); t[i] = v; lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const span = hi - lo || 1;
    paint(cv, size, center, (ctx) => {
      for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
        ctx.fillStyle = viridis((t[y * size + x] - lo) / span);
        ctx.fillRect(x * STAMP_PX, y * STAMP_PX, STAMP_PX + 0.5, STAMP_PX + 0.5);
      }
    });
  }, [b64, size, center, stretch]);
  return <div><canvas ref={ref} aria-label="σ stamp" role="img" /><div className="ens-trace__tier">σ</div></div>;
}

export function PixelTrace({ mode, pick, model, axis, cellLabel, targetLabel, onClose }: {
  mode: Mode; pick: Pick; model?: string; axis?: string; cellLabel: string; targetLabel: string; onClose: () => void;
}) {
  const q = new URLSearchParams({ mode, diag: pick.diag, i: String(pick.i), j: String(pick.j) });
  if (model) q.set("model", model);
  if (axis) q.set("axis", axis);
  const trace = useResource<Trace>(`/ensemble/pixel-trace.json?${q.toString()}`, [q.toString()]);
  const meta = useResource<{ color?: ColorMeta }>(url.viewerMeta(mode), [mode], { ttl: 5 * 60_000 });
  const display = useDisplay();
  const colorMeta = meta.data?.color;
  const K0 = colorMeta?.default_asinh ?? 100;
  const g = transferFor(display, "default");
  const color = RENDERABLE.has(display.color) ? display.color : "VIS";
  const opts: RenderOpts = useMemo(() => ({ color, knee: g.knee || K0, gain: g.gain || 1, K0 }), [color, g.knee, g.gain, K0]);
  const t = trace.data;
  return (
    <div className="ens-trace" aria-live="polite">
      <div className="ens-row" style={{ justifyContent: "space-between", marginBottom: "var(--s2)" }}>
        <span className="ens-bar__label">Back-trace · <span className="ens-mono" style={{ textTransform: "none" }}>{cellLabel}</span></span>
        <IconButton icon="close" label="Close the back-trace" size="sm" onClick={onClose} />
      </div>
      {trace.loading ? <Skeleton lines={3} />
        : trace.error ? <EmptyState compact icon="warn" title="Could not back-trace"
            action={<Button size="sm" onClick={trace.reload}>Retry</Button>}><span className="ens-mono">{trace.error.message}</span></EmptyState>
          : !t || !t.stamps.length ? <EmptyState compact icon="search" title="No pixels sampled in this cell">Try a denser cell.</EmptyState>
            : (
              <div className="ens-trace__grid">
                {t.stamps.map((s, k) => (
                  <div key={k} className="ens-trace__card">
                    <div className="ens-trace__stamps">
                      <ImageStamp b64={s.lr} size={t.size} center={s.center} bands={t.bands} meta={colorMeta} opts={opts} label="LR" />
                      <ImageStamp b64={s.hr} size={t.size} center={s.center} bands={t.bands} meta={colorMeta} opts={opts} label={targetLabel} />
                      <ImageStamp b64={s.sr} size={t.size} center={s.center} bands={t.bands} meta={colorMeta} opts={opts}
                        label={s.model_kind === "spatial_gate" ? "SR · gate" : s.sr_is_combiner ? "SR · combiner" : "SR · mean"} />
                      <SigmaStamp b64={s.std} size={t.size} center={s.center} stretch={t.stretch} />
                    </div>
                    <div className="ens-trace__nums">
                      field {s.field} · σ {fmtE(s.std_val)} · |err| {fmtE(s.err_val)} · {targetLabel} {fmtE(s.hr_val)} e⁻
                    </div>
                  </div>
                ))}
              </div>
            )}
    </div>
  );
}
