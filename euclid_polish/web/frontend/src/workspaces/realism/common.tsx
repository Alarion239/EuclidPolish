/* Small shared pieces of the Realism tabs: the sticky toolbar, load / error
   states that show the server's own message, an info popover for the long
   explanations (kept off the page), the state dot and the links into the sky
   atlas. Statistics use the kit's SummaryLine / FactsList / Caption / Details
   (ui/facts.tsx); there are no stat tiles. */
import { useCallback, useEffect, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import type { ApiError } from "../../api/client";
import Plot, { type PlotProps, type PlotView } from "../../charts/Plot";
import { formatNumber } from "../../format";
import { useUrlState } from "../../hooks/useUrlState";
import { Button, EmptyState, IconButton, Popover, RangeSlider, Skeleton, Tooltip, type Tone } from "../../ui";
import type { CheckState } from "./api";
import { autoTicks } from "../plotTicks";

export function RealismBar({ children, label }: { children: ReactNode; label: string }) {
  return <div className="rl-bar" role="toolbar" aria-label={label}>{children}</div>;
}

export function BarGroup({ label, children }: { label?: string; children: ReactNode }) {
  return (
    <div className="rl-bar__group">
      {label && <span className="rl-bar__label">{label}</span>}
      {children}
    </div>
  );
}

export const BarSpacer = () => <span className="rl-bar__spacer" aria-hidden="true" />;

/** Loading → skeleton; error (no data) → the server's message + Retry. */
export function LoadState(
  { loading, error, onRetry, children, lines = 5 }: {
    loading: boolean; error: ApiError | null; onRetry?: () => void; children?: ReactNode; lines?: number;
  },
) {
  if (loading) return <Skeleton lines={lines} />;
  if (error) {
    return (
      <EmptyState icon="warn" title="Could not load"
        action={onRetry && <Button size="sm" icon="reset" onClick={onRetry}>Retry</Button>}>
        <span className="rl-mono">{error.message}</span>
      </EmptyState>
    );
  }
  return <>{children}</>;
}

/** The longer explanation of a figure, one click away. */
export function Info({ label, children, width = 360 }: { label: string; children: ReactNode; width?: number }) {
  return (
    <Popover label={label} width={width}
      trigger={<IconButton icon="info" size="sm" label={label} />}>
      <div className="rl-info">{children}</div>
    </Popover>
  );
}

export const STATE_TONE: Record<CheckState, Tone> = { ok: "good", warn: "warn", bad: "bad", unknown: "neutral" };

export function StateDot({ state, label }: { state: CheckState; label?: string }) {
  return <span className="rl-dot" data-state={state} role="img" aria-label={label ?? state} />;
}

/** `/sky/atlas?layers=…` (W-SkyAtlas URL contract, atlas/urlState.ts). */
export function atlasHref(opts: {
  layers: string[]; ra?: number; dec?: number; fov?: number; inspect?: string;
}): string {
  const q = new URLSearchParams();
  if (opts.ra != null && opts.dec != null && Number.isFinite(opts.ra) && Number.isFinite(opts.dec)) {
    q.set("ra", String(Number(opts.ra.toFixed(5))));
    q.set("dec", String(Number(opts.dec.toFixed(5))));
  }
  if (opts.fov != null) q.set("fov", String(opts.fov));
  q.set("layers", opts.layers.join(","));
  if (opts.inspect) q.set("inspect", opts.inspect);
  return `/sky/atlas?${q.toString().replace(/%2C/g, ",").replace(/%3A/g, ":").replace(/%2F/g, "/")}`;
}

/** "Open on sky" link button. */
export function SkyLink(
  { layers, ra, dec, fov, inspect, children = "On sky", size = "sm", hint }: {
    layers: string[]; ra?: number; dec?: number; fov?: number; inspect?: string; children?: ReactNode;
    size?: "sm" | "md"; hint?: string;
  },
) {
  const link = (
    <Button asChild size={size} variant="ghost" icon="globe">
      <Link to={atlasHref({ layers, ra, dec, fov, inspect })}>{children}</Link>
    </Button>
  );
  return hint ? <Tooltip content={hint}>{link}</Tooltip> : link;
}

/** A legend whose hidden keys live in the URL (`key` = comma list), shaped
 *  like charts' `useLegend` (plotProps / legendProps). */
export function useUrlLegend(key: string) {
  const [hidden, setHidden] = useUrlState<string[]>(key, []);
  const [emphasis, setEmphasis] = useState<string | null>(null);
  const toggle = useCallback((k: string) => {
    setHidden(hidden.includes(k) ? hidden.filter((h) => h !== k) : [...hidden, k]);
  }, [hidden, setHidden]);
  return {
    hidden, setHidden, toggle, emphasis,
    plotProps: { hidden, onHiddenChange: setHidden, emphasis },
    legendProps: { hidden, onToggle: toggle, onHover: setEmphasis },
  };
}

/** A swatch for legends drawn in HTML (token colour passed in). */
export function Swatch({ color, dash = false }: { color: string; dash?: boolean }) {
  return <i className="rl-swatch" data-dash={dash || undefined} style={{ ["--sw" as string]: color }} aria-hidden="true" />;
}

/** Download a server URL (a figure) without leaving the page. */
export function downloadUrl(url: string, name: string): void {
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/** A Plot v2 whose zoom (drag / Ctrl-wheel / double-click reset) also has
 *  exact bounds one click away: a popover with a RangeSlider per axis over
 *  the data's full domain (log tracks on log axes). Replaces the old
 *  per-page AdjustablePlot / DualRange. */
export function BoundedPlot(props: PlotProps & { boundsLabel: string }) {
  const { boundsLabel, ...plot } = props;
  const [view, setView] = useState<PlotView | null>(null);
  // New data domains (a band toggle, a rebuild) start from the full range.
  const [d0, d1, d2, d3] = [...plot.xDomain, ...plot.yDomain];
  useEffect(() => { setView(null); }, [d0, d1, d2, d3]);
  const x = view?.x ?? plot.xDomain;
  const y = view?.y ?? plot.yDomain;
  const custom = view != null && (view.x != null || view.y != null);
  const fmt = (v: number) => formatNumber(v, { sig: 4 });
  const axis = (which: "x" | "y") => {
    const full = which === "x" ? plot.xDomain : plot.yDomain;
    const scale = (which === "x" ? plot.xScale : plot.yScale) === "log" && full[0] > 0 ? "log" : "linear";
    const value = which === "x" ? x : y;
    return (
      <label className="rl-bounds__axis">
        <span className="rl-subhead">{which} range</span>
        <RangeSlider min={full[0]} max={full[1]} scale={scale} value={[Math.max(full[0], value[0]), Math.min(full[1], value[1])]}
          step={scale === "log" ? undefined : (full[1] - full[0]) / 1000} format={fmt} showValue
          aria-label={`${boundsLabel} ${which}`}
          onChange={(v) => setView({ x: which === "x" ? v : view?.x ?? null, y: which === "y" ? v : view?.y ?? null })} />
      </label>
    );
  };
  return (
    <div className="rl-bounded">
      {/* a caller that picks no ticks still gets labelled axes (Plot draws only the ticks it is given) */}
      <Plot {...plot} view={view} onViewChange={setView}
        xTicks={plot.xTicks ?? autoTicks(plot.xDomain, plot.xScale, plot.xFormat)}
        yTicks={plot.yTicks ?? autoTicks(plot.yDomain, plot.yScale, plot.yFormat)} />
      <div className="rl-bounded__bar">
        <Popover label={`${boundsLabel}: axis bounds`} width={320}
          trigger={<Button size="sm" variant="ghost" icon="zoomIn">{custom ? "custom bounds" : "bounds"}</Button>}>
          <div className="rl-bounds">
            {axis("x")}
            {axis("y")}
            <Button size="sm" variant="ghost" icon="reset" disabled={!custom} onClick={() => setView(null)}>Full range</Button>
          </div>
        </Popover>
      </div>
    </div>
  );
}
