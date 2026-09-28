/* Small shared pieces of the Synthetic tabs: the toolbar (it scrolls with
   the page), load / error states that show the server's own message, an info
   popover for the long explanations (kept off the page), the state dot, the
   links into the sky atlas and the drawers every ingredient tab ends with
   (Prior, How this is produced, Real reference, Generate and sync).
   Statistics use the kit's SummaryLine / FactsList / Caption / Details
   (ui/facts.tsx); there are no stat tiles. */
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { holdInView } from "../../hooks/arrival";
import { Link } from "react-router-dom";
import type { ApiError } from "../../api/client";
import Plot, { type PlotProps, type PlotView } from "../../charts/Plot";
import { formatNumber } from "../../format";
import { useMediaQuery } from "../../hooks/useMediaQuery";
import { useUrlState } from "../../hooks/useUrlState";
import {
  Button, EmptyState, IconButton, Popover, RangeSlider, Section, Skeleton, Tooltip, type IconName, type Tone,
} from "../../ui";
import type { CheckState } from "./api";
import { autoTicks } from "../plotTicks";

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

/* ─── drawers ───────────────────────────────────────────────────────────── */

/** The DOM id of a drawer (its section), from its URL flag. */
export const drawerId = (flag: string) => `syn-drawer-${flag}`;

/** A drawer's state: open is the URL flag (`?prior=1`, `?how=1`, `?ref=1`,
 *  `?gen=1`; shareable, and the old pages' redirects set it), `reveal` opens
 *  it and scrolls it into view (instantly under reduced motion). */
export function useDrawer(flag: string) {
  const [open, setOpen] = useUrlState(flag, false);
  const reduce = useMediaQuery("(prefers-reduced-motion: reduce)");
  const reveal = useCallback(() => {
    setOpen(true);
    requestAnimationFrame(() => requestAnimationFrame(() => document.getElementById(drawerId(flag))
      ?.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" })));
  }, [flag, reduce, setOpen]);
  return { open, setOpen, reveal, id: drawerId(flag) };
}

/** One of a tab's drawers: a collapsible section at the foot of the page
 *  (it scrolls with the page and never covers the figures; its body renders
 *  only while open, so a closed drawer loads nothing). Opened from the URL,
 *  it scrolls itself into view and stays there while the page above loads. */
export function Drawer({ flag, title, sub, right, children }: {
  flag: string; title: string; sub?: ReactNode; right?: ReactNode; children: ReactNode;
}) {
  const d = useDrawer(flag);
  const opened = useRef(d.open);
  useEffect(() => {
    if (!opened.current) return undefined;
    opened.current = false;
    return holdInView(d.id);
  }, [d.id]);
  return (
    <Section id={d.id} className="syn-drawer" title={title} sub={sub} right={right} collapsible
      open={d.open} onOpenChange={d.setOpen}>
      {children}
    </Section>
  );
}

/** The toolbar button that opens (and scrolls to) a drawer. */
export function DrawerButton({ flag, icon, children, hint }: {
  flag: string; icon?: IconName; children: ReactNode; hint?: string;
}) {
  const d = useDrawer(flag);
  const button = (
    <Button size="sm" variant="ghost" icon={icon} aria-expanded={d.open} aria-controls={d.open ? d.id : undefined}
      onClick={d.reveal}>{children}</Button>
  );
  return hint ? <Tooltip content={hint}>{button}</Tooltip> : button;
}
