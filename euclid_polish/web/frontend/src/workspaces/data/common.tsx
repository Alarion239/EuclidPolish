/* Data workspace — shared pieces: the tab toolbar (one plain row that
 * scrolls with the page), freshness badges,
 * load/error states that show the server's text, sky links, and the job
 * starter every sync / generate button uses (confirm → POST → job tray →
 * invalidate the data resources → toast). */
import { useLayoutEffect, useRef, useState, type ReactNode, type RefObject } from "react";
import { Link } from "react-router-dom";
import { invalidate } from "../../api/query";
import type { Job, UseJob } from "../../api/jobs";
import type { FormRecord } from "../../api/client";
import { useFasrcStatus } from "../../app/status";
import { formatDateTime, formatRelative } from "../../format";
import { Badge, Button, Callout, Icon, JobProgress, Skeleton, Tooltip, confirm, toast, type IconName, type Tone } from "../../ui";
import { DATA_PREFIXES } from "./api";
import { atlasHref, compactLevelFor } from "./model";

/** The tab's toolbar: one plain row under the workspace tab strip (never
 *  pinned). With `compactable`, when the row does not fit with every label it
 *  steps through compact levels instead of wrapping onto a second row (so the
 *  viewer starts one row higher): level 1 — the buttons in its
 *  `<BarActions>` go icon-only (they keep their aria-label and title); a page
 *  may define more (`compactable={3}`: Records' badges go dot-only at 2, its
 *  overlay control becomes a menu at 3), styled by `data-compact="<level>"`. */
export function DataBar({ label, children, compactable = false }: {
  label: string; children: ReactNode; compactable?: boolean | number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const levels = compactable === true ? 1 : compactable === false ? 0 : Math.max(0, Math.floor(compactable));
  const level = useCompactLevel(ref, levels);
  return <div ref={ref} className="dt-bar" role="toolbar" aria-label={label} data-compact={level ? String(level) : undefined}>{children}</div>;
}

/** The toolbar's action buttons (icon-only while the bar is compact). */
export function BarActions({ children }: { children: ReactNode }) {
  return <div className="dt-bar__act">{children}</div>;
}

/** The least compact level (0…`levels`) at which a flex row fits on one
 *  line: at each level its children's natural widths plus the gaps are
 *  measured with that level applied (`compactLevelFor`), on resize and
 *  whenever a child changes size. The attribute is restored after reading, so
 *  the answer does not depend on the current form (no flip-flop). */
function useCompactLevel(ref: RefObject<HTMLElement>, levels: number): number {
  const [level, setLevel] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!levels || !el) return;
    let raf = 0;
    const rowWidth = () => {
      const kids = [...el.children] as HTMLElement[];
      const gap = parseFloat(getComputedStyle(el).columnGap) || 0;
      // Fractional widths (offsetWidth rounds): a row that fits to the pixel still wraps.
      return kids.reduce((w, k) => w + (k.classList.contains("dt-bar__spacer") ? 0 : k.getBoundingClientRect().width), 0)
        + gap * Math.max(0, kids.length - 1);
    };
    const measure = () => {
      raf = 0;
      const had = el.getAttribute("data-compact");
      const widths: number[] = [];
      for (let l = 0; l <= levels; l++) {
        if (l) el.setAttribute("data-compact", String(l)); else el.removeAttribute("data-compact");
        widths.push(rowWidth());
      }
      if (had == null) el.removeAttribute("data-compact"); else el.setAttribute("data-compact", had);
      setLevel(compactLevelFor(widths, el.clientWidth));
    };
    const schedule = () => { if (!raf) raf = requestAnimationFrame(measure); };
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(schedule) : null;
    const observeKids = () => { ro?.disconnect(); ro?.observe(el); for (const k of el.children) ro?.observe(k); };
    const mo = typeof MutationObserver !== "undefined" ? new MutationObserver(() => { observeKids(); schedule(); }) : null;
    observeKids();
    mo?.observe(el, { childList: true });
    measure();
    return () => { if (raf) cancelAnimationFrame(raf); ro?.disconnect(); mo?.disconnect(); };
  }, [ref, levels]);
  return level;
}

export const Spacer = () => <span className="dt-bar__spacer" />;

/** A labelled cluster inside the toolbar. */
export function BarGroup({ label, children }: { label: string; children: ReactNode }) {
  return <div className="dt-bar__group" role="group" aria-label={label}>{children}</div>;
}

/** "synced 3 h ago" with the absolute time in a tooltip. */
export function Freshness({ at, label = "synced", tone, stale = 7 * 24 * 3600 }: {
  at: number | null | undefined; label?: string; tone?: Tone; stale?: number;
}) {
  if (at == null) return <Badge size="sm">never {label}</Badge>;
  const age = Date.now() / 1000 - at;
  return (
    <Tooltip content={`${label} ${formatDateTime(at * 1000)}`}>
      <span tabIndex={0} className="dt-fresh">
        <Badge size="sm" tone={tone ?? (age > stale ? "warn" : "neutral")}>{label} {formatRelative(at * 1000)}</Badge>
      </span>
    </Tooltip>
  );
}

/** Loading skeleton, or the server's error with a retry, else the children. */
export function LoadState({ loading, error, onRetry, lines = 4, children }: {
  loading: boolean; error: { message: string } | null | undefined; onRetry?: () => void; lines?: number; children: ReactNode;
}) {
  if (loading) return <Skeleton lines={lines} />;
  if (error) {
    return (
      <Callout tone="bad" title="Could not load" action={onRetry ? <Button size="sm" onClick={onRetry}>Retry</Button> : undefined}>
        <span className="dt-pre">{error.message}</span>
      </Callout>
    );
  }
  return <>{children}</>;
}

/** A router link styled as a small ghost button, with its icon and a label
 *  that a compact toolbar hides (the link keeps its accessible name).
 *  (`Button asChild` renders only its child, so the icon goes in here.) */
export function LinkButton({ to, icon, label, hint }: { to: string; icon: IconName; label: string; hint?: string }) {
  const link = (
    <Button asChild size="sm" variant="ghost">
      <Link to={to} aria-label={label}><Icon name={icon} /><span className="ui-btn__label">{label}</span></Link>
    </Button>
  );
  return hint ? <Tooltip content={hint}>{link}</Tooltip> : link;
}

/** Router link to the Sky atlas (a small ghost button). */
export function SkyButton({ ra, dec, fov = 0.05, layers, inspect, label = "On sky", hint }: {
  ra?: number | null; dec?: number | null; fov?: number; layers?: string[]; inspect?: string; label?: string; hint?: string;
}) {
  return <LinkButton to={atlasHref({ ra, dec, fov, layers, inspect })} icon="globe" label={label} hint={hint} />;
}

/** FASRC connection state for gating the remote actions. */
export function useFasrcOnline(): { online: boolean; known: boolean } {
  const s = useFasrcStatus().data;
  return { online: !!s?.ssh_connected, known: s != null };
}

export const OFFLINE_HINT = "Needs the FASRC connection (Settings › Connections).";

export type StartOpts = {
  /** The confirm dialog; omit for no confirmation. */
  question?: { title: string; message: string; confirmLabel: string; tone?: "danger" | "default" };
  label: string;
  onDone?: (job: Job) => void;
};

/** Confirm, POST a job endpoint through `job` (tray + re-attach by key), then
 *  refresh every Data resource and toast the outcome. */
export async function startDataJob(job: UseJob, url: string, data: FormRecord, opts: StartOpts): Promise<void> {
  if (opts.question && !(await confirm(opts.question))) return;
  await job.run(url, data, {
    onDone: (j) => {
      for (const prefix of DATA_PREFIXES) void invalidate(prefix);
      if (j.status === "done") toast.success(`${opts.label}: done`);
      else if (j.status === "failed") toast.error(`${opts.label}: failed`, { description: (j.error ?? "").split("\n")[0] });
      else if (j.status === "cancelled") toast.warning(`${opts.label}: cancelled`);
      opts.onDone?.(j);
    },
  });
}

/** A running / just-finished job under the toolbar (hidden when idle). */
export function JobStrip({ job }: { job: UseJob }) {
  if (!job.job && !job.error) return null;
  return (
    <div className="dt-jobstrip">
      <JobProgress job={job.job} error={job.error} />
      {job.job && job.job.status !== "running" && (
        <Button size="sm" variant="ghost" icon="close" onClick={job.reset} aria-label="Dismiss the job">Dismiss</Button>
      )}
    </div>
  );
}
