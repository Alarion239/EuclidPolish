/* Pure helpers of Runs › Resources (no React): the step list's line, the
 * summary counts, the idle share of allocated hours, the per-run chart
 * series (oldest → newest, OOM / TIMEOUT runs marked), the requested-vs-used
 * fractions and tones of the run table, and where a step is submitted.
 * Unit-tested in resourcesModel.test.ts. */
import { pagePath } from "../../app/nav";
import { formatCount, formatDate, formatNumber } from "../../format";
import type { Tick } from "../../ticks";
import type { RunUsage, StateCounts, StepSummary } from "./api";
import { stepHome } from "./model";

const fin = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

/** Runs that ended (every counted state but RUNNING). */
export const finishedRuns = (s: StateCounts): number => s.completed + s.oom + s.timeout + s.failed + s.cancelled;

/** "54%" of a 0–1 fraction ("<1%" for a sliver, never a false "0%"), "—" when unknown. */
export const pctText = (fraction: number | null | undefined): string => {
  if (!fin(fraction)) return "—";
  return fraction > 0 && fraction < 0.005 ? "<1%" : `${Math.round(fraction * 100)}%`;
};

/** "71%" of a 0–100 utilisation, "—" when unknown. */
export const utilText = (percent: number | null | undefined): string => (fin(percent) ? `${Math.round(percent)}%` : "—");

/** "766" / "1,230" CPU-hours: three significant figures. */
export const hoursText = (h: number | null | undefined): string => formatNumber(h, { sig: 3 });

export type Waste = { alloc: number; used: number; idle: number; share: number };

/** Allocated hours nobody used: `idle` and its share of `alloc`, or null
 *  when either side is unknown. */
export function waste(alloc: number | null | undefined, used: number | null | undefined): Waste | null {
  if (!fin(alloc) || alloc <= 0 || !fin(used)) return null;
  const idle = Math.max(0, alloc - used);
  return { alloc, used, idle, share: idle / alloc };
}

export type Clause = { n: number; text: string; tone?: "bad" | "warn" };

/** What happened to the runs that did not complete, worst first, for the
 *  summary sentence ("3 ran out of memory", "16 hit the time limit", …);
 *  zero counts are left out. */
export function outcomeClauses(s: StateCounts): Clause[] {
  const all: Clause[] = [
    { n: s.oom, text: "ran out of memory", tone: "bad" },
    { n: s.timeout, text: "hit the time limit", tone: "warn" },
    { n: s.failed, text: "failed" },
    { n: s.cancelled, text: s.cancelled === 1 ? "was cancelled" : "were cancelled" },
    { n: s.running, text: s.running === 1 ? "is still running" : "are still running" },
  ];
  return all.filter((c) => c.n > 0);
}

/** The step list's line under each step's label: "58 runs · 54% completed ·
 *  CPU 42% · GPU 71%" (the OOM / timeout counts are badges beside it). */
export function stepMeta(s: StepSummary): string {
  const parts = [`${formatCount(s.runs)} run${s.runs === 1 ? "" : "s"}`];
  if (fin(s.success_rate)) parts.push(`${pctText(s.success_rate)} completed`);
  if (fin(s.cpu_efficiency)) parts.push(`CPU ${pctText(s.cpu_efficiency)}`);
  if (s.needs_gpu && fin(s.gpu_util)) parts.push(`GPU ${utilText(s.gpu_util)}`);
  return parts.join(" · ");
}

/** Where a step is submitted: its home tab (Models › Train, Synthetic ›
 *  Records' generate drawer, …), else its card in Runs › Steps. */
export function submitHome(stepId: string): { label: string; to: string } {
  return stepHome(stepId) ?? {
    label: "Runs › Steps", to: `${pagePath("runs", { tab: "steps" })}?${new URLSearchParams({ step: stepId }).toString()}`,
  };
}

/* ── per-run charts ───────────────────────────────────────────────────────── */

const when = (r: RunUsage): number => {
  const t = r.submitted_at ? Date.parse(r.submitted_at) : NaN;
  return Number.isFinite(t) ? t : -Infinity;
};

/** The runs oldest → newest (the endpoint lists them newest first); runs
 *  submitted at the same time keep their relative order. */
export function chronological(runs: readonly RunUsage[]): RunUsage[] {
  return runs.map((r, i) => ({ r, i })).sort((a, b) => when(a.r) - when(b.r) || b.i - a.i).map((x) => x.r);
}

export const isOom = (state: string | null | undefined) => String(state ?? "").toUpperCase().startsWith("OUT_OF_ME");
export const isTimeout = (state: string | null | undefined) => String(state ?? "").toUpperCase() === "TIMEOUT";

export type RunSeries = {
  /** 1 … n, oldest first. */
  x: number[];
  requested: (number | null)[];
  used: (number | null)[];
  /** The runs that failed on this resource, at what they got to. */
  marked: { x: number[]; y: number[] };
};

const gb = (mb: number | null | undefined) => (fin(mb) ? mb / 1024 : null);
const hours = (s: number | null | undefined) => (fin(s) ? s / 3600 : null);

/** Peak memory vs requested per run, in GB; OUT_OF_MEMORY runs marked at
 *  the larger of the two (it needed more than it asked for). */
export function memorySeries(runs: readonly RunUsage[]): RunSeries {
  const out: RunSeries = { x: [], requested: [], used: [], marked: { x: [], y: [] } };
  runs.forEach((r, i) => {
    const req = gb(r.req_memory_mb), used = gb(r.peak_mem_mb);
    out.x.push(i + 1); out.requested.push(req); out.used.push(used);
    const top = Math.max(req ?? -Infinity, used ?? -Infinity);
    if (isOom(r.state) && Number.isFinite(top)) { out.marked.x.push(i + 1); out.marked.y.push(top); }
  });
  return out;
}

/** Elapsed vs time limit per run, in hours; TIMEOUT runs marked at their
 *  elapsed (it needed more than it got). */
export function timeSeries(runs: readonly RunUsage[]): RunSeries {
  const out: RunSeries = { x: [], requested: [], used: [], marked: { x: [], y: [] } };
  runs.forEach((r, i) => {
    const limit = hours(r.req_time_s), elapsed = hours(r.elapsed_s);
    out.x.push(i + 1); out.requested.push(limit); out.used.push(elapsed);
    const top = elapsed ?? limit;
    if (isTimeout(r.state) && top != null) { out.marked.x.push(i + 1); out.marked.y.push(top); }
  });
  return out;
}

/** The top of a chart's y axis: 8 % above the largest value (never below 1). */
export function yTop(s: RunSeries): number {
  const vals = [...s.requested, ...s.used, ...s.marked.y].filter(fin);
  return Math.max(1, ...vals) * 1.08;
}

/** A chart's y axis: linear from 0 to `yTop`, unless the positive values span
 *  more than `LOG_SPAN`× (one 30 h time limit among 10 min runs), where a
 *  linear axis flattens every other run into the floor — then log. */
export const LOG_SPAN = 30;
export function yAxis(s: RunSeries): { scale: "linear" | "log"; domain: [number, number] } {
  const pos = [...s.requested, ...s.used, ...s.marked.y].filter((v): v is number => fin(v) && v > 0);
  const lo = Math.min(...pos), hi = Math.max(...pos);
  if (pos.length < 2 || hi / lo <= LOG_SPAN) return { scale: "linear", domain: [0, yTop(s)] };
  return { scale: "log", domain: [lo / 1.5, hi * 1.5] };
}

/** At most `max` x ticks at evenly spaced runs, labelled with the
 *  submission day ("09-14"); a day already labelled is not repeated. */
export function runTicks(runs: readonly RunUsage[], max = 5): Tick[] {
  const n = runs.length;
  if (!n) return [];
  const count = Math.min(n, Math.max(1, max));
  const idx = count === 1 ? [0] : Array.from({ length: count }, (_, k) => Math.round((k * (n - 1)) / (count - 1)));
  const ticks: Tick[] = [];
  let last = "";
  for (const i of [...new Set(idx)]) {
    const day = formatDate(runs[i].submitted_at, { fallback: "" }).slice(5);
    if (!day || day === last) continue;
    ticks.push({ v: i + 1, label: day });
    last = day;
  }
  return ticks;
}

/** The chart tooltip's name of run `x`: "#48107719 · 2026-09-14". */
export function runAt(runs: readonly RunUsage[], x: number): string {
  const r = runs[Math.round(x) - 1];
  if (!r) return "";
  const day = formatDate(r.submitted_at, { fallback: "" });
  return day ? `#${r.jobid} · ${day}` : `#${r.jobid}`;
}

/* ── the run table ────────────────────────────────────────────────────────── */

/** used ÷ requested, or null when either is unknown (or nothing was requested). */
export function usedFraction(used: number | null | undefined, requested: number | null | undefined): number | null {
  return fin(used) && fin(requested) && requested > 0 ? used / requested : null;
}

/** The bar tone of a requested-vs-used cell: bad for a run that ran out of
 *  memory (or used more than it asked), warn for one that hit its limit. */
export function barTone(kind: "cpu" | "memory" | "time", fraction: number | null, state: string | null | undefined):
  "bad" | "warn" | undefined {
  if (kind === "memory" && (isOom(state) || (fraction ?? 0) > 1)) return "bad";
  if (kind === "time" && isTimeout(state)) return "warn";
  return undefined;
}

/** "12,000 images" / "70,000 steps", or "" when the work is unknown. */
export function workText(r: Pick<RunUsage, "units" | "units_label">): string {
  return fin(r.units) ? `${formatCount(r.units)}${r.units_label ? ` ${r.units_label}` : ""}` : "";
}
