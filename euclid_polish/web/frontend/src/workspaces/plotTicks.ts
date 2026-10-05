/* Ticks for the workspaces' charts.
 *
 * charts/Plot once drew numbers (and the matching grid lines) only for the
 * ticks it was given — generated ticks appeared only once the view was
 * zoomed — so a chart whose caller passed no `xTicks` / `yTicks` had bare
 * axes (Ensemble › Curves, Combiners, the r(k) / T(k) y axis). Plot now
 * generates ticks itself for an axis whose ticks are undefined (and thins
 * them to the plot's size); `autoTicks(domain, scale, format)` is the
 * generator underneath (charts/plotModel `viewTicks`, unthinned) for a
 * caller that wants the ticks up front — about five nice linear ticks,
 * decades on a log axis. */
import { viewTicks } from "../charts/plotModel";
import type { AxisScale } from "../charts/types";
import type { Tick } from "../ticks";

export function autoTicks(domain: readonly [number, number], scale: AxisScale = "linear", format?: (v: number) => string): Tick[] {
  const [a, b] = domain;
  if (!Number.isFinite(a) || !Number.isFinite(b) || a === b) return [];
  return viewTicks(undefined, [a, b], scale === "log" && Math.min(a, b) > 0 ? "log" : "linear", format);
}

/** Fixed fractions for a correlation / transfer axis starting at 0 (r(k)). */
export function unitTicks(top: number): Tick[] {
  const out: Tick[] = [];
  for (let v = 0; v <= top + 1e-9; v += 0.25) out.push({ v: Math.round(v * 100) / 100, label: String(Math.round(v * 100) / 100) });
  return out;
}
