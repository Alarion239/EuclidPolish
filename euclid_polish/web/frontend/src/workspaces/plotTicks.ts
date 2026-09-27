/* Ticks for the workspaces' charts.
 *
 * charts/Plot draws numbers (and the matching grid lines) only for the ticks
 * it is given — generated ticks appear only once the view is zoomed — so a
 * chart whose caller passes no `xTicks` / `yTicks` had bare axes (Ensemble ›
 * Curves, Combiners, the r(k) / T(k) y axis). Every call site that does not
 * pick its own ticks passes `autoTicks(domain, scale, format)`: the same
 * generator the zoomed view uses (charts/plotModel `viewTicks`) — about five
 * nice linear ticks, decades on a log axis. */
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
