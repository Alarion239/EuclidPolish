/* The flux SR/LR strip of Sky › Targets: one row per chosen target set, a
 * dot per target (its total VIS flux SR over LR), the set's median as a
 * diamond and the flux-kept line at 1. A click opens the nearest target's
 * card. The counts are the chips' and the sentences' (once per screen). */
import { useMemo } from "react";
import Plot from "../../../charts/Plot";
import { C, categorical } from "../../../colors";
import { formatNumber } from "../../../format";
import { useResolvedTheme } from "../../../state/prefs";
import { Caption } from "../../../ui";
import { TARGET_SETS, fluxStrip, stripCoverage, type TargetRow, type TargetSetId } from "./model";

/** A set keeps its colour whichever sets are shown (its place in the chips). */
const setColor = (set: TargetSetId) => categorical(Math.max(0, TARGET_SETS.findIndex((s) => s.id === set)));

const ROW_PX = 46;

export function FluxStrip({ sets, rows, onPick }: {
  sets: readonly TargetSetId[]; rows: readonly TargetRow[]; onPick: (row: { ref: string | null; key: string }) => void;
}) {
  const theme = useResolvedTheme();
  const strip = useMemo(() => fluxStrip(sets, rows), [sets, rows]);
  const shown = strip.rows;
  const coverage = stripCoverage(shown);
  const yTicks = useMemo(() => shown.map((r, i) => ({ v: i, label: r.label })), [shown]);
  const series = useMemo(() => {
    void theme;
    const dots = shown.map((r) => {
      const pts = strip.points.filter((p) => p.set === r.set);
      return {
        x: pts.map((p) => p.x), y: pts.map((p) => p.y), color: setColor(r.set), mode: "scatter" as const,
        width: 1.6, name: r.label, key: r.set, alpha: 0.7,
      };
    });
    const medians = shown.filter((r) => r.median != null);
    return [...dots, {
      x: medians.map((r) => r.median as number), y: medians.map((r) => shown.indexOf(r)),
      color: C.cross, mode: "scatter" as const, marker: "diamond" as const, width: 4, name: "Median", key: "median",
    }];
  }, [shown, strip.points, theme]);
  if (!strip.points.length) return null;
  const onPlotClick = ({ x, y }: { x: number; y: number }) => {
    const [x0, x1] = strip.domain;
    const sx = (x1 - x0) || 1;
    let best: (typeof strip.points)[number] | null = null;
    let bestD = Infinity;
    for (const p of strip.points) {
      const d = ((p.x - x) / sx) ** 2 + ((p.y - y) / (shown.length || 1)) ** 2;
      if (d < bestD) { bestD = d; best = p; }
    }
    if (best) onPick(best);
  };
  return (
    <figure className="tg-strip" aria-label="Flux SR/LR per target">
      <Plot xDomain={strip.domain} yDomain={[-0.6, shown.length - 0.4]} yTicks={yTicks}
        height={52 + ROW_PX * shown.length} series={series} zoomAxes="x"
        guides={[{ axis: "x", v: 1, dash: [4, 3], label: "flux kept", labelSide: "before" }]}
        xLabel="Total flux SR / LR (VIS)" xFormat={(v) => formatNumber(v, { digits: 2 })}
        yFormat={(v) => shown[Math.round(v)]?.label ?? ""} onPlotClick={onPlotClick}
        exportName="targets-flux" aria-label="Total flux SR over LR per target, by target set" />
      <Caption>
        One dot per target with a flux, its set's median as a diamond; left of 1 the SR lost flux.
        Click a dot to open its card.{coverage ? ` ${coverage}` : ""}
      </Caption>
    </figure>
  );
}
