/* Small charts of the Inspect workspace: a histogram (pixel values, table
   columns) and a 1-D series (vector HDUs). */
import { useMemo, useState } from "react";
import Plot, { type Guide } from "../../charts/Plot";
import { C } from "../../colors";
import { formatNumber } from "../../format";
import { formatTick, linearTicks, logTicks, paddedDomain } from "../../ticks";
import { Segmented } from "../../ui";
import type { Histogram } from "./api";
import { axisLabel, histogramSeries } from "./model";

export function HistogramPlot({ hist, label, unit, exportName, markers, height = 190 }: {
  hist: Histogram; label: string; unit?: string; exportName: string;
  markers?: { v: number | null | undefined; label: string }[]; height?: number;
}) {
  const [scale, setScale] = useState<"log" | "linear">("log");
  const { x, y } = useMemo(() => histogramSeries(hist), [hist]);
  const lo = hist.edges[0], hi = hist.edges[hist.edges.length - 1];
  const peak = Math.max(1, ...y);
  const yDomain: [number, number] = scale === "log" ? [0.8, peak * 1.6] : [0, peak * 1.08];
  const guides: Guide[] = (markers ?? [])
    .filter((m): m is { v: number; label: string } => typeof m.v === "number" && Number.isFinite(m.v) && m.v >= lo && m.v <= hi)
    .map((m) => ({ axis: "x", v: m.v, label: m.label, dash: [3, 3], color: C.cross, alpha: 0.8 }));
  const xLabel = unit ? `${label} [${unit}]` : label;
  // Big values as "10k" and only 4 ticks: the card is often half of a ~720 px pane.
  const big = Math.max(Math.abs(lo), Math.abs(hi)) >= 1e3;
  const xTicks = linearTicks([lo, hi], { count: 4, format: (v, step) => (big ? axisLabel(v) : formatTick(v, step)) });
  return (
    <div className="insp-hist">
      <div className="insp-hist__bar">
        <span className="insp-dim">
          {hist.below + hist.above > 0 ? `${formatNumber(hist.below)} below · ${formatNumber(hist.above)} above the range` : "full range"}
        </span>
        <Segmented size="sm" aria-label="Count axis" value={scale} onChange={setScale}
          options={[{ value: "log", label: "log" }, { value: "linear", label: "linear" }]} />
      </div>
      <Plot
        xDomain={[lo, hi]} yDomain={yDomain} yScale={scale}
        xTicks={xTicks}
        yTicks={scale === "log" ? logTicks(yDomain, { maxTicks: 5 }) : linearTicks(yDomain, { count: 4 })}
        xLabel={xLabel} yLabel="count" height={height}
        series={[{ x, y, color: C.mean, mode: "histogram", name: "count" }]}
        guides={guides} exportName={exportName} aria-label={`Histogram of ${label}`}
        xFormat={axisLabel} />
    </div>
  );
}

export function SeriesPlot({ x, y, label, unit, exportName }: {
  x: number[]; y: (number | null)[]; label: string; unit?: string; exportName: string;
}) {
  const xDomain = paddedDomain(x, { pad: 0 });
  const yDomain = paddedDomain(y, { pad: 0.06 });
  return (
    <Plot xDomain={xDomain} yDomain={yDomain}
      xTicks={linearTicks(xDomain, { count: 6 })} yTicks={linearTicks(yDomain, { count: 5 })}
      xLabel="pixel" yLabel={unit ? `${label} [${unit}]` : label} height={300}
      series={[{ x, y, color: C.mean, name: label, width: 1.5 }]}
      exportName={exportName} aria-label={`${label} values`} />
  );
}
