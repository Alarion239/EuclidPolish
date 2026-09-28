/* A 1-D image HDU (a spectrum, a profile, a vector of values) as a line plot
   with its statistics. */
import { useResource } from "../../api/query";
import { formatCount, formatNumber } from "../../format";
import { Callout, DefList, Skeleton } from "../../ui";
import { imageStatsUrl, type HduSummary, type ImageStats } from "./api";
import { SeriesPlot } from "./charts";
import { basename } from "./model";

export function VectorPanel({ fits, hdu, unit }: { fits: string; hdu: HduSummary; unit: string }) {
  const res = useResource<ImageStats>(imageStatsUrl(fits, hdu.index), [], { ttl: 60_000 });
  const s = res.data;
  if (res.loading) return <Skeleton height={300} />;
  if (res.error || !s?.series) return <Callout tone="bad" title="Could not read this HDU">{res.error?.message ?? "No values."}</Callout>;
  return (
    <div className="insp-vector">
      <SeriesPlot x={s.series.x} y={s.series.y} label={hdu.name} unit={unit}
        exportName={`${basename(fits)}_hdu${hdu.index}`} />
      <DefList dense items={[
        ["points", `${formatCount(s.n_points)}${s.step && s.step > 1 ? ` (every ${s.step}th shown)` : ""}`],
        ["min / max", `${formatNumber(s.min)} / ${formatNumber(s.max)}`],
        ["mean ± σ", `${formatNumber(s.mean)} ± ${formatNumber(s.std)}`],
        ["median", formatNumber(s.median)],
        s.n_nan ? ["NaN", formatCount(s.n_nan)] : null,
      ]} />
    </div>
  );
}
