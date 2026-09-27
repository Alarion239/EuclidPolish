/* Small shared pieces of the Sky results tabs: state badge, position with
 * copy, the per-band metrics table, gate core weights, file download. */
import { formatDec, formatDeg, formatRA } from "../../../format";
import { Badge, CopyButton, Tooltip } from "../../../ui";
import { BANDS, type BandMetrics, type GateCoreWeights } from "./api";
import { bandLabel, formatMetric, METRICS, STATE_TONE, type MetricKey } from "./model";

export function StateBadge({ state, prefix, title }: { state?: string | null; prefix?: string; title?: string | null }) {
  const s = state || "missing";
  return (
    <Badge size="sm" dot tone={STATE_TONE[s] ?? "neutral"} title={title ?? undefined}>
      {prefix ? `${prefix} ${s}` : s}
    </Badge>
  );
}

export function Position({ ra, dec }: { ra: number | null | undefined; dec: number | null | undefined }) {
  if (ra == null || dec == null || !Number.isFinite(ra) || !Number.isFinite(dec)) return <span className="muted">—</span>;
  return (
    <span className="res-pos">
      <span className="mono">{formatRA(ra)} {formatDec(dec)}</span>
      <span className="mono muted">{formatDeg(ra, 5)} {formatDeg(dec, 5, { signed: true })}</span>
      <CopyButton value={() => `${ra.toFixed(6)} ${dec.toFixed(6)}`} label="Copy RA Dec (degrees)" />
    </span>
  );
}

/** Start a browser download of a server file (FITS etc.). */
export function downloadUrl(url: string): void {
  const a = document.createElement("a");
  a.href = url;
  a.download = "";
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

const TABLE_METRICS: MetricKey[] = [
  "hole_pct", "hole_pct_100sigma", "pct_R_lt_0p8", "pct_R_lt_0p5", "median_R", "min_R", "flux_ratio",
  "n_peaks", "n_edge", "n_artifacts",
];

/** Metrics × bands for one model (rows = metrics, columns = bands). */
export function BandMetricsTable({ perBand, caption }: { perBand: Record<string, BandMetrics> | null | undefined; caption?: string }) {
  const bands = BANDS.filter((b) => perBand?.[b]);
  if (!perBand || !bands.length) return <p className="muted res-note">No per-band metrics.</p>;
  return (
    <div className="res-mtable-wrap">
      <table className="res-mtable">
        {caption && <caption className="sr-only">{caption}</caption>}
        <thead>
          <tr><th scope="col">metric</th>{bands.map((b) => <th key={b} scope="col">{bandLabel(b)}</th>)}</tr>
        </thead>
        <tbody>
          {TABLE_METRICS.map((key) => {
            const def = METRICS.find((m) => m.key === key)!;
            return (
              <tr key={key}>
                <th scope="row">
                  <Tooltip content={def.hint}><span tabIndex={0}>{def.label}</span></Tooltip>
                </th>
                {bands.map((b) => <td key={b} className="mono">{formatMetric(key, perBand[b]?.[key])}</td>)}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** The gate's top members over the brightest 1 % of pixels, per band. */
export function CoreWeights({ weights }: { weights: GateCoreWeights | null | undefined }) {
  const bands = BANDS.filter((b) => weights?.[b]?.length);
  if (!weights || !bands.length) return null;
  return (
    <div className="res-weights" aria-label="Gate core weights">
      {bands.map((b) => (
        <div key={b} className="res-weights__band">
          <span className="res-weights__name">{bandLabel(b)}</span>
          {weights[b].map(([label, w]) => (
            <span key={label} className="res-weights__item mono" title={`${label}: mean weight ${w}`}>
              <span className="res-weights__bar" style={{ inlineSize: `${Math.max(4, Math.min(100, w * 100))}%` }} />
              <span className="res-weights__text">{label.replace("·psnr", "")} {(w * 100).toFixed(0)}%</span>
            </span>
          ))}
        </div>
      ))}
    </div>
  );
}
