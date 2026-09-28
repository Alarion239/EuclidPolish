/* Sky › Compare, the pieces of one comparison (tabs/Compare.tsx): the Δm
 * footer under the viewer, the band strip (one metric per band, a dot per
 * model), the pivot table (models × VIS / Y / J / H for that metric, the best
 * value per band marked), the history's columns and the one sentence. A
 * model has ONE colour here: its dot in the strip and its swatch in the
 * table (the order of recordSpecs). */
import { useMemo } from "react";
import Plot from "../../../charts/Plot";
import { categorical } from "../../../colors";
import { formatCount, formatNumber, formatRelative } from "../../../format";
import { Badge, DataTable, Num, SummaryLine, Tooltip, type DataColumn } from "../../../ui";
import { BANDS, type ExperimentRecord, type ExperimentSummary } from "../results/api";
import { StateBadge } from "../results/common";
import {
  bandLabel, bandSeries, formatMetric, METRIC_BY_KEY, recordSpecs, seriesDomain, specShort, specWords, type MetricKey,
} from "../results/model";
import { compareSentence, deltaMags, headlineText, pivotBest, pivotRows, type PivotRow } from "./model";

type Band = (typeof BANDS)[number];

/** Each spec's colour, in the record's model order. */
export function useSpecColors(record: ExperimentRecord): Record<string, string> {
  return useMemo(() => Object.fromEntries(recordSpecs(record).map((s, i) => [s, categorical(i)])), [record]);
}

/** The page's one sentence (production's worst-band holes against the mean). */
export function CompareSentence({ record, scope }: { record: ExperimentRecord; scope: string }) {
  const pieces = compareSentence(record, scope);
  if (!pieces) return null;
  return (
    <SummaryLine className="cmp-sentence">
      {pieces.map((p, i) => (p.num ? <Num key={i} tone={p.warn ? "warn" : undefined}>{p.text}</Num> : <span key={i}>{p.text}</span>))}
    </SummaryLine>
  );
}

/** Under the viewer: each model's total VIS flux against the LR on the tile
 *  the viewer shows, as Δm (warned beyond 0.1 mag). */
export function DeltaFooter({ record, tile }: { record: ExperimentRecord; tile: string }) {
  const rows = deltaMags(record, tile, "VIS");
  if (!rows.length) return null;
  return (
    <p className="cmp-delta" aria-label="Flux against the LR">
      <span className="cmp-delta__k">VIS, Δm vs LR on {tile}:</span>{" "}
      {rows.map((d, i) => (
        <span key={d.spec} className="cmp-delta__item" data-warn={d.warn || undefined}>
          {i > 0 && <span className="cmp-delta__sep" aria-hidden="true"> · </span>}
          {specWords(d.spec)} <span className="mono">{d.dm == null ? "—" : formatNumber(d.dm, { digits: 2, signed: true })}</span>
        </span>
      ))}
    </p>
  );
}

const BAND_TICKS = BANDS.map((b, i) => ({ v: i, label: bandLabel(b) }));

/** One metric per band, a dot per model (a small offset per model keeps the
 *  dots apart); the legend toggles models. */
export function BandStrip({ record, scope, metric, colors }: {
  record: ExperimentRecord; scope: string; metric: MetricKey; colors: Record<string, string>;
}) {
  const series = useMemo(() => bandSeries(record, scope, metric), [record, scope, metric]);
  const def = METRIC_BY_KEY[metric];
  if (!series.some((s) => s.y.some((v) => v != null))) return null;
  const domain = seriesDomain(series, metric);
  const guides = def?.better === "one" ? [{ axis: "y" as const, v: 1, dash: [3, 3] }] : [];
  return (
    <Plot xDomain={[-0.5, BANDS.length - 0.5]} yDomain={domain} xTicks={BAND_TICKS} height={220} zoomAxes="y"
      yLabel={def?.label} guides={guides}
      series={series.map((s) => ({
        x: s.x, y: s.y, color: colors[s.spec] ?? categorical(0), mode: "scatter" as const, width: 3,
        name: specWords(s.spec), key: s.spec,
      }))}
      legend="auto" legendToggle exportName={`comparison-${record.id}-${metric}`}
      yFormat={(v) => formatMetric(metric, v)} aria-label={`${def?.label ?? metric} per band, a dot per model`} />
  );
}

/** Models × bands for one metric: the reference (production) first; the
 *  best value per band is bold. The unit is in the headers. */
export function PivotTable({ record, scope, metric, colors }: {
  record: ExperimentRecord; scope: string; metric: MetricKey; colors: Record<string, string>;
}) {
  const rows = useMemo(() => pivotRows(record, scope, metric), [record, scope, metric]);
  const best = useMemo(() => pivotBest(rows, metric), [rows, metric]);
  const def = METRIC_BY_KEY[metric];
  const columns = useMemo<DataColumn<PivotRow>[]>(() => [
    {
      id: "spec", header: "Model", accessor: (r) => specWords(r.spec), width: 200, sortable: false,
      cell: (r) => (
        <Tooltip content={r.label}>
          <span className="cmp-model" tabIndex={0}>
            <span className="cmp-swatch" style={{ background: colors[r.spec] }} aria-hidden="true" />
            {specWords(r.spec)}
          </span>
        </Tooltip>
      ),
    },
    ...BANDS.map((b: Band): DataColumn<PivotRow> => ({
      id: b, header: def?.unit ? `${bandLabel(b)} (${def.unit})` : bandLabel(b), headerText: `${bandLabel(b)} ${def?.label ?? ""}`.trim(),
      numeric: true, accessor: (r) => r[b], width: 84,
      cell: (r) => (r[b] == null ? <span className="muted">—</span>
        : best[b] === r.spec ? <strong className="cmp-best">{formatMetric(metric, r[b])}</strong> : formatMetric(metric, r[b])),
    })),
  ], [best, colors, def, metric]);
  if (!rows.length) return null;
  return (
    <DataTable rows={rows} columns={columns} rowKey={(r) => r.spec} aria-label={`${def?.label ?? metric} per model and band`}
      dense height="auto" hideToolbar />
  );
}

/* ── history ───────────────────────────────────────────────────────────── */

/* Narrow tables drop Models first, then Status (a badge only on a problem);
 * the headline result stays with the label. */

export function historyColumns(): DataColumn<ExperimentSummary>[] {
  return [
    { id: "created", header: "Created", accessor: (e) => e.created ?? "", width: 104,
      cell: (e) => (e.created ? <span title={e.created}>{formatRelative(e.created)}</span> : "—") },
    { id: "label", header: "Label", accessor: (e) => e.label || e.id, width: 200,
      cell: (e) => <span className="res-ellipsis" title={e.id}>{e.label || <code className="mono">{e.id}</code>}</span> },
    { id: "headline", header: "Worst-band holes", headerText: "Worst-band holes (production · mean)", width: 240,
      accessor: (e) => headlineText(e), cell: (e) => headlineText(e) || <span className="muted">not scored</span>, priority: 1 },
    { id: "tiles", header: "Tiles", numeric: true, accessor: (e) => e.tiles?.length ?? 0, width: 56,
      filterText: (e) => (e.tiles ?? []).join(" "), cell: (e) => formatCount(e.tiles?.length ?? 0) },
    { id: "models", header: "Models", accessor: (e) => (e.models ?? []).join(" "), width: 220, priority: 3,
      cell: (e) => (
        <span className="res-chips res-chips--tight">
          {(e.models ?? []).slice(0, 4).map((m) => <Badge key={m} size="sm" tone={e.skipped?.[m] ? "warn" : undefined}>{specShort(m)}</Badge>)}
          {(e.models?.length ?? 0) > 4 && <span className="muted">+{(e.models?.length ?? 0) - 4}</span>}
        </span>
      ) },
    // A badge only on a problem (running, failed, errors); a finished run is quiet.
    { id: "status", header: "Status", accessor: (e) => e.status ?? "", width: 96, priority: 2,
      cell: (e) => {
        const errors = Object.keys(e.errors ?? {}).length;
        if (e.status && e.status !== "done") return <StateBadge state={e.status} />;
        return errors ? <Badge size="sm" tone="bad">{errors} error{errors === 1 ? "" : "s"}</Badge> : null;
      } },
    { id: "id", header: "Id", cell: (e) => <code className="mono">{e.id}</code>, hidden: true },
  ];
}
