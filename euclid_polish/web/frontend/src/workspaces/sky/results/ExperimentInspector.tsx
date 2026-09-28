/* Inspector kind `experiment:<id>` — one model comparison on real tiles, as
 * Sky › Compare shows it, without the viewer: its one sentence, the metric
 * select with the pivot table (models × bands), its tiles (→ the tile
 * cards) and "Open in Compare" for the viewer and the history. Polls while
 * the comparison runs (the record is written progressively). */
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { Badge, Button, Callout, Section, Select, Skeleton } from "../../../ui";
import { comparisonLabel } from "../compare/model";
import { CompareSentence, PivotTable, useSpecColors } from "../compare/parts";
import "../compare/compare.css";
import { URLS, type ExperimentRecord } from "./api";
import { StateBadge } from "./common";
import { CHART_METRICS, METRIC_BY_KEY, type MetricKey } from "./model";
import "./results.css";

function Body({ record }: { record: ExperimentRecord }) {
  const navigate = useNavigate();
  const [scope, setScope] = useState("pooled");
  const [metric, setMetric] = useState<MetricKey>("hole_pct");
  const colors = useSpecColors(record);
  const tiles = record.tiles ?? [];
  return (
    <div className="res-card">
      <div className="res-card__actions">
        <strong className="res-exp__title" title={record.id}>{comparisonLabel(record)}</strong>
        {record.status !== "done" && <StateBadge state={record.status} />}
        {!!Object.keys(record.errors ?? {}).length && <Badge size="sm" tone="bad">{Object.keys(record.errors ?? {}).length} errors</Badge>}
      </div>
      <div className="res-card__actions">
        <Button size="sm" variant="primary" onClick={() => navigate(`/sky/compare?exp=${encodeURIComponent(record.id)}`)}>
          Open in Compare
        </Button>
        <Select size="sm" value={scope} onChange={setScope} aria-label="Scope"
          options={[{ value: "pooled", label: tiles.length === 1 ? "Pooled (1 tile)" : `Pooled (${tiles.length} tiles)` },
            ...tiles.map((t) => ({ value: t, label: t }))]} />
      </div>
      <CompareSentence record={record} scope={scope} />
      <Select size="sm" value={metric} onChange={(v) => setMetric(v as MetricKey)} aria-label="Metric"
        options={CHART_METRICS.map((k) => ({ value: k, label: METRIC_BY_KEY[k].label }))} />
      <PivotTable record={record} scope={scope} metric={metric} colors={colors} />
      <Section title="Tiles" sub={String(tiles.length)} collapsible defaultOpen={false}>
        <div className="res-chips">
          {tiles.map((t) => (
            <Button key={t} size="sm" variant="ghost" onClick={() => openInspector({ kind: "tile", id: t })}>
              <span className="mono">{t}</span>
            </Button>
          ))}
        </div>
      </Section>
    </div>
  );
}

export default function ExperimentInspector({ id }: { id: string }) {
  const [running, setRunning] = useState(false);
  const res = useResource<ExperimentRecord>(URLS.experiment(id), [], { ttl: 5_000, poll: running ? 3_000 : undefined });
  const record = res.data;
  const nowRunning = record?.status === "running";
  useEffect(() => { setRunning(nowRunning); }, [nowRunning]);
  if (res.loading) return <Skeleton lines={6} />;
  if (!record) {
    return (
      <Callout tone="bad" title={res.error?.status === 404 ? "Unknown comparison" : "Could not load the comparison"}
        action={<Button size="sm" onClick={res.reload}>Retry</Button>}>
        {res.error?.message ?? "No data."}
      </Callout>
    );
  }
  return <Body record={record} />;
}
