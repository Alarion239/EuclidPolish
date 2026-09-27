/* Inspector kind `experiment:<id>` — one real-data experiment: status, tiles
 * (→ realtile inspectors), models, per-band metrics (pooled or per tile) and
 * the band chart; opens the full comparison in Sky › Experiments. Polls
 * while the experiment runs (the record is written progressively). */
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { Button, Callout, Section, Skeleton } from "../../../ui";
import { URLS, type ExperimentRecord } from "./api";
import { ExperimentDetail } from "./ExperimentDetail";
import type { MetricKey } from "./model";
import "./results.css";

export default function ExperimentInspector({ id }: { id: string }) {
  const navigate = useNavigate();
  const [scope, setScope] = useState("pooled");
  const [metric, setMetric] = useState<MetricKey>("hole_pct");
  const [running, setRunning] = useState(false);
  const res = useResource<ExperimentRecord>(URLS.experiment(id), [], { ttl: 5_000, poll: running ? 3_000 : undefined });
  const record = res.data;
  const nowRunning = record?.status === "running";
  useEffect(() => { setRunning(nowRunning); }, [nowRunning]);
  if (res.loading) return <Skeleton lines={6} />;
  if (!record) {
    return (
      <Callout tone="bad" title={res.error?.status === 404 ? "Unknown experiment" : "Could not load the experiment"}
        action={<Button size="sm" onClick={res.reload}>Retry</Button>}>
        {res.error?.message ?? "No data."}
      </Callout>
    );
  }
  return (
    <div className="res-card">
      <div className="res-card__actions">
        <Button size="sm" variant="primary" onClick={() => navigate(`/sky/experiments?exp=${encodeURIComponent(record.id)}`)}>
          Open in Experiments
        </Button>
      </div>
      <ExperimentDetail record={record} scope={scope} onScope={setScope} metric={metric} onMetric={setMetric} viewer={false} />
      <Section title="Tiles" sub={String(record.tiles?.length ?? 0)} collapsible defaultOpen={false}>
        <div className="res-chips">
          {(record.tiles ?? []).map((t) => (
            <Button key={t} size="sm" variant="ghost" onClick={() => openInspector({ kind: "realtile", id: t })}>
              <span className="mono">{t}</span>
            </Button>
          ))}
        </div>
      </Section>
    </div>
  );
}
