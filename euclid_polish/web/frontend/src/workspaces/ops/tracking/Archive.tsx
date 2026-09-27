/* Archived (saved) campaigns: a table of snapshots and, for one campaign,
 * its detail — metadata, backups with per-model time travel, the notebook
 * and its job records. The detail is also the `campaign:<dir>` inspector. */
import { useMemo, useState } from "react";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { useFasrcStatus } from "../../../app/status";
import { formatDateTime, formatRelative } from "../../../format";
import {
  Badge, Button, Callout, DataTable, DefList, IconButton, Section, Skeleton, type DataColumn,
} from "../../../ui";
import { campaignUrl, type Archived, type CampaignResp } from "../api";
import { Markdown } from "../markdown";
import { BackupTables } from "./Backups";
import { TimeTravelDialog, commitText, type TimeTravelTarget } from "./TimeTravel";
import { TrackedJobs } from "./TrackedJobs";

export function ArchiveTable({ archived, onTimeTravel }: {
  archived: Archived[]; onTimeTravel: (t: TimeTravelTarget) => void;
}) {
  const columns = useMemo<DataColumn<Archived>[]>(() => [
    { id: "title", header: "Campaign", cell: (a) => <span className="ops-cell2"><strong>{a.title}</strong>
      {a.description && <span className="ops-dim ops-small">{a.description}</span>}</span> },
    { id: "saved_at", header: "Saved", width: 110,
      cell: (a) => <span className="ops-dim ops-small" title={formatDateTime(a.saved_at)}>{formatRelative(a.saved_at)}</span> },
    { id: "commit", header: "Commit", width: 100, accessor: (a) => commitText(a.saved_commit ?? a.created_commit),
      cell: (a) => <code className="mono">{commitText(a.saved_commit ?? a.created_commit)}</code> },
    { id: "models", header: "Models", numeric: true, width: 80, accessor: (a) => a.models?.length ?? 0 },
    { id: "actions", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 80,
      cell: (a) => (
        <span className="ops-row-actions">
          <IconButton size="sm" icon="reset" label={`Time-travel to ${a.title}`}
            onClick={() => onTimeTravel({ campaign: a._dir, title: a.title, commit: a.saved_commit ?? a.created_commit })} />
          <IconButton size="sm" icon="panelRight" label={`Open ${a.title}`} onClick={() => openInspector({ kind: "campaign", id: a._dir })} />
        </span>
      ) },
  ], [onTimeTravel]);
  return (
    <DataTable rows={archived} columns={columns} rowKey={(a) => a._dir} aria-label="Archived campaigns"
      inspect={(a) => ({ kind: "campaign", id: a._dir })} height={560} exportName="campaigns"
      empty="No campaign has been saved yet." />
  );
}

/** One campaign in full (the inspector body). */
export function CampaignDetail({ id }: { id: string }) {
  const res = useResource<CampaignResp>(campaignUrl(id), [id], { ttl: 60_000 });
  const fasrc = useFasrcStatus();
  const [tt, setTt] = useState<TimeTravelTarget | null>(null);
  const d = res.data;
  if (res.loading && !d) return <Skeleton lines={6} />;
  if (res.error && !d) return <Callout tone="bad" title="Campaign not found">{res.error.message}</Callout>;
  if (!d) return null;
  const m = d.metadata;
  return (
    <div className="ops-stack">
      <div className="ops-campaign">
        <span className="ops-campaign__title">{m.title}</span>
        <Badge size="sm" tone={d.active ? "info" : undefined}>{d.active ? "active" : "archived"}</Badge>
      </div>
      {m.description && <p className="ops-note">{m.description}</p>}
      <DefList dense items={[
        ["created", <span>{formatDateTime(m.created_at)} · <code className="mono">{commitText(m.created_commit)}</code></span>],
        m.saved_at ? ["saved", <span>{formatDateTime(m.saved_at)} · <code className="mono">{commitText(m.saved_commit)}</code></span>] : null,
        ["dir", <code className="mono">{d.dir}</code>],
        ["jobs", String(d.jobs_count)],
      ]} />
      <Button size="sm" icon="reset" onClick={() => setTt({ campaign: d.dir === "current" ? "current" : id, title: m.title,
        commit: m.saved_commit ?? m.created_commit })}>Time-travel to this campaign</Button>
      <Section title="Backups" collapsible defaultOpen>
        <BackupTables backups={d.backups} campaign={d.active ? "current" : id} title={m.title} onTimeTravel={setTt} />
      </Section>
      <Section title="Notebook" collapsible defaultOpen={false}>
        <Markdown text={d.log_md || "_(empty)_"} />
      </Section>
      {d.jobs_count > 0 && (
        <Section title={`FASRC jobs (${d.jobs_count})`} collapsible defaultOpen={false}>
          <TrackedJobs campaign={d.active ? "current" : id} compact />
        </Section>
      )}
      <TimeTravelDialog target={tt} onClose={() => setTt(null)} fasrcConnected={!!fasrc.data?.ssh_connected} />
    </div>
  );
}
