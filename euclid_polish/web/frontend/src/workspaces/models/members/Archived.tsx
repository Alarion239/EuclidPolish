/* Models › Members, archived view: the tombstones (names are never reused)
   with their zip in the tracking campaign, and Restore (confirmed). A row
   opens the member inspector. */
import { useMemo } from "react";
import { useJob } from "../../../api/jobs";
import { formatBytes, formatRelative } from "../../../format";
import { Button, Caption, DataTable, JobProgress, Tooltip, confirm, type DataColumn } from "../../../ui";
import type { Tombstone } from "../api";
import { JOB, useOnJobEnd } from "../jobs";
import { memberNumber } from "../model";

function columns(onRestore: (name: string) => void, busy: boolean): DataColumn<Tombstone>[] {
  return [
    { id: "name", header: "Member", sortFn: (a, b) => Number(memberNumber(a.name)) - Number(memberNumber(b.name)),
      cell: (t) => <span className="mdl-mono">#{memberNumber(t.name)}</span>, width: 90 },
    { id: "archived_at", header: "Archived", cell: (t) => (t.archived_at
      ? <Tooltip content={t.archived_at}><span tabIndex={0}>{formatRelative(t.archived_at)}</span></Tooltip> : "—") },
    { id: "zip", header: "Zip", accessor: (t) => t.zip ?? null,
      cell: (t) => (t.zip_found ? <span className="mdl-mono">{t.zip?.replace(/^models\//, "")} <span className="mdl-faint">· {t.campaign}</span></span>
        : <span className="mdl-warn">not found</span>) },
    { id: "size", header: "Size", numeric: true, accessor: (t) => t.size_bytes ?? null, cell: (t) => formatBytes(t.size_bytes) },
    { id: "commit", header: "Commit", hidden: true, cell: (t) => <code>{t.commit ?? "—"}</code> },
    { id: "actions", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 96,
      cell: (t) => (
        <Button size="sm" variant="ghost" disabled={!t.zip_found || busy}
          onClick={async () => {
            if (await confirm({ title: `Restore ${t.name}?`, message: "Unzips the archive back into the ensemble and makes the member active again. The evaluation and the production gate then read stale until re-run.", confirmLabel: "Restore" })) onRestore(t.name);
          }}>Restore</Button>
      ) },
  ];
}

export function Archived({ rows }: { rows: Tombstone[] }) {
  const restore = useJob(JOB.restore);
  useOnJobEnd(restore.job);
  const cols = useMemo(() => columns((name) => void restore.run("/ensemble/restore-member", { member: name }), restore.busy),
    [restore.busy]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div className="mdl-stack">
      <JobProgress job={restore.job} error={restore.error} />
      <DataTable rows={rows} columns={cols} rowKey={(t) => t.name} aria-label="Archived members"
        inspect={(t) => ({ kind: "member", id: t.name })} urlKey="a" height={480} dense
        exportName="ensemble-archived" empty="No archived members." />
      <Caption>Archived member names are never reused; each zip lives in its tracking campaign (Notebook › Backups).</Caption>
    </div>
  );
}
