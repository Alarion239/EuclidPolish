/* Time-travel sandboxes: open (restart) / stop / remove (confirmed). The
 * `source` is an object; its text is `source_label`. */
import { useMemo, useState } from "react";
import { apiPost } from "../../api/client";
import { invalidate } from "../../api/query";
import { formatRelative } from "../../format";
import { Badge, Button, DataTable, confirm, toast, type DataColumn } from "../../ui";
import { TRACKING_STATE_URL, type Sandbox } from "./api";

export function SandboxTable({ sandboxes }: { sandboxes: Sandbox[] }) {
  const [busy, setBusy] = useState<string | null>(null);
  async function act(sb: Sandbox, what: "open" | "stop" | "remove") {
    if (what === "remove" && !(await confirm({ title: `Remove sandbox ${sb.short}?`,
      message: "Stops its server and deletes its worktree and sandbox data.", tone: "danger", confirmLabel: "Remove" }))) return;
    if (what === "stop" && !(await confirm({ title: `Stop sandbox ${sb.short}?`, message: "Its second console shuts down; the worktree stays.",
      confirmLabel: "Stop" }))) return;
    setBusy(`${sb.short}:${what}`);
    try {
      const r = await apiPost<{ ok?: boolean; error?: string; url?: string }>(`/api/tracking/timetravel/${what}`, { short: sb.short });
      if (r.ok === false || r.error) toast.error(r.error || `${what} failed`);
      else if (what === "open" && r.url) {
        toast.success(`Sandbox ${sb.short} is up`, { action: { label: "Open", onClick: () => window.open(r.url, "_blank", "noopener") } });
      } else toast.success(`${what === "stop" ? "Stopped" : what === "remove" ? "Removed" : "Started"} ${sb.short}`);
    } catch (e) { toast.error(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(null); void invalidate(TRACKING_STATE_URL); }
  }
  const columns = useMemo<DataColumn<Sandbox>[]>(() => [
    { id: "short", header: "Sandbox", width: 96, cell: (s) => <code className="mono">{s.short}</code> },
    { id: "source_label", header: "Source", cell: (s) => <span>{s.source_label || "—"}</span> },
    { id: "running", header: "State", width: 96, accessor: (s) => (s.running ? "running" : "stopped"),
      cell: (s) => <Badge size="sm" tone={s.running ? "good" : undefined} dot={s.running}>{s.running ? "running" : "stopped"}</Badge> },
    { id: "where", header: "Where", width: 90, accessor: (s) => (s.remote?.ok ? "local + FASRC" : "local") },
    { id: "created_at", header: "Created", width: 100, cell: (s) => <span className="nb-dim nb-small">{formatRelative(s.created_at)}</span> },
    { id: "actions", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 210,
      cell: (s) => (
        <span className="nb-row">
          {s.running && s.url
            ? <Button asChild size="sm" variant="primary"><a href={s.url} target="_blank" rel="noreferrer noopener">Open</a></Button>
            : <Button size="sm" loading={busy === `${s.short}:open`} onClick={() => void act(s, "open")}>Start</Button>}
          {s.running && <Button size="sm" variant="ghost" loading={busy === `${s.short}:stop`} onClick={() => void act(s, "stop")}>Stop</Button>}
          <Button size="sm" variant="ghost" loading={busy === `${s.short}:remove`} onClick={() => void act(s, "remove")}>Remove</Button>
        </span>
      ) },
  ], [busy]);
  return (
    <DataTable rows={sandboxes} columns={columns} rowKey={(s) => s.short} aria-label="Time-travel sandboxes"
      height="auto" dense hideToolbar={sandboxes.length < 8}
      empty="No sandbox yet: time-travel a backup or a campaign (⏱ in Backups) to create one." />
  );
}
