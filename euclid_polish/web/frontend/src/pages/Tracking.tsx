/* Tracking — the experiment lab notebook: campaign manager, log.md editor,
   model/FITS/image backups, and time-travel sandboxes. Everything reads from
   one /api/tracking/state and mutates via small POSTs. */
import { useEffect, useState } from "react";
import { postForm } from "../api";
import { asArray } from "../data";
import { useResource } from "../hooks";
import {
  Badge, Button, Card, CardBody, CardHead, ConnBadge, DefList, Empty, Field,
  Input, LogTail, Page, PageHead, Spinner, Table, Textarea, type Column,
} from "../ui";

type Commit = { short?: string; hash?: string; branch?: string; dirty?: boolean } | string | null;
const commitStr = (c?: Commit): string => !c ? "—" : typeof c === "string" ? c : (c.short ?? c.hash ?? "—");

type Campaign = { title: string; description?: string; slug: string; created_at?: string; created_commit?: Commit };
type ModelRec = { name: string; size_bytes: number; created_at?: string; commit?: Commit };
type Archived = Campaign & { _dir: string; saved_at?: string; saved_commit?: Commit; models: ModelRec[] };
type BackupRec = { name: string; kind: string; comment?: string; size_bytes: number; created_at?: string; commit?: Commit };
type Sandbox = { short: string; created_at?: string; source?: string; running: boolean; url?: string; remote?: boolean };
type TrackState = {
  active: Campaign | null;
  archived: Archived[];
  backups: { models: BackupRec[]; fits: BackupRec[]; images: BackupRec[] };
  jobs: { jobid: string; logged_at?: string; step_id?: string; label?: string }[];
  log_md: string;
  tracking_dir?: string; remote_dir?: string;
  ssh_connected: boolean;
  sandboxes: Sandbox[];
};

const mb = (b: number) => `${(b / 1e6).toFixed(1)} MB`;

const backupCols = (): Column<BackupRec>[] => [
  { header: "name", cell: (r) => <code className="mono">{r.name}</code> },
  { header: "comment", cell: (r) => <span className="muted">{r.comment || "—"}</span> },
  { header: "size", cell: (r) => mb(r.size_bytes), align: "right" },
  { header: "commit", cell: (r) => r.commit ? <code className="mono">{commitStr(r.commit)}</code> : "—" },
  { header: "when", cell: (r) => <span className="muted">{r.created_at || ""}</span>, align: "right" },
];

export default function TrackingPage() {
  const { data, loading, reload } = useResource<TrackState>("/api/tracking/state");
  const [title, setTitle] = useState("");
  const [desc, setDesc] = useState("");
  const [log, setLog] = useState("");
  const [logMode, setLogMode] = useState<"append" | "replace">("append");
  const [busy, setBusy] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);

  useEffect(() => { if (data && logMode === "replace") setLog(data.log_md ?? ""); }, [data, logMode]);

  async function act(url: string, body?: Record<string, string>, msg?: string) {
    setBusy(url); setNote(null);
    try {
      const r = await postForm<{ ok?: boolean; error?: string; warning?: string; url?: string }>(url, body ?? {});
      if (r.url && url.includes("timetravel")) window.open(r.url, "_blank");
      setNote(r.error ? `✗ ${r.error}` : (r.warning ? `⚠ ${r.warning}` : (msg ?? "✓ done")));
      reload();
    } catch (e) {
      setNote(`✗ ${e instanceof Error ? e.message : String(e)}`);
    } finally { setBusy(null); }
  }

  const s = data;
  const archived = asArray<Archived>(s?.archived);
  const sandboxes = asArray<Sandbox>(s?.sandboxes);
  const backups = {
    models: asArray<BackupRec>(s?.backups?.models),
    fits: asArray<BackupRec>(s?.backups?.fits),
    images: asArray<BackupRec>(s?.backups?.images),
  };
  return (
    <Page>
      <PageHead eyebrow="ops · tracking" title="Experiment tracking"
        sub="Lab notebook + campaign snapshots: back up models/FITS/images with a commit stamp, and time-travel a backup's exact code."
        right={s && <ConnBadge ok={s.ssh_connected} labels={["holylabs connected", "holylabs offline"]} />} />

      {loading && <Card><CardBody><Empty><Spinner /> loading…</Empty></CardBody></Card>}
      {note && <div className="job-panel" style={{ marginBottom: "var(--s4)" }}><LogTail text={note} /></div>}

      {s && (
        <div className="grid" style={{ gridTemplateColumns: "1fr", gap: "var(--s4)" }}>
          <Card>
            <CardHead title="Active campaign"
              right={s.active
                ? <div className="row" style={{ gap: 8 }}>
                    <Button onClick={() => act("/api/tracking/save", {}, "✓ snapshot saved")} disabled={busy != null} variant="primary">Save snapshot</Button>
                    <Button onClick={() => act("/api/tracking/sync", {}, "✓ pushed")} disabled={busy != null}>Push to holylabs</Button>
                  </div>
                : undefined} />
            <CardBody>
              {s.active ? (
                <DefList items={[
                  ["title", s.active.title],
                  ["description", s.active.description || "—"],
                  ["created", <span className="muted">{s.active.created_at} · <code className="mono">{commitStr(s.active.created_commit)}</code></span>],
                  ["dir", <code className="mono">{s.tracking_dir}</code>],
                ]} />
              ) : <Empty>no active campaign — create one below</Empty>}
              <div className="row" style={{ marginTop: "var(--s4)", alignItems: "flex-end", gap: "var(--s3)" }}>
                <Field label="new campaign title"><Input value={title} onChange={setTitle} placeholder="e.g. starless-combiner" /></Field>
                <Field label="description"><Input value={desc} onChange={setDesc} placeholder="what are we testing?" /></Field>
                <Button onClick={() => { act("/api/tracking/new", { title, description: desc }, "✓ campaign created"); setTitle(""); setDesc(""); }}
                  disabled={!title.trim() || busy != null}>Create campaign</Button>
              </div>
            </CardBody>
          </Card>

          <Card>
            <CardHead title="Notebook · log.md"
              right={<div className="row" style={{ gap: 8 }}>
                <Button size="sm" variant={logMode === "append" ? "primary" : "ghost"} onClick={() => setLogMode("append")}>append</Button>
                <Button size="sm" variant={logMode === "replace" ? "primary" : "ghost"} onClick={() => setLogMode("replace")}>replace</Button>
              </div>} />
            <CardBody>
              <Textarea value={log} onChange={setLog} rows={6}
                placeholder={logMode === "append" ? "append a note to the campaign log…" : "full log.md text…"} />
              <div className="row" style={{ marginTop: "var(--s3)" }}>
                <Button variant="primary" disabled={!log.trim() || busy != null}
                  onClick={() => { act("/api/tracking/log", { text: log, mode: logMode }, "✓ logged"); if (logMode === "append") setLog(""); }}>
                  Save note
                </Button>
              </div>
              {s.log_md && (
                <details style={{ marginTop: "var(--s3)" }}>
                  <summary className="muted" style={{ cursor: "pointer" }}>rendered log</summary>
                  <LogTail text={s.log_md} style={{ maxHeight: 360 }} />
                </details>
              )}
            </CardBody>
          </Card>

          {(["models", "fits", "images"] as const).map((kind) => (
            <Card key={kind}>
              <CardHead title={`Backups · ${kind}`} sub={`${backups[kind].length} saved`} />
              <CardBody><Table columns={backupCols()} rows={backups[kind]} rowKey={(r) => r.name} empty={`no ${kind} backed up yet`} /></CardBody>
            </Card>
          ))}

          <Card>
            <CardHead title="Archived campaigns" sub={`${archived.length} snapshots`} />
            <CardBody>
              {archived.length === 0 ? <Empty>no snapshots yet</Empty> : (
                <div className="grid" style={{ gap: "var(--s3)" }}>
                  {archived.map((a) => (
                    <div key={a.slug} className="track-arch">
                      <div className="row" style={{ justifyContent: "space-between" }}>
                        <div><b>{a.title}</b> <span className="muted mono">{commitStr(a.saved_commit)}</span> <span className="muted">· {a.saved_at}</span></div>
                        <Button size="sm" onClick={() => act("/api/tracking/timetravel/restore", { campaign: a.slug, remote: "0" }, "✓ sandbox launching")}
                          disabled={busy != null}>⏱ time-travel</Button>
                      </div>
                      {asArray<ModelRec>(a.models).length > 0 && (
                        <div className="muted" style={{ fontSize: 12, marginTop: 4 }}>
                          {asArray<ModelRec>(a.models).map((m) => `${m.name} (${mb(m.size_bytes)})`).join(" · ")}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </CardBody>
          </Card>

          <Card>
            <CardHead title="Time-travel sandboxes" sub={`${sandboxes.length} active`} />
            <CardBody>
              {sandboxes.length === 0 ? <Empty>no sandboxes running</Empty> : (
                <Table
                  rows={sandboxes}
                  rowKey={(sb) => sb.short}
                  columns={[
                    { header: "id", cell: (sb) => <code className="mono">{sb.short}</code> },
                    { header: "source", cell: (sb) => <span className="muted">{sb.source || "—"}</span> },
                    { header: "state", cell: (sb) => <Badge tone={sb.running ? "good" : undefined}>{sb.running ? "running" : "stopped"}</Badge> },
                    { header: "where", cell: (sb) => sb.remote ? "FASRC" : "local" },
                    { header: "", align: "right", cell: (sb) => (
                      <div className="row" style={{ gap: 6, justifyContent: "flex-end" }}>
                        <Button size="sm" onClick={() => act("/api/tracking/timetravel/open", { short: sb.short })} disabled={busy != null}>open</Button>
                        <Button size="sm" onClick={() => act("/api/tracking/timetravel/stop", { short: sb.short }, "✓ stopped")} disabled={busy != null}>stop</Button>
                        <Button size="sm" variant="ghost" onClick={() => act("/api/tracking/timetravel/remove", { short: sb.short }, "✓ removed")} disabled={busy != null}>remove</Button>
                      </div>
                    ) },
                  ]}
                />
              )}
            </CardBody>
          </Card>
        </div>
      )}
    </Page>
  );
}
