/* settings/about (spec §8.8): which code the server runs (boot commit vs
 * HEAD, C3), the committed console bundle, the runtime (Python, platform,
 * packages, Node) and the disk: free space with its warning level, the
 * experiments' member-cache budget and the disk usage of every data root
 * (GET /api/system; measured by a local job, POST
 * /api/system/disk-usage/refresh — started automatically when the last
 * measurement is stale). Data roots open in the inspector (`root:<id>`). */
import { useEffect, useMemo, version as reactVersion, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { useJobsStore } from "../../../api/jobs";
import { invalidate, useResource } from "../../../api/query";
import { registerInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import { startJob } from "../../../app/RunActions";
import { useVersion } from "../../../app/status";
import { formatBytes, formatCount, formatDateTime, formatPercent, formatRelative } from "../../../format";
import {
  Badge, Button, Callout, Card, CardBody, CardHead, CopyButton, DataTable, DefList, EmptyState, Page, PageHead,
  ProgressBar, Skeleton, type DataColumn, type Tone,
} from "../../../ui";
import "../settings.css";

type Level = "ok" | "warn" | "bad" | "unknown";
type RootRow = { id: string; label: string; path: string; group: string; bytes: number; files: number; exists: boolean };
export type SystemInfo = {
  python: { version: string; implementation: string; executable: string };
  platform: { system: string; release: string; machine: string; platform: string };
  packages: Record<string, string | null>;
  node: string | null;
  pid: number;
  data_dir: string;
  noise_model: string;
  disk: {
    path: string; total_bytes: number; free_bytes: number; used_bytes: number; used_fraction: number | null;
    level: Level; warn_below_bytes: number; bad_below_bytes: number; warn_used_fraction: number;
  };
  roots: { items: RootRow[]; computed_at: string | null; total_bytes: number | null; stale: boolean; refresh_job: string | null };
  experiments: { cache_budget_bytes: number; min_free_bytes: number; cache_bytes: number | null; outputs_bytes: number | null };
};

export const SYSTEM_URL = "/api/system";
const DISK_KEY = "run:disk-usage";
const LEVEL_TONE: Record<Level, Tone> = { ok: "good", warn: "warn", bad: "bad", unknown: "neutral" };
const DISK_JOB = { key: DISK_KEY, label: "Measure disk usage", url: "/api/system/disk-usage/refresh" };

function useSystem() {
  return useResource<SystemInfo>(SYSTEM_URL, [], { ttl: 30_000 });
}

/** Inspector kind `root:<id>`: one data root's size. */
function RootInspector({ id }: { id: string }) {
  const sys = useSystem();
  const root = sys.data?.roots.items.find((r) => r.id === id) ?? null;
  if (sys.loading && !sys.data) return <Skeleton lines={3} />;
  if (!root) return <EmptyState compact icon="database" title="Not measured">{id}</EmptyState>;
  const total = sys.data?.roots.total_bytes ?? 0;
  return (
    <div className="settings-stack">
      <DefList dense items={[
        ["path", <span className="settings-row"><code className="mono">{root.path}</code><CopyButton value={root.path} label="Copy path" /></span>],
        ["size", formatBytes(root.bytes)],
        ["files", formatCount(root.files)],
        total > 0 ? ["share", formatPercent(root.bytes / total)] : null,
        ["measured", sys.data?.roots.computed_at ? formatRelative(sys.data.roots.computed_at) : "—"],
        !root.exists ? ["state", "does not exist"] : null,
      ]} />
      <Button asChild size="sm"><Link to="/inspect">Browse files in Inspect</Link></Button>
    </div>
  );
}
registerInspector("root", RootInspector, { title: (id) => `Data root · ${id}` });

/** A short commit hash with the full one in its tooltip and a copy button. */
function Commit({ full, short, label }: { full: string | null | undefined; short: string | null | undefined; label: string }) {
  if (!full) return <span className="muted">—</span>;
  return (
    <span className="settings-row settings-row--tight">
      <code className="mono" title={full}>{short || full.slice(0, 7)}</code>
      <CopyButton value={full} label={`Copy the ${label} hash`} />
    </span>
  );
}

function DiskCard({ sys }: { sys: SystemInfo }) {
  const d = sys.disk;
  const e = sys.experiments;
  return (
    <Card>
      <CardHead title="Disk" sub={d.path}
        right={<Badge tone={LEVEL_TONE[d.level]} dot>{d.level === "ok" ? "healthy" : d.level === "unknown" ? "unknown" : d.level === "bad" ? "critically low" : "low"}</Badge>} />
      <CardBody>
        <div className="settings-stack">
          <div className="disk-meter" data-level={d.level}>
            <div className="disk-meter__nums">
              <span className="disk-meter__free">{formatBytes(d.free_bytes)} free</span>
              <span className="muted">of {formatBytes(d.total_bytes)} · {d.used_fraction != null ? formatPercent(d.used_fraction, 0) : "—"} used</span>
            </div>
            <ProgressBar value={d.used_fraction != null ? d.used_fraction * 100 : null} max={100}
              aria-label="Data disk used" tone={LEVEL_TONE[d.level]} />
          </div>
          {d.level !== "ok" && d.level !== "unknown" && (
            <Callout tone={d.level === "bad" ? "bad" : "warn"} title="Free space is low">
              Below {formatBytes(d.warn_below_bytes)} free (or {formatPercent(d.warn_used_fraction, 0)} used) the console warns;
              below {formatBytes(d.bad_below_bytes)} experiments start being refused.
            </Callout>
          )}
          <DefList dense items={[
            sys.roots.total_bytes != null
              ? ["console data", `${formatBytes(sys.roots.total_bytes)} in the data roots · ${formatBytes(Math.max(0, d.used_bytes - sys.roots.total_bytes))} used elsewhere on this disk`]
              : null,
            ["member-SR cache", e.cache_bytes != null
              ? `${formatBytes(e.cache_bytes)} of ${formatBytes(e.cache_budget_bytes)} budget` : `budget ${formatBytes(e.cache_budget_bytes)}`],
            e.outputs_bytes != null ? ["experiment outputs", formatBytes(e.outputs_bytes)] : null,
            ["experiments keep free", formatBytes(e.min_free_bytes)],
          ]} />
          {e.cache_bytes != null && (
            <ProgressBar value={Math.min(100, (100 * e.cache_bytes) / e.cache_budget_bytes)} max={100}
              aria-label="Member-SR cache budget used" label={formatPercent(e.cache_bytes / e.cache_budget_bytes, 0)} />
          )}
        </div>
      </CardBody>
    </Card>
  );
}

export default function About() {
  const version = useVersion();
  const v = version.data;
  const sys = useSystem();
  const s = sys.data;
  const jobId = useJobsStore((st) => st.keyed[DISK_KEY] ?? null);
  const jobStatus = useJobsStore((st) => (jobId ? st.jobs[jobId]?.status ?? null : null));
  const measuring = jobStatus === "running" || !!s?.roots.refresh_job;

  // A finished measurement → reload the numbers.
  useEffect(() => { if (jobStatus === "done") void invalidate(SYSTEM_URL); }, [jobStatus, jobId]);
  // Poll while a measurement started elsewhere runs.
  useResource<SystemInfo>(s?.roots.refresh_job ? SYSTEM_URL : null, [s?.roots.refresh_job], { poll: 1500 });
  // Opening the page never starts a job: a stale measurement is flagged and
  // re-measured from "Measure now" (or the palette).

  const measure = () => { void startJob(DISK_JOB); };
  usePageActions([
    { id: "about:measure", label: "Measure disk usage per data root", group: "Settings", keywords: ["disk", "du", "space"],
      disabled: measuring, run: measure },
    { id: "about:refresh", label: "Refresh the About page", group: "Settings", run: () => { void version.reload(); void sys.reload(); } },
  ]);

  const total = s?.roots.total_bytes ?? 0;
  const columns = useMemo<DataColumn<RootRow>[]>(() => [
    { id: "label", header: "Root", cell: (r) => <span title={r.path}>{r.label}</span> },
    { id: "group", header: "Where", width: 70 },
    { id: "bytes", header: "Size", numeric: true, cell: (r) => (r.exists ? formatBytes(r.bytes) : "—") },
    { id: "files", header: "Files", numeric: true, cell: (r) => (r.exists ? formatCount(r.files) : "—") },
    { id: "share", header: "Share", sortable: false, filterable: false, csv: (r) => (total ? (r.bytes / total).toFixed(4) : ""),
      accessor: (r) => (total ? r.bytes / total : 0),
      cell: (r) => (
        <span className="root-bar" aria-label={total ? formatPercent(r.bytes / total) : "—"}>
          <span style={{ width: `${total ? Math.max(0.5, (100 * r.bytes) / total) : 0}%` }} />
        </span>
      ) },
    { id: "path", header: "Path", hidden: true, cell: (r) => <code className="mono">{r.path}</code> },
  ], [total]);

  return (
    <Page className="settings-about">
      <PageHead eyebrow="settings · about" title="About" sub="Server code, bundle, runtime and disk."
        right={<Button size="sm" variant="ghost" icon="reset" onClick={() => { void version.reload(); void sys.reload(); }}>Refresh</Button>} />
      <div className="settings-cards">
        <Card>
          <CardHead title="Server" right={v ? (v.behind ? <Badge tone="warn" dot>behind HEAD</Badge> : <Badge tone="good" dot>at HEAD</Badge>) : undefined} />
          <CardBody>
            {version.loading && !v && <Skeleton lines={5} />}
            {version.error && !v && <Callout tone="bad" title="Could not read /api/version">{version.error.message}</Callout>}
            {v && (
              <div className="settings-stack">
                {v.behind && (
                  <Callout tone="warn" title="Restart the server">
                    It runs <code className="mono">{v.boot_short}</code>; the checkout is at <code className="mono">{v.head_short}</code>.
                  </Callout>
                )}
                <DefList dense items={[
                  ["boot commit", <Commit full={v.boot_commit} short={v.boot_short} label="boot commit" />],
                  ["HEAD", <Commit full={v.head_commit} short={v.head_short} label="HEAD" />],
                  ["working tree", v.dirty ? <Badge tone="warn">uncommitted changes</Badge> : <Badge tone="good">clean</Badge>],
                  ["started", v.started_at ? `${formatDateTime(v.started_at)} (${formatRelative(v.started_at)})` : "—"],
                  ["pid", v.pid != null ? <code className="mono">{v.pid}</code> : "—"],
                  ["dist built", v.dist?.built_at ? `${formatDateTime(v.dist.built_at)} (${formatRelative(v.dist.built_at)})` : "—"],
                  ["index hash", v.dist?.index_hash ? <code className="mono">{v.dist.index_hash}</code> : "—"],
                ]} />
              </div>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHead title="Runtime" sub={s ? `${s.platform.system} ${s.platform.release} · ${s.platform.machine}` : undefined} />
          <CardBody>
            {sys.loading && !s && <Skeleton lines={5} />}
            {sys.error && !s && <Callout tone="bad" title="Could not read /api/system">{sys.error.message}</Callout>}
            {s && (
              <DefList dense items={[
                ["Python", <span title={s.python.executable}>{s.python.implementation} {s.python.version}</span>],
                ["Node", s.node ?? <span className="muted">not on the server PATH</span>],
                ["bundle", `React ${reactVersion} · ${import.meta.env.MODE}`],
                ...Object.entries(s.packages).map(([name, ver]) => [name, ver ?? <span className="muted">not installed</span>] as [string, ReactNode]),
                ["noise model", <code className="mono">{s.noise_model}</code>],
                ["data dir", <code className="mono">{s.data_dir}</code>],
              ]} />
            )}
          </CardBody>
        </Card>

        {s && <DiskCard sys={s} />}
      </div>

      <Card style={{ marginTop: "var(--s4)" }}>
        <CardHead title="Disk usage per data root"
          sub={s?.roots.computed_at ? `measured ${formatRelative(s.roots.computed_at)} · ${formatBytes(total)} in total` : "not measured yet"}
          right={(
            <div className="settings-row">
              {s?.roots.stale && !measuring && s.roots.computed_at && <Badge tone="warn">stale</Badge>}
              <Button size="sm" loading={measuring} onClick={measure}>Measure now</Button>
            </div>
          )} />
        <CardBody>
          {s && s.roots.items.length === 0 && !measuring && (
            <EmptyState compact icon="database" title="Not measured yet" action={<Button size="sm" onClick={measure}>Measure</Button>} />
          )}
          {(s?.roots.items.length ?? 0) > 0 && (
            <DataTable rows={s!.roots.items} columns={columns} rowKey={(r) => r.id} aria-label="Data roots"
              defaultSort={[{ id: "bytes", desc: true }]} exportName="data-roots" urlKey="roots" height="auto"
              inspect={(r) => ({ kind: "root", id: r.id })} dense />
          )}
          {measuring && !(s?.roots.items.length) && <Skeleton lines={4} />}
        </CardBody>
      </Card>
    </Page>
  );
}
