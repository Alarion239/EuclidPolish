/* System › Storage (`/system/storage`): is the console within its disk
 * budget, here and on FASRC? This laptop's data disk — one bar whose tone
 * follows the ONE rule that also raises the Home alert (routes/system.py
 * `disk_level`), and how much of the used space is ours — the experiments'
 * member-SR cache, the disk usage per data root (measured by a job, only on
 * "Measure now"), then FASRC: the remote sizes as a sortable table, the
 * remote file browser and Re-link data; and the maintenance actions (sync the
 * catalogue-evaluation results from FASRC, drop their cached PNGs), each
 * confirmed. Data roots open in the inspector (`root:<id>`).
 * URL: `side=fasrc` (open on FASRC), `dir` (the remote browser's folder), the
 * tables' `roots.*` / `du.*` / `rf.*`. */
import { useEffect, useMemo, useRef } from "react";
import { Link } from "react-router-dom";
import { apiPost } from "../../../api/client";
import { useJobsStore } from "../../../api/jobs";
import { invalidate, useResource } from "../../../api/query";
import { pagePath } from "../../../app/nav";
import { usePageActions } from "../../../app/palette";
import { startJob } from "../../../app/RunActions";
import { useFasrcStatus } from "../../../app/status";
import { ConnectionBar } from "../../../fasrc";
import { formatBytes, formatCount, formatDateTime, formatPercent, formatRelative } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Badge, Button, Callout, Caption, Card, CardBody, CardHead, DataTable, Details, EmptyState, IconButton,
  LogTail, Page, ProgressBar, Skeleton, Tooltip, confirm, toast, type DataColumn, type Tone,
} from "../../../ui";
import {
  SYSTEM_URL, filesUrl, remoteDownloadHref, remoteInspectHref, type DataListing, type DiskLevel, type FilesResp,
  type RemoteEntry, type RootRow, type SystemInfo,
} from "../api";
import { basename, diskCaption, diskThresholdText, duBytes, experimentsLine } from "../model";
import { useArrivalScroll } from "../../../hooks/arrival";
import "../system.css";

const DISK_KEY = "run:disk-usage";
const DISK_JOB = { key: DISK_KEY, label: "Measure disk usage", url: "/api/system/disk-usage/refresh" };
const LEVEL_TONE: Record<DiskLevel, Tone> = { ok: "good", warn: "warn", bad: "bad", unknown: "neutral" };
const LEVEL_WORDS: Record<DiskLevel, string> = { ok: "", warn: "low", bad: "critically low", unknown: "unknown" };

type DuRow = { path: string; size: string; bytes: number | null };

function DiskCard({ sys }: { sys: SystemInfo }) {
  const d = sys.disk;
  const e = sys.experiments;
  const problem = d.level !== "ok";
  return (
    <Card>
      <CardHead title="This laptop's data disk" sub={<code className="mono">{d.path}</code>}
        right={problem ? <Badge tone={LEVEL_TONE[d.level]} dot>{LEVEL_WORDS[d.level]}</Badge> : undefined} />
      <CardBody>
        <div className="sys-stack">
          <div className="sys-disk">
            <div className="sys-disk__nums">
              <span className="sys-disk__free">{formatBytes(d.free_bytes)} free</span>
              <span className="sys-dim">of {formatBytes(d.total_bytes)} · {d.used_fraction != null ? formatPercent(d.used_fraction, 0) : "—"} used</span>
            </div>
            <ProgressBar value={d.used_fraction != null ? d.used_fraction * 100 : null} max={100}
              aria-label="Data disk used" tone={problem ? LEVEL_TONE[d.level] : "neutral"} />
            <Caption>{diskCaption(d, sys.roots.total_bytes)}. {diskThresholdText(d)}</Caption>
          </div>
          <Caption>{experimentsLine(e)}</Caption>
        </div>
      </CardBody>
    </Card>
  );
}

function RootsCard({ sys, measuring, measure }: { sys: SystemInfo | null; measuring: boolean; measure: () => void }) {
  const total = sys?.roots.total_bytes ?? 0;
  const columns = useMemo<DataColumn<RootRow>[]>(() => [
    { id: "label", header: "Root", cell: (r) => <span title={r.path}>{r.label}</span> },
    { id: "group", header: "Where", width: 90 },
    { id: "bytes", header: "Size", numeric: true, cell: (r) => (r.exists ? formatBytes(r.bytes) : "—") },
    { id: "files", header: "Files", numeric: true, cell: (r) => (r.exists ? formatCount(r.files) : "—") },
    { id: "share", header: "Share", sortable: false, filterable: false, csv: (r) => (total ? (r.bytes / total).toFixed(4) : ""),
      accessor: (r) => (total ? r.bytes / total : 0),
      cell: (r) => (
        <span className="root-bar" role="img" aria-label={total ? formatPercent(r.bytes / total) : "—"}>
          <span style={{ width: `${total ? Math.max(0.5, (100 * r.bytes) / total) : 0}%` }} />
        </span>
      ) },
    { id: "path", header: "Path", hidden: true, cell: (r) => <code className="mono">{r.path}</code> },
  ], [total]);
  return (
    <Card>
      <CardHead title="Disk usage per data root"
        sub={sys?.roots.computed_at ? `measured ${formatRelative(sys.roots.computed_at)}` : "not measured yet"}
        right={(
          <div className="sys-row">
            {sys?.roots.stale && !measuring && sys.roots.computed_at && <Badge tone="warn">stale</Badge>}
            <Button size="sm" loading={measuring} onClick={measure}>Measure now</Button>
          </div>
        )} />
      <CardBody>
        {sys && sys.roots.items.length === 0 && !measuring && (
          <EmptyState compact icon="database" title="Not measured yet" action={<Button size="sm" onClick={measure}>Measure</Button>}>
            A local job walks every data root once (a minute or two).
          </EmptyState>
        )}
        {(sys?.roots.items.length ?? 0) > 0 && (
          <DataTable rows={sys!.roots.items} columns={columns} rowKey={(r) => r.id} aria-label="Data roots"
            defaultSort={[{ id: "bytes", desc: true }]} exportName="data-roots" urlKey="roots" height="auto"
            inspect={(r) => ({ kind: "root", id: r.id })} dense />
        )}
        {measuring && !(sys?.roots.items.length) && <Skeleton lines={4} />}
      </CardBody>
    </Card>
  );
}

function RemoteFiles() {
  const [dir, setDir] = useUrlState("dir", "");
  const files = useResource<FilesResp>(filesUrl(dir), [dir], { ttl: 30_000 });
  const d = files.data;
  const columns = useMemo<DataColumn<RemoteEntry>[]>(() => [
    { id: "name", header: "Name", accessor: (e) => e.name,
      cell: (e) => (e.type === "dir" || (e.type === "link" && !e.inspectable)
        ? <button type="button" className="sys-linkbtn" onClick={() => setDir(e.path)}>{e.name}{e.type === "dir" ? "/" : ""}</button>
        : <span className="mono">{e.name}</span>) },
    { id: "type", header: "Type", width: 70, priority: 2, cell: (e) => <span className="sys-dim sys-small">{e.type}</span> },
    { id: "size", header: "Size", numeric: true, width: 92, accessor: (e) => e.size ?? -1,
      cell: (e) => <span className="mono sys-small">{e.size == null ? "—" : formatBytes(e.size)}</span> },
    { id: "mtime", header: "Modified", width: 110, priority: 1, accessor: (e) => e.mtime ?? 0,
      cell: (e) => <span className="sys-dim sys-small" title={formatDateTime(e.mtime)}>{e.mtime ? formatRelative(e.mtime) : "—"}</span> },
    { id: "actions", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 76,
      cell: (e) => (e.type === "file" ? (
        <span className="sys-row-actions">
          {e.inspectable && (
            <Tooltip content="Fetch (cached) and open in Files">
              <a className="sys-iconlink" href={remoteInspectHref(e.path)} aria-label={`Open ${e.name} in Files`}>⌕</a>
            </Tooltip>
          )}
          <Tooltip content="Download (fetched through the local cache)">
            <a className="sys-iconlink" href={remoteDownloadHref(e.path)} download aria-label={`Download ${e.name}`}>↓</a>
          </Tooltip>
        </span>
      ) : null) },
  ], [setDir]);
  return (
    <Card>
      <CardHead title="Remote files" sub={d?.dir ?? "allowed roots"} right={
        <div className="sys-row">
          {dir && <IconButton size="sm" icon="arrowUp" label="Up" onClick={() => {
            const crumbs = d?.crumbs ?? [];
            setDir(crumbs.length > 1 ? crumbs[crumbs.length - 2].path : "");
          }} />}
          <IconButton size="sm" icon="reset" label="Reload" onClick={() => files.reload()} />
        </div>} />
      <CardBody>
        {d && d.crumbs.length > 0 && (
          <nav className="sys-crumbs" aria-label="Remote path">
            <button type="button" className="sys-linkbtn" onClick={() => setDir("")}>roots</button>
            {d.crumbs.map((c) => <span key={c.path}> / <button type="button" className="sys-linkbtn" onClick={() => setDir(c.path)}>{c.name}</button></span>)}
          </nav>
        )}
        {files.error && !d ? <Callout tone={files.error.status === 403 ? "warn" : "bad"} title="Cannot list this folder">{files.error.message}</Callout> : (
          <DataTable rows={d?.entries ?? []} columns={columns} rowKey={(e) => e.path} dense height={440}
            loading={files.loading && !d} aria-label="Remote files" exportName="fasrc-files" urlKey="rf"
            onRowClick={(e) => { if (e.type === "dir") setDir(e.path); }}
            empty="Empty folder." />
        )}
        {d?.truncated && <p className="sys-note">Only the first 2000 entries are listed.</p>}
      </CardBody>
    </Card>
  );
}

function FasrcStorage({ connected }: { connected: boolean }) {
  const listing = useResource<DataListing>(connected ? "/api/fasrc/data-listing" : null, [connected], { ttl: 5 * 60_000 });
  const d = listing.data;
  const du = useMemo<DuRow[]>(() => (d?.du ?? []).map(([size, path]) => ({ path, size, bytes: duBytes(size) })), [d]);
  const columns = useMemo<DataColumn<DuRow>[]>(() => [
    { id: "path", header: "Folder", accessor: (r) => basename(r.path), cell: (r) => <span className="mono" title={r.path}>{basename(r.path)}</span> },
    { id: "bytes", header: "Size", numeric: true, width: 110, accessor: (r) => r.bytes, cell: (r) => <span className="mono">{r.size}</span>,
      csv: (r) => r.size },
  ], []);

  async function relink() {
    if (!(await confirm({ title: "Re-create the data symlinks on FASRC?",
      message: "Links euclid_psf and COSMOS2025 under the netscratch data dir to their holylabs copies (idempotent).",
      confirmLabel: "Re-link" }))) return;
    try {
      const r = await apiPost<{ ok: boolean; output?: string; error?: string }>("/api/fasrc/bootstrap-data");
      if (r.ok) toast.success("Data symlinks re-created"); else toast.error(r.error || "failed");
    } catch (e) { toast.error(e instanceof Error ? e.message : String(e)); }
    void invalidate("/api/fasrc/data-listing");
  }
  usePageActions([
    { id: "storage-relink", label: "Re-link the FASRC data symlinks", group: "Storage", disabled: !connected, run: () => void relink() },
  ]);

  if (!connected) {
    return <Callout tone="warn" title="FASRC offline" action={<ConnectionBar />}>Connect to read the remote storage.</Callout>;
  }
  return (
    <div className="sys-stack">
      <Card>
        <CardHead title="Remote storage" sub={d?.data_dir ? <code className="mono">{d.data_dir}</code> : undefined} right={
          <div className="sys-row">
            <Button size="sm" variant="ghost" onClick={() => void relink()}>Re-link data</Button>
            <IconButton size="sm" icon="reset" label="Rescan (remote du, ~20 s)" onClick={() => listing.reload()} />
          </div>} />
        <CardBody>
          {listing.loading && !d ? <Skeleton lines={5} />
            : listing.error && !d ? <Callout tone="bad">{listing.error.message}</Callout>
            : d && d.ok === false ? <Callout tone="warn">{d.error || "Listing unavailable"}</Callout>
            : (
              <DataTable rows={du} columns={columns} rowKey={(r) => r.path} aria-label="Remote folder sizes" dense height="auto"
                defaultSort={[{ id: "bytes", desc: true }]} urlKey="du" exportName="fasrc-du" hideToolbar={du.length < 12}
                empty="No listing: is the data dir set?" />
            )}
          {d && (d.tfrecords?.length || d.checkpoints?.length) ? (
            <Details summary={`Records and checkpoints on FASRC (${d.tfrecords?.length ?? 0} TFRecords · ${d.checkpoints?.length ?? 0} checkpoint files)`}>
              <LogTail text={[...(d.tfrecords ?? []).map((t) => `${formatBytes(t.size).padStart(9)}  ${t.path}`),
                ...(d.checkpoints ?? []).map((c) => `${formatBytes(c.size).padStart(9)}  ${c.mtime ?? ""}  ${c.path}`)].join("\n")}
                style={{ maxHeight: 260 }} />
            </Details>
          ) : null}
          <p className="sys-note">
            Pull checkpoints from Models › <Link to={pagePath("models", { tab: "members", params: { mode: "starfull" } })}>Members</Link>.
            FASRC files opened here are cached locally (see <Link to="/files">Files</Link>).
          </p>
        </CardBody>
      </Card>
      <RemoteFiles />
    </div>
  );
}

/** Sync the catalogue evaluation's results from FASRC (rsync --delete-after, confirmed). */
async function syncEvaluation() {
  if (!(await confirm({ title: "Sync the evaluation results from FASRC?",
    message: "rsync --delete-after will delete local-only results in data/eval_results that FASRC does not have.",
    tone: "danger", confirmLabel: "Sync and delete local-only" }))) return;
  try {
    const r = (await apiPost<{ ok?: boolean; error?: string; n_ok?: number; n?: number }>("/api/evaluation/sync", { confirm: "1" })) ?? {};
    if (r.ok === false || r.error) throw new Error(String(r.error ?? "refused"));
    toast.success(`Synced: ${formatCount(r.n_ok)} of ${formatCount(r.n)} objects reconstructed`);
    void invalidate("/api/sky/");
    void invalidate("/api/evaluation");
  } catch (e) {
    toast.error("FASRC sync failed", { description: e instanceof Error ? e.message : String(e) });
  }
}

/** Drop the evaluation run's cached eye/solar PNGs (they re-render from the FITS). */
async function dropCachedPngs() {
  if (!(await confirm({ title: "Drop the cached eye/solar PNGs?",
    message: "Deletes the evaluation run's cached eye and solar PNG renders in data/eval_results. The FITS stay; each image re-renders from them the next time it is shown.",
    confirmLabel: "Drop cached PNGs" }))) return;
  try {
    const r = (await apiPost<{ ok?: boolean; error?: string; removed?: number }>("/api/evaluation/rerender", {})) ?? {};
    if (r.ok === false || r.error) throw new Error(String(r.error ?? "refused"));
    toast.success(`Dropped ${formatCount(r.removed ?? 0)} cached PNG${r.removed === 1 ? "" : "s"}`);
    void invalidate("/api/sky/");
  } catch (e) {
    toast.error("Cached PNGs: failed", { description: e instanceof Error ? e.message : String(e) });
  }
}

export default function Storage() {
  const [side] = useUrlState("side", "");
  const sys = useResource<SystemInfo>(SYSTEM_URL, [], { ttl: 30_000 });
  const s = sys.data;
  const fasrc = useFasrcStatus();
  const connected = !!fasrc.data?.ssh_connected;
  const jobId = useJobsStore((st) => st.keyed[DISK_KEY] ?? null);
  const jobStatus = useJobsStore((st) => (jobId ? st.jobs[jobId]?.status ?? null : null));
  const measuring = jobStatus === "running" || !!s?.roots.refresh_job;
  const fasrcRef = useRef<HTMLElement>(null);

  // A finished measurement → reload the numbers; poll while one started elsewhere runs.
  useEffect(() => { if (jobStatus === "done") void invalidate(SYSTEM_URL); }, [jobStatus, jobId]);
  useResource<SystemInfo>(s?.roots.refresh_job ? SYSTEM_URL : null, [s?.roots.refresh_job], { poll: 1500 });
  // ?side=fasrc (the old FASRC › Storage view) opens on FASRC, once the local cards have their height.
  useArrivalScroll(fasrcRef, side === "fasrc", !!s);
  // Opening the page never starts a job: a stale measurement is flagged and
  // re-measured from "Measure now" (or the palette).
  const measure = () => { void startJob(DISK_JOB); };
  usePageActions([
    { id: "storage:measure", label: "Measure disk usage per data root", group: "Storage", keywords: ["disk", "du", "space"],
      disabled: measuring, run: measure },
    { id: "storage:sync-eval", label: "Sync the evaluation results from FASRC", group: "Storage", disabled: !connected,
      run: () => void syncEvaluation() },
    { id: "storage:drop-pngs", label: "Drop the evaluation's cached PNGs", group: "Storage", run: () => void dropCachedPngs() },
  ]);

  return (
    <Page className="sys-page">
      <section className="sys-stack" aria-labelledby="storage-local">
        <h2 id="storage-local" className="sys-section">This laptop</h2>
        {sys.loading && !s && <Skeleton lines={5} />}
        {sys.error && !s && <Callout tone="bad" title="Could not read /api/system">{sys.error.message}</Callout>}
        {s && <DiskCard sys={s} />}
        <RootsCard sys={s ?? null} measuring={measuring} measure={measure} />
      </section>
      <section className="sys-stack" aria-labelledby="storage-fasrc" ref={fasrcRef}>
        <h2 id="storage-fasrc" className="sys-section">FASRC</h2>
        {fasrc.data ? <FasrcStorage connected={connected} /> : <Skeleton lines={3} />}
      </section>
      <section className="sys-stack" aria-labelledby="storage-maint">
        <h2 id="storage-maint" className="sys-section">Maintenance</h2>
        <Card>
          <CardBody>
            <div className="sys-maint">
              <div>
                <strong>Sync the evaluation results from FASRC</strong>
                <p className="sys-note">The catalogue evaluation's reconstructions (data/eval_results), with rsync --delete-after.</p>
              </div>
              <Button size="sm" disabled={!connected} onClick={() => void syncEvaluation()}>Sync from FASRC…</Button>
              <div>
                <strong>Drop the cached PNGs</strong>
                <p className="sys-note">The evaluation's eye / solar renders; they re-render from the FITS when next shown.</p>
              </div>
              <Button size="sm" onClick={() => void dropCachedPngs()}>Drop cached PNGs…</Button>
            </div>
          </CardBody>
        </Card>
      </section>
    </Page>
  );
}
