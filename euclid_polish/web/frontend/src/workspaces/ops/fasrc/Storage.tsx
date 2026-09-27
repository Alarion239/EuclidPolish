/* Ops › FASRC › Storage: the remote data/checkpoint sizes, a remote file
 * browser (inspect FITS in the Inspect workspace, download anything), the
 * holylabs data-symlink repair and the confirmed checkpoint pull
 * (`rsync --delete-after`, a local job). URL: `dir` (browser folder). */
import { useMemo } from "react";
import { Link } from "react-router-dom";
import { apiPost } from "../../../api/client";
import { useJob } from "../../../api/jobs";
import { invalidate, useResource } from "../../../api/query";
import { formatBytes, formatDateTime, formatRelative } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Badge, Button, Callout, Card, CardBody, CardHead, DataTable, IconButton, JobProgress, LogTail, Section,
  Skeleton, Tooltip, confirm, toast, type DataColumn,
} from "../../../ui";
import {
  filesUrl, remoteDownloadHref, remoteInspectHref, type DataListing, type FilesResp, type MirrorStatus,
  type RemoteEntry,
} from "../api";
import { basename } from "../model";

function FileBrowser() {
  const [dir, setDir] = useUrlState("dir", "");
  const files = useResource<FilesResp>(filesUrl(dir), [dir], { ttl: 30_000 });
  const d = files.data;
  const columns = useMemo<DataColumn<RemoteEntry>[]>(() => [
    { id: "name", header: "Name", accessor: (e) => e.name,
      cell: (e) => (e.type === "dir" || (e.type === "link" && !e.inspectable)
        ? <button type="button" className="ops-linkbtn" onClick={() => setDir(e.path)}>{e.name}{e.type === "dir" ? "/" : ""}</button>
        : <span className="mono">{e.name}</span>) },
    { id: "type", header: "Type", width: 70, cell: (e) => <span className="ops-dim ops-small">{e.type}</span> },
    { id: "size", header: "Size", numeric: true, width: 92, accessor: (e) => e.size ?? -1,
      cell: (e) => <span className="mono ops-small">{e.size == null ? "—" : formatBytes(e.size)}</span> },
    { id: "mtime", header: "Modified", width: 110, accessor: (e) => e.mtime ?? 0,
      cell: (e) => <span className="ops-dim ops-small" title={formatDateTime(e.mtime)}>{e.mtime ? formatRelative(e.mtime) : "—"}</span> },
    { id: "actions", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 76,
      cell: (e) => (e.type === "file" ? (
        <span className="ops-row-actions">
          {e.inspectable && (
            <Tooltip content="Fetch (cached) and open in Inspect">
              <a className="ops-iconlink" href={remoteInspectHref(e.path)} aria-label={`Inspect ${e.name}`}>⌕</a>
            </Tooltip>
          )}
          <Tooltip content="Download (fetched through the local cache)">
            <a className="ops-iconlink" href={remoteDownloadHref(e.path)} download aria-label={`Download ${e.name}`}>↓</a>
          </Tooltip>
        </span>
      ) : null) },
  ], [setDir]);
  return (
    <Card>
      <CardHead title="Remote files" sub={d?.dir ?? "allowed roots"} right={
        <div className="ops-row">
          {dir && <IconButton size="sm" icon="arrowUp" label="Up" onClick={() => {
            const crumbs = d?.crumbs ?? [];
            setDir(crumbs.length > 1 ? crumbs[crumbs.length - 2].path : "");
          }} />}
          <IconButton size="sm" icon="reset" label="Reload" onClick={() => files.reload()} />
        </div>} />
      <CardBody>
        {d && d.crumbs.length > 0 && (
          <nav className="ops-crumbs" aria-label="Remote path">
            <button type="button" className="ops-linkbtn" onClick={() => setDir("")}>roots</button>
            {d.crumbs.map((c) => <span key={c.path}> / <button type="button" className="ops-linkbtn" onClick={() => setDir(c.path)}>{c.name}</button></span>)}
          </nav>
        )}
        {files.error && !d ? <Callout tone={files.error.status === 403 ? "warn" : "bad"} title="Cannot list this folder">{files.error.message}</Callout> : (
          <DataTable rows={d?.entries ?? []} columns={columns} rowKey={(e) => e.path} dense height={440}
            loading={files.loading && !d} aria-label="Remote files" exportName="fasrc-files" urlKey="rf"
            onRowClick={(e) => { if (e.type === "dir") setDir(e.path); }}
            empty="Empty folder." />
        )}
        {d?.truncated && <p className="ops-note">Only the first 2000 entries are listed.</p>}
      </CardBody>
    </Card>
  );
}

export function StoragePanel({ fasrcConnected }: { fasrcConnected: boolean }) {
  const listing = useResource<DataListing>(fasrcConnected ? "/api/fasrc/data-listing" : null, [fasrcConnected], { ttl: 5 * 60_000 });
  const mirror = useResource<MirrorStatus>("/api/fasrc/mirror/status", [], { ttl: 10_000 });
  const pull = useJob("fasrc:mirror");
  const d = listing.data;

  async function bootstrap() {
    if (!(await confirm({ title: "Re-create the data symlinks on FASRC?",
      message: "Links euclid_psf and COSMOS2025 under the netscratch data dir to their holylabs copies (idempotent).",
      confirmLabel: "Re-link" }))) return;
    try {
      const r = await apiPost<{ ok: boolean; output?: string; error?: string }>("/api/fasrc/bootstrap-data");
      if (r.ok) toast.success("Data symlinks re-created"); else toast.error(r.error || "failed");
    } catch (e) { toast.error(e instanceof Error ? e.message : String(e)); }
    void invalidate("/api/fasrc/data-listing");
  }
  async function pullCheckpoints() {
    if (!(await confirm({ title: "Pull the ensemble checkpoints from FASRC?",
      message: `rsync --delete-after into ${mirror.data?.local_dir || "the local ensemble dir"}: local files the FASRC copy lacks are DELETED.`,
      tone: "danger", confirmLabel: "Pull and delete", requireText: "pull" }))) return;
    await pull.run("/api/fasrc/mirror/trigger", { confirm: "1" }, {
      onDone: (j) => { void invalidate("/api/fasrc/mirror/status"); if (j.status === "done") toast.success("Checkpoints pulled"); },
    });
  }

  if (!fasrcConnected) return <Callout tone="warn" title="FASRC offline">Connect to browse the remote storage.</Callout>;
  const du = d?.du ?? [];
  return (
    <div className="ops-stack">
      <Card>
        <CardHead title="Remote storage" sub={d?.data_dir} right={
          <div className="ops-row">
            <Button size="sm" variant="ghost" onClick={bootstrap}>Re-link data</Button>
            <Button size="sm" icon="download" loading={pull.busy} onClick={pullCheckpoints}>Pull checkpoints</Button>
            <IconButton size="sm" icon="reset" label="Rescan (remote du, ~20 s)" onClick={() => listing.reload()} />
          </div>} />
        <CardBody>
          <JobProgress job={pull.job} error={pull.error} />
          {mirror.data?.last_run_at && (
            <p className="ops-note">
              Last pull {formatRelative(mirror.data.last_run_at)} ·{" "}
              <Badge size="sm" tone={mirror.data.last_rc === 0 ? "good" : "bad"}>{mirror.data.last_rc === 0 ? "ok" : `rc ${mirror.data.last_rc}`}</Badge>
              {mirror.data.last_error && <span className="ops-bad"> {mirror.data.last_error}</span>}
            </p>
          )}
          {listing.loading && !d ? <Skeleton lines={5} />
            : listing.error && !d ? <Callout tone="bad">{listing.error.message}</Callout>
            : d && d.ok === false ? <Callout tone="warn">{d.error || "Listing unavailable"}</Callout>
            : (
              <div className="ops-du" role="list" aria-label="Directory sizes">
                {du.map(([size, path]) => (
                  <div role="listitem" key={path} className="ops-du__row">
                    <span className="mono ops-du__size">{size}</span>
                    <span className="mono ops-ellipsis" title={path}>{basename(path)}</span>
                  </div>
                ))}
                {!du.length && <span className="ops-dim">No listing — is the data dir set?</span>}
              </div>
            )}
          {d && (d.tfrecords?.length || d.checkpoints?.length) ? (
            <Section title="Records and checkpoints" sub={`${d.tfrecords?.length ?? 0} TFRecords · ${d.checkpoints?.length ?? 0} checkpoint files`} collapsible defaultOpen={false}>
              <LogTail text={[...(d.tfrecords ?? []).map((t) => `${formatBytes(t.size).padStart(9)}  ${t.path}`),
                ...(d.checkpoints ?? []).map((c) => `${formatBytes(c.size).padStart(9)}  ${c.mtime ?? ""}  ${c.path}`)].join("\n")}
                style={{ maxHeight: 260 }} />
            </Section>
          ) : null}
          <p className="ops-note">FASRC files inspected here are cached locally (see <Link to="/inspect">Inspect</Link>).</p>
        </CardBody>
      </Card>
      <FileBrowser />
    </div>
  );
}
