/* ops/provenance (spec §8.7): the lineage browser over data/_prov, the
 * sidecars next to the data and the checkpoint stamps. Search the records
 * (server-side, ANDed tokens), filter by kind / verdict / source, and walk a
 * record's ancestors and descendants; the verdict compares a product's model
 * with the active ensemble members (current / stale / unknown).
 * URL: `q`, `kind`, `verdict`, `source`, `id` (the open record). */
import { useEffect, useMemo, useState } from "react";
import { apiPost } from "../../../api/client";
import { invalidate, setResourceData, useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import { formatCount, formatDateTime, formatRelative } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Badge, Callout, Card, CardBody, CardHead, DataTable, IconButton, Input, Kpi, Page, Segmented,
  Select, Skeleton, toast, type DataColumn,
} from "../../../ui";
import {
  PROV_SUMMARY_URL, provRecordsUrl, type ProvRecordsResp, type ProvRow, type ProvSummary,
} from "../api";
import { ProvDetail, VerdictBadge } from "../provenance/ProvDetail";
import "../ops.css";

type VerdictFilter = "" | "current" | "stale" | "unknown";

export default function Provenance() {
  const summary = useResource<ProvSummary>(PROV_SUMMARY_URL, [], { ttl: 60_000 });
  const [q, setQ] = useUrlState("q", "");
  const [kind, setKind] = useUrlState("kind", "");
  const [verdict, setVerdict] = useUrlState<VerdictFilter>("verdict", "",
    { parse: (r) => (["current", "stale", "unknown"].includes(r) ? r as VerdictFilter : undefined) });
  const [source, setSource] = useUrlState("source", "");
  const [id, setId] = useUrlState("id", "", { replace: false });
  const [draft, setDraft] = useState(q);
  useEffect(() => { setDraft(q); }, [q]);
  // Debounced server search: typing updates the URL 300 ms after the last key.
  useEffect(() => {
    if (draft === q) return undefined;
    const t = setTimeout(() => setQ(draft.trim()), 300);
    return () => clearTimeout(t);
  }, [draft, q, setQ]);
  const records = useResource<ProvRecordsResp>(provRecordsUrl({ q, kind, verdict, source }), [q, kind, verdict, source], { ttl: 60_000 });
  const [rebuilding, setRebuilding] = useState(false);
  const s = summary.data;

  async function rebuild() {
    setRebuilding(true);
    try {
      const r = await apiPost<ProvSummary>("/api/provenance/rebuild");
      setResourceData(PROV_SUMMARY_URL, r);
      toast.success(`Index rebuilt: ${formatCount(r.total)} records in ${r.build_seconds.toFixed(1)} s`);
      void invalidate("/api/provenance/");
    } catch (e) { toast.error(e instanceof Error ? e.message : String(e)); }
    finally { setRebuilding(false); }
  }
  usePageActions([
    { id: "prov-rebuild", label: "Rebuild the provenance index", group: "Provenance", run: () => void rebuild() },
    { id: "prov-stale", label: "Show stale products", group: "Provenance", keywords: ["model", "outdated"], run: () => setVerdict("stale") },
    { id: "prov-unknown", label: "Show products with no model", group: "Provenance", run: () => setVerdict("unknown") },
  ]);

  const kindOptions = useMemo(() => [{ value: "", label: "All kinds" },
    ...Object.entries(s?.counts.kinds ?? {}).map(([k, n]) => ({ value: k, label: `${k} (${formatCount(n)})` }))], [s]);
  const columns = useMemo<DataColumn<ProvRow>[]>(() => [
    { id: "id", header: "Id", width: 96, cell: (r) => <code className="mono">{r.id}</code> },
    { id: "kind", header: "Kind", width: 150, cell: (r) => <span className="ops-small">{r.kind}</span> },
    { id: "label", header: "Label", cell: (r) => <span className="ops-ellipsis" title={r.path ?? r.label}>{r.label}</span> },
    { id: "verdict", header: "Model", width: 104, accessor: (r) => r.verdict ?? "",
      cell: (r) => <VerdictBadge verdict={r.verdict} /> },
    { id: "models", header: "Member", width: 110, accessor: (r) => r.models.map((m) => m.member ?? m.id).join(" ") || r.member || "",
      cell: (r) => <span className="mono ops-small">{r.models.map((m) => m.member ?? m.id).join(", ") || r.member || "—"}</span> },
    { id: "created_at", header: "Created", width: 104, accessor: (r) => r.created_at ?? "",
      cell: (r) => <span className="ops-dim ops-small" title={formatDateTime(r.created_at)}>{formatRelative(r.created_at)}</span> },
    { id: "git", header: "Commit", width: 84, cell: (r) => <code className="mono ops-small">{r.git ?? "—"}</code> },
    { id: "links", header: "Up / down", numeric: true, width: 84, accessor: (r) => r.n_upstream + r.n_downstream,
      cell: (r) => <span className="mono ops-small">{r.n_upstream} / {r.n_downstream}</span> },
    { id: "source", header: "Source", hidden: true },
  ], []);

  const d = records.data;
  return (
    <Page className="ops-page">
      {summary.error && !s && <Callout tone="bad" title="Could not index the provenance records">{summary.error.message}</Callout>}
      <div className="ops-kpis">
        <Kpi label="Records" value={s ? formatCount(s.total) : "…"} loading={summary.loading && !s}
          hint={s ? s.roots.map((r) => `${r.path}: ${formatCount(r.records)}`).join(" · ") : undefined} onClick={() => { setKind(""); setVerdict(""); }} />
        <Kpi label="Current" value={s ? formatCount(s.counts.verdicts.current) : "…"} tone="good" loading={summary.loading && !s}
          hint="SR products made by an active member" onClick={() => setVerdict("current")} />
        <Kpi label="Stale" value={s ? formatCount(s.counts.verdicts.stale) : "…"} tone={s?.counts.verdicts.stale ? "bad" : undefined}
          loading={summary.loading && !s} hint="Made by a model that is no longer active" onClick={() => setVerdict("stale")} />
        <Kpi label="No model" value={s ? formatCount(s.counts.verdicts.unknown) : "…"} loading={summary.loading && !s}
          hint="Legacy / un-stamped products (no model id)" onClick={() => setVerdict("unknown")} />
        <Kpi label="Active models" value={s ? s.current_models.length : "…"} loading={summary.loading && !s}
          hint="Active ensemble members with a provenance id" onClick={() => { setKind("checkpointartifact"); setVerdict(""); }} />
      </div>
      <div className="ops-bar" role="toolbar" aria-label="Search provenance">
        <Input size="sm" value={draft} onChange={setDraft} icon="search" clearable placeholder="id, kind, path, member, commit…"
          aria-label="Search records" style={{ flex: "1 1 220px", maxWidth: 360 }} />
        <Select size="sm" value={kind} onChange={setKind} options={kindOptions} aria-label="Kind" />
        <Segmented<VerdictFilter> size="sm" value={verdict} onChange={setVerdict} aria-label="Verdict"
          options={[{ value: "", label: "All" }, { value: "current", label: "Current" }, { value: "stale", label: "Stale" }, { value: "unknown", label: "No model" }]} />
        <Select size="sm" value={source} onChange={setSource} aria-label="Source"
          options={[{ value: "", label: "All sources" }, { value: "prov", label: "data/_prov" }, { value: "sidecar", label: "Sidecars" }, { value: "checkpoint", label: "Checkpoints" }]} />
        <span className="ops-spacer" />
        {s && <span className="ops-dim ops-small" title={`Built in ${s.build_seconds.toFixed(1)} s`}>indexed {formatRelative(s.built_at)}</span>}
        <IconButton size="sm" icon="reset" label="Rebuild the index" loading={rebuilding} onClick={() => void rebuild()} />
      </div>
      {s?.truncated && <Callout tone="warn">The scan stopped at its file cap: some sidecars are not indexed.</Callout>}
      <div className="ops-split">
        <div className="ops-split__main">
          {records.error && !d && <Callout tone="bad">{records.error.message}</Callout>}
          <DataTable rows={d?.records ?? []} columns={columns} rowKey={(r) => r.id} aria-label="Provenance records"
            loading={records.loading && !d} height={600} activeKey={id || null} onRowClick={(r) => setId(r.id)}
            exportName="provenance" urlKey="pr" filterPlaceholder="Filter these rows…"
            empty={q || kind || verdict || source ? "No record matches." : "No provenance record found locally."}
            toolbar={d ? <Badge size="sm">{d.total > d.records.length ? `${formatCount(d.records.length)} of ${formatCount(d.total)}` : formatCount(d.total)}</Badge> : undefined} />
        </div>
        {id && (
          <aside className="ops-split__side" aria-label={`Record ${id}`}>
            <Card>
              <CardHead title="Record" right={<div className="ops-row">
                <IconButton size="sm" icon="panelRight" label="Open in the inspector" onClick={() => openInspector({ kind: "prov", id })} />
                <IconButton size="sm" icon="close" label="Close" onClick={() => setId("")} />
              </div>} />
              <CardBody><ProvDetail key={id} id={id} onSelect={setId} /></CardBody>
            </Card>
          </aside>
        )}
      </div>
      {!s && summary.loading && <Skeleton lines={2} />}
      {s && !s.current_models.length && (
        <p className="ops-note">No active member carries a provenance id, so every product reads “no model”.</p>
      )}
    </Page>
  );
}
