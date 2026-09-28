/* System › Lineage (`/system/lineage`, was Ops › Provenance): a lookup
 * tool over data/_prov, the sidecars next to the data and the checkpoint
 * stamps. Search a product and see its lineage in the side card. Search the
 * records (server-side, ANDed tokens), filter by kind / verdict / source, and
 * walk a record's ancestors and descendants.
 *
 * The verdicts come from the staleness service Home's Loop reads (GET
 * /api/system/loop): each record takes the verdict of its Loop stage
 * (generation runs → Records, checkpoints → Members, real-sky SR runs and
 * cutouts → Real SR; model.ts stageOfKind), so Home and Lineage never
 * disagree. The verdict control counts records per verdict, and filtering by
 * a verdict narrows the server's kind filter to that verdict's kinds. The
 * per-record model check of the index is not a verdict here: while most
 * records carry no model id, one callout says so (no stat tiles).
 * URL: `q`, `kind`, `verdict` (current | stale | blocked | unknown),
 * `source`, `id` (the open record). */
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { apiPost } from "../../../api/client";
import { invalidate, setResourceData, useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import { formatCount, formatDateTime, formatRelative } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Callout, Caption, Card, CardBody, CardHead, DataTable, IconButton, Input, Page, Segmented,
  Select, Skeleton, Toolbar, ToolbarSpacer, ToolbarText, toast, type DataColumn,
} from "../../../ui";
import {
  LINEAGE_VERDICTS, lineageCallout, lineageVerdicts, stageOfKind, verdictKindFilter, type LineageVerdict,
} from "../model";
import {
  PROV_SUMMARY_URL, provRecordsUrl, type ProvRecordsResp, type ProvRow, type ProvSummary,
} from "../api";
import { ProvDetail, StageBadge, useLoop } from "../provenance/ProvDetail";
import "../system.css";

type VerdictFilter = "" | LineageVerdict;
const VERDICT_LABEL: Record<LineageVerdict, string> = { current: "Current", stale: "Stale", blocked: "Blocked", unknown: "Unknown" };

export default function Provenance() {
  const summary = useResource<ProvSummary>(PROV_SUMMARY_URL, [], { ttl: 60_000 });
  const [q, setQ] = useUrlState("q", "");
  const [kind, setKind] = useUrlState("kind", "");
  const [verdict, setVerdict] = useUrlState<VerdictFilter>("verdict", "",
    { parse: (r) => ((LINEAGE_VERDICTS as readonly string[]).includes(r) ? r as VerdictFilter : undefined) });
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
  const loop = useLoop();
  const [rebuilding, setRebuilding] = useState(false);
  const s = summary.data;
  const stages = loop.data?.stages;
  const verdicts = useMemo(() => lineageVerdicts(s?.counts.kinds ?? {}, stages), [s, stages]);
  // A verdict narrows the server's kind filter to that verdict's kinds; null:
  // no record can match, so nothing is requested. Until the Loop answers, a
  // verdict filter waits rather than showing every record.
  const kindParam = verdict && !stages ? null : verdictKindFilter(kind, verdict, verdicts);
  const records = useResource<ProvRecordsResp>(kindParam == null ? null : provRecordsUrl({ q, kind: kindParam, source }),
    [q, kindParam, source], { ttl: 60_000 });

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
    { id: "prov-rebuild", label: "Rebuild the lineage index", group: "Lineage", run: () => void rebuild() },
    { id: "prov-stale", label: "Show stale products", group: "Lineage", keywords: ["model", "outdated"], run: () => setVerdict("stale") },
  ]);

  // Current and Stale always; Blocked and Unknown only when a record has them.
  const verdictOptions = useMemo(() => [{ value: "" as VerdictFilter, label: "All" },
    ...LINEAGE_VERDICTS.filter((v) => v === "current" || v === "stale" || verdicts.counts[v] > 0 || verdict === v).map((v) => ({
      value: v as VerdictFilter, label: s && stages ? `${VERDICT_LABEL[v]} · ${formatCount(verdicts.counts[v])}` : VERDICT_LABEL[v],
    }))], [s, stages, verdicts, verdict]);
  // The stages the indexed kinds belong to, in Loop order, for the caption.
  const indexedStages = useMemo(() => (stages ?? []).filter((st) =>
    Object.keys(s?.counts.kinds ?? {}).some((k) => stageOfKind(k, stages)?.id === st.id)), [s, stages]);
  // The index's own model check compares a product's model id with the active
  // members; while most records carry none, each takes its stage's verdict.
  const callout = lineageCallout(s);
  const kindOptions = useMemo(() => [{ value: "", label: "All kinds" },
    ...Object.entries(s?.counts.kinds ?? {}).map(([k, n]) => ({ value: k, label: `${k} (${formatCount(n)})` }))], [s]);
  const columns = useMemo<DataColumn<ProvRow>[]>(() => [
    { id: "id", header: "Id", width: 96, cell: (r) => <code className="mono">{r.id}</code> },
    { id: "kind", header: "Kind", width: 150, cell: (r) => <span className="sys-small">{r.kind}</span> },
    { id: "label", header: "Label", cell: (r) => <span className="sys-ellipsis" title={r.path ?? r.label}>{r.label}</span> },
    { id: "verdict", header: "Stage verdict", width: 150, accessor: (r) => stageOfKind(r.kind, stages)?.state ?? "",
      cell: (r) => <StageBadge stage={stageOfKind(r.kind, stages)} /> },
    { id: "models", header: "Member", width: 110, priority: 1, accessor: (r) => r.models.map((m) => m.member ?? m.id).join(" ") || r.member || "",
      cell: (r) => <span className="mono sys-small">{r.models.map((m) => m.member ?? m.id).join(", ") || r.member || "—"}</span> },
    { id: "created_at", header: "Created", width: 104, accessor: (r) => r.created_at ?? "",
      cell: (r) => <span className="sys-dim sys-small" title={formatDateTime(r.created_at)}>{formatRelative(r.created_at)}</span> },
    { id: "git", header: "Commit", width: 84, priority: 2, cell: (r) => <code className="mono sys-small">{r.git ?? "—"}</code> },
    { id: "links", header: "Up / down", numeric: true, width: 84, priority: 3, accessor: (r) => r.n_upstream + r.n_downstream,
      cell: (r) => <span className="mono sys-small">{r.n_upstream} / {r.n_downstream}</span> },
    { id: "source", header: "Source", hidden: true },
  ], [stages]);

  const d = records.data;
  return (
    <Page className="sys-page">
      {summary.error && !s && <Callout tone="bad" title="Could not index the provenance records">{summary.error.message}</Callout>}
      {callout && <div className="sys-lead"><Callout tone="info" dense>{callout}</Callout></div>}
      {loop.error && !loop.data && (
        <Callout tone="warn" title="The staleness service did not answer">
          The verdicts come from the Loop (GET /api/system/loop); without it no record has one. {loop.error.message}
        </Callout>
      )}
      <Toolbar label="Search lineage">
        <Input size="sm" value={draft} onChange={setDraft} icon="search" clearable placeholder="id, kind, path, member, commit…"
          aria-label="Search records" style={{ flex: "1 1 220px", maxWidth: 360 }} />
        <Select size="sm" value={kind} onChange={setKind} options={kindOptions} aria-label="Kind" />
        <Segmented<VerdictFilter> size="sm" value={verdict} onChange={setVerdict} aria-label="Verdict" options={verdictOptions} />
        <Select size="sm" value={source} onChange={setSource} aria-label="Source"
          options={[{ value: "", label: "All sources" }, { value: "prov", label: "data/_prov" }, { value: "sidecar", label: "Sidecars" }, { value: "checkpoint", label: "Checkpoints" }]} />
        <ToolbarSpacer />
        {s && <ToolbarText><span title={`Built in ${s.build_seconds.toFixed(1)} s`}>indexed {formatRelative(s.built_at)}</span></ToolbarText>}
        <IconButton size="sm" icon="reset" label="Rebuild the index" loading={rebuilding} onClick={() => void rebuild()} />
      </Toolbar>
      {indexedStages.length > 0 && (
        <Caption>
          Verdicts are the Loop's, as on Home, one per stage:{" "}
          {indexedStages.map((st, i) => (
            <span key={st.id}>{i ? " · " : ""}<Link to={st.to}>{st.label}</Link> {st.state === "loading" ? "checking" : st.state}
              {st.state !== "current" && st.reason ? ` (${st.reason})` : ""}</span>
          ))}
          {loop.data?.computed_at ? `; checked ${formatRelative(loop.data.computed_at)}.` : "."}
        </Caption>
      )}
      {s?.truncated && <Callout tone="warn">The scan stopped at its file cap: some sidecars are not indexed.</Callout>}
      <div className="sys-split">
        <div className="sys-split__main">
          {records.error && !d && <Callout tone="bad">{records.error.message}</Callout>}
          <DataTable rows={d?.records ?? []} columns={columns} rowKey={(r) => r.id} aria-label="Provenance records"
            loading={(records.loading && !d) || (!!verdict && !stages && !loop.error)} height={600} activeKey={id || null} onRowClick={(r) => setId(r.id)}
            exportName="provenance" urlKey="pr" filterPlaceholder="Filter these rows…"
            empty={q || kind || verdict || source ? "No record matches." : "No provenance record found locally."}
            countText={d && d.total > d.records.length
              ? `showing ${formatCount(d.records.length)} of ${formatCount(d.total)}` : undefined} />
        </div>
        {id && (
          <aside className="sys-split__side" aria-label={`Record ${id}`}>
            <Card>
              <CardHead title="Record" right={<div className="sys-row">
                <IconButton size="sm" icon="panelRight" label="Open in the inspector" onClick={() => openInspector({ kind: "prov", id })} />
                <IconButton size="sm" icon="close" label="Close" onClick={() => setId("")} />
              </div>} />
              <CardBody><ProvDetail key={id} id={id} onSelect={setId} /></CardBody>
            </Card>
          </aside>
        )}
      </div>
      {!s && summary.loading && <Skeleton lines={2} />}
      {s && !s.current_models.length && !callout && (
        <p className="sys-note">No active member carries a provenance id, so every product reads “no model”.</p>
      )}
    </Page>
  );
}
