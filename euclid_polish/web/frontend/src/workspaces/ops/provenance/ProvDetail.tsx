/* One provenance record: what it is, the model(s) behind it and its
 * current/stale verdict, direct upstream/downstream, the transitive
 * ancestors/descendants (with hop depth) and the stored JSON. The body of
 * the `prov:<id>` inspector and of the Provenance tab's detail pane.
 * `onSelect` walks to another record (the tab keeps it in the URL); without
 * it the ids open in the inspector. */
import { Link } from "react-router-dom";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { formatDateTime, formatRaDec } from "../../../format";
import {
  Badge, Callout, CopyButton, DefList, JsonTree, Section, Skeleton, Tooltip, type Tone,
} from "../../../ui";
import { provRecordUrl, type ProvRecordResp, type ProvRef, type ProvVerdict } from "../api";

export const VERDICT_TONE: Record<ProvVerdict, Tone> = { current: "good", stale: "bad", unknown: "neutral" };
export const VERDICT_HINT: Record<ProvVerdict, string> = {
  current: "Made by an active ensemble member.",
  stale: "Made by a model that is no longer an active member.",
  unknown: "No model id was recorded (a legacy or un-stamped product).",
};

export function VerdictBadge({ verdict }: { verdict: ProvVerdict | null }) {
  if (!verdict) return <span className="ops-dim">—</span>;
  return (
    <Tooltip content={VERDICT_HINT[verdict]}>
      <span tabIndex={0}><Badge size="sm" tone={VERDICT_TONE[verdict]}>{verdict}</Badge></span>
    </Tooltip>
  );
}

function RefList({ refs, onSelect, depth = false }: { refs: ProvRef[]; onSelect?: (id: string) => void; depth?: boolean }) {
  if (!refs.length) return <span className="ops-dim ops-small">none</span>;
  return (
    <ul className="ops-lineage">
      {refs.map((r) => (
        <li key={`${r.role}:${r.id}`}>
          {depth && <span className="ops-lineage__depth">{r.depth ?? ""}</span>}
          <button type="button" className="ops-linkbtn mono" disabled={!r.exists}
            onClick={() => (onSelect ? onSelect(r.id) : openInspector({ kind: "prov", id: r.id }))}>{r.id}</button>
          <span className="ops-dim">{r.kind ?? "not in the local store"}</span>
          {r.member && <Badge size="sm">{r.member}</Badge>}
          {r.role !== "child" && !depth && <span className="ops-dim ops-small">{r.role.replace("_", " ")}</span>}
          {r.label && r.label !== r.kind && <span className="ops-ellipsis ops-small">{r.label}</span>}
        </li>
      ))}
    </ul>
  );
}

export function ProvDetail({ id, onSelect }: { id: string; onSelect?: (id: string) => void }) {
  const res = useResource<ProvRecordResp>(provRecordUrl(id), [id], { ttl: 60_000 });
  const d = res.data;
  if (res.loading && !d) return <Skeleton lines={6} />;
  if (res.error && !d) return <Callout tone={res.error.status === 404 ? "warn" : "bad"} title={`Record ${id}`}>{res.error.message}</Callout>;
  if (!d) return null;
  const e = d.entry;
  return (
    <div className="ops-stack">
      <div className="ops-row">
        <code className="mono">{e.id}</code><CopyButton value={e.id} label="Copy the id" />
        <Badge size="sm">{e.kind}</Badge>
        <VerdictBadge verdict={e.verdict} />
      </div>
      <DefList dense items={[
        ["label", <span className="ops-ellipsis" title={e.label}>{e.label}</span>],
        ["created", formatDateTime(e.created_at)],
        e.git ? ["commit", <span><code className="mono">{e.git}</code>{e.dirty ? <Badge size="sm" tone="warn">dirty</Badge> : null}</span>] : null,
        e.status ? ["status", e.status] : null,
        e.config_type ? ["config", <code className="mono">{e.config_type}</code>] : null,
        e.seed != null ? ["seed", <code className="mono">{e.seed}</code>] : null,
        e.path ? ["path", <span className="mono ops-small ops-break">{e.path}</span>] : null,
        e.ra != null && e.dec != null ? ["position", formatRaDec(e.ra, e.dec, { mode: "both" })] : null,
        ["sidecar", <span className="mono ops-small ops-break">{e.file}</span>],
      ]} />
      {(d.inspect_path || (e.ra != null && e.dec != null)) && (
        <div className="ops-row">
          {d.inspect_path && <Link className="ops-small" to={`/inspect?fits=${encodeURIComponent(d.inspect_path)}`}>Open the FITS in Inspect</Link>}
          {e.ra != null && e.dec != null && <Link className="ops-small" to={`/sky/atlas?ra=${e.ra}&dec=${e.dec}`}>Show on the sky</Link>}
        </div>
      )}
      {e.verdict && (
        <Callout tone={e.verdict === "stale" ? "warn" : e.verdict === "current" ? "good" : "info"} title={`Model check: ${e.verdict}`}>
          {d.models.length ? <>Model{d.models.length > 1 ? "s" : ""}: {d.models.map((m) => `${m.id}${m.member ? ` (${m.member})` : ""}`).join(", ")}. </> : null}
          {VERDICT_HINT[e.verdict]} {d.current_models.length} active members carry a provenance id.
        </Callout>
      )}
      <Section title={`Upstream (${d.upstream.length})`} collapsible defaultOpen>
        <RefList refs={d.upstream} onSelect={onSelect} />
      </Section>
      <Section title={`Downstream (${d.downstream.length})`} collapsible defaultOpen={d.downstream.length > 0 && d.downstream.length <= 20}>
        <RefList refs={d.downstream} onSelect={onSelect} />
      </Section>
      <Section title={`Ancestors (${d.ancestors.total})`} collapsible defaultOpen={false}>
        <RefList refs={d.ancestors.items} onSelect={onSelect} depth />
        {d.ancestors.total > d.ancestors.items.length && <p className="ops-note">First {d.ancestors.items.length} shown.</p>}
      </Section>
      <Section title={`Descendants (${d.descendants.total})`} collapsible defaultOpen={false}>
        <RefList refs={d.descendants.items} onSelect={onSelect} depth />
        {d.descendants.total > d.descendants.items.length && <p className="ops-note">First {d.descendants.items.length} shown.</p>}
      </Section>
      <Section title="Stored record" collapsible defaultOpen={false}>
        {d.record ? <JsonTree data={d.record} expandDepth={1} /> : <span className="ops-dim">The sidecar could not be read.</span>}
      </Section>
    </div>
  );
}
