/* One provenance record: what it is, the model(s) behind it, the verdict of
 * its Loop stage (the staleness service Home's Loop reads, GET
 * /api/system/loop — not a per-record model check, which contradicted
 * Home), direct upstream/downstream, the transitive
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
import { LOOP_URL, provRecordUrl, type LoopResp, type ProvRecordResp, type ProvRef } from "../api";
import { stageOfKind, type LoopStageSlice } from "../model";

export const STAGE_TONE: Record<string, Tone> = { current: "good", stale: "warn", blocked: "bad", unknown: "neutral", loading: "neutral" };

/** The Loop's verdicts (shared with Home: one request, one cache entry). */
export const useLoop = () => useResource<LoopResp>(LOOP_URL, [], { ttl: 60_000 });

/** "Real SR · stale" with the stage's reason as the tooltip; "—" for a kind
 *  no Loop stage covers. */
export function StageBadge({ stage }: { stage: LoopStageSlice | null }) {
  if (!stage) return <span className="sys-dim">—</span>;
  const state = stage.state === "loading" ? "checking" : stage.state;
  return (
    <Tooltip content={`${stage.label}: ${stage.reason}${stage.detail ? `. ${stage.detail}` : ""}`}>
      <span tabIndex={0}><Badge size="sm" tone={STAGE_TONE[stage.state] ?? "neutral"}>{stage.label} · {state}</Badge></span>
    </Tooltip>
  );
}

function RefList({ refs, onSelect, depth = false }: { refs: ProvRef[]; onSelect?: (id: string) => void; depth?: boolean }) {
  if (!refs.length) return <span className="sys-dim sys-small">none</span>;
  return (
    <ul className="sys-lineage">
      {refs.map((r) => (
        <li key={`${r.role}:${r.id}`}>
          {depth && <span className="sys-lineage__depth">{r.depth ?? ""}</span>}
          <button type="button" className="sys-linkbtn mono" disabled={!r.exists}
            onClick={() => (onSelect ? onSelect(r.id) : openInspector({ kind: "prov", id: r.id }))}>{r.id}</button>
          <span className="sys-dim">{r.kind ?? "not in the local store"}</span>
          {r.member && <Badge size="sm">{r.member}</Badge>}
          {r.role !== "child" && !depth && <span className="sys-dim sys-small">{r.role.replace("_", " ")}</span>}
          {r.label && r.label !== r.kind && <span className="sys-ellipsis sys-small">{r.label}</span>}
        </li>
      ))}
    </ul>
  );
}

export function ProvDetail({ id, onSelect }: { id: string; onSelect?: (id: string) => void }) {
  const res = useResource<ProvRecordResp>(provRecordUrl(id), [id], { ttl: 60_000 });
  const loop = useLoop();
  const d = res.data;
  if (res.loading && !d) return <Skeleton lines={6} />;
  if (res.error && !d) return <Callout tone={res.error.status === 404 ? "warn" : "bad"} title={`Record ${id}`}>{res.error.message}</Callout>;
  if (!d) return null;
  const e = d.entry;
  const stage = stageOfKind(e.kind, loop.data?.stages);
  return (
    <div className="sys-stack">
      <div className="sys-row">
        <code className="mono">{e.id}</code><CopyButton value={e.id} label="Copy the id" />
        <Badge size="sm">{e.kind}</Badge>
        <StageBadge stage={stage} />
      </div>
      <DefList dense items={[
        ["label", <span className="sys-ellipsis" title={e.label}>{e.label}</span>],
        ["created", formatDateTime(e.created_at)],
        e.git ? ["commit", <span><code className="mono">{e.git}</code>{e.dirty ? <Badge size="sm" tone="warn">dirty</Badge> : null}</span>] : null,
        e.status ? ["status", e.status] : null,
        e.config_type ? ["config", <code className="mono">{e.config_type}</code>] : null,
        e.seed != null ? ["seed", <code className="mono">{e.seed}</code>] : null,
        e.path ? ["path", <span className="mono sys-small sys-break">{e.path}</span>] : null,
        e.ra != null && e.dec != null ? ["position", formatRaDec(e.ra, e.dec, { mode: "both" })] : null,
        ["sidecar", <span className="mono sys-small sys-break">{e.file}</span>],
        e.verdict ? ["model id", d.models.length
          ? <span className="mono sys-small">{d.models.map((m) => `${m.id}${m.member ? ` (${m.member})` : ""}`).join(", ")}</span>
          : <span className="sys-dim">none recorded</span>] : null,
      ]} />
      {(d.inspect_path || (e.ra != null && e.dec != null)) && (
        <div className="sys-row">
          {d.inspect_path && <Link className="sys-small" to={`/files?fits=${encodeURIComponent(d.inspect_path)}`}>Open the FITS in Files</Link>}
          {e.ra != null && e.dec != null && <Link className="sys-small" to={`/sky/atlas?ra=${e.ra}&dec=${e.dec}`}>Show on the sky</Link>}
        </div>
      )}
      {stage && stage.state !== "current" && stage.state !== "loading" && (
        <Callout tone={stage.state === "blocked" ? "bad" : stage.state === "stale" ? "warn" : "info"} title={`${stage.label}: ${stage.reason}`}
          action={<Link className="sys-small" to={stage.to}>Open the tab that fixes it</Link>}>
          The verdict of the {stage.label} stage of the Loop (Home), which this {e.kind} belongs to.
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
        {d.ancestors.total > d.ancestors.items.length && <p className="sys-note">First {d.ancestors.items.length} shown.</p>}
      </Section>
      <Section title={`Descendants (${d.descendants.total})`} collapsible defaultOpen={false}>
        <RefList refs={d.descendants.items} onSelect={onSelect} depth />
        {d.descendants.total > d.descendants.items.length && <p className="sys-note">First {d.descendants.items.length} shown.</p>}
      </Section>
      <Section title="Stored record" collapsible defaultOpen={false}>
        {d.record ? <JsonTree data={d.record} expandDepth={1} /> : <span className="sys-dim">The sidecar could not be read.</span>}
      </Section>
    </div>
  );
}
