/* The provenance of a FITS file: the PROVID stamp in its primary header, the
   co-located sidecar records describing it (the current one first) and the
   records of its producing run and parents. */
import { useResource } from "../../api/query";
import { formatDateTime } from "../../format";
import { Badge, Callout, CopyButton, DefList, EmptyState, JsonTree, Section, Skeleton } from "../../ui";
import { provenanceUrl, type Provenance, type RelatedRecord, type Sidecar } from "./api";

function Id({ id }: { id: string | null | undefined }) {
  if (!id) return <span className="insp-dim">—</span>;
  return <span className="insp-id"><code className="mono">{id}</code><CopyButton value={id} label={`Copy ${id}`} /></span>;
}

function gitOf(record: Record<string, unknown>): string {
  const git = record.git as { short?: string; dirty?: boolean } | undefined;
  return git?.short ? `${git.short}${git.dirty ? " (dirty)" : ""}` : "—";
}

function SidecarBody({ s }: { s: Sidecar }) {
  return (
    <>
      <DefList dense items={[
        ["file", <code className="mono insp-break">{s.file}</code>],
        ["created", s.record.created_at ? formatDateTime(String(s.record.created_at)) : "—"],
        ["git", <code className="mono">{gitOf(s.record)}</code>],
      ]} />
      <JsonTree data={s.record} expandDepth={1} />
    </>
  );
}

function RelatedBody({ r }: { r: RelatedRecord }) {
  if (!r.record) return <p className="insp-dim">Not found in data/_prov, next to the file, or in a checkpoint.</p>;
  return (
    <>
      <DefList dense items={[
        ["file", r.file ? <code className="mono insp-break">{r.file}</code> : "—"],
        r.checkpoint ? ["checkpoint", <code className="mono insp-break">{r.checkpoint}</code>] : null,
      ]} />
      <JsonTree data={r.record} expandDepth={1} />
    </>
  );
}

export function ProvenancePanel({ fits }: { fits: string }) {
  const res = useResource<Provenance>(provenanceUrl(fits), [], { ttl: 60_000 });
  const p = res.data;
  if (res.loading) return <Skeleton lines={5} />;
  if (res.error || !p) return <Callout tone="bad" title="Could not read the provenance">{res.error?.message ?? "No data."}</Callout>;
  if (!p.stamp && !p.sidecars.length) {
    return (
      <EmptyState compact icon="info" title="No provenance recorded">
        The primary header has no PROVID card and no sidecar next to it names this file.
      </EmptyState>
    );
  }
  const current = p.sidecars.find((s) => s.current) ?? null;
  const earlier = p.sidecars.filter((s) => !s.current);
  return (
    <div className="insp-prov">
      <DefList items={[
        ["PROVID", <Id id={p.stamp?.id} />],
        ["produced by", <Id id={p.stamp?.produced_by} />],
        ["parents", p.stamp?.parents.length
          ? <span className="insp-prov__ids">{p.stamp.parents.map((id) => <Id key={id} id={id} />)}</span>
          : <span className="insp-dim">none</span>],
        ["record", current ? <Badge tone="good" dot>{current.kind}</Badge>
          : <Badge tone="warn" dot>{p.stamp ? "stamp only (no sidecar)" : "sidecars without a stamp"}</Badge>],
        earlier.length ? ["earlier runs", <Badge tone="neutral">{earlier.length} superseded record{earlier.length > 1 ? "s" : ""}</Badge>] : null,
      ]} />
      {current && (
        <Section title="This file's record" sub={current.id}><SidecarBody s={current} /></Section>
      )}
      {p.related.map((r) => (
        <Section key={r.id} collapsible defaultOpen={false}
          title={<>{r.role === "produced_by" ? "Produced by" : "Parent"} · <code className="mono">{r.id}</code></>}
          sub={r.kind ?? "not found"}>
          <RelatedBody r={r} />
        </Section>
      ))}
      {earlier.length > 0 && (
        <Section collapsible defaultOpen={false} title="Earlier runs of this file" sub={`${earlier.length}`}>
          {earlier.map((s) => (
            <Section key={s.id} collapsible defaultOpen={false} title={<code className="mono">{s.id}</code>}
              sub={s.record.created_at ? formatDateTime(String(s.record.created_at)) : s.kind}>
              <SidecarBody s={s} />
            </Section>
          ))}
        </Section>
      )}
    </div>
  );
}
