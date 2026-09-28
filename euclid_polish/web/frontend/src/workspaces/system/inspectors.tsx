/* The System workspace's inspector kinds (lazy chunk, see ./register.ts):
 *   prov:<id8>       a provenance record — lineage, model verdict, JSON
 *   commit:<hash>    a local git commit — message, stat, patch
 *   root:<id>        one data root's disk usage (System › Storage) */
import { Link } from "react-router-dom";
import { useResource } from "../../api/query";
import { formatBytes, formatCount, formatPercent, formatRelative } from "../../format";
import { Button, CopyButton, DefList, EmptyState, Skeleton } from "../../ui";
import { SYSTEM_URL, type SystemInfo } from "./api";
import { CommitDetail } from "./git/CommitDetail";
import { ProvDetail } from "./provenance/ProvDetail";
import "./system.css";

export function ProvInspector({ id }: { id: string }) {
  return <ProvDetail id={id} />;
}

export function CommitInspector({ id }: { id: string }) {
  return <CommitDetail id={id} />;
}

export function RootInspector({ id }: { id: string }) {
  const sys = useResource<SystemInfo>(SYSTEM_URL, [], { ttl: 30_000 });
  const root = sys.data?.roots.items.find((r) => r.id === id) ?? null;
  if (sys.loading && !sys.data) return <Skeleton lines={3} />;
  if (!root) return <EmptyState compact icon="database" title="Not measured">{id}</EmptyState>;
  const total = sys.data?.roots.total_bytes ?? 0;
  return (
    <div className="sys-stack">
      <DefList dense items={[
        ["path", <span className="sys-row"><code className="mono">{root.path}</code><CopyButton value={root.path} label="Copy path" /></span>],
        ["size", formatBytes(root.bytes)],
        ["files", formatCount(root.files)],
        total > 0 ? ["share", formatPercent(root.bytes / total)] : null,
        ["measured", sys.data?.roots.computed_at ? formatRelative(sys.data.roots.computed_at) : "—"],
        !root.exists ? ["state", "does not exist"] : null,
      ]} />
      <Button asChild size="sm"><Link to="/files">Browse files in Files</Link></Button>
    </div>
  );
}
