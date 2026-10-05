/* One commit (`git show`): message, author, --stat and the patch in the diff
 * viewer. The body of the `commit:<hash>` inspector, which System › Code's
 * history rows open. */
import { useResource } from "../../../api/query";
import { formatDateTime } from "../../../format";
import { Callout, CopyButton, DefList, LogTail, Section, Skeleton } from "../../../ui";
import { gitShowUrl, type GitShow } from "../api";
import { DiffView } from "./DiffView";

export function CommitDetail({ id }: { id: string }) {
  const res = useResource<GitShow>(gitShowUrl(id), [id], { ttl: 10 * 60_000 });
  const d = res.data;
  if (res.loading && !d) return <Skeleton lines={6} />;
  if (res.error && !d) return <Callout tone="bad" title="Could not read the commit">{res.error.message}</Callout>;
  if (!d) return null;
  return (
    <div className="sys-stack">
      <strong>{d.subject}</strong>
      {d.body && <p className="sys-note sys-pre">{d.body}</p>}
      <DefList dense items={[
        ["commit", <span className="sys-row"><code className="mono">{d.full.slice(0, 12)}</code><CopyButton value={d.full} label="Copy the hash" /></span>],
        ["author", `${d.author} <${d.email}>`],
        ["date", formatDateTime(d.date)],
      ]} />
      {d.stat && <Section title="Files" collapsible defaultOpen={false}><LogTail text={d.stat} style={{ maxHeight: 220 }} /></Section>}
      <DiffView text={d.patch} empty="No textual changes." />
      {d.truncated && <p className="sys-note">The patch is truncated.</p>}
    </div>
  );
}
