/* The Ops workspace's inspector kinds (lazy chunk, see ./register.ts):
 *   prov:<id8>       a provenance record — lineage, model verdict, JSON
 *   campaign:<dir>   a tracking campaign — backups, notebook, jobs, time travel
 *   commit:<hash>    a local git commit — message, stat, patch */
import { CommitDetail } from "./git/CommitDetail";
import { ProvDetail } from "./provenance/ProvDetail";
import { CampaignDetail } from "./tracking/Archive";
import "./ops.css";

export function ProvInspector({ id }: { id: string }) {
  return <ProvDetail id={id} />;
}

export function CampaignInspector({ id }: { id: string }) {
  return <CampaignDetail id={id} />;
}

export function CommitInspector({ id }: { id: string }) {
  return <CommitDetail id={id} />;
}
