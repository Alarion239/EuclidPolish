/* The Notebook inspector kind (lazy chunk, see ./register.ts):
 *   campaign:<dir>   a tracking campaign — backups, notebook, jobs, time travel */
import { CampaignDetail } from "./Archive";
import "./notebook.css";

export default function CampaignInspector({ id }: { id: string }) {
  return <CampaignDetail id={id} />;
}
