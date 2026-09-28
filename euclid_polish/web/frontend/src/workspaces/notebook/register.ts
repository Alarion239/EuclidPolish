/* Registers the Notebook inspector kind `campaign` (`campaign:<dir>`, a
   tracking campaign: backups, notebook, jobs, time travel). The component is
   a lazy chunk, so importing this module is cheap: the workspace imports it,
   and the shell does too (app/Shell.tsx) so any page can open a
   campaign before Notebook was visited. */
import { lazy } from "react";
import { registerInspector } from "../../app/inspector";

const CampaignInspector = lazy(() => import("./inspectors"));

export const unregisterNotebookInspectors = [
  registerInspector("campaign", CampaignInspector, { title: (id) => `Campaign ${id}` }),
];
