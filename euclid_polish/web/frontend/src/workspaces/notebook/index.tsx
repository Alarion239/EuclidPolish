/* Notebook workspace, `/notebook/<tab>` (console regrouping): what we tried
   and concluded, and how to get the exact state back. Tabs:
     log        the active campaign's lab notebook (campaign bar, a new
                entry — prefilled from a page's "Log to notebook" — and the
                notebook itself)
     backups    model, FITS and image backups and the archived campaigns,
                each with ⏱ time travel (?show=)
     sandboxes  the running time-travel sandbox servers
   The `campaign` inspector kind is registered by ./register.ts (imported
   here, and by the shell). */
import { Workspace, defineTabs } from "../../app/workspace";
import "./register";

export const TABS = defineTabs("notebook", {
  log: { load: () => import("./tabs/Log") },
  backups: { load: () => import("./tabs/Backups") },
  sandboxes: { load: () => import("./tabs/Sandboxes") },
});

export default function NotebookWorkspace() {
  return <Workspace id="notebook" tabs={TABS} />;
}
