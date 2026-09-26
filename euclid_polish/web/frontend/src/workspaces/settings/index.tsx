/* Settings workspace (spec §8.8). Phase 1: config → the old Config page, a
   minimal connections tab, and the new appearance and about pages;
   W-Settings+Home completes it in phase 3. */
import { Workspace, defineTabs } from "../../app/workspace";

export const TABS = defineTabs("settings", {
  config: { load: () => import("./tabs/Config") },
  connections: { load: () => import("./tabs/Connections") },
  appearance: { load: () => import("./tabs/Appearance") },
  about: { load: () => import("./tabs/About") },
});

export default function SettingsWorkspace() {
  return <Workspace id="settings" tabs={TABS} />;
}
