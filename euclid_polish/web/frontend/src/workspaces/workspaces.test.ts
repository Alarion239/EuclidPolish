/* Every workspace folder honours the C8 contract: `index.tsx` default-exports
 * the workspace and declares exactly the manifest's tabs, and every tab
 * module loads to a component — so every old page is reachable at its new
 * URL and no manifest tab is left without a module. */
import { describe, expect, it } from "vitest";
import { MANIFEST } from "../app/manifest";
import { workspaceComponents } from "../app/routes";
import type { WorkspaceTabDefs } from "../app/workspace";

type WorkspaceIndex = { default: unknown; TABS?: WorkspaceTabDefs };

describe("workspace folders", () => {
  for (const ws of MANIFEST.workspaces) {
    it(`${ws.id}: default export + exactly the manifest tabs, each loading a component`, async () => {
      const mod = (await workspaceComponents[ws.id]()) as WorkspaceIndex;
      expect(typeof mod.default).toBe("function");
      const tabs = mod.TABS ?? {};
      expect(Object.keys(tabs).sort()).toEqual([...ws.tabs].sort());
      for (const [tab, def] of Object.entries(tabs)) {
        const loaded = await def.load();
        expect(typeof loaded.default, `${ws.id}/${tab}`).toBe("function");
      }
    }, 30_000);
  }
});

/* The phase-1 legacy adapters (tabs that re-exported a pre-rework
   `src/pages/<X>` page) are all replaced: every tab is a workspace module. */
