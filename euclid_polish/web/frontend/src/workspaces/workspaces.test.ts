/* Every workspace folder honours the C8 contract: `index.tsx` default-exports
 * the workspace and declares exactly the manifest's tabs, and every tab
 * module loads to a component — so every old page is reachable at its new
 * URL (the legacy adapters) and no manifest tab is left without a module. */
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

/* Which legacy page each tab adapts (plan WP-F T8): a page listed here must
   be what the tab module exports. */
const ADAPTERS: [string, string, () => Promise<{ default: unknown }>][] = [
  ["sky/atlas", "JwstEuclid", () => import("../pages/JwstEuclid")],
  ["sky/results", "Inference", () => import("../pages/Inference")],
  ["sky/catalog-eval", "Evaluation", () => import("../pages/Evaluation")],
  ["ensemble/overview", "Ensemble", () => import("../pages/Ensemble")],
  ["ensemble/disagreement", "Ensemble", () => import("../pages/Ensemble")],
  ["ensemble/train", "TrainMembers", () => import("../pages/TrainMembers")],
  ["realism/noise", "Noise", () => import("../pages/Noise")],
  ["realism/galaxies", "GalaxyDistributions", () => import("../pages/GalaxyDistributions")],
  ["realism/stars", "StarDistribution", () => import("../pages/StarDistribution")],
  ["realism/pixels", "PopulationComparison", () => import("../pages/PopulationComparison")],
  ["realism/visual", "SyntheticReal", () => import("../pages/SyntheticReal")],
  ["data/records", "Sky", () => import("../pages/Sky")],
  ["data/catalog", "Catalog", () => import("../pages/Catalog")],
  ["data/cutouts", "Cutouts", () => import("../pages/Cutouts")],
  ["data/psfs", "Psfs", () => import("../pages/Psfs")],
  ["data/tng", "Tng", () => import("../pages/Tng")],
  ["figures/plates", "Visualization", () => import("../pages/Visualization")],
  ["ops/fasrc", "Fasrc", () => import("../pages/Fasrc")],
  ["ops/tracking", "Tracking", () => import("../pages/Tracking")],
  ["ops/git", "Git", () => import("../pages/Git")],
  ["settings/config", "Config", () => import("../pages/Config")],
];

describe("legacy adapters", () => {
  for (const [path, page, loadPage] of ADAPTERS) {
    it(`${path} renders pages/${page}`, async () => {
      const [wsId, tab] = path.split("/");
      const mod = (await workspaceComponents[wsId]()) as WorkspaceIndex;
      const tabMod = await mod.TABS![tab].load();
      expect(tabMod.default).toBe((await loadPage()).default);
    }, 30_000);
  }
});
