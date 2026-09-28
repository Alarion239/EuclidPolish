import { describe, expect, it } from "vitest";
import { ICON_NAMES } from "../ui/icons";
import { MANIFEST, isPagePath } from "./manifest";
import {
  WORKSPACE_META, allPages, describePath, humanize, landingPath, pageHeading, pagePath, pageTitle, tabLabel,
} from "./nav";

describe("nav metadata ↔ route manifest", () => {
  it("has an entry for every manifest workspace and nothing else", () => {
    expect(Object.keys(WORKSPACE_META).sort()).toEqual(MANIFEST.workspaces.map((w) => w.id).sort());
  });

  it("labels exactly the manifest tabs of each workspace", () => {
    for (const ws of MANIFEST.workspaces) {
      expect(Object.keys(WORKSPACE_META[ws.id].tabs).sort(), ws.id).toEqual([...ws.tabs].sort());
    }
  });

  it("uses real icons and unique go-keys", () => {
    const keys = Object.values(WORKSPACE_META).map((m) => m.goKey);
    expect(new Set(keys).size).toBe(keys.length);
    for (const m of Object.values(WORKSPACE_META)) expect(ICON_NAMES).toContain(m.icon);
  });

  it("gives every tab a label that is unique across the rail, in sentence case", () => {
    const labels = MANIFEST.workspaces.flatMap((ws) => ws.tabs.map((t) => tabLabel(ws.id, t)));
    expect(new Set(labels).size).toBe(labels.length);
    for (const label of labels) expect(label, label).toMatch(/^[A-Z][a-z]*( [a-z]+)*$|^PSF$/);
    const ws = MANIFEST.workspaces.map((w) => w.label);
    expect(new Set([...ws, ...labels]).size).toBe(ws.length + labels.length);
  });
});

describe("paths", () => {
  it("lands on the default tab with default params", () => {
    expect(landingPath("home")).toBe("/");
    expect(landingPath("synthetic")).toBe("/synthetic/status");
    expect(landingPath("models")).toBe("/models/starfull/leaderboard");
    expect(landingPath("sky")).toBe("/sky/atlas");
    expect(landingPath("figures")).toBe("/figures/plates");
    expect(landingPath("files")).toBe("/files");
    expect(landingPath("runs")).toBe("/runs/live");
    expect(landingPath("notebook")).toBe("/notebook/log");
    expect(landingPath("system")).toBe("/system/connections");
  });

  it("builds tab paths with params", () => {
    expect(pagePath("models", { tab: "combiner", params: { mode: "starless" } })).toBe("/models/starless/combiner");
    // an unknown param value falls back to the default
    expect(pagePath("models", { tab: "combiner", params: { mode: "bogus" } })).toBe("/models/starfull/combiner");
    // an unknown tab falls back to the base
    expect(pagePath("sky", { tab: "nope" })).toBe("/sky");
  });

  it("every palette target is a page path", () => {
    const pages = allPages();
    expect(pages.length).toBeGreaterThan(30);
    for (const p of pages) expect(isPagePath(p.path), p.path).toBe(true);
    // both regimes of every models tab
    expect(pages.filter((p) => p.workspace === "models")).toHaveLength(12);
    expect(pages.find((p) => p.path === "/models/starless/train")?.label).toBe("Models (starless) › Train");
    expect(pages.find((p) => p.path === "/files")?.label).toBe("Files");
  });
});

describe("labels", () => {
  it("humanizes slugs", () => {
    expect(humanize("catalog-eval")).toBe("Catalog eval");
    expect(tabLabel("synthetic", "psf")).toBe("PSF");
    expect(tabLabel("sky", "zzz-top")).toBe("Zzz top");
  });

  it("describes a location and titles the document", () => {
    const d = describePath("/models/starless/members");
    expect(d?.workspaceLabel).toBe("Models");
    expect(d?.tabLabel).toBe("Members");
    expect(d?.paramLabels).toEqual(["starless"]);
    expect(pageTitle("/models/starless/members")).toBe("Members · Models (starless) · EuclidPolish");
    expect(pageTitle("/runs/history")).toBe("History · Runs · EuclidPolish");
    expect(pageTitle("/")).toBe("Home · EuclidPolish");
    expect(pageTitle("/sky")).toBe("Sky · EuclidPolish");
    expect(pageTitle("/nope")).toBe("Not found · EuclidPolish");
    expect(describePath("/ensemble/status.json")).toBeNull();
  });

  it("gives every page a heading: the tab, then where it lives", () => {
    expect(pageHeading("/models/starless/members")).toBe("Members, Models (starless)");
    expect(pageHeading("/synthetic/records")).toBe("Records, Synthetic");
    expect(pageHeading("/")).toBe("Home");
    expect(pageHeading("/files")).toBe("Files");
    expect(pageHeading("/inspect")).toBe("Not found");
    expect(pageHeading("/nope")).toBe("Not found");
  });
});
