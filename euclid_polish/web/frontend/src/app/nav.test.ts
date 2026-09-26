import { describe, expect, it } from "vitest";
import { ICON_NAMES } from "../ui/icons";
import { MANIFEST, isPagePath } from "./manifest";
import {
  WORKSPACE_META, allPages, describePath, humanize, landingPath, pagePath, pageTitle, tabLabel,
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
});

describe("paths", () => {
  it("lands on the default tab with default params", () => {
    expect(landingPath("home")).toBe("/");
    expect(landingPath("sky")).toBe("/sky/atlas");
    expect(landingPath("ensemble")).toBe("/ensemble/starfull/overview");
    expect(landingPath("inspect")).toBe("/inspect");
    expect(landingPath("settings")).toBe("/settings/config");
  });

  it("builds tab paths with params", () => {
    expect(pagePath("ensemble", { tab: "knee", params: { mode: "starless" } })).toBe("/ensemble/starless/knee");
    // an unknown param value falls back to the default
    expect(pagePath("ensemble", { tab: "knee", params: { mode: "bogus" } })).toBe("/ensemble/starfull/knee");
    // an unknown tab falls back to the base
    expect(pagePath("sky", { tab: "nope" })).toBe("/sky");
  });

  it("every palette target is a page path", () => {
    const pages = allPages();
    expect(pages.length).toBeGreaterThan(30);
    for (const p of pages) expect(isPagePath(p.path), p.path).toBe(true);
    // both regimes of every ensemble tab
    expect(pages.filter((p) => p.workspace === "ensemble")).toHaveLength(16);
    expect(pages.find((p) => p.path === "/ensemble/starless/train")?.label).toBe("Ensemble (starless) › Train");
  });
});

describe("labels", () => {
  it("humanizes slugs", () => {
    expect(humanize("catalog-eval")).toBe("Catalog eval");
    expect(tabLabel("sky", "catalog-eval")).toBe("Catalog eval");
    expect(tabLabel("sky", "zzz-top")).toBe("Zzz top");
  });

  it("describes a location and titles the document", () => {
    const d = describePath("/ensemble/starless/members");
    expect(d?.workspaceLabel).toBe("Ensemble");
    expect(d?.tabLabel).toBe("Members");
    expect(d?.paramLabels).toEqual(["starless"]);
    expect(pageTitle("/ensemble/starless/members")).toBe("Members · Ensemble (starless) · EuclidPolish");
    expect(pageTitle("/")).toBe("Home · EuclidPolish");
    expect(pageTitle("/sky")).toBe("Sky · EuclidPolish");
    expect(pageTitle("/nope")).toBe("Not found · EuclidPolish");
    expect(describePath("/ensemble/status.json")).toBeNull();
  });
});
