import { describe, expect, it } from "vitest";
import {
  MANIFEST,
  isPagePath,
  matchPage,
  pagePaths,
  redirectTarget,
  workspacePaths,
} from "./manifest";

/* Mirrors tests/test_spa_routes.py so the Flask matcher (spa_routes.py) and the
   SPA/dev-proxy matcher agree on what a page path is (contract C1). */
describe("route manifest (C1)", () => {
  it("expands every workspace path and tab into a page", () => {
    for (const p of [
      "/", "/sky", "/sky/atlas", "/sky/catalog-eval", "/ensemble/starfull",
      "/ensemble/starless/train", "/ensemble/starfull/overview", "/inspect",
      "/settings/about", "/ops/provenance", "/data/records", "/realism/visual",
    ]) expect(isPagePath(p), p).toBe(true);
  });

  it("ignores a trailing slash like Flask does", () => {
    expect(isPagePath("/sky/")).toBe(true);
    expect(isPagePath("/ensemble/starfull/train/")).toBe(true);
  });

  it("rejects data URLs that share a page prefix", () => {
    for (const p of [
      "/ensemble/status.json", "/ensemble/foo", "/ensemble", "/sky/unknown",
      "/inspect/preview.png", "/api/jobs", "/static/x.js", "/ensemble/starfull/nope",
      "/overview", "",
    ]) expect(isPagePath(p), p).toBe(false);
  });

  it("substitutes params from their allowed values only", () => {
    const ensemble = MANIFEST.workspaces.find((w) => w.id === "ensemble")!;
    expect(workspacePaths(ensemble).sort()).toEqual(["/ensemble/starfull", "/ensemble/starless"]);
    expect(isPagePath("/ensemble/starfree")).toBe(false);
  });

  it("lists every concrete page path once", () => {
    const pages = pagePaths();
    expect(new Set(pages).size).toBe(pages.length);
    expect(pages).toContain("/");
    expect(pages).toContain("/ensemble/starless/combiners");
    expect(pages).not.toContain("//atlas");
  });

  it("matches a page to its workspace, params and tab", () => {
    expect(matchPage("/ensemble/starless/knee")).toEqual({
      workspace: "ensemble", params: { mode: "starless" }, tab: "knee",
      base: "/ensemble/starless",
    });
    expect(matchPage("/sky")).toEqual({ workspace: "sky", params: {}, tab: null, base: "/sky" });
    expect(matchPage("/")).toEqual({ workspace: "home", params: {}, tab: null, base: "/" });
    expect(matchPage("/ensemble/status.json")).toBeNull();
  });

  it("redirects every legacy entry, preserving the query string", () => {
    for (const [from, to] of Object.entries(MANIFEST.redirects)) {
      expect(redirectTarget(from)).toBe(to);
      expect(redirectTarget(from, "?a=1&b=2")).toBe(`${to}?a=1&b=2`);
      expect(redirectTarget(`${from}/`, "x=1")).toBe(`${to}?x=1`);
    }
    expect(redirectTarget("/config")).toBe("/settings/config");
    expect(redirectTarget("/sky")).toBeNull();
    expect(redirectTarget("/ensemble/status.json")).toBeNull();
  });

  it("maps the old /app prefix onto the bare path", () => {
    expect(redirectTarget("/app/x", "y=1")).toBe("/x?y=1");
    expect(redirectTarget("/app/inference")).toBe("/inference");
    expect(redirectTarget("/app/sky/atlas")).toBe("/sky/atlas");
    expect(redirectTarget("/app")).toBe("/");
    expect(redirectTarget("/app/", "q=1")).toBe("/?q=1");
    expect(redirectTarget("/application")).toBeNull();
  });

  // tests/test_spa_routes.py::test_app_prefix_redirect_never_leaves_the_host —
  // `//host` and `/\host` are protocol-relative to a browser, so every leading
  // slash, backslash, space or control character after `/app` collapses into
  // one `/` (spa_routes._same_host_path): the target is always on this host.
  it.each([
    ["/app//evil.example", "/evil.example"],
    ["/app//evil.example/x", "/evil.example/x"],
    ["/app///evil.example", "/evil.example"],
    ["/app/\\evil.example", "/evil.example"],
    ["/app/\\/evil.example", "/evil.example"],
    ["/app//\\evil.example", "/evil.example"],
    ["/app/\\\\evil.example", "/evil.example"],
    ["/app/\t/evil.example", "/evil.example"],
    ["/app/\n//evil.example", "/evil.example"],
    ["/app//", "/"],
    ["/app/ \u0000\u001f /evil.example", "/evil.example"],
  ])("the /app redirect never leaves the host: %j → %s", (path, target) => {
    const location = redirectTarget(path);
    expect(location).toBe(target);
    expect(location!.startsWith("//")).toBe(false);
    expect(location!.startsWith("/\\")).toBe(false);
  });

  it.each(["/sky", "/sky/atlas", "/api/jobs", "/ensemble/status.json", "/apple", "/application/x", "/", "/configure"])(
    "%s has no redirect target", (path) => {
      expect(redirectTarget(path, "q=1")).toBeNull();
    },
  );

  it("sends every redirect to a page and never shadows one", () => {
    for (const [source, target] of Object.entries(MANIFEST.redirects)) {
      expect(isPagePath(target), `${source} → ${target}`).toBe(true);
      expect(isPagePath(source), `redirect source ${source}`).toBe(false);
    }
  });
});
