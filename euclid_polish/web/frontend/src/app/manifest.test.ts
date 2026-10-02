import { describe, expect, it } from "vitest";
import cases from "../../../spa_redirect_cases.json";
import {
  MANIFEST,
  isPagePath,
  matchPage,
  pagePaths,
  redirectTarget,
  workspacePaths,
  type RouteManifest,
} from "./manifest";

type RedirectCase = { from: string; to: string | null };
const CASES = cases as RedirectCase[];

/* Mirrors tests/test_spa_routes.py so the Flask matcher (spa_routes.py) and the
   SPA/dev-proxy matcher agree on what a page path is (contract C1). */
describe("route manifest (C1)", () => {
  it("expands every workspace path and tab into a page", () => {
    for (const p of [
      "/", "/sky", "/sky/atlas", "/sky/targets", "/sky/compare", "/models",
      "/models/train", "/models/leaderboard", "/files", "/synthetic",
      "/synthetic/psf", "/runs/history", "/notebook/sandboxes", "/system/storage", "/figures/sheet",
    ]) expect(isPagePath(p), p).toBe(true);
  });

  it("ignores a trailing slash like Flask does", () => {
    expect(isPagePath("/sky/")).toBe(true);
    expect(isPagePath("/models/train/")).toBe(true);
  });

  it("rejects data URLs that share a page prefix, and the old pages", () => {
    for (const p of [
      "/ensemble/status.json", "/ensemble/foo", "/ensemble", "/sky/unknown", "/models/foo",
      "/models/starfull", "/models/starless/train", "/inspect/preview.png", "/api/jobs", "/static/x.js",
      "/ensemble/starfull/overview", "/sky/results", "/inspect", "/settings/about", "/overview", "",
    ]) expect(isPagePath(p), p).toBe(false);
  });

  it("substitutes params from their allowed values only", () => {
    const lab: RouteManifest = { version: 2, redirects: {}, workspaces: [
      { id: "lab", label: "Lab", path: "/lab/:mode", params: { mode: ["wet", "dry"] }, tabs: ["bench"] },
    ] };
    expect(workspacePaths(lab.workspaces[0]).sort()).toEqual(["/lab/dry", "/lab/wet"]);
    expect(isPagePath("/lab/wet/bench", lab)).toBe(true);
    expect(matchPage("/lab/dry/bench", lab)?.params).toEqual({ mode: "dry" });
    expect(isPagePath("/lab/damp", lab)).toBe(false);
  });

  it("lists every concrete page path once", () => {
    const pages = pagePaths();
    expect(new Set(pages).size).toBe(pages.length);
    expect(pages).toContain("/");
    expect(pages).toContain("/models/combiner");
    expect(pages).not.toContain("//atlas");
  });

  it("matches a page to its workspace, params and tab", () => {
    expect(matchPage("/models/images")).toEqual({
      workspace: "models", params: {}, tab: "images",
      base: "/models",
    });
    expect(matchPage("/sky")).toEqual({ workspace: "sky", params: {}, tab: null, base: "/sky" });
    expect(matchPage("/")).toEqual({ workspace: "home", params: {}, tab: null, base: "/" });
    expect(matchPage("/files")).toEqual({ workspace: "files", params: {}, tab: null, base: "/files" });
    expect(matchPage("/ensemble/status.json")).toBeNull();
  });

  it("orders the rail as approved", () => {
    expect(MANIFEST.workspaces.map((w) => w.id)).toEqual([
      "home", "synthetic", "models", "sky", "figures", "files", "runs", "notebook", "system",
    ]);
  });

  it("redirects every exact legacy entry, preserving the query string", () => {
    for (const [from, to] of Object.entries(MANIFEST.redirects)) {
      const q = "?a=1&b=2";
      expect(redirectTarget(from)).toBe(to);
      expect(redirectTarget(from, q)).toBe(`${to}${q}`);
      expect(redirectTarget(`${from}/`, "x=1")).toBe(`${to}?x=1`);
    }
    expect(redirectTarget("/config")).toBe("/system/config");
    expect(redirectTarget("/sky")).toBeNull();
    expect(redirectTarget("/ensemble/status.json")).toBeNull();
  });

  it("maps the old /app prefix onto the bare path", () => {
    expect(redirectTarget("/app/x", "y=1")).toBe("/x?y=1");
    expect(redirectTarget("/app/inference")).toBe("/sky/targets"); // one hop
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

  it("sends every exact redirect to a page without a query and never shadows one", () => {
    for (const [source, target] of Object.entries(MANIFEST.redirects)) {
      expect(isPagePath(target), `${source} → ${target}`).toBe(true);
      expect(target.includes("?") || target.includes("#"), `${source} → ${target}`).toBe(false);
      expect(isPagePath(source), `redirect source ${source}`).toBe(false);
    }
  });
});

/* The query-aware rules: the same case file as tests/test_spa_routes.py, so
   Flask's 308 and the router's <Navigate> land on byte-identical URLs. */
describe("redirect rules (spa_redirect_cases.json)", () => {
  it.each(CASES.map((c) => [c.from, c.to] as const))("%s → %s", (from, to) => {
    const i = from.indexOf("?");
    const path = i < 0 ? from : from.slice(0, i);
    const query = i < 0 ? "" : from.slice(i + 1);
    expect(redirectTarget(path, query)).toBe(to);
    // the router hands over `location.search` with its "?"
    if (query) expect(redirectTarget(path, `?${query}`)).toBe(to);
  });

  it("constrains every path param, so no data endpoint is captured", () => {
    for (const rule of MANIFEST.redirectRules ?? []) {
      const names = rule.from.split("/").filter((s) => s.startsWith(":")).map((s) => s.slice(1)).sort();
      expect(Object.keys(rule.params ?? {}).sort(), rule.from).toEqual(names);
    }
  });

  it("lands every rule on a page, never on another redirect, and never shadows a page", () => {
    for (const rule of MANIFEST.redirectRules ?? []) {
      let sources = [rule.from];
      for (const [name, values] of Object.entries(rule.params ?? {})) {
        sources = sources.flatMap((s) => values.map((v) => s.replace(`:${name}`, v)));
      }
      const q = new URLSearchParams(Object.entries(rule.query ?? {}).map(([k, want]) => [
        k, want === "*" ? "x" : Array.isArray(want) ? want[0] : want,
      ])).toString();
      for (const source of sources) {
        expect(isPagePath(source), `rule source ${source}`).toBe(false);
        const target = redirectTarget(source, q);
        expect(target, source).not.toBeNull();
        const [path, search = ""] = target!.split("?");
        expect(isPagePath(path), `${source}?${q} → ${target}`).toBe(true);
        expect(redirectTarget(path, search), `${target} redirects again`).toBeNull();
      }
    }
  });

  it("applies drop, rename, map, prefix and set in that order", () => {
    const manifest: RouteManifest = {
      version: 2,
      workspaces: [{ id: "lab", label: "Lab", path: "/lab", tabs: ["bench", "log"] }],
      redirectRules: [
        {
          from: "/old/:kind", params: { kind: ["a", "b"] }, query: { v: ["x", "y"] }, to: "/lab/bench",
          drop: ["v"], rename: { q: "find" }, map: { find: { old: "new" } }, prefix: { id: "run:" }, set: { kind: ":kind" },
        },
        { from: "/old/:kind", params: { kind: ["a", "b"] }, to: "/lab/log" },
      ],
      redirects: { "/old/a": "/lab/bench" },
    };
    expect(redirectTarget("/old/a", "v=x&q=old&id=7&z=1", manifest)).toBe("/lab/bench?find=new&id=run%3A7&z=1&kind=a");
    expect(redirectTarget("/old/b", "v=z", manifest)).toBe("/lab/log?v=z");
    expect(redirectTarget("/old/a", "", manifest)).toBe("/lab/log");
    expect(redirectTarget("/old/c", "v=x", manifest)).toBeNull();
  });
});
