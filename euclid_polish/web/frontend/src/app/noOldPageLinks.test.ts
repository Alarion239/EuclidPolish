/* Guard (console regrouping, plan "Console phases 2–6", task F5): no source
 * file links to an old page URL. Every page moved (spa_routes.json v2), and
 * an old URL only survives as a redirect; a link to it costs a redirect hop,
 * drops query keys the rule does not carry, and hides the page's new name.
 *
 * A string or template literal is an old page link when
 *   1. its path (up to `?`, `#` or a `${…}` that follows a whole segment; a
 *      `${…}` inside a segment stands for a path parameter) is a redirect
 *      source of the manifest (`redirectTarget`), which Flask and the router
 *      both answer with the move — backend URLs never are (tests/test_spa_routes.py
 *      pins that no Flask endpoint is a redirect source); or
 *   2. it starts with one of the old page prefixes the plan lists, except the
 *      backend URLs under them (`API_UNDER_OLD_PREFIXES`).
 * Comments are not scanned (they may say where a page came from), nor are
 * test files (they mount old URLs to prove the redirects), nor the `/app`
 * prefix the router itself redirects. */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { redirectTarget } from "./manifest";

const SRC = join(__dirname, "..");

function files(dir: string): string[] {
  return readdirSync(dir).flatMap((n) => {
    const p = join(dir, n);
    return statSync(p).isDirectory() ? files(p) : /\.tsx?$/.test(n) && !/\.test\.tsx?$/.test(n) ? [p] : [];
  });
}

/** The source without block comments and `//` line comments (a `//` inside
 *  a URL such as `https://` follows a colon, so it stays). */
export function stripComments(source: string): string {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, (c) => c.replace(/[^\n]/g, " "))
    .replace(/(^|[^:"'`\\])\/\/[^\n]*/gm, "$1");
}

/** Old page prefixes named by the plan (task F5). */
const OLD_PREFIXES = [
  "/sky/results", "/sky/experiments", "/sky/catalog-eval", "/realism", "/data", "/ops", "/settings",
  "/inspect", "/figures/grid", "/figures/results",
];
/** Backend URLs that share an old page prefix. */
const API_UNDER_OLD_PREFIXES = ["/inspect/preview.png", "/inspect/download"];
/** Backend cache prefixes that spell an old page path: `invalidate("/ensemble/")`
 *  refetches every `/ensemble/*.json` endpoint. */
const BACKEND_PREFIXES = new Set(["/ensemble/"]);

/** The literal starts: a quote or backtick, then a root-relative path. */
const LITERAL = /["'`](\/[^"'`\s]*)/g;

/** `path` starts with the whole `prefix` segment(s). */
const underPrefix = (path: string, prefix: string) =>
  path.startsWith(prefix) && (path.length === prefix.length || /^[/?#$]/.test(path.slice(prefix.length)));

/** The page path a literal addresses, for the manifest check: a `${…}`
 *  that starts a segment stands for a path parameter (read as a regime:
 *  `/ensemble/${mode}/members`, `/ensemble/${name}.json`); any other `${…}`
 *  starts a query or hash suffix (`/data/records${q}`), where the path ends. */
export function literalPaths(literal: string): string[] {
  const params = literal.replace(/\/\$\{[^}]*\}/g, "/starfull");
  const path = params.split(/[?#]|\$\{/)[0];
  return path && path !== "/" && !path.startsWith("/app") ? [path] : [];
}

/** Why `literal` is an old page link, or null. */
export function oldPageLink(literal: string): string | null {
  if (BACKEND_PREFIXES.has(literal)) return null;
  for (const path of literalPaths(literal)) {
    if (redirectTarget(path) != null) return `${path} redirects to ${redirectTarget(path)}`;
  }
  const prefix = OLD_PREFIXES.find((p) => underPrefix(literal, p));
  if (prefix && !API_UNDER_OLD_PREFIXES.some((api) => literal.startsWith(api))) return `old page prefix ${prefix}`;
  return null;
}

describe("the old-page-link matcher", () => {
  it("flags old page URLs, in strings and templates", () => {
    expect(oldPageLink("/sky/results")).toMatch(/redirects/);
    expect(oldPageLink("/sky/results?inspect=tile%3Aa")).toMatch(/redirects/);
    expect(oldPageLink("/data/records${q({ subset })}")).toMatch(/redirects/);
    expect(oldPageLink("/ensemble/${mode}/disagreement${q(pos)}")).toMatch(/redirects/);
    expect(oldPageLink("/ops/fasrc?view=steps&step=${id}")).toMatch(/redirects/);
    expect(oldPageLink("/inspect?fits=${encodeURIComponent(p)}")).toMatch(/redirects/);
    expect(oldPageLink("/realism/unknown")).toMatch(/prefix/);
    expect(oldPageLink("/settings")).toMatch(/redirects/);
    // the star regime left the Models URLs
    expect(oldPageLink("/models/starfull/combiner")).toMatch(/redirects/);
    expect(oldPageLink("/models/${mode}/images")).toMatch(/redirects/);
  });

  it("lets backend URLs and v2 pages through", () => {
    for (const ok of [
      "/ensemble/status.json", "/ensemble/members.json?mode=${mode}", "/ensemble/combiners/compare",
      "/ensemble/train/preview", "/ensemble/${name}.json", "/inspect/preview.png?path=${p}", "/inspect/download",
      "/api/realism/overview", "/tng/result/grid.png", "/git/commit", "/fasrc/file/inspect${qs(x)}",
      "/inference/refresh-combiners", "/viewer/meta/sky", "/models/images", "/synthetic/psf?view=cutouts",
      "/sky/targets", "/files?fits=${p}", "/runs/history", "/app/*", "/datasets", "/opsx", "/ensemble/",
      "/ensemble/${endpoint}.json",
    ]) expect(oldPageLink(ok), ok).toBeNull();
  });

  it("does not scan comments", () => {
    const src = stripComments("/* was `/sky/results` */ const a = 1; // see '/ops/jobs'\nconst u = \"https://x.org/a\";");
    expect(src).not.toMatch(/sky\/results|ops\/jobs/);
    expect(src).toContain("https://x.org/a");
  });
});

it("no source file links to an old page URL", () => {
  const offenders: string[] = [];
  for (const file of files(SRC)) {
    const text = stripComments(readFileSync(file, "utf8"));
    for (const m of text.matchAll(LITERAL)) {
      const why = oldPageLink(m[1]);
      if (why) offenders.push(`${file.slice(SRC.length + 1)}: ${m[1]} (${why})`);
    }
  }
  expect(offenders).toEqual([]);
});
