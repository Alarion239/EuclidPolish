/* The inline pre-paint script in index.html repeats the prefs sanitising of
   state/prefs.ts (theme, accent, density, the legacy "ep-theme" fallback) so
   the first paint already has the right theme. This suite runs that script
   against seeded storage + a stubbed `prefers-color-scheme` and checks that it
   writes exactly the <html> attributes `bindPrefsToDocument()` writes once the
   app has started from the same storage — so the two cannot drift (e.g. a new
   accent added to ACCENTS but not to the script's regex). */
import { afterEach, describe, expect, it, vi } from "vitest";
import indexHtml from "../../index.html?raw";
import { ACCENTS, DENSITIES, LEGACY_THEME_KEY, PREFS_STORAGE_KEY, THEME_PREFS } from "./prefs";

const ATTRS = ["data-theme", "data-theme-pref", "data-accent", "data-density"] as const;

/** The classic (non-module) inline <script> of index.html. */
function prePaintSource(): string {
  const scripts = [...indexHtml.matchAll(/<script(\s[^>]*)?>([\s\S]*?)<\/script>/g)]
    .filter((m) => !/\bsrc=|type="module"/.test(m[1] ?? ""));
  if (scripts.length !== 1) throw new Error(`expected one inline script in index.html, found ${scripts.length}`);
  return scripts[0][2];
}
const PRE_PAINT = prePaintSource();

type Seed = {
  prefs?: string | null;
  legacy?: string | null;
  systemDark?: boolean | "throws" | "missing";
  storage?: "ok" | "throws";
};

function seed({ prefs = null, legacy = null, systemDark = false, storage = "ok" }: Seed) {
  localStorage.clear();
  if (prefs != null) localStorage.setItem(PREFS_STORAGE_KEY, prefs);
  if (legacy != null) localStorage.setItem(LEGACY_THEME_KEY, legacy);
  if (storage === "throws") {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw new Error("SecurityError"); });
  }
  if (systemDark === "missing") vi.stubGlobal("matchMedia", undefined);
  else if (systemDark === "throws") vi.stubGlobal("matchMedia", () => { throw new Error("no media queries"); });
  else vi.stubGlobal("matchMedia", (q: string) => ({ matches: systemDark && q === "(prefers-color-scheme: dark)", media: q }));
}

const read = (el: Element) => Object.fromEntries(ATTRS.map((a) => [a, el.getAttribute(a)]));

/** What the inline script writes to <html> before first paint. */
function runPrePaint(): Record<string, string | null> {
  const root = document.documentElement;
  for (const a of ATTRS) root.removeAttribute(a);
  new Function(PRE_PAINT)();
  return read(root);
}

/** What the app writes once started from the same storage: a fresh prefs
 *  module (store construction + persist hydration) bound to an element. */
async function runApp(): Promise<Record<string, string | null>> {
  vi.resetModules();
  const prefs = await import("./prefs");
  const el = document.createElement("div");
  const unbind = prefs.bindPrefsToDocument(el);
  unbind();
  return read(el);
}

const state = (s: Record<string, unknown>) => JSON.stringify({ state: s, version: 1 });

afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); localStorage.clear(); });

describe("index.html pre-paint script ≡ prefs.ts", () => {
  it("is found in index.html and keys on the prefs storage keys", () => {
    expect(PRE_PAINT).toContain(`"${PREFS_STORAGE_KEY}"`);
    expect(PRE_PAINT).toContain(`"${LEGACY_THEME_KEY}"`);
  });

  const cases: [string, Seed][] = [
    ["nothing saved (defaults)", {}],
    ...THEME_PREFS.flatMap((theme) => [false, true].map((dark): [string, Seed] => [
      `theme ${theme}, OS ${dark ? "dark" : "light"}`, { prefs: state({ theme }), systemDark: dark },
    ])),
    ...ACCENTS.map((accent): [string, Seed] => [`accent ${accent}`, { prefs: state({ accent }) }]),
    ...DENSITIES.map((density): [string, Seed] => [`density ${density}`, { prefs: state({ density }) }]),
    ["every field at once", { prefs: state({ theme: "dark", accent: "rose", density: "compact" }) }],
    ["invalid values fall back to defaults", { prefs: state({ theme: "neon", accent: "plaid", density: 3 }) }],
    ["accent with a matching prefix is rejected", { prefs: state({ accent: "blueish" }) }],
    ["non-string accent", { prefs: state({ accent: ["teal"] }) }],
    ["corrupt JSON", { prefs: "{not json" }],
    ["JSON null", { prefs: "null" }],
    ["no state object", { prefs: JSON.stringify({ version: 1 }) }],
    ["legacy ep-theme dark (no prefs saved)", { legacy: "dark" }],
    ["legacy ep-theme light", { legacy: "light" }],
    ["legacy ep-theme garbage", { legacy: "system" }],
    ["saved prefs win over the legacy key", { prefs: state({ theme: "light" }), legacy: "dark" }],
    ["empty saved prefs still win over the legacy key", { prefs: state({}), legacy: "dark" }],
    ["system theme without matchMedia", { prefs: state({ theme: "system" }), systemDark: "missing" }],
    ["system theme with a throwing matchMedia", { prefs: state({ theme: "system" }), systemDark: "throws" }],
    ["storage that throws", { prefs: state({ theme: "dark" }), storage: "throws" }],
  ];

  it.each(cases)("%s", async (_name, s) => {
    seed(s);
    const prePaint = runPrePaint();
    const app = await runApp();
    expect(prePaint).toEqual(app);
    for (const a of ATTRS) expect(prePaint[a], a).toBeTruthy();
  });

  it("really resolves the cases it claims to (spot checks)", async () => {
    seed({ prefs: state({ theme: "system", accent: "teal", density: "compact" }), systemDark: true });
    expect(runPrePaint()).toEqual({
      "data-theme": "dark", "data-theme-pref": "system", "data-accent": "teal", "data-density": "compact",
    });
    seed({ legacy: "dark" });
    expect(runPrePaint()["data-theme"]).toBe("dark");
    seed({ prefs: "{not json", legacy: "dark" });
    expect(runPrePaint()["data-theme"]).toBe("light");
  });
});
