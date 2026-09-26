import { describe, expect, it } from "vitest";
import tokensCss from "./tokens.css?raw";
import baseCss from "./base.css?raw";

/* ── tiny CSS/colour helpers (hex only; the token file uses hex colours) ── */
function block(css: string, selector: string): Record<string, string> {
  const start = css.indexOf(`${selector} {`);
  if (start < 0) throw new Error(`no block ${selector}`);
  const body = css.slice(css.indexOf("{", start) + 1, css.indexOf("\n}", start));
  const out: Record<string, string> = {};
  for (const m of body.replace(/\/\*[\s\S]*?\*\//g, "").matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)) {
    out[m[1]] = m[2].trim();
  }
  return out;
}

type RGBA = [number, number, number, number];
function hex(value: string): RGBA {
  const h = value.trim().replace("#", "");
  if (!/^[0-9a-f]{6}([0-9a-f]{2})?$/i.test(h)) throw new Error(`not a hex colour: ${value}`);
  const n = (i: number) => parseInt(h.slice(i, i + 2), 16);
  return [n(0), n(2), n(4), h.length === 8 ? n(6) / 255 : 1];
}
const over = (fg: RGBA, bg: RGBA): RGBA =>
  [0, 1, 2].map((i) => fg[i] * fg[3] + bg[i] * (1 - fg[3])).concat(1) as RGBA;
const lum = ([r, g, b]: RGBA) => {
  const f = (v: number) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
  return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
};
const contrast = (a: RGBA, b: RGBA) => {
  const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p);
  return (x + 0.05) / (y + 0.05);
};

const LIGHT = block(tokensCss, ":root");
const DARK = { ...LIGHT, ...block(tokensCss, ':root[data-theme="dark"]') };

/** Resolve `var(--x)` aliases within one theme. */
function resolve(theme: Record<string, string>, name: string): string {
  let v = theme[name];
  for (let i = 0; v && i < 8; i++) {
    const m = /^var\((--[\w-]+)\)$/.exec(v);
    if (!m) break;
    v = theme[m[1]];
  }
  if (v == null) throw new Error(`${name} undefined`);
  return v;
}

const REQUIRED = [
  // surfaces + ink
  "--bg-0", "--bg-1", "--surface-1", "--surface-2", "--surface-3", "--surface-grad", "--border",
  "--border-strong", "--rail-bg", "--app-bg", "--overlay", "--tooltip-bg", "--tooltip-text",
  "--text", "--text-dim", "--text-faint",
  // accent + status
  "--accent", "--accent-press", "--accent-soft", "--on-accent", "--amber", "--cyan", "--violet",
  "--magenta", "--good", "--warn", "--bad", "--info", "--good-soft", "--warn-soft", "--bad-soft",
  "--info-soft",
  // series, categorical, loss, bands
  "--series-baseline", "--series-mean", "--series-comb", "--series-muted", "--series-cross",
  "--series-guide", "--series-visfwhm", ...[0, 1, 2, 3, 4, 5, 6, 7].map((i) => `--cat-${i}`),
  "--loss-l1", "--loss-l2", "--loss-l3", "--loss-mse", "--band-vis", "--band-y", "--band-j", "--band-h",
  // type, space, radius, elevation, z, motion, layout
  "--font-sans", "--font-mono", "--fs-2xs", "--fs-xs", "--fs-sm", "--fs-md", "--fs-base", "--fs-lg",
  "--fs-xl", "--fs-2xl", "--fs-3xl", "--lh-tight", "--lh-base", "--fw-medium", "--fw-semibold",
  "--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--s8", "--r-sm", "--r-md", "--r-lg",
  "--r-pill", "--shadow-1", "--shadow-2", "--shadow-3", "--ring", "--z-sticky", "--z-rail",
  "--z-topbar", "--z-popover", "--z-dialog", "--z-toast", "--z-tooltip", "--dur-fast", "--dur-base",
  "--dur-slow", "--ease-standard", "--rail-w", "--rail-w-collapsed", "--topbar-h", "--inspector-w",
  "--ctl-h", "--row-h", "--card-pad",
  // formerly undefined names used by page CSS/TSX
  "--line", "--muted", "--mono", "--panel", "--bg", "--surface", "--surface-subtle",
  "--warning-text", "--radius-sm", "--r-1", "--r-2", "--danger", "--guide", "--text-sm",
];

const THEMED = [
  "--bg-0", "--bg-1", "--surface-1", "--surface-2", "--surface-3", "--border", "--text",
  "--text-dim", "--text-faint", "--accent", "--on-accent", "--good", "--warn", "--bad",
  "--series-mean", "--cat-0", "--loss-l1", "--band-vis", "--band-h", "--tooltip-bg",
];

describe("theme tokens", () => {
  it("defines the full token contract", () => {
    for (const name of REQUIRED) expect(LIGHT[name], name).toBeTruthy();
  });

  it("gives the dark theme its own value for every themed colour", () => {
    const dark = block(tokensCss, ':root[data-theme="dark"]');
    for (const name of THEMED) expect(dark[name], name).toBeTruthy();
  });

  it.each([["light", LIGHT], ["dark", DARK]] as const)("%s: text inks meet WCAG AA", (_n, theme) => {
    for (const ink of ["--text", "--text-dim", "--text-faint"]) {
      for (const surface of ["--surface-1", "--bg-0", "--bg-1", "--surface-2", "--surface-3"]) {
        const ratio = contrast(hex(resolve(theme, ink)), hex(resolve(theme, surface)));
        expect(ratio, `${ink} on ${surface}`).toBeGreaterThanOrEqual(4.5);
      }
    }
  });

  // Every opaque surface a badge or a status line can sit on: cards, the app
  // background, the rail/sunken bg-1 (rail badges) and the active/input
  // surface-3.
  const STATUS_SURFACES = ["--surface-1", "--surface-2", "--surface-3", "--bg-0", "--bg-1"];

  it.each([["light", LIGHT], ["dark", DARK]] as const)("%s: status badges meet WCAG AA", (_n, theme) => {
    for (const tone of ["--good", "--warn", "--bad", "--info"]) {
      for (const surface of STATUS_SURFACES) {
        const bg = over(hex(resolve(theme, `${tone}-soft`)), hex(resolve(theme, surface)));
        const ratio = contrast(hex(resolve(theme, tone)), bg);
        expect(ratio, `${tone} on ${tone}-soft over ${surface}`).toBeGreaterThanOrEqual(4.5);
      }
    }
  });

  it.each([["light", LIGHT], ["dark", DARK]] as const)("%s: status text meets WCAG AA on every surface", (_n, theme) => {
    for (const tone of ["--good", "--warn", "--bad", "--info"]) {
      for (const surface of STATUS_SURFACES) {
        const ratio = contrast(hex(resolve(theme, tone)), hex(resolve(theme, surface)));
        expect(ratio, `${tone} on ${surface}`).toBeGreaterThanOrEqual(4.5);
      }
    }
  });

  it("keeps the default accent and every preset readable (AA) on every surface of both themes", () => {
    // "blue" is the default: no [data-accent] block, the :root values apply.
    const accents = ["blue", "violet", "teal", "amber", "rose"];
    const surfaces = ["--surface-1", "--bg-0", "--bg-1", "--surface-2", "--surface-3"];
    for (const accent of accents) {
      for (const [name, theme, sel] of [
        ["light", LIGHT, `:root[data-accent="${accent}"]`],
        ["dark", DARK, `:root[data-theme="dark"][data-accent="${accent}"]`],
      ] as const) {
        const t = accent === "blue" ? theme : { ...theme, ...block(tokensCss, sel) };
        const where = `${name}/${accent}`;
        const a = hex(resolve(t, "--accent"));
        const press = hex(resolve(t, "--accent-press"));
        for (const surface of surfaces) {
          expect(contrast(a, hex(resolve(t, surface))), `${where}: --accent on ${surface}`).toBeGreaterThanOrEqual(4.5);
        }
        // selected chips / tabs / rail items: accent or accent-press ink on the soft tint
        for (const surface of ["--surface-1", "--bg-0", "--bg-1", "--surface-2"]) {
          const tint = over(hex(resolve(t, "--accent-soft")), hex(resolve(t, surface)));
          expect(contrast(a, tint), `${where}: --accent on --accent-soft over ${surface}`).toBeGreaterThanOrEqual(4.5);
          expect(contrast(press, tint), `${where}: --accent-press on --accent-soft over ${surface}`).toBeGreaterThanOrEqual(4.5);
        }
        expect(contrast(hex(resolve(t, "--on-accent")), a), `${where}: --on-accent on --accent`).toBeGreaterThanOrEqual(4.5);
      }
    }
  });

  it("keeps keyboard focus visible in forced-colors mode (box-shadows are not painted there)", () => {
    const rule = /:focus-visible\s*\{([^}]*)\}/.exec(baseCss);
    expect(rule, "global :focus-visible rule").toBeTruthy();
    expect(rule![1]).not.toMatch(/outline:\s*none/);
    expect(rule![1]).toMatch(/outline:\s*2px solid transparent/);   // forced-colors repaints it
    // …and component rules with `outline: none` are overridden there too
    expect(baseCss).toMatch(/@media\s*\(forced-colors:\s*active\)\s*\{\s*:focus-visible\s*\{[^}]*outline:\s*2px solid CanvasText !important/);
  });

  it("defines the .muted utility class in the base stylesheet", () => {
    expect(baseCss).toMatch(/\.muted\s*\{[^}]*color:\s*var\(--text-dim\)/);
  });
});

/* Every `var(--x)` the SPA source reads must be defined somewhere: in a
   stylesheet (`--x:`) or as an inline style key (`"--x"`). Catches the class
   of bug where a page silently fell back because a token never existed. */
describe("token usage", () => {
  const css = import.meta.glob("../**/*.css", { query: "?raw", import: "default", eager: true }) as Record<string, string>;
  const tsx = import.meta.glob("../**/*.{ts,tsx}", { query: "?raw", import: "default", eager: true }) as Record<string, string>;
  // Set at runtime by the imperative viewer engine (static/cutout_viewer.js,
  // outside src/) via style.setProperty.
  const EXTERNALLY_SET = ["--cv-columns", "--cv-frame-size"];

  it("has no undefined custom properties", () => {
    const defined = new Set<string>(EXTERNALLY_SET);
    for (const text of Object.values(css)) for (const m of text.matchAll(/(--[\w-]+)\s*:/g)) defined.add(m[1]);
    for (const text of Object.values(tsx)) for (const m of text.matchAll(/["'`](--[\w-]+)["'`]/g)) defined.add(m[1]);
    const missing = new Set<string>();
    for (const [file, text] of [...Object.entries(css), ...Object.entries(tsx)]) {
      for (const m of text.matchAll(/var\(\s*(--[\w-]+)/g)) {
        const name = m[1];
        if (name.endsWith("-")) continue;          // template prefix: var(--cat-${i})
        if (name.startsWith("--radix-")) continue; // set at runtime by Radix (popover/menu sizes)
        if (!defined.has(name)) missing.add(`${name} (${file})`);
      }
    }
    expect([...missing]).toEqual([]);
  });
});
