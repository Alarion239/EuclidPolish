/* The kit's chrome rules (FOUNDATION §8–9), checked on the stylesheets
 * themselves:
 *   - labels are sentence case as authored, in the UI face: no
 *     text-transform (ALL-CAPS turned "σ" into "Σ" and "HR" choices into
 *     "hr"), no letter-spaced eyebrow;
 *   - every colour of the kit, the charts and the shell is a token: no raw
 *     hex / rgb() / hsl() outside theme/tokens.css (a hard-coded #fff gallery
 *     cell carried 2.4:1 captions in the dark theme).
 * Data values (table numbers, readouts, ticks, badges' numbers in cells) may
 * stay tabular mono; that is not checked here. */
import { describe, expect, it } from "vitest";

const sheets = import.meta.glob(["../ui/*.css", "../charts/*.css", "../app/*.css", "./base.css"], {
  query: "?raw", import: "default", eager: true,
}) as Record<string, string>;

const stripComments = (css: string) => css.replace(/\/\*[\s\S]*?\*\//g, "");

/** The declarations of the rule whose selector list is exactly `selector`. */
function rule(css: string, selector: string): string {
  const src = stripComments(css);
  const esc = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const m = new RegExp(`(^|[}\\n])\\s*${esc}\\s*\\{([^}]*)\\}`).exec(src);
  if (!m) throw new Error(`no rule ${selector}`);
  return m[2];
}

describe("kit chrome stylesheets", () => {
  it("are all found", () => {
    const names = Object.keys(sheets);
    for (const f of ["../ui/ui.css", "../ui/pages-compat.css", "../charts/plot.css", "../app/shell.css", "./base.css"]) {
      expect(names).toContain(f);
    }
  });

  it("never transform the case of a label or letter-space an eyebrow", () => {
    for (const [file, css] of Object.entries(sheets)) {
      const src = stripComments(css);
      expect(src.match(/text-transform:\s*(uppercase|lowercase|capitalize)/g), file).toBeNull();
      expect(src.match(/letter-spacing:\s*var\(--ls-eyebrow\)/g), file).toBeNull();
    }
  });

  it("write every colour as a token (no raw hex, rgb() or hsl())", () => {
    for (const [file, css] of Object.entries(sheets)) {
      const src = stripComments(css);
      expect(src.match(/#[0-9a-f]{3,8}\b/gi), file).toBeNull();
      expect(src.match(/\b(rgb|rgba|hsl|hsla)\(/g), file).toBeNull();
    }
  });

  it("set labels in the UI face: table headers, field / KPI / stat labels, segmented choices, badges, tabs, chips", () => {
    const ui = sheets["../ui/ui.css"];
    for (const sel of [".ui-dt__table thead th", ".ui-table th", ".ui-field__label", ".ui-kpi__label", ".ui-stat__k",
      ".ui-seg__item, .ui-seg button", ".ui-badge", ".ui-tab", ".ui-chip", ".ui-menu__label"]) {
      const body = rule(ui, sel);
      expect(body, sel).toMatch(/var\(--font-sans\)/);
      expect(body, sel).not.toMatch(/var\(--font-mono\)/);
    }
    expect(sheets["./base.css"]).not.toMatch(/\.eyebrow\s*\{[^}]*font-mono/);
  });

  it("shade a table's scrolled edge with a darkening token in both themes (never mixed from --text, a light haze in dark)", () => {
    const ui = sheets["../ui/ui.css"];
    for (const sel of [".ui-dt__frame::before", ".ui-dt__frame::after"]) {
      expect(rule(ui, sel), sel).toMatch(/var\(--scroll-shade\)/);
      expect(rule(ui, sel), sel).not.toMatch(/var\(--text\)/);
    }
  });

  it("keep numeric table cells tabular mono (they are data)", () => {
    expect(rule(sheets["../ui/ui.css"], ".ui-dt__table td.is-num")).toMatch(/var\(--font-mono\)/);
  });

  it("back a gallery image with the neutral image ink and its cell with a themed surface", () => {
    const ui = sheets["../ui/ui.css"];
    expect(rule(ui, ".ui-gallery__cell")).toMatch(/background:\s*var\(--surface-1\)/);
    expect(rule(ui, ".ui-gallery__cell img")).toMatch(/background:\s*var\(--image-ink\)/);
    expect(rule(ui, ".ui-figure__paper")).toMatch(/background:\s*var\(--paper\)/);
  });
});
