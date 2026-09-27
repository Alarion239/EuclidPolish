import { describe, expect, it } from "vitest";
import { viewerTakesEscape } from "./viewerEscape";

const html = (s: string) => { const d = document.createElement("div"); d.innerHTML = s; return d; };

describe("viewerTakesEscape (the inspector sheet leaves Esc to a viewer that uses it)", () => {
  it("is true for a viewer in focus mode, with a frozen lens, an open Display dock or a profile", () => {
    expect(viewerTakesEscape(html('<div class="cv-root" data-focus=""></div>'))).toBe(true);
    expect(viewerTakesEscape(html('<div class="cv-root"><div class="cv-lens cv-lens--frozen"></div></div>'))).toBe(true);
    expect(viewerTakesEscape(html('<div class="cv-root"><div class="cv-body" data-dock="true"></div></div>'))).toBe(true);
    expect(viewerTakesEscape(html('<div class="cv-root"><svg class="cv-svg"><g class="cv-prof"></g></svg></div>'))).toBe(true);
  });
  it("is false for an idle viewer or no viewer (Esc closes the sheet)", () => {
    expect(viewerTakesEscape(html('<div class="cv-root"><div class="cv-body"></div></div>'))).toBe(false);
    expect(viewerTakesEscape(html("<p>job</p>"))).toBe(false);
    expect(viewerTakesEscape(null)).toBe(false);
  });
});
