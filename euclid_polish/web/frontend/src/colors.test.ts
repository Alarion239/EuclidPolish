import { afterEach, describe, expect, it } from "vitest";
import { C, bandColor, categorical } from "./colors";

const root = () => document.documentElement;

afterEach(() => { root().removeAttribute("style"); });

describe("chart colours", () => {
  it("read the live token values", () => {
    root().style.setProperty("--band-vis", "#123456");
    root().style.setProperty("--series-mean", "#abcdef");
    root().style.setProperty("--cat-3", "#010203");
    expect(bandColor("VIS")).toBe("#123456");
    expect(C.mean).toBe("#abcdef");
    expect(categorical(11)).toBe("#010203");          // wraps modulo 8
  });

  it("fall back to the light palette before styles apply", () => {
    expect(bandColor("VIS")).toBe("#2563eb");
    expect(bandColor("H")).toBe("#dc2626");            // short band name
    expect(bandColor("F200W")).toBe("#9aa6b6");        // unknown → muted
    expect(C.baseline).toBe("#e11d48");
  });
});
