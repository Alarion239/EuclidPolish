import { describe, expect, it } from "vitest";
import {
  IMG_CODEC, MAX_PIXEL_OVERLAYS, overlayKey, overlayLabel, overlayUrl, patchPixelOverlay, resolveOverlays, tierLabel,
  withOverlays, withoutOverlay, type PixelOverlaySetting,
} from "./pixelOverlays";

describe("pixel overlays (the `img` URL param)", () => {
  it("round-trips entries: ref|tier|band, @opacity, ! hidden", () => {
    const raw = "nexus/f200w-0200|m:rbf|VIS@0.8,!archive/007|lr|Y_E,pair/p1|jwst|";
    const list = IMG_CODEC.parse(raw)!;
    expect(list).toEqual([
      { ref: "nexus/f200w-0200", tier: "m:rbf", band: "VIS", opacity: 0.8 },
      { ref: "archive/007", tier: "lr", band: "Y_E", hidden: true },
      { ref: "pair/p1", tier: "jwst", band: "" },
    ]);
    expect(IMG_CODEC.serialize(list)).toBe(raw);
  });

  it("keeps model specs with colons (member:member_170) and drops malformed entries", () => {
    const list = IMG_CODEC.parse("nexus/f200w-0001|m:member:member_170|J_E,bogus,nexus/x|hr|VIS,/x|lr|VIS,nexus/a b|lr|VIS")!;
    expect(list).toEqual([{ ref: "nexus/f200w-0001", tier: "m:member:member_170", band: "J_E" }]);
  });

  it("clamps opacity, collapses duplicates (last settings win, first position kept), empty → no param", () => {
    expect(IMG_CODEC.parse("nexus/a|lr|VIS@7,nexus/b|lr|VIS,nexus/a|lr|VIS@0.25")).toEqual([
      { ref: "nexus/a", tier: "lr", band: "VIS", opacity: 0.25 },
      { ref: "nexus/b", tier: "lr", band: "VIS" },
    ]);
    expect(IMG_CODEC.parse("nexus/a|lr|VIS@7")).toEqual([{ ref: "nexus/a", tier: "lr", band: "VIS", opacity: 1 }]);
    expect(IMG_CODEC.serialize([])).toBeNull();
    expect(IMG_CODEC.serialize([{ ref: "nexus/a", tier: "lr", band: "VIS", opacity: 1 }])).toBe("nexus/a|lr|VIS");
  });

  it("adding re-shows an existing overlay, appends new ones and caps the list", () => {
    const cur: PixelOverlaySetting[] = [{ ref: "nexus/a", tier: "lr", band: "VIS", hidden: true, opacity: 0.5 }];
    const next = withOverlays(cur, [{ ref: "nexus/a", tier: "lr", band: "VIS" }, { ref: "nexus/b", tier: "m:rbf", band: "VIS" }]);
    expect(next).toEqual([
      { ref: "nexus/a", tier: "lr", band: "VIS", opacity: 0.5 },
      { ref: "nexus/b", tier: "m:rbf", band: "VIS" },
    ]);
    const many = Array.from({ length: MAX_PIXEL_OVERLAYS + 5 }, (_, i) => ({ ref: `nexus/t${i}`, tier: "lr", band: "VIS" }));
    const capped = withOverlays([], many);
    expect(capped).toHaveLength(MAX_PIXEL_OVERLAYS);
    expect(capped[capped.length - 1].ref).toBe(`nexus/t${MAX_PIXEL_OVERLAYS + 4}`); // the newest are kept
  });

  it("patch / remove by key", () => {
    const cur: PixelOverlaySetting[] = [{ ref: "nexus/a", tier: "lr", band: "VIS" }, { ref: "nexus/b", tier: "lr", band: "VIS" }];
    const k = overlayKey("nexus/b", "lr", "VIS");
    expect(patchPixelOverlay(cur, k, { opacity: 0.3, hidden: true })[1]).toEqual({ ref: "nexus/b", tier: "lr", band: "VIS", opacity: 0.3, hidden: true });
    expect(patchPixelOverlay(cur, k, { hidden: false })[1]).toEqual({ ref: "nexus/b", tier: "lr", band: "VIS" });
    expect(withoutOverlay(cur, k)).toEqual([cur[0]]);
  });

  it("resolves the FITS url and a label from ref / tier / band", () => {
    expect(overlayUrl("nexus/f200w-0012", "m:rbf", "VIS")).toBe("/api/real/nexus/f200w-0012/image.fits?tier=m%3Arbf&band=VIS");
    expect(overlayUrl("pair/p1", "jwst", "")).toBe("/api/real/pair/p1/image.fits?tier=jwst");
    expect(tierLabel("lr")).toBe("LR");
    expect(tierLabel("m:member:member_170")).toBe("SR · member:member_170");
    expect(overlayLabel("nexus/f200w-0012", "m:rbf", "Y_E")).toBe("f200w-0012 · SR · rbf · Y");
    const [o] = resolveOverlays([{ ref: "nexus/a", tier: "lr", band: "VIS", hidden: true }]);
    expect(o).toMatchObject({ key: "nexus/a|lr|VIS", visible: false, opacity: 1, url: "/api/real/nexus/a/image.fits?tier=lr&band=VIS" });
  });
});
