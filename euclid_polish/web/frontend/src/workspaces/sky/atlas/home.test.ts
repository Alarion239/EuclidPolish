import { describe, expect, it } from "vitest";
import { LAST_TILE_KEY, Q1_HOME, atlasHome, parseLastTile, rememberTile } from "./home";

describe("the atlas's home framing", () => {
  it("opens on EDF-N (a Q1 deep field), not the all-sky view, when nothing was inspected", () => {
    expect(atlasHome()).toEqual(Q1_HOME);
    expect(Q1_HOME.fov).toBeLessThan(30);
  });

  it("opens on the last inspected tile, framed a few times its size", () => {
    rememberTile("nexus/f200w-0040", 268.47, 65.14, 25.5 / 3600);
    const home = atlasHome();
    expect(home).toMatchObject({ ra: 268.47, dec: 65.14, ref: "nexus/f200w-0040" });
    expect(home.fov).toBeGreaterThan(0.03);
    expect(home.fov).toBeLessThan(0.1);
  });

  it("ignores a malformed or out-of-range memory", () => {
    for (const raw of ["", "{", "[]", '{"ra":1}', '{"ra":400,"dec":0,"fov":1}', '{"ra":1,"dec":95,"fov":1}', '{"ra":1,"dec":0,"fov":0}']) {
      expect(parseLastTile(raw)).toBeNull();
    }
    localStorage.setItem(LAST_TILE_KEY, "{bad");
    expect(atlasHome()).toEqual(Q1_HOME);
  });

  it("does not remember a tile without a position", () => {
    rememberTile("eval/x", null, 1, 0.01);
    expect(localStorage.getItem(LAST_TILE_KEY)).toBeNull();
  });
});
