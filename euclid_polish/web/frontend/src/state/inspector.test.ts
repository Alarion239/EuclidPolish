import { beforeEach, describe, expect, it } from "vitest";
import {
  HISTORY_LIMIT,
  INSPECTOR_STORAGE_KEY,
  formatInspectParam,
  parseInspectParam,
  sameTarget,
  useInspector,
} from "./inspector";

const s = () => useInspector.getState();
const member = (n: number) => ({ kind: "member", id: `member_${n}` });

beforeEach(() => { localStorage.clear(); s().reset(); });

describe("inspect URL param", () => {
  it("round-trips kind:id, splitting at the first colon only", () => {
    expect(formatInspectParam({ kind: "member", id: "member_196" })).toBe("member:member_196");
    expect(parseInspectParam("tile:nexus/123")).toEqual({ kind: "tile", id: "nexus/123" });
    expect(parseInspectParam("fits:/data/a:b.fits")).toEqual({ kind: "fits", id: "/data/a:b.fits" });
    expect(parseInspectParam("job:local/ab12")).toEqual({ kind: "job", id: "local/ab12" });
  });

  it("rejects malformed values", () => {
    for (const bad of [null, undefined, "", "member", ":x", "member:", "Bad Kind:x", "k k:x"]) {
      expect(parseInspectParam(bad as string | null), String(bad)).toBeNull();
    }
  });

  it("compares targets by value", () => {
    expect(sameTarget(member(1), member(1))).toBe(true);
    expect(sameTarget(member(1), member(2))).toBe(false);
    expect(sameTarget(null, member(1))).toBe(false);
  });
});

describe("inspector store", () => {
  it("starts closed and empty", () => {
    expect(s()).toMatchObject({ open: false, current: null, back: [], forward: [], pinned: [] });
  });

  it("show() opens a target and pushes the previous one onto the back stack", () => {
    s().show(member(1));
    s().show(member(2));
    expect(s().open).toBe(true);
    expect(s().current).toEqual(member(2));
    expect(s().back).toEqual([member(1)]);
    s().show(member(2));                                  // same target: no new history entry
    expect(s().back).toEqual([member(1)]);
  });

  it("goBack / goForward walk the history; a new show clears forward", () => {
    s().show(member(1));
    s().show(member(2));
    s().show(member(3));
    s().goBack();
    expect(s().current).toEqual(member(2));
    s().goBack();
    expect(s().current).toEqual(member(1));
    s().goBack();                                          // nothing further back
    expect(s().current).toEqual(member(1));
    s().goForward();
    expect(s().current).toEqual(member(2));
    expect(s().forward).toEqual([member(3)]);
    s().show(member(9));
    expect(s().forward).toEqual([]);
    expect(s().back).toEqual([member(1), member(2)]);
  });

  it("show(target, {replace}) swaps the current entry without history", () => {
    s().show(member(1));
    s().show(member(2), { replace: true });
    expect(s().current).toEqual(member(2));
    expect(s().back).toEqual([]);
  });

  it("caps the history", () => {
    for (let i = 0; i < HISTORY_LIMIT + 10; i++) s().show(member(i));
    expect(s().back.length).toBe(HISTORY_LIMIT);
    expect(s().back[0]).toEqual(member(9));             // 0…58 behind current 59 → last 50
  });

  it("hide() closes but keeps the current target; setOpen reopens it", () => {
    s().show(member(1));
    s().hide();
    expect(s().open).toBe(false);
    expect(s().current).toEqual(member(1));
    s().setOpen(true);
    expect(s().open).toBe(true);
  });

  it("clear() closes and forgets the current target and history", () => {
    s().show(member(1));
    s().show(member(2));
    s().clear();
    expect(s()).toMatchObject({ open: false, current: null, back: [], forward: [] });
  });

  it("togglePin pins/unpins (current by default) and persists the pins", () => {
    s().show(member(1));
    s().togglePin();
    s().togglePin(member(5));
    expect(s().pinned).toEqual([member(1), member(5)]);
    expect(s().isPinned(member(5))).toBe(true);
    s().togglePin(member(1));
    expect(s().pinned).toEqual([member(5)]);
    const saved = JSON.parse(localStorage.getItem(INSPECTOR_STORAGE_KEY)!);
    expect(saved.state).toEqual({ pinned: [member(5)] });
  });
});
