import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { useIsSelected, useSelected, useSelection } from "./selection";

const s = () => useSelection.getState();

beforeEach(() => { s().clear(); });

describe("selection store", () => {
  it("keeps one ordered, de-duplicated id list per scope", () => {
    s().select("member", ["m1", "m2", "m1"]);
    s().select("tile", ["nexus/1"]);
    expect(s().get("member")).toEqual(["m1", "m2"]);
    expect(s().get("tile")).toEqual(["nexus/1"]);
    expect(s().get("nothing")).toEqual([]);
  });

  it("add / remove / toggle", () => {
    s().add("member", ["m1", "m2"]);
    s().add("member", ["m2", "m3"]);
    expect(s().get("member")).toEqual(["m1", "m2", "m3"]);
    s().remove("member", ["m2"]);
    expect(s().get("member")).toEqual(["m1", "m3"]);
    s().toggle("member", "m1");
    s().toggle("member", "m9");
    expect(s().get("member")).toEqual(["m3", "m9"]);
    expect(s().has("member", "m9")).toBe(true);
    expect(s().has("member", "m1")).toBe(false);
  });

  it("clear(scope) empties one scope; clear() empties all", () => {
    s().select("member", ["m1"]);
    s().select("tile", ["t1"]);
    s().clear("member");
    expect(s().get("member")).toEqual([]);
    expect(s().get("tile")).toEqual(["t1"]);
    s().clear();
    expect(s().sets).toEqual({});
  });

  it("does not notify subscribers of an unchanged scope", () => {
    s().select("member", ["m1"]);
    const before = s().sets;
    s().select("member", ["m1"]);
    expect(s().sets).toBe(before);
    s().remove("tile", ["x"]);
    expect(s().sets).toBe(before);
  });

  it("hooks re-render with a stable empty list", () => {
    const { result } = renderHook(() => ({ ids: useSelected("tile"), on: useIsSelected("tile", "t2") }));
    const empty = result.current.ids;
    expect(empty).toEqual([]);
    act(() => s().select("member", ["m1"]));
    expect(result.current.ids).toBe(empty);
    act(() => s().add("tile", ["t1", "t2"]));
    expect(result.current.ids).toEqual(["t1", "t2"]);
    expect(result.current.on).toBe(true);
  });

  it("hands out read-only lists: typed readonly, and a cast mutation throws instead of corrupting the store", () => {
    const none = s().get("none");
    // @ts-expect-error — selection lists are readonly
    expect(() => none.push("x")).toThrow(TypeError);
    s().select("member", ["m1"]);
    const got = s().get("member");
    // @ts-expect-error — selection lists are readonly
    expect(() => got.push("m2")).toThrow(TypeError);
    expect(s().get("member")).toEqual(["m1"]);
    const { result } = renderHook(() => useSelected("member"));
    // @ts-expect-error — selection lists are readonly
    expect(() => result.current.push("m3")).toThrow(TypeError);
    expect(s().get("member")).toEqual(["m1"]);
  });
});
