import { describe, expect, it } from "vitest";
import { followStep } from "./follow";

describe("followStep (keep a nav-less viewer on one object with the page's tiers)", () => {
  it("waits for the meta", () => {
    expect(followStep(null, "a", ["lr"], false)).toEqual({ kind: "wait" });
    expect(followStep({ id: null, tiers: [] }, "a", ["lr"], false)).toEqual({ kind: "wait" });
  });
  it("moves to the wanted object first", () => {
    expect(followStep({ id: "x", tiers: ["lr"] }, "a", ["lr", "m:mean"], false)).toEqual({ kind: "go", id: "a" });
  });
  it("then restores tiers the viewer dropped on mount, once", () => {
    expect(followStep({ id: "a", tiers: ["lr"] }, "a", ["lr", "m:mean"], false)).toEqual({ kind: "tiers", tiers: ["lr", "m:mean"] });
    expect(followStep({ id: "a", tiers: ["lr"] }, "a", ["lr", "m:mean"], true)).toEqual({ kind: "done" });
    expect(followStep({ id: "a", tiers: ["lr", "m:mean"] }, "a", ["lr", "m:mean"], false)).toEqual({ kind: "done" });
  });
  it("without a wanted id or tiers there is nothing to do", () => {
    expect(followStep({ id: "a", tiers: ["lr"] }, "", null, false)).toEqual({ kind: "done" });
    expect(followStep({ id: "a", tiers: undefined as never }, "a", ["lr"], false)).toEqual({ kind: "tiers", tiers: ["lr"] });
  });
});
