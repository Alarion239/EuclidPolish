import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useShortcutRegistry } from "../hooks/useShortcut";
import {
  paletteSuggestions, usePageActions, usePaletteActions, usePaletteRegistry, type PageAction,
} from "./palette";

afterEach(() => {
  usePaletteRegistry.getState().reset();
  useShortcutRegistry.getState().reset();
});

describe("usePageActions", () => {
  it("registers the page's actions while mounted", () => {
    const run = vi.fn();
    const { unmount } = renderHook(() => usePageActions([
      { id: "evaluate", label: "Evaluate on test set", group: "Ensemble", run },
    ]));
    const listed = renderHook(() => usePaletteActions());
    expect(listed.result.current.map((a) => a.label)).toEqual(["Evaluate on test set"]);
    act(() => listed.result.current[0].run());
    expect(run).toHaveBeenCalledTimes(1);
    unmount();
    expect(usePaletteRegistry.getState().list()).toEqual([]);
  });

  it("runs the latest closure without re-registering on every render", () => {
    const first = vi.fn(); const second = vi.fn();
    const set = vi.spyOn(usePaletteRegistry.getState(), "set");
    const { rerender } = renderHook(({ run }: { run: () => void }) => usePageActions([
      { id: "a", label: "A", run },
    ]), { initialProps: { run: first } });
    rerender({ run: second });
    rerender({ run: second });
    expect(set).toHaveBeenCalledTimes(1);
    usePaletteRegistry.getState().list()[0].run();
    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledTimes(1);
  });

  it("re-registers when labels or the list change", () => {
    const { rerender } = renderHook(({ actions }: { actions: PageAction[] }) => usePageActions(actions), {
      initialProps: { actions: [{ id: "a", label: "A", run: () => {} }] },
    });
    rerender({ actions: [{ id: "a", label: "A2", run: () => {} }, { id: "b", label: "B", run: () => {} }] });
    expect(usePaletteRegistry.getState().list().map((a) => a.label)).toEqual(["A2", "B"]);
  });

  it("keeps several pages' actions apart", () => {
    const a = renderHook(() => usePageActions([{ id: "x", label: "X", run: () => {} }]));
    renderHook(() => usePageActions([{ id: "y", label: "Y", run: () => {} }]));
    expect(usePaletteRegistry.getState().list().map((x) => x.label)).toEqual(["X", "Y"]);
    a.unmount();
    expect(usePaletteRegistry.getState().list().map((x) => x.label)).toEqual(["Y"]);
  });

  it("binds an action's shortcut and lists it in the ? sheet", () => {
    const run = vi.fn();
    const { unmount } = renderHook(() => usePageActions([
      { id: "e", label: "Evaluate", group: "Ensemble", shortcut: "Shift+E", run },
    ]));
    expect(useShortcutRegistry.getState().entries.map((e) => [e.combo, e.description, e.scope]))
      .toEqual([["Shift+E", "Evaluate", "Ensemble"]]);
    act(() => { window.dispatchEvent(new KeyboardEvent("keydown", { key: "E", code: "KeyE", shiftKey: true, bubbles: true })); });
    expect(run).toHaveBeenCalledTimes(1);
    unmount();
    expect(useShortcutRegistry.getState().entries).toEqual([]);
  });

  it("does not run a disabled action from its shortcut", () => {
    const run = vi.fn();
    renderHook(() => usePageActions([{ id: "e", label: "E", shortcut: "Shift+E", run, disabled: true }]));
    act(() => { window.dispatchEvent(new KeyboardEvent("keydown", { key: "E", code: "KeyE", shiftKey: true, bubbles: true })); });
    expect(run).not.toHaveBeenCalled();
  });
});

describe("paletteSuggestions", () => {
  const parse = (t: string) => (/^(\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)$/.test(t)
    ? { ra: Number(t.split(/\s+/)[0]), dec: Number(t.split(/\s+/)[1]) } : null);

  it("offers nothing for an empty query", () => {
    expect(paletteSuggestions("  ", parse)).toEqual([]);
  });

  it("turns coordinates into a sky jump", () => {
    const [s] = paletteSuggestions("269.2 66.1", parse);
    expect(s).toMatchObject({ kind: "navigate", to: "/sky/atlas?ra=269.2&dec=66.1" });
  });

  it("opens members, NEXUS tiles and FITS files", () => {
    expect(paletteSuggestions("member 196", parse)[0]).toMatchObject({ kind: "inspect", target: { kind: "member", id: "member_196" } });
    expect(paletteSuggestions("member_171", parse)[0]).toMatchObject({ target: { id: "member_171" } });
    expect(paletteSuggestions("nexus 12", parse)[0]).toMatchObject({ target: { kind: "tile", id: "nexus/12" } });
    expect(paletteSuggestions("tile#7", parse)[0]).toMatchObject({ target: { id: "nexus/7" } });
    expect(paletteSuggestions("data/eval_results/a b.fits", parse)[0])
      .toMatchObject({ kind: "navigate", to: "/inspect?fits=data%2Feval_results%2Fa%20b.fits" });
  });

  it("falls back to a sky name lookup for other text", () => {
    expect(paletteSuggestions("M 87", parse)[0]).toMatchObject({ to: "/sky/atlas?goto=M%2087" });
    expect(paletteSuggestions("7", parse)).toEqual([]);
  });
});
