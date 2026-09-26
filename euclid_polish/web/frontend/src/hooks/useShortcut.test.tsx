import { act, fireEvent, render, renderHook } from "@testing-library/react";
import { useRef } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  bindShortcut, comboParts, useShortcut, useShortcutRegistry, type ShortcutEntry,
} from "./useShortcut";

const key = (k: string, init: KeyboardEventInit = {}, target: EventTarget = window) => {
  const ev = new KeyboardEvent("keydown", { key: k, code: init.code ?? `Key${k.toUpperCase()}`, bubbles: true, cancelable: true, ...init });
  act(() => { target.dispatchEvent(ev); });
  return ev;
};

afterEach(() => {
  useShortcutRegistry.getState().reset();
});

describe("useShortcut", () => {
  it("fires on its combo and unbinds on unmount", () => {
    const fn = vi.fn();
    const { unmount } = renderHook(() => useShortcut("Shift+?", fn, { description: "Help" }));
    key("?", { shiftKey: true, code: "Slash" });
    expect(fn).toHaveBeenCalledTimes(1);
    unmount();
    key("?", { shiftKey: true, code: "Slash" });
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("supports sequences (g s)", () => {
    const fn = vi.fn();
    renderHook(() => useShortcut("g s", fn, { description: "Go to Sky" }));
    key("g");
    expect(fn).not.toHaveBeenCalled();
    key("s");
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("ignores typing in inputs unless allowInInputs", () => {
    const plain = vi.fn();
    const mod = vi.fn();
    renderHook(() => {
      useShortcut("x", plain, { description: "plain" });
      useShortcut("$mod+k", mod, { description: "palette", allowInInputs: true });
    });
    const input = document.createElement("input");
    document.body.appendChild(input);
    key("x", {}, input);
    expect(plain).not.toHaveBeenCalled();
    const mac = /Mac|iPod|iPhone|iPad/.test(navigator.platform);
    key("k", mac ? { metaKey: true } : { ctrlKey: true }, input);
    expect(mod).toHaveBeenCalledTimes(1);
    input.remove();
  });

  it("skips events another handler already consumed (preventDefault)", () => {
    const fn = vi.fn();
    renderHook(() => useShortcut("q", fn, { description: "q" }));
    const stop = (e: Event) => e.preventDefault();
    document.addEventListener("keydown", stop);
    key("q", {}, document.body);
    document.removeEventListener("keydown", stop);
    expect(fn).not.toHaveBeenCalled();
    key("q", {}, document.body);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("ignores plain keys inside a modal dialog", () => {
    const fn = vi.fn();
    renderHook(() => useShortcut("g h", fn, { description: "home" }));
    const dlg = document.createElement("div");
    dlg.setAttribute("role", "dialog");
    const btn = document.createElement("button");
    dlg.appendChild(btn);
    document.body.appendChild(dlg);
    key("g", {}, btn); key("h", {}, btn);
    expect(fn).not.toHaveBeenCalled();
    dlg.remove();
  });

  it("uses the latest handler without rebinding", () => {
    const a = vi.fn(); const b = vi.fn();
    const { rerender } = renderHook(({ h }) => useShortcut("z", h, { description: "z" }), { initialProps: { h: a } });
    rerender({ h: b });
    key("z");
    expect(a).not.toHaveBeenCalled();
    expect(b).toHaveBeenCalledTimes(1);
  });

  it("binds to an element target (scoped shortcuts)", () => {
    const fn = vi.fn();
    function Scoped() {
      const ref = useRef<HTMLDivElement>(null);
      useShortcut("ArrowRight", fn, { description: "next", target: ref, scope: "Viewer" });
      return <div ref={ref} tabIndex={0} data-testid="scope" />;
    }
    const { getByTestId } = render(<Scoped />);
    key("ArrowRight", { code: "ArrowRight" });
    expect(fn).not.toHaveBeenCalled();
    fireEvent.keyDown(getByTestId("scope"), { key: "ArrowRight", code: "ArrowRight" });
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("does nothing while disabled", () => {
    const fn = vi.fn();
    renderHook(() => useShortcut("y", fn, { description: "y", enabled: false }));
    key("y");
    expect(fn).not.toHaveBeenCalled();
    expect(useShortcutRegistry.getState().entries).toHaveLength(0);
  });
});

describe("registry (the ? sheet)", () => {
  it("lists registered shortcuts with scope and removes them on unbind", () => {
    const off = bindShortcut("g s", () => {}, { description: "Go to Sky", scope: "Navigation" });
    const off2 = bindShortcut("$mod+k", () => {}, { description: "Command palette" });
    const entries: ShortcutEntry[] = useShortcutRegistry.getState().entries;
    expect(entries.map((e) => [e.combo, e.description, e.scope])).toEqual([
      ["g s", "Go to Sky", "Navigation"],
      ["$mod+k", "Command palette", "Global"],
    ]);
    off();
    expect(useShortcutRegistry.getState().entries.map((e) => e.combo)).toEqual(["$mod+k"]);
    off2();
    expect(useShortcutRegistry.getState().entries).toEqual([]);
  });

  it("hides entries marked hidden from the sheet listing", () => {
    const off = bindShortcut("Escape", () => {}, { description: "close", hidden: true });
    expect(useShortcutRegistry.getState().entries[0].hidden).toBe(true);
    off();
  });
});

describe("comboParts", () => {
  it("splits a combo into key presses and keys for <Kbd>", () => {
    expect(comboParts("$mod+k")).toEqual(["mod+k"]);
    expect(comboParts("g s")).toEqual(["g", "s"]);
    expect(comboParts("Shift+?")).toEqual(["?"]);
    expect(comboParts("[Shift]+?")).toEqual(["?"]);
    expect(comboParts("$mod+Shift+p")).toEqual(["mod+shift+p"]);
  });
});
