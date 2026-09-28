import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { holdInView } from "./arrival";

describe("holdInView", () => {
  let top = 0;
  let node: HTMLElement;
  beforeEach(() => {
    vi.useFakeTimers();
    node = document.createElement("section");
    node.id = "syn-drawer-prior";
    document.body.append(node);
    top = 600;
    node.getBoundingClientRect = () => ({ top } as DOMRect);
    node.scrollIntoView = vi.fn(() => { top = 0; });
  });
  afterEach(() => { vi.useRealTimers(); node.remove(); });

  it("re-aligns the section while content above it keeps loading", () => {
    holdInView("syn-drawer-prior");
    expect(node.scrollIntoView).toHaveBeenCalledTimes(1);
    top = 420;                           // a figure above finished loading
    vi.advanceTimersByTime(300);
    expect(node.scrollIntoView).toHaveBeenCalledTimes(2);
    vi.advanceTimersByTime(300);         // nothing moved: no extra scroll
    expect(node.scrollIntoView).toHaveBeenCalledTimes(2);
  });

  it("lets go as soon as the user scrolls, and after 5 s", () => {
    holdInView("syn-drawer-prior");
    window.dispatchEvent(new Event("wheel"));
    top = 300;
    vi.advanceTimersByTime(1000);
    expect(node.scrollIntoView).toHaveBeenCalledTimes(1);

    holdInView("syn-drawer-prior");
    vi.advanceTimersByTime(5100);
    top = 250;
    vi.advanceTimersByTime(1000);
    expect(node.scrollIntoView).toHaveBeenCalledTimes(2);
  });
});
