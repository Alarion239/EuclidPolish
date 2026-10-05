import { act, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it } from "vitest";
import { useInspector } from "../state/inspector";
import { openInspector, useInspectorRegistry } from "./inspector";
import { InspectorPanel } from "./InspectorPanel";

afterEach(() => {
  useInspector.getState().reset();
  useInspectorRegistry.getState().reset();
});

describe("<InspectorPanel>", () => {
  it("explains an unknown kind without promising a workspace will register it later", () => {
    render(
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <InspectorPanel />
      </MemoryRouter>,
    );
    act(() => openInspector({ kind: "nosuchkind", id: "x/1" }));
    expect(screen.getByText("No inspector for “nosuchkind”")).toBeTruthy();
    const text = document.querySelector(".insp-unknown")!.textContent ?? "";
    // every workspace registers its kinds at app start (Shell.tsx), not when it loads
    expect(text).not.toMatch(/when it loads/);
    expect(text).toMatch(/at start/);
  });
});
