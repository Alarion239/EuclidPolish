import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { MANIFEST } from "./manifest";
import { Workspace, defineTabs } from "./workspace";

const ROUTER_FLAGS = { v7_startTransition: true, v7_relativeSplatPath: true };

describe("<Workspace>", () => {
  it("shows Not found for a manifest tab its workspace defines no module for", async () => {
    const ws = MANIFEST.workspaces.find((w) => w.id === "synthetic")!;
    const [missing, ...rest] = ws.tabs;
    const tabs = defineTabs(ws.id, Object.fromEntries(rest.map((t) => [t, {
      load: async () => ({ default: () => <p>{`${ws.id}:${t}`}</p> }),
    }])));
    render(
      <MemoryRouter initialEntries={[`/synthetic/${missing}`]} future={ROUTER_FLAGS}>
        <Workspace id="synthetic" tabs={tabs} />
      </MemoryRouter>,
    );
    expect(await screen.findByText("No page here")).toBeTruthy();
    expect(screen.queryByText(/arrives in phase 3|not built yet/)).toBeNull();
    // exactly one (visually hidden) h1, as for any unknown URL
    const h1s = document.querySelectorAll("h1");
    expect(h1s.length).toBe(1);
    expect(h1s[0].textContent).toBe("Not found");
  });
});
