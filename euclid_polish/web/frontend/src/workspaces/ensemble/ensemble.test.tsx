/* The ensemble workspace's regime switching: the workspace switch keeps the
 * tab, and so does the legacy page's own switch (it navigates to the bare
 * `/ensemble/<mode>`, which returns to the last tab visited). The tab modules
 * are stubbed so the legacy pages do not load. */
import { act, fireEvent, render, screen } from "@testing-library/react";
import { RouterProvider, createMemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import EnsembleWorkspace from "./index";

vi.mock("./tabs/Overview", () => ({ default: () => <p>tab:overview</p> }));
vi.mock("./tabs/Members", () => ({ default: () => <p>tab:members</p> }));
vi.mock("./tabs/Curves", () => ({ default: () => <p>tab:curves</p> }));
vi.mock("./tabs/Knee", () => ({ default: () => <p>tab:knee</p> }));
vi.mock("./tabs/Diagnostics", () => ({ default: () => <p>tab:diagnostics</p> }));
vi.mock("./tabs/Combiners", () => ({ default: () => <p>tab:combiners</p> }));
vi.mock("./tabs/Disagreement", () => ({ default: () => <p>tab:disagreement</p> }));
vi.mock("./tabs/Train", () => ({ default: () => <p>tab:train</p> }));

function go(url: string) {
  const router = createMemoryRouter([{ path: "/ensemble/:mode/*", element: <EnsembleWorkspace /> }], {
    initialEntries: [url],
    future: { v7_relativeSplatPath: true },
  });
  render(<RouterProvider router={router} future={{ v7_startTransition: true }} />);
  return router;
}

describe("ensemble workspace", () => {
  it("opens a bare regime path on the default tab", async () => {
    const router = go("/ensemble/starless");
    await screen.findByText("tab:overview");
    expect(router.state.location.pathname).toBe("/ensemble/starless/overview");
  });

  it("keeps the tab when the legacy page switches the regime (bare /ensemble/<mode>)", async () => {
    const router = go("/ensemble/starfull/knee?inspect=job%3Alocal%2Fa");
    await screen.findByText("tab:knee");
    await act(() => router.navigate("/ensemble/starless"));
    await screen.findByText("tab:knee");
    expect(router.state.location.pathname).toBe("/ensemble/starless/knee");
    await act(() => router.navigate("/ensemble/starless/curves"));
    await act(() => router.navigate("/ensemble/starfull"));
    await screen.findByText("tab:curves");
    expect(router.state.location.pathname).toBe("/ensemble/starfull/curves");
  });

  it("keeps the tab and ?inspect= with the workspace switch", async () => {
    const router = go("/ensemble/starfull/combiners?inspect=job%3Alocal%2Fa&other=1");
    await screen.findByText("tab:combiners");
    fireEvent.click(screen.getByRole("radio", { name: "starless" }));
    await screen.findByText("tab:combiners");
    expect(router.state.location.pathname + router.state.location.search)
      .toBe("/ensemble/starless/combiners?inspect=job%3Alocal%2Fa");
  });
});
