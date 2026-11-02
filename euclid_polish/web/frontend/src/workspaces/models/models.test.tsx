/* The Models workspace (`/models/<tab>`): a bare `/models` (a deep link, the
 * palette) returns to the last tab visited, and nothing sits beside the tab
 * strip (no regime switch). The tab modules are stubbed so no data loads. */
import { act, render, screen } from "@testing-library/react";
import { RouterProvider, createMemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import ModelsWorkspace from "./index";

vi.mock("./tabs/Universal", () => ({ default: () => <p>tab:universal</p> }));
vi.mock("./tabs/Knees", () => ({ default: () => <p>tab:knees</p> }));
vi.mock("./tabs/Spectra", () => ({ default: () => <p>tab:spectra</p> }));
vi.mock("./tabs/Train", () => ({ default: () => <p>tab:train</p> }));

function go(url: string) {
  const router = createMemoryRouter([{ path: "/models/*", element: <ModelsWorkspace /> }], {
    initialEntries: [url],
    future: { v7_relativeSplatPath: true },
  });
  render(<RouterProvider router={router} future={{ v7_startTransition: true }} />);
  return router;
}

describe("models workspace", () => {
  it("opens a bare /models on Universal, the first of its four tabs", async () => {
    const router = go("/models");
    await screen.findByText("tab:universal");
    expect(router.state.location.pathname).toBe("/models/universal");
    expect(screen.getAllByRole("link").map((a) => a.textContent)).toEqual(["Universal", "Knees", "Spectra", "Train"]);
  });

  it("returns a bare /models to the last tab visited, keeping the query", async () => {
    const router = go("/models/train");
    await screen.findByText("tab:train");
    await act(() => router.navigate("/models?inspect=job%3Alocal%2Fa"));
    await screen.findByText("tab:train");
    expect(router.state.location.pathname + router.state.location.search).toBe("/models/train?inspect=job%3Alocal%2Fa");
    await act(() => router.navigate("/models/spectra"));
    await act(() => router.navigate("/models"));
    await screen.findByText("tab:spectra");
    expect(router.state.location.pathname).toBe("/models/spectra");
  });

  it("puts nothing beside the tab strip: no regime switch, no member popover", async () => {
    go("/models/knees?inspect=job%3Alocal%2Fa");
    await screen.findByText("tab:knees");
    expect(screen.queryByRole("radiogroup", { name: "Star regime" })).toBeNull();
    expect(screen.queryByRole("radio", { name: /starless|starfull/ })).toBeNull();
    expect(screen.queryByRole("button", { name: /^Members:/ })).toBeNull();
    expect(document.querySelector(".ws__aside")).toBeNull();
  });
});
