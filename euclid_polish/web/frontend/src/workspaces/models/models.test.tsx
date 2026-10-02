/* The Models workspace (`/models/<tab>`): a bare `/models` (a deep link, the
 * palette) returns to the last tab visited, and nothing sits beside the tab
 * strip (no regime switch). The tab modules are stubbed so no data loads. */
import { act, render, screen } from "@testing-library/react";
import { RouterProvider, createMemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import ModelsWorkspace from "./index";

vi.mock("./tabs/Leaderboard", () => ({ default: () => <p>tab:leaderboard</p> }));
vi.mock("./tabs/Members", () => ({ default: () => <p>tab:members</p> }));
vi.mock("./tabs/Train", () => ({ default: () => <p>tab:train</p> }));
vi.mock("./tabs/Combiner", () => ({ default: () => <p>tab:combiner</p> }));
vi.mock("./tabs/Diagnostics", () => ({ default: () => <p>tab:diagnostics</p> }));
vi.mock("./tabs/Images", () => ({ default: () => <p>tab:images</p> }));

function go(url: string) {
  const router = createMemoryRouter([{ path: "/models/*", element: <ModelsWorkspace /> }], {
    initialEntries: [url],
    future: { v7_relativeSplatPath: true },
  });
  render(<RouterProvider router={router} future={{ v7_startTransition: true }} />);
  return router;
}

describe("models workspace", () => {
  it("opens a bare /models on the Leaderboard", async () => {
    const router = go("/models");
    await screen.findByText("tab:leaderboard");
    expect(router.state.location.pathname).toBe("/models/leaderboard");
  });

  it("returns a bare /models to the last tab visited, keeping the query", async () => {
    const router = go("/models/train");
    await screen.findByText("tab:train");
    await act(() => router.navigate("/models?inspect=job%3Alocal%2Fa"));
    await screen.findByText("tab:train");
    expect(router.state.location.pathname + router.state.location.search).toBe("/models/train?inspect=job%3Alocal%2Fa");
    await act(() => router.navigate("/models/combiner"));
    await act(() => router.navigate("/models"));
    await screen.findByText("tab:combiner");
    expect(router.state.location.pathname).toBe("/models/combiner");
  });

  it("puts nothing beside the tab strip: no regime switch, no member popover", async () => {
    go("/models/images?inspect=job%3Alocal%2Fa");
    await screen.findByText("tab:images");
    expect(screen.queryByRole("radiogroup", { name: "Star regime" })).toBeNull();
    expect(screen.queryByRole("radio", { name: /starless|starfull/ })).toBeNull();
    expect(screen.queryByRole("button", { name: /^Members:/ })).toBeNull();
    expect(document.querySelector(".ws__aside")).toBeNull();
  });
});
