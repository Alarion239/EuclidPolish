/* The Models workspace (`/models/:mode`)'s ONE regime switch keeps the tab,
 * and a bare `/models/<mode>` (a deep link, the palette) returns to the last
 * tab visited. The tab modules are stubbed so no data loads. */
import { act, fireEvent, render, screen } from "@testing-library/react";
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
  const router = createMemoryRouter([{ path: "/models/:mode/*", element: <ModelsWorkspace /> }], {
    initialEntries: [url],
    future: { v7_relativeSplatPath: true },
  });
  render(<RouterProvider router={router} future={{ v7_startTransition: true }} />);
  return router;
}

describe("models workspace", () => {
  it("opens a bare regime path on the Leaderboard", async () => {
    const router = go("/models/starless");
    await screen.findByText("tab:leaderboard");
    expect(router.state.location.pathname).toBe("/models/starless/leaderboard");
  });

  it("keeps the tab when a link switches the regime (bare /models/<mode>)", async () => {
    const router = go("/models/starfull/train?inspect=job%3Alocal%2Fa");
    await screen.findByText("tab:train");
    await act(() => router.navigate("/models/starless"));
    await screen.findByText("tab:train");
    expect(router.state.location.pathname).toBe("/models/starless/train");
    await act(() => router.navigate("/models/starless/combiner"));
    await act(() => router.navigate("/models/starfull"));
    await screen.findByText("tab:combiner");
    expect(router.state.location.pathname).toBe("/models/starfull/combiner");
  });

  it("keeps the tab and ?inspect= with the regime switch, and nothing else sits in the strip", async () => {
    const router = go("/models/starfull/images?inspect=job%3Alocal%2Fa&other=1");
    await screen.findByText("tab:images");
    expect(screen.queryByRole("button", { name: /^Members:/ })).toBeNull();     // no member popover in the tab strip
    fireEvent.click(screen.getByRole("radio", { name: "starless" }));
    await screen.findByText("tab:images");
    expect(router.state.location.pathname + router.state.location.search)
      .toBe("/models/starless/images?inspect=job%3Alocal%2Fa");
  });
});
