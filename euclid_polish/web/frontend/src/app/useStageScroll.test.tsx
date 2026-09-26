import { act, fireEvent, render } from "@testing-library/react";
import { Outlet, RouterProvider, createMemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { useStageScroll } from "./useStageScroll";

function Layout() {
  const ref = useStageScroll<HTMLDivElement>();
  return <div ref={ref} data-testid="stage" style={{ overflow: "auto", height: 100 }}><Outlet /></div>;
}

function setup(url: string) {
  const router = createMemoryRouter([{ Component: Layout, children: [{ path: "*", element: <p>page</p> }] }], {
    initialEntries: [url], future: { v7_relativeSplatPath: true },
  });
  const r = render(<RouterProvider router={router} future={{ v7_startTransition: true }} />);
  const stage = r.getByTestId("stage");
  const scrollTo = (y: number) => { stage.scrollTop = y; fireEvent.scroll(stage); };
  return { router, stage, scrollTo };
}

describe("useStageScroll", () => {
  it("resets on a new pathname, keeps the scroll on a query-only change", async () => {
    const { router, stage, scrollTo } = setup("/sky/atlas");
    scrollTo(240);
    await act(() => router.navigate("/sky/atlas?ra=10", { replace: true }));
    expect(stage.scrollTop).toBe(240);
    await act(() => router.navigate("/sky/results"));
    expect(stage.scrollTop).toBe(0);
  });

  it("restores the position of a history entry on back", async () => {
    const { router, stage, scrollTo } = setup("/data/records");
    scrollTo(300);
    await act(() => router.navigate("/data/psfs"));
    expect(stage.scrollTop).toBe(0);
    scrollTo(50);
    await act(() => router.navigate(-1));
    expect(stage.scrollTop).toBe(300);
    await act(() => router.navigate(1));
    expect(stage.scrollTop).toBe(50);
  });
});
