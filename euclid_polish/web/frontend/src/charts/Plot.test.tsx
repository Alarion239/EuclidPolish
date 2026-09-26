import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import Plot, { Legend, useLegend, type PlotView, type Series } from "./Plot";

const downloads = vi.hoisted(() => ({ text: [] as { name: string; text: string }[] }));
vi.mock("../ui/download", async (orig) => ({
  ...(await orig<typeof import("../ui/download")>()),
  downloadText: (name: string, text: string) => { downloads.text.push({ name, text }); },
}));

afterEach(() => { downloads.text = []; });

/* Without layout (happy-dom) the plot is 640×320 px with default margins
   l=42 r=22 t=12 b=38 → inner 576×270. These helpers map data → client px. */
const W = 640, L = 42, T = 12, IW = 576, IH = 270;
const px = (x: number, [a, b]: [number, number]) => L + ((x - a) / (b - a)) * IW;
const py = (y: number, [a, b]: [number, number]) => T + (1 - (y - a) / (b - a)) * IH;

const X: [number, number] = [0, 10], Y: [number, number] = [0, 100];
const SERIES: Series[] = [
  { x: [0, 5, 10], y: [10, 50, 90], color: "red", name: "mean" },
  { x: [0, 5, 10], y: [20, 30, 40], color: "blue", name: "gate" },
];

const surface = () => document.querySelector("canvas") as HTMLCanvasElement;
function hover(x: number, y: number) { fireEvent.pointerMove(surface(), { clientX: x, clientY: y, pointerType: "mouse" }); }
const tooltip = () => document.querySelector(".plot__tooltip") as HTMLElement | null;

describe("Plot v2 hover", () => {
  it("shows a crosshair and a tooltip with the nearest series and every line at that x", () => {
    render(<Plot xDomain={X} yDomain={Y} series={SERIES} aria-label="demo" />);
    hover(px(5, X) + 2, py(31, Y));
    const tip = tooltip()!;
    expect(tip).toBeTruthy();
    const rows = within(tip).getAllByRole("listitem", { hidden: true }).map((li) => li.textContent);
    expect(rows).toEqual(["mean50", "gate30"]);
    expect(tip.querySelector("[data-nearest='true']")!.textContent).toBe("gate30");
    expect(document.querySelector(".plot__cross-x")).toBeTruthy();
    fireEvent.pointerLeave(surface());
    expect(tooltip()).toBeNull();
  });

  it("always lists and marks the nearest series when there are more lines than rows", () => {
    // 26 flat lines y = 4i (m0 … m25); m20 sits at y = 80, the ensemble-mean case.
    const many: Series[] = Array.from({ length: 26 }, (_, i) => ({
      x: [0, 5, 10], y: [4 * i, 4 * i, 4 * i], color: `hsl(${i * 13} 60% 45%)`, name: `m${i}`,
    }));
    render(<Plot xDomain={X} yDomain={Y} series={many} />);
    hover(px(5, X) + 2, py(81, Y));
    const tip = tooltip()!;
    const rows = within(tip).getAllByRole("listitem", { hidden: true }).map((li) => li.textContent);
    // the 8 lines nearest the cursor (not the first 8 in array order), in series order
    expect(rows).toEqual(["m1768", "m1872", "m1976", "m2080", "m2184", "m2288", "m2392", "m2496"]);
    expect(tip.querySelector("[data-nearest='true']")!.textContent).toBe("m2080");
    expect(tip.textContent).toContain("+18 more");
  });

  it("keeps the hovered series in the readout even when it is far from the others", () => {
    const many: Series[] = [
      ...Array.from({ length: 12 }, (_, i) => ({ x: [0, 5, 10], y: [i, i, i], color: "gray", name: `p${i}` })),
      { x: [0, 5, 10], y: [90, 90, 90], color: "red", name: "mean" },
    ];
    render(<Plot xDomain={X} yDomain={Y} series={many} />);
    hover(px(5, X), py(90, Y));
    const tip = tooltip()!;
    const rows = within(tip).getAllByRole("listitem", { hidden: true }).map((li) => li.textContent);
    expect(rows).toHaveLength(8);
    expect(rows).toContain("mean90");
    expect(tip.querySelector("[data-nearest='true']")!.textContent).toBe("mean90");
    expect(tip.textContent).toContain("+5 more");
  });

  it("can turn the tooltip off", () => {
    render(<Plot xDomain={X} yDomain={Y} series={SERIES} tooltip={false} />);
    hover(px(5, X), py(30, Y));
    expect(tooltip()).toBeNull();
  });

  it("reads the heat cell under the cursor", () => {
    render(<Plot xDomain={[0, 2]} yDomain={[0, 2]} series={[]}
      heat={{ z: [[1, 2], [3, 40]], xEdges: [0, 1, 2], yEdges: [0, 1, 2] }} />);
    // heat plots reserve a 74 px right margin → inner width 524
    fireEvent.pointerMove(surface(), { clientX: L + 0.75 * 524, clientY: T + 0.25 * IH });
    expect(tooltip()!.textContent).toContain("40");
  });

  it("links cursors across plots sharing a syncKey", () => {
    render(<>
      <div data-testid="a"><Plot xDomain={X} yDomain={Y} series={SERIES} syncKey="curves" /></div>
      <div data-testid="b"><Plot xDomain={X} yDomain={Y} series={SERIES} syncKey="curves" /></div>
    </>);
    const [a] = document.querySelectorAll("canvas");
    fireEvent.pointerMove(a, { clientX: px(5, X), clientY: py(50, Y) });
    const remote = screen.getByTestId("b").querySelector(".plot__cross-x") as HTMLElement;
    expect(remote).toBeTruthy();
    expect(parseFloat(remote.style.left)).toBeCloseTo(px(5, X), 0);
    fireEvent.pointerLeave(a);
    expect(screen.getByTestId("b").querySelector(".plot__cross-x")).toBeNull();
  });
});

describe("Plot v2 legend", () => {
  it("builds an auto legend whose entries toggle series", () => {
    const onHiddenChange = vi.fn();
    render(<Plot xDomain={X} yDomain={Y} series={SERIES} legend="auto" onHiddenChange={onHiddenChange} />);
    const gate = screen.getByRole("button", { name: "gate" });
    expect(gate.getAttribute("aria-pressed")).toBe("true");
    fireEvent.click(gate);
    expect(gate.getAttribute("aria-pressed")).toBe("false");
    expect(onHiddenChange).toHaveBeenLastCalledWith(["gate"]);
    hover(px(5, X), py(31, Y));
    expect(within(tooltip()!).getAllByRole("listitem", { hidden: true }).map((li) => li.textContent)).toEqual(["mean50"]);
  });

  it("keeps a static legend when legendToggle is false", () => {
    render(<Plot xDomain={X} yDomain={Y} series={SERIES} legend="auto" legendToggle={false} />);
    expect(screen.queryByRole("button", { name: "gate" })).toBeNull();
    expect(screen.getByText("gate")).toBeTruthy();
  });

  it("connects an external Legend through useLegend", () => {
    function Demo() {
      const lg = useLegend();
      return <>
        <Plot xDomain={X} yDomain={Y} series={SERIES} {...lg.plotProps} />
        <Legend items={[{ label: "mean", color: "red" }, { label: "gate", color: "blue" }]} {...lg.legendProps} />
      </>;
    }
    render(<Demo />);
    fireEvent.click(screen.getByRole("button", { name: "mean" }));
    hover(px(5, X), py(50, Y));
    expect(within(tooltip()!).getAllByRole("listitem", { hidden: true }).map((li) => li.textContent)).toEqual(["gate30"]);
  });

  it("renders the compat static Legend unchanged", () => {
    render(<Legend items={[{ label: "LR", color: "red", dash: true }]} />);
    expect(screen.getByText("LR")).toBeTruthy();
    expect(screen.queryByRole("button")).toBeNull();
  });
});

describe("Plot v2 zoom", () => {
  it("box-zooms on drag, suppresses the click, and resets on double-click", () => {
    const onViewChange = vi.fn();
    const onPlotClick = vi.fn();
    render(<Plot xDomain={X} yDomain={Y} series={SERIES} onViewChange={onViewChange} onPlotClick={onPlotClick} />);
    const c = surface();
    fireEvent.pointerDown(c, { clientX: px(2, X), clientY: py(80, Y), button: 0, pointerId: 1 });
    fireEvent.pointerMove(c, { clientX: px(6, X), clientY: py(20, Y), buttons: 1, pointerId: 1 });
    expect(document.querySelector(".plot__box")).toBeTruthy();
    fireEvent.pointerUp(c, { clientX: px(6, X), clientY: py(20, Y), pointerId: 1 });
    fireEvent.click(c, { clientX: px(6, X), clientY: py(20, Y) });
    expect(onPlotClick).not.toHaveBeenCalled();
    const v = onViewChange.mock.lastCall![0] as PlotView;
    expect(v.x![0]).toBeCloseTo(2, 5);
    expect(v.x![1]).toBeCloseTo(6, 5);
    expect(v.y![0]).toBeCloseTo(20, 5);
    expect(v.y![1]).toBeCloseTo(80, 5);
    expect(screen.getByRole("button", { name: "Reset zoom" })).toBeTruthy();
    fireEvent.doubleClick(c);
    expect(onViewChange).toHaveBeenLastCalledWith(null);
    expect(screen.queryByRole("button", { name: "Reset zoom" })).toBeNull();
  });

  it("zooms only the x axis when zoomAxes='x'", () => {
    const onViewChange = vi.fn();
    render(<Plot xDomain={X} yDomain={Y} series={SERIES} zoomAxes="x" onViewChange={onViewChange} />);
    const c = surface();
    fireEvent.pointerDown(c, { clientX: px(2, X), clientY: py(80, Y), button: 0, pointerId: 1 });
    fireEvent.pointerMove(c, { clientX: px(4, X), clientY: py(20, Y), buttons: 1, pointerId: 1 });
    fireEvent.pointerUp(c, { clientX: px(4, X), clientY: py(20, Y), pointerId: 1 });
    const v = onViewChange.mock.lastCall![0] as PlotView;
    expect(v.y).toBeNull();
    expect(v.x![1]).toBeCloseTo(4, 5);
  });

  it("wheel-zooms with Ctrl held and not otherwise", () => {
    const onViewChange = vi.fn();
    render(<Plot xDomain={X} yDomain={Y} series={SERIES} onViewChange={onViewChange} />);
    const c = surface();
    // happy-dom's WheelEvent drops the MouseEvent init fields: set them by hand.
    const wheel = (init: { ctrlKey?: boolean }) => {
      const ev = new WheelEvent("wheel", { deltaY: -100, bubbles: true, cancelable: true });
      Object.defineProperties(ev, {
        ctrlKey: { value: !!init.ctrlKey }, metaKey: { value: false },
        clientX: { value: px(5, X) }, clientY: { value: py(50, Y) },
      });
      act(() => { c.dispatchEvent(ev); });
      return ev;
    };
    expect(wheel({}).defaultPrevented).toBe(false);
    expect(onViewChange).not.toHaveBeenCalled();
    expect(wheel({ ctrlKey: true }).defaultPrevented).toBe(true);
    const v = onViewChange.mock.lastCall![0] as PlotView;
    expect(v.x![1] - v.x![0]).toBeLessThan(10);
  });

  it("does not trap the page wheel after a mouse click focuses the plot; keyboard focus zooms", () => {
    const onViewChange = vi.fn();
    render(<Plot xDomain={X} yDomain={Y} series={SERIES} onViewChange={onViewChange} aria-label="curves" />);
    const c = surface();
    const fig = screen.getByRole("figure", { name: "curves" });
    const wheel = () => {
      const ev = new WheelEvent("wheel", { deltaY: -100, bubbles: true, cancelable: true });
      Object.defineProperties(ev, {
        ctrlKey: { value: false }, metaKey: { value: false },
        clientX: { value: px(5, X) }, clientY: { value: py(50, Y) },
      });
      act(() => { c.dispatchEvent(ev); });
      return ev;
    };
    // a mouse click: pointerdown, then the browser focuses the tabIndex=0 frame
    fireEvent.pointerDown(c, { clientX: px(5, X), clientY: py(50, Y), button: 0, pointerId: 1 });
    act(() => { fig.focus(); });
    fireEvent.pointerUp(c, { clientX: px(5, X), clientY: py(50, Y), pointerId: 1 });
    expect(document.activeElement).toBe(fig);
    expect(wheel().defaultPrevented).toBe(false);
    expect(onViewChange).not.toHaveBeenCalled();
    // a key the plot acts on switches to keyboard mode
    fireEvent.keyDown(fig, { key: "ArrowRight" });
    expect(wheel().defaultPrevented).toBe(true);
    // keyboard focus (Tab) without a pointer: wheel zooms
    act(() => { fig.blur(); });
    onViewChange.mockClear();
    act(() => { fig.focus(); });
    expect(wheel().defaultPrevented).toBe(true);
    expect(onViewChange).toHaveBeenCalled();
    // once blurred, plain wheel scrolls again
    act(() => { fig.blur(); });
    expect(wheel().defaultPrevented).toBe(false);
  });

  it("keeps the page wheel after a Ctrl-wheel or any key the plot does not handle", () => {
    const onViewChange = vi.fn();
    render(<Plot xDomain={X} yDomain={Y} series={SERIES} onViewChange={onViewChange} aria-label="curves" />);
    const c = surface();
    const fig = screen.getByRole("figure", { name: "curves" });
    const wheel = (ctrlKey = false) => {
      const ev = new WheelEvent("wheel", { deltaY: -100, bubbles: true, cancelable: true });
      Object.defineProperties(ev, {
        ctrlKey: { value: ctrlKey }, metaKey: { value: false },
        clientX: { value: px(5, X) }, clientY: { value: py(50, Y) },
      });
      act(() => { c.dispatchEvent(ev); });
      return ev;
    };
    fireEvent.pointerDown(c, { clientX: px(5, X), clientY: py(50, Y), button: 0, pointerId: 1 });
    act(() => { fig.focus(); });                                   // a mouse click
    fireEvent.pointerUp(c, { clientX: px(5, X), clientY: py(50, Y), pointerId: 1 });
    // the documented mouse gesture: hold Ctrl, wheel, release
    fireEvent.keyDown(fig, { key: "Control", ctrlKey: true });
    expect(wheel(true).defaultPrevented).toBe(true);
    fireEvent.keyUp(fig, { key: "Control" });
    expect(wheel().defaultPrevented).toBe(false);
    // other modifiers and page keys the plot ignores do not trap the wheel either
    for (const k of ["Meta", "Shift", "Alt", " ", "PageDown", "Tab", "a"]) {
      fireEvent.keyDown(fig, { key: k });
      expect(wheel().defaultPrevented, k).toBe(false);
    }
    // browser shortcuts (Ctrl/⌘ + = − 0: page zoom) are left to the browser
    onViewChange.mockClear();
    for (const k of ["=", "-", "0"]) {
      expect(fireEvent.keyDown(fig, { key: k, ctrlKey: true }), k).toBe(true);   // not prevented
      expect(fireEvent.keyDown(fig, { key: k, metaKey: true }), k).toBe(true);
    }
    expect(onViewChange).not.toHaveBeenCalled();
    expect(wheel().defaultPrevented).toBe(false);
  });

  it("reads a Tab into the plot as keyboard focus after a click on one of its tool buttons", () => {
    render(<Plot xDomain={X} yDomain={Y} series={SERIES} view={{ x: [2, 4], y: null }} aria-label="curves" />);
    const fig = screen.getByRole("figure", { name: "curves" });
    const reset = screen.getByRole("button", { name: "Reset zoom" });
    fireEvent.pointerDown(reset, { button: 0 });
    act(() => { reset.focus(); });
    fireEvent.pointerUp(reset, { button: 0 });
    act(() => { reset.blur(); fig.focus(); });                   // Tab / shift+Tab back onto the plot
    const ev = new WheelEvent("wheel", { deltaY: -100, bubbles: true, cancelable: true });
    Object.defineProperties(ev, {
      ctrlKey: { value: false }, metaKey: { value: false },
      clientX: { value: px(3, [2, 4]) }, clientY: { value: py(50, Y) },
    });
    act(() => { surface().dispatchEvent(ev); });
    expect(ev.defaultPrevented).toBe(true);
  });

  it("zooms, pans and resets from the keyboard; arrows step the readout", () => {
    const onViewChange = vi.fn();
    render(<Plot xDomain={X} yDomain={Y} series={SERIES} onViewChange={onViewChange} aria-label="curves" />);
    const fig = screen.getByRole("figure", { name: "curves" });
    fig.focus();
    fireEvent.keyDown(fig, { key: "+" });
    expect((onViewChange.mock.lastCall![0] as PlotView).x).toEqual([1, 9]);
    fireEvent.keyDown(fig, { key: "ArrowRight", shiftKey: true });
    expect((onViewChange.mock.lastCall![0] as PlotView).x![0]).toBeGreaterThan(1);
    fireEvent.keyDown(fig, { key: "0" });
    expect(onViewChange).toHaveBeenLastCalledWith(null);
    fireEvent.keyDown(fig, { key: "ArrowRight" });
    expect(tooltip()).toBeTruthy();
    expect(screen.getByRole("status").textContent).toMatch(/mean.*0.*10/);
  });

  it("steps the keyboard readout over values a log axis cannot draw (no NaN overlay)", () => {
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    render(<Plot xDomain={X} yDomain={[1, 100]} yScale="log" aria-label="log"
      series={[{ x: [0, 5, 10], y: [0, 10, 50], color: "red", name: "a" }]} />);
    const fig = screen.getByRole("figure", { name: "log" });
    act(() => { fig.focus(); });
    fireEvent.keyDown(fig, { key: "ArrowRight" });                 // x 0 (y 0) is a gap: start at x 5
    expect(screen.getByRole("status").textContent).toBe("a: x 5, y 10");
    const tipStyle = (tooltip() as HTMLElement).style;
    expect(tipStyle.top || tipStyle.bottom).not.toBe("");          // vertically placed (not a dropped NaN)
    fireEvent.keyDown(fig, { key: "ArrowLeft" });                  // nothing drawable to the left: stay
    expect(screen.getByRole("status").textContent).toBe("a: x 5, y 10");
    fireEvent.keyDown(fig, { key: "ArrowRight" });
    expect(screen.getByRole("status").textContent).toBe("a: x 10, y 50");
    expect(err).not.toHaveBeenCalled();
  });

  it("does nothing on drag when zoom is off but still reports clicks", () => {
    const onPlotClick = vi.fn();
    render(<Plot xDomain={X} yDomain={Y} series={SERIES} zoom={false} onPlotClick={onPlotClick} />);
    const c = surface();
    fireEvent.pointerDown(c, { clientX: px(2, X), clientY: py(80, Y), button: 0, pointerId: 1 });
    fireEvent.pointerMove(c, { clientX: px(6, X), clientY: py(20, Y), buttons: 1, pointerId: 1 });
    expect(document.querySelector(".plot__box")).toBeNull();
    fireEvent.pointerUp(c, { clientX: px(6, X), clientY: py(20, Y), pointerId: 1 });
    fireEvent.click(c, { clientX: px(5, X), clientY: py(50, Y) });
    const pt = onPlotClick.mock.lastCall![0];
    expect(pt.x).toBeCloseTo(5, 5);
    expect(pt.y).toBeCloseTo(50, 5);
  });

  it("maps clicks through a log y axis", () => {
    const onPlotClick = vi.fn();
    render(<Plot xDomain={X} yDomain={[1, 1000]} yScale="log" series={SERIES} onPlotClick={onPlotClick} />);
    fireEvent.click(surface(), { clientX: px(5, X), clientY: T + IH / 3 });   // 2/3 of 3 decades
    expect(onPlotClick.mock.lastCall![0].y).toBeCloseTo(100, 3);
  });

  it("reports heat cells on click", () => {
    const onHeatClick = vi.fn();
    render(<Plot xDomain={[0, 2]} yDomain={[0, 2]} series={[]} onHeatClick={onHeatClick}
      heat={{ z: [[1, 2], [3, 4]], xEdges: [0, 1, 2], yEdges: [0, 1, 2] }} />);
    fireEvent.click(surface(), { clientX: L + 0.75 * 524, clientY: T + 0.25 * IH });
    expect(onHeatClick).toHaveBeenCalledWith({ i: 1, j: 1 });
  });
});

describe("Plot v2 export and redraw", () => {
  it("exports the visible series as CSV", () => {
    render(<Plot xDomain={X} yDomain={Y} series={SERIES} exportName="Knee PSNR" legend="auto" />);
    fireEvent.click(screen.getByRole("button", { name: "mean" }));        // hide mean
    fireEvent.click(screen.getByRole("button", { name: "Download CSV" }));
    expect(downloads.text[0].name).toBe("knee-psnr.csv");
    expect(downloads.text[0].text).toBe("series,x,y\r\ngate,0,20\r\ngate,5,30\r\ngate,10,40\r\n");
    expect(screen.getByRole("button", { name: "Download PNG" })).toBeTruthy();
  });

  it("redraws only when the drawn inputs change", () => {
    const spy = vi.spyOn(HTMLCanvasElement.prototype, "getContext");
    const x = [0, 5, 10], y = [1, 2, 3];
    const { rerender } = render(<Plot xDomain={[0, 10]} yDomain={[0, 5]} series={[{ x, y, color: "red" }]} />);
    const n = spy.mock.calls.length;
    expect(n).toBeGreaterThan(0);
    rerender(<Plot xDomain={[0, 10]} yDomain={[0, 5]} series={[{ x, y, color: "red" }]} onPlotClick={() => {}} />);
    expect(spy.mock.calls.length).toBe(n);
    rerender(<Plot xDomain={[0, 10]} yDomain={[0, 5]} series={[{ x, y: [1, 2, 4], color: "red" }]} />);
    expect(spy.mock.calls.length).toBeGreaterThan(n);
    spy.mockRestore();
  });

  it("redraws when a drawn function changes: heat.color, and the formatters of zoomed ticks", () => {
    const spy = vi.spyOn(HTMLCanvasElement.prototype, "getContext");
    const z = [[1, 2], [3, 4]], e = [0, 1, 2];
    const red = () => "rgb(255, 0, 0)";
    const { rerender } = render(<Plot xDomain={[0, 2]} yDomain={[0, 2]} series={[]}
      heat={{ z, xEdges: e, yEdges: e, color: red }} />);
    let n = spy.mock.calls.length;
    rerender(<Plot xDomain={[0, 2]} yDomain={[0, 2]} series={[]} heat={{ z, xEdges: e, yEdges: e, color: red }} />);
    expect(spy.mock.calls.length).toBe(n);                      // same function: no redraw
    rerender(<Plot xDomain={[0, 2]} yDomain={[0, 2]} series={[]}
      heat={{ z, xEdges: e, yEdges: e, color: () => "rgb(0, 0, 255)" }} />);  // a new tint
    expect(spy.mock.calls.length).toBeGreaterThan(n);

    // zoomed: generated ticks are labelled by xFormat; a new formatter giving the
    // same labels costs nothing, one giving new labels redraws
    const view: PlotView = { x: [2, 4], y: null };
    rerender(<Plot xDomain={X} yDomain={Y} series={SERIES} view={view} xFormat={(v) => `${v} e`} />);
    n = spy.mock.calls.length;
    rerender(<Plot xDomain={X} yDomain={Y} series={SERIES} view={view} xFormat={(v) => `${v} e`} />);
    expect(spy.mock.calls.length).toBe(n);
    rerender(<Plot xDomain={X} yDomain={Y} series={SERIES} view={view} xFormat={(v) => `${v} dB`} />);
    expect(spy.mock.calls.length).toBeGreaterThan(n);
    spy.mockRestore();
  });

  it("has an accessible name and a text summary", () => {
    render(<Plot xDomain={X} yDomain={Y} series={SERIES} title="Knee PSNR" xLabel="knee" />);
    const fig = screen.getByRole("figure", { name: "Knee PSNR" });
    expect(fig.textContent).toMatch(/2 series/);
    act(() => { fig.focus(); });
    expect(document.activeElement).toBe(fig);
    expect(W).toBe(640);
  });
});
