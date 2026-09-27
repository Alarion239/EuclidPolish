/* ViewerStage measures the viewer's DOM in its stage and applies frameFit /
 * besideWidth (viewerFit.test.ts holds the maths). jsdom has no layout, so
 * the boxes the component reads are stubbed per class. */
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactElement } from "react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { ViewerStage } from "./ViewerStage";

type Box = { top?: number; width?: number; height?: number; clientHeight?: number };
let boxes: Record<string, Box>;

const boxOf = (el: Element): Box => {
  for (const cls of el.classList) if (boxes[cls]) return boxes[cls];
  return {};
};

const saved: [string, PropertyDescriptor | undefined][] = [];
function stub(prop: "clientWidth" | "clientHeight" | "offsetHeight", get: (el: HTMLElement) => number) {
  saved.push([prop, Object.getOwnPropertyDescriptor(HTMLElement.prototype, prop)]);
  Object.defineProperty(HTMLElement.prototype, prop, { configurable: true, get() { return get(this as HTMLElement); } });
}
let savedRect: typeof Element.prototype.getBoundingClientRect;

beforeEach(() => {
  // 792 × 720: a 672 px stage, the table 97 px under its top, bar 37 + readout 28
  // around a 599 px frame (the viewer's own full-height fit), 717 px wide.
  boxes = {
    stage: { top: 48, clientHeight: 672 },
    "dt-vstage": { width: 717 },
    "cv-table": { top: 48 + 97, width: 717, height: 37 + 599 + 28 },
    "cv-frames": { width: 717, height: 599 },
    "cv-bar": { height: 37 },
    "cv-readout": { height: 28 },
  };
  stub("clientWidth", (el) => boxOf(el).width ?? 0);
  stub("clientHeight", (el) => boxOf(el).clientHeight ?? boxOf(el).height ?? 0);
  stub("offsetHeight", (el) => boxOf(el).height ?? 0);
  savedRect = Element.prototype.getBoundingClientRect;
  Element.prototype.getBoundingClientRect = function rect(this: Element) {
    const b = boxOf(this);
    return { top: b.top ?? 0, left: 0, right: b.width ?? 0, bottom: (b.top ?? 0) + (b.height ?? 0), width: b.width ?? 0, height: b.height ?? 0, x: 0, y: b.top ?? 0, toJSON: () => ({}) } as DOMRect;
  };
});
afterEach(() => {
  for (const [prop, d] of saved.splice(0)) {
    if (d) Object.defineProperty(HTMLElement.prototype, prop, d);
    else delete (HTMLElement.prototype as unknown as Record<string, unknown>)[prop];
  }
  Element.prototype.getBoundingClientRect = savedRect;
});

/** A stand-in for <ImageViewer>'s DOM: the light table with `frames` frames. */
const FakeViewer = ({ focus = false, frames = 1, stacked = false, barRows = [] }: {
  focus?: boolean; frames?: number; stacked?: boolean; barRows?: string[][];
}) => (
  <div className="cv-host">
    <div className="cv-root" data-focus={focus || undefined}>
      <div className="cv-table">
        <div className="cv-bar">
          {barRows.map((row, i) => (
            <div key={i} className="cv-bar__row">{row.map((cls) => <span key={cls} className={cls} />)}</div>
          ))}
        </div>
        <div className="cv-frames">
          {stacked ? <div className="cv-stack" /> : Array.from({ length: frames }, (_x, i) => <div key={i} className="cv-frame" />)}
        </div>
        <div className="cv-readout" />
      </div>
    </div>
  </div>
);

const inStage = (el: ReactElement) => render(<div className="stage" style={{ overflowY: "auto" }}>{el}</div>);

const rowOf = (c: HTMLElement) => c.querySelector(".dt-vstage") as HTMLElement;

describe("ViewerStage", () => {
  it("fits a single height-limited frame under the viewer's top, the viewer staying full width", () => {
    const { container } = inStage(<ViewerStage frames={1}><FakeViewer /></ViewerStage>);
    const row = rowOf(container);
    expect(row.hasAttribute("data-fit")).toBe(true);
    expect(row.style.getPropertyValue("--dt-side")).toBe("502px");
    expect(row.style.getPropertyValue("--dt-cols")).toBe("1");
    expect(row.style.getPropertyValue("--dt-vw")).toBe("");
    expect(row.hasAttribute("data-beside")).toBe(false);
  });

  it("fits a blink / swipe stack as one frame at the chrome the viewer has (no narrowing)", () => {
    // Records two-up is width-limited (357 px each); blinking stacks them into
    // one frame: it gets the whole height under the top, not less than 357
    boxes["cv-frames"] = { width: 717, height: 357 };
    const { container } = inStage(<ViewerStage frames={2}><FakeViewer stacked /></ViewerStage>);
    const row = rowOf(container);
    expect(row.style.getPropertyValue("--dt-side")).toBe("502px");
    expect(row.style.getPropertyValue("--dt-vw")).toBe("");
  });

  it("puts the side panel beside the fitted viewer when it leaves room, as tall as the viewer", () => {
    boxes["dt-vstage__viewer"] = { height: 664 };
    const { container } = inStage(
      <ViewerStage frames={1} aside={(p) => <span>{p.beside ? `beside ${p.height}` : "below"}</span>} asideLabel="Gallery" asideMin={150} asideFill>
        <FakeViewer />
      </ViewerStage>,
    );
    const row = rowOf(container);
    expect(row.hasAttribute("data-beside")).toBe(true);
    expect(row.style.getPropertyValue("--dt-vw")).toBe("502px");
    expect(row.style.getPropertyValue("--dt-vh")).toBe("664px");
    expect(screen.getByRole("region", { name: "Gallery" }).textContent).toBe("beside 664");
    expect(screen.getByRole("region", { name: "Gallery" }).getAttribute("data-fill")).toBe("true");
  });

  it("never narrows the viewer below its bar's two-row width", () => {
    // the widest bar row is 540 px: 717 − 542 − 12 leaves the panel 163 px
    boxes.wide = { width: 540 };
    boxes.narrow = { width: 200 };
    const { container } = inStage(
      <ViewerStage frames={1} aside={<span>thumbs</span>} asideLabel="Gallery" asideMin={150}>
        <FakeViewer barRows={[["narrow"], ["wide"]]} />
      </ViewerStage>,
    );
    expect(rowOf(container).style.getPropertyValue("--dt-vw")).toBe("542px");
  });

  it("measures an icon-only (compact) bar at its labelled width and leaves it compact", () => {
    boxes.wide = { width: 540 };
    // the label's width only counts while the bar is not compact (as in the browser)
    const labelWidth = (el: HTMLElement) => (el.closest(".cv-bar")?.hasAttribute("data-compact") ? 0 : 60);
    boxes.label = {};
    const rect = Element.prototype.getBoundingClientRect;
    Element.prototype.getBoundingClientRect = function r(this: Element) {
      const d = rect.call(this);
      return this.classList.contains("label") ? { ...d, width: labelWidth(this as HTMLElement) } as DOMRect : d;
    };
    const { container } = inStage(
      <ViewerStage frames={1} aside={<span>thumbs</span>} asideLabel="Gallery" asideMin={100}>
        <FakeViewer barRows={[["wide", "label"]]} />
      </ViewerStage>,
    );
    container.querySelector(".cv-bar")!.setAttribute("data-compact", "");
    fireEvent(window, new Event("resize"));
    return waitFor(() => {
      expect(rowOf(container).style.getPropertyValue("--dt-vw")).toBe("602px");
      expect(container.querySelector(".cv-bar")!.hasAttribute("data-compact")).toBe(true);
    });
  });

  it("keeps the panel below when the fitted viewer leaves too little width", () => {
    const { container } = inStage(
      <ViewerStage frames={1} aside={<span>thumbs</span>} asideLabel="Gallery" asideMin={260}><FakeViewer /></ViewerStage>,
    );
    const row = rowOf(container);
    expect(row.hasAttribute("data-fit")).toBe(true);
    expect(row.hasAttribute("data-beside")).toBe(false);
  });

  it("leaves a width-limited viewer alone", () => {
    boxes["cv-frames"] = { width: 717, height: 357 };
    boxes["cv-table"] = { top: 48 + 97, width: 717, height: 37 + 357 + 28 };
    const { container } = inStage(<ViewerStage frames={2} aside={<span>list</span>} asideLabel="List"><FakeViewer frames={2} /></ViewerStage>);
    // two frames side by side are 357 px: under the 502 px left, nothing to fit
    const row = rowOf(container);
    expect(row.hasAttribute("data-fit")).toBe(false);
    expect(row.hasAttribute("data-beside")).toBe(false);
  });

  it("counts the page's expected frames while the viewer is still loading", () => {
    const Loading = () => (
      <div className="cv-host"><div className="cv-root"><div className="cv-table">
        <div className="cv-bar" /><div className="cv-frames"><div className="cv-frame cv-frame--message" /></div><div className="cv-readout" />
      </div></div></div>
    );
    // two expected tiers: width-limited, so no fit (and no jump when the meta arrives)
    const { container } = inStage(<ViewerStage frames={2}><Loading /></ViewerStage>);
    expect(rowOf(container).hasAttribute("data-fit")).toBe(false);
  });

  it("measures nothing in focus mode (the viewer is lifted out of the page)", () => {
    const { container } = inStage(<ViewerStage frames={1}><FakeViewer focus /></ViewerStage>);
    expect(rowOf(container).hasAttribute("data-fit")).toBe(false);
  });
});
