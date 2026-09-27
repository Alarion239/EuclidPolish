import { describe, expect, it } from "vitest";
import { DEFAULT_PREFS, INSPECTOR_WIDTH_RANGE } from "../state/prefs";
import {
  INSPECTOR_FLOOR_PX, INSPECTOR_TARGET_PX, MAIN_MIN_PX, SEPARATOR_PX, STAGE_MIN_PX, defaultInspectorWidth, dockedInspectorWidth, railWidth,
} from "./inspectorWidth";

describe("the docked inspector's width", () => {
  it("defaults to a ~480 px viewer inside (512 px panel) when the window allows", () => {
    expect(INSPECTOR_TARGET_PX).toBe(512);
    // 1440 wide, expanded rail: the stage keeps 1440 − 232 − 6 − 512 = 690 px
    expect(defaultInspectorWidth(1440 - railWidth(false))).toBe(512);
    expect(defaultInspectorWidth(1920 - railWidth(false))).toBe(512);
  });

  it("gives way so the main content keeps about 560 px, down to the panel's minimum", () => {
    // 1280: 1280 − 232 − 6 − 560 = 482 (a 450 px viewer inside)
    expect(defaultInspectorWidth(1280 - railWidth(false))).toBe(482);
    // 1024 with the rail collapsed: 1024 − 56 − 6 − 560 = 402
    expect(defaultInspectorWidth(1024 - railWidth(true))).toBe(402);
    // 1024, expanded rail (a 792 px body): both targets cannot be met, so the
    // viewer wins — never narrower than the old 380 px default (a 348 px viewer)
    expect(defaultInspectorWidth(1024 - railWidth(false))).toBeGreaterThanOrEqual(INSPECTOR_FLOOR_PX);
    expect(defaultInspectorWidth(792)).toBe(380);
    // …while the stage keeps its hard 320 px minimum (900 px window, expanded rail)
    expect(defaultInspectorWidth(900 - railWidth(false))).toBe(900 - 232 - SEPARATOR_PX - STAGE_MIN_PX);
    expect(defaultInspectorWidth(400)).toBe(INSPECTOR_WIDTH_RANGE[0]);
    const body = 1280 - railWidth(false);
    expect(body - SEPARATOR_PX - defaultInspectorWidth(body)).toBe(MAIN_MIN_PX);
  });

  it("uses a width the user dragged to, and the default while none was chosen", () => {
    const body = 1440 - railWidth(false);
    expect(dockedInspectorWidth(DEFAULT_PREFS.inspectorWidth, body)).toBe(512);   // never resized
    expect(dockedInspectorWidth(640, body)).toBe(640);
    expect(dockedInspectorWidth(300, body)).toBe(300);
    // a saved width the window cannot hold next to a 320 px stage is trimmed
    expect(dockedInspectorWidth(1100, 1024 - railWidth(false))).toBe(1024 - 232 - 6 - 320);
  });

  it("knows the rail widths (tokens.css --rail-w / --rail-w-collapsed)", () => {
    expect(railWidth(false)).toBe(232);
    expect(railWidth(true)).toBe(56);
  });
});
