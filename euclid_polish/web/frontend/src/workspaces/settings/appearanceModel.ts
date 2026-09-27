/* Settings › Appearance "Images": a plain-words summary of the live Display
 * settings (state/display.ts, the store the Display panel edits). The panel
 * is the one place to change them; this card only reads them and opens it.
 * Pure (appearanceModel.test.ts). */
import { CMAP_LABEL, COLOR_LABEL, STRETCH_LABEL, WHEEL_LABEL } from "../../app/DisplayPanel";
import { DEFAULT_DISPLAY, transferFor, type DisplaySettings } from "../../state/display";

export type DisplayFact = { label: string; value: string; changed: boolean };

const num = (v: number) => String(Number(v.toPrecision(3)));

export function displaySummary(d: DisplaySettings): DisplayFact[] {
  const t = transferFor(d), t0 = transferFor(DEFAULT_DISPLAY);
  const fact = (label: string, value: string, changed: boolean): DisplayFact => ({ label, value, changed });
  return [
    fact("Colour", COLOR_LABEL[d.color] ?? d.color, d.color !== DEFAULT_DISPLAY.color),
    fact("Stretch", STRETCH_LABEL[d.stretch] ?? d.stretch, d.stretch !== DEFAULT_DISPLAY.stretch),
    fact("Knee", `${num(t.knee)} e⁻, brightness ×${num(t.gain)}`, t.knee !== t0.knee || t.gain !== t0.gain || t.black !== t0.black),
    fact("Colormap", CMAP_LABEL[d.colormap] ?? d.colormap, d.colormap !== DEFAULT_DISPLAY.colormap),
    fact("Residual colormap", CMAP_LABEL[d.residualColormap] ?? d.residualColormap, d.residualColormap !== DEFAULT_DISPLAY.residualColormap),
    fact("NaN colour", d.nanColor, d.nanColor !== DEFAULT_DISPLAY.nanColor),
    fact("Invert", d.invert ? "On" : "Off", d.invert !== DEFAULT_DISPLAY.invert),
    fact("Viewers", d.linked ? "Every viewer follows these settings" : "Each viewer keeps its own settings", d.linked !== DEFAULT_DISPLAY.linked),
    fact("Mouse wheel", WHEEL_LABEL[d.wheel] ?? d.wheel, d.wheel !== DEFAULT_DISPLAY.wheel),
  ];
}
