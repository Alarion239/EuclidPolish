/* The global Display (colour) panel (spec §4, C7): edits `useDisplay`, which
 * every linked image viewer follows. It is a NON-modal side sheet on the
 * right, under the top bar: no overlay, the page stays live, and clicking or
 * dragging an image does not close it — you watch the knee, stretch or
 * colormap change on the images while you adjust. Esc, Done or the close
 * button closes it (Shift+D opens it).
 * Sections: the current page's own section first (`registerDisplaySection`,
 * e.g. the Sky atlas's "Sky"), then Image (colour mode, custom RGB mapping,
 * stretch, colormaps, NaN colour, invert, and the knee / brightness / black
 * point of one transfer group at a time) and Viewers (link all viewers,
 * wheel behaviour). The absolute-asinh stretch with knee 100 e⁻ is the
 * locked default. Labels match the viewer's own Display menu.
 * From 640 px up the sheet takes its width FROM the stage (the shell reserves
 * it, `data-display`), so the images refit beside it instead of sitting under
 * it; below that it overlays. Closing it hands focus back to what opened it
 * (the Display settings button, or whatever had focus for Shift+D; else the
 * button) — unless focus already moved to the page. */
import * as RDialog from "@radix-ui/react-dialog";
import { useEffect, useRef, useState } from "react";
import {
  COLORMAPS, COLOR_MODES, STRETCHES, TRANSFER_GROUPS, WHEEL_MODES,
  useDisplay, type ColorMode, type Colormap, type Stretch, type WheelMode,
} from "../state/display";
import { Button, Field, Icon, NumberField, Section, Segmented, Select, Slider, Switch, type SelectOption } from "../ui";
import { GAIN_SLIDER_RANGE, KNEE_SLIDER_RANGE, formatSig, parseNumber } from "../viewer/barModel";
import { useDisplaySections } from "./displaySections";
import { useShellUi } from "./shellStore";

/** Shared with System › Appearance. */
export const COLOR_LABEL: Record<ColorMode, string> = {
  VIS: "VIS", Y_E: "Y", J_E: "J", H_E: "H", lupton: "Lupton RGB", temp: "Temperature",
  rgb: "Custom RGB", native: "Native (tier default)",
};
export const STRETCH_LABEL: Record<Stretch, string> = {
  "asinh-abs": "Asinh (absolute)", linear: "Linear", log: "Log", sqrt: "Square root",
  "asinh-auto": "Asinh (auto)", zscale: "Zscale",
};
export const CMAP_LABEL: Record<Colormap, string> = {
  gray: "Gray", viridis: "Viridis", magma: "Magma", inferno: "Inferno", cividis: "Cividis", rdbu: "Red–blue (diverging)",
};
export const WHEEL_LABEL: Record<WheelMode, string> = {
  "zoom-when-focused": "Zoom when the viewer is focused (or ⌘/Ctrl)",
  "always-zoom": "Always zoom",
  scroll: "Scroll the page (zoom with ⌘/Ctrl)",
};
const GROUP_LABEL: Record<string, string> = { default: "Default", euclid: "Euclid", jwst: "JWST" };
/** The unit a group's knee and black point are in (JWST tiers are MJy/sr). */
const GROUP_UNIT: Record<string, string> = { default: "e⁻", euclid: "e⁻", jwst: "MJy/sr" };
const BANDS = ["VIS", "Y_E", "J_E", "H_E"];

const opts = <T extends string>(values: readonly T[], labels: Record<T, string>): SelectOption<T>[] =>
  values.map((v) => ({ value: v, label: labels[v] }));

/** The black point as a draft string, so "-" or "1." can be typed. */
function BlackPoint({ group, label, unit }: { group: string; label: string; unit: string }) {
  const value = useDisplay((st) => st.groups[group]?.black ?? 0);
  const setGroup = useDisplay((st) => st.setGroup);
  const [draft, setDraft] = useState(String(value));
  useEffect(() => {
    setDraft((d) => (Number(d) === value && d.trim() !== "" ? d : String(value)));
  }, [value]);
  return (
    <NumberField label={label} unit={unit} value={draft} step="any"
      onChange={(v) => {
        setDraft(v);
        const black = Number(v);
        if (v.trim() !== "" && Number.isFinite(black)) setGroup(group, { black });
      }} />
  );
}

const fmtX = (v: number) => `×${formatSig(v)}`;

/** A log slider with a typed value beside it, so an exact knee (1, 3 e⁻ —
 *  the training knees) is one entry away. Enter or leaving the field
 *  applies it; a value that is not a positive number is ignored. */
function KneeInput({ label, value, unit, onChange }: {
  label: string; value: number; unit: string; onChange: (v: number) => void;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const commit = () => {
    const v = parseNumber(draft ?? "");
    setDraft(null);
    if (v != null && v > 0) onChange(v);
  };
  return (
    <span className="display-knee">
      <Slider value={value} min={KNEE_SLIDER_RANGE[0]} max={KNEE_SLIDER_RANGE[1]} scale="log"
        format={(v) => `${formatSig(v)} ${unit}`} aria-label={label} onChange={onChange} />
      <input className="ui-input ui-input--sm display-knee__num" type="text" inputMode="decimal"
        aria-label={`${label} (${unit})`} value={draft ?? formatSig(value)}
        onChange={(e) => setDraft(e.target.value)} onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") commit();
          if (e.key === "Escape") setDraft(null);
        }} />
      <span className="display-knee__unit">{unit}</span>
    </span>
  );
}

/** Knee, brightness and black point of one transfer group at a time. */
function TransferGroups() {
  const d = useDisplay();
  const [group, setGroup] = useState<string>(TRANSFER_GROUPS[0]);
  const g = d.groups[group];
  const name = GROUP_LABEL[group] ?? group;
  const unit = GROUP_UNIT[group] ?? "";
  return (
    <fieldset className="display-group">
      <legend>Transfer</legend>
      <Segmented<string> size="sm" value={group} onChange={setGroup} aria-label="Transfer group"
        options={TRANSFER_GROUPS.map((n) => ({ value: n, label: GROUP_LABEL[n] ?? n }))} />
      <Field label="Knee">
        <KneeInput key={group} label={`${name} knee`} value={g.knee} unit={unit}
          onChange={(knee) => d.setGroup(group, { knee })} />
      </Field>
      <Field label="Brightness">
        <Slider value={g.gain} min={GAIN_SLIDER_RANGE[0]} max={GAIN_SLIDER_RANGE[1]} scale="log" showValue
          format={fmtX} aria-label={`${name} brightness`}
          onChange={(gain) => d.setGroup(group, { gain })} />
      </Field>
      <BlackPoint key={group} group={group} label="Black point" unit={unit} />
    </fieldset>
  );
}

function ImageSection() {
  const d = useDisplay();
  return (
    <Section title="Image" sub="Every linked viewer">
      <div className="display-grid">
        <Field label="Colour">
          <Select<ColorMode> value={d.color} onChange={(color) => d.set({ color })} options={opts(COLOR_MODES, COLOR_LABEL)} />
        </Field>
        <Field label="Stretch">
          <Select<Stretch> value={d.stretch} onChange={(stretch) => d.set({ stretch })} options={opts(STRETCHES, STRETCH_LABEL)} />
        </Field>
        {d.color === "rgb" && (["R", "G", "B"] as const).map((ch, i) => (
          <Field key={ch} label={`${ch} band`}>
            <Select value={d.rgb[i]} options={BANDS.map((b) => ({ value: b, label: COLOR_LABEL[b as ColorMode] }))}
              onChange={(band) => {
                const rgb = [...d.rgb] as [string, string, string];
                rgb[i] = band;
                d.set({ rgb });
              }} />
          </Field>
        ))}
        <Field label="Colormap">
          <Select<Colormap> value={d.colormap} onChange={(colormap) => d.set({ colormap })} options={opts(COLORMAPS, CMAP_LABEL)} />
        </Field>
        <Field label="Residual colormap">
          <Select<Colormap> value={d.residualColormap} onChange={(residualColormap) => d.set({ residualColormap })}
            options={opts(COLORMAPS, CMAP_LABEL)} />
        </Field>
        <Field label="NaN colour">
          <input type="color" className="display-color" value={d.nanColor}
            onChange={(e) => d.set({ nanColor: e.target.value })} />
        </Field>
        <div className="display-switch"><Switch checked={d.invert} onChange={(invert) => d.set({ invert })}>Invert</Switch></div>
        <div className="display-switch">
          <Switch checked={d.matchSurfaceBrightness} onChange={(matchSurfaceBrightness) => d.set({ matchSurfaceBrightness })}>
            Match surface brightness across pixel scales
          </Switch>
        </div>
      </div>
      <TransferGroups />
    </Section>
  );
}

function ViewersSection() {
  const d = useDisplay();
  return (
    <Section title="Viewers">
      <div className="display-grid display-grid--one">
        <div className="display-switch">
          <Switch checked={d.linked} onChange={(linked) => d.set({ linked })}>Link all viewers to these settings</Switch>
        </div>
        <Field label="Mouse wheel">
          <Select<WheelMode> value={d.wheel} onChange={(wheel) => d.set({ wheel })} options={opts(WHEEL_MODES, WHEEL_LABEL)} />
        </Field>
      </div>
    </Section>
  );
}

/** The panel's sections: the current page's own first (it is what the page
 *  shows), then Image and Viewers. */
export function DisplaySettingsForm() {
  const sections = useDisplaySections((st) => st.sections);
  return (
    <div className="display-panel">
      {sections.map((sec) => (
        <Section key={sec.id} title={sec.title}><sec.Component /></Section>
      ))}
      <ImageSection />
      <ViewersSection />
    </div>
  );
}

/** What had focus when the sheet opened (captured synchronously in the
 *  store update, before Radix moves focus into the sheet). */
let displayOpener: HTMLElement | null = null;
useShellUi.subscribe((st, prev) => {
  if (!st.display || prev.display || typeof document === "undefined") return;
  const a = document.activeElement;
  displayOpener = a instanceof HTMLElement && a !== document.body && !a.closest(".display-sheet") ? a : null;
});

/** Where focus goes when the sheet closes: the opener, else the top bar's
 *  Display settings button, else the stage. */
export function displayReturnTarget(): HTMLElement | null {
  if (displayOpener?.isConnected) return displayOpener;
  return document.querySelector<HTMLElement>('.topbar button[aria-label="Display settings"]')
    ?? document.getElementById("main");
}

export function DisplayPanel() {
  const open = useShellUi((st) => st.display);
  const setOpen = (v: boolean) => useShellUi.getState().setOpen("display", v);
  const reset = useDisplay((st) => st.reset);
  const contentRef = useRef<HTMLDivElement>(null);
  return (
    <RDialog.Root open={open} onOpenChange={setOpen} modal={false}>
      <RDialog.Portal>
        {/* Non-modal: no overlay, and working on the page does not close it. */}
        <RDialog.Content ref={contentRef} className="display-sheet" onInteractOutside={(e) => e.preventDefault()}
          onCloseAutoFocus={(e) => {
            e.preventDefault();
            const a = document.activeElement;
            // Focus already moved to the page (e.g. Shift+D from a viewer): leave it there.
            if (a && a !== document.body && !contentRef.current?.contains(a)) return;
            displayReturnTarget()?.focus({ preventScroll: true });
          }}>
          <header className="display-sheet__head">
            <RDialog.Title className="display-sheet__title">Display</RDialog.Title>
            <RDialog.Close asChild>
              <button type="button" className="ui-iconbtn ui-iconbtn--ghost ui-iconbtn--sm" aria-label="Close the Display panel">
                <Icon name="close" />
              </button>
            </RDialog.Close>
          </header>
          <RDialog.Description className="display-sheet__desc">
            Colour and stretch for every linked image viewer; the images update as you change them. Saved in this browser.
          </RDialog.Description>
          <div className="display-sheet__body">{open && <DisplaySettingsForm />}</div>
          <footer className="display-sheet__foot">
            <Button variant="ghost" onClick={reset}>Reset to defaults</Button>
            <Button variant="primary" onClick={() => setOpen(false)}>Done</Button>
          </footer>
        </RDialog.Content>
      </RDialog.Portal>
    </RDialog.Root>
  );
}
