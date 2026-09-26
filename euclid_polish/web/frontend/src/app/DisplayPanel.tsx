/* The global Display (colour) panel (spec §4, C7): edits `useDisplay`, which
 * every linked image viewer follows. Sections: Image (colour mode, custom RGB
 * mapping, stretch, per-group transfer knee/gain/black, colormaps, invert,
 * NaN colour), Viewers (link all viewers, wheel behaviour) and any section a
 * workspace registers (`registerDisplaySection`, e.g. the Sky atlas's "Sky").
 * The absolute-asinh stretch with knee 100 e⁻ is the locked default. */
import { useEffect, useState } from "react";
import {
  COLORMAPS, COLOR_MODES, GAIN_RANGE, KNEE_RANGE, STRETCHES, TRANSFER_GROUPS, WHEEL_MODES,
  useDisplay, type ColorMode, type Colormap, type Stretch, type WheelMode,
} from "../state/display";
import { Button, Dialog, Field, NumberField, Section, Select, Slider, Switch, type SelectOption } from "../ui";
import { useDisplaySections } from "./displaySections";
import { useShellUi } from "./shellStore";

const COLOR_LABEL: Record<ColorMode, string> = {
  VIS: "VIS", Y_E: "Y", J_E: "J", H_E: "H", lupton: "Lupton RGB", temp: "Temperature",
  rgb: "Custom RGB", native: "Native (tier default)",
};
const STRETCH_LABEL: Record<Stretch, string> = {
  "asinh-abs": "asinh · absolute (default)", linear: "linear", log: "log", sqrt: "sqrt",
  "asinh-auto": "asinh · auto", zscale: "zscale / percentile",
};
const CMAP_LABEL: Record<Colormap, string> = {
  gray: "gray", viridis: "viridis", magma: "magma", inferno: "inferno", cividis: "cividis", rdbu: "RdBu (diverging)",
};
const WHEEL_LABEL: Record<WheelMode, string> = {
  "zoom-when-focused": "zoom when the viewer is focused (or ⌘/Ctrl)",
  "always-zoom": "always zoom",
  scroll: "scroll the page (zoom with ⌘/Ctrl)",
};
const GROUP_LABEL: Record<string, string> = { default: "Default", euclid: "Euclid", jwst: "JWST" };
const BANDS = ["VIS", "Y_E", "J_E", "H_E"];

const opts = <T extends string>(values: readonly T[], labels: Record<T, string>): SelectOption<T>[] =>
  values.map((v) => ({ value: v, label: labels[v] }));

/** The black point as a draft string, so "-" or "1." can be typed. */
function BlackPoint({ group, label }: { group: string; label: string }) {
  const value = useDisplay((s) => s.groups[group]?.black ?? 0);
  const setGroup = useDisplay((s) => s.setGroup);
  const [draft, setDraft] = useState(String(value));
  useEffect(() => {
    setDraft((d) => (Number(d) === value && d.trim() !== "" ? d : String(value)));
  }, [value]);
  return (
    <NumberField label={label} unit="e⁻" value={draft} step="any"
      onChange={(v) => {
        setDraft(v);
        const black = Number(v);
        if (v.trim() !== "" && Number.isFinite(black)) setGroup(group, { black });
      }} />
  );
}

const fmtE = (v: number) => `${Number(v.toPrecision(3))} e⁻`;
const fmtX = (v: number) => `×${Number(v.toPrecision(3))}`;

function ImageSection() {
  const d = useDisplay();
  return (
    <Section title="Image" sub="every linked viewer">
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
      </div>
      <div className="display-groups">
        {TRANSFER_GROUPS.map((name) => {
          const g = d.groups[name];
          return (
            <fieldset key={name} className="display-group">
              <legend>{GROUP_LABEL[name] ?? name} transfer</legend>
              <Field label="Knee">
                <Slider value={g.knee} min={KNEE_RANGE[0]} max={KNEE_RANGE[1]} scale="log" showValue
                  format={fmtE} aria-label={`${GROUP_LABEL[name] ?? name} knee`}
                  onChange={(knee) => d.setGroup(name, { knee })} />
              </Field>
              <Field label="Gain">
                <Slider value={g.gain} min={GAIN_RANGE[0]} max={GAIN_RANGE[1]} scale="log" showValue
                  format={fmtX} aria-label={`${GROUP_LABEL[name] ?? name} gain`}
                  onChange={(gain) => d.setGroup(name, { gain })} />
              </Field>
              <BlackPoint group={name} label="Black point" />
            </fieldset>
          );
        })}
      </div>
    </Section>
  );
}

function ViewersSection() {
  const d = useDisplay();
  return (
    <Section title="Viewers">
      <div className="display-grid">
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

export function DisplaySettingsForm() {
  const sections = useDisplaySections((s) => s.sections);
  return (
    <div className="display-panel">
      <ImageSection />
      <ViewersSection />
      {sections.map((s) => (
        <Section key={s.id} title={s.title}><s.Component /></Section>
      ))}
    </div>
  );
}

export function DisplayPanel() {
  const open = useShellUi((s) => s.display);
  const setOpen = (v: boolean) => useShellUi.getState().setOpen("display", v);
  const reset = useDisplay((s) => s.reset);
  return (
    <Dialog open={open} onOpenChange={setOpen} title="Display" size="lg"
      description="Colour and stretch for every linked image viewer. Saved in this browser."
      footer={(
        <>
          <Button variant="ghost" onClick={reset}>Reset to defaults</Button>
          <Button variant="primary" onClick={() => setOpen(false)}>Done</Button>
        </>
      )}>
      {open && <DisplaySettingsForm />}
    </Dialog>
  );
}
