/* settings/appearance (spec §8.8): theme (light / dark / system), accent,
 * density, the rail and the inspector width, and the image defaults every
 * linked viewer follows (the Display panel's store, C7): colour, stretch,
 * colormaps, NaN colour, invert, linking and the mouse-wheel behaviour.
 * Saved in this browser; nothing goes to the server. */
import type { ReactNode } from "react";
import { CMAP_LABEL, COLOR_LABEL, STRETCH_LABEL, WHEEL_LABEL } from "../../../app/DisplayPanel";
import { usePageActions } from "../../../app/palette";
import { openDisplayPanel } from "../../../app/shellStore";
import {
  COLORMAPS, COLOR_MODES, DEFAULT_DISPLAY, STRETCHES, WHEEL_MODES, useDisplay,
  type ColorMode, type Colormap, type Stretch, type WheelMode,
} from "../../../state/display";
import {
  ACCENTS, DEFAULT_PREFS, DENSITIES, THEME_PREFS, usePrefs, useResolvedTheme,
  type Accent, type Density, type ThemePref,
} from "../../../state/prefs";
import {
  Badge, Button, Card, CardBody, CardHead, Chip, Field, Page, PageHead, Segmented, Select, Switch, toast,
} from "../../../ui";
import "../settings.css";

/* A caption over a group control (Segmented, chips): a <Field> would wrap
   several buttons in one <label> and rename them. */
function Group({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="appearance-group" role="group" aria-label={label}>
      <span className="appearance-group__label" aria-hidden="true">{label}</span>
      {children}
    </div>
  );
}

const THEME_LABEL: Record<ThemePref, string> = { light: "Light", dark: "Dark", system: "System" };
const DENSITY_LABEL: Record<Density, string> = { comfortable: "Comfortable", compact: "Compact" };
const opts = <T extends string>(values: readonly T[], labels: Record<T, string>) =>
  values.map((v) => ({ value: v, label: labels[v] }));

export default function Appearance() {
  const prefs = usePrefs();
  const resolved = useResolvedTheme();
  const d = useDisplay();
  usePageActions([
    { id: "appearance:reset", label: "Reset the appearance", group: "Settings", keywords: ["theme", "accent", "density"],
      run: () => { prefs.reset(); toast("Appearance reset to the defaults"); } },
    { id: "appearance:display-reset", label: "Reset the image display defaults", group: "Settings",
      keywords: ["display", "stretch", "knee", "colormap"],
      run: () => { d.reset(); toast("Display settings reset (absolute asinh, knee 100 e⁻)"); } },
  ]);
  const displayChanged = d.color !== DEFAULT_DISPLAY.color || d.stretch !== DEFAULT_DISPLAY.stretch
    || d.colormap !== DEFAULT_DISPLAY.colormap || d.residualColormap !== DEFAULT_DISPLAY.residualColormap
    || d.nanColor !== DEFAULT_DISPLAY.nanColor || d.invert !== DEFAULT_DISPLAY.invert
    || d.linked !== DEFAULT_DISPLAY.linked || d.wheel !== DEFAULT_DISPLAY.wheel;
  return (
    <Page className="settings-appearance">
      <PageHead eyebrow="settings · appearance" title="Appearance" sub="Saved in this browser only." />
      <div className="settings-cards">
        <Card>
          <CardHead title="Theme" sub={prefs.theme === "system" ? `following the system (now ${resolved})` : undefined} />
          <CardBody>
            <div className="settings-stack">
              <Group label="Theme">
                <Segmented<ThemePref> aria-label="Theme" value={prefs.theme} onChange={prefs.setTheme}
                  options={THEME_PREFS.map((t) => ({ value: t, label: THEME_LABEL[t] }))} />
              </Group>
              <Group label="Accent">
                <div className="settings-row">
                  {ACCENTS.map((a) => (
                    <Chip key={a} on={prefs.accent === a} onClick={() => prefs.set({ accent: a as Accent })}
                      aria-label={`Accent ${a}`}>{a}</Chip>
                  ))}
                </div>
              </Group>
              <Group label="Density">
                <Segmented<Density> aria-label="Density" value={prefs.density} onChange={(density) => prefs.set({ density })}
                  options={DENSITIES.map((x) => ({ value: x, label: DENSITY_LABEL[x] }))} />
              </Group>
              <div className="preview-strip" aria-label="Preview">
                <Button size="sm" variant="primary">Primary</Button>
                <Button size="sm">Default</Button>
                <Badge tone="accent">accent</Badge>
                <Badge tone="good" dot>good</Badge>
                <Badge tone="warn" dot>warn</Badge>
                <Badge tone="bad" dot>bad</Badge>
                <a href="#appearance" onClick={(e) => e.preventDefault()}>a link</a>
              </div>
            </div>
          </CardBody>
        </Card>

        <Card>
          <CardHead title="Layout" />
          <CardBody>
            <div className="settings-stack">
              <Switch checked={prefs.railCollapsed} onChange={(railCollapsed) => prefs.set({ railCollapsed })}>
                Collapse the navigation rail to icons (shortcut [)
              </Switch>
              <div className="settings-row">
                <span className="muted">Inspector width <b className="mono">{prefs.inspectorWidth}px</b></span>
                <Button size="sm" variant="ghost" disabled={prefs.inspectorWidth === DEFAULT_PREFS.inspectorWidth}
                  onClick={() => prefs.set({ inspectorWidth: DEFAULT_PREFS.inspectorWidth })}>Reset width</Button>
              </div>
              <div>
                <Button size="sm" onClick={() => { prefs.reset(); toast("Appearance reset to the defaults"); }}>
                  Reset appearance
                </Button>
              </div>
            </div>
          </CardBody>
        </Card>

        <Card className="settings-cards__wide">
          <CardHead title="Images" sub="defaults every linked viewer follows"
            right={displayChanged ? <Badge tone="info">customised</Badge> : <Badge>defaults</Badge>} />
          <CardBody>
            <div className="settings-stack">
              <div className="conn-form">
                <Field label="Colour" hint="The band or composite a viewer shows (Q–Y keys in a viewer).">
                  <Select<ColorMode> value={d.color} onChange={(color) => d.set({ color })} options={opts(COLOR_MODES, COLOR_LABEL)} />
                </Field>
                <Field label="Stretch" hint="Absolute asinh (black / knee / white) is the locked default; the others are opt-in.">
                  <Select<Stretch> value={d.stretch} onChange={(stretch) => d.set({ stretch })} options={opts(STRETCHES, STRETCH_LABEL)} />
                </Field>
                <Field label="Colormap" hint="Single-band colormap (gray by default).">
                  <Select<Colormap> value={d.colormap} onChange={(colormap) => d.set({ colormap })} options={opts(COLORMAPS, CMAP_LABEL)} />
                </Field>
                <Field label="Residual colormap" hint="Diverging map of residual tiers (A−B, (A−B)/σ).">
                  <Select<Colormap> value={d.residualColormap} onChange={(residualColormap) => d.set({ residualColormap })}
                    options={opts(COLORMAPS, CMAP_LABEL)} />
                </Field>
                <Field label="NaN colour" hint="Colour of missing / NaN pixels (blank NISP holes, masked cores).">
                  <span className="settings-row">
                    <input type="color" className="color-input" value={d.nanColor}
                      onChange={(e) => d.set({ nanColor: e.target.value })} aria-label="NaN colour" />
                    <code className="mono">{d.nanColor}</code>
                  </span>
                </Field>
                <Field label="Mouse wheel over a viewer" hint="A plain wheel scrolls the page unless you choose otherwise.">
                  <Select<WheelMode> value={d.wheel} onChange={(wheel) => d.set({ wheel })} options={opts(WHEEL_MODES, WHEEL_LABEL)} />
                </Field>
              </div>
              <Switch checked={d.invert} onChange={(invert) => d.set({ invert })}>Invert</Switch>
              <Switch checked={d.linked} onChange={(linked) => d.set({ linked })}>Link every viewer to these settings</Switch>
              <div className="settings-row">
                <Button size="sm" variant="primary" icon="contrast" onClick={openDisplayPanel}>
                  Open the Display panel
                </Button>
                <Button size="sm"
                  onClick={() => { d.reset(); toast("Display settings reset (absolute asinh, knee 100 e⁻)"); }}>
                  Reset display settings
                </Button>
              </div>
            </div>
          </CardBody>
        </Card>
      </div>
    </Page>
  );
}
