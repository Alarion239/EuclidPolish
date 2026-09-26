/* settings/appearance (spec §8.8): theme (light / dark / system), accent,
   density, the navigation rail, the inspector width, and the Display-panel
   defaults with the viewer wheel behaviour. Everything is saved in this
   browser (prefs + display stores). */
import type { ReactNode } from "react";
import { openDisplayPanel } from "../../../app/shellStore";
import { WHEEL_MODES, useDisplay, type WheelMode } from "../../../state/display";
import {
  ACCENTS, DEFAULT_PREFS, DENSITIES, THEME_PREFS, usePrefs, useResolvedTheme,
  type Accent, type Density, type ThemePref,
} from "../../../state/prefs";
import {
  Button, Card, CardBody, CardHead, Chip, Field, Page, PageHead, Segmented, Select, Switch, toast,
} from "../../../ui";
import "./settings.css";

/* A caption over a group control (Segmented, chips): a <Field> would wrap
   several buttons in one <label> and rename them. */
function Group({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="appearance-group">
      <span className="appearance-group__label">{label}</span>
      {children}
    </div>
  );
}

const THEME_LABEL: Record<ThemePref, string> = { light: "Light", dark: "Dark", system: "System" };
const DENSITY_LABEL: Record<Density, string> = { comfortable: "Comfortable", compact: "Compact" };
const WHEEL_LABEL: Record<WheelMode, string> = {
  "zoom-when-focused": "Zoom when the viewer is focused (or ⌘/Ctrl held)",
  "always-zoom": "Always zoom",
  scroll: "Scroll the page (zoom with ⌘/Ctrl)",
};

export default function Appearance() {
  const prefs = usePrefs();
  const resolved = useResolvedTheme();
  const wheel = useDisplay((s) => s.wheel);
  const setDisplay = useDisplay((s) => s.set);
  const resetDisplay = useDisplay((s) => s.reset);
  return (
    <Page>
      <PageHead eyebrow="settings · appearance" title="Appearance"
        sub="How the console looks in this browser. Saved locally; nothing is sent to the server." />
      <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))", gap: "var(--s4)" }}>
        <Card>
          <CardHead title="Theme" sub={prefs.theme === "system" ? `following the system (now ${resolved})` : undefined} />
          <CardBody>
            <div className="grid" style={{ gap: "var(--s4)" }}>
              <Group label="Theme">
                <Segmented<ThemePref> aria-label="Theme" value={prefs.theme} onChange={prefs.setTheme}
                  options={THEME_PREFS.map((t) => ({ value: t, label: THEME_LABEL[t] }))} />
              </Group>
              <Group label="Accent">
                <div className="row" role="group" aria-label="Accent" style={{ gap: "var(--s2)" }}>
                  {ACCENTS.map((a) => (
                    <Chip key={a} on={prefs.accent === a} onClick={() => prefs.set({ accent: a as Accent })}
                      aria-label={`Accent ${a}`}>
                      {a}
                    </Chip>
                  ))}
                </div>
              </Group>
              <Group label="Density">
                <Segmented<Density> aria-label="Density" value={prefs.density} onChange={(density) => prefs.set({ density })}
                  options={DENSITIES.map((d) => ({ value: d, label: DENSITY_LABEL[d] }))} />
              </Group>
            </div>
          </CardBody>
        </Card>
        <Card>
          <CardHead title="Layout" />
          <CardBody>
            <div className="grid" style={{ gap: "var(--s4)" }}>
              <Switch checked={prefs.railCollapsed} onChange={(railCollapsed) => prefs.set({ railCollapsed })}>
                Collapse the navigation rail to icons (shortcut [)
              </Switch>
              <div className="row" style={{ gap: "var(--s3)" }}>
                <span className="muted">Inspector width: <b className="mono">{prefs.inspectorWidth}px</b></span>
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
        <Card>
          <CardHead title="Images" sub="defaults of every linked viewer" />
          <CardBody>
            <div className="grid" style={{ gap: "var(--s4)" }}>
              <Field label="Mouse wheel over a viewer">
                <Select<WheelMode> value={wheel} onChange={(w) => setDisplay({ wheel: w })}
                  options={WHEEL_MODES.map((w) => ({ value: w, label: WHEEL_LABEL[w] }))} />
              </Field>
              <div className="row" style={{ gap: "var(--s2)" }}>
                <Button size="sm" variant="primary" onClick={openDisplayPanel}>Open the Display panel</Button>
                <Button size="sm" onClick={() => { resetDisplay(); toast("Display settings reset (absolute asinh, knee 100 e⁻)"); }}>
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
