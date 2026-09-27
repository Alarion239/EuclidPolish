/* settings/appearance (spec §8.8): theme (light / dark / system), accent,
 * density, the rail and the inspector width. "Images" summarises the live
 * Display settings every linked viewer follows (the Display panel's store,
 * C7) and opens the Display panel — the one place to change them (a second
 * set of controls here edited the same live store while calling it
 * "defaults"). Saved in this browser; nothing goes to the server. */
import type { ReactNode } from "react";
import { usePageActions } from "../../../app/palette";
import { openDisplayPanel } from "../../../app/shellStore";
import { useDisplay } from "../../../state/display";
import {
  ACCENTS, DEFAULT_PREFS, DENSITIES, THEME_PREFS, usePrefs, useResolvedTheme,
  type Accent, type Density, type ThemePref,
} from "../../../state/prefs";
import {
  Badge, Button, Card, CardBody, CardHead, Chip, DefList, Page, Segmented, Switch, toast,
} from "../../../ui";
import { PageLead } from "../../shared/PageLead";
import { displaySummary } from "../appearanceModel";
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
  const facts = displaySummary(d);
  const changed = facts.filter((f) => f.changed).length;
  return (
    <Page className="settings-appearance">
      <PageLead>Theme, layout and how images are shown. Saved in this browser only.</PageLead>
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
          <CardHead title="Images" sub="How every linked viewer shows images now. Change them in the Display panel (Shift-D, or the ◐ button in the top bar)."
            right={changed ? <Badge tone="info">{changed} changed</Badge> : <Badge>as installed</Badge>} />
          <CardBody>
            <div className="settings-stack">
              <DefList dense items={facts.map((f) => [f.label, f.changed ? <span className="appearance-changed">{f.value}</span> : f.value])} />
              <div className="settings-row">
                <Button size="sm" variant="primary" icon="contrast" onClick={openDisplayPanel}>
                  Open the Display panel
                </Button>
                <Button size="sm" disabled={!changed}
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
