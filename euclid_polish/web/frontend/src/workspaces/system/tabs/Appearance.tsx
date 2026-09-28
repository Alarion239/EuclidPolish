/* System › Appearance (`/system/appearance`): theme (light / dark / system),
 * accent, density with a preview; the layout (rail, inspector width); and one
 * line saying how images are shown now — the live Display settings every
 * linked viewer follows (C7) — with the Display panel, the one place to
 * change them. Saved in this browser; nothing goes to the server. */
import type { ReactNode } from "react";
import { usePageActions } from "../../../app/palette";
import { openDisplayPanel } from "../../../app/shellStore";
import { useDisplay } from "../../../state/display";
import {
  ACCENTS, DEFAULT_PREFS, DENSITIES, THEME_PREFS, usePrefs, useResolvedTheme,
  type Accent, type Density, type ThemePref,
} from "../../../state/prefs";
import { Badge, Button, Card, CardBody, CardHead, Chip, Page, Segmented, Switch, toast } from "../../../ui";
import { PageLead } from "../../shared/PageLead";
import { imagesLine } from "../model";
import "../system.css";

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
    { id: "appearance:reset", label: "Reset the appearance", group: "Appearance", keywords: ["theme", "accent", "density"],
      run: () => { prefs.reset(); toast("Appearance reset to the defaults"); } },
  ]);
  return (
    <Page className="sys-page">
      <PageLead>Theme and layout. Saved in this browser only.</PageLead>
      <div className="sys-cards">
        <Card>
          <CardHead title="Theme" sub={prefs.theme === "system" ? `following the system (now ${resolved})` : undefined} />
          <CardBody>
            <div className="sys-stack">
              <Group label="Theme">
                <Segmented<ThemePref> aria-label="Theme" value={prefs.theme} onChange={prefs.setTheme}
                  options={THEME_PREFS.map((t) => ({ value: t, label: THEME_LABEL[t] }))} />
              </Group>
              <Group label="Accent">
                <div className="sys-row">
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
            <div className="sys-stack">
              <Switch checked={prefs.railCollapsed} onChange={(railCollapsed) => prefs.set({ railCollapsed })}>
                Collapse the navigation rail to icons (shortcut [)
              </Switch>
              <div className="sys-row">
                <span className="sys-dim">Inspector width <b className="mono">{prefs.inspectorWidth}px</b></span>
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
      </div>
      <p className="sys-images">
        <span><span className="sys-dim">Images:</span> {imagesLine(d)}</span>
        <Button size="sm" variant="ghost" icon="contrast" onClick={openDisplayPanel}>Open Display panel</Button>
      </p>
    </Page>
  );
}
