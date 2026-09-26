/* The shell's global shortcuts (listed in the ? sheet):
 *   ⌘/Ctrl-K  palette (also inside fields)   ?        this sheet
 *   g h / g s / g e / g r / g d / g f / g i / g o / g ,   go to a workspace
 *   [  collapse / expand the rail             ]        show / hide the inspector
 *   Shift-D  Display panel   Shift-J  jobs   Shift-T  toggle theme */
import { useNavigate } from "react-router-dom";
import { useShortcut } from "../hooks/useShortcut";
import { useInspector } from "../state/inspector";
import { usePrefs } from "../state/prefs";
import { MANIFEST } from "./manifest";
import { WORKSPACE_META, landingPath } from "./nav";
import { useShellUi } from "./shellStore";

function GoTo({ id }: { id: string }) {
  const navigate = useNavigate();
  const ws = MANIFEST.workspaces.find((w) => w.id === id)!;
  useShortcut(`g ${WORKSPACE_META[id].goKey}`, () => navigate(landingPath(id)), {
    description: `Go to ${ws.label}`, scope: "Navigation",
  });
  return null;
}

/** Renders nothing; binds the global shortcuts while mounted. */
export function GlobalShortcuts() {
  const shell = useShellUi.getState;
  useShortcut("$mod+k", () => { shell().toggle("palette"); }, {
    description: "Command palette", allowInInputs: true,
  });
  useShortcut("[Shift]+?", () => { shell().openOnly("shortcuts"); }, { description: "Keyboard shortcuts" });
  useShortcut("[", () => { usePrefs.getState().toggleRail(); }, { description: "Collapse / expand the navigation" });
  useShortcut("]", () => {
    const s = useInspector.getState();
    if (s.open) s.hide(); else if (s.current) s.setOpen(true); else return false;
    return true;
  }, { description: "Show / hide the inspector" });
  useShortcut("Shift+D", () => { shell().openOnly("display"); }, { description: "Display panel" });
  useShortcut("Shift+J", () => { shell().openOnly("tray"); }, { description: "Running jobs" });
  useShortcut("Shift+T", () => { usePrefs.getState().toggleTheme(); }, { description: "Toggle light / dark theme" });
  return (
    <>
      {MANIFEST.workspaces.map((ws) => <GoTo key={ws.id} id={ws.id} />)}
    </>
  );
}
