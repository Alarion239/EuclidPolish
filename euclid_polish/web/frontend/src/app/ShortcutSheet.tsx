/* The "?" cheat sheet: every shortcut bound right now (global, navigation and
 * the current page's), grouped by scope, from the useShortcut registry. */
import { comboParts, useShortcutRegistry, type ShortcutEntry } from "../hooks/useShortcut";
import { Dialog, Kbd } from "../ui";
import { useShellUi } from "./shellStore";

function Keys({ combo }: { combo: string }) {
  const parts = comboParts(combo);
  return (
    <span className="shortcuts__keys">
      {parts.map((p, i) => (
        <span key={i}>
          {i > 0 && <span className="shortcuts__then">then</span>}
          <Kbd keys={p} />
        </span>
      ))}
    </span>
  );
}

export function groupShortcuts(entries: ShortcutEntry[]): [string, ShortcutEntry[]][] {
  const groups = new Map<string, ShortcutEntry[]>();
  const seen = new Set<string>();
  for (const e of entries) {
    if (e.hidden) continue;
    const id = `${e.scope}\u0000${e.combo}`;
    if (seen.has(id)) continue; // the same shortcut bound twice (e.g. two viewers)
    seen.add(id);
    groups.set(e.scope, [...(groups.get(e.scope) ?? []), e]);
  }
  return [...groups.entries()];
}

export function ShortcutSheet() {
  const open = useShellUi((s) => s.shortcuts);
  const entries = useShortcutRegistry((s) => s.entries);
  const setOpen = (v: boolean) => useShellUi.getState().setOpen("shortcuts", v);
  return (
    <Dialog open={open} onOpenChange={setOpen} title="Keyboard shortcuts" size="md"
      description="Shortcuts do not fire while you type in a field.">
      <div className="shortcuts">
        {groupShortcuts(entries).map(([scope, list]) => (
          <section key={scope} className="shortcuts__group">
            <h3 className="eyebrow">{scope}</h3>
            <dl>
              {list.map((e) => (
                <div key={e.id} className="shortcuts__row">
                  <dt>{e.description}</dt>
                  <dd><Keys combo={e.combo} /></dd>
                </div>
              ))}
            </dl>
          </section>
        ))}
      </div>
    </Dialog>
  );
}
