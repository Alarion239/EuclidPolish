/* ⌘K command palette (spec §4, cmdk): search over
 *   - suggestions from the typed text (RA/Dec, member N, NEXUS tile N, FITS
 *     path — `paletteSuggestions`), first;
 *   - the current page's actions (`usePageActions`);
 *   - every page: each workspace × tab (× ensemble regime);
 *   - global commands: theme, Display panel, jobs, run a FASRC step, shortcuts,
 *     rail, inspector, refresh data, copy link;
 *   - the global "Run a job" launchers (RunActions);
 *   - "Find <text> on the sky" (the name resolver), always last.
 * The ranking is ours (`rankPalette`, cmdk's filter is off): the page or
 * command the user named comes first — an exact or word-start match beats a
 * scattered-letter one, pages and commands win ties over job launchers — so
 * Enter does what was typed. Selecting an entry runs it and closes the
 * palette. "Nothing matches" shows only when there is nothing to pick. */
import * as RDialog from "@radix-ui/react-dialog";
import { Command } from "cmdk";
import { useMemo, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { invalidate } from "../api/query";
import { comboParts } from "../hooks/useShortcut";
import { parseSkyCoord } from "../format";
import { useInspector } from "../state/inspector";
import { usePrefs } from "../state/prefs";
import { Icon, Kbd, copyText, toast, type IconName } from "../ui";
import { closeInspector, openInspector } from "./inspector";
import { WORKSPACE_META, allPages, pagePath } from "./nav";
import { paletteSuggestions, usePaletteActions, type PageAction } from "./palette";
import { rankPalette, type RankGroup } from "./paletteRank";
import { RUN_GROUP } from "./RunActions";
import { useShellUi } from "./shellStore";

type Entry = {
  key: string;
  label: string;
  hint?: string;
  icon?: IconName;
  keywords?: string[];
  shortcut?: string;
  disabled?: boolean;
  run: () => void;
};

function Row({ entry, onPick }: { entry: Entry; onPick: (e: Entry) => void }) {
  return (
    <Command.Item value={entry.key} disabled={entry.disabled} onSelect={() => onPick(entry)} className="cmdk__item">
      {entry.icon && <Icon name={entry.icon} size={15} className="cmdk__icon" />}
      <span className="cmdk__label">{entry.label}</span>
      {entry.hint && <span className="cmdk__hint">{entry.hint}</span>}
      {entry.shortcut && (
        <span className="cmdk__keys">{comboParts(entry.shortcut).map((k, i) => <Kbd key={i} keys={k} />)}</span>
      )}
    </Command.Item>
  );
}

function Group({ heading, children }: { heading: string; children: ReactNode }) {
  return <Command.Group heading={heading} className="cmdk__group">{children}</Command.Group>;
}

/** Page actions by group, the page's own groups first; the shell's global
 *  "Run a job" group (RunActions) comes after them. */
export function groupActions(actions: PageAction[]): [string, PageAction[]][] {
  const groups = new Map<string, PageAction[]>();
  for (const a of actions) {
    const g = a.group ?? "This page";
    groups.set(g, [...(groups.get(g) ?? []), a]);
  }
  const entries = [...groups.entries()];
  return [...entries.filter(([g]) => g !== RUN_GROUP), ...entries.filter(([g]) => g === RUN_GROUP)];
}

export function CommandPalette() {
  const open = useShellUi((s) => s.palette);
  const [query, setQuery] = useState("");
  const navigate = useNavigate();
  const actions = usePaletteActions();
  const inspectorOpen = useInspector((s) => s.open);
  const pages = useMemo(() => allPages(), []);

  const setOpen = (v: boolean) => {
    useShellUi.getState().setOpen("palette", v);
    if (!v) setQuery("");
  };
  const pick = (entry: Entry) => {
    setOpen(false);
    entry.run();
  };

  const typed = paletteSuggestions(query, parseSkyCoord);
  const toEntry = (s: (typeof typed)[number]): Entry => ({
    key: `suggest:${s.id}`, label: s.label, hint: s.hint,
    icon: s.kind === "inspect" ? "panelRight" : s.id.startsWith("sky") ? "globe" : "fileSearch",
    run: s.kind === "navigate" ? () => navigate(s.to) : () => openInspector(s.target),
  });
  const suggestions = typed.filter((s) => !s.fallback).map(toEntry);
  const fallbacks = typed.filter((s) => s.fallback).map(toEntry);

  const pageEntries: Entry[] = pages.map((p) => ({
    key: `page:${p.path}`, label: p.label, hint: p.description, icon: WORKSPACE_META[p.workspace]?.icon,
    keywords: [p.workspace, p.tab ?? "", p.path, ...(p.keywords ?? [])],
    run: () => navigate(p.path),
  }));

  const prefs = usePrefs.getState();
  const commands: Entry[] = [
    { key: "cmd:theme", label: "Toggle light / dark theme", icon: "sun", shortcut: "Shift+T", run: () => prefs.toggleTheme() },
    { key: "cmd:theme-light", label: "Theme: light", icon: "sun", keywords: ["appearance"], run: () => prefs.setTheme("light") },
    { key: "cmd:theme-dark", label: "Theme: dark", icon: "moon", keywords: ["appearance"], run: () => prefs.setTheme("dark") },
    { key: "cmd:theme-system", label: "Theme: follow the system", icon: "monitor", keywords: ["appearance", "auto"], run: () => prefs.setTheme("system") },
    { key: "cmd:display", label: "Open the Display panel", icon: "contrast", shortcut: "Shift+D", keywords: ["colour", "color", "stretch", "knee"], run: () => useShellUi.getState().openOnly("display") },
    { key: "cmd:jobs", label: "Show running jobs", icon: "activity", shortcut: "Shift+J", keywords: ["tray", "slurm"], run: () => useShellUi.getState().openOnly("tray") },
    { key: "cmd:settings-connections", label: "FASRC & archive connections", hint: "System › Connections", icon: "server",
      keywords: ["ssh", "connect", "login", "euclid", "tng", "token", "credentials", "settings"], run: () => navigate(pagePath("system", { tab: "connections" })) },
    {
      key: "cmd:fasrc-step", label: "Run a FASRC step…", hint: "Runs › Steps", icon: "server",
      keywords: ["run job", "job", "slurm", "submit", "step", "pipeline", "cluster", "sbatch"],
      run: () => navigate(pagePath("runs", { tab: "steps" })),
    },
    { key: "cmd:files", label: "Open a file", hint: "Files", icon: "fileSearch",
      keywords: ["fits", "inspect", "hdu", "header", "browse"], run: () => navigate(pagePath("files")) },
    { key: "cmd:shortcuts", label: "Keyboard shortcuts", icon: "keyboard", shortcut: "Shift+?", run: () => useShellUi.getState().openOnly("shortcuts") },
    { key: "cmd:rail", label: "Collapse / expand the navigation", icon: "sidebar", shortcut: "[", run: () => prefs.toggleRail() },
    ...(inspectorOpen ? [{ key: "cmd:close-inspector", label: "Close the inspector", icon: "close" as IconName, run: closeInspector }] : []),
    { key: "cmd:refresh", label: "Refresh all data", icon: "reset", keywords: ["reload", "invalidate"], run: () => { void invalidate(); toast("Refreshing data…"); } },
    {
      key: "cmd:copy-link", label: "Copy a link to this view", icon: "link", keywords: ["share", "url"],
      run: () => { void copyText(window.location.href).then((ok) => (ok ? toast.success("Link copied") : toast.error("Could not copy"))); },
    },
  ];

  // Pages and commands before the job launchers; a small bias breaks ties.
  const actionGroups = groupActions(actions).map(([heading, list]): RankGroup<Entry> => ({
    heading,
    bias: heading === RUN_GROUP ? -1 : 0,
    items: list.map((a) => ({
      key: `action:${a.id}`, label: a.label, keywords: a.keywords, shortcut: a.shortcut,
      disabled: a.disabled, icon: "command" as IconName, run: a.run,
    })),
  }));
  const groups = rankPalette(query, [
    ...actionGroups.filter((g) => g.heading !== RUN_GROUP),
    { heading: "Pages", items: pageEntries, bias: 2 },
    { heading: "Commands", items: commands, bias: 1 },
    ...actionGroups.filter((g) => g.heading === RUN_GROUP),
  ]);
  const nothing = suggestions.length === 0 && groups.length === 0 && fallbacks.length === 0;

  return (
    <RDialog.Root open={open} onOpenChange={setOpen}>
      <RDialog.Portal>
        <RDialog.Overlay className="ui-dialog__overlay cmdk__overlay" />
        <RDialog.Content className="cmdk" aria-describedby={undefined}>
          <RDialog.Title className="sr-only">Command palette</RDialog.Title>
          {/* vimBindings off: Ctrl-K is the palette's own toggle, not "up".
              shouldFilter off: `rankPalette` filters and orders. */}
          <Command label="Command palette" loop vimBindings={false} shouldFilter={false}>
            <div className="cmdk__search">
              <Icon name="search" size={16} />
              <Command.Input value={query} onValueChange={setQuery} className="cmdk__input"
                placeholder="Search pages and actions, or type RA Dec, member 196, nexus 12, a .fits path…" />
              <Kbd keys="escape" />
            </div>
            <Command.List className="cmdk__list" label="Results">
              {nothing && <div className="cmdk__empty" role="status">Nothing matches “{query}”.</div>}
              {suggestions.length > 0 && (
                <Group heading="Go to">{suggestions.map((e) => <Row key={e.key} entry={e} onPick={pick} />)}</Group>
              )}
              {groups.map((g) => (
                <Group key={g.heading} heading={g.heading}>
                  {g.items.map((e) => <Row key={e.key} entry={e} onPick={pick} />)}
                </Group>
              ))}
              {fallbacks.length > 0 && (
                <Group heading="Search the sky">{fallbacks.map((e) => <Row key={e.key} entry={e} onPick={pick} />)}</Group>
              )}
            </Command.List>
          </Command>
        </RDialog.Content>
      </RDialog.Portal>
    </RDialog.Root>
  );
}
