/* Navigation metadata for the nine workspaces (the "Loop console" rail:
 * Home, Synthetic, Models, Sky, Figures, Files, Runs, Notebook, System): rail icon, description,
 * "g <key>" shortcut and the human label of every tab.
 *
 * The URLs themselves come from the route manifest (`spa_routes.json`, C1,
 * via `manifest.ts`); this module only adds the presentation the manifest
 * does not carry. `nav.test.ts` fails when a manifest workspace or tab has no
 * entry here (or when an entry names a tab the manifest lacks), so adding a
 * tab to the manifest forces a label.
 *
 * Pure (no React), so the palette, breadcrumbs, rail, document titles and
 * tests share it.
 */
import type { IconName } from "../ui/icons";
import { MANIFEST, matchPage, workspace, workspacePaths, type PageMatch, type WorkspaceDef } from "./manifest";

/** `keywords`: other words the palette finds the page by (e.g. the page's
 *  old name: "git" finds System › Code). */
export type TabMeta = { label: string; description?: string; keywords?: string[] };

export type WorkspaceMeta = {
  icon: IconName;
  description: string;
  /** Second key of the "g <key>" go-to shortcut. */
  goKey: string;
  tabs: Record<string, TabMeta>;
  /** Palette words for every page of the workspace (e.g. its old name). */
  keywords?: string[];
  /** Labels of the values of each `:param` of the workspace path. */
  paramLabels?: Record<string, Record<string, string>>;
};

export const WORKSPACE_META: Record<string, WorkspaceMeta> = {
  home: {
    icon: "home", goKey: "h", tabs: {},
    description: "What changed, what is running and what needs you now",
  },
  synthetic: {
    icon: "wave", goKey: "y", keywords: ["realism", "data"],
    description: "What goes into a synthetic scene, how each ingredient matches Euclid Q1, and whether you can generate",
    tabs: {
      status: { label: "Status", description: "Can you generate: every ingredient's state and the generation gate", keywords: ["overview", "readiness", "generate"] },
      records: { label: "Records", description: "Generated records against their truth sources, and the census", keywords: ["tfrecords", "census", "truth"] },
      galaxies: { label: "Galaxies", description: "Galaxy distributions against Q1, the prior and the TNG templates", keywords: ["tng", "templates", "galaxy distributions"] },
      stars: { label: "Stars", description: "Star density and colours against Q1, and the stellar prior" },
      noise: { label: "Noise", description: "How a scene gets its noise, from Q1 noise maps" },
      psf: { label: "PSF", description: "Real Euclid stars, their cutouts and the empirical PSFs", keywords: ["catalog", "catalogue", "cutouts", "psfs", "epsf"] },
      fields: { label: "Fields", description: "Synthetic against real LR fields: look, statistics, detection", keywords: ["pixels", "visual", "field statistics", "detection"] },
    },
  },
  models: {
    icon: "layers", goKey: "m", keywords: ["ensemble"],
    description: "Which model is best on synthetic truth and real data, how the members trained, and which combiner is production",
    tabs: {
      leaderboard: { label: "Leaderboard", description: "Production gate, plain mean and members ranked, with the knee curves", keywords: ["overview", "knee", "psnr"] },
      members: { label: "Members", description: "The roster, training curves and archived members", keywords: ["curves", "roster"] },
      train: { label: "Train", description: "Train, continue or fork members on FASRC" },
      combiner: { label: "Combiner", description: "Spatial-gate variants: fit, compare, promote", keywords: ["combiners", "gate", "spatial gate"] },
      diagnostics: { label: "Diagnostics", description: "Spectrum, transfer, coherence, spread, real field, recovery", keywords: ["spectrum", "coherence", "calibration"] },
      images: { label: "Images", description: "SR on synthetic test fields and stamps, with the members' disagreement", keywords: ["disagreement", "stamps"] },
    },
  },
  sky: {
    icon: "globe", goKey: "s",
    description: "What SR does on real Euclid sky: tiles and targets, production SR on each, and the safest model",
    tabs: {
      atlas: { label: "Atlas", description: "Where it is: the celestial sphere with coverage, tiles and targets", keywords: ["jwst", "map"] },
      targets: { label: "Targets", description: "Production SR on each science target, and whether it is current", keywords: ["results", "real results", "catalog eval", "lenses", "evaluation", "inference"] },
      compare: { label: "Compare", description: "Models on real tiles, no truth: holes, flux, JWST", keywords: ["experiments", "holes"] },
    },
  },
  figures: {
    icon: "image", goKey: "f",
    description: "Figures for the paper and poster, whether they use the current model, and their export",
    tabs: {
      plates: { label: "Plates", description: "Presentation and publication plates", keywords: ["publication", "poster"] },
      sheet: { label: "Sheet", description: "Contact sheets of saved crops", keywords: ["grid", "results", "crops"] },
      studies: { label: "Studies", description: "Frozen whole-ensemble comparisons for the paper: charts, exports and attached fields", keywords: ["study", "freeze", "paired", "bootstrap", "loss choice"] },
    },
  },
  files: {
    icon: "fileSearch", goKey: "i", tabs: {}, keywords: ["inspect", "fits", "hdu", "header"],
    description: "What is inside a FITS file, and where it came from",
  },
  runs: {
    icon: "activity", goKey: "r", keywords: ["ops", "jobs", "fasrc", "slurm"],
    description: "What is running or queued, locally and on FASRC, how past runs went, and which step makes what",
    tabs: {
      live: { label: "Live", description: "Running and queued jobs, local and SLURM", keywords: ["queue", "running"] },
      history: { label: "History", description: "Every past run, with its logs", keywords: ["logs", "ledger"] },
      resources: { label: "Resources", description: "What past runs asked for vs used, and what to ask for next",
        keywords: ["cpu", "memory", "time", "gpu", "efficiency", "sacct"] },
      steps: { label: "Steps", description: "The FASRC step catalogue by stage", keywords: ["pipeline", "submit"] },
    },
  },
  notebook: {
    icon: "copy", goKey: "n", keywords: ["tracking", "campaign"],
    description: "What we tried and concluded, and how to get the exact state back",
    tabs: {
      log: { label: "Log", description: "The campaign's lab notebook", keywords: ["notebook"] },
      backups: { label: "Backups", description: "Model, FITS and image backups with time travel", keywords: ["archive", "time travel"] },
      sandboxes: { label: "Sandboxes", description: "Running time-travel servers" },
    },
  },
  system: {
    icon: "settings", goKey: ",", keywords: ["settings"],
    description: "Connections, the job config, lineage, code and disk",
    tabs: {
      connections: { label: "Connections", description: "FASRC, the Euclid archive and the TNG token", keywords: ["ssh", "login"] },
      config: { label: "Config", description: "The universal job config", keywords: ["job config", "knobs"] },
      lineage: { label: "Lineage", description: "Where a product came from, and whether it is stale", keywords: ["provenance", "staleness"] },
      code: { label: "Code", description: "Laptop, server and FASRC commits", keywords: ["git", "commit", "about", "version"] },
      storage: { label: "Storage", description: "Local and FASRC disk, data roots and caches", keywords: ["disk", "about"] },
      appearance: { label: "Appearance", description: "Theme, accent, density and layout", keywords: ["theme"] },
    },
  },
};

export const APP_NAME = "EuclidPolish";

export function workspaceMeta(id: string): WorkspaceMeta {
  const meta = WORKSPACE_META[id];
  if (!meta) throw new Error(`no navigation metadata for workspace "${id}" (app/nav.ts)`);
  return meta;
}

/** "catalog-eval" → "Catalog eval" (fallback for a tab without a label). */
export function humanize(slug: string): string {
  const s = slug.replace(/[-_]+/g, " ").trim();
  return s ? s[0].toUpperCase() + s.slice(1) : slug;
}

export function workspaceLabel(id: string): string {
  return MANIFEST.workspaces.find((w) => w.id === id)?.label ?? humanize(id);
}

export function tabLabel(workspaceId: string, tab: string): string {
  return WORKSPACE_META[workspaceId]?.tabs[tab]?.label ?? humanize(tab);
}

export function paramLabel(workspaceId: string, name: string, value: string): string {
  return WORKSPACE_META[workspaceId]?.paramLabels?.[name]?.[value] ?? value;
}

/** The concrete base path of a workspace for `params` (defaults filled in). */
export function basePath(ws: WorkspaceDef, params: Record<string, string> = {}): string {
  let path = ws.path;
  for (const [name, values] of Object.entries(ws.params ?? {})) {
    const wanted = params[name] ?? ws.defaultParams?.[name] ?? values[0];
    const value = values.includes(wanted) ? wanted : (ws.defaultParams?.[name] ?? values[0]);
    path = path.replace(`:${name}`, value);
  }
  return path;
}

/** The URL of one tab (or of the workspace itself when it has no tabs). */
export function pagePath(workspaceId: string, opts: { tab?: string | null; params?: Record<string, string> } = {}): string {
  const ws = workspace(workspaceId);
  const base = basePath(ws, opts.params);
  const tab = opts.tab ?? ws.defaultTab ?? ws.tabs[0] ?? null;
  if (!tab || !ws.tabs.includes(tab)) return base;
  return base === "/" ? `/${tab}` : `${base}/${tab}`;
}

/** Where a rail entry / "g <key>" goes: the default tab with default params. */
export function landingPath(workspaceId: string): string {
  return pagePath(workspaceId);
}

export type LocationInfo = {
  match: PageMatch;
  workspaceLabel: string;
  tabLabel: string | null;
  /** Labels of the path params (empty for a workspace without params). */
  paramLabels: string[];
};

/** Labels for the page a pathname addresses (null for non-pages). */
export function describePath(pathname: string): LocationInfo | null {
  const match = matchPage(pathname);
  if (!match) return null;
  return {
    match,
    workspaceLabel: workspaceLabel(match.workspace),
    tabLabel: match.tab ? tabLabel(match.workspace, match.tab) : null,
    paramLabels: Object.entries(match.params).map(([k, v]) => paramLabel(match.workspace, k, v)),
  };
}

/** "Workspace (param)": the workspace with its path parameters, if any. */
function workspaceWithParams(info: LocationInfo): string {
  return info.paramLabels.length
    ? `${info.workspaceLabel} (${info.paramLabels.join(", ")})` : info.workspaceLabel;
}

/** `document.title` for a page: "Members · Models · EuclidPolish". */
export function pageTitle(pathname: string): string {
  const info = describePath(pathname);
  if (!info) return `Not found · ${APP_NAME}`;
  return [info.tabLabel, workspaceWithParams(info), APP_NAME].filter(Boolean).join(" · ");
}

/** The page's one h1 (visually hidden; screen readers and the outline), in
 *  plain words: "Members, Models", "Records, Synthetic", "Home". */
export function pageHeading(pathname: string): string {
  const info = describePath(pathname);
  if (!info) return "Not found";
  const ws = workspaceWithParams(info);
  return info.tabLabel ? `${info.tabLabel}, ${ws}` : ws;
}

export type NavTarget = {
  workspace: string;
  tab: string | null;
  params: Record<string, string>;
  path: string;
  label: string;
  description?: string;
  /** Palette words (the workspace's and the tab's `keywords`). */
  keywords?: string[];
};

/** Every addressable page (each param value × tab), for the palette. */
export function allPages(): NavTarget[] {
  const out: NavTarget[] = [];
  for (const ws of MANIFEST.workspaces) {
    const meta = WORKSPACE_META[ws.id];
    const names = Object.keys(ws.params ?? {});
    for (const concrete of workspacePaths(ws)) {
      const m = matchPage(concrete);
      const params = m?.params ?? {};
      const suffix = names.length
        ? ` (${names.map((n) => paramLabel(ws.id, n, params[n])).join(", ")})` : "";
      if (!ws.tabs.length) {
        out.push({
          workspace: ws.id, tab: null, params, path: concrete, label: `${ws.label}${suffix}`,
          description: meta?.description, keywords: meta?.keywords,
        });
        continue;
      }
      for (const tab of ws.tabs) {
        out.push({
          workspace: ws.id, tab, params,
          path: pagePath(ws.id, { tab, params }),
          label: `${ws.label}${suffix} › ${tabLabel(ws.id, tab)}`,
          description: meta?.tabs[tab]?.description,
          keywords: [...(meta?.keywords ?? []), ...(meta?.tabs[tab]?.keywords ?? [])],
        });
      }
    }
  }
  return out;
}
