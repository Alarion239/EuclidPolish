/* Navigation metadata for the nine workspaces: rail icon, description,
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

export type TabMeta = { label: string; description?: string };

export type WorkspaceMeta = {
  icon: IconName;
  description: string;
  /** Second key of the "g <key>" go-to shortcut. */
  goKey: string;
  tabs: Record<string, TabMeta>;
  /** Labels of the values of each `:param` (e.g. the ensemble regime). */
  paramLabels?: Record<string, Record<string, string>>;
};

export const WORKSPACE_META: Record<string, WorkspaceMeta> = {
  home: {
    icon: "home", goKey: "h", tabs: {},
    description: "Connection, server version, running work and the production model at a glance",
  },
  sky: {
    icon: "globe", goKey: "s",
    description: "Euclid and JWST coverage and every real result on the celestial sphere",
    tabs: {
      atlas: { label: "Atlas", description: "The celestial sphere with coverage and result layers" },
      results: { label: "Real results", description: "Every real tile source with its SR products" },
      experiments: { label: "Experiments", description: "Compare models on real tiles with real-data metrics" },
      "catalog-eval": { label: "Catalog eval", description: "Reconstruction browser and the grouped analysis" },
    },
  },
  ensemble: {
    icon: "layers", goKey: "e",
    description: "Members, training, evaluation and combiners of the SR ensemble",
    paramLabels: { mode: { starfull: "starfull", starless: "starless" } },
    tabs: {
      overview: { label: "Overview" },
      members: { label: "Members" },
      curves: { label: "Curves", description: "Training curves" },
      knee: { label: "Knee PSNR", description: "PSNR-vs-knee curves and the integrated leaderboard" },
      diagnostics: { label: "Diagnostics", description: "Power spectrum, coherence, calibration" },
      combiners: { label: "Combiners", description: "Spatial-gate variants: fit, compare, promote" },
      disagreement: { label: "Disagreement", description: "Where the members disagree" },
      train: { label: "Train", description: "Train, continue or fork members on FASRC" },
    },
  },
  realism: {
    icon: "wave", goKey: "r",
    description: "How close the synthetic training data are to real Euclid",
    tabs: {
      overview: { label: "Overview", description: "Readiness of every prior" },
      noise: { label: "Noise" },
      galaxies: { label: "Galaxies", description: "Galaxy distributions" },
      stars: { label: "Stars", description: "Star distribution" },
      pixels: { label: "Pixels", description: "Field statistics" },
      visual: { label: "Visual", description: "Synthetic vs real, side by side" },
    },
  },
  data: {
    icon: "database", goKey: "d",
    description: "Training records, the star catalogue, cutouts, PSFs and TNG",
    tabs: {
      records: { label: "Records", description: "Training TFRecords" },
      catalog: { label: "Catalog", description: "Star catalogue" },
      cutouts: { label: "Cutouts" },
      psfs: { label: "PSFs" },
      tng: { label: "TNG" },
    },
  },
  figures: {
    icon: "image", goKey: "f",
    description: "Figure grids, publication plates and saved viewer results",
    tabs: {
      grid: { label: "Grid", description: "Figure grid builder" },
      plates: { label: "Plates", description: "Presentation and publication plates" },
      results: { label: "Results", description: "Saved viewer results" },
    },
  },
  inspect: {
    icon: "fileSearch", goKey: "i", tabs: {},
    description: "Open any FITS file: headers, HDUs and a preview",
  },
  ops: {
    icon: "server", goKey: "o",
    description: "Local jobs, FASRC, experiment tracking, git and provenance",
    tabs: {
      jobs: { label: "Jobs", description: "Local background jobs" },
      fasrc: { label: "FASRC", description: "The SLURM cluster console" },
      tracking: { label: "Tracking", description: "Experiment lab notebook" },
      git: { label: "Git" },
      provenance: { label: "Provenance", description: "Lineage and staleness" },
    },
  },
  settings: {
    icon: "settings", goKey: ",",
    description: "Job config, connections, appearance and version",
    tabs: {
      config: { label: "Config", description: "Universal job config" },
      connections: { label: "Connections", description: "FASRC and archive sessions" },
      appearance: { label: "Appearance", description: "Theme, accent, density, display defaults" },
      about: { label: "About", description: "Server and build version" },
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
  /** Labels of the path params (e.g. ["starless"]). */
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

/** `document.title` for a page: "Members · Ensemble (starless) · EuclidPolish". */
export function pageTitle(pathname: string): string {
  const info = describePath(pathname);
  if (!info) return `Not found · ${APP_NAME}`;
  const ws = info.paramLabels.length
    ? `${info.workspaceLabel} (${info.paramLabels.join(", ")})` : info.workspaceLabel;
  return [info.tabLabel, ws, APP_NAME].filter(Boolean).join(" · ");
}

export type NavTarget = {
  workspace: string;
  tab: string | null;
  params: Record<string, string>;
  path: string;
  label: string;
  description?: string;
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
        out.push({ workspace: ws.id, tab: null, params, path: concrete, label: `${ws.label}${suffix}`, description: meta?.description });
        continue;
      }
      for (const tab of ws.tabs) {
        out.push({
          workspace: ws.id, tab, params,
          path: pagePath(ws.id, { tab, params }),
          label: `${ws.label}${suffix} › ${tabLabel(ws.id, tab)}`,
          description: meta?.tabs[tab]?.description,
        });
      }
    }
  }
  return out;
}
