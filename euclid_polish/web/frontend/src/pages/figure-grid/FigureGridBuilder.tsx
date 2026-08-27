import { useEffect, useMemo, useRef, useState } from "react";
import { useResource } from "../../hooks";
import { Badge, Button, Empty, Spinner } from "../../ui";
import type {
  FigureMode,
  FigureRecipe,
  FigureRecipeKey,
  FigureRegime,
  FigureTier,
  RecipeRow,
  SavedResult,
  SavedResultSource,
  ViewerResultsIndex,
} from "./types";
import {
  canAddGridRow,
  MAX_GRID_RESULTS,
  MAX_GRID_ROWS,
  selectionForPreset,
  toggleGridResult,
} from "./limits";
import "./figure-grid.css";

type Preset = {
  id: string;
  label: string;
  regime: FigureRegime;
  rows: FigureRecipe[];
};

type NormalizedIndex = {
  index: ViewerResultsIndex;
  malformed: boolean;
  droppedResults: number;
};

const PRESETS: Preset[] = [
  {
    id: "real-vis-h",
    label: "Real · VIS/H composite",
    regime: "real",
    rows: [
      { tier: "dirty", mode: "VIS", title: "VIS Dirty" },
      { tier: "dirty", mode: "H_E", title: "H_E Dirty" },
      { tier: "sr", mode: "VIS_H", title: "VIS+H_E SR" },
      { tier: "jwst", mode: "native", title: "NEXUS F200W" },
    ],
  },
  {
    id: "real-bandwise",
    label: "Real · bandwise SR",
    regime: "real",
    rows: [
      { tier: "dirty", mode: "VIS", title: "VIS Dirty" },
      { tier: "sr", mode: "VIS", title: "VIS SR" },
      { tier: "dirty", mode: "H_E", title: "H_E Dirty" },
      { tier: "sr", mode: "H_E", title: "H_E SR" },
      { tier: "jwst", mode: "native", title: "NEXUS F200W" },
    ],
  },
  {
    id: "real-input-composite",
    label: "Real · input composite",
    regime: "real",
    rows: [
      { tier: "dirty", mode: "VIS", title: "VIS Dirty" },
      { tier: "dirty", mode: "VIS_H", title: "VIS+H_E Dirty" },
      { tier: "sr", mode: "VIS_H", title: "VIS+H_E SR" },
      { tier: "jwst", mode: "native", title: "NEXUS F200W" },
    ],
  },
  {
    id: "synthetic-bandwise",
    label: "Synthetic · bandwise",
    regime: "synthetic",
    rows: [
      { tier: "dirty", mode: "VIS", title: "VIS Dirty" },
      { tier: "sr", mode: "VIS", title: "VIS SR" },
      { tier: "hr", mode: "VIS", title: "VIS HR" },
      { tier: "dirty", mode: "H_E", title: "H_E Dirty" },
      { tier: "sr", mode: "H_E", title: "H_E SR" },
      { tier: "hr", mode: "H_E", title: "H_E HR" },
    ],
  },
  {
    id: "synthetic-composite",
    label: "Synthetic · VIS/H composite",
    regime: "synthetic",
    rows: [
      { tier: "dirty", mode: "VIS_H", title: "VIS+H_E Dirty" },
      { tier: "sr", mode: "VIS_H", title: "VIS+H_E SR" },
      { tier: "hr", mode: "VIS_H", title: "VIS+H_E HR" },
    ],
  },
];

const TIERS: { value: FigureTier; label: string }[] = [
  { value: "dirty", label: "Dirty / LR" },
  { value: "sr", label: "SR" },
  { value: "hr", label: "HR truth" },
  { value: "bhr", label: "Blurred HR" },
  { value: "jwst", label: "JWST reference" },
];

const MODES: { value: FigureMode; label: string }[] = [
  { value: "VIS", label: "VIS" },
  { value: "H_E", label: "H_E" },
  { value: "VIS_H", label: "VIS + H_E" },
  { value: "native", label: "native band" },
];

const FIGURE_TIERS = new Set<string>(TIERS.map((item) => item.value));
const FIGURE_MODES = new Set<string>(MODES.map((item) => item.value));
const SYNTHETIC_COLLECTIONS = new Set(["sky", "evaluation", "ensemble"]);
const REAL_COLLECTIONS = new Set(["real-field", "jwst-euclid", "nexus-field"]);

let nextRowId = 0;
const withIds = (rows: FigureRecipe[]): RecipeRow[] => rows.map((row) => ({
  ...row,
  id: `recipe-${++nextRowId}`,
}));

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string");
}

function stringArrayMap(value: unknown): Record<string, string[]> {
  if (!isRecord(value)) return {};
  return Object.fromEntries(
    Object.entries(value).map(([key, item]) => [key, stringArray(item)]),
  );
}

function numberMap(value: unknown): Record<string, number> {
  if (!isRecord(value)) return {};
  return Object.fromEntries(
    Object.entries(value).filter(
      (entry): entry is [string, number] => (
        typeof entry[1] === "number" && Number.isFinite(entry[1])
      ),
    ),
  );
}

function positiveInteger(value: unknown): number | undefined {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0
    ? value
    : undefined;
}

function isRecipeKey(value: unknown): value is FigureRecipeKey {
  if (typeof value !== "string" || value.split(":").length !== 2) return false;
  const [tier, mode] = value.split(":");
  return FIGURE_TIERS.has(tier) && FIGURE_MODES.has(mode);
}

function normalizeSource(value: unknown): SavedResultSource | undefined {
  if (!isRecord(value)) return undefined;
  const object = isRecord(value.object) && typeof value.object.label === "string"
    ? { label: value.object.label }
    : undefined;
  return {
    collection: typeof value.collection === "string" ? value.collection : undefined,
    index: typeof value.index === "number" && Number.isInteger(value.index)
      ? value.index
      : undefined,
    object,
  };
}

function normalizeResult(value: unknown): SavedResult | null {
  if (!isRecord(value) || typeof value.id !== "string" || !value.id.trim()) return null;
  const source = normalizeSource(value.source);
  const regime = value.regime === "real" || value.regime === "synthetic"
    ? value.regime
    : undefined;
  const recipes = Array.isArray(value.recipes)
    ? value.recipes.filter(isRecipeKey)
    : [];
  const label = typeof value.label === "string" && value.label.trim()
    ? value.label
    : source?.object?.label || value.id;
  return {
    id: value.id,
    label,
    regime,
    source,
    selection: isRecord(value.selection) ? value.selection : undefined,
    logical_tiers: stringArray(value.logical_tiers),
    bands: stringArrayMap(value.bands),
    pixscale_arcsec: numberMap(value.pixscale_arcsec),
    recipes,
    wcs_preserved: value.wcs_preserved === true,
  };
}

function normalizeIndex(value: unknown): NormalizedIndex {
  if (value == null) {
    return { index: { results: [] }, malformed: false, droppedResults: 0 };
  }
  if (!isRecord(value) || !Array.isArray(value.results)) {
    return { index: { results: [] }, malformed: true, droppedResults: 0 };
  }
  const results = value.results.map(normalizeResult);
  const validResults = results.filter((result): result is SavedResult => result !== null);
  const supported = isRecord(value.supported) ? value.supported : {};
  const limits = isRecord(value.limits) ? value.limits
    : isRecord(supported.limits) ? supported.limits : {};
  const axisDefaults = isRecord(value.axis_defaults) ? value.axis_defaults : {};
  return {
    index: {
      schema_version: typeof value.schema_version === "number"
        ? value.schema_version
        : undefined,
      limits: {
        max_results: positiveInteger(limits.max_results),
        max_rows: positiveInteger(limits.max_rows),
      },
      axis_defaults: {
        columns: typeof axisDefaults.columns === "string" ? axisDefaults.columns : undefined,
        rows: typeof axisDefaults.rows === "string" ? axisDefaults.rows : undefined,
      },
      supported: {
        logical_tiers: stringArray(supported.logical_tiers),
        modes: stringArray(supported.modes),
      },
      results: validResults,
    },
    malformed: false,
    droppedResults: results.length - validResults.length,
  };
}

function recipeKey(row: Pick<FigureRecipe, "tier" | "mode">): FigureRecipeKey {
  return `${row.tier}:${row.mode}`;
}

function rowTitle(tier: FigureTier, mode: FigureMode): string {
  const band = mode === "VIS_H" ? "VIS+H_E" : mode === "native" ? "Native" : mode;
  if (tier === "jwst") return mode === "native" ? "NEXUS F200W" : `JWST ${band}`;
  const product = tier === "dirty" ? "Dirty" : tier === "bhr" ? "BHR" : tier.toUpperCase();
  return `${band} ${product}`;
}

function resultRegime(result: SavedResult): FigureRegime | null {
  if (result.regime) return result.regime;
  const collection = result.source?.collection;
  if (collection && SYNTHETIC_COLLECTIONS.has(collection)) return "synthetic";
  if (collection && REAL_COLLECTIONS.has(collection)) return "real";
  if (result.recipes.some((key) => key.startsWith("jwst:"))) return "real";
  if (result.recipes.some((key) => key.startsWith("hr:"))) return "synthetic";
  return null;
}

function missingRecipeCount(result: SavedResult, rows: FigureRecipe[]): number {
  return rows.filter((row) => !result.recipes.includes(recipeKey(row))).length;
}

function supportsRows(result: SavedResult, rows: FigureRecipe[]): boolean {
  return missingRecipeCount(result, rows) === 0;
}

function modeClass(mode: FigureMode): string {
  return mode === "VIS_H" ? "is-vis-h" : mode === "VIS" ? "is-vis"
    : mode === "H_E" ? "is-h" : "is-native";
}

export default function FigureGridBuilder() {
  const resource = useResource<unknown>("/viewer/results", [], { ttl: 0 });
  const normalized = useMemo(() => normalizeIndex(resource.data), [resource.data]);
  const results = normalized.index.results;
  const indexLoaded = resource.data !== null;
  const indexError = resource.error || normalized.malformed;
  const maxGridResults = normalized.index.limits?.max_results ?? MAX_GRID_RESULTS;
  const maxGridRows = normalized.index.limits?.max_rows ?? MAX_GRID_ROWS;

  const [presetId, setPresetId] = useState(PRESETS[0].id);
  const [activeRegime, setActiveRegime] = useState<FigureRegime>(PRESETS[0].regime);
  const [rows, setRows] = useState<RecipeRow[]>(() => withIds(PRESETS[0].rows));
  const [selected, setSelected] = useState<string[]>([]);
  const initializedSelection = useRef(false);

  const supportedTiers = new Set(normalized.index.supported?.logical_tiers ?? []);
  const supportedModes = new Set(normalized.index.supported?.modes ?? []);
  const tierOptions = supportedTiers.size
    ? TIERS.filter((item) => supportedTiers.has(item.value))
    : TIERS;
  const modeOptions = supportedModes.size
    ? MODES.filter((item) => supportedModes.has(item.value))
    : MODES;

  const visibleResults = useMemo(
    () => results.filter((result) => resultRegime(result) === activeRegime),
    [activeRegime, results],
  );

  useEffect(() => {
    setRows((current) => current.length > maxGridRows
      ? current.slice(0, maxGridRows)
      : current);
  }, [maxGridRows]);

  useEffect(() => {
    if (initializedSelection.current || !indexLoaded || indexError) return;
    initializedSelection.current = true;
    const initialRows = rows.slice(0, maxGridRows);
    const compatible = results.filter((result) => (
      resultRegime(result) === activeRegime && supportsRows(result, initialRows)
    ));
    setSelected(compatible.slice(0, Math.min(5, maxGridResults)).map((result) => result.id));
  }, [activeRegime, indexError, indexLoaded, maxGridResults, results, rows]);

  useEffect(() => {
    if (!indexLoaded || indexError) return;
    const byId = new Map(results.map((result) => [result.id, result]));
    setSelected((current) => {
      const next = current.filter((id) => {
        const result = byId.get(id);
        return result != null && resultRegime(result) === activeRegime;
      }).slice(0, maxGridResults);
      return next.length === current.length && next.every((id, index) => id === current[index])
        ? current
        : next;
    });
  }, [activeRegime, indexError, indexLoaded, maxGridResults, results]);

  const selectedResults = useMemo(() => selected
    .map((id) => results.find((result) => result.id === id))
    .filter((result): result is SavedResult => result != null), [results, selected]);
  const staleSelection = selectedResults.length !== selected.length;
  const unsupported = selectedResults.reduce(
    (total, result) => total + missingRecipeCount(result, rows),
    0,
  );
  const columnLimitReached = selected.length >= maxGridResults;
  const rowLimitReached = rows.length >= maxGridRows;
  const overBackendLimit = selected.length > maxGridResults || rows.length > maxGridRows;

  const query = useMemo(() => {
    const params = new URLSearchParams();
    for (const result of selectedResults) params.append("result", result.id);
    for (const row of rows) params.append("row", recipeKey(row));
    return params.toString();
  }, [rows, selectedResults]);
  const canRender = !indexError && !staleSelection && !overBackendLimit
    && selectedResults.length > 0
    && rows.length > 0 && unsupported === 0;
  const previewSrc = canRender
    ? `/viewer/results/grid.png?${query}&dpi=120&inline=1`
    : null;
  const pngHref = canRender ? `/viewer/results/grid.png?${query}&dpi=300` : undefined;
  const pdfHref = canRender ? `/viewer/results/grid.pdf?${query}&dpi=300` : undefined;
  const [loadedPreviewSrc, setLoadedPreviewSrc] = useState<string | null>(null);
  const [failedPreviewSrc, setFailedPreviewSrc] = useState<string | null>(null);
  const previewError = previewSrc != null && failedPreviewSrc === previewSrc;
  const previewLoading = previewSrc != null && !previewError && loadedPreviewSrc !== previewSrc;

  const choosePreset = (id: string) => {
    const preset = PRESETS.find((candidate) => candidate.id === id);
    if (!preset) return;
    const presetRows = preset.rows.slice(0, maxGridRows);
    const compatible = results.filter((result) => (
      resultRegime(result) === preset.regime && supportsRows(result, presetRows)
    ));
    const compatibleIds = compatible.map((result) => result.id);
    setPresetId(id);
    setActiveRegime(preset.regime);
    setRows(withIds(presetRows));
    setSelected((current) => selectionForPreset(current, compatibleIds, maxGridResults));
  };

  const toggleResult = (id: string) => {
    const result = results.find((candidate) => candidate.id === id);
    if (!result || resultRegime(result) !== activeRegime) return;
    setSelected((current) => toggleGridResult(current, id, maxGridResults));
  };

  const moveResult = (id: string, offset: -1 | 1) => setSelected((current) => {
    const index = current.indexOf(id);
    const target = index + offset;
    if (index < 0 || target < 0 || target >= current.length) return current;
    const copy = current.slice();
    [copy[index], copy[target]] = [copy[target], copy[index]];
    return copy;
  });

  const patchRow = (id: string, patch: Partial<Pick<RecipeRow, "tier" | "mode">>) => {
    setRows((current) => current.map((row) => {
      if (row.id !== id) return row;
      const tier = patch.tier ?? row.tier;
      const mode = patch.mode ?? row.mode;
      return { ...row, tier, mode, title: rowTitle(tier, mode) };
    }));
    setPresetId("custom");
  };

  const moveRow = (id: string, offset: -1 | 1) => {
    setRows((current) => {
      const index = current.findIndex((row) => row.id === id);
      const target = index + offset;
      if (index < 0 || target < 0 || target >= current.length) return current;
      const copy = current.slice();
      [copy[index], copy[target]] = [copy[target], copy[index]];
      return copy;
    });
    setPresetId("custom");
  };

  const duplicateRow = (row: RecipeRow) => {
    setRows((current) => {
      if (!canAddGridRow(current.length, maxGridRows)) return current;
      const index = current.findIndex((candidate) => candidate.id === row.id);
      if (index < 0) return current;
      const next = current.slice();
      next.splice(index + 1, 0, { ...row, id: `recipe-${++nextRowId}` });
      return next;
    });
    setPresetId("custom");
  };

  const removeRow = (id: string) => {
    setRows((current) => current.length > 1
      ? current.filter((row) => row.id !== id)
      : current);
    setPresetId("custom");
  };

  const addRow = () => {
    setRows((current) => canAddGridRow(current.length, maxGridRows) ? [...current, {
      id: `recipe-${++nextRowId}`,
      tier: "sr",
      mode: "VIS_H",
      title: "VIS+H_E SR",
    }] : current);
    setPresetId("custom");
  };

  const statusText = (() => {
    if (resource.loading && !indexLoaded) return "loading saved crops";
    if (indexError) return "saved crops unavailable";
    if (previewError) return "preview render failed";
    if (selected.length > maxGridResults) {
      return `reduce the grid to ${maxGridResults} columns`;
    }
    if (rows.length > maxGridRows) return `reduce the grid to ${maxGridRows} rows`;
    if (canRender) return previewLoading ? "rendering preview" : "render ready";
    if (staleSelection) return "saved crop list changed";
    if (unsupported) {
      return `${unsupported} unavailable cell${unsupported === 1 ? "" : "s"}`;
    }
    return "choose at least one saved crop";
  })();
  const statusBad = indexError || previewError || unsupported > 0 || staleSelection
    || overBackendLimit;

  return (
    <section className="figure-grid" aria-labelledby="figure-grid-title">
      <header className="figure-grid__head">
        <div>
          <div className="eyebrow">saved crops · contact sheet</div>
          <h2 id="figure-grid-title">Result grid assembler</h2>
          <p>Recipes run down the page; frozen galaxy crops run across it. Every export is rebuilt from the saved raw cubes.</p>
        </div>
        <div className="figure-grid__summary">
          <Badge tone={statusBad ? "bad" : canRender && !previewLoading ? "good" : "warn"}>
            {rows.length} × {selectedResults.length}
          </Badge>
          <Button size="sm" variant="ghost" onClick={resource.reload}>↻ results</Button>
        </div>
      </header>

      <div className="figure-grid__instrument">
        <aside className="figure-grid__rail" aria-label="Grid controls">
          <label className="figure-grid__preset">
            <span>Template</span>
            <select value={presetId} onChange={(event) => choosePreset(event.target.value)}>
              {presetId === "custom" && <option value="custom">Custom</option>}
              {PRESETS.map((preset) => (
                <option key={preset.id} value={preset.id}>{preset.label}</option>
              ))}
            </select>
          </label>

          <section className="figure-grid__control-group" aria-labelledby="figure-grid-columns-title"
            aria-busy={resource.loading}>
            <div className="figure-grid__control-head">
              <span id="figure-grid-columns-title">Columns · {activeRegime} saved crops</span>
              <b aria-label={`${selectedResults.length} of ${maxGridResults} columns`}>
                {selectedResults.length}/{maxGridResults}
              </b>
            </div>
            {resource.loading && !indexLoaded ? <Empty><Spinner /> loading results…</Empty>
              : indexError ? (
                <div className="figure-grid__empty figure-grid__empty--error" role="alert">
                  <strong>Saved crops could not be loaded.</strong>
                  <button type="button" onClick={resource.reload}>Try again</button>
                </div>
              ) : !results.length ? (
                <div className="figure-grid__empty">
                  Freeze a region in any viewer, then choose <strong>Save crop to results</strong>.
                </div>
              ) : !visibleResults.length ? (
                <div className="figure-grid__empty">
                  No {activeRegime} saved crops are available for this template.
                </div>
              ) : (
                <>
                  {normalized.droppedResults > 0 && (
                    <div className="figure-grid__notice" role="status">
                      {normalized.droppedResults} malformed saved result{normalized.droppedResults === 1 ? " was" : "s were"} omitted.
                    </div>
                  )}
                  {columnLimitReached && (
                    <div className="figure-grid__notice" id="figure-grid-column-limit" role="status">
                      {maxGridResults}-column limit reached. Remove a crop before adding another.
                    </div>
                  )}
                  <div className="figure-grid__targets">
                    {visibleResults.map((result) => {
                      const active = selected.includes(result.id);
                      const missing = missingRecipeCount(result, rows);
                      const limitBlocked = columnLimitReached && !active;
                      return (
                        <div className="figure-grid__target" data-on={active}
                          data-compatible={missing === 0} data-limit-blocked={limitBlocked}
                          key={result.id}>
                          <button type="button" className="figure-grid__target-toggle"
                            aria-pressed={active}
                            aria-label={`${active ? "Remove" : "Add"} ${result.label} ${activeRegime} crop`}
                            aria-describedby={limitBlocked ? "figure-grid-column-limit" : undefined}
                            disabled={limitBlocked}
                            onClick={() => toggleResult(result.id)}>
                            <span className="figure-grid__target-index">
                              {active ? selected.indexOf(result.id) + 1 : "–"}
                            </span>
                            <span><strong title={result.label}>{result.label}</strong><small>
                              {result.source?.collection ?? "saved result"}
                              {result.source?.index != null ? ` · ${result.source.index}` : ""}
                              {missing ? ` · missing ${missing} recipe${missing === 1 ? "" : "s"}` : ""}
                            </small></span>
                          </button>
                          {active && <span className="figure-grid__move">
                            <button type="button" onClick={() => moveResult(result.id, -1)}
                              disabled={selected[0] === result.id}
                              aria-label={`Move ${result.label} crop left`}>←</button>
                            <button type="button" onClick={() => moveResult(result.id, 1)}
                              disabled={selected[selected.length - 1] === result.id}
                              aria-label={`Move ${result.label} crop right`}>→</button>
                          </span>}
                        </div>
                      );
                    })}
                  </div>
                </>
              )}
          </section>

          <section className="figure-grid__control-group" aria-labelledby="figure-grid-rows-title">
            <div className="figure-grid__control-head">
              <span id="figure-grid-rows-title">Rows · display recipes</span>
              <span className="figure-grid__control-actions">
                <b aria-label={`${rows.length} of ${maxGridRows} rows`}>
                  {rows.length}/{maxGridRows}
                </b>
                <button type="button" onClick={addRow} disabled={rowLimitReached}
                  aria-describedby={rowLimitReached ? "figure-grid-row-limit" : undefined}
                  title={rowLimitReached ? `Maximum ${maxGridRows} recipe rows` : undefined}>
                  + add
                </button>
              </span>
            </div>
            {rowLimitReached && (
              <div className="figure-grid__notice" id="figure-grid-row-limit" role="status">
                {maxGridRows}-row limit reached. Remove a recipe before adding or duplicating one.
              </div>
            )}
            <div className="figure-grid__recipes">
              {rows.map((row, index) => (
                <article className={`figure-grid__recipe ${modeClass(row.mode)}`} key={row.id}
                  aria-label={`Row ${index + 1}: ${row.title}`}>
                  <span className="figure-grid__spectrum" aria-hidden />
                  <div className="figure-grid__recipe-title">
                    <b>{index + 1}</b><strong>{row.title}</strong>
                  </div>
                  <div className="figure-grid__recipe-fields">
                    <select value={row.tier}
                      onChange={(event) => patchRow(row.id, { tier: event.target.value as FigureTier })}
                      aria-label={`${row.title} product`}>
                      {tierOptions.map((tier) => <option key={tier.value} value={tier.value}>{tier.label}</option>)}
                    </select>
                    <select value={row.mode}
                      onChange={(event) => patchRow(row.id, { mode: event.target.value as FigureMode })}
                      aria-label={`${row.title} band`}>
                      {modeOptions.map((mode) => <option key={mode.value} value={mode.value}>{mode.label}</option>)}
                    </select>
                  </div>
                  <div className="figure-grid__recipe-actions">
                    <button type="button" onClick={() => moveRow(row.id, -1)} disabled={index === 0}
                      aria-label={`Move ${row.title} up`}>↑</button>
                    <button type="button" onClick={() => moveRow(row.id, 1)} disabled={index === rows.length - 1}
                      aria-label={`Move ${row.title} down`}>↓</button>
                    <button type="button" onClick={() => duplicateRow(row)} disabled={rowLimitReached}
                      aria-describedby={rowLimitReached ? "figure-grid-row-limit" : undefined}
                      aria-label={`Duplicate ${row.title}`}>⧉</button>
                    <button type="button" onClick={() => removeRow(row.id)} disabled={rows.length === 1}
                      aria-label={`Remove ${row.title}`}>×</button>
                  </div>
                </article>
              ))}
            </div>
          </section>
        </aside>

        <div className="figure-grid__workbench">
          <header className="figure-grid__workbench-head">
            <div role="status" aria-live="polite" aria-atomic="true">
              <span className="figure-grid__status-light" data-ready={canRender && !previewError}
                data-error={statusBad} aria-hidden />
              <span>{statusText}</span>
            </div>
            <div className="figure-grid__downloads">
              <a aria-disabled={!canRender} href={pngHref} download
                tabIndex={canRender ? undefined : -1} aria-label="Download result grid as PNG">PNG</a>
              <a aria-disabled={!canRender} href={pdfHref} download
                tabIndex={canRender ? undefined : -1} aria-label="Download result grid as PDF">PDF</a>
            </div>
          </header>
          <div className="figure-grid__paper" aria-busy={previewLoading}>
            {previewError ? (
              <div className="figure-grid__paper-empty figure-grid__paper-empty--error" role="alert">
                <strong>Preview could not be rendered</strong>
                <p>The saved matrix may have changed or the server rejected this recipe.</p>
                <button type="button" onClick={() => {
                  setFailedPreviewSrc(null);
                  setLoadedPreviewSrc(null);
                }}>Retry preview</button>
              </div>
            ) : previewSrc ? (
              <>
                <img key={previewSrc} src={previewSrc}
                  className={previewLoading ? "is-loading" : ""}
                  onLoad={() => {
                    setLoadedPreviewSrc(previewSrc);
                    setFailedPreviewSrc(null);
                  }}
                  onError={() => setFailedPreviewSrc(previewSrc)}
                  alt={`Result grid preview with ${rows.length} recipe rows and ${selectedResults.length} saved-result columns`} />
                {previewLoading && (
                  <div className="figure-grid__preview-state" role="status">
                    <Spinner /> rendering preview…
                  </div>
                )}
              </>
            ) : (
              <div className="figure-grid__paper-empty">
                <span className="figure-grid__reticle" aria-hidden />
                <strong>Build a compatible matrix</strong>
                <p>Select saved crops and recipes that exist in every column.</p>
              </div>
            )}
          </div>
          <footer className="figure-grid__legend">
            <span><i className="is-mono" aria-hidden />VIS and H_E alone → grayscale</span>
            <span><i className="is-vis-h" aria-hidden />VIS+H_E → VIS cyan-blue · H_E amber-red</span>
            <span>Raw cubes remain unchanged; color is applied only while rendering.</span>
          </footer>
        </div>
      </div>
    </section>
  );
}
