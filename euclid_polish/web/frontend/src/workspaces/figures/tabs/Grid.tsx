/* Figures › Grid (spec §8.5): the publication contact sheet — saved crops
 * run across (columns), display recipes run down (rows); the A4 PNG/PDF is
 * rebuilt server-side from the saved raw cubes with the locked absolute
 * asinh transfer. Templates (presets + saved layouts), live preview,
 * thumbnails of every saved crop. The preview is as large as the window
 * allows (above the editors in a narrow pane); a click on it (or "Full
 * size") opens the sheet in the Lightbox at FULL_DPI with Fit / Actual size,
 * and a crop's thumbnail opens that crop full size. Columns, rows, regime,
 * template and the export dpi live in the URL. */
import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { Link } from "react-router-dom";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Badge, Button, Callout, DataTable, EmptyState, IconButton, Input, Page, Popover, Segmented, Select,
  Tooltip, confirm, toast, type DataColumn, type SelectOption,
} from "../../../ui";
import { URLS, deleteLayout, gridUrl, saveLayout, type FigureMode, type FigureRegime, type FigureTier, type GridLayout, type RecipeKey, type SavedResult } from "../api";
import { ServerImage, ThumbButton, WcsBadge } from "../common";
import { Lightbox, ResultLightbox } from "../Lightbox";
import { canAddGridRow, selectionForPreset } from "../grid/limits";
import {
  DEFAULT_PRESET, MODES, PREVIEW, PRESETS, TIERS, commonRecipes, previewRows, gridSizeText, gridStatus, isRecipeKey, layoutValue,
  missingRecipes, modeTone, moveItem, normalizeIndex, recipeLabel, resultRegime, sanitizeColumns, splitRecipe,
} from "../model";
import "../figures.css";
import "../register";

const PREVIEW_DPI = 120;
/** The full-size view's render: sharp at 100 % on a 2× screen, ~0.4 s. */
const FULL_DPI = 200;
/** The preview's cap in the CSS comes from model.ts PREVIEW (the full-size
 *  view is tested to be taller than it). */
const PREVIEW_GEOMETRY = {
  "--fig-sticky-top": `${PREVIEW.stickyTop}px`, "--fig-preview-head": `${PREVIEW.head}px`,
  "--fig-preview-stacked": `${PREVIEW.stackedChrome}px`, "--fig-preview-min": `${PREVIEW.stackedMin}px`, "--fig-paper-pad": `${PREVIEW.pad}px`,
} as CSSProperties;
const DPI_OPTIONS: SelectOption[] = [150, 300, 600].map((d) => ({ value: String(d), label: `${d} dpi` }));
const REGIMES: { value: FigureRegime; label: string }[] = [{ value: "real", label: "Real" }, { value: "synthetic", label: "Synthetic" }];

/** A value that settles `ms` after its last change (the live preview). */
function useSettled<T>(value: T, ms: number): T {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setSettled(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return settled;
}

export default function Grid() {
  const [regimeRaw, setRegime] = useUrlState<string>("regime", DEFAULT_PRESET.regime);
  const [cols, setCols] = useUrlState<string[]>("cols", []);
  const [rowsRaw, setRows] = useUrlState<string[]>("rows", DEFAULT_PRESET.rows);
  const [tpl, setTpl] = useUrlState<string>("tpl", DEFAULT_PRESET.id);
  const [dpi, setDpi] = useUrlState<string>("dpi", "300");
  const [showPreview, setShowPreview] = useUrlState<boolean>("preview", true);
  const [fullGrid, setFullGrid] = useState(false);
  const [fullCrop, setFullCrop] = useState<SavedResult | null>(null);
  const regime: FigureRegime = regimeRaw === "synthetic" ? "synthetic" : "real";
  const rows = useMemo(() => rowsRaw.filter(isRecipeKey), [rowsRaw]);

  const index = useResource<unknown>(URLS.results, [], { ttl: 15_000 });
  const layoutsRes = useResource<{ layouts: GridLayout[] }>(URLS.layouts, [], { ttl: 30_000 });
  const norm = useMemo(() => normalizeIndex(index.data), [index.data]);
  const layouts = useMemo(() => layoutsRes.data?.layouts ?? [], [layoutsRes.data]);
  const byId = useMemo(() => new Map(norm.results.map((r) => [r.id, r])), [norm.results]);
  const pool = useMemo(() => norm.results.filter((r) => resultRegime(r) === regime), [norm.results, regime]);
  const colResults = cols.map((id) => byId.get(id)).filter((r): r is SavedResult => !!r);
  const loaded = index.data != null;
  const staleCols = loaded ? cols.filter((id) => !byId.has(id)) : [];
  const status = gridStatus({
    loading: index.loading, error: !!index.error || norm.malformed, results: norm.results,
    columns: loaded ? cols : [], rows, maxResults: norm.maxResults, maxRows: norm.maxRows,
  });
  const activeLayout = tpl.startsWith("layout:") ? layouts.find((l) => layoutValue(l) === tpl) : undefined;

  // With unavailable cells the preview still draws the rows every column has
  // (the renderer needs every cell); the status says what is left out.
  const drawable = useMemo(() => previewRows(rows, colResults), [rows, colResults]);
  const partial = status.unsupported > 0 && drawable.length > 0;
  const shown = partial ? gridStatus({
    loading: index.loading, error: !!index.error || norm.malformed, results: norm.results,
    columns: loaded ? cols : [], rows: drawable, maxResults: norm.maxResults, maxRows: norm.maxRows,
  }) : status;
  const previewQuery = shown.canRender ? gridUrl(cols, partial ? drawable : rows, "png", PREVIEW_DPI, true) : null;
  const settledSrc = useSettled(previewQuery, 450);
  /* the first render goes out at once; later edits settle for 450 ms while the
     last render stays up (dimmed) instead of blanking the preview */
  const previewSrc = previewQuery == null ? null : settledSrc ?? previewQuery;
  const previewPending = previewSrc != null && previewSrc !== previewQuery;
  const exportDpi = Number(dpi) || 300;

  /* ─── edits ─── */
  const custom = () => { if (tpl !== "custom" && !activeLayout) setTpl("custom"); };
  const choosePreset = (id: string) => {
    const preset = PRESETS.find((p) => p.id === id);
    if (!preset) return;
    const presetRows = preset.rows.slice(0, norm.maxRows);
    const compatible = norm.results.filter((r) => resultRegime(r) === preset.regime && !missingRecipes(r, presetRows).length).map((r) => r.id);
    setTpl(preset.id);
    setRegime(preset.regime);
    setRows(presetRows);
    setCols(selectionForPreset(sanitizeColumns(cols, norm.results, preset.regime, norm.maxResults), compatible, norm.maxResults));
  };
  const chooseLayout = (layout: GridLayout) => {
    const known = layout.results.filter((id) => byId.has(id));
    const lr = layout.regime ?? (known.length ? resultRegime(byId.get(known[0])!) : null) ?? regime;
    setTpl(layoutValue(layout));
    setRegime(lr);
    setRows(layout.rows.filter(isRecipeKey).slice(0, norm.maxRows));
    setCols(sanitizeColumns(known, norm.results, lr, norm.maxResults));
    if (known.length < layout.results.length) toast.warning(`${layout.results.length - known.length} column(s) of “${layout.name}” are no longer saved`);
  };
  const chooseTemplate = (value: string) => {
    const layout = layouts.find((l) => layoutValue(l) === value);
    if (layout) chooseLayout(layout);
    else choosePreset(value);
  };
  const switchRegime = (next: FigureRegime) => {
    if (next === regime) return;
    const preset = PRESETS.find((p) => p.id === tpl);
    setRegime(next);
    setCols(sanitizeColumns(cols, norm.results, next, norm.maxResults));
    if (preset && preset.regime !== next) {
      const first = PRESETS.find((p) => p.regime === next)!;
      setTpl(first.id);
      setRows(first.rows.slice(0, norm.maxRows));
    } else if (activeLayout) setTpl("custom");
  };
  const setColumnKeys = (keys: string[]) => {
    const kept = cols.filter((id) => keys.includes(id));
    const added = keys.filter((id) => !kept.includes(id));
    setCols([...kept, ...added].slice(0, norm.maxResults));
    custom();
  };
  /** Fill up to five columns (or one more) with the newest crops of the
   *  regime, preferring those that support every row. */
  const addNewest = () => {
    const current = cols.filter((id) => byId.has(id));
    const target = Math.min(norm.maxResults, Math.max(current.length + 1, 5));
    const fresh = pool.filter((r) => !current.includes(r.id));
    const compatible = fresh.filter((r) => !missingRecipes(r, rows).length);
    const extra = (compatible.length ? compatible : fresh).map((r) => r.id);
    if (!extra.length) { toast.info("Every saved crop of this regime is already a column"); return; }
    setCols([...current, ...extra].slice(0, target));
    custom();
  };
  const patchRow = (i: number, tier: FigureTier, mode: FigureMode) => {
    setRows(rows.map((r, j) => (j === i ? (`${tier}:${mode}` as RecipeKey) : r)));
    custom();
  };
  const editRows = (next: string[]) => { setRows(next); custom(); };
  const useCommon = () => {
    const common = commonRecipes(colResults).slice(0, norm.maxRows);
    if (common.length) editRows(common);
  };

  /* ─── layouts ─── */
  const [saveOpen, setSaveOpen] = useState(false);
  const [layoutName, setLayoutName] = useState("");
  const [savingLayout, setSavingLayout] = useState(false);
  const openSave = (open: boolean) => { setSaveOpen(open); if (open) setLayoutName(activeLayout?.name ?? ""); };
  const doSave = async () => {
    if (!layoutName.trim()) return;
    setSavingLayout(true);
    try {
      const { layout, created } = await saveLayout({ name: layoutName.trim(), results: colResults.map((r) => r.id), rows, regime });
      setTpl(layoutValue(layout));
      toast.success(created ? `Layout “${layout.name}” saved` : `Layout “${layout.name}” updated`);
      setSaveOpen(false);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setSavingLayout(false);
    }
  };
  const removeLayout = async () => {
    if (!activeLayout) return;
    if (!(await confirm({ title: `Delete layout “${activeLayout.name}”?`, message: "The saved results themselves are kept.", tone: "danger", confirmLabel: "Delete" }))) return;
    try {
      await deleteLayout(activeLayout.id);
      setTpl("custom");
      toast.success("Layout deleted");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    }
  };

  const download = (format: "png" | "pdf") => {
    if (status.canRender) window.location.assign(gridUrl(cols, rows, format, exportDpi));
  };
  usePageActions([
    { id: "fig-grid-save", label: "Save the grid layout…", group: "Figure grid", disabled: !rows.length, run: () => openSave(true) },
    { id: "fig-grid-png", label: `Download the grid as PNG (${exportDpi} dpi)`, group: "Figure grid", disabled: !status.canRender, run: () => download("png") },
    { id: "fig-grid-pdf", label: "Download the grid as PDF (A4)", group: "Figure grid", disabled: !status.canRender, run: () => download("pdf") },
    { id: "fig-grid-fill", label: "Add the newest compatible saved crops", group: "Figure grid", disabled: !pool.length, run: addNewest },
    { id: "fig-grid-common", label: "Use the rows every column supports", group: "Figure grid", disabled: !colResults.length, run: useCommon },
    { id: "fig-grid-clear", label: "Clear the grid columns", group: "Figure grid", disabled: !cols.length, run: () => { setCols([]); custom(); } },
    { id: "fig-grid-preview", label: showPreview ? "Collapse the live preview" : "Show the live preview", group: "Figure grid", run: () => setShowPreview(!showPreview) },
    { id: "fig-grid-full", label: "View the grid full size", group: "Figure grid", disabled: !status.canRender, run: () => setFullGrid(true) },
    { id: "fig-grid-refresh", label: "Refresh saved crops", group: "Figure grid", run: () => { void index.reload(); void layoutsRes.reload(); } },
  ]);

  const templateOptions: SelectOption[] = [
    ...(tpl === "custom" || (tpl.startsWith("layout:") && !activeLayout) ? [{ value: tpl, label: tpl === "custom" ? "Custom" : "Layout (deleted)" }] : []),
    ...PRESETS.map((p) => ({ value: p.id, label: p.label })),
    ...layouts.map((l) => ({ value: layoutValue(l), label: `★ ${l.name}`, hint: `${l.rows.length} × ${l.results.length}` })),
  ];

  const pickColumns = useMemo<DataColumn<SavedResult>[]>(() => [
    { id: "thumb", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 52,
      cell: (r) => <ThumbButton result={r} size={40} onOpen={() => setFullCrop(r)} /> },
    { id: "label", header: "Saved crop", width: 190, filterText: (r) => `${r.label} ${r.source?.collection ?? ""} ${r.id}`,
      cell: (r) => {
        const missing = missingRecipes(r, rows);
        return (
          <span className="fig-cell-stack">
            <span className="fig-ellipsis" title={r.label}>{r.label}</span>
            <span className="fig-ellipsis muted">
              {r.source?.collection ?? "viewer"}
              {missing.length ? <span className="fig-warn"> · missing {missing.length}</span> : null}
            </span>
          </span>
        );
      } },
    { id: "wcs", header: "", sortable: false, filterable: false, csv: false, width: 70, cell: (r) => <WcsBadge result={r} /> },
  ], [rows]);

  return (
    <Page className="fig-page">
      <div className="fig-bar" role="toolbar" aria-label="Figure grid">
        <label className="fig-bar__field">
          <span className="fig-bar__label">Template</span>
          <Select size="sm" value={tpl} onChange={chooseTemplate} options={templateOptions} aria-label="Template" />
        </label>
        <Popover open={saveOpen} onOpenChange={openSave} label="Save layout" align="start"
          trigger={<Button size="sm" icon="pin" disabled={!rows.length}>Save</Button>}>
          <div className="fig-pop">
            <Input value={layoutName} onChange={setLayoutName} placeholder="Layout name" aria-label="Layout name" onEnter={() => void doSave()} autoFocus />
            <div className="fig-pop__row">
              <span className="muted fig-pop__hint">{layouts.some((l) => l.name.toLowerCase() === layoutName.trim().toLowerCase()) ? "Updates the layout with this name" : gridSizeText(rows.length, colResults.length)}</span>
              <Button size="sm" variant="primary" loading={savingLayout} disabled={!layoutName.trim()} onClick={() => void doSave()}>Save layout</Button>
            </div>
          </div>
        </Popover>
        {activeLayout && <IconButton icon="close" size="sm" label={`Delete layout ${activeLayout.name}`} onClick={() => void removeLayout()} />}
        <Segmented size="sm" value={regime} onChange={switchRegime} aria-label="Regime" options={REGIMES} />
        <span className="fig-bar__spacer" />
        <Tooltip content={status.text}>
          <span tabIndex={0} className="fig-badge-wrap" role="status" aria-live="polite">
            <Badge tone={status.tone} dot>{rows.length} × {colResults.length}</Badge>
          </span>
        </Tooltip>
        <Select size="sm" value={dpi} onChange={setDpi} options={DPI_OPTIONS} aria-label="Export resolution" />
        <Button size="sm" icon="download" disabled={!status.canRender} href={status.canRender ? gridUrl(cols, rows, "png", exportDpi) : undefined} download>PNG</Button>
        <Button size="sm" icon="download" disabled={!status.canRender} href={status.canRender ? gridUrl(cols, rows, "pdf", exportDpi) : undefined} download>PDF</Button>
        <IconButton icon="reset" size="sm" label="Refresh saved crops" onClick={() => { void index.reload(); void layoutsRes.reload(); }} />
      </div>

      {index.error && !index.data && (
        <Callout tone="bad" title="Saved crops did not load" action={<Button size="sm" onClick={() => void index.reload()}>Retry</Button>}>{index.error.message}</Callout>
      )}
      {staleCols.length > 0 && (
        <Callout tone="warn" title={`${staleCols.length} column${staleCols.length === 1 ? " is" : "s are"} no longer saved`}
          action={<Button size="sm" onClick={() => setCols(cols.filter((id) => byId.has(id)))}>Drop</Button>} />
      )}

      <div className="fig-grid">
        <div className="fig-grid__controls">
          <section className="fig-panel" aria-labelledby="fig-grid-cols">
            <header className="fig-panel__head">
              <h3 id="fig-grid-cols">Columns</h3>
              <span className="muted">{colResults.length}/{norm.maxResults}</span>
              <span className="fig-bar__spacer" />
              <Button size="sm" variant="ghost" disabled={!pool.length} onClick={addNewest}>Add newest</Button>
            </header>
            {colResults.length > 0 && (
              <ol className="fig-order" aria-label="Column order">
                {colResults.map((r, i) => (
                  <li key={r.id} className="fig-order__item">
                    <span className="fig-order__n">{i + 1}</span>
                    <button type="button" className="fig-order__label fig-ellipsis" title={`${r.label} — open its card`}
                      onClick={() => openInspector({ kind: "figure", id: r.id })}>{r.label}</button>
                    <IconButton icon="chevronLeft" size="sm" label={`Move ${r.label} left`} disabled={i === 0}
                      onClick={() => { setCols(moveItem(cols, cols.indexOf(r.id), -1)); custom(); }} />
                    <IconButton icon="chevronRight" size="sm" label={`Move ${r.label} right`} disabled={i === colResults.length - 1}
                      onClick={() => { setCols(moveItem(cols, cols.indexOf(r.id), 1)); custom(); }} />
                    <IconButton icon="close" size="sm" label={`Remove ${r.label}`}
                      onClick={() => { setCols(cols.filter((id) => id !== r.id)); custom(); }} />
                  </li>
                ))}
              </ol>
            )}
            {!index.loading && !pool.length ? (
              <EmptyState compact icon="image" title={norm.results.length ? `No ${regime} saved crops` : "No saved crops yet"}
                action={<Button asChild size="sm" variant="ghost"><Link to="/figures/results">Saved results</Link></Button>}>
                Freeze a region in any viewer and press S.
              </EmptyState>
            ) : (
              <DataTable rows={pool} columns={pickColumns} rowKey={(r) => r.id} aria-label={`${regime} saved crops`}
                selectable selected={colResults.map((r) => r.id)} onSelectedChange={(keys) => setColumnKeys(keys)}
                rowHeight={50} height={pool.length > 5 ? 290 : "auto"} dense loading={index.loading && !index.data}
                filterPlaceholder="Filter crops" empty="No saved crops match." />
            )}
          </section>

          <section className="fig-panel" aria-labelledby="fig-grid-rows">
            <header className="fig-panel__head">
              <h3 id="fig-grid-rows">Rows</h3>
              <span className="muted">{rows.length}/{norm.maxRows}</span>
              <span className="fig-bar__spacer" />
              <Tooltip content="Replace the rows with every recipe all columns support">
                <Button size="sm" variant="ghost" disabled={!colResults.length} onClick={useCommon}>Common</Button>
              </Tooltip>
              <Button size="sm" icon="plus" disabled={!canAddGridRow(rows.length, norm.maxRows)}
                onClick={() => editRows([...rows, "sr:VIS_H"])}>Row</Button>
            </header>
            <ol className="fig-rows">
              {rows.map((key, i) => {
                const [tier, mode] = splitRecipe(key);
                const missing = colResults.filter((r) => !r.recipes.includes(key)).length;
                return (
                  <li key={`${key}-${i}`} className="fig-row" data-tone={modeTone(mode)} data-missing={missing > 0 || undefined}>
                    <span className="fig-row__n" aria-hidden>{i + 1}</span>
                    <span className="fig-row__title fig-ellipsis" title={recipeLabel(key)}>{recipeLabel(key)}
                      {missing > 0 && <span className="fig-warn"> · {missing} missing</span>}</span>
                    <span className="fig-row__fields">
                      <Select size="sm" value={tier} onChange={(t) => patchRow(i, t, mode)} aria-label={`Row ${i + 1} product`}
                        options={TIERS.filter((t) => norm.tiers.includes(t.value)).map((t) => ({ value: t.value, label: t.label }))} />
                      <Select size="sm" value={mode} onChange={(m) => patchRow(i, tier, m)} aria-label={`Row ${i + 1} band`}
                        options={MODES.filter((m) => norm.modes.includes(m.value)).map((m) => ({ value: m.value, label: m.label }))} />
                    </span>
                    <span className="fig-row__actions">
                      <IconButton icon="chevronUp" size="sm" label={`Move row ${i + 1} up`} disabled={i === 0} onClick={() => editRows(moveItem(rows, i, -1))} />
                      <IconButton icon="chevronDown" size="sm" label={`Move row ${i + 1} down`} disabled={i === rows.length - 1} onClick={() => editRows(moveItem(rows, i, 1))} />
                      <IconButton icon="copy" size="sm" label={`Duplicate row ${i + 1}`} disabled={!canAddGridRow(rows.length, norm.maxRows)}
                        onClick={() => editRows([...rows.slice(0, i + 1), key, ...rows.slice(i + 1)])} />
                      <IconButton icon="close" size="sm" label={`Remove row ${i + 1}`} disabled={rows.length === 1}
                        onClick={() => editRows(rows.filter((_, j) => j !== i))} />
                    </span>
                  </li>
                );
              })}
            </ol>
            <p className="fig-legend"><i data-tone="vis" /> VIS <i data-tone="h" /> H_E alone: grey · <i data-tone="vis-h" /> VIS + H_E: VIS cyan, H_E amber</p>
          </section>
        </div>

        <section className="fig-grid__preview" aria-label="Live preview" data-collapsed={!showPreview || undefined} style={PREVIEW_GEOMETRY}>
          <div className="fig-grid__previewhead">
            <span className="fig-grid__status fig-ellipsis" data-tone={status.canRender ? status.tone : partial ? "warn" : undefined} role="status">
              {status.canRender ? status.text
                : partial ? `${status.text}: the preview leaves out ${rows.length - drawable.length} row${rows.length - drawable.length === 1 ? "" : "s"}`
                  : "Preview"}
            </span>
            {partial && <Button size="sm" variant="ghost" onClick={useCommon}>Use the rows every column has</Button>}
            {showPreview && status.canRender && (
              <Button size="sm" variant="ghost" icon="zoomIn" onClick={() => setFullGrid(true)}>Full size</Button>
            )}
            <IconButton icon={showPreview ? "chevronUp" : "chevronDown"} size="sm"
              label={showPreview ? "Collapse the preview" : "Show the preview"} pressed={!showPreview}
              onClick={() => setShowPreview(!showPreview)} />
          </div>
          {showPreview && (
            <ServerImage src={previewSrc} keepPrevious pending={previewPending}
              alt={`Figure grid preview, ${gridSizeText(partial ? drawable.length : rows.length, colResults.length)}`}
              className="fig-grid__paper" minHeight={280}
              overlay={status.canRender ? (
                <button type="button" className="fig-grid__zoom" aria-label="View the grid full size" title="View full size"
                  onClick={() => setFullGrid(true)} />
              ) : undefined}>
              {!shown.canRender && (
                <EmptyState compact icon="columns" title={status.text}
                  action={status.unsupported > 0 && commonRecipes(colResults).length
                    ? <Button size="sm" onClick={useCommon}>Use the rows every column has</Button>
                    : pool.length && !cols.length ? <Button size="sm" onClick={addNewest}>Add the newest crops</Button> : undefined}>
                  Columns are saved crops; rows are recipes every column must support.
                </EmptyState>
              )}
            </ServerImage>
          )}
        </section>
      </div>
      <Lightbox open={fullGrid && status.canRender} onOpenChange={setFullGrid}
        title="Figure grid" description={`${gridSizeText(rows.length, colResults.length)} · rendered at ${FULL_DPI} dpi`}
        src={fullGrid && status.canRender ? gridUrl(cols, rows, "png", FULL_DPI, true) : null}
        alt={`Figure grid, ${gridSizeText(rows.length, colResults.length)}`}
        footer={<>
          <Button size="sm" icon="download" href={status.canRender ? gridUrl(cols, rows, "png", exportDpi) : undefined} download>PNG · {exportDpi} dpi</Button>
          <Button size="sm" icon="download" href={status.canRender ? gridUrl(cols, rows, "pdf", exportDpi) : undefined} download>PDF</Button>
        </>} />
      <ResultLightbox result={fullCrop} onClose={() => setFullCrop(null)} />
    </Page>
  );
}
