/* The viewer's control bar on the light table: ONE slim row when it fits,
 * else two — what is shown above how it is shown (barModel's barLayout
 * decides from the measured group widths). A narrow viewer (the 380 px
 * inspector panel, the bottom sheet: 300–480 px) keeps two rows: its band
 * chips collapse into one select and the rarely used groups (export,
 * layout, tools, compare, zoom — in that order) move into a More menu (⋯);
 * only below ~300 px does a row still wrap. No control is ever cut off or
 * scrolled out of sight.
 *
 *   what  tier chips ▾ · band / colour
 *   how   Display · compare · tools ▾ ·······  − + fit · ◀ i / n ▶ ▶▶ · layout ▾ · export ▾ · ⋯ · Open large
 *
 * Everything touched every minute is one click; the rest lives in the
 * popovers of BarMenus.tsx. "Display" opens the Display ROW under the bar
 * (knee, brightness, stretch; "More display settings" in a popover). "Open
 * large" (focus mode, F) is always a visible button. Modes: "full";
 * "compact" (the Display row with knee + brightness, no tools or export
 * menus — a lens toggle stands in for the tools); "nav" (toolbar "none"
 * with navigation: only the navigation, export and Open large). `nav`
 * false drops the navigation only. */
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Button, IconButton, Kbd, Menu, Popover, Segmented, Select, Tooltip, type MenuItem } from "../ui";
import { barLayout, chipTiers, colourOptions, isSinglePlane, navPosition, parsePosition, sentenceLabel, shortTierLabel, type BarItem, type BarLayout } from "./barModel";
import { TierMenu, ToolsMenu } from "./BarMenus";
import { LAYOUT_HINT, LAYOUT_LABEL, LAYOUT_MODES } from "./fit";
import { ZOOM_STEP } from "./controller";
import { useController, useSettings, useViewer } from "./hooks";
import { VIcon } from "./icons";
import type { Compare } from "./types";

export type BarMode = "full" | "compact" | "nav";

const tip = (label: string, keys?: string) => (keys ? <span className="cv-tip">{label} <Kbd keys={keys} /></span> : label);

/** The tier chips; `collapsed` (a narrow bar): only the selected tiers'
 *  chips, the tier menu (▾) holds every tier. */
function TierChips({ collapsed = false }: { collapsed?: boolean }) {
  const ctrl = useController();
  const meta = useViewer((s) => s.meta);
  const tiers = useViewer((s) => s.tiers);
  useViewer((s) => s.index);   // per-object tier availability
  if (!meta) return null;
  const all = meta.tiers ?? [];
  const chips = chipTiers(all, tiers).filter((t) => !collapsed || tiers.includes(t.key));
  const hasMenu = collapsed || all.length > chips.length || !!meta.bhr_fwhm_control || (meta.jwst_band_options?.length ?? 0) > 1 || all.some((t) => t.key === "morph");
  if (all.length < 2 && !hasMenu) return null;
  return (
    <div className="cv-bar__group cv-bar__tiers" role="group" aria-label="Tiers" data-g="tiers" data-collapsed={collapsed ? "" : undefined}>
      {all.length > 1 && chips.map((t) => {
        const on = tiers.includes(t.key);
        const disabled = !on && ctrl.tierDisabled(t.key);
        const name = shortTierLabel(t.label);
        const full = sentenceLabel(t.label);
        // A tier this object does not have: dimmed, not clickable, its tooltip says why.
        // Else the tooltip is the tier's own hint (what it is), or its full name.
        const why = disabled ? `${full}: ${ctrl.missingTierLabel(t.key)}` : name !== full ? full : "";
        const tip = disabled || !t.hint ? why : `${full}: ${t.hint}`;
        const chip = (
          <button key={t.key} type="button" className="cv-chip" aria-pressed={on} aria-disabled={disabled || undefined}
            aria-label={disabled ? why : name !== full ? full : undefined}
            onClick={() => { if (!disabled) ctrl.toggleTier(t.key); }}>{name}</button>
        );
        return tip ? <Tooltip key={t.key} content={tip}>{chip}</Tooltip> : chip;
      })}
      {hasMenu && (
        <Popover label="Tiers" width={340} trigger={
          <IconButton size="sm" icon={<VIcon name="chevron" />} label="All tiers and tier options" className="cv-ib" />}>
          <TierMenu />
        </Popover>
      )}
    </div>
  );
}

function ColourChoice({ collapsed = false }: { collapsed?: boolean }) {
  const ctrl = useController();
  const meta = useViewer((s) => s.meta);
  const shown = useViewer((s) => s.shown);
  useViewer((s) => s.tiers);
  const settings = useSettings();
  if (!meta) return null;
  const recs = ctrl.frameKeys().map((k) => shown[k]).filter((s) => s?.kind === "cube").map((s) => s!.rec);
  const options = colourOptions(meta.band_names ?? [], { logMode: meta.render_mode === "log", singlePlane: isSinglePlane(recs) });
  if (!options.length) return null;
  const known = options.some((o) => o.key === settings.color);
  const label = meta.color_label || "Band or colour";
  if (collapsed) {
    // A narrow bar: the band chips as ONE select (Q–Y still switch them)
    return (
      <span className="cv-bar__group" data-g="colour" data-collapsed="">
        <Tooltip content={`${label} (keys ${options.map((o) => o.shortcut).filter(Boolean).join(" ")})`}>
          <span className="cv-colsel">
            <Select size="sm" value={settings.color} aria-label={label}
              onChange={(c) => { if (options.some((o) => o.key === c)) ctrl.setColor(c); }}
              options={[
                ...options.map((o) => ({ value: o.key, label: o.label })),
                ...(known ? [] : [{ value: settings.color, label: settings.color, disabled: true }]),
              ]} />
          </span>
        </Tooltip>
      </span>
    );
  }
  return (
    <span className="cv-bar__group" data-g="colour">
      <Segmented size="sm" className="cv-seg" value={settings.color} aria-label={label}
        onChange={(c) => { if (options.some((o) => o.key === c)) ctrl.setColor(c); }}
        options={[
          ...options.map((o) => ({ value: o.key, label: o.label, title: o.shortcut ? `${o.title} (${o.shortcut})` : o.title })),
          ...(known ? [] : [{ value: settings.color, label: settings.color, title: "Set in the Display panel", disabled: true }]),
        ]} />
    </span>
  );
}

const COMPARE: { value: Compare; label: string; icon: "sideBySide" | "blink" | "swipe"; tip: string }[] = [
  { value: "off", label: "Side by side", icon: "sideBySide", tip: "Side by side" },
  { value: "blink", label: "Blink", icon: "blink", tip: "Blink: cycle the tiers in one frame (B)" },
  { value: "swipe", label: "Swipe", icon: "swipe", tip: "Swipe: reveal the second tier over the first" },
];

function CompareChoice() {
  const ctrl = useController();
  const compare = useViewer((s) => s.compare);
  useViewer((s) => s.tiers);
  useViewer((s) => s.residuals);
  const n = ctrl.frameKeys().length;
  if (n < 2 && compare === "off") return null;
  return (
    <span className="cv-bar__group" data-g="compare">
      <Segmented<Compare> size="sm" className="cv-seg cv-seg--icons" value={compare} aria-label="Compare" disabled={n < 2}
        onChange={(c) => ctrl.setCompare(c)}
        options={COMPARE.map((c) => ({ value: c.value, title: c.tip, label: <><VIcon name={c.icon} /><span className="sr-only">{c.label}</span></> }))} />
    </span>
  );
}

function Navigation() {
  const ctrl = useController();
  const count = useViewer((s) => s.meta?.count ?? 0);
  const index = useViewer((s) => s.index);
  const playing = useViewer((s) => s.playing);
  const [draft, setDraft] = useState<string | null>(null);
  useEffect(() => setDraft(null), [index]);
  if (count < 2) return null;
  const { position, total } = navPosition(index, count);
  const commit = () => {
    const v = parsePosition(draft ?? "");
    setDraft(null);
    if (v != null) ctrl.go(v, false);   // explicit jump → clamp, don't wrap
  };
  return (
    <div className="cv-bar__group cv-bar__nav" role="group" aria-label="Navigation" data-g="nav">
      <IconButton size="sm" className="cv-ib" icon={<VIcon name="prev" />} label="Previous" tooltip={tip("Previous", "ArrowLeft")} onClick={() => ctrl.go(index - 1)} />
      <Tooltip content={`Object ${position} of ${total} (index ${index} in labels and links, which count from 0). Type a number from 1 to ${total} and press Enter.`}>
        <input className="cv-idx" type="text" inputMode="numeric" aria-label={`Object number (1 to ${total})`} value={draft ?? position}
          size={Math.max(2, total.length)}
          onChange={(e) => setDraft(e.target.value)} onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") { commit(); (e.target as HTMLInputElement).blur(); }
            if (e.key === "Escape") { setDraft(null); (e.target as HTMLInputElement).blur(); }
          }} />
      </Tooltip>
      <span className="cv-idx-total" aria-hidden="true">/ {total}</span>
      <IconButton size="sm" className="cv-ib" icon={<VIcon name="next" />} label="Next" tooltip={tip("Next", "ArrowRight")} onClick={() => ctrl.go(index + 1)} />
      <IconButton size="sm" className="cv-ib" icon={<VIcon name={playing ? "pause" : "play"} />} label={playing ? "Stop the run-through" : "Run through the objects"}
        tooltip={tip(playing ? "Stop the run-through" : "Run through the objects", "Space")}
        pressed={playing} onClick={() => ctrl.togglePlay()} />
    </div>
  );
}

function Zoom() {
  const ctrl = useController();
  const view = useViewer((s) => s.view);
  return (
    <div className="cv-bar__group" role="group" aria-label="Zoom" data-g="zoom">
      <IconButton size="sm" className="cv-ib" icon={<VIcon name="zoomOut" />} label="Zoom out" tooltip={tip("Zoom out", "-")}
        disabled={!view} onClick={() => ctrl.zoomBy(1 / ZOOM_STEP)} />
      <IconButton size="sm" className="cv-ib" icon={<VIcon name="zoomIn" />} label="Zoom in" tooltip={tip("Zoom in: whole screen pixels per image pixel (or the wheel on the focused viewer)", "+")}
        onClick={() => ctrl.zoomBy(ZOOM_STEP)} />
      <IconButton size="sm" className="cv-ib" icon={<VIcon name="fit" />} label="Show the whole image" tooltip={tip("Show the whole image (or double-click)", "0")}
        disabled={!view} onClick={() => ctrl.resetView()} />
    </div>
  );
}

function LayoutMenu({ onFullscreen }: { onFullscreen: () => void }) {
  const ctrl = useController();
  const layout = useViewer((s) => s.layout);
  const focus = useViewer((s) => s.focus);
  const pixelExact = useViewer((s) => s.pixelExact);
  const items: MenuItem[] = [
    ...LAYOUT_MODES.map((m) => ({
      type: "checkbox" as const, id: m, checked: layout === m, keepOpen: false,
      label: <span className="cv-menuitem">{LAYOUT_LABEL[m]}<span className="cv-menuitem__hint">{LAYOUT_HINT[m]}</span></span>,
      onCheckedChange: () => ctrl.setLayout(m),
    })),
    { type: "separator" },
    {
      type: "checkbox" as const, id: "pixel-exact", checked: pixelExact, keepOpen: false,
      label: <span className="cv-menuitem">Pixel-exact fit<span className="cv-menuitem__hint">Whole screen pixels per image pixel, even when the image then fills less of its frame (otherwise only when it keeps 80 % of it)</span></span>,
      onCheckedChange: () => ctrl.setPixelExact(!pixelExact),
    },
    { type: "separator" },
    { id: "focus", label: focus ? "Leave focus mode" : "Open large (focus mode)", shortcut: focus ? "Esc" : "F", onSelect: () => ctrl.setFocus(!focus) },
    { id: "fullscreen", label: "Full screen", onSelect: onFullscreen },
  ];
  return <Menu label="Layout" align="end" items={items}
    trigger={<IconButton size="sm" className="cv-ib" icon={<VIcon name="layout" />} label="Arrange the frames, focus mode, full screen" tooltip="Arrange the frames" data-g="layout" />} />;
}

function ExportMenu({ onPng, onFigure, onRecord }: { onPng: () => void; onFigure: () => void; onRecord: () => void }) {
  const ctrl = useController();
  const recording = useViewer((s) => s.recording);
  const saveInFlight = useViewer((s) => s.saveInFlight);
  useViewer((s) => s.frozen);
  useViewer((s) => s.status);
  const reason = ctrl.saveBlockReason();
  const items: MenuItem[] = [
    { id: "png", label: "Save the frames as PNG", onSelect: onPng },
    { id: "figure", label: <span className="cv-menuitem">Publication figure<span className="cv-menuitem__hint">High-resolution plate of the frozen crop, else the current view</span></span>, onSelect: onFigure },
    { id: "video", label: recording ? "Stop recording and save the video" : "Record a video of the frames", onSelect: onRecord },
    { type: "separator" },
    {
      id: "crop", disabled: !!reason, shortcut: "S", onSelect: () => { void ctrl.saveCropToResults(); },
      label: <span className="cv-menuitem">{saveInFlight ? "Saving the crop…" : "Save the crop to results"}<span className="cv-menuitem__hint">{reason || "The frozen matched raw cubes and a manifest"}</span></span>,
    },
  ];
  return <Menu label="Export" align="end" items={items}
    trigger={<IconButton size="sm" className="cv-ib" icon={<VIcon name="download" />} label="Export: PNG, figure, video, save the crop" tooltip="Export" data-g="export" />} />;
}

/* ---- one row or two, and the More menu ------------------------------------------ */

type GroupSpec = { id: string; row: 1 | 2; overflow?: number; present: boolean };

/** Measure the groups and decide one row / two rows / icon-only / what goes
 *  into More (barModel.barLayout); the result lives in the viewer store, so
 *  the frame grid refits in the same layout pass. Widths are cached per
 *  group (`data-g`): a group in the More menu, or the band chips while
 *  collapsed, keep the width they had in the bar, so the decision is the
 *  same in both states (no flip-flop). The button texts likewise. */
function useBarLayout(ref: React.RefObject<HTMLDivElement>, specs: GroupSpec[]): BarLayout {
  const ctrl = useController();
  const lay = useViewer((s) => s.bar);
  const textWidth = useRef(0);
  const widths = useRef<Record<string, number>>({});
  const specsRef = useRef(specs);
  specsRef.current = specs;
  useLayoutEffect(() => {
    const bar = ref.current;
    if (!bar || typeof ResizeObserver === "undefined") return;
    const measure = () => {
      const cs = getComputedStyle(bar);
      const available = bar.clientWidth - (parseFloat(cs.paddingLeft) || 0) - (parseFloat(cs.paddingRight) || 0);
      const gap = parseFloat(cs.columnGap) || 0;
      const cur = ctrl.s.bar;
      if (!cur.compact) {
        let t = 0;
        // what icon-only mode hides: each text button's label and chevron (+ their gaps)
        bar.querySelectorAll<HTMLElement>(".cv-btn .ui-btn__label, .cv-btn .cv-btn__chev").forEach((el) => {
          const btn = el.closest<HTMLElement>(".cv-btn");
          t += el.getBoundingClientRect().width + (btn ? parseFloat(getComputedStyle(btn).columnGap) || 0 : 0);
        });
        textWidth.current = t;
      }
      const rows = Array.from(bar.children).filter((el) => el.classList.contains("cv-bar__row")) as HTMLElement[];
      if (!rows.length) return;   // before the meta: keep the reserved height
      const inBar: Record<string, boolean> = {};
      bar.querySelectorAll<HTMLElement>("[data-g]").forEach((el) => {
        const id = el.dataset.g as string;
        const w = el.getBoundingClientRect().width;
        inBar[id] = true;
        if (el.hasAttribute("data-collapsed")) { widths.current[`${id}:collapsed`] = w; return; }
        widths.current[id] = w;
        if (id === "tiers") {
          // what it would take collapsed: the selected chips and the menu
          const kept = Array.from(el.children).filter((k) => k.getAttribute("aria-pressed") === "true" || !k.classList.contains("cv-chip")) as HTMLElement[];
          const g = parseFloat(getComputedStyle(el).columnGap) || 0;
          const est = kept.reduce((a, k) => a + k.getBoundingClientRect().width, 0) + g * Math.max(0, kept.length - 1) + (el.querySelector(".cv-ib") ? 0 : 30);
          widths.current["tiers:collapsed"] = est;
        }
      });
      const items: BarItem[] = [];
      for (const g of specsRef.current) {
        if (!g.present) continue;
        const moved = cur.overflow.includes(g.id) || cur.collapsed.includes(g.id);
        // a group the bar shows is measured; one in More keeps its last width
        if (!inBar[g.id] && !moved) continue;
        const w = widths.current[g.id];
        if (w == null) continue;
        items.push({
          id: g.id, row: g.row, overflow: g.overflow,
          width: w + (cur.compact && g.id === "display" ? textWidth.current : 0),
          shrink: g.id === "colour" ? widths.current["colour:collapsed"] ?? 80 : g.id === "tiers" ? widths.current["tiers:collapsed"] : undefined,
        });
      }
      ctrl.setBarLayout(barLayout({ items, textWidth: textWidth.current, gap, available: available - 1, moreWidth: widths.current.more ?? 28 }));
    };
    measure();
    const ro = new ResizeObserver(() => measure());
    ro.observe(bar);
    bar.querySelectorAll<HTMLElement>("[data-g]").forEach((el) => ro.observe(el));
    return () => ro.disconnect();
  });
  return lay;
}

/** The More menu lists its groups in the bar's own order. */
const MORE_ORDER = ["compare", "tools", "zoom", "layout", "export"];

/** The groups a narrow bar moved into its More menu, in a popover (it
 *  follows the app theme): each with its plain-words name. */
function MoreMenu({ ids, mode, onPng, onFigure, onRecord, onFullscreen }: {
  ids: string[]; mode: BarMode; onPng: () => void; onFigure: () => void; onRecord: () => void; onFullscreen: () => void;
}) {
  const ctrl = useController();
  const tool = useViewer((s) => s.tool);
  const row = (id: string, label: string, body: React.ReactNode) => (
    <div key={id} className="cv-more__row"><span className="cv-more__label">{label}</span><span className="cv-more__ctl">{body}</span></div>
  );
  return (
    <Popover label="More controls" align="end" width={300} trigger={
      <IconButton size="sm" className="cv-ib" icon={<VIcon name="more" />} label="More controls" tooltip="More controls: compare, zoom, tools, arrange, export" data-g="more" />}>
      <div className="cv-menu cv-more">
        {MORE_ORDER.filter((id) => ids.includes(id)).map((id) => {
          if (id === "compare") return row(id, "Compare", <CompareChoice />);
          if (id === "zoom") return row(id, "Zoom", <Zoom />);
          if (id === "tools") {
            return row(id, "Tools", mode === "full" ? (
              <Popover label="Tools" width={400} trigger={<Button size="sm" variant="ghost" icon={<VIcon name={tool === "lens" ? "lens" : "tools"} />}>Lens, profiles and residuals</Button>}>
                <ToolsMenu />
              </Popover>
            ) : (
              <Button size="sm" variant="ghost" icon={<VIcon name="lens" />} aria-pressed={tool === "lens"}
                onClick={() => ctrl.setTool(tool === "lens" ? "pan" : "lens")}>Magnifier lens <Kbd keys="L" /></Button>
            ));
          }
          if (id === "layout") return row(id, "Arrange", <LayoutMenu onFullscreen={onFullscreen} />);
          if (id === "export") return row(id, "Export", <ExportMenu onPng={onPng} onFigure={onFigure} onRecord={onRecord} />);
          return null;
        })}
      </div>
    </Popover>
  );
}

export function Bar({ mode, nav, onPng, onFigure, onRecord, onFullscreen }: {
  mode: BarMode; nav: boolean; onPng: () => void; onFigure: () => void; onRecord: () => void; onFullscreen: () => void;
}) {
  const ctrl = useController();
  const meta = useViewer((s) => s.meta);
  const tool = useViewer((s) => s.tool);
  const focus = useViewer((s) => s.focus);
  const recording = useViewer((s) => s.recording);
  const compare = useViewer((s) => s.compare);
  useViewer((s) => s.tiers);
  useViewer((s) => s.residuals);
  const ref = useRef<HTMLDivElement>(null);
  const full = mode === "full";
  const dock = useViewer((s) => s.dock);
  const withNav = nav || mode === "nav";
  // Export in the full bar (with or without the navigation) and the nav-only bar.
  const withExport = mode !== "compact";
  const how = mode !== "nav";
  const n = meta ? ctrl.frameKeys().length : 0;
  const specs: GroupSpec[] = [
    { id: "tiers", row: 1, present: how },
    { id: "colour", row: 1, present: how },
    { id: "display", row: 2, present: how },
    { id: "compare", row: 2, overflow: 4, present: how && (n >= 2 || compare !== "off") },
    { id: "tools", row: 2, overflow: 3, present: how },
    { id: "zoom", row: 2, overflow: 5, present: how },
    { id: "nav", row: 2, present: withNav && (meta?.count ?? 0) >= 2 },
    { id: "layout", row: 2, overflow: 2, present: how },
    { id: "rec", row: 2, present: withExport && recording },
    { id: "export", row: 2, overflow: 1, present: withExport },
    { id: "focus", row: 2, present: true },
  ];
  const lay = useBarLayout(ref, specs);
  const moved = lay.overflow;
  const shows = (id: string) => !moved.includes(id);
  return (
    <div ref={ref} className="cv-bar" role="group" aria-label="Image viewer controls"
      data-rows={lay.rows} data-compact={lay.compact || undefined}>
      {meta && how && (
        <div className="cv-bar__row cv-bar__row--what" data-row="1" data-wrap={lay.wrap.includes(1) || undefined}>
          <TierChips collapsed={lay.collapsed.includes("tiers")} />
          <ColourChoice collapsed={lay.collapsed.includes("colour")} />
        </div>
      )}
      {meta && (
        <div className="cv-bar__row cv-bar__row--how" data-row="2" data-wrap={lay.wrap.includes(2) || undefined}>
          {how && <>
            <Tooltip content={dock ? "Close the Display settings" : "Knee, brightness and stretch of this viewer, in a row under the bar"}>
              <Button size="sm" variant="ghost" className="cv-btn" aria-label="Display settings for this viewer" aria-pressed={dock}
                aria-expanded={dock} icon={<VIcon name="display" />} onClick={() => ctrl.setDock(!dock)} data-g="display">Display</Button>
            </Tooltip>
            {shows("compare") && <CompareChoice />}
            {shows("tools") && (full ? (
              <Popover label="Tools" width={400} trigger={
                <IconButton size="sm" className="cv-ib" label="Tools: lens, profiles, residuals, playback" tooltip="Tools: lens, profiles, residuals"
                  icon={<VIcon name={tool === "lens" ? "lens" : "tools"} />} data-g="tools" />}>
                <ToolsMenu />
              </Popover>
            ) : (
              <IconButton size="sm" className="cv-ib" icon={<VIcon name="lens" />} label="Magnifier lens"
                tooltip={tip("Magnifier lens", "L")} data-g="tools"
                pressed={tool === "lens"} onClick={() => ctrl.setTool(tool === "lens" ? "pan" : "lens")} />
            ))}
          </>}
          <span className="cv-bar__spacer" aria-hidden="true" />
          {how && shows("zoom") && <Zoom />}
          {withNav && <Navigation />}
          <div className="cv-bar__group" role="group" aria-label="View">
            {how && shows("layout") && <LayoutMenu onFullscreen={onFullscreen} />}
            {withExport && recording && (
              <Button size="sm" variant="ghost" className="cv-btn cv-rec" aria-label="Stop recording and save the video" icon={<VIcon name="stopRecord" />} onClick={onRecord} data-g="rec">Stop recording</Button>
            )}
            {withExport && shows("export") && <ExportMenu onPng={onPng} onFigure={onFigure} onRecord={onRecord} />}
            {moved.length > 0 && <MoreMenu ids={moved} mode={mode} onPng={onPng} onFigure={onFigure} onRecord={onRecord} onFullscreen={onFullscreen} />}
            <IconButton size="sm" className="cv-ib" icon={<VIcon name={focus ? "unfocus" : "focus"} />} data-g="focus"
              tooltip={tip(focus ? "Leave focus mode" : "Open large: the images fill the page", focus ? "Escape" : "F")}
              label={focus ? "Leave focus mode" : "Open large"} pressed={focus} onClick={() => ctrl.setFocus(!focus)} />
          </div>
        </div>
      )}
    </div>
  );
}
