/* The viewer's control bar on the light table: ONE slim row when it fits,
 * else two — what is shown above how it is shown (barModel's barLayout
 * decides from the measured group widths). Only a viewer too narrow for a
 * row even icon-only (~300 px beside the inspector, a phone) wraps that row
 * onto another line: no control is ever cut off or scrolled out of sight.
 *
 *   what  tier chips ▾ · band / colour
 *   how   Display · compare · tools ▾ ·······  − + fit · ◀ i / n ▶ ▶▶ · layout ▾ · export ▾ · focus
 *
 * Everything touched every minute is one click; the rest lives in the
 * popovers of BarMenus.tsx. "Display" opens the Display DOCK beside the
 * frames (not a popover: the image stays in sight while the knee changes).
 * Modes: "full"; "compact" (the dock with knee + brightness only, no tools
 * or export menus — a lens toggle stands in for the tools); "nav" (toolbar
 * "none" with navigation: only the navigation, export and focus). `nav`
 * false drops the navigation only. */
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Button, IconButton, Kbd, Menu, Popover, Segmented, Tooltip, type MenuItem } from "../ui";
import { barLayout, chipTiers, colourOptions, isSinglePlane, navPosition, parsePosition, sentenceLabel, shortTierLabel, type BarLayout } from "./barModel";
import { TierMenu, ToolsMenu } from "./BarMenus";
import { LAYOUT_HINT, LAYOUT_LABEL, LAYOUT_MODES } from "./fit";
import { useController, useSettings, useViewer } from "./hooks";
import { VIcon } from "./icons";
import type { Compare } from "./types";

export type BarMode = "full" | "compact" | "nav";

const tip = (label: string, keys?: string) => (keys ? <span className="cv-tip">{label} <Kbd keys={keys} /></span> : label);

function TierChips() {
  const ctrl = useController();
  const meta = useViewer((s) => s.meta);
  const tiers = useViewer((s) => s.tiers);
  useViewer((s) => s.index);   // per-object tier availability
  if (!meta) return null;
  const all = meta.tiers ?? [];
  const chips = chipTiers(all, tiers);
  const hasMenu = all.length > chips.length || !!meta.bhr_fwhm_control || (meta.jwst_band_options?.length ?? 0) > 1 || all.some((t) => t.key === "morph");
  if (all.length < 2 && !hasMenu) return null;
  return (
    <div className="cv-bar__group cv-bar__tiers" role="group" aria-label="Tiers">
      {all.length > 1 && chips.map((t) => {
        const on = tiers.includes(t.key);
        const disabled = !on && ctrl.tierDisabled(t.key);
        const name = shortTierLabel(t.label);
        const full = sentenceLabel(t.label);
        const why = disabled ? `${full}: ${ctrl.missingTierLabel(t.key)}` : name !== full ? full : "";
        const chip = (
          <button key={t.key} type="button" className="cv-chip" aria-pressed={on} aria-disabled={disabled || undefined}
            aria-label={name !== full ? full : undefined}
            onClick={() => { if (!disabled) ctrl.toggleTier(t.key); }}>{name}</button>
        );
        return why ? <Tooltip key={t.key} content={why}>{chip}</Tooltip> : chip;
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

function ColourChoice() {
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
  return (
    <Segmented size="sm" className="cv-seg" value={settings.color} aria-label={meta.color_label || "Band or colour"}
      onChange={(c) => { if (options.some((o) => o.key === c)) ctrl.setColor(c); }}
      options={[
        ...options.map((o) => ({ value: o.key, label: o.label, title: o.shortcut ? `${o.title} (${o.shortcut})` : o.title })),
        ...(known ? [] : [{ value: settings.color, label: settings.color, title: "Set in the Display panel", disabled: true }]),
      ]} />
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
    <Segmented<Compare> size="sm" className="cv-seg cv-seg--icons" value={compare} aria-label="Compare" disabled={n < 2}
      onChange={(c) => ctrl.setCompare(c)}
      options={COMPARE.map((c) => ({ value: c.value, title: c.tip, label: <><VIcon name={c.icon} /><span className="sr-only">{c.label}</span></> }))} />
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
    <div className="cv-bar__group cv-bar__nav" role="group" aria-label="Navigation">
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
  const first = () => ctrl.frameKeys()[0];
  return (
    <div className="cv-bar__group" role="group" aria-label="Zoom">
      <IconButton size="sm" className="cv-ib" icon={<VIcon name="zoomOut" />} label="Zoom out" tooltip={tip("Zoom out", "-")}
        disabled={!view} onClick={() => { const k = first(); if (k) ctrl.zoomView(k, 1 / 1.5); }} />
      <IconButton size="sm" className="cv-ib" icon={<VIcon name="zoomIn" />} label="Zoom in" tooltip={tip("Zoom in (or the wheel on the focused viewer)", "+")}
        onClick={() => { const k = first(); if (k) ctrl.zoomView(k, 1.5); }} />
      <IconButton size="sm" className="cv-ib" icon={<VIcon name="fit" />} label="Show the whole image" tooltip={tip("Show the whole image (or double-click)", "0")}
        disabled={!view} onClick={() => ctrl.resetView()} />
    </div>
  );
}

function LayoutMenu({ onFullscreen }: { onFullscreen: () => void }) {
  const ctrl = useController();
  const layout = useViewer((s) => s.layout);
  const focus = useViewer((s) => s.focus);
  const items: MenuItem[] = [
    ...LAYOUT_MODES.map((m) => ({
      type: "checkbox" as const, id: m, checked: layout === m, keepOpen: false,
      label: <span className="cv-menuitem">{LAYOUT_LABEL[m]}<span className="cv-menuitem__hint">{LAYOUT_HINT[m]}</span></span>,
      onCheckedChange: () => ctrl.setLayout(m),
    })),
    { type: "separator" },
    { id: "focus", label: focus ? "Leave focus mode" : "Focus mode", shortcut: focus ? "Esc" : "F", onSelect: () => ctrl.setFocus(!focus) },
    { id: "fullscreen", label: "Full screen", onSelect: onFullscreen },
  ];
  return <Menu label="Layout" align="end" items={items}
    trigger={<IconButton size="sm" className="cv-ib" icon={<VIcon name="layout" />} label="Arrange the frames, focus mode, full screen" tooltip="Layout" />} />;
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
    trigger={<IconButton size="sm" className="cv-ib" icon={<VIcon name="download" />} label="Export: PNG, figure, video, save the crop" tooltip="Export" />} />;
}

/* ---- one row or two ------------------------------------------------------------- */

/** The natural width of a bar row: its groups side by side (not the spacer). */
function rowWidth(row: HTMLElement): number {
  const kids = Array.from(row.children).filter((k) => !k.classList.contains("cv-bar__spacer")) as HTMLElement[];
  const gap = parseFloat(getComputedStyle(row).columnGap) || 0;
  return kids.reduce((s, k) => s + k.getBoundingClientRect().width, 0) + gap * Math.max(0, kids.length - 1);
}

/** Measure the rows and decide one row / two rows / icon-only (barModel.barLayout);
 *  the result lives in the viewer store, so the frame grid refits in the same
 *  layout pass. The button texts are measured while shown and added back while
 *  hidden, so the decision is the same in both states (no flip-flop). */
function useBarLayout(ref: React.RefObject<HTMLDivElement>): BarLayout {
  const ctrl = useController();
  const lay = useViewer((s) => s.bar);
  const textWidth = useRef(0);
  useLayoutEffect(() => {
    const bar = ref.current;
    if (!bar || typeof ResizeObserver === "undefined") return;
    const measure = () => {
      const cs = getComputedStyle(bar);
      const available = bar.clientWidth - (parseFloat(cs.paddingLeft) || 0) - (parseFloat(cs.paddingRight) || 0);
      const gap = parseFloat(cs.columnGap) || 0;
      const compact = ctrl.s.bar.compact;
      if (!compact) {
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
      const items = rows.map((r) => ({
        row: (r.dataset.row === "1" ? 1 : 2) as 1 | 2,
        width: rowWidth(r) + (compact && r.querySelector(".cv-btn") ? textWidth.current : 0),
      }));
      ctrl.setBarLayout(barLayout({ items, textWidth: textWidth.current, gap, available: available - 1 }));
    };
    measure();
    const ro = new ResizeObserver(() => measure());
    ro.observe(bar);
    bar.querySelectorAll<HTMLElement>(".cv-bar__row > *").forEach((el) => ro.observe(el));
    return () => ro.disconnect();
  });
  return lay;
}

export function Bar({ mode, nav, onPng, onFigure, onRecord, onFullscreen }: {
  mode: BarMode; nav: boolean; onPng: () => void; onFigure: () => void; onRecord: () => void; onFullscreen: () => void;
}) {
  const ctrl = useController();
  const meta = useViewer((s) => s.meta);
  const tool = useViewer((s) => s.tool);
  const focus = useViewer((s) => s.focus);
  const recording = useViewer((s) => s.recording);
  const ref = useRef<HTMLDivElement>(null);
  const lay = useBarLayout(ref);
  const full = mode === "full";
  const dock = useViewer((s) => s.dock);
  const withNav = nav || mode === "nav";
  // Export in the full bar (with or without the navigation) and the nav-only bar.
  const withExport = mode !== "compact";
  return (
    <div ref={ref} className="cv-bar" role="group" aria-label="Image viewer controls"
      data-rows={lay.rows} data-compact={lay.compact || undefined}>
      {meta && mode !== "nav" && (
        <div className="cv-bar__row cv-bar__row--what" data-row="1" data-wrap={lay.wrap.includes(1) || undefined}>
          <TierChips />
          <ColourChoice />
        </div>
      )}
      {meta && (
        <div className="cv-bar__row cv-bar__row--how" data-row="2" data-wrap={lay.wrap.includes(2) || undefined}>
          {mode !== "nav" && <>
            <Tooltip content={dock ? "Close the Display settings" : "Stretch, knee and brightness of this viewer, beside the images"}>
              <Button size="sm" variant="ghost" className="cv-btn" aria-label="Display settings for this viewer" aria-pressed={dock}
                aria-expanded={dock} icon={<VIcon name="display" />} onClick={() => ctrl.setDock(!dock)}>Display</Button>
            </Tooltip>
            <CompareChoice />
            {full ? (
              <Popover label="Tools" width={400} trigger={
                <IconButton size="sm" className="cv-ib" label="Tools: lens, profiles, residuals, playback" tooltip="Tools"
                  icon={<VIcon name={tool === "lens" ? "lens" : "tools"} />} />}>
                <ToolsMenu />
              </Popover>
            ) : (
              <IconButton size="sm" className="cv-ib" icon={<VIcon name="lens" />} label="Magnifier lens"
                tooltip={tip("Magnifier lens", "L")}
                pressed={tool === "lens"} onClick={() => ctrl.setTool(tool === "lens" ? "pan" : "lens")} />
            )}
          </>}
          <span className="cv-bar__spacer" aria-hidden="true" />
          {mode !== "nav" && <Zoom />}
          {withNav && <Navigation />}
          <div className="cv-bar__group" role="group" aria-label="View">
            {mode !== "nav" && <LayoutMenu onFullscreen={onFullscreen} />}
            {withExport && recording && (
              <Button size="sm" variant="ghost" className="cv-btn cv-rec" aria-label="Stop recording and save the video" icon={<VIcon name="stopRecord" />} onClick={onRecord}>Stop recording</Button>
            )}
            {withExport && <ExportMenu onPng={onPng} onFigure={onFigure} onRecord={onRecord} />}
            <IconButton size="sm" className="cv-ib" icon={<VIcon name={focus ? "unfocus" : "focus"} />}
              tooltip={tip(focus ? "Leave focus mode" : "Focus mode: the images fill the page", focus ? "Escape" : "F")}
              label={focus ? "Leave focus mode" : "Focus mode"} pressed={focus} onClick={() => ctrl.setFocus(!focus)} />
          </div>
        </div>
      )}
    </div>
  );
}
