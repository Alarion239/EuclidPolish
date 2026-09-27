/* <ImageViewer> — viewer engine v2 (spec §6). See src/viewer/README.md.
 *
 *   <ImageViewer collection="nexus-field" params={{ field }} tiers={["lr", "sr", "jwst"]}
 *     urlKey="nexus" onReady={(api) => …} toolbar="full" />
 *
 * One ViewerController per (collection, params); the Display panel store
 * (C7) drives the colour of every linked viewer. */
import { useEffect, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore, type CSSProperties } from "react";
import { useInRouterContext, useLocation } from "react-router-dom";
import { useUrlState } from "../hooks/useUrlState";
import { useShortcutRegistry } from "../hooks/useShortcut";
import { useDisplay } from "../state/display";
import { Bar, type BarMode } from "./Bar";
import { DisplayDock } from "./BarMenus";
import { reservedBarRows, safeStorage } from "./barModel";
import { ViewerController } from "./controller";
import {
  compositeFrames, publicationFigureCanvas, publicationPanelName, recordCanvas, saveCanvasPng,
  type CompositeFrame, type FigurePanel,
} from "./export";
import { figureLayout } from "./fit";
import { ViewerContext, useController, useViewer } from "./hooks";
import { LensLayer } from "./Lens";
import { MarkersContext } from "./markers";
import { ProfilePanel } from "./ProfilePanel";
import { ReadoutBar } from "./ReadoutBar";
import { parseResidualKey } from "./residual";
import type { Selection } from "./selection";
import { TierGrid } from "./TierGrid";
import type { ImageViewerProps } from "./types";
import "./viewer.css";

// ---- URL state ------------------------------------------------------------------

/** "u,v,3.2" (angular side, arcsec) or "u,v,r0.25" (relative side). */
export function serializeView(v: Selection | null): string {
  if (!v) return "";
  const r = (x: number) => String(Math.round(x * 1e5) / 1e5);
  const side = v.angularSideArcsec && v.angularSideArcsec > 0 ? r(v.angularSideArcsec) : `r${r(v.relativeSide ?? 1)}`;
  return `${r(v.u)},${r(v.v)},${side}`;
}
export function parseView(raw: string | null): Selection | null {
  if (!raw) return null;
  const m = /^(-?[\d.]+),(-?[\d.]+),(r?)([\d.]+)$/.exec(raw.trim());
  if (!m) return null;
  const u = Number(m[1]), v = Number(m[2]), side = Number(m[4]);
  if (![u, v, side].every(Number.isFinite) || !(side > 0)) return null;
  return m[3] ? { u, v, angularSideArcsec: null, relativeSide: side } : { u, v, angularSideArcsec: side, relativeSide: null };
}

type UrlInit = { index?: number; id?: string; tiers?: string[]; residuals?: string[]; view?: Selection | null; color?: string };

function readUrlInit(key: string, search: string): UrlInit {
  const q = new URLSearchParams(search);
  const p = (k: string) => q.get(`v.${key}.${k}`);
  const out: UrlInit = {};
  const i = Number(p("i"));
  if (p("i") != null && Number.isInteger(i) && i >= 0) out.index = i;
  if (p("id")) out.id = p("id") as string;
  if (p("t")) out.tiers = (p("t") as string).split(",").filter(Boolean);
  if (p("r")) out.residuals = (p("r") as string).split(",").filter((k) => !!parseResidualKey(k));
  if (p("z")) out.view = parseView(p("z"));
  if (p("c")) out.color = p("c") as string;
  return out;
}

/* Every mounted viewer's URL writes are flushed in ONE tick: useUrlState
   coalesces setters within a tick, but two viewers writing in separate
   macrotasks before a re-render could overwrite each other's params. */
const pendingUrlWrites = new Map<object, () => void>();
let urlFlushTimer: ReturnType<typeof setTimeout> | null = null;
function scheduleUrlWrite(owner: object, write: () => void) {
  pendingUrlWrites.set(owner, write);
  if (urlFlushTimer) return;
  urlFlushTimer = setTimeout(() => {
    urlFlushTimer = null;
    const writes = [...pendingUrlWrites.values()];
    pendingUrlWrites.clear();
    for (const w of writes) w();
  }, 200);
}

/** What the viewer showed at mount without URL state, and which of its keys
 *  the URL carried: the default state is never written (visiting a page must
 *  not rewrite its URL), a key the URL had stays written. */
type UrlBase = { index: number; id: string | null; tiers: string[] | null; had: { id: boolean; i: boolean; t: boolean; c: boolean } };

/** Mirrors the viewer into `v.<key>.*` (replace mode, debounced). */
function UrlSync({ urlKey, base }: { urlKey: string; base: UrlBase }) {
  const ctrl = useController();
  const [, setI] = useUrlState(`v.${urlKey}.i`, "");
  const [, setId] = useUrlState(`v.${urlKey}.id`, "");
  const [, setT] = useUrlState(`v.${urlKey}.t`, "");
  const [, setR] = useUrlState(`v.${urlKey}.r`, "");
  const [, setZ] = useUrlState(`v.${urlKey}.z`, "");
  const [, setC] = useUrlState(`v.${urlKey}.c`, "");
  const setters = useRef({ setI, setId, setT, setR, setZ, setC });
  setters.current = { setI, setId, setT, setR, setZ, setC };
  const meta = useViewer((s) => s.meta);
  const index = useViewer((s) => s.index);
  const tiers = useViewer((s) => s.tiers);
  const residuals = useViewer((s) => s.residuals);
  const view = useViewer((s) => s.view);
  const override = useViewer((s) => s.override);
  const owner = useRef({});
  // The colour override in place when the meta arrived (the `display` prop, or
  // a page's setView from onReady) is the page's default, not URL state.
  const baseColor = useRef<string | null | undefined>(undefined);
  if (meta && baseColor.current === undefined) baseColor.current = typeof override.color === "string" ? override.color : null;
  useEffect(() => () => { pendingUrlWrites.delete(owner.current); }, []);
  useEffect(() => {
    if (!meta) return;
    scheduleUrlWrite(owner.current, () => {
      const S = setters.current;
      const objs = meta.objects ?? [];
      const obj = objs[index];
      const id = typeof obj?.id === "string" ? obj.id : "";
      // The object: only once it differs from the mount-time one (or the URL named one).
      const byId = base.id != null ? objs.findIndex((o) => o.id === base.id) : -1;
      const defaultIndex = byId >= 0 ? byId : base.index;
      const writeObject = index !== defaultIndex || base.had.id || base.had.i;
      S.setId(writeObject ? id : "");
      S.setI(writeObject && !id ? String(index) : "");
      // The tiers: only when they differ from the mount-time tiers (the page's
      // `tiers` prop, else meta.default_tier), compared as the engine keeps them.
      const keys = (meta.tiers ?? []).map((t) => t.key);
      const canon = (list: string[]) => keys.filter((k) => list.includes(k) && (tiers.includes(k) || !ctrl.tierDisabled(k))).join(",");
      const defaults = base.tiers?.length ? base.tiers : meta.default_tier ? [meta.default_tier] : [];
      S.setT(base.had.t || canon(tiers) !== canon(defaults) ? tiers.join(",") : "");
      S.setR(residuals.join(","));
      // The view in the first frame's coordinates (restored with that frame as its source tier).
      const first = ctrl.frameKeys()[0];
      S.setZ(serializeView(view && first ? ctrl.selectionOn(first, view) : view));
      const color = typeof override.color === "string" ? override.color : "";
      S.setC(base.had.c || color !== (baseColor.current ?? "") ? color : "");
    });
  }, [ctrl, base, meta, index, tiers, residuals, view, override]);
  return null;
}

// ---- keyboard help in the ? sheet ----------------------------------------------------

const HELP: [string, string][] = [
  ["q", "Colour: VIS (then W Y, E J, R H, T Lupton, Y Temp)"],
  ["ArrowLeft", "Previous object"],
  ["ArrowRight", "Next object"],
  ["Space", "Run through the objects"],
  ["s", "Save the frozen crop to results (Shift+letter = the letter, unless the shell or page binds it)"],
  ["+", "Zoom in (wheel zooms when the viewer is focused or ⌘/Ctrl is held)"],
  ["-", "Zoom out"],
  ["0", "Fit the whole image (or double-click)"],
  ["l", "Toggle the magnifier lens (hold Alt for a moment's lens)"],
  ["b", "Blink the selected tiers"],
  ["f", "Focus mode: the images fill the page (F or Esc returns)"],
  ["Escape", "Unfreeze the lens crop, clear the profile, leave focus mode"],
];
let helpUsers = 0;
function useViewerHelp() {
  useEffect(() => {
    helpUsers++;
    if (helpUsers === 1) {
      const reg = useShortcutRegistry.getState();
      HELP.forEach(([combo, description], i) => reg.add({ id: -1000 - i, combo, description, scope: "Image viewer (hovered or focused)", hidden: false }));
    }
    return () => {
      helpUsers--;
      if (helpUsers === 0) HELP.forEach((_, i) => useShortcutRegistry.getState().remove(-1000 - i));
    };
  }, []);
}

// ---- the viewer ---------------------------------------------------------------------------

function ViewerBody({ ctrl, toolbar, nav, urlKey, urlBase, onFullscreen }: {
  ctrl: ViewerController; toolbar: ImageViewerProps["toolbar"]; nav: boolean; urlKey?: string; urlBase: UrlBase | null;
  onFullscreen: () => void;
}) {
  const inRouter = useInRouterContext();
  const profileOpen = useViewer((s) => s.profileOpen);
  const recording = useViewer((s) => s.recording);
  const dock = useViewer((s) => s.dock);

  const compositeInput = (): CompositeFrame[] => ctrl.frameKeys().flatMap((k) => {
    const h = ctrl.frames.get(k);
    if (!h || h.element.classList.contains("cv-frame--hidden")) return [];
    const st = ctrl.s.status[k];
    return [{
      // The canvas's own rect: the frame's border box is 2 px larger.
      canvas: h.visible, rect: h.visible.getBoundingClientRect(),
      label: ctrl.s.overlay[k] ?? ctrl.tierLabel(k),
      message: st && (st.kind === "error" || st.kind === "missing") ? st.message : "",
    }];
  });

  const savePng = () => {
    const out = compositeFrames(compositeInput());
    if (out) saveCanvasPng(out, `${ctrl.stem()}.png`);
  };

  const exportFigure = () => {
    const settings = ctrl.settings();
    // The frozen crop, else the current pan/zoom view, else the whole image.
    const region = ctrl.s.frozen ?? ctrl.s.view;
    const panels: FigurePanel[] = ctrl.frameKeys().flatMap((k) => {
      const h = ctrl.frames.get(k);
      const g = ctrl.geomOf(k);
      const heatbar = ctrl.heatbarInfo(k, settings);
      if (!h || !g || !heatbar || h.source.width < 2) return [];
      return [{
        source: h.source, width: h.source.width, height: h.source.height,
        crop: region ? ctrl.cropOf(k, region) : null,
        name: publicationPanelName(k, ctrl.tierLabel(k)),
        pixscale: g.pixscale,
        heatbar,
      }];
    });
    // The plate follows the on-screen arrangement (one row, or two rows).
    const out = publicationFigureCanvas(panels, figureLayout(ctrl.s.fit, panels.length), ctrl.K0());
    if (out) saveCanvasPng(out, `${ctrl.stem()}_figure.png`);
  };

  const toggleRecord = () => {
    const r = ctrl.getRecorder();
    if (r) { r.stop(); return; }
    const rec = recordCanvas((target) => { compositeFrames(compositeInput(), target); }, ctrl.stem(), () => ctrl.setRecorder(null));
    ctrl.setRecorder(rec);
  };
  void recording;

  useEffect(() => {
    ctrl.onExportFigure = exportFigure;
    ctrl.onSavePng = savePng;
  });

  const mode: BarMode | null = toolbar === "none" ? (nav ? "nav" : null) : toolbar === "compact" ? "compact" : "full";
  return (
    <>
      {urlKey && inRouter && urlBase && <UrlSync urlKey={urlKey} base={urlBase} />}
      <div className="cv-table">
        {mode && <Bar mode={mode} nav={nav} onPng={savePng} onFigure={exportFigure} onRecord={toggleRecord} onFullscreen={onFullscreen} />}
        <div className="cv-body" data-dock={(dock && (mode === "full" || mode === "compact")) || undefined}>
          <TierGrid />
          {dock && (mode === "full" || mode === "compact") && <DisplayDock basic={mode === "compact"} />}
        </div>
        <ReadoutBar />
      </div>
      {profileOpen && <div className="cv-panels"><ProfilePanel /></div>}
      <LensLayer />
    </>
  );
}

/** The light table before the controller exists: the same bar / frame /
 *  readout boxes, so nothing jumps when the meta arrives. */
function Placeholder({ bar, collection }: { bar: boolean; collection: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [rows, setRows] = useState<1 | 2>(() => reservedBarRows(collection, 0, safeStorage()));
  // Two rows when the viewer is narrower than a one-row bar needs (unless
  // this collection's bar was one row last time).
  useLayoutEffect(() => { setRows(reservedBarRows(collection, ref.current?.clientWidth ?? 0, safeStorage())); }, [collection]);
  return (
    <div ref={ref} className="cv-table">
      {bar && <div className="cv-bar" data-rows={rows} data-reserve aria-hidden="true" />}
      <div className="cv-frames"><div className="cv-frame cv-frame--message cv-loading"><div className="cv-msg"><span>Loading…</span></div></div></div>
      <div className="cv-readout" aria-hidden="true" />
    </div>
  );
}

const noFocus = { subscribe: () => () => {}, get: () => false };

/** Focus mode: where the viewer sits (the stage below the top bar, or the
 *  whole screen in full screen), and full screen through the Fullscreen API
 *  (on the document, so the menus and tooltips — portalled to <body> — stay
 *  visible). Leaving full screen leaves focus mode too. */
function useFocusMode(ctrl: ViewerController | null, root: React.RefObject<HTMLDivElement>) {
  const subscribe = useMemo(() => (ctrl ? (fn: () => void) => ctrl.store.subscribe(fn) : noFocus.subscribe), [ctrl]);
  const focus = useSyncExternalStore(subscribe, ctrl ? () => ctrl.store.getState().focus : noFocus.get);
  const [rect, setRect] = useState<{ top: number; left: number; width: number; height: number } | null>(null);
  const [placeholder, setPlaceholder] = useState(0);
  const fsByUs = useRef(false);
  const lastHeight = useRef(0);

  useLayoutEffect(() => {
    if (!focus && root.current) lastHeight.current = root.current.offsetHeight;
  });

  useLayoutEffect(() => {
    if (!focus) { setRect(null); setPlaceholder(0); return; }
    setPlaceholder(lastHeight.current);
    const stage = document.querySelector<HTMLElement>("main.stage");
    const place = () => {
      if (document.fullscreenElement) { setRect({ top: 0, left: 0, width: window.innerWidth, height: window.innerHeight }); return; }
      const r = stage?.getBoundingClientRect();
      const top = r ? Math.max(0, r.top) : 0, left = r ? Math.max(0, r.left) : 0;
      setRect({ top, left, width: Math.max(0, window.innerWidth - left), height: Math.max(0, window.innerHeight - top) });
    };
    place();
    root.current?.focus({ preventScroll: true });
    window.addEventListener("resize", place);
    document.addEventListener("fullscreenchange", place);
    const ro = typeof ResizeObserver !== "undefined" && stage ? new ResizeObserver(place) : null;
    if (stage) ro?.observe(stage);
    return () => {
      window.removeEventListener("resize", place);
      document.removeEventListener("fullscreenchange", place);
      ro?.disconnect();
    };
  }, [focus, root]);

  // Leaving focus mode leaves the full screen we entered; leaving full
  // screen (the browser's Esc) leaves focus mode.
  useEffect(() => {
    if (!ctrl) return;
    const onChange = () => {
      if (!document.fullscreenElement && fsByUs.current) { fsByUs.current = false; ctrl.setFocus(false); }
    };
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, [ctrl]);
  useEffect(() => {
    if (!focus && fsByUs.current && document.fullscreenElement) { fsByUs.current = false; void document.exitFullscreen?.().catch(() => {}); }
  }, [focus]);
  useEffect(() => () => {
    if (fsByUs.current && document.fullscreenElement) void document.exitFullscreen?.().catch(() => {});
  }, []);

  const enterFullscreen = () => {
    if (!ctrl) return;
    ctrl.setFocus(true);
    const el = document.documentElement;
    if (!document.fullscreenElement && el.requestFullscreen) {
      el.requestFullscreen().then(() => { fsByUs.current = true; }, () => { /* refused: focus mode stays */ });
    }
  };

  const style: CSSProperties | undefined = focus && rect
    ? { top: rect.top, left: rect.left, width: rect.width, height: rect.height }
    : undefined;
  return { focus, style, placeholder, enterFullscreen };
}

/** The viewer reads its initial URL state from the router's location (or the
 *  window's outside a router). */
export function ImageViewer(props: ImageViewerProps) {
  return useInRouterContext()
    ? <RoutedViewer {...props} />
    : <ViewerCore {...props} search={typeof window === "undefined" ? "" : window.location.search} />;
}

function RoutedViewer(props: ImageViewerProps) {
  return <ViewerCore {...props} search={useLocation().search} />;
}

function ViewerCore(props: ImageViewerProps & { search: string }) {
  const { collection, params, tiers, initialIndex, initialId, id, urlKey, onState, onReady, toolbar = "full", nav = true, className, display, markers } = props;
  const searchRef = useRef(props.search);
  searchRef.current = props.search;
  const paramsKey = JSON.stringify(params ?? {});
  const [ctrl, setCtrl] = useState<ViewerController | null>(null);
  const [urlBase, setUrlBase] = useState<UrlBase | null>(null);
  const onReadyRef = useRef(onReady);
  onReadyRef.current = onReady;
  const onStateRef = useRef(onState);
  onStateRef.current = onState;
  const init = useRef({ tiers, initialIndex, initialId, id, display, params });
  init.current = { tiers, initialIndex, initialId, id, display, params };
  useViewerHelp();
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const i = init.current;
    const fromUrl = urlKey ? readUrlInit(urlKey, searchRef.current) : {};
    const c = new ViewerController({
      collection,
      params: i.params,
      tiers: fromUrl.tiers ?? i.tiers,
      initialIndex: fromUrl.index ?? i.initialIndex,
      initialId: fromUrl.id ?? i.initialId,
      id: i.id,
      display: fromUrl.color ? { ...(i.display ?? {}), color: fromUrl.color as never } : i.display,
    });
    if (fromUrl.residuals?.length) c.store.setState({ residuals: fromUrl.residuals });
    // Reserve the bar's height (rows) until the meta arrives and it is measured.
    c.store.setState({ bar: { ...c.store.getState().bar, rows: reservedBarRows(collection, rootRef.current?.clientWidth ?? 0, safeStorage()) } });
    c.setOnState((s) => onStateRef.current?.(s));
    setUrlBase({
      index: i.initialIndex ?? 0, id: i.initialId ?? null, tiers: i.tiers ?? null,
      had: { id: fromUrl.id != null, i: fromUrl.index != null, t: fromUrl.tiers != null, c: fromUrl.color != null },
    });
    setCtrl(c);
    onReadyRef.current?.(c.api);
    // A URL view is in the first frame's coordinates (UrlSync writes it so).
    void c.start().then(() => { if (fromUrl.view) c.setViewSelection({ ...fromUrl.view, sourceTier: c.frameKeys()[0] }); });
    return () => {
      onReadyRef.current?.(null);
      c.destroy();
    };
  }, [collection, paramsKey, urlKey]);

  // Colour-mode changes from the Display panel refresh the magnitude overlays.
  useEffect(() => {
    if (!ctrl) return;
    return useDisplay.subscribe((s, prev) => {
      if (s.color !== prev.color || s.groups !== prev.groups) ctrl.afterDisplayChange();
    });
  }, [ctrl]);

  // Releasing Alt ends the temporary lens.
  useEffect(() => {
    if (!ctrl) return;
    const up = (e: KeyboardEvent) => { if (e.key === "Alt") ctrl.setAltLens(false); };
    window.addEventListener("keyup", up);
    return () => window.removeEventListener("keyup", up);
  }, [ctrl]);

  const focusMode = useFocusMode(ctrl, rootRef);

  return (
    <div className="cv-host" style={focusMode.placeholder ? { minHeight: focusMode.placeholder } : undefined}>
      <div ref={rootRef} className={`cv-root${className ? ` ${className}` : ""}`} tabIndex={0} data-collection={collection}
        data-focus={focusMode.focus || undefined} style={focusMode.style}
        aria-label={`Image viewer, ${collection}${focusMode.focus ? " (focus mode: F or Esc returns)" : ""}`}
        onMouseEnter={() => ctrl?.activate()}
        onMouseLeave={() => { if (!ctrl) return; ctrl.deactivate(); ctrl.hideHover(); ctrl.clearReadout(); }}
        onFocus={() => ctrl?.activate()}
        onBlur={(e) => { if (ctrl && !e.currentTarget.contains(e.relatedTarget as Node | null)) ctrl.deactivate(); }}>
        {ctrl
          ? <ViewerContext.Provider value={ctrl}><MarkersContext.Provider value={markers ?? null}>
            <ViewerBody ctrl={ctrl} toolbar={toolbar} nav={nav} urlKey={urlKey} urlBase={urlBase} onFullscreen={focusMode.enterFullscreen} />
          </MarkersContext.Provider></ViewerContext.Provider>
          : <Placeholder bar={toolbar !== "none" || nav} collection={collection} />}
      </div>
    </div>
  );
}
