/* The sky itself: the slot the one Aladin engine is attached to, and the
 * glue between the engine and the atlas state.
 *
 * URL ⇄ engine: the view (ra/dec/fov) is written to the URL (debounced,
 * replace) when the user moves; a URL change (palette, Back, a link, a card's
 * "show on sky") moves the engine. Layers, the image stack, the selection
 * region, the inspected feature, the theme, the grid and the projection are
 * pushed into the engine by effects. Aladin callbacks come in through the
 * engine's fan-out: click → inspector, hover → tooltip, move → status bar,
 * right-click → our context menu. */
import { useCallback, useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { openInspector } from "../../../app/inspector";
import { formatDec, formatRA } from "../../../format";
import {
  SkyEngineError, currentSkyEngine, getSkyEngine, type BaseColor, type ContextEvent, type SkyEngine, type StackEntry,
} from "../../../sky/engine";
import { angularDistance, type Region } from "../../../sky/geometry";
import { DEFAULT_VIEW } from "../../../sky/surveys";
import { useResolvedTheme } from "../../../state/prefs";
import { Button, Callout, EmptyState, Menu, Spinner, copyText, toast, type MenuItem } from "../../../ui";
import { cacheTileAt, coordText, downloadPair, esaskyUrl, simbadUrl } from "./actions";
import { readSkyPalette } from "./colorScale";
import { rendererFor } from "./engineHooks";
import { featureAt, preferHit } from "./hitTest";
import { featureFacts, type SkyFeature } from "./layerModel";
import type { RenderSpec } from "./render";
import { useAtlas, useSkyDisplay, type AtlasState } from "./store";
import type { AtlasUrl } from "./useAtlasUrl";
import { parseGoto, pointTargetId } from "./urlState";

const URL_WRITE_DEBOUNCE_MS = 350;

function readBackground(): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue("--sky-bg").trim();
  return v || "black";
}

function sameView(a: { ra: number; dec: number; fov: number }, b: { ra: number; dec: number; fov: number }): boolean {
  const tol = Math.max(a.fov / 4000, 1e-6);
  return angularDistance(a.ra, a.dec, b.ra, b.dec) <= tol && Math.abs(a.fov - b.fov) <= a.fov * 0.002;
}

export type SkyStageProps = {
  url: AtlasUrl;
  /** The view without URL coordinates (atlas/home.ts: the last tile, else EDF-N). */
  home?: { ra: number; dec: number; fov: number };
  specs: readonly RenderSpec[];
  base: StackEntry | null;
  overlays: readonly StackEntry[];
  baseColor: BaseColor;
  region: Region | null;
  focus: SkyFeature | null;
  children?: ReactNode;
};

export function SkyStage({ url, home = DEFAULT_VIEW, specs, base, overlays, baseColor, region, focus, children }: SkyStageProps) {
  const slot = useRef<HTMLDivElement>(null);
  const [engine, setEngine] = useState<SkyEngine | null>(null);
  const [attempt, setAttempt] = useState(0);
  const status = useAtlas((s) => s.status);
  const error = useAtlas((s) => s.error);
  const errorCode = useAtlas((s) => s.errorCode);
  const theme = useResolvedTheme();
  const urlRef = useRef(url);
  urlRef.current = url;
  const homeRef = useRef(home);
  homeRef.current = home;
  const specsRef = useRef(specs);
  specsRef.current = specs;
  const [ctx, setCtx] = useState<ContextEvent | null>(null);
  const hovering = useAtlas((s) => s.hover != null);

  /* create / attach / park */
  useLayoutEffect(() => {
    const el = slot.current;
    if (!el) return;
    let alive = true;
    const store = useAtlas.getState();
    const existing = currentSkyEngine();
    if (existing) {
      existing.attach(el);
      setEngine(existing);
      store.set({ status: "ready", error: null, errorCode: null });
    } else {
      store.set({ status: "loading", error: null, errorCode: null });
      const u = urlRef.current;
      const h = homeRef.current;
      getSkyEngine(() => ({
        view: {
          ra: u.ra ?? h.ra, dec: u.dec ?? h.dec,
          fov: u.fov ?? (u.ra != null ? 0.5 : h.fov), proj: u.proj,
        },
        theme: document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light",
        background: readBackground(),
        size: { width: el.clientWidth, height: el.clientHeight },
      })).then((e) => {
        if (!alive) return;
        e.attach(el);
        setEngine(e);
        useAtlas.getState().set({ status: "ready" });
      }).catch((err: unknown) => {
        if (!alive) return;
        useAtlas.getState().set({
          status: "error",
          error: err instanceof Error ? err.message : String(err),
          errorCode: err instanceof SkyEngineError ? err.code : "load",
        });
      });
    }
    return () => {
      alive = false;
      currentSkyEngine()?.detach(el);
    };
  }, [attempt]);

  /* engine → store / URL / inspector */
  useEffect(() => {
    if (!engine) return;
    const r = rendererFor(engine);
    const store = useAtlas.getState;
    let writeTimer: ReturnType<typeof setTimeout> | undefined;
    let moveFrame = 0;
    let lastMove: { ra: number; dec: number; x: number; y: number } | null = null;
    let pendingHit: SkyFeature | null = null;
    let nativeHover = false;
    const syncView = () => {
      const v = engine.getView();
      store().set({ view: v, size: engine.size() });
      clearTimeout(writeTimer);
      writeTimer = setTimeout(() => {
        const u = urlRef.current;
        const h = homeRef.current;
        const cur = engine.getView();
        const inUrl = { ra: u.ra ?? h.ra, dec: u.dec ?? h.dec, fov: u.fov ?? h.fov };
        if (!sameView(cur, inUrl)) u.setView(cur);
      }, URL_WRITE_DEBOUNCE_MS);
    };
    syncView();
    const offs = [
      engine.events.on("viewChanged", syncView),
      /* Aladin reports shapes only on their outline, then calls `click`
         in the same handler: the outline hit waits for our own inside
         test (hitTest.ts) and the more specific of the two opens. */
      engine.events.on("objectClicked", (obj) => {
        if (store().selecting) return; // drawing a region: clicks are vertices
        const f = r.featureOf(obj);
        if (!f?.inspect) return;
        pendingHit = f;
        queueMicrotask(() => {
          if (pendingHit === f) { pendingHit = null; openInspector(f.inspect!); }
        });
      }),
      engine.events.on("click", (e) => {
        const native = pendingHit;
        pendingHit = null;
        if (store().selecting || (e.isDragging && !native)) return;
        const hit = preferHit(native, e.isDragging ? null : featureAt(e.ra, e.dec, specsRef.current));
        if (hit?.inspect) openInspector(hit.inspect);
      }),
      engine.events.on("objectHovered", (obj, xy) => {
        const f = r.featureOf(obj);
        nativeHover = !!f;
        if (f && xy) store().set({ hover: { feature: f, x: xy.x, y: xy.y } });
      }),
      engine.events.on("objectHoveredStop", () => { nativeHover = false; store().set({ hover: null }); }),
      engine.events.on("mouseMove", (e) => {
        lastMove = e;
        if (moveFrame) return;
        moveFrame = requestAnimationFrame(() => {
          moveFrame = 0;
          const m = lastMove;
          if (!m) return;
          if (!Number.isFinite(m.ra) || !Number.isFinite(m.dec)) { // off the sky
            store().set({ cursor: null, pixel: null, ...(nativeHover ? {} : { hover: null }) });
            return;
          }
          const patch: Partial<AtlasState> = { cursor: m, pixel: engine.readPixel(m.x, m.y) };
          if (!nativeHover) {
            const f = featureAt(m.ra, m.dec, specsRef.current);
            const cur = store().hover;
            if (f) patch.hover = { feature: f, x: m.x, y: m.y };
            else if (cur) patch.hover = null;
          }
          store().set(patch);
        });
      }),
      engine.events.on("contextMenu", (e) => setCtx(e)),
    ];
    return () => {
      offs.forEach((off) => off());
      clearTimeout(writeTimer);
      if (moveFrame) cancelAnimationFrame(moveFrame);
      store().set({ hover: null, cursor: null });
    };
  }, [engine]);

  /* URL → engine view */
  useEffect(() => {
    if (!engine || url.ra == null || url.dec == null) return;
    const v = engine.getView();
    const fov = url.fov ?? (v.fov > 10 ? 0.5 : v.fov);
    const target = { ra: url.ra, dec: url.dec, fov };
    if (!sameView(v, target)) engine.setView(target);
  }, [engine, url.ra, url.dec, url.fov]);

  useEffect(() => { if (engine && url.fov != null && url.ra == null) engine.setView({ fov: url.fov }); }, [engine, url.fov, url.ra]);
  useEffect(() => { engine?.setProjection(url.proj); }, [engine, url.proj]);
  useEffect(() => { engine?.setFrame(url.gal ? "Galactic" : "ICRS"); }, [engine, url.gal]);
  useEffect(() => { engine?.setTheme(theme); }, [engine, theme]);

  const grid = useSkyDisplay((s) => s.grid);
  useEffect(() => {
    if (engine) engine.setGrid(grid, readSkyPalette().ink);
  }, [engine, grid]);

  /* one-shot `goto` (palette: a name for Sesame, or coordinates) */
  useEffect(() => {
    if (!engine || !url.goto) return;
    const target = parseGoto(url.goto);
    const u = urlRef.current;
    u.setGoto("");
    if (!target) return;
    if (target.kind === "coord") {
      u.setView({ ra: target.ra, dec: target.dec, fov: Math.min(engine.getView().fov, 0.5) });
      return;
    }
    const pending = toast.loading(`Resolving “${target.name}”…`);
    engine.gotoObject(target.name).then(([ra, dec]) => {
      toast.success(`${target.name}: ${formatRA(ra)} ${formatDec(dec)}`, { id: pending });
      urlRef.current.setView({ ra, dec, fov: Math.min(engine.getView().fov, 0.5) });
    }).catch((err: unknown) => {
      toast.error(err instanceof Error ? err.message : String(err), { id: pending });
    });
  }, [engine, url.goto]);

  /* image stack: background HiPS + overlay HiPS + pixel overlays */
  const stackSig = `${base?.signature ?? "-"}|${overlays.map((o) => `${o.key}:${o.signature}:${o.opacity ?? ""}:${o.color ? JSON.stringify(o.color) : ""}`).join(",")}`;
  const stackRef = useRef({ base, overlays });
  stackRef.current = { base, overlays };
  useEffect(() => {
    if (!engine) return;
    engine.syncImageStack(stackRef.current.base, stackRef.current.overlays);
  }, [engine, stackSig]);

  const colorSig = JSON.stringify(baseColor);
  useEffect(() => { if (engine) engine.applyBaseColor(baseColor); /* eslint-disable-line react-hooks/exhaustive-deps */ }, [engine, colorSig, base?.signature]);

  /* vector layers */
  useEffect(() => { if (engine) rendererFor(engine).sync(specs); }, [engine, specs]);

  const regionSig = JSON.stringify(region);
  useEffect(() => { if (engine) rendererFor(engine).setRegion(region); /* eslint-disable-line react-hooks/exhaustive-deps */ }, [engine, regionSig]);

  const focusKey = focus ? `${focus.layer}/${focus.key}` : "";
  useEffect(() => { if (engine) rendererFor(engine).setFocus(focus); /* eslint-disable-line react-hooks/exhaustive-deps */ }, [engine, focusKey]);

  /* keyboard: +/− zoom, arrows pan (the canvas itself has no keyboard) */
  const onKey = useCallback((e: KeyboardEvent<HTMLDivElement>) => {
    if (!engine || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "Escape" && useAtlas.getState().selecting) { e.preventDefault(); engine.cancelSelect(); return; }
    if (e.target !== e.currentTarget) return;
    const v = engine.getView();
    const step = v.fov * 0.15;
    const cos = Math.max(0.05, Math.cos((v.dec * Math.PI) / 180));
    let next: { ra?: number; dec?: number; fov?: number } | null = null;
    if (e.key === "+" || e.key === "=") next = { fov: Math.max(v.fov / 1.5, 1e-4) };
    else if (e.key === "-" || e.key === "_") next = { fov: Math.min(v.fov * 1.5, 360) };
    else if (e.key === "ArrowLeft") next = { ra: v.ra + step / cos, dec: v.dec };
    else if (e.key === "ArrowRight") next = { ra: v.ra - step / cos, dec: v.dec };
    else if (e.key === "ArrowUp") next = { ra: v.ra, dec: Math.min(90, v.dec + step) };
    else if (e.key === "ArrowDown") next = { ra: v.ra, dec: Math.max(-90, v.dec - step) };
    if (!next) return;
    e.preventDefault();
    engine.setView(next);
  }, [engine]);

  return (
    <div className="sky-stage" data-status={status} data-hover={hovering || undefined}>
      <div ref={slot} className="sky-stage__slot" tabIndex={0} role="application"
        aria-label="Sky view: drag to pan, wheel or + / − to zoom, arrow keys to pan, right-click for actions"
        onKeyDown={onKey}
        onMouseLeave={() => { if (useAtlas.getState().hover || useAtlas.getState().cursor) useAtlas.getState().set({ hover: null, cursor: null }); }} />
      {status !== "ready" && <EngineFallback status={status} error={error} code={errorCode} onRetry={() => setAttempt((n) => n + 1)} />}
      <HoverTip />
      {ctx && <SkyContextMenu at={ctx} onClose={() => setCtx(null)} onCentre={(ra, dec) => url.setView({ ra, dec })} />}
      {children}
    </div>
  );
}

function EngineFallback({ status, error, code, onRetry }: { status: string; error: string | null; code: string | null; onRetry: () => void }) {
  if (status === "loading" || status === "idle") {
    return (
      <div className="sky-stage__cover" aria-live="polite">
        <Spinner label="Loading the sky engine" />
        <span className="muted">Loading Aladin Lite…</span>
      </div>
    );
  }
  if (code === "webgl2") {
    return (
      <div className="sky-stage__cover">
        <EmptyState icon="globe" title="WebGL2 is not available">
          {error ?? "This browser cannot run the sky engine."} The layers, selection and cards still work from the panel.
        </EmptyState>
      </div>
    );
  }
  return (
    <div className="sky-stage__cover">
      <Callout tone="bad" title="The sky engine failed to load" action={<Button size="sm" onClick={onRetry}>Retry</Button>}>
        {error ?? "Unknown error"}
      </Callout>
    </div>
  );
}

function HoverTip() {
  const hover = useAtlas((s) => s.hover);
  const size = useAtlas((s) => s.size);
  if (!hover) return null;
  const facts = featureFacts(hover.feature);
  // Flip to the cursor's left / top near the right / bottom edge.
  const flipX = size.width > 0 && hover.x > size.width - 280;
  const flipY = size.height > 0 && hover.y > size.height - 110;
  const style = {
    ...(flipX ? { right: size.width - hover.x + 12 } : { left: hover.x + 14 }),
    ...(flipY ? { bottom: size.height - hover.y + 12 } : { top: hover.y + 14 }),
  };
  return (
    <div className="sky-tip" style={style} role="tooltip">
      <strong>{hover.feature.label}</strong>
      {facts.map(([k, v]) => <span key={k}><em>{k}</em> {v}</span>)}
    </div>
  );
}

function SkyContextMenu({ at, onClose, onCentre }: { at: ContextEvent; onClose: () => void; onCentre: (ra: number, dec: number) => void }) {
  const { ra, dec } = at;
  const onSky = ra != null && dec != null;
  const copy = async (text: string) => { if (await copyText(text)) toast.success(`Copied ${text}`); };
  const items: MenuItem[] = onSky ? [
    { type: "label", label: coordText(ra, dec, "sex") },
    { label: "What covers this point", onSelect: () => openInspector({ kind: "source", id: pointTargetId(ra, dec) }) },
    { label: "Centre here", onSelect: () => onCentre(ra, dec) },
    { type: "separator" },
    { label: "Cache a 25.6″ tile here…", onSelect: () => { void cacheTileAt(ra, dec); } },
    { label: "Cache a tile + run production & mean…", onSelect: () => { void cacheTileAt(ra, dec, { run: true }); } },
    { label: "Download JWST × Euclid pair here…", onSelect: () => { void downloadPair({ ra, dec }); } },
    { type: "separator" },
    {
      type: "sub", label: "Copy coordinates", items: [
        { label: `Degrees · ${coordText(ra, dec)}`, onSelect: () => { void copy(coordText(ra, dec)); } },
        { label: `Sexagesimal · ${coordText(ra, dec, "sex")}`, onSelect: () => { void copy(coordText(ra, dec, "sex")); } },
      ],
    },
    { label: "Open in ESASky", onSelect: () => window.open(esaskyUrl(ra, dec), "_blank", "noopener") },
    { label: "Open in SIMBAD", onSelect: () => window.open(simbadUrl(ra, dec), "_blank", "noopener") },
  ] : [{ type: "label", label: "Outside the sky" }];
  return (
    <Menu open onOpenChange={(o) => { if (!o) onClose(); }} label="Sky actions" items={items}
      trigger={<span className="sky-ctx-anchor" style={{ left: at.x, top: at.y }} aria-hidden="true" />} />
  );
}
