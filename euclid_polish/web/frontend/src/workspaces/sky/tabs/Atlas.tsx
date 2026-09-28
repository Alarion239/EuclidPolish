/* Sky › Atlas — where is it? (console regrouping): Aladin Lite v3 with the
 * Euclid / JWST / all-sky HiPS and every local dataset as layers, grouped
 * Real tiles / Targets / Scene inputs / Coverage (each linked to the tab that
 * owns its data); click → inspector card, region selection → bulk compare /
 * run production, right-click → "cache a tile here" / "what covers this
 * point"; the JWST menu discovers observations and caches the NEXUS mosaic
 * and pairs (all confirmed), `?obs=1` opens the discovered observations.
 * Without URL coordinates it opens on the last inspected tile, else EDF-N
 * (atlas/home.ts). Every view is a URL (atlas/useAtlasUrl.ts). The engine is
 * created once and survives navigation (src/sky/engine.ts). */
import { useCallback, useEffect, useMemo, useRef, useState, type RefObject } from "react";
import { registerDisplaySection } from "../../../app/displaySections";
import { usePageActions, type PageAction } from "../../../app/palette";
import { useResource } from "../../../api/query";
import { currentSkyEngine, type StackEntry } from "../../../sky/engine";
import { fitView, type RaDec } from "../../../sky/geometry";
import { footprintsQuery } from "../../../sky/lod";
import {
  BASE_SURVEYS, DEFAULT_VIEW, OVERLAY_SURVEYS, PROJECTIONS, QUICK_JUMPS, baseSurvey, type QuickJump,
} from "../../../sky/surveys";
import { useUrlState } from "../../../hooks/useUrlState";
import { useDisplay } from "../../../state/display";
import { useInspector } from "../../../state/inspector";
import { useResolvedTheme } from "../../../state/prefs";
import { IconButton, downloadBlob, toast } from "../../../ui";
import { discoverJwst, viewRegion } from "../atlas/actions";
import { AtlasToolbar } from "../atlas/AtlasToolbar";
import { atlasHome } from "../atlas/home";
import { readSkyPalette } from "../atlas/colorScale";
import { useLayerData } from "../atlas/layerData";
import {
  findFeatureByTarget, footprintFeatures, withClientLayers, withStubLayers, type FootprintsResponse, type LayersResponse,
  type SkyFeature,
} from "../atlas/layerModel";
import { JwstObservations } from "../atlas/JwstObservations";
import { LayersPanel } from "../atlas/LayersPanel";
import { featuresInRegion, shapeToRegion } from "../atlas/selection";
import { overlayColor } from "../atlas/overlayColor";
import { resolveOverlays } from "../atlas/pixelOverlays";
import { SelectionPanel } from "../atlas/SelectionPanel";
import { SkyDisplaySection } from "../atlas/SkyDisplaySection";
import { SkyStage } from "../atlas/SkyStage";
import { buildSpecs } from "../atlas/specs";
import { StatusBar } from "../atlas/StatusBar";
import { useAtlas, useSkyDisplay } from "../atlas/store";
import { useAtlasUrl } from "../atlas/useAtlasUrl";
import { featureView } from "../atlas/urlState";
import "../atlas/atlas.css";

const NARROW_PX = 620;
const BLINK_MS = 800;

function useContainerWidth(ref: RefObject<HTMLElement>): number {
  const [w, setW] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([e]) => setW(Math.round(e.contentRect.width)));
    ro.observe(el);
    setW(el.clientWidth);
    return () => ro.disconnect();
  }, [ref]);
  return w;
}

function dataUrlToBlob(dataUrl: string): Blob {
  const [head, body] = dataUrl.split(",");
  const mime = /data:([^;]+)/.exec(head)?.[1] ?? "image/png";
  const bin = atob(body ?? "");
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}

export default function Atlas() {
  const url = useAtlasUrl();
  // Without URL coordinates the atlas opens on the last inspected tile, else EDF-N (read once).
  const [home] = useState(atlasHome);
  const [obsOpen, setObsOpen] = useUrlState<boolean>("obs", false);
  const theme = useResolvedTheme();
  const rootRef = useRef<HTMLDivElement>(null);
  const width = useContainerWidth(rootRef);
  const narrow = width > 0 && width < NARROW_PX;
  const [wideOpen, setWideOpen] = useState(true);
  const narrowOpen = useAtlas((s) => s.panelOpen);
  const panelOpen = narrow ? narrowOpen : wideOpen;
  const togglePanel = useCallback(() => {
    if (narrow) useAtlas.getState().set({ panelOpen: !useAtlas.getState().panelOpen });
    else setWideOpen((v) => !v);
  }, [narrow]);

  /* layers */
  const catalogue = useResource<LayersResponse>("/api/sky/layers", [], { ttl: 60_000 });
  const known = useMemo(() => withClientLayers(catalogue.data?.layers ?? []), [catalogue.data]);
  const enabledIds = useMemo(() => url.layers.map((l) => l.id), [url.layers]);
  // Until the catalogue arrives the enabled layers load (and draw) on their own.
  const data = useLayerData(known, enabledIds, !catalogue.data);
  const layers = useMemo(() => (catalogue.data ? known : withStubLayers(known, enabledIds, data)), [catalogue.data, known, enabledIds, data]);
  const palette = useMemo(() => { void theme; return readSkyPalette(); }, [theme]);
  const view = useAtlas((s) => s.view);
  const size = useAtlas((s) => s.size);
  const selecting = useAtlas((s) => s.selecting);
  const markerScale = useSkyDisplay((s) => s.markerScale);
  const fov = view?.fov ?? url.fov ?? home.fov;
  const skyWidth = size.width || 800;

  const fpUrl = enabledIds.includes("jwst-mast") ? footprintsQuery(view) : null;
  const fp = useResource<FootprintsResponse>(fpUrl, [], { ttl: 5 * 60_000 });
  const footprints = useMemo(
    () => (fp.data ? { features: footprintFeatures(fp.data), version: fp.updatedAt ?? 0 } : null),
    [fp.data, fp.updatedAt],
  );

  const specs = useMemo(() => buildSpecs({
    layers, settings: url.layers, data, palette, fov, width: skyWidth, markerScale, footprints,
  }), [layers, url.layers, data, palette, fov, skyWidth, markerScale, footprints]);

  const byLayer = useMemo(() => {
    const out: Record<string, SkyFeature[]> = {};
    for (const [id, d] of Object.entries(data)) out[id] = d.features;
    if (footprints) out["jwst-footprints"] = footprints.features;
    return out;
  }, [data, footprints]);

  /* image stack */
  const survey = baseSurvey(url.base);
  const byBase = useSkyDisplay((s) => s.byBase);
  const overlayStretch = useSkyDisplay((s) => s.overlay);
  const dispColormap = useDisplay((s) => s.colormap);
  const dispStretch = useDisplay((s) => s.stretch);
  const dispInvert = useDisplay((s) => s.invert);
  const dispGroups = useDisplay((s) => s.groups);
  const pixelOverlays = useMemo(() => resolveOverlays(url.img), [url.img]);
  const [blinkIndex, setBlinkIndex] = useState(0);
  const visibleOverlays = useMemo(() => pixelOverlays.filter((o) => o.visible), [pixelOverlays]);
  const blinking = url.blink && visibleOverlays.length >= 2;
  useEffect(() => {
    if (!blinking) return;
    const t = setInterval(() => setBlinkIndex((i) => i + 1), BLINK_MS);
    return () => clearInterval(t);
  }, [blinking]);

  const base = useMemo<StackEntry | null>(() => {
    if (!survey.url) return null;
    const hipsUrl = survey.url;
    return {
      key: "base", signature: `${survey.id}|${hipsUrl}`,
      make: (A) => A.HiPS(hipsUrl, { name: survey.label, ...(survey.format === "fits" ? { imgFormat: "fits" } : {}), ...(survey.color ?? {}) }),
    };
  }, [survey]);

  const stackOverlays = useMemo<StackEntry[]>(() => {
    const hips: StackEntry[] = url.ov.flatMap((o) => {
      const s = OVERLAY_SURVEYS.find((x) => x.id === o.id);
      if (!s) return [];
      return [{ key: `hips:${s.id}`, signature: s.url, opacity: o.opacity ?? s.defaultOpacity, make: (A) => A.HiPS(s.url, { name: s.label }) }];
    });
    const shown = blinking ? visibleOverlays[blinkIndex % visibleOverlays.length]?.key : null;
    const display = { colormap: dispColormap, stretch: dispStretch, invert: dispInvert, groups: dispGroups };
    const imgs: StackEntry[] = pixelOverlays.map((o) => {
      const color = overlayColor(o.tier, display, overlayStretch);
      return {
        key: `img:${o.key}`, signature: o.url, color,
        opacity: !o.visible ? 0 : blinking ? (o.key === shown ? o.opacity : 0) : o.opacity,
        make: (A) => A.image(o.url, {
          name: o.label, ...color,
          successCallback: () => {},
          errorCallback: (e: unknown) => toast.error(`${o.label}: ${e instanceof Error ? e.message : String(e ?? "could not load the FITS")}`),
        }),
      };
    });
    return [...hips, ...imgs];
  }, [url.ov, pixelOverlays, overlayStretch, blinking, blinkIndex, visibleOverlays, dispColormap, dispStretch, dispInvert, dispGroups]);

  const baseColor = useMemo(() => ({ ...(survey.color ?? {}), ...(byBase[survey.id] ?? {}) }), [survey, byBase]);

  /* selection, focus */
  const groups = useMemo(() => featuresInRegion(url.sel, byLayer, [...enabledIds, ...(footprints ? ["jwst-footprints"] : [])]), [url.sel, byLayer, enabledIds, footprints]);
  const inspected = useInspector((s) => (s.open ? s.current : null));
  const focus = useMemo(() => findFeatureByTarget(inspected, byLayer), [inspected, byLayer]);
  // A link that names only the inspected feature (`?inspect=tile:…`, no
  // ra/dec) opens framed on it, not on the whole sky — once per feature.
  const framed = useRef("");
  useEffect(() => {
    if (!focus || url.ra != null || url.dec != null) return;
    const key = `${focus.layer}/${focus.key}`;
    if (framed.current === key) return;
    framed.current = key;
    url.setView(featureView(focus));
  }, [focus, url]);

  /* actions */
  const jump = useCallback((j: Pick<QuickJump, "ra" | "dec" | "fov">) => url.setView({ ra: j.ra, dec: j.dec, fov: j.fov }), [url]);

  const startSelect = useCallback(async (mode: "rect" | "circle" | "poly") => {
    const engine = currentSkyEngine();
    if (!engine || !engine.attachedTo) { toast.error("The sky engine is not ready"); return; }
    useAtlas.getState().set({ selecting: mode });
    const hint = toast.info(mode === "poly" ? "Click the polygon's vertices on the sky; click the first one to close it." : "Drag on the sky to draw the region.", { duration: 60_000 });
    const shape = await engine.select(mode);
    toast.dismiss(hint);
    useAtlas.getState().set({ selecting: null });
    if (!shape) return; // cancelled
    const region = shapeToRegion(shape, (x, y) => engine.pix2world(x, y));
    if (!region) { toast.error("The region must lie on the sky."); return; }
    url.setSel(region);
  }, [url]);

  const exportPng = useCallback(async () => {
    const engine = currentSkyEngine();
    if (!engine) return;
    try {
      const v = engine.getView();
      downloadBlob(`sky-${v.ra.toFixed(3)}_${v.dec.toFixed(3)}.png`, dataUrlToBlob(await engine.exportPng()));
    } catch (e) {
      toast.error(`Export failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }, []);

  const zoomTo = useCallback((id: string) => {
    const info = layers.find((l) => l.id === id);
    if (info?.kind === "moc") { url.setView({ ra: DEFAULT_VIEW.ra, dec: DEFAULT_VIEW.dec, fov: 360 }); return; }
    const feats = data[id]?.features ?? [];
    const pts: RaDec[] = feats.length
      ? feats.flatMap((f) => f.polygon ?? [[f.ra, f.dec] as RaDec])
      : info?.bbox ? [[info.bbox.ra_min, info.bbox.dec_min], [info.bbox.ra_max, info.bbox.dec_max]] : [];
    if (!pts.length) { toast.warning(`${info?.label ?? id} has nothing to zoom to yet`); return; }
    url.setView(fitView(pts, { minFov: 0.01 }));
  }, [layers, data, url]);

  useEffect(() => registerDisplaySection({ id: "sky", title: "Sky", order: 10, Component: SkyDisplaySection }), []);
  // Leaving the atlas mid-drawing leaves Aladin in pan mode.
  useEffect(() => () => currentSkyEngine()?.cancelSelect(), []);

  const actions = useMemo<PageAction[]>(() => [
    ...QUICK_JUMPS.map((j) => ({ id: `sky-jump-${j.id}`, label: `Zoom to ${j.label}`, group: "Sky", keywords: ["go", "fly", "field"], run: () => jump(j) })),
    { id: "sky-allsky", label: "All-sky view", group: "Sky", run: () => jump({ ra: DEFAULT_VIEW.ra, dec: DEFAULT_VIEW.dec, fov: 360 }) },
    ...PROJECTIONS.map((p) => ({ id: `sky-proj-${p}`, label: `Projection: ${p}`, group: "Sky", run: () => url.setProj(p) })),
    { id: "sky-select-rect", label: "Select a rectangle on the sky", group: "Sky", keywords: ["region", "selection"], run: () => { void startSelect("rect"); } },
    { id: "sky-select-circle", label: "Select a circle on the sky", group: "Sky", keywords: ["region", "selection", "cone"], run: () => { void startSelect("circle"); } },
    { id: "sky-select-poly", label: "Select a polygon on the sky", group: "Sky", keywords: ["region", "selection"], run: () => { void startSelect("poly"); } },
    { id: "sky-select-clear", label: "Clear the sky selection", group: "Sky", disabled: !url.sel, run: () => url.setSel(null) },
    { id: "sky-select-cancel", label: "Cancel the region drawing", group: "Sky", disabled: !selecting, run: () => currentSkyEngine()?.cancelSelect() },
    {
      id: "sky-discover", label: "Discover JWST observations in this view", group: "Sky", keywords: ["mast", "jwst"], disabled: !view,
      run: () => { if (view) void discoverJwst({ region: viewRegion(view), label: "the current view" }); },
    },
    { id: "sky-observations", label: "Discovered JWST observations", group: "Sky", keywords: ["mast", "jwst", "pairs"], run: () => setObsOpen(true) },
    { id: "sky-export", label: "Export the sky view as PNG", group: "Sky", run: () => { void exportPng(); } },
    { id: "sky-panel", label: panelOpen ? "Hide the layers panel" : "Show the layers panel", group: "Sky", run: togglePanel },
    { id: "sky-galactic", label: url.gal ? "Equatorial coordinates" : "Galactic coordinates", group: "Sky", run: () => url.setGal(!url.gal) },
    ...BASE_SURVEYS.map((b) => ({ id: `sky-base-${b.id}`, label: `Background: ${b.label}`, group: "Sky", keywords: ["hips", "survey"], run: () => url.setBase(b.id) })),
    { id: "sky-overlays-clear", label: "Remove the pixel overlays", group: "Sky", disabled: !pixelOverlays.length, run: () => { url.setImg([]); url.setBlink(false); } },
  ], [jump, url, startSelect, view, exportPng, panelOpen, togglePanel, pixelOverlays.length, setObsOpen, selecting]);
  usePageActions(actions);

  return (
    <div ref={rootRef} className="sky-atlas" data-narrow={narrow || undefined} data-panel={panelOpen ? "open" : "closed"}>
      <AtlasToolbar url={url} panelOpen={panelOpen} onTogglePanel={togglePanel} onSelect={(m) => { void startSelect(m); }}
        onExport={() => { void exportPng(); }} onJump={jump} onObservations={() => setObsOpen(true)} />
      <JwstObservations open={obsOpen} onOpenChange={setObsOpen} view={view}
        onShow={(ra, dec) => url.setView({ ra, dec, fov: Math.min(view?.fov ?? 0.5, 0.5) })} />
      <div className="sky-atlas__body">
        {panelOpen && (
          <aside className="sky-panel" aria-label="Sky layers"
            onKeyDown={(e) => { if (narrow && e.key === "Escape") useAtlas.getState().set({ panelOpen: false }); }}>
            {narrow && (
              <div className="sky-panel__close">
                <IconButton icon="close" size="sm" label="Close the layers panel" onClick={togglePanel} />
              </div>
            )}
            <LayersPanel url={url} layers={layers} data={data} specs={specs}
              loading={catalogue.loading} error={catalogue.error?.message ?? null} onRetry={catalogue.reload} onZoomTo={zoomTo} />
          </aside>
        )}
        <div className="sky-atlas__main">
          <SkyStage url={url} home={home} specs={specs} base={base} overlays={stackOverlays} baseColor={baseColor}
            region={url.sel} focus={focus}>
            {url.sel && <SelectionPanel groups={groups} layers={layers} onClear={() => url.setSel(null)} />}
          </SkyStage>
          <StatusBar url={url} />
        </div>
      </div>
    </div>
  );
}
