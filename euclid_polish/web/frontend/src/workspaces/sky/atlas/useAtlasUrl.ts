/* The atlas's URL state (codecs in urlState.ts). Everything that defines a
 * view is here, so any atlas view is a shareable link:
 *   ra, dec, fov (deg) · proj · base · ov · layers · sel · gal · img (pixel
 *   overlays, pixelOverlays.ts) · blink · goto (one-shot) */
import { useMemo } from "react";
import { useUrlState } from "../../../hooks/useUrlState";
import type { Region } from "../../../sky/geometry";
import { DEFAULT_BASE, DEFAULT_VIEW, type Projection } from "../../../sky/surveys";
import { DEFAULT_LAYERS } from "./layerModel";
import { IMG_CODEC, type PixelOverlaySetting } from "./pixelOverlays";
import {
  LAYERS_CODEC, OVERLAYS_CODEC, REGION_CODEC, fmtCoord, fmtFovParam, parseProjection,
  type LayerSetting, type OverlaySetting,
} from "./urlState";

const optNumber = (fmt: (v: number) => string) => ({
  parse: (raw: string) => {
    const n = Number(raw);
    return raw.trim() !== "" && Number.isFinite(n) ? n : undefined;
  },
  serialize: (v: number | null) => (v == null ? null : fmt(v)),
});

const RA_CODEC = optNumber(fmtCoord);
const FOV_CODEC = {
  parse: (raw: string) => { const n = Number(raw); return Number.isFinite(n) && n > 0 && n <= 360 ? n : undefined; },
  serialize: (v: number | null) => (v == null ? null : fmtFovParam(v)),
};
const PROJ_CODEC = { parse: parseProjection, serialize: (v: Projection) => v };
const LAYERS_DEFAULT: LayerSetting[] = DEFAULT_LAYERS.map((id) => ({ id }));
const OV_DEFAULT: OverlaySetting[] = [];
const IMG_DEFAULT: PixelOverlaySetting[] = [];

export type AtlasUrl = ReturnType<typeof useAtlasUrl>;

export function useAtlasUrl() {
  const [ra, setRa] = useUrlState<number | null>("ra", null, RA_CODEC);
  const [dec, setDec] = useUrlState<number | null>("dec", null, RA_CODEC);
  const [fov, setFov] = useUrlState<number | null>("fov", null, FOV_CODEC);
  const [proj, setProj] = useUrlState<Projection>("proj", DEFAULT_VIEW.proj, PROJ_CODEC);
  const [base, setBase] = useUrlState<string>("base", DEFAULT_BASE);
  const [ov, setOv] = useUrlState<OverlaySetting[]>("ov", OV_DEFAULT, OVERLAYS_CODEC);
  const [layers, setLayers] = useUrlState<LayerSetting[]>("layers", LAYERS_DEFAULT, LAYERS_CODEC);
  const [sel, setSel] = useUrlState<Region | null>("sel", null, REGION_CODEC);
  const [gal, setGal] = useUrlState<boolean>("gal", false);
  const [img, setImg] = useUrlState<PixelOverlaySetting[]>("img", IMG_DEFAULT, IMG_CODEC);
  const [blink, setBlink] = useUrlState<boolean>("blink", false);
  const [goto, setGoto] = useUrlState<string>("goto", "");
  return useMemo(() => ({
    ra, dec, fov, proj, base, ov, layers, sel, gal, img, blink, goto,
    setRa, setDec, setFov, setProj, setBase, setOv, setLayers, setSel, setGal, setImg, setBlink, setGoto,
    /** Write a view (all three in one tick → one history entry). */
    setView: (v: { ra: number; dec: number; fov?: number }) => {
      setRa(v.ra); setDec(v.dec);
      if (v.fov != null) setFov(v.fov);
    },
  }), [ra, dec, fov, proj, base, ov, layers, sel, gal, img, blink, goto,
    setRa, setDec, setFov, setProj, setBase, setOv, setLayers, setSel, setGal, setImg, setBlink, setGoto]);
}
