/* Pixel overlays on the sky: LR / SR(<spec>) / JWST planes of real tiles as
 * `A.image` layers (spec §7.1, `GET /api/real/<source>/<id>/image.fits`).
 * They are URL state, so a reload, Back or a shared link keeps them:
 *
 *   img=nexus/f200w-0200|m:rbf|VIS@0.8,!archive/007|lr|Y_E&blink=1
 *
 * entry = `[!]<ref>|<tier>|<band>[@opacity]` — `!` = hidden, opacity
 * defaults to 1, `band` is empty for JWST's default filter. The FITS url and
 * the label are derived from ref / tier / band. */

export type PixelOverlaySetting = { ref: string; tier: string; band: string; opacity?: number; hidden?: boolean };

/** A resolved overlay (what the stack and the Layers panel draw). */
export type ImageOverlay = {
  /** `<ref>|<tier>|<band>`. */
  key: string;
  ref: string;
  tier: string;
  band: string;
  label: string;
  url: string;
  opacity: number;
  visible: boolean;
};

/** More FITS layers than this make Aladin (and the URL) unwieldy. */
export const MAX_PIXEL_OVERLAYS = 60;

const REF_RE = /^[a-z][a-z0-9_-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/;
const TIER_RE = /^(?:lr|jwst|m:[^\s|,@!]+)$/;
const BAND_RE = /^[A-Za-z0-9_]*$/;

export const overlayKey = (ref: string, tier: string, band: string) => `${ref}|${tier}|${band}`;
const keyOf = (o: PixelOverlaySetting) => overlayKey(o.ref, o.tier, o.band);

export function overlayUrl(ref: string, tier: string, band: string): string {
  const i = ref.indexOf("/");
  const source = ref.slice(0, i), id = ref.slice(i + 1);
  const p = new URLSearchParams(band ? { tier, band } : { tier });
  return `/api/real/${encodeURIComponent(source)}/${encodeURIComponent(id)}/image.fits?${p.toString()}`;
}

export function tierLabel(tier: string): string {
  if (tier === "lr") return "LR";
  if (tier === "jwst") return "JWST";
  return `SR · ${tier.replace(/^m:/, "")}`;
}

export function overlayLabel(ref: string, tier: string, band: string): string {
  const id = ref.slice(ref.indexOf("/") + 1);
  return `${id} · ${tierLabel(tier)}${band ? ` · ${band.replace(/_E$/, "")}` : ""}`;
}

function clean(o: PixelOverlaySetting): PixelOverlaySetting {
  const out: PixelOverlaySetting = { ref: o.ref, tier: o.tier, band: o.band };
  if (o.opacity != null && o.opacity !== 1) out.opacity = o.opacity;
  if (o.hidden) out.hidden = true;
  return out;
}

function parseEntry(part: string): PixelOverlaySetting | null {
  let s = part.trim();
  const hidden = s.startsWith("!");
  if (hidden) s = s.slice(1);
  let opacity: number | undefined;
  const at = /@([0-9.]+)$/.exec(s);
  if (at) {
    const n = Number(at[1]);
    if (Number.isFinite(n)) opacity = Math.max(0, Math.min(1, n));
    s = s.slice(0, at.index);
  }
  const bits = s.split("|");
  if (bits.length !== 3) return null;
  const [ref, tier, band] = bits;
  if (!REF_RE.test(ref) || !TIER_RE.test(tier) || !BAND_RE.test(band)) return null;
  return { ref, tier, band, ...(opacity != null ? { opacity } : {}), ...(hidden ? { hidden } : {}) };
}

function trimOpacity(v: number): string {
  return String(Number(v.toFixed(2)));
}

export const IMG_CODEC = {
  parse: (raw: string): PixelOverlaySetting[] | undefined => {
    const byKey = new Map<string, PixelOverlaySetting>();
    for (const part of raw.split(",")) {
      const e = parseEntry(part);
      if (e) byKey.set(keyOf(e), e); // a duplicate's settings win; it keeps its first position
    }
    return [...byKey.values()];
  },
  serialize: (list: readonly PixelOverlaySetting[]): string | null => {
    if (!list.length) return null;
    return list.map((raw) => {
      const o = clean(raw);
      return `${o.hidden ? "!" : ""}${keyOf(o)}${o.opacity != null ? `@${trimOpacity(o.opacity)}` : ""}`;
    }).join(",");
  },
};

/** Add overlays: an existing one is shown again (its opacity kept), new ones
 *  are appended; the oldest go beyond MAX_PIXEL_OVERLAYS. */
export function withOverlays(list: readonly PixelOverlaySetting[], adds: readonly PixelOverlaySetting[]): PixelOverlaySetting[] {
  const out = list.map(clean);
  for (const a of adds) {
    const k = keyOf(a);
    const i = out.findIndex((o) => keyOf(o) === k);
    if (i >= 0) out[i] = clean({ ...out[i], hidden: false });
    else out.push(clean(a));
  }
  return out.length > MAX_PIXEL_OVERLAYS ? out.slice(out.length - MAX_PIXEL_OVERLAYS) : out;
}

export function patchPixelOverlay(
  list: readonly PixelOverlaySetting[], key: string, patch: Partial<Pick<PixelOverlaySetting, "opacity" | "hidden">>,
): PixelOverlaySetting[] {
  return list.map((o) => (keyOf(o) === key ? clean({ ...o, ...patch }) : o));
}

export function withoutOverlay(list: readonly PixelOverlaySetting[], key: string): PixelOverlaySetting[] {
  return list.filter((o) => keyOf(o) !== key);
}

export function resolveOverlays(list: readonly PixelOverlaySetting[]): ImageOverlay[] {
  return list.map((o) => ({
    key: keyOf(o), ref: o.ref, tier: o.tier, band: o.band,
    label: overlayLabel(o.ref, o.tier, o.band), url: overlayUrl(o.ref, o.tier, o.band),
    opacity: o.opacity ?? 1, visible: !o.hidden,
  }));
}
