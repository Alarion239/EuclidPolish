/* Colormaps of the image viewer (C7 `Colormap`): 256-entry RGB lookup tables.
 * viridis, magma, inferno and cividis are matplotlib 3.11's maps and `rdbu`
 * is matplotlib's RdBu_r (negative = blue, zero = white, positive = red — the
 * residual convention), sampled at 33 evenly spaced stops and linearly
 * interpolated; gray is exact. Theme-independent by design (science colour). */
import type { Colormap } from "../state/display";

const STOPS: Record<Exclude<Colormap, "gray">, string> = {
  viridis: "440154470d6048186a482374472d7b4537814240863e49893b528b375b8d33638d2f6b8e2c728e297a8e26828e23898e21918c1f988b1fa08822a78528ae8032b67a3fbc734ec36b5ec96270cf5784d44b98d83eaddc30c2df23d8e219ece51bfde725",
  magma: "0000040303120a0822130d341d114729115a36106b440f7651127c5d177f6a1c81762181832681902a819c2e7faa337db73779c43c75d0416fdc4869e75263ef5d5ef56b5cf9795dfc8961fd9869fea772feb67cfec488fed395fde2a3fcf0b2fcfdbf",
  inferno: "0000040403120b0724150b37210c4a2f0a5b3d09654a0c6b57106e64156e71196e7d1e6d8a226a972766a32c61b0315bbc3754c73e4cd24644db503be45a31eb6628f1731df68013f98e09fb9d07fcac11fbbc21f9cb35f5db4cf2ea69f3f68afcffa4",
  cividis: "00224e00285b002e6a0533711a386f273e6e32436d3b496c434e6c4b546c535a6d5a5f6e61656f686a716f70737676767d7c788482798c8878938e789b9476a39a74aba072b4a76fbcae6cc4b468cdbb63d5c25edec958e7d150f0d846f9e03afee838",
  rdbu: "0530610e41791752901f63a82a71b23480b93f8ec0529dc86bacd184bcd99bc9e0aed3e6c2ddecd4e6f1e0ecf3ecf2f5f7f6f6f9eee7fbe5d8fddcc9fbccb4f8bb9ef5aa89ee9677e48066db6b55d05548c53e3dba2832ab162a930e267c072267001f",
};

function build(hex: string): Uint8ClampedArray {
  const n = hex.length / 6;
  const stops: number[][] = [];
  for (let i = 0; i < n; i++) {
    stops.push([0, 2, 4].map((o) => parseInt(hex.slice(i * 6 + o, i * 6 + o + 2), 16)));
  }
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const x = (i / 255) * (n - 1);
    const j = Math.min(n - 2, Math.floor(x));
    const f = x - j;
    for (let k = 0; k < 3; k++) lut[i * 3 + k] = Math.round(stops[j][k] + (stops[j + 1][k] - stops[j][k]) * f);
  }
  return lut;
}

function grayLut(): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) lut[i * 3] = lut[i * 3 + 1] = lut[i * 3 + 2] = i;
  return lut;
}

export const COLORMAP_LUTS: Record<Colormap, Uint8ClampedArray> = {
  gray: grayLut(),
  viridis: build(STOPS.viridis),
  magma: build(STOPS.magma),
  inferno: build(STOPS.inferno),
  cividis: build(STOPS.cividis),
  rdbu: build(STOPS.rdbu),
};

export function colormapLut(name: Colormap): Uint8ClampedArray {
  return COLORMAP_LUTS[name] ?? COLORMAP_LUTS.gray;
}

/** A CSS gradient of a colormap (legends, the histogram bar). */
export function colormapGradient(name: Colormap, stops = 9): string {
  const lut = colormapLut(name);
  const parts: string[] = [];
  for (let i = 0; i < stops; i++) {
    const k = Math.round((i / (stops - 1)) * 255) * 3;
    parts.push(`rgb(${lut[k]}, ${lut[k + 1]}, ${lut[k + 2]}) ${((i / (stops - 1)) * 100).toFixed(1)}%`);
  }
  return `linear-gradient(90deg, ${parts.join(", ")})`;
}

/** "#rgb" / "#rrggbb" → [r, g, b]; anything else → magenta (the C7 default). */
export function parseCssColor(css: string): [number, number, number] {
  const s = String(css || "").trim();
  let m = /^#([0-9a-f]{6})$/i.exec(s);
  if (m) return [0, 2, 4].map((o) => parseInt(m![1].slice(o, o + 2), 16)) as [number, number, number];
  m = /^#([0-9a-f]{3})$/i.exec(s);
  if (m) return [0, 1, 2].map((o) => parseInt(m![1][o] + m![1][o], 16)) as [number, number, number];
  return [255, 0, 255];
}
