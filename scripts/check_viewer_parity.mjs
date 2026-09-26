/* Viewer colour parity (Node ≥ 22.6; native TypeScript type stripping).
 *
 * 1. Parity check of the SPA colour module against visualization/color.py:
 *
 *      python scripts/_viewer_parity_ref.py | node scripts/check_viewer_parity.mjs
 *
 *    Checks euclid_polish/web/frontend/src/viewer/color.ts: per-band
 *    calibration constants, the Planckian-locus hue chain, and the Temp render
 *    at the default knee/brightness against eye_rgb twice — the unrounded
 *    float chain (prepareCore's I and hue through asinhTransfer, 1e-4, as the
 *    pre-rework script checked it) and the rendered RGBA bytes of transferCore
 *    (2e-3 + half a level). Exits non-zero and prints the worst offender of
 *    each group on drift.
 *
 * 2. Golden emitter (used by `_viewer_parity_ref.py --write-golden`):
 *
 *      node scripts/check_viewer_parity.mjs --emit-engine <old cutout_viewer.js> < request.json
 *
 *    Runs the OLD engine's `_internals` on the request's cases and prints its
 *    outputs as JSON (primitives, prepared frames, RGBA bytes). The vitest suite
 *    src/viewer/color.test.ts holds the TS port to these values.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

// transferCore allocates `new ImageData(w, h)`; Node has no DOM.
if (typeof globalThis.ImageData === "undefined") {
  globalThis.ImageData = class ImageData {
    constructor(width, height) {
      this.width = width; this.height = height;
      this.data = new Uint8ClampedArray(width * height * 4);
    }
  };
}

const decode = (v) => (v === "NaN" ? NaN : v === "Infinity" ? Infinity : v === "-Infinity" ? -Infinity : v);
const encode = (v) => (Number.isNaN(v) ? "NaN" : v === Infinity ? "Infinity" : v === -Infinity ? "-Infinity" : v);
const list = (arr) => Array.from(arr, (v) => encode(Number(v)));

function recOf(c) {
  return {
    key: c.name, ...c.rec,
    data: Float32Array.from(c.data.map(decode)),
  };
}

async function emitEngine(enginePath) {
  const request = JSON.parse(readFileSync(0, "utf8"));
  const M = (await import(pathToFileURL(resolve(enginePath)).href))._internals;
  const meta = request.color_meta;
  const bands = Object.keys(meta.bands);
  const temps = [1000, 1667, 2000, 2222, 2500, 3500, 4000, 4001, 5800, 6500, 8000, 12000, 20000, 25000, 40000];
  const primitives = {
    abFluxNorm: Object.fromEntries(bands.map((n) => [n, M.abFluxNorm(meta.bands[n])])),
    solarBalance: Object.fromEntries(bands.filter((n) => meta.bands[n].solar_ab_mag != null)
      .map((n) => [n, M.solarBalance(meta.bands[n])])),
    planck: temps.map((T) => {
      const xy = M.planckianXY(T);
      const lin = M.xyToLinearSrgb(xy[0], xy[1]);
      return { T, xy, lin, srgb: lin.map(M.srgbGamma) };
    }),
    srgbGamma: [-0.5, 0, 0.001, 0.0031308, 0.004, 0.2, 0.5, 0.99, 1, 1.5].map((c) => [c, M.srgbGamma(c)]),
    eyeTGrid96: Array.from(M.eyeTGrid(96)),
    planckFnu: [2000, 5800, 20000].map((T) => ({ T, lam: [0.715, 1.081, 1.367, 1.771], f: Array.from(M.planckFnu([0.715, 1.081, 1.367, 1.771], T)) })),
    asinhTransfer: [[0, 1, 100, 4], [50, 1, 100, 4], [-5, 1, 100, 4], [1e5, 1, 100, 4], [250, 2.5, 37, 5.1]]
      .map(([I, g, k, n]) => [I, g, k, n, M.asinhTransfer(I, g, k, n)]),
  };
  const cases = request.cases.map((c) => {
    const colorMeta = c.render_mode ? { ...meta, render_mode: c.render_mode } : meta;
    const prep = M.prepareCore(recOf(c), colorMeta, c.color);
    const prepared = { mode: prep.mode, factor: prep.factor, h: prep.h, w: prep.w, npx: prep.npx };
    for (const k of ["I", "R", "G", "B", "hueR", "hueG", "hueB"]) if (prep[k]) prepared[k] = list(prep[k]);
    if (prep.mode === "gray-log") { prepared.logLo = prep.logLo; prepared.logHi = prep.logHi; }
    const renders = c.transfers.map(([knee, gain, K0]) => ({
      knee, gain, K0, rgba: Array.from(M.transferCore(prep, knee, gain, K0).data),
    }));
    return { name: c.name, prepared, renders };
  });
  process.stdout.write(JSON.stringify({ primitives, cases }));
}

async function checkPort() {
  const ref = JSON.parse(readFileSync(0, "utf8"));
  const here = new URL(".", import.meta.url);
  const M = await import(new URL("../euclid_polish/web/frontend/src/viewer/color.ts", here).href);
  const consts = ref.color_meta.bands;
  const TOL = 1e-4;
  let fail = 0;
  // Worst offender per group: the float chain must stay at float precision;
  // only the rounded RGBA bytes carry the half-level rounding allowance.
  const worst = { float: { d: 0, what: "" }, bytes: { d: 0, what: "" } };
  const check = (what, got, exp, tol = TOL, group = "float") => {
    const d = Math.abs(got - exp);
    const w = worst[group];
    if (d > w.d || Number.isNaN(d)) { w.d = d; w.what = what; }
    if (!(d <= tol)) { fail++; if (fail <= 12) console.error(`✗ ${what}: got ${got}, exp ${exp} (Δ${d.toExponential(2)})`); }
  };

  // 1) calibration constants
  for (const [name, v] of Object.entries(ref.bands)) {
    check(`abFluxNorm[${name}]`, M.abFluxNorm(consts[name]) / v.ab_flux_norm, 1.0, 1e-6);
    check(`solarBalance[${name}]`, M.solarBalance(consts[name]) / v.solar_balance, 1.0, 1e-6);
  }
  // 2) Planckian hue chain
  for (const row of ref.planck_chain) {
    const [x, y] = M.planckianXY(row.T);
    const srgb = M.xyToLinearSrgb(x, y).map(M.srgbGamma);
    for (let k = 0; k < 3; k++) check(`planck[${row.T}][${k}]`, srgb[k], row.srgb[k]);
  }
  // 3) the full Temp render at the default knee/brightness vs eye_rgb.
  const cube = ref.eye_rgb.cube;
  const rec = { h: 1, w: cube.length, c: 4, bands: ["VIS", "Y_E", "J_E", "H_E"], data: Float32Array.from(cube.flat()) };
  const prep = M.prepareCore(rec, ref.color_meta, "temp");
  const k0 = ref.eye_rgb.asinh_scale_e;
  // 3a) the unrounded float chain (as the pre-rework script checked it): the
  //     port's prepared luminance I and per-pixel hue through the same
  //     transfer transferCore runs (Kc = knee·factor, norm = asinh(30·K0/knee)).
  const Kc = Math.max(k0 * prep.factor, 1e-30);
  const norm = Math.max(Math.asinh((30.0 * k0 * prep.factor) / Kc), 1e-6);
  const hues = [prep.hueR, prep.hueG, prep.hueB];
  cube.forEach((_, i) => {
    const lum = M.asinhTransfer(prep.I[i], 1.0, Kc, norm);
    for (let k = 0; k < 3; k++) check(`eye-float[${i}][${k}]`, M.srgbGamma(hues[k][i] * lum), ref.eye_rgb.rgb[i][k]);
  });
  // 3b) the rendered RGBA bytes: rounded, so half a level on top of the 2e-3 hue tolerance.
  const img = M.transferCore(prep, k0, 1.0, k0);
  cube.forEach((_, i) => {
    for (let k = 0; k < 3; k++) {
      check(`eye-bytes[${i}][${k}]`, img.data[i * 4 + k] / 255, ref.eye_rgb.rgb[i][k], 2e-3 + 0.5 / 255, "bytes");
    }
  });

  console.log(`worst Δ (float chain) = ${worst.float.d.toExponential(3)} @ ${worst.float.what}`);
  console.log(`worst Δ (RGBA bytes)  = ${worst.bytes.d.toExponential(3)} @ ${worst.bytes.what}`);
  if (fail) { console.error(`PARITY FAIL: ${fail} value(s) over tolerance`); process.exit(1); }
  console.log("✓ parity OK");
}

const i = process.argv.indexOf("--emit-engine");
if (i >= 0) await emitEngine(process.argv[i + 1]);
else await checkPort();
