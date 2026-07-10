/* ============================================================================
 * cutout_viewer.js — unified client-side cutout viewer (ES module).
 *
 * One reusable component for every page that shows LR / HR / SR field cutouts.
 * It fetches the raw N-band float cube from /viewer/cube/<collection>/<i> and
 * renders it entirely in the browser, so band / colour / asinh / brightness
 * controls are instant (no server round-trip) and next/prev can prefetch.
 *
 * Mount with:
 *     import { mountCutoutViewer } from ".../cutout_viewer.js";
 *     const v = mountCutoutViewer(rootEl, { collection: "evaluation" });
 *
 * The colour maths are ported verbatim from
 * euclid_polish/visualization/color.py so the in-browser render matches the
 * server renderer. See scripts/check_viewer_parity.mjs for the parity check.
 * ==========================================================================*/

// ----------------------------------------------------------------------------
// Colour maths — ported from visualization/color.py
// ----------------------------------------------------------------------------

const EYE_T_MIN = 1667.0;
const EYE_T_MAX = 25000.0;

/** AB-flux normalisation: e⁻-over-stack → proportional AB flux density.
 *  1 / (electrons of an AB=0 source over the stack), anchored on the served
 *  zeropoint_ab_e_total (= BandConfig.sim_zeropoint_e) — the same single
 *  anchor as euclid_polish/photometry.py; equals the historical
 *  1/(t_total · 10^(0.4·zp_rate)). */
function abFluxNorm(band) {
  return 1.0 / Math.pow(10, 0.4 * band.zeropoint_ab_e_total);
}
/** Solar-balance factor (whitens a G2V SED on top of abFluxNorm). */
function solarBalance(band) {
  return 1.0 / Math.pow(10, -0.4 * band.solar_ab_mag);
}

/** Blackbody f_ν (arbitrary norm) at wavelengths `lam` (μm) for temperature T. */
function planckFnu(lam, T) {
  const out = new Float64Array(lam.length);
  for (let i = 0; i < lam.length; i++) {
    const x = 14387.77 / (lam[i] * T);
    out[i] = Math.pow(1.0 / lam[i], 3) / Math.expm1(x);
  }
  return out;
}

/** CIE 1931 (x, y) chromaticity of a blackbody at T (K). Clamped to locus fit. */
function planckianXY(T) {
  T = Math.min(Math.max(T, EYE_T_MIN), EYE_T_MAX);
  const u = 1e3 / T;
  let x;
  if (T <= 4000.0) {
    x = -0.2661239 * u ** 3 - 0.2343589 * u ** 2 + 0.8776956 * u + 0.179910;
  } else {
    x = -3.0258469 * u ** 3 + 2.1070379 * u ** 2 + 0.2226347 * u + 0.240390;
  }
  let y;
  if (T <= 2222.0) {
    y = -1.1063814 * x ** 3 - 1.34811020 * x ** 2 + 2.18555832 * x - 0.20219683;
  } else if (T <= 4000.0) {
    y = -0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867;
  } else {
    y = 3.0817580 * x ** 3 - 5.87338670 * x ** 2 + 3.75112997 * x - 0.37001483;
  }
  return [x, y];
}

/** (x, y) chromaticity → linear sRGB (D65), normalised so max channel = 1. */
function xyToLinearSrgb(x, y) {
  const Y = 1.0;
  const X = x / y;
  const Z = (1.0 - x - y) / y;
  let r = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z;
  let g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z;
  let b = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z;
  r = Math.max(r, 0); g = Math.max(g, 0); b = Math.max(b, 0);
  const peak = Math.max(r, g, b, 1e-12);
  return [r / peak, g / peak, b / peak];
}

/** Linear-light → sRGB-encoded (standard piecewise transfer). */
function srgbGamma(c) {
  c = Math.min(Math.max(c, 0), 1);
  return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1.0 / 2.4) - 0.055;
}

/** Log-spaced colour-temperature grid over the locus fit's validity. */
function eyeTGrid(n) {
  const out = new Float64Array(n);
  const lo = Math.log(EYE_T_MIN), hi = Math.log(EYE_T_MAX);
  for (let i = 0; i < n; i++) out[i] = Math.exp(lo + (hi - lo) * (i / (n - 1)));
  return out;
}

/** asinh contrast transfer: arcsinh(I·G / Kc) / arcsinh(Wref/Kc), clipped. */
function asinhTransfer(I, gain, Kc, norm) {
  const t = Math.asinh((I * gain) / Kc) / norm;
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

// ----------------------------------------------------------------------------
// Small DOM helper
// ----------------------------------------------------------------------------

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on") && typeof v === "function") {
      node.addEventListener(k.slice(2), v);
    } else if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c) node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return node;
}

// ----------------------------------------------------------------------------
// The viewer
// ----------------------------------------------------------------------------

const COLOR_MODES_EXTRA = [
  { key: "lupton", label: "Lupton", title: "4-band solar-balanced Lupton RGB" },
  { key: "temp", label: "Temp", title: "Per-pixel blackbody-T colour (Planckian locus)" },
];

//: Auto-run ("Run through") cadence — slow enough to read each object.
const PLAY_INTERVAL_MS = 1500;

export function mountCutoutViewer(root, opts = {}) {
  const collection = opts.collection;
  const state = {
    meta: null,
    params: Object.assign({}, opts.params || {}),
    index: opts.initialIndex || 0,
    tiers: [],             // selected tier keys (multi-select), canonical order
    color: "VIS",          // band name | "lupton" | "temp"
    knee: 100,             // K (e⁻)
    gain: 1.0,             // brightness multiplier
    K0: 100,               // knee captured on first load → fixes the white ref
    cubeCache: new Map(),  // "tier:index" → cube record
    prepCache: new Map(),  // "tier:index:color" → colour/intensity decomposition
    shown: new Map(),      // tier → rec currently displayed (for cheap re-render)
    frames: [],            // [{ tier, frame, canvas, ctx, overlay, legendWrap, msg }]
    playTimer: null,
    playMs: PLAY_INTERVAL_MS,   // auto-run cadence (live-tunable via the nav slider)
    hot: false,
    morphAmp: 1.6,              // "morph" tier: amplitude (member-σ) + speed
    morphSpeed: 0.5,
    morphMembers: null,         // CSV of member indices → subset movie (null=all)
  };
  let morphRaf = null;          // requestAnimationFrame id for the "morph" tier

  // --- DOM scaffold --------------------------------------------------------
  root.classList.add("cutout-viewer");
  root.innerHTML = "";
  const toolbar = el("div", { class: "cv-toolbar" });
  const framesRow = el("div", { class: "cv-frames" });
  const nav = el("div", { class: "cv-nav" });
  root.append(toolbar, framesRow, nav);
  // Compact mode (e.g. the morphology hover preview): hide the toolbar/nav
  // chrome so only the image frame shows.
  if (opts.compact) { toolbar.style.display = "none"; nav.style.display = "none"; }

  // --- helpers -------------------------------------------------------------
  const band = (name) => state.meta.color.bands[name];
  // `extra` (e.g. the movie's member subset) belongs in the key so a subset
  // frame never collides with the plain tier's cached cube.
  const cacheKey = (tier, index, extra) =>
    `${tier}:${index}${extra ? ":" + new URLSearchParams(extra).toString() : ""}`;
  /** Selected tiers in the meta's canonical order ([{key,label}]). */
  const orderedSelected = () =>
    ((state.meta && state.meta.tiers) || []).filter((t) => state.tiers.includes(t.key));
  /** Tiers available for the current object (eval gates per object). */
  const tierAvail = (key) => {
    const obj = state.meta && state.meta.objects && state.meta.objects[state.index];
    return !obj || !obj.tiers || obj.tiers.includes(key);
  };
  const tierMeta = (key) =>
    ((state.meta && state.meta.tiers) || []).find((t) => t.key === key);
  /** Not selectable: globally disabled (e.g. SR not generated yet) or, for
   *  per-object collections, missing for the current object. */
  const tierDisabled = (key) => {
    const tm = tierMeta(key);
    return !!(tm && tm.disabled) || !tierAvail(key);
  };

  function notify() {
    if (opts.onIndexChange) opts.onIndexChange(state.index);
    if (opts.onChange) opts.onChange(api.getState());
  }

  async function fetchCube(tier, index, extra) {
    const key = cacheKey(tier, index, extra);
    if (state.cubeCache.has(key)) return state.cubeCache.get(key);
    const qs = new URLSearchParams(Object.assign({ tier }, state.params, extra || {}));
    const r = await fetch(`/viewer/cube/${collection}/${index}?${qs}`);
    if (!r.ok) throw new Error(`cube ${r.status}`);
    const shape = (r.headers.get("X-Cube-Shape") || "").split(",").map(Number);
    const buf = await r.arrayBuffer();
    const amp = parseFloat(r.headers.get("X-Cube-Amp"));
    const rec = {
      key, h: shape[0], w: shape[1], c: shape[2],
      data: new Float32Array(buf),
      label: r.headers.get("X-Cube-Label") || "",
      asinh: parseFloat(r.headers.get("X-Cube-Asinh")) || 100,
      pixscale: parseFloat(r.headers.get("X-Cube-Pixscale")) || 0,
      // PCA eigen-image amplitude/variance (subset-aware) for the movie.
      amp: Number.isFinite(amp) ? amp : null,
      varexp: parseFloat(r.headers.get("X-Cube-Var")) || 0,
    };
    state.cubeCache.set(key, rec);
    if (state.cubeCache.size > 96) state.cubeCache.delete(state.cubeCache.keys().next().value);
    return rec;
  }

  // Warm the cubes for the upcoming indices so next/prev is instant. Crucially
  // this now includes the "morph" tier: the movie's sr + pcaN cubes (subset-
  // aware) are the expensive ones — a subset PCA is a fresh server SVD — so
  // pre-fetching them ahead is what kills the per-switch lag.
  function prefetch(index) {
    const subset = state.morphMembers;
    const extra = subset ? { members: subset } : undefined;
    const nSub = subset ? subset.split(",").filter(Boolean).length : 0;
    const pcaMax = (state.meta && state.meta.pca_max) || 3;
    const nPca = subset ? Math.max(0, Math.min(pcaMax, nSub - 1))
                        : ((state.meta && state.meta.pca_n) | 0);
    const warm = (t, j) => fetchCube(t, j).catch(() => {});
    for (const di of [1, 2, 3, -1]) {
      const j = index + di;
      if (j < 0 || j >= state.meta.count) continue;
      for (const t of state.tiers) {
        if (t === "morph") {
          fetchCube("sr", j, extra).catch(() => {});
          for (let k = 0; k < nPca; k++) fetchCube(`pca${k}`, j, extra).catch(() => {});
        } else if (tierAvail(t)) {
          warm(t, j);
        }
      }
    }
  }

  // --- colour prepare (pure, memoised by cube + colour mode) ---------------
  // Returns { mode, factor, I, ... } with I = per-pixel linear intensity in
  // the unit `factor` converts the e⁻ sliders into. Heavy work (temp fit)
  // happens here; transfer() is cheap on every slider tick.
  function prepare(rec) {
    const key = `${rec.key}:${state.color}`;
    // The morph frame's data changes every tick (rec.noCache) → never cache it.
    if (!rec.noCache && state.prepCache.has(key)) return state.prepCache.get(key);

    const C = state.meta.color;
    const names = C.band_names;
    const npx = rec.h * rec.w;
    const at = (p, k) => rec.data[p * rec.c + k];   // band k at pixel p

    let prepared;
    if (state.color === "lupton" || state.color === "temp") {
      const useSolar = state.color === "lupton";
      const calib = names.map((n, k) => {
        let f = abFluxNorm(band(n));
        if (useSolar) f *= solarBalance(band(n));
        return f;
      });
      if (state.color === "lupton") {
        const sel = C.rgb_scheme.map((n) => names.indexOf(n)); // [H_E,J_E,VIS]
        const R = new Float32Array(npx), G = new Float32Array(npx), B = new Float32Array(npx);
        const I = new Float32Array(npx);
        for (let p = 0; p < npx; p++) {
          const r = at(p, sel[0]) * calib[sel[0]];
          const g = at(p, sel[1]) * calib[sel[1]];
          const b = at(p, sel[2]) * calib[sel[2]];
          R[p] = r; G[p] = g; B[p] = b; I[p] = (r + g + b) / 3.0;
        }
        const visB = band("VIS");
        prepared = { mode: "lupton", R, G, B, I,
                     factor: abFluxNorm(visB) * solarBalance(visB) };
      } else {
        const lam = names.map((n) => band(n).pivot_um);
        const ts = eyeTGrid(96);
        const pvecs = [];
        for (const T of ts) {
          const p = planckFnu(lam, T);
          let nrm = 0; for (const v of p) nrm += v * v; nrm = Math.sqrt(nrm);
          pvecs.push(p.map((v) => v / nrm));
        }
        const hueR = new Float32Array(npx), hueG = new Float32Array(npx), hueB = new Float32Array(npx);
        const I = new Float32Array(npx);
        const cal = new Float64Array(names.length);
        for (let p = 0; p < npx; p++) {
          let sum = 0;
          for (let k = 0; k < names.length; k++) { cal[k] = at(p, k) * calib[k]; sum += cal[k]; }
          I[p] = Math.max(sum / names.length, 0);
          let bestScore = -Infinity, bestT = 6500.0;
          for (let ti = 0; ti < ts.length; ti++) {
            const pv = pvecs[ti];
            let s = 0; for (let k = 0; k < names.length; k++) s += cal[k] * pv[k];
            if (s > bestScore) { bestScore = s; bestT = ts[ti]; }
          }
          const T = bestScore > 0 ? bestT : 6500.0;
          const [x, y] = planckianXY(T);
          const [hr, hg, hb] = xyToLinearSrgb(x, y);
          hueR[p] = hr; hueG[p] = hg; hueB[p] = hb;
        }
        const visB = band("VIS");
        prepared = { mode: "temp", hueR, hueG, hueB, I, factor: abFluxNorm(visB) };
      }
    } else {
      const k = Math.max(0, names.indexOf(state.color));
      const I = new Float32Array(npx);
      for (let p = 0; p < npx; p++) I[p] = at(p, k);
      prepared = { mode: "gray", I, factor: 1.0 };
    }
    prepared.h = rec.h; prepared.w = rec.w; prepared.npx = npx;
    if (!rec.noCache) {
      state.prepCache.set(key, prepared);
      if (state.prepCache.size > 48) state.prepCache.delete(state.prepCache.keys().next().value);
    }
    return prepared;
  }

  // --- transfer (per slider tick) → one frame's canvas ---------------------
  function renderInto(fr, rec) {
    const prep = prepare(rec);
    const { h, w, npx, I, factor } = prep;
    // Kc = knee·factor; white reference Wref = 30·K0·factor (e⁻). norm cancels
    // factor → asinh(30·K0/knee); at knee=K0, gain=1 this matches eye_rgb.
    const Kc = Math.max(state.knee * factor, 1e-30);
    const norm = Math.max(Math.asinh((30.0 * state.K0 * factor) / Kc), 1e-6);
    const G = state.gain;
    const canvas = fr.canvas;
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const img = fr.ctx.createImageData(w, h);
    const out = img.data;

    if (prep.mode === "gray") {
      for (let p = 0; p < npx; p++) {
        const v = (asinhTransfer(I[p], G, Kc, norm) * 255) | 0;
        const o = p * 4; out[o] = v; out[o + 1] = v; out[o + 2] = v; out[o + 3] = 255;
      }
    } else if (prep.mode === "lupton") {
      const { R, G: GG, B } = prep;
      for (let p = 0; p < npx; p++) {
        const t = asinhTransfer(I[p], G, Kc, norm);
        const rescale = I[p] > 1e-30 ? t / I[p] : 0;
        const o = p * 4;
        out[o] = Math.min(Math.max(R[p] * rescale, 0), 1) * 255;
        out[o + 1] = Math.min(Math.max(GG[p] * rescale, 0), 1) * 255;
        out[o + 2] = Math.min(Math.max(B[p] * rescale, 0), 1) * 255;
        out[o + 3] = 255;
      }
    } else { // temp
      const { hueR, hueG, hueB } = prep;
      for (let p = 0; p < npx; p++) {
        const lum = asinhTransfer(I[p], G, Kc, norm);
        const o = p * 4;
        out[o] = srgbGamma(hueR[p] * lum) * 255;
        out[o + 1] = srgbGamma(hueG[p] * lum) * 255;
        out[o + 2] = srgbGamma(hueB[p] * lum) * 255;
        out[o + 3] = 255;
      }
    }
    fr.ctx.putImageData(img, 0, 0);
    fr.overlay.textContent = rec.label + magLabel(rec);
    fr.legendWrap.style.display = prep.mode === "temp" ? "flex" : "none";
    setFrameMsg(fr, "");
    showPlens(fr);
    addSigmaToSR(fr, rec);
  }

  /** Integrated flux of the cube in the shown band (colour composites fall
   *  back to band 0) and its approximate AB magnitude,
   *  mag = zeropoint_ab_e_total − 2.5·log₁₀(Σe⁻), where the stack zeropoint
   *  is served precomputed (BandConfig.sim_zeropoint_e — the same anchor as
   *  euclid_polish/photometry.py; never re-derive it here). Sums cached per
   *  cube+band; the morph tier (data mutates every animation tick) is
   *  excluded. */
  function magInfo(rec) {
    const bands = state.meta && state.meta.color && state.meta.color.bands;
    if (!bands || rec.noCache || !rec.data) return null;
    const names = state.meta.band_names || [];
    const bi = Math.max(0, names.indexOf(state.color));   // composite → band 0
    const c = rec.c || 1;
    const idx = Math.min(bi, c - 1);
    if (!rec._sums) rec._sums = {};
    if (!(idx in rec._sums)) {
      let s = 0;
      const d = rec.data;
      for (let i = idx; i < d.length; i += c) s += d[i];
      rec._sums[idx] = s;
    }
    const tot = rec._sums[idx];
    const name = names[Math.min(idx, names.length - 1)] || "VIS";
    const b = bands[name];
    const mag = (b && tot > 0 && b.zeropoint_ab_e_total)
      ? b.zeropoint_ab_e_total - 2.5 * Math.log10(tot) : null;
    return { name, tot, mag };
  }

  /** " · VIS 21.43 AB" — on stdSR that is the magnitude of the ONE-SIGMA
   *  ensemble disagreement (how bright the hallucinated flux is). */
  function magLabel(rec) {
    const mi = magInfo(rec);
    return mi && mi.mag != null ? ` · ${mi.name} ${mi.mag.toFixed(2)} AB` : "";
  }

  /** On the ensemble-mean SR frame, extend the magnitude with a ± from the
   *  stdSR cube: δm = 1.0857·(Σσ/ΣSR) — the one-sigma disagreement expressed
   *  as a magnitude error. The std cube is fetched (cached) on demand; stale
   *  responses (index/band/frame moved on) are dropped. */
  async function addSigmaToSR(fr, rec) {
    if (fr.tier.toLowerCase() !== "sr") return;
    if (!(state.meta.tiers || []).some((t) => t.key === "std")) return;
    const mi = magInfo(rec);
    if (!mi || mi.mag == null) return;
    const idx = state.index, color0 = state.color;
    let std;
    try { std = await fetchCube("std", idx); } catch { return; }
    if (state.index !== idx || state.color !== color0
        || state.shown.get(fr.tier) !== rec) return;
    const si = magInfo(std);
    if (!si || !(si.tot > 0)) return;
    const dm = (2.5 / Math.LN10) * (si.tot / mi.tot);
    fr.overlay.textContent =
      `${rec.label} · ${mi.name} ${mi.mag.toFixed(2)} ± ${dm.toFixed(2)} AB`;
  }

  /** Show the headed lens-finder's P(lens) for this frame's tier (eval only). */
  function showPlens(fr) {
    const obj = state.meta && state.meta.objects && state.meta.objects[state.index];
    const p = obj && obj.plens ? obj.plens[fr.tier] : undefined;
    if (p == null || !isFinite(p)) { fr.plens.style.display = "none"; return; }
    const t = Math.max(0, Math.min(1, p));
    const hue = Math.round(120 * t);                 // 0 → red (galaxy), 120 → green (lens)
    fr.plens.textContent = `P(lens) ${p.toFixed(2)}`;
    fr.plens.style.borderLeftColor = `hsl(${hue}, 65%, 52%)`;
    fr.plens.style.color = `hsl(${hue}, 72%, 82%)`;
    fr.plens.style.display = "block";
  }

  function setFrameMsg(fr, text) {
    fr.msg.textContent = text || "";
    fr.msg.style.display = text ? "flex" : "none";
    if (text) {
      fr.overlay.textContent = "";
      fr.legendWrap.style.display = "none";
      fr.plens.style.display = "none";
    }
  }

  function makeFrame(tier) {
    const canvas = el("canvas", { class: "cv-canvas", width: 1, height: 1 });
    const legendCanvas = el("canvas", { class: "cv-legend", width: 14, height: 160 });
    const legendWrap = el("div", { class: "cv-legend-wrap" }, [
      el("span", { class: "cv-legend-tick", text: "20k" }),
      legendCanvas,
      el("span", { class: "cv-legend-tick", text: "3k" }),
    ]);
    const overlay = el("div", { class: "cv-overlay" });
    const plens = el("div", { class: "cv-plens" });   // headed-model P(lens) for this tier
    const msg = el("div", { class: "cv-msg" });
    const frame = el("div", { class: "cv-frame" }, [canvas, legendWrap, overlay, plens, msg]);
    const fr = { tier, frame, canvas, ctx: canvas.getContext("2d"),
                 legendWrap, legendCanvas, overlay, plens, msg };
    drawLegend(fr);
    return fr;
  }

  function drawLegend(fr) {
    const n = 160, ctx = fr.legendCanvas.getContext("2d");
    const img = ctx.createImageData(14, n);
    const lo = Math.log(3000), hi = Math.log(20000);
    for (let row = 0; row < n; row++) {
      const T = Math.exp(hi - (hi - lo) * (row / (n - 1)));   // top hot, bottom cool
      const [x, y] = planckianXY(T);
      const [r, g, b] = xyToLinearSrgb(x, y);
      for (let col = 0; col < 14; col++) {
        const o = (row * 14 + col) * 4;
        img.data[o] = srgbGamma(r) * 255;
        img.data[o + 1] = srgbGamma(g) * 255;
        img.data[o + 2] = srgbGamma(b) * 255;
        img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  /** Rebuild the frame row to match the selected tiers (canonical order). */
  function rebuildFrames() {
    const want = orderedSelected().map((t) => t.key);
    const have = state.frames.map((f) => f.tier);
    if (want.length === have.length && want.every((t, i) => t === have[i])) return;
    stopMorph();               // frames are being recreated → drop the old loop
    framesRow.innerHTML = "";
    state.frames = want.map((t) => {
      const fr = makeFrame(t);
      framesRow.appendChild(fr.frame);
      return fr;
    });
    framesRow.classList.toggle("cv-multi", state.frames.length > 1);
  }

  // --- "morph" tier: an animated frame in the row (ensemble disagreement) ---
  function stopMorph() {
    if (morphRaf != null) cancelAnimationFrame(morphRaf);
    morphRaf = null;
  }

  // Drive one frame with mean + Σ aᵢ·sin(2π fᵢ t)·componentᵢ, rendered through
  // the SAME colour pipeline as the static frames (so band/colour/asinh match).
  async function startMorph(fr, index) {
    stopMorph();
    // Member subset (null → all): the sr/pcaN cubes are then recomputed server-
    // side over just those members, so their mean + amplitudes are subset-aware.
    const subset = state.morphMembers;
    const extra = subset ? { members: subset } : undefined;
    const nSub = subset ? subset.split(",").filter(Boolean).length : 0;
    // Kept PCs: min(pca_max, |subset|−1) for a subset, else the baked pca_n.
    const pcaMax = (state.meta && state.meta.pca_max) || 3;
    const n = subset ? Math.max(0, Math.min(pcaMax, nSub - 1))
                     : ((state.meta && state.meta.pca_n) | 0);
    let sr;
    const comps = [];
    try {
      sr = await fetchCube("sr", index, extra);
      for (let k = 0; k < n; k++) {
        try { comps.push(await fetchCube(`pca${k}`, index, extra)); }
        catch { /* fewer components than requested */ }
      }
    } catch { setFrameMsg(fr, "movie unavailable"); return; }
    fr.frame.classList.remove("cv-loading");
    // Amplitudes/variance now ride on each PC's cube header (subset-aware),
    // falling back to the baked per-field manifest for the full-ensemble movie.
    const metaAmps = (state.meta.pca_amps && state.meta.pca_amps[index]) || [];
    const amps = comps.map((c, k) => (c.amp != null ? c.amp : (metaAmps[k] || 0)));
    const varTot = comps.reduce((a, c) => a + (c.varexp || 0), 0);
    const subLbl = subset ? ` · ${nSub} members` : "";
    const varLbl = varTot > 0
      ? ` · ${comps.length} PCs ≈ ${(varTot * 100).toFixed(0)}% of variance`
      : "";
    const len = sr.data.length;
    const data = new Float32Array(len);
    const rec = { key: `morph:${index}`, h: sr.h, w: sr.w, c: sr.c, data,
                  label: `disagreement movie${subLbl}${varLbl}`, asinh: sr.asinh,
                  noCache: true };
    const FRQ = [1, 2, 3], PH = [0, Math.PI / 2, Math.PI / 3];
    let phase = 0, last = performance.now();          // accumulate phase so a
    const tick = (now) => {                            // speed change never jumps
      phase += ((now - last) / 1000) * state.morphSpeed;
      last = now;
      const amp = state.morphAmp;                     // live from the sliders
      data.set(sr.data);                             // mean
      for (let k = 0; k < comps.length; k++) {
        const ck = (amps[k] || 0) * amp
          * Math.sin(2 * Math.PI * FRQ[k % 3] * phase + PH[k % 3]);
        const cd = comps[k].data;
        for (let i = 0; i < len; i++) data[i] += ck * cd[i];
      }
      renderInto(fr, rec);
      morphRaf = requestAnimationFrame(tick);
    };
    morphRaf = requestAnimationFrame(tick);
  }

  // --- load + show current index across all selected tiers -----------------
  async function show() {
    if (!state.meta || state.meta.count === 0) {
      rebuildFrames();
      for (const fr of state.frames) setFrameMsg(fr, "No cutouts available.");
      updateNav();
      return;
    }
    state.index = Math.max(0, Math.min(state.index, state.meta.count - 1));
    stopMorph();
    rebuildFrames();
    state.shown.clear();
    await Promise.all(state.frames.map(async (fr) => {
      if (!tierAvail(fr.tier)) { setFrameMsg(fr, `no ${fr.tier} for this object`); return; }
      if (fr.tier === "morph") { await startMorph(fr, state.index); return; }
      fr.frame.classList.add("cv-loading");
      try {
        const rec = await fetchCube(fr.tier, state.index);
        state.shown.set(fr.tier, rec);
        renderInto(fr, rec);
      } catch (e) {
        setFrameMsg(fr, tierAvail(fr.tier) ? "not synced yet" : `no ${fr.tier}`);
      } finally {
        fr.frame.classList.remove("cv-loading");
      }
    }));
    updateNav();
    prefetch(state.index);
    notify();
  }

  // Re-render the already-loaded cubes (slider ticks — no fetch).
  function rerender() {
    for (const fr of state.frames) {
      const rec = state.shown.get(fr.tier);
      if (rec) renderInto(fr, rec);
    }
  }

  // --- toolbar -------------------------------------------------------------
  function chip(group, value, label, title) {
    return el("button", {
      class: "cv-chip", "data-group": group, "data-value": value,
      title: title || "", type: "button",
      onclick: () => onChip(group, value),
    }, label);
  }

  function syncChips() {
    toolbar.querySelectorAll(".cv-chip[data-group]").forEach((c) => {
      const g = c.dataset.group;
      let on;
      if (g === "tier") on = state.tiers.includes(c.dataset.value);
      else if (g === "color") on = c.dataset.value === state.color;
      else on = c.dataset.value === String(state.params[g]);
      c.classList.toggle("active", on);
      if (g === "tier") c.classList.toggle("cv-disabled", tierDisabled(c.dataset.value));
    });
    if (toolbar._morphGroup)
      toolbar._morphGroup.style.display = state.tiers.includes("morph") ? "" : "none";
  }

  function onChip(group, value) {
    if (group === "tier") {
      // Multi-select: toggle membership, keep at least one tier selected.
      // A disabled tier (e.g. SR before it's generated) can't be added.
      if (!state.tiers.includes(value) && tierDisabled(value)) return;
      const set = new Set(state.tiers);
      if (set.has(value)) { if (set.size > 1) set.delete(value); }
      else set.add(value);
      state.tiers = (state.meta.tiers || [])
        .map((t) => t.key).filter((k) => set.has(k));
      syncChips();
      show();
    } else if (group === "color") {
      state.color = value;
      syncChips();
      rerender();
      notify();
    } else {                                  // param control (e.g. sky subset)
      state.params[group] = value;
      syncChips();
      loadMeta().then(show);
    }
  }

  function slider(label, min, max, val, fmt, onInput) {
    const input = el("input", { type: "range", min: 0, max: 1000, value: 0, class: "cv-range" });
    const out = el("span", { class: "cv-val" });
    const toSlider = (v) => Math.round(1000 * (Math.log(v) - Math.log(min)) / (Math.log(max) - Math.log(min)));
    const fromSlider = (s) => Math.exp(Math.log(min) + (s / 1000) * (Math.log(max) - Math.log(min)));
    input.value = toSlider(val);
    out.textContent = fmt(val);
    input.addEventListener("input", () => {
      const v = fromSlider(+input.value);
      out.textContent = fmt(v);
      onInput(v);
    });
    return el("div", { class: "cv-slider" }, [el("label", { text: label }), input, out]);
  }

  function buildToolbar() {
    toolbar.innerHTML = "";
    const tiers = state.meta.tiers || [];
    // `hidden` tiers (e.g. the 22 individual member SRs) stay loadable via
    // setTiers but are kept out of the chip row — an external panel drives them.
    const shown = tiers.filter((t) => !t.hidden);
    if (shown.length > 1) {
      toolbar.append(el("span", { class: "cv-grouplabel", text: "Tier" }));
      const g = el("div", { class: "cv-group" });
      shown.forEach((t) => g.append(chip("tier", t.key, t.label,
        "toggle — select more than one to compare side by side")));
      toolbar.append(g);
    }
    for (const pc of (opts.paramControls || [])) {
      toolbar.append(el("span", { class: "cv-grouplabel", text: pc.label }));
      const g = el("div", { class: "cv-group" });
      pc.options.forEach((o) => g.append(chip(pc.key, o.value, o.label)));
      toolbar.append(g);
    }
    toolbar.append(el("span", { class: "cv-grouplabel", text: "Colour" }));
    const cg = el("div", { class: "cv-group" });
    state.meta.band_names.forEach((n) => cg.append(chip("color", n, n)));
    COLOR_MODES_EXTRA.forEach((m) => cg.append(chip("color", m.key, m.label, m.title)));
    toolbar.append(cg);

    const sg = el("div", { class: "cv-group cv-sliders" });
    sg.append(
      slider("asinh knee", 5, 5000, state.knee,
        (v) => `${Math.round(v)} e⁻`, (v) => { state.knee = v; rerender(); }),
      slider("brightness", 0.1, 10, state.gain,
        (v) => `${v.toFixed(2)}×`, (v) => { state.gain = v; rerender(); }),
    );
    toolbar.append(sg);

    // "morph" tier controls — amplitude + speed (live; the rAF loop reads them).
    // Shown only while the disagreement-movie tier is selected (see syncChips).
    if ((state.meta.tiers || []).some((t) => t.key === "morph")) {
      const mg = el("div", { class: "cv-group cv-sliders cv-morph-ctl", style: "display:none;" });
      mg.append(
        slider("morph amplitude", 0.05, 3, state.morphAmp,
          (v) => `${v.toFixed(2)}σ`, (v) => { state.morphAmp = v; }),
        slider("morph speed", 0.1, 2, state.morphSpeed,
          (v) => `${v.toFixed(2)}×`, (v) => { state.morphSpeed = v; }),
      );
      toolbar.append(mg);
      toolbar._morphGroup = mg;
    } else {
      toolbar._morphGroup = null;
    }
  }

  // --- nav (prev / editable index / next / play) ---------------------------
  let idxInput, idxTotal;
  function buildNav() {
    nav.innerHTML = "";
    const prev = el("button", { class: "cv-navbtn", type: "button", text: "◀", title: "Previous (←)",
      onclick: () => go(state.index - 1) });
    const next = el("button", { class: "cv-navbtn", type: "button", text: "▶", title: "Next (→)",
      onclick: () => go(state.index + 1) });
    const play = el("button", { class: "cv-navbtn cv-play", type: "button", text: "▶▶", title: "Run through (Space)",
      onclick: togglePlay });
    nav._play = play;
    idxInput = el("input", { class: "cv-idx-input", type: "text", inputmode: "numeric",
      title: "type an index and press Enter to jump" });
    idxTotal = el("span", { class: "cv-idx-total" });
    const commit = () => {
      const v = parseInt(idxInput.value, 10);
      if (Number.isFinite(v)) go(v, false);     // explicit jump → clamp, don't wrap
      else updateNav();
    };
    idxInput.addEventListener("change", commit);
    idxInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { commit(); idxInput.blur(); }
    });
    const idxWrap = el("span", { class: "cv-idx" }, [idxInput, idxTotal]);
    const speed = slider("speed", 0.3, 3.0, state.playMs / 1000,
      (v) => `${v.toFixed(1)} s`, (v) => setPlaySpeed(v * 1000));
    speed.classList.add("cv-speed");
    speed.title = "auto-run cadence — seconds per slide";
    const hint = el("span", { class: "cv-kbd", text: "← →  ·  Space to run" });
    const png = el("button", { class: "cv-navbtn", type: "button", text: "⬇ PNG",
      title: "Save the current view (all selected tiers, side by side) as a PNG",
      onclick: savePNG });
    const rec = el("button", { class: "cv-navbtn cv-rec", type: "button", text: "⏺ video",
      title: "Record the current view (all selected tiers, side by side) — click to start, click again to stop and download a .webm clip",
      onclick: () => toggleRecord(rec) });
    nav.append(prev, idxWrap, next, play, speed, hint, png, rec);
  }

  // --- save PNG / record video ----------------------------------------------
  function _download(blob, name) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  }

  function _stem() {
    return `${collection || "cutout"}_idx${state.index}_`
      + `${state.tiers.join("-") || "view"}_${state.color}`;
  }

  function exportFrames() {
    return state.frames.map((fr) => {
      const rect = fr.frame.getBoundingClientRect();
      return { fr, rect };
    }).filter(({ fr, rect }) => fr.canvas.width > 1 && rect.width > 1 && rect.height > 1);
  }

  function drawLabel(ctx, text, x, y, maxW, opts = {}) {
    if (!text) return;
    ctx.save();
    ctx.font = opts.font || '11px "JetBrains Mono", Menlo, monospace';
    const padX = opts.padX || 8;
    const padY = opts.padY || 4;
    const lineH = opts.lineH || 15;
    let label = text;
    const limit = Math.max(20, maxW - 18);
    while (label.length > 1 && ctx.measureText(label).width + 2 * padX > limit) {
      label = `${label.slice(0, -2)}…`;
    }
    const w = Math.min(limit, ctx.measureText(label).width + 2 * padX);
    ctx.fillStyle = opts.background || "rgba(6, 9, 16, 0.68)";
    ctx.fillRect(x, y, w, lineH);
    ctx.fillStyle = opts.color || "#cdd6e6";
    ctx.fillText(label, x + padX, y + lineH - padY);
    ctx.restore();
  }

  function compositeVisibleFrames(target) {
    const frames = exportFrames();
    if (!frames.length) return null;
    const left = Math.min(...frames.map(({ rect }) => rect.left));
    const top = Math.min(...frames.map(({ rect }) => rect.top));
    const right = Math.max(...frames.map(({ rect }) => rect.right));
    const bottom = Math.max(...frames.map(({ rect }) => rect.bottom));
    const cssW = Math.max(1, right - left);
    const cssH = Math.max(1, bottom - top);
    const scale = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
    const out = target || document.createElement("canvas");
    const pxW = Math.max(1, Math.round(cssW * scale));
    const pxH = Math.max(1, Math.round(cssH * scale));
    if (out.width !== pxW || out.height !== pxH) {
      out.width = pxW;
      out.height = pxH;
    }
    const ctx = out.getContext("2d");
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.fillStyle = "#05070d";
    ctx.fillRect(0, 0, cssW, cssH);
    for (const { fr, rect } of frames) {
      const x = rect.left - left;
      const y = rect.top - top;
      ctx.fillStyle = "#05070d";
      ctx.fillRect(x, y, rect.width, rect.height);
      ctx.drawImage(fr.canvas, x, y, rect.width, rect.height);
      drawLabel(ctx, fr.overlay.textContent, x + 9, y + 8, rect.width);
      if (fr.plens.style.display !== "none") {
        drawLabel(ctx, fr.plens.textContent, x + 9, y + rect.height - 27,
          rect.width, {
            font: '600 12px "JetBrains Mono", Menlo, monospace',
            background: "rgba(6, 9, 16, 0.74)",
            color: "#e6ecf7",
          });
      }
      if (fr.msg.style.display !== "none") {
        drawLabel(ctx, fr.msg.textContent, x + 20, y + rect.height / 2 - 8,
          rect.width - 40, { background: "rgba(6, 9, 16, 0.82)" });
      }
    }
    return out;
  }

  function savePNG() {
    const out = compositeVisibleFrames();
    if (!out) return;
    out.toBlob((b) => { if (b) _download(b, `${_stem()}.png`); }, "image/png");
  }

  let recorder = null;
  let recordRaf = null;
  function stopRecordPainter() {
    if (recordRaf != null) cancelAnimationFrame(recordRaf);
    recordRaf = null;
  }

  function toggleRecord(btn) {
    if (recorder) { recorder.stop(); return; }
    if (!exportFrames().length || typeof MediaRecorder === "undefined") return;
    const out = document.createElement("canvas");
    const paint = () => {
      compositeVisibleFrames(out);
      recordRaf = requestAnimationFrame(paint);
    };
    paint();
    const stream = out.captureStream(30);
    const mime = ["video/webm;codecs=vp9", "video/webm"]
      .find((m) => MediaRecorder.isTypeSupported(m));
    const chunks = [];
    recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    recorder.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
    recorder.onstop = () => {
      stopRecordPainter();
      stream.getTracks().forEach((track) => track.stop());
      _download(new Blob(chunks, { type: "video/webm" }), `${_stem()}.webm`);
      recorder = null;
      btn.textContent = "⏺ video";
      btn.classList.remove("cv-recording");
    };
    recorder.start();
    btn.textContent = "⏹ stop";
    btn.classList.add("cv-recording");
  }

  function updateNav() {
    if (!idxInput) return;
    if (document.activeElement !== idxInput) idxInput.value = state.index;
    idxTotal.textContent = ` / ${Math.max(0, state.meta.count - 1)}`;
  }

  function go(i, wrap = true) {
    const n = state.meta.count;
    if (n === 0) return;
    state.index = wrap ? ((i % n) + n) % n : Math.max(0, Math.min(i, n - 1));
    show();
    syncChips();
  }

  function togglePlay() {
    if (state.playTimer) {
      clearInterval(state.playTimer); state.playTimer = null;
      nav._play.classList.remove("active");
    } else {
      nav._play.classList.add("active");
      state.playTimer = setInterval(() => go(state.index + 1), state.playMs);
    }
  }

  /** Set the auto-run cadence; restart a running timer so it takes effect now. */
  function setPlaySpeed(ms) {
    state.playMs = ms;
    if (state.playTimer) {
      clearInterval(state.playTimer);
      state.playTimer = setInterval(() => go(state.index + 1), state.playMs);
    }
  }

  function onKey(e) {
    if (!state.hot) return;     // only when the viewer is hovered/focused
    if (e.target.matches("input, textarea, select")) return;
    if (e.key === "ArrowLeft") { go(state.index - 1); e.preventDefault(); }
    else if (e.key === "ArrowRight") { go(state.index + 1); e.preventDefault(); }
    else if (e.key === " ") { togglePlay(); e.preventDefault(); }
  }
  root.addEventListener("mouseenter", () => { state.hot = true; });
  root.addEventListener("mouseleave", () => { state.hot = false; });
  root.addEventListener("focusin", () => { state.hot = true; });

  // --- meta + lifecycle ----------------------------------------------------
  async function loadMeta() {
    const qs = new URLSearchParams(state.params);
    const r = await fetch(`/viewer/meta/${collection}?${qs}`);
    if (!r.ok) throw new Error(`meta ${r.status}`);
    state.meta = await r.json();
    const tierKeys = (state.meta.tiers || []).map((t) => t.key);
    // Keep only still-present, still-enabled tiers; never auto-select a
    // disabled one (e.g. SR before it's been generated).
    state.tiers = state.tiers.filter((t) => tierKeys.includes(t) && !tierDisabled(t));
    if (!state.tiers.length) {
      const def = state.meta.default_tier;
      const firstEnabled = (state.meta.tiers || [])
        .find((t) => !tierDisabled(t.key));
      state.tiers = (tierKeys.includes(def) && !tierDisabled(def)) ? [def]
        : (firstEnabled ? [firstEnabled.key] : tierKeys.slice(0, 1));
    }
    if (!state.meta.band_names.includes(state.color)
        && !["lupton", "temp"].includes(state.color)) {
      state.color = state.meta.band_names[0];
    }
    state.knee = state.meta.color.default_asinh || 100;
    state.K0 = state.knee;
    state.cubeCache.clear();
    state.prepCache.clear();
    buildToolbar();
    buildNav();
    syncChips();
  }

  document.addEventListener("keydown", onKey);

  const api = {
    goTo(i) { go(i, false); },
    /** Programmatically set the selected tier(s), mirroring a chip click:
     *  keep only tiers that exist and aren't disabled, then re-render. */
    setTiers(keys) {
      if (!state.meta) return;
      const wanted = (Array.isArray(keys) ? keys : [keys])
        .filter((k) => !tierDisabled(k));
      const next = (state.meta.tiers || [])
        .map((t) => t.key).filter((k) => wanted.includes(k));
      if (!next.length) return;              // nothing valid → leave as-is
      state.tiers = next;
      syncChips();
      show();
    },
    /** Set the disagreement movie's member subset — a CSV of member indices
     *  (e.g. "0,3,7"), or null/"" for the full ensemble. The sr/pcaN cubes are
     *  recomputed server-side over the subset. Re-arms the movie if shown. */
    setMorphMembers(csv) {
      const v = (csv == null || csv === "") ? null : String(csv);
      if (v === state.morphMembers) return;
      state.morphMembers = v;
      if (state.tiers.includes("morph")) show();
    },
    getIndex() { return state.index; },
    isReady() { return !!state.meta; },
    getState() {
      return { index: state.index, tier: state.tiers[0],
               tiers: state.tiers.slice(), color: state.color,
               params: Object.assign({}, state.params) };
    },
    reload() { return loadMeta().then(show); },
    destroy() {
      document.removeEventListener("keydown", onKey);
      if (state.playTimer) clearInterval(state.playTimer);
      root.innerHTML = "";
    },
  };

  // initial load — a plain placeholder until meta arrives, then show()
  // builds the real per-tier frames.
  function placeholder(text) {
    framesRow.innerHTML = "";
    state.frames = [];
    const m = el("div", { class: "cv-msg", text }); m.style.display = "flex";
    framesRow.appendChild(el("div", { class: "cv-frame" }, [m]));
  }
  placeholder("Loading…");
  loadMeta().then(show).catch((e) => placeholder(`Could not load viewer (${e.message}).`));
  return api;
}

/** Pure colour primitives, exported for the parity check (see
 *  scripts/check_viewer_parity.mjs). Not part of the public viewer API. */
export const _internals = {
  abFluxNorm, solarBalance, planckFnu, planckianXY,
  xyToLinearSrgb, srgbGamma, eyeTGrid, asinhTransfer,
};
