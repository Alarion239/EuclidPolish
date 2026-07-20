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
// Colour render core (pure) — the two halves of turning an N-band cube into
// display pixels, factored out of the viewer's closure so OTHER surfaces (e.g.
// the ensemble back-trace stamps) render byte-identically to the field viewer.
//   prepareCore : cube + colour mode → { mode, factor, I, … }  (EXPENSIVE:
//                 per-pixel temp fit for "temp", band composite for "lupton")
//   transferCore: prepared + knee/gain/K0 → ImageData          (CHEAP: the
//                 asinh brightness transfer, re-run on every slider tick)
// `colorMeta` is the served meta's `color` block (band_names, bands, rgb_scheme).
// ----------------------------------------------------------------------------

function prepareCore(rec, colorMeta, color) {
  const names = colorMeta.band_names;
  const bandOf = (n) => colorMeta.bands[n];
  const npx = rec.h * rec.w;
  const at = (p, k) => rec.data[p * rec.c + k];   // band k at pixel p

  let prepared;
  if (color === "lupton" || color === "temp") {
    const useSolar = color === "lupton";
    const calib = names.map((n) => {
      let f = abFluxNorm(bandOf(n));
      if (useSolar) f *= solarBalance(bandOf(n));
      return f;
    });
    if (color === "lupton") {
      const sel = colorMeta.rgb_scheme.map((n) => names.indexOf(n)); // [H_E,J_E,VIS]
      const R = new Float32Array(npx), G = new Float32Array(npx), B = new Float32Array(npx);
      const I = new Float32Array(npx);
      for (let p = 0; p < npx; p++) {
        const r = at(p, sel[0]) * calib[sel[0]];
        const g = at(p, sel[1]) * calib[sel[1]];
        const b = at(p, sel[2]) * calib[sel[2]];
        R[p] = r; G[p] = g; B[p] = b; I[p] = (r + g + b) / 3.0;
      }
      const visB = bandOf("VIS");
      prepared = { mode: "lupton", R, G, B, I,
                   factor: abFluxNorm(visB) * solarBalance(visB) };
    } else {
      const lam = names.map((n) => bandOf(n).pivot_um);
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
      const visB = bandOf("VIS");
      prepared = { mode: "temp", hueR, hueG, hueB, I, factor: abFluxNorm(visB) };
    }
  } else {
    const k = Math.max(0, names.indexOf(color));
    const I = new Float32Array(npx);
    if (colorMeta.render_mode === "log") {
      let hi = -Infinity;
      for (let p = 0; p < npx; p++) {
        I[p] = Math.log10(Math.max(at(p, k), 1e-12));
        hi = Math.max(hi, I[p]);
      }
      prepared = {
        mode: "gray-log", I, factor: 1.0,
        logLo: Math.max(hi - 6.0, -12.0), logHi: hi,
      };
    } else {
      for (let p = 0; p < npx; p++) I[p] = at(p, k);
      prepared = { mode: "gray", I, factor: 1.0 };
    }
  }
  prepared.h = rec.h; prepared.w = rec.w; prepared.npx = npx;
  return prepared;
}

function transferCore(prep, knee, gain, K0) {
  const { h, w, npx, I, factor } = prep;
  // Kc = knee·factor; white reference Wref = 30·K0·factor (e⁻). norm cancels
  // factor → asinh(30·K0/knee); at knee=K0, gain=1 this matches eye_rgb.
  const Kc = Math.max(knee * factor, 1e-30);
  const norm = Math.max(Math.asinh((30.0 * K0 * factor) / Kc), 1e-6);
  const G = gain;
  const img = new ImageData(w, h);
  const out = img.data;

  if (prep.mode === "gray-log") {
    const lo = prep.logLo, span = Math.max(prep.logHi - lo, 1e-6);
    for (let p = 0; p < npx; p++) {
      const t = Math.min(Math.max((I[p] - lo) / span, 0), 1);
      const v = Math.min(1, t * G) * 255;
      const o = p * 4; out[o] = v; out[o + 1] = v; out[o + 2] = v; out[o + 3] = 255;
    }
  } else if (prep.mode === "gray") {
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
  return img;
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
const LENS_MIN_ZOOM = 1.0;
const LENS_MAX_ZOOM = 16.0;
const LENS_DEFAULT_ZOOM = 3.0;
const LENS_SIDE = 280;
const COLOR_KEYS = ["q", "w", "e", "r", "t", "y"];

export function mountCutoutViewer(root, opts = {}) {
  const collection = opts.collection;
  const state = {
    meta: null,
    params: Object.assign({}, opts.params || {}),
    index: opts.initialIndex || 0,
    tiers: opts.initialTier ? [String(opts.initialTier)] : [],
                           // selected tier keys (multi-select), canonical order
    color: "VIS",          // band name | "lupton" | "temp"
    knee: 100,             // K (e⁻)
    gain: 1.0,             // brightness multiplier
    K0: 100,               // knee captured on first load → fixes the white ref
    viewOverride: {},      // externally driven colour / transfer-function values
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
  let paramRefreshToken = 0;    // rejects a slower, superseded PSF warp response
  let brightnessControl = null;

  // --- DOM scaffold --------------------------------------------------------
  root.classList.add("cutout-viewer");
  root.innerHTML = "";
  const toolbar = el("div", { class: "cv-toolbar" });
  const framesRow = el("div", { class: "cv-frames" });
  const nav = el("div", { class: "cv-nav" });
  root.append(toolbar, framesRow, nav);
  const makeLensPopup = () => {
    const canvas = el("canvas", { class: "cv-lens__canvas", width: 560, height: 560 });
    const label = el("span", { class: "cv-lens__label" });
    const popup = el("div", { class: "cv-lens", "aria-hidden": "true" }, [canvas, label]);
    document.body.appendChild(popup);
    return { popup, canvas, label, ctx: canvas.getContext("2d") };
  };
  const hoverLens = makeLensPopup();
  const frozenLenses = new Map(); // frame → {u, v, zoom, popup}
  const lensState = {
    frame: null,
    x: 0,
    y: 0,
    zoom: LENS_DEFAULT_ZOOM,
    angularSide: null,
    relativeSide: null,
    visible: false,
  };
  // Compact mode (e.g. the morphology hover preview): hide the toolbar/nav
  // chrome so only the image frame shows.
  if (opts.compact) { toolbar.style.display = "none"; nav.style.display = "none"; }
  else if (opts.hideToolbar) toolbar.style.display = "none";

  const onViewportChange = () => requestAnimationFrame(() => {
    fitFramesToViewport();
    refreshAllLenses();
  });
  const onDocumentMouseMove = (event) => {
    if (!lensState.frame || !lensState.visible) return;
    const rect = lensState.frame.canvas.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right
        || event.clientY < rect.top || event.clientY > rect.bottom) {
      hideHoverLens();
    }
  };
  const scrollParents = [];
  for (let parent = root.parentElement; parent; parent = parent.parentElement) {
    scrollParents.push(parent);
    parent.addEventListener("scroll", onViewportChange, { passive: true });
  }
  window.addEventListener("resize", onViewportChange);
  window.addEventListener("scroll", onViewportChange, { passive: true });
  window.visualViewport?.addEventListener("resize", onViewportChange);
  document.addEventListener("mousemove", onDocumentMouseMove);

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
    // The PSF page changes its replay seed every few seconds.  Warming four
    // invisible neighbours for every seed would spend most of the preview's
    // CPU on arrays the user never sees.
    if (state.params.psf_warp === "1") return;
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
    const prepared = prepareCore(rec, state.meta.color, state.color);
    if (!rec.noCache) {
      state.prepCache.set(key, prepared);
      if (state.prepCache.size > 48) state.prepCache.delete(state.prepCache.keys().next().value);
    }
    return prepared;
  }

  // --- transfer (per slider tick) → ImageData ------------------------------
  // The CHEAP half of rendering: maps a prepared frame's linear intensity to
  // display pixels via the asinh transfer, reading the knee/brightness sliders
  // live. Split out from renderInto so the movie can cache the EXPENSIVE half
  // (prepare's per-pixel colour/temp-fit) once per frame and re-run only this
  // on every blit — so the brightness slider rescales instantly with no rebuild.
  function transferPrepared(prep) {
    return transferCore(prep, state.knee, state.gain, state.K0);
  }

  function renderInto(fr, rec) {
    fr.pixscale = Number.isFinite(rec.pixscale) && rec.pixscale > 0
      ? rec.pixscale : null;
    const prep = prepare(rec);
    if (fr.canvas.width !== prep.w || fr.canvas.height !== prep.h) {
      fr.canvas.width = prep.w; fr.canvas.height = prep.h;
    }
    fr.ctx.putImageData(transferPrepared(prep), 0, 0);
    fr.overlay.textContent = rec.label + magLabel(rec);
    fr.legendWrap.style.display = prep.mode === "temp" ? "flex" : "none";
    setFrameMsg(fr, "");
    showPlens(fr);
    addSigmaToSR(fr, rec);
    refreshAllLenses();
  }

  /** Integrated flux of the cube in the shown band (colour composites fall
   *  back to band 0) and its approximate AB magnitude,
   *  mag = zeropoint_ab_e_total − 2.5·log₁₀(Σe⁻), where the stack zeropoint
   *  is served precomputed (BandConfig.sim_zeropoint_e — the same anchor as
   *  euclid_polish/photometry.py; never re-derive it here). Sums cached per
   *  cube+band; the morph tier (data mutates every animation tick) is
   *  excluded. */
  function magInfo(rec) {
    if (state.meta && state.meta.render_mode === "log") return null;
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

  function lensSide() {
    return Math.max(160, Math.min(LENS_SIDE, window.innerWidth - 24, window.innerHeight - 24));
  }

  function hideHoverLens() {
    lensState.visible = false;
    const oldFrame = lensState.frame;
    lensState.frame = null;
    hoverLens.popup.style.display = "none";
    if (oldFrame && !frozenLenses.has(oldFrame)) oldFrame.lensBox.style.display = "none";
  }

  function clearFrozenLenses() {
    for (const frozen of frozenLenses.values()) frozen.popup.remove();
    frozenLenses.clear();
  }

  function clearAllLenses() {
    hideHoverLens();
    clearFrozenLenses();
    lensState.angularSide = null;
    lensState.relativeSide = null;
    lensState.zoom = LENS_DEFAULT_ZOOM;
  }

  function placeLensPopup(popup, side, x, y) {
    const pad = 12;
    const gap = 18;
    let left = x + gap;
    let top = y + gap;
    if (left + side > window.innerWidth - pad) left = x - side - gap;
    if (top + side > window.innerHeight - pad) top = y - side - gap;
    popup.style.width = `${side}px`;
    popup.style.height = `${side}px`;
    left = Math.max(pad, Math.min(left, window.innerWidth - side - pad));
    top = Math.max(pad, Math.min(top, window.innerHeight - side - pad));
    // Hover lenses follow the viewport. Frozen lenses follow the document:
    // x/y are recomputed from the source tile on every refresh, so adding the
    // scroll offset keeps the popup beside the same image location while the
    // page moves underneath it.
    const documentPosition = popup.classList.contains("cv-lens--frozen");
    popup.style.left = `${left + (documentPosition ? (window.scrollX || 0) : 0)}px`;
    popup.style.top = `${top + (documentPosition ? (window.scrollY || 0) : 0)}px`;
  }

  function validPixscale(fr) {
    return Number.isFinite(fr.pixscale) && fr.pixscale > 0 ? fr.pixscale : null;
  }

  function currentAngularSide(fr, zoom) {
    const pixscale = validPixscale(fr);
    if (!pixscale) return null;
    const sourceSide = Math.min(
      fr.canvas.width,
      fr.canvas.height,
      Math.max(1, lensSide() / zoom),
    );
    return sourceSide * pixscale;
  }

  function currentRelativeSide(fr, zoom) {
    if (!(fr.canvas.width > 0 && fr.canvas.height > 0)) return null;
    const extent = Math.min(fr.canvas.width, fr.canvas.height);
    const sourceSide = Math.min(extent, Math.max(1, lensSide() / zoom));
    return sourceSide / extent;
  }

  function zoomForAngularSide(fr, angularSide) {
    const pixscale = validPixscale(fr);
    if (!pixscale || !(angularSide > 0)) return lensState.zoom;
    const sourceSide = Math.min(
      fr.canvas.width,
      fr.canvas.height,
      Math.max(1, angularSide / pixscale),
    );
    return Math.max(LENS_MIN_ZOOM,
      Math.min(LENS_MAX_ZOOM, lensSide() / sourceSide));
  }

  function zoomForRelativeSide(fr, relativeSide) {
    if (!(relativeSide > 0)) return lensState.zoom;
    const extent = Math.min(fr.canvas.width, fr.canvas.height);
    const sourceSide = Math.min(extent, Math.max(1, relativeSide * extent));
    return Math.max(LENS_MIN_ZOOM,
      Math.min(LENS_MAX_ZOOM, lensSide() / sourceSide));
  }

  function drawLens(fr, lens, position, pinned) {
    if (fr.canvas.width <= 1 || fr.canvas.height <= 1) return;
    const canvasRect = fr.canvas.getBoundingClientRect();
    if (!(canvasRect.width > 0 && canvasRect.height > 0)) return;
    const side = lensSide();
    const x = position.u == null
      ? position.x : canvasRect.left + position.u * canvasRect.width;
    const y = position.v == null
      ? position.y : canvasRect.top + position.v * canvasRect.height;
    const pixscale = validPixscale(fr);
    const extent = Math.min(fr.canvas.width, fr.canvas.height);
    const requestedSourceSide = pixscale && position.angularSide != null
      ? position.angularSide / pixscale
      : position.relativeSide != null
        ? position.relativeSide * extent
      : side / position.zoom;
    const sourceSide = Math.min(
      fr.canvas.width,
      fr.canvas.height,
      Math.max(1, requestedSourceSide),
    );
    const effectiveZoom = side / sourceSide;
    position.zoom = effectiveZoom;
    if (pixscale) position.angularSide = sourceSide * pixscale;
    position.relativeSide = sourceSide / extent;
    const px = ((x - canvasRect.left) / canvasRect.width) * fr.canvas.width;
    const py = ((y - canvasRect.top) / canvasRect.height) * fr.canvas.height;
    const cx = Math.max(sourceSide / 2,
      Math.min(fr.canvas.width - sourceSide / 2, px));
    const cy = Math.max(sourceSide / 2,
      Math.min(fr.canvas.height - sourceSide / 2, py));
    const cropX = cx - sourceSide / 2;
    const cropY = cy - sourceSide / 2;
    lens.ctx.clearRect(0, 0, lens.canvas.width, lens.canvas.height);
    lens.ctx.imageSmoothingEnabled = false;
    lens.ctx.drawImage(fr.canvas, cropX, cropY, sourceSide, sourceSide,
      0, 0, lens.canvas.width, lens.canvas.height);

    const frameRect = fr.frame.getBoundingClientRect();
    const cropLeft = canvasRect.left - frameRect.left
      + (cropX / fr.canvas.width) * canvasRect.width;
    const cropTop = canvasRect.top - frameRect.top
      + (cropY / fr.canvas.height) * canvasRect.height;
    fr.lensBox.style.display = "block";
    fr.lensBox.style.left = `${cropLeft}px`;
    fr.lensBox.style.top = `${cropTop}px`;
    fr.lensBox.style.width = `${(sourceSide / fr.canvas.width) * canvasRect.width}px`;
    fr.lensBox.style.height = `${(sourceSide / fr.canvas.height) * canvasRect.height}px`;
    lens.label.textContent = `${effectiveZoom.toFixed(1)}× · ${pinned ? "click to unfreeze" : "↑↓ zoom · ←→ brightness"}`;
    lens.popup.classList.toggle("cv-lens--frozen", pinned);
    placeLensPopup(lens.popup, side, x, y);
    lens.popup.style.display = "block";
  }

  function refreshLens(fr) {
    const frozen = frozenLenses.get(fr);
    if (frozen) {
      drawLens(fr, frozen, frozen, true);
    } else if (lensState.visible && lensState.frame === fr) {
      drawLens(fr, hoverLens, lensState, false);
    } else {
      fr.lensBox.style.display = "none";
    }
  }

  function refreshAllLenses() {
    for (const fr of state.frames) refreshLens(fr);
    resolveLensOverlaps();
  }

  function popupDocumentRect(popup) {
    const rect = popup.getBoundingClientRect();
    const scrollX = window.scrollX || 0;
    const scrollY = window.scrollY || 0;
    return {
      left: rect.left + scrollX,
      top: rect.top + scrollY,
      right: rect.right + scrollX,
      bottom: rect.bottom + scrollY,
      width: rect.width,
      height: rect.height,
    };
  }

  function lensRectsOverlap(a, b, gap = 18) {
    return a.left < b.right + gap && a.right + gap > b.left
      && a.top < b.bottom + gap && a.bottom + gap > b.top;
  }

  // Keep frozen popups close to their source locations, but move them onto
  // nearby grid slots when several source locations would make them overlap.
  // The grid expands as needed, so this remains collision-free for any number
  // of frozen tiles without tying the popups to viewport coordinates.
  function resolveLensOverlaps() {
    const placed = [];
    if (hoverLens.popup.style.display !== "none") {
      placed.push(popupDocumentRect(hoverLens.popup));
    }
    for (const frozen of frozenLenses.values()) {
      if (frozen.popup.style.display === "none") continue;
      const current = popupDocumentRect(frozen.popup);
      const stepX = current.width + 22;
      const stepY = current.height + 22;
      let chosen = null;
      for (let radius = 0; radius < 256 && !chosen; radius++) {
        for (let dx = -radius; dx <= radius && !chosen; dx++) {
          for (let dy = -radius; dy <= radius; dy++) {
            if (Math.max(Math.abs(dx), Math.abs(dy)) !== radius) continue;
            const candidate = {
              left: Math.max(12, current.left + dx * stepX),
              top: Math.max(12, current.top + dy * stepY),
              width: current.width,
              height: current.height,
            };
            candidate.right = candidate.left + candidate.width;
            candidate.bottom = candidate.top + candidate.height;
            if (placed.every((other) => !lensRectsOverlap(candidate, other))) {
              chosen = candidate;
              break;
            }
          }
        }
      }
      if (!chosen) continue;
      frozen.popup.style.left = `${chosen.left}px`;
      frozen.popup.style.top = `${chosen.top}px`;
      placed.push(chosen);
    }
  }

  function enterLens(fr, event) {
    if (frozenLenses.has(fr)) return;
    if (lensState.frame && lensState.frame !== fr) {
      const angularSide = currentAngularSide(lensState.frame, lensState.zoom);
      if (angularSide != null) lensState.angularSide = angularSide;
      const relativeSide = currentRelativeSide(lensState.frame, lensState.zoom);
      if (relativeSide != null) lensState.relativeSide = relativeSide;
      hideHoverLens();
    }
    state.hot = true;
    lensState.frame = fr;
    lensState.visible = true;
    lensState.x = event.clientX;
    lensState.y = event.clientY;
    if (lensState.angularSide != null) {
      lensState.zoom = validPixscale(fr)
        ? zoomForAngularSide(fr, lensState.angularSide)
        : zoomForRelativeSide(fr, lensState.relativeSide);
    } else if (lensState.relativeSide != null) {
      lensState.zoom = zoomForRelativeSide(fr, lensState.relativeSide);
    }
    refreshLens(fr);
    resolveLensOverlaps();
  }

  function moveLens(fr, event) {
    if (lensState.frame !== fr) {
      enterLens(fr, event);
      return;
    }
    lensState.x = event.clientX;
    lensState.y = event.clientY;
    refreshLens(fr);
    resolveLensOverlaps();
  }

  function zoomLens(fr, event) {
    event.preventDefault();
    event.stopPropagation();
    if (Math.abs(event.deltaX) > Math.abs(event.deltaY)) {
      state.gain = Math.max(0.1, Math.min(10, state.gain * Math.exp(-event.deltaX * 0.002)));
      if (brightnessControl) {
        const input = brightnessControl.querySelector("input");
        const output = brightnessControl.querySelector(".cv-val");
        input.value = Math.round(1000 * (Math.log(state.gain) - Math.log(0.1))
          / (Math.log(10) - Math.log(0.1)));
        output.textContent = `${state.gain.toFixed(2)}×`;
      }
      rerender();
      notify();
      return;
    }
    if (frozenLenses.has(fr)) return;
    if (lensState.frame !== fr) enterLens(fr, event);
    const factor = event.deltaY < 0 ? 1.18 : 1 / 1.18;
    lensState.zoom = Math.max(LENS_MIN_ZOOM,
      Math.min(LENS_MAX_ZOOM, lensState.zoom * factor));
    const angularSide = currentAngularSide(fr, lensState.zoom);
    if (angularSide != null) lensState.angularSide = angularSide;
    const relativeSide = currentRelativeSide(fr, lensState.zoom);
    if (relativeSide != null) lensState.relativeSide = relativeSide;
    refreshLens(fr);
    resolveLensOverlaps();
  }

  function toggleFrozen(fr, event) {
    if (frozenLenses.has(fr)) {
      frozenLenses.get(fr).popup.remove();
      frozenLenses.delete(fr);
      hideHoverLens();
      refreshAllLenses();
      return;
    }
    const rect = fr.canvas.getBoundingClientRect();
    const u = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    const v = Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height));
    const popup = makeLensPopup();
    const frozen = {
      u, v,
      zoom: lensState.frame === fr ? lensState.zoom : LENS_DEFAULT_ZOOM,
      angularSide: lensState.frame === fr ? lensState.angularSide : null,
      relativeSide: lensState.frame === fr ? lensState.relativeSide : null,
      popup: popup.popup, canvas: popup.canvas, label: popup.label, ctx: popup.ctx,
    };
    frozenLenses.set(fr, frozen);
    hideHoverLens();
    refreshAllLenses();
  }

  function makeFrame(tier) {
    const canvas = el("canvas", { class: "cv-canvas", width: 1, height: 1 });
    const legendCanvas = el("canvas", { class: "cv-legend", width: 14, height: 160 });
    const legendWrap = el("div", { class: "cv-legend-wrap" }, [
      el("span", { class: "cv-legend-tick", text: "20k" }),
      legendCanvas,
      el("span", { class: "cv-legend-tick", text: "3k" }),
    ]);
    const lensBox = el("div", { class: "cv-lens-box", "aria-hidden": "true" });
    const overlay = el("div", { class: "cv-overlay" });
    const plens = el("div", { class: "cv-plens" });   // headed-model P(lens) for this tier
    const msg = el("div", { class: "cv-msg" });
    const frame = el("div", { class: "cv-frame" }, [
      canvas, lensBox, legendWrap, overlay, plens, msg,
    ]);
    canvas.addEventListener("mouseenter", (event) => enterLens(fr, event));
    canvas.addEventListener("mousemove", (event) => moveLens(fr, event));
    canvas.addEventListener("mouseleave", () => {
      if (lensState.frame === fr && !frozenLenses.has(fr)) hideHoverLens();
    });
    canvas.addEventListener("wheel", (event) => zoomLens(fr, event), { passive: false });
    canvas.addEventListener("click", (event) => toggleFrozen(fr, event));
    const fr = { tier, frame, canvas, ctx: canvas.getContext("2d"), pixscale: null,
                 legendWrap, legendCanvas, lensBox, overlay, plens, msg };
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
    clearAllLenses();
    stopMorph();               // frames are being recreated → drop the old loop
    framesRow.innerHTML = "";
    state.frames = want.map((t) => {
      const fr = makeFrame(t);
      framesRow.appendChild(fr.frame);
      return fr;
    });
    framesRow.classList.toggle("cv-multi", state.frames.length > 1);
    requestAnimationFrame(fitFramesToViewport);
  }

  /** Choose the grid that gives every visible tile the largest square side.
   *
   * Reserve the viewer's own toolbar and navigation rows, but do not subtract
   * the viewer's document position: a page header can place the viewer below
   * the fold even though the tile should still be screen-sized when reached.
   * Try every possible column count because the best choice depends on both
   * the screen aspect ratio and the number of selected tiers. */
  function fitFramesToViewport() {
    if (!state.frames.length) {
      root.style.removeProperty("--cv-frame-size");
      root.style.removeProperty("--cv-columns");
      return;
    }
    const frameStyle = getComputedStyle(framesRow);
    const columnGap = parseFloat(frameStyle.columnGap) || 14;
    const rowGap = parseFloat(frameStyle.rowGap) || columnGap;
    const width = framesRow.clientWidth;
    const toolbarStyle = getComputedStyle(toolbar);
    const toolbarMargin = parseFloat(toolbarStyle.marginBottom) || 0;
    const navStyle = getComputedStyle(nav);
    const navMargin = parseFloat(navStyle.marginTop) || 0;
    const bottomMargin = 16;
    const visibleHeight = window.visualViewport?.height ?? window.innerHeight;
    const viewportHeight = visibleHeight - toolbar.offsetHeight - toolbarMargin
      - nav.offsetHeight - navMargin - bottomMargin;
    const count = state.frames.length;
    const maxColumns = window.innerWidth > 760 ? count : 1;
    let best = { side: 0, columns: 1 };

    for (let columns = 1; columns <= maxColumns; columns++) {
      const rows = Math.ceil(count / columns);
      const widthLimit = (width - columnGap * (columns - 1)) / columns;
      const heightLimit = (viewportHeight - rowGap * (rows - 1)) / rows;
      const side = Math.floor(Math.min(widthLimit, heightLimit));
      if (side > best.side || (side === best.side && columns > best.columns)) {
        best = { side, columns };
      }
    }

    if (best.side > 0) {
      root.style.setProperty("--cv-frame-size", `${best.side}px`);
      root.style.setProperty("--cv-columns", `${best.columns}`);
    }
  }

  // --- "morph" tier: an animated ensemble-disagreement frame ----------------
  //
  // The movie is a fixed loop of MORPH_FRAMES frames (periodic: mean +
  // Σ aᵢ·sin(2π·kᵢ·φ)·PCᵢ has period φ=1). Drawing a frame has an EXPENSIVE half
  // — prepare()'s per-pixel colour/temp-fit — and a CHEAP half — the asinh
  // brightness transfer. So we PRE-RENDER every frame's prepare() output once,
  // cache it, and each animation tick re-runs only the cheap transfer. Hence:
  //   • first arm shows a progress bar while the loop is cached, then plays;
  //   • the brightness/knee sliders rescale instantly (just a fresh transfer
  //     over the cached frames — no rebuild);
  //   • neighbouring fields are cached in the background so next/prev is instant.
  // Keyed by field+subset; colour or morph-amplitude changes (they alter
  // prepare) rebuild; an LRU bounds memory (each full-res movie is sizeable).
  const MORPH_FRAMES = 48;
  const FRQ = [1, 2, 3], MPH = [0, Math.PI / 2, Math.PI / 3];
  const MOVIE_RADIUS = 10;                 // fields to pre-cache in each direction
  const MOVIE_BUDGET_BYTES = 1.4e9;        // ~1.4 GB cap on the movie LRU
  const movieStore = new Map();            // key → cached movie entry (LRU order)
  let buildToken = 0;                      // bump to abort an in-flight build
  let playingKey = null;                   // never evict the movie on screen

  const movieKey = (index, subset) => `${index}|${subset || ""}`;
  const movieFresh = (e) =>
    e && e.done && e.color === state.color && e.amp === state.morphAmp;

  function stopMorph() {
    if (morphRaf != null) cancelAnimationFrame(morphRaf);
    morphRaf = null;
  }

  function evictMovies() {                 // drop LRU entries until under budget
    let total = 0;
    for (const e of movieStore.values()) total += e.bytes || 0;
    for (const k of [...movieStore.keys()]) {
      if (total <= MOVIE_BUDGET_BYTES) break;
      if (k === playingKey) continue;      // keep what's on screen
      total -= (movieStore.get(k).bytes || 0);
      movieStore.delete(k);
    }
  }

  // Progress bar overlaid on the frame while a movie loop is being cached.
  function movieProgress(fr, p) {
    let bar = fr.frame.querySelector(".cv-movie-prog");
    if (p == null) { if (bar) bar.remove(); return; }
    if (!bar) {
      bar = el("div", { class: "cv-movie-prog", style:
        "position:absolute;left:14%;right:14%;bottom:16px;height:20px;z-index:6;" +
        "border-radius:10px;background:rgba(0,0,0,.5);overflow:hidden;" +
        "font:600 11px 'IBM Plex Mono',monospace;color:#fff;" });
      bar.append(
        el("div", { class: "cv-movie-prog__fill", style:
          "position:absolute;inset:0;width:0;background:rgba(120,170,255,.55);" +
          "transition:width .08s linear;" }),
        el("div", { class: "cv-movie-prog__lbl", style:
          "position:absolute;inset:0;display:flex;align-items:center;justify-content:center;" }));
      fr.frame.appendChild(bar);
    }
    bar.querySelector(".cv-movie-prog__fill").style.width = `${Math.round(p * 100)}%`;
    bar.querySelector(".cv-movie-prog__lbl").textContent = `caching movie… ${Math.round(p * 100)}%`;
  }

  // Fetch a field's movie ingredients: sr (mean) + PCA cubes (subset-aware).
  async function movieCubes(index, subset) {
    const extra = subset ? { members: subset } : undefined;
    const nSub = subset ? subset.split(",").filter(Boolean).length : 0;
    const pcaMax = (state.meta && state.meta.pca_max) || 3;
    const n = subset ? Math.max(0, Math.min(pcaMax, nSub - 1))
                     : ((state.meta && state.meta.pca_n) | 0);
    const sr = await fetchCube("sr", index, extra);
    const comps = [];
    for (let k = 0; k < n; k++) {
      try { comps.push(await fetchCube(`pca${k}`, index, extra)); } catch { /* fewer PCs */ }
    }
    const metaAmps = (state.meta.pca_amps && state.meta.pca_amps[index]) || [];
    const amps = comps.map((c, k) => (c.amp != null ? c.amp : (metaAmps[k] || 0)));
    const varTot = comps.reduce((a, c) => a + (c.varexp || 0), 0);
    const subLbl = subset ? ` · ${nSub} members` : "";
    const varLbl = varTot > 0
      ? ` · ${comps.length} PCs ≈ ${(varTot * 100).toFixed(0)}% of variance` : "";
    return { sr, comps, amps, pixscale: sr.pixscale,
      label: `disagreement movie${subLbl}${varLbl}` };
  }

  // Build (cache) one field's movie loop — MORPH_FRAMES prepared frames at the
  // current colour + amplitude. Async, yields between frames so the progress bar
  // animates and the UI stays responsive; abortable via `token`. Returns the
  // entry or null (aborted / no cubes).
  async function buildMovie(index, subset, { token, onProgress } = {}) {
    const key = movieKey(index, subset);
    const existing = movieStore.get(key);
    if (movieFresh(existing)) {
      movieStore.delete(key); movieStore.set(key, existing);   // touch LRU
      return existing;
    }
    const color = state.color, amp = state.morphAmp;
    let ing;
    try { ing = await movieCubes(index, subset); } catch { return null; }
    if (token != null && token !== buildToken) return null;
    const { sr, comps, amps, pixscale, label } = ing;
    const len = sr.data.length, data = new Float32Array(len);
    const entry = { frames: new Array(MORPH_FRAMES).fill(null), w: sr.w, h: sr.h,
                    mode: "gray", bytes: 0, color, amp, done: false, label, pixscale };
    movieStore.set(key, entry);
    for (let slot = 0; slot < MORPH_FRAMES; slot++) {
      if (token != null && token !== buildToken) { movieStore.delete(key); return null; }
      const ph = slot / MORPH_FRAMES;
      data.set(sr.data);                              // mean (full res)
      for (let k = 0; k < comps.length; k++) {
        const ck = (amps[k] || 0) * amp * Math.sin(2 * Math.PI * FRQ[k % 3] * ph + MPH[k % 3]);
        const cd = comps[k].data;
        for (let i = 0; i < len; i++) data[i] += ck * cd[i];
      }
      entry.frames[slot] = prepare({ key: `mv:${key}:${slot}`, h: sr.h, w: sr.w,
                                     c: sr.c, data, noCache: true });
      if (onProgress) onProgress((slot + 1) / MORPH_FRAMES);
      await new Promise((r) => setTimeout(r, 0));     // yield to the event loop
    }
    entry.mode = entry.frames[0] ? entry.frames[0].mode : "gray";
    // ~bytes: frames × pixels × Float32 planes kept (temp/lupton 4, gray 1).
    entry.bytes = MORPH_FRAMES * entry.w * entry.h * 4 * (entry.mode === "gray" ? 1 : 4);
    entry.done = true;
    evictMovies();
    return entry;
  }

  // Play a cached movie: each tick blits transferPrepared(frame) — only the
  // cheap brightness transfer, so the sliders rescale live with no rebuild. If
  // colour / amplitude change (they alter prepare), settle then re-arm.
  function playMovie(fr, index, entry) {
    stopMorph();
    let phase = 0, last = performance.now();
    let lastAmp = state.morphAmp, lastColor = state.color, changedAt = 0;
    const drawSlot = (slot) => {
      const prep = entry.frames[slot];
      if (!prep) return;
      if (fr.canvas.width !== entry.w || fr.canvas.height !== entry.h) {
        fr.canvas.width = entry.w; fr.canvas.height = entry.h;
      }
      fr.pixscale = Number.isFinite(entry.pixscale) && entry.pixscale > 0
        ? entry.pixscale : null;
      fr.ctx.putImageData(transferPrepared(prep), 0, 0);
      fr.overlay.textContent = entry.label;           // magLabel="" for morph
      fr.legendWrap.style.display = entry.mode === "temp" ? "flex" : "none";
      setFrameMsg(fr, "");
      refreshLens(fr);
    };
    drawSlot(0);                                       // instant first frame
    const tick = (now) => {
      // Colour / amplitude drive prepare → need a rebuild, but debounce so a
      // slider drag doesn't storm rebuilds: only re-arm once it settles.
      if (state.morphAmp !== lastAmp || state.color !== lastColor) {
        lastAmp = state.morphAmp; lastColor = state.color; changedAt = now;
      }
      if ((entry.amp !== state.morphAmp || entry.color !== state.color)
          && now - changedAt > 250) { startMorph(fr, index); return; }
      phase += ((now - last) / 1000) * state.morphSpeed;
      last = now;
      drawSlot(Math.min(MORPH_FRAMES - 1,
                        Math.floor((phase - Math.floor(phase)) * MORPH_FRAMES)));
      morphRaf = requestAnimationFrame(tick);
    };
    morphRaf = requestAnimationFrame(tick);
  }

  // Cache neighbouring fields' movies in the background (interleaved outward) so
  // next/prev is instant. Stops if a newer arm bumps the token or budget is hit.
  async function prefetchMovies(index, subset, token) {
    for (let d = 1; d <= MOVIE_RADIUS; d++) {
      for (const j of [index + d, index - d]) {
        if (token !== buildToken) return;
        if (j < 0 || j >= state.meta.count) continue;
        let total = 0; for (const e of movieStore.values()) total += e.bytes || 0;
        if (total > MOVIE_BUDGET_BYTES) return;        // buffer full → stop reaching
        if (movieFresh(movieStore.get(movieKey(j, subset)))) continue;
        await buildMovie(j, subset, { token });
      }
    }
  }

  async function startMorph(fr, index) {
    stopMorph();
    const token = ++buildToken;                        // supersede any older build
    const subset = state.morphMembers;
    const key = movieKey(index, subset);
    playingKey = key;
    let entry = movieStore.get(key);
    fr.frame.classList.remove("cv-loading");
    if (!movieFresh(entry)) {                          // build with a progress bar
      movieProgress(fr, 0);
      entry = await buildMovie(index, subset, { token, onProgress: (p) => movieProgress(fr, p) });
      movieProgress(fr, null);
      if (token !== buildToken) return;                // superseded while building
      if (!entry) { setFrameMsg(fr, "movie unavailable"); return; }
    }
    playMovie(fr, index, entry);
    prefetchMovies(index, subset, token);              // background, not awaited
  }

  // --- load + show current index across all selected tiers -----------------
  async function show() {
    clearAllLenses();
    if (!state.meta || state.meta.count === 0) {
      rebuildFrames();
      for (const fr of state.frames) {
        setFrameMsg(fr, state.meta?.empty_label || "No cutouts available.");
      }
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

  // Refresh only the currently visible cubes after a parameter change.  Keep
  // the existing frames mounted so a live PSF warp swaps cleanly instead of
  // resetting the selected cluster/tier or flashing the viewer chrome.
  async function refreshVisible() {
    if (!state.meta || state.meta.count === 0) return;
    const token = ++paramRefreshToken;
    stopMorph();
    state.cubeCache.clear();
    state.prepCache.clear();
    state.shown.clear();
    await Promise.all(state.frames.map(async (fr) => {
      if (!tierAvail(fr.tier)) return;
      if (fr.tier === "morph") { await startMorph(fr, state.index); return; }
      fr.frame.classList.add("cv-loading");
      try {
        const rec = await fetchCube(fr.tier, state.index);
        if (token !== paramRefreshToken) return;
        state.shown.set(fr.tier, rec);
        renderInto(fr, rec);
      } catch {
        if (token !== paramRefreshToken) return;
        setFrameMsg(fr, tierAvail(fr.tier) ? "not synced yet" : `no ${fr.tier}`);
      } finally {
        if (token === paramRefreshToken) fr.frame.classList.remove("cv-loading");
      }
    }));
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
    brightnessControl = null;
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
    const logMode = state.meta.render_mode === "log";
    if (!logMode) {
      toolbar.append(el("span", { class: "cv-grouplabel", text: "Colour" }));
      const cg = el("div", { class: "cv-group" });
      let colorIndex = 0;
      state.meta.band_names.forEach((n) => {
        const key = COLOR_KEYS[colorIndex++]?.toUpperCase();
        cg.append(chip("color", n, n, key ? `${key} — ${n}` : n));
      });
      COLOR_MODES_EXTRA.forEach((m) => {
        const key = COLOR_KEYS[colorIndex++]?.toUpperCase();
        cg.append(chip("color", m.key, m.label,
          key ? `${key} — ${m.title}` : m.title));
      });
      toolbar.append(cg);
    }

    const sg = el("div", { class: "cv-group cv-sliders" });
    if (!logMode) {
      sg.append(slider("asinh knee", 5, 5000, state.knee,
        (v) => `${Math.round(v)} e⁻`, (v) => { state.knee = v; rerender(); }));
    }
    brightnessControl = slider("brightness", 0.1, 10, state.gain,
      (v) => `${v.toFixed(2)}×`, (v) => { state.gain = v; rerender(); });
    sg.append(brightnessControl);
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
    if (e.target.matches("input, textarea, select") || e.ctrlKey || e.metaKey || e.altKey) return;
    const key = e.key.toLowerCase();
    const colorIndex = COLOR_KEYS.indexOf(key);
    if (colorIndex >= 0 && state.meta && state.meta.render_mode !== "log") {
      const colors = [...(state.meta.band_names || []), ...COLOR_MODES_EXTRA.map((m) => m.key)];
      const color = colors[colorIndex];
      if (color && color !== state.color) {
        state.color = color;
        syncChips();
        rerender();
        notify();
      }
      if (color) e.preventDefault();
      return;
    }
    if (e.key === "ArrowLeft") { go(state.index - 1); e.preventDefault(); }
    else if (e.key === "ArrowRight") { go(state.index + 1); e.preventDefault(); }
    else if (e.key === " ") { togglePlay(); e.preventDefault(); }
  }
  root.addEventListener("mouseenter", () => { state.hot = true; });
  root.addEventListener("mouseleave", () => { state.hot = false; hideHoverLens(); });
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
    const defaultKnee = state.meta.color.default_asinh || 100;
    state.K0 = defaultKnee;
    state.knee = Number.isFinite(state.viewOverride.knee)
      ? state.viewOverride.knee : defaultKnee;
    if (Number.isFinite(state.viewOverride.gain)) state.gain = state.viewOverride.gain;
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
    /** Patch cube-query parameters without reloading metadata or rebuilding
     *  frames. Used by the PSF page's replayable live-warp preview. */
    setParams(patch) {
      Object.assign(state.params, patch || {});
      return refreshVisible();
    },
    /** Drive the client-side colour transfer from an external control surface.
     *  Values are remembered even when called before metadata finishes loading,
     *  which lets two separately mounted viewers share one exact transfer. */
    setView(patch) {
      const next = patch || {};
      if (typeof next.color === "string") {
        const allowed = !state.meta || state.meta.band_names.includes(next.color)
          || ["lupton", "temp"].includes(next.color);
        if (allowed) {
          state.color = next.color;
          state.viewOverride.color = next.color;
        }
      }
      if (Number.isFinite(next.knee) && next.knee > 0) {
        state.knee = next.knee;
        state.viewOverride.knee = next.knee;
      }
      if (Number.isFinite(next.gain) && next.gain > 0) {
        state.gain = next.gain;
        state.viewOverride.gain = next.gain;
      }
      if (!state.meta) return;
      buildToolbar();
      syncChips();
      rerender();
      notify();
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
               knee: state.knee, gain: state.gain,
               params: Object.assign({}, state.params) };
    },
    reload() { return loadMeta().then(show); },
    destroy() {
      paramRefreshToken++;
      clearAllLenses();
      hoverLens.popup.remove();
      window.removeEventListener("resize", onViewportChange);
      window.removeEventListener("scroll", onViewportChange);
      for (const parent of scrollParents) {
        parent.removeEventListener("scroll", onViewportChange);
      }
      window.visualViewport?.removeEventListener("resize", onViewportChange);
      document.removeEventListener("mousemove", onDocumentMouseMove);
      document.removeEventListener("keydown", onKey);
      if (state.playTimer) clearInterval(state.playTimer);
      stopMorph();
      buildToken++;                 // abort any in-flight movie build
      movieStore.clear();           // free the cached movie frames
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
  prepareCore, transferCore,
};

/** Render one N-band cube to an ImageData with the SAME pipeline the field
 *  viewer uses — so the ensemble back-trace stamps match the viewer's colour /
 *  knee / brightness exactly. `rec` = { data:Float32Array, h, w, c }; `colorMeta`
 *  = the served meta's `color` block; `opts` = { color, knee, gain, K0 }. */
export function renderCubeImageData(rec, colorMeta, opts = {}) {
  const prep = prepareCore(rec, colorMeta, opts.color || "VIS");
  const knee = opts.knee || 100;
  return transferCore(prep, knee, opts.gain != null ? opts.gain : 1.0,
                      opts.K0 || knee);
}
