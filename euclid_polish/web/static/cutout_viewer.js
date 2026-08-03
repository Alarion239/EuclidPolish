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
  // A collection can mix cameras: e.g. Euclid's four calibrated channels
  // beside a one-filter JWST image.  The response header, not the collection
  // default, is the authority for a particular frame's channel order.
  const names = Array.isArray(rec.bands) && rec.bands.length === rec.c
    ? rec.bands : colorMeta.band_names.slice(0, rec.c);
  const bandOf = (n) => colorMeta.bands[n];
  const npx = rec.h * rec.w;
  // A heterogeneous archive panel may ask for a display-only contrast scale.
  // It never modifies the served Float32/FITS science values.
  const displayScale = Number.isFinite(rec.displayScale) && rec.displayScale > 0
    ? rec.displayScale : 1.0;
  const at = (p, k) => rec.data[p * rec.c + k] * displayScale; // band k at pixel p
  const calibrated = names.every((n) => !!bandOf(n));
  const canLupton = calibrated && colorMeta.rgb_scheme.every((n) => names.includes(n));
  const canTemp = calibrated && names.length >= 2;

  let prepared;
  if (rec.directRgb && rec.c >= 3) {
    const R = new Float32Array(npx), G = new Float32Array(npx), B = new Float32Array(npx);
    const I = new Float32Array(npx);
    for (let p = 0; p < npx; p++) {
      R[p] = Math.max(at(p, 0), 0); G[p] = Math.max(at(p, 1), 0); B[p] = Math.max(at(p, 2), 0);
      I[p] = (R[p] + G[p] + B[p]) / 3.0;
    }
    prepared = { mode: "direct-rgb", R, G, B, I, factor: 1.0 };
  } else if ((color === "lupton" && canLupton) || (color === "temp" && canTemp)) {
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
      // JWST's F### approximation carries no invented AB zero point; retain
      // its display-normalised native scale instead of borrowing Euclid VIS.
      const factor = names.some((n) => bandOf(n).display_only) ? 1.0 : abFluxNorm(visB);
      prepared = { mode: "temp", hueR, hueG, hueB, I, factor };
    }
  } else {
    // A colour chosen for a different camera is never applied to this image.
    // For a one-filter JWST frame this resolves to its sole native channel.
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
      prepared = rec.tint && rec.tint.length === 3
        ? { mode: "tint", I, factor: 1.0, tint: rec.tint }
        : { mode: "gray", I, factor: 1.0 };
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
  } else if (prep.mode === "tint") {
    const [tr, tg, tb] = prep.tint;
    for (let p = 0; p < npx; p++) {
      const v = asinhTransfer(I[p], G, Kc, norm) * 255;
      const o = p * 4;
      out[o] = tr * v; out[o + 1] = tg * v; out[o + 2] = tb * v; out[o + 3] = 255;
    }
  } else if (prep.mode === "lupton" || prep.mode === "direct-rgb") {
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
const LENS_ZOOM_STEP = 1.18;
const LENS_SIDE = 280;
const LENS_LAYOUT_GAP = 12;
const PUBLICATION_MAX_WIDTH = 4800;
const PUBLICATION_PAPER = "#ffffff";
const PUBLICATION_INK = "#111111";
const COLOR_KEYS = ["q", "w", "e", "r", "t", "y"];
const VIEW_LAYOUT_STORAGE_KEY = "euclid-polish.cutout-viewer.layout";

function savedViewLayout() {
  try {
    const value = window.localStorage.getItem(VIEW_LAYOUT_STORAGE_KEY);
    return value === "two-rows" ? value : "one-row";
  } catch (_) {
    return "one-row";
  }
}

function saveViewLayout(value) {
  try { window.localStorage.setItem(VIEW_LAYOUT_STORAGE_KEY, value); } catch (_) { /* optional */ }
}

export function mountCutoutViewer(root, opts = {}) {
  const collection = opts.collection;
  const state = {
    meta: null,
    params: Object.assign({}, opts.params || {}),
    index: opts.initialIndex || 0,
    tiers: Array.isArray(opts.initialTiers) ? opts.initialTiers.map(String)
      : (opts.initialTier ? [String(opts.initialTier)] : []),
                           // selected tier keys (multi-select), canonical order
    layout: savedViewLayout(), // one-row | two-rows; applies to every shared viewer
    color: "VIS",          // band name | "lupton" | "temp"
    knee: 100,             // K (e⁻)
    gain: 1.0,             // brightness multiplier
    transfers: {},         // transfer group → { knee, gain }; default retains legacy behaviour
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
  const brightnessControls = new Map();

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
  let frozenSequence = 0;
  const lensState = {
    frame: null,
    x: 0,
    y: 0,
    zoom: LENS_DEFAULT_ZOOM,
    angularSide: null,
    relativeSide: null,
    visible: false,
  };
  // Last inspected sky location, retained after the pointer leaves a canvas.
  // Publication export projects this normalized position and physical crop
  // size onto every selected tier, so LR/SR/HR/JWST insets compare the same
  // patch instead of unrelated pixel coordinates.
  let publicationRegion = null;
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
  const cacheKey = (tier, index, extra) => {
    const params = Object.assign({}, state.params, extra || {});
    const suffix = new URLSearchParams(params).toString();
    return `${tier}:${index}${suffix ? ":" + suffix : ""}`;
  };
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
  const missingTierLabel = (key) =>
    (state.meta && state.meta.missing_tier_labels && state.meta.missing_tier_labels[key])
      || `no ${key}`;
  const jwstBandAvailable = (value) => {
    if (value === "colour") return true;
    const obj = state.meta && state.meta.objects && state.meta.objects[state.index];
    return !obj || !obj.jwst_bands || obj.jwst_bands.includes(value);
  };
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
      transferGroup: r.headers.get("X-Cube-Transfer-Group") || "default",
      displayScale: parseFloat(r.headers.get("X-Cube-Display-Scale")) || 1,
      bands: (r.headers.get("X-Cube-Bands") || "").split(",").filter(Boolean),
      tint: (r.headers.get("X-Cube-Tint") || "").split(",").map(Number)
        .filter(Number.isFinite),
      directRgb: r.headers.get("X-Cube-Direct-RGB") === "1",
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
  function transferGroupKeys() {
    const groups = state.meta && Array.isArray(state.meta.transfer_groups)
      ? state.meta.transfer_groups.filter((group) => group === "euclid" || group === "jwst")
      : [];
    return groups.length ? Array.from(new Set(groups)) : ["default"];
  }

  function resetTransferSettings() {
    state.transfers = {};
    for (const group of transferGroupKeys()) {
      state.transfers[group] = { knee: state.knee, gain: state.gain };
    }
  }

  function copiedTransferSettings() {
    const copied = {};
    for (const group of Object.keys(state.transfers)) {
      const transfer = state.transfers[group];
      copied[group] = { knee: transfer.knee, gain: transfer.gain };
    }
    return copied;
  }

  function transferSetting(group) {
    if (!state.transfers[group]) state.transfers[group] = { knee: state.knee, gain: state.gain };
    return state.transfers[group];
  }

  function transferFor(rec) {
    return transferSetting(rec && rec.transferGroup ? rec.transferGroup : "default");
  }

  function transferPrepared(prep, rec) {
    const transfer = transferFor(rec);
    return transferCore(prep, transfer.knee, transfer.gain, state.K0);
  }

  function renderInto(fr, rec) {
    fr.pixscale = Number.isFinite(rec.pixscale) && rec.pixscale > 0
      ? rec.pixscale : null;
    const prep = prepare(rec);
    if (fr.canvas.width !== prep.w || fr.canvas.height !== prep.h) {
      fr.canvas.width = prep.w; fr.canvas.height = prep.h;
    }
    fr.ctx.putImageData(transferPrepared(prep, rec), 0, 0);
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
    const names = rec.bands && rec.bands.length === rec.c
      ? rec.bands : (state.meta.band_names || []);
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
    if (!b || b.display_only) return null; // native JWST approximation: no fake AB magnitude
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

  function snapshotFrozenLenses() {
    const snapshot = new Map();
    for (const [fr, frozen] of frozenLenses) {
      snapshot.set(fr.tier, {
        u: frozen.u,
        v: frozen.v,
        zoom: frozen.zoom,
        angularSide: frozen.angularSide,
        relativeSide: frozen.relativeSide,
        corner: frozen.corner,
        order: frozen.order,
      });
    }
    return snapshot;
  }

  function restoreFrozenLenses(snapshot) {
    if (!snapshot || !snapshot.size) return;
    for (const fr of state.frames) {
      const position = snapshot.get(fr.tier);
      if (!position) continue;
      const popup = makeLensPopup();
      const order = Number.isFinite(position.order) ? position.order : ++frozenSequence;
      frozenSequence = Math.max(frozenSequence, order);
      popup.popup.style.zIndex = String(1000 + order);
      frozenLenses.set(fr, {
        ...position,
        popup: popup.popup,
        canvas: popup.canvas,
        label: popup.label,
        ctx: popup.ctx,
      });
    }
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
    const documentPosition = popup.classList.contains("cv-lens--frozen");
    let left = x + gap;
    let top = y + gap;
    if (left + side > window.innerWidth - pad) left = x - side - gap;
    if (top + side > window.innerHeight - pad) top = y - side - gap;
    popup.style.width = `${side}px`;
    popup.style.height = `${side}px`;
    // Frozen popups follow the source tile, including when that tile is
    // outside the viewport. Clamping them here would pin them to the top or
    // bottom edge while the scroll container moved the source away.
    if (documentPosition) {
      popup.style.left = `${left + (window.scrollX || 0)}px`;
      popup.style.top = `${top + (window.scrollY || 0)}px`;
      return;
    }
    left = Math.max(pad, Math.min(left, window.innerWidth - side - pad));
    top = Math.max(pad, Math.min(top, window.innerHeight - side - pad));
    popup.style.left = `${left}px`;
    popup.style.top = `${top}px`;
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

  function rememberPublicationRegion(fr, x, y) {
    const rect = fr.canvas.getBoundingClientRect();
    if (!(rect.width > 0 && rect.height > 0)) return;
    const u = Math.max(0, Math.min(1, (x - rect.left) / rect.width));
    const v = Math.max(0, Math.min(1, (y - rect.top) / rect.height));
    publicationRegion = {
      u,
      v,
      zoom: lensState.zoom,
      angularSide: currentAngularSide(fr, lensState.zoom),
      relativeSide: currentRelativeSide(fr, lensState.zoom),
    };
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

  function receptiveFieldLabels(fr, position) {
    const fields = state.meta && Array.isArray(state.meta.receptive_fields)
      ? state.meta.receptive_fields : [];
    const angularSide = validPixscale(fr) ? position.angularSide : null;
    if (!(angularSide > 0) || !fields.length) return [];
    // Wheel zoom is discrete. Treat the two neighboring wheel positions as
    // the valid match for a field that lies between them, using log-space so
    // the tolerance is symmetric for zooming in and out.
    const tolerance = Math.log(LENS_ZOOM_STEP) / 2 + 1e-9;
    return fields
      .filter((field) => {
        const target = Number(field && field.angular_side_arcsec);
        return target > 0
          && Math.abs(Math.log(angularSide / target)) <= tolerance;
      })
      .sort((a, b) => Math.abs(Math.log(angularSide / a.angular_side_arcsec))
        - Math.abs(Math.log(angularSide / b.angular_side_arcsec)))
      .map((field) => String(field.label || `${field.blocks}b`));
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
    position.sourceRect = {
      left: canvasRect.left + (cropX / fr.canvas.width) * canvasRect.width,
      top: canvasRect.top + (cropY / fr.canvas.height) * canvasRect.height,
      width: (sourceSide / fr.canvas.width) * canvasRect.width,
      height: (sourceSide / fr.canvas.height) * canvasRect.height,
    };
    const rfLabels = receptiveFieldLabels(fr, position);
    const labelParts = [`${effectiveZoom.toFixed(1)}×`, ...rfLabels.map((label) => `${label} RF`)];
    if (pinned) labelParts.push("click to unfreeze");
    lens.label.textContent = labelParts.join(" · ");
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

  function lensRectsOverlap(a, b) {
    return a.left < b.right && a.right > b.left
      && a.top < b.bottom && a.bottom > b.top;
  }

  function sourceDocumentRect(position) {
    if (!position.sourceRect) return null;
    const source = position.sourceRect;
    const scrollX = window.scrollX || 0;
    const scrollY = window.scrollY || 0;
    return {
      left: source.left + scrollX,
      top: source.top + scrollY,
      right: source.left + source.width + scrollX,
      bottom: source.top + source.height + scrollY,
    };
  }

  function cornerCandidates(source, width, height) {
    if (!source) return [];
    const gap = LENS_LAYOUT_GAP;
    return [
      { corner: "top-left", left: source.left - width - gap, top: source.top - height - gap },
      { corner: "top-right", left: source.right + gap, top: source.top - height - gap },
      { corner: "bottom-left", left: source.left - width - gap, top: source.bottom + gap },
      { corner: "bottom-right", left: source.right + gap, top: source.bottom + gap },
    ].map((candidate) => Object.assign(candidate, {
      width, height,
      right: candidate.left + width,
      bottom: candidate.top + height,
    }));
  }

  function candidateFitsViewport(candidate) {
    const left = candidate.left - (window.scrollX || 0);
    const top = candidate.top - (window.scrollY || 0);
    const pad = 12;
    return left >= pad && top >= pad
      && candidate.right - (window.scrollX || 0) <= window.innerWidth - pad
      && candidate.bottom - (window.scrollY || 0) <= window.innerHeight - pad;
  }

  function candidateVisibleArea(candidate) {
    const scrollX = window.scrollX || 0;
    const scrollY = window.scrollY || 0;
    const left = candidate.left - scrollX;
    const top = candidate.top - scrollY;
    const right = candidate.right - scrollX;
    const bottom = candidate.bottom - scrollY;
    return Math.max(0, Math.min(window.innerWidth, right) - Math.max(0, left))
      * Math.max(0, Math.min(window.innerHeight, bottom) - Math.max(0, top));
  }

  function sourceIsVisible(position) {
    const source = position.sourceRect;
    if (!source) return false;
    return source.left < window.innerWidth && source.left + source.width > 0
      && source.top < window.innerHeight && source.top + source.height > 0;
  }

  function chooseLensPosition(current, placed, position) {
    const candidates = cornerCandidates(
      sourceDocumentRect(position), current.width, current.height,
    );
    if (!candidates.length) {
      candidates.push(Object.assign({}, current, { corner: null }));
    }
    const free = candidates.filter((candidate) =>
      placed.every((other) => !lensRectsOverlap(candidate, other)));
    if (!free.length) return null;

    const preferred = (items) => {
      if (position.corner) {
        const saved = items.find((candidate) => candidate.corner === position.corner);
        if (saved) return saved;
      }
      return items.slice().sort((a, b) => {
        const da = Math.hypot(a.left - current.left, a.top - current.top);
        const db = Math.hypot(b.left - current.left, b.top - current.top);
        return da - db;
      })[0];
    };

    // While the source is visible, prefer a fully visible corner. This is
    // what prevents a lens near the screen edge from oscillating off-screen.
    if (sourceIsVisible(position)) {
      const fullyVisible = free.filter(candidateFitsViewport);
      if (fullyVisible.length) return preferred(fullyVisible);
      const partiallyVisible = free.filter((candidate) => candidateVisibleArea(candidate) > 0);
      if (partiallyVisible.length) {
        return partiallyVisible.slice().sort((a, b) =>
          candidateVisibleArea(b) - candidateVisibleArea(a))[0];
      }
    }

    // Once the source itself has scrolled away, preserve its selected corner
    // so the popup follows it and returns from the same side.
    return preferred(free);
  }

  function applyLensPosition(popup, rect) {
    const documentPosition = popup.classList.contains("cv-lens--frozen");
    const scrollX = window.scrollX || 0;
    const scrollY = window.scrollY || 0;
    popup.style.left = `${rect.left - (documentPosition ? 0 : scrollX)}px`;
    popup.style.top = `${rect.top - (documentPosition ? 0 : scrollY)}px`;
  }

  // Frozen popups are authoritative: each one uses one of the four corners
  // around its source square. If that angle collides, choose another corner;
  // the live hover popup is the flexible one and moves around frozen popups.
  function resolveLensOverlaps() {
    const placed = [];
    const frozen = [...frozenLenses.values()]
      .sort((a, b) => (a.order || 0) - (b.order || 0));
    for (const position of frozen) {
      if (position.popup.style.display === "none") continue;
      const current = popupDocumentRect(position.popup);
      const chosen = chooseLensPosition(current, placed, position);
      // If every corner is occupied, keep the newest popup source-attached
      // and let its higher z-index make it the visible one.
      const fallback = chooseLensPosition(current, [], position) || current;
      const finalPosition = chosen || fallback;
      position.corner = finalPosition.corner || null;
      applyLensPosition(position.popup, finalPosition);
      placed.push(finalPosition);
    }
    if (hoverLens.popup.style.display !== "none") {
      const chosen = chooseLensPosition(
        popupDocumentRect(hoverLens.popup), placed, lensState,
      );
      if (chosen) applyLensPosition(hoverLens.popup, chosen);
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
    rememberPublicationRegion(fr, event.clientX, event.clientY);
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
    rememberPublicationRegion(fr, event.clientX, event.clientY);
    refreshLens(fr);
    resolveLensOverlaps();
  }

  function zoomLens(fr, event) {
    event.preventDefault();
    event.stopPropagation();
    if (Math.abs(event.deltaX) > Math.abs(event.deltaY)) {
      const rec = state.shown.get(fr.tier);
      const group = rec && rec.transferGroup ? rec.transferGroup : "default";
      const transfer = transferFor(rec);
      transfer.gain = Math.max(0.1, Math.min(10, transfer.gain * Math.exp(-event.deltaX * 0.002)));
      if (group === "default") state.gain = transfer.gain;
      const brightnessControl = brightnessControls.get(group);
      if (brightnessControl) {
        const input = brightnessControl.querySelector("input");
        const output = brightnessControl.querySelector(".cv-val");
        input.value = Math.round(1000 * (Math.log(transfer.gain) - Math.log(0.1))
          / (Math.log(10) - Math.log(0.1)));
        output.textContent = `${transfer.gain.toFixed(2)}×`;
      }
      rerender();
      notify();
      return;
    }
    if (frozenLenses.has(fr)) return;
    if (lensState.frame !== fr) enterLens(fr, event);
    const factor = event.deltaY < 0 ? LENS_ZOOM_STEP : 1 / LENS_ZOOM_STEP;
    lensState.zoom = Math.max(LENS_MIN_ZOOM,
      Math.min(LENS_MAX_ZOOM, lensState.zoom * factor));
    const angularSide = currentAngularSide(fr, lensState.zoom);
    if (angularSide != null) lensState.angularSide = angularSide;
    const relativeSide = currentRelativeSide(fr, lensState.zoom);
    if (relativeSide != null) lensState.relativeSide = relativeSide;
    rememberPublicationRegion(fr, event.clientX, event.clientY);
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
    const order = ++frozenSequence;
    popup.popup.style.zIndex = String(1000 + order);
    const frozen = {
      u, v,
      zoom: lensState.frame === fr ? lensState.zoom : LENS_DEFAULT_ZOOM,
      angularSide: lensState.frame === fr ? lensState.angularSide : null,
      relativeSide: lensState.frame === fr ? lensState.relativeSide : null,
      corner: null,
      order,
      popup: popup.popup, canvas: popup.canvas, label: popup.label, ctx: popup.ctx,
    };
    publicationRegion = {
      u,
      v,
      zoom: frozen.zoom,
      angularSide: frozen.angularSide,
      relativeSide: frozen.relativeSide,
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
  function rebuildFrames({ preserveFrozen = false } = {}) {
    const want = orderedSelected().map((t) => t.key);
    const have = state.frames.map((f) => f.tier);
    if (want.length === have.length && want.every((t, i) => t === have[i])) return;
    const frozenSnapshot = preserveFrozen ? snapshotFrozenLenses() : null;
    clearAllLenses();
    stopMorph();               // frames are being recreated → drop the old loop
    framesRow.innerHTML = "";
    state.frames = want.map((t) => {
      const fr = makeFrame(t);
      framesRow.appendChild(fr.frame);
      return fr;
    });
    framesRow.classList.toggle("cv-multi", state.frames.length > 1);
    restoreFrozenLenses(frozenSnapshot);
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
    // The layout control is intentionally a column limit rather than a CSS
    // reordering trick: exports and the frame-size calculation then match the
    // exact layout the user sees.  On narrow screens retain the single-column
    // fallback for legibility.
    const forcedColumns = window.innerWidth <= 760 ? 1
      : state.layout === "two-rows" ? Math.ceil(count / 2) : count;
    let best = { side: 0, columns: 1 };

    // The control chooses an exact layout.  Do not let the former
    // tile-size optimiser silently replace the requested one-row layout with
    // two columns on a shorter viewport.
    const columns = forcedColumns;
    const rows = Math.ceil(count / columns);
    const widthLimit = (width - columnGap * (columns - 1)) / columns;
    const heightLimit = (viewportHeight - rowGap * (rows - 1)) / rows;
    const side = Math.floor(Math.min(widthLimit, heightLimit));
    if (side > best.side || (side === best.side && columns > best.columns)) {
      best = { side, columns };
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
  async function show({ preserveFrozen = false } = {}) {
    if (!preserveFrozen) clearAllLenses();
    if (!state.meta || state.meta.count === 0) {
      rebuildFrames({ preserveFrozen });
      for (const fr of state.frames) {
        setFrameMsg(fr, state.meta?.empty_label || "No cutouts available.");
      }
      updateNav();
      return;
    }
    state.index = Math.max(0, Math.min(state.index, state.meta.count - 1));
    if (!jwstBandAvailable(state.params.jwst_band || "colour")) {
      state.params.jwst_band = "colour";
      syncChips();
    }
    stopMorph();
    rebuildFrames({ preserveFrozen });
    state.shown.clear();
    await Promise.all(state.frames.map(async (fr) => {
      if (!tierAvail(fr.tier)) { setFrameMsg(fr, missingTierLabel(fr.tier)); return; }
      if (fr.tier === "morph") { await startMorph(fr, state.index); return; }
      fr.frame.classList.add("cv-loading");
      try {
        const rec = await fetchCube(fr.tier, state.index);
        state.shown.set(fr.tier, rec);
        renderInto(fr, rec);
      } catch (e) {
        setFrameMsg(fr, tierAvail(fr.tier) ? (state.meta.missing_tier_labels?.[fr.tier] || "not synced yet")
          : missingTierLabel(fr.tier));
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
        setFrameMsg(fr, tierAvail(fr.tier) ? (state.meta.missing_tier_labels?.[fr.tier] || "not synced yet")
          : missingTierLabel(fr.tier));
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
      else if (g === "layout") on = c.dataset.value === state.layout;
      else on = c.dataset.value === String(state.params[g]);
      c.classList.toggle("active", on);
      if (g === "tier") c.classList.toggle("cv-disabled", tierDisabled(c.dataset.value));
      if (g === "jwst_band") c.classList.toggle("cv-disabled", !jwstBandAvailable(c.dataset.value));
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
      show({ preserveFrozen: true });
    } else if (group === "color") {
      state.color = value;
      syncChips();
      rerender();
      notify();
    } else if (group === "layout") {
      state.layout = value === "two-rows" ? "two-rows" : "one-row";
      saveViewLayout(state.layout);
      syncChips();
      requestAnimationFrame(fitFramesToViewport);
      notify();
    } else {                                  // param control (e.g. sky subset)
      if (group === "jwst_band" && !jwstBandAvailable(value)) return;
      if (group === "jwst_band") {
        if (value === "temperature") state.color = "temp";
        else if (state.color === "temp") state.color = state.meta.band_names[0];
      }
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

  function transferGroupLabel(group) {
    return group === "jwst" ? "JWST" : group === "euclid" ? "Euclid" : "";
  }

  function buildToolbar() {
    toolbar.innerHTML = "";
    brightnessControls.clear();
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
    if (shown.length >= 3) {
      toolbar.append(el("span", { class: "cv-grouplabel", text: "Layout" }));
      const layoutGroup = el("div", { class: "cv-group" });
      layoutGroup.append(
        chip("layout", "one-row", "one row", "place all selected images side by side"),
        chip("layout", "two-rows", "two rows", "place selected images across two rows"),
      );
      toolbar.append(layoutGroup);
    }
    for (const pc of (opts.paramControls || [])) {
      toolbar.append(el("span", { class: "cv-grouplabel", text: pc.label }));
      const g = el("div", { class: "cv-group" });
      pc.options.forEach((o) => g.append(chip(pc.key, o.value, o.label)));
      toolbar.append(g);
    }
    const jwstOptions = state.meta.jwst_band_options || [];
    if (jwstOptions.length > 1) {
      toolbar.append(el("span", { class: "cv-grouplabel", text: "JWST band" }));
      const jwstGroup = el("div", { class: "cv-group" });
      jwstOptions.forEach((option) => jwstGroup.append(chip(
        "jwst_band", option.value, option.label,
        option.value === "colour" ? "all available filters as display colour" : `show native ${option.label}`,
      )));
      toolbar.append(jwstGroup);
    }
    const logMode = state.meta.render_mode === "log";
    if (!logMode) {
      toolbar.append(el("span", { class: "cv-grouplabel", text: state.meta.color_label || "Colour" }));
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
    const transferGroups = transferGroupKeys();
    for (const group of transferGroups) {
      const transfer = transferSetting(group);
      const prefix = transferGroups.length > 1 ? `${transferGroupLabel(group)} ` : "";
      if (!logMode) {
        sg.append(slider(`${prefix}asinh knee`, 5, 5000, transfer.knee,
          (v) => `${Math.round(v)} e⁻`,
          (v) => {
            transfer.knee = v;
            if (group === "default") state.knee = v;
            rerender();
          }));
      }
      const brightnessControl = slider(`${prefix}brightness`, 0.1, 10, transfer.gain,
        (v) => `${v.toFixed(2)}×`,
        (v) => {
          transfer.gain = v;
          if (group === "default") state.gain = v;
          rerender();
        });
      brightnessControls.set(group, brightnessControl);
      sg.append(brightnessControl);
    }
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
    const figure = el("button", { class: "cv-navbtn cv-figure", type: "button", text: "⬇ Figure",
      title: "Export a high-resolution publication plate. The last inspected region becomes a matched bottom-left inset in every panel.",
      onclick: savePublicationFigure });
    const rec = el("button", { class: "cv-navbtn cv-rec", type: "button", text: "⏺ video",
      title: "Record the current view (all selected tiers, side by side) — click to start, click again to stop and download a .webm clip",
      onclick: () => toggleRecord(rec) });
    nav.append(prev, idxWrap, next, play, speed, hint, png, figure, rec);
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

  function niceAngularScale(fullSideArcsec) {
    if (!(fullSideArcsec > 0)) return null;
    const target = fullSideArcsec * 0.2;
    const candidates = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200];
    let best = candidates[0];
    for (const value of candidates) if (value <= target) best = value;
    return best;
  }

  function publicationCrop(fr) {
    const region = publicationRegion;
    if (!region || !(fr.canvas.width > 1) || !(fr.canvas.height > 1)) return null;
    const extent = Math.min(fr.canvas.width, fr.canvas.height);
    const pixscale = validPixscale(fr);
    let sourceSide = pixscale && region.angularSide > 0
      ? region.angularSide / pixscale
      : region.relativeSide > 0 ? region.relativeSide * extent : extent / LENS_DEFAULT_ZOOM;
    sourceSide = Math.max(1, Math.min(extent, sourceSide));
    const cx = Math.max(sourceSide / 2,
      Math.min(fr.canvas.width - sourceSide / 2, region.u * fr.canvas.width));
    const cy = Math.max(sourceSide / 2,
      Math.min(fr.canvas.height - sourceSide / 2, region.v * fr.canvas.height));
    return {
      x: cx - sourceSide / 2,
      y: cy - sourceSide / 2,
      side: sourceSide,
      zoom: Math.max(1, Math.min(fr.canvas.width, fr.canvas.height) / sourceSide),
      angularSide: pixscale ? sourceSide * pixscale : null,
    };
  }

  function publicationPanelName(fr) {
    const tier = String(fr.tier || "").toLowerCase();
    const label = String(tierMeta(fr.tier)?.label || fr.tier || "Image");
    const normalized = label.trim().toLowerCase();
    if (tier === "lr" || tier === "dirty" || normalized === "lr") return "Euclid Image";
    if (tier === "sr" || normalized === "sr") return "Super-resolved Image";
    return label;
  }

  function publicationBandLabel(fr) {
    const rec = state.shown.get(fr.tier);
    if (state.color === "lupton") return "Lupton RGB";
    if (state.color === "temp") return "temperature composite";
    if (Array.isArray(rec?.bands) && rec.bands.length === 1) return rec.bands[0];
    return state.color;
  }

  function publicationDisplayInfo(fr) {
    const rec = state.shown.get(fr.tier);
    const transfer = transferFor(rec);
    return {
      band: publicationBandLabel(fr),
      gain: transfer.gain,
      knee: transfer.knee,
      log: state.meta?.color?.render_mode === "log",
    };
  }

  function publicationElectronLabel(value) {
    if (value >= 1000) return `${(value / 1000).toFixed(value >= 10000 ? 0 : 1)}k`;
    if (value >= 10) return `${Math.round(value)}`;
    if (value >= 1) return value.toFixed(1);
    return value.toFixed(2);
  }

  function drawPublicationHeatbar(ctx, info, x, imageBottom, side) {
    const parameterSize = Math.max(23, side * 0.022);
    ctx.fillStyle = PUBLICATION_INK;
    ctx.font = `400 ${parameterSize}px Arial, Helvetica, sans-serif`;
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    const parameterText = info.log
      ? `Band: ${info.band}  ·  logarithmic display`
      : `Band: ${info.band}  ·  asinh knee: ${Math.round(info.knee)} e⁻`;
    ctx.fillText(parameterText, x, imageBottom + side * 0.034);

    const barWidth = side * 0.56;
    const barHeight = Math.max(20, side * 0.018);
    const barX = x + (side - barWidth) / 2;
    const barY = imageBottom + side * 0.060;
    const gradient = ctx.createLinearGradient(barX, 0, barX + barWidth, 0);
    gradient.addColorStop(0, "#000000");
    gradient.addColorStop(1, "#ffffff");
    ctx.fillStyle = gradient;
    ctx.fillRect(barX, barY, barWidth, barHeight);
    ctx.strokeStyle = PUBLICATION_INK;
    ctx.lineWidth = 1;
    ctx.strokeRect(barX, barY, barWidth, barHeight);

    const fractions = [0, 0.25, 0.5, 0.75, 1];
    const norm = Math.max(Math.asinh((30 * state.K0) / Math.max(info.knee, 1e-30)), 1e-6);
    const tickSize = Math.max(18, side * 0.017);
    ctx.font = `400 ${tickSize}px Arial, Helvetica, sans-serif`;
    ctx.fillStyle = PUBLICATION_INK;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const fraction of fractions) {
      const tickX = barX + fraction * barWidth;
      ctx.beginPath();
      ctx.moveTo(tickX, barY + barHeight);
      ctx.lineTo(tickX, barY + barHeight + side * 0.009);
      ctx.stroke();
      const value = info.log
        ? fraction
        : Math.sinh(fraction * norm) * info.knee / Math.max(info.gain, 1e-30);
      const label = info.log ? fraction.toFixed(2) : publicationElectronLabel(value);
      ctx.fillText(label, tickX, barY + barHeight + side * 0.013);
    }
    ctx.font = `400 ${tickSize}px Arial, Helvetica, sans-serif`;
    ctx.textBaseline = "top";
    ctx.fillText(info.log ? "relative display intensity" : "input-equivalent signal (e⁻)",
      barX + barWidth / 2, barY + barHeight + side * 0.045);
  }

  function drawPublicationPanel(ctx, fr, x, y, side) {
    ctx.save();
    ctx.fillStyle = PUBLICATION_INK;
    ctx.font = `600 ${Math.max(32, side * 0.030)}px Arial, Helvetica, sans-serif`;
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.fillText(publicationPanelName(fr), x, y - side * 0.014);

    ctx.fillStyle = "#05070d";
    ctx.fillRect(x, y, side, side);
    // Keep detector/reconstruction pixels literal. Browser interpolation can
    // make an upscaled science image look smoother than the sampled data.
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(fr.canvas, x, y, side, side);

    const crop = publicationCrop(fr);
    if (crop) {
      const cropFraction = crop.side / Math.min(fr.canvas.width, fr.canvas.height);
      if (cropFraction < 0.5) {
        const sx = x + crop.x / fr.canvas.width * side;
        const sy = y + crop.y / fr.canvas.height * side;
        const sw = crop.side / fr.canvas.width * side;
        const sh = crop.side / fr.canvas.height * side;
        ctx.strokeStyle = "rgba(0, 0, 0, .88)";
        ctx.lineWidth = Math.max(4, side * 0.004);
        ctx.strokeRect(sx, sy, sw, sh);
        ctx.strokeStyle = "rgba(255, 255, 255, .92)";
        ctx.lineWidth = Math.max(1.5, side * 0.0015);
        ctx.setLineDash([Math.max(5, side * 0.006), Math.max(4, side * 0.004)]);
        ctx.strokeRect(sx, sy, sw, sh);
        ctx.setLineDash([]);
      }

      // The inset is deliberately embedded at bottom-left in every panel.
      // This makes the comparison survive slide reflow and keeps all panels
      // registered to the same cursor-selected sky location.
      const inset = Math.round(side * 0.255);
      const pad = Math.round(side * 0.022);
      const ix = x + pad;
      const iy = y + side - inset - pad;
      ctx.fillStyle = "#05070d";
      const insetHalo = Math.max(4, side * 0.004);
      ctx.fillRect(ix - insetHalo, iy - insetHalo,
        inset + 2 * insetHalo, inset + 2 * insetHalo);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(fr.canvas, crop.x, crop.y, crop.side, crop.side,
        ix, iy, inset, inset);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = Math.max(2, side * 0.002);
      ctx.strokeRect(ix, iy, inset, inset);
      if (crop.angularSide > 0) {
        const angular = crop.angularSide < 1
          ? crop.angularSide.toFixed(2) : crop.angularSide.toFixed(1);
        const insetLabel = `${angular}″ × ${angular}″`;
        ctx.font = `600 ${Math.max(20, side * 0.020)}px Arial, Helvetica, sans-serif`;
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.lineWidth = Math.max(3, side * 0.0035);
        ctx.strokeStyle = "rgba(0, 0, 0, .9)";
        ctx.strokeText(insetLabel, ix + inset * 0.045, iy + inset * 0.04);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(insetLabel, ix + inset * 0.045, iy + inset * 0.04);
      }
    }

    const pixscale = validPixscale(fr);
    if (pixscale) {
      const barArcsec = niceAngularScale(fr.canvas.width * pixscale);
      const barWidth = barArcsec / (fr.canvas.width * pixscale) * side;
      const right = x + side - side * 0.035;
      const bottom = y + side - side * 0.035;
      ctx.strokeStyle = "rgba(0, 0, 0, .9)";
      ctx.lineWidth = Math.max(6, side * 0.007);
      ctx.beginPath(); ctx.moveTo(right - barWidth, bottom); ctx.lineTo(right, bottom); ctx.stroke();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = Math.max(2.5, side * 0.003);
      ctx.beginPath(); ctx.moveTo(right - barWidth, bottom); ctx.lineTo(right, bottom); ctx.stroke();
      ctx.font = `600 ${Math.max(20, side * 0.020)}px Arial, Helvetica, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.lineWidth = Math.max(3, side * 0.0035);
      ctx.strokeStyle = "rgba(0, 0, 0, .9)";
      const shownScale = barArcsec < 0.1 ? barArcsec.toFixed(2)
        : barArcsec < 1 ? barArcsec.toFixed(1) : barArcsec.toFixed(0);
      ctx.strokeText(`${shownScale}″`, right - barWidth / 2, bottom - side * 0.009);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(`${shownScale}″`, right - barWidth / 2, bottom - side * 0.009);
    }

    ctx.strokeStyle = "#111111";
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, side, side);
    drawPublicationHeatbar(ctx, publicationDisplayInfo(fr), x, y + side, side);
    ctx.restore();
  }

  function publicationFigureCanvas() {
    const frames = exportFrames().map(({ fr }) => fr);
    if (!frames.length) return null;
    const count = frames.length;
    const columns = state.layout === "two-rows" ? Math.ceil(count / 2) : count;
    const rows = Math.ceil(count / columns);
    const outer = 28;
    const gap = 14;
    const panelHeader = 64;
    const panelFooter = 184;
    const rowGap = 22;
    const preferredSide = count === 1 ? 1600 : count === 2 ? 1380 : 1260;
    const widthBound = Math.floor(
      (PUBLICATION_MAX_WIDTH - 2 * outer - gap * (columns - 1)) / columns,
    );
    const side = Math.max(640, Math.min(preferredSide, widthBound));
    const width = 2 * outer + columns * side + (columns - 1) * gap;
    const height = 2 * outer + rows * (panelHeader + side + panelFooter)
      + (rows - 1) * rowGap;
    const out = document.createElement("canvas");
    out.width = width;
    out.height = height;
    const ctx = out.getContext("2d");
    ctx.fillStyle = PUBLICATION_PAPER;
    ctx.fillRect(0, 0, width, height);

    frames.forEach((fr, index) => {
      const col = index % columns;
      const row = Math.floor(index / columns);
      drawPublicationPanel(ctx, fr,
        outer + col * (side + gap),
        outer + panelHeader + row * (panelHeader + side + panelFooter + rowGap), side);
    });
    return out;
  }

  function savePublicationFigure() {
    const out = publicationFigureCanvas();
    if (!out) return;
    out.toBlob((blob) => {
      if (blob) _download(blob, `${_stem()}_figure.png`);
    }, "image/png");
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
    const jwstOptions = state.meta.jwst_band_options || [];
    if (jwstOptions.length
        && !jwstOptions.some((option) => option.value === state.params.jwst_band)) {
      state.params.jwst_band = jwstOptions[0].value;
    }
    const defaultKnee = state.meta.color.default_asinh || 100;
    state.K0 = defaultKnee;
    state.knee = Number.isFinite(state.viewOverride.knee)
      ? state.viewOverride.knee : defaultKnee;
    state.gain = Number.isFinite(state.viewOverride.gain) ? state.viewOverride.gain : 1.0;
    resetTransferSettings();
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
      show({ preserveFrozen: true });
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
        for (const group of transferGroupKeys()) transferSetting(group).knee = next.knee;
      }
      if (Number.isFinite(next.gain) && next.gain > 0) {
        state.gain = next.gain;
        state.viewOverride.gain = next.gain;
        for (const group of transferGroupKeys()) transferSetting(group).gain = next.gain;
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
      if (state.tiers.includes("morph")) show({ preserveFrozen: true });
    },
    getIndex() { return state.index; },
    isReady() { return !!state.meta; },
    getState() {
      return { index: state.index, tier: state.tiers[0],
               tiers: state.tiers.slice(), color: state.color, layout: state.layout,
               knee: state.knee, gain: state.gain,
               transfers: copiedTransferSettings(),
               params: Object.assign({}, state.params) };
    },
    /** Download the same fixed-resolution, annotated figure as the viewer's
     *  Figure button. Useful to external React controls without DOM capture. */
    exportFigure() { savePublicationFigure(); },
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
