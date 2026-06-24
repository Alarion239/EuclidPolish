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

/** AB-flux normalisation: e⁻-over-stack → proportional AB flux density. */
function abFluxNorm(band) {
  return 1.0 / (band.t_total_s * Math.pow(10, 0.4 * band.zeropoint_ab));
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

export function mountCutoutViewer(root, opts = {}) {
  const collection = opts.collection;
  const state = {
    meta: null,
    params: Object.assign({}, opts.params || {}),
    index: opts.initialIndex || 0,
    tier: null,
    color: "VIS",          // band name | "lupton" | "temp"
    knee: 100,             // K (e⁻)
    gain: 1.0,             // brightness multiplier
    K0: 100,               // knee captured on first load → fixes the white ref
    cache: new Map(),      // "tier:index" → cube record
    prepared: null,        // memoised colour/intensity decomposition
    preparedKey: "",
    playTimer: null,
  };

  // --- DOM scaffold --------------------------------------------------------
  root.classList.add("cutout-viewer");
  root.innerHTML = "";

  const toolbar = el("div", { class: "cv-toolbar" });
  const frame = el("div", { class: "cv-frame" });
  const canvas = el("canvas", { class: "cv-canvas", width: 1, height: 1 });
  const legend = el("canvas", { class: "cv-legend", width: 14, height: 160 });
  const legendWrap = el("div", { class: "cv-legend-wrap" }, [
    el("span", { class: "cv-legend-tick cv-legend-top", text: "20k" }),
    legend,
    el("span", { class: "cv-legend-tick cv-legend-bot", text: "3k" }),
  ]);
  const overlay = el("div", { class: "cv-overlay" });
  const msg = el("div", { class: "cv-msg" });
  frame.append(canvas, legendWrap, overlay, msg);

  const nav = el("div", { class: "cv-nav" });
  root.append(toolbar, frame, nav);
  const ctx = canvas.getContext("2d");

  // --- helpers -------------------------------------------------------------
  const band = (name) => state.meta.color.bands[name];
  const cacheKey = (tier, index) => `${tier}:${index}`;

  function setMsg(text) {
    msg.textContent = text || "";
    msg.style.display = text ? "flex" : "none";
  }

  function notify() {
    if (opts.onIndexChange) opts.onIndexChange(state.index);
    if (opts.onChange) opts.onChange(api.getState());
  }

  async function fetchCube(tier, index) {
    const key = cacheKey(tier, index);
    if (state.cache.has(key)) return state.cache.get(key);
    const qs = new URLSearchParams(Object.assign({ tier }, state.params));
    const url = `/viewer/cube/${collection}/${index}?${qs}`;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`cube ${r.status}`);
    const shape = (r.headers.get("X-Cube-Shape") || "").split(",").map(Number);
    const buf = await r.arrayBuffer();
    const rec = {
      h: shape[0], w: shape[1], c: shape[2],
      data: new Float32Array(buf),
      label: r.headers.get("X-Cube-Label") || "",
      asinh: parseFloat(r.headers.get("X-Cube-Asinh")) || 100,
      pixscale: parseFloat(r.headers.get("X-Cube-Pixscale")) || 0,
    };
    state.cache.set(key, rec);
    if (state.cache.size > 80) state.cache.delete(state.cache.keys().next().value);
    return rec;
  }

  function prefetch(index) {
    const tiers = (state.meta.tiers || []).map((t) => t.key);
    const jobs = [];
    for (const di of [1, -1, 2]) {
      const j = index + di;
      if (j >= 0 && j < state.meta.count) jobs.push([state.tier, j]);
    }
    for (const t of tiers) if (t !== state.tier) jobs.push([t, index]);
    for (const [t, j] of jobs) fetchCube(t, j).catch(() => {});
  }

  // --- colour prepare (per cube + colour mode) -----------------------------
  // Returns { mode, factor, I, ... } where I is per-pixel linear intensity in
  // the same units `factor` converts the e⁻ sliders into. Heavy work (temp
  // fit) happens here, then transfer() is cheap on every slider tick.
  function prepare(rec) {
    const key = `${cacheKey(state.tier, state.index)}|${state.color}`;
    if (state.preparedKey === key && state.prepared) return state.prepared;

    const C = state.meta.color;
    const names = C.band_names;
    const npx = rec.h * rec.w;
    // value of band k at pixel p inside the interleaved (H,W,C) buffer
    const at = (p, k) => rec.data[p * rec.c + k];

    let prepared;
    if (state.color === "lupton" || state.color === "temp") {
      const useSolar = state.color === "lupton";
      // AB (+ solar) calibrated channels.
      const calib = [];
      for (let k = 0; k < names.length; k++) {
        const b = band(names[k]);
        let f = abFluxNorm(b);
        if (useSolar) f *= solarBalance(b);
        calib.push(f);
      }
      if (state.color === "lupton") {
        // RGB scheme [H_E, J_E, VIS] → R,G,B; intensity = mean of the three.
        const sel = C.rgb_scheme.map((n) => names.indexOf(n));
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
        // Temperature: per-pixel blackbody-T fit → Planckian-locus hue.
        const lam = names.map((n) => band(n).pivot_um);
        const ts = eyeTGrid(96);
        // Pre-normalise each grid temperature's Planck vector.
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
      // Single band grayscale → I = that band's raw electrons.
      const k = names.indexOf(state.color);
      const kk = k < 0 ? 0 : k;
      const I = new Float32Array(npx);
      for (let p = 0; p < npx; p++) I[p] = at(p, kk);
      prepared = { mode: "gray", I, factor: 1.0 };
    }
    prepared.h = rec.h; prepared.w = rec.w; prepared.npx = npx;
    state.prepared = prepared; state.preparedKey = key;
    return prepared;
  }

  // --- transfer (per slider tick) → canvas ---------------------------------
  function render(rec) {
    const prep = prepare(rec);
    const { h, w, npx, I, factor } = prep;
    // Kc = knee·factor; white reference Wref = 30·K0·factor (e⁻). norm cancels
    // factor → asinh(30·K0/knee); at knee=K0, gain=1 this matches eye_rgb.
    const Kc = Math.max(state.knee * factor, 1e-30);
    const norm = Math.max(Math.asinh((30.0 * state.K0 * factor) / Kc), 1e-6);
    const G = state.gain;

    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const img = ctx.createImageData(w, h);
    const out = img.data;

    if (prep.mode === "gray") {
      for (let p = 0; p < npx; p++) {
        const t = asinhTransfer(I[p], G, Kc, norm);
        const v = (t * 255) | 0;
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
    ctx.putImageData(img, 0, 0);
    legendWrap.style.display = prep.mode === "temp" ? "flex" : "none";
  }

  function renderLegend() {
    const n = 160;
    const img = legend.getContext("2d").createImageData(14, n);
    const lo = Math.log(3000), hi = Math.log(20000);
    for (let row = 0; row < n; row++) {
      // top (row 0) = hot (20k), bottom = cool (3k)
      const T = Math.exp(hi - (hi - lo) * (row / (n - 1)));
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
    legend.getContext("2d").putImageData(img, 0, 0);
  }

  // --- load + show current cube --------------------------------------------
  async function show() {
    if (!state.meta || state.meta.count === 0) { setMsg("No cutouts available."); return; }
    state.index = Math.max(0, Math.min(state.index, state.meta.count - 1));
    frame.classList.add("cv-loading");
    try {
      const rec = await fetchCube(state.tier, state.index);
      setMsg("");
      render(rec);
      overlay.textContent = rec.label;
      updateNav();
      prefetch(state.index);
      notify();
    } catch (e) {
      setMsg(tierAvailable() ? `Could not load cutout (${e.message}).`
                             : "Not synced yet — pull the data first.");
    } finally {
      frame.classList.remove("cv-loading");
    }
  }

  // Re-render the current (already-cached) cube without a fetch — slider ticks.
  function rerender() {
    const rec = state.cache.get(cacheKey(state.tier, state.index));
    if (rec) render(rec);
  }

  function tierAvailable() {
    const obj = state.meta.objects && state.meta.objects[state.index];
    return !obj || !obj.tiers || obj.tiers.includes(state.tier);
  }

  // --- toolbar / nav building ----------------------------------------------
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
      const cur = g === "tier" ? state.tier : g === "color" ? state.color : state.params[g];
      c.classList.toggle("active", c.dataset.value === String(cur));
      if (g === "tier") {
        const obj = state.meta.objects && state.meta.objects[state.index];
        const avail = !obj || !obj.tiers || obj.tiers.includes(c.dataset.value);
        c.classList.toggle("cv-disabled", !avail);
      }
    });
  }

  function onChip(group, value) {
    if (group === "tier") { state.tier = value; state.preparedKey = ""; show(); }
    else if (group === "color") { state.color = value; state.preparedKey = ""; rerender(); notify(); }
    else { // param (e.g. subset) → reload meta
      state.params[group] = value;
      loadMeta().then(show);
    }
    syncChips();
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
    const wrap = el("div", { class: "cv-slider" }, [
      el("label", { text: label }), input, out,
    ]);
    wrap._set = (v) => { input.value = toSlider(v); out.textContent = fmt(v); };
    return wrap;
  }

  let kneeSlider, brightSlider;
  function buildToolbar() {
    toolbar.innerHTML = "";
    const tiers = state.meta.tiers || [];
    if (tiers.length > 1) {
      toolbar.append(el("span", { class: "cv-grouplabel", text: "Tier" }));
      const g = el("div", { class: "cv-group" });
      tiers.forEach((t) => g.append(chip("tier", t.key, t.label)));
      toolbar.append(g);
    }
    // Param controls (e.g. sky subset).
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
    kneeSlider = slider("asinh knee", 5, 5000, state.knee,
      (v) => `${v < 100 ? v.toFixed(0) : Math.round(v)} e⁻`,
      (v) => { state.knee = v; rerender(); });
    brightSlider = slider("brightness", 0.1, 10, state.gain,
      (v) => `${v.toFixed(2)}×`,
      (v) => { state.gain = v; rerender(); });
    sg.append(kneeSlider, brightSlider);
    toolbar.append(sg);
  }

  function buildNav() {
    nav.innerHTML = "";
    const prev = el("button", { class: "cv-navbtn", type: "button", text: "◀", title: "Previous (←)",
      onclick: () => go(state.index - 1) });
    const next = el("button", { class: "cv-navbtn", type: "button", text: "▶", title: "Next (→)",
      onclick: () => go(state.index + 1) });
    const play = el("button", { class: "cv-navbtn cv-play", type: "button", text: "▶▶", title: "Run through (Space)",
      onclick: togglePlay });
    const idx = el("span", { class: "cv-idx" });
    nav._idx = idx; nav._play = play;
    const hint = el("span", { class: "cv-kbd", text: "← →  ·  Space to run" });
    nav.append(prev, idx, next, play, hint);
  }

  function updateNav() {
    if (nav._idx) nav._idx.textContent = `${state.index} / ${Math.max(0, state.meta.count - 1)}`;
  }

  function go(i) {
    const n = state.meta.count;
    if (n === 0) return;
    state.index = (i + n) % n;     // wrap for smooth run-through
    state.preparedKey = "";
    show();
    syncChips();
  }
  function togglePlay() {
    if (state.playTimer) {
      clearInterval(state.playTimer); state.playTimer = null;
      nav._play.classList.remove("active");
    } else {
      nav._play.classList.add("active");
      state.playTimer = setInterval(() => go(state.index + 1), 450);
    }
  }

  function onKey(e) {
    // Only when the pointer/focus is on the viewer, so arrow/space don't
    // hijack pages that embed it alongside other content (e.g. /evaluation).
    if (!state.hot) return;
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
    if (!state.tier || !state.meta.tiers.some((t) => t.key === state.tier)) {
      state.tier = state.meta.default_tier;
    }
    if (!state.meta.band_names.includes(state.color)
        && !["lupton", "temp"].includes(state.color)) {
      state.color = state.meta.band_names[0];
    }
    state.knee = state.meta.color.default_asinh || 100;
    state.K0 = state.knee;
    state.cache.clear();
    buildToolbar();
    buildNav();
    syncChips();
    renderLegend();
  }

  document.addEventListener("keydown", onKey);

  const api = {
    goTo(i) { go(i); },
    getIndex() { return state.index; },
    getState() {
      return { index: state.index, tier: state.tier, color: state.color,
               params: Object.assign({}, state.params) };
    },
    reload() { return loadMeta().then(show); },
    destroy() {
      document.removeEventListener("keydown", onKey);
      if (state.playTimer) clearInterval(state.playTimer);
      root.innerHTML = "";
    },
  };

  // initial load
  setMsg("Loading…");
  loadMeta().then(show).catch((e) => setMsg(`Could not load viewer (${e.message}).`));
  return api;
}

/** Pure colour primitives, exported for the parity check (see
 *  scripts/check_viewer_parity.mjs). Not part of the public viewer API. */
export const _internals = {
  abFluxNorm, solarBalance, planckFnu, planckianXY,
  xyToLinearSrgb, srgbGamma, eyeTGrid, asinhTransfer,
};
