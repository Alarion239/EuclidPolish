/**
 * Ensemble disagreement animation — a "morphing movie" of an SR field.
 *
 * The ensemble's per-member disagreement lives in a tiny subspace: the members
 * minus their mean span (M-1) dims. The server PCAs that per field and caches
 * the mean (SR) + top-K unit eigen-images (pca0..) + each component's amplitude
 * aᵢ. This widget fetches them and renders, at 60fps in a canvas:
 *
 *     frame(t) = mean + Σ aᵢ · amp · sin(2π fᵢ t + φᵢ) · componentᵢ
 *
 * a smooth loop through the real (spatially-correlated) disagreement subspace —
 * hallucinated regions morph, LR-constrained structure stays still.
 *
 * Rendering is single-band + a colour LUT (cheap; no per-frame colour fit), with
 * band and colormap selectable. Mounted alongside the field viewer.
 */

const COLLECTION = "ensemble";
const FREQS = [1, 2, 3];
const PHASES = [0, Math.PI / 2, Math.PI / 3];

//: Compact control-point colormaps (interpolated per pixel — a few ops each).
const CMAPS = {
  gray: [[0, 0, 0], [255, 255, 255]],
  magma: [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
          [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 253, 191]],
  temperature: [[9, 30, 90], [40, 110, 200], [220, 235, 245],
                [240, 150, 70], [200, 30, 30]],   // cool → white → hot
};
function cmap(stops, v) {
  const n = stops.length - 1;
  const x = (v < 0 ? 0 : v > 1 ? 1 : v) * n;
  const i = Math.min(n - 1, x | 0);
  const f = x - i, a = stops[i], b = stops[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

function el(tag, attrs = {}, kids = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "style") e.setAttribute("style", v);
    else if (k in e) e[k] = v; else e.setAttribute(k, v);
  }
  for (const c of kids) e.append(c);
  return e;
}

async function fetchCube(index, tier) {
  const r = await fetch(`/viewer/cube/${COLLECTION}/${index}?tier=${tier}`);
  if (!r.ok) throw new Error(`${tier} ${r.status}`);
  const shape = (r.headers.get("X-Cube-Shape") || "").split(",").map(Number);
  const asinh = parseFloat(r.headers.get("X-Cube-Asinh")) || 100;
  return { data: new Float32Array(await r.arrayBuffer()),
           h: shape[0], w: shape[1], c: shape[2], asinh };
}

export async function mountEnsembleAnim(root, opts = {}) {
  const canvas = el("canvas", { style: "max-width:100%;height:auto;image-rendering:pixelated;background:#000;border-radius:6px;" });
  const ctx = canvas.getContext("2d");
  const msg = el("div", { class: "muted", style: "font-size:12px;margin-top:4px;" });
  const idxLabel = el("span", { class: "muted", style: "min-width:9ch;display:inline-block;" });

  const prev = el("button", { type: "button", textContent: "‹ prev" });
  const next = el("button", { type: "button", textContent: "next ›" });
  const playBtn = el("button", { type: "button", textContent: "⏸ pause" });
  const speed = el("input", { type: "range", min: "0.1", max: "2", step: "0.05", value: "0.5", style: "width:110px;" });
  const amp = el("input", { type: "range", min: "0", max: "3", step: "0.1", value: "1.6", style: "width:110px;" });
  const bandSel = el("select");
  const cmapSel = el("select");
  for (const [k, label] of [["gray", "grayscale"], ["magma", "magma"], ["temperature", "temperature"]])
    cmapSel.append(el("option", { value: k, textContent: label }));

  const controls = el("div", { style: "display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:8px;" }, [
    prev, next, idxLabel, playBtn,
    el("label", { style: "font-size:12px;" }, [document.createTextNode("band "), bandSel]),
    el("label", { style: "font-size:12px;" }, [document.createTextNode("colour "), cmapSel]),
    el("label", { style: "font-size:12px;" }, [document.createTextNode("speed "), speed]),
    el("label", { style: "font-size:12px;", title: "morph amplitude in units of member σ" },
      [document.createTextNode("amplitude "), amp]),
  ]);
  root.replaceChildren(canvas, controls, msg);

  let meta = null;
  let index = Number.isFinite(opts.index) ? opts.index : 0;
  let raw = null;      // { sr:{data,h,w,c}, comps:[{data,c}], amps:[] } — full 4-band
  let field = null;    // { h, w, mean, comps:[Float32], norm, Kc } — chosen band
  let playing = true;
  let t0 = performance.now();

  async function loadMeta() {
    const r = await fetch(`/viewer/meta/${COLLECTION}`);
    meta = r.ok ? await r.json() : null;
    if (meta && meta.band_names) {
      bandSel.replaceChildren(...meta.band_names.map((n, i) =>
        el("option", { value: String(i), textContent: n })));
    }
    return meta && meta.count ? meta : null;
  }

  // Extract the currently-selected band into single-band arrays + norm.
  function setBand() {
    if (!raw) return;
    const bi = Math.min(parseInt(bandSel.value || "0", 10) || 0, raw.sr.c - 1);
    const { h, w, c } = raw.sr;
    const pull = (cube) => { const o = new Float32Array(h * w); for (let p = 0; p < h * w; p++) o[p] = cube.data[p * c + bi]; return o; };
    const mean = pull(raw.sr);
    const comps = raw.comps.map(pull);
    const Kc = raw.sr.asinh || 100;
    let mx = 1e-6;
    for (let p = 0; p < mean.length; p++) { const a = Math.asinh(mean[p] / Kc); if (a > mx) mx = a; }
    field = { h, w, mean, comps, norm: mx, Kc };
    canvas.width = w; canvas.height = h;
  }

  async function loadField(i) {
    msg.textContent = "loading…";
    const n = meta.pca_n | 0;
    const sr = await fetchCube(i, "sr");
    const comps = [];
    for (let k = 0; k < n; k++) { try { comps.push(await fetchCube(i, `pca${k}`)); } catch { /* <2 members */ } }
    raw = { sr, comps, amps: (meta.pca_amps && meta.pca_amps[i]) || [] };
    setBand();
    idxLabel.textContent = `field ${i + 1}/${meta.count}`;
    msg.textContent = comps.length
      ? `morphing through ${comps.length} disagreement component(s)`
      : "single member — nothing to morph";
    t0 = performance.now();
  }

  let imageData = null;
  function draw(now) {
    if (field) {
      if (!imageData || imageData.width !== field.w) imageData = ctx.createImageData(field.w, field.h);
      const { mean, comps, norm, Kc, w, h } = field;
      const amps = raw.amps;
      const t = ((now - t0) / 1000) * parseFloat(speed.value);
      const A = parseFloat(amp.value);
      const stops = CMAPS[cmapSel.value] || CMAPS.gray;
      const coeff = comps.map((_, k) =>
        (amps[k] || 0) * A * Math.sin(2 * Math.PI * FREQS[k % FREQS.length] * t + PHASES[k % PHASES.length]));
      const px = imageData.data;
      for (let p = 0; p < w * h; p++) {
        let v = mean[p];
        for (let k = 0; k < comps.length; k++) v += coeff[k] * comps[k][p];
        let g = Math.asinh(v / Kc) / norm;
        g = g < 0 ? 0 : g > 1 ? 1 : g;
        const rgb = cmap(stops, g), o = p * 4;
        px[o] = rgb[0]; px[o + 1] = rgb[1]; px[o + 2] = rgb[2]; px[o + 3] = 255;
      }
      ctx.putImageData(imageData, 0, 0);
    }
    if (playing) requestAnimationFrame(draw);
  }

  const setPlaying = (on) => {
    playing = on;
    playBtn.textContent = on ? "⏸ pause" : "▶ play";
    if (on) { t0 = performance.now(); requestAnimationFrame(draw); }
  };
  playBtn.onclick = () => setPlaying(!playing);
  prev.onclick = async () => { index = (index - 1 + meta.count) % meta.count; await loadField(index); if (!playing) draw(performance.now()); };
  next.onclick = async () => { index = (index + 1) % meta.count; await loadField(index); if (!playing) draw(performance.now()); };
  bandSel.onchange = () => { setBand(); if (!playing) draw(performance.now()); };
  for (const c of [speed, amp, cmapSel]) c.oninput = c.onchange = () => { if (!playing) draw(performance.now()); };

  if (!(await loadMeta())) {
    root.replaceChildren(el("p", { class: "muted", textContent: "No ensemble fields cached yet — run “Evaluate on test set” first." }));
    return;
  }
  index = Math.min(Math.max(0, index), meta.count - 1);
  await loadField(index);
  requestAnimationFrame(draw);
}
