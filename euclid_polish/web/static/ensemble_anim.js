/**
 * Ensemble disagreement animation — a "morphing movie" of an SR field.
 *
 * The ensemble's per-member disagreement lives in a tiny subspace: the M=5
 * members minus their mean span (M-1) dims. The server PCAs that per field and
 * caches the mean (SR) + the top-K unit eigen-images (pca0..pca{K-1}) + each
 * component's amplitude aᵢ (the population std the members actually spread along
 * it). This widget fetches them and renders, at 60fps in a canvas:
 *
 *     frame(t) = mean + Σ aᵢ · amp · sin(2π fᵢ t + φᵢ) · componentᵢ
 *
 * a smooth, looping path through the *real* (spatially-correlated) disagreement
 * subspace. Hallucinated regions visibly morph; LR-constrained structure stays
 * still. (Per-pixel Gaussian draws would instead be spatial white noise.)
 *
 *   import { mountEnsembleAnim } from ".../ensemble_anim.js";
 *   mountEnsembleAnim(document.getElementById("ens-anim"));
 */

const COLLECTION = "ensemble";
//: Per-component loop frequencies (cycles per loop) + phases — distinct small
//: integers ⇒ a clean-looping Lissajous through the top components.
const FREQS = [1, 2, 3];
const PHASES = [0, Math.PI / 2, Math.PI / 3];

function el(tag, attrs = {}, kids = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "style") e.setAttribute("style", v);
    else if (k in e) e[k] = v;
    else e.setAttribute(k, v);
  }
  for (const c of kids) e.append(c);
  return e;
}

async function fetchCube(index, tier) {
  const r = await fetch(`/viewer/cube/${COLLECTION}/${index}?tier=${tier}`);
  if (!r.ok) throw new Error(`${tier} ${r.status}`);
  const shape = (r.headers.get("X-Cube-Shape") || "").split(",").map(Number);
  const asinh = parseFloat(r.headers.get("X-Cube-Asinh")) || 100;
  const data = new Float32Array(await r.arrayBuffer());
  return { data, h: shape[0], w: shape[1], c: shape[2], asinh };
}

export async function mountEnsembleAnim(root, opts = {}) {
  // ---- DOM ----------------------------------------------------------------
  const canvas = el("canvas", { style: "max-width:100%;height:auto;image-rendering:pixelated;background:#000;border-radius:6px;" });
  const ctx = canvas.getContext("2d");
  const msg = el("div", { class: "muted", style: "font-size:12px;margin-top:4px;" });
  const idxLabel = el("span", { class: "muted", style: "min-width:9ch;display:inline-block;" });

  const prev = el("button", { type: "button", textContent: "‹ prev" });
  const next = el("button", { type: "button", textContent: "next ›" });
  const playBtn = el("button", { type: "button", textContent: "⏸ pause" });
  const speed = el("input", { type: "range", min: "0.1", max: "2", step: "0.05", value: "0.5", style: "width:120px;" });
  const amp = el("input", { type: "range", min: "0", max: "3", step: "0.1", value: "1.6", style: "width:120px;" });

  const controls = el("div", { style: "display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:8px;" }, [
    prev, next, idxLabel, playBtn,
    el("label", { style: "font-size:12px;" }, [document.createTextNode("speed "), speed]),
    el("label", { style: "font-size:12px;", title: "morph amplitude in units of member σ (1 = the members' actual spread)" },
      [document.createTextNode("amplitude "), amp]),
  ]);
  root.replaceChildren(canvas, controls, msg);

  // ---- state --------------------------------------------------------------
  let meta = null;
  let index = Number.isFinite(opts.index) ? opts.index : 0;
  let field = null;          // { h, w, mean, comps:[Float32Array], amps:[..], norm, Kc }
  let playing = true;
  let t0 = performance.now();

  async function loadMeta() {
    const r = await fetch(`/viewer/meta/${COLLECTION}`);
    meta = r.ok ? await r.json() : null;
    return meta && meta.count ? meta : null;
  }

  // Extract the VIS band (band 0) of a cube into a contiguous H*W array.
  const visBand = (cube) => {
    const { data, h, w, c } = cube;
    const out = new Float32Array(h * w);
    for (let p = 0; p < h * w; p++) out[p] = data[p * c];
    return out;
  };

  async function loadField(i) {
    msg.textContent = "loading…";
    const n = (meta.pca_n | 0);
    const sr = await fetchCube(i, "sr");
    const comps = [];
    for (let k = 0; k < n; k++) {
      try { comps.push(visBand(await fetchCube(i, `pca${k}`))); }
      catch { /* a field with <2 members has no components */ }
    }
    const mean = visBand(sr);
    // Stable brightness: normalise the asinh stretch by the mean field's max so
    // the morph changes structure, not global brightness.
    const Kc = sr.asinh || 100;
    let mx = 1e-6;
    for (let p = 0; p < mean.length; p++) {
      const a = Math.asinh(mean[p] / Kc);
      if (a > mx) mx = a;
    }
    const amps = (meta.pca_amps && meta.pca_amps[i]) || [];
    field = { h: sr.h, w: sr.w, mean, comps, amps, norm: mx, Kc };
    canvas.width = sr.w; canvas.height = sr.h;
    idxLabel.textContent = `field ${i + 1}/${meta.count}`;
    msg.textContent = comps.length
      ? `morphing through ${comps.length} disagreement component(s)`
      : "single member — nothing to morph";
    t0 = performance.now();
  }

  // ---- render loop --------------------------------------------------------
  const img = () => ctx.createImageData(field.w, field.h);
  let imageData = null;

  function draw(now) {
    if (field) {
      if (!imageData || imageData.width !== field.w) imageData = img();
      const { mean, comps, amps, norm, Kc, w, h } = field;
      const t = ((now - t0) / 1000) * parseFloat(speed.value);
      const A = parseFloat(amp.value);
      // Per-component coefficients for this instant.
      const coeff = comps.map((_, k) =>
        (amps[k] || 0) * A * Math.sin(2 * Math.PI * FREQS[k % FREQS.length] * t + PHASES[k % PHASES.length]));
      const px = imageData.data;
      for (let p = 0; p < w * h; p++) {
        let v = mean[p];
        for (let k = 0; k < comps.length; k++) v += coeff[k] * comps[k][p];
        let g = Math.asinh(v / Kc) / norm;          // asinh stretch, stable norm
        g = g < 0 ? 0 : g > 1 ? 1 : g;
        const b = (g * 255) | 0;
        const o = p * 4;
        px[o] = px[o + 1] = px[o + 2] = b; px[o + 3] = 255;
      }
      ctx.putImageData(imageData, 0, 0);
    }
    if (playing) requestAnimationFrame(draw);
  }

  // ---- wire controls ------------------------------------------------------
  const setPlaying = (on) => {
    playing = on;
    playBtn.textContent = on ? "⏸ pause" : "▶ play";
    if (on) { t0 = performance.now() - 0; requestAnimationFrame(draw); }
  };
  playBtn.onclick = () => setPlaying(!playing);
  prev.onclick = async () => { index = (index - 1 + meta.count) % meta.count; await loadField(index); if (!playing) draw(performance.now()); };
  next.onclick = async () => { index = (index + 1) % meta.count; await loadField(index); if (!playing) draw(performance.now()); };
  for (const s of [speed, amp]) s.oninput = () => { if (!playing) draw(performance.now()); };

  // ---- go -----------------------------------------------------------------
  if (!(await loadMeta())) {
    root.replaceChildren(el("p", { class: "muted", textContent: "No ensemble fields cached yet — run “Evaluate on test set” first." }));
    return;
  }
  index = Math.min(Math.max(0, index), meta.count - 1);
  await loadField(index);
  requestAnimationFrame(draw);
}
