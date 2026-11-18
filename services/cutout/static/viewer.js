// POLISH Euclid viewer: fetch Q1 MER blocks from this server, super-resolve them
// in the browser with ONNX Runtime Web, and show Euclid vs POLISH side by side.
//
// Geometry. A MER tile is a 19200×19200 grid of 0.1″ pixels. The view is cut into
// blocks of B×B tile pixels aligned on that grid; each block is fetched with an
// M-pixel margin on every side (the model's receptive field) and its centre is kept,
// so neighbouring SR blocks meet without seams. Array row 0 is the first FITS row
// (south); the screen shows north up, east left.
//
// Model. Input (1, H, W, 32): asinh(e⁻ / q) of the four bands at 8 knees, knee-major
// (channel k·4 + band). Output (1, 2H, 2W, 4): electrons per 0.05″ pixel.

const B = 256;                 // block side, LR pixels
const M = 32;                  // fetch margin, LR pixels
const N = B + 2 * M;           // fetched side
const SCALE = 2;               // SR factor
const KNEES = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4];
const BANDS = ["VIS", "Y_E", "J_E", "H_E"];
// Display: asinh knee at 5σ of each band's MER noise (e⁻ per LR pixel); white
// points balanced like the console's "vivid" mode (poster galaxy levels).
const SIGMA = { VIS: 29.81, Y_E: 9.4, J_E: 9.9, H_E: 9.69 };
const WHITE_BALANCE = { VIS: 1.0, Y_E: 0.2088, J_E: 0.2929, H_E: 0.3799 };
const PRESETS = [
  ["Poster galaxy (EDF-N)", 273.2308875, 68.3636556],
  ["EDF-N centre", 270.0791195, 66.0000139],
  ["EDF-S centre", 61.2735175, -48.4999861],
  ["EDF-F centre", 52.9869358, -27.9999861],
  ["LDN 1641 (Orion)", 85.8353297, -8.4999861],
];
const ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";
const FETCH_CONCURRENCY = 4;
const CACHE_BLOCKS = 48;
const D2R = Math.PI / 180;

const $ = (id) => document.getElementById(id);
const canvas = $("view");
const ctx = canvas.getContext("2d");

const state = {
  tiles: [],
  tile: null,            // {id, width, height, wcs, bands}
  cx: 0, cy: 0,          // view centre, tile LR pixel coordinates (0-based, float)
  zoom: 1,               // screen CSS px per SR pixel
  mode: "swipe",         // swipe | lr | sr
  divider: 0.5,          // swipe divider, fraction of the canvas width
  colour: "colour",
  white: 3000,           // VIS-equivalent white level, e⁻ per LR pixel
  tta: false,
  blink: false,
  hover: null,
};
const blocks = new Map();      // key → block
let session = null, providerName = "", modelMs = [];
let fetchQueue = [], fetching = 0, inferChain = Promise.resolve(), pendingInfer = 0;
let drawScheduled = false;

// ───────────────────────── WCS (gnomonic) and tile choice ─────────────────────────

function world2pix(wcs, ra, dec) {
  const a0 = wcs.CRVAL1 * D2R, d0 = wcs.CRVAL2 * D2R, a = ra * D2R, d = dec * D2R;
  const cosc = Math.sin(d0) * Math.sin(d) + Math.cos(d0) * Math.cos(d) * Math.cos(a - a0);
  const xi = Math.cos(d) * Math.sin(a - a0) / cosc / D2R;
  const eta = (Math.cos(d0) * Math.sin(d) - Math.sin(d0) * Math.cos(d) * Math.cos(a - a0)) / cosc / D2R;
  const det = wcs.CD1_1 * wcs.CD2_2 - wcs.CD1_2 * wcs.CD2_1;
  const dx = (wcs.CD2_2 * xi - wcs.CD1_2 * eta) / det;
  const dy = (-wcs.CD2_1 * xi + wcs.CD1_1 * eta) / det;
  return [wcs.CRPIX1 + dx - 1, wcs.CRPIX2 + dy - 1];      // 0-based array col, row
}

function pix2world(wcs, x, y) {
  const dx = x + 1 - wcs.CRPIX1, dy = y + 1 - wcs.CRPIX2;
  const xi = (wcs.CD1_1 * dx + wcs.CD1_2 * dy) * D2R;
  const eta = (wcs.CD2_1 * dx + wcs.CD2_2 * dy) * D2R;
  const a0 = wcs.CRVAL1 * D2R, d0 = wcs.CRVAL2 * D2R;
  const den = Math.cos(d0) - eta * Math.sin(d0);
  const ra = (a0 + Math.atan2(xi, den)) / D2R;
  const dec = Math.atan2(eta * Math.cos(d0) + Math.sin(d0), Math.hypot(xi, den)) / D2R;
  return [((ra % 360) + 360) % 360, dec];
}

// Tangent-plane offsets (deg) of (ra, dec) about (ra0, dec0).
function project(ra0, dec0, ra, dec) {
  const a0 = ra0 * D2R, d0 = dec0 * D2R, a = ra * D2R, d = dec * D2R;
  const cosc = Math.sin(d0) * Math.sin(d) + Math.cos(d0) * Math.cos(d) * Math.cos(a - a0);
  return [Math.cos(d) * Math.sin(a - a0) / cosc / D2R,
          (Math.cos(d0) * Math.sin(d) - Math.sin(d0) * Math.cos(d) * Math.cos(a - a0)) / cosc / D2R];
}

// The tile whose footprint contains (ra, dec) deepest (MER tiles overlap).
function locateTile(ra, dec) {
  let best = null, bestDepth = -Infinity;
  for (const t of state.tiles) {
    if (Math.abs(t.dec - dec) > 1.5) continue;
    const [px, py] = project(t.ra, t.dec, ra, dec);
    const poly = t.polygon.map(([r, d]) => project(t.ra, t.dec, r, d));
    let inside = false, depth = Infinity;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i], [xj, yj] = poly[j];
      if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) inside = !inside;
      const ex = xj - xi, ey = yj - yi, len2 = ex * ex + ey * ey;
      const u = Math.max(0, Math.min(1, ((px - xi) * ex + (py - yi) * ey) / len2));
      depth = Math.min(depth, Math.hypot(px - xi - u * ex, py - yi - u * ey));
    }
    if (inside && depth > bestDepth) { best = t; bestDepth = depth; }
  }
  return best;
}

async function openTile(id) {
  if (state.tile && state.tile.id === id) return state.tile;
  const resp = await fetch(`v1/tile/${id}`);
  if (!resp.ok) throw new Error(`tile ${id}: ${resp.status} ${await resp.text()}`);
  const info = await resp.json();
  const vis = info.bands[0];
  state.tile = { id, width: vis.width, height: vis.height, wcs: vis.wcs, bands: info.bands };
  return state.tile;
}

async function goTo(ra, dec) {
  const t = locateTile(ra, dec);
  if (!t) { setStatus(`No Q1 tile covers RA ${ra.toFixed(4)}, Dec ${dec.toFixed(4)}`, true); return; }
  await openTile(t.tile);
  [state.cx, state.cy] = world2pix(state.tile.wcs, ra, dec);
  saveHash();
  schedule();
}

// After a pan, move to the tile that holds the view centre deepest.
async function retileIfNeeded() {
  if (!state.tile) return;
  const [ra, dec] = pix2world(state.tile.wcs, state.cx, state.cy);
  const t = locateTile(ra, dec);
  if (t && t.tile !== state.tile.id) await goTo(ra, dec);
}

// ───────────────────────────── fetching and the model ─────────────────────────────

function blockKey(tile, i, j) { return `${tile}:${i}:${j}`; }

function decodeFrame(buf) {
  const dv = new DataView(buf);
  const magic = String.fromCharCode(...new Uint8Array(buf, 0, 4));
  if (magic !== "EPCT") throw new Error("bad cutout frame");
  const n = dv.getUint32(4, true);
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n)));
  const start = 8 + n + ((8 - ((8 + n) % 8)) % 8);
  return { meta, data: new Float32Array(buf, start, meta.bands.length * meta.h * meta.w) };
}

function requestBlock(tile, i, j) {
  const key = blockKey(tile.id, i, j);
  let blk = blocks.get(key);
  if (!blk) {
    blk = { key, tile: tile.id, i, j, state: "queued", lr: null, sr: {}, canvases: {}, used: 0 };
    blocks.set(key, blk);
    fetchQueue.push(blk);
  }
  blk.used = performance.now();
  if (blk.state === "lr" && !blk.sr[state.tta] && !blk.inferQueued) queueInfer(blk);
  return blk;
}

function pumpFetches() {
  // Nearest the view centre first.
  fetchQueue.sort((a, b) => dist2(b) - dist2(a));
  while (fetching < FETCH_CONCURRENCY && fetchQueue.length) {
    const blk = fetchQueue.pop();
    if (!blocks.has(blk.key)) continue;
    fetching++;
    blk.state = "fetching";
    fetchBlock(blk).catch((e) => { blk.state = "error"; blk.error = String(e); setStatus(String(e), true); })
      .finally(() => { fetching--; pumpFetches(); schedule(); });
  }
}

function dist2(blk) {
  if (!state.tile || blk.tile !== state.tile.id) return 1e18;
  const dx = (blk.i + 0.5) * B - state.cx, dy = (blk.j + 0.5) * B - state.cy;
  return dx * dx + dy * dy;
}

async function fetchBlock(blk) {
  const x0 = blk.i * B - M, y0 = blk.j * B - M;
  const resp = await fetch(`v1/cutout?tile=${blk.tile}&x0=${x0}&y0=${y0}&size=${N}`);
  if (!resp.ok) throw new Error(`cutout ${blk.key}: ${resp.status} ${await resp.text()}`);
  const { meta, data } = decodeFrame(await resp.arrayBuffer());
  // Electrons, band-major (4, N, N).
  const e = new Float32Array(data.length);
  for (let b = 0; b < 4; b++) {
    const f = meta.bands[b].to_electrons, off = b * N * N;
    for (let p = 0; p < N * N; p++) e[off + p] = data[off + p] * f;
  }
  blk.input = e;
  blk.lr = crop(e, N, 4, M, B);
  blk.state = "lr";
  blk.canvases = {};
  queueInfer(blk);
}

// (C, n, n) band-major → its (C, size, size) window starting at (off, off).
function crop(src, n, channels, off, size) {
  const out = new Float32Array(channels * size * size);
  for (let c = 0; c < channels; c++)
    for (let r = 0; r < size; r++) {
      const s = c * n * n + (r + off) * n + off;
      out.set(src.subarray(s, s + size), c * size * size + r * size);
    }
  return out;
}

function queueInfer(blk) {
  const tta = state.tta;
  if (blk.sr[tta] || !blk.input) return;
  blk.inferQueued = true;
  pendingInfer++;
  inferChain = inferChain.then(async () => {
    try {
      if (!blocks.has(blk.key) || !session) return;
      if (!isVisible(blk)) return;             // panned away: skip, re-queued if seen again
      const t0 = performance.now();
      blk.sr[tta] = await superResolve(blk.input, tta);
      modelMs.push((performance.now() - t0) / (tta ? 8 : 1));
      if (modelMs.length > 20) modelMs.shift();
      blk.canvases = {};
      if (blk.sr[false] && blk.sr[true]) blk.input = null;   // both done: drop the raw input
    } catch (e) {
      setStatus(`Model error: ${e.message || e}`, true);
    } finally {
      blk.inferQueued = false;
      pendingInfer--;
      schedule();
    }
  });
}

// (4, N, N) electrons → (4, 2B, 2B) electrons per SR pixel (NaN where the input is).
async function superResolve(e, tta) {
  const hw = N * N, C = 32;
  const x = new Float32Array(hw * C);
  for (let p = 0; p < hw; p++) {
    for (let b = 0; b < 4; b++) {
      let v = e[b * hw + p];
      if (!Number.isFinite(v)) v = 0;
      for (let k = 0; k < 8; k++) x[p * C + k * 4 + b] = Math.asinh(v / KNEES[k]);
    }
  }
  const n2 = 2 * N, acc = new Float32Array(n2 * n2 * 4);
  const variants = tta ? 8 : 1;
  for (let t = 0; t < variants; t++) {
    const input = t === 0 ? x : dihedral(x, N, C, t, false);
    const res = await session.run({ lr: new ort.Tensor("float32", input, [1, N, N, C]) });
    let y = res[session.outputNames[0]].data;
    if (t !== 0) y = dihedral(y, n2, 4, t, true);
    for (let q = 0; q < acc.length; q++) acc[q] += y[q];
  }
  // NHWC (2N, 2N, 4) → band-major centre (4, 2B, 2B), masked where the LR pixel is NaN.
  const s = 2 * B, off = 2 * M, out = new Float32Array(4 * s * s);
  for (let r = 0; r < s; r++)
    for (let c = 0; c < s; c++) {
      const src = ((r + off) * n2 + (c + off)) * 4;
      const lrp = ((r >> 1) + M) * N + ((c >> 1) + M);
      for (let b = 0; b < 4; b++)
        out[b * s * s + r * s + c] = Number.isFinite(e[b * hw + lrp]) ? acc[src + b] / variants : NaN;
    }
  return out;
}

// Dihedral transform t (0–7) of a square NHWC image: t&3 quarter turns, t&4 a
// transpose first. inverse=true undoes it.
function dihedral(src, n, ch, t, inverse) {
  const turns = t & 3, flip = (t & 4) !== 0;
  let a = src;
  if (!inverse) {
    if (flip) a = transpose(a, n, ch);
    for (let k = 0; k < turns; k++) a = rot90(a, n, ch);
  } else {
    for (let k = 0; k < (4 - turns) % 4; k++) a = rot90(a, n, ch);
    if (flip) a = transpose(a, n, ch);
  }
  return a;
}
function transpose(src, n, ch) {
  const out = new Float32Array(src.length);
  for (let r = 0; r < n; r++)
    for (let c = 0; c < n; c++) out.set(src.subarray((c * n + r) * ch, (c * n + r + 1) * ch), (r * n + c) * ch);
  return out;
}
function rot90(src, n, ch) {        // out[r][c] = src[c][n-1-r]
  const out = new Float32Array(src.length);
  for (let r = 0; r < n; r++)
    for (let c = 0; c < n; c++)
      out.set(src.subarray((c * n + n - 1 - r) * ch, (c * n + n - r) * ch), (r * n + c) * ch);
  return out;
}

async function loadModel() {
  ort.env.wasm.wasmPaths = ORT_DIST;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 4, 8) : 1;
  const tries = navigator.gpu ? ["webgpu", "wasm"] : ["wasm"];
  for (const ep of tries) {
    try {
      session = await ort.InferenceSession.create("static/m257.onnx", { executionProviders: [ep] });
      providerName = ep === "webgpu" ? "WebGPU" : `CPU (${ort.env.wasm.numThreads} thread${ort.env.wasm.numThreads > 1 ? "s" : ""})`;
      return;
    } catch (e) {
      console.warn(`ONNX Runtime ${ep} failed`, e);
    }
  }
  throw new Error("could not start the model (no WebGPU or WebAssembly)");
}

// ─────────────────────────────────── rendering ───────────────────────────────────

function lrPerScreenPx() { return 1 / (SCALE * state.zoom); }

function viewRect() {
  const w = canvas.clientWidth, h = canvas.clientHeight, k = lrPerScreenPx();
  return { w, h, x0: state.cx - (w / 2) * k, x1: state.cx + (w / 2) * k,
           y0: state.cy - (h / 2) * k, y1: state.cy + (h / 2) * k };
}

function isVisible(blk) {
  if (!state.tile || blk.tile !== state.tile.id) return false;
  const v = viewRect();
  return (blk.i + 1) * B > v.x0 && blk.i * B < v.x1 && (blk.j + 1) * B > v.y0 && blk.j * B < v.y1;
}

// Screen position of tile pixel coordinates (north up: array row grows upward).
function toScreen(x, y) {
  const k = lrPerScreenPx(), w = canvas.clientWidth, h = canvas.clientHeight;
  return [w / 2 + (x - state.cx) / k, h / 2 - (y - state.cy) / k];
}
function fromScreen(sx, sy) {
  const k = lrPerScreenPx(), w = canvas.clientWidth, h = canvas.clientHeight;
  return [state.cx + (sx - w / 2) * k, state.cy - (sy - h / 2) * k];
}

function stretch(v, knee, white) {
  if (!(v > 0)) return 0;
  const t = Math.asinh(v / knee) / Math.asinh(white / knee);
  return t >= 1 ? 255 : (t * 255) | 0;
}

// (4, n, n) electrons → a canvas, rows flipped so north is up. `sb` turns values
// into e⁻ per LR-pixel area (4 for SR) so both sides share one brightness scale.
function renderBlock(data, n, sb) {
  const c = document.createElement("canvas");
  c.width = c.height = n;
  const g = c.getContext("2d"), img = g.createImageData(n, n), px = img.data;
  const plane = n * n, W = state.white;
  const vis = data.subarray(0, plane), y = data.subarray(plane, 2 * plane),
        j = data.subarray(2 * plane, 3 * plane), hb = data.subarray(3 * plane, 4 * plane);
  const kV = 5 * SIGMA.VIS, kG = 5 * (SIGMA.Y_E + SIGMA.J_E) / 2, kR = 5 * SIGMA.H_E;
  const wG = W * (WHITE_BALANCE.Y_E + WHITE_BALANCE.J_E) / 2, wR = W * WHITE_BALANCE.H_E;
  for (let r = 0; r < n; r++) {
    const row = (n - 1 - r) * n;
    for (let col = 0; col < n; col++) {
      const p = row + col, o = (r * n + col) * 4;
      const v = vis[p];
      if (Number.isNaN(v)) { px[o] = px[o + 1] = px[o + 2] = 28; px[o + 3] = 255; continue; }
      if (state.colour === "vis") {
        px[o] = px[o + 1] = px[o + 2] = stretch(v * sb, kV, W);
      } else {
        px[o] = stretch(hb[p] * sb, kR, wR);
        px[o + 1] = stretch((y[p] + j[p]) * 0.5 * sb, kG, wG);
        px[o + 2] = stretch(v * sb, kV, W);
      }
      px[o + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  return c;
}

function blockCanvas(blk, which) {
  const key = `${which}:${state.colour}:${state.white}:${state.tta}`;
  if (blk.canvases[key]) return blk.canvases[key];
  let c = null;
  if (which === "lr" && blk.lr) c = renderBlock(blk.lr, B, 1);
  if (which === "sr" && blk.sr[state.tta]) c = renderBlock(blk.sr[state.tta], 2 * B, SCALE * SCALE);
  if (c) blk.canvases = { ...Object.fromEntries(Object.entries(blk.canvases).filter(([k]) => !k.startsWith(which))), [key]: c };
  return c;
}

function schedule() {
  if (drawScheduled) return;
  drawScheduled = true;
  requestAnimationFrame(() => { drawScheduled = false; draw(); });
}

function draw() {
  const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#0b0d10";
  ctx.fillRect(0, 0, w, h);
  if (!state.tile) return;

  const v = viewRect(), tile = state.tile;
  const i0 = Math.max(0, Math.floor(v.x0 / B)), i1 = Math.min(Math.ceil(tile.width / B) - 1, Math.floor(v.x1 / B));
  const j0 = Math.max(0, Math.floor(v.y0 / B)), j1 = Math.min(Math.ceil(tile.height / B) - 1, Math.floor(v.y1 / B));
  const visible = [];
  if ((i1 - i0 + 1) * (j1 - j0 + 1) <= 64)                 // don't fetch a whole tile when zoomed out
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) visible.push(requestBlock(tile, i, j));
  pumpFetches();
  evict();

  // Swipe: Euclid left of the divider, POLISH right. Otherwise one image fills the
  // view; B (blink) swaps it for the other.
  const swipe = state.mode === "swipe";
  const split = swipe ? state.divider * w : w;
  const fullKind = (state.mode === "lr") !== state.blink ? "lr" : "sr";
  ctx.imageSmoothingEnabled = false;
  for (const blk of visible) {
    const [sx, sy] = toScreen(blk.i * B, (blk.j + 1) * B);
    const size = B / lrPerScreenPx();
    const lr = blockCanvas(blk, "lr"), sr = blockCanvas(blk, "sr");
    const left = swipe || fullKind === "lr" ? lr : sr;
    drawClipped(left, sx, sy, size, 0, split);
    if (swipe) drawClipped(sr, sx, sy, size, split, w);
    if (!left && sx < split) placeholder(blk, sx, sy, size, 0, split);
    if (swipe && !sr && sx + size > split) placeholder(blk, sx, sy, size, split, w);
  }

  if (state.mode === "swipe") {
    ctx.fillStyle = "rgba(255,255,255,.85)";
    ctx.fillRect(split - 1, 0, 2, h);
    ctx.beginPath(); ctx.arc(split, h / 2, 9, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = "#0b0d10"; ctx.font = "12px system-ui"; ctx.textAlign = "center"; ctx.fillText("⇔", split, h / 2 + 4);
  }
  ctx.font = "600 12px system-ui"; ctx.textAlign = "left";
  const labels = swipe ? [["EUCLID 0.1″", 12, "#e6e8eb"], ["POLISH 0.05″", split + 12, "#f2c14e"]]
    : [[fullKind === "lr" ? "EUCLID 0.1″" : "POLISH 0.05″", 12, fullKind === "lr" ? "#e6e8eb" : "#f2c14e"]];
  for (const [text, x, colour] of labels) {
    if (x > w - 40) continue;
    ctx.fillStyle = "rgba(11,13,16,.7)"; ctx.fillRect(x - 6, 8, ctx.measureText(text).width + 12, 20);
    ctx.fillStyle = colour; ctx.fillText(text, x, 22);
  }
  drawScaleBar(w, h);
  updateHud();
  updateStatus(visible);
}

function drawClipped(img, sx, sy, size, xa, xb) {
  if (!img || xb <= xa || sx >= xb || sx + size <= xa) return;
  ctx.save();
  ctx.beginPath(); ctx.rect(xa, sy, xb - xa, size); ctx.clip();
  ctx.drawImage(img, sx, sy, size, size);
  ctx.restore();
}

function placeholder(blk, sx, sy, size, xa, xb) {
  xa = Math.max(xa, sx);
  xb = Math.min(xb, sx + size);
  if (xb <= xa) return;
  ctx.save();
  ctx.beginPath(); ctx.rect(xa, sy, xb - xa, size); ctx.clip();
  ctx.strokeStyle = "rgba(108,182,255,.25)"; ctx.strokeRect(sx + .5, sy + .5, size - 1, size - 1);
  if (size > 90) {
    ctx.fillStyle = "rgba(139,147,160,.9)"; ctx.font = "12px system-ui"; ctx.textAlign = "center";
    const text = blk.state === "error" ? "fetch failed" : blk.state === "lr" ? "super-resolving…" : "fetching…";
    ctx.fillText(text, (xa + xb) / 2, sy + size / 2);
  }
  ctx.restore();
}

function drawScaleBar(w, h) {
  const px = 10 / 0.1 / lrPerScreenPx();            // 10″ on screen
  const len = px > w / 3 ? px / 10 : px, label = px > w / 3 ? "1″" : "10″";
  ctx.fillStyle = "#e6e8eb"; ctx.fillRect(w - 16 - len, h - 52, len, 3);
  ctx.font = "12px system-ui"; ctx.textAlign = "right"; ctx.fillText(label, w - 16, h - 58);
}

function evict() {
  if (blocks.size <= CACHE_BLOCKS) return;
  const old = [...blocks.values()].filter((b) => !isVisible(b) && b.state !== "fetching" && !b.inferQueued)
    .sort((a, b) => a.used - b.used);
  for (const b of old.slice(0, blocks.size - CACHE_BLOCKS)) blocks.delete(b.key);
  fetchQueue = fetchQueue.filter((b) => blocks.has(b.key));
}

// ───────────────────────────────── HUD and status ─────────────────────────────────

function sexagesimal(deg, hours) {
  const sign = deg < 0 ? "-" : hours ? "" : "+";
  let v = Math.abs(deg) / (hours ? 15 : 1);
  const a = Math.floor(v); v = (v - a) * 60; const b = Math.floor(v); const c = (v - b) * 60;
  return `${sign}${String(a).padStart(2, "0")}:${String(b).padStart(2, "0")}:${c.toFixed(hours ? 2 : 1).padStart(hours ? 5 : 4, "0")}`;
}

function sampleAt(x, y) {
  const i = Math.floor(x / B), j = Math.floor(y / B), blk = blocks.get(blockKey(state.tile.id, i, j));
  if (!blk || !blk.lr) return null;
  const c = Math.floor(x - i * B), r = Math.floor(y - j * B);
  const lr = blk.lr[r * B + c];
  const sr = blk.sr[state.tta];
  const s = sr ? sr[Math.floor(2 * (y - j * B)) * 2 * B + Math.floor(2 * (x - i * B))] : null;
  return { lr, sr: s };
}

function updateHud() {
  if (!state.tile) { $("hud").textContent = ""; return; }
  const [x, y] = state.hover ? fromScreen(...state.hover) : [state.cx, state.cy];
  const [ra, dec] = pix2world(state.tile.wcs, x, y);
  let text = `RA ${ra.toFixed(5)}  Dec ${dec.toFixed(5)}   ${sexagesimal(ra, true)} ${sexagesimal(dec, false)}\n` +
    `tile ${state.tile.id}  x ${x.toFixed(1)}  y ${y.toFixed(1)}`;
  const s = state.hover && sampleAt(x, y);
  if (s) text += `\nVIS  Euclid ${fmt(s.lr)} e⁻/px   POLISH ${s.sr == null ? "…" : fmt(s.sr * 4)} e⁻/(0.1″)²`;
  $("hud").textContent = text;
}
const fmt = (v) => (Number.isFinite(v) ? (Math.abs(v) >= 1000 ? v.toExponential(2) : v.toFixed(1)) : "—");

let statusError = "";
function setStatus(text, isError) {
  if (isError) { statusError = text; console.error(text); }
  $("status").innerHTML = isError ? `<span class="err">${escapeHtml(text)}</span>` : escapeHtml(text);
}
function escapeHtml(s) { return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]); }

function updateStatus(visible) {
  if (statusError) return;
  const waiting = visible.filter((b) => b.state === "queued" || b.state === "fetching").length;
  const srDone = visible.filter((b) => b.sr[state.tta]).length;
  const ms = modelMs.length ? modelMs.reduce((a, b) => a + b, 0) / modelMs.length : null;
  const model = session ? `${providerName}${ms ? ` · ${Math.round(ms)} ms/block${state.tta ? " ×8" : ""}` : ""}` : "loading model…";
  const work = visible.length === 0 ? "zoom in to load blocks" : `${srDone}/${visible.length} blocks` +
    (waiting ? ` · fetching ${waiting}` : "") + (pendingInfer ? ` · model queue ${pendingInfer}` : "");
  setStatus(`${model} · ${work}`);
}

// ───────────────────────────────── interaction ─────────────────────────────────

function saveHash() {
  if (!state.tile) return;
  const [ra, dec] = pix2world(state.tile.wcs, state.cx, state.cy);
  history.replaceState(null, "", `#ra=${ra.toFixed(6)}&dec=${dec.toFixed(6)}&z=${state.zoom}`);
}

function parseWhere(text) {
  const nums = text.trim().split(/[\s,]+/).filter(Boolean);
  if (nums[0] && nums[0].toLowerCase() === "tile" && nums.length === 4) return { tile: nums[1], x: +nums[2], y: +nums[3] };
  if (nums.length === 3 && /^\d{6,}$/.test(nums[0])) return { tile: nums[0], x: +nums[1], y: +nums[2] };
  if (nums.length === 2 && nums.every((s) => Number.isFinite(+s))) return { ra: +nums[0], dec: +nums[1] };
  if (nums.length === 6) {                                   // h m s d m s
    const [h, m, s, d, dm, ds] = nums.map(Number);
    const sign = nums[3].startsWith("-") ? -1 : 1;
    return { ra: 15 * (h + m / 60 + s / 3600), dec: sign * (Math.abs(d) + dm / 60 + ds / 3600) };
  }
  return null;
}

async function onSearch(ev) {
  ev.preventDefault();
  statusError = "";
  const where = parseWhere($("where").value);
  if (!where) { setStatus("Enter 'RA Dec' in degrees, 'hh mm ss dd mm ss', or 'tile x y'", true); return; }
  try {
    if (where.tile) {
      await openTile(where.tile); state.cx = where.x; state.cy = where.y; saveHash(); schedule();
    } else {
      await goTo(where.ra, where.dec);
    }
  } catch (e) { setStatus(String(e.message || e), true); }
}

function setZoom(z, sx, sy) {
  z = Math.max(1 / 16, Math.min(8, z));
  if (sx != null) {                               // keep the point under the cursor fixed
    const [x, y] = fromScreen(sx, sy);
    state.zoom = z;
    const [x2, y2] = fromScreen(sx, sy);
    state.cx += x - x2; state.cy += y - y2;
  } else state.zoom = z;
  saveHash(); schedule();
}

function bindUi() {
  for (const [name, ra, dec] of PRESETS) $("presets").add(new Option(name, `${ra} ${dec}`));
  $("presets").addEventListener("change", (e) => {
    if (!e.target.value) return;
    $("where").value = e.target.value; e.target.value = "";
    $("search").requestSubmit();
  });
  $("search").addEventListener("submit", onSearch);
  document.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  $("colour").addEventListener("change", (e) => { state.colour = e.target.value; schedule(); });
  const white = () => { state.white = Math.round(300 * Math.pow(10, (100 - +$("white").value) / 50)); schedule(); };
  $("white").addEventListener("input", white); white();
  $("tta").addEventListener("change", (e) => { state.tta = e.target.checked; schedule(); });
  $("info").addEventListener("click", () => $("about").classList.toggle("show"));

  let drag = null;
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    const nearDivider = state.mode === "swipe" && Math.abs(e.offsetX - state.divider * canvas.clientWidth) < 14;
    drag = { x: e.offsetX, y: e.offsetY, cx: state.cx, cy: state.cy, divider: nearDivider };
    canvas.classList.add("dragging");
  });
  canvas.addEventListener("pointermove", (e) => {
    state.hover = [e.offsetX, e.offsetY];
    if (drag && drag.divider) state.divider = Math.max(0, Math.min(1, e.offsetX / canvas.clientWidth));
    else if (drag) {
      const k = lrPerScreenPx();
      state.cx = drag.cx - (e.offsetX - drag.x) * k;
      state.cy = drag.cy + (e.offsetY - drag.y) * k;
    }
    schedule();
  });
  canvas.addEventListener("pointerup", () => {
    if (drag && !drag.divider) { saveHash(); retileIfNeeded().catch((e) => setStatus(String(e), true)); }
    drag = null; canvas.classList.remove("dragging");
  });
  canvas.addEventListener("pointerleave", () => { state.hover = null; schedule(); });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    setZoom(state.zoom * Math.pow(2, -e.deltaY / 300), e.offsetX, e.offsetY);
  }, { passive: false });
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;
    if (e.key === "1") setMode("swipe");
    else if (e.key === "2") setMode("lr");
    else if (e.key === "3") setMode("sr");
    else if (e.key === "b" || e.key === "B") { state.blink = !state.blink; schedule(); }
    else if (e.key === "+" || e.key === "=") setZoom(state.zoom * 2);
    else if (e.key === "-") setZoom(state.zoom / 2);
  });
  window.addEventListener("resize", schedule);
}

function setMode(mode) {
  state.mode = mode; state.blink = false;
  document.querySelectorAll("[data-mode]").forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));
  schedule();
}

async function main() {
  bindUi();
  const tilesReq = fetch("static/tiles.json").then((r) => r.json());
  const modelReq = loadModel().catch((e) => setStatus(String(e.message || e), true));
  state.tiles = (await tilesReq).tiles;
  const hash = new URLSearchParams(location.hash.slice(1));
  const ra = +hash.get("ra"), dec = +hash.get("dec");
  if (hash.get("z")) state.zoom = +hash.get("z");
  if (hash.has("ra") && Number.isFinite(ra) && Number.isFinite(dec)) await goTo(ra, dec);
  else await goTo(PRESETS[0][1], PRESETS[0][2]);
  await modelReq;
  for (const blk of blocks.values()) if (blk.state === "lr") queueInfer(blk);
  schedule();
}

// For checks from the console: the block cache and the model call.
window.polish = { state, blocks, superResolve: (e, tta) => superResolve(e, tta), decodeFrame };

main().catch((e) => setStatus(String(e.message || e), true));
