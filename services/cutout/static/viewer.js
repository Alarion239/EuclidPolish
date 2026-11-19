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
// AB-balanced colour (default). Electrons → "VIS-equivalent electrons" (same AB
// flux) with the bands' electron zeropoints (BandConfig.sim_zeropoint_e), so a
// flat-spectrum source is white. The "vivid" balance above is white only at
// VIS−H ≈ +1.7 AB, redder than nearly every galaxy (Q1 median VIS−H ≈ +0.4 at
// VIS 24–27, +1.2 at VIS 18–22), so it shows almost everything blue.
const ZP_E = { VIS: 34.29662776165843, Y_E: 31.396441190651323, J_E: 31.616441190651326, H_E: 31.566441190651325 };
const TO_VIS_EQ = Object.fromEntries(Object.entries(ZP_E).map(([b, zp]) => [b, 10 ** (0.4 * (ZP_E.VIS - zp))]));
// Measured per-pixel MER noise (Q1, 2026-11-18): VIS 21.7, NISP 2.6–2.8 e⁻.
// Each channel gets the same asinh stretch in VIS-equivalent units (so the balance
// is AB at every level), as "vivid" does per channel: bright cores compress toward
// white instead of saturating into one hue. (A hue-preserving Lupton stretch of
// the mean was tried first: bright galaxies came out as flat orange blobs.)
const NOISE_E = { VIS: 21.7, Y_E: 2.6, J_E: 2.8, H_E: 2.7 };
const SIGMA_AB = Math.hypot(NOISE_E.VIS, Math.hypot(NOISE_E.Y_E * TO_VIS_EQ.Y_E, NOISE_E.J_E * TO_VIS_EQ.J_E) / 2,
                            NOISE_E.H_E * TO_VIS_EQ.H_E) / 3;
// Knee 600 VIS-equivalent e⁻ (~18σ of H, the noisiest channel at 33 e⁻): about
// vivid's NISP knees, whose near-linear stretch gives its contrast. A 5σ knee
// (~170) log-compresses every resolved galaxy into a flat, washed-out disc.
const KNEE_AB = 600;
// Black point at 1.5σ of the channel mean: without it the Euclid side's sky is a
// speckle of coloured noise.
const BLACK_AB = 1.5 * SIGMA_AB;
// Releases: IRSA's MER mosaics, served by this server (?release=…). DR1 has no
// HiPS yet, so its coverage is drawn from its own MOCs.
const DR1_MOC = "https://nasa-irsa-euclid-dr1.s3.us-east-1.amazonaws.com/dr1/MOC/";
const RELEASES = {
  q1: { label: "Q1", mocs: [] },
  dr1: { label: "DR1", mocs: [`${DR1_MOC}euclid_dr1_wide.moc.moc.fits`, `${DR1_MOC}euclid_dr1_deep.moc.moc.fits`] },
};
// Draw every tile outline up to this many tiles; beyond, only the ones near the view.
const ALL_OUTLINES_MAX = 1000;
// [name, RA, Dec, map field of view in degrees, release]
const PRESETS = [
  ["Poster galaxy (EDF-N)", 273.2308875, 68.3636556, 0.04, "q1"],
  ["EDF-N", 270.0791195, 66.0000139, 3, "q1"],
  ["EDF-S", 61.2735175, -48.4999861, 5, "q1"],
  ["EDF-F", 52.9869358, -27.9999861, 3, "q1"],
  ["LDN 1641 (Orion)", 85.8353297, -8.4999861, 1, "q1"],
  ["DR1 · Wide Survey", 0, 0, 40, "dr1"],          // centred on the DR1 tiles (main)
  ["DR1 · poster galaxy", 273.2308875, 68.3636556, 0.04, "dr1"],
  ["DR1 · EDF-F", 52.9869358, -27.9999861, 3, "dr1"],
];
const ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";
// The map: Aladin Lite showing Euclid's own pre-rendered HiPS (no model, no
// cutouts). The first survey that answers is used; CDS's colour HiPS is the
// nicer one, ESA's VIS HiPS the fallback.
const ALADIN_URL = "https://cdn.jsdelivr.net/npm/aladin-lite@3.8.2/dist/aladin.js";
const SURVEYS = [
  { id: "cds-color", label: "Euclid Q1 colour (CDS)", url: "https://alasky.cds.unistra.fr/Euclid/Q1/CDS_P_Euclid_Q1_color" },
  // ESA's PNG tiles are a near-black linear stretch; its FITS tiles (ADU/s, sky σ ≈ 0.003)
  // are stretched here instead.
  { id: "cds-dss", label: "DSS2 colour (CDS)", url: "https://alasky.cds.unistra.fr/DSS/DSSColor", format: "jpeg" },
  { id: "esa-vis", label: "Euclid VIS (ESA)", url: "https://esdchips.esac.esa.int/Euclid/Q1_MER/VIS/ESAC_P_EUC_VIS",
    format: "fits", color: { colormap: "grayscale", stretch: "asinh", minCut: -0.003, maxCut: 0.3 } },
];
// Zooming the map in to one Euclid pixel (0.1″) per screen pixel switches to the
// POLISH view; zooming that out past ENTER/EXIT (hysteresis) switches back.
const ENTER_ZOOM = 0.5;        // screen px per SR px: 1 screen px = 0.1″
const EXIT_ZOOM = 0.4;
const SR_ARCSEC = 0.05;
const FETCH_CONCURRENCY = 4;
const CACHE_BLOCKS = 48;
const D2R = Math.PI / 180;

const $ = (id) => document.getElementById(id);
const canvas = $("view");
const ctx = canvas.getContext("2d");

const state = {
  view: "map",           // map | detail
  release: "q1",         // q1 | dr1
  tile: null,            // {id, width, height, wcs, bands}
  cx: 0, cy: 0,          // view centre, tile LR pixel coordinates (0-based, float)
  zoom: 1,               // screen CSS px per SR pixel
  mode: "swipe",         // swipe | lr | sr
  divider: 0.5,          // swipe divider, fraction of the canvas width
  colour: "ab",
  white: 3000,           // VIS-equivalent white level, e⁻ per LR pixel
  saturation: 1,         // AB colour: 0 grey, 1 as stretched, 2 doubled
  tta: false,
  blink: false,
  hover: null,
};
const blocks = new Map();      // key → block
const map = { A: null, al: null, survey: null, outlines: null, mocs: [] };
const indexes = {};            // release → {cd, tiles: [{id, ra, dec, w, h, wcs, deep}]}
let session = null, providerName = "", modelMs = [], modelPromise = null;
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

// The release's tile index (server: /v1/tiles), with each tile's TAN WCS.
async function loadIndex(release) {
  if (!indexes[release]) {
    indexes[release] = fetch(`v1/tiles?release=${release}`).then(async (resp) => {
      if (!resp.ok) throw new Error(`tile index ${release}: ${resp.status} ${await resp.text()}`);
      const raw = await resp.json();
      const [c11, c12, c21, c22] = raw.cd, col = Object.fromEntries(raw.columns.map((c, i) => [c, i]));
      const tiles = raw.tiles.map((r) => ({
        id: r[col.tile], ra: r[col.ra], dec: r[col.dec], w: r[col.naxis1], h: r[col.naxis2], deep: !!r[col.deep],
        wcs: { CRVAL1: r[col.ra], CRVAL2: r[col.dec], CRPIX1: r[col.crpix1], CRPIX2: r[col.crpix2],
               CD1_1: c11, CD1_2: c12, CD2_1: c21, CD2_2: c22 },
      }));
      return { release, dataset: raw.dataset, tiles };
    });
    indexes[release].catch(() => { delete indexes[release]; });
  }
  return indexes[release];
}

let index = null;              // the current release's resolved index

// The tile whose image holds (ra, dec) deepest (MER tile images overlap).
function locateTile(ra, dec) {
  if (!index) return null;
  let best = null, bestDepth = 0;
  const cosd = Math.max(0.05, Math.cos(dec * D2R));
  for (const t of index.tiles) {
    if (Math.abs(t.dec - dec) > 0.6) continue;
    let dra = Math.abs(t.ra - ra);
    if (dra > 180) dra = 360 - dra;
    if (dra * cosd > 0.6) continue;
    const [x, y] = world2pix(t.wcs, ra, dec);
    const depth = Math.min(x + 0.5, t.w - 0.5 - x, y + 0.5, t.h - 0.5 - y);
    if (depth > bestDepth) { best = t; bestDepth = depth; }
  }
  return best;
}

// A tile's outline on the sky: its image corners.
function tileOutline(t) {
  return [[-0.5, -0.5], [t.w - 0.5, -0.5], [t.w - 0.5, t.h - 0.5], [-0.5, t.h - 0.5]]
    .map(([x, y]) => pix2world(t.wcs, x, y));
}

const releaseLabel = () => RELEASES[state.release].label;

async function openTile(id) {
  const key = `${state.release}:${id}`;
  if (state.tile && state.tile.key === key) return state.tile;
  const resp = await fetch(`v1/tile/${id}?release=${state.release}`);
  if (!resp.ok) throw new Error(`tile ${id}: ${resp.status} ${await resp.text()}`);
  const info = await resp.json();
  const vis = info.bands[0];
  state.tile = { id, key, release: state.release, dataset: info.dataset, deep: !!info.deep,
                 width: vis.width, height: vis.height,
                 wcs: vis.wcs, bands: info.bands };
  return state.tile;
}

async function goTo(ra, dec) {
  const t = locateTile(ra, dec);
  if (!t) {
    setStatus(`No ${releaseLabel()} tile covers RA ${ra.toFixed(4)}, Dec ${dec.toFixed(4)}`, true);
    return false;
  }
  await openTile(t.id);
  [state.cx, state.cy] = world2pix(state.tile.wcs, ra, dec);
  saveHash();
  schedule();
  return true;
}

// ─────────────────────────────── map ⇄ POLISH view ───────────────────────────────

function setView(view) {
  state.view = view;
  document.body.classList.toggle("map-mode", view === "map");
  document.body.classList.toggle("detail-mode", view === "detail");
}

// Switch to the POLISH view at (ra, dec); stays on the map outside Q1 coverage.
async function enterDetail(ra, dec, zoom) {
  if (!locateTile(ra, dec)) {
    $("hint").innerHTML = `<span>No Euclid ${releaseLabel()} data here — zoom in on an <b>outlined</b> tile.</span>`;
    return false;
  }
  state.zoom = Math.max(ENTER_ZOOM, zoom);
  statusError = "";
  setView("detail");
  await goTo(ra, dec);
  ensureModel();
  return true;
}

function enterMap(ra, dec, fov) {
  setView("map");
  statusError = "";
  if (map.al) {
    map.al.gotoRaDec(ra, dec);
    map.al.setFoV(fov);
  }
  saveHash();
  updateMapStatus();
}

function detailToMap() {
  if (!state.tile) { enterMap(PRESETS[0][1], PRESETS[0][2], PRESETS[0][3]); return; }
  const [ra, dec] = pix2world(state.tile.wcs, state.cx, state.cy);
  enterMap(ra, dec, (canvas.clientWidth * SR_ARCSEC) / state.zoom / 3600);
}

// Map zoom expressed as the POLISH view's zoom (screen px per 0.05″ SR pixel).
function mapZoom() {
  const [fov] = map.al.getFov();
  return SR_ARCSEC / ((fov * 3600) / Math.max(1, $("map").clientWidth));
}

function mapCentre() {
  const [ra, dec] = map.al.getRaDec();
  return [ra, dec];
}

async function loadMap() {
  const A = (await import(ALADIN_URL)).default;
  await A.init;
  map.A = A;
  const [ra, dec] = [PRESETS[0][1], PRESETS[0][2]];
  map.al = A.aladin($("map"), {
    survey: [], target: `${ra} ${dec}`, fov: PRESETS[0][3], projection: "TAN", cooFrame: "ICRS",
    hipsList: [], log: false, samp: false, showReticle: false, showCooGrid: false,
    showCooGridControl: false, showProjectionControl: false, showLayersControl: false,
    showFullscreenControl: false, showSettingsControl: false, showShareControl: false,
    showSimbadPointerControl: false, showColorPickerControl: false, showContextMenu: false,
    showFrame: false, showFov: false, showCooLocation: true, showZoomControl: true, showStatusBar: true,
    backgroundColor: "rgb(11,13,16)",
  });
  map.al.contextMenu = null;
  showCoverage();
  for (const s of SURVEYS) $("survey").add(new Option(s.label, s.id));
  $("survey").addEventListener("change", (e) => setSurvey(e.target.value));
  setSurvey(await firstLiveSurvey());
  // Aladin's WASM core is borrowed while it fires these events: calling back into
  // it (getFov, getRaDec, setFoV…) from inside the callback throws "recursive use
  // of an object". Handle them on the next tick instead, coalesced.
  let pending = false;
  const later = () => {
    if (pending) return;
    pending = true;
    setTimeout(() => { pending = false; onMapMove(); }, 0);
  };
  map.al.on("zoomChanged", later);
  map.al.on("positionChanged", (p) => { if (!p || !p.dragging) later(); });
}

async function firstLiveSurvey() {
  for (const s of SURVEYS) {
    try {
      const ctrl = new AbortController(), timer = setTimeout(() => ctrl.abort(), 5000);
      const resp = await fetch(`${s.url}/properties`, { signal: ctrl.signal });
      clearTimeout(timer);
      if (resp.ok) return s.id;
    } catch { /* try the next one */ }
  }
  return SURVEYS[SURVEYS.length - 1].id;
}

function setSurvey(id) {
  const s = SURVEYS.find((x) => x.id === id) || SURVEYS[0];
  map.survey = s;
  $("survey").value = s.id;
  const layer = map.A.HiPS(s.url, { name: s.label, imgFormat: s.format || "png" });
  map.al.setBaseImageLayer(layer);
  if (s.color) {
    layer.setColormap(s.color.colormap, { stretch: s.color.stretch });
    layer.setCuts(s.color.minCut, s.color.maxCut);
  }
  updateMapStatus();
}

// Coverage on the map: tile outlines (all of them for Q1; for DR1's ~11,000 only
// those near the view once zoomed in) and the release's MOCs.
function showCoverage() {
  if (!map.al) return;
  for (const m of map.mocs) { try { map.al.removeOverlay(m); } catch { /* gone */ } }
  map.mocs = RELEASES[state.release].mocs.map((url) => {
    const moc = map.A.MOCFromURL(url, { color: "#6cb6ff", opacity: 0.12, lineWidth: 1, name: `${releaseLabel()} coverage` });
    map.al.addMOC(moc);
    return moc;
  });
  map.outlinesKey = null;
  refreshOutlines();
}

function refreshOutlines() {
  if (!map.al || !index) return;
  let tiles = index.tiles, key = state.release;
  if (tiles.length > ALL_OUTLINES_MAX) {
    const [fov] = map.al.getFov(), [ra, dec] = mapCentre();
    if (fov > 3) tiles = [];
    else {
      const r = fov * 0.75 + 0.4, cosd = Math.max(0.05, Math.cos(dec * D2R));
      tiles = tiles.filter((t) => {
        let dra = Math.abs(t.ra - ra);
        if (dra > 180) dra = 360 - dra;
        return Math.abs(t.dec - dec) < r && dra * cosd < r;
      });
    }
    key += ":" + tiles.map((t) => t.id).join(",");
  }
  if (key === map.outlinesKey) return;
  map.outlinesKey = key;
  if (map.outlines) { try { map.al.removeOverlay(map.outlines); } catch { /* gone */ } }
  map.outlines = map.A.graphicOverlay({ color: "#6cb6ff", lineWidth: 1, opacity: 0.55, name: `${releaseLabel()} tiles` });
  map.al.addOverlay(map.outlines);
  for (const t of tiles) map.outlines.add(map.A.polygon(tileOutline(t)));
}

async function setRelease(release, { keepView = true } = {}) {
  if (!RELEASES[release]) release = "q1";
  const changed = release !== state.release;
  state.release = release;
  $("release").value = release;
  index = await loadIndex(release);
  if (index.release !== state.release) return;     // switched again meanwhile
  if (changed || !map.outlines) showCoverage();
  if (keepView && changed && state.view === "detail" && state.tile) {
    const tileWcs = state.tile.wcs;
    const [ra, dec] = pix2world(tileWcs, state.cx, state.cy);
    state.tile = null;
    if (!(await enterDetail(ra, dec, state.zoom))) {
      enterMap(ra, dec, 0.3);
      setStatus(`No Euclid ${releaseLabel()} data here`, true);
    }
  }
  saveHash();
  updateMapStatus();
}

function onMapMove() {
  if (state.view !== "map" || !map.al) return;
  refreshOutlines();
  updateMapStatus();
  saveHash();
  if (mapZoom() >= ENTER_ZOOM) {
    const [ra, dec] = mapCentre();
    enterDetail(ra, dec, mapZoom()).catch((e) => setStatus(String(e.message || e), true));
  }
}

function updateMapStatus() {
  if (state.view !== "map") return;
  if (!map.al) { setStatus(statusError || "Loading the map…", !!statusError); return; }
  const arcsecPerPx = (map.al.getFov()[0] * 3600) / Math.max(1, $("map").clientWidth);
  const [ra, dec] = mapCentre();
  const covered = !!locateTile(ra, dec);
  const scale = arcsecPerPx >= 60 ? `${(arcsecPerPx / 60).toFixed(1)}′/px` : `${arcsecPerPx.toFixed(arcsecPerPx < 10 ? 2 : 0)}″/px`;
  setStatus(`${releaseLabel()} · map · ${map.survey ? map.survey.label : ""} · ${scale}`);
  $("hint").innerHTML = covered
    ? `<span>Zoom in to <b>0.1″/px</b> to super-resolve (now ${scale})</span><button id="enter-now">Super-resolve here</button>`
    : `<span>Euclid ${releaseLabel()} coverage is outlined. <b>Zoom in</b> on it to super-resolve.</span>`;
  const btn = document.getElementById("enter-now");
  if (btn) btn.onclick = () => enterDetail(ra, dec, ENTER_ZOOM);
}

function ensureModel() {
  if (!modelPromise) {
    modelPromise = loadModel()
      .then(() => { for (const blk of blocks.values()) if (blk.state === "lr") queueInfer(blk); schedule(); })
      .catch((e) => { modelPromise = null; setStatus(String(e.message || e), true); });
  }
  return modelPromise;
}

// After a pan, move to the tile that holds the view centre deepest.
async function retileIfNeeded() {
  if (!state.tile) return;
  const [ra, dec] = pix2world(state.tile.wcs, state.cx, state.cy);
  const t = locateTile(ra, dec);
  if (t && t.id !== state.tile.id) await goTo(ra, dec);
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
  const key = blockKey(tile.key, i, j);
  let blk = blocks.get(key);
  if (!blk) {
    blk = { key, tile: tile.key, tileId: tile.id, release: tile.release, i, j,
            state: "queued", lr: null, sr: {}, canvases: {}, used: 0 };
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
  if (!state.tile || blk.tile !== state.tile.key) return 1e18;
  const dx = (blk.i + 0.5) * B - state.cx, dy = (blk.j + 0.5) * B - state.cy;
  return dx * dx + dy * dy;
}

async function fetchBlock(blk) {
  const x0 = blk.i * B - M, y0 = blk.j * B - M;
  const resp = await fetch(`v1/cutout?release=${blk.release}&tile=${blk.tileId}&x0=${x0}&y0=${y0}&size=${N}`);
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
  if (blk.sr[tta] || !blk.input || !session) return;      // ensureModel re-queues once loaded
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
  if (!state.tile || blk.tile !== state.tile.key) return false;
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
  const abNorm = Math.asinh((W - BLACK_AB) / KNEE_AB);
  for (let r = 0; r < n; r++) {
    const row = (n - 1 - r) * n;
    for (let col = 0; col < n; col++) {
      const p = row + col, o = (r * n + col) * 4;
      const v = vis[p];
      if (Number.isNaN(v)) { px[o] = px[o + 1] = px[o + 2] = 28; px[o + 3] = 255; continue; }
      if (state.colour === "vis") {
        px[o] = px[o + 1] = px[o + 2] = stretch(v * sb, kV, W);
      } else if (state.colour === "ab") {
        const R = hb[p] * sb * TO_VIS_EQ.H_E, G = (y[p] * TO_VIS_EQ.Y_E + j[p] * TO_VIS_EQ.J_E) * 0.5 * sb, Bl = v * sb;
        const f = (c) => (c > BLACK_AB ? Math.min(1, Math.asinh((c - BLACK_AB) / KNEE_AB) / abNorm) : 0);
        const tr = f(R), tg = f(G), tb = f(Bl), L = (tr + tg + tb) / 3, k = state.saturation;
        px[o] = Math.max(0, Math.min(255, (L + k * (tr - L)) * 255)) | 0;   // saturation: scale
        px[o + 1] = Math.max(0, Math.min(255, (L + k * (tg - L)) * 255)) | 0; // the departure
        px[o + 2] = Math.max(0, Math.min(255, (L + k * (tb - L)) * 255)) | 0; // from grey
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
  const key = `${which}:${state.colour}:${state.white}:${state.saturation}:${state.tta}`;
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
  if (!state.tile || state.view !== "detail") return;

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
  // Block edges snapped to device pixels, shared by neighbours: no hairline seams.
  const snap = (v) => Math.round(v * dpr) / dpr;
  for (const blk of visible) {
    const [ax, ay] = toScreen(blk.i * B, (blk.j + 1) * B);
    const [bx] = toScreen((blk.i + 1) * B, blk.j * B);
    const sx = snap(ax), sy = snap(ay), size = snap(bx) - sx;
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
  ctx.beginPath(); ctx.rect(Math.max(xa, sx), sy, Math.min(xb, sx + size) - Math.max(xa, sx), size); ctx.clip();
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
  const i = Math.floor(x / B), j = Math.floor(y / B), blk = blocks.get(blockKey(state.tile.key, i, j));
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
    `${releaseLabel()} tile ${state.tile.id}${state.tile.dataset ? ` (${state.tile.dataset})` : ""}  x ${x.toFixed(1)}  y ${y.toFixed(1)}`;
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
  const rel = `${releaseLabel()}${state.tile && state.tile.deep ? " deep field (model trained on Wide depth)" : ""}`;
  const model = session ? `${rel} · ${providerName}${ms ? ` · ${Math.round(ms)} ms/block${state.tta ? " ×8" : ""}` : ""}` : "loading model…";
  const work = visible.length === 0 ? "zoom in to load blocks" : `${srDone}/${visible.length} blocks` +
    (waiting ? ` · fetching ${waiting}` : "") + (pendingInfer ? ` · model queue ${pendingInfer}` : "");
  setStatus(`${model} · ${work}`);
}

// ───────────────────────────────── interaction ─────────────────────────────────

function saveHash() {
  const rel = `&rel=${state.release}`;
  if (state.view === "map") {
    if (!map.al) return;
    const [ra, dec] = mapCentre();
    history.replaceState(null, "", `#ra=${ra.toFixed(6)}&dec=${dec.toFixed(6)}&fov=${map.al.getFov()[0].toPrecision(4)}${rel}`);
    return;
  }
  if (!state.tile) return;
  const [ra, dec] = pix2world(state.tile.wcs, state.cx, state.cy);
  history.replaceState(null, "", `#ra=${ra.toFixed(6)}&dec=${dec.toFixed(6)}&z=${+state.zoom.toPrecision(4)}${rel}`);
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
  const text = $("where").value.trim();
  const where = parseWhere(text);
  try {
    if (!where) {                                  // an object name: Sesame, via Aladin
      if (!text || !map.al) { setStatus("Enter a name, 'RA Dec' in degrees, or 'tile x y'", true); return; }
      map.al.gotoObject(text, {
        success: ([ra, dec]) => enterMap(ra, dec, 0.05),
        error: () => setStatus(`Could not resolve "${text}"`, true),
      });
      return;
    }
    if (where.tile) {
      setView("detail"); ensureModel();
      await openTile(where.tile); state.cx = where.x; state.cy = where.y; saveHash(); schedule();
    } else if (!(await enterDetail(where.ra, where.dec, Math.max(state.zoom, 1)))) {
      enterMap(where.ra, where.dec, 0.2);
      setStatus(`No Euclid ${releaseLabel()} data at RA ${where.ra.toFixed(4)}, Dec ${where.dec.toFixed(4)}`, true);
    }
  } catch (e) { setStatus(String(e.message || e), true); }
}

function setZoom(z, sx, sy) {
  if (z < EXIT_ZOOM) { state.zoom = EXIT_ZOOM; detailToMap(); return; }
  z = Math.min(8, z);
  if (sx != null) {                               // keep the point under the cursor fixed
    const [x, y] = fromScreen(sx, sy);
    state.zoom = z;
    const [x2, y2] = fromScreen(sx, sy);
    state.cx += x - x2; state.cy += y - y2;
  } else state.zoom = z;
  saveHash(); schedule();
}

function bindUi() {
  PRESETS.forEach(([name], i) => $("presets").add(new Option(name, String(i))));
  $("presets").addEventListener("change", async (e) => {
    if (!e.target.value) return;
    let [, ra, dec, fov, release] = PRESETS[+e.target.value];
    e.target.value = "";
    await setRelease(release, { keepView: false });
    if (release === "dr1" && ra === 0 && dec === 0) [ra, dec] = medianTile();
    enterMap(ra, dec, fov);
  });
  $("release").addEventListener("change", (e) => {
    setRelease(e.target.value).catch((err) => setStatus(String(err.message || err), true));
  });
  $("tomap").addEventListener("click", detailToMap);
  $("enter").addEventListener("click", () => {
    if (!map.al) return;
    const [ra, dec] = mapCentre();
    enterDetail(ra, dec, Math.max(ENTER_ZOOM, mapZoom()));
  });
  $("search").addEventListener("submit", onSearch);
  document.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  $("colour").addEventListener("change", (e) => { state.colour = e.target.value; schedule(); });
  // White level 300 → 3·10⁵ e⁻ (VIS-equivalent per LR pixel): bright nearby galaxies
  // need the top decade to keep their cores unclipped.
  const white = () => { state.white = Math.round(300 * Math.pow(10, 3 * (100 - +$("white").value) / 100)); schedule(); };
  $("white").addEventListener("input", white); white();
  const saturation = () => { state.saturation = +$("saturation").value / 50; schedule(); };
  $("saturation").addEventListener("input", saturation); saturation();
  const satLabel = () => { $("saturation").parentElement.hidden = state.colour !== "ab"; };
  $("colour").addEventListener("change", satLabel); satLabel();
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
    if (e.target.tagName === "INPUT" || state.view !== "detail") return;
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

// A representative point of the current index: the median tile centre.
function medianTile() {
  const ras = index.tiles.map((t) => t.ra).sort((a, b) => a - b);
  const decs = index.tiles.map((t) => t.dec).sort((a, b) => a - b);
  return [ras[ras.length >> 1], decs[decs.length >> 1]];
}

async function main() {
  bindUi();
  setView("map");
  const hash = new URLSearchParams(location.hash.slice(1));
  state.release = RELEASES[hash.get("rel")] ? hash.get("rel") : "q1";
  $("release").value = state.release;
  index = await loadIndex(state.release);
  const hasPos = hash.has("ra") && Number.isFinite(+hash.get("ra")) && Number.isFinite(+hash.get("dec"));
  const [ra, dec] = hasPos ? [+hash.get("ra"), +hash.get("dec")] : [PRESETS[0][1], PRESETS[0][2]];
  try {
    await loadMap();
  } catch (e) {                                  // no WebGL2 or Aladin unreachable: POLISH view only
    console.error("map unavailable", e);
    statusError = "";
    await enterDetail(ra, dec, hash.get("z") ? +hash.get("z") : 1);
    setStatus("Map unavailable here (needs WebGL2); showing the POLISH view only", true);
    return;
  }
  if (hasPos && hash.get("z")) {
    if (await enterDetail(ra, dec, +hash.get("z"))) return;
  }
  enterMap(ra, dec, hash.get("fov") ? +hash.get("fov") : PRESETS[0][3]);
}

// For checks from the console: the block cache and the model call.
window.polish = { state, blocks, map, superResolve: (e, tta) => superResolve(e, tta), decodeFrame };

main().catch((e) => setStatus(String(e.message || e), true));
