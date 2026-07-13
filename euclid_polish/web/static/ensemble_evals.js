/**
 * Frontend renderers for the /ensemble "Evaluations" card.
 *
 * All four figures (power spectrum, std-vs-error, std-vs-brightness,
 * calibration) are drawn CLIENT-SIDE from /ensemble/evals.json — the server
 * only computes numbers (once per Evaluate, or on an explicit ↻ Recompute).
 * Styling choices (member-line coloring by loss/depth, tab switches) are
 * instant redraws of the cached payload; nothing touches the cubes.
 *
 * Every plot works in TRANSFORMED coordinates (log10 electrons, asinh
 * brightness, log10 angular scale), so one linear pixel mapper + custom tick
 * labels covers everything. Canvases render at a fixed internal resolution
 * and scale responsively via CSS, so hidden tabs can be drawn eagerly.
 */

const LOSS_COLORS = { l1: "#3b6fb0", l2: "#5aae61", l3: "#e08214",
                      berhu: "#9970ab" };
const GROUP_PALETTE = ["#3b6fb0", "#5aae61", "#e08214", "#d6604d",
                      "#9970ab", "#e8c944", "#59c7d6", "#e07356"];
const VIS_COLOR = "#3b6fb0";
const ACCENT = "#d6604d";
const NO_DATA = "#7a8292";

const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴",
              5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
const pow10Label = (n) =>
  (n === 0 ? "1" : n === 1 ? "10" :
   "10" + String(n).split("").map((c) => SUP[c] ?? c).join(""));

// Compact viridis (t ∈ [0,1]).
const VIRIDIS = [[68, 1, 84], [59, 82, 139], [33, 145, 140],
                 [94, 201, 98], [253, 231, 37]];
function viridis(t) {
  const x = Math.max(0, Math.min(1, t)) * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(x));
  const f = x - i;
  const c = VIRIDIS[i].map((a, k) => Math.round(a + (VIRIDIS[i + 1][k] - a) * f));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

// ── canvas plot kit ─────────────────────────────────────────────────

const MARGIN = { l: 62, r: 14, t: 30, b: 44 };

function makePlot(parent, w, h, title) {
  const cv = document.createElement("canvas");
  cv.width = w * 2;                       // 2× internal → crisp downscale
  cv.height = h * 2;
  cv.style.cssText = "width:100%;max-width:" + w + "px;height:auto;display:block;";
  parent.appendChild(cv);
  const ctx = cv.getContext("2d");
  ctx.scale(2, 2);
  ctx.font = "11px system-ui, sans-serif";
  const X = (x, d) => MARGIN.l + (x - d[0]) / (d[1] - d[0]) * (w - MARGIN.l - MARGIN.r);
  const Y = (y, d) => h - MARGIN.b - (y - d[0]) / (d[1] - d[0]) * (h - MARGIN.t - MARGIN.b);
  if (title) {
    ctx.fillStyle = "#222";
    ctx.font = "600 12px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(title, MARGIN.l + (w - MARGIN.l - MARGIN.r) / 2, 16);
    ctx.font = "11px system-ui, sans-serif";
  }
  return { cv, ctx, w, h, X, Y };
}

function frameAxes(p, xd, yd, { xticks, yticks, xlabel, ylabel }) {
  const { ctx, w, h } = p;
  ctx.strokeStyle = "#888";
  ctx.lineWidth = 1;
  ctx.strokeRect(MARGIN.l, MARGIN.t, w - MARGIN.l - MARGIN.r,
                 h - MARGIN.t - MARGIN.b);
  ctx.fillStyle = "#333";
  ctx.textAlign = "center";
  for (const [v, lbl] of xticks) {
    if (v < xd[0] - 1e-9 || v > xd[1] + 1e-9) continue;
    const x = p.X(v, xd);
    ctx.strokeStyle = "#00000018";
    ctx.beginPath(); ctx.moveTo(x, MARGIN.t); ctx.lineTo(x, h - MARGIN.b); ctx.stroke();
    ctx.fillText(lbl, x, h - MARGIN.b + 14);
  }
  ctx.textAlign = "right";
  for (const [v, lbl] of yticks) {
    if (v < yd[0] - 1e-9 || v > yd[1] + 1e-9) continue;
    const y = p.Y(v, yd);
    ctx.strokeStyle = "#00000018";
    ctx.beginPath(); ctx.moveTo(MARGIN.l, y); ctx.lineTo(w - MARGIN.r, y); ctx.stroke();
    ctx.fillText(lbl, MARGIN.l - 6, y + 3.5);
  }
  ctx.textAlign = "center";
  ctx.fillText(xlabel, MARGIN.l + (w - MARGIN.l - MARGIN.r) / 2, h - 8);
  ctx.save();
  ctx.translate(13, MARGIN.t + (h - MARGIN.t - MARGIN.b) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(ylabel, 0, 0);
  ctx.restore();
}

function drawLine(p, xd, yd, xs, ys, { color, width = 1, alpha = 1, dash = [],
                                       dots = false } = {}) {
  const { ctx } = p;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.globalAlpha = alpha;
  ctx.lineWidth = width;
  ctx.setLineDash(dash);
  ctx.beginPath();
  let pen = false;
  for (let i = 0; i < xs.length; i++) {
    if (xs[i] == null || ys[i] == null || !isFinite(xs[i]) || !isFinite(ys[i])) {
      pen = false;
      continue;
    }
    const x = p.X(xs[i], xd), y = p.Y(Math.min(Math.max(ys[i], yd[0]), yd[1]), yd);
    if (pen) ctx.lineTo(x, y); else { ctx.moveTo(x, y); pen = true; }
  }
  ctx.stroke();
  if (dots) {
    for (let i = 0; i < xs.length; i++) {
      if (xs[i] == null || ys[i] == null) continue;
      ctx.beginPath();
      ctx.arc(p.X(xs[i], xd), p.Y(Math.min(Math.max(ys[i], yd[0]), yd[1]), yd),
              2, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
  ctx.restore();
}

function vline(p, xd, yd, x, opts) { drawLine(p, xd, yd, [x, x], [yd[0], yd[1]], opts); }
function hline(p, xd, yd, y, opts) { drawLine(p, xd, yd, [xd[0], xd[1]], [y, y], opts); }

function heatmap(p, xd, yd, xEdges, yEdges, hist) {
  // hist[i][j]: i → x bin, j → y bin; 0 = transparent; log color scale.
  const { ctx } = p;
  let max = 1;
  for (const row of hist) for (const v of row) if (v > max) max = v;
  const lmax = Math.log(max);
  for (let i = 0; i < hist.length; i++) {
    const x0 = p.X(xEdges[i], xd), x1 = p.X(xEdges[i + 1], xd);
    for (let j = 0; j < hist[i].length; j++) {
      const v = hist[i][j];
      if (!v) continue;
      const y0 = p.Y(yEdges[j], yd), y1 = p.Y(yEdges[j + 1], yd);
      ctx.fillStyle = viridis(Math.log(v) / lmax);
      ctx.fillRect(x0, y1, x1 - x0 + 0.5, y0 - y1 + 0.5);
    }
  }
}

function legendHtml(entries) {
  const div = document.createElement("div");
  div.style.cssText =
    "display:flex;flex-wrap:wrap;gap:12px;font-size:11px;color:#444;margin:2px 0 6px;";
  for (const [label, color, dash] of entries) {
    const item = document.createElement("span");
    const sw = document.createElement("span");
    sw.style.cssText = `display:inline-block;width:18px;height:0;margin-right:4px;` +
      `vertical-align:middle;border-top:2px ${dash ? "dashed" : "solid"} ${color};`;
    item.appendChild(sw);
    item.appendChild(document.createTextNode(label));
    div.appendChild(item);
  }
  return div;
}

const decadeTicks = (lo, hi) => {
  const out = [];
  for (let n = Math.ceil(lo); n <= Math.floor(hi); n++) out.push([n, pow10Label(n)]);
  return out;
};

// ── member coloring ─────────────────────────────────────────────────

function memberColors(members, mode) {
  if (mode === "loss") {
    return members.map((m) => {
      const loss = (m.loss || "l1").toLowerCase();
      return [loss.toUpperCase(), LOSS_COLORS[loss] || NO_DATA];
    });
  }
  if (mode === "depth") {
    const depths = [...new Set(members.map((m) => m.blocks).filter(Boolean))]
      .sort((a, b) => a - b);
    const byDepth = new Map(depths.map((d, i) =>
      [d, GROUP_PALETTE[i % GROUP_PALETTE.length]]));
    return members.map((m) =>
      m.blocks ? [`${m.blocks}b`, byDepth.get(m.blocks)] : ["?", NO_DATA]);
  }
  return members.map(() => ["individual models", VIS_COLOR]);
}

// ── figure renderers ────────────────────────────────────────────────

function renderPS(el, payload, mode) {
  el.innerHTML = "";
  const ps = payload.ps;
  if (!ps || !ps.theta) {
    el.innerHTML = '<p class="muted">no power-spectrum data in the payload — run Evaluate.</p>';
    return;
  }
  const g = payload.guides || {};
  const lt = (a) => (a || []).map((v) => (v == null || v <= 0 ? null : Math.log10(v)));
  const theta = lt(ps.theta);
  const xd = [Math.log10(g.theta_min || 0.05),
              Math.max(...theta.filter((v) => v != null))];
  const xticks = [0.05, 0.1, 0.2, 0.5, 1, 2, 5].map((v) => [Math.log10(v), String(v)]);
  const colors = memberColors(payload.members || [], mode);
  const uniformAlpha = mode ? 0.55 : 0.3;

  const COMB_COLOR = "#ee7733";
  const COMBINED_COLOR = "#9d6cff";
  const LR_COLOR = "#c23b3b";                 // baseline-to-beat (no SR)
  const hasComb = (arr) => (arr || []).some((v) => v != null && isFinite(v));
  const hasLR = hasComb(ps.r_lr);

  // A single, LARGE cross-correlation r(k) panel (the transfer-function panel
  // was removed — r(k) is the headline metric). Fills the card width.
  const yd = [0, 1.05];
  const box = document.createElement("div");
  box.style.cssText = "width:100%;";
  el.appendChild(box);
  const w = Math.max(720, Math.min(1180, (el.clientWidth || 980) - 8));
  const p = makePlot(box, w, Math.round(w * 0.5),
                     "cross-correlation  r(k) vs HR  (1 = perfect; higher is better)");
  frameAxes(p, xd, yd, {
    xticks,
    yticks: [[0, "0"], [0.25, "0.25"], [0.5, "0.5"], [0.75, "0.75"], [1.0, "1"]],
    xlabel: "angular scale θ = 1/2k [arcsec]",
    ylabel: "r(k)  [VIS]",
  });
  hline(p, xd, yd, 1.0, { color: "#888", dash: [2, 3] });
  vline(p, xd, yd, Math.log10(g.lr_scale || 0.1),
        { color: "#333", width: 1.3, dash: [6, 3] });
  vline(p, xd, yd, Math.log10(g.vis_fwhm || 0.16),
        { color: VIS_COLOR, alpha: 0.55, width: 1.5, dash: [5, 2] });
  // per-member r(k)
  for (let j = 0; j < (ps.r_members || []).length; j++) {
    const [, color] = colors[j] || ["", VIS_COLOR];
    drawLine(p, xd, yd, theta, ps.r_members[j],
             { color, width: 0.9, alpha: uniformAlpha });
  }
  // truth-free model–model agreement
  for (const pair of ps.r_pairs || []) {
    drawLine(p, xd, yd, theta, pair, { color: "#777", width: 0.6, alpha: 0.15 });
  }
  drawLine(p, xd, yd, theta, ps.r_cross || [],
           { color: "#555", width: 2, dash: [6, 3] });
  // LR baseline (no SR) — the floor the network must beat
  if (hasLR) {
    drawLine(p, xd, yd, theta, ps.r_lr, { color: LR_COLOR, width: 2.5, dash: [7, 4] });
  }
  drawLine(p, xd, yd, theta, ps.r, { color: VIS_COLOR, width: 2.5, dots: true });
  if (hasComb(ps.r_comb)) {
    drawLine(p, xd, yd, theta, ps.r_comb, { color: COMB_COLOR, width: 2, dots: true });
  }
  if (hasComb(ps.r_combined)) {
    drawLine(p, xd, yd, theta, ps.r_combined, { color: COMBINED_COLOR, width: 2, dots: true });
  }

  const groups = [...new Map(colors.map(([l, c]) => [l, c])).entries()];
  el.appendChild(legendHtml([
    ...(hasLR ? [["LR baseline (no SR)", LR_COLOR, true]] : []),
    ...groups.map(([l, c]) => [l, c, false]),
    ["ensemble mean", VIS_COLOR, false],
    ...(hasComb(ps.r_comb) ? [["combiner", COMB_COLOR, false]] : []),
    ...(hasComb(ps.r_combined) ? [["combined combiner", COMBINED_COLOR, false]] : []),
    ["model–model r̃(k) (no HR)", "#555", true],
    ["LR sampling (0.1″)", "#333", true],
    ["VIS PSF FWHM", VIS_COLOR, true],
  ]));
}

function renderStdErr(el, payload) {
  el.innerHTML = "";
  const d = payload.std_err;
  if (!d) { el.innerHTML = '<p class="muted">no data — run Evaluate.</p>'; return; }
  const xd = [d.edges[0], d.edges[d.edges.length - 1]];
  const p = makePlot(el, 620, 480,
    `Does disagreement predict error?  (VIS, ${payload.n_fields} fields, ` +
    `${payload.n_members} members)`);
  frameAxes(p, xd, xd, {
    xticks: decadeTicks(xd[0], xd[1]),
    yticks: decadeTicks(xd[0], xd[1]),
    xlabel: "cross-member per-pixel std σ [e⁻]",
    ylabel: "actual error |mean − HR| [e⁻]",
  });
  heatmap(p, xd, xd, d.edges, d.edges, d.hist);
  drawLine(p, xd, xd, [xd[0], xd[1]], [xd[0], xd[1]],
           { color: "#333", dash: [6, 3], width: 1.2 });
  const off = Math.log10(0.6745);
  drawLine(p, xd, xd, [xd[0], xd[1]], [xd[0] + off, xd[1] + off],
           { color: "#333", dash: [2, 3], width: 1.2 });
  drawLine(p, xd, xd, d.med_std, d.med_err,
           { color: ACCENT, width: 2, dots: true });
  el.appendChild(legendHtml([
    ["median |error| per std bin", ACCENT, false],
    ["|error| = std", "#333", true],
    ["0.674·std (calibrated Gaussian)", "#333", true],
  ]));
}

function renderBrightStd(el, payload) {
  el.innerHTML = "";
  const d = payload.bright_std;
  if (!d) { el.innerHTML = '<p class="muted">no data — run Evaluate.</p>'; return; }
  const xd = [d.bright_edges[0], d.bright_edges[d.bright_edges.length - 1]];
  const yd = [d.std_edges[0], d.std_edges[d.std_edges.length - 1]];
  const stretch = d.stretch || 100;
  const tickE = [[0, "0"], [100, "100"], [1e3, "10³"], [1e4, "10⁴"],
                 [1e5, "10⁵"], [1e6, "10⁶"]];
  const p = makePlot(el, 620, 480,
    `Where does disagreement live?  (VIS, ${payload.n_fields} fields)`);
  frameAxes(p, xd, yd, {
    xticks: tickE.map(([e, l]) => [Math.asinh(e / stretch), l]),
    yticks: decadeTicks(yd[0], yd[1]),
    xlabel: "HR pixel brightness [e⁻] (asinh-spaced axis)",
    ylabel: "cross-member per-pixel std σ [e⁻]",
  });
  heatmap(p, xd, yd, d.bright_edges, d.std_edges, d.hist);
  drawLine(p, xd, yd, d.bright, d.lo, { color: ACCENT, width: 1, alpha: 0.5 });
  drawLine(p, xd, yd, d.bright, d.hi, { color: ACCENT, width: 1, alpha: 0.5 });
  drawLine(p, xd, yd, d.bright, d.med, { color: ACCENT, width: 2, dots: true });
  el.appendChild(legendHtml([
    ["median std per brightness bin", ACCENT, false],
    ["16–84%", ACCENT, true],
  ]));
}

function renderCalibration(el, payload) {
  el.innerHTML = "";
  const c = payload.calibration;
  if (!c) { el.innerHTML = '<p class="muted">no data — run Evaluate.</p>'; return; }
  const st = c.stats || {};
  const row = document.createElement("div");
  row.style.cssText = "display:flex;flex-wrap:wrap;gap:10px;";
  el.appendChild(row);

  // Left: z-score PDF (log y) vs N(0, 1).
  const zc = [];
  for (let i = 0; i < c.pdf.length; i++) {
    zc.push((c.z_edges[i] + c.z_edges[i + 1]) / 2);
  }
  const lpdf = c.pdf.map((v) => (v > 0 ? Math.log10(v) : null));
  const ymin = Math.max(-8, Math.min(...lpdf.filter((v) => v != null)) - 0.3);
  const xd = [c.z_edges[0], c.z_edges[c.z_edges.length - 1]];
  const yd = [ymin, 0.4];
  const box1 = document.createElement("div");
  box1.style.cssText = "flex:1 1 420px;min-width:320px;";
  row.appendChild(box1);
  const p1 = makePlot(box1, 560, 380, "Is the member std a calibrated error bar?");
  frameAxes(p1, xd, yd, {
    xticks: [-8, -6, -4, -2, 0, 2, 4, 6, 8].map((v) => [v, String(v)]),
    yticks: decadeTicks(Math.ceil(ymin), 0),
    xlabel: "z = (mean − HR) / σ",
    ylabel: "pixel PDF (log)",
  });
  const zz = [], nn = [];
  for (let z = xd[0]; z <= xd[1]; z += 0.1) {
    zz.push(z);
    const v = Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
    nn.push(v > 1e-12 ? Math.log10(v) : null);
  }
  drawLine(p1, xd, yd, zz, nn, { color: "#333", dash: [6, 3], width: 1.2 });
  drawLine(p1, xd, yd, zc, lpdf, { color: VIS_COLOR, width: 1.6 });
  const fmt = (v, d = 1) => (v == null ? "—" : (v * 100).toFixed(d));
  const stats = document.createElement("div");
  stats.style.cssText = "font-size:11px;color:#444;margin-top:2px;";
  stats.textContent =
    `σ(z) = ${st.sigma_z == null ? "—" : st.sigma_z.toFixed(2)} (1 if calibrated)` +
    ` · coverage |z|<1: ${fmt(st.cover1)}% (68.3) · |z|<2: ${fmt(st.cover2)}%` +
    ` (95.4) · |z|<3: ${fmt(st.cover3)}% (99.7)`;
  box1.appendChild(stats);

  // Right: per-field mean std vs RMSE (log-log scatter).
  const xs = [], ys = [];
  for (let i = 0; i < (c.field_std || []).length; i++) {
    const a = c.field_std[i], b = (c.field_rmse || [])[i];
    if (a > 0 && b > 0) { xs.push(Math.log10(a)); ys.push(Math.log10(b)); }
  }
  const box2 = document.createElement("div");
  box2.style.cssText = "flex:1 1 420px;min-width:320px;";
  row.appendChild(box2);
  if (xs.length >= 2) {
    const pad = 0.3;
    const xd2 = [Math.min(...xs) - pad, Math.max(...xs) + pad];
    const yd2 = [Math.min(...ys) - pad, Math.max(...ys) + pad];
    const mx = xs.reduce((s, v) => s + v, 0) / xs.length;
    const my = ys.reduce((s, v) => s + v, 0) / ys.length;
    let sxy = 0, sxx = 0, syy = 0;
    for (let i = 0; i < xs.length; i++) {
      sxy += (xs[i] - mx) * (ys[i] - my);
      sxx += (xs[i] - mx) ** 2;
      syy += (ys[i] - my) ** 2;
    }
    const pearson = sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : NaN;
    const p2 = makePlot(box2, 560, 380,
      `Does disagreement rank fields?  (r = ${pearson.toFixed(2)}, ` +
      `${xs.length} fields)`);
    frameAxes(p2, xd2, yd2, {
      xticks: decadeTicks(xd2[0], xd2[1]),
      yticks: decadeTicks(yd2[0], yd2[1]),
      xlabel: "per-field mean member std [e⁻]",
      ylabel: "per-field RMSE (mean vs HR) [e⁻]",
    });
    const { ctx } = p2;
    ctx.fillStyle = VIS_COLOR;
    ctx.globalAlpha = 0.7;
    for (let i = 0; i < xs.length; i++) {
      ctx.beginPath();
      ctx.arc(p2.X(xs[i], xd2), p2.Y(ys[i], yd2), 2.6, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  } else {
    box2.innerHTML = '<p class="muted">not enough per-field data.</p>';
  }
}

// ── mount ───────────────────────────────────────────────────────────

export async function mountEnsembleEvals(card, url) {
  const panels = {
    "power-spectrum": card.querySelector('[data-plot="ps"]'),
    "std-error": card.querySelector('[data-plot="std-err"]'),
    "std-brightness": card.querySelector('[data-plot="bright-std"]'),
    calibration: card.querySelector('[data-plot="calibration"]'),
  };
  const colorSel = card.querySelector("#ens-ps-color");
  // The evals payload is detached per star regime; follow the top mode toggle.
  const modeSel = document.getElementById("ens-mode");
  const curMode = () => (modeSel ? modeSel.value : "starfull");
  let payload = null;

  function renderAll() {
    if (!payload) return;
    renderPS(panels["power-spectrum"], payload, colorSel ? colorSel.value : "");
    renderStdErr(panels["std-error"], payload);
    renderBrightStd(panels["std-brightness"], payload);
    renderCalibration(panels.calibration, payload);
  }

  async function load(fresh) {
    const q = new URLSearchParams({ mode: curMode() });
    if (fresh) { q.set("fresh", "1"); q.set("_t", Date.now()); }
    const r = await fetch(url + "?" + q.toString());
    if (!r.ok) {
      const msg = '<p class="muted">no evaluation data cached — run ' +
        "“Evaluate on test set”, then ↻ Recompute.</p>";
      for (const el of Object.values(panels)) if (el) el.innerHTML = msg;
      return false;
    }
    payload = await r.json();
    renderAll();
    return true;
  }

  // Instant client-side re-style — the whole point of the JSON payload.
  if (colorSel) {
    colorSel.addEventListener("change", () =>
      renderPS(panels["power-spectrum"], payload,
               colorSel ? colorSel.value : ""));
  }
  const btn = card.querySelector("#ens-eval-plot-regen");
  if (btn) {
    btn.addEventListener("click", async () => {
      const label = btn.textContent;
      btn.disabled = true;
      btn.textContent = "↻ Recomputing…";
      try { await load(true); } finally {
        btn.disabled = false;
        btn.textContent = label;
      }
    });
  }
  // Switching the star regime swaps to that regime's detached evals payload.
  document.addEventListener("ensemble-mode-change", () => { load(false); });
  await load(false);
}
