// Combiner card renderer. Pulls fit-time info from /ensemble/combiner.json
// (member weight-vs-brightness gate curves, val loss) and test-time metrics from
// /ensemble/evals.json (combiner vs mean vs best member, VIS stretched PSNR).
// All drawn client-side; no server round-trips after load.

const COLORS = [
  "#4477aa", "#ee6677", "#228833", "#ccbb44", "#66ccee", "#aa3377",
  "#ee8866", "#44bb99", "#bbbb44", "#99ddff", "#cc6677", "#555555",
];
const BAND_COLORS = { VIS: "#4477aa", Y_E: "#66ccee", J_E: "#228833", H_E: "#ccbb44" };
const TXT = "var(--text-primary,#333)";
// Member-line coloring, matching the power-spectrum plot's palettes.
const LOSS_COLORS = { l1: "#3b6fb0", l2: "#5aae61", l3: "#e08214", mse: "#1b9e9a", berhu: "#9970ab" };
const GROUP_PALETTE = ["#3b6fb0", "#5aae61", "#e08214", "#d6604d",
                       "#9970ab", "#e8c944", "#59c7d6", "#e07356"];
const NO_DATA = "#7a8292";
const VIRIDIS = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
function viridis(t) {
  const x = Math.max(0, Math.min(1, t)) * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(x)), f = x - i;
  const c = VIRIDIS[i].map((a, k) => Math.round(a + (VIRIDIS[i + 1][k] - a) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

// Per-member metadata aligned to comb.member_labels (from the combiner payload,
// falling back to evals.members), then a color+tag per member for a color mode.
function memberMeta(comb, evals) {
  const src = (comb.members && comb.members.length) ? comb.members
    : ((evals && evals.members) || []);
  const byLabel = new Map(src.map((m) => [m.label, m]));
  return (comb.member_labels || []).map((lbl, i) => byLabel.get(lbl) || src[i] || { label: lbl });
}

function memberColorsFor(members, mode) {
  if (members.some((m) => m.expert_color)) {
    return members.map((m, i) => ({
      color: m.expert_color || COLORS[i % COLORS.length], tag: "expert",
    }));
  }
  if (mode === "loss") {
    return members.map((m) => {
      const l = (m.loss || "l1").toLowerCase();
      return { color: LOSS_COLORS[l] || NO_DATA, tag: l.toUpperCase() };
    });
  }
  if (mode === "depth") {
    const ds = [...new Set(members.map((m) => m.blocks).filter(Boolean))].sort((a, b) => a - b);
    const by = new Map(ds.map((d, i) => [d, GROUP_PALETTE[i % GROUP_PALETTE.length]]));
    return members.map((m) => (m.blocks ? { color: by.get(m.blocks), tag: `${m.blocks}b` }
      : { color: NO_DATA, tag: "?" }));
  }
  if (mode === "psnr") {
    const ps = members.map((m) => m.psnr).filter((v) => v != null && isFinite(v));
    const lo = Math.min(...ps), hi = Math.max(...ps);
    return members.map((m) => ((m.psnr != null && isFinite(m.psnr))
      ? { color: viridis((m.psnr - lo) / ((hi - lo) || 1)), tag: (+m.psnr).toFixed(2) }
      : { color: NO_DATA, tag: "?" }));
  }
  return members.map(() => ({ color: null, tag: null }));   // uniform → per-member COLORS
}

function fmt(x, d = 3) {
  return (x == null || !isFinite(x)) ? "—" : (Math.round(x * 10 ** d) / 10 ** d);
}

function fmtE(e) {
  const a = Math.abs(e);
  if (a < 1000) return String(Math.round(e));
  if (a < 1e6) return (e / 1e3).toFixed(a < 1e4 ? 1 : 0) + "k";
  return (e / 1e6).toFixed(1) + "M";
}

function metricsHtml(comb, evals) {
  const kind = comb.kind || "raw_incremental_minmeanmax_rbf";
  const modelLabel = "minibatched convex all-asinh RBF";
  const c = evals && ((evals.model_combiners || {})[kind] || evals.combiner);
  let rows = "<tr><th>series</th><th>VIS asinh L1</th><th>VIS PSNR (dB, asinh)</th></tr>";
  if (c && c.available) {
    const g = (v) => (v == null ? "" : ` <span class="muted">(${v >= 0 ? "+" : ""}${fmt(v, 2)} dB)</span>`);
    const vsMean = c.psnr != null && c.ensemble_mean_psnr != null ? c.psnr - c.ensemble_mean_psnr : null;
    const vsBest = c.psnr != null && c.best_member_psnr != null ? c.psnr - c.best_member_psnr : null;
    rows += `<tr><td><b>${modelLabel}</b></td><td><b>${fmt(c.asinh_l1, 5)}</b></td><td><b>${fmt(c.psnr, 3)}</b></td></tr>`;
    rows += `<tr><td>ensemble mean</td><td>${fmt(c.ensemble_mean_asinh_l1, 5)}</td><td>${fmt(c.ensemble_mean_psnr, 3)}${g(vsMean)}</td></tr>`;
    rows += `<tr><td>best PSNR member${c.best_member_label ? ` (${c.best_member_label})` : ""}</td>`
          + `<td>—</td><td>${fmt(c.best_member_psnr, 3)}${g(vsBest)}</td></tr>`;
    rows += `<tr><td>best L1 member${c.best_member_l1_label ? ` (${c.best_member_l1_label})` : ""}</td>`
          + `<td>${fmt(c.best_member_asinh_l1, 5)}</td><td>—</td></tr>`;
  } else {
    rows += `<tr><td colspan="3" class="muted">Run “Evaluate on test set” (starfull) to score the combiner on test.</td></tr>`;
  }
  const stale = comb.stale
    ? ` · <b style="color:#c33">STALE</b> (membership changed since fit — re-fit)` : "";
  const preview = comb.fit_meta && comb.fit_meta.preview
    ? ` · <b style="color:#b06c00">PREVIEW</b> (${comb.fit_meta.num_images || "?"} validation fields)` : "";
  return `<div style="display:flex;justify-content:center"><table class="mini-table">${rows}</table></div>`
       + `<p class="hint" style="text-align:center">Fit asinh L1 on validate: <b>${fmt(comb.val_l1, 4)}</b>`
       + ` · ${modelLabel}, K=${comb.n_kernels} kernels`
       + ` · shared convex weights${stale}${preview}</p>`;
}

function fmtSteps(s) {
  if (s == null || !isFinite(s)) return null;
  return s >= 1000 ? (s / 1000).toFixed(s < 10000 ? 1 : 0) + "k" : String(s);
}
function paramStr(meta) {
  if (!meta) return "";
  const p = [];
  if (meta.loss) p.push(String(meta.loss));
  if (meta.blocks) p.push(`${meta.blocks}b`);
  const st = fmtSteps(meta.step);
  if (st) p.push(st);
  if (meta.psnr != null && isFinite(meta.psnr)) p.push(`${(+meta.psnr).toFixed(1)}dB`);
  return p.join(" · ");
}

// One row per member with validation peak and integral gate shares per band.
function drawImportance(host, comb, members) {
  const bands = comb.band_names || [];
  const labels = comb.member_labels || [];
  const peaks = comb.member_weight_peaks || {};
  const integrals = comb.member_weight_integrals || {};
  if (!labels.length) {
    host.innerHTML = `<p class="muted">no members</p>`;
    return;
  }
  const pct = (v) => v == null || !isFinite(v) ? "—" : `${(100 * v).toFixed(2)}%`;
  const totals = labels.map((_, member) => bands.reduce((sum, band) =>
    sum + ((peaks[band] || [])[member] || 0) + ((integrals[band] || [])[member] || 0), 0));
  const routed = "member";
  const head = `<tr><th>${routed}</th><th>8-channel weight sum</th>` + bands.map((band) =>
    `<th>${band}<br><span class="muted">peak</span></th>`
    + `<th>${band}<br><span class="muted">integral</span></th>`).join("") + `</tr>`;
  const rows = labels.map((label, member) => `<tr><td class="mono">${label}</td><td>${pct(totals[member])}</td>`
    + bands.map((band) => `<td>${pct((peaks[band] || [])[member])}</td>`
      + `<td>${pct((integrals[band] || [])[member])}</td>`).join("") + `</tr>`).join("");
  host.innerHTML = `<div style="overflow-x:auto"><table class="mini-table">${head}${rows}</table></div>`
    + `<p class="hint">Peak is the maximum gate share on represented validation pixels. Integral is the mean gate share over brightness-stratified validation rows. No ${routed} is removed.</p>`;
}

// Gate weight (convex, sums to 1) of each member vs pixel brightness, one line
// per member, for the selected band — with axis ticks, labels and a legend.
function drawEffWeights(host, comb, band, members, mode) {
  const ew = (comb.eff_weights || {})[band];
  host.innerHTML = "";
  // x = asinh brightness levels; derive from brightness_e if not sent explicitly.
  const bx = (ew && ew.brightness_asinh)
    || ((ew && ew.brightness_e) || []).map((e) => (e == null ? null : Math.asinh(e / 100)));
  if (!ew || !ew.jacobian || !bx || !bx.length) {
    host.innerHTML = `<p class="muted">no curve for ${band}</p>`;
    return;
  }
  const jac = ew.jacobian;                        // [L][M], rows sum to 1
  const nM = (jac[0] || []).length;
  const labels = comb.member_labels || [];
  const W = 800, H = 300, PL = 56, PR = 150, PT = 16, PB = 46;
  const fin = bx.filter((v) => v != null && isFinite(v));
  const xmin = Math.min(...fin), xmax = Math.max(...fin);
  const sx = (v) => PL + (v - xmin) / (xmax - xmin || 1) * (W - PL - PR);
  const sy = (w) => H - PB - w * (H - PT - PB);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="gate weight vs brightness" style="color:${TXT};font:12px sans-serif">`;
  [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
    const y = sy(t);
    s += `<line x1="${PL}" y1="${y}" x2="${W - PR}" y2="${y}" stroke="currentColor" opacity="0.12"/>`;
    s += `<text x="${PL - 8}" y="${y}" text-anchor="end" dominant-baseline="central" fill="currentColor" opacity="0.7">${t}</text>`;
  });
  for (let v = Math.max(0, Math.ceil(xmin)); v <= xmax; v += 2) {
    const x = sx(v);
    s += `<line x1="${x}" y1="${PT}" x2="${x}" y2="${H - PB}" stroke="currentColor" opacity="0.08"/>`;
    s += `<text x="${x}" y="${H - PB + 16}" text-anchor="middle" fill="currentColor" opacity="0.7">${fmtE(100 * Math.sinh(v))}</text>`;
  }
  s += `<line x1="${PL}" y1="${H - PB}" x2="${W - PR}" y2="${H - PB}" stroke="currentColor" opacity="0.4"/>`;
  s += `<line x1="${PL}" y1="${PT}" x2="${PL}" y2="${H - PB}" stroke="currentColor" opacity="0.4"/>`;
  s += `<text x="${(PL + W - PR) / 2}" y="${H - 6}" text-anchor="middle" fill="currentColor" opacity="0.85">pixel brightness [e⁻]</text>`;
  s += `<text transform="translate(14 ${(PT + H - PB) / 2}) rotate(-90)" text-anchor="middle" fill="currentColor" opacity="0.85">gate weight</text>`;
  const cm = memberColorsFor(members || [], mode);
  for (let m = 0; m < nM; m++) {
    const col = (cm[m] && cm[m].color) || COLORS[m % COLORS.length];
    const tag = cm[m] && cm[m].tag;
    let d = "";
    for (let i = 0; i < bx.length; i++) {
      const v = jac[i] && jac[i][m];
      if (v == null || !isFinite(v) || bx[i] == null || !isFinite(bx[i])) continue;
      d += (d ? "L" : "M") + sx(bx[i]).toFixed(1) + "," + sy(v).toFixed(1) + " ";
    }
    if (d) s += `<path d="${d}" fill="none" stroke="${col}" stroke-width="2"/>`;
    const ly = PT + 14 + m * 16;
    s += `<line x1="${W - PR + 8}" y1="${ly - 4}" x2="${W - PR + 24}" y2="${ly - 4}" stroke="${col}" stroke-width="2"/>`;
    s += `<text x="${W - PR + 28}" y="${ly}" dominant-baseline="central" fill="currentColor" opacity="0.9">${labels[m] || m}${tag ? ` · ${tag}` : ""}</text>`;
  }
  host.innerHTML = s + "</svg>";
}

// Empirical diagnostic: the combiner receives only the member stack, while
// validation HR brightness is used afterward to group the resulting weights.
function drawHRWeights(host, comb, band, members, mode) {
  const d = ((comb.hr_weights || {}).bands || {})[band];
  host.innerHTML = "";
  if (!d || !d.mean || !d.brightness_asinh) {
    host.innerHTML = `<p class="muted">${(comb.hr_weights || {}).reason || "no HR-conditioned diagnostic"}</p>`;
    return;
  }
  const bx = d.brightness_asinh;
  const mean = d.mean;
  const counts = d.counts || [];
  const nM = (mean[0] || []).length;
  const labels = comb.member_labels || [];
  const W = 800, H = 300, PL = 56, PR = 150, PT = 16, PB = 46;
  const fin = bx.filter((v, i) => v != null && isFinite(v) && (counts[i] || 0) > 0);
  if (!fin.length) { host.innerHTML = `<p class="muted">no populated HR brightness bins</p>`; return; }
  const xmin = Math.min(...fin), xmax = Math.max(...fin);
  const sx = (v) => PL + (v - xmin) / (xmax - xmin || 1) * (W - PL - PR);
  const sy = (w) => H - PB - w * (H - PT - PB);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="weights conditioned on HR brightness" style="color:${TXT};font:12px sans-serif">`;
  [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
    const y = sy(t);
    s += `<line x1="${PL}" y1="${y}" x2="${W - PR}" y2="${y}" stroke="currentColor" opacity="0.12"/>`;
    s += `<text x="${PL - 8}" y="${y}" text-anchor="end" dominant-baseline="central" fill="currentColor" opacity="0.7">${t}</text>`;
  });
  for (let v = Math.max(0, Math.ceil(xmin)); v <= xmax; v += 2) {
    const x = sx(v);
    s += `<line x1="${x}" y1="${PT}" x2="${x}" y2="${H - PB}" stroke="currentColor" opacity="0.08"/>`;
    s += `<text x="${x}" y="${H - PB + 16}" text-anchor="middle" fill="currentColor" opacity="0.7">${fmtE(100 * Math.sinh(v))}</text>`;
  }
  s += `<line x1="${PL}" y1="${H - PB}" x2="${W - PR}" y2="${H - PB}" stroke="currentColor" opacity="0.4"/>`;
  s += `<line x1="${PL}" y1="${PT}" x2="${PL}" y2="${H - PB}" stroke="currentColor" opacity="0.4"/>`;
  s += `<text x="${(PL + W - PR) / 2}" y="${H - 6}" text-anchor="middle" fill="currentColor" opacity="0.85">HR pixel brightness [e⁻]</text>`;
  s += `<text transform="translate(14 ${(PT + H - PB) / 2}) rotate(-90)" text-anchor="middle" fill="currentColor" opacity="0.85">mean member weight</text>`;
  const cm = memberColorsFor(members || [], mode);
  for (let m = 0; m < nM; m++) {
    const col = (cm[m] && cm[m].color) || COLORS[m % COLORS.length];
    let dpath = "";
    for (let i = 0; i < bx.length; i++) {
      const v = mean[i] && mean[i][m];
      if (v == null || !isFinite(v) || bx[i] == null || !isFinite(bx[i]) || !(counts[i] || 0)) continue;
      dpath += (dpath ? "L" : "M") + sx(bx[i]).toFixed(1) + "," + sy(v).toFixed(1) + " ";
    }
    if (dpath) s += `<path d="${dpath}" fill="none" stroke="${col}" stroke-width="2"/>`;
    const ly = PT + 14 + m * 16;
    s += `<line x1="${W - PR + 8}" y1="${ly - 4}" x2="${W - PR + 24}" y2="${ly - 4}" stroke="${col}" stroke-width="2"/>`;
    s += `<text x="${W - PR + 28}" y="${ly}" dominant-baseline="central" fill="currentColor" opacity="0.9">${labels[m] || m}</text>`;
  }
  host.innerHTML = s + "</svg>";
}

// Interactive 3-D RBF gate-weight surface. The floor is the two actual model
// features (mean×log-std or min×max); the height is one member's gate weight.
function drawFeatureSurface(host, comb, band, member, suppliedGrid) {
  const g = suppliedGrid || ((comb.feature_grid || {})[band] || {});
  const x = g.mean_asinh || [];
  const yRaw = g.std_asinh || [];
  const y = g.y_is_log === false ? yRaw : (g.std_log || yRaw);
  const weights = g.weights || [];
  if (x.length < 2 || y.length < 2 || !weights.length) {
    host.innerHTML = `<p class="muted">no two-feature surface for ${band}</p>`;
    return;
  }
  host.innerHTML = `<canvas aria-label="interactive RBF feature gate-weight surface" tabindex="0" style="display:block;width:100%;height:480px;cursor:grab;border:1px solid #c8cdd7;border-radius:6px"></canvas>`
    + `<p class="hint" style="text-align:center">drag to rotate · scroll to zoom · ${g.x_label || "mean"} × ${g.y_label || "std"}</p>`;
  const canvas = host.querySelector("canvas"), W = 820, H = 480;
  const ctx = canvas.getContext("2d");
  let yaw = -0.72, pitch = 0.58, zoom = 1, drag = null;
  const fmtAxis = (v) => Math.abs(v) >= 10 ? v.toFixed(0) : (Math.abs(v) >= 1 ? v.toFixed(1) : v.toPrecision(2));
  function paint() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const scale = Math.min(W * .34, H * .42) * zoom;
    const ymax = y[y.length - 1], ymin = y[0];
    const all = [];
    for (let i = 0; i < y.length; i++) for (let j = 0; j < x.length; j++) {
      const w = weights[i] && weights[i][j] && weights[i][j][member];
      if (w != null && isFinite(w)) all.push(w);
    }
    const wmax = Math.max(...all, 1e-8);
    const project = (xx, yy, zz) => {
      const rx = xx * cy - yy * sy, ry = xx * sy + yy * cy;
      const py = ry * cp - zz * sp, depth = ry * sp + zz * cp;
      const p = .82 + (depth + 1.8) * .08;
      return { x: W * .5 + rx * scale * p, y: H * .66 - py * scale * p, depth };
    };
    const point = (i, j) => {
      const w = weights[i] && weights[i][j] && weights[i][j][member];
      const zz = isFinite(w) ? Math.max(0, Math.min(1, w)) : 0;
      const yy = ymax === ymin ? 0 : -1 + 2 * (y[i] - ymin) / (ymax - ymin);
      return project(-1 + 2 * j / (x.length - 1), yy, zz);
    };
    const cells = [];
    for (let i = 0; i < y.length - 1; i++) for (let j = 0; j < x.length - 1; j++) {
      const q = [point(i, j), point(i, j + 1), point(i + 1, j + 1), point(i + 1, j)];
      const z = [weights[i][j][member], weights[i][j + 1][member], weights[i + 1][j + 1][member], weights[i + 1][j][member]]
        .filter((v) => isFinite(v)).reduce((a, v, _, a0) => a + v / a0.length, 0);
      cells.push({ q, depth: q.reduce((a, p) => a + p.depth, 0) / 4, z });
    }
    cells.sort((a, b) => a.depth - b.depth).forEach((c) => {
      ctx.beginPath(); c.q.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)); ctx.closePath();
      ctx.fillStyle = viridis(Math.max(0, Math.min(1, c.z))); ctx.globalAlpha = .82; ctx.fill();
      ctx.strokeStyle = "#77808c"; ctx.globalAlpha = .25; ctx.stroke();
    });
    ctx.globalAlpha = 1; ctx.strokeStyle = "#56606c"; ctx.fillStyle = "#303845"; ctx.lineWidth = 1.2; ctx.font = "12px ui-monospace,monospace";
    const axis = (a, b, label) => { const p = project(...a), q = project(...b); ctx.beginPath(); ctx.moveTo(p.x,p.y); ctx.lineTo(q.x,q.y); ctx.stroke(); ctx.fillText(label,q.x+6,q.y-4); };
    axis([-1,-1,0],[1,-1,0],g.x_label || "mean");
    axis([-1,-1,0],[-1,1,0],(g.y_is_log === false ? "" : "log ") + (g.y_label || "std"));
    axis([-1,-1,0],[-1,-1,1],"relative weight [0–1]");
    ctx.fillStyle = "#56606c"; ctx.fillText(`${fmtAxis(x[0])} … ${fmtAxis(x[x.length-1])}`, 12, 18);
    ctx.fillText(`${fmtAxis(yRaw[0])} … ${fmtAxis(yRaw[yRaw.length-1])}`, 12, 34);
  }
  canvas.addEventListener("pointerdown", (e) => { canvas.setPointerCapture(e.pointerId); drag = { x:e.clientX, y:e.clientY, yaw, pitch }; });
  canvas.addEventListener("pointermove", (e) => { if (!drag) return; yaw = drag.yaw + (e.clientX-drag.x)*.012; pitch = Math.max(-1.35,Math.min(1.35,drag.pitch+(e.clientY-drag.y)*.01)); paint(); });
  canvas.addEventListener("pointerup", () => { drag = null; });
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); zoom = Math.max(.55,Math.min(2.5,zoom*(e.deltaY>0?.9:1.1))); paint(); }, { passive:false });
  paint();
}

function render(root, comb, evals) {
  const bands = comb.band_names || Object.keys(comb.eff_weights || {});
  const featureRBF = true;
  const sharedPCA = comb.pca_weight_surface && comb.pca_weight_surface.available;
  const featureLabel = "all-inference PCA";
  const routed = "band";
  root.innerHTML = `
    <div style="max-width:860px;margin:0 auto">
      <div style="margin-bottom:1rem">${metricsHtml(comb, evals)}</div>
      <h4 style="margin:.2rem 0;text-align:center">${routed} weight diagnostics</h4>
      <div id="ens-comb-imp"></div>
      ${sharedPCA ? `<h4 style="margin:1rem 0 .2rem;text-align:center">shared gate weight on validation PCA · PC1 × PC2
        <button type="button" id="ens-comb-surface-prev" title="previous member" style="margin-left:.4rem">←</button>
        <select id="ens-comb-surface-member" style="margin-left:.4rem">
          ${(comb.member_labels || []).map((m, i) => `<option value="${i}">${m}</option>`).join("")}
        </select>
        <button type="button" id="ens-comb-surface-next" title="next member">→</button></h4>
        <div id="ens-comb-pca-surface"></div>
        <p class="hint" style="text-align:center">PC1 and PC2 vary jointly through the validation mean; every remaining principal component stays at zero.</p>`
        : featureRBF ? `<h4 style="margin:1rem 0 .2rem;text-align:center">interactive ${routed} gate weight across ${featureLabel}
        <select id="ens-comb-band" style="margin-left:.4rem">
          ${bands.map((b) => `<option value="${b}">${b}</option>`).join("")}
        </select>
        <button type="button" id="ens-comb-surface-prev" title="previous member" style="margin-left:.4rem">←</button>
        <select id="ens-comb-surface-member" style="margin-left:.4rem">
          ${(comb.member_labels || []).map((m, i) => `<option value="${i}">${m}</option>`).join("")}
        </select>
        <button type="button" id="ens-comb-surface-next" title="next member">→</button></h4><div id="ens-comb-surface"></div>
        ` : `<h4 style="margin:1rem 0 .2rem;text-align:center">gate weight vs brightness
        <select id="ens-comb-band" style="margin-left:.4rem">
          ${bands.map((b) => `<option value="${b}">${b}</option>`).join("")}
        </select>
        <select id="ens-comb-color" style="margin-left:.4rem">
          <option value="">uniform</option>
          <option value="loss">by loss</option>
          <option value="depth">by depth</option>
          <option value="psnr">by psnr</option>
        </select></h4>
      <div id="ens-comb-eff"></div>
      <p class="hint" style="text-align:center">Convex gate weight of each member (sums to 1) as pixels get brighter — faint → the L1 members, star cores → the star-reproducing members.</p>`}
      <h4 style="margin:1rem 0 .2rem;text-align:center">weights conditioned on underlying HR brightness</h4>
      <div id="ens-comb-hr"></div>
      <p class="hint" style="text-align:center">Offline diagnostic: weights are computed from the member stack only,
        then grouped by the corresponding HR brightness. HR is never supplied to the combiner.</p>
    </div>`;
  const members = memberMeta(comb, evals);
  drawImportance(root.querySelector("#ens-comb-imp"), comb, members);
  const sel = root.querySelector("#ens-comb-band");
  const colorSel = root.querySelector("#ens-comb-color");
  const host = root.querySelector("#ens-comb-eff");
  const hrHost = root.querySelector("#ens-comb-hr");
  const surfaceHost = root.querySelector("#ens-comb-surface");
  const pcaSurfaceHost = root.querySelector("#ens-comb-pca-surface");
  const surfaceSel = root.querySelector("#ens-comb-surface-member");
  const surfacePrev = root.querySelector("#ens-comb-surface-prev");
  const surfaceNext = root.querySelector("#ens-comb-surface-next");
  const draw = () => {
    if (host && sel) drawEffWeights(host, comb, sel.value, members, colorSel ? colorSel.value : "");
  };
  const drawBoth = () => {
    draw();
    const hrBand = sel ? sel.value.split(" ")[0] : bands[0];
    drawHRWeights(hrHost, comb, hrBand, members, colorSel ? colorSel.value : "");
    if (surfaceHost && sel) drawFeatureSurface(surfaceHost, comb, sel.value, +(surfaceSel && surfaceSel.value || 0));
    if (pcaSurfaceHost) drawFeatureSurface(
      pcaSurfaceHost, comb, "PC1 × PC2", +(surfaceSel && surfaceSel.value || 0),
      comb.pca_weight_surface);
  };
  if (sel) sel.addEventListener("change", drawBoth);
  if (colorSel) colorSel.addEventListener("change", drawBoth);
  if (surfaceSel) surfaceSel.addEventListener("change", drawBoth);
  const moveSurface = (step) => {
    if (!surfaceSel) return;
    const next = Math.max(0, Math.min(surfaceSel.options.length - 1, surfaceSel.selectedIndex + step));
    if (next === surfaceSel.selectedIndex) return;
    surfaceSel.selectedIndex = next;
    drawBoth();
  };
  if (surfacePrev) surfacePrev.addEventListener("click", () => moveSurface(-1));
  if (surfaceNext) surfaceNext.addEventListener("click", () => moveSurface(1));
  if (surfaceHost) surfaceHost.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    moveSurface(event.key === "ArrowLeft" ? -1 : 1);
  });
  drawBoth();
}

export async function mountEnsembleCombiner(card, combinerUrl, evalsUrl) {
  if (!card) return;
  const results = card.querySelector("[data-combiner-results]")
    || card.querySelector("#ens-combiner-results");
  if (!results) return;
  // Combiner + evals are detached per star regime — fetch the current mode's,
  // and reload when the top toggle changes.
  const modeSel = document.getElementById("ens-mode");
  const modelSel = card.querySelector('select[name="model_kind"]');
  const curMode = () => (modeSel ? modeSel.value : "starfull");
  const curModel = () => (modelSel ? modelSel.value : "raw_incremental_minmeanmax_rbf");
  const withMode = (u) =>
    u + (u.includes("?") ? "&" : "?") + "mode=" + encodeURIComponent(curMode())
      + "&model_kind=" + encodeURIComponent(curModel());

  async function load() {
    let comb = null, evals = null;
    try { const r = await fetch(withMode(combinerUrl)); if (r.ok) comb = await r.json(); } catch (_) {}
    try { const r = await fetch(withMode(evalsUrl)); if (r.ok) evals = await r.json(); } catch (_) {}
    if (!comb || !comb.available) {
      results.innerHTML = `<p class="muted">No combiner fitted yet for the`
        + ` <b>${curMode()}</b> regime — sync validate records (/sky), then`
        + ` press <b>Fit combiner</b>.</p>`;
      return;
    }
    render(results, comb, evals);
  }

  document.addEventListener("ensemble-mode-change", () => { load(); });
  if (modelSel) modelSel.addEventListener("change", () => {
    const kernels = card.querySelector('input[name="n_kernels"]');
    if (kernels) kernels.value = "128";
    load();
  });
  await load();
}
