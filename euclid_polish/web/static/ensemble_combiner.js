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
const LOSS_COLORS = { l1: "#3b6fb0", l2: "#5aae61", l3: "#e08214", berhu: "#9970ab" };
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
  const c = evals && evals.combiner;
  let rows = "<tr><th>series</th><th>VIS PSNR (dB, asinh)</th></tr>";
  if (c && c.available) {
    const g = (v) => (v == null ? "" : ` <span class="muted">(${v >= 0 ? "+" : ""}${fmt(v, 2)} dB)</span>`);
    const vsMean = c.psnr != null && c.ensemble_mean_psnr != null ? c.psnr - c.ensemble_mean_psnr : null;
    const vsBest = c.psnr != null && c.best_member_psnr != null ? c.psnr - c.best_member_psnr : null;
    rows += `<tr><td><b>combiner</b></td><td><b>${fmt(c.psnr, 3)}</b></td></tr>`;
    rows += `<tr><td>ensemble mean</td><td>${fmt(c.ensemble_mean_psnr, 3)}${g(vsMean)}</td></tr>`;
    rows += `<tr><td>best member${c.best_member_label ? ` (${c.best_member_label})` : ""}</td>`
          + `<td>${fmt(c.best_member_psnr, 3)}${g(vsBest)}</td></tr>`;
  } else {
    rows += `<tr><td colspan="2" class="muted">Run “Evaluate on test set” (starfull) to score the combiner on test.</td></tr>`;
  }
  const stale = comb.stale
    ? ` · <b style="color:#c33">STALE</b> (membership changed since fit — re-fit)` : "";
  return `<div style="display:flex;justify-content:center"><table class="mini-table">${rows}</table></div>`
       + `<p class="hint" style="text-align:center">Fit L1 on validate: <b>${fmt(comb.val_l1, 4)}</b>`
       + ` · RBF gate, K=${comb.n_kernels} kernels · prune &lt;${comb.min_usage} cumulative importance${stale}</p>`;
}

// Per-member, per-band importance = mean gate weight over the brightness sweep.
function importanceData(comb) {
  const bands = comb.band_names || Object.keys(comb.eff_weights || {});
  const labels = comb.member_labels || [];
  const eff = comb.eff_weights || {};
  const imp = labels.map(() => bands.map(() => 0));
  bands.forEach((b, bi) => {
    const J = (eff[b] || {}).jacobian || [];              // [L][M]
    for (let m = 0; m < labels.length; m++) {
      let s = 0, c = 0;
      for (let l = 0; l < J.length; l++) {
        const v = J[l][m];
        if (v != null && isFinite(v)) { s += v; c += 1; }
      }
      imp[m][bi] = c ? s / c : 0;
    }
  });
  return { bands, labels, imp, totals: imp.map((r) => r.reduce((a, v) => a + v, 0)) };
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

// One horizontal bar per member, stacked into the 4 band segments; sorted by
// total importance. Pruned members are omitted; each row is annotated with the
// member's essential params (loss · depth · steps · PSNR).
function drawImportance(host, comb, members) {
  const { bands, labels, imp, totals } = importanceData(comb);
  const surv = comb.surviving || {};
  const kept = labels.map((_, m) => bands.some((b) => (surv[b] || [])[m] !== false));
  const order = labels.map((_, i) => i).filter((i) => kept[i])
    .sort((a, b) => totals[b] - totals[a]);
  const M = order.length;
  if (!M) { host.innerHTML = `<p class="muted">no surviving members</p>`; return; }
  const xmax = Math.max(1e-6, ...order.map((i) => totals[i]));
  const W = 800, PL = 72, PR = 214, PT = 10, rowH = 26;
  const H = PT + M * rowH + 46;
  const barW = W - PL - PR;
  const sx = (v) => v / xmax * barW;
  const xR = W - PR + 8;
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="member importance" style="color:${TXT};font:12px sans-serif">`;
  [0, 0.5, 1].forEach((f) => {
    const tv = f * xmax, x = PL + sx(tv);
    s += `<line x1="${x}" y1="${PT}" x2="${x}" y2="${PT + M * rowH}" stroke="currentColor" opacity="0.1"/>`;
    s += `<text x="${x}" y="${PT + M * rowH + 16}" text-anchor="middle" fill="currentColor" opacity="0.7">${tv.toFixed(2)}</text>`;
  });
  order.forEach((m, row) => {
    const y = PT + row * rowH + 4, h = rowH - 10;
    s += `<text x="${PL - 8}" y="${y + h / 2}" text-anchor="end" dominant-baseline="central" fill="currentColor" opacity="0.9">${labels[m]}</text>`;
    let x0 = PL;
    bands.forEach((b, bi) => {
      const w = sx(imp[m][bi]);
      if (w > 0.3) s += `<rect x="${x0.toFixed(1)}" y="${y}" width="${w.toFixed(1)}" height="${h}" fill="${BAND_COLORS[b] || "#888"}"/>`;
      x0 += w;
    });
    const ps = paramStr((members || [])[m]);
    s += `<text x="${xR}" y="${y + h / 2}" dominant-baseline="central" fill="currentColor" opacity="0.7">`
       + `<tspan opacity="0.6">${totals[m].toFixed(2)}</tspan>${ps ? `  ${ps}` : ""}</text>`;
  });
  let lx = PL;
  const ly = PT + M * rowH + 36;
  bands.forEach((b) => {
    s += `<rect x="${lx}" y="${ly - 10}" width="12" height="12" fill="${BAND_COLORS[b] || "#888"}"/>`;
    s += `<text x="${lx + 16}" y="${ly}" dominant-baseline="central" fill="currentColor" opacity="0.8">${b}</text>`;
    lx += 34 + b.length * 7;
  });
  host.innerHTML = s + "</svg>";
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
  const surviving = (comb.surviving || {})[band] || [];
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
    const dead = surviving[m] === false;
    let d = "";
    for (let i = 0; i < bx.length; i++) {
      const v = jac[i] && jac[i][m];
      if (v == null || !isFinite(v) || bx[i] == null || !isFinite(bx[i])) continue;
      d += (d ? "L" : "M") + sx(bx[i]).toFixed(1) + "," + sy(v).toFixed(1) + " ";
    }
    if (d) s += `<path d="${d}" fill="none" stroke="${col}" stroke-width="${dead ? 1 : 2}" opacity="${dead ? 0.3 : 1}"/>`;
    const ly = PT + 14 + m * 16;
    s += `<line x1="${W - PR + 8}" y1="${ly - 4}" x2="${W - PR + 24}" y2="${ly - 4}" stroke="${col}" stroke-width="2" opacity="${dead ? 0.3 : 1}"/>`;
    s += `<text x="${W - PR + 28}" y="${ly}" dominant-baseline="central" fill="currentColor" opacity="${dead ? 0.4 : 0.9}">${labels[m] || m}${tag ? ` · ${tag}` : ""}${dead ? " (pruned)" : ""}</text>`;
  }
  host.innerHTML = s + "</svg>";
}

function render(root, comb, evals) {
  const bands = comb.band_names || Object.keys(comb.eff_weights || {});
  root.innerHTML = `
    <div style="max-width:860px;margin:0 auto">
      <div style="margin-bottom:1rem">${metricsHtml(comb, evals)}</div>
      <h4 style="margin:.2rem 0;text-align:center">member importance (mean gate weight, stacked by band)</h4>
      <div id="ens-comb-imp"></div>
      <h4 style="margin:1rem 0 .2rem;text-align:center">gate weight vs brightness
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
      <p class="hint" style="text-align:center">Convex gate weight of each member (sums to 1)
        as pixels get brighter — faint → the L1 members, star cores → the
        star-reproducing members.</p>
    </div>`;
  const members = memberMeta(comb, evals);
  drawImportance(root.querySelector("#ens-comb-imp"), comb, members);
  const sel = root.querySelector("#ens-comb-band");
  const colorSel = root.querySelector("#ens-comb-color");
  const host = root.querySelector("#ens-comb-eff");
  const draw = () => drawEffWeights(host, comb, sel.value, members, colorSel.value);
  sel.addEventListener("change", draw);
  colorSel.addEventListener("change", draw);
  draw();
}

export async function mountEnsembleCombiner(card, combinerUrl, evalsUrl) {
  if (!card) return;
  const results = card.querySelector("#ens-combiner-results");
  if (!results) return;
  // Combiner + evals are detached per star regime — fetch the current mode's,
  // and reload when the top toggle changes.
  const modeSel = document.getElementById("ens-mode");
  const curMode = () => (modeSel ? modeSel.value : "starfull");
  const withMode = (u) =>
    u + (u.includes("?") ? "&" : "?") + "mode=" + encodeURIComponent(curMode());

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
  await load();
}
