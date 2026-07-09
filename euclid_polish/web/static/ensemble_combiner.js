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
       + ` · RBF gate, K=${comb.n_kernels} kernels · prune ${comb.min_usage}${stale}</p>`;
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

// One horizontal bar per member, stacked into the 4 band segments; sorted by
// total importance. Shows which members the gate leans on, and in which bands.
function drawImportance(host, comb) {
  const { bands, labels, imp, totals } = importanceData(comb);
  const M = labels.length;
  const order = labels.map((_, i) => i).sort((a, b) => totals[b] - totals[a]);
  const xmax = Math.max(1e-6, ...totals);
  const W = 800, PL = 66, PR = 64, PT = 10, rowH = 26;
  const H = PT + M * rowH + 46;
  const barW = W - PL - PR;
  const sx = (v) => v / xmax * barW;
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
    s += `<text x="${x0 + 6}" y="${y + h / 2}" dominant-baseline="central" fill="currentColor" opacity="0.6">${totals[m].toFixed(2)}</text>`;
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
function drawEffWeights(host, comb, band) {
  const ew = (comb.eff_weights || {})[band];
  host.innerHTML = "";
  if (!ew || !ew.brightness_asinh || !ew.jacobian) {
    host.innerHTML = `<p class="muted">no curve for ${band}</p>`;
    return;
  }
  const bx = ew.brightness_asinh;                 // asinh brightness levels
  const jac = ew.jacobian;                        // [L][M], rows sum to 1
  const nM = (jac[0] || []).length;
  const surviving = (comb.surviving || {})[band] || [];
  const labels = comb.member_labels || [];
  const W = 800, H = 300, PL = 56, PR = 150, PT = 16, PB = 46;
  const xmin = Math.min(...bx), xmax = Math.max(...bx);
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
  for (let m = 0; m < nM; m++) {
    const col = COLORS[m % COLORS.length];
    const dead = surviving[m] === false;
    let d = "";
    for (let i = 0; i < bx.length; i++) {
      const v = jac[i] && jac[i][m];
      if (v == null || !isFinite(v)) continue;
      d += (d ? "L" : "M") + sx(bx[i]).toFixed(1) + "," + sy(v).toFixed(1) + " ";
    }
    if (d) s += `<path d="${d}" fill="none" stroke="${col}" stroke-width="${dead ? 1 : 2}" opacity="${dead ? 0.3 : 1}"/>`;
    const ly = PT + 14 + m * 16;
    s += `<line x1="${W - PR + 8}" y1="${ly - 4}" x2="${W - PR + 24}" y2="${ly - 4}" stroke="${col}" stroke-width="2" opacity="${dead ? 0.3 : 1}"/>`;
    s += `<text x="${W - PR + 28}" y="${ly}" dominant-baseline="central" fill="currentColor" opacity="${dead ? 0.4 : 0.9}">${labels[m] || m}${dead ? " (pruned)" : ""}</text>`;
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
        </select></h4>
      <div id="ens-comb-eff"></div>
      <p class="hint" style="text-align:center">Convex gate weight of each member (sums to 1)
        as pixels get brighter — faint → the L1 members, star cores → the
        star-reproducing members.</p>
    </div>`;
  drawImportance(root.querySelector("#ens-comb-imp"), comb);
  const sel = root.querySelector("#ens-comb-band");
  const host = root.querySelector("#ens-comb-eff");
  const draw = () => drawEffWeights(host, comb, sel.value);
  sel.addEventListener("change", draw);
  draw();
}

export async function mountEnsembleCombiner(card, combinerUrl, evalsUrl) {
  if (!card) return;
  const results = card.querySelector("#ens-combiner-results");
  if (!results) return;
  let comb = null, evals = null;
  try { const r = await fetch(combinerUrl); if (r.ok) comb = await r.json(); } catch (_) {}
  try { const r = await fetch(evalsUrl); if (r.ok) evals = await r.json(); } catch (_) {}
  if (!comb || !comb.available) {
    results.innerHTML = `<p class="muted">No combiner fitted yet — make sure the`
      + ` page mode is <b>starfull</b>, validate records are synced (/sky), then`
      + ` press <b>Fit combiner</b>.</p>`;
    return;
  }
  render(results, comb, evals);
}
