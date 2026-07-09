// Combiner card renderer. Pulls fit-time info from /ensemble/combiner.json
// (survivors, val loss, effective-weight-vs-brightness Jacobian curves) and
// test-time metrics from /ensemble/evals.json (combiner vs mean vs best member,
// VIS stretched PSNR). All drawn client-side; no server round-trips after load.

const COLORS = [
  "#4477aa", "#ee6677", "#228833", "#ccbb44", "#66ccee", "#aa3377",
  "#ee8866", "#44bb99", "#bbbb44", "#99ddff", "#cc6677", "#000000",
];

function fmt(x, d = 3) {
  return (x == null || !isFinite(x)) ? "—" : (Math.round(x * 10 ** d) / 10 ** d);
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
  return `<table class="mini-table">${rows}</table>`
       + `<p class="hint">Fit L1 on validate: <b>${fmt(comb.val_l1, 4)}</b>`
       + ` · RBF gate, K=${comb.n_kernels} kernels · prune ${comb.min_usage}${stale}</p>`;
}

function survivorsHtml(comb) {
  const labels = comb.member_labels || [];
  const bands = comb.band_names || Object.keys(comb.surviving || {});
  let h = `<table class="mini-table"><tr><th>member</th>`
        + bands.map((b) => `<th>${b}</th>`).join("") + `</tr>`;
  labels.forEach((lab, i) => {
    h += `<tr><td>${lab}</td>` + bands.map((b) => {
      const alive = ((comb.surviving || {})[b] || [])[i];
      return `<td style="text-align:center;color:${alive ? "#228833" : "#bbb"}">`
           + `${alive ? "✔" : "·"}</td>`;
    }).join("") + `</tr>`;
  });
  return h + `</table><p class="hint">✔ = member kept for that band; · = pruned by group-L1.</p>`;
}

// Compact SVG line plot: effective weight (Jacobian) vs pixel brightness, one
// line per surviving member, for the selected band.
function drawEffWeights(host, comb, band) {
  const ew = (comb.eff_weights || {})[band];
  host.innerHTML = "";
  if (!ew || !ew.brightness_e || !ew.jacobian) {
    host.innerHTML = `<p class="muted">no curve for ${band}</p>`;
    return;
  }
  const xs = ew.brightness_e.map((v) => (v == null ? NaN : Math.log10(Math.max(v, 1e-3))));
  const jac = ew.jacobian;                      // [nLevels][nMembers]
  const nM = (jac[0] || []).length;
  const surviving = (comb.surviving || {})[band] || [];
  const W = 480, H = 240, PL = 44, PR = 90, PT = 12, PB = 30;
  const finite = [];
  jac.forEach((row) => row.forEach((v) => { if (v != null && isFinite(v)) finite.push(v); }));
  const xmin = Math.min(...xs.filter(isFinite)), xmax = Math.max(...xs.filter(isFinite));
  let ymin = Math.min(0, ...finite), ymax = Math.max(1, ...finite);
  if (!isFinite(ymin)) ymin = 0; if (!isFinite(ymax)) ymax = 1;
  const pad = (ymax - ymin) * 0.08 || 0.1;
  ymin -= pad; ymax += pad;
  const sx = (x) => PL + (x - xmin) / (xmax - xmin || 1) * (W - PL - PR);
  const sy = (y) => H - PB - (y - ymin) / (ymax - ymin || 1) * (H - PT - PB);
  const labels = comb.member_labels || [];
  let svg = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:${W}px;font:11px sans-serif">`;
  // axes
  svg += `<line x1="${PL}" y1="${sy(0)}" x2="${W - PR}" y2="${sy(0)}" stroke="#ccc"/>`;
  svg += `<line x1="${PL}" y1="${PT}" x2="${PL}" y2="${H - PB}" stroke="#888"/>`;
  svg += `<text x="${PL}" y="${H - 8}" fill="#666">faint</text>`;
  svg += `<text x="${W - PR}" y="${H - 8}" text-anchor="end" fill="#666">bright →</text>`;
  svg += `<text x="8" y="${PT + 8}" fill="#666">weight</text>`;
  for (let m = 0; m < nM; m++) {
    const col = COLORS[m % COLORS.length];
    const dead = surviving[m] === false;
    let d = "";
    for (let i = 0; i < xs.length; i++) {
      const v = jac[i] && jac[i][m];
      if (!isFinite(xs[i]) || v == null || !isFinite(v)) continue;
      d += (d ? "L" : "M") + sx(xs[i]).toFixed(1) + "," + sy(v).toFixed(1) + " ";
    }
    if (d) {
      svg += `<path d="${d}" fill="none" stroke="${col}" stroke-width="${dead ? 1 : 2}" `
           + `opacity="${dead ? 0.25 : 1}"/>`;
    }
    const ly = PT + 12 + m * 14;
    svg += `<line x1="${W - PR + 6}" y1="${ly - 4}" x2="${W - PR + 22}" y2="${ly - 4}" `
         + `stroke="${col}" stroke-width="2" opacity="${dead ? 0.25 : 1}"/>`;
    svg += `<text x="${W - PR + 26}" y="${ly}" fill="#333" opacity="${dead ? 0.4 : 1}">`
         + `${labels[m] || m}${dead ? " (pruned)" : ""}</text>`;
  }
  svg += `</svg>`;
  host.innerHTML = svg;
}

function render(root, comb, evals) {
  root.innerHTML = "";
  const bands = comb.band_names || Object.keys(comb.eff_weights || {});
  root.innerHTML = `
    <div class="ens-comb-grid" style="display:flex;flex-wrap:wrap;gap:1.5rem;align-items:flex-start">
      <div><h4 style="margin:.2rem 0">test metrics</h4>${metricsHtml(comb, evals)}</div>
      <div><h4 style="margin:.2rem 0">surviving members</h4>${survivorsHtml(comb)}</div>
    </div>
    <div style="margin-top:.75rem">
      <h4 style="margin:.2rem 0;display:inline-block">effective member weight vs brightness</h4>
      <select id="ens-comb-band" style="margin-left:.5rem">
        ${bands.map((b) => `<option value="${b}">${b}</option>`).join("")}
      </select>
      <div id="ens-comb-eff"></div>
      <p class="hint">The gate weight of each member vs pixel brightness (convex,
        sums to 1): how much the combiner leans on each member as pixels get
        brighter. Faint → L1 members; bright → star-reproducing members.</p>
    </div>`;
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
