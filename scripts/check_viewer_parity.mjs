/* Parity check: in-browser colour maths (static/cutout_viewer.js) vs the
 * Python renderer (visualization/color.py).
 *
 *   python scripts/_viewer_parity_ref.py | node scripts/check_viewer_parity.mjs
 *
 * Exits non-zero (and prints the worst offender) if any value drifts past the
 * tolerance. Covers: per-band calibration constants, the Planckian-locus hue
 * chain, and full eye_rgb on a small cube (which the viewer's Temp mode at
 * default knee/brightness reproduces exactly).
 */
import { readFileSync } from "node:fs";
import { _internals as M } from "../euclid_polish/web/static/cutout_viewer.js";

const ref = JSON.parse(readFileSync(0, "utf8")); // stdin
const TOL = 1e-4;
let worst = 0, worstWhat = "";
let fail = 0;

function check(what, got, exp, tol = TOL) {
  const d = Math.abs(got - exp);
  if (d > worst) { worst = d; worstWhat = what; }
  if (d > tol) { fail++; if (fail <= 12) console.error(`✗ ${what}: got ${got}, exp ${exp} (Δ${d.toExponential(2)})`); }
}

// Band-config approximations of what the JS reads from /viewer/meta.
import { execSync } from "node:child_process";
const PYBIN = process.env.PYBIN || "python3";
const consts = JSON.parse(execSync(
  `${PYBIN} -c "import json;from euclid_polish.config import Config as C;` +
  `print(json.dumps({n:{'t_total_s':b.t_total_s,'zeropoint_ab':b.zeropoint_ab_e_per_s,` +
  `'zeropoint_ab_e_total':b.sim_zeropoint_e,` +
  `'solar_ab_mag':C.Color.SOLAR_AB_MAG[n],'pivot_um':C.Color.PIVOT_WAVELENGTH_UM[n]} ` +
  `for n,b in ((x.name,x) for x in C.BANDS)}))"`,
  { encoding: "utf8" }));

// 1) calibration constants
for (const [name, v] of Object.entries(ref.bands)) {
  const b = consts[name];
  check(`abFluxNorm[${name}]`, M.abFluxNorm(b) / v.ab_flux_norm, 1.0, 1e-6);
  check(`solarBalance[${name}]`, M.solarBalance(b) / v.solar_balance, 1.0, 1e-6);
}

// 2) Planckian hue chain
for (const row of ref.planck_chain) {
  const [x, y] = M.planckianXY(row.T);
  const lin = M.xyToLinearSrgb(x, y);
  const srgb = lin.map(M.srgbGamma);
  for (let k = 0; k < 3; k++) check(`planck[${row.T}][${k}]`, srgb[k], row.srgb[k]);
}

// 3) full eye_rgb on the sample cube (Temp mode at default knee/brightness).
{
  const names = ["VIS", "Y_E", "J_E", "H_E"];
  const lam = names.map((n) => consts[n].pivot_um);
  const abnorm = names.map((n) => M.abFluxNorm(consts[n]));
  const ts = M.eyeTGrid(96);
  // NB: Array.from — a Float64Array.map would coerce the returned vectors to NaN.
  const pvecs = Array.from(ts, (T) => {
    const p = M.planckFnu(lam, T);
    let nrm = 0; for (const v of p) nrm += v * v; nrm = Math.sqrt(nrm);
    return Array.from(p, (v) => v / nrm);
  });
  const asinhScale = ref.eye_rgb.asinh_scale_e;
  const kneeCal = asinhScale * abnorm[0];
  const norm = Math.asinh(30.0); // white_e = 30·asinh_scale, VIS-equiv

  ref.eye_rgb.cube.forEach((px, i) => {
    const cal = px.map((v, k) => v * abnorm[k]);
    let bestS = -Infinity, bestT = 6500;
    for (let ti = 0; ti < ts.length; ti++) {
      let s = 0; for (let k = 0; k < 4; k++) s += cal[k] * pvecs[ti][k];
      if (s > bestS) { bestS = s; bestT = ts[ti]; }
    }
    const T = bestS > 0 ? bestT : 6500;
    const [x, y] = M.planckianXY(T);
    const hue = M.xyToLinearSrgb(x, y);
    const I = Math.max((cal[0] + cal[1] + cal[2] + cal[3]) / 4, 0);
    const lum = Math.min(Math.max(Math.asinh(I / kneeCal) / norm, 0), 1);
    const rgb = hue.map((h) => M.srgbGamma(h * lum));
    for (let k = 0; k < 3; k++) check(`eye[${i}][${k}]`, rgb[k], ref.eye_rgb.rgb[i][k], 2e-3);
  });
}

console.log(`worst Δ = ${worst.toExponential(3)} @ ${worstWhat}`);
if (fail) { console.error(`PARITY FAIL: ${fail} value(s) over tolerance`); process.exit(1); }
console.log("✓ parity OK");
