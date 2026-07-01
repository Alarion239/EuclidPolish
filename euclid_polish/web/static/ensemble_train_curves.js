// In-browser ensemble training curves: two SVG line charts (PSNR + combined
// loss vs step), one colored line per member. Fetches the rollback-deduped
// series from /ensemble/training-curves.json — no server-side matplotlib.

const SVGNS = "http://www.w3.org/2000/svg";
const AXIS = "var(--muted, #8a94a6)";
const GRID = "var(--border, #2a2f3a)";
const TEXT = "var(--fg, #c8d0dc)";

function svg(tag, attrs, children) {
  const n = document.createElementNS(SVGNS, tag);
  for (const k in (attrs || {})) n.setAttribute(k, attrs[k]);
  for (const c of children || []) n.appendChild(c);
  return n;
}

// Distinct, evenly-spread hue per member (golden-angle so neighbours differ).
function memberColor(i, total) {
  const hue = (i * 137.508) % 360;
  return `hsl(${hue.toFixed(1)}, 62%, 58%)`;
}

function niceTicks(min, max, n) {
  if (!(max > min)) return [min];
  const span = max - min;
  const step0 = span / Math.max(1, n);
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const norm = step0 / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const ticks = [];
  for (let t = Math.ceil(min / step) * step; t <= max + step * 1e-6; t += step)
    ticks.push(t);
  return ticks;
}

function decadeTicks(min, max) {
  const ticks = [];
  const lo = Math.floor(Math.log10(min));
  const hi = Math.ceil(Math.log10(max));
  for (let e = lo; e <= hi; e++) {
    const base = Math.pow(10, e);
    for (const m of [1, 2, 5]) {
      const v = m * base;
      if (v >= min * 0.999 && v <= max * 1.001) ticks.push(v);
    }
  }
  return ticks.length ? ticks : [min, max];
}

function fmtStep(v) {
  if (Math.abs(v) >= 1000) return (v / 1000).toFixed(v % 1000 ? 1 : 0) + "k";
  return String(Math.round(v));
}

function fmtVal(v, logY) {
  if (logY) {
    if (v >= 1) return v.toFixed(v < 10 ? 1 : 0);
    return v.toPrecision(1);
  }
  return v.toFixed(Math.abs(v) < 10 ? 1 : 0);
}

// Draw one panel. `key` selects series[i][key] = [[step, value], ...].
function drawPanel(series, key, opts) {
  const W = 520, H = 300;
  const m = { t: 30, r: 14, b: 40, l: 54 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const logY = !!opts.logY;

  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (const s of series) {
    for (const [x, y] of s[key] || []) {
      if (x < xmin) xmin = x;
      if (x > xmax) xmax = x;
      const yy = logY ? (y > 0 ? y : NaN) : y;
      if (Number.isFinite(yy)) {
        if (yy < ymin) ymin = yy;
        if (yy > ymax) ymax = yy;
      }
    }
  }
  if (!Number.isFinite(xmin)) return svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "etc-panel" });
  if (xmax === xmin) xmax = xmin + 1;
  if (logY) { ymin *= 0.9; ymax *= 1.1; }
  else {
    const pad = (ymax - ymin || 1) * 0.06;
    ymin -= pad; ymax += pad;
  }

  const X = (x) => m.l + ((x - xmin) / (xmax - xmin)) * iw;
  const Y = logY
    ? (y) => m.t + ih - ((Math.log10(y) - Math.log10(ymin)) /
        (Math.log10(ymax) - Math.log10(ymin))) * ih
    : (y) => m.t + ih - ((y - ymin) / (ymax - ymin)) * ih;

  const kids = [];
  // Title
  kids.push(svg("text", {
    x: m.l, y: 18, fill: TEXT, "font-size": 13, "font-weight": 600,
  }, [document.createTextNode(opts.title)]));

  // Y grid + labels
  const yticks = logY ? decadeTicks(ymin, ymax) : niceTicks(ymin, ymax, 5);
  for (const yt of yticks) {
    const y = Y(yt);
    if (y < m.t - 1 || y > m.t + ih + 1) continue;
    kids.push(svg("line", { x1: m.l, y1: y, x2: m.l + iw, y2: y, stroke: GRID, "stroke-width": 1, opacity: 0.5 }));
    kids.push(svg("text", { x: m.l - 8, y: y + 4, fill: AXIS, "font-size": 11, "text-anchor": "end" },
      [document.createTextNode(fmtVal(yt, logY))]));
  }
  // X grid + labels
  for (const xt of niceTicks(xmin, xmax, 5)) {
    const x = X(xt);
    if (x < m.l - 1 || x > m.l + iw + 1) continue;
    kids.push(svg("line", { x1: x, y1: m.t, x2: x, y2: m.t + ih, stroke: GRID, "stroke-width": 1, opacity: 0.35 }));
    kids.push(svg("text", { x, y: m.t + ih + 20, fill: AXIS, "font-size": 11, "text-anchor": "middle" },
      [document.createTextNode(fmtStep(xt))]));
  }
  // Axes
  kids.push(svg("line", { x1: m.l, y1: m.t + ih, x2: m.l + iw, y2: m.t + ih, stroke: AXIS, "stroke-width": 1.2 }));
  kids.push(svg("line", { x1: m.l, y1: m.t, x2: m.l, y2: m.t + ih, stroke: AXIS, "stroke-width": 1.2 }));
  kids.push(svg("text", { x: m.l + iw / 2, y: H - 4, fill: AXIS, "font-size": 11, "text-anchor": "middle" },
    [document.createTextNode("step")]));

  // Lines
  series.forEach((s, i) => {
    const pts = (s[key] || []).filter(([, y]) => !logY || y > 0);
    if (pts.length < 2) return;
    const d = pts.map(([x, y], j) => `${j ? "L" : "M"}${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join(" ");
    kids.push(svg("path", {
      d, fill: "none", stroke: memberColor(i, series.length),
      "stroke-width": 1.6, "stroke-linejoin": "round", "stroke-opacity": 0.9,
    }));
  });

  return svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "etc-panel", preserveAspectRatio: "xMidYMid meet" }, kids);
}

function legend(series) {
  const wrap = document.createElement("div");
  wrap.className = "etc-legend";
  series.forEach((s, i) => {
    const item = document.createElement("span");
    item.className = "etc-legend-item";
    const sw = document.createElement("span");
    sw.className = "etc-swatch";
    sw.style.background = memberColor(i, series.length);
    const lbl = document.createElement("span");
    lbl.textContent = s.name;
    item.appendChild(sw);
    item.appendChild(lbl);
    wrap.appendChild(item);
  });
  return wrap;
}

export async function mountEnsembleTrainCurves(el, jsonUrl) {
  if (!el) return;
  let data;
  try {
    const resp = await fetch(jsonUrl, { cache: "no-store" });
    if (!resp.ok) throw new Error(String(resp.status));
    data = await resp.json();
  } catch (e) {
    el.style.display = "none";
    return;
  }
  const series = (data && data.members) || [];
  if (!series.length) { el.style.display = "none"; return; }

  el.innerHTML = "";
  const panels = document.createElement("div");
  panels.className = "etc-panels";
  panels.appendChild(drawPanel(series, "psnr", { title: "PSNR (stretched, dB)" }));
  panels.appendChild(drawPanel(series, "loss", { title: "Combined loss (held-out)", logY: true }));
  el.appendChild(panels);
  el.appendChild(legend(series));
}
