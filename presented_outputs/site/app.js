"use strict";
/* Mini-site for the WISDM results. All data comes from data.js (built by build_site.py). */
const D = window.DATA;
if (!D) document.getElementById("app").innerHTML = "<p>data.js is missing – run <code>python presented_outputs/build_site.py</code>.</p>";
const CLASSES = D.classes;

// ------------------------------------------------------------------ small helpers
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const isNum = v => typeof v === "number" && isFinite(v);
const f3 = v => isNum(v) ? v.toFixed(3) : "–";
const f1d = v => isNum(v) ? v.toFixed(1) : "–";
const f0 = v => isNum(v) ? Math.round(v).toString() : "–";
const pct = v => isNum(v) ? (100 * v).toFixed(1) + " %" : "–";
const sgn = v => isNum(v) ? (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(3) : "–";
const pfmt = p => !isNum(p) ? "–" : p < 0.001 ? "<0.001" : p.toFixed(3);
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "html") e.innerHTML = v;
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) e.append(kid instanceof Node ? kid : document.createTextNode(kid));
  return e;
}
const NS = "http://www.w3.org/2000/svg";
function S(tag, attrs = {}, text) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
}
const tipEl = document.getElementById("tip");
function hover(node, html) {
  node.addEventListener("mousemove", ev => {
    tipEl.innerHTML = typeof html === "function" ? html() : html; tipEl.style.display = "block";
    const x = Math.min(ev.clientX + 14, window.innerWidth - tipEl.offsetWidth - 8);
    tipEl.style.left = x + "px"; tipEl.style.top = ev.clientY + 14 + "px";
  });
  node.addEventListener("mouseleave", () => tipEl.style.display = "none");
}
function stat(values) {
  const a = values.filter(isNum);
  if (!a.length) return null;
  const mean = a.reduce((s, x) => s + x, 0) / a.length;
  const std = a.length > 1 ? Math.sqrt(a.reduce((s, x) => s + (x - mean) ** 2, 0) / (a.length - 1)) : 0;
  return { mean, std, n: a.length, min: Math.min(...a), max: Math.max(...a) };
}
const pmStr = (s, fmt = f3) => s ? `${fmt(s.mean)} ± ${fmt(s.std)}` : "–";

// ------------------------------------------------------------------ labels, colours
const TOKENS = { central: "centralised", fedavg: "FedAvg", fedbn: "FedBN", globalbn: "global BN",
  k1check: "K=1 (sanity check)", r80: "R=80 rounds", e3: "E=3 local epochs" };
const methodLabel = m => { const [model, ...rest] = m.split("-");
  return [model.toUpperCase(), ...rest.map(t => TOKENS[t] || t)].join(" · "); };
const expLabel = e => { if (e === "central") return "Centralised";
  const [s, k] = e.split("_K"); return `${(D.split_names || {})[s] || s} · K=${k}`; };
const COLORS = { "fcn-central": "#6e7781", "lstm-central": "#b08800", "fcn-fedavg": "#0969da", "fcn-fedbn": "#1a7f37",
  "fcn-fedbn-globalbn": "#e16f24", "lstm-fedavg": "#8250df", "lstm-fedavg-r80": "#cf222e", "fcn-fedavg-e3": "#54aeff",
  "fcn-fedbn-e3": "#4ac26b", "fcn-fedbn-e3-globalbn": "#f0a35e", "fcn-fedavg-k1check": "#bf3989" };
const EXTRA = ["#1b7c83", "#953800", "#6639ba", "#0550ae", "#116329", "#a40e26"];
const colorOf = m => COLORS[m] || (COLORS[m] = EXTRA[Object.keys(COLORS).length % EXTRA.length]);
const CLASS_COLORS = ["#0969da", "#1a7f37", "#cf222e", "#8250df", "#bf8700", "#1b7c83"];

// ------------------------------------------------------------------ data model
const RUNS = D.runs.map((r, i) => ({ ...r, idx: i }));
const groupsMap = new Map();
for (const r of RUNS) {
  const key = `${r.experiment}|${r.method}`;
  if (!groupsMap.has(key)) groupsMap.set(key, { key, experiment: r.experiment, method: r.method, model: r.model, strategy: r.strategy, runs: [] });
  groupsMap.get(key).runs.push(r);
}
const expOrder = e => e === "central" ? [-1, 0, 0] : [Number(e.slice(1, 2)), Number(e.split("_K")[1]), 0];
const cmpExp = (a, b) => { const x = expOrder(a), y = expOrder(b); return x[0] - y[0] || y[1] - x[1] || a.localeCompare(b); };
const METHOD_ORDER = Object.keys(COLORS);
const cmpMethod = (a, b) => { const i = METHOD_ORDER.indexOf(a), j = METHOD_ORDER.indexOf(b);
  return (i < 0 ? 99 : i) - (j < 0 ? 99 : j) || a.localeCompare(b); };
const GROUPS = [...groupsMap.values()].sort((a, b) => cmpExp(a.experiment, b.experiment) || cmpMethod(a.method, b.method));
const EXPERIMENTS = [...new Set(GROUPS.map(g => g.experiment))].sort(cmpExp);
const METHODS = [...new Set(GROUPS.map(g => g.method))].sort(cmpMethod);
const CENTRAL = {};
for (const r of RUNS) if (r.kind === "central") (CENTRAL[r.model] ||= {})[`${r.fold}-${r.seed}`] = r;
const groupOf = (exp, method) => groupsMap.get(`${exp}|${method}`);
const centralGroup = model => groupOf("central", `${model}-central`);

const METRICS = {
  test_f1: { label: "Test macro-F1", get: r => r.test.f1, hi: true, fmt: f3, central: true },
  test_bacc: { label: "Test balanced accuracy", get: r => r.test.bacc, hi: true, fmt: f3, central: true },
  test_acc: { label: "Test accuracy", get: r => r.test.acc, hi: true, fmt: f3, central: true },
  val_f1: { label: "Validation macro-F1", get: r => r.val.f1, hi: true, fmt: f3, central: true },
  worst_client: { label: "Worst-client F1", get: r => r.worst_client, hi: true, fmt: f3 },
  client_std: { label: "F1 std across clients", get: r => r.client_std, hi: false, fmt: f3 },
  median_user: { label: "Median per-user accuracy", get: r => r.users.median, hi: true, fmt: f3, central: true },
  below50: { label: "Users with acc. < 50 %", get: r => r.users.below50, hi: false, fmt: pct, central: true },
  best: { label: "Selected epoch / round", get: r => r.best, hi: false, fmt: f1d, central: true },
  seconds: { label: "Training time (s)", get: r => r.seconds, hi: false, fmt: f1d, central: true },
  comm: { label: "MB communicated up to the selected round", get: r => r.comm_mb_to_best, hi: false, fmt: f0 },
};
const groupStat = (g, metric) => stat(g.runs.map(METRICS[metric].get));

/** Paired difference to the centralised model of the same architecture, per (fold, seed). */
function vsCentral(g, metric = "test_f1") {
  if (g.experiment === "central" || !CENTRAL[g.model]) return null;
  const get = METRICS[metric].get, d = [];
  for (const r of g.runs) { const c = CENTRAL[g.model][`${r.fold}-${r.seed}`];
    if (c && isNum(get(r)) && isNum(get(c))) d.push(get(r) - get(c)); }
  if (!d.length) return null;
  const s = stat(d);
  return { mean: s.mean, std: s.std, n: d.length, wins: d.filter(x => x > 0).length };
}
const findTest = (exp, method, kind) => (D.tests || []).find(t => t.experiment === exp && t.method === method && t.kind === kind);
function sigBadge(t) {
  if (!t || !isNum(t.p_holm ?? t.p)) return "";
  const p = t.p_holm ?? t.p;
  const cls = p < 0.05 ? (t.mean_diff >= 0 ? "sig" : "sigbad") : "nsig";
  return `<span class="badge ${cls}">p${t.p_holm != null ? "<sub>Holm</sub>" : ""} ${pfmt(p)}</span>`;
}

// ------------------------------------------------------------------ chart primitives
function niceTicks(a, b, n = 5) {
  if (a === b) { a -= 0.5; b += 0.5; }
  const raw = (b - a) / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(s => s * mag).find(s => s >= raw) || raw;
  const out = []; for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
function padDomain(values, pad = 0.05, clamp01 = false) {
  const a = values.filter(isNum); if (!a.length) return [0, 1];
  let lo = Math.min(...a), hi = Math.max(...a); const span = (hi - lo) || Math.abs(hi) || 1;
  lo -= span * pad; hi += span * pad;
  if (clamp01) { lo = Math.max(0, lo); hi = Math.min(1, hi); }
  return [lo, hi];
}
function frame({ W = 520, H = 260, m = { l: 48, r: 14, t: 26, b: 36 }, x, y, title, xlabel, ylabel, xticks, yticks, xfmt = String, yfmt = v => +v.toFixed(3) }) {
  const svg = S("svg", { viewBox: `0 0 ${W} ${H}`, width: W });
  const X = v => m.l + (v - x[0]) / ((x[1] - x[0]) || 1) * (W - m.l - m.r);
  const Y = v => H - m.b - (v - y[0]) / ((y[1] - y[0]) || 1) * (H - m.t - m.b);
  if (title) svg.append(S("text", { x: m.l, y: 15, class: "title" }, title));
  for (const t of yticks ?? niceTicks(y[0], y[1])) {
    svg.append(S("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "grid" }));
    svg.append(S("text", { x: m.l - 6, y: Y(t) + 3.5, "text-anchor": "end", class: "tick" }, yfmt(t)));
  }
  if (xticks !== null) for (const t of xticks ?? niceTicks(x[0], x[1], 6))
    svg.append(S("text", { x: X(t), y: H - m.b + 14, "text-anchor": "middle", class: "tick" }, xfmt(t)));
  svg.append(S("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }));
  svg.append(S("line", { x1: m.l, x2: m.l, y1: m.t, y2: H - m.b, class: "axis" }));
  if (xlabel) svg.append(S("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle", class: "tick" }, xlabel));
  if (ylabel) svg.append(S("text", { x: 11, y: (m.t + H - m.b) / 2, "text-anchor": "middle", class: "tick",
    transform: `rotate(-90 11 ${(m.t + H - m.b) / 2})` }, ylabel));
  return { svg, X, Y, W, H, m };
}
function legend(items) {
  return el("div", { class: "legend" }, items.map(([name, color, dash]) =>
    el("span", {}, el("span", { class: "sw", style: `background:${color};${dash ? "opacity:.55" : ""}` }), " " + name)));
}

/** series: [{name, color, x[], y[], lo?[], hi?[], dash?}] */
function lineChart(series, { title, xlabel, ylabel, vlines = [], hlines = [], W = 520, H = 250, y01 = false, xfmt } = {}) {
  series = series.filter(s => s.y.some(isNum));
  if (!series.length) return el("div", { class: "muted" }, `${title || ""}: no data`);
  const xs = series.flatMap(s => s.x), ys = series.flatMap(s => [...s.y, ...(s.lo || []), ...(s.hi || [])]);
  const f = frame({ W, H, title, xlabel, ylabel, xfmt, x: [Math.min(...xs), Math.max(...xs)],
    y: padDomain([...ys, ...hlines.map(h => h.y)], 0.06, y01) });
  for (const h of hlines) {
    f.svg.append(S("line", { x1: f.m.l, x2: W - f.m.r, y1: f.Y(h.y), y2: f.Y(h.y), stroke: h.color, "stroke-dasharray": "5 4", "stroke-width": 1.3 }));
    f.svg.append(S("text", { x: W - f.m.r - 2, y: f.Y(h.y) - 3, "text-anchor": "end", class: "tick" }, h.label));
  }
  for (const v of vlines) f.svg.append(S("line", { x1: f.X(v.x), x2: f.X(v.x), y1: f.m.t, y2: H - f.m.b, stroke: v.color, "stroke-dasharray": "3 3", "stroke-width": 1.4 }));
  for (const s of series) {
    const pts = s.x.map((x, i) => [x, s.y[i], s.lo?.[i], s.hi?.[i]]).filter(p => isNum(p[1]));
    if (s.lo) {
      const up = pts.filter(p => isNum(p[3])).map(p => `${f.X(p[0])},${f.Y(p[3])}`);
      const down = pts.filter(p => isNum(p[2])).reverse().map(p => `${f.X(p[0])},${f.Y(p[2])}`);
      f.svg.append(S("polygon", { points: [...up, ...down].join(" "), fill: s.color, opacity: 0.12 }));
    }
    f.svg.append(S("path", { d: "M" + pts.map(p => `${f.X(p[0])},${f.Y(p[1])}`).join("L"), fill: "none", stroke: s.color,
      "stroke-width": 1.8, "stroke-dasharray": s.dash ? "5 3" : null }));
  }
  // hover column: values of every series at the nearest x
  const allX = [...new Set(xs)].sort((a, b) => a - b);
  const cursor = S("line", { y1: f.m.t, y2: H - f.m.b, stroke: "currentColor", opacity: 0, "stroke-width": 1 });
  const overlay = S("rect", { x: f.m.l, y: f.m.t, width: W - f.m.l - f.m.r, height: H - f.m.t - f.m.b, fill: "transparent" });
  f.svg.append(cursor, overlay);
  let near = allX[0];
  overlay.addEventListener("mousemove", ev => {
    const box = f.svg.getBoundingClientRect(), px = (ev.clientX - box.left) * W / box.width;
    near = allX.reduce((b, x) => Math.abs(f.X(x) - px) < Math.abs(f.X(b) - px) ? x : b, allX[0]);
    cursor.setAttribute("x1", f.X(near)); cursor.setAttribute("x2", f.X(near)); cursor.setAttribute("opacity", 0.35);
  });
  overlay.addEventListener("mouseleave", () => cursor.setAttribute("opacity", 0));
  hover(overlay, () => `<b>${esc(xlabel || "x")} ${esc(xfmt ? xfmt(near) : near)}</b><br>` + series.map(s => {
    const i = s.x.indexOf(near); return i < 0 || !isNum(s.y[i]) ? "" :
      `<span class="sw" style="background:${s.color}"></span> ${esc(s.name)}: <b>${f3(s.y[i])}</b>${s.lo && isNum(s.hi[i]) ? ` <span class="muted">± ${f3(s.hi[i] - s.y[i])}</span>` : ""}<br>`;
  }).join(""));
  const wrap = el("div", {}, f.svg);
  if (series.length > 1) wrap.append(legend(series.map(s => [s.name, s.color, s.dash])));
  return wrap;
}

/** Grouped vertical bars. cats: labels; series: [{name, color, values[], err?[]}] */
function barChart(cats, series, { title, ylabel, W = 560, H = 260, y = [0, 1], refs = [] } = {}) {
  const f = frame({ W, H, title, ylabel, x: [0, cats.length], y, xticks: null, m: { l: 44, r: 10, t: 26, b: 44 } });
  const bw = (f.X(1) - f.X(0)) * 0.8 / series.length;
  cats.forEach((c, i) => {
    f.svg.append(S("text", { x: f.X(i + 0.5), y: H - f.m.b + 14, "text-anchor": "middle", class: "tick" }, c));
    series.forEach((s, j) => {
      const v = s.values[i]; if (!isNum(v)) return;
      const x = f.X(i + 0.1) + j * bw, top = f.Y(Math.min(v, y[1]));
      const rect = S("rect", { x, y: top, width: Math.max(bw - 1, 1), height: Math.max(f.Y(y[0]) - top, 0), fill: s.color, rx: 1.5 });
      hover(rect, `<b>${esc(c)}</b><br>${esc(s.name)}: <b>${f3(v)}</b>${isNum(s.err?.[i]) ? ` ± ${f3(s.err[i])}` : ""}`);
      f.svg.append(rect);
      if (isNum(s.err?.[i])) f.svg.append(S("line", { x1: x + bw / 2, x2: x + bw / 2, y1: f.Y(Math.min(v + s.err[i], y[1])), y2: f.Y(Math.max(v - s.err[i], y[0])), stroke: "currentColor", opacity: 0.5 }));
    });
  });
  for (const r of refs) f.svg.append(S("line", { x1: f.m.l, x2: W - f.m.r, y1: f.Y(r.y), y2: f.Y(r.y), stroke: r.color, "stroke-dasharray": "5 4" }));
  const wrap = el("div", {}, f.svg);
  if (series.length > 1) wrap.append(legend(series.map(s => [s.name, s.color])));
  return wrap;
}

/** Row-normalised confusion matrix with counts in the tooltip. */
function confusion(cm, { title = "", color = "#0969da" } = {}) {
  const k = cm.length, cell = 46, m = { l: 80, t: 34, r: 8, b: 8 };
  const W = m.l + k * cell + m.r, H = m.t + k * cell + 66;
  const svg = S("svg", { viewBox: `0 0 ${W} ${H}`, width: W });
  svg.append(S("text", { x: 0, y: 14, class: "title" }, title));
  cm.forEach((row, i) => {
    const tot = row.reduce((a, b) => a + b, 0);
    svg.append(S("text", { x: m.l - 6, y: m.t + i * cell + cell / 2 + 4, "text-anchor": "end", class: "tick" }, CLASSES[i]));
    row.forEach((v, j) => {
      const share = tot ? v / tot : 0;
      const r = S("rect", { x: m.l + j * cell, y: m.t + i * cell, width: cell - 2, height: cell - 2, fill: color, "fill-opacity": 0.06 + 0.94 * share, rx: 3 });
      hover(r, `true <b>${CLASSES[i]}</b> → predicted <b>${CLASSES[j]}</b><br>${v} windows (${pct(share)} of the row)`);
      svg.append(r, S("text", { x: m.l + j * cell + cell / 2 - 1, y: m.t + i * cell + cell / 2 + 3, "text-anchor": "middle",
        style: `fill:${share > 0.55 ? "#fff" : "var(--fg)"};font-size:10.5px;pointer-events:none` }, Math.round(share * 100)));
    });
  });
  CLASSES.forEach((c, j) => svg.append(S("text", { x: m.l + j * cell + cell / 2, y: m.t + k * cell + 12, "text-anchor": "end", class: "tick",
    transform: `rotate(-35 ${m.l + j * cell + cell / 2} ${m.t + k * cell + 12})` }, c)));
  svg.append(S("text", { x: m.l, y: 28, class: "tick" }, "row = true class, column = prediction (% of row)"));
  return svg;
}

/** Horizontal mean ± std dots; rows: [{label, color, stat, group, onClick, note}] */
function dotPlot(rows, { title, xlabel, refs = [], W = 940 } = {}) {
  const rowH = 21, m = { l: 290, r: 70, t: 18 + 12 * Math.max(refs.length, 1), b: 34 };
  const H = m.t + m.b + rows.length * rowH;
  const vals = rows.flatMap(r => r.stat ? [r.stat.mean - r.stat.std, r.stat.mean + r.stat.std] : []);
  const f = frame({ W, H, m, title, xlabel, x: padDomain([...vals, ...refs.map(r => r.x)], 0.04), y: [0, rows.length],
    yticks: [], xfmt: v => +v.toFixed(3) });
  let prev = null;
  rows.forEach((r, i) => {
    const cy = m.t + i * rowH + rowH / 2;
    if (r.group !== prev) {
      if (i) f.svg.append(S("line", { x1: 4, x2: W - m.r, y1: m.t + i * rowH, y2: m.t + i * rowH, class: "axis" }));
      f.svg.append(S("text", { x: 4, y: cy + 4, class: "title", style: "font-size:11.5px" }, r.groupLabel)); prev = r.group;
    }
    const lbl = S("text", { x: m.l - 8, y: cy + 4, "text-anchor": "end", style: r.onClick ? "cursor:pointer" : "" }, r.label);
    if (r.onClick) lbl.addEventListener("click", r.onClick);
    f.svg.append(lbl);
    if (!r.stat) return;
    const s = r.stat;
    f.svg.append(S("line", { x1: f.X(s.mean - s.std), x2: f.X(s.mean + s.std), y1: cy, y2: cy, stroke: r.color, "stroke-width": 2, opacity: 0.6 }));
    const dot = S("circle", { cx: f.X(s.mean), cy, r: 5, fill: r.color, style: r.onClick ? "cursor:pointer" : "" });
    if (r.onClick) dot.addEventListener("click", r.onClick);
    hover(dot, `<b>${esc(r.groupLabel)} – ${esc(r.label)}</b><br>${f3(s.mean)} ± ${f3(s.std)} (n = ${s.n})${r.note ? "<br>" + r.note : ""}`);
    f.svg.append(dot, S("text", { x: W - m.r + 6, y: cy + 4, class: "tick" }, f3(s.mean)));
  });
  refs.forEach((r, i) => {
    f.svg.append(S("line", { x1: f.X(r.x), x2: f.X(r.x), y1: m.t - 8 - 12 * i, y2: H - m.b, stroke: r.color, "stroke-dasharray": "5 4", "stroke-width": 1.3 }));
    f.svg.append(S("text", { x: f.X(r.x) + 4, y: m.t - 6 - 12 * i, class: "tick", style: `fill:${r.color}` }, r.label));
  });
  return f.svg;
}

/** Scatter: points [{x, y, color, label, onClick}] */
function scatter(points, { title, xlabel, ylabel, W = 560, H = 320 } = {}) {
  const f = frame({ W, H, title, xlabel, ylabel, x: padDomain(points.map(p => p.x), 0.08), y: padDomain(points.map(p => p.y), 0.08), m: { l: 52, r: 14, t: 26, b: 40 } });
  for (const p of points) {
    const c = S("circle", { cx: f.X(p.x), cy: f.Y(p.y), r: p.r || 5, fill: p.color, "fill-opacity": 0.8, stroke: p.stroke || "none", style: p.onClick ? "cursor:pointer" : "" });
    hover(c, p.label + `<br>x = ${f3(p.x)}, y = ${f3(p.y)}`);
    if (p.onClick) c.addEventListener("click", p.onClick);
    f.svg.append(c);
  }
  return f.svg;
}

/** Jittered strip plot per category with mean marker. cats: [{label, color, values}] */
function strip(cats, { title, ylabel, W = 560, H = 340, y = [0, 1] } = {}) {
  const f = frame({ W, H, title, ylabel, x: [0, cats.length], y, xticks: null, m: { l: 44, r: 10, t: 26, b: 130 } });
  cats.forEach((c, i) => {
    const cx = f.X(i + 0.5), lx = cx;
    f.svg.append(S("text", { x: lx, y: H - f.m.b + 12, "text-anchor": "end", class: "tick", transform: `rotate(-38 ${lx} ${H - f.m.b + 12})` }, c.label));
    c.values.forEach((v, j) => { if (!isNum(v.y)) return;
      const jit = ((j * 0.618034) % 1 - 0.5) * (f.X(1) - f.X(0)) * 0.6;
      const d = S("circle", { cx: cx + jit, cy: f.Y(v.y), r: 3, fill: c.color, "fill-opacity": 0.55 });
      if (v.label) hover(d, v.label);
      f.svg.append(d); });
    const s = stat(c.values.map(v => v.y));
    if (s) { const w = (f.X(1) - f.X(0)) * 0.35;
      f.svg.append(S("line", { x1: cx - w, x2: cx + w, y1: f.Y(s.mean), y2: f.Y(s.mean), stroke: "currentColor", "stroke-width": 2.5 })); }
  });
  return f.svg;
}

function histogram(values, { title, xlabel, bins = 10, W = 460, H = 230, color = "#0969da" } = {}) {
  const counts = Array(bins).fill(0);
  for (const v of values.filter(isNum)) counts[Math.min(bins - 1, Math.floor(v * bins))]++;
  const f = frame({ W, H, title, xlabel, ylabel: "users", x: [0, 1], y: [0, Math.max(1, ...counts) * 1.1], yfmt: v => Math.round(v) });
  counts.forEach((c, i) => { const r = S("rect", { x: f.X(i / bins) + 1, y: f.Y(c), width: f.X(1 / bins) - f.X(0) - 2, height: f.Y(0) - f.Y(c), fill: color, opacity: 0.8 });
    hover(r, `acc. ${pct(i / bins)} – ${pct((i + 1) / bins)}: <b>${c}</b> users`); f.svg.append(r); });
  return f.svg;
}

/** Sortable table. cols: [{label, get, fmt?, html?, cls?, title?}] */
function table(cols, rows, { onClick, sortBy = null, desc = true, selected } = {}) {
  let key = sortBy, down = desc;
  const wrap = el("div", { class: "tablewrap" });
  const draw = () => {
    const sorted = key == null ? rows : [...rows].sort((a, b) => {
      const x = cols[key].get(a), y = cols[key].get(b);
      if (x == null) return 1; if (y == null) return -1;
      return (typeof x === "string" ? x.localeCompare(y) : x - y) * (down ? -1 : 1);
    });
    const t = el("table", {}, el("thead", {}, el("tr", {}, cols.map((c, i) =>
      el("th", { class: `sortable ${c.cls || ""}`, title: c.title, onclick: () => { down = key === i ? !down : true; key = i; draw(); } },
        c.label + (key === i ? (down ? " ▾" : " ▴") : ""))))));
    const tb = el("tbody");
    for (const r of sorted) {
      const tr = el("tr", { class: (onClick ? "click " : "") + (selected && selected(r) ? "sel" : ""), onclick: onClick ? () => onClick(r) : null });
      for (const c of cols) tr.append(el("td", { class: c.cls, html: c.html ? c.html(r) : esc((c.fmt || (v => v ?? "–"))(c.get(r))) }));
      tb.append(tr);
    }
    t.append(tb); wrap.replaceChildren(t);
  };
  draw();
  return wrap;
}

// ------------------------------------------------------------------ aggregation helpers
/** Mean ± std learning curve of one history key over the runs of a group. */
function meanCurve(g, key) {
  const len = Math.max(...g.runs.map(r => (r.history.x || []).length));
  const x = [], y = [], lo = [], hi = [];
  for (let i = 0; i < len; i++) {
    const s = stat(g.runs.map(r => r.history[key]?.[i]));
    if (!s) continue;
    x.push(i + 1); y.push(s.mean); lo.push(s.mean - s.std); hi.push(s.mean + s.std);
  }
  return { name: methodLabel(g.method), color: colorOf(g.method), x, y, lo, hi };
}
const sumCm = runs => runs.reduce((acc, r) => r.cm ? acc.map((row, i) => row.map((v, j) => v + r.cm[i][j])) : acc,
  CLASSES.map(() => CLASSES.map(() => 0)));
const card = (label, value, sub) => el("div", { class: "card" }, el("div", { class: "v", html: value }), el("div", { class: "l" }, label), sub ? el("div", { class: "s", html: sub }) : null);
const go = (page, params = {}) => { location.hash = page + (Object.keys(params).length ? "?" + new URLSearchParams(params) : ""); };
function select(options, value, onChange, labelFn = String) {
  const s = el("select", { onchange: e => onChange(e.target.value) }, options.map(o => el("option", { value: o, selected: o === value }, labelFn(o))));
  return s;
}

// ------------------------------------------------------------------ pages
function pageOverview(app) {
  const cf = centralGroup("fcn"), cl = centralGroup("lstm");
  const fed = GROUPS.filter(g => g.experiment !== "central");
  const ranked = fed.map(g => ({ g, s: groupStat(g, "test_f1"), d: vsCentral(g) })).filter(x => x.s).sort((a, b) => b.s.mean - a.s.mean);
  const best = ranked[0], worst = ranked[ranked.length - 1];
  app.append(el("h1", {}, "What happened in the experiments"),
    el("p", { class: "note", html: `Recognising 6 activities (${CLASSES.join(", ")}) from 10 s windows of the phone accelerometer.
      Every configuration was run on <b>5 folds × 3 seeds = 15 runs</b>. The test set only contains <b>people never seen</b> during training.
      The main metric is <b>macro-F1</b>: the mean F1 over classes, so rare classes (Stairs, LyingDown) count as much as Walking.` }));
  app.append(el("div", { class: "cards" },
    card("runs found in outputs/", RUNS.filter(r => !r.id.endsWith("[global BN]")).length, `${D.scanned_files} JSON files scanned`),
    card("configurations (experiment × method)", GROUPS.length, `${EXPERIMENTS.length - 1} federated scenarios`),
    cf && card("Centralised FCN · test F1", f3(groupStat(cf, "test_f1").mean), `± ${f3(groupStat(cf, "test_f1").std)}`),
    cl && card("Centralised LSTM · test F1", f3(groupStat(cl, "test_f1").mean), `± ${f3(groupStat(cl, "test_f1").std)}`),
    best && card("best federated", f3(best.s.mean), `${esc(methodLabel(best.g.method))}<br>${esc(expLabel(best.g.experiment))}`),
    worst && card("worst federated", f3(worst.s.mean), `${esc(methodLabel(worst.g.method))}<br>${esc(expLabel(worst.g.experiment))}`)));

  app.append(el("h2", {}, "Test macro-F1 for every configuration"),
    el("p", { class: "note", html: "Dot = mean of the 15 runs, bar = ± one standard deviation. The dashed lines are the centralised models (all data pooled in one place), i.e. the reference. Click a configuration for details." }));
  const refs = [cf && { x: groupStat(cf, "test_f1").mean, color: colorOf("fcn-central"), label: "FCN centralised" },
    cl && { x: groupStat(cl, "test_f1").mean, color: colorOf("lstm-central"), label: "LSTM centralised" }].filter(Boolean);
  app.append(el("div", { class: "panel" }, dotPlot(GROUPS.map(g => {
    const d = vsCentral(g), t = findTest(g.experiment, g.method, "vs_central");
    return { label: methodLabel(g.method), color: colorOf(g.method), stat: groupStat(g, "test_f1"), group: g.experiment,
      groupLabel: expLabel(g.experiment), onClick: () => go("experiment", { e: g.experiment }),
      note: d ? `Δ vs. centralised: <b>${sgn(d.mean)}</b> (better in ${d.wins}/${d.n} pairs) ${sigBadge(t)}` : "" };
  }), { xlabel: "test macro-F1", refs })));

  app.append(el("h2", {}, "How to read the results"), el("div", { class: "explain", html: `
    <div><b>Centralised</b>All training data is in one place. This is the reference: what can be achieved without any privacy constraints.</div>
    <div><b>FedAvg</b>Each client (a “site”) trains locally and the server takes a weighted average of the models. The data never leaves the client.</div>
    <div><b>FedBN</b>Like FedAvg, but the BatchNorm layers stay local: every client keeps its own normalisation. Helps when the clients’ data differs.</div>
    <div><b>FedBN · global BN</b>The same FedBN model evaluated with averaged BatchNorm: simulates a new client without its own statistics.</div>
    <div><b>S0 IID</b>Windows are split at random, so all clients have identical distributions. This is the ideal scenario.</div>
    <div><b>S1 natural</b>Each user belongs to exactly one client. Differences come from how people carry their phone.</div>
    <div><b>S2 user-Dirichlet</b>Users are assigned unevenly by their dominant activity. Clients differ in size and in activity mix.</div>
    <div><b>S3 window-Dirichlet</b>The windows of each class are split very unevenly across clients. This is the stress test.</div>
    <div><b>K, R, E</b>K = number of clients, R = number of federated rounds, E = local training epochs per round.</div>
    <div><b>Worst client / std across clients</b>Fairness across clients: how well the weakest client does and how much F1 varies between clients.</div>` }));
  if (D.skipped?.length) app.append(el("details", {}, el("summary", {}, `${D.skipped.length} JSON files without run metrics`), el("pre", {}, D.skipped.join("\n"))));
}

function pageCompare(app, params) {
  const metric = METRICS[params.get("m")] ? params.get("m") : "test_f1";
  const M = METRICS[metric];
  app.append(el("h1", {}, "Comparison across scenarios and methods"),
    el("div", { class: "controls" }, el("label", {}, "Metric:", select(Object.keys(METRICS), metric, v => go("compare", { m: v }), k => METRICS[k].label))));
  const exps = EXPERIMENTS.filter(e => e !== "central"), meths = METHODS.filter(m => !m.endsWith("-central"));
  const cells = [];
  for (const e of exps) for (const m of meths) { const g = groupOf(e, m); if (g) { const s = groupStat(g, metric); if (s) cells.push(s.mean); } }
  const [lo, hi] = [Math.min(...cells), Math.max(...cells)];
  const colour = v => { if (!isNum(v)) return "transparent"; let t = (v - lo) / ((hi - lo) || 1); if (!M.hi) t = 1 - t;
    return `hsla(${Math.round(t * 120)}, 65%, 45%, ${0.18 + 0.5 * Math.abs(t - 0.5) * 2})`; };
  app.append(el("h2", {}, `Matrix: ${M.label}`), el("p", { class: "note", html:
    `Green = better, red = worse (relative to the other cells). ${M.central ? "The bottom row is the centralised reference. " : ""}
     ${metric === "test_f1" ? "Below the value: mean difference to the centralised model with the same architecture, on the same folds and seeds; below 0 means a loss caused by federation. " : ""}Click a cell to open the experiment.` }));
  const t = el("table", { class: "matrix" }, el("thead", {}, el("tr", {}, el("th", {}, "scenario \\ method"), meths.map(m => el("th", {}, methodLabel(m))))));
  const tb = el("tbody");
  const rowsFor = M.central ? [...exps, "central"] : exps;
  for (const e of rowsFor) {
    const tr = el("tr", {}, el("td", { class: "l", style: "cursor:default" }, expLabel(e)));
    for (const m of meths) {
      const g = e === "central" ? centralGroup(m.split("-")[0]) : groupOf(e, m);
      const s = g && (e !== "central" || m.split("-").length === 2 && m.endsWith("fedavg")) ? groupStat(g, metric) : null;
      if (!s) { tr.append(el("td", { class: "muted", style: "cursor:default" }, "")); continue; }
      const d = metric === "test_f1" && e !== "central" ? vsCentral(g) : null;
      const test = d && findTest(e, m, "vs_central");
      tr.append(el("td", { style: `background:${colour(s.mean)}`, onclick: () => go("experiment", { e: g.experiment }),
        html: `${esc(M.fmt(s.mean))}${d ? `<small>${sgn(d.mean)} · ${d.wins}/${d.n}${test && (test.p_holm ?? 1) < 0.05 ? " *" : ""}</small>` : ""}` }));
    }
    tb.append(tr);
  }
  t.append(tb);
  app.append(el("div", { class: "tablewrap" }, t), el("p", { class: "note" }, "* = statistically significant (Wilcoxon with Holm correction, from summary_fl.json)."));

  // fairness scatter + K / skew effect
  const grid = el("div", { class: "grid" });
  const pts = GROUPS.filter(g => g.experiment !== "central").map(g => ({ g, x: groupStat(g, "test_f1"), y: groupStat(g, "worst_client") }))
    .filter(p => p.x && p.y).map(p => ({ x: p.x.mean, y: p.y.mean, color: colorOf(p.g.method), r: 3 + 2 * Number(p.g.experiment.split("_K")[1] || 1) / 2,
      label: `<b>${esc(methodLabel(p.g.method))}</b><br>${esc(expLabel(p.g.experiment))}`, onClick: () => go("experiment", { e: p.g.experiment }) }));
  grid.append(el("div", { class: "panel" }, el("h3", {}, "Overall performance vs. fairness across clients"),
    el("p", { class: "note" }, "Each dot is a configuration. Top-right is ideal: a good model on average whose weakest client is not left behind."),
    scatter(pts, { xlabel: "test macro-F1 (all clients)", ylabel: "worst-client F1" }),
    legend([...new Set(GROUPS.filter(g => g.experiment !== "central").map(g => g.method))].map(m => [methodLabel(m), colorOf(m)]))));

  const families = ["fcn-fedavg", "fcn-fedbn", "fcn-fedbn-globalbn", "lstm-fedavg"].filter(m => METHODS.includes(m));
  for (const K of [...new Set(exps.map(e => e.split("_K")[1]))].filter(k => k !== "1")) {
    const kexps = exps.filter(e => e.endsWith(`_K${K}`));
    const series = families.map(m => ({ name: methodLabel(m), color: colorOf(m), values: kexps.map(e => groupOf(e, m) && groupStat(groupOf(e, m), metric)?.mean),
      err: kexps.map(e => groupOf(e, m) && groupStat(groupOf(e, m), metric)?.std) })).filter(s => s.values.some(isNum));
    const all = series.flatMap(s => s.values.map((v, i) => [v - (s.err[i] || 0), v + (s.err[i] || 0)]).flat());
    const centralRef = M.central ? ["fcn", "lstm"].map(mm => centralGroup(mm) && { y: groupStat(centralGroup(mm), metric).mean, color: colorOf(`${mm}-central`) }).filter(Boolean) : [];
    grid.append(el("div", { class: "panel" }, el("h3", {}, `Effect of data skew (K = ${K})`),
      el("p", { class: "note" }, "From left to right, the clients’ data becomes increasingly different. The dashed lines are the centralised models."),
      barChart(kexps.map(e => e.split("_K")[0].toUpperCase()), series, { ylabel: M.label, refs: centralRef,
        y: padDomain([...all, ...centralRef.map(r => r.y)], 0.1, M.fmt === f3 || M.fmt === pct) })));
  }
  app.append(el("h2", {}, "Trends"), grid);
}

function pageExperiment(app, params) {
  const exp = EXPERIMENTS.includes(params.get("e")) ? params.get("e") : EXPERIMENTS.find(e => e !== "central") || EXPERIMENTS[0];
  const groups = GROUPS.filter(g => g.experiment === exp);
  const hidden = new Set((params.get("hide") || "").split(",").filter(Boolean));
  const shown = groups.filter(g => !hidden.has(g.method));
  const toggle = m => { hidden.has(m) ? hidden.delete(m) : hidden.add(m); go("experiment", { e: exp, ...(hidden.size ? { hide: [...hidden].join(",") } : {}) }); };
  const refCentral = exp === "central" ? [] : [...new Set(groups.map(g => g.model))].map(centralGroup).filter(Boolean);

  app.append(el("h1", {}, expLabel(exp)), el("div", { class: "controls" },
    el("label", {}, "Experiment:", select(EXPERIMENTS, exp, v => go("experiment", { e: v }), expLabel)),
    el("div", { class: "chips" }, groups.map(g => el("span", { class: "chip" + (hidden.has(g.method) ? " off" : ""), onclick: () => toggle(g.method) },
      el("span", { class: "sw", style: `background:${colorOf(g.method)}` }), methodLabel(g.method))))));

  // summary table
  const rows = [...shown, ...refCentral];
  app.append(el("h2", {}, "Summary (mean of the 15 runs)"), table([
    { label: "method", get: g => methodLabel(g.method) + (g.experiment !== exp ? " (reference)" : ""), cls: "l",
      html: g => `<span class="sw" style="background:${colorOf(g.method)}"></span> ${esc(methodLabel(g.method))}${g.experiment !== exp ? ' <span class="muted">(reference)</span>' : ""}` },
    { label: "n", get: g => g.runs.length },
    { label: "test F1", get: g => groupStat(g, "test_f1")?.mean, html: g => pmStr(groupStat(g, "test_f1")) },
    { label: "Δ vs central", title: "paired difference to the centralised model (same folds and seeds)", get: g => vsCentral(g)?.mean,
      html: g => { const d = vsCentral(g); return d ? `<span class="${d.mean >= 0 ? "pos" : "neg"}">${sgn(d.mean)}</span> <span class="muted">${d.wins}/${d.n}</span> ${sigBadge(findTest(g.experiment, g.method, "vs_central"))}` : "–"; } },
    { label: "vs FedAvg", get: g => findTest(g.experiment, g.method, "vs_fedavg")?.mean_diff,
      html: g => { const t = findTest(g.experiment, g.method, "vs_fedavg"); return t ? `${sgn(t.mean_diff)} <span class="muted">${t.wins}/${t.pairs}</span> ${sigBadge(t)}` : "–"; } },
    { label: "bal. acc.", get: g => groupStat(g, "test_bacc")?.mean, fmt: f3 },
    { label: "acc.", get: g => groupStat(g, "test_acc")?.mean, fmt: f3 },
    { label: "worst client F1", get: g => groupStat(g, "worst_client")?.mean, fmt: f3 },
    { label: "client std", get: g => groupStat(g, "client_std")?.mean, fmt: f3 },
    { label: "users < 50 %", get: g => groupStat(g, "below50")?.mean, fmt: pct },
    { label: "selected round/epoch", get: g => groupStat(g, "best")?.mean, fmt: f1d },
    { label: "MB to best", get: g => groupStat(g, "comm")?.mean, fmt: f0 },
    { label: "seconds", get: g => groupStat(g, "seconds")?.mean, fmt: f1d },
  ], rows, { sortBy: 2 }));

  // partition
  const part = groups.find(g => g.runs[0].partition)?.runs[0].partition;
  if (part) {
    const k = part.train_windows.length;
    app.append(el("h2", {}, "How the data is split across clients (fold 0)"), el("p", { class: "note", html:
      `Mean JSD = ${f3(part.jsd_mean)}, max = ${f3(part.jsd_max)}. JSD measures how different a client’s activity distribution is from the overall one: 0 = identical, 1 = completely different.` }),
      el("div", { class: "grid" }, el("div", { class: "panel" }, barChart([...Array(k).keys()].map(i => `client ${i}`),
        [{ name: "training windows", color: "#0969da", values: part.train_windows }], { title: "Training windows per client", y: [0, Math.max(...part.train_windows) * 1.1] })),
      el("div", { class: "panel" }, barChart([...Array(k).keys()].map(i => `client ${i}`),
        [{ name: "users", color: "#8250df", values: part.train_users }], { title: "Training users per client", y: [0, Math.max(...part.train_users) * 1.15] }))));
  }

  // learning curves
  const unit = exp === "central" ? "epoch" : "round";
  const curves = [["val_f1", "Validation macro-F1"], ["test_f1", "Test macro-F1"], ["train_loss", "Training loss"], ["val_loss", "Validation loss"]];
  app.append(el("h2", {}, `Learning curves (mean ± std per ${unit})`), el("p", { class: "note", html:
    `The reported model is the one from the ${unit} with the best <b>validation F1</b>, not the last one. If the validation loss rises while F1 stays flat, the model becomes overconfident without becoming more accurate.` }),
    el("div", { class: "grid" }, curves.map(([k, label]) => el("div", { class: "panel" },
      lineChart(shown.map(g => meanCurve(g, k)), { title: label, xlabel: unit, y01: k.endsWith("f1"),
        hlines: k === "test_f1" ? refCentral.map(c => ({ y: groupStat(c, "test_f1").mean, color: colorOf(c.method), label: methodLabel(c.method) })) : [] })))));

  // per class + confusion
  const cmMethod = groups.some(g => g.method === params.get("cm")) ? params.get("cm") : shown[0]?.method;
  const pcSeries = rows.map(g => ({ name: methodLabel(g.method) + (g.experiment !== exp ? " (ref.)" : ""), color: colorOf(g.method),
    values: CLASSES.map(c => stat(g.runs.map(r => r.per_class[c]?.f1))?.mean), err: CLASSES.map(c => stat(g.runs.map(r => r.per_class[c]?.f1))?.std) }));
  const cmGroup = groupOf(exp, cmMethod);
  app.append(el("h2", {}, "Per activity class"), el("div", { class: "grid" },
    el("div", { class: "panel wide" }, barChart(CLASSES, pcSeries, { title: "Test F1 per class (mean ± std)", W: 980, H: 280 })),
    cmGroup && el("div", { class: "panel" }, el("div", { class: "controls" }, el("label", {}, "Confusion matrix for:",
      select(groups.map(g => g.method), cmMethod, v => go("experiment", { e: exp, cm: v, ...(hidden.size ? { hide: [...hidden].join(",") } : {}) }), methodLabel))),
      confusion(sumCm(cmGroup.runs), { title: `${methodLabel(cmMethod)} – sum of the ${cmGroup.runs.length} runs (test)`, color: colorOf(cmMethod) })),
    refCentral[0] && el("div", { class: "panel" }, confusion(sumCm(refCentral[0].runs), { title: `Reference: ${methodLabel(refCentral[0].method)}`, color: colorOf(refCentral[0].method) }))));

  // per client + per run
  const grid = el("div", { class: "grid" });
  if (shown.some(g => g.runs.some(r => r.clients.length > 1))) {
    grid.append(el("div", { class: "panel" }, el("h3", {}, "Test F1 per client (all runs)"),
      el("p", { class: "note" }, "One dot = one client in one run. Line = mean. The more spread out the dots, the more unequally the clients are served."),
      strip(shown.map(g => ({ label: methodLabel(g.method).replace(" local epochs", "").replace(" rounds", ""), color: colorOf(g.method),
        values: g.runs.flatMap(r => r.clients.map(c => ({ y: c.f1, label: `fold ${r.fold} seed ${r.seed} · client ${c.id}<br>F1 ${f3(c.f1)}, ${c.users} test users` }))) })), { ylabel: "macro-F1" })));
  }
  const keys = [...new Set(groups.flatMap(g => g.runs.map(r => `${r.fold}-${r.seed}`)))].sort();
  const runSeries = [...shown, ...refCentral].map(g => { const by = Object.fromEntries(g.runs.map(r => [`${r.fold}-${r.seed}`, r.test.f1]));
    return { name: methodLabel(g.method), color: colorOf(g.method), dash: g.experiment !== exp, x: keys.map((_, i) => i + 1), y: keys.map(k => by[k]) }; });
  grid.append(el("div", { class: "panel" }, el("h3", {}, "Test F1 of every run (fold × seed)"),
    el("p", { class: "note" }, "Shows how much the result varies between folds, i.e. between different groups of test users. Dashed lines are the centralised reference."),
    lineChart(runSeries, { xlabel: "run", xfmt: i => keys[i - 1] ? "f" + keys[i - 1].replace("-", "s") : "" })));
  app.append(el("h2", {}, "Clients and individual runs"), grid);
  app.append(el("p", {}, el("a", { href: "#runs?" + new URLSearchParams({ e: exp }) }, "→ see all runs of this experiment")));
}

function pageRuns(app, params) {
  const f = { e: params.get("e") || "", m: params.get("m") || "", fold: params.get("fold") || "", seed: params.get("seed") || "" };
  const sel = params.get("run") != null ? RUNS[Number(params.get("run"))] : null;
  const setF = (k, v) => go("runs", Object.fromEntries(Object.entries({ ...f, [k]: v, ...(sel ? { run: sel.idx } : {}) }).filter(([, x]) => x !== "")));
  const opt = vals => ["", ...vals];
  const runs = RUNS.filter(r => (!f.e || r.experiment === f.e) && (!f.m || r.method === f.m) && (!f.fold || String(r.fold) === f.fold) && (!f.seed || String(r.seed) === f.seed));
  app.append(el("h1", {}, "All runs"), el("div", { class: "controls" },
    el("label", {}, "Experiment", select(opt(EXPERIMENTS), f.e, v => setF("e", v), v => v ? expLabel(v) : "all")),
    el("label", {}, "Method", select(opt(METHODS), f.m, v => setF("m", v), v => v ? methodLabel(v) : "all")),
    el("label", {}, "Fold", select(opt([0, 1, 2, 3, 4].map(String)), f.fold, v => setF("fold", v), v => v || "all")),
    el("label", {}, "Seed", select(opt(["1", "2", "3"]), f.seed, v => setF("seed", v), v => v || "all")),
    el("span", { class: "muted" }, `${runs.length} runs · click a row for details`)));
  const detail = el("div", { id: "detail" });
  if (sel) { detail.append(runDetail(sel)); }
  app.append(detail);
  app.append(table([
    { label: "experiment", get: r => expLabel(r.experiment), cls: "l" },
    { label: "method", get: r => methodLabel(r.method), cls: "l", html: r => `<span class="sw" style="background:${colorOf(r.method)}"></span> ${esc(methodLabel(r.method))}` },
    { label: "fold", get: r => r.fold }, { label: "seed", get: r => r.seed },
    { label: "test F1", get: r => r.test.f1, fmt: f3 }, { label: "val F1", get: r => r.val.f1, fmt: f3 },
    { label: "Δ vs central", get: r => { const c = r.kind !== "central" && CENTRAL[r.model]?.[`${r.fold}-${r.seed}`]; return c ? r.test.f1 - c.test.f1 : null; },
      html: r => { const c = r.kind !== "central" && CENTRAL[r.model]?.[`${r.fold}-${r.seed}`]; if (!c) return "–"; const d = r.test.f1 - c.test.f1; return `<span class="${d >= 0 ? "pos" : "neg"}">${sgn(d)}</span>`; } },
    { label: "bal. acc.", get: r => r.test.bacc, fmt: f3 }, { label: "acc.", get: r => r.test.acc, fmt: f3 },
    { label: "worst client", get: r => r.worst_client, fmt: f3 },
    { label: "best / total", get: r => r.best, html: r => `${r.best ?? "–"} / ${r.length ?? "–"}` },
    { label: "seconds", get: r => r.seconds, fmt: f1d },
  ], runs, { sortBy: 4, onClick: r => go("runs", Object.fromEntries(Object.entries({ ...f, run: r.idx }).filter(([, x]) => x !== ""))), selected: r => r === sel }));
  if (sel) setTimeout(() => detail.scrollIntoView({ behavior: "smooth", block: "start" }), 0);
}

function runDetail(r) {
  const unit = r.kind === "central" ? "epoch" : "round";
  const c = r.kind !== "central" && CENTRAL[r.model]?.[`${r.fold}-${r.seed}`];
  const box = el("div", { class: "panel", style: "margin:10px 0 18px" });
  box.append(el("h2", { style: "margin-top:4px" }, `${methodLabel(r.method)} · ${expLabel(r.experiment)} · fold ${r.fold} · seed ${r.seed}`),
    el("div", { class: "note" }, el("code", {}, "outputs/" + r.path)),
    el("div", { class: "cards" },
      card("test macro-F1", f3(r.test.f1), c ? `centralised: ${f3(c.test.f1)} (<span class="${r.test.f1 >= c.test.f1 ? "pos" : "neg"}">${sgn(r.test.f1 - c.test.f1)}</span>)` : ""),
      card("validation macro-F1", f3(r.val.f1)), card("balanced acc.", f3(r.test.bacc)), card("accuracy", f3(r.test.acc)),
      card(`selected ${unit}`, `${r.best ?? "–"} <span class="muted" style="font-size:13px">/ ${r.length ?? "–"}</span>`, r.stop ? `early stop at ${r.stop}` : ""),
      r.worst_client != null && card("worst client", f3(r.worst_client), `std across clients ${f3(r.client_std)}`),
      card("test users", r.users.n ?? "–", `median acc. ${f3(r.users.median)} · ${pct(r.users.below50)} below 50 %`),
      card("time", f1d(r.seconds) + " s", r.comm_mb_to_best ? `${f0(r.comm_mb_to_best)} MB sent up to best` : "")));
  const h = r.history, x = h.x || [], vl = [{ x: r.best, color: "#9a6700" }];
  const line = (k, name, color) => h[k] ? { name, color, x, y: h[k] } : null;
  const grid = el("div", { class: "grid" });
  grid.append(
    el("div", { class: "panel" }, lineChart([line("train_acc", "train acc", "#8250df"), line("val_f1", "val F1", "#1a7f37"), line("test_f1", "test F1", "#cf222e")].filter(Boolean),
      { title: `Macro-F1 per ${unit} (vertical line = selected ${unit})`, xlabel: unit, vlines: vl, y01: true })),
    el("div", { class: "panel" }, lineChart([line("train_loss", "train", "#8250df"), line("val_loss", "validation", "#1a7f37"), line("test_loss", "test", "#cf222e")].filter(Boolean),
      { title: "Loss", xlabel: unit, vlines: vl })),
    r.cm && el("div", { class: "panel" }, confusion(r.cm, { title: "Confusion matrix (test)", color: colorOf(r.method) })),
    el("div", { class: "panel" }, barChart(CLASSES, ["precision", "recall", "f1"].map((k, i) => ({ name: k, color: ["#0969da", "#bf8700", "#1a7f37"][i],
      values: CLASSES.map(cl => r.per_class[cl]?.[k]) })), { title: "Per class (test)" })));
  if (r.clients.length > 1) grid.append(el("div", { class: "panel" }, barChart(r.clients.map(cl => `client ${cl.id}`),
    [{ name: "test F1", color: colorOf(r.method), values: r.clients.map(cl => cl.f1) }, { name: "val F1", color: "#8c959f", values: r.clients.map(cl => cl.val_f1) }],
    { title: "F1 per client" }), el("div", { class: "note" }, "Test users per client: " + r.clients.map(cl => cl.users).join(", "))));
  if (r.users.values) grid.append(el("div", { class: "panel" }, histogram(r.users.values, { title: "Accuracy per test user", xlabel: "accuracy", color: colorOf(r.method) })));
  box.append(grid, el("details", {}, el("summary", {}, "configuration"), el("pre", {}, JSON.stringify({ config: r.config, partition: r.partition, params: r.params, fedbn_distinct_bn: r.fedbn_distinct }, null, 1))));
  return box;
}

function pageTests(app, params) {
  const e = params.get("e") || "";
  const tests = (D.tests || []).filter(t => !e || t.experiment === e);
  const KIND = { vs_central: "vs. centralised", vs_fedavg: "vs. FedAvg", vs_fedavg_worst: "worst client vs. FedAvg", fcn_vs_lstm: "FCN vs. LSTM" };
  app.append(el("h1", {}, "Statistical tests"), el("p", { class: "note", html:
    `Taken from the summary files found in outputs/ (${[...new Set((D.tests || []).map(t => t.source))].map(esc).join(", ") || "none"}).
     Each test compares two methods on the same 15 fold × seed pairs (Wilcoxon). <b>Δ</b> = mean difference, <b>wins</b> = in how many pairs the first method is better.
     <span class="badge sig">green</span> = significant in favour of the first method, <span class="badge sigbad">red</span> = significantly against it (p &lt; 0.05 after Holm correction).` }),
    el("div", { class: "controls" }, el("label", {}, "Experiment", select(["", ...new Set((D.tests || []).map(t => t.experiment))].sort(cmpExp), e,
      v => go("tests", v ? { e: v } : {}), v => v ? expLabel(v) : "all"))));
  app.append(table([
    { label: "experiment", get: t => expLabel(t.experiment), cls: "l" },
    { label: "comparison", get: t => t.label || t.method, cls: "l" },
    { label: "type", get: t => KIND[t.kind] || t.kind, cls: "l" },
    { label: "mean Δ", get: t => t.mean_diff, html: t => `<span class="${t.mean_diff >= 0 ? "pos" : "neg"}">${sgn(t.mean_diff)}</span>` },
    { label: "wins", get: t => t.wins, html: t => `${t.wins ?? "–"}/${t.pairs ?? "–"}` },
    { label: "p", get: t => t.p, fmt: pfmt },
    { label: "p Holm", get: t => t.p_holm ?? t.p, html: t => sigBadge(t) },
  ], tests, { sortBy: 6, desc: false }));
}

// ------------------------------------------------------------------ router
const PAGES = { overview: pageOverview, compare: pageCompare, experiment: pageExperiment, runs: pageRuns, tests: pageTests };
function route() {
  const [page, q] = location.hash.slice(1).split("?");
  const name = PAGES[page] ? page : "overview";
  document.querySelectorAll("#nav a").forEach(a => a.classList.toggle("active", a.getAttribute("href") === "#" + name));
  const app = document.getElementById("app");
  app.replaceChildren();
  tipEl.style.display = "none";
  PAGES[name](app, new URLSearchParams(q || ""));
  if (name !== "runs") window.scrollTo(0, 0);
}
if (D) {
  document.getElementById("meta").textContent = `generated ${D.generated.replace("T", " ")} · ${D.runs.length} records`;
  window.addEventListener("hashchange", route);
  route();
}
