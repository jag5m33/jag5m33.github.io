/* =====================================================================
   Jag Sandhu — portfolio
   orb → spider-web map → full-screen chapters and project pages

   Content lives in two places, so you rarely touch this file:
     data.json        the map: notes (left half), projects + models (right half)
     notes/<id>.md    one chapter per topic, written in Markdown

   The map is a radial tree. Everything is always open:
     - notes fan out across the LEFT half, numbered in reading order
     - projects and the models inside them fan out across the RIGHT half
     - dashed threads join a model or project to the notes it uses
       (they light up when you hover)
   Scroll to zoom, drag to move, click a gold branch to zoom into it.
   ===================================================================== */

const SVGNS = "http://www.w3.org/2000/svg";
const intro  = document.getElementById("intro");
const map    = document.getElementById("map");
const reader = document.getElementById("reader");
const paper  = document.getElementById("paper");
const crumbs = document.getElementById("crumbs");

// ---------------- layout settings (map units) ----------------
const LEFT_SPAN  = 70;                    // notes fan ±70° around due left
const RIGHT_SPAN = 62;                    // projects fan ±62° around due right
const RING       = [0, 215, 360, 490];    // radius of branch levels 1, 2, 3
const LEAF_R     = 640;                   // every note sits on this outer ring
const PROJ_R     = 300, MODEL_R = 580;    // right half: projects, then their models
const SQUASH     = 0.6;                   // < 1 flattens the web to suit wide screens
const GAP_TOP    = 1.8, GAP_SUB = 1.3;    // extra space between groups (in leaf slots)
// --------------------------------------------------------------

let data;
const byId = {};               // every node, by id
const order = [];              // notes in reading order (the numbered sequence)
const view = {};               // id -> { el, x, y, o } what's on screen now
const edgeEls = {};            // "from>to" -> path element
let vb = { x: -800, y: -500, w: 1600, h: 1000 };
let home = null;               // the view box that fits the whole web
let mapOpen = false;

// ---------------------------------------------------------------------
// LOAD
// ---------------------------------------------------------------------
const ready = fetch("data.json").then(r => r.json()).then(d => {
  data = d;
  byId.centre = { id: "centre", label: "Jag Sandhu", type: "centre", depth: 0 };

  (function index(list, parent, depth) {
    list.forEach(n => {
      n.parent = parent;
      n.depth = depth;
      n.type = n.children ? "branch" : "leaf";
      n.siblings = list;
      byId[n.id] = n;
      if (n.children) index(n.children, n.id, depth + 1);
      else { order.push(n); n.num = order.length; }
    });
  })(data.tree, "centre", 1);

  data.projects.forEach(p => {
    p.type = "project"; p.parent = "centre"; p.depth = 1;
    byId[p.id] = p;
    (p.models || []).forEach(m => { m.type = "model"; m.parent = p.id; m.depth = 2; m.project = p; byId[m.id] = m; });
  });
  return d;
});

// ---------------------------------------------------------------------
// ORB → MAP
// ---------------------------------------------------------------------
async function enter() {
  if (mapOpen) return;
  await ready;
  // the orb bursts, then shrinks into the map's centre node as the map opens
  Orb.burst(centreOnScreen, () => openMap(true));
}
intro.addEventListener("click", enter);
intro.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); enter(); } });

// where the centre node is on screen right now (it moves while the map settles)
function centreOnScreen() {
  const v = view.centre;
  if (!v || !v.el.isConnected) return null;
  const b = v.el.querySelector(".hit").getBoundingClientRect();
  return { x: b.left + b.width / 2, y: b.top + b.height / 2, r: b.width / 2 * 0.8 };
}

const phone = () => window.innerWidth < 700;

function openMap(animated) {
  mapOpen = true;
  document.body.classList.add("map-open");
  if (phone()) { buildIndex(); document.body.classList.add("list-mode"); return; }
  document.body.classList.remove("list-mode");
  build(animated);
}

// phone view: the web as a numbered list, same order and same links
function buildIndex() {
  const box = document.getElementById("index");
  if (box.dataset.built) return;
  const list = items => items.map(n => n.children
    ? `<li class="group"><h3>${n.label}</h3><ol>${list(n.children)}</ol></li>`
    : `<li class="${n.soon ? "soon" : ""}"><a href="#/t/${n.id}"><span class="num">${n.num}</span>${n.label}${n.soon ? " <em>soon</em>" : ""}</a></li>`).join("");
  box.innerHTML = `
    <header><button type="button" id="index-back">← Sphere</button><h1>Jag Sandhu</h1></header>
    <section><h2>Notes</h2>${data.tree.map(t => `<h3 class="top">${t.label}</h3><ol>${list(t.children)}</ol>`).join("")}</section>
    <section><h2>Projects</h2>${data.projects.map(p => `
      <a class="proj" href="#/p/${p.id}"><strong>${p.label}</strong>
        <span>${(p.models || []).map(m => m.label + (m.status ? ` (${m.status})` : "")).join(" · ")}</span></a>`).join("")}</section>`;
  box.dataset.built = "1";
  document.getElementById("index-back").onclick = closeMap;
}

function closeMap() {
  if (document.body.classList.contains("list-mode")) {
    document.body.classList.remove("map-open", "list-mode");
    Orb.restore(null);
    mapOpen = false;
    return;
  }
  flyTo(home, 400);
  setTimeout(() => {
    const from = centreOnScreen();
    document.body.classList.remove("map-open");
    Orb.restore(from);                                  // the orb grows back out of the centre node
    setTimeout(() => {
      map.innerHTML = "";
      layers = null;
      for (const k in view) delete view[k];
      for (const k in edgeEls) delete edgeEls[k];
      mapOpen = false;
    }, 700);
  }, 420);
}

// ---------------------------------------------------------------------
// LAYOUT: angles and radii for every node
// ---------------------------------------------------------------------
const rad = d => d * Math.PI / 180;
// polar → screen. SQUASH flattens the circle into an ellipse.
const P = (deg, r) => ({ x: r * Math.cos(rad(deg)), y: r * Math.sin(rad(deg)) * SQUASH, a: deg, r });

function layout() {
  const pos = { centre: { x: 0, y: 0, a: 180, r: 0 } };

  // ---- left half: one slot per note, with gaps between groups ----
  let slots = 0;
  (function count(list, depth) {
    list.forEach((n, i) => {
      if (i > 0) slots += depth === 1 ? GAP_TOP : (n.children || list[i - 1].children ? GAP_SUB : 0);
      if (n.children) count(n.children, depth + 1); else slots += 1;
    });
  })(data.tree, 1);
  slots -= 1;

  // top of the half-circle first, so reading order runs top → bottom
  const start = 180 + LEFT_SPAN, step = (2 * LEFT_SPAN) / Math.max(slots, 1);
  let s = 0;
  (function place(list, depth) {
    list.forEach((n, i) => {
      if (i > 0) s += depth === 1 ? GAP_TOP : (n.children || list[i - 1].children ? GAP_SUB : 0);
      if (n.children) {
        place(n.children, depth + 1);
        const kids = n.children.map(c => pos[c.id].a);
        const a = (Math.min(...kids) + Math.max(...kids)) / 2;
        pos[n.id] = P(a, RING[depth]);
      } else {
        pos[n.id] = P(start - s * step, LEAF_R);
        s += 1;
      }
    });
  })(data.tree, 1);

  // ---- right half: projects, each with its models ----
  const projs = data.projects;
  const nModels = projs.reduce((k, p) => k + Math.max(1, (p.models || []).length), 0) + (projs.length - 1) * 1.2;
  const rstep = (2 * RIGHT_SPAN) / Math.max(nModels - 1, 1);
  let t = 0;
  projs.forEach((p, i) => {
    if (i > 0) t += 1.2;
    const ms = p.models || [];
    const first = t;
    ms.forEach(m => { pos[m.id] = P(-RIGHT_SPAN + t * rstep, MODEL_R); t += 1; });
    if (!ms.length) t += 1;
    const a = -RIGHT_SPAN + ((first + Math.max(t - 1, first)) / 2) * rstep;
    pos[p.id] = P(a, PROJ_R);
  });

  return pos;
}

// tree edges, plus dashed threads from projects/models to the notes they use
function edgeList() {
  const out = [];
  Object.values(byId).forEach(n => { if (n.id !== "centre") out.push({ from: n.parent, to: n.id, kind: "tree" }); });
  data.projects.forEach(p => {
    (p.uses || []).forEach(u => byId[u] && out.push({ from: u, to: p.id, kind: "cross" }));
    (p.models || []).forEach(m => m.topic && byId[m.topic] && out.push({ from: m.topic, to: m.id, kind: "cross model" }));
  });
  return out;
}

// ---------------------------------------------------------------------
// DRAWING
// ---------------------------------------------------------------------
function el(tag, attrs = {}) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}

let layers = null;

function makeLayers() {
  layers = { web: el("g", { class: "web" }), edges: el("g"), nodes: el("g") };
  map.append(layers.web, layers.edges, layers.nodes);

  // the "spider web": faint rings at each level, and a few spokes
  const rings = [...RING.slice(1), LEAF_R];
  rings.forEach(r => {
    const leftArc = arc(180 - LEFT_SPAN - 4, 180 + LEFT_SPAN + 4, r);
    layers.web.append(el("path", { d: leftArc, class: "ring" }));
  });
  [PROJ_R, MODEL_R].forEach(r => layers.web.append(el("path", { d: arc(-RIGHT_SPAN - 6, RIGHT_SPAN + 6, r), class: "ring right" })));
  for (let a = 180 - LEFT_SPAN; a <= 180 + LEFT_SPAN + 0.1; a += LEFT_SPAN / 4) {
    const p = P(a, LEAF_R + 10);
    layers.web.append(el("line", { x1: 0, y1: 0, x2: p.x, y2: p.y, class: "spoke" }));
  }
  for (let a = -RIGHT_SPAN; a <= RIGHT_SPAN + 0.1; a += RIGHT_SPAN / 2) {
    const p = P(a, MODEL_R + 10);
    layers.web.append(el("line", { x1: 0, y1: 0, x2: p.x, y2: p.y, class: "spoke right" }));
  }
}

function arc(a0, a1, r) {
  let d = "";
  for (let a = a0; a <= a1 + 0.01; a += 2) {
    const p = P(a, r);
    d += (d ? "L" : "M") + p.x.toFixed(1) + "," + p.y.toFixed(1);
  }
  return d;
}

function makeNode(id, p) {
  const n = byId[id];
  const side = n.type === "project" || n.type === "model" ? "right" : n.type === "centre" ? "" : "left";
  const cls = ["node", n.type, side, "d" + n.depth];
  if (n.soon) cls.push("soon");
  if (n.status) cls.push("pending");
  const g = el("g", { class: cls.join(" ") });

  const r = { centre: 42, branch: n.depth === 1 ? 12 : 9, leaf: 6.5, project: 12, model: 7.5 }[n.type];
  if (n.type === "centre") {
    g.append(el("circle", { r, class: "hit" }));       // dark disc behind, and the click target
    const dots = el("g");
    g.append(dots);
    Orb.mini(dots, 34);                                // small spinning sphere of dots
  } else {
    g.append(el("circle", { r }));
  }

  // labels: notes and models radiate outwards along their spoke;
  // branches and projects sit level, just outside their dot
  const t = el("text");
  if (n.type === "centre") {
    t.setAttribute("y", 68); t.setAttribute("text-anchor", "middle");
    t.textContent = n.label;
  } else if (n.type === "leaf" || n.type === "model") {
    const ang = Math.atan2(p.y, p.x) * 180 / Math.PI;     // the label follows the spoke on screen
    const flip = Math.cos(rad(ang)) < 0;                  // left half: turn text the right way up
    t.setAttribute("transform", `rotate(${flip ? ang + 180 : ang})`);
    t.setAttribute("x", flip ? -(r + 8) : r + 8);
    t.setAttribute("y", 4.5);
    t.setAttribute("text-anchor", flip ? "end" : "start");
    if (n.type === "leaf") {
      const num = el("tspan", { class: "num" }); num.textContent = n.num + "  ";
      const lab = el("tspan"); lab.textContent = n.label + (n.soon ? " · soon" : "");
      // number sits next to the dot: before the label on the right, after it on the left
      if (flip) { lab.textContent += "  "; t.append(lab); num.textContent = String(n.num); t.append(num); }
      else { t.append(num, lab); }
    } else {
      t.textContent = n.label + (n.status ? ` · ${n.status}` : "");
    }
  } else {
    // branches and projects: level text placed away from the centre
    const outward = p.x <= 0 ? -1 : 1;
    t.setAttribute("x", outward * (r + 8));
    t.setAttribute("y", p.y > 20 ? r + 17 : -r - 6);      // lower half: label under the dot
    t.setAttribute("text-anchor", outward < 0 ? "end" : "start");
    t.textContent = n.label;
  }
  g.append(t);

  g.addEventListener("click", e => { e.stopPropagation(); if (!dragged) onNodeClick(id); });
  g.addEventListener("mouseenter", () => focus(id, true));
  g.addEventListener("mouseleave", () => focus(id, false));
  layers.nodes.append(g);
  return g;
}

// curved edge that follows the web: d3-style radial link through the mid-ring
function curve(a, b, kind) {
  if (kind.startsWith("cross")) {
    // threads between the halves bow through the space above/below the centre
    const my = (a.y + b.y) / 2;
    return `M${a.x},${a.y} Q${(a.x + b.x) / 2 * 0.25},${my * 1.25} ${b.x},${b.y}`;
  }
  const pa = polar(a), pb = polar(b);
  const angA = pa.r < 1 ? pb.a : pa.a;
  const m = (pa.r + pb.r) / 2;
  const c1 = P(angA, m), c2 = P(pb.a, m);
  return `M${a.x},${a.y} C${c1.x},${c1.y} ${c2.x},${c2.y} ${b.x},${b.y}`;
}
const polar = p => ({ a: Math.atan2(p.y / SQUASH, p.x) * 180 / Math.PI, r: Math.hypot(p.x, p.y / SQUASH) });

// ---------------------------------------------------------------------
// BUILD + OPENING ANIMATION: everything grows out of the centre
// ---------------------------------------------------------------------
let anim = null;

function build(animated) {
  if (!layers) makeLayers();
  const pos = layout();
  const ids = Object.keys(byId);
  const D = animated ? 1300 : 0;

  // nodes start at the centre and fly out, inner rings first
  ids.forEach(id => {
    if (view[id]) return;
    const p = pos[id];
    const delay = animated ? (byId[id].depth || 0) * 160 + (p.r / LEAF_R) * 120 : 0;
    view[id] = { el: makeNode(id, p), x: 0, y: 0, o: 0, tx: p.x, ty: p.y, delay };
  });

  edgeList().forEach(e => {
    const key = e.from + ">" + e.to;
    if (edgeEls[key]) return;
    const path = el("path", { class: "edge " + e.kind });
    path.dataset.from = e.from; path.dataset.to = e.to;
    layers.edges.append(path);
    edgeEls[key] = path;
  });

  // fit the whole web, including where each angled label ends, to the screen
  const xs = [], ys = [];
  ids.forEach(id => {
    const p = pos[id], n = byId[id];
    xs.push(p.x); ys.push(p.y);
    if (n.type === "leaf" || n.type === "model") {
      const len = 20 + (n.label.length + (n.soon ? 7 : 0) + (n.status ? n.status.length + 3 : 0) + 4) * 7.3;
      const d = Math.hypot(p.x, p.y) || 1;
      xs.push(p.x + p.x / d * len); ys.push(p.y + p.y / d * len);
    }
  });
  const pad = 30;
  home = {
    x: Math.min(...xs) - pad, y: Math.min(...ys) - pad,
    w: Math.max(...xs) - Math.min(...xs) + pad * 2, h: Math.max(...ys) - Math.min(...ys) + pad * 2 + 30
  };
  vb = { ...home };
  setVB();

  const t0 = performance.now();
  const ease = t => 1 - Math.pow(1 - t, 3);
  const maxDelay = Math.max(...Object.values(view).map(v => v.delay));
  if (anim) cancelAnimationFrame(anim);

  (function frame(now) {
    const el_ = now - t0;
    for (const id in view) {
      const v = view[id];
      const k = D ? ease(Math.min(1, Math.max(0, (el_ - v.delay) / D))) : 1;
      v.x = v.tx * k; v.y = v.ty * k; v.o = k;
      v.el.setAttribute("transform", `translate(${v.x},${v.y})`);
      v.el.style.opacity = v.o;
    }
    drawEdges();
    const done = !D || el_ > D + maxDelay;
    map.classList.toggle("settled", done);
    if (!done) anim = requestAnimationFrame(frame);
    else anim = null;
  })(t0);
}

function drawEdges() {
  for (const key in edgeEls) {
    const p = edgeEls[key], a = view[p.dataset.from], b = view[p.dataset.to];
    if (!a || !b) continue;
    p.setAttribute("d", curve(a, b, p.getAttribute("class").includes("cross") ? "cross" : "tree"));
    p.style.opacity = Math.min(a.o, b.o);
  }
}

// ---------------------------------------------------------------------
// HOVER: light up a node's path to the centre and its threads
// ---------------------------------------------------------------------
function focus(id, on) {
  map.querySelectorAll(".lit, .hot").forEach(x => x.classList.remove("lit", "hot"));
  map.classList.toggle("focus", on && id !== "centre");
  if (!on || id === "centre") return;

  // the hovered node and everything below it...
  const below = new Set([id]);
  (function down(x) { (x.children || x.models || []).forEach(c => { below.add(c.id); down(c); }); })(byId[id]);
  // ...plus its path back to the centre
  const lit = new Set(below);
  let n = byId[id];
  while (n && n.parent) { lit.add(n.parent); n = byId[n.parent]; }

  Object.values(edgeEls).forEach(p => {
    const f = p.dataset.from, t = p.dataset.to;
    if (p.getAttribute("class").includes("cross")) {
      // threads across the halves that start or end at the hovered part
      if (below.has(f) || below.has(t)) { p.classList.add("hot"); lit.add(f); lit.add(t); }
    } else if (lit.has(f) && lit.has(t)) {
      p.classList.add("hot");
    }
  });
  lit.forEach(x => view[x] && view[x].el.classList.add("lit"));
}

// ---------------------------------------------------------------------
// ZOOM AND PAN: wheel to zoom, drag to move, buttons in the corner
// ---------------------------------------------------------------------
function setVB() { map.setAttribute("viewBox", `${vb.x} ${vb.y} ${vb.w} ${vb.h}`); }

function toMap(cx, cy) {                    // screen pixel → map units
  const pt = map.createSVGPoint(); pt.x = cx; pt.y = cy;
  return pt.matrixTransform(map.getScreenCTM().inverse());
}

function zoomAt(cx, cy, factor) {
  const p = toMap(cx, cy);
  const w = Math.min(home.w * 1.5, Math.max(home.w / 6, vb.w * factor));
  const k = w / vb.w;
  vb = { x: p.x - (p.x - vb.x) * k, y: p.y - (p.y - vb.y) * k, w: vb.w * k, h: vb.h * k };
  setVB();
}

let flight = null;
function flyTo(box, ms = 600) {
  if (!box) return;
  const from = { ...vb }, t0 = performance.now();
  const ease = t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  if (flight) cancelAnimationFrame(flight);
  (function f(now) {
    const k = ease(Math.min(1, (now - t0) / ms));
    for (const key of ["x", "y", "w", "h"]) vb[key] = from[key] + (box[key] - from[key]) * k;
    setVB();
    if (k < 1) flight = requestAnimationFrame(f);
  })(t0);
}

map.addEventListener("wheel", e => {
  if (!mapOpen) return;
  e.preventDefault();
  zoomAt(e.clientX, e.clientY, Math.exp(e.deltaY * 0.0015));
}, { passive: false });

let drag = null, dragged = false;
map.addEventListener("pointerdown", e => {
  if (!mapOpen) return;
  drag = { x: e.clientX, y: e.clientY, vb: { ...vb } };
  dragged = false;
});
window.addEventListener("pointermove", e => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  if (Math.hypot(dx, dy) > 4) { dragged = true; map.classList.add("dragging"); }
  if (!dragged) return;
  const ctm = map.getScreenCTM();
  vb = { ...drag.vb, x: drag.vb.x - dx / ctm.a, y: drag.vb.y - dy / ctm.d };
  setVB();
});
window.addEventListener("pointerup", () => {
  drag = null;
  map.classList.remove("dragging");
  setTimeout(() => dragged = false, 0);
});

document.getElementById("zoom-in").onclick  = () => zoomAt(innerWidth / 2, innerHeight / 2, 0.75);
document.getElementById("zoom-out").onclick = () => zoomAt(innerWidth / 2, innerHeight / 2, 1.33);
document.getElementById("zoom-fit").onclick = () => flyTo(home);

// ---------------------------------------------------------------------
// CLICKS
// ---------------------------------------------------------------------
function onNodeClick(id) {
  const n = byId[id];
  if (n.type === "centre") return closeMap();
  if (n.type === "project") return go(`#/p/${id}`);
  if (n.type === "model") return go(`#/p/${n.project.id}`);
  if (n.type === "leaf") return go(`#/t/${id}`);
  zoomToBranch(id);
}

// zoom so a branch and everything under it fills the screen
function zoomToBranch(id) {
  const ids = [id];
  (function down(x) { (x.children || []).forEach(c => { ids.push(c.id); down(c); }); })(byId[id]);
  const xs = ids.map(i => view[i].tx), ys = ids.map(i => view[i].ty);
  const pad = 260;
  let box = { x: Math.min(...xs, 0) - pad, y: Math.min(...ys) - 80, w: 0, h: 0 };
  box.w = Math.max(...xs, 0) + 120 - box.x;
  box.h = Math.max(...ys) + 80 - box.y;
  // keep the screen's shape
  const aspect = innerWidth / innerHeight;
  if (box.w / box.h < aspect) { const w = box.h * aspect; box.x -= (w - box.w) / 2; box.w = w; }
  else { const h = box.w / aspect; box.y -= (h - box.h) / 2; box.h = h; }
  flyTo(box);
}


// ---------------------------------------------------------------------
// ROUTER: #/t/<topic>  #/p/<project>  (empty = map)
// so the browser back button works and every page has its own link
// ---------------------------------------------------------------------
function go(hash) { if (location.hash === hash) route(); else location.hash = hash; }

async function route() {
  await ready;
  const [, kind, id] = location.hash.match(/^#\/(t|p)\/(.+)$/) || [];

  if (!kind || !byId[id]) {
    reader.classList.remove("open");
    reader.setAttribute("aria-hidden", "true");
    return;
  }

  if (!mapOpen) {                     // arrived straight from a link: skip the intro
    Orb.hideNow();
    openMap(false);
  }

  reader.classList.add("open");
  reader.setAttribute("aria-hidden", "false");
  document.getElementById("reader-scroll").scrollTop = 0;

  if (kind === "t") await showTopic(id);
  else showProject(id);
}

window.addEventListener("hashchange", route);
document.getElementById("back").onclick = () => go("#/");
document.addEventListener("keydown", e => { if (e.key === "Escape" && reader.classList.contains("open")) go("#/"); });
if (location.hash.length > 2) route();

function trail(id) {
  const out = [];
  let n = byId[id];
  while (n && n.id !== "centre") { out.unshift(n); n = byId[n.parent]; }
  return out;
}

// ---------------------------------------------------------------------
// CHAPTERS
// ---------------------------------------------------------------------
const noteCache = {};

async function showTopic(id) {
  const n = byId[id];
  crumbs.innerHTML = trail(id).map(t => `<span>${t.label}</span>`).join("<i>›</i>");
  paper.className = "paper";

  let body;
  if (n.soon) {
    body = `<h1>${n.label}</h1><div class="soon-box"><p>Coming soon.</p><p class="muted">This chapter gets written once I've studied the topic.</p></div>`;
  } else {
    paper.innerHTML = `<p class="muted">Loading…</p>`;
    try {
      if (!noteCache[id]) {
        const r = await fetch(`notes/${id}.md`);
        if (!r.ok) throw new Error(r.status);
        noteCache[id] = await r.text();
      }
      body = renderMarkdown(noteCache[id]);
    } catch (e) {
      body = `<h1>${n.label}</h1><p>Couldn't load <code>notes/${id}.md</code> (${e.message}).</p>`;
    }
  }

  // footer: projects that use this topic, and previous / next in the reading order
  const used = data.projects.filter(p => (p.uses || []).includes(id) || (p.models || []).some(m => m.topic === id));
  let foot = "";
  if (used.length) {
    foot += `<h3>Used in my projects</h3>` +
      used.map(p => `<a class="chip" href="#/p/${p.id}">${p.label} →</a>`).join("");
  }
  const path = order.filter(o => !o.soon || o.id === id), i = path.indexOf(n);
  const prev = path[i - 1], next = path[i + 1];
  foot += `<nav class="pager">
      ${prev ? `<a href="#/t/${prev.id}">← ${prev.num}. ${prev.label}</a>` : "<span></span>"}
      ${next ? `<a href="#/t/${next.id}">${next.num}. ${next.label} →</a>` : "<span></span>"}
    </nav>`;

  paper.innerHTML = body + `<footer class="chapter-foot">${foot}</footer>`;
  paper.querySelectorAll("pre code").forEach(c => window.hljs && hljs.highlightElement(c));
}

// Markdown → HTML, with $maths$ and $$display maths$$ rendered by KaTeX.
// Maths is lifted out first so Markdown doesn't mangle underscores and stars.
function renderMarkdown(src) {
  const maths = [];
  const stash = (tex, display) => {
    maths.push({ tex, display });
    return `@@MATH${maths.length - 1}@@`;
  };

  // leave code untouched: split on ``` blocks and `inline code`
  const parts = src.split(/(```[\s\S]*?```|`[^`\n]*`)/g);
  src = parts.map((part, i) => {
    if (i % 2 === 1) return part;
    return part
      .replace(/\\\$/g, "@@DOLLAR@@")
      .replace(/\$\$([\s\S]+?)\$\$/g, (_, t) => "\n\n" + stash(t, true) + "\n\n")
      .replace(/\$([^\$\n]+?)\$/g, (_, t) => stash(t, false));
  }).join("");

  let html = marked.parse(src);

  html = html.replace(/@@MATH(\d+)@@/g, (_, k) => {
    const m = maths[k];
    try { return katex.renderToString(m.tex, { displayMode: m.display, throwOnError: false }); }
    catch { return m.tex; }
  }).replace(/@@DOLLAR@@/g, "$");

  // an image on its own line becomes a figure with its alt text as the caption
  html = html.replace(/<p>\s*<img src="([^"]+)" alt="([^"]*)"[^>]*>\s*<\/p>/g,
    (_, s, a) => `<figure><img src="${s}" alt="${a}" loading="lazy">${a ? `<figcaption>${a}</figcaption>` : ""}</figure>`);

  // links out of the site open in a new tab
  html = html.replace(/<a href="(https?:[^"]+)"/g, '<a href="$1" target="_blank" rel="noopener"');
  return html;
}

// ---------------------------------------------------------------------
// PROJECT PAGES
// ---------------------------------------------------------------------
function showProject(id) {
  const p = byId[id];
  crumbs.innerHTML = `<span>Projects</span><i>›</i><span>${p.label}</span>`;
  paper.className = "paper project";
  paper.innerHTML = `
    <p class="kicker">Project</p>
    <h1>${p.label}</h1>
    <div class="tabs">
      <button data-tab="overview" class="active">Overview</button>
      <button data-tab="code">Code</button>
      <button data-tab="results">Results</button>
    </div>
    <div class="tab-body"></div>`;
  const body = paper.querySelector(".tab-body");
  const tabs = paper.querySelectorAll(".tabs button");
  tabs.forEach(b => b.onclick = () => {
    tabs.forEach(x => x.classList.toggle("active", x === b));
    renderTab(p, b.dataset.tab, body);
  });
  renderTab(p, "overview", body);
}

function renderTab(p, tab, body) {
  if (tab === "overview") {
    body.innerHTML = `<p class="lead">${p.summary || ""}</p>
      <h3>Methods used</h3>
      ${(p.uses || []).filter(u => byId[u]).map(u => `<a class="chip" href="#/t/${u}">${byId[u].label}</a>`).join("")}
      <p class="repo-link"><a href="https://github.com/${p.repo}" target="_blank" rel="noopener">Open the repository on GitHub ↗</a></p>`;
  }

  if (tab === "code") {
    body.innerHTML = `<div class="files">
        ${(p.code || []).map(f => `<button class="chip" data-file="${f}">${f.split("/").pop()}</button>`).join("")}
      </div>
      <div class="code-box"><p class="muted">Pick a file to view it.</p></div>`;
    const box = body.querySelector(".code-box");
    body.querySelectorAll("[data-file]").forEach(b => b.onclick = () => {
      body.querySelectorAll("[data-file]").forEach(x => x.classList.toggle("active", x === b));
      loadCode(p, b.dataset.file, box);
    });
  }

  if (tab === "results") {
    const rs = p.results || [], m = p.metrics;
    let html = "";
    if (m) {
      html += `<h3>${m.title || "Metrics"}</h3><table class="metrics">
        <tr>${m.columns.map(c => `<th>${c}</th>`).join("")}</tr>
        ${m.rows.map(r => `<tr>${r.map(v => `<td>${v}</td>`).join("")}</tr>`).join("")}
      </table>`;
    }
    html += rs.map(r => `<figure><img src="${r.img}" alt="${r.caption || ""}" loading="lazy"><figcaption>${r.caption || ""}</figcaption></figure>`).join("");
    body.innerHTML = html || `<p class="muted">Results coming soon.</p>`;
  }
}

async function loadCode(p, path, box) {
  const branch = p.branch || "main";
  const ghLink = `https://github.com/${p.repo}/blob/${branch}/${path}`;

  if (path.endsWith(".ipynb")) {      // GitHub shows notebooks with their outputs, so link out
    box.innerHTML = `<p>Notebooks display best on GitHub, with all their outputs.</p><a href="${ghLink}" target="_blank" rel="noopener">Open ${path.split("/").pop()} ↗</a>`;
    return;
  }

  box.innerHTML = `<p class="muted">Loading…</p>`;
  try {
    const res = await fetch(`https://raw.githubusercontent.com/${p.repo}/${branch}/${path}`);
    if (!res.ok) throw new Error(res.status);
    const code = await res.text();
    const lang = { py: "python", sql: "sql", r: "r", js: "javascript", txt: "plaintext" }[path.split(".").pop()] || "plaintext";
    box.innerHTML = `<a href="${ghLink}" target="_blank" rel="noopener">View on GitHub ↗</a><pre><code class="language-${lang}"></code></pre>`;
    const c = box.querySelector("code");
    c.textContent = code;
    if (window.hljs) hljs.highlightElement(c);
  } catch (e) {
    box.innerHTML = `<p>Couldn't load this file (${e.message}). Check the repo name, branch and path in data.json.</p>
      <a href="${ghLink}" target="_blank" rel="noopener">Try it on GitHub ↗</a>`;
  }
}
