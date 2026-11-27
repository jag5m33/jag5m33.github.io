/* =====================================================================
   Jag Sandhu — portfolio
   orb → spider-web map → full-screen chapters and project pages

   Content lives in two places, so you rarely touch this file:
     data.json        the map: notes (left half), projects + models (right half)
     notes/<id>.md    one chapter per topic, written in Markdown

   The map is a ring of MINI MIND MAPS around me (the sphere in the centre):
     - one mini map per subject (Statistics, Linear algebra, SQL, Databricks ...):
       the subject in the middle, its chapters around it, numbered in reading order
     - a Models map: projects in the middle ring, the models I built around them
     - an MLOps map, and an AI engineering map: deep learning so far, the rest "coming soon"
     - dashed threads join a model or project to the chapters it uses
       (they light up when you hover)
   Scroll to zoom, drag to move, click a subject to zoom into its mini map.
   ===================================================================== */

const SVGNS = "http://www.w3.org/2000/svg";
const intro  = document.getElementById("intro");
const map    = document.getElementById("map");
const reader = document.getElementById("reader");
const paper  = document.getElementById("paper");
const crumbs = document.getElementById("crumbs");

// ---------------- layout settings (map units) ----------------
const ONE_R      = 120;                   // smallest radius for a one-ring mini map
const SUB_R      = 105;                   // two-ring maps: the inner ring (groups / projects)
const TWO_R      = 225;                   // two-ring maps: the outer ring (chapters / models)
const SLOT       = 36;                    // space each point needs around its ring
const CHAR_W     = 7.6;                   // rough width of one label character
const RING_GAP   = 1.0;                   // breathing room between neighbouring mini maps
const TITLE_MIN  = 260;                   // the ring of titles is never smaller than this
const X_STRETCH  = 1.25;                  // wider than tall, to suit screens
// --------------------------------------------------------------

let data;
const byId = {};               // every node, by id
const order = [];              // notes in reading order (the numbered sequence)
const view = {};               // id -> { el, x, y, o } what's on screen now
const edgeEls = {};            // "from>to" -> path element
let vb = { x: -800, y: -500, w: 1600, h: 1000 };
let home = null;               // the view box that fits the whole web
let mapOpen = false;
let zoomedHub = null;          // the mini map currently open and zoomed into (null = only titles showing)
let clusters = [];             // the hub of each mini map, in order around the ring

const kids = n => n.children || n.models || n.tools || n.projects || [];
function hubOf(id) { let n = byId[id]; while (n && n.parent && n.parent !== "centre") n = byId[n.parent]; return n; }
// only the titles show until a mini map is opened; then that map's points show too
const shown = id => { const n = byId[id]; return !n.parent || n.depth === 1 || hubOf(id).id === zoomedHub; };
const MOVE_MS = 700;                                    // how long opening / closing a mini map takes

// ---------------------------------------------------------------------
// LOAD
// ---------------------------------------------------------------------
const ready = fetch("data.json", { cache: "no-cache" }).then(r => r.json()).then(d => {
  data = d;
  byId.centre = { id: "centre", label: "Jag Sandhu", type: "centre", depth: 0 };

  function index(list, parent, depth) {
    list.forEach(n => {
      n.parent = parent;
      n.depth = depth;
      n.type = n.children ? "branch" : "leaf";
      n.siblings = list;
      byId[n.id] = n;
      if (n.children) index(n.children, n.id, depth + 1);
      else { order.push(n); n.num = order.length - (order[0].zero ? 1 : 0); }   // a "zero" first page is page 0
    });
  }
  index(data.tree, "centre", 1);

  // the Models mini map: projects round a "Models" hub, each project's models round it
  byId.models = { id: "models", label: "Models", type: "hub", parent: "centre", depth: 1, projects: data.projects };
  data.projects.forEach(p => {
    p.type = "project"; p.parent = "models"; p.depth = 2;
    byId[p.id] = p;
    (p.models || []).forEach(m => { m.type = "model"; m.parent = p.id; m.depth = 3; m.project = p; byId[m.id] = m; });
  });

  if (data.mlops) {                 // MLOps: its own mini map, tools round the hub
    const o = data.mlops;
    o.type = "ops"; o.parent = "centre"; o.depth = 1; byId[o.id] = o;
    (o.tools || []).forEach(t => { t.type = "tool"; t.parent = o.id; t.depth = 2; byId[t.id] = t; });
  }

  if (data.ai) {                    // AI engineering: what I'm learning now; unstarted parts stay "coming soon"
    const a = data.ai;
    a.type = "branch"; a.parent = "centre"; a.depth = 1; byId[a.id] = a;
    index(a.children, a.id, 2);
  }

  // round the ring, anticlockwise from the right: Models, AI, the subjects in reading order, MLOps
  clusters = [byId.models, ...(data.ai ? [data.ai] : []), ...data.tree, ...(data.mlops ? [data.mlops] : [])];
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
    : `<li class="${n.soon ? "soon" : ""}"><a href="#/t/${n.id}"><span class="num">${n.num}</span>${n.label}${n.soon ? " <em>(coming up soon)</em>" : ""}</a></li>`).join("");
  box.innerHTML = `
    <header><button type="button" id="index-back">← Sphere</button><h1>Jag Sandhu</h1></header>
    <section><h2>Notes</h2>${data.tree.map(t => `<h3 class="top">${t.label}</h3><ol>${list(t.children)}</ol>`).join("")}</section>
    <section><h2>Projects</h2>${data.projects.map(p => `
      <div class="proj"><a href="#/p/${p.id}"><strong>${p.label}</strong></a>
        <span>${(p.models || []).map(m => `<a href="#/m/${m.id}">${m.label}${m.status ? ` (${m.status})` : ""}</a>`).join(" · ")}</span></div>`).join("")}</section>
    ${data.mlops ? `<section><h2>${data.mlops.label}</h2><div class="proj ops"><a href="#/o/${data.mlops.id}"><strong>Tools</strong></a>
        <span>${data.mlops.tools.map(t => `<a href="#/o/${t.id}">${t.label}${t.status ? ` (${t.status})` : ""}</a>`).join(" · ")}</span></div></section>` : ""}
    ${data.ai ? `<section><h2>${data.ai.label}</h2><ol>${list(data.ai.children)}</ol></section>` : ""}`;
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

// one mini map, laid out round its own hub at (0, 0). Returns local positions and its size.
// Every end point (chapter, model, tool) sits on the outer ring, in reading order, with a small gap
// between groups. Groups (Supervised, Regression ..., or a project) sit on inner rings, in the middle
// of their own points. A gap is left on the side facing the centre, where the line from me comes in.
function miniMap(hub, inward) {
  const slots = [];                                   // outer-ring points, in order; null = a gap
  let depth = 1;
  (function walk(list, level) {
    list.forEach((k, i) => {
      if (i > 0 && (kids(k).length || kids(list[i - 1]).length)) slots.push(null);
      if (kids(k).length) { depth = Math.max(depth, level + 1); walk(kids(k), level + 1); }
      else slots.push(k);
    });
  })(kids(hub), 1);
  const n = slots.length + 1.5;                       // + the gap facing the centre
  const minR = depth === 1 ? ONE_R : depth === 2 ? TWO_R : TWO_R + 85 * (depth - 2);
  const R = Math.max(minR, n * SLOT / (2 * Math.PI));
  const local = {};
  const put = (c, a, rr) => local[c.id] = { x: rr * Math.cos(rad(a)), y: rr * Math.sin(rad(a)), deg: a };
  slots.forEach((c, i) => { if (c) put(c, inward + (i + 1.25) * 360 / n, R); });   // start just past the gap
  (function groups(list, level) {                     // each group: middle of its points, on its own ring
    list.forEach(k => {
      if (!kids(k).length) return;
      groups(kids(k), level + 1);
      const ds = kids(k).map(c => local[c.id].deg);
      put(k, (Math.min(...ds) + Math.max(...ds)) / 2, R * level / depth * (depth === 2 ? 0.94 : 0.85));
    });
  })(kids(hub), 1);
  const longest = Math.max(0, ...slots.filter(Boolean).map(c => labelText(c).length));
  return { local, R, size: R + longest * CHAR_W * 0.8 + 30 };
}

function layout() {
  const pos = { centre: { x: 0, y: 0, r: 0 } };

  // size every mini map first, then pick a ring big enough to fit them side by side
  // a closed map only needs room for its title; the open one needs its full size
  const maps = clusters.map(h => {
    const m = miniMap(h, 0);
    if (h.id !== zoomedHub) m.size = Math.max(70, h.label.length * 6.2);
    return { h, m };
  });
  const RING_R = Math.max(TITLE_MIN, maps.reduce((k, c) => k + c.m.size, 0) * RING_GAP / Math.PI);

  // each mini map gets a slice of the ring in proportion to its size, so they always meet up all round
  const total = maps.reduce((k, c) => k + c.m.size, 0);
  let ang = 0;
  maps.forEach((c, i) => {
    if (i > 0) ang -= (maps[i - 1].m.size + c.m.size) / total * 180;                 // anticlockwise
    const H = { x: RING_R * X_STRETCH * Math.cos(rad(ang)), y: RING_R * Math.sin(rad(ang)) };
    const inward = Math.atan2(-H.y, -H.x) * 180 / Math.PI;                         // direction back to me
    const { local, R } = miniMap(c.h, inward);
    pos[c.h.id] = { ...H, r: Math.hypot(H.x, H.y), R };
    const open = c.h.id === zoomedHub;                                               // closed: points wait on the hub
    for (const id in local) {
      const p = open ? local[id] : { x: 0, y: 0 };
      pos[id] = { x: H.x + p.x, y: H.y + p.y, hx: H.x, hy: H.y, r: Math.hypot(H.x + p.x, H.y + p.y), lx: local[id].x, ly: local[id].y };
    }
  });
  pos.ringR = RING_R;
  return pos;
}

// tree edges, plus dashed threads from projects/models to the notes they use
function edgeList() {
  const out = [];
  Object.values(byId).forEach(n => { if (n.id !== "centre" && n.parent) out.push({ from: n.parent, to: n.id, kind: n.type === "tool" || n.type === "ops" ? "tree ops" : "tree" }); });
  ((data.mlops || {}).tools || []).forEach(t =>
    (t.used || []).forEach(u => byId[u.project] && out.push({ from: t.id, to: u.project, kind: "cross tool" })));
  data.projects.forEach(p => {
    (p.uses || []).forEach(u => byId[u] && out.push({ from: u, to: p.id, kind: "cross" }));
    (p.models || []).forEach(m => m.topic && byId[m.topic] && out.push({ from: m.topic, to: m.id, kind: "cross model" }));
  });
  return out;
}

const labelText = n => n.type === "leaf" && n.num != null ? `${n.num}  ${n.short || n.label}`
  : (n.short || n.label) + (n.status ? ` · ${n.status}` : "");

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
}

// the faint "web": a ring through every mini map, and a circle round each one
function drawWeb(pos) {
  layers.web.innerHTML = "";
  layers.web.append(el("ellipse", { cx: 0, cy: 0, rx: pos.ringR * X_STRETCH, ry: pos.ringR, class: "ring" }));
  clusters.filter(h => h.id === zoomedHub).forEach(h => {
    const p = pos[h.id];
    layers.web.append(el("circle", { cx: p.x, cy: p.y, r: p.R, class: "ring mini" }));
  });
}

function makeNode(id, p) {
  const n = byId[id];
  const side = n.type === "project" || n.type === "model" || n.type === "hub" ? "right"
             : n.type === "ops" || n.type === "tool" ? "opsside" : n.type === "centre" ? "" : "left";
  const cls = ["node", n.type, side, "d" + n.depth];
  if (n.soon) cls.push("soon");
  if (n.teaser) cls.push("teaser");
  if (n.status) cls.push("pending");
  const g = el("g", { class: cls.join(" ") });

  const r = { centre: 42, branch: n.depth === 1 ? 16 : 9, leaf: 6.5, hub: 16, project: 11, model: 7, ops: 16, tool: 7 }[n.type];
  if (n.type === "centre") {
    g.append(el("circle", { r, class: "hit" }));       // dark disc behind, and the click target
    const dots = el("g");
    g.append(dots);
    Orb.mini(dots, 34);                                // small spinning sphere of dots
  } else {
    g.append(el("circle", { r }));
  }
  g.dataset.r = r;

  const t = el("text");
  g.append(t);
  placeLabel(g, n, p, r);

  g.addEventListener("click", e => { e.stopPropagation(); if (!dragged) onNodeClick(id); });
  g.addEventListener("mouseenter", () => focus(id, true));
  g.addEventListener("mouseleave", () => focus(id, false));
  layers.nodes.append(g);
  return g;
}

// labels: every point radiates outwards from its own hub; hubs have their name underneath;
// the middle ring (groups, projects) sits level, above its dot
function placeLabel(g, n, p, r) {
  const t = g.querySelector("text");
  t.removeAttribute("transform"); t.textContent = "";
  const hubLevel = n.depth === 1;
  if (n.type === "centre") {
    const tip = el("title"); tip.textContent = "Jag Sandhu · click to go back";
    g.append(tip);                                         // name shows on hover; the web needs the space
  } else if (hubLevel) {
    t.setAttribute("x", 0); t.setAttribute("y", r + 30); t.setAttribute("text-anchor", "middle");
    t.textContent = n.label;
    if (n.tag) {                                           // the AI map: a small "in progress" under its name
      const s = el("tspan", { x: 0, dy: 26, class: "soon-tag" }); s.textContent = n.tag;
      t.append(s);
    }
  } else if (n.type === "branch" || n.type === "project") {
    t.setAttribute("x", 0); t.setAttribute("y", -r - 8); t.setAttribute("text-anchor", "middle");
    t.textContent = n.short || n.label;
  } else {
    const ang = Math.atan2(p.ly, p.lx) * 180 / Math.PI;                // the label follows its spoke
    const flip = Math.cos(rad(ang)) < 0;                               // left side: turn text the right way up
    t.setAttribute("transform", `rotate(${flip ? ang + 180 : ang})`);
    t.setAttribute("x", flip ? -(r + 8) : r + 8);
    t.setAttribute("y", 4.5);
    t.setAttribute("text-anchor", flip ? "end" : "start");
    if (n.type === "leaf" && n.num != null) {
      const num = el("tspan", { class: "num" }); num.textContent = n.num + "  ";
      const lab = el("tspan"); lab.textContent = (n.short || n.label);
      // number sits next to the dot: before the label on the right, after it on the left
      if (flip) { lab.textContent += "  "; t.append(lab); num.textContent = String(n.num); t.append(num); }
      else { t.append(num, lab); }
    } else {
      t.textContent = labelText(n);
    }
  }
}

// edges: straight inside a mini map and out to the centre; threads between maps bow towards me
function curve(a, b, kind) {
  if (kind.startsWith("cross") || kind === "tool") {
    return `M${a.x},${a.y} Q${(a.x + b.x) * 0.25},${(a.y + b.y) * 0.25} ${b.x},${b.y}`;
  }
  return `M${a.x},${a.y} L${b.x},${b.y}`;
}

// ---------------------------------------------------------------------
// BUILD + OPENING ANIMATION: everything grows out of the centre
// ---------------------------------------------------------------------
let anim = null;

function build(animated) {
  if (!layers) makeLayers();
  const pos = layout();
  drawWeb(pos);
  const ids = Object.keys(byId);
  const D = animated ? 1300 : 0;

  // nodes start at the centre and fly out, inner rings first
  ids.forEach(id => {
    if (view[id]) return;
    const p = pos[id];
    const delay = animated ? (byId[id].depth || 0) * 160 + (p.r / pos.ringR) * 120 : 0;
    view[id] = { el: makeNode(id, p), x: 0, y: 0, o: 0, fx: 0, fy: 0, fo: 0, tx: p.x, ty: p.y, to: shown(id) ? 1 : 0, delay, p };
  });

  edgeList().forEach(e => {
    const key = e.from + ">" + e.to;
    if (edgeEls[key]) return;
    const path = el("path", { class: "edge " + e.kind });
    path.dataset.from = e.from; path.dataset.to = e.to;
    layers.edges.append(path);
    edgeEls[key] = path;
  });

  home = fitBox(Object.keys(byId).filter(shown));
  vb = { ...home };
  setVB();
  animate(animated ? 1300 : 0);
}

// the box that fits these nodes, including where each label ends
function fitBox(ids, pad = 30) {
  const xs = [], ys = [];
  ids.forEach(id => {
    const n = byId[id], p = view[id] ? view[id].p : null;
    if (!p) return;
    xs.push(p.x); ys.push(p.y);
    if (p.hx != null && n.type !== "branch" && n.type !== "project") {     // radial label: where it ends
      const len = 16 + labelText(n).length * CHAR_W, dx = p.x - p.hx, dy = p.y - p.hy, d = Math.hypot(dx, dy) || 1;
      xs.push(p.x + dx / d * len); ys.push(p.y + dy / d * len);
    }
    if (n.depth === 1) {                                                    // hub name underneath
      const half = n.label.length * 8;
      xs.push(p.x - half, p.x + half); ys.push(p.y + (n.tag ? 80 : 54));
    }
  });
  let box = {
    x: Math.min(...xs) - pad, y: Math.min(...ys) - pad,
    w: Math.max(...xs) - Math.min(...xs) + pad * 2, h: Math.max(...ys) - Math.min(...ys) + pad * 2 + 30
  };
  // keep the screen's shape, so flying between boxes doesn't stretch
  const aspect = innerWidth / innerHeight;
  if (box.w / box.h < aspect) { const w = box.h * aspect; box.x -= (w - box.w) / 2; box.w = w; }
  else { const h = box.w / aspect; box.y -= (h - box.h) / 2; box.h = h; }
  return box;
}

// move every node from where it is (fx, fy, fo) to where it should be (tx, ty, to)
function animate(D) {
  const t0 = performance.now();
  const ease = t => 1 - Math.pow(1 - t, 3);
  const maxDelay = Math.max(...Object.values(view).map(v => v.delay));
  if (anim) cancelAnimationFrame(anim);

  (function frame(now) {
    const el_ = now - t0;
    for (const id in view) {
      const v = view[id];
      const k = D ? ease(Math.min(1, Math.max(0, (el_ - v.delay) / D))) : 1;
      v.x = v.fx + (v.tx - v.fx) * k; v.y = v.fy + (v.ty - v.fy) * k; v.o = v.fo + (v.to - v.fo) * k;
      v.el.setAttribute("transform", `translate(${v.x},${v.y})`);
      v.el.style.opacity = v.o;
      v.el.style.pointerEvents = v.to ? "" : "none";
    }
    drawEdges();
    const done = !D || el_ > D + maxDelay;
    map.classList.toggle("settled", done);
    if (!done) anim = requestAnimationFrame(frame);
    else anim = null;
  })(t0);
}

// open one mini map (or none): the ring makes room for it, its points fan out, the view flies to it
function setOpen(id, animated = true) {
  zoomedHub = id;
  if (!layers) return;
  const pos = layout();
  for (const k in view) {
    const v = view[k], p = pos[k];
    Object.assign(v, { fx: v.x, fy: v.y, fo: v.o, tx: p.x, ty: p.y, to: shown(k) ? 1 : 0, delay: 0, p });
    if (byId[k].depth > 1) placeLabel(v.el, byId[k], p, +v.el.dataset.r);        // spokes may have turned
  }
  drawWeb(pos);
  home = fitBox(Object.keys(byId).filter(k => byId[k].depth <= 1));
  animate(animated ? MOVE_MS : 0);
  let box = home;
  if (id) {
    const ids = Object.keys(byId).filter(k => hubOf(k) && hubOf(k).id === id);
    const b = fitBox(ids, 60), k = Math.max(1, 1100 / b.w);        // small maps: don't zoom in too far
    box = { x: b.x - b.w * (k - 1) / 2, y: b.y - b.h * (k - 1) / 2, w: b.w * k, h: b.h * k };
  }
  if (animated) flyTo(box, MOVE_MS); else { vb = { ...box }; setVB(); }
}

function drawEdges() {
  for (const key in edgeEls) {
    const p = edgeEls[key];
    const c = p.getAttribute("class");
    const end = id => view[c.includes("cross") && !shown(id) ? hubOf(id).id : id];    // a closed map: thread to its title
    const a = end(p.dataset.from), b = end(p.dataset.to);
    if (!a || !b) continue;
    p.setAttribute("d", curve(a, b, c.includes("cross") ? "cross" : "tree"));
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
  (function down(x) { kids(x).forEach(c => { below.add(c.id); down(c); }); })(byId[id]);
  // ...plus its path back to the centre
  const lit = new Set(below);
  let n = byId[id];
  while (n && n.parent) { lit.add(n.parent); n = byId[n.parent]; }

  Object.values(edgeEls).forEach(p => {
    const f = p.dataset.from, t = p.dataset.to;
    if (p.getAttribute("class").includes("cross")) {
      // threads across the halves that start or end at the hovered part
      if (below.has(f) || below.has(t)) {
        p.classList.add("hot"); lit.add(f); lit.add(t);
        if (!shown(f)) lit.add(hubOf(f).id);
        if (!shown(t)) lit.add(hubOf(t).id);
      }
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
  const w = Math.min(home.w * 1.5, Math.max(home.w / 14, vb.w * factor));
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
document.getElementById("zoom-fit").onclick = () => setOpen(null);

// ---------------------------------------------------------------------
// CLICKS
// ---------------------------------------------------------------------
function onNodeClick(id) {
  const n = byId[id];
  if (n.type === "centre") return closeMap();
  if (n.type === "project") return go(`#/p/${id}`);
  if (n.type === "model") return go(`#/m/${id}`);
  if (n.type === "leaf") return go(`#/t/${id}`);
  if (n.type === "tool") return go(`#/o/${id}`);
  if (n.type === "ops" && zoomedHub === id) return go(`#/o/${id}`);   // MLOps: zoom in first, then its page
  zoomToMini(id);
}

// open a mini map (zooms into it); clicking its title again closes it
function zoomToMini(id) {
  const hub = hubOf(id);
  setOpen(zoomedHub === hub.id && id === hub.id ? null : hub.id);
}


// ---------------------------------------------------------------------
// ROUTER: #/t/<topic>  #/p/<project>  #/m/<model>  #/o/<mlops tool>  (empty = map)
// so the browser back button works and every page has its own link
// ---------------------------------------------------------------------
function go(hash) { if (location.hash === hash) route(); else location.hash = hash; }

async function route() {
  await ready;
  if (location.hash === "#/about") return showAbout();
  const [, kind, id] = location.hash.match(/^#\/(t|p|m|o)\/(.+)$/) || [];

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
  else if (kind === "m") showModel(id);
  else if (kind === "o") showOps(id);
  else showProject(id);
}

// ABOUT ME page (#/about), written in about.md
async function showAbout() {
  if (!mapOpen) { Orb.hideNow(); openMap(false); }
  reader.classList.add("open");
  reader.setAttribute("aria-hidden", "false");
  document.getElementById("reader-scroll").scrollTop = 0;
  crumbs.innerHTML = `<span>About me</span>`;
  paper.className = "paper";
  paper.innerHTML = `<p class="muted">Loading…</p>`;
  try {
    const r = await fetch("about.md", { cache: "no-cache" });
    if (!r.ok) throw new Error(r.status);
    paper.innerHTML = renderMarkdown(await r.text());
  } catch (e) {
    paper.innerHTML = `<h1>About me</h1><p>Couldn't load <code>about.md</code> (${e.message}).</p>`;
  }
}
document.getElementById("about-link").addEventListener("click", e => e.stopPropagation());

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
  if (layers && hubOf(id).id !== zoomedHub) setOpen(hubOf(id).id, false);
  crumbs.innerHTML = trail(id).map(t => `<span>${t.label}</span>`).join("<i>›</i>");
  paper.className = "paper";

  let body;
  if (n.soon) {
    body = `<h1>${n.label}</h1><div class="soon-box"><p>Coming up soon.</p><p class="muted">I'm studying this now; my notes will go here as I write them up.</p></div>`;
  } else {
    paper.innerHTML = `<p class="muted">Loading…</p>`;
    try {
      if (!noteCache[id]) {
        const r = await fetch(`notes/${id}.md`, { cache: "no-cache" });
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
  const prev = i < 0 ? null : path[i - 1], next = i < 0 ? null : path[i + 1];
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

  // ✓ / ✗ example sentences (vocabulary page)
  html = html.replace(/<p>✓/g, '<p class="ok">✓').replace(/<p>✗/g, '<p class="no">✗');

  // colour-code callouts by their opening word
  html = html.replace(/<blockquote>\s*<p><strong>([^<:]+):?<\/strong>/g, (m, w) => {
    const k = { "definition": "def", "where it goes wrong": "warn", "check": "warn", "say it like this": "say" }[w.trim().replace(/:$/, "").toLowerCase()];
    return k ? m.replace("<blockquote>", `<blockquote class="${k}">`) : m;
  });

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

// MODEL PAGES: short on purpose, the detail lives on GitHub
function showModel(id) {
  const m = byId[id], p = m.project;
  const gh = f => `https://github.com/${p.repo}/blob/${p.branch || "main"}/${f}`;
  crumbs.innerHTML = `<span>Projects</span><i>›</i><a href="#/p/${p.id}">${p.label}</a><i>›</i><span>${m.label}</span>`;
  paper.className = "paper project model";
  const t = m.metrics;
  paper.innerHTML = `
    <p class="kicker">Model · ${p.label}</p>
    <h1>${m.label}${m.status ? ` <span class="status">${m.status}</span>` : ""}</h1>
    <p class="lead">${m.summary || ""}</p>
    ${(m.points || []).length ? `<ul>${m.points.map(x => `<li>${x}</li>`).join("")}</ul>` : ""}
    ${t ? `<table class="metrics"><tr>${t.columns.map(c => `<th>${c}</th>`).join("")}</tr>
      ${t.rows.map(r => `<tr>${r.map(v => `<td>${v}</td>`).join("")}</tr>`).join("")}</table>` : ""}
    <h3>Code</h3>
    ${(m.code || []).map(f => `<a class="chip" href="${gh(f)}" target="_blank" rel="noopener">${f.split("/").pop()} ↗</a>`).join("")}
    <p class="repo-link"><a href="https://github.com/${p.repo}" target="_blank" rel="noopener">Full project on GitHub ↗</a></p>
    <footer class="chapter-foot">
      ${m.topic && byId[m.topic] ? `<a class="chip" href="#/t/${m.topic}">My notes: ${byId[m.topic].label} →</a>` : ""}
      <a class="chip" href="#/p/${p.id}">Project page →</a>
    </footer>`;
}

// MLOPS PAGES: no notes, just what each tool is for and where I've used it
function showOps(id) {
  const o = data.mlops, n = byId[id];
  const gh = (p, f) => `https://github.com/${p.repo}/blob/${p.branch || "main"}/${f}`;
  paper.className = "paper project model ops";
  if (n.type === "ops") {
    crumbs.innerHTML = `<span>${o.label}</span>`;
    paper.innerHTML = `
      <p class="kicker">MLOps</p>
      <h1>${o.label}</h1>
      <p class="lead">${o.summary || ""}</p>
      ${o.tools.map(t => `<div class="tool-row${t.status ? " pending" : ""}">
        <a class="chip" href="#/o/${t.id}">${t.label}${t.status ? ` (${t.status})` : ""} →</a>
        <p>${t.what}</p>
        ${(t.used || []).length ? `<p class="muted">Used in ${t.used.map(u => `<a href="#/p/${u.project}">${byId[u.project].label}</a>`).join(", ")}</p>` : ""}
      </div>`).join("")}`;
    return;
  }
  crumbs.innerHTML = `<a href="#/o/${o.id}">${o.label}</a><i>›</i><span>${n.label}</span>`;
  const used = n.used || [];
  paper.innerHTML = `
    <p class="kicker">MLOps tool</p>
    <h1>${n.label}${n.status ? ` <span class="status">${n.status}</span>` : ""}</h1>
    <p class="lead">${n.what}</p>
    <h3>Used in</h3>
    ${used.length ? used.map(u => { const p = byId[u.project]; return `
      <p><a href="#/p/${p.id}"><strong>${p.label}</strong></a>: ${u.note || ""}</p>
      ${(u.code || []).map(f => `<a class="chip" href="${gh(p, f)}" target="_blank" rel="noopener">${f.split("/").pop()} ↗</a>`).join("")}`; }).join("")
      : `<p class="muted">Not in a project yet: this is what I'm learning next.</p>`}
    <footer class="chapter-foot"><a class="chip" href="#/o/${o.id}">All MLOps tools →</a></footer>`;
}

function renderTab(p, tab, body) {
  if (tab === "overview") {
    body.innerHTML = `<p class="lead">${p.summary || ""}</p>
      ${(p.models || []).length ? `<h3>Models</h3>${p.models.map(m => `<a class="chip" href="#/m/${m.id}">${m.label}${m.status ? ` (${m.status})` : ""} →</a>`).join("")}` : ""}
      ${opsFor(p.id).length ? `<h3>MLOps</h3>${opsFor(p.id).map(t => `<a class="chip" href="#/o/${t.id}">${t.label} →</a>`).join("")}` : ""}
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

const opsFor = pid => ((data.mlops || {}).tools || []).filter(t => (t.used || []).some(u => u.project === pid));

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
