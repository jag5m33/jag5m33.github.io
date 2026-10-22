/* =====================================================================
   Jag Sandhu — portfolio
   book → expandable mind map → full-screen chapters and project pages

   Content lives in two places, so you rarely touch this file:
     data.json        the map: topics (left), projects (right)
     notes/<id>.md    one chapter per topic, written in Markdown
   ===================================================================== */

const SVGNS = "http://www.w3.org/2000/svg";
const book   = document.getElementById("book");
const map    = document.getElementById("map");
const reader = document.getElementById("reader");
const paper  = document.getElementById("paper");
const crumbs = document.getElementById("crumbs");

// layout settings (map units, not pixels)
const ROW      = 40;     // vertical gap between topics
const GROUP    = 24;     // extra gap between the top-level branches
const COL      = 200;    // horizontal gap between levels
const FIRST    = 230;    // distance from the centre to the first column
const HUB_X    = 230, PROJ_X = 450, PROJ_ROW = 110;

let data;
const byId = {};                  // every node, by id
const expanded = new Set();       // branches currently open
const view = {};                  // id -> { el, x, y, o } what's on screen now
const edgeEls = {};               // "from>to" -> path element
let vb = { x: -700, y: -450, w: 1400, h: 900 };
let mapOpen = false;

// ---------------------------------------------------------------------
// LOAD
// ---------------------------------------------------------------------
const ready = fetch("data.json").then(r => r.json()).then(d => {
  data = d;
  byId.centre = { id: "centre", label: "Jag Sandhu", type: "centre" };
  byId.hub    = { id: "hub", label: "Projects", type: "hub", parent: "centre" };

  (function index(list, parent, depth) {
    list.forEach((n, i) => {
      n.parent = parent;
      n.depth = depth;
      n.type = n.children ? "branch" : "leaf";
      n.siblings = list;
      byId[n.id] = n;
      if (n.children) index(n.children, n.id, depth + 1);
    });
  })(data.tree, "centre", 1);

  data.projects.forEach(p => { p.type = "project"; p.parent = "hub"; byId[p.id] = p; });
  return d;
});

// ---------------------------------------------------------------------
// BOOK
// ---------------------------------------------------------------------
book.addEventListener("click", async () => {
  if (mapOpen) return;
  await ready;
  book.classList.add("open");
  setTimeout(() => {
    book.classList.add("gone");
    openMap(true);
  }, 1600);
});

function openMap(animated) {
  mapOpen = true;
  document.body.classList.add("map-open");
  relayout(animated ? 1100 : 0, true);
}

function closeMap() {
  expanded.clear();
  relayout(600);
  setTimeout(() => {
    document.body.classList.remove("map-open");
    setTimeout(() => {
      map.innerHTML = "";
      for (const k in view) delete view[k];
      for (const k in edgeEls) delete edgeEls[k];
      book.classList.remove("gone");
      setTimeout(() => { book.classList.remove("open"); mapOpen = false; }, 900);
    }, 500);
  }, 650);
}

// ---------------------------------------------------------------------
// WHAT'S VISIBLE, AND WHERE IT GOES
// ---------------------------------------------------------------------
function isVisible(id) {
  const n = byId[id];
  if (!n || n.type === "centre" || n.type === "hub" || n.type === "project") return true;
  let p = n.parent;
  while (p !== "centre") {
    if (!expanded.has(p)) return false;
    p = byId[p].parent;
  }
  return true;
}

function layout() {
  const pos = { centre: { x: 0, y: 0 } };
  let y = 0, first = true;

  function place(n) {
    const x = -FIRST - (n.depth - 1) * COL;
    if (n.children && expanded.has(n.id)) {
      const ys = n.children.map(place);
      pos[n.id] = { x, y: (ys[0] + ys[ys.length - 1]) / 2 };
    } else {
      pos[n.id] = { x, y };
      y += ROW;
    }
    return pos[n.id].y;
  }

  data.tree.forEach(n => {
    if (!first) y += GROUP;
    first = false;
    place(n);
  });

  // centre the left side vertically on the book
  const ids = Object.keys(pos).filter(k => k !== "centre");
  const ys = ids.map(k => pos[k].y);
  const mid = (Math.min(...ys) + Math.max(...ys)) / 2;
  ids.forEach(k => pos[k].y -= mid);

  // right side
  pos.hub = { x: HUB_X, y: 0 };
  const ps = data.projects;
  ps.forEach((p, i) => pos[p.id] = { x: PROJ_X, y: (i - (ps.length - 1) / 2) * PROJ_ROW });

  return pos;
}

function edgesFor(visibleIds) {
  const out = [];
  const seen = new Set();
  visibleIds.forEach(id => {
    const n = byId[id];
    if (id !== "centre") out.push({ from: n.parent, to: id, kind: "tree" });
  });
  // dashed links from topics to the projects that use them;
  // if a topic is folded away, the link starts from its nearest visible branch
  data.projects.forEach(p => (p.uses || []).forEach(u => {
    let a = u;
    while (byId[a] && !isVisible(a)) a = byId[a].parent;
    if (!byId[a] || a === "centre") return;
    const key = a + ">" + p.id;
    if (!seen.has(key)) { seen.add(key); out.push({ from: a, to: p.id, kind: "cross" }); }
  }));
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

let edgeLayer, nodeLayer;

function makeNode(id) {
  const n = byId[id];
  const side = n.type === "project" || n.type === "hub" ? "right" : n.type === "centre" ? "" : "left";
  const g = el("g", { class: `node ${n.type} ${side} d${n.depth || 0}${n.soon ? " soon" : ""}` });

  const r = n.type === "centre" ? 40 : n.type === "hub" ? 13 : n.type === "project" ? 11
          : n.type === "branch" ? (n.depth === 1 ? 13 : 10) : 7;
  g.append(el("circle", { r }));

  if (n.type === "centre") {
    const icon = el("text", { "text-anchor": "middle", y: 2, class: "icon" });
    icon.textContent = "📖";
    g.append(icon);
  }

  const t = el("text");
  g.append(t);
  g._label = t;
  g._r = r;

  g.addEventListener("click", e => { e.stopPropagation(); onNodeClick(id); });
  g.addEventListener("mouseenter", () => hot(id, true));
  g.addEventListener("mouseleave", () => hot(id, false));
  nodeLayer.append(g);
  return g;
}

// label sits left of topics, above open branches, right of projects
function placeLabel(id) {
  const n = byId[id], g = view[id].el, t = g._label, r = g._r;
  let txt = n.label, x = 0, y = 0, anchor = "start";
  if (n.type === "centre")       { y = 66; anchor = "middle"; }
  else if (n.type === "hub")     { y = -24; anchor = "middle"; }
  else if (n.type === "project") { x = r + 10; y = 5; }
  else if (n.type === "branch" && expanded.has(id)) { y = -20; anchor = "middle"; txt += "  −"; }
  else {
    x = -r - 10; y = 5; anchor = "end";
    if (n.type === "branch") txt += "  +";
    if (n.soon) txt += " · soon";
  }
  t.textContent = txt;
  t.setAttribute("x", x); t.setAttribute("y", y); t.setAttribute("text-anchor", anchor);
  g.classList.toggle("expanded", expanded.has(id));
}

function curve(a, b, kind) {
  if (kind === "cross") {
    const mid = (a.y + b.y) / 2;
    const bend = mid <= 0 ? Math.min(a.y, b.y) - 160 : Math.max(a.y, b.y) + 160;
    return `M${a.x},${a.y} Q${(a.x + b.x) / 2},${bend} ${b.x},${b.y}`;
  }
  const mx = (a.x + b.x) / 2;
  return `M${a.x},${a.y} C${mx},${a.y} ${mx},${b.y} ${b.x},${b.y}`;
}

// ---------------------------------------------------------------------
// ANIMATION: move everything from where it is to where it should be
// ---------------------------------------------------------------------
let anim = null;

function relayout(duration = 700, stagger = false) {
  if (!edgeLayer) {
    edgeLayer = el("g"); nodeLayer = el("g");
    map.append(edgeLayer, nodeLayer);
  }

  const pos = layout();
  const visibleIds = Object.keys(byId).filter(id => isVisible(id) && pos[id]);
  const targets = {};
  visibleIds.forEach(id => targets[id] = { x: pos[id].x, y: pos[id].y, o: 1 });

  // nodes being hidden fold back into their nearest visible parent
  Object.keys(view).forEach(id => {
    if (!targets[id]) {
      let p = byId[id].parent;
      while (p !== "centre" && !targets[p]) p = byId[p].parent;
      targets[id] = { x: targets[p].x, y: targets[p].y, o: 0, remove: true };
    }
  });

  // new nodes start at their parent (or the centre)
  visibleIds.forEach((id, i) => {
    if (!view[id]) {
      let p = byId[id].parent, start = { x: 0, y: 0 };
      while (p && p !== "centre") { if (view[p]) { start = view[p]; break; } p = byId[p].parent; }
      view[id] = { el: makeNode(id), x: start.x, y: start.y, o: 0, delay: stagger ? i * 35 : 0 };
    }
  });
  visibleIds.forEach(placeLabel);

  // edges
  const edges = edgesFor(visibleIds);
  const keep = new Set();
  edges.forEach(e => {
    const key = e.from + ">" + e.to;
    keep.add(key);
    if (!edgeEls[key]) {
      const p = el("path", { class: "edge " + e.kind });
      p.dataset.from = e.from; p.dataset.to = e.to;
      p._o = 0;
      edgeLayer.append(p);
      edgeEls[key] = p;
    }
    edgeEls[key]._target = 1;
  });
  Object.keys(edgeEls).forEach(k => { if (!keep.has(k)) edgeEls[k]._target = 0; });

  // view box: fit everything that will be visible
  const xs = visibleIds.map(id => pos[id].x), ys = visibleIds.map(id => pos[id].y);
  const pad = 70;
  const box = {
    x: Math.min(...xs) - 240, y: Math.min(...ys) - pad,
    w: Math.max(...xs) + 260 - (Math.min(...xs) - 240), h: Math.max(...ys) - Math.min(...ys) + pad * 2 + 40
  };
  box.h = Math.max(box.h, 700);
  box.y = Math.min(box.y, -box.h / 2);

  // tween
  const from = {};
  Object.keys(view).forEach(id => from[id] = { x: view[id].x, y: view[id].y, o: view[id].o });
  const vbFrom = { ...vb };
  const eFrom = {};
  Object.keys(edgeEls).forEach(k => eFrom[k] = edgeEls[k]._o);
  const maxDelay = Math.max(0, ...Object.values(view).map(v => v.delay || 0));
  const total = duration + maxDelay;
  const t0 = performance.now();
  if (anim) cancelAnimationFrame(anim);

  const ease = t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

  function frame(now) {
    const el_ = now - t0;
    for (const id in view) {
      const v = view[id], f = from[id], tg = targets[id];
      if (!tg || !f) continue;
      const k = duration ? ease(Math.min(1, Math.max(0, (el_ - (v.delay || 0)) / duration))) : 1;
      v.x = f.x + (tg.x - f.x) * k;
      v.y = f.y + (tg.y - f.y) * k;
      v.o = f.o + (tg.o - f.o) * k;
      v.el.setAttribute("transform", `translate(${v.x},${v.y})`);
      v.el.style.opacity = v.o;
    }
    const kAll = total ? ease(Math.min(1, el_ / total)) : 1;
    for (const key in edgeEls) {
      const p = edgeEls[key], a = view[p.dataset.from], b = view[p.dataset.to];
      if (!a || !b) continue;
      p.setAttribute("d", curve(a, b, p.classList.contains("cross") ? "cross" : "tree"));
      // edges fade in at the end of the move, fade out at the start
      const target = p._target;
      const ke = target ? Math.max(0, (kAll - .6) / .4) : Math.min(1, kAll / .4);
      p._o = (eFrom[key] ?? 0) + (target - (eFrom[key] ?? 0)) * ke;
      p.style.opacity = p._o;
    }
    vb.x = vbFrom.x + (box.x - vbFrom.x) * kAll;
    vb.y = vbFrom.y + (box.y - vbFrom.y) * kAll;
    vb.w = vbFrom.w + (box.w - vbFrom.w) * kAll;
    vb.h = vbFrom.h + (box.h - vbFrom.h) * kAll;
    map.setAttribute("viewBox", `${vb.x} ${vb.y} ${vb.w} ${vb.h}`);

    if (el_ < total) { anim = requestAnimationFrame(frame); return; }

    // tidy up
    for (const id in view) {
      view[id].delay = 0;
      if (targets[id] && targets[id].remove) { view[id].el.remove(); delete view[id]; }
    }
    for (const key in edgeEls) if (!edgeEls[key]._target) { edgeEls[key].remove(); delete edgeEls[key]; }
    anim = null;
  }
  anim = requestAnimationFrame(frame);
}

function hot(id, on) {
  map.querySelectorAll(`.edge[data-from="${id}"], .edge[data-to="${id}"]`)
     .forEach(p => p.classList.toggle("hot", on));
}

// ---------------------------------------------------------------------
// CLICKS
// ---------------------------------------------------------------------
function onNodeClick(id) {
  const n = byId[id];
  if (n.type === "centre") return closeMap();
  if (n.type === "hub") return;
  if (n.type === "project") return go(`#/p/${id}`);
  if (n.type === "leaf") return go(`#/t/${id}`);

  // branch: open it, and fold away its siblings so the map stays readable
  if (expanded.has(id)) {
    collapse(id);
  } else {
    n.siblings.forEach(s => { if (s.id !== id) collapse(s.id); });
    expanded.add(id);
  }
  relayout(700);
}

function collapse(id) {
  expanded.delete(id);
  (byId[id].children || []).forEach(c => collapse(c.id));
}

// make sure a topic is visible on the map (used when arriving by link)
function reveal(id) {
  let p = byId[id] && byId[id].parent;
  while (p && p !== "centre") { expanded.add(p); p = byId[p].parent; }
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

  if (!mapOpen) {                     // arrived straight from a link: skip the book
    book.classList.add("open", "gone");
    if (kind === "t") reveal(id);
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

  // footer: projects that use this topic, and previous / next topic
  const used = data.projects.filter(p => (p.uses || []).includes(id));
  let foot = "";
  if (used.length) {
    foot += `<h3>Used in my projects</h3>` +
      used.map(p => `<a class="chip" href="#/p/${p.id}">${p.label} →</a>`).join("");
  }
  const sib = n.siblings, i = sib.indexOf(n);
  const prev = sib[i - 1], next = sib[i + 1];
  foot += `<nav class="pager">
      ${prev && !prev.children ? `<a href="#/t/${prev.id}">← ${prev.label}</a>` : "<span></span>"}
      ${next && !next.children ? `<a href="#/t/${next.id}">${next.label} →</a>` : "<span></span>"}
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
