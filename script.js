// ============ SETUP ============
const book  = document.getElementById("book");
const map   = document.getElementById("map");
const panel = document.getElementById("panel");

const W = 1400, H = 900, CX = W / 2, CY = H / 2;   // map drawing space
const SVGNS = "http://www.w3.org/2000/svg";

const dataReady = fetch("data.json").then(r => r.json());
let data, nodes = {}, opened = false;

// ============ BOOK → MAP ============
book.addEventListener("click", async () => {
  if (opened) return;
  opened = true;
  data = await dataReady;
  book.classList.add("open");                 // 1. cover swings open
  setTimeout(() => {
    book.classList.add("gone");               // 2. book shrinks away
    buildMap();                               // 3. map grows out
  }, 1600);
});

function reset() {                            // click the centre node to go back
  panel.classList.remove("open");
  map.classList.remove("show", "edges-in");
  setTimeout(() => {
    map.innerHTML = "";
    nodes = {};
    book.classList.remove("gone");
    setTimeout(() => { book.classList.remove("open"); opened = false; }, 900);
  }, 700);
}

// ============ LAYOUT: work out where every node sits ============
function layout() {
  const list = [], edges = [];
  list.push({ id: "centre", label: "Jag Sandhu", x: CX, y: CY, type: "centre" });

  // LEFT side: knowledge categories and their topics
  const cats = data.knowledge;
  const slots = cats.reduce((s, c) => s + c.children.length, 0) + cats.length - 1;
  const gapL = Math.min(60, (H - 100) / Math.max(slots - 1, 1));
  let y = CY - gapL * (slots - 1) / 2;

  cats.forEach(cat => {
    const ys = [];
    cat.children.forEach(ch => {
      list.push({ id: ch.id, label: ch.label, x: CX - 470, y, type: "leaf", side: "left", item: ch, parent: cat });
      edges.push({ from: cat.id, to: ch.id, kind: "tree" });
      ys.push(y);
      y += gapL;
    });
    y += gapL;                                // blank row between categories
    const catY = (ys[0] + ys[ys.length - 1]) / 2;
    list.push({ id: cat.id, label: cat.label, x: CX - 250, y: catY, type: "cat", side: "left", item: cat });
    edges.push({ from: "centre", to: cat.id, kind: "tree" });
  });

  // RIGHT side: projects
  list.push({ id: "hub", label: "Projects", x: CX + 230, y: CY, type: "hub", side: "right" });
  edges.push({ from: "centre", to: "hub", kind: "tree" });

  const ps = data.projects;
  const gapR = Math.min(110, (H - 100) / Math.max(ps.length - 1, 1));
  ps.forEach((p, i) => {
    const py = CY - gapR * (ps.length - 1) / 2 + i * gapR;
    list.push({ id: p.id, label: p.label, x: CX + 450, y: py, type: "project", side: "right", item: p });
    edges.push({ from: "hub", to: p.id, kind: "tree" });
    (p.uses || []).forEach(u => edges.push({ from: u, to: p.id, kind: "cross" }));   // the links across
  });

  return { list, edges };
}

// ============ DRAW ============
function el(tag, attrs = {}) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}

function curve(a, b, kind) {
  if (kind === "cross") {                     // arc over the top or under the bottom
    const mid = (a.y + b.y) / 2;
    const cy = mid < CY ? mid - 260 : mid + 260;
    return `M${a.x},${a.y} Q${CX},${cy} ${b.x},${b.y}`;
  }
  const mx = (a.x + b.x) / 2;                 // smooth S-curve
  return `M${a.x},${a.y} C${mx},${a.y} ${mx},${b.y} ${b.x},${b.y}`;
}

function buildMap() {
  const { list, edges } = layout();
  list.forEach(n => nodes[n.id] = n);

  const edgeG = el("g"), nodeG = el("g");
  map.append(edgeG, nodeG);

  edges.forEach(e => {
    const a = nodes[e.from], b = nodes[e.to];
    if (!a || !b) { console.warn("Unknown id in data.json:", e.from, "→", e.to); return; }
    const p = el("path", { d: curve(a, b, e.kind), class: "edge " + e.kind });
    p.dataset.from = e.from;
    p.dataset.to = e.to;
    edgeG.append(p);
  });

  list.forEach((n, i) => {
    const g = el("g", { class: `node ${n.type} ${n.side || ""}` });
    g.style.transform = `translate(${CX}px, ${CY}px)`;        // everything starts in the centre

    const r = { centre: 40, cat: 13, hub: 13, leaf: 7, project: 11 }[n.type];
    g.append(el("circle", { r }));

    const t = el("text");
    if (n.type === "centre") {
      const icon = el("text", { "text-anchor": "middle", y: 3, class: "icon" });
      icon.textContent = "📖";
      g.append(icon);
      t.setAttribute("y", 64); t.setAttribute("text-anchor", "middle");
    } else if (n.type === "cat" || n.type === "hub") {
      t.setAttribute("y", -28); t.setAttribute("text-anchor", "middle");
    } else if (n.side === "left") {
      t.setAttribute("x", -r - 10); t.setAttribute("text-anchor", "end");
    } else {
      t.setAttribute("x", r + 10);
    }
    t.textContent = n.label;
    g.append(t);

    g.addEventListener("click", () => select(n));
    g.addEventListener("mouseenter", () => highlight(n.id, true));
    g.addEventListener("mouseleave", () => highlight(n.id, false));
    nodeG.append(g);

    setTimeout(() => { g.style.transform = `translate(${n.x}px, ${n.y}px)`; }, 80 + i * 45);   // fly out
  });

  map.classList.add("show");
  setTimeout(() => map.classList.add("edges-in"), 1100 + list.length * 45);
}

function highlight(id, on) {
  map.querySelectorAll(`.edge[data-from="${id}"], .edge[data-to="${id}"]`)
     .forEach(p => p.classList.toggle("hot", on));
}

// ============ SIDE PANEL ============
function select(n) {
  if (n.type === "centre") return reset();
  if (n.type === "hub") return;
  const it = n.item;
  let html = `<button class="close" aria-label="Close">×</button>`;

  if (n.type === "project") {
    html += `<p class="kicker">Project</p><h2>${it.label}</h2>
      <div class="tabs">
        <button data-tab="overview" class="active">Overview</button>
        <button data-tab="code">Code</button>
        <button data-tab="results">Results</button>
      </div>
      <div class="tab-body"></div>`;
  } else {
    const kicker = n.type === "cat" ? "Topic" : n.parent.label;
    html += `<p class="kicker">${kicker}</p><h2>${it.label}</h2><p>${it.text || ""}</p>`;
    if (n.type === "cat") {
      html += `<h3>Covers</h3><ul>${it.children.map(c => `<li>${c.label}</li>`).join("")}</ul>`;
    }
    const used = data.projects.filter(p => (p.uses || []).includes(it.id));
    if (used.length) {
      html += `<h3>Used in</h3>` + used.map(p => `<button class="chip" data-go="${p.id}">${p.label} →</button>`).join("");
    }
  }

  panel.innerHTML = html;
  panel.classList.add("open");
  panel.querySelector(".close").onclick = () => panel.classList.remove("open");
  panel.querySelectorAll("[data-go]").forEach(b => b.onclick = () => select(nodes[b.dataset.go]));

  if (n.type === "project") {
    const body = panel.querySelector(".tab-body");
    const tabs = panel.querySelectorAll(".tabs button");
    tabs.forEach(b => b.onclick = () => {
      tabs.forEach(x => x.classList.toggle("active", x === b));
      renderTab(it, b.dataset.tab, body);
    });
    renderTab(it, "overview", body);
  }
}

function renderTab(p, tab, body) {
  if (tab === "overview") {
    body.innerHTML = `<p>${p.summary || ""}</p>
      <h3>Methods used</h3>
      ${(p.uses || []).map(u => nodes[u] ? `<button class="chip" data-go="${u}">${nodes[u].label}</button>` : "").join("")}
      <p class="repo-link"><a href="https://github.com/${p.repo}" target="_blank" rel="noopener">Open repository ↗</a></p>`;
    body.querySelectorAll("[data-go]").forEach(b => b.onclick = () => select(nodes[b.dataset.go]));
  }

  if (tab === "code") {
    body.innerHTML = `<div class="files">
        ${(p.code || []).map(f => `<button class="chip" data-file="${f}">${f.split("/").pop()}</button>`).join("")}
      </div>
      <div class="code-box"><p class="muted">Pick a file to view it.</p></div>`;
    const box = body.querySelector(".code-box");
    body.querySelectorAll("[data-file]").forEach(b => b.onclick = () => loadCode(p, b.dataset.file, box));
  }

  if (tab === "results") {
    const rs = p.results || [];
    body.innerHTML = rs.length
      ? rs.map(r => `<figure><img src="${r.img}" alt="${r.caption || ""}" loading="lazy"><figcaption>${r.caption || ""}</figcaption></figure>`).join("")
      : `<p class="muted">Results coming soon.</p>`;
  }
}

async function loadCode(p, path, box) {
  const branch = p.branch || "main";
  const ghLink = `https://github.com/${p.repo}/blob/${branch}/${path}`;

  if (path.endsWith(".ipynb")) {                 // GitHub renders notebooks nicely, so link out
    box.innerHTML = `<p>Notebooks display best on GitHub.</p><a href="${ghLink}" target="_blank" rel="noopener">Open notebook ↗</a>`;
    return;
  }

  box.innerHTML = `<p class="muted">Loading…</p>`;
  try {
    const res = await fetch(`https://raw.githubusercontent.com/${p.repo}/${branch}/${path}`);
    if (!res.ok) throw new Error(res.status);
    const code = await res.text();
    const lang = { py: "python", sql: "sql", r: "r", js: "javascript" }[path.split(".").pop()] || "plaintext";
    box.innerHTML = `<a href="${ghLink}" target="_blank" rel="noopener">View on GitHub ↗</a><pre><code class="language-${lang}"></code></pre>`;
    const c = box.querySelector("code");
    c.textContent = code;
    if (window.hljs) hljs.highlightElement(c);
  } catch (e) {
    box.innerHTML = `<p>Couldn't load this file (${e.message}). Check the repo name, branch and path in data.json.</p>
      <a href="${ghLink}" target="_blank" rel="noopener">Try it on GitHub ↗</a>`;
  }
}

document.addEventListener("keydown", e => { if (e.key === "Escape") panel.classList.remove("open"); });
