/* =====================================================================
   THE ORB — a rotating sphere of dots on the landing screen.
   Gold half = topics, teal half = projects (same colours as the map).

   Used by script.js:
     Orb.burst(getTarget, onMidway)  click: dots burst out, then shrink into
                                     the map's centre node
     Orb.restore(target)             closing the map: dots grow back out
     Orb.hideNow()                   arriving by a link: skip the intro
     Orb.mini(svgGroup, radius)      the small spinning sphere at the
                                     centre of the map
   ===================================================================== */
(function () {
  const canvas = document.getElementById("orb");
  const intro  = document.getElementById("intro");
  const ctx    = canvas.getContext("2d");

  // ---------------- settings you might want to tweak ----------------
  const N_DOTS   = 1100;               // dots on the sphere
  const N_RING   = 200;                // dots on the ring
  const RING_R   = 1.45;               // ring radius, in sphere radii
  const SPIN     = 0.00018;            // radians per ms when idle
  const GOLD = [232, 200, 114], TEAL = [95, 201, 184], INK = [245, 240, 230];
  // -------------------------------------------------------------------

  // Fibonacci sphere: N points spread almost perfectly evenly
  function fibonacci(n) {
    const pts = [], golden = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < n; i++) {
      const y = 1 - (i / (n - 1)) * 2;
      const r = Math.sqrt(1 - y * y);
      const t = golden * i;
      pts.push({ x: Math.cos(t) * r, y, z: Math.sin(t) * r });
    }
    return pts;
  }

  // colour by side: gold on one side, teal on the other, blended in between
  const mix = (a, b, k) => a.map((v, i) => Math.round(v + (b[i] - v) * k));
  const smooth = (e0, e1, x) => { const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t); };

  function colourFor(p, i) {
    if (i % 29 === 0) return INK;                      // a few bright sparkles
    return mix(GOLD, TEAL, smooth(-0.35, 0.35, p.x));
  }

  const dots = fibonacci(N_DOTS).map((p, i) => ({
    ...p, c: colourFor(p, i), k: 0.7 + Math.random() * 0.8, j: Math.random() * Math.PI * 2
  }));

  const ring = [];
  for (let i = 0; i < N_RING; i++) {
    const a = (i / N_RING) * Math.PI * 2;
    ring.push({ x: Math.cos(a) * RING_R, y: 0, z: Math.sin(a) * RING_R, k: 0.8 + Math.random() * 0.6 });
  }
  const RING_TILT = 0.42;

  // rotate around y (yaw) then x (pitch), then perspective-project
  function project(p, yaw, pitch, R, cx, cy, scale = 1) {
    const ca = Math.cos(yaw), sa = Math.sin(yaw);
    const x1 = p.x * ca + p.z * sa, z1 = -p.x * sa + p.z * ca;
    const cb = Math.cos(pitch), sb = Math.sin(pitch);
    const y2 = p.y * cb - z1 * sb, z2 = p.y * sb + z1 * cb;
    const persp = 1 / (1 - z2 * 0.22 / Math.max(scale, 0.3));
    return { x: cx + x1 * R * scale * persp, y: cy + y2 * R * scale * persp, z: z2, s: persp };
  }

  // ------------------------------------------------------------------
  // state
  // ------------------------------------------------------------------
  let W = 0, H = 0, DPR = 1, R = 200, CX = 0, CY = 0;
  let yaw = 0, pitch = -0.28, ringYaw = 0;
  let tiltX = 0, tiltY = 0, aimX = 0, aimY = 0;   // pointer parallax
  let hover = 0, hoverAim = 0;
  let mode = "idle";                              // idle | burst | gone | restore
  let t0 = 0, target = null, getTarget = null, onMid = null, midFired = false;
  let last = performance.now(), running = false;

  function resize() {
    DPR = Math.min(window.devicePixelRatio || 1, 2);
    W = window.innerWidth; H = window.innerHeight;
    canvas.width = W * DPR; canvas.height = H * DPR;
    canvas.style.width = W + "px"; canvas.style.height = H + "px";
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    R = Math.min(W, H) * (W < 600 ? 0.3 : 0.24);
    CX = W / 2; CY = H * 0.44;
  }
  window.addEventListener("resize", resize);
  resize();

  window.addEventListener("pointermove", e => {
    aimX = (e.clientX / W - 0.5);
    aimY = (e.clientY / H - 0.5);
    const d = Math.hypot(e.clientX - CX, e.clientY - CY);
    hoverAim = mode === "idle" && d < R * 1.1 ? 1 : 0;
    intro.style.cursor = hoverAim ? "pointer" : "default";
  });

  const easeOut = t => 1 - Math.pow(1 - t, 3);
  const easeInOut = t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

  // ------------------------------------------------------------------
  // draw one frame
  // ------------------------------------------------------------------
  function frame(now) {
    const dt = Math.min(50, now - last); last = now;

    tiltX += (aimX - tiltX) * 0.04;
    tiltY += (aimY - tiltY) * 0.04;
    hover += (hoverAim - hover) * 0.08;

    let speed = SPIN * (1 + hover * 2.5);
    let cx = CX, cy = CY, rad = 1, fade = 1, scatter = 0, ringFade = 1;

    if (mode === "burst" || mode === "restore") {
      const D = mode === "burst" ? 1500 : 1300;
      let t = Math.min(1, (now - t0) / D);
      if (mode === "restore") t = 1 - t;          // play the burst backwards

      if (getTarget) { const g = getTarget(); if (g) target = g; }
      const tg = target || { x: CX, y: CY, r: 30 };

      speed = SPIN * (1 + 12 * Math.sin(Math.PI * Math.min(1, t * 1.2)));
      if (t < 0.32) {                             // phase 1: breathe out
        const a = easeOut(t / 0.32);
        scatter = a;
      } else {                                    // phase 2: shrink into the centre node
        const a = easeInOut((t - 0.32) / 0.68);
        scatter = 1 - a;
        rad = 1 + (tg.r / R - 1) * a;
        cx = CX + (tg.x - CX) * a;
        cy = CY + (tg.y - CY) * a;
        fade = 1 - a * 0.85;
        ringFade = 1 - a;
      }

      if (mode === "burst" && !midFired && t > 0.27) { midFired = true; onMid && onMid(); }
      if ((now - t0) >= D) {
        if (mode === "burst") { mode = "gone"; canvas.style.opacity = 0; }
        else { mode = "idle"; intro.classList.remove("away"); }
      }
    }

    yaw += speed * dt;
    ringYaw -= speed * 0.6 * dt;
    const breathe = 1 + Math.sin(now / 1400) * 0.012 + hover * 0.03;
    const pitchNow = pitch + tiltY * 0.5;
    const yawNow = yaw + tiltX * 0.8;

    ctx.clearRect(0, 0, W, H);
    if (mode === "gone") { running = false; return; }

    // soft glow behind the sphere
    const glowR = R * rad * 1.6;
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, glowR);
    g.addColorStop(0, `rgba(95,201,184,${0.10 * fade})`);
    g.addColorStop(0.5, `rgba(232,200,114,${0.04 * fade})`);
    g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g;
    ctx.fillRect(cx - glowR, cy - glowR, glowR * 2, glowR * 2);

    // project everything, then draw back to front
    const pts = [];
    const rr = R * breathe;

    for (const p of dots) {
      const out = 1 + scatter * 0.6 * p.k;
      const q = project({ x: p.x * out, y: p.y * out, z: p.z * out }, yawNow, pitchNow, rr, cx, cy, rad);
      pts.push({ q, c: p.c, base: p.c === INK ? 1.9 : 1.35, a: fade });
    }

    if (ringFade > 0.01) {
      for (const p of ring) {
        // tilt the ring plane, then let it spin on its own
        const ca = Math.cos(ringYaw), sa = Math.sin(ringYaw);
        const x = p.x * ca + p.z * sa, z = -p.x * sa + p.z * ca;
        const ct = Math.cos(RING_TILT), st = Math.sin(RING_TILT);
        const y = -z * st, z2 = z * ct;
        const out = 1 + scatter * 1.2 * p.k;
        const q = project({ x: x * out, y: y * out, z: z2 * out }, tiltX * 0.8, pitchNow * 0.3, rr, cx, cy, rad);
        pts.push({ q, c: GOLD, base: 1.05, a: ringFade * 0.55 });
      }
    }

    pts.sort((a, b) => a.q.z - b.q.z);
    for (const { q, c, base, a } of pts) {
      const depth = (q.z + 1.5) / 3;                       // 0 at the back, 1 at the front
      const alpha = Math.max(0, (0.12 + 0.88 * depth) * a);
      const size = Math.max(0.35, base * (0.55 + 0.9 * depth) * q.s * Math.max(rad, 0.35));
      ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${alpha.toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(q.x, q.y, size, 0, Math.PI * 2);
      ctx.fill();
    }

    requestAnimationFrame(frame);
  }

  function start() {
    if (running) return;
    running = true; last = performance.now();
    requestAnimationFrame(frame);
  }
  start();

  // ------------------------------------------------------------------
  // public API
  // ------------------------------------------------------------------
  function burst(targetFn, onMidway) {
    if (mode !== "idle") return false;
    mode = "burst"; t0 = performance.now();
    getTarget = targetFn; onMid = onMidway; midFired = false; target = null;
    hoverAim = 0;
    intro.classList.add("away");
    return true;
  }

  function restore(tg) {
    target = tg; getTarget = null;
    canvas.style.opacity = 1;
    mode = "restore"; t0 = performance.now();
    start();
  }

  function hideNow() {
    mode = "gone";
    canvas.style.opacity = 0;
    intro.classList.add("away");
  }

  // the small spinning sphere drawn in SVG at the centre of the map
  function mini(group, radius) {
    const NS = "http://www.w3.org/2000/svg";
    const pts = fibonacci(150).map((p, i) => {
      const c = document.createElementNS(NS, "circle");
      const col = colourFor(p, i);
      c.setAttribute("fill", `rgb(${col[0]},${col[1]},${col[2]})`);
      c.setAttribute("class", "dot");
      group.append(c);
      return { ...p, el: c };
    });
    let a = yaw;
    let prev = performance.now(), seen = false;
    (function spin(now) {
      if (group.isConnected) seen = true;
      else if (seen) return;                          // stops once the map is cleared
      a += SPIN * 1.4 * Math.min(50, now - prev); prev = now;
      for (const p of pts) {
        const q = project(p, a, -0.28, radius, 0, 0);
        const depth = (q.z + 1) / 2;
        p.el.setAttribute("cx", q.x.toFixed(2));
        p.el.setAttribute("cy", q.y.toFixed(2));
        p.el.setAttribute("r", (0.7 + 1.1 * depth).toFixed(2));
        p.el.setAttribute("opacity", (0.15 + 0.85 * depth).toFixed(2));
      }
      requestAnimationFrame(spin);
    })(prev);
  }

  window.Orb = { burst, restore, hideNow, mini };
})();
