// 3D "physical twin": a clothed, animated person whose body shows the twin's state.
// Realistic mode: skin/clothes, face (blinks, chews), walking gait, eating, sleeping; a CGM patch on the arm blinks the glucose zone.
// X-ray mode: body turns glass and organs show model state (stomach = carbs on board, pancreas = insulin action, heart, liver, brain,
// glucose + absorbed-carb particles in the vessels).
import * as THREE from './vendor/three.module.min.js';

const ZONE = g => g < 70 ? 0x3b6fd4 : g <= 180 ? 0x2e9e5b : g <= 250 ? 0xe07a1f : 0xc9372c;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

export function initHuman(el) {
  const probe = document.createElement('canvas');
  if (!(probe.getContext('webgl2') || probe.getContext('webgl'))) throw new Error('WebGL is not available in this browser');
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  el.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(30, 1, 0.1, 50);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a96a6, 0.95));
  const key = new THREE.DirectionalLight(0xfff2e6, 1.5); key.position.set(2.5, 3.5, 3.5); scene.add(key);
  const rim = new THREE.DirectionalLight(0x9ec8ff, 0.8); rim.position.set(-3, 2, -3); scene.add(rim);

  const root = new THREE.Group(); scene.add(root);
  const body = new THREE.Group(); root.add(body);

  // ---------- materials
  const std = (color, rough = 0.65) => new THREE.MeshStandardMaterial({ color, roughness: rough, metalness: 0.02 });
  const M = { skin: std(0xc98f6b, 0.7), shirt: std(0x2f6f8f, 0.85), pants: std(0x2b3445, 0.85), shoe: std(0xf0f0f0, 0.6), hair: std(0x1b1512, 0.9),
              white: std(0xffffff, 0.3), dark: std(0x15110f, 0.4), lip: std(0x9a4a3c, 0.6) };
  const xr = [M.skin, M.shirt, M.pants, M.shoe, M.hair];   // become glass in X-ray mode
  const auraMat = new THREE.MeshBasicMaterial({ color: 0x2e9e5b, transparent: true, opacity: 0, side: THREE.BackSide, blending: THREE.AdditiveBlending, depthWrite: false });

  // ---------- geometry helpers (lathe profiles, bottom -> top)
  const lathe = pts => new THREE.LatheGeometry(pts.map(([r, y]) => new THREE.Vector2(r, y)), 28);
  const tube = (len, rTop, rBot, bulge = 0) => {      // hangs from y=0 down to y=-len, rounded ends
    const p = [[0.001, -len - rBot * 0.8], [rBot * 0.8, -len - rBot * 0.35]];
    for (let i = 0; i <= 10; i++) { const s = i / 10; p.push([rBot + (rTop - rBot) * s + bulge * Math.sin(Math.PI * s), -len + len * s]); }
    p.push([rTop * 0.8, rTop * 0.35], [0.001, rTop * 0.8]);
    return lathe(p);
  };
  const mesh = (geo, mat, parent, pos = [0, 0, 0]) => { const m = new THREE.Mesh(geo, mat); m.position.set(...pos); parent.add(m); return m; };
  const sph = (r, mat, parent, pos, sc = [1, 1, 1]) => { const m = mesh(new THREE.SphereGeometry(r, 24, 18), mat, parent, pos); m.scale.set(...sc); return m; };

  // ---------- torso (pelvis -> shoulders), clothed
  const torsoProfile = [[0.001, -0.03], [0.13, 0], [0.152, 0.08], [0.136, 0.2], [0.146, 0.32], [0.166, 0.42], [0.172, 0.5], [0.15, 0.56], [0.075, 0.6], [0.001, 0.62]];
  const torso = new THREE.Group(); torso.position.y = 0.9; body.add(torso);
  const tGeo = lathe(torsoProfile);
  const shirt = mesh(tGeo, M.shirt, torso); shirt.scale.set(1.2, 1, 0.66);
  const hips = mesh(lathe([[0.001, -0.06], [0.15, -0.02], [0.158, 0.06], [0.14, 0.14], [0.001, 0.16]]), M.pants, torso); hips.scale.set(1.2, 1, 0.7);
  const tAura = mesh(tGeo, auraMat, torso); tAura.scale.set(1.3, 1.02, 0.78);
  mesh(new THREE.CylinderGeometry(0.048, 0.056, 0.1, 18), M.skin, torso, [0, 0.64, 0]);   // neck

  // ---------- head + face
  const head = new THREE.Group(); head.position.set(0, 1.62, 0.005); body.add(head);
  const skull = sph(0.105, M.skin, head, [0, 0, 0], [0.93, 1.12, 1.0]);
  const hAura = sph(0.105, auraMat, head, [0, 0, 0], [1.05, 1.24, 1.12]);
  const hair = new THREE.Mesh(new THREE.SphereGeometry(0.112, 28, 20, 0, Math.PI * 2, 0, Math.PI * 0.5), M.hair); hair.scale.set(0.97, 1.13, 1.06); hair.position.set(0, 0.006, -0.012); hair.rotation.x = -0.4; head.add(hair);
  const eyes = new THREE.Group(); head.add(eyes);
  for (const sx of [-1, 1]) {
    const e = new THREE.Group(); e.position.set(sx * 0.039, 0.016, 0.093); eyes.add(e);
    sph(0.0145, M.white, e, [0, 0, 0], [1.2, 0.8, 0.6]); sph(0.0075, M.dark, e, [0, 0, 0.007]);
    mesh(new THREE.BoxGeometry(0.036, 0.006, 0.01), M.hair, e, [0, 0.026, 0.002]).rotation.z = -sx * 0.12;   // brow
    sph(0.012, M.skin, head, [sx * 0.098, -0.005, -0.005], [0.5, 1.2, 0.9]);   // ear
  }
  const nose = mesh(new THREE.ConeGeometry(0.013, 0.036, 12), M.skin, head, [0, -0.012, 0.108]); nose.rotation.x = Math.PI / 2 * 0.9;
  const mouth = mesh(new THREE.TorusGeometry(0.022, 0.0042, 8, 20, Math.PI), M.lip, head, [0, -0.052, 0.088]); mouth.rotation.z = Math.PI; mouth.scale.set(1, 0.7, 1);
  const mouthIn = sph(0.016, M.dark, head, [0, -0.056, 0.084], [1.2, 0.01, 0.4]);

  // ---------- limbs (pivot at joint)
  const arm = sx => {
    const sh = new THREE.Group(); sh.position.set(sx * 0.215, 1.46, 0); body.add(sh);
    sph(0.058, M.shirt, sh, [0, 0, 0]);
    mesh(tube(0.27, 0.05, 0.04), M.skin, sh);
    mesh(tube(0.17, 0.056, 0.05), M.shirt, sh, [0, 0.0, 0]);           // short sleeve
    const el = new THREE.Group(); el.position.set(0, -0.28, 0); sh.add(el);
    sph(0.04, M.skin, el, [0, 0, 0]);
    mesh(tube(0.24, 0.04, 0.03), M.skin, el);
    const hand = sph(0.03, M.skin, el, [0, -0.29, 0], [0.7, 1.4, 0.5]);
    sph(0.012, M.skin, el, [sx * -0.022, -0.265, 0.014]);                // thumb
    return { sh, el, hand };
  };
  const leg = sx => {
    const hp = new THREE.Group(); hp.position.set(sx * 0.088, 0.92, 0); body.add(hp);
    mesh(tube(0.43, 0.088, 0.062), M.pants, hp);
    const kn = new THREE.Group(); kn.position.set(0, -0.44, 0); hp.add(kn);
    sph(0.06, M.pants, kn, [0, 0, 0]);
    mesh(tube(0.40, 0.06, 0.044, 0.012), M.pants, kn);
    const ft = new THREE.Group(); ft.position.set(0, -0.46, 0); kn.add(ft);
    sph(0.045, M.shoe, ft, [0, -0.015, 0.04], [0.95, 0.65, 2.0]);
    return { hp, kn };
  };
  const aL = arm(-1), aR = arm(1), lL = leg(-1), lR = leg(1);

  // CGM patch on the left upper arm (LED blinks glucose zone)
  const patch = mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.008, 20), std(0xf5f5f5, 0.4), aL.sh, [-0.052, -0.14, 0.0]); patch.rotation.z = Math.PI / 2;
  const led = mesh(new THREE.SphereGeometry(0.006, 10, 8), new THREE.MeshBasicMaterial({ color: 0x2e9e5b }), aL.sh, [-0.058, -0.14, 0.0]);

  // ---------- organs (X-ray)
  const organs = new THREE.Group(); body.add(organs);
  const organ = (geo, color, pos, sc) => { const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.3, roughness: 0.5 })); m.position.set(...pos); if (sc) m.scale.set(...sc); organs.add(m); return m; };
  const stomach = organ(new THREE.SphereGeometry(0.06, 20, 20), 0xe07a1f, [0.07, 1.28, 0.04], [1.1, 1.3, 0.8]);
  const liver = organ(new THREE.SphereGeometry(0.075, 20, 20), 0x8c2f2f, [-0.09, 1.34, 0.03], [1.5, 0.8, 0.9]);
  const pancreas = organ(new THREE.CapsuleGeometry(0.018, 0.09, 6, 12), 0x0f8b8d, [0, 1.22, -0.02]); pancreas.rotation.z = Math.PI / 2;
  const heart = organ(new THREE.SphereGeometry(0.045, 20, 20), 0xc9372c, [0.05, 1.46, 0.04]);
  const brain = organ(new THREE.SphereGeometry(0.075, 20, 20), 0xd9a0c0, [0, 1.65, -0.005], [1, 0.9, 1.05]);
  const gut = organ(new THREE.CapsuleGeometry(0.03, 0.12, 6, 12), 0xd8b88a, [0, 1.07, 0.03], [1, 1, 1.2]); gut.rotation.z = Math.PI / 2;
  const V = (...p) => new THREE.CatmullRomCurve3(p.map(a => new THREE.Vector3(...a)));
  const curves = [V([.05, 1.46, .04], [0, 1.27, 0], [0, 1.02, 0], [-.088, .88, 0], [-.088, .45, 0], [-.088, .08, 0]),
    V([.05, 1.46, .04], [0, 1.27, 0], [0, 1.02, 0], [.088, .88, 0], [.088, .45, 0], [.088, .08, 0]),
    V([.05, 1.46, .04], [.18, 1.45, 0], [.215, 1.2, 0], [.22, .98, 0]), V([.05, 1.46, .04], [-.18, 1.45, 0], [-.215, 1.2, 0], [-.22, .98, 0]),
    V([.05, 1.46, .04], [.02, 1.56, 0], [0, 1.65, 0])];
  const absorb = V([.07, 1.28, .04], [0, 1.32, .05], [-.08, 1.34, .04], [0, 1.4, .05], [.05, 1.46, .04]);
  const mkPts = (n, color, size) => { const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    const p = new THREE.Points(g, new THREE.PointsMaterial({ color, size, transparent: true, opacity: 0.95, depthWrite: false, blending: THREE.AdditiveBlending })); organs.add(p); return p; };
  const N = 160, NA = 70, gl = mkPts(N, 0x2e9e5b, 0.026), ab = mkPts(NA, 0xffa040, 0.03);
  const gu = Array.from({ length: N }, (_, i) => ({ c: i % curves.length, u: Math.random() })), au = Array.from({ length: NA }, () => Math.random());

  // ---------- set: table + plate, bed + pillow + blanket, floor shadow, zzz
  const table = new THREE.Group(); root.add(table); table.visible = false;
  mesh(new THREE.BoxGeometry(0.9, 0.04, 0.5), std(0x9b7653, 0.7), table, [0.1, 0.93, 0.44]);
  for (const [x, z] of [[-.3, .22], [.5, .22], [-.3, .66], [.5, .66]]) mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.93, 8), std(0x7d5d40), table, [x, 0.465, z]);
  mesh(new THREE.CylinderGeometry(0.13, 0.1, 0.02, 28), M.white, table, [0.1, 0.96, 0.3]);
  const food = sph(0.06, std(0xe8c47a, 0.8), table, [0.1, 0.985, 0.3], [1, 0.5, 1]);
  const bed = new THREE.Group(); root.add(bed); bed.visible = false;
  mesh(new THREE.BoxGeometry(1.0, 0.22, 2.2), std(0xdfe6ee, 0.8), bed, [0, 0.11, 0]);
  mesh(new THREE.BoxGeometry(0.5, 0.1, 0.32), M.white, bed, [0, 0.27, -0.76]);
  const blanket = mesh(new THREE.BoxGeometry(0.92, 0.1, 1.15), new THREE.MeshStandardMaterial({ color: 0x6b8fc4, roughness: 0.9, transparent: true, opacity: 0.96 }), bed, [0, 0.42, 0.35]);
  const shadowTex = (() => { const c = document.createElement('canvas'); c.width = c.height = 128; const g = c.getContext('2d'), gr = g.createRadialGradient(64, 64, 4, 64, 64, 64); gr.addColorStop(0, 'rgba(0,0,0,.35)'); gr.addColorStop(1, 'rgba(0,0,0,0)'); g.fillStyle = gr; g.fillRect(0, 0, 128, 128); return new THREE.CanvasTexture(c); })();
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(1.3, 1.3), new THREE.MeshBasicMaterial({ map: shadowTex, transparent: true, depthWrite: false })); shadow.rotation.x = -Math.PI / 2; shadow.position.y = 0.004; root.add(shadow);
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.62, 0.65, 64), new THREE.MeshBasicMaterial({ color: 0x2e9e5b, side: THREE.DoubleSide, transparent: true, opacity: 0.8 })); ring.rotation.x = -Math.PI / 2; ring.position.y = 0.01; root.add(ring);
  const zTex = (() => { const c = document.createElement('canvas'); c.width = c.height = 64; const g = c.getContext('2d'); g.font = 'bold 48px sans-serif'; g.fillStyle = '#5a6b85'; g.fillText('Z', 14, 50); return new THREE.CanvasTexture(c); })();
  const zs = [0, 1, 2].map(i => { const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: zTex, transparent: true })); s.visible = false; root.add(s); return s; });

  // ---------- interaction + state
  let drag = null, userYaw = 0, xray = false;
  renderer.domElement.addEventListener('pointerdown', e => drag = e.clientX);
  addEventListener('pointerup', () => drag = null);
  addEventListener('pointermove', e => { if (drag != null) { userYaw += (e.clientX - drag) * 0.01; drag = e.clientX; } });
  const S = { g: 120, cob: 0, x: 0, risk: 0, steps: 0, hour: 12, meal: -1, alert: false };
  let eatUntil = 0, lastMeal = -1, phase = 0, sleepK = 0, walkK = 0, eatK = 0, last = performance.now(), nextBlink = 2, blinkT = -1;
  function setXray(on) {
    xray = on; organs.visible = on;
    xr.forEach(m => { m.transparent = on; m.opacity = on ? 0.13 : 1; m.depthWrite = !on; m.needsUpdate = true; });
  }
  setXray(false);
  function resize() { renderer.setSize(el.clientWidth, el.clientHeight); cam.aspect = el.clientWidth / el.clientHeight; cam.updateProjectionMatrix(); }
  new ResizeObserver(resize).observe(el); resize();

  function frame(now) {
    const dt = Math.min((now - last) / 1000, 0.05), t = now / 1000; last = now;
    try {
      const sleeping = (S.hour >= 23 || S.hour < 6) && S.steps < 5;
      if (S.meal !== lastMeal) { if (S.meal > lastMeal && lastMeal !== -1 && !sleeping) eatUntil = t + 6; lastMeal = S.meal; }
      const eating = t < eatUntil, walking = !eating && !sleeping && S.steps > 25;
      const ease = (a, b) => a + (b - a) * clamp(dt * 5, 0, 1);
      sleepK = ease(sleepK, sleeping ? 1 : 0); walkK = ease(walkK, walking ? clamp(S.steps / 100, 0.35, 1) : 0); eatK = ease(eatK, eating ? 1 : 0);
      const idle = 1 - Math.max(walkK, eatK, sleepK);
      phase += dt * (2.2 + 4.2 * walkK);
      const w = walkK, sp = Math.sin(phase), cp = Math.cos(phase);

      // gait: thighs swing, knees flex in swing phase, arms counter-swing, torso counter-rotates, body bobs
      lL.hp.rotation.x = -0.5 * w * sp + 0.02 * idle * Math.sin(t * 0.7); lR.hp.rotation.x = 0.5 * w * sp;
      lL.kn.rotation.x = 0.85 * w * Math.max(0, cp) + 0.04 * sleepK; lR.kn.rotation.x = 0.85 * w * Math.max(0, -cp) + 0.04 * sleepK;
      aL.sh.rotation.x = 0.5 * w * sp; aR.sh.rotation.x = -0.5 * w * sp;
      aL.sh.rotation.z = 0.08 + 0.02 * Math.sin(t * 1.3) * idle; aR.sh.rotation.z = -0.08 - 0.02 * Math.sin(t * 1.3) * idle;
      aL.el.rotation.x = -0.2 - 0.5 * w * (0.5 + 0.5 * sp); aR.el.rotation.x = -0.2 - 0.5 * w * (0.5 - 0.5 * sp);
      // eating: poses solved by inverse kinematics offline (hand centre reaches the mouth / the plate); bite cycles between them
      const cyc = 0.5 + 0.5 * Math.sin(t * 2.0), bite = cyc * cyc * (3 - 2 * cyc);       // smoothstep: 0 = at plate, 1 = at mouth
      const mix = (a, b) => a + (b - a) * bite, ek = eatK;
      aR.sh.rotation.set(aR.sh.rotation.x * (1 - ek) + ek * mix(-0.401, -1.287), ek * mix(-0.036, -0.799), aR.sh.rotation.z * (1 - ek) + ek * mix(-0.203, -0.375));
      aR.el.rotation.x = aR.el.rotation.x * (1 - ek) + ek * mix(-0.354, -2.265);
      aL.sh.rotation.set(aL.sh.rotation.x * (1 - ek) + ek * -0.401, ek * 0.036, aL.sh.rotation.z * (1 - ek) + ek * 0.203);   // left hand rests by the plate
      aL.el.rotation.x = aL.el.rotation.x * (1 - ek) + ek * -0.354;
      torso.rotation.y = 0.14 * w * sp; torso.rotation.x = 0.05 * eatK + 0.01 * Math.sin(t * 1.5) * idle;
      head.rotation.y = -0.1 * w * sp + 0.12 * Math.sin(t * 0.35) * idle; head.rotation.x = 0.06 * eatK * (0.4 + 0.6 * bite) + 0.03 * Math.sin(t * 0.9) * idle;
      torso.scale.set(1 + 0.012 * Math.sin(t * 1.7), 1, 1 + 0.02 * Math.sin(t * 1.7));            // breathing
      body.position.y = 0.022 * w * Math.abs(cp) + 0.28 * sleepK + 0.01 * Math.sin(t * 1.7) * sleepK;
      body.position.z = 0.85 * sleepK + 0.0;
      body.position.x = 0.012 * Math.sin(t * 0.5) * idle;       // weight shift
      body.rotation.x = -sleepK * (Math.PI / 2);
      // face: blink, chew, smile
      if (t > nextBlink && blinkT < 0) blinkT = t;
      const bl = blinkT >= 0 ? (t - blinkT < 0.14 ? 1 - Math.sin((t - blinkT) / 0.14 * Math.PI) : (blinkT = -1, nextBlink = t + 2 + Math.random() * 3, 1)) : 1;
      eyes.scale.y = clamp(sleepK > 0.5 ? 0.08 : bl, 0.08, 1);
      const chew = eatK * (bite > 0.85 ? 0.9 : 0.15 + 0.2 * Math.sin(t * 9));
      mouthIn.scale.y = 0.01 + 0.35 * chew; mouth.scale.y = 0.7 - 0.2 * chew;
      table.visible = eatK > 0.15; table.scale.setScalar(clamp(eatK * 1.2, 0.01, 1)); food.scale.set(1 - 0.35 * (1 - bite) * eatK, 0.5, 1 - 0.35 * (1 - bite) * eatK);
      bed.visible = sleepK > 0.1; shadow.visible = sleepK < 0.5;
      zs.forEach((z, i) => { z.visible = sleepK > 0.8; const u = ((t * 0.4 + i / 3) % 1); z.position.set(0.15 + u * 0.3, 0.9 + u * 0.6, -0.6); z.scale.setScalar(0.12 + 0.12 * u); z.material.opacity = 1 - u; });
      root.rotation.y = userYaw + Math.sin(t * 0.22) * 0.3 + sleepK * 1.15;
      cam.position.set(0, 1.12 + 0.55 * sleepK, 4.6 + 1.2 * sleepK); cam.lookAt(0, 0.86 - 0.1 * sleepK, 0);

      // state -> indicators
      const zc = ZONE(S.g);
      led.material.color.setHex(zc); led.scale.setScalar(1 + 0.5 * (Math.sin(t * 4) > 0.6 ? 1 : 0));
      ring.material.color.setHex(zc); ring.material.opacity = 0.55 + 0.35 * (S.alert ? Math.sin(t * 6) : 0);
      auraMat.color.setHex(zc); auraMat.opacity = xray ? 0.2 + 0.25 * clamp(S.risk, 0, 1) + (S.alert ? 0.12 * (1 + Math.sin(t * 6)) : 0) : 0;
      if (xray) {
        const ins = clamp(S.x / 15, 0, 1);
        heart.scale.setScalar(1 + 0.18 * Math.max(0, Math.sin(t * Math.PI * 2 * (62 + S.steps * 0.5) / 60)) ** 3);
        stomach.material.emissiveIntensity = 0.15 + 1.6 * clamp(S.cob / 70, 0, 1); stomach.scale.set(1.1, 1.3, 0.8).multiplyScalar(1 + 0.25 * clamp(S.cob / 80, 0, 1));
        pancreas.material.emissiveIntensity = 0.2 + 1.8 * ins * (0.6 + 0.4 * Math.sin(t * 3)); pancreas.scale.setScalar(1 + 0.2 * ins);
        brain.material.emissiveIntensity = S.g < 70 ? 0.2 + 0.6 * Math.abs(Math.sin(t * 5)) : 0.4; liver.material.emissiveIntensity = 0.3 + 0.5 * ins;
        gl.geometry.setDrawRange(0, Math.round(N * clamp((S.g - 40) / 260, 0.1, 1))); gl.material.color.setHex(zc);
        const pos = gl.geometry.attributes.position, v = new THREE.Vector3();
        gu.forEach((p, i) => { p.u += dt * 0.12 * (1 + ins); if (p.u > 1) p.u = 0; curves[p.c].getPoint(p.u, v); pos.setXYZ(i, v.x + Math.sin(i) * 0.012, v.y, v.z + Math.cos(i) * 0.012); }); pos.needsUpdate = true;
        ab.geometry.setDrawRange(0, Math.round(NA * clamp(S.cob / 50, 0, 1)));
        const pa = ab.geometry.attributes.position;
        au.forEach((u, i) => { au[i] = (u + dt * 0.35) % 1; absorb.getPoint(au[i], v); pa.setXYZ(i, v.x + Math.sin(i * 3) * 0.015, v.y, v.z + 0.02); }); pa.needsUpdate = true;
      }
      renderer.render(scene, cam);
    } catch (e) { console.error('human frame', e); return; }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  return { update: s => Object.assign(S, s), setXray };
}
