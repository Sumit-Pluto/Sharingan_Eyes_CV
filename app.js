// Sharingan Eyes — browser version.
// YOLO's best.pt cannot run in a browser, so pupil/iris detection uses
// MediaPipe Face Landmarker (478 landmarks incl. iris centres) running
// locally via WASM/WebGPU. Blink = eyeBlink blendshapes. Everything else
// (stages, rotation, sound, blink-to-evolve) mirrors sharingan.py.

// MediaPipe is imported dynamically inside initLandmarker(): a static
// top-level import that fails (no internet, CDN down) would kill the
// whole module and leave Start Camera dead with no message at all.
const MEDIAPIPE_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

const ASSET_SIZE = 256;          // assets are downscaled once: fast rotation
const BG_TOLERANCE = 40;
const BLINK_SCORE = 0.55;        // eyeBlink blendshape threshold
const MIN_CLOSED_FRAMES = 2;
const MIN_CLOSED_MS = 350;   // eyes must STAY shut this long to evolve
const COOLDOWN_MS = 500;
const ROTATION_MS = 2400;
const OPACITY = 0.9;   // near-opaque: the real pupil must not bleed through
const MAX_STAGE_PROBE = 20;      // assets are probed as 1.png..20.png

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const stageLabel = document.getElementById("stageLabel");
const fpsLabel = document.getElementById("fpsLabel");
const errorEl = document.getElementById("error");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const prevBtn = document.getElementById("prevBtn");
const nextBtn = document.getElementById("nextBtn");
// One Euro filter (Casiez et al.): adaptive low-pass that is heavy
// when still (kills jitter and hand-shake tremor) and light when
// moving fast (no lag on a real head/eye move). Tuned for phones,
// whose hands add an ~8 Hz tremor on top of landmark noise: a low
// resting cutoff (1.1) rejects that tremor, while a large beta (18)
// opens the filter right up the moment the filtered velocity says
// the movement is real. Two guards sit in front of it: a deadband
// holds the output perfectly still under sub-pixel noise, and a
// max-speed clamp stops one-frame landmark spikes from yanking the
// eye across the face. Positions are then drawn with a short,
// time-based velocity prediction in loop() (see latestEyes), which
// — not frame-count guesswork — is what cancels detector latency.
class OneEuro {
  // Coordinates are normalised (0..1), so velocities are numerically
  // small: beta must be large or the filter stays in its slow regime
  // during fast pupil moves and the overlay trails behind.
  constructor(minCutoff = 1.1, beta = 18.0, dCutoff = 1.0,
              { deadband = 0.0006, maxSpeed = 1.7 } = {}) {
    this.minCutoff = minCutoff; this.beta = beta; this.dCutoff = dCutoff;
    this.deadband = deadband; this.maxSpeed = maxSpeed;
    this.xPrev = null; this.dxPrev = 0; this.tPrev = null;
  }
  static alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }
  filter(x, t) {
    if (this.xPrev === null) { this.xPrev = x; this.tPrev = t; return x; }
    // Clamp dt: a dropped frame must not make the filter leap, and
    // a duplicate timestamp must not divide by ~zero.
    const dt = Math.min(0.1, Math.max(1 / 240, t - this.tPrev));
    this.tPrev = t;
    // Outlier clamp: the iris cannot cross the face in one frame.
    let target = x;
    const maxStep = this.maxSpeed * dt;
    if (Math.abs(target - this.xPrev) > maxStep) {
      target = this.xPrev + Math.sign(target - this.xPrev) * maxStep;
    }
    const dx = (target - this.xPrev) / dt;
    const aD = OneEuro.alpha(this.dCutoff, dt);
    const dxHat = aD * dx + (1 - aD) * this.dxPrev;
    this.dxPrev = dxHat;
    // Deadband: under micro-noise (still face, shaking hand) hold
    // exactly still instead of trembling around the true position.
    if (Math.abs(target - this.xPrev) < this.deadband
        && Math.abs(dxHat) < 0.05) return this.xPrev;
    const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
    const a = OneEuro.alpha(cutoff, dt);
    this.xPrev = a * target + (1 - a) * this.xPrev;
    return this.xPrev;
  }
}
let eyeFilters = [];     // per-eye OneEuro sets {x, y, r, gx, gy, ...}
let prevFiltered = [];   // last filtered centres, for velocity
let prevFilteredAt = 0;  // timestamp (s) of prevFiltered
let latestEyes = [];     // last filtered eyes, drawn every rAF
let latestEyeAt = 0;     // performance.now() of the last detection
let eyeVelocities = [];  // per-eye smoothed velocity (normalised/s)
let eyeLatencyMs = 50;   // smoothed detection time, for prediction
let lastDrawAt = 0;      // last overlay draw, for draw throttling
let headYaw = 0;         // from the facial transformation matrix
let earBaseline = null;  // personal open-eye EAR, calibrated at start
let earSamples = [];
const IS_TOUCH = window.matchMedia?.("(pointer: coarse)")?.matches ?? false;

let landmarker = null;
let stream = null;
let running = false;
let rafId = null;
let lastVideoTime = -1;
let stages = [];        // processed canvases, background already stripped
let audioFiles = [];
let currentIndex = 0;
let lastTransition = 0;
let rotationStart = 0;
let awakeningStart = 0;   // when the current stage awakened (glow flare)
let zoomOrigin = "50% 50%";
let bloodStart = 0;       // when an Amaterasu stage (8/9) became active
let lightFactor = 1;      // overlay brightness matched to the webcam
const sampleCanvas = document.createElement("canvas");
sampleCanvas.width = 64; sampleCanvas.height = 36;
const sampleCtx = sampleCanvas.getContext("2d", { willReadFrequently: true });
const noiseCanvas = (() => {   // webcam-like grain, tiled inside the disc
  const c = document.createElement("canvas");
  c.width = 128; c.height = 128;
  const g = c.getContext("2d");
  const data = g.createImageData(128, 128);
  for (let i = 0; i < data.data.length; i += 4) {
    const v = 110 + Math.random() * 90;
    data.data[i] = data.data[i + 1] = data.data[i + 2] = v;
    data.data[i + 3] = 255;
  }
  g.putImageData(data, 0, 0);
  return c;
})();

const fibreCanvas = (() => {   // radial iris fibres, rotates with the eye
  const c = document.createElement("canvas");
  c.width = 256; c.height = 256;
  const g = c.getContext("2d");
  g.translate(128, 128);
  for (let i = 0; i < 220; i++) {
    const a = Math.random() * Math.PI * 2;
    const r0 = 18 + Math.random() * 20;
    const r1 = 90 + Math.random() * 36;
    g.strokeStyle = Math.random() > 0.5
      ? "rgba(60, 0, 0, 0.55)" : "rgba(255, 120, 90, 0.40)";
    g.lineWidth = 1 + Math.random();
    g.beginPath();
    g.moveTo(Math.cos(a) * r0, Math.sin(a) * r0);
    g.lineTo(Math.cos(a + 0.05) * r1, Math.sin(a + 0.05) * r1);
    g.stroke();
  }
  return c;
})();

let fxReduced = IS_TOUCH;   // phones start lite: grain/ripples cost frames

function updateLightFactor() {
  // Average luminance of the live feed, sampled cheaply every so
  // often, so the clean vector asset is dimmed/warmed to live in the
  // same photo as the grainy webcam face.
  try {
    sampleCtx.drawImage(video, 0, 0, 64, 36);
    const d = sampleCtx.getImageData(0, 0, 64, 36).data;
    let sum = 0, n = 0;
    for (let i = 0; i < d.length; i += 16) {
      sum += 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];
      n++;
    }
    const avg = n ? (sum / n) / 255 : 0.5;
    const target = 0.92 + (avg - 0.45) * 0.5;   // gentle, not a slave
    lightFactor += (Math.min(1.12, Math.max(0.72, target)) - lightFactor) * 0.2;
  } catch { /* video not ready yet */ }
}
const AMATERASU_STAGES = new Set([7, 8]);  // assets 8.png / 9.png (Sasuke)
let eyesWereOpen = true;
let closedFrames = 0;
let closedSince = null;   // timestamp the current closure began
let frames = 0;
let fpsTimer = performance.now();

// --- Loading screen: pixel Naruto runners (Google-Naruto style) -----
// Sprite is drawn from a pixel map in code (no extra asset to load);
// two leg frames are swapped while the runners cross the screen.
const SPRITE_HEAD = [
  "....Y..Y..Y.....",
  "...YYYYYYYY.....",
  "...YYYYYYYY.....",
  "...YSSSSSSY.....",
  "....SKSSKS......",
  "....SSSSSS......",
  ".....SSSS.......",
  "..O.OOOOOO......",
  ".OO.OOOOOOO.....",
  ".S...OOOOO......",
  "....OOOOOO......",
];
const SPRITE_LEGS_A = [
  "...OOO..........",
  "..OO..OOO.......",
  ".OO.....OOO.....",
  "UU........UUU...",
  "................",
];
const SPRITE_LEGS_B = [
  "....OOO.........",
  ".....OOOO.......",
  "..OO...OOO......",
  ".UU......UUU....",
  "................",
];
const SPRITE_COLOURS = {
  Y: "#ffd21f", S: "#f6c48f", K: "#20242c", O: "#f07d18", U: "#2456a6",
};

function spriteDataUrl(legs) {
  const rows = [...SPRITE_HEAD, ...legs];
  const px = 4;
  const c = document.createElement("canvas");
  c.width = 16 * px; c.height = rows.length * px;
  const g = c.getContext("2d");
  rows.forEach((row, y) => {
    [...row].forEach((ch, x) => {
      if (SPRITE_COLOURS[ch]) {
        g.fillStyle = SPRITE_COLOURS[ch];
        g.fillRect(x * px, y * px, px, px);
      }
    });
  });
  return c.toDataURL();
}

const loadingEl = document.getElementById("loading");
const loadingTextEl = document.getElementById("loadingText");
const runnersEl = document.getElementById("runners");
let runnerTimer = null;

function showLoading(text) {
  loadingTextEl.textContent = text;
  if (!loadingEl.hidden) return;
  loadingEl.hidden = false;
  runnersEl.innerHTML = "";
  const frames = [spriteDataUrl(SPRITE_LEGS_A), spriteDataUrl(SPRITE_LEGS_B)];
  const runners = [];
  for (let i = 0; i < 6; i++) {
    const el = document.createElement("div");
    el.className = "runner";
    const size = 44 + (i % 3) * 14;
    el.style.width = `${size}px`;
    el.style.height = `${size}px`;
    el.style.top = `${(i % 3) * 34 + 4}px`;
    el.style.backgroundImage = `url(${frames[0]})`;
    el.style.animationDuration = `${2.6 + i * 0.55}s, 0.36s`;
    el.style.animationDelay = `${-i * 0.9}s, 0s`;
    runnersEl.appendChild(el);
    runners.push(el);
  }
  let frame = 0;
  runnerTimer = setInterval(() => {
    frame = 1 - frame;
    runners.forEach((el) => {
      el.style.backgroundImage = `url(${frames[frame]})`;
    });
  }, 180);
}

function hideLoading() {
  loadingEl.hidden = true;
  if (runnerTimer) { clearInterval(runnerTimer); runnerTimer = null; }
}

function setStatus(text) {
  statusEl.textContent = text;
  if (!loadingEl.hidden) loadingTextEl.textContent = text;
}
function setError(text) {
  errorEl.textContent = text || "";
  if (text) setStatus(text);   // also on the video overlay: always visible
}
window.addEventListener("error", (e) => setError(`Error: ${e.message}`));
window.addEventListener("unhandledrejection", (e) =>
  setError(`Error: ${e.reason?.message || e.reason}`));

// --- Asset loading -------------------------------------------------
// Best path: optimized/manifest.json (written by optimized/optimize.py)
// lists exactly what exists: one fetch, zero 404s. Fallback: probe
// 1.png, 2.png, ... and stop at the first missing number (assets are
// consecutively numbered), instead of blindly probing up to 20.
let manifestCache = null;
async function loadManifest() {
  if (manifestCache !== null) return manifestCache;
  try {
    const r = await fetch("optimized/manifest.json", { cache: "no-store" });
    manifestCache = r.ok ? await r.json() : false;
  } catch { manifestCache = false; }
  return manifestCache;
}

async function probeFiles(makeUrl) {
  const found = [];
  for (let i = 1; i <= MAX_STAGE_PROBE; i++) {
    const url = makeUrl(i);
    let ok = false;
    try {
      ok = (await fetch(url, { method: "HEAD", cache: "no-store" })).ok;
    } catch {}
    if (!ok) break;            // consecutive numbering: stop at first gap
    found.push(url);
  }
  return found;
}

async function stageUrls() {
  const manifest = await loadManifest();
  if (manifest?.assets?.length) {
    return manifest.assets.map((a) => `optimized/assets/${a.file}`);
  }
  const optimized = await probeFiles((i) => `optimized/assets/${i}.png`);
  return optimized.length ? optimized : probeFiles((i) => `assets/${i}.png`);
}

async function audioUrls(stageCount) {
  if (audioFiles.length) return audioFiles;   // probe at most once
  const manifest = await loadManifest();
  if (manifest?.audio?.length) {
    return manifest.audio.map((name) => `optimized/audio/${name}`);
  }
  if (manifest?.assets?.length) {
    // Older manifests: audio shares the stage numbering (1.mp3..N.mp3).
    return manifest.assets.map((a) =>
      `optimized/audio/${a.file.replace(/\.png$/i, ".mp3")}`);
  }
  const optimized = await probeFiles((i) => `optimized/audio/${i}.mp3`);
  return optimized.length ? optimized : probeFiles((i) => `audio/${i}.mp3`);
}

// Downscale to ASSET_SIZE and strip a baked-in background (same fix as
// the Python strip_opaque_background): opaque corners -> pixels matching
// the corner colour and connected to the border become transparent.
function processAsset(img) {
  const c = document.createElement("canvas");
  c.width = ASSET_SIZE; c.height = ASSET_SIZE;
  const g = c.getContext("2d", { willReadFrequently: true });
  g.drawImage(img, 0, 0, ASSET_SIZE, ASSET_SIZE);
  const data = g.getImageData(0, 0, ASSET_SIZE, ASSET_SIZE);
  const px = data.data;
  const cornerAlpha = px[3];
  if (cornerAlpha < 128) {           // already transparent: clear hidden RGB
    for (let p = 0; p < px.length; p += 4) {
      if (px[p + 3] === 0) { px[p] = px[p + 1] = px[p + 2] = 0; }
    }
    g.putImageData(data, 0, 0);
    return c;
  }
  const cr = px[0], cg = px[1], cb = px[2];
  const similar = new Uint8Array(ASSET_SIZE * ASSET_SIZE);
  for (let p = 0, q = 0; p < px.length; p += 4, q++) {
    if (Math.max(Math.abs(px[p] - cr), Math.abs(px[p + 1] - cg),
                 Math.abs(px[p + 2] - cb)) <= BG_TOLERANCE) similar[q] = 1;
  }
  // Flood fill from the borders over similar pixels.
  const seen = new Uint8Array(ASSET_SIZE * ASSET_SIZE);
  const stack = [];
  for (let x = 0; x < ASSET_SIZE; x++) {
    stack.push(x, (ASSET_SIZE - 1) * ASSET_SIZE + x);
  }
  for (let y = 0; y < ASSET_SIZE; y++) {
    stack.push(y * ASSET_SIZE, y * ASSET_SIZE + ASSET_SIZE - 1);
  }
  while (stack.length) {
    const q = stack.pop();
    if (seen[q] || !similar[q]) continue;
    seen[q] = 1;
    const x = q % ASSET_SIZE, y = (q / ASSET_SIZE) | 0;
    if (x > 0) stack.push(q - 1);
    if (x < ASSET_SIZE - 1) stack.push(q + 1);
    if (y > 0) stack.push(q - ASSET_SIZE);
    if (y < ASSET_SIZE - 1) stack.push(q + ASSET_SIZE);
  }
  for (let q = 0; q < seen.length; q++) {
    if (seen[q]) { px[q * 4 + 3] = 0; px[q * 4] = px[q * 4 + 1] = px[q * 4 + 2] = 0; }
  }
  g.putImageData(data, 0, 0);
  return c;
}

function bakeFeather(sprite) {
  // Pre-render the feathered rim into the sprite once at load, so
  // the per-frame loop just blits a finished disc instead of
  // building an erase-gradient for every eye every frame.
  const g = sprite.getContext("2d");
  const s = sprite.width;
  g.save();
  g.globalCompositeOperation = "destination-out";
  const grad = g.createRadialGradient(s / 2, s / 2, s * 0.36, s / 2, s / 2, s / 2);
  grad.addColorStop(0, "rgba(0, 0, 0, 0)");
  grad.addColorStop(0.8, "rgba(0, 0, 0, 0)");
  grad.addColorStop(1, "rgba(0, 0, 0, 0.9)");
  g.fillStyle = grad;
  g.fillRect(0, 0, s, s);
  g.restore();
  return sprite;
}

async function loadStages() {
  const urls = await stageUrls();
  const loaded = [];
  // No cache-busting query: stage URLs are stable, so the browser and
  // the CDN can cache them (matters on mobile data). Fresh deploys
  // revalidate via Cache-Control instead.
  for (const url of urls) {
    const img = new Image();
    img.src = url;
    try { await img.decode(); } catch { continue; }
    loaded.push(bakeFeather(processAsset(img)));
  }
  return loaded;
}

async function initLandmarker() {
  let vision;
  try {
    vision = await import(MEDIAPIPE_URL);
  } catch {
    throw new Error("Could not download the MediaPipe library. "
      + "Check your internet connection and refresh.");
  }
  const { FaceLandmarker, FilesetResolver } = vision;
  let fileset;
  try {
    fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  } catch {
    throw new Error("Could not download the MediaPipe WASM files. "
      + "Check your internet connection and refresh.");
  }
  const make = (delegate) => FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true,
    // Phones skip the pose matrix (only a cosmetic squash input):
    // less work per detection means detections land sooner.
    outputFacialTransformationMatrixes: !IS_TOUCH,
  });
  try { return await make("GPU"); }
  catch {
    try { return await make("CPU"); }
    catch {
      throw new Error("Could not load the face model file. "
        + "Check your internet connection and refresh.");
    }
  }
}

const RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155,
                           133, 173, 157, 158, 159, 160, 161, 246];
const LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249,
                          263, 466, 388, 387, 386, 385, 384, 398];

function irisEye(landmarks, w, h, centreIdx, ringStart, upperIdx, lowerIdx,
                 contourIdx, side, cornerA, cornerB) {
  // Pupil/iris centre (468 right, 473 left), radius from the iris ring
  // in real pixels (landmark x/y are normalised differently, so mixing
  // them unscaled distorts the size), plus how open the eye is: the
  // eyelid gap divided by the iris diameter. Squinting shrinks the
  // gap, so the overlay contracts with the eye, like the YOLO pupil
  // box did; opening wide expands it back.
  if (landmarks.length <= centreIdx) return null;
  const c = landmarks[centreIdx];
  let rPx = 0;
  for (let i = ringStart; i < ringStart + 4; i++) {
    const p = landmarks[i];
    rPx = Math.max(rPx, Math.hypot((p.x - c.x) * w, (p.y - c.y) * h));
  }
  const gapPx = Math.abs(landmarks[upperIdx].y - landmarks[lowerIdx].y) * h;
  const openness = Math.min(1, Math.max(0.12, gapPx / Math.max(1, 2 * rPx)));
  // Eyeball foreshortening: the visible iris is an ellipse, wider
  // than tall when facing forward and narrowing sideways as the
  // gaze/face turns. squash = visible width / height of the iris ring.
  const ring = [];
  for (let i = ringStart; i < ringStart + 4; i++) ring.push(landmarks[i]);
  const ringW = (Math.max(...ring.map((p) => p.x)) - Math.min(...ring.map((p) => p.x))) * w;
  const ringH = (Math.max(...ring.map((p) => p.y)) - Math.min(...ring.map((p) => p.y))) * h;
  const squash = Math.min(1, Math.max(0.55, ringW / Math.max(1, ringH)));
  // The eye's almond outline, used to clip the overlay so the eyelids
  // hide the top/bottom of the circle like they hide the real iris.
  const contour = contourIdx.map((i) => ({ x: landmarks[i].x, y: landmarks[i].y }));
  // Gaze offset: iris centre relative to the eye-corner midpoint,
  // roughly -1..1 per axis. Used for the corneal-refraction shift:
  // the visible iris appears displaced toward the gaze direction.
  const midX = (landmarks[cornerA].x + landmarks[cornerB].x) / 2;
  const midY = (landmarks[cornerA].y + landmarks[cornerB].y) / 2;
  const eyeW = Math.max(1e-4, Math.abs(landmarks[cornerA].x - landmarks[cornerB].x));
  const gx = Math.max(-1, Math.min(1, (c.x - midX) / (eyeW / 2)));
  const gy = Math.max(-1, Math.min(1, (c.y - midY) / (eyeW / 2)));
  return { x: c.x, y: c.y, r: rPx / w, openness, contour, side, squash, gx, gy };
}

function eyeCentres(landmarks, w, h) {
  // Landmark space is unmirrored; the video is shown mirrored, so the
  // caller flips x. Upper/lower eyelid: 159/145 right, 386/374 left.
  const eyes = [];
  const right = irisEye(landmarks, w, h, 468, 469, 159, 145, RIGHT_EYE_CONTOUR, "right", 33, 133);
  const left = irisEye(landmarks, w, h, 473, 474, 386, 374, LEFT_EYE_CONTOUR, "left", 362, 263);
  if (right) eyes.push(right);
  if (left) eyes.push(left);
  return eyes;
}

function distPx(a, b, w, h) {
  return Math.hypot((a.x - b.x) * w, (a.y - b.y) * h);
}

function eyeEAR(landmarks, w, h, idx) {
  // Eye Aspect Ratio: vertical eyelid distances over eye width.
  const [c1, c2, u1, u2, l1, l2] = idx.map((i) => landmarks[i]);
  const width = distPx(c1, c2, w, h);
  if (width < 1) return 1;
  return (distPx(u1, l1, w, h) + distPx(u2, l2, w, h)) / (2 * width);
}

const EAR_IDX = {
  right: [33, 133, 159, 158, 145, 153],
  left: [362, 263, 386, 385, 374, 380],
};

function averageEAR(landmarks, w, h) {
  if (!landmarks || landmarks.length < 387) return null;
  return (eyeEAR(landmarks, w, h, EAR_IDX.right)
        + eyeEAR(landmarks, w, h, EAR_IDX.left)) / 2;
}

function blinkClosed(result, ear) {
  const cats = result.faceBlendshapes?.[0]?.categories || [];
  let l = 0, r = 0;
  for (const c of cats) {
    if (c.categoryName === "eyeBlinkLeft") l = c.score;
    if (c.categoryName === "eyeBlinkRight") r = c.score;
  }
  if (l > BLINK_SCORE && r > BLINK_SCORE) return true;
  // Personal threshold: EAR falling under 65% of YOUR calibrated
  // open-eye baseline counts as closed, whatever your eye shape.
  if (earBaseline && ear !== null) return ear < earBaseline * 0.65;
  return false;
}

function playStageAudio() {
  if (!audioFiles.length) return;
  const url = audioFiles[currentIndex % audioFiles.length];
  new Audio(url).play().catch(() => {});
}

function awakeningEnergy(now) {
  // 1 at the moment of awakening, decaying to 0 over the spin window.
  const p = (now - awakeningStart) / ROTATION_MS;
  if (p >= 1 || awakeningStart === 0) return 0;
  return Math.pow(1 - Math.max(0, p), 2);
}

function setStage(index, now) {
  currentIndex = ((index % stages.length) + stages.length) % stages.length;
  rotationStart = now;
  awakeningStart = now;
  // Black blood (Amaterasu) only flows on Sasuke's stages 8 and 9:
  // stage 8 bleeds after 0.5s, stage 9 after 1.5s.
  bloodStart = AMATERASU_STAGES.has(currentIndex)
    ? now + (currentIndex === 8 ? 1500 : 500)
    : 0;
  lastTransition = now;
  stageLabel.textContent = `Stage ${currentIndex + 1} / ${stages.length}`;
  playStageAudio();
}

function tryTransition(now) {
  if (now - lastTransition < COOLDOWN_MS) return;
  setStage(currentIndex + 1, now);
}

function snapshot() {
  // Composite mirrored video + overlay into a downloadable PNG.
  const out = document.createElement("canvas");
  out.width = canvas.width; out.height = canvas.height;
  const g = out.getContext("2d");
  g.translate(out.width, 0); g.scale(-1, 1);
  g.drawImage(video, 0, 0, out.width, out.height);
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.drawImage(canvas, 0, 0);
  const a = document.createElement("a");
  a.href = out.toDataURL("image/png");
  a.download = `sharingan_stage_${currentIndex + 1}.png`;
  // iOS Safari ignores clicks on detached anchors: attach first.
  document.body.appendChild(a);
  a.click();
  a.remove();
}

// Per-stage motion, anime-accurate:
//  stages 1-3 (Tomoe): slow spin. 4-10 (Mangekyo): faster, direction
//  alternates per stage. 11 (Rinnegan): no spin, ripple rings expand
//  inside the eye. 12 (yellow Sage cross): no spin, the pupil dilates
//  and contracts slowly like a real pupil.
function stageMotion(index) {
  if (index <= 2) return { mode: "spin", dir: 1, speed: 0.45 };
  if (index <= 9) return { mode: "spin", dir: index % 2 ? -1 : 1, speed: 1.15 };
  if (index === 10) return { mode: "ripple" };
  return { mode: "dilate" };
}

function drawEye(eye, now) {
  const w = canvas.width, h = canvas.height;
  // Corneal refraction: the iris seen through the cornea appears
  // shifted toward the gaze direction by a few % of the eye size.
  const cx = (1 - eye.x) * w + (eye.gx || 0) * eye.r * w * 0.14;
  const cy = eye.y * h + (eye.gy || 0) * eye.r * h * 0.14;
  // eye.r is the iris radius. The overlay is the iris diameter plus a
  // small margin (x1.18) so the user's real pupil/iris edge is fully
  // covered — at exactly x1.0 the dark pupil rim stayed visible.
  const energy = awakeningEnergy(now);
  const motion = stageMotion(currentIndex);
  let size = Math.max(10, eye.r * 2 * w * 1.18);   // fixed size, no pulse
  // Rinnegan is a WHOLE-EYE dojutsu: its ripple pattern spreads over
  // the entire eyeball, sclera included. So stage 11 is sized to the
  // eye's full corner-to-corner width and visible height (from the
  // eyelid contour), not to the iris.
  let drawW = size, drawH = size;
  if (motion.mode === "ripple" && eye.contour?.length) {
    const xs = eye.contour.map((p) => (1 - p.x) * w);
    const ys = eye.contour.map((p) => p.y * h);
    drawW = Math.max(size, (Math.max(...xs) - Math.min(...xs)) * 1.04);
    drawH = Math.max(size, (Math.max(...ys) - Math.min(...ys)) * 1.12);
  }
  if (motion.mode === "dilate") {
    // Sage cross pupil breathes: slow dilation/contraction cycle.
    size *= 1 + 0.055 * Math.sin(now / 650);
  }
  const img = stages[currentIndex];
  if (!img) return;

  ctx.save();
  // Clip to the eye's almond shape first: the eyelids then crop the
  // top/bottom of the circle exactly like they crop the real iris
  // (nobody sees a complete pupil). The circle itself is NOT squashed.
  if (eye.contour?.length > 2) {
    // Expand the almond from the eye centre: clipping exactly on the
    // eyelid line shaved too much and flattened circles into
    // rectangles. ~20% taller / ~5% wider: eyelids visibly crop the
    // circle, but gently.
    // Rinnegan hugs the real lid line (whole-eye coverage); other
    // stages keep the gentler expanded crop.
    const expandX = motion.mode === "ripple" ? 1.0 : 1.05;
    const expandY = motion.mode === "ripple" ? 1.02 : 1.20;
    ctx.beginPath();
    eye.contour.forEach((p, i) => {
      const rawX = (1 - p.x) * w, rawY = p.y * h;
      const px = cx + (rawX - cx) * expandX;
      const py = cy + (rawY - cy) * expandY;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.closePath();
    ctx.clip();
  }
  // Eyelash shadow: a soft dark line hugging the upper eyelid,
  // cast onto the eyeball (drawn before the disc, under it at rim).
  if (eye.contour?.length > 8) {
    ctx.save();
    ctx.strokeStyle = "rgba(0, 0, 0, 0.38)";
    ctx.lineWidth = Math.max(1.5, size * 0.055);
    ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
    ctx.shadowBlur = IS_TOUCH ? 0 : size * 0.08;   // shadowBlur is dear on phones
    ctx.lineCap = "round";
    ctx.beginPath();
    // Upper lid runs from one corner over the top to the other:
    // the second half of the contour loop in our ordering.
    const upper = eye.contour.slice(8);
    upper.forEach((p, i) => {
      const px = (1 - p.x) * w, py = p.y * h;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();
    ctx.restore();
  }
  ctx.globalAlpha = OPACITY;
  ctx.translate(cx, cy);
  // Eyeball curvature: squash horizontally by the visible iris
  // ellipse, so turning your gaze foreshortens the design like a
  // real iris on a sphere instead of a flat sticker facing forward.
  // Rinnegan instead scales to the whole-eye size computed above.
  if (motion.mode === "ripple") ctx.scale(drawW / size, drawH / size);
  else ctx.scale(eye.squash ?? 1, 1);
  if (motion.mode === "spin") {
    // Idle: the Copy Wheel Eye never sits still; tomoe creep around
    // the pupil continuously (faster for Mangekyo), direction per
    // stage. Awakening adds one big ease-out spin on top, ending on
    // a whole number of turns so it lands back on the idle angle.
    const idleDeg = (now / 1000) * 16 * motion.speed * motion.dir;
    let angle = idleDeg;
    const elapsed = now - rotationStart;
    if (elapsed < ROTATION_MS) {
      const p = elapsed / ROTATION_MS;
      const eased = 1 - Math.pow(1 - p, 3);    // ease-out: soft landing
      angle += 1800 * eased * motion.dir;      // 5 whole turns
    }
    ctx.rotate(((angle % 360) + 360) % 360 * Math.PI / 180);
  }
  // Live in the same photo as the face: slight blur, desaturation
  // and brightness matched to the webcam feed (sampled in the loop).
  // Canvas filters are very expensive on phone GPUs: skip on touch.
  ctx.filter = IS_TOUCH ? "none" : `blur(0.4px) brightness(${lightFactor.toFixed(3)}) saturate(0.88)`;
  // Spin motion blur: while the awakening spin is fast, ghost the
  // image at trailing angles instead of one crisp CGI rotation.
  const spinning = motion.mode === "spin"
    && (now - rotationStart) < ROTATION_MS;
  if (spinning && !fxReduced) {
    ctx.globalAlpha = OPACITY * 0.28;
    for (const off of [-14, -7, 7, 14]) {
      ctx.save();
      ctx.rotate(off * Math.PI / 180 * motion.dir);
      ctx.drawImage(img, -size / 2, -size / 2, size, size);
      ctx.restore();
    }
    ctx.globalAlpha = OPACITY;
  }
  ctx.drawImage(img, -size / 2, -size / 2, size, size);
  // Iris fibre texture, rotating with the design (not painted flat).
  // Skipped on touch, where the extra blended blit costs frames.
  if (!IS_TOUCH) {
    ctx.globalAlpha = 0.13;
    ctx.globalCompositeOperation = "overlay";
    ctx.drawImage(fibreCanvas, -size / 2, -size / 2, size, size);
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = OPACITY;
  }
  ctx.filter = "none";
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  // Glow INSIDE the circle only: a soft red brightening clipped to
  // the Sharingan disc (and already to the eyelid shape above), at
  // full strength on awakening and fading to nothing as it settles.
  // Nothing is drawn outside the circle.
  {
    // Feathered edge is baked into the sprite at load (bakeFeather).
    // 2) Webcam grain inside the disc, shifting per frame, so the
    // clean vector asset carries the same noise as the face.
    if (!fxReduced) {
      ctx.save();
      ctx.beginPath();
      ctx.ellipse(cx, cy, drawW / 2, drawH / 2, 0, 0, Math.PI * 2);
      ctx.clip();
      ctx.globalAlpha = 0.05;
      ctx.globalCompositeOperation = "overlay";
      const nx = (now % 300) / 300 * 64, ny = (now % 230) / 230 * 64;
      ctx.drawImage(noiseCanvas, cx - drawW / 2 - nx, cy - drawH / 2 - ny, drawW * 1.6, drawH * 1.6);
      ctx.restore();
    }

    ctx.beginPath();
    ctx.ellipse(cx, cy, drawW / 2, drawH / 2, 0, 0, Math.PI * 2);
    ctx.clip();
    if (energy > 0) {
      const inner = ctx.createRadialGradient(cx, cy, 0, cx, cy, size / 2);
      inner.addColorStop(0, `rgba(255, 120, 90, ${0.50 * energy})`);
      inner.addColorStop(0.6, `rgba(255, 50, 35, ${0.28 * energy})`);
      inner.addColorStop(1, `rgba(200, 0, 0, ${0.10 * energy})`);
      ctx.globalAlpha = 1;
      ctx.fillStyle = inner;
      ctx.fillRect(cx - drawW / 2, cy - drawH / 2, drawW, drawH);
    }
    // 4) Eyelid shadow: the upper lid shades the top of the iris,
    // deepening as the eye closes (openness -> shadow strength).
    const lidStrength = 0.16 + 0.30 * (1 - (eye.openness ?? 1));
    const lid = ctx.createLinearGradient(cx, cy - drawH / 2, cx, cy + drawH * 0.05);
    lid.addColorStop(0, `rgba(0, 0, 0, ${lidStrength})`);
    lid.addColorStop(1, "rgba(0, 0, 0, 0)");
    ctx.globalAlpha = 1;
    ctx.fillStyle = lid;
    ctx.fillRect(cx - drawW / 2, cy - drawH / 2, drawW, drawH / 2);
    // 5a) Limbal ring: the dark rim where iris meets the white of
    // the eye, shading the sphere edge (replaces the sticker edge).
    const shade = ctx.createRadialGradient(cx, cy, size * 0.18, cx, cy, size / 2);
    shade.addColorStop(0, "rgba(0, 0, 0, 0)");
    shade.addColorStop(0.78, "rgba(0, 0, 0, 0)");
    shade.addColorStop(1, "rgba(0, 0, 0, 0.34)");
    ctx.fillStyle = shade;
    ctx.fillRect(cx - drawW / 2, cy - drawH / 2, drawW, drawH);
    // 5b) Reflection that MOVES with your head (opposite the motion,
    // like light fixed in the room while the eye moves under it).
    const shiftX = Math.max(-1, Math.min(1, -(eye.vx || 0) * 22)) * size * 0.05;
    const shiftY = Math.max(-1, Math.min(1, -(eye.vy || 0) * 22)) * size * 0.03;
    ctx.fillStyle = "rgba(255, 255, 255, 0.18)";
    ctx.beginPath();
    ctx.ellipse(cx - size * 0.17 + shiftX, cy - size * 0.20 + shiftY,
                size * 0.042, size * 0.024, -0.6, 0, Math.PI * 2);
    ctx.fill();
    if (motion.mode === "ripple" && !fxReduced) {
      // Rinnegan: rings continuously expand from the pupil to the
      // rim of the eye, fading as they grow. Inside the disc only.
      const phase = (now % 2200) / 2200;
      ctx.globalAlpha = 1;
      ctx.strokeStyle = "rgba(190, 140, 230, 0.85)";
      for (let k = 0; k < 2; k++) {
        const rp = ((phase + k / 2) % 1);
        ctx.globalAlpha = 0.75 * (1 - rp);
        ctx.lineWidth = Math.max(1, size * 0.045);
        ctx.beginPath();
        // Whole-eye rings: ellipses reaching the corners, not just
        // circles inside the iris.
        ctx.ellipse(cx, cy, Math.max(0.5, rp * drawW / 2),
                    Math.max(0.5, rp * drawH / 2), 0, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
  }
  ctx.restore();
}

function clipToEye(eye, expandX = 1, expandY = 1) {
  const w = canvas.width, h = canvas.height;
  const cx = (1 - eye.x) * w, cy = eye.y * h;
  ctx.beginPath();
  (eye.contour || []).forEach((p, i) => {
    const rawX = (1 - p.x) * w, rawY = p.y * h;
    const px = cx + (rawX - cx) * expandX;
    const py = cy + (rawY - cy) * expandY;
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.closePath();
  ctx.clip();
}

function drawVeins(eye, now) {
  // Mangekyo strain (stages 8/9): red vessels spread through the
  // white of the eye before/with the blood tears. Clipped to the
  // sclera, stopping at the iris rim, drawn under the Sharingan.
  if (!bloodStart || !eye.contour?.length) return;
  const w = canvas.width, h = canvas.height;
  const cx = (1 - eye.x) * w, cy = eye.y * h;
  const size = Math.max(10, eye.r * 2 * w * 1.18);
  const intensity = Math.min(1, Math.max(0, (now - bloodStart) / 1400) + 0.25);
  ctx.save();
  clipToEye(eye);
  ctx.lineCap = "round";
  for (let i = 0; i < 11; i++) {
    const seed = i * 37.7 + (eye.side === "left" ? 91 : 13);
    const ang = (i / 11) * Math.PI * 2 + Math.sin(seed) * 0.3;
    const x0 = cx + Math.cos(ang) * size * 0.95;
    const y0 = cy + Math.sin(ang) * size * 0.52;
    const x1 = cx + Math.cos(ang) * size * (0.60 + (Math.sin(seed * 2) * 0.5 + 0.5) * 0.14);
    const y1 = cy + Math.sin(ang) * size * 0.34;
    const mx = (x0 + x1) / 2 + Math.sin(seed * 3) * size * 0.08;
    const my = (y0 + y1) / 2 + Math.cos(seed * 3) * size * 0.05;
    ctx.strokeStyle = `rgba(178, 24, 28, ${0.5 * intensity})`;
    ctx.lineWidth = Math.max(0.7, size * 0.018);
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.quadraticCurveTo(mx, my, x1, y1);
    ctx.stroke();
  }
  ctx.restore();
}

function drawRinneganField(eye, now) {
  // Stage 11 (Rinnegan): the ripple spreads over the WHOLE eyeball,
  // white included: faint purple tint + rings expanding past the iris.
  if (currentIndex !== 10 || !eye.contour?.length) return;
  const w = canvas.width, h = canvas.height;
  const cx = (1 - eye.x) * w, cy = eye.y * h;
  const size = Math.max(10, eye.r * 2 * w * 1.18);
  ctx.save();
  clipToEye(eye, 1.02, 1.05);
  ctx.fillStyle = "rgba(148, 118, 205, 0.16)";
  ctx.fillRect(cx - size, cy - size, size * 2, size * 2);
  const phase = (now % 2200) / 2200;
  ctx.strokeStyle = "rgba(150, 110, 210, 0.8)";
  for (let k = 0; k < 2; k++) {
    const rp = (phase + k / 2) % 1;
    ctx.globalAlpha = 0.5 * (1 - rp);
    ctx.lineWidth = Math.max(1, size * 0.03);
    ctx.beginPath();
    ctx.arc(cx, cy, Math.max(0.5, rp * size * 1.05), 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
}

function drawRivulet(cx, cy, size, now, delayS, lenFactor, wobbleSeed) {
  // One glossy blood stream: pooled bead at the lower eyelid, then a
  // tapering, slightly wandering trail with a specular highlight
  // down its left edge, like light on wet blood (see reference look).
  const elapsed = (now - bloodStart) / 1000 - delayS;
  if (elapsed <= 0) return;
  const maxLen = size * 3.4 * lenFactor;
  const len = Math.min(maxLen, elapsed * size * 0.8);
  if (len <= 2) return;
  const y0 = cy + size * 0.26;
  const steps = 16;
  const points = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    points.push({
      x: cx + Math.sin(t * 4.5 + wobbleSeed) * size * 0.05,
      y: y0 + t * len,
      half: size * 0.075 * (1 - t * 0.5) + size * 0.012,
    });
  }
  ctx.save();
  ctx.shadowColor = "rgba(20, 0, 0, 0.55)";
  ctx.shadowBlur = size * 0.06;
  const grad = ctx.createLinearGradient(0, y0, 0, y0 + len);
  grad.addColorStop(0, "rgba(46, 1, 4, 0.97)");
  grad.addColorStop(0.45, "rgba(122, 7, 16, 0.96)");
  grad.addColorStop(1, "rgba(74, 2, 7, 0.94)");
  ctx.fillStyle = grad;
  ctx.beginPath();
  points.forEach((p, i) => {
    if (i === 0) ctx.moveTo(p.x - p.half, p.y);
    else ctx.lineTo(p.x - p.half, p.y);
  });
  [...points].reverse().forEach((p) => ctx.lineTo(p.x + p.half, p.y));
  ctx.closePath();
  ctx.fill();
  ctx.shadowBlur = 0;
  // Specular highlight: a thin bright stripe along the stream.
  ctx.strokeStyle = "rgba(255, 170, 150, 0.38)";
  ctx.lineWidth = Math.max(1, size * 0.022);
  ctx.lineCap = "round";
  ctx.beginPath();
  points.forEach((p, i) => {
    const hx = p.x - p.half * 0.45, hy = p.y;
    if (i === 0) ctx.moveTo(hx, hy); else ctx.lineTo(hx, hy);
  });
  ctx.stroke();
  // Glossy pooled bead where the stream leaves the eyelid.
  const bead = ctx.createRadialGradient(
    cx - size * 0.03, y0 - size * 0.03, 0, cx, y0, size * 0.11);
  bead.addColorStop(0, "rgba(190, 40, 45, 0.98)");
  bead.addColorStop(0.6, "rgba(110, 5, 12, 0.97)");
  bead.addColorStop(1, "rgba(50, 1, 4, 0.95)");
  ctx.fillStyle = bead;
  ctx.beginPath();
  ctx.ellipse(cx, y0, size * 0.10, size * 0.075, 0, 0, Math.PI * 2);
  ctx.fill();
  // Rounded tip, then falling drops once fully grown.
  const tip = points[points.length - 1];
  ctx.fillStyle = "rgba(96, 4, 10, 0.95)";
  ctx.beginPath();
  ctx.arc(tip.x, tip.y, tip.half * 1.2, 0, Math.PI * 2);
  ctx.fill();
  if (len >= maxLen - 1) {
    const cycle = (((now - bloodStart) / 1600) + wobbleSeed) % 1;
    const dropY = tip.y + cycle * cycle * size * 1.9;
    const dg = ctx.createRadialGradient(
      tip.x - 1, dropY - 1, 0, tip.x, dropY, tip.half);
    dg.addColorStop(0, `rgba(200, 60, 60, ${0.95 * (1 - cycle)})`);
    dg.addColorStop(1, `rgba(80, 2, 7, ${0.95 * (1 - cycle)})`);
    ctx.fillStyle = dg;
    ctx.beginPath();
    ctx.ellipse(tip.x, dropY, tip.half * 0.75, tip.half * 1.2, 0, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawBlood(eye, now, eyeIndex) {
  // Amaterasu blood, as in the reference: TWO streams per eye, from
  // the inner and outer lower eyelid, the second starting later and
  // running shorter. Amaterasu is cast from the LEFT eye, so that
  // eye bleeds fully; the right eye only weeps a little.
  if (!bloodStart) return;
  const w = canvas.width, h = canvas.height;
  const cx = (1 - eye.x) * w;
  const cy = eye.y * h;
  const size = Math.max(10, eye.r * 2 * w * 1.18);
  if (eye.side === "left") {
    drawRivulet(cx - size * 0.30, cy, size, now, 0, 1.0, eyeIndex * 2.1);
    drawRivulet(cx + size * 0.31, cy, size, now, 0.8, 0.72, eyeIndex * 2.1 + 3.4);
  } else {
    drawRivulet(cx - size * 0.18, cy, size, now, 1.1, 0.42, eyeIndex * 2.1 + 5.2);
  }
}

function loop() {
  if (!running) return;
  rafId = requestAnimationFrame(loop);
  const now = performance.now();
  // Detection runs only on a new camera frame; drawing runs on every
  // rAF. On a phone the detector may only manage 15-20 fps while the
  // screen refreshes at 60: redrawing the last filtered eyes between
  // detections (predicted a few ms ahead, below) is what stops the
  // overlay visibly stepping and trailing behind the real eye.
  const isNewFrame = video.currentTime !== lastVideoTime;
  let result = null;
  let landmarks = null;
  let detectFailed = false;
  if (isNewFrame) {
    lastVideoTime = video.currentTime;
    try { result = landmarker.detectForVideo(video, now); }
    catch { result = null; detectFailed = true; }
    // How long detection itself took: a result describes where the
    // eye WAS when its frame was captured, so the draw must predict
    // this far ahead (plus the result's age) to sit on the pupil NOW.
    eyeLatencyMs += (Math.min(120, performance.now() - now) - eyeLatencyMs) * 0.2;

    if (video.videoWidth) {
      // Cap the overlay canvas: phones can expose 4K streams and
      // per-frame full-res work drains battery. Landmarks are
      // normalised, so a smaller canvas loses no tracking accuracy.
      const scale = Math.min(1, (IS_TOUCH ? 720 : 960) / Math.max(video.videoWidth, video.videoHeight));
      const tw = Math.round(video.videoWidth * scale);
      const th = Math.round(video.videoHeight * scale);
      if (canvas.width !== tw || canvas.height !== th) {
        canvas.width = tw;
        canvas.height = th;
      }
    }
    landmarks = result?.faceLandmarks?.[0];
    // Head pose from the transformation matrix (separates head turn
    // from eye movement; used to correct foreshortening below).
    // Lightly smoothed: the raw matrix jitters under hand shake.
    const matrix = result?.facialTransformationMatrixes?.[0]?.data;
    if (matrix?.length >= 16) {
      const yaw = Math.atan2(matrix[8], matrix[10]);
      headYaw += (yaw - headYaw) * 0.35;
    }
    if (landmarks) {
      const raw = eyeCentres(landmarks, canvas.width, canvas.height);
      if (eyeFilters.length !== raw.length) {
        eyeFilters = raw.map(() => ({
          // Centre: anti-tremor at rest, near-raw on a real move.
          // Phones run a faster tuning (speed is the priority there);
          // prediction below, not raw looseness, does most of the work.
          x: new OneEuro(IS_TOUCH ? 1.6 : 1.1, IS_TOUCH ? 24 : 18, 1.0,
                         { deadband: 0.0005, maxSpeed: IS_TOUCH ? 2.2 : 1.7 }),
          y: new OneEuro(IS_TOUCH ? 1.6 : 1.1, IS_TOUCH ? 24 : 18, 1.0,
                         { deadband: 0.0005, maxSpeed: IS_TOUCH ? 2.2 : 1.7 }),
          // Radius and shape change slowly; filter them harder so
          // the disc does not pulse with iris-ring noise.
          r: new OneEuro(0.9, 6, 1.0, { deadband: 0.0003, maxSpeed: 0.7 }),
          gx: new OneEuro(1.2, 8, 1.0, { deadband: 0.004, maxSpeed: 2.5 }),
          gy: new OneEuro(1.2, 8, 1.0, { deadband: 0.004, maxSpeed: 2.5 }),
          squash: new OneEuro(0.9, 6, 1.0, { deadband: 0.004, maxSpeed: 1.2 }),
          open: new OneEuro(1.4, 8, 1.0, { deadband: 0.01, maxSpeed: 3 }),
        }));
        eyeVelocities = raw.map(() => ({ x: 0, y: 0 }));
        prevFiltered = [];
      }
      const t = now / 1000;
      const dtPrev = prevFilteredAt ? Math.min(0.1, Math.max(1 / 240, t - prevFilteredAt)) : 0;
      const newPrev = [];
      latestEyes = raw.map((eye, i) => {
        const f = eyeFilters[i];
        const fx = f.x.filter(eye.x, t);
        const fy = f.y.filter(eye.y, t);
        const prev = prevFiltered[i];
        newPrev.push({ x: fx, y: fy });
        // Smoothed velocity of the FILTERED centre (per second),
        // for the time-based prediction at draw time. ~70 ms time
        // constant: fast enough to follow a real move, slow enough
        // that 8 Hz hand tremor mostly cancels out of it.
        if (prev && dtPrev > 0) {
          const v = eyeVelocities[i];
          const a = 1 - Math.exp(-dtPrev / 0.07);
          v.x += (Math.max(-2.0, Math.min(2.0, (fx - prev.x) / dtPrev)) - v.x) * a;
          v.y += (Math.max(-2.0, Math.min(2.0, (fy - prev.y) / dtPrev)) - v.y) * a;
        }
        return {
          ...eye,
          x: fx,
          y: fy,
          r: f.r.filter(eye.r, t),
          gx: f.gx.filter(eye.gx || 0, t),
          gy: f.gy.filter(eye.gy || 0, t),
          openness: f.open.filter(eye.openness, t),
          // Anchor the eyelid contour to the filtered centre: the
          // shape stays live, but its position inherits the centre's
          // smoothing instead of jittering on its own landmarks.
          contour: eye.contour?.map((p) => ({
            x: p.x + (fx - eye.x), y: p.y + (fy - eye.y),
          })),
          vx: fx - eye.x, vy: fy - eye.y,   // residual, for reflection
          // Head turn foreshortens the eye beyond the landmark ellipse.
          squash: Math.min(f.squash.filter(eye.squash, t),
                           0.60 + 0.40 * Math.abs(Math.cos(headYaw))),
        };
      });
      prevFiltered = newPrev;
      prevFilteredAt = t;
      latestEyeAt = performance.now();   // when the result landed
    } else if (!detectFailed) {
      // A real "no face" clears the overlay; a transient detector
      // error keeps the last eyes instead of blinking them off.
      latestEyes = [];
      eyeVelocities = [];
    }
  }

  // Draw: at most every ~24 ms between detections (full-rate draws
  // would starve the detector of main-thread time on a phone). The
  // last filtered eyes are predicted ahead by the smoothed velocity
  // for the result's age PLUS its measured detection latency — a
  // result is already that old when it arrives, and drawing it raw
  // is exactly the visible trail behind the pupil. Lead is capped
  // (110 ms / 0.014 of frame) so a direction change cannot fling it.
  if (!isNewFrame && now - lastDrawAt < 24) return;
  const drawNow = performance.now();
  lastDrawAt = drawNow;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (latestEyes.length) {
    const ageMs = Math.max(0, drawNow - latestEyeAt);
    const leadS = Math.min(ageMs + eyeLatencyMs, 110) / 1000;
    const drawn = latestEyes.map((eye, i) => {
      if (ageMs > 180) return eye;
      const v = eyeVelocities[i] || { x: 0, y: 0 };
      const dx = Math.max(-0.014, Math.min(0.014, v.x * leadS));
      const dy = Math.max(-0.014, Math.min(0.014, v.y * leadS));
      if (!dx && !dy) return eye;
      return {
        ...eye,
        x: eye.x + dx,
        y: eye.y + dy,
        contour: eye.contour?.map((p) => ({ x: p.x + dx, y: p.y + dy })),
      };
    });
    // Under-overlay effects first: Rinnegan field and Mangekyo
    // strain veins live on the eye/skin beneath the Sharingan.
    drawn.forEach((eye) => {
      drawRinneganField(eye, drawNow);
      drawVeins(eye, drawNow);
    });
    for (const eye of drawn) drawEye(eye, drawNow);
    // Amaterasu: black blood flows from both eyes on stages 8 and 9.
    if (bloodStart) drawn.forEach((eye, i) => drawBlood(eye, drawNow, i));

    // Almighty Push: one expanding shockwave ring from the face when
    // the Rinnegan (stage 11) awakens.
    if (currentIndex === 10 && drawn.length) {
      const energy = awakeningEnergy(drawNow);
      if (energy > 0) {
        const p = 1 - Math.sqrt(energy);
        const mx = drawn.reduce((s, e) => s + (1 - e.x), 0) / drawn.length * canvas.width;
        const my = drawn.reduce((s, e) => s + e.y, 0) / drawn.length * canvas.height;
        ctx.save();
        ctx.strokeStyle = `rgba(190, 160, 240, ${0.55 * (1 - p)})`;
        ctx.lineWidth = 2 + 5 * (1 - p);
        ctx.beginPath();
        ctx.arc(mx, my, Math.max(1, p * canvas.width * 0.55), 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }
    }
  }

  // Between detections (or on a transient detector error) there is
  // nothing new to calibrate or blink on — the draw above is done.
  if (!isNewFrame || detectFailed) return;
  const eyesOpen = !!landmarks;

  // Personal calibration first: ~1.5s of open eyes at start sets
  // this user's EAR baseline. No stage changes until it is done.
  const ear = averageEAR(landmarks, canvas.width, canvas.height);
  if (!earBaseline && eyesOpen && ear !== null) {
    const cats = result.faceBlendshapes?.[0]?.categories || [];
    const blinkScore = Math.max(0, ...cats
      .filter((c) => c.categoryName.startsWith("eyeBlink"))
      .map((c) => c.score));
    if (blinkScore < 0.3) earSamples.push(ear);
    setStatus("Calibrating: look at the camera, eyes open…");
    if (earSamples.length >= 40) {
      const sorted = [...earSamples].sort((a, b) => a - b);
      earBaseline = sorted[Math.floor(sorted.length / 2)];
      setStatus("Calibrated. Blink to evolve");
    }
  }

  // Deliberate blink: eyes must stay shut for MIN_CLOSED_MS, not
  // just flutter, before reopening evolves the stage.
  const closed = eyesOpen ? blinkClosed(result, ear) : false;
  if (!closed && eyesOpen) {
    if (earBaseline && !eyesWereOpen && closedFrames >= MIN_CLOSED_FRAMES
        && closedSince !== null && now - closedSince >= MIN_CLOSED_MS) {
      tryTransition(now);
    }
    closedFrames = 0;
    closedSince = null;
    eyesWereOpen = true;
    if (earBaseline) setStatus("Eyes open");
  } else if (closed || !eyesOpen) {
    if (eyesWereOpen) closedSince = now;
    closedFrames++;
    eyesWereOpen = false;
    if (earBaseline) {
      setStatus(closed
        ? (closedSince && now - closedSince >= MIN_CLOSED_MS
            ? "Release to evolve…"
            : "Hold… keep eyes closed")
        : "No face detected");
    }
  }

  if (frames % 15 === 0) updateLightFactor();
  frames++;
  if (now - fpsTimer >= 1000) {
    fpsLabel.textContent = `${frames} fps`;
    // Auto-quality: sustained low FPS sheds the expensive extras
    // (grain, spin ghosts, ripples); recovering FPS restores them.
    if (frames < 20) fxReduced = true;
    else if (frames > 26 && !IS_TOUCH) fxReduced = false;
    frames = 0; fpsTimer = now;
  }
}

// --- Intro flow ---------------------------------------------------
// On page open the intro video loops while EVERYTHING prepares
// behind it: stage images, audio list, face model, camera stream
// and the personal EAR calibration. The face is never shown during
// this. Only when all of it is ready does the red START appear.
const introScreen = document.getElementById("introScreen");
const introVideo = document.getElementById("introVideo");
const introLoadingEl = document.getElementById("introLoading");
const introErrorEl = document.getElementById("introError");
const introStartBtn = document.getElementById("introStartBtn");
const helpModal = document.getElementById("helpModal");
const helpBtn = document.getElementById("helpBtn");
const helpCloseBtn = document.getElementById("helpCloseBtn");
let prepared = false;
let helpShownOnce = false;

function introStatus(text) {
  introLoadingEl.firstChild.textContent = text;
  setStatus(text);
}

async function calibrateBehindIntro(deadlineMs = 4500) {
  // Run detection (without showing anything) until the personal
  // EAR baseline is set, or the deadline passes — calibration then
  // simply finishes in the main loop instead of blocking START.
  const t0 = performance.now();
  let lastTime = -1;
  while (!earBaseline && performance.now() - t0 < deadlineMs) {
    await new Promise((r) => requestAnimationFrame(r));
    if (video.currentTime === lastTime || !video.videoWidth) continue;
    lastTime = video.currentTime;
    let result = null;
    try { result = landmarker.detectForVideo(video, performance.now()); }
    catch { continue; }
    const landmarks = result?.faceLandmarks?.[0];
    if (!landmarks) continue;
    const ear = averageEAR(landmarks, video.videoWidth, video.videoHeight);
    if (ear === null) continue;
    const cats = result.faceBlendshapes?.[0]?.categories || [];
    const blinkScore = Math.max(0, ...cats
      .filter((c) => c.categoryName.startsWith("eyeBlink"))
      .map((c) => c.score));
    if (blinkScore < 0.3) earSamples.push(ear);
    if (earSamples.length >= 40) {
      const sorted = [...earSamples].sort((a, b) => a - b);
      earBaseline = sorted[Math.floor(sorted.length / 2)];
    }
  }
}

async function prepare() {
  if (prepared) return;
  introErrorEl.textContent = "";
  introLoadingEl.hidden = false;
  introStartBtn.hidden = true;
  try {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      throw new Error("Camera API unavailable. Open this site over "
        + "HTTPS or http://localhost (not file://).");
    }
    introStatus("Loading stages");
    if (!stages.length) stages = await loadStages();
    if (!stages.length) throw new Error("No stage images found in assets/.");
    audioFiles = await audioUrls(stages.length);
    introStatus("Loading face model");
    if (!landmarker) landmarker = await initLandmarker();
    introStatus("Asking camera permission");
    // Mobile-first constraints: front camera at a sane resolution.
    // Unconstrained requests can resolve to 4K on phones (heat,
    // battery, frame drops). Fall back to a bare request on devices
    // that reject the ideal constraints.
    // Phones get a smaller feed: MediaPipe's landmark model works on
    // a cropped face, so 480-ish pixels loses no useful accuracy but
    // runs detection far more often — the single biggest latency win
    // on mobile. Desktops keep the sharper 1280 feed they can afford.
    const wantCamera = {
      video: {
        facingMode: "user",
        width: IS_TOUCH ? { ideal: 480, max: 640 } : { ideal: 1280, max: 1280 },
        height: IS_TOUCH ? { ideal: 480, max: 640 } : { ideal: 720, max: 1280 },
      },
      audio: false,
    };
    try {
      try {
        stream = await navigator.mediaDevices.getUserMedia(wantCamera);
      } catch {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
      }
    } catch (err) {
      throw new Error(err.name === "NotAllowedError"
        ? "Camera was blocked. Allow camera access for this site, "
          + "then press START to retry."
        : "No camera available. Close other apps using the camera, "
          + "then press START to retry.");
    }
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    await video.play();
    introStatus("Calibrating your eyes");
    await calibrateBehindIntro();
    prepared = true;
    // Everything ready: swap Loading... for the red START button.
    introLoadingEl.hidden = true;
    introStartBtn.hidden = false;
    introStartBtn.textContent = "START";
  } catch (err) {
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
      video.srcObject = null;
    }
    introLoadingEl.hidden = true;
    introErrorEl.textContent = err.message;
    introStartBtn.textContent = "RETRY";
    introStartBtn.hidden = false;
  }
}

let awaitingFirstEntry = false;

function enterApp() {
  // START was clicked, but the face is NOT shown yet: the help
  // instructions come first (once). Only "Got it" opens the camera
  // view. On later restarts (help already seen) enter directly.
  if (!helpShownOnce) {
    helpShownOnce = true;
    awaitingFirstEntry = true;
    helpModal.hidden = false;
    return;
  }
  openCameraView();
}

function openCameraView() {
  awaitingFirstEntry = false;
  helpModal.hidden = true;
  introVideo.pause();
  introScreen.hidden = true;
  currentIndex = 0;
  rotationStart = performance.now();
  awakeningStart = performance.now();
  bloodStart = 0;
  stageLabel.textContent = `Stage 1 / ${stages.length}`;
  running = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  prevBtn.disabled = nextBtn.disabled = false;
  hideLoading();
  setStatus(earBaseline ? "Eyes open" : "Calibrating: look at the camera, eyes open…");
  // First stage audio, on this user gesture so the browser allows it.
  if (audioFiles.length) {
    new Audio(audioFiles[0]).play().catch(() => {});
  }
  loop();
}

function closeHelp() {
  helpModal.hidden = true;
  if (awaitingFirstEntry) openCameraView();
}

async function start() {
  // The in-app Start Camera button restarts after a Stop.
  setError("");
  await prepare();
  if (prepared) enterApp();
}

function stop() {
  running = false;
  prepared = false;
  if (rafId) cancelAnimationFrame(rafId);
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  startBtn.disabled = false;
  stopBtn.disabled = true;
  prevBtn.disabled = nextBtn.disabled = true;
  eyeFilters = [];
  prevFiltered = [];
  prevFilteredAt = 0;
  latestEyes = [];
  latestEyeAt = 0;
  eyeVelocities = [];
  lastDrawAt = 0;
  closedSince = null;
  closedFrames = 0;
  earBaseline = null;
  earSamples = [];
  bloodStart = 0;
  video.style.transform = "scaleX(-1)";
  canvas.style.transform = "";
  setStatus("Stopped");
}

// Intro playback plan (no sound prompt, by design): the
// FIRST run plays WITH voice, from the start. When that run ends,
// the same video restarts MUTED and loops forever as loading-screen
// ambience. Browsers may block autoplay-with-sound on a first-ever
// visit; then the muted loop starts silently, and the first touch or
// keypress anywhere (except START, which enters the app) quietly
// restarts the voiced run — the user is never asked to tap for sound.
let introVoiceDone = false;
let introVoicePlaying = false;
let introVoiceTrying = false;

function playIntroMutedLoop() {
  introVideo.muted = true;
  introVideo.loop = true;
  introVideo.currentTime = 0;
  introVideo.play().catch(() => {});
}

async function playIntroWithVoice() {
  introVideo.muted = false;
  introVideo.volume = 1;
  introVideo.loop = false;
  introVideo.currentTime = 0;      // voiced run always from the start
  await introVideo.play();
  introVoicePlaying = true;
}

introVideo.addEventListener("ended", () => {
  if (introVoiceDone) return;
  introVoiceDone = true;           // voice has completed, exactly once
  introVoicePlaying = false;
  playIntroMutedLoop();
});

(async () => {
  try {
    await playIntroWithVoice();    // first run, with voice
  } catch {
    // Autoplay-with-sound blocked: muted loop now, voiced run on the
    // first natural interaction — silently, with no prompt button.
    playIntroMutedLoop();
    const wakeOnFirstTouch = async (e) => {
      if (introVoiceDone || introVoicePlaying || introVoiceTrying) return;
      if (introScreen.hidden) return;
      if (e.target?.closest?.("#introStartBtn")) return;   // enters app
      introVoiceTrying = true;
      try {
        await playIntroWithVoice();
        window.removeEventListener("pointerdown", wakeOnFirstTouch);
        window.removeEventListener("keydown", wakeOnFirstTouch);
      } catch { /* still blocked: stay on the muted loop */ }
      introVoiceTrying = false;
    };
    window.addEventListener("pointerdown", wakeOnFirstTouch, { passive: true });
    window.addEventListener("keydown", wakeOnFirstTouch);
  }
})();

introStartBtn.addEventListener("click", () => {
  if (prepared) enterApp();
  else prepare();          // RETRY after a failure
});
helpBtn.addEventListener("click", () => { helpModal.hidden = false; });
helpCloseBtn.addEventListener("click", closeHelp);
helpModal.addEventListener("click", (e) => {
  if (e.target === helpModal) closeHelp();
});
// Begin loading everything behind the intro video immediately.
prepare();

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);
nextBtn.addEventListener("click", () => setStage(currentIndex + 1, performance.now()));
prevBtn.addEventListener("click", () => setStage(currentIndex - 1, performance.now()));
// Reset (r) and Snapshot (s) stay on the keyboard, per the Help card.
window.addEventListener("keydown", (e) => {
  if (!running) return;
  if (e.key === " ") e.preventDefault();   // space also scrolls the page
  if (e.key === "n" || e.key === " ") setStage(currentIndex + 1, performance.now());
  else if (e.key === "p") setStage(currentIndex - 1, performance.now());
  else if (e.key === "r") setStage(0, performance.now());
  else if (e.key === "s") snapshot();
});

