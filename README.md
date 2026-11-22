# Sharingan_Eyes_CV

Real-time webcam overlay: a YOLO pupil detector places the current Sharingan stage over each pupil. Blinking advances to the next stage with a rotation animation and sound.

## Setup

Requires Python 3.10+, a webcam, and these files in the project root (paths are relative to `sharingan.py`, no absolute paths):

- `best.pt` — trained YOLO pupil model
- `assets/` — Sharingan stage images (`.png`/`.jpg`/`.jpeg`, loaded in natural order: `2.png` before `10.png`, non-numeric names also work)
- `audio/` — stage sounds (`.mp3`/`.wav`/`.ogg`, same ordering; missing or mismatched counts only disable/cycle audio, never crash)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python sharingan.py
```

Close both eyes and reopen to advance a stage (closing one eye will not work — the other pupil is still detected). Any closure past ~2 frames counts, however long you hold it. Detections below 0.5 confidence are ignored.

Keys: `n`/Space next stage, `p` previous, `r` reset, `+`/`-` overlay size, `s` snapshot (saved to `snapshots/`), `f` toggle HUD, `q`/`Esc` quit.

Options: `python sharingan.py --help` — `--camera N`, `--imgsz 320` (lower is faster), `--detect-every 2` (run YOLO every Nth frame), `--scale 1.0`.

Performance/stability built in: assets are downscaled to 256px at load, rotation happens at pupil size (not full resolution), YOLO runs at 320 every 2nd frame with box smoothing to stop overlay jitter, the spin eases out instead of stopping dead, and the HUD shows stage, FPS, and eye state.

## Web version (browser, no Python)

Runs fully on the visitor's device — deploy the project folder to any static host (Vercel, Cloudflare Pages, Netlify, GitHub Pages) for free. Note: `best.pt` (YOLO) cannot run in a browser, so detection uses MediaPipe Face Landmarker iris tracking instead; stages, blink behaviour, rotation, and sounds behave the same as the Python version.

```bash
npm start          # node server.js
# open http://localhost:8000
```

Requires internet once to load the MediaPipe library/model from CDN. Camera permission is needed; on a deployed site this requires HTTPS (localhost is exempt).

Web controls mirror the Python keys: Prev/Next/Reset/Snapshot buttons, `n`/`p`/`r`/`s` keys, iris smoothing, and the same eased spin (last stage still does not rotate). Run `python optimized/optimize.py` once to generate the pre-shrunk assets/ONNX bundle the page loads first.

## Tests

```bash
python -m unittest discover -s tests -v
```
