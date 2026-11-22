# Optimized bundle

Smaller, faster copy of the model and stage images. Originals are untouched.

## Generate it

From the project root (one time, needs `pip install -r requirements.txt`):

```bash
python optimized/optimize.py
```

That creates:

- `model/best.onnx` — `../best.pt` exported to ONNX at `imgsz=320`. Requires the `onnx` package (`pip install --no-deps "onnx==1.17.0"`); if it is missing the script skips export instead of triggering ultralytics' slow AutoUpdate, and still produces the resized assets. Use it with ONNX Runtime (Python: `onnxruntime`; browser: ONNX Runtime Web) instead of the PyTorch `.pt` for faster, lighter inference. The current browser app detects with MediaPipe, so this file is for the Python/ONNX path or a future YOLO-in-browser build.
- `assets/N.png` — each stage downscaled to max 256px (they are only ever shown at pupil size), background stripped, PNG recompressed at max compression. The browser app loads these first and falls back to `../assets/` if they are missing.
- `audio/` — stage sounds, copied unchanged.
- `manifest.json` — before/after sizes for every file.

## Why

The originals run up to 6000x6000 and several MB each; the browser had to download, downscale, and background-strip them on every visit. Pre-baking that work here makes the web version load and spin noticeably faster, and the ONNX model avoids shipping PyTorch.
