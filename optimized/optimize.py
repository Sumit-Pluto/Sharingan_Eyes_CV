"""Build the optimized bundle in this folder.

Run once from the project root:

    python optimized/optimize.py

Produces, next to this script:
    optimized/model/best.onnx   YOLO model exported from ../best.pt
                                (imgsz=320, simplified, opset 12)
    optimized/assets/N.png      stage images downscaled to <=256px,
                                background stripped, PNG recompressed
    optimized/audio/            stage sounds copied unchanged
    optimized/manifest.json     before/after file sizes

Originals are never modified. Requires: ultralytics, opencv, numpy
(all already in ../requirements.txt).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

MAX_SIZE = 256          # assets are displayed at pupil size; 256 is plenty
MODEL_IMGSZ = 320       # down from 640: pupils are large, easy targets
PNG_PARAMS = None       # set after cv2 import


def export_onnx() -> Path | None:
    src = ROOT / "best.pt"
    if not src.is_file():
        print(f"Skip ONNX: model not found: {src}")
        return None
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"Skip ONNX: ultralytics not installed ({exc})")
        return None
    # Never let ultralytics AutoUpdate: its pip install of onnx/onnxslim
    # can hang for an hour in old conda envs. If onnx is missing, say so
    # and skip; the user installs one pinned wheel manually instead.
    try:
        import onnx  # noqa: F401
    except ImportError:
        print("Skip ONNX: package 'onnx' is not installed.")
        print('  Install it fast with: pip install --no-deps "onnx==1.17.0"')
        print("  then re-run: python optimized/optimize.py")
        return None
    print(f"Exporting {src.name} -> ONNX (imgsz={MODEL_IMGSZ}) ...")
    try:
        # simplify=False: onnxslim is not needed (it was the other half
        # of the hanging AutoUpdate) and barely shrinks this model.
        exported = Path(YOLO(str(src)).export(
            format="onnx", imgsz=MODEL_IMGSZ, simplify=False, opset=12,
        ))
    except Exception as exc:
        # A checkpoint saved by a newer ultralytics cannot be unpickled
        # by an older one (e.g. "Can't get attribute 'DFLoss'"). Report
        # it and continue: resized assets are still worth generating.
        print(f"Skip ONNX: could not load/export the model ({exc})")
        print("  Hint: best.pt needs the ultralytics version it was "
              "trained with. Run: pip install -r requirements.txt")
        return None
    dest_dir = HERE / "model"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "best.onnx"
    if exported.resolve() != dest.resolve():
        shutil.copy2(exported, dest)
    print(f"  {dest}  ({dest.stat().st_size / 1e6:.2f} MB, "
          f"was {src.stat().st_size / 1e6:.2f} MB)")
    return dest


def resize_assets() -> list[dict]:
    import cv2

    from sharingan import ensure_bgra, natural_sort_key, strip_opaque_background

    src_dir, dest_dir = ROOT / "assets", HERE / "assets"
    dest_dir.mkdir(parents=True, exist_ok=True)
    report = []
    paths = sorted(
        (p for p in src_dir.iterdir()
         if p.suffix.lower() in {".png", ".jpg", ".jpeg"}),
        key=natural_sort_key,
    )
    for path in paths:
        image = ensure_bgra(cv2.imread(str(path), cv2.IMREAD_UNCHANGED))
        if image is None:
            print(f"  skip (unreadable): {path.name}")
            continue
        image = strip_opaque_background(image)
        h, w = image.shape[:2]
        scale = min(1.0, MAX_SIZE / max(h, w))
        if scale < 1.0:
            image = cv2.resize(
                image, (round(w * scale), round(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        dest = dest_dir / (path.stem + ".png")
        cv2.imwrite(str(dest), image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        before, after = path.stat().st_size, dest.stat().st_size
        report.append({"file": dest.name, "before": before, "after": after})
        print(f"  {path.name}: {w}x{h} {before/1e3:.0f} KB -> "
              f"{image.shape[1]}x{image.shape[0]} {after/1e3:.0f} KB")
    return report


def copy_audio() -> list[str]:
    src, dest = ROOT / "audio", HERE / "audio"
    if not src.is_dir():
        print("No audio folder, skipping audio.")
        return []
    from sharingan import natural_sort_key

    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(
        (p for p in src.iterdir() if p.is_file()), key=natural_sort_key,
    ):
        shutil.copy2(path, dest / path.name)
        copied.append(path.name)
    print(f"Audio copied to {dest}")
    return copied


def main() -> int:
    onnx = export_onnx()
    report = resize_assets()
    audio = copy_audio()
    total_before = sum(r["before"] for r in report)
    total_after = sum(r["after"] for r in report)
    manifest = {
        "model_onnx": str(onnx.relative_to(HERE)) if onnx else None,
        "model_imgsz": MODEL_IMGSZ,
        "asset_max_size": MAX_SIZE,
        "assets": report,
        "audio": audio,
        "assets_total_before": total_before,
        "assets_total_after": total_after,
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if total_before:
        print(f"\nAssets total: {total_before/1e6:.2f} MB -> "
              f"{total_after/1e6:.2f} MB "
              f"({100 * (1 - total_after / total_before):.0f}% smaller)")
    print(f"Manifest: {HERE / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
