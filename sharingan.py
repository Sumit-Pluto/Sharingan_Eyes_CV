"""Real-time Sharingan overlay.

Detects pupils with a YOLO model, overlays the current Sharingan stage on
each pupil, and advances to the next stage (with rotation + sound) on a blink.
Keys: blink/n = next, p = previous, r = reset, +/- = overlay size,
s = snapshot, f = HUD, q/Esc = quit. Options: --camera, --imgsz,
--detect-every, --scale (see --help).
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"
ASSETS_DIR = BASE_DIR / "assets"
AUDIO_DIR = BASE_DIR / "audio"

CONFIDENCE_THRESHOLD = 0.5
OVERLAY_OPACITY = 0.7
ASSET_MAX_SIZE = 256       # stages are shown at pupil size; load small
DEFAULT_IMGSZ = 320        # YOLO input; pupils are large, 640 was waste
DEFAULT_DETECT_EVERY = 2   # run YOLO every Nth frame, reuse in between
WEBCAM_WIDTH, WEBCAM_HEIGHT = 640, 480
SMOOTHING_ALPHA = 0.5      # box smoothing: 0 = frozen, 1 = raw YOLO boxes

TRANSITION_COOLDOWN_S = 0.5
# A blink counts once the eyes have been closed for at least this long.
# There is deliberately no upper limit: YOLO runs at a low frame rate,
# so a normal close-and-open easily spans >0.6s, and rejecting longer
# closures made stage changes never fire.
MIN_BLINK_FRAMES = 2
MIN_BLINK_DURATION_S = 0.05

ROTATION_DURATION_S = 2.4
ROTATION_SPEED_DEG_S = 720.0

WINDOW_NAME = "Real-time Sharingan Overlay"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
AUDIO_SUFFIXES = {".mp3", ".wav", ".ogg"}


def natural_sort_key(path: Path) -> tuple:
    """Sort ``2.png`` before ``10.png``; tolerate non-numeric names."""
    parts = re.split(r"(\d+)", path.stem)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def ensure_bgra(image: np.ndarray | None) -> np.ndarray | None:
    """Normalise a loaded image to 4-channel BGRA; ``None`` stays ``None``."""
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    if image.shape[2] == 4:
        return image
    return None


BACKGROUND_TOLERANCE = 40  # colour distance from the corner colour


def strip_opaque_background(image: np.ndarray) -> np.ndarray:
    """Make a baked-in background transparent; keep real alpha as-is.

    Some assets (e.g. 7.png, 10.png) have no usable alpha: their corners
    are opaque white/black, so the overlay painted a whole square. When
    the corners are already transparent, only clear the RGB hiding under
    transparent pixels, which otherwise bleeds white when resized/rotated.
    When the corners are opaque, drop every pixel that matches the corner
    colour and is connected to the image border. Interior pixels of the
    same colour (the black pupil centre, outlines) are not connected to
    the border through that colour alone once enclosed, so they survive.
    """
    result = image.copy()
    corners = np.array([
        result[0, 0, 3], result[0, -1, 3],
        result[-1, 0, 3], result[-1, -1, 3],
    ])
    if (corners < 128).any():
        result[result[:, :, 3] == 0, :3] = 0
        return result

    patches = np.concatenate([
        result[:5, :5, :3].reshape(-1, 3), result[:5, -5:, :3].reshape(-1, 3),
        result[-5:, :5, :3].reshape(-1, 3), result[-5:, -5:, :3].reshape(-1, 3),
    ])
    corner_colour = np.median(patches, axis=0).astype(np.int16)
    diff = np.abs(result[:, :, :3].astype(np.int16) - corner_colour)
    similar = (diff.max(axis=2) <= BACKGROUND_TOLERANCE).astype(np.uint8)
    count, labels = cv2.connectedComponents(similar, connectivity=4)
    if count <= 1:
        return result
    border_labels = set(np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1],
    ]))) - {0}
    if border_labels:
        result[np.isin(labels, list(border_labels)), 3] = 0
    result[result[:, :, 3] == 0, :3] = 0
    return result


def load_images(folder: Path) -> list[np.ndarray]:
    if not folder.is_dir():
        raise SystemExit(f"Error: assets folder not found: {folder}")
    paths = sorted(
        (p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
        key=natural_sort_key,
    )
    images: list[np.ndarray] = []
    for path in paths:
        image = ensure_bgra(cv2.imread(str(path), cv2.IMREAD_UNCHANGED))
        if image is None:
            print(f"Warning: could not load image, skipping: {path.name}")
            continue
        # Shrink first: background stripping and every later resize/rotate
        # then work on <=256px images instead of up to 6000px originals.
        height, width = image.shape[:2]
        scale = min(1.0, ASSET_MAX_SIZE / max(height, width))
        if scale < 1.0:
            image = cv2.resize(
                image, (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        images.append(strip_opaque_background(image))
    if not images:
        raise SystemExit(f"Error: no loadable Sharingan images in {folder}")
    return images


def load_audio_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        print(f"Warning: audio folder not found, sound disabled: {folder}")
        return []
    files = sorted(
        (p for p in folder.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES),
        key=natural_sort_key,
    )
    if not files:
        print("Warning: no audio files found, sound disabled.")
    return files


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    image = image.copy()
    image[image[:, :, 3] == 0, :3] = 0  # stop white RGB bleeding into edges
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT,
    )


ROTATION_TOTAL_DEG = ROTATION_SPEED_DEG_S * ROTATION_DURATION_S


def rotation_angle(elapsed_s: float) -> float | None:
    """Ease-out spin angle, or ``None`` when the spin has finished.

    Linear rotation stopped dead at 720 deg/s; an ease-out cubic covers
    the same total angle but decelerates into the final position.
    """
    if elapsed_s >= ROTATION_DURATION_S:
        return None
    progress = max(0.0, elapsed_s) / ROTATION_DURATION_S
    eased = 1.0 - (1.0 - progress) ** 3
    return (ROTATION_TOTAL_DEG * eased) % 360.0


def smooth_detections(previous: list, current: list,
                      alpha: float = SMOOTHING_ALPHA) -> list:
    """Exponentially smooth YOLO boxes so the overlay does not jitter.

    Boxes are matched left-to-right; if the count changed, the new set
    is used as-is (snapping is better than blending unrelated boxes).
    """
    if not previous or len(previous) != len(current):
        return list(current)
    prev_sorted = sorted(previous, key=lambda d: d[0])
    curr_sorted = sorted(current, key=lambda d: d[0])
    smoothed = []
    for prev, curr in zip(prev_sorted, curr_sorted):
        coords = tuple(
            int(round(prev[i] + alpha * (curr[i] - prev[i]))) for i in range(4)
        )
        smoothed.append((*coords, curr[4]))
    return smoothed


def overlay_image(frame: np.ndarray, overlay_bgra: np.ndarray,
                  center_x: int, center_y: int, size: int,
                  angle: float = 0.0) -> None:
    """Alpha-blend a square ``overlay_bgra`` centred on a pupil, in place.

    Resizes to pupil size first and only then rotates, so rotation costs
    a few hundred pixels instead of the full-resolution asset.
    """
    if size <= 0 or overlay_bgra is None:
        return
    clean = overlay_bgra.copy()
    clean[clean[:, :, 3] == 0, :3] = 0  # stop white RGB bleeding into edges
    resized = cv2.resize(clean, (size, size))
    if angle:
        resized = rotate_image(resized, angle)
    # Apply opacity once to a float copy instead of mutating the source.
    src = resized.astype(np.float32)
    src[:, :, 3] *= OVERLAY_OPACITY

    x0, y0 = center_x - size // 2, center_y - size // 2
    x1, y1 = x0 + size, y0 + size
    # Clip destination to the frame and source by the same amount.
    dx0, dy0 = max(x0, 0), max(y0, 0)
    dx1, dy1 = min(x1, frame.shape[1]), min(y1, frame.shape[0])
    if dx1 <= dx0 or dy1 <= dy0:
        return
    sx0, sy0 = dx0 - x0, dy0 - y0
    cropped = src[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]

    alpha = (cropped[:, :, 3:4] / 255.0).astype(np.float32)
    roi = frame[dy0:dy1, dx0:dx1].astype(np.float32)
    frame[dy0:dy1, dx0:dx1] = (
        alpha * cropped[:, :, :3] + (1.0 - alpha) * roi
    ).astype(np.uint8)


def should_rotate(index: int, total: int) -> bool:
    """Every stage spins on transition except the last (yellow) one."""
    return index != total - 1


def is_valid_blink(closed_frames: int, closed_duration_s: float) -> bool:
    """Any closure past the noise floor counts, however long it lasted."""
    return (
        closed_frames >= MIN_BLINK_FRAMES
        and closed_duration_s >= MIN_BLINK_DURATION_S
    )


def extract_detections(result) -> list[tuple[int, int, int, int, float]]:
    """Return (x_min, y_min, x_max, y_max, confidence) from a YOLO result.

    Uses ``boxes.conf`` alongside ``boxes.xyxy``; the old code iterated
    ``xyxy`` alone and unpacked 6 values that were never there, so the
    confidence threshold silently never applied.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
    detections = []
    for (x_min, y_min, x_max, y_max), confidence in zip(xyxy, conf):
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue
        detections.append(
            (int(x_min), int(y_min), int(x_max), int(y_max), float(confidence))
        )
    return detections


def parse_args(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description="Real-time Sharingan overlay")
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help="YOLO input size (lower = faster)")
    parser.add_argument("--detect-every", type=int, default=DEFAULT_DETECT_EVERY,
                        help="run YOLO every Nth frame")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="overlay size multiplier (0.5-2.0)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not MODEL_PATH.is_file():
        raise SystemExit(f"Error: YOLO model not found: {MODEL_PATH}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(f"Error: ultralytics is not installed: {exc}") from exc

    sharingan_images = load_images(ASSETS_DIR)
    audio_files = load_audio_files(AUDIO_DIR)
    if audio_files and len(audio_files) != len(sharingan_images):
        print(
            f"Warning: {len(sharingan_images)} images but {len(audio_files)} "
            "audio files; audio will cycle independently."
        )

    # Audio is optional: the overlay must still run without a mixer/device.
    music = None
    try:
        import pygame

        pygame.mixer.init()
        music = pygame.mixer.music
    except Exception as exc:  # pygame.error, ImportError, no audio device...
        print(f"Warning: audio disabled ({exc})")

    def play_audio(path: Path) -> None:
        if music is None:
            return
        try:
            music.load(str(path))
            music.play()
        except Exception as exc:
            print(f"Warning: could not play {path.name} ({exc})")

    def stop_audio() -> None:
        if music is not None:
            music.stop()

    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
    if not cap.isOpened():
        stop_audio()
        raise SystemExit(f"Error: could not open webcam (index {args.camera}).")

    current_index = 0
    last_transition = time.monotonic() - TRANSITION_COOLDOWN_S
    is_rotating = False
    rotation_start = 0.0
    scaling_factor = min(2.0, max(0.5, args.scale))
    show_hud = True
    detections: list = []
    frame_count = 0
    fps = 0.0
    fps_last = time.monotonic()

    eyes_were_open = True
    closed_frames = 0
    closed_since: float | None = None

    def set_stage(index: int, now: float) -> None:
        nonlocal current_index, is_rotating, rotation_start, last_transition
        stop_audio()
        current_index = index % len(sharingan_images)
        is_rotating = should_rotate(current_index, len(sharingan_images))
        rotation_start = now
        last_transition = now
        if audio_files:  # modulo: never index past the audio list
            play_audio(audio_files[current_index % len(audio_files)])

    def try_transition(now: float) -> None:
        if now - last_transition < TRANSITION_COOLDOWN_S:
            return
        set_stage(current_index + 1, now)

    def save_snapshot(frame) -> None:
        folder = BASE_DIR / "snapshots"
        folder.mkdir(exist_ok=True)
        path = folder / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(str(path), frame)
        print(f"Snapshot saved: {path}")

    print("Controls: blink / n = next, p = previous, r = reset, "
          "+/- = size, s = snapshot, f = HUD, q = quit")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: lost webcam frame, stopping.")
                break
            now = time.monotonic()
            frame_count += 1
            if now > fps_last:  # exponential moving average FPS
                instant = 1.0 / max(now - fps_last, 1e-6) if frame_count > 1 else 0
                fps = instant if fps == 0 else 0.9 * fps + 0.1 * instant
            fps_last = now

            # YOLO is the expensive part: run it every Nth frame and
            # reuse (smoothed) detections in between. Blink state only
            # advances on fresh detections, so timing stays honest.
            fresh = (frame_count - 1) % max(1, args.detect_every) == 0
            if fresh:
                results = model(frame, imgsz=args.imgsz,
                                conf=CONFIDENCE_THRESHOLD, verbose=False)
                raw = extract_detections(results[0]) if results else []
                detections = smooth_detections(detections, raw)
            eyes_open = bool(detections)

            # Ease-out spin, computed on the pupil-size image inside
            # overlay_image. Last (yellow) stage never rotates.
            angle = 0.0
            if is_rotating and should_rotate(current_index, len(sharingan_images)):
                computed = rotation_angle(now - rotation_start)
                if computed is None:
                    is_rotating = False
                else:
                    angle = computed

            for x_min, y_min, x_max, y_max, _conf in detections:
                width, height = x_max - x_min, y_max - y_min
                if width <= 0 or height <= 0:
                    continue
                # Square target: a circular Sharingan must not be stretched
                # into the rectangular pupil box.
                size = int(max(width, height) * scaling_factor)
                overlay_image(
                    frame, sharingan_images[current_index],
                    x_min + width // 2, y_min + height // 2, size, angle,
                )

            # Blink = closure sandwiched by open eyes. Any closure past
            # the noise floor counts, however long it lasted.
            if fresh:
                if eyes_open:
                    if not eyes_were_open and closed_since is not None:
                        if is_valid_blink(closed_frames, now - closed_since):
                            try_transition(now)
                    closed_frames = 0
                    closed_since = None
                else:
                    if eyes_were_open:
                        closed_since = now
                    closed_frames += 1
                eyes_were_open = eyes_open

            if show_hud:
                cv2.putText(
                    frame,
                    f"Stage {current_index + 1}/{len(sharingan_images)}"
                    f"  {fps:.0f} fps  {'Open' if eyes_open else 'Closed'}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
                )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or Esc
                break
            elif key in (ord("n"), ord(" ")):
                set_stage(current_index + 1, now)
            elif key == ord("p"):
                set_stage(current_index - 1, now)
            elif key == ord("r"):
                set_stage(0, now)
            elif key in (ord("+"), ord("=")):
                scaling_factor = min(2.0, scaling_factor + 0.1)
            elif key in (ord("-"), ord("_")):
                scaling_factor = max(0.5, scaling_factor - 0.1)
            elif key == ord("f"):
                show_hud = not show_hud
            elif key == ord("s"):
                save_snapshot(frame)
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        stop_audio()
        if music is not None:
            try:
                import pygame

                pygame.mixer.quit()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

