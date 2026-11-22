"""Focused tests for the pure helpers in sharingan.py (no webcam/model)."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sharingan


class NaturalSortTests(unittest.TestCase):
    def test_numeric_order(self):
        names = ["10.png", "2.png", "1.png"]
        ordered = sorted((Path(n) for n in names), key=sharingan.natural_sort_key)
        self.assertEqual([p.name for p in ordered], ["1.png", "2.png", "10.png"])

    def test_non_numeric_names_do_not_crash(self):
        names = ["sharingan-final.png", "stage1.png"]
        ordered = sorted((Path(n) for n in names), key=sharingan.natural_sort_key)
        self.assertEqual(len(ordered), 2)


class EnsureBgraTests(unittest.TestCase):
    def test_none_stays_none(self):
        self.assertIsNone(sharingan.ensure_bgra(None))

    def test_three_channel_jpg_gains_alpha(self):
        out = sharingan.ensure_bgra(np.zeros((4, 4, 3), dtype=np.uint8))
        self.assertEqual(out.shape, (4, 4, 4))
        self.assertTrue((out[:, :, 3] == 255).all())

    def test_four_channel_unchanged(self):
        img = np.zeros((4, 4, 4), dtype=np.uint8)
        self.assertEqual(sharingan.ensure_bgra(img).shape, (4, 4, 4))


class StripBackgroundTests(unittest.TestCase):
    def _disc_on_background(self, bg_bgr):
        img = np.zeros((40, 40, 4), dtype=np.uint8)
        img[:, :, :3] = bg_bgr
        img[:, :, 3] = 255
        # Red disc with a black centre, like a Sharingan on a baked-in bg.
        import cv2

        cv2.circle(img, (20, 20), 14, (0, 0, 255, 255), -1)
        cv2.circle(img, (20, 20), 4, (0, 0, 0, 255), -1)
        return img

    def test_white_background_becomes_transparent(self):
        out = sharingan.strip_opaque_background(self._disc_on_background((255, 255, 255)))
        self.assertEqual(out[0, 0, 3], 0)       # corner transparent
        self.assertEqual(out[20, 33, 3], 255)   # red disc kept
        self.assertEqual(out[20, 20, 3], 255)   # enclosed black centre kept

    def test_black_background_becomes_transparent(self):
        out = sharingan.strip_opaque_background(self._disc_on_background((0, 0, 0)))
        self.assertEqual(out[0, 0, 3], 0)
        self.assertEqual(out[20, 33, 3], 255)

    def test_existing_alpha_kept_and_rgb_cleared(self):
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        img[:, :, :3] = 255  # white RGB hiding under transparency
        img[:, :, 3] = 0
        out = sharingan.strip_opaque_background(img)
        self.assertEqual(out[0, 0, 3], 0)
        self.assertEqual(int(out[0, 0, 0]), 0)  # no white bleed on resize


class ParseArgsTests(unittest.TestCase):
    def test_defaults_are_the_fast_settings(self):
        args = sharingan.parse_args([])
        self.assertEqual(args.imgsz, 320)
        self.assertEqual(args.detect_every, 2)
        self.assertEqual(args.camera, 0)


class SmoothingTests(unittest.TestCase):
    def test_smooths_towards_new_box(self):
        prev = [(100, 100, 150, 150, 0.9)]
        curr = [(110, 90, 160, 140, 0.8)]
        out = sharingan.smooth_detections(prev, curr, alpha=0.5)
        self.assertEqual(out[0][:4], (105, 95, 155, 145))
        self.assertAlmostEqual(out[0][4], 0.8, places=5)  # fresh confidence

    def test_count_change_snaps_to_new(self):
        curr = [(0, 0, 10, 10, 0.9), (20, 20, 30, 30, 0.9)]
        self.assertEqual(sharingan.smooth_detections([(0, 0, 5, 5, 0.5)], curr), curr)

    def test_empty_previous_returns_current(self):
        curr = [(0, 0, 10, 10, 0.9)]
        self.assertEqual(sharingan.smooth_detections([], curr), curr)


class RotationAngleTests(unittest.TestCase):
    def test_finished_after_duration(self):
        self.assertIsNone(sharingan.rotation_angle(sharingan.ROTATION_DURATION_S + 0.1))

    def test_starts_at_zero_and_eases(self):
        self.assertEqual(sharingan.rotation_angle(0), 0.0)
        quarter = sharingan.rotation_angle(sharingan.ROTATION_DURATION_S / 4)
        half = sharingan.rotation_angle(sharingan.ROTATION_DURATION_S / 2)
        self.assertIsNotNone(quarter)
        self.assertIsNotNone(half)


class RotationTests(unittest.TestCase):
    def test_last_stage_does_not_rotate(self):
        self.assertFalse(sharingan.should_rotate(11, 12))

    def test_other_stages_rotate(self):
        self.assertTrue(sharingan.should_rotate(0, 12))
        self.assertTrue(sharingan.should_rotate(10, 12))


class BlinkTests(unittest.TestCase):
    def test_brief_closure_is_blink(self):
        self.assertTrue(sharingan.is_valid_blink(5, 0.2))

    def test_noise_is_not_blink(self):
        self.assertFalse(sharingan.is_valid_blink(1, 0.01))

    def test_long_closure_still_counts(self):
        # Regression: an upper duration/frame cap rejected normal
        # close-and-open at YOLO's low frame rate, so stages never changed.
        self.assertTrue(sharingan.is_valid_blink(60, 2.0))
        self.assertTrue(sharingan.is_valid_blink(3, 1.5))


class FakeTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class FakeBoxes:
    def __init__(self, xyxy, conf):
        self.xyxy = FakeTensor(xyxy)
        self.conf = FakeTensor(conf)

    def __len__(self):
        return len(self.xyxy.numpy())


class FakeResult:
    def __init__(self, xyxy, conf):
        self.boxes = FakeBoxes(xyxy, conf)


class ExtractDetectionsTests(unittest.TestCase):
    def test_confidence_threshold_applies(self):
        result = FakeResult(
            [[0, 0, 10, 10], [20, 20, 30, 30]], [0.9, 0.3]
        )
        detections = sharingan.extract_detections(result)
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0][4], 0.9, places=5)

    def test_no_boxes(self):
        class Empty:
            boxes = None

        self.assertEqual(sharingan.extract_detections(Empty()), [])


class OverlayTests(unittest.TestCase):
    def test_overlay_at_frame_edge_does_not_crash_or_misalign(self):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        overlay = np.zeros((8, 8, 4), dtype=np.uint8)
        overlay[:, :, :3] = 255
        overlay[:, :, 3] = 255
        sharingan.overlay_image(frame, overlay, 0, 0, 8)  # pupil at corner
        self.assertGreater(frame[0:4, 0:4].sum(), 0)

    def test_overlay_inside_frame(self):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        overlay = np.zeros((8, 8, 4), dtype=np.uint8)
        overlay[:, :, :3] = 255
        overlay[:, :, 3] = 255
        sharingan.overlay_image(frame, overlay, 10, 10, 8)
        self.assertGreater(frame[10, 10].sum(), 0)


if __name__ == "__main__":
    unittest.main()
