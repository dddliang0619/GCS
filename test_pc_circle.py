"""离线回归：实际图像算法、目标关联、对准状态和标定上下文。"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import pc_circle_camera as camera
import pc_circle_vision as vision
from pc_circle_vision import CircleTracker, Detection


def detection(x=160, y=120, radius=30, score=1000):
    return Detection(x, y, radius, score)


def disk(center=(160, 120), radius=35):
    mask = np.zeros((240, 320), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return mask


class DetectionTests(unittest.TestCase):
    def test_three_colors_and_red_wraparound(self):
        for color, hue in (("RED", 0), ("RED", 179), ("GREEN", 60), ("BLUE", 115)):
            with self.subTest(color=color, hue=hue):
                hsv = np.zeros((240, 320, 3), dtype=np.uint8)
                hsv[disk() != 0] = (hue, 220, 220)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                converted = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                raw = vision.make_mask(converted, color, clean=False)
                results = vision.detect_circles(vision.clean_mask(raw), raw)
                self.assertEqual(len(results), 1)
                self.assertAlmostEqual(results[0].cx, 160, delta=0.5)
                self.assertAlmostEqual(results[0].cy, 120, delta=0.5)

    def test_sizes_and_position_accuracy(self):
        for radius in (15, 20, 50, 90):
            with self.subTest(radius=radius):
                raw = disk((155, 122), radius)
                result = vision.detect_circles(vision.clean_mask(raw), raw)[0]
                self.assertAlmostEqual(result.cx, 155, delta=0.5)
                self.assertAlmostEqual(result.cy, 122, delta=0.5)
                self.assertAlmostEqual(result.radius, radius, delta=1)

    def test_blank_noise_and_undersized_objects(self):
        raw = np.zeros((240, 320), dtype=np.uint8)
        raw[::20, ::20] = 255
        cv2.circle(raw, (100, 100), 9, 255, -1)
        self.assertEqual(vision.detect_circles(vision.clean_mask(raw), raw), [])

    def test_square_is_not_circle(self):
        raw = np.zeros((240, 320), dtype=np.uint8)
        cv2.rectangle(raw, (100, 60), (180, 140), 255, -1)
        self.assertEqual(vision.detect_circles(vision.clean_mask(raw), raw), [])

    def test_ring_and_large_hole_rejected(self):
        for hole in (18, 28):
            raw = disk()
            cv2.circle(raw, (160, 120), hole, 0, -1)
            self.assertEqual(vision.detect_circles(vision.clean_mask(raw), raw), [])

    def test_small_highlight_tolerated(self):
        raw = disk()
        cv2.circle(raw, (155, 120), 3, 0, -1)
        self.assertEqual(len(vision.detect_circles(vision.clean_mask(raw), raw)), 1)

    def test_raw_mask_exposes_holes_closed_by_morphology(self):
        raw = disk()
        raw[95:145:4, 135:185:4] = 0
        # 模拟一个外形完整、内部颜色实际大量缺失的候选。
        raw[105:135, 145:175] = 0
        self.assertEqual(vision.detect_circles(disk(), raw), [])

    def test_cut_off_and_elongated_shapes_rejected(self):
        raw = disk()
        raw[:, :155] = 0
        self.assertEqual(vision.detect_circles(vision.clean_mask(raw), raw), [])
        raw[:] = 0
        cv2.ellipse(raw, (160, 120), (65, 30), 0, 0, 360, 255, -1)
        self.assertEqual(vision.detect_circles(vision.clean_mask(raw), raw), [])

    def test_border_clipped_circle_rejected(self):
        for center in ((0, 120), (30, 120), (319, 120), (160, 0), (160, 239)):
            raw = disk(center)
            self.assertEqual(vision.detect_circles(vision.clean_mask(raw), raw), [])

    def test_roi_offset_and_multiple_candidates(self):
        raw = disk((70, 70), 25)
        cv2.circle(raw, (230, 150), 40, 255, -1)
        results = vision.detect_circles(vision.clean_mask(raw), raw, offset=(40, 30))
        self.assertEqual(len(results), 2)
        self.assertEqual(sorted((round(item.cx), round(item.cy)) for item in results),
                         [(110, 100), (270, 180)])

    def test_degenerate_fit(self):
        contour = np.array([[[1, 1]], [[2, 2]], [[3, 3]]], dtype=np.int32)
        self.assertIsNone(vision.fit_circle(contour))


class TrackerTests(unittest.TestCase):
    def test_requires_stability_then_consecutive_alignment(self):
        tracker = CircleTracker()
        total = vision.STABLE_FRAMES + vision.ALIGN_FRAMES - 1
        for index in range(total):
            tracker.update([detection()], (160, 120))
            self.assertEqual(tracker.aligned, index == total - 1)
        self.assertTrue(tracker.stable)

    def test_no_calibration_never_aligned(self):
        tracker = CircleTracker()
        for _ in range(20):
            tracker.update([detection()])
        self.assertTrue(tracker.stable)
        self.assertFalse(tracker.aligned)

    def test_slow_motion_not_stable(self):
        tracker = CircleTracker()
        for index in range(20):
            tracker.update([detection(x=150 + index)], (160, 120))
            self.assertFalse(tracker.stable)
            self.assertFalse(tracker.aligned)

    def test_radius_variation_not_stable(self):
        tracker = CircleTracker()
        for index in range(20):
            tracker.update([detection(radius=30 + (index % 2) * 4)], (160, 120))
            self.assertFalse(tracker.aligned)

    def test_larger_distractor_does_not_steal_lock(self):
        tracker = CircleTracker()
        tracker.update([detection()])
        for _ in range(10):
            target = tracker.update([detection(score=500), detection(x=240, radius=50, score=99999)])
            self.assertEqual(target.cx, 160)

    def test_initial_acquisition_nearest_grab_point(self):
        tracker = CircleTracker()
        target = tracker.update([detection(x=70, score=99999), detection(x=162)], (160, 120))
        self.assertEqual(target.cx, 162)

    def test_loss_invalidates_output_and_reserves_anchor(self):
        tracker = CircleTracker()
        for _ in range(12):
            tracker.update([detection()], (160, 120))
        self.assertTrue(tracker.aligned)
        self.assertIsNone(tracker.update([detection(x=240)], (160, 120)))
        self.assertIsNone(tracker.target)
        self.assertFalse(tracker.aligned)
        self.assertIsNotNone(tracker.anchor)
        tracker.update([detection()], (160, 120))
        self.assertFalse(tracker.stable)
        self.assertEqual(tracker.align_count, 0)

    def test_reacquisition_after_timeout_starts_fresh(self):
        tracker = CircleTracker()
        tracker.update([detection()])
        for _ in range(vision.MAX_MISSED_FRAMES):
            self.assertIsNone(tracker.update([detection(x=250)]))
        self.assertIsNone(tracker.anchor)
        target = tracker.update([detection(x=250)], (250, 120))
        self.assertEqual(target.cx, 250)
        self.assertFalse(tracker.stable)
        self.assertFalse(tracker.aligned)

    def test_raw_outside_tolerance_cancels_alignment_despite_smoothing(self):
        tracker = CircleTracker()
        for _ in range(12):
            tracker.update([detection(x=164)], (160, 120))
        self.assertTrue(tracker.aligned)
        tracker.update([detection(x=166)], (160, 120))
        self.assertLess(tracker.target.cx, 165)
        self.assertFalse(tracker.aligned)
        self.assertEqual(tracker.align_count, 0)

    def test_reset_alignment_on_calibration_or_selection(self):
        tracker = CircleTracker()
        for _ in range(12):
            tracker.update([detection()], (160, 120))
        tracker.reset_alignment()
        self.assertFalse(tracker.aligned)
        self.assertEqual(tracker.align_count, 0)


class CalibrationAndCameraTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix=".circle-test-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "calibration.json"
        self.patcher = patch.object(camera, "CONFIG_FILE", self.path)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.context = camera.calibration_context(0, 640, 480, (20, 30, 500, 400))

    def test_calibration_roundtrip_and_changed_context(self):
        with contextlib.redirect_stdout(io.StringIO()):
            camera.save_grab_point((320.25, 240.5), self.context)
            self.assertEqual(camera.load_grab_point(self.context), (320.25, 240.5))
            for key, value in (("camera", 1), ("width", 1280), ("height", 720),
                               ("roi", [0, 0, 640, 480])):
                changed = dict(self.context, **{key: value})
                self.assertIsNone(camera.load_grab_point(changed))
        self.assertFalse(self.path.with_suffix(".json.tmp").exists())

    def test_bad_and_legacy_calibrations_ignored(self):
        for data in ("{bad", "[]", '{"grab_x": 320, "grab_y": 240}',
                     json.dumps({"version": 2, "context": self.context, "grab_x": -1, "grab_y": 100}),
                     json.dumps({"version": 2, "context": self.context, "grab_x": float("nan"), "grab_y": 100}),
                     json.dumps({"version": 2, "context": self.context, "grab_x": True, "grab_y": 100})):
            self.path.write_text(data, encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertIsNone(camera.load_grab_point(self.context))

    def test_roi_validation(self):
        self.assertEqual(camera.validate_roi(None, 640, 480), (0, 0, 640, 480))
        for roi in ((-1, 0, 20, 20), (0, 0, 0, 20), (600, 0, 100, 100)):
            with self.assertRaises(ValueError):
                camera.validate_roi(roi, 640, 480)

    def test_camera_released_if_window_creation_fails(self):
        device = Mock()
        with patch.object(camera, "open_camera", return_value=device), \
                patch.object(camera.cv2, "namedWindow", side_effect=RuntimeError("UI failed")), \
                patch.object(camera.cv2, "destroyAllWindows"), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "UI failed"):
                camera.run(0, 640, 480)
        device.release.assert_called_once()

    def test_mock_camera_end_to_end_calibrate_align_and_lose(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.circle(image, (160, 120), 35, (255, 0, 0), -1)
        blank = np.zeros_like(image)
        frames = [image.copy() for _ in range(15)] + [blank]
        device = Mock()
        device.read.side_effect = [(True, frame) for frame in frames]
        keys = [-1] * len(frames)
        keys[5] = ord("c")
        keys[-1] = ord("q")
        with patch.object(camera, "open_camera", return_value=device), \
                patch.object(camera.cv2, "namedWindow"), \
                patch.object(camera.cv2, "imshow"), \
                patch.object(camera.cv2, "destroyAllWindows"), \
                patch.object(camera.cv2, "waitKey", side_effect=keys), \
                patch.object(camera.cv2, "putText", wraps=cv2.putText) as overlay, \
                contextlib.redirect_stdout(io.StringIO()):
            camera.run(0, 320, 240)
        labels = [call.args[1] for call in overlay.call_args_list]
        self.assertTrue(any("ALIGNED" in label for label in labels))
        self.assertEqual(labels[-2], "BLUE: NO CIRCLE")
        self.assertTrue(self.path.exists())
        device.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
