"""电脑摄像头红绿蓝实心圆定位；算法位于 pc_circle_vision.py。"""

import argparse
import json
import math
import time
from pathlib import Path

import cv2

from pc_circle_vision import (
    ALIGN_FRAMES, HSV_RANGES, STABLE_FRAMES, CircleTracker,
    clean_mask, detect_circle, detect_circles, make_mask,
)

CONFIG_FILE = Path(__file__).with_name("pc_circle_config.json")
WINDOW = "RGB circle centers | 1 Red 2 Green 3 Blue | C calibrate | M mask | Q quit"
MASK_WINDOW = "Selected color mask (raw | cleaned)"
DRAW_COLORS = {"RED": (0, 0, 255), "GREEN": (0, 255, 0), "BLUE": (255, 0, 0)}


def validate_roi(roi, width, height):
    if roi is None:
        return (0, 0, width, height)
    x, y, w, h = roi
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
        raise ValueError("ROI must be inside the actual camera frame: %dx%d" % (width, height))
    return tuple(roi)


def calibration_context(camera_index, width, height, roi):
    return {"camera": camera_index, "width": width, "height": height, "roi": list(roi)}


def load_grab_point(context):
    if not CONFIG_FILE.exists():
        return None
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != 2 or data.get("context") != context:
            raise ValueError("camera/resolution/ROI changed or legacy calibration")
        point = data["grab_x"], data["grab_y"]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in point):
            raise ValueError("invalid coordinates")
        x, y, w, h = context["roi"]
        if not (x <= point[0] < x + w and y <= point[1] < y + h):
            raise ValueError("calibration outside ROI")
        return point
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("Calibration ignored; press C to recalibrate:", exc)
        return None


def save_grab_point(point, context):
    data = {"version": 2, "grab_x": point[0], "grab_y": point[1], "context": context}
    temporary = CONFIG_FILE.with_suffix(".json.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(CONFIG_FILE)
    finally:
        temporary.unlink(missing_ok=True)
    print("Saved grab point:", point)


def open_camera(index, width, height):
    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Cannot open camera. Check USB/index, camera permissions and other camera apps.")
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


def run(camera_index, width, height, roi=None):
    if width <= 0 or height <= 0:
        raise ValueError("Requested width and height must be positive")
    camera = open_camera(camera_index, width, height)
    selected, show_mask = "BLUE", False
    trackers = {color: CircleTracker() for color in HSV_RANGES}
    grab_point = context = frame_size = None
    fps = 0.0
    previous_time = time.perf_counter()
    frame_number = 0
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        print("1/2/3=RED/GREEN/BLUE | C=calibrate | M=raw/clean masks | R=reset | Q/Esc=quit")
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera frame read failed; no current target.")
                break
            actual_height, actual_width = frame.shape[:2]
            if frame_size != (actual_width, actual_height):
                active_roi = validate_roi(roi, actual_width, actual_height)
                context = calibration_context(camera_index, actual_width, actual_height, active_roi)
                # 实际分辨率变化后，清空全部历史，不继续使用旧坐标。
                trackers = {color: CircleTracker() for color in HSV_RANGES}
                grab_point = load_grab_point(context)
                frame_size = actual_width, actual_height
                print("Actual frame:", frame_size, "ROI:", active_roi)
            x, y, w, h = active_roi
            hsv = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
            raw_masks = {color: make_mask(hsv, color, clean=False) for color in HSV_RANGES}
            masks = {color: clean_mask(raw_masks[color]) for color in HSV_RANGES}
            for color, tracker in trackers.items():
                candidates = detect_circles(masks[color], raw_masks[color], offset=(x, y))
                tracker.update(candidates, grab_point)
            cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), (160, 160, 160), 1)
            if grab_point is not None:
                cv2.drawMarker(frame, tuple(round(v) for v in grab_point), (255, 255, 255),
                               cv2.MARKER_CROSS, 24, 2)
            for color, tracker in trackers.items():
                if tracker.target is None:
                    continue
                cx, cy, radius = (round(v) for v in tracker.target[:3])
                ink = DRAW_COLORS[color]
                cv2.circle(frame, (cx, cy), radius, ink, 2)
                cv2.drawMarker(frame, (cx, cy), ink, cv2.MARKER_CROSS, 18, 2)
                cv2.putText(frame, "%s (%d,%d)" % (color, cx, cy),
                            (max(5, cx - radius), max(90, cy - radius - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, ink, 2, cv2.LINE_AA)
            tracker = trackers[selected]
            target = tracker.target
            if target is None:
                status = "%s: NO CIRCLE" % selected
            elif grab_point is None:
                status = "%s: CALIBRATE (%.1f, %.1f) - press C when stable" % (selected, *target[:2])
            else:
                dx, dy = target.cx - grab_point[0], target.cy - grab_point[1]
                status = "%s: %s dx=%+.1f dy=%+.1f px" % (
                    selected, "ALIGNED" if tracker.aligned else "OFFSET", dx, dy)
                cv2.line(frame, (round(target.cx), round(target.cy)),
                         tuple(round(v) for v in grab_point), (0, 255, 255), 1)
            now = time.perf_counter()
            dt, previous_time = now - previous_time, now
            if dt > 0:
                fps = 0.85 * fps + 0.15 / dt if fps else 1 / dt
            cv2.rectangle(frame, (0, 0), (actual_width, 68), (0, 0, 0), -1)
            cv2.putText(frame, status, (10, 27), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, DRAW_COLORS[selected], 1, cv2.LINE_AA)
            info = "FPS %.1f | stable %s | align %d/%d | M mask" % (
                fps, "YES" if tracker.stable else "NO", tracker.align_count, ALIGN_FRAMES)
            cv2.putText(frame, info, (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, frame)
            if show_mask:
                cv2.imshow(MASK_WINDOW, cv2.hconcat([raw_masks[selected], masks[selected]]))
            frame_number += 1
            if frame_number % 10 == 0:
                print(status, "stable=", tracker.stable, "align=", tracker.align_count,
                      "fps=%.1f" % fps)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("1"), ord("2"), ord("3")):
                selected = {ord("1"): "RED", ord("2"): "GREEN", ord("3"): "BLUE"}[key]
                for item in trackers.values():
                    item.reset_alignment()
            if key == ord("c"):
                tracker = trackers[selected]
                if tracker.target is not None and tracker.stable:
                    # 标定使用最近原始检测的均值，避免沿用滞后的平滑坐标。
                    point = tuple(sum(item[i] for item in tracker.history) / len(tracker.history)
                                  for i in (0, 1))
                    try:
                        save_grab_point(point, context)
                    except OSError as exc:
                        print("Calibration save failed:", exc)
                    else:
                        grab_point = point
                        for item in trackers.values():
                            item.reset_alignment()
                else:
                    print("Keep the selected circle still for at least %d frames." % STABLE_FRAMES)
            if key == ord("r"):
                grab_point = None
                for item in trackers.values():
                    item.reset_alignment()
                try:
                    CONFIG_FILE.unlink(missing_ok=True)
                except OSError as exc:
                    print("Calibration cleared in memory; file removal failed:", exc)
            if key == ord("m"):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow(MASK_WINDOW)
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera released.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect red/green/blue solid circle centers")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                        help="Search region in actual frame pixels; default is full frame")
    args = parser.parse_args()
    run(args.camera, args.width, args.height, args.roi)
