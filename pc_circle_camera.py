"""电脑摄像头识别红、绿、蓝圆形物料的圆心，并显示夹取偏差。"""

import argparse
import json
import math
import time
from pathlib import Path

import cv2


CONFIG_FILE = Path(__file__).with_name("pc_circle_config.json")
WINDOW = "RGB circle centers | 1 Red 2 Green 3 Blue | C calibrate | M mask | Q quit"
MASK_WINDOW = "Selected color mask"

# OpenCV HSV: H=0..179, S/V=0..255。现场光照不同，先按 M 检查二值图再调范围。
HSV_RANGES = {
    "RED": [((0, 90, 50), (12, 255, 255)),
            ((170, 90, 50), (179, 255, 255))],
    "GREEN": [((35, 65, 45), (85, 255, 255))],
    "BLUE": [((90, 65, 45), (135, 255, 255))],
}
DRAW_COLORS = {"RED": (0, 0, 255), "GREEN": (0, 255, 0), "BLUE": (255, 0, 0)}

MIN_AREA = 500
MIN_RADIUS = 14
MAX_RADIUS = 150
MIN_CIRCULARITY = 0.65
MIN_FILL_RATIO = 0.58
MAX_CENTER_DISAGREEMENT = 0.28  # 色块质心与外接圆圆心相差不得超过半径的此比例
STABLE_FRAMES = 5
MAX_CENTER_JUMP = 16
MAX_RADIUS_JUMP = 10
SMOOTHING = 0.45
ALIGN_TOLERANCE_PX = 5


def load_grab_point():
    if not CONFIG_FILE.exists():
        return None
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        return int(data["grab_x"]), int(data["grab_y"])
    except (OSError, ValueError, KeyError, TypeError):
        print("Calibration file invalid; starting uncalibrated.")
        return None


def save_grab_point(point):
    CONFIG_FILE.write_text(
        json.dumps({"grab_x": point[0], "grab_y": point[1]}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Saved grab point:", point)


def make_mask(hsv, color):
    mask = None
    for low, high in HSV_RANGES[color]:
        part = cv2.inRange(hsv, low, high)
        mask = part if mask is None else cv2.bitwise_or(mask, part)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_circle(mask):
    """从指定颜色的实心圆色块求质心；返回 (x,y,r,score) 或 None。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue
        (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
        if not MIN_RADIUS <= radius <= MAX_RADIUS:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        fill_ratio = area / (math.pi * radius * radius)
        if circularity < MIN_CIRCULARITY or fill_ratio < MIN_FILL_RATIO:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        if math.hypot(cx - circle_x, cy - circle_y) > max(
            5, MAX_CENTER_DISAGREEMENT * radius
        ):
            continue
        score = area * min(circularity, 1.0) * min(fill_ratio, 1.0)
        if score > best_score:
            best = (cx, cy, radius, score)
            best_score = score
    return best


def update_track(previous, detection):
    """一帧丢失就复位，防止旧坐标被误用；对有效圆心做轻度平滑。"""
    if detection is None:
        return None
    if previous is None:
        return (detection[0], detection[1], detection[2], 1)
    jump = max(abs(detection[0] - previous[0]),
               abs(detection[1] - previous[1]))
    radius_jump = abs(detection[2] - previous[2])
    if jump > MAX_CENTER_JUMP or radius_jump > MAX_RADIUS_JUMP:
        return (detection[0], detection[1], detection[2], 1)
    return (
        SMOOTHING * detection[0] + (1 - SMOOTHING) * previous[0],
        SMOOTHING * detection[1] + (1 - SMOOTHING) * previous[1],
        SMOOTHING * detection[2] + (1 - SMOOTHING) * previous[2],
        min(previous[3] + 1, 999),
    )


def open_camera(index, width, height):
    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            "Cannot open camera. Check USB connection/index, close MATLAB or other "
            "camera apps, and check Windows camera permissions."
        )
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


def run(camera_index, width, height):
    camera = open_camera(camera_index, width, height)
    grab_point = load_grab_point()
    selected = "BLUE"
    show_mask = False
    tracks = {color: None for color in HSV_RANGES}
    fps = 0.0
    previous_time = time.perf_counter()
    frame_number = 0
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print("1=RED 2=GREEN 3=BLUE | C=calibrate selected stable circle | M=mask | R=reset | Q/Esc=quit")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera frame read failed.")
                break
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            masks = {color: make_mask(hsv, color) for color in HSV_RANGES}
            detections = {color: detect_circle(masks[color]) for color in HSV_RANGES}
            for color in HSV_RANGES:
                tracks[color] = update_track(tracks[color], detections[color])

            if grab_point is not None:
                cv2.drawMarker(frame, grab_point, (255, 255, 255),
                               cv2.MARKER_CROSS, 24, 2)
            for color, target in tracks.items():
                if target is None:
                    continue
                cx, cy, radius = map(int, target[:3])
                ink = DRAW_COLORS[color]
                cv2.circle(frame, (cx, cy), radius, ink, 2)
                cv2.drawMarker(frame, (cx, cy), ink, cv2.MARKER_CROSS, 18, 2)
                cv2.putText(frame, "%s (%d,%d)" % (color, cx, cy),
                            (max(5, cx - radius), max(20, cy - radius - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, ink, 2, cv2.LINE_AA)

            target = tracks[selected]
            if target is None:
                status = "%s: NO CIRCLE" % selected
            elif grab_point is None:
                status = "%s: (%.1f, %.1f) - place at grab point, press C" % (
                    selected, target[0], target[1]
                )
            else:
                dx = target[0] - grab_point[0]
                dy = target[1] - grab_point[1]
                aligned = (target[3] >= STABLE_FRAMES
                           and abs(dx) <= ALIGN_TOLERANCE_PX
                           and abs(dy) <= ALIGN_TOLERANCE_PX)
                status = "%s: %s dx=%+.1f dy=%+.1f px" % (
                    selected, "ALIGNED" if aligned else "OFFSET", dx, dy
                )
                cv2.line(frame, (int(target[0]), int(target[1])),
                         grab_point, (0, 255, 255), 1)

            now = time.perf_counter()
            dt = now - previous_time
            previous_time = now
            if dt > 0:
                fps = 0.85 * fps + 0.15 / dt if fps else 1 / dt
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 68), (0, 0, 0), -1)
            cv2.putText(frame, status, (10, 27), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, DRAW_COLORS[selected], 2, cv2.LINE_AA)
            cv2.putText(frame, "FPS %.1f | stable %d | 1 RED 2 GREEN 3 BLUE | M mask" %
                        (fps, target[3] if target else 0), (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255),
                        1, cv2.LINE_AA)
            cv2.imshow(WINDOW, frame)
            if show_mask:
                cv2.imshow(MASK_WINDOW, masks[selected])

            frame_number += 1
            if frame_number % 10 == 0:
                summary = ["%s:%s" % (color, "-" if tracks[color] is None else
                           "(%.1f,%.1f)" % tracks[color][:2])
                           for color in HSV_RANGES]
                print(" ".join(summary), "selected=", selected,
                      "fps=%.1f" % fps)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("1"), ord("2"), ord("3")):
                selected = {ord("1"): "RED", ord("2"): "GREEN",
                            ord("3"): "BLUE"}[key]
            if key == ord("c"):
                target = tracks[selected]
                if target is not None and target[3] >= STABLE_FRAMES:
                    grab_point = round(target[0]), round(target[1])
                    save_grab_point(grab_point)
                else:
                    print("Wait for the selected color circle to stabilize for 5 frames.")
            if key == ord("r"):
                grab_point = None
                if CONFIG_FILE.exists():
                    CONFIG_FILE.unlink()
                print("Grab point reset.")
            if key == ord("m"):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow(MASK_WINDOW)
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera released.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect red/green/blue circle centers")
    parser.add_argument("--camera", type=int, default=0, help="camera index, default 0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    run(args.camera, args.width, args.height)
