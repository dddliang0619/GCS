# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
"""K230 顶视圆心定位。用 VS Code 的 CanMV 扩展运行，不用 Python/F5 调试。"""

import sys

if sys.implementation.name != "micropython":
    raise SystemExit(
        "K230 firmware script: do not run with desktop Python. "
        "Connect the board and use CanMV: Run Active Python Script."
    )

import gc
import os
import time

from media.display import Display
from media.media import MediaManager
from media.sensor import Sensor


# ===== 首次运行只调整这一段 =====
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
ROI = (40, 30, 240, 180)  # x, y, w, h
X_STRIDE = 3
Y_STRIDE = 2
HOUGH_THRESHOLD = 2000
MIN_RADIUS = 12
MAX_RADIUS = 65

# 先把物料放在夹爪确实能夹中的位置，读取稳定圆心，然后修改下面 3 项。
GRAB_X = 160
GRAB_Y = 120
GRAB_POINT_CALIBRATED = False

ALIGN_TOLERANCE_X = 4
ALIGN_TOLERANCE_Y = 4
STABLE_FRAMES = 5
MAX_CENTER_JUMP = 12
MAX_RADIUS_JUMP = 7
PRINT_EVERY_N_FRAMES = 5


def choose_circle(img, previous):
    best = None
    best_score = None
    circles = img.find_circles(
        roi=ROI,
        x_stride=X_STRIDE,
        y_stride=Y_STRIDE,
        threshold=HOUGH_THRESHOLD,
    )
    for circle in circles:
        radius = circle.r()
        if radius < MIN_RADIUS or radius > MAX_RADIUS:
            continue
        cx, cy = circle.x(), circle.y()
        strength = circle.magnitude()
        if previous is None:
            score = strength
        else:
            distance2 = (cx - previous[0]) ** 2 + (cy - previous[1]) ** 2
            score = -distance2 - 25 * abs(radius - previous[2]) + strength // 50
        if best_score is None or score > best_score:
            best = (cx, cy, radius, strength)
            best_score = score
    return best


def draw_result(img, target, stable_count):
    img.draw_rectangle(ROI[0], ROI[1], ROI[2], ROI[3],
                       color=(0, 255, 0), thickness=1)
    img.draw_cross(GRAB_X, GRAB_Y, color=(255, 0, 0),
                   size=12, thickness=2)
    if target is None:
        img.draw_string(5, 5, "NO CIRCLE", color=(255, 0, 0))
        return "CIR,0,0,0,0,0,0,0"

    cx, cy, radius, _strength = target
    dx, dy = cx - GRAB_X, cy - GRAB_Y
    stable = stable_count >= STABLE_FRAMES
    aligned = (GRAB_POINT_CALIBRATED and stable
               and abs(dx) <= ALIGN_TOLERANCE_X
               and abs(dy) <= ALIGN_TOLERANCE_Y)

    img.draw_circle(cx, cy, radius, color=(0, 255, 0), thickness=2)
    img.draw_cross(cx, cy, color=(0, 255, 0), size=10, thickness=2)
    img.draw_line(cx, cy, GRAB_X, GRAB_Y,
                  color=(255, 255, 0), thickness=1)
    status = "ALIGNED" if aligned else ("STABLE" if stable else "SEARCH")
    if not GRAB_POINT_CALIBRATED:
        status = "CALIBRATE"
    img.draw_string(5, 5, "%s dx=%d dy=%d" % (status, dx, dy),
                    color=(255, 255, 0))
    return "CIR,1,%d,%d,%d,%d,%d,%d" % (
        1 if aligned else 0, cx, cy, radius, dx, dy)


sensor = None
try:
    sensor = Sensor(width=FRAME_WIDTH, height=FRAME_HEIGHT)
    sensor.reset()
    sensor.set_framesize(width=FRAME_WIDTH, height=FRAME_HEIGHT)
    sensor.set_pixformat(Sensor.RGB565)
    Display.init(Display.VIRT, width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=30)
    MediaManager.init()
    sensor.run()

    clock = time.clock()
    previous = None
    stable_count = 0
    frame_count = 0
    while True:
        os.exitpoint()
        clock.tick()
        img = sensor.snapshot()
        target = choose_circle(img, previous)

        if target is None:
            previous = None
            stable_count = 0
        else:
            if previous is not None and (max(abs(target[0] - previous[0]),
                                              abs(target[1] - previous[1])) <= MAX_CENTER_JUMP
                                         and abs(target[2] - previous[2]) <= MAX_RADIUS_JUMP):
                stable_count += 1
            else:
                stable_count = 1
            previous = target

        packet = draw_result(img, target, stable_count)
        Display.show_image(img)
        frame_count += 1
        if frame_count % PRINT_EVERY_N_FRAMES == 0:
            print(packet, "fps=", clock.fps())
        if frame_count % 30 == 0:
            gc.collect()

except KeyboardInterrupt:
    print("user stop")
except BaseException as exc:
    print("circle vision exception:", exc)
finally:
    if isinstance(sensor, Sensor):
        sensor.stop()
    Display.deinit()
    os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
    time.sleep_ms(100)
    MediaManager.deinit()
