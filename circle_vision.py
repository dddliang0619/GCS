"""圆检测、目标连续性和抓取偏差计算；不依赖具体摄像头驱动。"""

import circle_config as cfg


def choose_circle(img, previous=None):
    """返回 (cx, cy, radius, magnitude)，或 None。"""
    circles = img.find_circles(
        roi=cfg.DETECT_ROI,
        x_stride=cfg.X_STRIDE,
        y_stride=cfg.Y_STRIDE,
        threshold=cfg.HOUGH_THRESHOLD,
    )

    best = None
    best_score = None
    for circle in circles:
        radius = circle.r()
        if radius < cfg.MIN_RADIUS or radius > cfg.MAX_RADIUS:
            continue
        cx, cy = circle.x(), circle.y()
        strength = circle.magnitude()

        # 已追踪到物料时优先保持同一个圆；失去追踪时取最强边缘。
        if previous is None:
            score = strength
        else:
            distance2 = (cx - previous[0]) ** 2 + (cy - previous[1]) ** 2
            radius_delta = abs(radius - previous[2])
            score = -distance2 - 25 * radius_delta + strength // 50
        if best_score is None or score > best_score:
            best = (cx, cy, radius, strength)
            best_score = score
    return best


class CircleTracker:
    def __init__(self):
        self.previous = None
        self.stable_count = 0

    def update(self, candidate):
        if candidate is None:
            self.previous = None
            self.stable_count = 0
            return None

        if self.previous is not None:
            jump = max(abs(candidate[0] - self.previous[0]),
                       abs(candidate[1] - self.previous[1]))
            radius_jump = abs(candidate[2] - self.previous[2])
            if jump <= cfg.MAX_CENTER_JUMP and radius_jump <= cfg.MAX_RADIUS_JUMP:
                self.stable_count += 1
            else:
                self.stable_count = 1
        else:
            self.stable_count = 1

        self.previous = candidate
        dx = candidate[0] - cfg.GRAB_X
        dy = candidate[1] - cfg.GRAB_Y
        stable = self.stable_count >= cfg.STABLE_FRAMES
        aligned = (cfg.GRAB_POINT_CALIBRATED and stable
                   and abs(dx) <= cfg.ALIGN_TOLERANCE_X
                   and abs(dy) <= cfg.ALIGN_TOLERANCE_Y)
        return {
            "cx": candidate[0], "cy": candidate[1],
            "r": candidate[2], "strength": candidate[3],
            "dx": dx, "dy": dy,
            "stable": stable, "aligned": aligned,
            "count": self.stable_count,
        }


def draw_overlay(img, result):
    x, y, w, h = cfg.DETECT_ROI
    img.draw_rectangle(x, y, w, h, color=(0, 255, 0), thickness=1)
    img.draw_cross(cfg.GRAB_X, cfg.GRAB_Y, color=(255, 0, 0), size=12, thickness=2)
    if result is None:
        img.draw_string(5, 5, "NO CIRCLE", color=(255, 0, 0))
        return
    cx, cy, radius = result["cx"], result["cy"], result["r"]
    img.draw_circle(cx, cy, radius, color=(0, 255, 0), thickness=2)
    img.draw_cross(cx, cy, color=(0, 255, 0), size=10, thickness=2)
    img.draw_line(cx, cy, cfg.GRAB_X, cfg.GRAB_Y, color=(255, 255, 0), thickness=1)
    status = "ALIGNED" if result["aligned"] else ("STABLE" if result["stable"] else "SEARCH")
    if not cfg.GRAB_POINT_CALIBRATED:
        status = "CALIBRATE"
    img.draw_string(5, 5, "%s dx=%d dy=%d" %
                    (status, result["dx"], result["dy"]), color=(255, 255, 0))


def format_result(result):
    """调试文本；像素坐标系 x 向右、y 向下，尚非小车运动方向。"""
    if result is None:
        return "CIR,0,0,0,0,0,0,0"
    return "CIR,1,%d,%d,%d,%d,%d,%d" % (
        1 if result["aligned"] else 0,
        result["cx"], result["cy"], result["r"],
        result["dx"], result["dy"])
