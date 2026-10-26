"""电脑端圆形检测与跟踪；不打开摄像头，便于离线回归。"""

from collections import deque
import math
from typing import NamedTuple

import cv2
import numpy as np


HSV_RANGES = {
    "RED": [((0, 90, 50), (12, 255, 255)),
            ((170, 90, 50), (179, 255, 255))],
    "GREEN": [((35, 65, 45), (85, 255, 255))],
    "BLUE": [((90, 65, 45), (135, 255, 255))],
}
MIN_AREA = 500
MIN_RADIUS, MAX_RADIUS = 14, 150
MIN_CIRCULARITY = 0.65
MIN_FILL_RATIO = 0.58
MAX_FIT_ERROR = 0.06  # 轮廓径向误差的 RMS / 拟合半径
MAX_CENTER_DISAGREEMENT = 0.10
MIN_COLOR_OCCUPANCY = 0.90  # 0.85 倍半径内，原始颜色掩膜的有效像素占比
STABLE_FRAMES = 5
MAX_CENTER_JUMP = 16
MAX_RADIUS_JUMP = 10
MAX_CENTER_SPREAD = 3.0  # 最近 5 帧原始中心的单轴极差
MAX_RADIUS_SPREAD = 3.0
MAX_RAW_STEP = 1.5  # 每帧单轴位移；帧率变化后需现场调整
SMOOTHING = 0.45
ALIGN_TOLERANCE_PX = 5
ALIGN_FRAMES = 5
MAX_MISSED_FRAMES = 3
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


class Detection(NamedTuple):
    cx: float
    cy: float
    radius: float
    score: float


def make_mask(hsv, color, clean=True):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in HSV_RANGES[color]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, low, high))
    return clean_mask(mask) if clean else mask


def clean_mask(mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)


def fit_circle(contour):
    """中心化后最小二乘拟合圆，返回 (cx, cy, r, 相对径向 RMS)。"""
    points = contour.reshape(-1, 2).astype(np.float64)
    origin = points.mean(axis=0)
    local = points - origin
    matrix = np.column_stack((2 * local, np.ones(len(local))))
    values = np.sum(local * local, axis=1)
    solution, _, rank, _ = np.linalg.lstsq(matrix, values, rcond=None)
    radius2 = solution[2] + np.dot(solution[:2], solution[:2])
    if rank < 3 or radius2 <= 0 or not np.isfinite(solution).all():
        return None
    radius = math.sqrt(radius2)
    residual = np.linalg.norm(local - solution[:2], axis=1) - radius
    error = float(np.sqrt(np.mean(residual ** 2)) / radius)
    center = origin + solution[:2]
    return float(center[0]), float(center[1]), radius, error


def detect_circles(mask, raw_mask=None, offset=(0, 0)):
    """保留全部合格候选；圆心采用拟合值，质心用于一致性检查。

    raw_mask 为形态学处理前的掩膜，防止闭运算掩盖实际孔洞。
    offset 将 ROI 局部坐标还原成全图坐标。触及 ROI/画面边界的轮廓拒绝。
    """
    if raw_mask is None:
        raw_mask = mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    height, width = mask.shape
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if x <= 0 or y <= 0 or x + w >= width or y + h >= height:
            continue
        _, enclosing_radius = cv2.minEnclosingCircle(contour)
        if not MIN_RADIUS <= enclosing_radius <= MAX_RADIUS:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4 * math.pi * area / perimeter ** 2
        fill = area / (math.pi * enclosing_radius ** 2)
        if circularity < MIN_CIRCULARITY or fill < MIN_FILL_RATIO:
            continue
        fitted = fit_circle(contour)
        if fitted is None:
            continue
        cx, cy, radius, error = fitted
        if not MIN_RADIUS <= radius <= MAX_RADIUS or error > MAX_FIT_ERROR:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        mx = moments["m10"] / moments["m00"]
        my = moments["m01"] / moments["m00"]
        if math.hypot(cx - mx, cy - my) > max(2, MAX_CENTER_DISAGREEMENT * radius):
            continue
        # 只分配候选包围盒大小的网格，避免每个候选都扫描全图。
        yy, xx = np.ogrid[y:y + h, x:x + w]
        inner_disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= (0.85 * radius) ** 2
        occupancy = np.count_nonzero(raw_mask[y:y + h, x:x + w][inner_disk]) / max(
            1, np.count_nonzero(inner_disk))
        if occupancy < MIN_COLOR_OCCUPANCY:
            continue
        score = area * min(circularity, 1) * min(fill, 1) * occupancy * (1 - error)
        candidates.append(Detection(cx + offset[0], cy + offset[1], radius, score))
    return candidates


def detect_circle(mask):
    """单候选兼容接口；实时入口使用全部候选进行关联。"""
    return max(detect_circles(mask), key=lambda item: item.score, default=None)


class CircleTracker:
    """近邻关联、短暂保留关联锚点、原始坐标稳定检查和连续对准。"""

    def __init__(self):
        self.anchor = None
        self.target = None
        self.raw = None
        self.history = deque(maxlen=STABLE_FRAMES)
        self.missed = 0
        self.stable = False
        self.align_count = 0
        self.aligned = False

    def reset_alignment(self):
        self.align_count = 0
        self.aligned = False

    def update(self, candidates, grab_point=None):
        if self.anchor is None:
            if grab_point is None:
                candidate = max(candidates, key=lambda item: item.score, default=None)
            else:
                candidate = min(candidates, key=lambda item: math.hypot(
                    item.cx - grab_point[0], item.cy - grab_point[1]), default=None)
        else:
            nearby = [item for item in candidates
                      if max(abs(item.cx - self.anchor.cx),
                             abs(item.cy - self.anchor.cy)) <= MAX_CENTER_JUMP
                      and abs(item.radius - self.anchor.radius) <= MAX_RADIUS_JUMP]
            candidate = min(nearby, key=lambda item:
                            ((item.cx - self.anchor.cx) / MAX_CENTER_JUMP) ** 2
                            + ((item.cy - self.anchor.cy) / MAX_CENTER_JUMP) ** 2
                            + ((item.radius - self.anchor.radius) / MAX_RADIUS_JUMP) ** 2,
                            default=None)
        if candidate is None:
            self.missed += 1
            if self.missed >= MAX_MISSED_FRAMES:
                self.anchor = None
            # 保留的只是关联锚点；丢失当帧不输出旧坐标、不允许对准。
            self.target = self.raw = None
            self.history.clear()
            self.stable = False
            self.reset_alignment()
            return None

        previous_raw = self.raw
        self.missed = 0
        self.anchor = self.raw = candidate
        if self.target is None:
            self.target = candidate
        else:
            self.target = Detection(*(SMOOTHING * new + (1 - SMOOTHING) * old
                                      for new, old in zip(candidate[:3], self.target[:3])),
                                    candidate.score)
        self.history.append(candidate)
        spread = [max(item[i] for item in self.history)
                  - min(item[i] for item in self.history) for i in range(3)]
        step = (max(abs(candidate.cx - previous_raw.cx),
                    abs(candidate.cy - previous_raw.cy))
                if previous_raw is not None else float("inf"))
        self.stable = (len(self.history) == STABLE_FRAMES
                       and max(spread[:2]) <= MAX_CENTER_SPREAD
                       and spread[2] <= MAX_RADIUS_SPREAD and step <= MAX_RAW_STEP)
        within = (grab_point is not None and self.stable
                  and all(abs(point[i] - grab_point[i]) <= ALIGN_TOLERANCE_PX
                          for point in (candidate, self.target) for i in (0, 1)))
        self.align_count = min(self.align_count + 1, ALIGN_FRAMES) if within else 0
        self.aligned = self.align_count >= ALIGN_FRAMES
        return self.target
