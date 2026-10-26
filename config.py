FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# 只在夹爪前方区域识别，格式为 (x, y, w, h)。
DETECT_ROI = (20, 45, 280, 180)

# LAB 阈值仅为启动占位值，必须用实际物料和赛场灯光重新标定。
COLOR_THRESHOLDS = {
    "RED": (20, 80, 20, 127, -20, 80),
    "GREEN": (20, 85, -128, -15, -20, 90),
    "BLUE": (10, 75, -20, 70, -128, -10),
}

PIXELS_THRESHOLD = 120
AREA_THRESHOLD = 150

# 画面中的抓取基准点，不一定等于几何中心；装车后标定。
TARGET_X = FRAME_WIDTH // 2
TARGET_Y = 175

STABLE_REQUIRED = 4
MAX_CENTER_JUMP = 18

