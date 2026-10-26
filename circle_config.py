"""K230 顶视圆形物料定位参数。先在 CanMV 预览中标定，再接小车控制。"""

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# 搜索区域 (x, y, w, h)，只框住取料区可明显提升速度并减少误检。
DETECT_ROI = (40, 30, 240, 180)
X_STRIDE = 3
Y_STRIDE = 2
HOUGH_THRESHOLD = 2000
MIN_RADIUS = 12
MAX_RADIUS = 65

# 把圆形物料手动放到“夹爪确实能夹中的位置”，读取串口连续稳定的 cx,cy，填到这里。
# 初始值只是显示参考，不代表已经标定。
GRAB_X = FRAME_WIDTH // 2
GRAB_Y = FRAME_HEIGHT // 2
GRAB_POINT_CALIBRATED = False

# 只有完成夹爪基准点标定后，才允许输出 aligned=1。
ALIGN_TOLERANCE_X = 4
ALIGN_TOLERANCE_Y = 4
STABLE_FRAMES = 5
MAX_CENTER_JUMP = 12
MAX_RADIUS_JUMP = 7

# 调试打印过于频繁会拖慢显示；每隔若干帧输出一次。
PRINT_EVERY_N_FRAMES = 5
