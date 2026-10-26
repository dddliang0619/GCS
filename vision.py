import config


class TargetResult:
    def __init__(self, valid=False, color="NONE", cx=0, cy=0,
                 pixels=0, confidence=0):
        self.valid = valid
        self.color = color
        self.cx = cx
        self.cy = cy
        self.dx = cx - config.TARGET_X if valid else 0
        self.dy = cy - config.TARGET_Y if valid else 0
        self.pixels = pixels
        self.confidence = confidence


class StableTargetFilter:
    def __init__(self):
        self.last_color = "NONE"
        self.last_cx = 0
        self.last_cy = 0
        self.stable_count = 0

    def reset(self):
        self.last_color = "NONE"
        self.last_cx = 0
        self.last_cy = 0
        self.stable_count = 0

    def update(self, candidate):
        if candidate is None:
            self.reset()
            return TargetResult()

        same_color = candidate.color == self.last_color
        close_enough = (
            abs(candidate.cx - self.last_cx) <= config.MAX_CENTER_JUMP and
            abs(candidate.cy - self.last_cy) <= config.MAX_CENTER_JUMP
        )

        if same_color and close_enough:
            self.stable_count += 1
        else:
            self.stable_count = 1

        self.last_color = candidate.color
        self.last_cx = candidate.cx
        self.last_cy = candidate.cy

        candidate.confidence = min(100,
                                   self.stable_count * 100 // config.STABLE_REQUIRED)
        candidate.valid = self.stable_count >= config.STABLE_REQUIRED
        return candidate


def find_largest_color_blob(img, requested_color=None):
    colors = (requested_color,) if requested_color else tuple(config.COLOR_THRESHOLDS)
    best = None

    for color in colors:
        threshold = config.COLOR_THRESHOLDS[color]
        blobs = img.find_blobs(
            [threshold],
            roi=config.DETECT_ROI,
            pixels_threshold=config.PIXELS_THRESHOLD,
            area_threshold=config.AREA_THRESHOLD,
            merge=True,
        )

        for blob in blobs:
            if best is None or blob.pixels() > best.pixels:
                best = TargetResult(
                    valid=False,
                    color=color,
                    cx=blob.cx(),
                    cy=blob.cy(),
                    pixels=blob.pixels(),
                )

    return best


def draw_debug(img, result):
    x, y, w, h = config.DETECT_ROI
    img.draw_rectangle((x, y, w, h), color=(255, 255, 255), thickness=1)
    img.draw_cross(config.TARGET_X, config.TARGET_Y,
                   color=(255, 255, 0), size=12)

    if result.color != "NONE":
        color = (0, 255, 0) if result.valid else (255, 165, 0)
        img.draw_cross(result.cx, result.cy, color=color, size=16)

