import gc
import os
import time

from media.display import Display
from media.media import MediaManager
from media.sensor import Sensor

import config
from protocol import encode_result
from vision import StableTargetFilter, draw_debug, find_largest_color_blob


sensor = None
target_filter = StableTargetFilter()

# 第一阶段固定识别全部颜色。接入 UART 后由 STM32 的 SEARCH 命令修改。
requested_color = None


try:
    sensor = Sensor(width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT)
    sensor.reset()
    sensor.set_framesize(width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT)
    sensor.set_pixformat(Sensor.RGB565)

    # 使用 IDE 虚拟显示，便于调阈值和观察识别结果。
    Display.init(Display.VIRT,
                 width=config.FRAME_WIDTH,
                 height=config.FRAME_HEIGHT,
                 fps=30)
    MediaManager.init()
    sensor.run()

    clock = time.clock()

    while True:
        clock.tick()
        os.exitpoint()

        img = sensor.snapshot()
        candidate = find_largest_color_blob(img, requested_color)
        result = target_filter.update(candidate)

        draw_debug(img, result)
        Display.show_image(img)

        # 先从 IDE 串口观察协议；接线后把这里替换为 uart.write(...).
        print(encode_result(result), "fps=", clock.fps())
        gc.collect()

except KeyboardInterrupt:
    print("user stop")
except BaseException as exc:
    print("K230 vision exception:", exc)
finally:
    if isinstance(sensor, Sensor):
        sensor.stop()
    Display.deinit()
    os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
    time.sleep_ms(100)
    MediaManager.deinit()
