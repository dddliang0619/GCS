# GCS — 视觉模块

小车物料识别与抓取对准的视觉代码，包含电脑端 Python/OpenCV、K230/CanMV 和 MATLAB 验证脚本。

当前主要入口是 `pc_circle_camera.py`：同时识别红、绿、蓝色圆形物料，显示圆心，并计算选中物料相对抓取基准点的像素偏差。

## 电脑端运行

在仓库根目录打开 PowerShell，安装 Python 后执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe pc_circle_camera.py
```

多个摄像头时可加 `--camera 1`。也可以在 VS Code 中打开本目录，按 F5 选择“电脑摄像头：圆心识别”。

| 按键 | 功能 |
| --- | --- |
| `1` / `2` / `3` | 选择红 / 绿 / 蓝目标 |
| `m` | 显示或关闭选中颜色的二值图 |
| `c` | 稳定检测后保存抓取基准点 |
| `r` | 清除抓取基准点 |
| `q` / Esc | 退出 |

`dx = 圆心 x - 基准点 x`，`dy = 圆心 y - 基准点 y`；图像坐标 x 向右、y 向下。标定生成的 `pc_circle_config.json` 属于本机配置，不纳入版本控制。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `pc_circle_camera.py` | 电脑摄像头、显示、ROI 和标定入口 |
| `pc_circle_vision.py` | 颜色分割、圆拟合、内部颜色验证、目标关联与连续对准 |
| `test_pc_circle.py` | 不依赖实体摄像头的回归测试 |
| `circle_main.py` | K230 圆检测独立入口，参数直接写在脚本中 |
| `circle_config.py`、`circle_vision.py` | K230 圆检测参数和模块化算法参考；独立入口当前没有导入这两个模块 |
| `main.py`、`config.py`、`vision.py`、`protocol.py` | K230 颜色块识别第一版、稳定滤波和通信协议辅助代码 |
| `matlab_circle_centering.m` | MATLAB 圆心定位与对准验证 |
| `matlab_face_detection.m` | MATLAB 摄像头人脸检测实验 |
| `.vscode/`、`k230.code-workspace` | VS Code / CanMV 配置 |

K230 脚本需要开发板上的 CanMV 固件及其 `media` 模块，不能用电脑端的 Python/OpenCV 直接运行。MATLAB 脚本依赖各自文件开头列出的工具箱和摄像头支持包。

## 使用说明与当前进度

- [红绿蓝圆心识别使用指南](红绿蓝圆心识别使用指南.md)
- [电脑摄像头圆心说明](电脑摄像头圆心说明.md)
- [圆心定位说明](圆心定位说明.md)
- [K230 第一版说明](docs/K230第一版说明.md)

上述原始文档保留了开发机目录 `F:\k230`；在其他电脑上使用时，应替换为自己的仓库路径，以本页的相对路径启动命令为准。

电脑端已增加圆拟合误差、原始掩膜内部颜色占比、同色近邻关联、原始测量稳定窗口，以及连续 5 帧对准确认。旧版标定文件需要重新按 c 生成。新增离线回归覆盖形状误检、跟踪丢失、标定和模拟相机完整流程；实物抓取精度和实时摄像头运行仍待验收。K230 第一版说明中的串口接线与控制步骤是后续接入计划；现有主程序只打印结果，尚未实现物理 UART 收发或小车、夹爪控制。像素偏差需要另行标定才能换算成实际移动量。

## ROI 与离线回归

```powershell
.\.venv\Scripts\python.exe pc_circle_camera.py --roi 80 60 480 360
.\.venv\Scripts\python.exe -m unittest discover -s . -p 'test_pc_circle.py' -v
```

算法阈值位于 `pc_circle_vision.py`。`m` 显示原始/处理后掩膜对照。新版要求先满足最近 5 帧原始测量稳定，再连续 5 帧满足原始/平滑中心均进入容差；目标丢失当帧取消对准。K230/CanMV 脚本尚未移植这些电脑端优化。
