%% 小车物料顶部圆心识别与抓取对准验证
% 依赖：Image Processing Toolbox、MATLAB Support Package for USB Webcams
% 用途：电脑端验证圆检测、圆心误差、毫米换算和稳定对准逻辑。
% 注意：最终比赛需将相同逻辑移植到 K230，并由 STM32 控制底盘。

clear;
clc;

%% ===== 需要按实物修改的参数 =====
CAMERA_INDEX = 1;

% 物料顶部圆的真实直径，用于由检测半径估算 mm/px。
CIRCLE_DIAMETER_MM = 40.0;

% 预计圆半径范围，单位为像素。先看画面再调整。
RADIUS_RANGE_PX = [18, 80];

% 圆比背景更亮用 "bright"，更暗用 "dark"。
CIRCLE_POLARITY = "dark";

% 灵敏度越接近 1 越容易检出，也越可能误检。
CIRCLE_SENSITIVITY = 0.92;
EDGE_THRESHOLD = 0.08;

% 抓取允许误差和连续稳定帧数。
ALIGN_TOLERANCE_MM = 2.0;
STABLE_FRAMES_REQUIRED = 5;

% 抓取基准点。NaN 表示启动后用鼠标点击标定。
% 基准点应通过“把物料放到夹爪实际能准确抓取的位置”来标定。
GRAB_TARGET_X = NaN;
GRAB_TARGET_Y = NaN;

%% ===== 摄像头和界面初始化 =====
cameraList = webcamlist;
if isempty(cameraList)
    error("未检测到电脑摄像头，请检查摄像头支持包和系统权限。");
end

disp("检测到的摄像头：");
disp(cameraList);

cam = webcam(CAMERA_INDEX);

% 优先使用 640x480，若摄像头不支持则保持默认分辨率。
if any(strcmp(cam.AvailableResolutions, "640x480"))
    cam.Resolution = "640x480";
end

firstFrame = snapshot(cam);

fig = figure( ...
    "Name", "物料圆心识别与抓取对准", ...
    "NumberTitle", "off", ...
    "Color", "white");
fig.UserData = false;
fig.CloseRequestFcn = @(src, ~) set(src, "UserData", true);

ax = axes(fig, "Position", [0.05, 0.12, 0.90, 0.83]);
imageHandle = imshow(firstFrame, "Parent", ax);
hold(ax, "on");

if isnan(GRAB_TARGET_X) || isnan(GRAB_TARGET_Y)
    title(ax, ["将物料放在夹爪实际抓取位置，" ...
               "然后点击物料顶部圆心完成基准点标定"]);
    [GRAB_TARGET_X, GRAB_TARGET_Y] = ginput(1);
end

targetMarker = plot(ax, GRAB_TARGET_X, GRAB_TARGET_Y, ...
    "y+", "MarkerSize", 24, "LineWidth", 3);

theta = linspace(0, 2*pi, 100);
circleLine = plot(ax, nan, nan, "g-", "LineWidth", 3);
centerMarker = plot(ax, nan, nan, "g+", "MarkerSize", 20, "LineWidth", 3);

statusText = text(ax, 10, 25, "等待识别", ...
    "Color", "yellow", ...
    "FontSize", 14, ...
    "FontWeight", "bold", ...
    "BackgroundColor", "black", ...
    "Margin", 4);

uicontrol(fig, ...
    "Style", "pushbutton", ...
    "String", "停止并释放摄像头", ...
    "Units", "normalized", ...
    "Position", [0.38, 0.02, 0.24, 0.06], ...
    "FontSize", 12, ...
    "Callback", @(~, ~) set(fig, "UserData", true));

cleanupObject = onCleanup(@() cleanupCamera(cam));

stableCount = 0;
frameCounter = 0;
fpsTimer = tic;
displayFps = 0;

%% ===== 实时圆心检测 =====
while isvalid(fig) && ~fig.UserData
    frame = snapshot(cam);
    grayFrame = im2gray(frame);

    [centers, radii, metrics] = imfindcircles( ...
        grayFrame, ...
        RADIUS_RANGE_PX, ...
        "ObjectPolarity", CIRCLE_POLARITY, ...
        "Sensitivity", CIRCLE_SENSITIVITY, ...
        "EdgeThreshold", EDGE_THRESHOLD);

    imageHandle.CData = frame;

    if ~isempty(centers)
        % imfindcircles 已按强度排序，仍显式选最大置信度目标。
        [confidence, bestIndex] = max(metrics);
        center = centers(bestIndex, :);
        radius = radii(bestIndex);

        % 利用已知真实直径进行毫米/像素换算。
        mmPerPixel = CIRCLE_DIAMETER_MM / (2 * radius);
        errorXmm = (center(1) - GRAB_TARGET_X) * mmPerPixel;
        errorYmm = (center(2) - GRAB_TARGET_Y) * mmPerPixel;
        totalErrorMm = hypot(errorXmm, errorYmm);

        circleLine.XData = center(1) + radius * cos(theta);
        circleLine.YData = center(2) + radius * sin(theta);
        centerMarker.XData = center(1);
        centerMarker.YData = center(2);

        if totalErrorMm <= ALIGN_TOLERANCE_MM
            stableCount = stableCount + 1;
        else
            stableCount = 0;
        end

        aligned = stableCount >= STABLE_FRAMES_REQUIRED;

        if aligned
            centerMarker.Color = "cyan";
            circleLine.Color = "cyan";
            commandText = "已稳定对准，可以抓取";
        else
            centerMarker.Color = "green";
            circleLine.Color = "green";
            commandText = directionText(errorXmm, errorYmm, ALIGN_TOLERANCE_MM);
        end

        statusText.String = sprintf([ ...
            "圆心=(%.1f, %.1f)  半径=%.1f px\n" ...
            "X误差=%+.1f mm  Y误差=%+.1f mm\n" ...
            "%s  稳定=%d/%d  置信度=%.2f  FPS=%.1f"], ...
            center(1), center(2), radius, ...
            errorXmm, errorYmm, commandText, ...
            stableCount, STABLE_FRAMES_REQUIRED, confidence, displayFps);
    else
        stableCount = 0;
        circleLine.XData = nan;
        circleLine.YData = nan;
        centerMarker.XData = nan;
        centerMarker.YData = nan;
        statusText.String = sprintf("未检测到圆，请调整半径范围/明暗极性  FPS=%.1f", ...
            displayFps);
    end

    frameCounter = frameCounter + 1;
    elapsed = toc(fpsTimer);
    if elapsed >= 1
        displayFps = frameCounter / elapsed;
        frameCounter = 0;
        fpsTimer = tic;
    end

    drawnow limitrate;
end

if isvalid(fig)
    delete(fig);
end

clear cleanupObject;


function textValue = directionText(errorXmm, errorYmm, tolerance)
    commands = strings(0);

    if errorXmm < -tolerance
        commands(end + 1) = "目标在左";
    elseif errorXmm > tolerance
        commands(end + 1) = "目标在右";
    end

    if errorYmm < -tolerance
        commands(end + 1) = "目标在上";
    elseif errorYmm > tolerance
        commands(end + 1) = "目标在下";
    end

    if isempty(commands)
        textValue = "位置进入容差，等待稳定";
    else
        textValue = strjoin(commands, "，");
    end
end


function cleanupCamera(cam)
    closePreview(cam);
    clear cam;
end
