%% 电脑摄像头实时人脸检测
% 依赖：
% 1. Computer Vision Toolbox
% 2. MATLAB Support Package for USB Webcams

clear;
clc;

cameraList = webcamlist;
if isempty(cameraList)
    error(['未检测到电脑摄像头。请检查摄像头连接、Windows权限，' ...
           '以及 MATLAB Support Package for USB Webcams 是否已安装。']);
end

disp("检测到的摄像头：");
disp(cameraList);

% 如果有多个摄像头，可将 1 改成其他编号。
cam = webcam(1);

% 使用 Viola-Jones 级联检测器检测正面人脸。
faceDetector = vision.CascadeObjectDetector();
faceDetector.MergeThreshold = 4;

fig = figure( ...
    "Name", "电脑摄像头人脸检测", ...
    "NumberTitle", "off", ...
    "Color", "white");

% 关闭窗口时先设置停止标志，防止 imshow 自动创建新窗口。
fig.UserData = false;
fig.CloseRequestFcn = @(src, ~) set(src, "UserData", true);

uicontrol(fig, ...
    "Style", "pushbutton", ...
    "String", "停止并释放摄像头", ...
    "Units", "normalized", ...
    "Position", [0.38, 0.01, 0.24, 0.06], ...
    "FontSize", 12, ...
    "Callback", @(~, ~) set(fig, "UserData", true));

cleanupObject = onCleanup(@() cleanupCamera(cam, faceDetector));

while isvalid(fig) && ~fig.UserData
    frame = snapshot(cam);
    boxes = faceDetector(frame);

    if ~isempty(boxes)
        result = insertObjectAnnotation( ...
            frame, ...
            "rectangle", ...
            boxes, ...
            repmat("Face", size(boxes, 1), 1), ...
            "Color", "green", ...
            "LineWidth", 3, ...
            "FontSize", 18);
        statusText = "检测到 " + size(boxes, 1) + " 张人脸";
    else
        result = frame;
        statusText = "未检测到人脸";
    end

    imshow(result, "Parent", gca);
    title(statusText);
    drawnow limitrate;
end

if isvalid(fig)
    delete(fig);
end

clear cleanupObject;


function cleanupCamera(cam, detector)
    release(detector);
    clear cam;
end
