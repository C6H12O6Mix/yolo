@echo off
echo ===== YOLO视频流处理项目启动脚本 =====

REM 检查weights目录是否存在
if not exist weights mkdir weights
echo 已创建weights目录，请确保将yolo11n-obb.pt文件放入该目录

REM 检查weights/yolo11n-obb.pt文件是否存在
if not exist weights\yolo11n-obb.pt (
    echo 警告: weights\yolo11n-obb.pt文件不存在
    echo 请先下载YOLO11n-OBB模型权重文件并放入weights目录
    echo 下载完成后再次运行此脚本
    pause
    exit /b 1
)

REM 启动应用（跳过Nginx检查）
echo 正在启动应用（Windows模式，跳过Nginx）...
python main.py --skip-nginx

pause 